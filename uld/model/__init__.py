import inspect
import json
import os
import re
from pathlib import Path
import torch
from peft import PeftConfig, PeftModel, LoraConfig
from transformers import AutoConfig, AutoModelForCausalLM
try:
    from safetensors import safe_open
except ImportError:
    safe_open = None

from .utils import *
from ..utils import NameTimer


TRAIN_INIT_FUNCS = {
    "base": create_full_model,
}


def _resolve_eval_attn_implementation(base_model_config):
    env_impl = os.environ.get("EVAL_ATTN_IMPL", "").strip()
    if env_impl:
        return env_impl

    cfg_impl = getattr(base_model_config, "attn_implementation", None)
    if cfg_impl in ("", None):
        return None
    return cfg_impl


def _resolve_local_files_only() -> bool:
    return os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"


def _resolve_eval_devices(model_mode_config, device):
    if hasattr(model_mode_config, "get"):
        eval_devices = model_mode_config.get("eval_devices", device)
    else:
        eval_devices = getattr(model_mode_config, "eval_devices", device)
    if isinstance(eval_devices, str):
        eval_devices = [d.strip() for d in re.split(r"[,|]", eval_devices) if d.strip()]
    elif not isinstance(eval_devices, (list, tuple)):
        eval_devices = [eval_devices]
    if not eval_devices:
        eval_devices = [device]
    return eval_devices


def _adapter_layers_from_checkpoint(ckpt_path):
    adapter_safetensors = os.path.join(ckpt_path, "adapter_model.safetensors")
    adapter_bin = os.path.join(ckpt_path, "adapter_model.bin")
    keys = []
    try:
        if os.path.exists(adapter_safetensors) and safe_open is not None:
            with safe_open(adapter_safetensors, framework="pt") as f:
                keys = list(f.keys())
        elif os.path.exists(adapter_bin):
            state = torch.load(adapter_bin, map_location="cpu")
            keys = list(state.keys())
    except Exception as exc:
        print(f"[peft-load] failed to inspect adapter weights at {ckpt_path}: {exc}")
        return None

    layers = sorted(
        {
            int(match.group(1))
            for key in keys
            for match in [re.search(r"layers\.(\d+)\.", key)]
            if match
        }
    )
    return layers or None


def _peft_config_for_eval(base_model, ckpt_path):
    if not os.path.exists(os.path.join(ckpt_path, "adapter_config.json")):
        return None

    try:
        peft_config = PeftConfig.from_pretrained(ckpt_path)
    except TypeError as exc:
        adapter_cfg_path = os.path.join(ckpt_path, "adapter_config.json")
        with open(adapter_cfg_path, "r", encoding="utf-8") as f:
            raw_cfg = json.load(f)

        peft_type = str(raw_cfg.get("peft_type", "")).upper()
        if peft_type != "LORA":
            raise

        allowed = set(inspect.signature(LoraConfig.__init__).parameters.keys())
        filtered_cfg = {k: v for k, v in raw_cfg.items() if k in allowed}
        dropped = sorted(set(raw_cfg.keys()) - set(filtered_cfg.keys()))
        if dropped:
            print(f"[peft-load] drop unsupported adapter_config keys for eval: {dropped} ({exc})")
        peft_config = LoraConfig(**filtered_cfg)

    if getattr(peft_config, "layers_to_transform", None) is not None:
        return peft_config

    layers = _adapter_layers_from_checkpoint(ckpt_path)
    if not layers:
        return peft_config

    total_layers = getattr(getattr(base_model, "config", None), "num_hidden_layers", None)
    if total_layers is None:
        model_layers = getattr(getattr(base_model, "model", None), "layers", None)
        if model_layers is not None:
            total_layers = len(model_layers)
    if total_layers is not None and len(layers) >= total_layers:
        return peft_config

    peft_config.layers_to_transform = layers
    print(f"[peft-load] inferred layers_to_transform={layers} for {ckpt_path}")
    return peft_config


def _load_peft_for_eval(base_model, ckpt_path):
    peft_config = _peft_config_for_eval(base_model, ckpt_path)
    kwargs = {"torch_dtype": torch.bfloat16}
    if peft_config is not None:
        kwargs["config"] = peft_config
    return PeftModel.from_pretrained(base_model, ckpt_path, **kwargs)


def _is_pretrained_model_dir(path):
    if not path or not os.path.isdir(path):
        return False
    required_markers = (
        "config.json",
        "model.safetensors",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
        "model.safetensors.index.json",
    )
    return any(os.path.exists(os.path.join(path, marker)) for marker in required_markers)


def _resolve_small_full_path(base_model_config, ckpt_path):
    candidate = os.path.abspath(os.path.join(ckpt_path, "..", "fullmodel"))
    if _is_pretrained_model_dir(candidate):
        return candidate
    return base_model_config.model_path

def eval_create_base_model(base_model_config, model_mode_config, ckpt_path, device):
    with NameTimer("Loading Base model"):
        eval_devices = _resolve_eval_devices(model_mode_config, device)
        use_device_map = len(eval_devices) > 1
        attn_implementation = _resolve_eval_attn_implementation(base_model_config)
        base_kwargs = {
            "torch_dtype": torch.bfloat16,
            "local_files_only": _resolve_local_files_only(),
        }
        if attn_implementation:
            base_kwargs["attn_implementation"] = attn_implementation
        if use_device_map:
            base_kwargs.update({
                "device_map": "auto",
                "low_cpu_mem_usage": True,
            })

        if os.path.exists(os.path.join(ckpt_path, 'adapter_config.json')):
            #! A lora model
            base_path = _resolve_small_full_path(base_model_config, ckpt_path)
            model = AutoModelForCausalLM.from_pretrained(
                base_path, **base_kwargs
            )
            if not use_device_map:
                model = model.to(device)
            peftmod = _load_peft_for_eval(model, ckpt_path)
            if use_device_map:
                peftmod.eval()
                return peftmod

            peftmod = peftmod.merge_and_unload()
            peftmod = peftmod.to(device)
            return peftmod 
        else:
            # Base only
            base_kwargs["use_flash_attention_2"] = False
            model = AutoModelForCausalLM.from_pretrained(
                ckpt_path, **base_kwargs
            )
            if not use_device_map:
                model = model.to(device)
            return model


EVAL_INIT_FUNCS = {
    "base": eval_create_base_model,
}
