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

from .contrastllm import ContrastLLM
from .offsetllm import create_offset_model
from .utils import *
from .doubleassisllm import DoubleAssisLLM
try:
    from .tofu05assisllm import Tofu05AssisLLM
    from .tofu05assisllm_fast import Tofu05AssisLLMFast
    from .tofu05assisllm_lookup import Tofu05AssisLLMLookup
except ImportError:
    Tofu05AssisLLM = None
    Tofu05AssisLLMFast = None
    Tofu05AssisLLMLookup = None
from ..utils import NameTimer

DEFAULT_TINYLLAMA_ASSIST_PATH = os.environ.get(
    "ASSIST_MODEL",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
)

TRAIN_INIT_FUNCS = {
    "base": create_full_model,
    "uld": create_peft_model,
    "offset": create_offset_model,
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


def _infer_double_assis_routing_score_paths(finetuned_assist_path):
    env_paths = os.environ.get("ROUTING_SCORE_PATHS", "").strip()
    if env_paths:
        return [p for p in re.split(r"[,|]", env_paths) if p]

    path_str = str(finetuned_assist_path)
    match = re.search(
        r"artifacts/outputs_trained_models/cbd_dfb_tinyllama_(?P<run_tag>[^/]+)/hf_forget_train/.*/logs/(?P<project>cbd_dfb_[^/]+)/",
        path_str,
    )
    if not match:
        return []

    repo_root = Path(__file__).resolve().parents[2]
    ce_dir = repo_root / "artifacts" / "ce_results_cbd_dfb" / match.group("run_tag") / match.group("project")
    if not ce_dir.is_dir():
        return []

    return [str(p) for p in sorted(ce_dir.glob("tinyllama_comparison_results_*.json"))]


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


def eval_create_uld_model(base_model_config, model_mode_config, ckpt_path, device):
    with NameTimer("Loading ULD model"):
        eval_devices = _resolve_eval_devices(model_mode_config, device)
        base_device = eval_devices[0]
        assist_device = eval_devices[1] if len(eval_devices) > 1 else eval_devices[0]
        attn_implementation = _resolve_eval_attn_implementation(base_model_config)
        base_kwargs = {
            "torch_dtype": torch.bfloat16,
            "local_files_only": _resolve_local_files_only(),
        }
        if attn_implementation:
            base_kwargs["attn_implementation"] = attn_implementation
        basellm = AutoModelForCausalLM.from_pretrained(
            base_model_config.model_path, use_flash_attention_2=False, **base_kwargs
        ).to(base_device)
        with NameTimer("Loading assistant"):
            small_full_path = _resolve_small_full_path(base_model_config, ckpt_path)
            assistant = AutoModelForCausalLM.from_pretrained(
                small_full_path, use_flash_attention_2=False, **base_kwargs
            ).to(assist_device)
            peftmod = _load_peft_for_eval(assistant, ckpt_path)
            peftmod = peftmod.merge_and_unload()
            peftmod = peftmod.to(assist_device)

        model = ContrastLLM(
            basellm, peftmod, 
            weight=model_mode_config.weight, 
            top_logit_filter=model_mode_config.top_logit_filter,
        ) 
        return model

def eval_create_offset_model(base_model_config, model_mode_config, ckpt_path, device):
    with NameTimer("Loading Offset model"):
        try:
            config = AutoConfig.from_pretrained(ckpt_path)
        except Exception:
            config = None
        eval_devices = _resolve_eval_devices(model_mode_config, device)
        offset_kwargs = {}
        if len(eval_devices) >= 3:
            offset_kwargs.update(
                base_device=eval_devices[0],
                base_assist_device=eval_devices[1],
                assist_device=eval_devices[2],
            )
        elif len(eval_devices) == 2:
            offset_kwargs.update(
                base_device=eval_devices[0],
                base_assist_device=eval_devices[1],
                assist_device=eval_devices[1],
            )
        else:
            offset_kwargs.update(device=eval_devices[0])
        if config is not None and hasattr(config, 'is_offset') and config.is_offset:
            explicit_weight = getattr(model_mode_config, "weight", None)
            if explicit_weight is not None:
                weight = float(explicit_weight)
            elif hasattr(config, 'weight'):
                weight = config.weight
            else:
                weight = 1.0
            base_name = config.base_model_name
            base_assist_path = getattr(model_mode_config, "base_assist_path", None) or config.base_assist_path
            model = create_offset_model(
                base_name, 
                base_assist_path=base_assist_path, 
                weight=weight, 
                new_assist_path=ckpt_path,
                **offset_kwargs,
            )
            return model
        base_name = getattr(base_model_config, "model_path", None)
        base_assist_path = getattr(model_mode_config, "base_assist_path", None)
        if not base_name or not base_assist_path:
            return None
        weight = float(getattr(model_mode_config, "weight", 1.0))
        model = create_offset_model(
            base_name,
            base_assist_path=base_assist_path,
            weight=weight,
            new_assist_path=ckpt_path,
            **offset_kwargs,
        )
        return model

def eval_create_double_assis_model(base_model_config, model_mode_config, ckpt_path, device):
    """
    创建DoubleAssisLLM模型

    Args:
        base_model_config: 基础模型配置
        model_mode_config: 模型模式配置，应包含：
            - original_assist_path: 原始辅助模型路径
            - finetuned_assist_path: 微调后辅助模型路径
            - threshold: 交叉熵阈值 (默认12.6943)
            - max_new_tokens: 最大生成token数 (默认20)
        ckpt_path: 检查点路径 (这里可能不使用，但保持接口一致)
        device: 设备
    """
    with NameTimer("Loading DoubleAssisLLM model"):
        attn_implementation = _resolve_eval_attn_implementation(base_model_config)
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "local_files_only": _resolve_local_files_only(),
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        # 加载原始大模型
        print(f"Loading base model from: {base_model_config.model_path}")
        basellm = AutoModelForCausalLM.from_pretrained(
            base_model_config.model_path, **model_kwargs
        ).to(device)

        # 加载原始辅助模型
        original_assist_path = getattr(model_mode_config, 'original_assist_path', DEFAULT_TINYLLAMA_ASSIST_PATH)
        print(f"Loading original assist model from: {original_assist_path}")
        original_assist_llm = AutoModelForCausalLM.from_pretrained(
            original_assist_path, **model_kwargs
        ).to(device)

        # 加载微调后的辅助模型
        # 优先使用配置文件中的路径，如果没有则使用ckpt_path
        finetuned_assist_path = getattr(model_mode_config, 'finetuned_assist_path', None)
        if finetuned_assist_path is None:
            finetuned_assist_path = ckpt_path
            print(f"Using ckpt_path for finetuned assist model: {finetuned_assist_path}")
        else:
            print(f"Using config finetuned_assist_path: {finetuned_assist_path}")
        print(f"Loading finetuned assist model from: {finetuned_assist_path}")
        routing_score_paths = _infer_double_assis_routing_score_paths(finetuned_assist_path)
        if routing_score_paths:
            print(f"Preloading routing score cache from {len(routing_score_paths)} file(s)")

        # 检查是否是LoRA模型
        if os.path.exists(os.path.join(finetuned_assist_path, 'adapter_config.json')):
            # 这是一个LoRA模型，需要先加载基础模型再应用LoRA
            if os.path.exists(os.path.join(finetuned_assist_path, '../fullmodel')):
                base_path = os.path.join(finetuned_assist_path, '../fullmodel')
            else:
                base_path = original_assist_path  # 使用原始辅助模型作为基础

            finetuned_assist_llm = AutoModelForCausalLM.from_pretrained(
                base_path, **model_kwargs
            ).to(device)

            # 应用LoRA
            peftmod = _load_peft_for_eval(finetuned_assist_llm, finetuned_assist_path)
            finetuned_assist_llm = peftmod.merge_and_unload()
            finetuned_assist_llm = finetuned_assist_llm.to(device)
        else:
            # 直接加载完整模型
            finetuned_assist_llm = AutoModelForCausalLM.from_pretrained(
                finetuned_assist_path, **model_kwargs
            ).to(device)

        # 获取配置参数
        threshold = getattr(model_mode_config, 'threshold', 12.6943)
        max_new_tokens = getattr(model_mode_config, 'max_new_tokens', 20)

        print(f"Creating DoubleAssisLLM with threshold: {threshold}, max_new_tokens: {max_new_tokens}")

        # 创建DoubleAssisLLM模型
        model = DoubleAssisLLM(
            basellm=basellm,
            original_assist_llm=original_assist_llm,
            finetuned_assist_llm=finetuned_assist_llm,
            routing_score_paths=routing_score_paths,
            threshold=threshold,
            max_new_tokens=max_new_tokens
        )

        return model

def eval_create_tofu05_assis_model(base_model_config, model_mode_config, ckpt_path, device):
    """
    创建Tofu05AssisLLM模型
    """
    with NameTimer("Loading Tofu05AssisLLM model"):
        # 加载基础大模型
        print(f"Loading base model from: {base_model_config.model_path}")
        basellm = AutoModelForCausalLM.from_pretrained(
            base_model_config.model_path, torch_dtype=torch.bfloat16
        ).to(device)

        # 加载原始辅助模型
        original_assist_path = getattr(model_mode_config, 'original_assist_path', DEFAULT_TINYLLAMA_ASSIST_PATH)
        print(f"Loading original assist model from: {original_assist_path}")
        original_assist_llm = AutoModelForCausalLM.from_pretrained(
            original_assist_path, torch_dtype=torch.bfloat16
        ).to(device)

        # 获取辅助模型配置
        assist_models = getattr(model_mode_config, 'assist_models', [])
        if not assist_models:
            raise ValueError("assist_models configuration is required for tofu05_assis mode")

        # 获取其他参数
        max_new_tokens = getattr(model_mode_config, 'max_new_tokens', 20)

        # 创建Tofu05AssisLLM模型
        model = Tofu05AssisLLM(
            basellm=basellm,
            original_assist_llm=original_assist_llm,
            assist_models=assist_models,
            max_new_tokens=max_new_tokens
        )

        return model

def eval_create_tofu05_assis_fast_model(base_model_config, model_mode_config, ckpt_path, device):
    """
    创建Tofu05AssisLLMFast模型（快速版本）
    """
    with NameTimer("Loading Tofu05AssisLLMFast model"):
        # 加载基础大模型
        print(f"Loading base model from: {base_model_config.model_path}")
        basellm = AutoModelForCausalLM.from_pretrained(
            base_model_config.model_path,
            torch_dtype=torch.bfloat16,
            local_files_only=_resolve_local_files_only(),
        ).to(device)

        # 加载原始辅助模型
        original_assist_path = getattr(model_mode_config, 'original_assist_path', DEFAULT_TINYLLAMA_ASSIST_PATH)
        print(f"Loading original assist model from: {original_assist_path}")
        original_assist_llm = AutoModelForCausalLM.from_pretrained(
            original_assist_path,
            torch_dtype=torch.bfloat16,
            local_files_only=_resolve_local_files_only(),
        ).to(device)

        # 获取辅助模型配置
        assist_models = getattr(model_mode_config, 'assist_models', [])
        if not assist_models:
            raise ValueError("assist_models configuration is required for tofu05_assis_fast mode")

        # 获取其他参数
        max_new_tokens = getattr(model_mode_config, 'max_new_tokens', 10)  # 快速版本默认更少token

        # 创建Tofu05AssisLLMFast模型
        model = Tofu05AssisLLMFast(
            basellm=basellm,
            original_assist_llm=original_assist_llm,
            assist_models=assist_models,
            max_new_tokens=max_new_tokens
        )

        return model

def eval_create_tofu05_assis_lookup_model(base_model_config, model_mode_config, ckpt_path, device):
    """
    创建Tofu05AssisLLMLookup模型（查找表版本）
    """
    with NameTimer("Loading Tofu05AssisLLMLookup model"):
        # 加载基础大模型
        print(f"Loading base model from: {base_model_config.model_path}")
        basellm = AutoModelForCausalLM.from_pretrained(
            base_model_config.model_path,
            torch_dtype=torch.bfloat16,
            local_files_only=_resolve_local_files_only(),
        ).to(device)

        # 加载原始辅助模型
        original_assist_path = getattr(model_mode_config, 'original_assist_path', DEFAULT_TINYLLAMA_ASSIST_PATH)
        print(f"Loading original assist model from: {original_assist_path}")
        original_assist_llm = AutoModelForCausalLM.from_pretrained(
            original_assist_path,
            torch_dtype=torch.bfloat16,
            local_files_only=_resolve_local_files_only(),
        ).to(device)

        # 获取配置参数
        assist_models = getattr(model_mode_config, 'assist_models', [])
        if not assist_models:
            raise ValueError("assist_models configuration is required for tofu05_assis_lookup mode")

        ce_results_dir = getattr(model_mode_config, 'ce_results_dir', 'ce_results')
        max_new_tokens = getattr(model_mode_config, 'max_new_tokens', 20)

        # 创建Tofu05AssisLLMLookup模型
        model = Tofu05AssisLLMLookup(
            basellm=basellm,
            original_assist_llm=original_assist_llm,
            assist_models=assist_models,
            ce_results_dir=ce_results_dir,
            max_new_tokens=max_new_tokens
        )

        return model

TRAIN_INIT_FUNCS = {
    "base": create_full_model,
    "uld": create_peft_model,
    "offset": create_offset_model,
}

EVAL_INIT_FUNCS = {
    "base": eval_create_base_model,
    "uld": eval_create_uld_model,
    "offset": eval_create_offset_model,
    "double_assis": eval_create_double_assis_model,
    "tofu05_assis": eval_create_tofu05_assis_model,
    "tofu05_assis_fast": eval_create_tofu05_assis_fast_model,
    "tofu05_assis_lookup": eval_create_tofu05_assis_lookup_model,
}
