import os
import copy 
import torch

from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from omegaconf import ListConfig

from ..utils import NameTimer
from .peft_util import find_all_linear_names

def get_dtype(data_type):
    if data_type == 'bfloat16':
        return torch.bfloat16
    elif data_type == 'float16':
        return torch.float16

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def _summarize_trainable_parameters(model, prefix=""):
    total_params = model.num_parameters()
    trainable_params = model.num_parameters(only_trainable=True)
    trainable_ratio = (100 * trainable_params / total_params) if total_params else 0.0
    prefix = prefix or ""
    print(f"{prefix}总参数数量: {total_params:,}")
    print(f"{prefix}可训练参数数量: {trainable_params:,}")
    print(f"{prefix}可训练参数比例: {trainable_ratio:.2f}%")
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

def _unique_modules(candidates):
    mods = []
    seen = set()
    for module in candidates:
        if module is None:
            continue
        ident = id(module)
        if ident in seen:
            continue
        seen.add(ident)
        mods.append(module)
    return mods

def _resolve_backbone(module):
    candidates = _unique_modules([
        module,
        getattr(module, "model", None),
        getattr(getattr(module, "model", None), "model", None),
        getattr(module, "base_model", None),
        getattr(getattr(module, "base_model", None), "model", None),
        getattr(getattr(getattr(module, "base_model", None), "model", None), "model", None),
    ])
    for cand in candidates:
        if hasattr(cand, "embed_tokens") and hasattr(cand, "layers") and hasattr(cand, "norm"):
            return cand
    return None

def _resolve_lm_head(module):
    candidates = _unique_modules([
        module,
        getattr(module, "model", None),
        getattr(getattr(module, "model", None), "model", None),
        getattr(module, "base_model", None),
        getattr(getattr(module, "base_model", None), "model", None),
        getattr(getattr(getattr(module, "base_model", None), "model", None), "model", None),
    ])
    for cand in candidates:
        lm_head = getattr(cand, "lm_head", None)
        if lm_head is not None:
            return lm_head
    return None

def copy_weights(base_llm, model):
    config = model.config
    name = model.config._name_or_path.lower()
    if ('llama' in name) or ('zephyr' in name) or ('mistral' in name):
        print(f"Copying {name} first layer: {config.num_hidden_layers}")
        src_backbone = _resolve_backbone(base_llm)
        dst_backbone = _resolve_backbone(model)
        if src_backbone is None or dst_backbone is None:
            raise AttributeError(f"Cannot resolve model backbone for copy_weights: src={type(base_llm).__name__}, dst={type(model).__name__}")
        dst_backbone.embed_tokens.load_state_dict(
            src_backbone.embed_tokens.state_dict()
        )
        dst_backbone.norm.load_state_dict(
            src_backbone.norm.state_dict()
        )
        for layer_num in range(config.num_hidden_layers):
            dst_backbone.layers[layer_num].load_state_dict(
                src_backbone.layers[layer_num].state_dict()
            )
        src_lm_head = _resolve_lm_head(base_llm)
        dst_lm_head = _resolve_lm_head(model)
        if src_lm_head is None or dst_lm_head is None:
            raise AttributeError(f"Cannot resolve lm_head for copy_weights: src={type(base_llm).__name__}, dst={type(model).__name__}")
        dst_lm_head.load_state_dict(src_lm_head.state_dict())
        return model
    else:
        raise ValueError(f"Unsupported model: {name}")

def init_small_llm(origin_config, num_layer, device, hparams=None, base_llm=None, saved_path=None):
    config = copy.deepcopy(origin_config)
    config.num_hidden_layers = num_layer
    model = AutoModelForCausalLM.from_config(
        config,
        use_flash_attention_2=False, 
        torch_dtype=torch.bfloat16, 
    ).to('cuda')

    if base_llm is not None:
        copy_weights(base_llm, model)
        
    if saved_path is not None:
        model.load_state_dict(
            torch.load(saved_path)
        )

    return model

def save_pretrained_compat(model, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    try:
        model.save_pretrained(out_dir, safe_serialization=False)
        return
    except TypeError:
        # Older Transformers versions may not accept `safe_serialization`.
        try:
            model.save_pretrained(out_dir)
            return
        except ImportError as exc:
            if "DTensor" not in str(exc):
                raise
    except ImportError as exc:
        if "DTensor" not in str(exc):
            raise
    model.config.save_pretrained(out_dir)
    if hasattr(model, "generation_config") and model.generation_config is not None:
        try:
            model.generation_config.save_pretrained(out_dir)
        except Exception:
            pass
    try:
        from safetensors.torch import save_model as safetensors_save_model
        safetensors_save_model(model, os.path.join(out_dir, "model.safetensors"))
        bin_path = os.path.join(out_dir, "pytorch_model.bin")
        if os.path.exists(bin_path):
            os.remove(bin_path)
        return
    except Exception:
        pass
    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(state_dict, os.path.join(out_dir, "pytorch_model.bin"))

# def create_full_model(model_path,Lora, num_layer=0 ,data_type='bfloat16', **kwargs):
#     with NameTimer("Init full model"):
#         basellm = AutoModelForCausalLM.from_pretrained(
#             model_path, torch_dtype=get_dtype(data_type),
#             use_flash_attention_2=True, trust_remote_code=True,
#         )
#         if Lora.r != 0:
#             peftconfig = LoraConfig(
#                 r=Lora.r,
#                 lora_alpha=Lora.alpha,
#                 target_modules=find_all_linear_names(basellm), 
#                 lora_dropout=Lora.dropout,
#                 bias=Lora.bias, 
#                 task_type="CAUSAL_LM",
#             )
#             basellm = get_peft_model(basellm, peftconfig)

#         if num_layer != 0: #! Construct the small model
#             basellm = init_small_llm( 
#                 basellm.model.config,
#                 num_layer=num_layer,
#                 base_llm=basellm,
#                 device='cpu',
#             )
#         return basellm

def _check_pure_gpu_device_map(model, tag: str):
    """检查模型的 device_map 是否全部在 GPU 上（不能有 CPU/disk）"""
    hf_map = getattr(model, "hf_device_map", None)
    if hf_map is None:
        return
    bad = {}
    for k, v in hf_map.items():
        sv = str(v).lower()
        if "cpu" in sv or "disk" in sv:
            bad[k] = v
    if bad:
        raise RuntimeError(f"{tag}: found non-GPU placement: {bad}")
    print(f"[MP] {tag} device_map verified: all on GPU, {len(hf_map)} modules")

def create_full_model(
    model_path,
    Lora,
    num_layer=0,
    data_type='bfloat16',
    freeze_lora_a=False,
    attn_implementation=None,
    device_map=None,
    report_trainable_summary=True,
    **kwargs,
):
    print(f"Lora: {Lora}")
    print(f"freeze_lora_a: {freeze_lora_a}")
    print(f"device_map: {device_map}")
    print("=="*10)
    if os.environ.get("OFFICIAL_ULD_MODEL_UTILS", "0") == "1":
        with NameTimer("Init full model"):
            if attn_implementation is None:
                attn_implementation = os.environ.get("EVAL_ATTN_IMPL", "").strip() or "sdpa"
            load_kwargs = dict(
                torch_dtype=get_dtype(data_type),
                attn_implementation=attn_implementation,
                trust_remote_code=True,
            )
            if kwargs.get("local_files_only") is not None:
                load_kwargs["local_files_only"] = kwargs["local_files_only"]
            if device_map is not None:
                load_kwargs["device_map"] = device_map
                load_kwargs["low_cpu_mem_usage"] = kwargs.get("low_cpu_mem_usage", True)
            basellm = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
            if device_map is not None:
                _check_pure_gpu_device_map(basellm, "official_basellm")
            if num_layer != 0:
                basellm = init_small_llm(
                    basellm.model.config,
                    num_layer=num_layer,
                    base_llm=basellm,
                    device='cpu',
                )
            return basellm
    with NameTimer("Init full model"):
        if attn_implementation is None:
            # Prefer PyTorch SDPA kernels for speed without extra dependencies.
            attn_implementation = "sdpa"

        model_kwargs = {
            "torch_dtype": get_dtype(data_type),
            "attn_implementation": attn_implementation,
            "trust_remote_code": True,
        }

        # 如果指定了 device_map，使用模型并行
        if device_map is not None:
            model_kwargs["device_map"] = device_map
            model_kwargs["low_cpu_mem_usage"] = True
            print(f"[MP] Loading model with device_map={device_map}")

        basellm = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

        # 如果使用 device_map，检查是否纯GPU
        if device_map is not None:
            _check_pure_gpu_device_map(basellm, "train_model")
        else:
            # 单卡模式，移动到 cuda
            basellm = basellm.to('cuda')

        # 记录原始模型参数数量
        original_params = basellm.num_parameters()
        print(f"原始模型参数数量: {original_params:,}")

        if Lora.r != 0:
            # Convert ListConfig to list to avoid JSON serialization issues
            if hasattr(Lora, 'target_modules'):
                target_modules = list(Lora.target_modules) if isinstance(Lora.target_modules, ListConfig) else Lora.target_modules
            else:
                target_modules = find_all_linear_names(basellm)

            peftconfig = LoraConfig(
                r=Lora.r,
                lora_alpha=Lora.alpha,
                target_modules=target_modules,
                lora_dropout=Lora.dropout,
                bias=Lora.bias,
                task_type="CAUSAL_LM",
            )
            print(f"peftconfig: {peftconfig}")
            basellm = get_peft_model(basellm, peftconfig)
            print("已应用LoRA配置")

            # 如果需要冻结LoRA A矩阵，设置相应参数的requires_grad为False
            if freeze_lora_a:
                frozen_params = 0
                total_lora_params = 0
                for name, param in basellm.named_parameters():
                    if 'lora_A' in name:
                        param.requires_grad = False
                        frozen_params += param.numel()
                        print(f"冻结LoRA A矩阵参数: {name}")
                    if 'lora_' in name:
                        total_lora_params += param.numel()
                print(f"已冻结 {frozen_params:,} 个LoRA A矩阵参数，占LoRA参数总数的 {frozen_params/total_lora_params*100:.2f}%")
        else:
            print("未使用LoRA，将进行全参数微调")
        
        if report_trainable_summary:
            _summarize_trainable_parameters(basellm)
         
        if num_layer != 0: #! Construct the small model
            basellm = init_small_llm(                 
                basellm.model.config,
                num_layer=num_layer,
                base_llm=basellm,
                device='cuda',
            )
            
        return basellm

def create_peft_model(model_path, Lora, baseoutdir, num_layer=0, data_type='bfloat16', **kwargs):
    with NameTimer("Init peft model"):
        print("[PEFT] 先构建未挂载 LoRA 的基座模型，随后再附加 LoRA 适配器。")
        base_lora = copy.deepcopy(Lora)
        if isinstance(base_lora, dict):
            base_lora["r"] = 0
        else:
            if hasattr(base_lora, "r"):
                base_lora.r = 0
            try:
                base_lora["r"] = 0
            except Exception:
                pass
        basellm = create_full_model(
            model_path,
            base_lora,
            num_layer,
            data_type,
            report_trainable_summary=False,
            **kwargs,
        )
        if num_layer != 0:
            save_pretrained_compat(basellm, os.path.join(baseoutdir, 'fullmodel'))
        peftconfig = LoraConfig(
            r=Lora.r,
            lora_alpha=Lora.alpha,
            target_modules=find_all_linear_names(basellm), 
            lora_dropout=Lora.dropout,
            bias=Lora.bias, 
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(basellm, peftconfig)
        print(f"[PEFT] 已附加 LoRA 适配器: r={Lora.r}, alpha={Lora.alpha}, dropout={Lora.dropout}")
        _summarize_trainable_parameters(model, prefix="[PEFT] ")
        return model
