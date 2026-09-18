import os
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
         
            
        return basellm

