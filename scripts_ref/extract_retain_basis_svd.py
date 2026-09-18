#!/usr/bin/env python3
"""
按照 GPM 官方思路提取 retain 表示子空间。

当前脚本服务于表 3 的 matched 控变量实验：
- ToFU: 使用 retain split 的辅助模型激活
- WMDP: 使用 MMLU retain 提示的辅助模型激活
"""

import argparse
import json
import logging
import os
import pickle
from typing import Dict, List

import datasets
import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from uld.data.conv_util import create_template


TOFU_CONV_TEMPLATE = {
    "question_start_token": "question: ",
    "question_end_token": " answer:",
    "answer_token": "",
    "max_len": 200,
}

WMDP_CONV_TEMPLATE = {
    "question_start_token": "",
    "question_end_token": "",
    "answer_token": " ",
    "strip_prompt": False,
    "max_len": 512,
}


def load_local_tofu(split_name):
    local_tofu_path = os.environ.get("TOFU_DATA_NAME") or os.path.join(
        os.environ.get("CBD_DATA_ROOT", "data"), "TOFU"
    )
    json_file = os.path.join(local_tofu_path, f"{split_name}.json")
    if not os.path.exists(json_file):
        return load_dataset("locuslab/TOFU", split_name)["train"]

    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        data = []
        with open(json_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line.strip()))
    return datasets.Dataset.from_list(data)


def _read_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _format_mcq_prompt(subject: str, question: str, choices):
    subject = str(subject).replace("_", " ").strip()
    a, b, c, d = (list(choices) + ["", "", "", ""])[:4]
    return (
        f"The following are multiple choice questions (with answers) about {subject}.\n\n"
        f"{question}\n\n"
        f"A. {a}\n\n"
        f"B. {b}\n\n"
        f"C. {c}\n\n"
        f"D. {d}\n\n"
        f"Answer:"
    )


def _ans_letter(answer_idx: int) -> str:
    return ["A", "B", "C", "D"][int(answer_idx)]


def load_mmlu_mcq(jsonl_path: str, subjects_csv=None):
    rows = []
    subjects = None
    if subjects_csv:
        subjects = {s.strip() for s in subjects_csv.split(",") if s.strip()}
    for ex in _read_jsonl(jsonl_path):
        subject = ex.get("subject") or "general"
        if subjects is not None and subject not in subjects:
            continue
        prompt = _format_mcq_prompt(subject, ex["question"], ex["choices"])
        rows.append({"question": prompt, "answer": _ans_letter(ex["answer"])})
    return datasets.Dataset.from_list(rows)


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger(__name__)


def build_lora_model(base_model_name, r, alpha, dropout, target_modules):
    local_files_only = os.getenv("HF_HUB_OFFLINE") == "1" or os.getenv("TRANSFORMERS_OFFLINE") == "1"
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        local_files_only=local_files_only,
    )
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(base_model, lora_config)
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False
    return model


def prepare_item(tokenizer, conv_template, question, answer, max_len):
    item = {"question": question, "answer": answer}
    prefix_text = conv_template.prepare_gen_prompt(**item)
    full_text = conv_template.prepare_prompt(**item)
    inputs = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
    )
    input_ids = inputs["input_ids"][0]
    attention_mask = inputs["attention_mask"][0]

    mcq_mode = not bool(getattr(conv_template, "strip_prompt", True))
    if mcq_mode and isinstance(answer, str) and answer in {"A", "B", "C", "D"} and isinstance(question, str) and "Answer:" in question:
        labels = torch.full_like(input_ids, -100)
        nonpad = attention_mask.nonzero(as_tuple=False).flatten()
        if nonpad.numel() > 0:
            labels[int(nonpad[-1].item())] = input_ids[int(nonpad[-1].item())]
        return {"input_ids": input_ids.unsqueeze(0), "attention_mask": attention_mask.unsqueeze(0), "labels": labels.unsqueeze(0)}

    labels = input_ids.clone()
    prefix_num = len(tokenizer(prefix_text, truncation=True, max_length=max_len).input_ids)
    prefix_num = min(prefix_num, labels.size(0))
    labels[:prefix_num] = -100
    if (labels[1:] != -100).sum().item() == 0:
        return None
    return {"input_ids": input_ids.unsqueeze(0), "attention_mask": attention_mask.unsqueeze(0), "labels": labels.unsqueeze(0)}


def load_retain_dataset(args):
    if args.dataset == "tofu":
        return load_local_tofu(args.retain_split)
    if args.dataset == "wmdp_mmlu":
        return load_mmlu_mcq(args.mmlu_retain_file, args.mmlu_retain_subjects)
    raise ValueError(f"Unsupported dataset: {args.dataset}")


def extract_activations_svd(model, processed_data, device, max_samples=400):
    logger = logging.getLogger(__name__)
    model.eval()
    layer_activations: Dict[str, List[torch.Tensor]] = {}
    hooks = []

    def get_activation_hook(name):
        def hook(module, input_, output):
            if name not in layer_activations:
                layer_activations[name] = []
            activation = output[0] if isinstance(output, tuple) else output
            if activation.dim() == 3:
                pooled = activation[0].mean(dim=0)
            elif activation.dim() == 2:
                pooled = activation.mean(dim=0)
            else:
                return
            layer_activations[name].append(pooled.detach().to(dtype=torch.float32, device="cpu"))
        return hook

    found_layers = []
    for name, module in model.named_modules():
        if not name.endswith("up_proj"):
            continue
        found_layers.append(name)
        logger.info("注册 hook: %s", name)
        hooks.append(module.register_forward_hook(get_activation_hook(name)))

    if not found_layers:
        logger.error("未找到任何 up_proj 层")
        return {}

    with torch.no_grad():
        for batch in tqdm(processed_data[:max_samples], desc="提取激活表示"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

    for hook in hooks:
        hook.remove()

    activation_matrices = {}
    for layer_name, activations in layer_activations.items():
        if not activations:
            continue
        activation_stack = torch.stack(activations[:max_samples])
        activation_matrix = activation_stack.T.numpy()
        activation_matrices[layer_name] = activation_matrix
        logger.info("Layer %s: 激活矩阵形状 %s", layer_name, tuple(activation_matrix.shape))
    return activation_matrices


def compute_svd_basis(activation_matrices, threshold=0.95):
    logger = logging.getLogger(__name__)
    basis_info = {}

    for layer_name, activation_matrix in activation_matrices.items():
        logger.info("为层 %s 计算 SVD 基底...", layer_name)
        activation_matrix = activation_matrix.astype(np.float32)
        U, S, _ = np.linalg.svd(activation_matrix, full_matrices=False)

        sval_total = (S ** 2).sum()
        sval_ratio = (S ** 2) / max(sval_total, 1e-12)
        cumsum_ratio = np.cumsum(sval_ratio)

        r = int(np.sum(cumsum_ratio < threshold))
        if r == 0:
            r = 1
        basis_components = U[:, :r]
        explained = float(cumsum_ratio[r - 1])
        logger.info("Layer %s: 保留 %d 个主成分 (方差保留 %.4f)", layer_name, r, explained)

        basis_info[layer_name] = {
            "components": torch.from_numpy(basis_components.T),
            "explained_variance_ratio": explained,
            "n_components": r,
            "singular_values": S[:r],
            "total_samples": activation_matrix.shape[1],
        }

    return basis_info


def main():
    parser = argparse.ArgumentParser(description="使用激活 SVD 提取 GPM retain 基底")
    parser.add_argument("--model_path", type=str, default="", help="可选：LoRA 模型路径")
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="原始辅助模型名称",
    )
    parser.add_argument("--dataset", choices=["tofu", "wmdp_mmlu"], default="tofu")
    parser.add_argument("--retain_split", type=str, default="retain99", help="ToFU retain 数据分割名称")
    parser.add_argument("--mmlu_retain_file", type=str, default="eval-method/wmdp/data/mmlu/all_auxiliary_train.jsonl")
    parser.add_argument("--mmlu_retain_subjects", type=str, default=None)
    parser.add_argument("--target_variance", type=float, default=0.9)
    parser.add_argument("--max_samples", type=int, default=400)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--output_dir", type=str, default="./basis_gpm")

    args = parser.parse_args()
    logger = setup_logging()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("使用设备: %s", device)

    local_files_only = os.getenv("HF_HUB_OFFLINE") == "1" or os.getenv("TRANSFORMERS_OFFLINE") == "1"
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, local_files_only=local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.model_path and os.path.isdir(args.model_path) and os.path.exists(os.path.join(args.model_path, "adapter_config.json")):
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model_name,
            torch_dtype=torch.bfloat16,
            local_files_only=local_files_only,
        )
        model = PeftModel.from_pretrained(base_model, args.model_path, torch_dtype=torch.bfloat16)
    else:
        model = build_lora_model(
            args.base_model_name,
            args.lora_r,
            args.lora_alpha,
            args.lora_dropout,
            ["up_proj"],
        )
    model.to(device)
    model.eval()

    conv_template_cfg = TOFU_CONV_TEMPLATE if args.dataset == "tofu" else WMDP_CONV_TEMPLATE
    conv_template_cfg = dict(conv_template_cfg)
    conv_template_cfg["max_len"] = args.max_len
    conv_template = create_template(conv_template_cfg, tokenizer=tokenizer, max_len=args.max_len)

    dataset = load_retain_dataset(args)
    if len(dataset) > args.max_samples:
        dataset = dataset.select(range(args.max_samples))
    logger.info("数据集大小: %d", len(dataset))

    processed_data = []
    for idx in range(len(dataset)):
        example = dataset[idx]
        prepared = prepare_item(tokenizer, conv_template, example["question"], example["answer"], args.max_len)
        if prepared is not None:
            processed_data.append(prepared)

    logger.info("提取激活表示 (最多 %d 个样本)...", args.max_samples)
    activation_matrices = extract_activations_svd(model, processed_data, device, args.max_samples)
    if not activation_matrices:
        logger.error("未能提取到任何激活表示")
        return

    logger.info("计算 SVD 基底...")
    basis_info = compute_svd_basis(activation_matrices, args.target_variance)

    basis_file = os.path.join(args.output_dir, "gpm_retain_basis.pkl")
    with open(basis_file, "wb") as f:
        pickle.dump(basis_info, f)

    if args.dataset == "tofu":
        legacy_name = f"{args.retain_split}_svd_basis.pkl"
    else:
        legacy_name = "mmlu_retain_svd_basis.pkl"
    legacy_path = os.path.join(args.output_dir, legacy_name)
    if legacy_path != basis_file:
        with open(legacy_path, "wb") as f:
            pickle.dump(basis_info, f)

    config = {
        "model_path": args.model_path or args.base_model_name,
        "base_model_name": args.base_model_name,
        "dataset": args.dataset,
        "retain_split": args.retain_split,
        "mmlu_retain_file": args.mmlu_retain_file,
        "mmlu_retain_subjects": args.mmlu_retain_subjects,
        "target_variance": args.target_variance,
        "max_samples": args.max_samples,
        "max_len": args.max_len,
        "method": "GPM-SVD-activation",
        "conv_template": conv_template_cfg,
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": ["up_proj"],
        },
        "layers": list(basis_info.keys()),
        "basis_shapes": {k: list(v["components"].shape) for k, v in basis_info.items()},
        "explained_variance": {k: float(v["explained_variance_ratio"]) for k, v in basis_info.items()},
        "n_components": {k: int(v["n_components"]) for k, v in basis_info.items()},
    }

    config_file = os.path.join(args.output_dir, "basis_config.json")
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    logger.info("基底已保存到: %s", basis_file)
    if legacy_path != basis_file:
        logger.info("兼容旧路径副本: %s", legacy_path)
    logger.info("配置已保存到: %s", config_file)


if __name__ == "__main__":
    main()
