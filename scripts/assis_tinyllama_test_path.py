#!/usr/bin/env python
# -*- coding: utf-8 -*-
#! 这个脚本用于测试微调后的TinyLlama模型在TOFU数据集上的表现，并与原始未微调模型进行比较

import os
import sys
import torch
import json
import inspect
import argparse
import datetime
import numpy as np
from datasets import load_dataset
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from pathlib import Path
import torch.nn.functional as F
from peft import LoraConfig, PeftModel
from routing_score_reducers import (
    DEFAULT_ROUTING_REDUCER_ALPHA,
    DEFAULT_ROUTING_REDUCER_BETA,
    DEFAULT_ROUTING_REDUCER_GAMMA,
    DEFAULT_ROUTING_REDUCER_TOP_M,
    ROUTING_SCORE_SEMANTICS_VERSION,
    reduce_routing_scores,
    routing_entropy_from_logp,
    routing_reducer_params,
    routing_surprisal_from_actual_tokens,
)

import os


def load_peft_model_compat(base_model, adapter_path):
    try:
        return PeftModel.from_pretrained(base_model, adapter_path)
    except TypeError as exc:
        cfg_path = os.path.join(adapter_path, "adapter_config.json")
        with open(cfg_path, "r", encoding="utf-8") as f:
            raw_cfg = json.load(f)
        allowed = set(inspect.signature(LoraConfig.__init__).parameters.keys())
        filtered_cfg = {k: v for k, v in raw_cfg.items() if k in allowed}
        dropped = sorted(set(raw_cfg.keys()) - set(filtered_cfg.keys()))
        if dropped:
            print(f"[peft-load] drop unsupported adapter_config keys: {dropped} ({exc})")
        return PeftModel.from_pretrained(base_model, adapter_path, config=LoraConfig(**filtered_cfg))

# # 设置离线模式
# os.environ["HF_DATASETS_OFFLINE"] = "1"        # datasets离线
# os.environ["TRANSFORMERS_OFFLINE"] = "1"       # transformers离线  
# os.environ["HF_HUB_OFFLINE"] = "1"             # huggingface_hub离线

# 定义split映射
SPLIT_MAPPING = {
    0: "forget05_perturbed_1",      # 遗忘数据集
    1: "retain95",                  # 保留数据集
    2: "world_facts",               # 世界事实数据集
    3: "real_authors"               # 真实作者数据集
}

# 加载TOFU数据集
def load_tofu_dataset(dataset_name, split):
    # 如果split是数字字符串，转换为整数后使用映射
    if split.isdigit():
        split_idx = int(split)
        if split_idx in SPLIT_MAPPING:
            split = SPLIT_MAPPING[split_idx]
            print(f"使用split索引{split_idx}，对应分割: {split}")
        else:
            print(f"警告: split索引 {split_idx} 无效，使用原始值")
    
    print(f"加载数据集: {dataset_name}, 分割: {split}")

    # 尝试离线模式加载
    # import os
    # if os.environ.get('HF_DATASETS_OFFLINE') == '1':
    #     try:
    #         # 在离线模式下，尝试从缓存加载
    #         dataset = load_from_disk(dataset_name, split, download_mode='reuse_cache_if_exists')
    #     except Exception as e:
    #         print(f"离线模式加载失败: {e}")
    #         print("尝试直接从缓存目录加载...")
    #         # 如果离线模式失败，尝试直接指定缓存路径
    #         cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
    #         dataset = load_dataset(dataset_name, split, cache_dir=cache_dir, download_mode='reuse_cache_if_exists')
    # else:
    # 尝试从本地JSON文件加载TOFU数据
    import os
    import json
    from datasets import Dataset, DatasetDict
    
    # In the clean repo we do not copy datasets. Prefer the explicit dataset_name
    # path (or TOFU_DATA_NAME), then fall back to a local TOFU/ directory.
    local_tofu_candidates = []
    for candidate in (dataset_name, os.environ.get("TOFU_DATA_NAME"), "TOFU"):
        if candidate and candidate not in local_tofu_candidates:
            local_tofu_candidates.append(candidate)

    json_file = None
    for local_tofu_path in local_tofu_candidates:
        candidate_file = os.path.join(local_tofu_path, f"{split}.json")
        if os.path.exists(candidate_file):
            json_file = candidate_file
            break

    if json_file is not None:
        print(f"从本地加载TOFU数据: {json_file}")
        try:
            # 这些文件常见为 JSONL（每行一个 JSON），但扩展名仍为 .json；先 sniff 再解析
            with open(json_file, 'r', encoding='utf-8') as f:
                first_non_ws = ""
                while True:
                    ch = f.read(1)
                    if not ch:
                        break
                    if not ch.isspace():
                        first_non_ws = ch
                        break
                f.seek(0)

                if first_non_ws == "[":
                    data = json.load(f)
                    dataset = DatasetDict({"train": Dataset.from_list(data)})
                    print("✓ 成功从本地JSON文件加载数据集")
                else:
                    print("检测到JSONL格式（每行一个JSON对象），按行读取...")
                    data = []
                    for line in f:
                        if line.strip():
                            data.append(json.loads(line.strip()))
                    dataset = DatasetDict({"train": Dataset.from_list(data)})
                    print("✓ 成功从本地JSONL文件加载数据集")
        except Exception as e:
            print(f"本地TOFU文件加载失败: {e}")
            print("尝试从网络加载...")
            dataset = load_dataset(dataset_name, split)
    else:
        searched = ", ".join(os.path.join(p, f"{split}.json") for p in local_tofu_candidates)
        print(f"本地文件不存在（查找过: {searched}），尝试从网络加载...")
        dataset = load_dataset(dataset_name, split)

    # 限制数据集大小为300条（与TOFU官方实现一致）
    MAX_SAMPLES = 300
    original_size = len(dataset['train'])
    if original_size > MAX_SAMPLES:
        dataset['train'] = dataset['train'].select(range(MAX_SAMPLES))
        print(f"数据集大小: {len(dataset['train'])} (限制为前{MAX_SAMPLES}条，原始大小: {original_size})")
    else:
        print(f"数据集大小: {len(dataset['train'])}")

    print(f"数据集结构: {dataset['train'].column_names}")
    return dataset

# 格式化问题为模型输入格式
def format_prompt(question):
    return f"question: {question.strip()} answer:"
    #return f"[INST] {question} [/INST]"

# 计算两个模型输出的交叉熵
def compute_cross_entropy(
    finetuned_logits,
    original_logits,
    tokenizer,
    use_weighted_ce=False,
    use_length_factor=False,
    verbose=False,
):
    """
    计算两个模型输出的对称交叉熵，考虑序列长度差异

    Args:
        finetuned_logits: 微调模型的logits
        original_logits: 原始模型的logits
        tokenizer: 分词器
        use_weighted_ce: 是否使用加权交叉熵（CE(t)²/∑CE(t)），否则使用平均交叉熵（∑CE(t)/N）
        use_length_factor: 是否应用长度差异因子
    """
    # 确定哪个序列更长
    if verbose:
        print("logits原始形状")
        print(f"finetuned_logits.shape:{finetuned_logits.shape}")
        print(f"original_logits.shape:{original_logits.shape}")
    finetuned_length = finetuned_logits.size(0)
    original_length = original_logits.size(0)
    max_length = max(finetuned_length, original_length)

    # 计算长度差异因子
    length_diff = abs(finetuned_length - original_length)
    if use_length_factor:
        length_factor = 1.0 + 0.1*(length_diff / max(max_length, 1))
    else:
        length_factor = 1.0
    if verbose:
        print(f"长度差异因子: {length_factor} (use_length_factor: {use_length_factor})")
    
    a1_logits = finetuned_logits
    a2_logits = original_logits

    min_length = min(finetuned_logits.size(0), original_logits.size(0))

    # print(f'finetuned_logits.shape:{model_a_logits.shape}')
    # print(f'original_logits.shape:{model_b_logits.shape}')
    a1_logits = finetuned_logits[:min_length]
    a2_logits = original_logits[:min_length]
    if verbose:
        print("logits裁剪后形状")
        print(f"finetuned_logits.shape:{a1_logits.shape}")
        print(f"original_logits.shape:{a2_logits.shape}")
    # 获取每个位置的最大概率对应的 token id
    a1_ids = torch.argmax(a1_logits, dim=-1)
    a2_ids = torch.argmax(a2_logits, dim=-1)

    # 使用 tokenizer 解码
    a1_text = tokenizer.decode(a1_ids, skip_special_tokens=True)
    a2_text = tokenizer.decode(a2_ids, skip_special_tokens=True)

    if verbose:
        print("finetuned 解码结果：", a1_text)
        print("original 解码结果：", a2_text)
    # 将logits转换为概率分布
    a1_probs = F.softmax(a1_logits, dim=-1)
    a2_probs = F.softmax(a2_logits, dim=-1)
    a1_log_probs = F.log_softmax(a1_logits, dim=-1)
    a2_log_probs = F.log_softmax(a2_logits, dim=-1)

    # 计算两个方向的token交叉熵
    token_ce_a1_a2 = -(a1_probs * a2_log_probs).sum(dim=-1)
    token_ce_a2_a1 = -(a2_probs * a1_log_probs).sum(dim=-1)
    token_ce_sym = 0.5 * (token_ce_a1_a2 + token_ce_a2_a1)

    def aggregate_ce(token_ces):
        if use_weighted_ce:
            ce_squared = token_ces ** 2
            sum_ce = token_ces.sum()
            if sum_ce > 0:
                return (ce_squared.sum() / sum_ce).item()
            return 0.0
        return token_ces.mean().item()

    ce_a1_a2 = aggregate_ce(token_ce_a1_a2)
    ce_a2_a1 = aggregate_ce(token_ce_a2_a1)
    cross_entropy_value = 0.5 * (ce_a1_a2 + ce_a2_a1)
    if verbose:
        if use_weighted_ce:
            print("使用加权交叉熵: CE(t)²/∑CE(t)")
        else:
            print("使用平均交叉熵: ∑CE(t)/N")

    # 应用长度因子
    cross_entropy_value *= length_factor

    return {
        "cross_entropy": cross_entropy_value,
        "max_token_ce": token_ce_sym.max().item(),
        "min_token_ce": token_ce_sym.min().item(),
        "avg_token_ce": token_ce_sym.mean().item()
    }

# def compute_cross_entropy(model_a_logits, model_b_logits):
#     # print(f'finetuned_logits.shape:{model_a_logits.shape}')
#     # print(f'original_logits.shape:{model_b_logits.shape}')
#     a1_logits = model_a_logits
#     a2_logits = model_b_logits
#     # 将logits转换为概率分布
#     a1_probs = F.softmax(a1_logits, dim=-1)
#     a2_log_probs = F.log_softmax(a2_logits, dim=-1)

#     # 计算每个token位置的交叉熵
#     token_cross_entropies = -(a1_probs * a2_log_probs).sum(dim=-1)

#     # 使用CE(t)²/∑CE(t)的加权方案
#     ce_squared = token_cross_entropies ** 2
#     sum_ce = token_cross_entropies.sum()
#     if sum_ce > 0:  # 避免除以0
#         weighted_ce = (ce_squared.sum() / sum_ce).item()  # 确保转换为Python标量
#     else:
#         weighted_ce = 0.0
    
#     # 返回加权平均交叉熵值和一些调试信息 (确保所有值都是Python原生类型)
#     return {
#         "weighted_ce": weighted_ce,
#         "max_token_ce": token_cross_entropies.max().item(),
#         "min_token_ce": token_cross_entropies.min().item(),
#         "avg_token_ce": token_cross_entropies.mean().item()
#     }
# def compute_cross_entropy(model_a_logits, model_b_logits, gamma=2.0):
#     a1_probs = F.softmax(model_a_logits, dim=-1)
#     a2_log_probs = F.log_softmax(model_b_logits, dim=-1)
    
#     # 计算每个token位置的交叉熵
#     token_cross_entropies = -(a1_probs * a2_log_probs).sum(dim=-1)
    
#     # 使用指数加权，放大差异
#     exp_weights = torch.exp(gamma * token_cross_entropies)
#     weighted_ce = (token_cross_entropies * exp_weights).sum() / exp_weights.sum()
    
#     return {
#         "weighted_ce": weighted_ce.item(),
#         "max_token_ce": token_cross_entropies.max().item(),
#         "min_token_ce": token_cross_entropies.min().item(),
#         "avg_token_ce": token_cross_entropies.mean().item()
#     }

# 生成答案并获取logits
def generate_with_logits(model, inputs, tokenizer, max_new_tokens=20, stop_on_newline=False):
    """
    使用模型生成答案并返回生成token的logits
    """
    # 准备存储生成的token和logits
    # generated_tokens = inputs["input_ids"].clone()
    generated_tokens = inputs.clone()
    #print(f'generated_tokens.shape: {generated_tokens.shape}')
    all_logits = []
    
    # 逐token生成（使用 KV cache，加速生成）
    past_key_values = None
    for i in range(max_new_tokens):
        with torch.no_grad():
            if past_key_values is None:
                outputs = model(input_ids=generated_tokens, use_cache=True)
            else:
                outputs = model(
                    input_ids=generated_tokens[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            logits = outputs.logits[:, -1, :]
            past_key_values = getattr(outputs, "past_key_values", None)
            
            # 使用贪婪解码选择下一个token
            next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)

            #next_token_id = next_token.item()
            # next_token_text = tokenizer.decode([next_token_id], skip_special_tokens=False)
            # print(f'生成的token {i}: id={next_token_id}, 文本="{next_token_text}"')
            
            
            # 保存logits
            all_logits.append(logits)          

            
            # 添加新token到已生成序列
            generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)

            # 如果生成了结束标记，停止生成
            if next_token.item() == tokenizer.eos_token_id:
                break
            if stop_on_newline and tokenizer.decode([next_token.item()], skip_special_tokens=False) == "\n":
                break
    
    # 将所有logits堆叠成一个张量 [num_tokens, vocab_size]
    stacked_logits = torch.stack(all_logits, dim=1).squeeze(0) if all_logits else torch.tensor([])
    
    # 解码生成的文本
    generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    
    return generated_text, generated_tokens, stacked_logits


def _left_pad_tokenize(tokenizer, texts, device):
    # For batched generation in causal LM, left padding is more reliable.
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        max_len = getattr(tokenizer, "model_max_length", None)
        if not isinstance(max_len, int) or max_len <= 0 or max_len > 100000:
            max_len = 2048
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        )
    finally:
        tokenizer.padding_side = old_padding_side
    return {k: v.to(device) for k, v in enc.items()}


def _first_eos_lengths(gen_tokens: torch.LongTensor, eos_token_id: int) -> torch.LongTensor:
    """
    Args:
        gen_tokens: [bsz, steps] generated tokens (prompt excluded)
    Returns:
        lengths: [bsz] number of generated steps to include (include EOS step if present)
    """
    bsz, steps = gen_tokens.shape
    if steps == 0:
        return torch.zeros((bsz,), device=gen_tokens.device, dtype=torch.long)
    pos = torch.arange(steps, device=gen_tokens.device).unsqueeze(0).expand(bsz, steps)
    eos_mask = gen_tokens.eq(eos_token_id)
    eos_pos = torch.where(eos_mask, pos, torch.full_like(pos, steps))
    first = eos_pos.min(dim=1).values  # steps if none
    lengths = torch.where(first < steps, first + 1, torch.full_like(first, steps))
    return lengths.to(torch.long)


def _masked_reduce(token_scores: torch.Tensor, lengths: torch.LongTensor):
    """
    Args:
        token_scores: [bsz, steps]
        lengths: [bsz] valid steps per sample
    Returns:
        mean: [bsz], max: [bsz], min: [bsz]
    """
    bsz, steps = token_scores.shape
    if steps == 0:
        z = torch.zeros((bsz,), device=token_scores.device, dtype=token_scores.dtype)
        return z, z, z
    idx = torch.arange(steps, device=token_scores.device).unsqueeze(0)
    valid = idx < lengths.unsqueeze(1)
    denom = lengths.clamp(min=1).to(token_scores.dtype)

    masked_sum = token_scores.masked_fill(~valid, 0.0).sum(dim=1)
    mean = masked_sum / denom

    neg_inf = torch.finfo(token_scores.dtype).min
    pos_inf = torch.finfo(token_scores.dtype).max
    maxv = token_scores.masked_fill(~valid, neg_inf).max(dim=1).values
    minv = token_scores.masked_fill(~valid, pos_inf).min(dim=1).values

    zero = lengths.eq(0)
    if zero.any():
        z = torch.zeros_like(mean)
        mean = torch.where(zero, z, mean)
        maxv = torch.where(zero, z, maxv)
        minv = torch.where(zero, z, minv)

    return mean, maxv, minv


def _span_mean_max_score(token_scores: torch.Tensor, window: int) -> torch.Tensor:
    if token_scores.ndim != 1:
        raise ValueError(f"_span_mean_max_score expects 1D scores, got {tuple(token_scores.shape)}")
    if token_scores.numel() == 0:
        return torch.zeros((), device=token_scores.device, dtype=token_scores.dtype)
    win = max(1, min(int(window), int(token_scores.numel())))
    if win == 1:
        return token_scores.max()
    unfolded = token_scores.unfold(0, win, 1)
    return unfolded.mean(dim=-1).max()


def compute_fixed_path_kl_batch(
    original_model,
    finetuned_model,
    prompt_input_ids: torch.LongTensor,
    prompt_attention_mask: torch.LongTensor,
    tokenizer,
    max_new_tokens: int = 32,
    symmetric: bool = False,
    escort_alpha: float = DEFAULT_ROUTING_REDUCER_ALPHA,
    escort_beta: float = DEFAULT_ROUTING_REDUCER_BETA,
    sces_gamma: float = DEFAULT_ROUTING_REDUCER_GAMMA,
    sces_top_m: int = DEFAULT_ROUTING_REDUCER_TOP_M,
    span_window: int = 4,
):
    """
    Batched fixed-path KL:
      - Use original_model greedy path (generate) to obtain sequences + per-step logits
      - Compute finetuned logits on the same sequences in one forward pass
      - Return per-sample scores (mean over generated steps, include EOS step)
    """
    with torch.no_grad():
        gen_out = original_model.generate(
            input_ids=prompt_input_ids,
            attention_mask=prompt_attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    sequences = gen_out.sequences  # [bsz, prompt_len + steps]
    scores_list = list(getattr(gen_out, "scores", []) or [])
    steps = len(scores_list)

    if steps == 0:
        bsz = int(prompt_input_ids.size(0))
        empty = [
            {
                "cross_entropy": 0.0,
                "max_token_ce": 0.0,
                "min_token_ce": 0.0,
                "avg_token_ce": 0.0,
                "feis_score": 0.0,
                "cbd_weighted_kl": 0.0,
                "escort_score": 0.0,
            }
            for _ in range(bsz)
        ]
        texts = tokenizer.batch_decode(sequences, skip_special_tokens=True)
        return texts, sequences, empty

    orig_step_logits = torch.stack(scores_list, dim=1)  # [bsz, steps, vocab]

    prompt_len = int(prompt_input_ids.size(1))
    start = max(prompt_len - 1, 0)

    # Build an attention mask for the full sequence.
    # We treat all generated positions as valid tokens; later steps are masked out per-sample if EOS appears.
    gen_mask = torch.ones(
        (prompt_attention_mask.size(0), sequences.size(1) - prompt_len),
        device=prompt_attention_mask.device,
        dtype=prompt_attention_mask.dtype,
    )
    full_attention_mask = torch.cat([prompt_attention_mask, gen_mask], dim=1)

    with torch.no_grad():
        ft_out = finetuned_model(input_ids=sequences, attention_mask=full_attention_mask)
        ft_logits_full = ft_out.logits  # [bsz, seq_len, vocab]

    ft_step_logits = ft_logits_full[:, start : start + steps, :]  # [bsz, steps, vocab]

    orig_logp = F.log_softmax(orig_step_logits, dim=-1)
    ft_logp = F.log_softmax(ft_step_logits, dim=-1)
    orig_p = orig_logp.exp()

    kl_of = (orig_p * (orig_logp - ft_logp)).sum(dim=-1)  # [bsz, steps]
    if symmetric:
        ft_p = ft_logp.exp()
        kl_fo = (ft_p * (ft_logp - orig_logp)).sum(dim=-1)
        token_scores = 0.5 * (kl_of + kl_fo)
    else:
        token_scores = kl_of

    orig_entropy = routing_entropy_from_logp(orig_logp, probs=orig_p)  # [bsz, steps]

    # Get the actual token indices to compute surprisal at each step
    # sequences shape: [bsz, prompt_len + steps]
    gen_tokens = sequences[:, prompt_len:]  # [bsz, steps]

    # Gather log-probs of the actually generated tokens
    # orig_logp shape: [bsz, steps, vocab]
    orig_surprisal = routing_surprisal_from_actual_tokens(orig_logp, gen_tokens)  # [bsz, steps]

    lengths = _first_eos_lengths(gen_tokens, tokenizer.eos_token_id)
    mean, maxv, minv = _masked_reduce(token_scores, lengths)

    # Compute FEIS and CBD-Weighted scores per sample
    details = []
    for i in range(int(prompt_input_ids.size(0))):
        cur_len = int(lengths[i].item())
        if cur_len > 0:
            s_i = orig_surprisal[i, :cur_len]
            h_i = orig_entropy[i, :cur_len]
            kl_i = token_scores[i, :cur_len]

            feis_val = reduce_routing_scores(kl_i, "fsis", entropy=h_i, surprisal=s_i).item()
            cbd_val = reduce_routing_scores(kl_i, "cbd").item()
            escort_val = reduce_routing_scores(
                kl_i,
                "escort",
                entropy=h_i,
                surprisal=s_i,
                alpha=escort_alpha,
                beta=escort_beta,
            ).item()
            sces_val = reduce_routing_scores(
                kl_i,
                "sces",
                entropy=h_i,
                surprisal=s_i,
                gamma=sces_gamma,
                top_m=sces_top_m,
            ).item()
            span_val = _span_mean_max_score(kl_i, span_window).item()
        else:
            feis_val = 0.0
            cbd_val = 0.0
            escort_val = 0.0
            sces_val = 0.0
            span_val = 0.0

        details.append(
            {
                "cross_entropy": float(mean[i].item()),
                "max_token_ce": float(maxv[i].item()),
                "min_token_ce": float(minv[i].item()),
                "avg_token_ce": float(mean[i].item()),
                "feis_score": float(feis_val),
                "cbd_weighted_kl": float(cbd_val),
                "escort_score": float(escort_val),
                "sces_score": float(sces_val),
                "span_score": float(span_val),
            }
        )

    texts = tokenizer.batch_decode(sequences, skip_special_tokens=True)
    return texts, sequences, details


def compute_fixed_path_symmetric_kl(
    original_model,
    finetuned_model,
    prompt_input_ids,
    tokenizer,
    max_new_tokens=32,
    escort_alpha: float = DEFAULT_ROUTING_REDUCER_ALPHA,
    escort_beta: float = DEFAULT_ROUTING_REDUCER_BETA,
    sces_gamma: float = DEFAULT_ROUTING_REDUCER_GAMMA,
    sces_top_m: int = DEFAULT_ROUTING_REDUCER_TOP_M,
    span_window: int = 4,
):
    """
    固定路径 token-wise 对称 KL：
    1) 用 original_model greedy 生成 token 序列
    2) 在同一前缀路径上比较两模型下一 token 分布
    """
    prompt_len = prompt_input_ids.size(1)
    original_text, full_tokens, original_step_logits = generate_with_logits(
        original_model,
        prompt_input_ids,
        tokenizer,
        max_new_tokens=max_new_tokens,
        stop_on_newline=False,
    )

    gen_len = full_tokens.size(1) - prompt_len
    if gen_len <= 0 or original_step_logits.numel() == 0:
        return original_text, full_tokens, {
            "cross_entropy": 0.0,
            "max_token_ce": 0.0,
            "min_token_ce": 0.0,
            "avg_token_ce": 0.0,
            "feis_score": 0.0,
            "cbd_weighted_kl": 0.0,
            "escort_score": 0.0,
            "sces_score": 0.0,
            "span_score": 0.0,
        }

    # finetuned logits on the same fixed path via one forward pass
    with torch.no_grad():
        ft_out = finetuned_model(input_ids=full_tokens)
        ft_logits_full = ft_out.logits[0]  # [seq_len, vocab]

    start = max(prompt_len - 1, 0)
    end = start + gen_len
    ft_step_logits = ft_logits_full[start:end, :]  # [gen_len, vocab]
    orig_step_logits = original_step_logits[:gen_len]  # [gen_len, vocab]

    orig_logp = F.log_softmax(orig_step_logits, dim=-1)
    ft_logp = F.log_softmax(ft_step_logits, dim=-1)
    orig_p = orig_logp.exp()
    ft_p = ft_logp.exp()

    kl_of = (orig_p * (orig_logp - ft_logp)).sum(dim=-1)
    kl_fo = (ft_p * (ft_logp - orig_logp)).sum(dim=-1)
    sym_kl = 0.5 * (kl_of + kl_fo)
    gen_tokens = full_tokens[0, prompt_len:]
    orig_entropy = routing_entropy_from_logp(orig_logp, probs=orig_p)
    orig_surprisal = routing_surprisal_from_actual_tokens(orig_logp, gen_tokens)

    return original_text, full_tokens, {
        # keep key name for backward compatibility with analysis script
        "cross_entropy": sym_kl.mean().item(),
        "max_token_ce": sym_kl.max().item(),
        "min_token_ce": sym_kl.min().item(),
        "avg_token_ce": sym_kl.mean().item(),
        "feis_score": reduce_routing_scores(sym_kl, "fsis", entropy=orig_entropy, surprisal=orig_surprisal).item(),
        "cbd_weighted_kl": reduce_routing_scores(sym_kl, "cbd").item(),
        "escort_score": reduce_routing_scores(
            sym_kl,
            "escort",
            entropy=orig_entropy,
            surprisal=orig_surprisal,
            alpha=escort_alpha,
            beta=escort_beta,
        ).item(),
        "sces_score": reduce_routing_scores(
            sym_kl,
            "sces",
            entropy=orig_entropy,
            surprisal=orig_surprisal,
            gamma=sces_gamma,
            top_m=sces_top_m,
        ).item(),
        "span_score": _span_mean_max_score(sym_kl, span_window).item(),
    }


def compute_fixed_path_kl(
    original_model,
    finetuned_model,
    prompt_input_ids,
    tokenizer,
    max_new_tokens=32,
    escort_alpha: float = DEFAULT_ROUTING_REDUCER_ALPHA,
    escort_beta: float = DEFAULT_ROUTING_REDUCER_BETA,
    sces_gamma: float = DEFAULT_ROUTING_REDUCER_GAMMA,
    sces_top_m: int = DEFAULT_ROUTING_REDUCER_TOP_M,
    span_window: int = 4,
):
    """
    固定路径 token-wise KL (非对称):
    1) 用 original_model greedy 生成 token 序列
    2) 在同一前缀路径上比较两模型下一 token 分布
    3) 计算 KL(p0 || p1)
    """
    prompt_len = prompt_input_ids.size(1)
    original_text, full_tokens, original_step_logits = generate_with_logits(
        original_model,
        prompt_input_ids,
        tokenizer,
        max_new_tokens=max_new_tokens,
        stop_on_newline=False,
    )

    gen_len = full_tokens.size(1) - prompt_len
    if gen_len <= 0 or original_step_logits.numel() == 0:
        return original_text, full_tokens, {
            "cross_entropy": 0.0,
            "max_token_ce": 0.0,
            "min_token_ce": 0.0,
            "avg_token_ce": 0.0,
            "feis_score": 0.0,
            "cbd_weighted_kl": 0.0,
            "escort_score": 0.0,
            "sces_score": 0.0,
            "span_score": 0.0,
        }

    # finetuned logits on the same fixed path via one forward pass
    with torch.no_grad():
        ft_out = finetuned_model(input_ids=full_tokens)
        ft_logits_full = ft_out.logits[0]  # [seq_len, vocab]

    start = max(prompt_len - 1, 0)
    end = start + gen_len
    ft_step_logits = ft_logits_full[start:end, :]  # [gen_len, vocab]
    orig_step_logits = original_step_logits[:gen_len]  # [gen_len, vocab]

    orig_logp = F.log_softmax(orig_step_logits, dim=-1)
    ft_logp = F.log_softmax(ft_step_logits, dim=-1)
    orig_p = orig_logp.exp()

    kl_of = (orig_p * (orig_logp - ft_logp)).sum(dim=-1)
    gen_tokens = full_tokens[0, prompt_len:]
    orig_entropy = routing_entropy_from_logp(orig_logp, probs=orig_p)
    orig_surprisal = routing_surprisal_from_actual_tokens(orig_logp, gen_tokens)

    return original_text, full_tokens, {
        "cross_entropy": kl_of.mean().item(),
        "max_token_ce": kl_of.max().item(),
        "min_token_ce": kl_of.min().item(),
        "avg_token_ce": kl_of.mean().item(),
        "feis_score": reduce_routing_scores(kl_of, "fsis", entropy=orig_entropy, surprisal=orig_surprisal).item(),
        "cbd_weighted_kl": reduce_routing_scores(kl_of, "cbd").item(),
        "escort_score": reduce_routing_scores(
            kl_of,
            "escort",
            entropy=orig_entropy,
            surprisal=orig_surprisal,
            alpha=escort_alpha,
            beta=escort_beta,
        ).item(),
        "sces_score": reduce_routing_scores(
            kl_of,
            "sces",
            entropy=orig_entropy,
            surprisal=orig_surprisal,
            gamma=sces_gamma,
            top_m=sces_top_m,
        ).item(),
        "span_score": _span_mean_max_score(kl_of, span_window).item(),
    }

def main():
    parser = argparse.ArgumentParser(description="测试微调后的TinyLlama模型并与原始模型比较")
    parser.add_argument("--model_path", type=str,
                      default="artifacts/outputs_trained_models/2025-07-16T20-44-25/checkpoint-600",
                      help="微调模型路径")
    parser.add_argument("--pretrained_model_name", type=str, 
                      default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                      help="原始预训练模型名称")
    parser.add_argument("--dataset_name", type=str, default="locuslab/TOFU", help="数据集名称")
    parser.add_argument("--dataset_split", type=str, default="all",
                     help="数据集分割 (0=forget05_perturbed_1, 1=retain95, 2=world_facts, 3=real_authors, all=所有数据集)")
    parser.add_argument("--forget_split", type=str, default="forget05_perturbed_1",
                     help="要测试的forget数据分割 (forget05_perturbed_1 到 forget05_perturbed_5)")
    parser.add_argument("--output_file", type=str, default="tinyllama_comparison_results.json", help="输出结果文件")
    parser.add_argument("--output_dir", type=str, default="ce_results/freeze_gd+kl_600_0.8base", help="结果文件保存目录")
    parser.add_argument("--use_weighted_ce", type=str, default="False", help="是否使用加权交叉熵（CE(t)²/∑CE(t)）")
    parser.add_argument("--use_length_factor", type=str, default="False", help="是否应用长度差异因子")
    parser.add_argument("--metric", type=str, default="fixed_sym_kl",
                      choices=["greedy_sym_ce", "fixed_sym_kl", "fixed_kl"],
                      help="打分方式：greedy_sym_ce=两模型各自生成后做对称CE；fixed_sym_kl=固定original路径做对称KL；fixed_kl=固定original路径KL(p0||p1)")
    parser.add_argument("--verbose", type=str, default="False", help="是否打印每条样本的详细调试信息")
    parser.add_argument("--batch_size", type=int, default=1, help="批处理大小")
    parser.add_argument("--max_new_tokens", type=int, default=20, help="生成的最大新token数")
    parser.add_argument("--max_samples", type=int, default=300, help="最多评测多少条样本（默认300，与TOFU官方一致）")
    parser.add_argument("--gpu_id", type=str, default="", help="使用的GPU ID（留空则沿用外部CUDA_VISIBLE_DEVICES）")
    parser.add_argument("--question_key", type=str, default="question", help="数据集里用作输入问题的字段名")
    parser.add_argument("--answer_key", type=str, default="answer", help="数据集里用作标准答案的字段名")
    parser.add_argument("--score_reducer_alpha", type=float, default=DEFAULT_ROUTING_REDUCER_ALPHA,
                      help="escort 聚合的 alpha 参数")
    parser.add_argument("--score_reducer_beta", type=float, default=DEFAULT_ROUTING_REDUCER_BETA,
                      help="escort 聚合的 beta 参数")
    parser.add_argument("--sces_gamma", type=float, default=DEFAULT_ROUTING_REDUCER_GAMMA,
                      help="SCES 聚合的 gamma 参数")
    parser.add_argument("--sces_top_m", type=int, default=DEFAULT_ROUTING_REDUCER_TOP_M,
                      help="SCES 聚合保留的 token 个数")
    parser.add_argument("--span_window", type=int, default=4,
                      help="span 聚合使用的连续窗口大小")
    
    args = parser.parse_args()

    # 将字符串参数转换为布尔值
    args.use_weighted_ce = args.use_weighted_ce.lower() == 'true'
    args.use_length_factor = args.use_length_factor.lower() == 'true'
    args.verbose = args.verbose.lower() == 'true'

    # 打印打分参数
    print("="*60)
    print("打分配置:")
    print(f"  metric: {args.metric}")
    print(f"  max_new_tokens: {args.max_new_tokens}")
    print(f"  verbose: {args.verbose}")
    print(f"  use_weighted_ce (仅greedy_sym_ce有效): {args.use_weighted_ce}")
    print(f"  use_length_factor (仅greedy_sym_ce有效): {args.use_length_factor}")
    print("="*60)

    # 设置GPU（如需要覆盖外部CUDA_VISIBLE_DEVICES）
    if args.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"结果文件将保存到目录: {args.output_dir}")

    # 确定要测试的数据集split列表
    if args.dataset_split == "all":
        splits_to_test = list(SPLIT_MAPPING.keys())
        print(f"将测试所有数据集: {[SPLIT_MAPPING[i] for i in splits_to_test]}")
    else:
        raw_splits = [item.strip() for item in str(args.dataset_split).split(",") if item.strip()]
        if not raw_splits:
            print("错误: dataset_split 为空")
            return

        splits_to_test = []
        for raw_split in raw_splits:
            if raw_split.isdigit():
                split_idx = int(raw_split)
                if split_idx in SPLIT_MAPPING:
                    splits_to_test.append(split_idx)
                else:
                    print(f"错误: 无效的split索引 {split_idx}")
                    return
            else:
                splits_to_test.append(raw_split)
        print(f"将测试数据集: {splits_to_test}")

    # 更新SPLIT_MAPPING中的forget数据集
    SPLIT_MAPPING[0] = args.forget_split
    print(f"使用forget数据分割: {args.forget_split}")

    # 加载tokenizer（只需要加载一次）
    print(f"从原始预训练模型加载tokenizer: {args.pretrained_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name,local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载微调后的模型
    print(f"加载微调后的模型: {args.model_path}")
    
    # 检查是否是LoRA模型（通过路径名或adapter_config.json文件）
    adapter_config_path = os.path.join(args.model_path, "adapter_config.json")
    is_lora_model = "lora" in args.model_path.lower() or os.path.exists(adapter_config_path)
    
    if is_lora_model:
        print("加载lora模型")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.pretrained_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True
        )
        finetuned_model = load_peft_model_compat(base_model, args.model_path)
        finetuned_model = finetuned_model.merge_and_unload()
    else:
        print("加载非lora模型")
        finetuned_model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True
        )
    finetuned_model.eval()
    
    # 加载原始预训练模型
    print(f"加载原始预训练模型: {args.pretrained_model_name}")
    original_model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True
    )
    original_model.eval()

    # 循环测试每个数据集
    for split_to_test in splits_to_test:
        print(f"\n{'='*60}")
        if isinstance(split_to_test, int):
            split_name = SPLIT_MAPPING[split_to_test]
            print(f"开始测试数据集 {split_to_test}: {split_name}")
        else:
            split_name = split_to_test
            print(f"开始测试数据集: {split_name}")
        print(f"{'='*60}")

        # 加载数据集
        dataset = load_tofu_dataset(args.dataset_name, str(split_to_test))
        max_samples = min(int(args.max_samples), len(dataset["train"]))
        if max_samples < len(dataset["train"]):
            dataset["train"] = dataset["train"].select(range(max_samples))
            print(f"按参数 --max_samples 限制样本数: {len(dataset['train'])}")

        # 更新输出文件路径，包含split信息和目录
        output_path = Path(args.output_file)
        detailed_results_file = os.path.join(args.output_dir, f"{output_path.stem}_{split_name}{output_path.suffix}")

        # 初始化结果列表
        results = []

        # 对每个问题进行测试
        print("开始测试...")
        examples = list(dataset["train"])
        total = len(examples)
        batch_size = max(int(args.batch_size), 1)
        print(f"Batch size {batch_size}")

        for start_idx in tqdm(range(0, total, batch_size)):
            batch = examples[start_idx : start_idx + batch_size]
            questions = [ex[args.question_key] for ex in batch]
            answers = [ex[args.answer_key] for ex in batch]
            prompts = [format_prompt(q) for q in questions]

            if args.metric in ("fixed_sym_kl", "fixed_kl"):
                enc = _left_pad_tokenize(tokenizer, prompts, device=device)
                prompt_ids = enc["input_ids"]
                prompt_mask = enc.get("attention_mask", torch.ones_like(prompt_ids))
                original_texts, _, score_details_list = compute_fixed_path_kl_batch(
                    original_model=original_model,
                    finetuned_model=finetuned_model,
                    prompt_input_ids=prompt_ids,
                    prompt_attention_mask=prompt_mask,
                    tokenizer=tokenizer,
                    max_new_tokens=int(args.max_new_tokens),
                    symmetric=(args.metric == "fixed_sym_kl"),
                    escort_alpha=float(args.score_reducer_alpha),
                    escort_beta=float(args.score_reducer_beta),
                    sces_gamma=float(args.sces_gamma),
                    sces_top_m=int(args.sces_top_m),
                    span_window=int(args.span_window),
                )

                # fixed-path: we skip finetuned generation (same as previous code path)
                finetuned_texts = prompts
            else:
                # greedy 方式对称CE：逐条计算（保持旧行为）
                original_texts = []
                finetuned_texts = []
                score_details_list = []
                for prompt in prompts:
                    inputs = tokenizer(prompt, return_tensors="pt").to(device)
                    input_ids = inputs["input_ids"]
                    finetuned_text, _, finetuned_logits = generate_with_logits(
                        finetuned_model,
                        input_ids,
                        tokenizer,
                        max_new_tokens=args.max_new_tokens,
                        stop_on_newline=False,
                    )
                    original_text, _, original_logits = generate_with_logits(
                        original_model,
                        input_ids,
                        tokenizer,
                        max_new_tokens=args.max_new_tokens,
                        stop_on_newline=False,
                    )
                    score_details = compute_cross_entropy(
                        finetuned_logits,
                        original_logits,
                        tokenizer,
                        use_weighted_ce=args.use_weighted_ce,
                        use_length_factor=args.use_length_factor,
                        verbose=args.verbose,
                    )
                    original_texts.append(original_text)
                    finetuned_texts.append(finetuned_text)
                    score_details_list.append(score_details)

            for j in range(len(batch)):
                i = start_idx + j
                question = questions[j]
                original_answer = answers[j]
                prompt = prompts[j]
                original_text = original_texts[j]
                finetuned_text = finetuned_texts[j]
                score_details = score_details_list[j]

                if args.verbose:
                    print(f"\n问题 {i+1}: {question}")

                # 提取生成的答案（去除提示部分）
                finetuned_answer = finetuned_text[len(prompt):].strip()
                original_answer_generated = original_text[len(prompt):].strip()

                cross_entropy = score_details["cross_entropy"]

                if args.verbose:
                    print(f"原始数据集答案: {original_answer}")
                    print(f"微调模型回答: {finetuned_answer}")
                    print(f"原始模型回答: {original_answer_generated}")
                    if cross_entropy is not None:
                        print(f"score: {cross_entropy:.4f}")
                        print(f"最大token差异: {score_details['max_token_ce']:.4f}")
                        print(f"最小token差异: {score_details['min_token_ce']:.4f}")
                        print(f"平均token差异: {score_details['avg_token_ce']:.4f}")
                    print("="*50)

                results.append({
                    "id": i,
                    "question": question,
                    "dataset_answer": original_answer,
                    "finetuned_model_answer": finetuned_answer,
                    "original_model_answer": original_answer_generated,
                    "cross_entropy": cross_entropy,
                    "max_token_ce": score_details["max_token_ce"] if score_details else None,
                    "feis_score": score_details.get("feis_score") if score_details else None,
                    "cbd_weighted_kl": score_details.get("cbd_weighted_kl") if score_details else None,
                    "escort_score": score_details.get("escort_score") if score_details else None,
                    "sces_score": score_details.get("sces_score") if score_details else None,
                    "span_score": score_details.get("span_score") if score_details else None,
                    "cross_entropy_details": {
                        "metric": args.metric,
                        "routing_score_semantics_version": ROUTING_SCORE_SEMANTICS_VERSION,
                        "escort_params": routing_reducer_params(
                            "escort",
                            alpha=float(args.score_reducer_alpha),
                            beta=float(args.score_reducer_beta),
                        ),
                        "sces_params": routing_reducer_params(
                            "sces",
                            gamma=float(args.sces_gamma),
                            top_m=int(args.sces_top_m),
                        ),
                        "span_params": {
                            "window": int(args.span_window),
                        },
                        "max_token_ce": score_details["max_token_ce"] if score_details else None,
                        "min_token_ce": score_details["min_token_ce"] if score_details else None,
                        "avg_token_ce": score_details["avg_token_ce"] if score_details else None,
                        "feis_score": score_details.get("feis_score") if score_details else None,
                        "cbd_weighted_kl": score_details.get("cbd_weighted_kl") if score_details else None,
                        "escort_score": score_details.get("escort_score") if score_details else None,
                        "sces_score": score_details.get("sces_score") if score_details else None,
                        "span_score": score_details.get("span_score") if score_details else None,
                    }
                })

                # 每10个问题保存一次结果
                if (i + 1) % 10 == 0 or (i + 1) == total:
                    try:
                        with open(detailed_results_file, 'w', encoding='utf-8') as f:
                            json.dump(results, f, ensure_ascii=False, indent=2)
                        print(f"已保存中间结果到 {detailed_results_file}，完成 {i+1}/{total} 个问题")
                    except TypeError as e:
                        print(f"保存结果时出错: {e}")
                        for idx, item in enumerate(results):
                            try:
                                json.dumps(item)
                            except TypeError:
                                print(f"问题出现在结果 {idx}")
                                for k, v in item.items():
                                    try:
                                        json.dumps({k: v})
                                    except TypeError:
                                        print(f"  问题键: {k}, 类型: {type(v)}")

        # 保存完整结果（带时间戳）
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # output_path = Path(args.output_file)
        # final_output_file = f"{output_path.stem}_{timestamp}{output_path.suffix}"

        # with open(final_output_file, 'w', encoding='utf-8') as f:
        #     json.dump(results, f, ensure_ascii=False, indent=2)

        # print(f"\n测试完成! 结果已保存到 {final_output_file}")

        # 评估指标计算 - 微调模型
        finetuned_exact_match = 0
        for item in results:
            if item["dataset_answer"].strip() == item["finetuned_model_answer"].strip():
                finetuned_exact_match += 1

        finetuned_exact_match_rate = finetuned_exact_match / len(results) * 100

        # 评估指标计算 - 原始模型
        original_exact_match = 0
        for item in results:
            if item["dataset_answer"].strip() == item["original_model_answer"].strip():
                original_exact_match += 1

        original_exact_match_rate = original_exact_match / len(results) * 100

        # 计算平均交叉熵
        cross_entropies = [item["cross_entropy"] for item in results if item["cross_entropy"] is not None]
        avg_cross_entropy = sum(cross_entropies) / len(cross_entropies) if cross_entropies else None

        # 计算更多交叉熵统计指标
        ce_details = {}
        if cross_entropies:
            ce_details = {
                "max_weighted_ce": max(cross_entropies),
                "min_weighted_ce": min(cross_entropies),
                "std_weighted_ce": np.std(cross_entropies)
            }

            # 计算所有问题中最大和最小token交叉熵
            max_token_ces = [item["cross_entropy_details"]["max_token_ce"] for item in results if item["cross_entropy_details"]["max_token_ce"] is not None]
            min_token_ces = [item["cross_entropy_details"]["min_token_ce"] for item in results if item["cross_entropy_details"]["min_token_ce"] is not None]
            avg_token_ces = [item["cross_entropy_details"]["avg_token_ce"] for item in results if item["cross_entropy_details"]["avg_token_ce"] is not None]

            if max_token_ces:
                ce_details.update({
                    "global_max_token_ce": max(max_token_ces),
                    "global_min_token_ce": min(min_token_ces),
                    "avg_max_token_ce": sum(max_token_ces) / len(max_token_ces),
                    "avg_min_token_ce": sum(min_token_ces) / len(min_token_ces),
                    "avg_avg_token_ce": sum(avg_token_ces) / len(avg_token_ces)
                })

        # 创建一个包含评估结果的文件
        eval_results = {
            "total_questions": len(results),
            "finetuned_exact_matches": finetuned_exact_match,
            "finetuned_exact_match_rate": finetuned_exact_match_rate,
            "original_exact_matches": original_exact_match,
            "original_exact_match_rate": original_exact_match_rate,
            "average_weighted_cross_entropy": avg_cross_entropy,
            "cross_entropy_statistics": ce_details,
            "available_routing_score_keys": [
                "cross_entropy",
                "max_token_ce",
                "feis_score",
                "cbd_weighted_kl",
                "escort_score",
                "sces_score",
                "span_score",
            ],
            "routing_score_semantics_version": ROUTING_SCORE_SEMANTICS_VERSION,
            "escort_params": routing_reducer_params(
                "escort",
                alpha=float(args.score_reducer_alpha),
                beta=float(args.score_reducer_beta),
            ),
            "sces_params": routing_reducer_params(
                "sces",
                gamma=float(args.sces_gamma),
                top_m=int(args.sces_top_m),
            ),
            "span_params": {
                "window": int(args.span_window),
            },
            "finetuned_model_path": args.model_path,
            "original_model_path": args.pretrained_model_name,
            "dataset": f"{args.dataset_name}/{split_name}",
            "timestamp": timestamp
        }

        eval_output_file = os.path.join(args.output_dir, f"evaluation_comparison_results_{split_name}_{timestamp}.json")
        with open(eval_output_file, 'w', encoding='utf-8') as f:
            json.dump(eval_results, f, ensure_ascii=False, indent=2)

        print(f"评估结果已保存到 {eval_output_file}")
        print(f"微调模型精确匹配率: {finetuned_exact_match_rate:.2f}%")
        print(f"原始模型精确匹配率: {original_exact_match_rate:.2f}%")
        if avg_cross_entropy is not None:
            print(f"平均交叉熵: {avg_cross_entropy:.4f}")
            if ce_details:
                print(f"最大交叉熵: {ce_details['max_weighted_ce']:.4f}")
                print(f"最小交叉熵: {ce_details['min_weighted_ce']:.4f}")
                print(f"交叉熵标准差: {ce_details['std_weighted_ce']:.4f}")
                if 'global_max_token_ce' in ce_details:
                    print(f"所有问题中最大token交叉熵: {ce_details['global_max_token_ce']:.4f}")
                    print(f"所有问题中最小token交叉熵: {ce_details['global_min_token_ce']:.4f}")
                    print(f"每个问题的平均token交叉熵: {ce_details['avg_avg_token_ce']:.4f}")

if __name__ == "__main__":
    main()
