#!/usr/bin/env python3
"""
Select routing threshold for WMDP vs MMLU using symmetric KL between:
  - original assistant A0
  - finetuned assistant A1

For MCQ prompts ending with "Answer:", the score is computed at the next-token
position (equivalent to fixed_sym_kl when max_new_tokens=1).
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from routing_score_reducers import (
    DEFAULT_ROUTING_REDUCER_ALPHA,
    DEFAULT_ROUTING_REDUCER_BETA,
    ROUTING_SCORE_SEMANTICS_VERSION,
    normalize_routing_reducer,
    reduce_routing_scores,
    routing_entropy_from_logp,
    routing_reducer_metadata,
    routing_surprisal_from_argmax,
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_root() -> Path:
    return Path(os.environ.get("CBD_DATA_ROOT", "data")).resolve()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def format_mcq_prompt(subject: str, question: str, choices: List[str]) -> str:
    subject = str(subject).replace("_", " ").strip()
    a, b, c, d = (choices + ["", "", "", ""])[:4]
    return (
        f"The following are multiple choice questions (with answers) about {subject}.\n\n"
        f"{question}\n\n"
        f"A. {a}\n\n"
        f"B. {b}\n\n"
        f"C. {c}\n\n"
        f"D. {d}\n\n"
        f"Answer:"
    )


def load_assistant(model_id_or_path: str, base_if_lora, device: torch.device):
    local_files_only = os.getenv("HF_HUB_OFFLINE") == "1" or os.getenv("TRANSFORMERS_OFFLINE") == "1"
    if os.path.isdir(model_id_or_path) and os.path.exists(os.path.join(model_id_or_path, "adapter_config.json")):
        if not base_if_lora:
            raise ValueError("base_if_lora is required when loading a LoRA adapter directory")
        base = AutoModelForCausalLM.from_pretrained(
            base_if_lora, torch_dtype=torch.bfloat16, local_files_only=local_files_only
        )
        try:
            peft = PeftModel.from_pretrained(base, model_id_or_path, torch_dtype=torch.bfloat16)
        except TypeError as exc:
            cfg_path = os.path.join(model_id_or_path, "adapter_config.json")
            raw_cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
            allowed = set(inspect.signature(LoraConfig.__init__).parameters.keys())
            filtered_cfg = {k: v for k, v in raw_cfg.items() if k in allowed}
            dropped = sorted(set(raw_cfg.keys()) - set(filtered_cfg.keys()))
            if dropped:
                print(f"[peft-load] drop unsupported adapter_config keys for eval: {dropped} ({exc})")
            peft = PeftModel.from_pretrained(
                base,
                model_id_or_path,
                torch_dtype=torch.bfloat16,
                config=LoraConfig(**filtered_cfg),
            )
        model = peft.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id_or_path, torch_dtype=torch.bfloat16, local_files_only=local_files_only
        )
    model.eval()
    model.to(device)
    return model


def _tokenize_prompts(tokenizer, prompts: List[str]) -> List[List[int]]:
    ids = [tokenizer(p, add_special_tokens=False).input_ids for p in prompts]
    bos = tokenizer.bos_token_id
    if bos is not None:
        bos_id = int(bos)
        ids = [[bos_id] + seq for seq in ids]
    return ids


def _truncate_prompt(seq: List[int], max_len: int, mode: str) -> List[int]:
    if not max_len or max_len <= 0 or len(seq) <= max_len:
        return seq
    if mode == "left":
        return seq[-max_len:]
    if mode == "head_tail":
        head = max_len // 2
        tail = max_len - head
        if head <= 0:
            return seq[-tail:]
        if tail <= 0:
            return seq[:head]
        return seq[:head] + seq[-tail:]
    raise ValueError(f"Unknown truncate_mode: {mode!r} (expected 'left' or 'head_tail')")


def _pad_batch(seqs: List[List[int]], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = max((len(s) for s in seqs), default=0)
    input_ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(seqs), max_len), dtype=torch.long)
    for i, seq in enumerate(seqs):
        if not seq:
            continue
        seq_len = len(seq)
        input_ids[i, :seq_len] = torch.tensor(seq, dtype=torch.long)
        attention_mask[i, :seq_len] = 1
    return input_ids, attention_mask


def _align_kl_tensors(
    o_logp: torch.Tensor,
    f_logp: torch.Tensor,
    o_p: torch.Tensor,
    f_p: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    target = o_logp.device
    if f_logp.device != target:
        f_logp = f_logp.to(target)
    if o_p.device != target:
        o_p = o_p.to(target)
    if f_p.device != target:
        f_p = f_p.to(target)
    return o_logp, f_logp, o_p, f_p


@dataclass
class ChoiceSpec:
    mode: str
    token_ids: List[List[int]]
    answer_token_ids: List[int]
    prefix_token_id: int | None


def _choice_texts() -> List[str]:
    # Keep consistent with scripts/eval_wmdp_routing.py (prompts end with "Answer:").
    # Using leading-space choices enables prefix-mode scoring for LLaMA tokenizers.
    return [" A", " B", " C", " D"]


def _choice_spec(tokenizer) -> ChoiceSpec:
    texts = _choice_texts()
    token_ids = [tokenizer(t, add_special_tokens=False).input_ids for t in texts]
    if any(len(t) == 0 for t in token_ids):
        raise ValueError("Empty tokenization for choices")
    lens = {len(t) for t in token_ids}
    if lens == {1}:
        return ChoiceSpec(
            mode="single",
            token_ids=token_ids,
            answer_token_ids=[int(t[0]) for t in token_ids],
            prefix_token_id=None,
        )
    if lens == {2} and all(t[0] == token_ids[0][0] for t in token_ids):
        return ChoiceSpec(
            mode="prefix",
            token_ids=token_ids,
            answer_token_ids=[int(t[1]) for t in token_ids],
            prefix_token_id=int(token_ids[0][0]),
        )
    return ChoiceSpec(mode="full", token_ids=token_ids, answer_token_ids=[], prefix_token_id=None)


def _build_prompt_batch(
    prompt_ids: List[List[int]],
    max_len: int,
    pad_id: int,
    append_token_id: int | None = None,
    truncate_mode: str = "left",
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    seqs: List[List[int]] = []
    lens: List[int] = []
    for seq in prompt_ids:
        if append_token_id is not None:
            max_prompt = max_len - 1 if (max_len and max_len > 0) else 0
            if max_prompt < 1:
                max_prompt = 1
            seq_t = _truncate_prompt(seq, max_prompt, truncate_mode)
            seq_t = seq_t + [int(append_token_id)]
        else:
            seq_t = _truncate_prompt(seq, max_len, truncate_mode)
        seqs.append(seq_t)
        lens.append(len(seq_t))
    input_ids, attention_mask = _pad_batch(seqs, pad_id)
    return input_ids, attention_mask, lens


@torch.inference_mode()
def sym_kl_scores(
    tokenizer,
    orig_model,
    ft_model,
    prompts: List[str],
    batch_size: int,
    max_len: int,
    orig_device: torch.device,
    ft_device: torch.device,
    score_space: str = "vocab",
    score_pos: str = "prompt_last",
    score_probe_suffix: str = "",
    score_last_k: int = 1,
    score_last_k_reduce: str = "mean",
    score_k_mode: str = "last",
    score_reducer_alpha: float = DEFAULT_ROUTING_REDUCER_ALPHA,
    score_reducer_beta: float = DEFAULT_ROUTING_REDUCER_BETA,
    choice_spec: ChoiceSpec | None = None,
    truncate_mode: str = "left",
) -> np.ndarray:
    scores: List[np.ndarray] = []
    prompt_ids_all = _tokenize_prompts(tokenizer, prompts)
    score_probe_suffix = str(score_probe_suffix or "")
    if score_probe_suffix.strip():
        probe_ids = tokenizer(score_probe_suffix, add_special_tokens=False).input_ids
        if probe_ids:
            prompt_ids_all = [seq + probe_ids for seq in prompt_ids_all]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if (score_space in ("choices", "choices5") or score_pos == "after_choice_prefix") and choice_spec is None:
        choice_spec = _choice_spec(tokenizer)
    append_token_id = None
    if (
        choice_spec is not None
        and choice_spec.mode == "prefix"
        and (score_space in ("choices", "choices5") or score_pos == "after_choice_prefix")
    ):
        append_token_id = choice_spec.prefix_token_id
    score_last_k = int(score_last_k) if score_last_k is not None else 1
    if score_last_k < 1:
        score_last_k = 1
    score_last_k_reduce = normalize_routing_reducer(score_last_k_reduce)
    score_k_mode = str(score_k_mode or "last")
    if score_k_mode not in ("last", "uniform"):
        raise ValueError(f"Unknown score_k_mode: {score_k_mode!r} (expected 'last' or 'uniform')")
    for start in range(0, len(prompts), batch_size):
        batch_prompt_ids = prompt_ids_all[start : start + batch_size]
        input_ids, attention_mask, prompt_lens = _build_prompt_batch(
            batch_prompt_ids,
            max_len,
            pad_id,
            append_token_id=append_token_id,
            truncate_mode=truncate_mode,
        )
        bs = len(batch_prompt_ids)

        o_inp = input_ids.to(orig_device)
        o_mask = attention_mask.to(orig_device)
        o_last_pos = torch.tensor(prompt_lens, device=orig_device) - 1
        o_logits = orig_model(input_ids=o_inp, attention_mask=o_mask, use_cache=False).logits
        o_last = o_logits[torch.arange(bs, device=orig_device), o_last_pos, :].detach().float()

        f_inp = input_ids.to(ft_device)
        f_mask = attention_mask.to(ft_device)
        f_last_pos = torch.tensor(prompt_lens, device=ft_device) - 1
        f_logits = ft_model(input_ids=f_inp, attention_mask=f_mask, use_cache=False).logits
        f_last = f_logits[torch.arange(bs, device=ft_device), f_last_pos, :].detach().float()

        if score_last_k > 1:
            k = score_last_k
            if score_k_mode == "uniform":
                fracs_o = torch.linspace(0.0, 1.0, steps=k, device=orig_device, dtype=torch.float32)
                # Ensure k==1 still uses the last position.
                if k == 1:
                    pos_o = o_last_pos.unsqueeze(1)
                else:
                    pos_o = torch.floor(o_last_pos.float().unsqueeze(1) * fracs_o.unsqueeze(0)).to(o_last_pos.dtype)
            else:
                offs_o = torch.arange(k, device=orig_device, dtype=o_last_pos.dtype)
                pos_o = (o_last_pos.unsqueeze(1) - offs_o.unsqueeze(0)).clamp(min=0)
            b_o = torch.arange(bs, device=orig_device).unsqueeze(1)
            o_k_logits = o_logits[b_o, pos_o, :].detach().float()

            if score_k_mode == "uniform":
                fracs_f = torch.linspace(0.0, 1.0, steps=k, device=ft_device, dtype=torch.float32)
                if k == 1:
                    pos_f = f_last_pos.unsqueeze(1)
                else:
                    pos_f = torch.floor(f_last_pos.float().unsqueeze(1) * fracs_f.unsqueeze(0)).to(f_last_pos.dtype)
            else:
                offs_f = torch.arange(k, device=ft_device, dtype=f_last_pos.dtype)
                pos_f = (f_last_pos.unsqueeze(1) - offs_f.unsqueeze(0)).clamp(min=0)
            b_f = torch.arange(bs, device=ft_device).unsqueeze(1)
            f_k_logits = f_logits[b_f, pos_f, :].detach().float()

            if o_k_logits.device != f_k_logits.device:
                f_k_logits = f_k_logits.to(o_k_logits.device)

            if score_space == "choices":
                if choice_spec is None:
                    raise ValueError("choice_spec is required when score_space=choices")
                if choice_spec.mode == "full":
                    raise ValueError("score_space=choices is not supported for tokenizer choice mode=full; use vocab")
                ids = torch.tensor(choice_spec.answer_token_ids, device=o_k_logits.device, dtype=torch.long)
                o_sub = o_k_logits.index_select(dim=2, index=ids)
                f_sub = f_k_logits.index_select(dim=2, index=ids.to(f_k_logits.device))
                o_logp = F.log_softmax(o_sub, dim=-1)
                f_logp = F.log_softmax(f_sub, dim=-1)
                o_p = o_logp.exp()
                f_p = f_logp.exp()
            elif score_space == "choices5":
                if choice_spec is None:
                    raise ValueError("choice_spec is required when score_space=choices5")
                if choice_spec.mode == "full":
                    raise ValueError("score_space=choices5 is not supported for tokenizer choice mode=full; use vocab")
                ids = torch.tensor(choice_spec.answer_token_ids, device=o_k_logits.device, dtype=torch.long)
                o_logp_full = F.log_softmax(o_k_logits, dim=-1)
                f_logp_full = F.log_softmax(f_k_logits, dim=-1)
                o_p_full = o_logp_full.exp()
                f_p_full = f_logp_full.exp()
                o_choices = o_p_full.index_select(dim=2, index=ids)
                f_choices = f_p_full.index_select(dim=2, index=ids.to(f_p_full.device))
                o_other = (1.0 - o_choices.sum(dim=-1, keepdim=True)).clamp(min=1e-12)
                f_other = (1.0 - f_choices.sum(dim=-1, keepdim=True)).clamp(min=1e-12)
                o_p = torch.cat([o_choices, o_other], dim=-1)
                f_p = torch.cat([f_choices, f_other], dim=-1)
                o_logp = torch.log(o_p)
                f_logp = torch.log(f_p)
            else:
                o_logp = F.log_softmax(o_k_logits, dim=-1)
                f_logp = F.log_softmax(f_k_logits, dim=-1)
                o_p = o_logp.exp()
                f_p = f_logp.exp()
            o_logp, f_logp, o_p, f_p = _align_kl_tensors(o_logp, f_logp, o_p, f_p)
            kl_of = (o_p * (o_logp - f_logp)).sum(dim=-1)
            kl_fo = (f_p * (f_logp - o_logp)).sum(dim=-1)
            sym_k = 0.5 * (kl_of + kl_fo)
            entropy_k = routing_entropy_from_logp(o_logp, probs=o_p)
            surprisal_k = routing_surprisal_from_argmax(o_logp)
            sym = reduce_routing_scores(
                sym_k,
                score_last_k_reduce,
                entropy=entropy_k,
                surprisal=surprisal_k,
                alpha=score_reducer_alpha,
                beta=score_reducer_beta,
            )
            scores.append(sym.detach().float().cpu().numpy())
        else:
            if score_space == "choices":
                if choice_spec is None:
                    raise ValueError("choice_spec is required when score_space=choices")
                if choice_spec.mode == "full":
                    raise ValueError("score_space=choices is not supported for tokenizer choice mode=full; use vocab")
                ids = torch.tensor(choice_spec.answer_token_ids, device=o_last.device, dtype=torch.long)
                o_sub = o_last.index_select(dim=1, index=ids)
                f_sub = f_last.index_select(dim=1, index=ids.to(f_last.device))
                o_logp = F.log_softmax(o_sub, dim=-1)
                f_logp = F.log_softmax(f_sub, dim=-1)
                o_p = o_logp.exp()
                f_p = f_logp.exp()
            elif score_space == "choices5":
                if choice_spec is None:
                    raise ValueError("choice_spec is required when score_space=choices5")
                if choice_spec.mode == "full":
                    raise ValueError("score_space=choices5 is not supported for tokenizer choice mode=full; use vocab")
                ids = torch.tensor(choice_spec.answer_token_ids, device=o_last.device, dtype=torch.long)
                o_logp_full = F.log_softmax(o_last, dim=-1)
                f_logp_full = F.log_softmax(f_last, dim=-1)
                o_p_full = o_logp_full.exp()
                f_p_full = f_logp_full.exp()
                o_choices = o_p_full.index_select(dim=1, index=ids)
                f_choices = f_p_full.index_select(dim=1, index=ids.to(f_p_full.device))
                o_other = (1.0 - o_choices.sum(dim=-1, keepdim=True)).clamp(min=1e-12)
                f_other = (1.0 - f_choices.sum(dim=-1, keepdim=True)).clamp(min=1e-12)
                o_p = torch.cat([o_choices, o_other], dim=-1)
                f_p = torch.cat([f_choices, f_other], dim=-1)
                o_logp = torch.log(o_p)
                f_logp = torch.log(f_p)
            else:
                # Full-vocab sym-KL (original behavior).
                o_logp = F.log_softmax(o_last, dim=-1)
                f_logp = F.log_softmax(f_last, dim=-1)
                o_p = o_logp.exp()
                f_p = f_logp.exp()
            o_logp, f_logp, o_p, f_p = _align_kl_tensors(o_logp, f_logp, o_p, f_p)
            kl_of = (o_p * (o_logp - f_logp)).sum(dim=-1)
            kl_fo = (f_p * (f_logp - o_logp)).sum(dim=-1)
            sym_single = 0.5 * (kl_of + kl_fo)
            entropy_single = routing_entropy_from_logp(o_logp, probs=o_p)
            surprisal_single = routing_surprisal_from_argmax(o_logp)
            sym = reduce_routing_scores(
                sym_single.unsqueeze(-1),
                score_last_k_reduce,
                entropy=entropy_single.unsqueeze(-1),
                surprisal=surprisal_single.unsqueeze(-1),
                alpha=score_reducer_alpha,
                beta=score_reducer_beta,
            )

            scores.append(sym.detach().float().cpu().numpy())

    return np.concatenate(scores, axis=0) if scores else np.zeros((0,), dtype=np.float32)


def candidate_thresholds(forget: np.ndarray, retain: np.ndarray) -> np.ndarray:
    combined = np.concatenate([forget, retain]).astype(np.float64, copy=False)
    combined = combined[np.isfinite(combined)]
    unique = np.unique(combined)
    unique.sort()
    if unique.size == 0:
        return unique
    above_max = np.nextafter(unique[-1], np.inf)
    return np.concatenate([unique, np.array([above_max], dtype=unique.dtype)])


def metrics_at_threshold(forget: np.ndarray, retain: np.ndarray, threshold: float) -> Dict:
    if not np.isfinite(float(threshold)):
        raise ValueError(f"Non-finite threshold: {threshold}")
    # Match DoubleAssisLLM: is_forget = score > threshold
    forget_pred = forget > threshold
    retain_pred = retain > threshold

    tp = int(forget_pred.sum())
    fn = int((~forget_pred).sum())
    fp = int(retain_pred.sum())
    tn = int((~retain_pred).sum())

    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0.0
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tpr
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "threshold": float(threshold),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "accuracy": float(acc),
        "tpr": float(tpr),
        "fpr": float(fpr),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "gap": float(tpr - fpr),
    }


def select_threshold(
    forget: np.ndarray,
    retain: np.ndarray,
    optimize: str,
    min_tpr,
    max_fpr,
) -> Dict:
    cands = candidate_thresholds(forget, retain)
    if cands.size == 0:
        raise ValueError("Empty candidate threshold set")

    best_any = None
    best_any_key = None
    best_constrained = None
    best_constrained_key = None

    for thr in cands:
        if not np.isfinite(float(thr)):
            continue
        m = metrics_at_threshold(forget, retain, float(thr))
        if optimize == "gap":
            score = m["gap"]
        elif optimize == "f1":
            score = m["f1"]
        elif optimize == "tpr":
            score = m["tpr"]
        else:
            score = m["accuracy"]
        key = (float(score), float(m["tpr"]), -float(m["fpr"]))
        if best_any_key is None or key > best_any_key:
            best_any_key = key
            best_any = m

        ok = True
        if min_tpr is not None:
            ok = ok and (m["tpr"] >= min_tpr)
        if max_fpr is not None:
            ok = ok and (m["fpr"] <= max_fpr)
        if ok and (best_constrained_key is None or key > best_constrained_key):
            best_constrained_key = key
            best_constrained = m

    chosen = best_constrained if best_constrained is not None else best_any
    if chosen is None:
        raise ValueError("No valid finite threshold candidate")
    return {
        "best_threshold": float(chosen["threshold"]),
        "optimize": optimize,
        "constraints": {"min_tpr": min_tpr, "max_fpr": max_fpr},
        "constraints_satisfied": bool(best_constrained is not None),
        "metrics": chosen,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetuned_assist_path", required=True, help="A1 checkpoint dir (can be LoRA adapter dir)")
    parser.add_argument("--original_assist_path", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--base_if_lora", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--orig_device", default="cuda", help="Device for original assistant (e.g., cuda:0)")
    parser.add_argument("--ft_device", default="cuda", help="Device for finetuned assistant (e.g., cuda:1)")
    parser.add_argument("--wmdp_domains", default="bio,cyber", help="Forget domains (csv)")
    parser.add_argument("--mmlu_retain_file", default="eval-method/wmdp/data/mmlu/all_validation.jsonl")
    parser.add_argument(
        "--mmlu_retain_subjects",
        default=None,
        help="Optional comma-separated subject filter for MMLU retain prompts (only works if JSONL has non-empty subject)",
    )
    parser.add_argument("--max_forget", type=int, default=800)
    parser.add_argument("--max_retain", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--optimize", choices=["accuracy", "gap", "f1", "tpr"], default="gap")
    parser.add_argument("--min_tpr", type=float, default=None)
    parser.add_argument("--max_fpr", type=float, default=None)
    parser.add_argument("--score_space", choices=["vocab", "choices", "choices5"], default="vocab")
    parser.add_argument(
        "--score_pos",
        choices=["prompt_last", "after_choice_prefix"],
        default="prompt_last",
        help="Where to compute routing logits for A0/A1. "
        "Default 'prompt_last' scores at the last prompt token (typically ':'). "
        "'after_choice_prefix' appends the shared prefix token (e.g., space) when the tokenizer supports it.",
    )
    parser.add_argument(
        "--score_probe_suffix",
        type=str,
        default="",
        help="Optional: append a fixed probe suffix to the prompt when computing A0/A1 routing symKL "
        "(does NOT affect WMDP/MMLU answering prompts). Default empty keeps existing behavior.",
    )
    parser.add_argument(
        "--score_last_k",
        type=int,
        default=1,
        help="Average routing score over the last K prompt positions (default: 1).",
    )
    parser.add_argument(
        "--score_last_k_reduce",
        choices=["mean", "max", "cbd", "fsis", "escort"],
        default="mean",
        help="How to reduce symKL over the last K prompt positions when --score_last_k > 1. "
        "Default 'mean' keeps the old behavior; 'max' can amplify rare high-divergence signals. "
        "'cbd' uses sum(KL^2)/sum(KL); 'fsis' uses entropy-weighted average; "
        "'escort' combines log-KL and saliency z-score.",
    )
    parser.add_argument(
        "--score_reducer_alpha",
        type=float,
        default=DEFAULT_ROUTING_REDUCER_ALPHA,
        help="Escort reducer alpha. Ignored by other reducers.",
    )
    parser.add_argument(
        "--score_reducer_beta",
        type=float,
        default=DEFAULT_ROUTING_REDUCER_BETA,
        help="Escort reducer beta. Ignored by other reducers.",
    )
    parser.add_argument(
        "--score_k_mode",
        choices=["last", "uniform"],
        default="last",
        help="Which prompt positions to use when --score_last_k > 1. "
        "Default 'last' matches the old behavior (last-K tokens). "
        "'uniform' samples K positions uniformly over the prompt (can capture earlier divergence).",
    )
    parser.add_argument(
        "--truncate_mode",
        choices=["left", "head_tail"],
        default="left",
        help="Prompt truncation strategy when max_len is exceeded. "
        "'left' keeps the suffix (existing behavior). "
        "'head_tail' keeps both prefix+suffix to preserve early context for long prompts (useful for WMDP-cyber).",
    )
    parser.add_argument(
        "--dump_npz",
        default=None,
        help="Optional: write {forget_scores, retain_scores, meta_json} to a .npz (numeric arrays only).",
    )
    parser.add_argument("--output_json", default=None, help="Write JSON result here (default: alongside finetuned_assist_path)")
    args = parser.parse_args()

    def resolve_device(name: str) -> torch.device:
        if name is None or name == "":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if name.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(name)

    orig_device = resolve_device(args.orig_device)
    ft_device = resolve_device(args.ft_device)
    t0 = time.time()

    local_files_only = os.getenv("HF_HUB_OFFLINE") == "1" or os.getenv("TRANSFORMERS_OFFLINE") == "1"
    tokenizer = AutoTokenizer.from_pretrained(args.original_assist_path, local_files_only=local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    orig = load_assistant(args.original_assist_path, base_if_lora=None, device=orig_device)
    ft = load_assistant(args.finetuned_assist_path, base_if_lora=args.base_if_lora, device=ft_device)
    choice_spec = (
        _choice_spec(tokenizer)
        if (args.score_space in ("choices", "choices5") or args.score_pos == "after_choice_prefix")
        else None
    )

    # Load forget (WMDP)
    domain_map = {"bio": ("bio_questions.json", "biology"), "cyber": ("cyber_questions.json", "cybersecurity"), "chem": ("chem_questions.json", "chemistry")}
    domains = [d.strip().lower() for d in str(args.wmdp_domains).split(",") if d.strip()]
    forget_prompts: List[str] = []
    for d in domains:
        fname, subj = domain_map[d]
        path = data_root() / "eval-method" / "wmdp" / "data" / "wmdp_mcqs" / "wmdp-mcqs" / fname
        for ex in read_json(path):
            forget_prompts.append(format_mcq_prompt(subj, ex["question"], ex["choices"]))

    # deterministic sample
    rng = np.random.default_rng(int(args.seed))
    if args.max_forget and args.max_forget < len(forget_prompts):
        idx = rng.permutation(len(forget_prompts))[: int(args.max_forget)]
        forget_prompts = [forget_prompts[i] for i in idx.tolist()]

    # Load retain (MMLU jsonl)
    mmlu_path = Path(args.mmlu_retain_file)
    if not mmlu_path.is_absolute():
        mmlu_path = (data_root() / mmlu_path).resolve()
    if not mmlu_path.exists():
        raise FileNotFoundError(
            f"MMLU retain file not found: {mmlu_path}. Run `HF_ENDPOINT=https://hf-mirror.com python3 scripts/cache_mmlu.py`."
        )
    mmlu_rows = read_jsonl(mmlu_path)
    if args.mmlu_retain_subjects:
        keep = {s.strip() for s in str(args.mmlu_retain_subjects).split(",") if s.strip()}
        if keep:
            mmlu_rows = [r for r in mmlu_rows if (r.get("subject") in keep)]
    if args.max_retain and args.max_retain < len(mmlu_rows):
        idx = rng.permutation(len(mmlu_rows))[: int(args.max_retain)]
        mmlu_rows = [mmlu_rows[i] for i in idx.tolist()]
    retain_prompts = [format_mcq_prompt(r.get("subject") or "general", r["question"], r["choices"]) for r in mmlu_rows]

    print(f"[threshold] forget_prompts={len(forget_prompts)} retain_prompts={len(retain_prompts)} orig_device={orig_device} ft_device={ft_device}")

    forget_scores_raw = sym_kl_scores(
        tokenizer,
        orig,
        ft,
        forget_prompts,
        args.batch_size,
        args.max_len,
        orig_device,
        ft_device,
        score_space=args.score_space,
        score_pos=args.score_pos,
        score_probe_suffix=args.score_probe_suffix,
        score_last_k=int(args.score_last_k),
        score_last_k_reduce=str(args.score_last_k_reduce),
        score_k_mode=str(args.score_k_mode),
        score_reducer_alpha=float(args.score_reducer_alpha),
        score_reducer_beta=float(args.score_reducer_beta),
        choice_spec=choice_spec,
        truncate_mode=str(args.truncate_mode),
    )
    retain_scores_raw = sym_kl_scores(
        tokenizer,
        orig,
        ft,
        retain_prompts,
        args.batch_size,
        args.max_len,
        orig_device,
        ft_device,
        score_space=args.score_space,
        score_pos=args.score_pos,
        score_probe_suffix=args.score_probe_suffix,
        score_last_k=int(args.score_last_k),
        score_last_k_reduce=str(args.score_last_k_reduce),
        score_k_mode=str(args.score_k_mode),
        score_reducer_alpha=float(args.score_reducer_alpha),
        score_reducer_beta=float(args.score_reducer_beta),
        choice_spec=choice_spec,
        truncate_mode=str(args.truncate_mode),
    )

    reducer_meta = routing_reducer_metadata(
        str(args.score_last_k_reduce),
        alpha=float(args.score_reducer_alpha),
        beta=float(args.score_reducer_beta),
    )
    forget_finite_mask = np.isfinite(forget_scores_raw)
    retain_finite_mask = np.isfinite(retain_scores_raw)
    forget_scores = forget_scores_raw[forget_finite_mask]
    retain_scores = retain_scores_raw[retain_finite_mask]

    if forget_scores.size == 0 or retain_scores.size == 0:
        raise ValueError(
            "No finite routing scores for threshold selection: "
            f"forget={int(forget_scores.size)} retain={int(retain_scores.size)}"
        )

    result = {
        "wmdp_domains": domains,
        "mmlu_retain_file": str(mmlu_path),
        "score_space": str(args.score_space),
        "score_pos": str(args.score_pos),
        "score_probe_suffix": str(args.score_probe_suffix),
        "score_last_k": int(args.score_last_k),
        "score_last_k_reduce": str(args.score_last_k_reduce),
        "score_reducer_alpha": float(args.score_reducer_alpha),
        "score_reducer_beta": float(args.score_reducer_beta),
        "score_k_mode": str(args.score_k_mode),
        "truncate_mode": str(args.truncate_mode),
        **reducer_meta,
        "choice_mode": (choice_spec.mode if choice_spec is not None else None),
        "max_forget": int(len(forget_prompts)),
        "max_retain": int(len(retain_prompts)),
        "score_summary": {
            "forget_n_raw": int(forget_scores_raw.size),
            "retain_n_raw": int(retain_scores_raw.size),
            "forget_n_finite": int(forget_scores.size),
            "retain_n_finite": int(retain_scores.size),
            "forget_n_nonfinite_dropped": int(forget_scores_raw.size - forget_scores.size),
            "retain_n_nonfinite_dropped": int(retain_scores_raw.size - retain_scores.size),
            "forget_mean": float(forget_scores.mean()) if forget_scores.size else None,
            "forget_std": float(forget_scores.std()) if forget_scores.size else None,
            "retain_mean": float(retain_scores.mean()) if retain_scores.size else None,
            "retain_std": float(retain_scores.std()) if retain_scores.size else None,
        },
        "selection": select_threshold(
            forget_scores,
            retain_scores,
            optimize=args.optimize,
            min_tpr=args.min_tpr,
            max_fpr=args.max_fpr,
        ),
        "elapsed_sec": float(time.time() - t0),
    }

    if args.dump_npz:
        dump_path = Path(args.dump_npz)
        if not dump_path.is_absolute():
            dump_path = (repo_root() / dump_path).resolve()
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "best_threshold": float(result["selection"]["best_threshold"]),
            "optimize": str(args.optimize),
            "constraints": {"min_tpr": args.min_tpr, "max_fpr": args.max_fpr},
            "wmdp_domains": domains,
            "mmlu_retain_file": str(mmlu_path),
            "seed": int(args.seed),
            "n_forget": int(forget_scores.size),
            "n_retain": int(retain_scores.size),
            "score_space": str(args.score_space),
            "score_pos": str(args.score_pos),
            "score_probe_suffix": str(args.score_probe_suffix),
            "score_last_k": int(args.score_last_k),
            "score_last_k_reduce": str(args.score_last_k_reduce),
            "score_reducer_alpha": float(args.score_reducer_alpha),
            "score_reducer_beta": float(args.score_reducer_beta),
            "score_k_mode": str(args.score_k_mode),
            "truncate_mode": str(args.truncate_mode),
            "schema_version": 1,
            "routing_score_semantics_version": ROUTING_SCORE_SEMANTICS_VERSION,
        }
        np.savez_compressed(
            str(dump_path),
            forget_scores=forget_scores.astype(np.float32, copy=False),
            retain_scores=retain_scores.astype(np.float32, copy=False),
            meta_json=np.asarray(json.dumps(meta, ensure_ascii=False)),
        )
        print("dump_npz", str(dump_path))

    out_json = Path(args.output_json) if args.output_json else (Path(args.finetuned_assist_path) / "wmdp_threshold.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[threshold] wrote", out_json)
    print("[threshold] best_threshold", result["selection"]["best_threshold"])


if __name__ == "__main__":
    main()
