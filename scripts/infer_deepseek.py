#!/usr/bin/env python3
"""
Routing evaluation for DeepSeek CBD-DFB experiments.

Scoring follows the author's WMDP routing pipeline exactly (see
`scripts_ref/eval_wmdp_routing.py` and `scripts_ref/select_wmdp_routing_threshold.py`):
the routing score is the **symmetric KL divergence between A0 and A1 at the
next-token position after the prompt**, not a cross-entropy difference.

    score(x) = 0.5 * ( KL(p_A0 || p_A1) + KL(p_A1 || p_A0) )   at the last prompt token

  A0 = original assistant, A1 = unlearned assistant.

Only the data plumbing is DeepSeek-specific:

  * prompt            -> `probing input` (code that calls a deprecated API)
  * forget side       -> `probing input`   from the valid split of D_forget.json
  * retain side       -> `retain`          from the same records
  * test positive     -> D_test_U_dep.json      (expect score ABOVE threshold)
  * test negative     -> D_test_U_nondep.json   (expect score BELOW threshold)

The author's MCQ-only options (`score_space=choices/choices5`,
`score_pos=after_choice_prefix`) are not ported: they index the A/B/C/D answer
tokens, which do not exist for free-form code. Full-vocab scoring at the last
prompt token is the author's default and is what applies here.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from routing_score_reducers import (
    DEFAULT_ROUTING_REDUCER_ALPHA,
    DEFAULT_ROUTING_REDUCER_BETA,
    VALID_ROUTING_REDUCERS,
    normalize_routing_reducer,
    reduce_routing_scores,
    routing_entropy_from_logp,
    routing_reducer_metadata,
    routing_surprisal_from_argmax,
)


# ---------------------------------------------------------------------------
# Model / tokenizer plumbing (ported from scripts_ref/eval_wmdp_routing.py)
# ---------------------------------------------------------------------------

def load_model_maybe_lora(model_id_or_path: str, base_if_lora, device: torch.device):
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


def _build_prompt_batch(
    prompt_ids: List[List[int]],
    max_len: int,
    pad_id: int,
    truncate_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    seqs: List[List[int]] = []
    lens: List[int] = []
    for seq in prompt_ids:
        seq_t = _truncate_prompt(seq, max_len, truncate_mode)
        seqs.append(seq_t)
        lens.append(len(seq_t))
    input_ids, attention_mask = _pad_batch(seqs, pad_id)
    return input_ids, attention_mask, lens


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


# ---------------------------------------------------------------------------
# Symmetric-KL routing score (ported from scripts_ref/eval_wmdp_routing.py)
# ---------------------------------------------------------------------------

@torch.no_grad()
def sym_kl_scores(
    orig_model,
    ft_model,
    tokenizer,
    prompts: List[str],
    *,
    device: torch.device,
    batch_size: int,
    max_len: int,
    truncate_mode: str,
    score_last_k: int,
    score_last_k_reduce: str,
    score_k_mode: str,
    score_probe_suffix: str,
    score_reducer_alpha: float,
    score_reducer_beta: float,
    tag: str = "",
) -> np.ndarray:
    """Symmetric KL between A0 and A1 at the next-token position after each prompt."""
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    prompt_ids_all = _tokenize_prompts(tokenizer, prompts)

    score_probe_suffix = str(score_probe_suffix or "")
    if score_probe_suffix.strip():
        probe_ids = tokenizer(score_probe_suffix, add_special_tokens=False).input_ids
        if probe_ids:
            prompt_ids_all = [seq + probe_ids for seq in prompt_ids_all]

    out: List[np.ndarray] = []
    total = len(prompt_ids_all)
    for start in range(0, total, batch_size):
        batch_ids = prompt_ids_all[start : start + batch_size]
        input_ids, mask, lens = _build_prompt_batch(batch_ids, max_len, pad_id, truncate_mode)
        bs = input_ids.size(0)

        inp = input_ids.to(device)
        att = mask.to(device)
        last_pos = torch.tensor(lens, device=device) - 1

        a0_logits = orig_model(input_ids=inp, attention_mask=att, use_cache=False).logits
        a1_logits = ft_model(input_ids=inp, attention_mask=att, use_cache=False).logits

        if score_last_k > 1:
            k = score_last_k
            if score_k_mode == "uniform":
                fracs = torch.linspace(0.0, 1.0, steps=k, device=device, dtype=torch.float32)
                pos = torch.floor(last_pos.float().unsqueeze(1) * fracs.unsqueeze(0)).to(last_pos.dtype)
            else:
                offs = torch.arange(k, device=device, dtype=last_pos.dtype)
                pos = (last_pos.unsqueeze(1) - offs.unsqueeze(0)).clamp(min=0)
            rows = torch.arange(bs, device=device).unsqueeze(1)
            o_sel = a0_logits[rows, pos, :].detach().float()
            f_sel = a1_logits[rows, pos, :].detach().float()
        else:
            rows = torch.arange(bs, device=device)
            o_sel = a0_logits[rows, last_pos, :].detach().float().unsqueeze(1)
            f_sel = a1_logits[rows, last_pos, :].detach().float().unsqueeze(1)

        del a0_logits, a1_logits

        # Full-vocab sym-KL (the author's default `score_space=vocab`).
        o_logp = F.log_softmax(o_sel, dim=-1)
        f_logp = F.log_softmax(f_sel, dim=-1)
        o_p = o_logp.exp()
        f_p = f_logp.exp()
        o_logp, f_logp, o_p, f_p = _align_kl_tensors(o_logp, f_logp, o_p, f_p)

        kl_of = (o_p * (o_logp - f_logp)).sum(dim=-1)
        kl_fo = (f_p * (f_logp - o_logp)).sum(dim=-1)
        score_k = 0.5 * (kl_of + kl_fo)  # [bs, k]

        entropy_k = routing_entropy_from_logp(o_logp, probs=o_p)
        surprisal_k = routing_surprisal_from_argmax(o_logp)
        score = reduce_routing_scores(
            score_k,
            score_last_k_reduce,
            entropy=entropy_k,
            surprisal=surprisal_k,
            alpha=score_reducer_alpha,
            beta=score_reducer_beta,
        )

        out.append(score.detach().float().cpu().numpy())
        done = min(start + batch_size, total)
        if done % (batch_size * 50) < batch_size or done == total:
            print(f"  [{tag}] {done}/{total} scored...")

    if not out:
        return np.zeros((0,), dtype=np.float64)
    return np.concatenate(out).astype(np.float64, copy=False)


# ---------------------------------------------------------------------------
# Threshold selection (ported from scripts_ref/select_wmdp_routing_threshold.py)
# ---------------------------------------------------------------------------

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


def select_threshold(forget: np.ndarray, retain: np.ndarray, optimize: str, min_tpr, max_fpr) -> Dict:
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


# ---------------------------------------------------------------------------
# DeepSeek data plumbing
# ---------------------------------------------------------------------------

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def valid_split(raw_data: List[Dict], train_ratio: float, seed: int) -> List[Dict]:
    """Reproduce the split used by extract_cbd_dfb_basis.py -- must stay byte-identical."""
    n_total = len(raw_data)
    n_train = int(n_total * train_ratio)
    indices = list(range(n_total))
    rng = random.Random(seed)
    rng.shuffle(indices)
    valid_indices = sorted(indices[n_train:])
    return [raw_data[i] for i in valid_indices]


def extract_prompts(records: List[Dict], field: str, fallback: str = "probing input") -> List[str]:
    """Pull the prompt text out of each record.

    IMPORTANT: no .strip() here. `probing input new` is cut at the API decision point and
    1535/9667 records end in indentation whitespace that is part of the anchor context --
    stripping it would move the scored position.
    """
    out: List[str] = []
    for it in records:
        v = it.get(field)
        if not isinstance(v, str) or not v.strip():
            v = it.get(fallback)
        if isinstance(v, str) and v.strip():
            out.append(v)
    return out


def describe(name: str, scores: np.ndarray, threshold: float) -> Dict:
    above = int(np.sum(scores > threshold))
    total = int(scores.size)
    pct = 100.0 * above / total if total else 0.0
    print(
        f"  {name:24s} | {scores.mean():9.5f} | {scores.std():9.5f} | "
        f"{scores.min():9.5f} | {scores.max():9.5f} | {above:6d}/{total:6d} ({pct:5.1f}%)"
    )
    return {
        "n": total,
        "mean": float(scores.mean()) if total else 0.0,
        "std": float(scores.std()) if total else 0.0,
        "min": float(scores.min()) if total else 0.0,
        "max": float(scores.max()) if total else 0.0,
        "above_threshold": above,
        "above_threshold_pct": float(pct),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepSeek CBD-DFB symmetric-KL routing evaluation")
    parser.add_argument("--original_model_path", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        help="A0: original assistant")
    parser.add_argument("--finetuned_model_path", type=str, required=True,
                        help="A1: unlearned assistant (LoRA adapter dir or full model)")
    parser.add_argument("--valid_data_path", type=str, default="../Data-Collection/deepseek/D_forget.json")
    parser.add_argument("--test_dep_path", type=str, default="../Data-Collection/deepseek/D_test_U_dep.json")
    parser.add_argument("--test_nondep_path", type=str, default="../Data-Collection/deepseek/D_test_U_nondep.json")
    parser.add_argument("--output_dir", type=str, default="artifacts/eval_outputs/deepseek")
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Must match the ratio used at basis extraction / training time")
    parser.add_argument("--seed", type=int, default=42)

    # Where the scored position sits. `probing input new` is the DeepSeek analogue of the
    # author's "Answer:" -- it is cut exactly at the API qualifier (np. / torch. / scipy.integrate.),
    # so the next token IS the deprecated-or-updated API name. `probing input` stops one line
    # earlier, at a newline, which is not a decision point.
    parser.add_argument("--prompt_field", type=str, default="probing input new",
                        choices=["probing input new", "probing input"])
    # The valid split of D_forget contains ONLY deprecated cases, so the negative class for
    # threshold selection has to come from somewhere. See --retain_source.
    parser.add_argument("--retain_source", type=str, default="nondep_holdout",
                        choices=["nondep_holdout", "retain_field"])
    parser.add_argument("--nondep_holdout_frac", type=float, default=0.2,
                        help="Fraction of D_test_U_nondep held out for threshold calibration; "
                             "the reported test_nondep stats use the disjoint remainder")

    # Author's routing knobs (scripts_ref/eval_wmdp_routing.py), same defaults.
    parser.add_argument("--truncate_mode", choices=["left", "head_tail"], default="left")
    parser.add_argument("--score_last_k", type=int, default=1,
                        help="Score over the last K prompt positions instead of just the final one")
    parser.add_argument("--score_last_k_reduce", type=str, default="mean", choices=list(VALID_ROUTING_REDUCERS))
    parser.add_argument("--score_k_mode", choices=["last", "uniform"], default="last")
    parser.add_argument("--score_probe_suffix", type=str, default="",
                        help="Optional fixed suffix appended to the prompt before scoring")
    parser.add_argument("--score_reducer_alpha", type=float, default=DEFAULT_ROUTING_REDUCER_ALPHA)
    parser.add_argument("--score_reducer_beta", type=float, default=DEFAULT_ROUTING_REDUCER_BETA)
    parser.add_argument("--optimize", choices=["accuracy", "gap", "f1", "tpr"], default="accuracy")
    parser.add_argument("--min_tpr", type=float, default=None)
    parser.add_argument("--max_fpr", type=float, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    reducer = normalize_routing_reducer(args.score_last_k_reduce)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_files_only = os.getenv("HF_HUB_OFFLINE") == "1" or os.getenv("TRANSFORMERS_OFFLINE") == "1"

    print("=" * 72)
    print("Loading tokenizer + models...")
    tokenizer = AutoTokenizer.from_pretrained(args.original_model_path, local_files_only=local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    orig_model = load_model_maybe_lora(args.original_model_path, None, device)
    ft_model = load_model_maybe_lora(args.finetuned_model_path, args.original_model_path, device)

    # ---- Positive side of the valid split (every D_forget record is a deprecated case) ----
    raw_data = load_json(args.valid_data_path)
    valid_data = valid_split(raw_data, args.train_ratio, args.seed)
    valid_forget_prompts = extract_prompts(valid_data, args.prompt_field)
    print(f"  valid split: {len(valid_data)}/{len(raw_data)} records, "
          f"forget prompts={len(valid_forget_prompts)} (field={args.prompt_field!r})")

    test_nondep_all = load_json(args.test_nondep_path)

    # ---- Negative side for threshold calibration ----
    if args.retain_source == "retain_field":
        # Cheapest option, but the `retain` field holds COMPLETE functions: 0/9667 end in the
        # API-qualifier position that the forget prompts end in. The resulting threshold partly
        # separates "position type", not "deprecated vs not".
        valid_retain_prompts = extract_prompts(valid_data, "retain", fallback="retain")
        nondep_test = test_nondep_all
        calib_note = "retain field of the valid split (structurally different anchor)"
    else:
        # Hold out a slice of D_test_U_nondep. Same construction as the forget prompts, so the
        # positive and negative classes are structurally identical -- the property the author
        # relies on with WMDP (forget) vs MMLU (retain), both ending in "Answer:".
        idx = list(range(len(test_nondep_all)))
        random.Random(args.seed).shuffle(idx)
        n_cal = int(len(idx) * float(args.nondep_holdout_frac))
        cal_idx = sorted(idx[:n_cal])
        test_idx = sorted(idx[n_cal:])
        valid_retain_prompts = extract_prompts([test_nondep_all[i] for i in cal_idx], args.prompt_field)
        nondep_test = [test_nondep_all[i] for i in test_idx]
        calib_note = (f"{len(cal_idx)}-record holdout of D_test_U_nondep "
                      f"(reported nondep stats use the disjoint {len(test_idx)})")
    print(f"  retain/negative source: {calib_note}")
    print(f"  retain prompts={len(valid_retain_prompts)}")
    if not valid_forget_prompts or not valid_retain_prompts:
        raise ValueError("Valid split produced an empty forget or retain prompt set")

    score_kwargs = dict(
        device=device,
        batch_size=args.batch_size,
        max_len=args.max_len,
        truncate_mode=args.truncate_mode,
        score_last_k=max(1, int(args.score_last_k)),
        score_last_k_reduce=reducer,
        score_k_mode=args.score_k_mode,
        score_probe_suffix=args.score_probe_suffix,
        score_reducer_alpha=args.score_reducer_alpha,
        score_reducer_beta=args.score_reducer_beta,
    )

    print("\nScoring valid split (symmetric KL at last prompt token)...")
    valid_forget = sym_kl_scores(orig_model, ft_model, tokenizer, valid_forget_prompts,
                                 tag="valid/forget", **score_kwargs)
    valid_retain = sym_kl_scores(orig_model, ft_model, tokenizer, valid_retain_prompts,
                                 tag="valid/retain", **score_kwargs)

    selection = select_threshold(valid_forget, valid_retain, args.optimize, args.min_tpr, args.max_fpr)
    threshold = selection["best_threshold"]

    # ---- Test sets ----
    test_dep = load_json(args.test_dep_path)
    dep_prompts = extract_prompts(test_dep, args.prompt_field)
    nondep_prompts = extract_prompts(nondep_test, args.prompt_field)
    print(f"\nTest sets: D_test_U_dep={len(dep_prompts)} D_test_U_nondep={len(nondep_prompts)}")

    dep_scores = sym_kl_scores(orig_model, ft_model, tokenizer, dep_prompts, tag="test/dep", **score_kwargs)
    nondep_scores = sym_kl_scores(orig_model, ft_model, tokenizer, nondep_prompts, tag="test/nondep", **score_kwargs)

    # ---- Report ----
    m = selection["metrics"]
    print("\n" + "=" * 90)
    print("                        SYMMETRIC-KL ROUTING SCORES")
    print("=" * 90)
    print(f"  threshold (from valid): {threshold:.6f}   optimize={args.optimize}"
          f"   constraints_satisfied={selection['constraints_satisfied']}")
    print(f"  valid: acc={m['accuracy']*100:.2f}%  tpr={m['tpr']*100:.2f}%  fpr={m['fpr']*100:.2f}%  f1={m['f1']:.4f}")
    print("-" * 90)
    print(f"  {'dataset':24s} | {'mean':>9s} | {'std':>9s} | {'min':>9s} | {'max':>9s} | {'> thresh':>16s}")
    print("-" * 90)
    stats = {
        "valid_forget": describe("valid forget (deprecated)", valid_forget, threshold),
        "valid_retain": describe("valid retain", valid_retain, threshold),
        "test_dep": describe("D_test_U_dep", dep_scores, threshold),
        "test_nondep": describe("D_test_U_nondep", nondep_scores, threshold),
    }
    print("=" * 90)

    results = {
        "score": "symmetric_kl",
        "score_position": "last_prompt_token",
        "threshold": threshold,
        "selection": selection,
        "config": {
            "original_model_path": args.original_model_path,
            "finetuned_model_path": args.finetuned_model_path,
            "max_len": args.max_len,
            "batch_size": args.batch_size,
            "train_ratio": args.train_ratio,
            "seed": args.seed,
            "prompt_field": args.prompt_field,
            "retain_source": args.retain_source,
            "retain_source_note": calib_note,
            "nondep_holdout_frac": args.nondep_holdout_frac,
            "truncate_mode": args.truncate_mode,
            "score_last_k": max(1, int(args.score_last_k)),
            "score_k_mode": args.score_k_mode,
            "score_probe_suffix": args.score_probe_suffix,
            **routing_reducer_metadata(
                reducer, alpha=args.score_reducer_alpha, beta=args.score_reducer_beta
            ),
        },
        "stats": stats,
        "scores": {
            "valid_forget": valid_forget.tolist(),
            "valid_retain": valid_retain.tolist(),
            "test_dep": dep_scores.tolist(),
            "test_nondep": nondep_scores.tolist(),
        },
    }

    results_path = os.path.join(args.output_dir, "routing_statistics.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to {results_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.hist(dep_scores, bins=60, alpha=0.6, label=f"D_test_U_dep (n={dep_scores.size})", color="#a02f26")
        ax.hist(nondep_scores, bins=60, alpha=0.6, label=f"D_test_U_nondep (n={nondep_scores.size})", color="#0d5d6d")
        ax.axvline(x=threshold, color="black", linestyle="--", linewidth=2, label=f"threshold={threshold:.4f}")
        ax.set_xlabel("routing score: symmetric KL(A0, A1) at last prompt token")
        ax.set_ylabel("count")
        ax.set_title("Routing score distribution: deprecated vs non-deprecated")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plot_path = os.path.join(args.output_dir, "routing_histogram.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Histogram saved to {plot_path}")
    except ImportError:
        print("(matplotlib not available, skipping histogram plot)")


if __name__ == "__main__":
    main()
