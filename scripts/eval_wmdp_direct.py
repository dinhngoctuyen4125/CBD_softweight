#!/usr/bin/env python3
"""
Direct (non-routing) WMDP + MMLU evaluation for a single model checkpoint.

Implementation note:
- Reuses the routing evaluator's prompt/template/tokenization/choice scoring helpers.
- Runs only one target-model forward path; baseline direct eval does not need routing scores.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer

from eval_wmdp_routing import (
    EvalResult,
    _choice_spec,
    _score_choices_batch,
    _tokenize_prompts,
    ans_letter_idx,
    format_mcq_prompt,
    load_model_maybe_lora,
    read_json,
    read_jsonl,
    data_root,
    repo_root,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def resolve_data_path(path_like: str) -> Path:
    p = Path(path_like)
    if p.is_absolute():
        return p
    return (data_root() / p).resolve()


def resolve_output_path(path_like: str) -> Path:
    p = Path(path_like)
    if p.is_absolute():
        return p
    return (repo_root() / p).resolve()


def resolve_device(name: str) -> torch.device:
    if not name:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def load_eval_model(
    model_path: str,
    base_if_lora: str | None,
    device: torch.device,
    *,
    model_mode: str = "auto",
    uld_weight: float = -0.75,
    uld_top_logit_filter: float = 0.01,
    offset_base_assist_path: str | None = None,
    offset_weight: float = 1.0,
    eval_devices: str | None = None,
):
    if str(model_mode).lower() == "uld":
        if not base_if_lora:
            raise ValueError("--model_base_if_lora is required for --model_mode uld")
        from uld.model import eval_create_uld_model

        mode_cfg = SimpleNamespace(
            weight=float(uld_weight),
            top_logit_filter=float(uld_top_logit_filter),
        )
        if eval_devices:
            mode_cfg.eval_devices = str(eval_devices)
        model = eval_create_uld_model(
            base_model_config=SimpleNamespace(model_path=base_if_lora),
            model_mode_config=mode_cfg,
            ckpt_path=model_path,
            device=device,
        )
        model.eval()
        return model

    local_files_only = os.getenv("HF_HUB_OFFLINE") == "1" or os.getenv("TRANSFORMERS_OFFLINE") == "1"
    try:
        cfg = AutoConfig.from_pretrained(model_path, local_files_only=local_files_only)
    except Exception:
        cfg = None
    mode = str(model_mode).lower()
    if mode == "offset" or (mode == "auto" and cfg is not None and bool(getattr(cfg, "is_offset", False))):
        from uld.model import eval_create_offset_model

        mode_cfg = SimpleNamespace(
            base_assist_path=offset_base_assist_path,
            weight=float(offset_weight),
        )
        if eval_devices:
            mode_cfg.eval_devices = str(eval_devices)
        model = eval_create_offset_model(
            base_model_config=SimpleNamespace(model_path=(base_if_lora or "")),
            model_mode_config=mode_cfg,
            ckpt_path=model_path,
            device=device,
        )
        if model is None:
            raise RuntimeError(f"Failed to construct offset model for {model_path}")
        model.eval()
        return model

    try:
        return load_model_maybe_lora(
            model_path,
            base_if_lora=base_if_lora,
            device=device,
        )
    except Exception as exc:
        if mode in {"direct", "base"}:
            raise exc
        if cfg is None:
            raise exc
        if not bool(getattr(cfg, "is_offset", False)):
            raise exc
        from uld.model import eval_create_offset_model

        mode_cfg = SimpleNamespace(
            base_assist_path=offset_base_assist_path,
            weight=float(offset_weight),
        )
        if eval_devices:
            mode_cfg.eval_devices = str(eval_devices)
        model = eval_create_offset_model(
            base_model_config=SimpleNamespace(model_path=(base_if_lora or "")),
            model_mode_config=mode_cfg,
            ckpt_path=model_path,
            device=device,
        )
        if model is None:
            raise exc
        model.eval()
        return model


@torch.inference_mode()
def eval_mcq_direct(
    tokenizer,
    model,
    prompts,
    labels,
    *,
    name: str,
    truncate_mode: str,
    batch_size: int,
    max_len: int,
    device: torch.device,
    progress_every: int = 0,
) -> dict:
    choice_spec = _choice_spec(tokenizer)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0
    prompt_ids = _tokenize_prompts(tokenizer, prompts)
    result = EvalResult()
    pred_counts = [0, 0, 0, 0]
    gold_counts = [0, 0, 0, 0]
    total_batches = (len(prompt_ids) + max(int(batch_size), 1) - 1) // max(int(batch_size), 1)

    for batch_idx, start in enumerate(range(0, len(prompt_ids), int(batch_size)), start=1):
        end = min(start + int(batch_size), len(prompt_ids))
        scores = _score_choices_batch(
            model,
            prompt_ids[start:end],
            choice_spec,
            int(max_len),
            int(pad_id),
            device,
            str(truncate_mode),
        )
        pred = torch.argmax(scores, dim=-1)
        gold = torch.tensor(labels[start:end], dtype=torch.long)
        for idx in pred.tolist():
            if 0 <= int(idx) < 4:
                pred_counts[int(idx)] += 1
        for idx in gold.tolist():
            if 0 <= int(idx) < 4:
                gold_counts[int(idx)] += 1
        result.add(pred == gold)
        if progress_every and batch_idx % int(progress_every) == 0:
            print(f"[direct-eval] {name}: batch {batch_idx}/{total_batches}", flush=True)

    output = {
        "n": int(result.n),
        "correct_base": int(result.correct),
        "correct_a0": int(result.correct),
        "correct_routed": int(result.correct),
        "acc_base": float(result.acc),
        "acc_a0": float(result.acc),
        "acc_routed": float(result.acc),
        "routed_forget": 0,
        "routed_forget_ratio": 0.0,
        "score_mean": 0.0,
        "score_std": 0.0,
        "mode": "direct_single_model",
    }
    if os.getenv("WMDP_WRITE_PRED_DIAGNOSTICS") == "1":
        pred_ratios = [float(c / result.n) if result.n else 0.0 for c in pred_counts]
        gold_ratios = [float(c / result.n) if result.n else 0.0 for c in gold_counts]
        constant_pred = None
        if result.n:
            for i, count in enumerate(pred_counts):
                if count == result.n:
                    constant_pred = "ABCD"[i]
                    break
        output.update({
            "pred_counts": pred_counts,
            "pred_ratios": pred_ratios,
            "gold_counts": gold_counts,
            "gold_ratios": gold_ratios,
            "constant_pred": constant_pred,
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Target model checkpoint or LoRA adapter path")
    parser.add_argument("--model_base_if_lora", default=None, help="Required when --model_path is a LoRA adapter dir")
    parser.add_argument("--tokenizer_path", default=None, help="Optional tokenizer path (default: model_path)")
    parser.add_argument("--model_mode", choices=["auto", "direct", "base", "offset", "uld"], default="auto")
    parser.add_argument("--uld_weight", type=float, default=-0.75)
    parser.add_argument("--uld_top_logit_filter", type=float, default=0.01)
    parser.add_argument("--offset_base_assist_path", default=None)
    parser.add_argument("--offset_weight", type=float, default=1.0)
    parser.add_argument("--eval_devices", default=None, help="Optional comma-separated devices for composite eval models")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mmlu_test_file", default="eval-method/wmdp/data/mmlu/all_test.jsonl")
    parser.add_argument("--mmlu_subjects", default=None, help="Comma-separated MMLU subjects to keep (optional)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_mmlu", type=int, default=0)
    parser.add_argument("--max_wmdp", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--truncate_mode", choices=["left", "head_tail"], default="left")
    parser.add_argument("--progress_every", type=int, default=0)
    parser.add_argument("--out_json", required=True)
    args = parser.parse_args()

    t0 = time.time()
    device = resolve_device(args.device)
    rng = np.random.default_rng(int(args.seed))

    model = load_eval_model(
        args.model_path,
        base_if_lora=args.model_base_if_lora,
        device=device,
        model_mode=args.model_mode,
        uld_weight=float(args.uld_weight),
        uld_top_logit_filter=float(args.uld_top_logit_filter),
        offset_base_assist_path=args.offset_base_assist_path,
        offset_weight=float(args.offset_weight),
        eval_devices=args.eval_devices,
    )

    local_files_only = os.getenv("HF_HUB_OFFLINE") == "1" or os.getenv("TRANSFORMERS_OFFLINE") == "1"
    tok_name = args.tokenizer_path or args.model_base_if_lora or args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tok_name, local_files_only=local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    wmdp_root = data_root() / "eval-method" / "wmdp" / "data" / "wmdp_mcqs" / "wmdp-mcqs"
    wmdp_files = {
        "bio": ("bio_questions.json", "biology"),
        "cyber": ("cyber_questions.json", "cybersecurity"),
        "chem": ("chem_questions.json", "chemistry"),
    }

    wmdp_results = {}
    for dom_idx, (dom, (fname, subj)) in enumerate(wmdp_files.items()):
        items = read_json(wmdp_root / fname)
        if args.max_wmdp and args.max_wmdp > 0:
            k = int(args.max_wmdp)
            if k < len(items):
                r = np.random.default_rng(int(args.seed) + 1000 + dom_idx)
                idx = r.permutation(len(items))[:k]
                items = [items[i] for i in idx.tolist()]
            else:
                items = items[:k]
        prompts = [format_mcq_prompt(subj, ex["question"], ex["choices"]) for ex in items]
        labels = [ans_letter_idx(ex["answer"]) for ex in items]
        wmdp_results[dom] = eval_mcq_direct(
            tokenizer,
            model,
            prompts,
            labels,
            name=f"wmdp_{dom}",
            truncate_mode=str(args.truncate_mode),
            batch_size=int(args.batch_size),
            max_len=int(args.max_len),
            device=device,
            progress_every=int(args.progress_every),
        )

    w_total_n = int(sum(v["n"] for v in wmdp_results.values()))
    w_total_correct = int(sum(v["correct_routed"] for v in wmdp_results.values()))
    w_total_acc = float(w_total_correct / max(w_total_n, 1))

    mmlu_path = resolve_data_path(args.mmlu_test_file)
    mmlu_rows = read_jsonl(mmlu_path)
    if args.mmlu_subjects:
        keep = {s.strip() for s in str(args.mmlu_subjects).split(",") if s.strip()}
        if keep:
            mmlu_rows = [r for r in mmlu_rows if (r.get("subject") in keep)]
    if args.max_mmlu and args.max_mmlu > 0:
        k = int(args.max_mmlu)
        if k < len(mmlu_rows):
            idx = rng.permutation(len(mmlu_rows))[:k]
            mmlu_rows = [mmlu_rows[i] for i in idx.tolist()]
        else:
            mmlu_rows = mmlu_rows[:k]

    mmlu_prompts = [format_mcq_prompt(r.get("subject") or "general", r["question"], r["choices"]) for r in mmlu_rows]
    mmlu_labels = [ans_letter_idx(r["answer"]) for r in mmlu_rows]
    mmlu_all = eval_mcq_direct(
        tokenizer,
        model,
        mmlu_prompts,
        mmlu_labels,
        name="mmlu_all",
        truncate_mode=str(args.truncate_mode),
        batch_size=int(args.batch_size),
        max_len=int(args.max_len),
        device=device,
        progress_every=int(args.progress_every),
    )

    summary = {
        "model_path": str(args.model_path),
        "model_base_if_lora": str(args.model_base_if_lora) if args.model_base_if_lora else None,
        "model_mode": str(args.model_mode),
        "eval_devices": str(args.eval_devices) if args.eval_devices else None,
        "seed": int(args.seed),
        "max_wmdp": int(args.max_wmdp),
        "max_mmlu": int(args.max_mmlu),
        "mmlu_test_file": str(args.mmlu_test_file),
        "mmlu_subjects": str(args.mmlu_subjects) if args.mmlu_subjects else None,
        "batch_size": int(args.batch_size),
        "max_len": int(args.max_len),
        "truncate_mode": str(args.truncate_mode),
        "wmdp": wmdp_results,
        "wmdp_total_acc": float(w_total_acc),
        "wmdp_total_acc_routed": float(w_total_acc),
        "mmlu_all": mmlu_all,
        "mmlu_all_acc": float(mmlu_all.get("acc_routed", 0.0)),
        "elapsed_sec": float(time.time() - t0),
        "mode": "direct_single_model",
    }

    out_json = resolve_output_path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("out_json", str(out_json))
    print("wmdp_total_acc", float(summary["wmdp_total_acc"]))
    print("mmlu_all_acc", float(summary["mmlu_all_acc"]))
    print("elapsed_sec", float(summary["elapsed_sec"]))


if __name__ == "__main__":
    main()
