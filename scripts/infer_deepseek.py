#!/usr/bin/env python3
"""
Inference & weight statistics for DeepSeek CBD-DFB experiments.

Computes routing scores (CE difference) between the original and finetuned
TinyLlama models on:
  1. Valid split of D_forget.json (20%) → find optimal threshold
  2. D_test_U_dep.json → expect HIGH scores (deprecated API usage)
  3. D_test_U_nondep.json → expect LOW scores (non-deprecated)
"""

import os
import sys
import json
import argparse
import random
import math
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_ce_scores(model, tokenizer, texts, max_len=512, batch_size=4):
    """
    Compute per-sample cross-entropy (auto-regressive loss) on a list of texts.
    Returns a list of float CE values.
    """
    model.eval()
    device = next(model.parameters()).device
    ce_scores = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        # Compute per-sample CE: shift logits and labels
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous()

        vocab_size = shift_logits.size(-1)
        loss_flat = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            reduction="none",
        )
        loss_tok = loss_flat.view_as(shift_labels)

        # Mask out padding tokens
        masked_loss = loss_tok * shift_mask
        denom = shift_mask.sum(dim=1).clamp(min=1)
        per_sample_ce = (masked_loss.sum(dim=1) / denom).cpu().tolist()
        ce_scores.extend(per_sample_ce)

        if len(ce_scores) % 200 < batch_size:
            print(f"  [{len(ce_scores)}/{len(texts)}] computed...")

    return ce_scores


def find_optimal_threshold(
    forget_scores, retain_scores, n_steps=1000
):
    """
    Find threshold that maximizes accuracy on valid set.
    Forget samples should have score > threshold (positive),
    Retain samples should have score <= threshold (negative).
    """
    all_scores = forget_scores + retain_scores
    lo = min(all_scores)
    hi = max(all_scores)

    best_threshold = (lo + hi) / 2
    best_acc = 0.0

    for i in range(n_steps + 1):
        t = lo + (hi - lo) * i / n_steps
        # forget should be above threshold, retain below
        tp = sum(1 for s in forget_scores if s > t)
        tn = sum(1 for s in retain_scores if s <= t)
        acc = (tp + tn) / (len(forget_scores) + len(retain_scores))
        if acc > best_acc:
            best_acc = acc
            best_threshold = t

    return best_threshold, best_acc


def print_statistics(name, scores, threshold):
    """Print statistics for a dataset's scores."""
    scores_arr = np.array(scores)
    above = np.sum(scores_arr > threshold)
    total = len(scores_arr)
    pct = 100.0 * above / total if total > 0 else 0.0

    print(f"  {name:25s} | {scores_arr.mean():8.4f} | {scores_arr.std():8.4f} | "
          f"{scores_arr.min():8.4f} | {scores_arr.max():8.4f} | "
          f"{above:5d}/{total:5d} ({pct:5.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="DeepSeek CBD-DFB inference & weight statistics")
    parser.add_argument("--original_model_path", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        help="Path to the original (pre-unlearning) model")
    parser.add_argument("--finetuned_model_path", type=str, required=True,
                        help="Path to the finetuned (post-unlearning) LoRA checkpoint")
    parser.add_argument("--valid_data_path", type=str, default="data/deepseek/D_forget.json",
                        help="Path to D_forget.json (will use 20%% as valid)")
    parser.add_argument("--test_dep_path", type=str, default="data/deepseek/D_test_U_dep.json",
                        help="Path to D_test_U_dep.json")
    parser.add_argument("--test_nondep_path", type=str, default="data/deepseek/D_test_U_nondep.json",
                        help="Path to D_test_U_nondep.json")
    parser.add_argument("--output_dir", type=str, default="artifacts/eval_outputs/deepseek",
                        help="Output directory for results")
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Must match the ratio used during training to get correct valid split")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load tokenizer ──
    print("=" * 60)
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.original_model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load original model ──
    print("Loading original model...")
    original_model = AutoModelForCausalLM.from_pretrained(
        args.original_model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to(device)
    original_model.eval()

    # ── Load finetuned model (base + LoRA) ──
    print("Loading finetuned model (LoRA)...")
    finetuned_base = AutoModelForCausalLM.from_pretrained(
        args.original_model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to(device)
    finetuned_model = PeftModel.from_pretrained(finetuned_base, args.finetuned_model_path)
    finetuned_model = finetuned_model.merge_and_unload()
    finetuned_model.eval()

    # ── Prepare valid set from D_forget.json ──
    print(f"\nLoading valid data from {args.valid_data_path}...")
    raw_data = load_json(args.valid_data_path)
    n_total = len(raw_data)
    n_train = int(n_total * args.train_ratio)

    # Reproduce the same split as training
    indices = list(range(n_total))
    rng = random.Random(args.seed)
    rng.shuffle(indices)
    valid_indices = sorted(indices[n_train:])
    valid_data = [raw_data[i] for i in valid_indices]
    print(f"  Valid split: {len(valid_data)} samples")

    # Extract probing inputs for valid set
    valid_forget_texts = [item["probing input"] for item in valid_data if item.get("probing input") and item.get("y_neg")]
    valid_retain_texts = [item["probing input"] for item in valid_data if item.get("probing input") and item.get("y_pos")]

    # ── Compute CE scores on valid set ──
    print("\n" + "=" * 60)
    print("Computing CE scores on valid set (original model)...")
    valid_forget_ce_orig = compute_ce_scores(original_model, tokenizer, valid_forget_texts, args.max_len, args.batch_size)
    valid_retain_ce_orig = compute_ce_scores(original_model, tokenizer, valid_retain_texts, args.max_len, args.batch_size)

    print("Computing CE scores on valid set (finetuned model)...")
    valid_forget_ce_ft = compute_ce_scores(finetuned_model, tokenizer, valid_forget_texts, args.max_len, args.batch_size)
    valid_retain_ce_ft = compute_ce_scores(finetuned_model, tokenizer, valid_retain_texts, args.max_len, args.batch_size)

    # Routing score = CE_finetuned - CE_original
    # For forget data: finetuned model should have HIGHER CE (more "surprised") → positive score
    valid_forget_scores = [ft - orig for ft, orig in zip(valid_forget_ce_ft, valid_forget_ce_orig)]
    valid_retain_scores = [ft - orig for ft, orig in zip(valid_retain_ce_ft, valid_retain_ce_orig)]

    # ── Find optimal threshold ──
    threshold, valid_acc = find_optimal_threshold(valid_forget_scores, valid_retain_scores)
    print(f"\n  Optimal threshold: {threshold:.4f}")
    print(f"  Valid accuracy:    {valid_acc * 100:.1f}%")

    # ── Load test datasets ──
    print("\n" + "=" * 60)
    print("Loading test datasets...")
    test_dep_data = load_json(args.test_dep_path)
    test_nondep_data = load_json(args.test_nondep_path)
    print(f"  D_test_U_dep:    {len(test_dep_data)} samples")
    print(f"  D_test_U_nondep: {len(test_nondep_data)} samples")

    test_dep_texts = [item["probing input"] for item in test_dep_data if item.get("probing input")]
    test_nondep_texts = [item["probing input"] for item in test_nondep_data if item.get("probing input")]

    # ── Compute scores on test sets ──
    print("\nComputing CE scores on D_test_U_dep (original model)...")
    dep_ce_orig = compute_ce_scores(original_model, tokenizer, test_dep_texts, args.max_len, args.batch_size)
    print("Computing CE scores on D_test_U_dep (finetuned model)...")
    dep_ce_ft = compute_ce_scores(finetuned_model, tokenizer, test_dep_texts, args.max_len, args.batch_size)
    dep_scores = [ft - orig for ft, orig in zip(dep_ce_ft, dep_ce_orig)]

    print("\nComputing CE scores on D_test_U_nondep (original model)...")
    nondep_ce_orig = compute_ce_scores(original_model, tokenizer, test_nondep_texts, args.max_len, args.batch_size)
    print("Computing CE scores on D_test_U_nondep (finetuned model)...")
    nondep_ce_ft = compute_ce_scores(finetuned_model, tokenizer, test_nondep_texts, args.max_len, args.batch_size)
    nondep_scores = [ft - orig for ft, orig in zip(nondep_ce_ft, nondep_ce_orig)]

    # ── Print statistics ──
    print("\n" + "=" * 70)
    print("                         SCORE STATISTICS")
    print("=" * 70)
    print(f"  Threshold (from valid set): {threshold:.4f}")
    print(f"  Valid accuracy:             {valid_acc * 100:.1f}%")
    print("-" * 70)
    print(f"  {'Dataset':25s} | {'Mean':>8s} | {'Std':>8s} | "
          f"{'Min':>8s} | {'Max':>8s} | {'> Thresh':>15s}")
    print("-" * 70)
    print_statistics("Valid (forget/y_neg)", valid_forget_scores, threshold)
    print_statistics("Valid (retain/y_pos)", valid_retain_scores, threshold)
    print("-" * 70)
    print_statistics("D_test_U_dep", dep_scores, threshold)
    print_statistics("D_test_U_nondep", nondep_scores, threshold)
    print("=" * 70)

    # ── Save detailed results ──
    results = {
        "threshold": threshold,
        "valid_accuracy": valid_acc,
        "valid_forget_scores": valid_forget_scores,
        "valid_retain_scores": valid_retain_scores,
        "test_dep_scores": dep_scores,
        "test_nondep_scores": nondep_scores,
        "test_dep_mean": float(np.mean(dep_scores)),
        "test_dep_std": float(np.std(dep_scores)),
        "test_dep_above_threshold": int(np.sum(np.array(dep_scores) > threshold)),
        "test_dep_total": len(dep_scores),
        "test_nondep_mean": float(np.mean(nondep_scores)),
        "test_nondep_std": float(np.std(nondep_scores)),
        "test_nondep_above_threshold": int(np.sum(np.array(nondep_scores) > threshold)),
        "test_nondep_total": len(nondep_scores),
    }

    results_path = os.path.join(args.output_dir, "weight_statistics.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to {results_path}")

    # ── Optional: histogram plot ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.hist(dep_scores, bins=50, alpha=0.6, label=f"D_test_U_dep (n={len(dep_scores)})", color="red")
        ax.hist(nondep_scores, bins=50, alpha=0.6, label=f"D_test_U_nondep (n={len(nondep_scores)})", color="blue")
        ax.axvline(x=threshold, color="black", linestyle="--", linewidth=2, label=f"Threshold={threshold:.4f}")
        ax.set_xlabel("Routing Score (CE_finetuned - CE_original)")
        ax.set_ylabel("Count")
        ax.set_title("Score Distribution: Deprecated vs Non-deprecated")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plot_path = os.path.join(args.output_dir, "score_histogram.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Histogram saved to {plot_path}")
    except ImportError:
        print("(matplotlib not available, skipping histogram plot)")


if __name__ == "__main__":
    main()
