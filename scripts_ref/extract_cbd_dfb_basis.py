#!/usr/bin/env python3
"""
Extract CBD-DFB basis (generalized eigen subspace) from gradient statistics.

We build LoRA on the base assistant (A0) and compute per-sample gradients on
forget/retain splits. For each layer, we form G_f and G_r (column-wise) and
solve the generalized eigen problem using a low-rank Woodbury formulation.
"""

import os
import json
import argparse
import pickle
import random
import time
import math
import numpy as np
import torch
import torch.nn.functional as F
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from uld.data.conv_util import create_template


def load_local_tofu(split_name):
    local_tofu_path = os.environ.get("TOFU_DATA_NAME") or os.path.join(
        os.environ.get("CBD_DATA_ROOT", "data"), "TOFU"
    )
    json_file = os.path.join(local_tofu_path, f"{split_name}.json")
    if not os.path.exists(json_file):
        from datasets import load_dataset
        return load_dataset("locuslab/TOFU", split_name)["train"]

    try:
        with open(json_file, "r", encoding="utf-8") as f:
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
            else:
                data = []
                for line in f:
                    if line.strip():
                        data.append(json.loads(line.strip()))
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


def load_wmdp_mcq(domains_csv: str):
    data_root = os.environ.get("CBD_DATA_ROOT", "data")
    domain_map = {"bio": ("bio_questions.json", "biology"), "cyber": ("cyber_questions.json", "cybersecurity"), "chem": ("chem_questions.json", "chemistry")}
    domains = [d.strip().lower() for d in str(domains_csv).split(",") if d.strip()]
    rows = []
    for d in domains:
        if d not in domain_map:
            raise ValueError(f"Unknown WMDP domain: {d!r} (expected one of {sorted(domain_map)})")
        fname, subject = domain_map[d]
        path = os.path.join(data_root, "eval-method", "wmdp", "data", "wmdp_mcqs", "wmdp-mcqs", fname)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for ex in data:
            prompt = _format_mcq_prompt(subject, ex["question"], ex["choices"])
            rows.append({"question": prompt, "answer": _ans_letter(ex["answer"])})
    return datasets.Dataset.from_list(rows)


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


def build_lora_model(base_model_name, r, alpha, dropout, target_modules):
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
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
    # freeze base params
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

    # WMDP/MMLU MCQ: supervise only the last (answer) token to avoid prefix/full
    # tokenization mismatch from whitespace merging (SentencePiece).
    # NOTE: only enable this behavior for MCQ-style templates (strip_prompt=False), to avoid
    # altering other datasets (e.g., ToFU) that use this script.
    mcq_mode = not bool(getattr(conv_template, "strip_prompt", True))
    if mcq_mode and isinstance(answer, str) and answer in {"A", "B", "C", "D"} and isinstance(question, str) and "Answer:" in question:
        labels = torch.full_like(input_ids, -100)
        nonpad = attention_mask.nonzero(as_tuple=False).flatten()
        if nonpad.numel() > 0:
            last_pos = int(nonpad[-1].item())
            labels[last_pos] = input_ids[last_pos]
        return input_ids, attention_mask, labels

    # labels with question masked
    labels = input_ids.clone()
    prefix_num = len(tokenizer(prefix_text, truncation=True, max_length=max_len).input_ids)
    prefix_num = min(prefix_num, labels.size(0))
    labels[:prefix_num] = -100
    # HF CausalLM shifts labels by 1 internally; keep at least one supervised token after shift.
    if (labels[1:] != -100).sum().item() == 0:
        return None
    return input_ids, attention_mask, labels


def _pad_batch(tokenizer, items):
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [t[0] for t in items],
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [t[1] for t in items],
        batch_first=True,
        padding_value=0,
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        [t[2] for t in items],
        batch_first=True,
        padding_value=-100,
    )
    return input_ids, attention_mask, labels


def _compute_per_sample_loss(logits, labels):
    # Match HF CausalLM loss: shift by 1.
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    vocab_size = shift_logits.size(-1)
    loss_flat = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        reduction="none",
    )
    loss_tok = loss_flat.view_as(shift_labels)
    mask = shift_labels != -100
    denom = mask.sum(dim=1).clamp(min=1)
    loss_per_sample = (loss_tok * mask).sum(dim=1) / denom
    return loss_per_sample


def collect_gradients(
    model,
    tokenizer,
    conv_template,
    dataset,
    max_samples,
    max_len,
    batch_size=1,
    store_dtype="float16",
):
    grads = {}
    target_params = []
    for name, param in model.named_parameters():
        if "lora_B" in name and "up_proj" in name and "default" in name:
            target_params.append((name, param))
            grads[name] = []

    device = next(model.parameters()).device
    model.eval()

    total = min(len(dataset), max_samples)
    t0 = time.perf_counter()
    kept = 0
    batch_size = max(int(batch_size), 1)
    param_list = [p for _, p in target_params]

    for start in range(0, total, batch_size):
        batch = []
        end = min(start + batch_size, total)
        for idx in range(start, end):
            sample = dataset[idx]
            prepared = prepare_item(
                tokenizer, conv_template, sample["question"], sample["answer"], max_len
            )
            if prepared is not None:
                batch.append(prepared)

        if not batch:
            continue

        input_ids, attention_mask, labels = _pad_batch(tokenizer, batch)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        model.zero_grad(set_to_none=True)
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        )
        loss_vec = _compute_per_sample_loss(outputs.logits, labels)

        b = loss_vec.size(0)
        # Some torch builds on this machine cannot create eye directly in bfloat16.
        eye = torch.eye(b, device=loss_vec.device, dtype=torch.float32).to(loss_vec.dtype)
        grads_batched = torch.autograd.grad(
            loss_vec,
            param_list,
            grad_outputs=eye,
            is_grads_batched=True,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        for (name, _), g in zip(target_params, grads_batched):
            if g is None:
                continue
            g = g.detach()
            if store_dtype == "float16":
                g = g.to(torch.float16)
            elif store_dtype == "bfloat16":
                g = g.to(torch.bfloat16)
            else:
                g = g.to(torch.float32)
            grads[name].append(g.reshape(b, -1).cpu())

        kept += b
        if kept > 0 and kept % 50 == 0:
            dt = time.perf_counter() - t0
            print(f"[collect_gradients] kept={kept}/{total} avg_sec={dt/kept:.3f}")

    dt = time.perf_counter() - t0
    if kept == 0:
        print("[collect_gradients] WARNING: kept=0 (all samples had no supervised tokens after truncation)")
    else:
        print(f"[collect_gradients] done kept={kept}/{total} sec={dt:.1f} avg_sec={dt/kept:.3f}")
    return grads


def _concat_grad_chunks(chunks):
    if chunks is None:
        return None
    if torch.is_tensor(chunks):
        if chunks.dim() != 2:
            raise ValueError(f"Expected 2D gradient matrix, got shape={tuple(chunks.shape)}")
        return chunks
    if not chunks:
        return None
    first = chunks[0]
    if torch.is_tensor(first) and first.dim() == 2:
        return torch.cat(chunks, dim=0)
    if torch.is_tensor(first) and first.dim() == 1:
        return torch.stack(chunks, dim=0)
    raise ValueError(f"Unsupported gradient chunk type: {type(first)}")


def compute_cbd_dfb_basis(forget_grads, retain_grads, mu, mu_mode, mu_scale, target_variance, top_k):
    basis = {}
    compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if compute_device.type == "cuda":
        if os.environ.get("TRAIN_EXACT_DETERMINISTIC", "0") == "1":
            try:
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                if hasattr(torch, "set_float32_matmul_precision"):
                    torch.set_float32_matmul_precision("highest")
            except Exception:
                pass
        else:
            # Speed up large GEMMs on Ampere+; precision is sufficient for subspace estimation.
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
            except Exception:
                pass
    if compute_device.type == "cpu":
        # Avoid CPU thread oversubscription when running multiple splits in parallel.
        try:
            torch.set_num_threads(1)
        except Exception:
            pass

    for layer_name in forget_grads:
        layer_t0 = time.perf_counter()
        f_mat = _concat_grad_chunks(forget_grads[layer_name])  # [n_f, d]
        r_mat = _concat_grad_chunks(retain_grads.get(layer_name, []))  # [n_r, d]
        if f_mat is None or r_mat is None:
            continue

        # Build G_f and G_r from per-sample flattened gradients.
        # Each sample gradient is a vector in R^d where d = out_dim * r (LoRA-B).
        # We move the heavy linear algebra to GPU when available, otherwise limit CPU threads.
        # Keep compute in float32 for stable eigh/qr.
        G_f = f_mat.to(device=compute_device, dtype=torch.float32).t().contiguous()  # [d, n_f]
        G_r = r_mat.to(device=compute_device, dtype=torch.float32).t().contiguous()  # [d, n_r]

        n_f = G_f.size(1)
        n_r = G_r.size(1)

        # Determine mu (regularization) per layer if requested
        if mu_mode == "auto":
            # trace(F_r) = (1/n_r) * sum(||g||^2)
            trace_fr = G_r.pow(2).sum().item() / float(n_r)
            feature_dim = G_r.size(0)
            mu_layer = mu_scale * (trace_fr / float(feature_dim))
        else:
            mu_layer = mu
        mu_layer = float(mu_layer)
        if (not math.isfinite(mu_layer)) or mu_layer <= 0.0:
            mu_layer = 1e-8

        # Compute small matrices
        K_r = G_r.t().matmul(G_r)  # [n_r, n_r]
        K_rf = G_r.t().matmul(G_f)  # [n_r, n_f]
        eye_r = torch.eye(n_r, dtype=K_r.dtype, device=compute_device)

        # Numerical guard: for some large/ill-conditioned layers, low mu can produce
        # non-finite Z/M while solve itself succeeds. We only escalate mu on failure.
        M = None
        Z = None
        mu_eff = mu_layer
        for _attempt in range(8):
            try:
                A = K_r + (n_r * mu_eff) * eye_r  # [n_r, n_r], SPD
                try:
                    L = torch.linalg.cholesky(A)
                    X = torch.cholesky_solve(K_rf, L)  # [n_r, n_f]
                except RuntimeError:
                    # Fallback to generic solver if Cholesky fails.
                    X = torch.linalg.solve(A, K_rf)

                Z_try = (1.0 / mu_eff) * (G_f - G_r.matmul(X))
                M_try = (G_f.t().matmul(Z_try)) / float(n_f)
                if torch.isfinite(Z_try).all() and torch.isfinite(M_try).all():
                    Z = Z_try
                    M = M_try
                    mu_layer = mu_eff
                    break
            except Exception:
                pass
            mu_eff *= 10.0

        if M is None or Z is None:
            # CPU-float64 fallback for pathological layers; only active on numerical failure.
            G_f_cpu = G_f.detach().to(device="cpu", dtype=torch.float64)
            G_r_cpu = G_r.detach().to(device="cpu", dtype=torch.float64)
            K_r_cpu = G_r_cpu.t().matmul(G_r_cpu)
            K_rf_cpu = G_r_cpu.t().matmul(G_f_cpu)
            eye_r_cpu = torch.eye(n_r, dtype=K_r_cpu.dtype, device="cpu")
            mu_eff_cpu = max(mu_layer, 1e-8)
            ok_cpu = False
            for _attempt in range(10):
                try:
                    A_cpu = K_r_cpu + (n_r * mu_eff_cpu) * eye_r_cpu
                    try:
                        L_cpu = torch.linalg.cholesky(A_cpu)
                        X_cpu = torch.cholesky_solve(K_rf_cpu, L_cpu)
                    except RuntimeError:
                        X_cpu = torch.linalg.solve(A_cpu, K_rf_cpu)
                    Z_cpu = (1.0 / mu_eff_cpu) * (G_f_cpu - G_r_cpu.matmul(X_cpu))
                    M_cpu = (G_f_cpu.t().matmul(Z_cpu)) / float(n_f)
                    if torch.isfinite(Z_cpu).all() and torch.isfinite(M_cpu).all():
                        Z = Z_cpu.to(device=compute_device, dtype=torch.float32)
                        M = M_cpu.to(device=compute_device, dtype=torch.float32)
                        mu_layer = float(mu_eff_cpu)
                        ok_cpu = True
                        break
                except Exception:
                    pass
                mu_eff_cpu *= 10.0
            if not ok_cpu:
                print(f"[compute_cbd_dfb_basis] WARN skip_layer_nonfinite layer={layer_name}")
                del G_f, G_r, K_r, K_rf, eye_r
                if compute_device.type == "cuda":
                    torch.cuda.empty_cache()
                continue

        # eig on M (symmetric PSD in theory). In practice, some PyTorch/CUDA builds may
        # fail to converge for very ill-conditioned inputs (e.g., repeated eigenvalues).
        # Keep the default path unchanged, and only apply a numerical-stability fallback
        # if eigendecomposition fails to converge.
        M_sym = None
        eye_m = None
        try:
            eigvals, eigvecs = torch.linalg.eigh(M)
            if (not torch.isfinite(eigvals).all()) or (not torch.isfinite(eigvecs).all()):
                raise RuntimeError("non_finite_eigh_output")
        except Exception as e:
            print(f"[compute_cbd_dfb_basis] WARN eigh_failed layer={layer_name} err={type(e).__name__}")
            # Ensure exact symmetry (avoid tiny asymmetry from matmul numerics).
            M_sym = (M + M.t()) * 0.5

            scale = float(M_sym.diagonal().abs().max().item())
            if (not math.isfinite(scale)) or scale <= 0.0:
                scale = 1.0

            # Retry with adaptive diagonal jitter (Tikhonov regularization).
            # This only activates when eigh fails, so it won't affect normal runs.
            jitter = scale * 1e-6
            eye_m = torch.eye(M_sym.size(0), dtype=M_sym.dtype, device=compute_device)
            ok = False
            for attempt in range(6):
                try:
                    eigvals, eigvecs = torch.linalg.eigh(M_sym + jitter * eye_m)
                    if (not torch.isfinite(eigvals).all()) or (not torch.isfinite(eigvecs).all()):
                        raise RuntimeError("non_finite_eigh_retry_output")
                    print(
                        f"[compute_cbd_dfb_basis] eigh_retry_ok layer={layer_name} attempt={attempt} jitter={jitter:.3e}"
                    )
                    ok = True
                    break
                except Exception:
                    jitter *= 10.0

            if not ok:
                # CPU fallback in float64 for very ill-conditioned layers.
                # Keep this path strictly conditional on CUDA failures so golden runs stay unchanged.
                M_cpu = M_sym.detach().to(device="cpu", dtype=torch.float64)
                eye_cpu = torch.eye(M_cpu.size(0), dtype=M_cpu.dtype, device="cpu")

                cpu_scale = float(M_cpu.diagonal().abs().max().item())
                if (not math.isfinite(cpu_scale)) or cpu_scale <= 0.0:
                    cpu_scale = 1.0
                cpu_jitter = cpu_scale * 1e-10

                for attempt in range(8):
                    try:
                        eigvals_cpu, eigvecs_cpu = torch.linalg.eigh(M_cpu + cpu_jitter * eye_cpu)
                        if (not torch.isfinite(eigvals_cpu).all()) or (not torch.isfinite(eigvecs_cpu).all()):
                            raise RuntimeError("non_finite_eigh_cpu_output")
                        eigvals = eigvals_cpu.to(device=compute_device, dtype=M.dtype)
                        eigvecs = eigvecs_cpu.to(device=compute_device, dtype=M.dtype)
                        print(
                            f"[compute_cbd_dfb_basis] eigh_cpu_retry_ok layer={layer_name} attempt={attempt} jitter={cpu_jitter:.3e}"
                        )
                        ok = True
                        break
                    except Exception:
                        cpu_jitter *= 10.0

                if not ok:
                    # Final fallback: NumPy LAPACK on CPU.
                    M_np = ((M_cpu + M_cpu.t()) * 0.5).numpy()
                    try:
                        eigvals_np, eigvecs_np = np.linalg.eigh(M_np)
                        if (not np.isfinite(eigvals_np).all()) or (not np.isfinite(eigvecs_np).all()):
                            raise RuntimeError("non_finite_numpy_eigh_output")
                        eigvals = torch.from_numpy(eigvals_np).to(device=compute_device, dtype=M.dtype)
                        eigvecs = torch.from_numpy(eigvecs_np).to(device=compute_device, dtype=M.dtype)
                        ok = True
                        print(f"[compute_cbd_dfb_basis] numpy_eigh_fallback layer={layer_name}")
                    except Exception:
                        # Last-resort SVD on CPU; for symmetric PSD, U are eigenvectors.
                        U_np, S_np, _ = np.linalg.svd(M_np, full_matrices=False)
                        if (not np.isfinite(S_np).all()) or (not np.isfinite(U_np).all()):
                            print(f"[compute_cbd_dfb_basis] WARN skip_layer_nonfinite_svd layer={layer_name}")
                            del G_f, G_r, K_r, K_rf, eye_r, Z, M, M_sym, eye_m
                            if compute_device.type == "cuda":
                                torch.cuda.empty_cache()
                            continue
                        eigvals = torch.from_numpy(S_np).to(device=compute_device, dtype=M.dtype)
                        eigvecs = torch.from_numpy(U_np).to(device=compute_device, dtype=M.dtype)
                        print(f"[compute_cbd_dfb_basis] numpy_svd_fallback layer={layer_name}")

        idx = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # choose k
        if top_k is not None and top_k > 0:
            k = min(top_k, eigvecs.size(1))
        else:
            total = eigvals.clamp(min=0).sum().item()
            if total <= 0:
                k = min(8, eigvecs.size(1))
            else:
                cumsum = torch.cumsum(eigvals.clamp(min=0), dim=0) / total
                k = int((cumsum < target_variance).sum().item())
                if k <= 0:
                    k = 1
        u = eigvecs[:, :k]
        eigvals_k = eigvals[:k].clamp(min=0)
        Q = Z.matmul(u)  # [out_dim, k]

        # Orthonormalize Q
        Q, _ = torch.linalg.qr(Q)
        Q_T = Q.t().contiguous()  # [k, out_dim]

        # For §6.3 trust-region: we need ||G_r^T g_proj|| where g_proj = Q c.
        # Precompute R = G_r^T Q so that G_r^T g_proj = R c.
        retain_proj = G_r.t().matmul(Q).contiguous()  # [n_r, k]

        if (not torch.isfinite(Q_T).all()) or (not torch.isfinite(eigvals_k).all()) or (not torch.isfinite(retain_proj).all()):
            print(f"[compute_cbd_dfb_basis] WARN skip_layer_nonfinite_outputs layer={layer_name}")
            del G_f, G_r, K_r, K_rf, eye_r, Z, M, M_sym, eye_m, eigvals, eigvecs, u, Q, Q_T, retain_proj
            if compute_device.type == "cuda":
                torch.cuda.empty_cache()
            continue

        basis[layer_name] = {
            # Store in float16 to keep basis files small; training will upcast as needed.
            "components": Q_T.detach().to(dtype=torch.float16).cpu(),
            "n_components": int(Q_T.size(0)),
            "mu": float(mu_layer),
            "eigvals": eigvals_k.detach().to(dtype=torch.float32).cpu(),
            # Optional: trust-region helpers (方法描述.md §6.3)
            "retain_proj": retain_proj.detach().to(dtype=torch.float16).cpu(),  # [n_r, k]
            "n_retain": int(n_r),
            "n_forget": int(n_f),
        }

        layer_dt = time.perf_counter() - layer_t0
        print(f"[compute_cbd_dfb_basis] layer={layer_name} k={basis[layer_name]['n_components']} sec={layer_dt:.2f}")

        # Free per-layer tensors early to keep memory bounded.
        del G_f, G_r, K_r, K_rf, eye_r, A, X, Z, M, M_sym, eye_m, eigvals, eigvecs, u, Q, Q_T
        if compute_device.type == "cuda":
            torch.cuda.empty_cache()

    return basis


def main():
    parser = argparse.ArgumentParser(description="Extract CBD-DFB basis from gradients")
    parser.add_argument("--base_model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（用于对齐 LoRA A 初始化）")
    parser.add_argument("--dataset", type=str, default="tofu", choices=["tofu", "wmdp_mcq"], help="数据来源：ToFU(split) 或 WMDP/MMLU(MCQ)")
    parser.add_argument("--forget_split", type=str, default=None, help="ToFU forget split（dataset=tofu 时必填）")
    parser.add_argument("--retain_split", type=str, default=None, help="ToFU retain split（dataset=tofu 时必填）")
    parser.add_argument("--wmdp_domains", type=str, default="bio,cyber", help="WMDP domains for forget (dataset=wmdp_mcq)")
    parser.add_argument("--mmlu_retain_file", type=str, default="eval-method/wmdp/data/mmlu/all_auxiliary_train.jsonl", help="Local MMLU JSONL (run scripts/cache_mmlu.py)")
    parser.add_argument("--mmlu_retain_subjects", type=str, default=None, help="Optional comma-separated subject filter for MMLU retain")
    parser.add_argument("--max_forget", type=int, default=400)
    parser.add_argument("--max_retain", type=int, default=400)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1, help="梯度收集 batch size（>1 会用 per-sample autograd 批量提取）")
    parser.add_argument("--grad_store_dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"], help="CPU 上存储梯度的 dtype（仅影响内存/速度）")
    parser.add_argument("--refuse_forget", action="store_true", help="将 forget 的答案替换为固定拒答（需与训练 data_mode 对齐）")
    parser.add_argument("--refuse_answer", type=str, default="I don't know.", help="refuse_forget 时使用的拒答文本")
    parser.add_argument("--mu", type=float, default=1e-3)
    parser.add_argument("--mu-mode", type=str, default="fixed", choices=["fixed", "auto"])
    parser.add_argument("--mu-scale", type=float, default=1e-2)
    parser.add_argument("--target_variance", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=64)
    parser.add_argument("--output_dir", type=str, required=True)

    # LoRA config
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", type=str, default="up_proj")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if os.environ.get("TRAIN_EXACT_DETERMINISTIC", "0") == "1":
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as exc:
            print(f"[deterministic] torch.use_deterministic_algorithms skipped: {exc}")
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("highest")
        except Exception as exc:
            print(f"[deterministic] tf32 flags skipped: {exc}")
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception as exc:
            print(f"[deterministic] cudnn flags skipped: {exc}")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        print("[deterministic] TRAIN_EXACT_DETERMINISTIC=1")

    # Ensure LoRA A init matches training (freeze_a) for the same seed.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.dataset == "wmdp_mcq":
        # MCQ prompts can be long; keep the suffix containing choices + "Answer:".
        tokenizer.truncation_side = "left"
        conv_template_cfg = {
            "question_start_token": "",
            "question_end_token": "",
            # Add a trailing space so the next-token distribution corresponds to the choice letter.
            "answer_token": " ",
            # Preserve trailing space in answer_token (avoid .strip()).
            "strip_prompt": False,
            "max_len": args.max_len,
        }
    else:
        conv_template_cfg = {
            "question_start_token": "question: ",
            "question_end_token": " answer:",
            "answer_token": "",
            "max_len": args.max_len,
        }
    conv_template = create_template(conv_template_cfg, tokenizer=tokenizer, max_len=args.max_len)

    model = build_lora_model(
        args.base_model_name,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=[args.target_modules],
    )

    # freeze LoRA A for stability (optional)
    for name, param in model.named_parameters():
        if "lora_A" in name:
            param.requires_grad = False

    if args.dataset == "wmdp_mcq":
        mmlu_path = args.mmlu_retain_file
        if not os.path.isabs(mmlu_path):
            mmlu_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), mmlu_path)
        if not os.path.exists(mmlu_path):
            raise FileNotFoundError(
                f"MMLU retain file not found: {mmlu_path}. "
                f"Run: HF_ENDPOINT=https://hf-mirror.com python3 scripts/cache_mmlu.py"
            )
        forget_ds = load_wmdp_mcq(args.wmdp_domains)
        retain_ds = load_mmlu_mcq(mmlu_path, subjects_csv=args.mmlu_retain_subjects)
        forget_tag = f"wmdp_{args.wmdp_domains.replace(',', '_')}"
        retain_tag = "mmlu"
    else:
        if not args.forget_split or not args.retain_split:
            raise ValueError("--forget_split and --retain_split are required when --dataset=tofu")
        forget_ds = load_local_tofu(args.forget_split)
        retain_ds = load_local_tofu(args.retain_split)
        forget_tag = args.forget_split
        retain_tag = args.retain_split

    if args.refuse_forget:
        replaced = []
        for i in range(len(forget_ds)):
            s = dict(forget_ds[i])
            s["answer"] = args.refuse_answer
            replaced.append(s)
        forget_ds = datasets.Dataset.from_list(replaced)

    # use last retain samples like training
    if len(retain_ds) > args.max_retain:
        retain_ds = retain_ds.select(range(len(retain_ds) - args.max_retain, len(retain_ds)))
    if len(forget_ds) > args.max_forget:
        forget_ds = forget_ds.select(range(args.max_forget))

    print(f"Forget samples: {len(forget_ds)}, Retain samples: {len(retain_ds)}")

    t0 = time.perf_counter()
    forget_grads = collect_gradients(
        model,
        tokenizer,
        conv_template,
        forget_ds,
        len(forget_ds),
        args.max_len,
        batch_size=args.batch_size,
        store_dtype=args.grad_store_dtype,
    )
    t_forget = time.perf_counter()
    retain_grads = collect_gradients(
        model,
        tokenizer,
        conv_template,
        retain_ds,
        len(retain_ds),
        args.max_len,
        batch_size=args.batch_size,
        store_dtype=args.grad_store_dtype,
    )
    t_retain = time.perf_counter()

    basis = compute_cbd_dfb_basis(
        forget_grads,
        retain_grads,
        mu=args.mu,
        mu_mode=args.mu_mode,
        mu_scale=args.mu_scale,
        target_variance=args.target_variance,
        top_k=args.top_k,
    )
    t_basis = time.perf_counter()
    print(f"[timing] forget_grads_sec={t_forget - t0:.1f} retain_grads_sec={t_retain - t_forget:.1f} basis_sec={t_basis - t_retain:.1f} total_sec={t_basis - t0:.1f}")

    basis_path = os.path.join(args.output_dir, f"cbd_dfb_basis_{forget_tag}_vs_{retain_tag}.pkl")
    with open(basis_path, "wb") as f:
        pickle.dump(basis, f)

    config = {
        "base_model_name": args.base_model_name,
        "dataset": args.dataset,
        "forget_split": args.forget_split,
        "retain_split": args.retain_split,
        "wmdp_domains": args.wmdp_domains,
        "mmlu_retain_file": args.mmlu_retain_file,
        "mmlu_retain_subjects": args.mmlu_retain_subjects,
        "max_forget": args.max_forget,
        "max_retain": args.max_retain,
        "mu": args.mu,
        "mu_mode": args.mu_mode,
        "mu_scale": args.mu_scale,
        "target_variance": args.target_variance,
        "top_k": args.top_k,
        "batch_size": args.batch_size,
        "grad_store_dtype": args.grad_store_dtype,
        "refuse_forget": bool(args.refuse_forget),
        "refuse_answer": args.refuse_answer,
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": args.target_modules,
        },
        "basis_layers": list(basis.keys()),
    }

    with open(os.path.join(args.output_dir, "basis_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Saved basis to {basis_path}")


if __name__ == "__main__":
    main()
