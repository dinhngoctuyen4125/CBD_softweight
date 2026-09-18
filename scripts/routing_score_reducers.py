#!/usr/bin/env python3
"""
Shared routing score reducers for advanced_routing experiments.

These helpers keep ToFU fixed-path scoring and WMDP routing on the same
aggregation semantics without changing the default legacy behavior.
"""

from __future__ import annotations

from typing import Dict

import torch


ROUTING_SCORE_SEMANTICS_VERSION = 1
VALID_ROUTING_REDUCERS = ("mean", "max", "cbd", "fsis", "escort", "sces")
DEFAULT_ROUTING_REDUCER_ALPHA = 1.0
DEFAULT_ROUTING_REDUCER_BETA = 1.0
DEFAULT_ROUTING_REDUCER_GAMMA = 1.0
DEFAULT_ROUTING_REDUCER_TOP_M = 4
DEFAULT_ROUTING_REDUCER_EPS = 1e-6


def normalize_routing_reducer(name: str | None) -> str:
    reducer = str(name or "mean").strip().lower()
    if reducer == "feis":
        reducer = "fsis"
    if reducer not in VALID_ROUTING_REDUCERS:
        expected = ", ".join(VALID_ROUTING_REDUCERS)
        raise ValueError(f"Unknown routing reducer: {name!r} (expected one of {expected})")
    return reducer


def routing_entropy_from_logp(logp: torch.Tensor, probs: torch.Tensor | None = None) -> torch.Tensor:
    if probs is None:
        probs = logp.exp()
    return -(probs * logp).sum(dim=-1)


def routing_surprisal_from_actual_tokens(logp: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    return -torch.gather(logp, dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)


def routing_surprisal_from_argmax(logp: torch.Tensor) -> torch.Tensor:
    return -logp.max(dim=-1).values


def routing_reducer_params(
    reducer: str,
    *,
    alpha: float = DEFAULT_ROUTING_REDUCER_ALPHA,
    beta: float = DEFAULT_ROUTING_REDUCER_BETA,
    gamma: float = DEFAULT_ROUTING_REDUCER_GAMMA,
    top_m: int = DEFAULT_ROUTING_REDUCER_TOP_M,
) -> Dict[str, float]:
    reducer = normalize_routing_reducer(reducer)
    params: Dict[str, float] = {}
    if reducer == "escort":
        params["alpha"] = float(alpha)
        params["beta"] = float(beta)
    elif reducer == "sces":
        params["gamma"] = float(gamma)
        params["top_m"] = int(top_m)
    return params


def routing_reducer_metadata(
    reducer: str,
    *,
    alpha: float = DEFAULT_ROUTING_REDUCER_ALPHA,
    beta: float = DEFAULT_ROUTING_REDUCER_BETA,
    gamma: float = DEFAULT_ROUTING_REDUCER_GAMMA,
    top_m: int = DEFAULT_ROUTING_REDUCER_TOP_M,
) -> Dict[str, object]:
    reducer = normalize_routing_reducer(reducer)
    return {
        "routing_reducer": reducer,
        "routing_reducer_params": routing_reducer_params(
            reducer,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            top_m=top_m,
        ),
        "routing_score_semantics_version": ROUTING_SCORE_SEMANTICS_VERSION,
    }


def reduce_routing_scores(
    token_scores: torch.Tensor,
    reducer: str,
    *,
    entropy: torch.Tensor | None = None,
    surprisal: torch.Tensor | None = None,
    alpha: float = DEFAULT_ROUTING_REDUCER_ALPHA,
    beta: float = DEFAULT_ROUTING_REDUCER_BETA,
    gamma: float = DEFAULT_ROUTING_REDUCER_GAMMA,
    top_m: int = DEFAULT_ROUTING_REDUCER_TOP_M,
    eps: float = DEFAULT_ROUTING_REDUCER_EPS,
) -> torch.Tensor:
    reducer = normalize_routing_reducer(reducer)
    squeeze = token_scores.ndim == 1

    if squeeze:
        token_scores = token_scores.unsqueeze(0)
        if entropy is not None:
            entropy = entropy.unsqueeze(0)
        if surprisal is not None:
            surprisal = surprisal.unsqueeze(0)

    if token_scores.ndim < 2:
        raise ValueError(f"token_scores must be 1D or 2D, got shape={tuple(token_scores.shape)}")

    if reducer == "mean":
        reduced = token_scores.mean(dim=-1)
    elif reducer == "max":
        reduced = token_scores.max(dim=-1).values
    elif reducer == "cbd":
        reduced = (token_scores * token_scores).sum(dim=-1) / (token_scores.sum(dim=-1) + float(eps))
    else:
        if entropy is None or surprisal is None:
            raise ValueError(f"Reducer {reducer!r} requires both entropy and surprisal")
        if entropy.shape != token_scores.shape or surprisal.shape != token_scores.shape:
            raise ValueError(
                "entropy/surprisal must match token_scores shape: "
                f"scores={tuple(token_scores.shape)} entropy={tuple(entropy.shape)} surprisal={tuple(surprisal.shape)}"
            )

        saliency = surprisal / (entropy + float(eps))
        if reducer == "fsis":
            weights = torch.softmax(saliency, dim=-1)
            reduced = (weights * token_scores).sum(dim=-1)
        elif reducer == "sces":
            saliency_z = (saliency - saliency.mean(dim=-1, keepdim=True)) / (
                saliency.std(dim=-1, keepdim=True, unbiased=False) + float(eps)
            )
            evidence = torch.log(token_scores.clamp_min(float(eps))) + float(gamma) * saliency_z
            keep = max(1, min(int(top_m), int(token_scores.size(-1))))
            top_idx = torch.topk(evidence, k=keep, dim=-1).indices
            top_scores = token_scores.gather(dim=-1, index=top_idx)
            reduced = top_scores.mean(dim=-1)
        else:
            saliency_z = (saliency - saliency.mean(dim=-1, keepdim=True)) / (
                saliency.std(dim=-1, keepdim=True, unbiased=False) + float(eps)
            )
            reducer_logits = float(alpha) * torch.log(token_scores.clamp_min(float(eps))) + float(beta) * saliency_z
            weights = torch.softmax(reducer_logits, dim=-1)
            reduced = (weights * token_scores).sum(dim=-1)

    if squeeze:
        return reduced.squeeze(0)
    return reduced
