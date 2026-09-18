#!/usr/bin/env python3
"""
Cache the MMLU dataset locally as JSONL to avoid repeated HF downloads and to
avoid the repo-local `datasets.py` shim shadowing the HuggingFace `datasets` pkg.

Usage:
  HF_ENDPOINT=https://hf-mirror.com python3 scripts/cache_mmlu.py

Outputs:
  eval-method/wmdp/data/mmlu/all_<split>.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable, Mapping


def _import_hf_datasets():
    """
    Import HuggingFace `datasets` even when this repo's root contains a
    `datasets.py` shim.
    """
    repo_root = Path(__file__).resolve().parents[1]
    clean_sys_path = []
    for p in sys.path:
        if p in ("", str(repo_root)):
            continue
        clean_sys_path.append(p)
    sys.path = clean_sys_path

    import datasets  # type: ignore

    return datasets


def _write_jsonl(path: Path, rows: Iterable[Mapping]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Repo-relative output dir (default: eval-method/wmdp/data/mmlu)",
    )
    parser.add_argument(
        "--splits",
        default="test,validation,dev,auxiliary_train",
        help="Comma-separated splits to export",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing jsonl files",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir) if args.out_dir else (repo_root / "eval-method" / "wmdp" / "data" / "mmlu")
    splits = [s.strip() for s in str(args.splits).split(",") if s.strip()]

    hf_datasets = _import_hf_datasets()
    load_dataset = hf_datasets.load_dataset

    print("[cache_mmlu] Using HF_ENDPOINT=", os.environ.get("HF_ENDPOINT"))
    print("[cache_mmlu] Loading cais/mmlu (all)...")
    ds_dict = load_dataset("cais/mmlu", "all")

    for split in splits:
        if split not in ds_dict:
            raise ValueError(f"Split {split!r} not found in cais/mmlu: {list(ds_dict.keys())}")
        out_path = out_dir / f"all_{split}.jsonl"
        if out_path.exists() and not args.force:
            print(f"[cache_mmlu] Skip existing: {out_path}")
            continue

        ds = ds_dict[split]
        print(f"[cache_mmlu] Exporting split={split} rows={len(ds)} -> {out_path}")

        def rows():
            for ex in ds:
                yield {
                    "question": ex["question"],
                    "choices": ex["choices"],
                    "answer": ex["answer"],
                    "subject": ex.get("subject"),
                }

        _write_jsonl(out_path, rows())

    print("[cache_mmlu] Done.")


if __name__ == "__main__":
    main()

