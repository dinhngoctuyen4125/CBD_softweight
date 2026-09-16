import json
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import datasets
from omegaconf import ListConfig

from .conv_util import create_template
from .datamodule import TrainDataModule


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _data_root() -> Path:
    return Path(os.environ.get("CBD_DATA_ROOT", "data")).resolve()


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _wmdp_questions_path(domain: str) -> Path:
    root = _data_root()
    domain = domain.lower()
    mapping = {
        "bio": "bio_questions.json",
        "cyber": "cyber_questions.json",
        "chem": "chem_questions.json",
    }
    if domain not in mapping:
        raise ValueError(f"Unknown WMDP domain: {domain!r} (expected one of {sorted(mapping)})")
    return root / "eval-method" / "wmdp" / "data" / "wmdp_mcqs" / "wmdp-mcqs" / mapping[domain]


def _format_mcq_prompt(subject: str, question: str, choices: List[str]) -> str:
    subject = subject.replace("_", " ").strip()
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


def _ans_letter(answer_idx: int) -> str:
    return ["A", "B", "C", "D"][int(answer_idx)]


def _build_wmdp_qa(domains: List[str]) -> List[Dict[str, str]]:
    domain_to_subject = {"bio": "biology", "cyber": "cybersecurity", "chem": "chemistry"}
    rows: List[Dict[str, str]] = []
    for d in domains:
        d = d.lower()
        items = _read_json(_wmdp_questions_path(d))
        subject = domain_to_subject[d]
        for ex in items:
            prompt = _format_mcq_prompt(subject, ex["question"], ex["choices"])
            rows.append({"question": prompt, "answer": _ans_letter(ex["answer"])})
    return rows


def _build_mmlu_qa(jsonl_path: Path, keep_subjects: Optional[Set[str]] = None) -> List[Dict[str, str]]:
    items = _read_jsonl(jsonl_path)
    rows: List[Dict[str, str]] = []
    for ex in items:
        subject = ex.get("subject") or "general"
        if keep_subjects is not None and subject not in keep_subjects:
            continue
        prompt = _format_mcq_prompt(subject, ex["question"], ex["choices"])
        rows.append({"question": prompt, "answer": _ans_letter(ex["answer"])})
    return rows


def _sample(rows: List[Dict[str, str]], n: Optional[int], seed: int) -> List[Dict[str, str]]:
    if n is None or n <= 0 or n >= len(rows):
        if n is None or n <= 0 or n == len(rows) or len(rows) == 0:
            return rows
        # Oversample with replacement (useful when retain sets are small but we want a stronger retain signal).
        rng = random.Random(int(seed))
        return [rng.choice(rows) for _ in range(int(n))]
    rng = random.Random(int(seed))
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    idx = idx[: int(n)]
    return [rows[i] for i in idx]


class WMDP_DataModule(TrainDataModule):
    """
    WMDP MCQ + MMLU MCQ datamodule for training the assistant A1.

    - forget split: `split` string like "bio_cyber" (default paper setting) or "bio_cyber_chem".
    - retain set: local JSONL exported by `scripts/cache_mmlu.py`.
    """

    def __init__(
        self,
        split,
        tokenizer,
        conv_template_config,
        max_len=512,
        batch_size=4,
        with_retain=True,
        retain_num=2400,
        with_dpo=False,
        expand_forget=False,
        with_perturb=False,
        **kwargs,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.max_len = int(max_len)
        self.batch_size = int(batch_size)
        self.dpo_mode = bool(with_dpo)
        self.conv_template = create_template(conv_template_config, tokenizer=tokenizer)
        self.mcq_last_token_only = True

        # WMDP/MMLU MCQ prompts can be long; we must keep the *suffix* containing choices and "Answer:".
        # Use left truncation so the answer region is preserved under max_len.
        self.tokenizer.truncation_side = "left"

        # Parse forget domains from split like "bio_cyber".
        split = str(split or "bio_cyber")
        domains = [p for p in split.split("_") if p]
        for d in domains:
            if d.lower() not in {"bio", "cyber", "chem"}:
                raise ValueError(f"Invalid WMDP split token {d!r} in split={split!r}")
        self.domains = [d.lower() for d in domains]

        seed = int(kwargs.get("seed", 42))
        max_forget = kwargs.get("max_forget", None)
        if max_forget is not None:
            max_forget = int(max_forget)

        mmlu_retain_file = kwargs.get("mmlu_retain_file", "eval-method/wmdp/data/mmlu/all_auxiliary_train.jsonl")
        mmlu_retain_subjects = kwargs.get("mmlu_retain_subjects", None)
        keep_subjects: Optional[Set[str]] = None
        if mmlu_retain_subjects is not None:
            if isinstance(mmlu_retain_subjects, (list, tuple, set, ListConfig)):
                keep_subjects = {str(s).strip() for s in list(mmlu_retain_subjects) if str(s).strip()}
            elif str(mmlu_retain_subjects).strip():
                keep_subjects = {s.strip() for s in str(mmlu_retain_subjects).split(",") if s.strip()}
        mmlu_retain_file = (_data_root() / mmlu_retain_file).resolve() if not str(mmlu_retain_file).startswith("/") else Path(mmlu_retain_file)
        if not mmlu_retain_file.exists():
            raise FileNotFoundError(
                f"MMLU retain file not found: {mmlu_retain_file}. "
                f"Run `HF_ENDPOINT=https://hf-mirror.com python3 scripts/cache_mmlu.py` first."
            )

        # Build forget and retain QA pairs.
        forget_rows = _build_wmdp_qa(self.domains)
        forget_rows = _sample(forget_rows, max_forget, seed=seed)
        self.forget_length = len(forget_rows)

        retain_rows: List[Dict[str, str]] = []
        if with_retain:
            retain_rows = _build_mmlu_qa(mmlu_retain_file, keep_subjects=keep_subjects)
            retain_rows = _sample(retain_rows, int(retain_num) if retain_num is not None else None, seed=seed + 1)
        self.retain_length = len(retain_rows)

        self.forget_data = datasets.concatenate_datasets(
            [datasets.Dataset.from_list(forget_rows), datasets.Dataset.from_list(retain_rows)]
        )

        # Minimal eval sets (not used when eval_strategy=no).
        self.eval_sets = {
            "forget": datasets.Dataset.from_list(forget_rows[: min(128, len(forget_rows))]),
            "retain": datasets.Dataset.from_list(retain_rows[: min(128, len(retain_rows))]),
        }

        print(
            f"[WMDP_DataModule] forget_domains={self.domains} "
            f"train_forget={self.forget_length} train_retain={self.retain_length} "
            f"max_len={self.max_len} bs={self.batch_size}"
        )
