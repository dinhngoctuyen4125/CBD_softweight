import json
import os
import random
from typing import Dict, List

from .conv_util import create_template
from .datamodule import TrainDataModule


class DeepSeek_DataModule(TrainDataModule):
    """Forget/retain data for the DeepSeek deprecated-API unlearning task.

    Every record carries one prompt with two competing continuations:

        forget : `probing input` + `y_neg`   (the deprecated API call)
        retain : `probing input` + `y_pos`   (the updated API call)

    This mirrors ``scripts/extract_cbd_dfb_basis.py`` deliberately. The basis Q and the
    training gradients have to live in the same space, so the split, the field names and
    the conversation template must not drift apart between the two stages.

    NOTE on ordering: ``TorchDataset.__getitem__`` derives the forget/retain label purely
    from the index (``retainlabel = 0 if idx < forget_length else 1``). Concatenating in
    the wrong order silently swaps the forget and retain losses without raising anything,
    so ``forget`` must come first.
    """

    def __init__(
        self,
        tokenizer,
        conv_template_config,
        data_path="../Data-Collection/deepseek/D_forget.json",
        train_ratio=1.0,
        split_seed=42,
        max_len=512,
        batch_size=4,
        with_retain=True,
        retain_num=-1,
        max_forget=-1,
        with_dpo=False,
        **kwargs,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.batch_size = batch_size
        self.dpo_mode = bool(with_dpo)
        self.conv_template = create_template(
            conv_template_config, tokenizer=tokenizer, max_len=max_len
        )

        records = self._load_train_records(data_path, train_ratio, split_seed)

        forget = [
            {"question": r["probing input"], "answer": r["y_neg"]}
            for r in records
            if r.get("probing input") and r.get("y_neg")
        ]
        retain = [
            {"question": r["probing input"], "answer": r["y_pos"]}
            for r in records
            if r.get("probing input") and r.get("y_pos")
        ]

        if max_forget is not None and int(max_forget) > 0:
            forget = forget[: int(max_forget)]
        if not with_retain:
            retain = []
        elif retain_num is not None and int(retain_num) > 0:
            # Take the same leading slice as the forget side so y_neg and y_pos stay paired
            # per record; the generalized eigen problem assumes identical prompts on both
            # sides, differing only in the continuation.
            retain = retain[: int(retain_num)]

        self.forget_length = len(forget)
        self.retain_length = len(retain)
        self.forget_data = forget + retain

        # train_ratio=1.0 holds nothing out, so there is no internal validation set.
        # Run stage 2 with DISABLE_INTERNAL_EVAL=1; hf_forget_train.py still calls
        # val_set(), which walks this dict and returns {}.
        self.eval_sets = {}

        print(
            f"[DeepSeek] forget={self.forget_length} retain={self.retain_length} "
            f"total={len(self.forget_data)} max_len={self.max_len}"
        )

    @staticmethod
    def _load_train_records(data_path, train_ratio, split_seed) -> List[Dict]:
        if not os.path.isabs(data_path):
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            for cand in (
                data_path,
                os.path.join(os.getcwd(), data_path),
                os.path.join(repo_root, data_path),
            ):
                if os.path.exists(cand):
                    data_path = cand
                    break
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"DeepSeek data file not found: {data_path}")

        print(f"[DeepSeek] Loading data from {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # Byte-identical to extract_cbd_dfb_basis.py: shuffle with the same seed, keep the
        # head, then restore the original order. train_ratio=1.0 keeps every record.
        n_train = int(len(raw) * float(train_ratio))
        indices = list(range(len(raw)))
        random.Random(split_seed).shuffle(indices)
        train_indices = sorted(indices[:n_train])
        print(
            f"[DeepSeek] train split: {len(train_indices)}/{len(raw)} "
            f"(train_ratio={train_ratio}, split_seed={split_seed})"
        )
        return [raw[i] for i in train_indices]
