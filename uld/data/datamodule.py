import copy
import json
import os
import random
from typing import Dict

import torch
from torch.utils.data import DataLoader, Sampler
from transformers import DataCollatorForLanguageModeling
from pytorch_lightning import LightningDataModule
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS

class EqualForgetRetainSampler(Sampler):
    #! This sampler interleaves sample from forget subset (0 - forget_length) and retain subset (forget_length - (forget_length + retain_length))
    # used for deepspeed forget-retain training, otherwise the training would fail for invalidate source error (https://github.com/microsoft/DeepSpeed/discussions/4081)

    def __init__(self, forget_length, retain_length, generator=None):
        self.forget_length = forget_length
        self.retain_length = retain_length
        self.generator = generator
    
    def balanced_interleave(self, shorter, longer):
        if len(shorter) > len(longer):
            shorter, longer = longer, shorter
        if len(shorter) == 0: # no need to interleave
            return torch.tensor(longer)

        ratio = len(longer) / len(shorter)
        result = []
        long_idx = 0
        for s in shorter:
            result.append(s)
            steps = round(ratio)
            result.extend(longer[long_idx:long_idx+steps])
            long_idx += steps
        return torch.tensor(result)

    def __iter__(self):
        forget_indices = torch.randperm(self.forget_length, generator=self.generator)
        retain_indices = torch.randperm(self.retain_length, generator=self.generator) + self.forget_length
        interleaved_indices = self.balanced_interleave(forget_indices, retain_indices).tolist()
        return iter(interleaved_indices)

    def __len__(self):
        return self.forget_length + self.retain_length


class TorchDataset(torch.utils.data.Dataset):
    # conv_template can prepare_gen_prompt or prepare_prompt
    def __init__(
        self,
        data,
        tokenizer,
        conv_template,
        max_length=500,
        forget_length=None,
        retain_length=None,
        dpo_mode=False,
        mcq_last_token_only: bool = False,
    ):
        super(TorchDataset, self).__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.data_collator = DataCollatorForLanguageModeling(mlm=False, tokenizer=self.tokenizer)
        self.conv_template = conv_template
        self.forget_length = forget_length
        self.retain_length = retain_length
        self.max_length = max_length

        self.dpo_mode = dpo_mode
        self.mcq_last_token_only = bool(mcq_last_token_only)
        refusal_path = os.getenv("REFUSAL_DATA_PATH")
        if not refusal_path:
            data_root = os.getenv("CBD_DATA_ROOT", "data")
            refusal_path = os.path.join(data_root, "data", "refusal.jsonl")
        self.alternative_responses = [json.loads(x) for x in open(refusal_path).readlines()]
        cache_env = os.getenv("TOKENIZE_RESULT_CACHE", "1").strip().lower()
        self.enable_cache = (not self.dpo_mode) and cache_env not in {"0", "false", "no", "off"}
        self._item_cache = [None] * len(self.data) if self.enable_cache else None

    def __len__(self):
        return len(self.data)

    def tokenize_text(self, item : Dict):
        prefix_text = self.conv_template.prepare_gen_prompt(**item)
        full_text = self.conv_template.prepare_prompt(**item)
        inputs = self.tokenizer(
            full_text, 
            return_tensors='pt', 
            padding='max_length', 
            max_length=self.max_length, 
            truncation=True
        )
        input_ids = inputs['input_ids'][0]
        attention_mask = inputs['attention_mask'][0]
        collated = self.data_collator([input_ids])

        labels = collated['labels'][0]
        if self.mcq_last_token_only:
            # WMDP/MMLU MCQ: supervise only the last (answer) token to avoid tokenizer-dependent
            # whitespace merging issues between prefix and full prompt (e.g., SentencePiece "▁A").
            answer = item.get("answer")
            question = item.get("question")
            is_mcq = (
                isinstance(answer, str)
                and answer in {"A", "B", "C", "D"}
                and isinstance(question, str)
                and "Answer:" in question
            )
            if is_mcq:
                nonpad = attention_mask.nonzero(as_tuple=False).flatten()
                if nonpad.numel() > 0:
                    last_pos = int(nonpad[-1].item())
                    labels[:] = -100
                    labels[last_pos] = input_ids[last_pos]
                return input_ids, attention_mask, labels

        def prefix_len(text: str) -> int:
            # Keep ToFU behavior unchanged: only apply truncation-aware prefix length
            # when the tokenizer uses left truncation (WMDP/MMLU MCQ).
            if getattr(self.tokenizer, "truncation_side", "right") == "left":
                return len(
                    self.tokenizer(text, truncation=True, max_length=self.max_length).input_ids
                )
            return len(self.tokenizer(text).input_ids)

        if self.tokenizer.padding_side == 'right':
            prefix_num = prefix_len(prefix_text)
            labels[:prefix_num] = -100
        else:
            prefix_num = prefix_len(prefix_text)
            suffix_num = attention_mask.sum().item() - prefix_num
            labels[:-suffix_num] = -100
            
        return input_ids, attention_mask, labels
    
    def __getitem__(self, idx):
        if self._item_cache is not None:
            cached = self._item_cache[idx]
            if cached is not None:
                return {
                    k: (v.clone() if torch.is_tensor(v) else v)
                    for k, v in cached.items()
                }

        item = self.data[idx] # {question: , answer: }
        real_items = [item]
        if self.forget_length is not None:
            retainlabel = 0 if idx < self.forget_length else 1
        else:
            retainlabel = 0

        if self.dpo_mode:
            tempitem = copy.deepcopy(item)
            tempitem['answer'] = random.choice(self.alternative_responses)
            real_items.append(tempitem)
        
        result = {}
        for name, text in zip(["", "prefer_"], real_items):
            input_ids, attention_mask, labels = self.tokenize_text(text)
            result[f"{name}input_ids"] = input_ids
            result[f"{name}attention_mask"] = attention_mask
            result[f"{name}labels"] = labels
        
        result['retainlabels'] = retainlabel
        if self._item_cache is not None:
            self._item_cache[idx] = {
                k: (v.clone() if torch.is_tensor(v) else v)
                for k, v in result.items()
            }
        return result


class TrainDataModule(LightningDataModule):
    def __init__(self, split=None, tokenizer=None, conv_template=None, max_len=1000, batch_size=4, with_retain=False, expand_forget=False, with_perturb=False, with_dpo=False, **kwargs) -> None:
        super().__init__()
        
    def to_torch_dataset(self, data, forget_length=None, retain_length=None, dpo_mode=False):
        torchdataset = TorchDataset(
            data, self.tokenizer, 
            conv_template=self.conv_template, 
            max_length=self.max_len, 
            forget_length=forget_length, 
            retain_length=retain_length,
            dpo_mode=dpo_mode,
            mcq_last_token_only=getattr(self, "mcq_last_token_only", False),
        )
        return torchdataset

    def to_loader(self, data, shuffle=True, **kwargs):
        torchdataset = self.to_torch_dataset(data, **kwargs)
        num_workers = 16
        if getattr(self, "mcq_last_token_only", False):
            num_workers = int(os.getenv("DATA_NUM_WORKERS", "16"))
        return DataLoader(
            torchdataset,
            batch_size=self.batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers,
        )
    
    def train_set(self):
        return self.to_torch_dataset(
            self.forget_data, 
            forget_length=self.forget_length, 
            retain_length=self.retain_length, 
            dpo_mode=self.dpo_mode
        )

    def val_set(self):
        res = {}
        for k, v in self.eval_sets.items():
            res['val_'+k] = self.to_torch_dataset(v)
        return res

    def train_dataloader(self):
        return self.to_loader(self.forget_data, forget_length=self.forget_length, retain_length=self.retain_length, dpo_mode=self.dpo_mode)

    def val_dataloader(self) -> TRAIN_DATALOADERS:
        valsets = self.val_set()
        return [
            self.to_loader(valset, shuffle=False) for valset in valsets.values()
        ]
    
    def stats(self):
        return {
            "train": {"forget num": self.forget_length, "retain num": len(self.forget_train) - self.forget_length, "forget mode": self.forget_train.answer_key, "dpo mode": self.forget_train.as_dpo},
            "val": {
                k: len(v) for k, v in self.val_set()
            }
        }
