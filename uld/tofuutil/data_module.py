#! Adapted from ToFU repo
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import datasets
import os
import json
from functools import lru_cache
from .utils import get_model_identifiers_from_yaml


@lru_cache(maxsize=64)
def _load_tofu_dataset_cached(data_path, split):
    """Cache raw dataset loads within one process to avoid repeated local JSONL reads."""
    if os.environ.get("TOFU_DATASET_LOAD_MODE", "").strip().lower() == "legacy":
        if split and (data_path == "locuslab/TOFU" or os.path.isdir(str(data_path))):
            local_tofu_path = str(data_path) if os.path.isdir(str(data_path)) else "TOFU"
            json_file = os.path.join(local_tofu_path, f"{split}.json")

            if os.path.exists(json_file):
                print(f"从本地加载TOFU数据: {json_file}")
                with open(json_file, 'r', encoding='utf-8') as f:
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
                        return datasets.Dataset.from_list(data)

                    print("检测到JSONL格式（每行一个JSON对象），按行读取...")
                    data = []
                    for line in f:
                        if line.strip():
                            data.append(json.loads(line.strip()))
                    return datasets.Dataset.from_list(data)

            print(f"本地文件 {json_file} 不存在，尝试从网络加载...")

        return datasets.load_dataset(data_path, split)["train"]

    if split:
        local_tofu_candidates = []
        for candidate in (data_path, os.environ.get("TOFU_DATA_NAME"), "TOFU"):
            if candidate and candidate not in local_tofu_candidates:
                local_tofu_candidates.append(candidate)

        json_file = None
        for local_tofu_path in local_tofu_candidates:
            candidate_file = os.path.join(local_tofu_path, f"{split}.json")
            if os.path.exists(candidate_file):
                json_file = candidate_file
                break

        if json_file is not None:
            print(f"从本地加载TOFU数据: {json_file}")
            # These files often use JSONL despite the .json suffix.
            with open(json_file, 'r', encoding='utf-8') as f:
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
                    return datasets.Dataset.from_list(data)

                print("检测到JSONL格式（每行一个JSON对象），按行读取...")
                data = []
                for line in f:
                    if line.strip():
                        data.append(json.loads(line.strip()))
                return datasets.Dataset.from_list(data)

        if data_path == "locuslab/TOFU":
            searched = ", ".join(os.path.join(p, f"{split}.json") for p in local_tofu_candidates)
            print(f"本地文件不存在（查找过: {searched}），尝试从网络加载...")

    return datasets.load_dataset(data_path, split)["train"]

class TextDatasetQA(Dataset):
    def __init__(
        self,
        data_path,
        tokenizer,
        conv_template,
        split=None,
        question_key='question',
        answer_key='answer',
        max_num=-1,
    ):
        super(TextDatasetQA, self).__init__()
        self.conv_template = conv_template
        self.tokenizer = tokenizer

        # 尝试从本地TOFU数据加载
        self.data = self._load_dataset(data_path, split)

        if max_num != -1:
            self.data = self.data.select(range(min(len(self.data), max_num)))
        self.qk = question_key
        self.ak = answer_key

        self.qk_candidates = self._normalize_key_candidates(question_key, "question")
        self.ak_candidates = self._normalize_key_candidates(answer_key, "answer")

        # Some local TOFU eval splits do not carry every optional field
        # (for example `paraphrased_answer` on auxiliary splits). Resolve a
        # stable fallback order once, and keep the same order available for the
        # per-sample fallback in __getitem__.
        columns = set(getattr(self.data, "column_names", []) or [])
        self.qk = self._resolve_existing_key(columns, self.qk_candidates, "question")
        self.ak = self._resolve_existing_key(columns, self.ak_candidates, "answer")

    @staticmethod
    def _normalize_key_candidates(raw_key, fallback_key):
        if isinstance(raw_key, (list, tuple)):
            keys = [str(k) for k in raw_key if k]
        elif raw_key:
            keys = [str(raw_key)]
        else:
            keys = []
        if fallback_key and fallback_key not in keys:
            keys.append(fallback_key)
        return keys

    @staticmethod
    def _resolve_existing_key(columns, candidates, role):
        for key in candidates:
            if key in columns:
                return key
        if candidates:
            print(f"[tofu-eval] {role} field fallback unresolved in column_names, keep first candidate: {candidates[0]} candidates={candidates}")
            return candidates[0]
        raise KeyError(f"[tofu-eval] no candidate field configured for role={role}")

    @staticmethod
    def _get_sample_value(sample, candidates, role):
        for key in candidates:
            if key in sample:
                return sample[key]
        raise KeyError(f"[tofu-eval] missing {role} field, candidates={candidates}, sample_keys={list(sample.keys())}")

    def _load_dataset(self, data_path, split):
        """加载数据集，优先使用本地TOFU数据，并缓存原始读取结果。"""
        return _load_tofu_dataset_cached(data_path, split)

    def __len__(self):
        return len(self.data)
    
    def prepare_input_ids(self, 
        question, answer,  
        question_start_token, question_end_token, answer_token, 
        tokenizer=None, max_len=None
    ):
        tokenizer = tokenizer
        #! Important about the format
        new_question = question_start_token + " " + question + " " + question_end_token
        new_answer = answer_token + answer
        full_text = new_question + new_answer
        num_question_tokens = len(tokenizer.tokenize(new_question, add_special_tokens=True))
        encoded = tokenizer(
            full_text, 
            add_special_tokens=True, 
            max_length=max_len, 
            truncation=True, 
        )
        pad_length = max_len - len(encoded.input_ids)
        pad_input_ids = encoded['input_ids'] + [tokenizer.eos_token_id] * pad_length
        pad_attention_mask = encoded['attention_mask'] + [0] * pad_length
        if len(encoded.input_ids) == max_len:
            label = encoded.input_ids
        else:
            label = encoded['input_ids'] + [tokenizer.eos_token_id] + [-100] * (pad_length-1)

        # change label to -100 for question tokens
        label = torch.tensor(label)
        label[:num_question_tokens] = -100
        return (
            torch.tensor(pad_input_ids),
            label,
            torch.tensor(pad_attention_mask),
        )    

    def __getitem__(self, idx):
        sample = self.data[idx]
        question = self._get_sample_value(sample, self.qk_candidates, "question")
        answers = self._get_sample_value(sample, self.ak_candidates, "answer")

        if isinstance(answers, str):
            answers = [answers]

        pad_input_ids_list = []
        label_list = []
        pad_attention_mask_list = []

        for answer in answers:
            tensor_data = self.prepare_input_ids(
                question, answer, tokenizer=self.tokenizer, max_len=self.conv_template.max_len,
                question_start_token=self.conv_template.question_start_token, question_end_token=self.conv_template.question_end_token, answer_token=self.conv_template.answer_token,
            )
            pad_input_ids_list.append(tensor_data[0])
            label_list.append(tensor_data[1])
            pad_attention_mask_list.append(tensor_data[2])

        return (
            torch.stack(pad_input_ids_list).squeeze(),
            torch.stack(label_list).squeeze(),
            torch.stack(pad_attention_mask_list).squeeze()
        )
    

def collate_fn(batch):
    input_ids, attention_masks = zip(*batch)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=-100)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    return input_ids, attention_masks

def custom_data_collator(samples):
    input_ids = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    attention_mask = [s[2] for s in samples]
    return torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)

def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()
    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum loss for each sequence in a batch
    loss = loss_function(output.transpose(-1,-2), shifted_labels).sum(dim=-1)
    return loss
