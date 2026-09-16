"""Compatibility wrapper for the HuggingFace ``datasets`` package.

This repository historically carried a tiny local ``datasets.py`` shim for
offline JSON experiments.  Because the file lives at the repository root, it
would otherwise shadow the real HuggingFace package for normal users.  By
default we forward to the installed package; set ``CBD_FORCE_LOCAL_DATASETS_SHIM=1``
to use the historical shim.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path


def _try_load_real_datasets():
    if os.environ.get("CBD_FORCE_LOCAL_DATASETS_SHIM", "0") == "1":
        return None

    this_dir = Path(__file__).resolve().parent
    current_module = sys.modules.get(__name__)
    original_path = list(sys.path)
    cwd = Path.cwd().resolve()

    def _resolved(entry: str) -> Path:
        if entry in {"", "."}:
            return cwd
        try:
            return Path(entry).resolve()
        except Exception:
            return Path(entry)

    try:
        sys.modules.pop(__name__, None)
        sys.path = [entry for entry in original_path if _resolved(entry) != this_dir]
        return importlib.import_module(__name__)
    except Exception:
        if current_module is not None:
            sys.modules[__name__] = current_module
        return None
    finally:
        sys.path = original_path


_real_datasets = _try_load_real_datasets()

if _real_datasets is not None:
    globals().update(_real_datasets.__dict__)
    sys.modules[__name__] = _real_datasets
else:

    class IterableDataset:
        pass


    class Dataset:
        def __init__(self, data):
            self._data = list(data)

        @property
        def column_names(self):
            if not self._data:
                return []
            return list(self._data[0].keys())

        def __len__(self):
            return len(self._data)

        def __iter__(self):
            return iter(self._data)

        def __getitem__(self, idx):
            return self._data[idx]

        def select(self, indices):
            if hasattr(indices, "tolist"):
                indices = indices.tolist()
            if isinstance(indices, range):
                indices = list(indices)
            return Dataset([self._data[i] for i in indices])

        def remove_columns(self, columns):
            if isinstance(columns, str):
                columns = [columns]
            new_data = []
            for item in self._data:
                new_item = {k: v for k, v in item.items() if k not in columns}
                new_data.append(new_item)
            return Dataset(new_data)

        def rename_column(self, old_name, new_name):
            new_data = []
            for item in self._data:
                new_item = dict(item)
                if old_name in new_item:
                    new_item[new_name] = new_item.pop(old_name)
                new_data.append(new_item)
            return Dataset(new_data)

        @classmethod
        def from_list(cls, data):
            return cls(data)

        @classmethod
        def from_dict(cls, data_dict):
            if not data_dict:
                return cls([])
            keys = list(data_dict.keys())
            length = len(data_dict[keys[0]])
            data = []
            for idx in range(length):
                item = {k: data_dict[k][idx] for k in keys}
                data.append(item)
            return cls(data)

        @classmethod
        def from_generator(cls, generator, gen_kwargs=None):
            gen_kwargs = gen_kwargs or {}
            return cls(list(generator(**gen_kwargs)))


    class DatasetDict(dict):
        pass


    def concatenate_datasets(datasets_list):
        combined = []
        for ds in datasets_list:
            if isinstance(ds, Dataset):
                combined.extend(ds._data)
            else:
                combined.extend(list(ds))
        return Dataset(combined)


    def _load_json_or_jsonl(path):
        with open(path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                data = []
                f.seek(0)
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
                return data


    def load_dataset(path, split):
        if path in ("locuslab/TOFU", "TOFU"):
            json_file = os.path.join("TOFU", f"{split}.json")
            if not os.path.exists(json_file):
                raise FileNotFoundError(f"TOFU split not found: {json_file}")
            data = _load_json_or_jsonl(json_file)
            return {"train": Dataset.from_list(data)}
        raise ValueError(f"Unsupported dataset: {path}")


    def load_from_disk(path, **kwargs):
        raise NotImplementedError("load_from_disk is not supported in the local datasets shim.")
