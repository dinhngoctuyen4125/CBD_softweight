"""离线TOFU数据集加载器"""
from datasets import Dataset, DatasetDict
from pathlib import Path

def load_tofu_offline(split_name):
    """从本地缓存加载TOFU数据集，完全离线"""
    cache_paths = [
        Path.home() / '.cache/huggingface/datasets/locuslab___tofu',  # 原始缓存
        Path.home() / '.cache/huggingface/datasets/locuslab_TOFU',   # 复制的缓存
    ]
    
    for cache_path in cache_paths:
        split_path = cache_path / f"{split_name}/0.0.0"
        if not split_path.exists():
            continue
            
        # 找到版本目录
        version_dirs = [d for d in split_path.iterdir() if d.is_dir() and len(d.name) == 40]
        if not version_dirs:
            continue
            
        arrow_file = version_dirs[0] / "tofu-train.arrow"
        if arrow_file.exists():
            train_dataset = Dataset.from_file(str(arrow_file))
            return DatasetDict({'train': train_dataset})
    
    raise FileNotFoundError(f"No cache found for {split_name}")
