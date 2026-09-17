# CBD-DFB — Discriminative Subspace Unlearning for DeepSeek

Pipeline học unlearning trên dữ liệu DeepSeek deprecated API, sử dụng phương pháp CBD-DFB (gradient projection lên discriminative subspace).

## Cài Đặt

```bash
conda env create -f environment.yaml
conda activate cbd
```

## Dữ Liệu

```
../Data-Collection/deepseek/
├── D_forget.json          # 9667 mẫu (train + valid, 80/20)
│                          #   "probing input" → prompt
│                          #   "y_neg" → deprecated API (forget)
│                          #   "y_pos" → updated API (retain)
├── D_test_U_dep.json      # 581 mẫu test — kỳ vọng score CAO (deprecated)
└── D_test_U_nondep.json   # 17179 mẫu test — kỳ vọng score THẤP (bình thường)
```

## Chạy Pipeline

Chạy toàn bộ 3 giai đoạn:

```bash
bash run_cbd.sh
```

Hoặc chạy từng giai đoạn:

```bash
bash run_cbd.sh basis    # ① Trích xuất discriminative subspace
bash run_cbd.sh train    # ② Train LoRA + gradient projection
bash run_cbd.sh infer    # ③ Infer + thống kê score
```

### Giai đoạn ① — Trích xuất CBD-DFB Basis

Tạo LoRA trên TinyLlama, thu gradient per-sample trên forget/retain, giải bài toán trị riêng tổng quát → tìm top-k discriminative subspace.

```bash
python scripts/extract_cbd_dfb_basis.py \
    --data_path ../Data-Collection/deepseek/D_forget.json \
    --max_forget 7733 --max_retain 7733 \
    --top_k 192 --batch_size 4 --max_len 512 \
    --output_dir artifacts/basis_cbd_dfb/deepseek_seed42
```

**Output:** `artifacts/basis_cbd_dfb/deepseek_seed42/cbd_dfb_basis_deepseek_forget_vs_deepseek_retain.pkl`

### Giai đoạn ② — Train Unlearn Model

Dùng Hydra config để train LoRA với CBDDFBForgetTrainer — mỗi bước backward, gradient được project lên subspace Q: `g ← QQ^T g`.

```bash
python scripts/hf_forget_train.py \
    --config-name cbd_dfb_deepseek \
    enable_cbd_dfb=true \
    cbd_dfb_basis_path=artifacts/basis_cbd_dfb/deepseek_seed42/cbd_dfb_basis_deepseek_forget_vs_deepseek_retain.pkl \
    seed=42
```

**Output:** LoRA checkpoint trong `artifacts/outputs_trained_models/cbd_dfb_deepseek/`

### Giai đoạn ③ — Infer + Thống kê

Tính `score = CE_finetuned - CE_original` trên valid set → tìm threshold, sau đó test trên 2 tập test.

```bash
python scripts/infer_deepseek.py \
    --original_model_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --finetuned_model_path artifacts/outputs_trained_models/cbd_dfb_deepseek/<checkpoint> \
    --valid_data_path ../Data-Collection/deepseek/D_forget.json \
    --test_dep_path ../Data-Collection/deepseek/D_test_U_dep.json \
    --test_nondep_path ../Data-Collection/deepseek/D_test_U_nondep.json \
    --output_dir artifacts/eval_outputs/deepseek
```

**Output:** Bảng thống kê score + histogram

## Cấu Trúc Dự Án

```
CBD_softweight/
├── run_cbd.sh                         # 🚀 Script chạy pipeline
├── environment.yaml                   # Conda environment
│
├── configs/                           # ⚙️ Hydra configs (dùng bởi giai đoạn ②)
│   ├── cbd_dfb_deepseek.yaml          #   Config chính, merge 5 sub-configs
│   ├── data/deepseek.yaml             #   Dataset class, path, conv_template
│   ├── data_mode/forget_retain.yaml   #   with_retain, retain_num
│   ├── model/tinyllama.yaml           #   Model path, tokenizer path
│   ├── model_mode/base_freeze_a.yaml  #   LoRA rank, alpha, target modules
│   └── unlearn_loss/gd+kl.yaml       #   Loss = GradDescent(forget) + KL(retain)
│
├── scripts/                           # 🔧 Scripts chạy trực tiếp
│   ├── extract_cbd_dfb_basis.py       #   Giai đoạn ①: trích xuất basis
│   ├── hf_forget_train.py             #   Giai đoạn ②: train (Hydra entry point)
│   └── infer_deepseek.py              #   Giai đoạn ③: infer + thống kê
│
├── uld/                               # 📦 Core engine (import bởi scripts)
│   ├── data/
│   │   ├── deepseek.py                #   DeepSeek_DataModule
│   │   ├── datamodule.py              #   TorchDataset, TrainDataModule
│   │   └── conv_util.py               #   ConvTemplate
│   ├── model/
│   │   ├── __init__.py                #   Model factory (TRAIN/EVAL_INIT_FUNCS)
│   │   ├── utils.py                   #   create_full_model(), create_peft_model()
│   │   ├── forget_losses.py           #   GradDescentLoss, KLLoss, NPOLoss, ...
│   │   └── peft_util.py               #   LoRA utilities
│   └── hfutil/
│       ├── hf_trainers.py             #   ForgetTrainer (base)
│       ├── cbd_dfb_trainer.py         #   CBDDFBForgetTrainer (gradient projection)
│       ├── gmp_trainer.py             #   GPMForgetTrainer
│       └── hf_callbacks.py            #   SimpleProfileCallback
│
└── ../Data-Collection/deepseek/   # 📊 Dữ liệu
    ├── D_forget.json
    ├── D_test_U_dep.json
    └── D_test_U_nondep.json
```

## Artifacts (git-ignored)

```
artifacts/
├── basis_cbd_dfb/          # Output giai đoạn ①
├── outputs_trained_models/ # Output giai đoạn ②
└── eval_outputs/           # Output giai đoạn ③
```
