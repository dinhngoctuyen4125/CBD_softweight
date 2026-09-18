# CBD-DFB — Discriminative Subspace Unlearning cho DeepSeek

Pipeline dò code dùng API đã deprecated, bằng phương pháp CBD-DFB: chiếu gradient lên
không gian con phân biệt giữa `y_neg` (API cũ) và `y_pos` (API mới).

Hai mô hình nhỏ tạo thành bộ dò:

- **M_ref** — TinyLlama gốc, đóng băng
- **M_pro** — TinyLlama + LoRA, huấn luyện để lệch khỏi M_ref đúng trên ngữ cảnh deprecated

Điểm phân loại là **symmetric KL divergence** giữa hai phân phối dự đoán tại token cuối
của prompt:

```
s(x) = 0.5 * ( KL(p_M_ref || p_M_pro) + KL(p_M_pro || p_M_ref) )

x dùng API deprecated  →  kỳ vọng s(x) CAO
x bình thường          →  kỳ vọng s(x) THẤP
```

## Cài Đặt

```bash
conda env create -f environment.yaml
conda activate cbd
```

## Dữ Liệu

```
../Data-Collection/deepseek/
├── D_forget.json          # 9667 mẫu — dùng TOÀN BỘ để dựng Q và train
│                          #   "probing input" → prompt
│                          #   "y_neg" → API deprecated (forget)
│                          #   "y_pos" → API updated   (retain)
├── D_test_U_dep.json      # 581 mẫu  — kỳ vọng s(x) CAO
└── D_test_U_nondep.json   # 17179 mẫu — kỳ vọng s(x) THẤP
```

Threshold được dò trên **200 mẫu tách ra từ mỗi tập test**, phần còn lại dùng để báo cáo.
Các prompt trùng với `D_forget.json` bị loại khỏi cả hai (31 ở dep, 194 ở nondep).

## Chạy Pipeline

```bash
bash run_cbd.sh basis    # ① Trích xuất discriminative subspace Q
bash run_cbd.sh train    # ② Train LoRA + gradient projection
bash run_cbd.sh infer    # ③ Chấm symKL + thống kê
```

Tham số nằm ở đầu `run_cbd.sh`. Vài biến hay dùng:

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `TRAIN_EPOCHS` | 3 | ~2417 bước mỗi epoch |
| `TRAIN_GPU` | 0 | ép một GPU; nhiều GPU khiến Trainer bọc DataParallel và lệch lịch train |
| `TRAIN_SAVE_TOTAL_LIMIT` | 10 | giữ checkpoint từng epoch để chọn sau |
| `TEST_NONDEP_N` | -1 | giới hạn tập âm lúc test; đặt 500 để lặp nhanh |

### Giai đoạn ① — Trích xuất CBD-DFB Basis

Dựng LoRA trên TinyLlama, thu gradient per-sample trên forget/retain, giải bài toán trị
riêng tổng quát `F⁻v = λ(F⁺ + μI)v` → lấy top-k hướng phân biệt nhất.

Gradient được ghi ra memmap trên đĩa (~153 GB cho 9667 mẫu). Giữ hết trong RAM sẽ bị
OOM killer; script tự kiểm tra dung lượng trống trước khi chạy.

**Output:** `artifacts/basis_cbd_dfb/deepseek_seed42/cbd_dfb_basis_*.pkl` + `basis_config.json`

### Giai đoạn ② — Train Unlearn Model

LoRA (`r=32`, chỉ `up_proj`, `lora_A` đóng băng) với loss `GradDescent(y_neg) + β·KL(y_pos)`.
Mỗi bước backward, gradient được chiếu lên Q: `g ← QQᵀg`.

**Output:** LoRA checkpoint trong `artifacts/outputs_trained_models/cbd_dfb_deepseek/`

Các công tắc trong `configs/cbd_dfb_deepseek.yaml`:

| Cờ | Tác dụng |
|---|---|
| `cbd_dfb_project_forget_only` | chỉ chiếu nhánh forget: `g = QQᵀg⁻ + β·g⁺` |
| `cbd_dfb_eigval_weight` | nhân hệ số theo `√λ` |
| `cbd_dfb_trust_region` | co bước đi khi ảnh hưởng lên retain vượt ε |

### Giai đoạn ③ — Infer + Thống kê

**Output:** `artifacts/eval_outputs/deepseek/routing_statistics.json` + `routing_histogram.png`

## Cấu Trúc Dự Án

```
CBD_softweight/
├── run_cbd.sh                         # 🚀 Điều phối ba giai đoạn
├── environment.yaml                   # Conda environment
│
├── configs/                           # ⚙️ Hydra configs
│   ├── cbd_dfb_deepseek.yaml          #   Config chính (cũng là config mặc định)
│   ├── data/deepseek.yaml             #   Đường dẫn + conv_template
│   ├── data_mode/forget_retain.yaml   #   with_retain, retain_num
│   ├── model/tinyllama.yaml           #   Model path, attn_implementation
│   ├── model_mode/base_freeze_a.yaml  #   LoRA rank, alpha, target modules
│   └── unlearn_loss/gd+kl.yaml        #   GradDescent(forget) + KL(retain)
│
├── scripts/
│   ├── extract_cbd_dfb_basis.py       #   Giai đoạn ①
│   ├── hf_forget_train.py             #   Giai đoạn ② (Hydra entry point)
│   ├── infer_deepseek.py              #   Giai đoạn ③
│   └── routing_score_reducers.py      #   Gộp symKL trên nhiều vị trí token
│
└── uld/                               # 📦 Core engine
    ├── data/
    │   ├── deepseek.py                #   DeepSeek_DataModule
    │   ├── datamodule.py              #   TorchDataset, EqualForgetRetainSampler
    │   └── conv_util.py               #   ConvTemplate
    ├── model/
    │   ├── __init__.py                #   TRAIN_INIT_FUNCS
    │   ├── utils.py                   #   create_full_model()
    │   ├── forget_losses.py           #   GradDescent, KL, NPO, DPO, ...
    │   └── peft_util.py               #   LoRA utilities
    └── hfutil/
        ├── hf_trainers.py             #   ForgetTrainer (base)
        ├── cbd_dfb_trainer.py         #   CBDDFBForgetTrainer (gradient projection)
        └── hf_callbacks.py            #   SimpleProfileCallback
```

## Ràng buộc giữa các giai đoạn

Ba thứ này **phải khớp** giữa ① và ②, nếu không Q sẽ nằm ở không gian khác với gradient
lúc train và phép chiếu trở nên vô nghĩa — mà không có lỗi nào được ném ra:

1. **LoRA config** — `r=32`, `alpha=64`, `dropout=0.05`, target `up_proj`, `lora_A` đóng băng
2. **conv_template** — ba token phân cách đều rỗng, `max_len=512`
3. **`train_ratio`** — `1.0` ở cả hai

Kiểm tra (1) bằng cách so hash trọng số `lora_A` dựng theo hai đường; (2) và (3) đọc trực
tiếp trong `configs/data/deepseek.yaml` so với `scripts/extract_cbd_dfb_basis.py:main()`.

## Artifacts (git-ignored)

```
artifacts/
├── basis_cbd_dfb/          # Output giai đoạn ①
├── outputs_trained_models/ # Output giai đoạn ②
└── eval_outputs/           # Output giai đoạn ③
```
