#!/bin/bash
# ==============================================================================
# run_cbd.sh — Script chạy toàn bộ pipeline CBD-DFB cho dữ liệu DeepSeek
#
# Pipeline gồm 3 giai đoạn:
#   ① Trích xuất CBD-DFB basis từ gradient
#   ② Train unlearn model (LoRA + gradient projection)
#   ③ Infer trên test sets + thống kê score
#
# Cách dùng:
#   bash run_cbd.sh              # Chạy tiếp từ chỗ đang dở (bỏ qua giai đoạn đã xong)
#   bash run_cbd.sh status       # Chỉ xem đang ở đâu, không chạy gì
#   bash run_cbd.sh basis        # Chỉ chạy giai đoạn ①
#   bash run_cbd.sh train        # Chỉ chạy giai đoạn ②
#   bash run_cbd.sh infer        # Chỉ chạy giai đoạn ③
#
# Biến môi trường hay dùng:
#   FORCE=1                      # chạy lại cả giai đoạn đã có output
#   TEST_NONDEP_N=500            # giai đoạn ③ chỉ chấm 500 mẫu âm cho nhanh
#   TRAIN_EPOCHS=1               # train ngắn hơn
# ==============================================================================

set -e  # Dừng nếu có lỗi

# ── Cấu hình ──────────────────────────────────────────────────────────────────
SEED=42
DATA_PATH="../Data-Collection/deepseek/D_forget.json"
TEST_DEP_PATH="../Data-Collection/deepseek/D_test_U_dep.json"
TEST_NONDEP_PATH="../Data-Collection/deepseek/D_test_U_nondep.json"
BASE_MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Giai đoạn ① — Basis extraction
# Dùng TOÀN BỘ D_forget (không chia train/valid nữa): threshold được dò trên
# 200 mẫu tách ra từ hai tập test, xem giai đoạn ③.
BASIS_OUTPUT_DIR="artifacts/basis_cbd_dfb/deepseek_seed${SEED}"
TRAIN_RATIO=1.0
MAX_FORGET=9667      # toàn bộ D_forget
MAX_RETAIN=9667
TOP_K=192
BASIS_BATCH_SIZE=4
MAX_LEN=512

# Giai đoạn ② — Training
HYDRA_CONFIG="cbd_dfb_deepseek"
# Ép một GPU: nếu thấy nhiều GPU, HF Trainer tự bọc nn.DataParallel trong khi script
# vẫn tính lịch train theo WORLD_SIZE=1 → số bước, warmup và save_steps đều lệch.
# Muốn dùng nhiều GPU thì phải chạy bằng torchrun (DDP), không phải DataParallel.
TRAIN_GPU="${TRAIN_GPU:-0}"
# Giữ một checkpoint mỗi epoch. Thiết kế này không còn tập valid nên Trainer không có
# tín hiệu early-stop; để mặc định 1 thì chỉ còn đúng epoch cuối, không quay lại được.
TRAIN_SAVE_TOTAL_LIMIT="${TRAIN_SAVE_TOTAL_LIMIT:-10}"
# 19.334 mẫu / batch 4 / accum 2 = 2.417 bước mỗi epoch. Giá trị 10 là kế thừa từ setup
# ToFU cũ; LoRA ở đây chỉ có 3,96M tham số nên nhiều khả năng ít epoch là đủ. Vẫn giữ
# checkpoint từng epoch nhờ TRAIN_SAVE_TOTAL_LIMIT, chọn epoch sau bằng tập calib.
TRAIN_EPOCHS="${TRAIN_EPOCHS:-3}"

# Giai đoạn ③ — Inference
EVAL_OUTPUT_DIR="artifacts/eval_outputs/deepseek"
INFER_BATCH_SIZE=8
PROMPT_FIELD="probing input"
CALIB_DEP_N=200
CALIB_NONDEP_N=200
# Giới hạn tập âm lúc TEST. 500 đủ để lặp nhanh; đặt -1 (toàn bộ ~16.700) khi lấy số cuối,
# vì khoảng tin cậy của FPR hẹp hơn khoảng 7 lần.
TEST_NONDEP_N="${TEST_NONDEP_N:--1}"
# ──────────────────────────────────────────────────────────────────────────────

# Thêm đường dẫn hiện tại vào PYTHONPATH để import được uld
export PYTHONPATH="$(pwd):$PYTHONPATH"
export PYTHONUNBUFFERED=1

# Tự động tìm basis file path
BASIS_FILE="${BASIS_OUTPUT_DIR}/cbd_dfb_basis_deepseek_forget_vs_deepseek_retain.pkl"

# Tự động tìm checkpoint mới nhất (sau train)
find_latest_checkpoint() {
    local ckpt_dir="artifacts/outputs_trained_models/cbd_dfb_deepseek"
    if [ ! -d "$ckpt_dir" ]; then
        echo ""
        return
    fi
    # Checkpoint mới nhất theo THỜI GIAN, không theo tên: sort chuỗi sẽ coi
    # checkpoint-9 mới hơn checkpoint-100.
    find "$ckpt_dir" -name "adapter_config.json" -printf '%T@ %h\n' 2>/dev/null \
        | sort -rn | head -1 | cut -d' ' -f2-
}

FORCE="${FORCE:-0}"

# ══════════════════════════════════════════════════════════════════════════════
# PREFLIGHT — kiểm mọi thứ cần thiết TRƯỚC khi tốn hàng giờ tính toán
# ══════════════════════════════════════════════════════════════════════════════
preflight() {
    local fail=0

    for f in "${DATA_PATH}" "${TEST_DEP_PATH}" "${TEST_NONDEP_PATH}"; do
        if [ ! -f "$f" ]; then
            echo "❌ Thiếu dữ liệu: $f"; fail=1
        fi
    done

    # uld phải import được. `python script.py` đặt sys.path[0] = thư mục chứa script,
    # KHÔNG phải thư mục hiện tại — nên PYTHONPATH ở trên là bắt buộc.
    if ! python -c "import uld, uld.data.deepseek" 2>/dev/null; then
        echo "❌ Không import được uld — kiểm tra conda env đã activate chưa"; fail=1
    fi

    for m in torch transformers peft hydra omegaconf datasets; do
        python -c "import $m" 2>/dev/null || { echo "❌ Thiếu module: $m"; fail=1; }
    done

    # matplotlib chỉ cần cho biểu đồ ở giai đoạn ③ — thiếu thì cảnh báo, không chặn.
    if ! python -c "import matplotlib" 2>/dev/null; then
        echo "⚠️  Thiếu matplotlib → giai đoạn ③ sẽ bỏ qua routing_histogram.png"
        echo "    pip install matplotlib==3.8.4"
    fi

    if [ "$fail" = "1" ]; then
        exit 1
    fi
    return 0
}

# ══════════════════════════════════════════════════════════════════════════════
# STATUS — pipeline đang ở đâu
# ══════════════════════════════════════════════════════════════════════════════
show_status() {
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  TRẠNG THÁI PIPELINE"
    echo "════════════════════════════════════════════════════════════════"

    if [ -f "${BASIS_FILE}" ]; then
        echo "  ① basis   ✅ $(du -h "${BASIS_FILE}" | cut -f1)  ${BASIS_FILE}"
    else
        echo "  ① basis   ❌ chưa có"
    fi

    local n_ckpt
    n_ckpt=$(find artifacts/outputs_trained_models -name adapter_config.json 2>/dev/null | wc -l | tr -d ' ')
    if [ "${n_ckpt}" -gt 0 ]; then
        echo "  ② train   ✅ ${n_ckpt} checkpoint"
        find artifacts/outputs_trained_models -name adapter_config.json \
            -printf '              %TY-%Tm-%Td %TH:%TM  %h\n' 2>/dev/null | sort
    else
        echo "  ② train   ❌ chưa có checkpoint"
    fi

    if [ -f "${EVAL_OUTPUT_DIR}/routing_statistics.json" ]; then
        echo "  ③ infer   ✅ ${EVAL_OUTPUT_DIR}/routing_statistics.json"
    else
        echo "  ③ infer   ❌ chưa chạy"
    fi

    # pgrep trả 1 khi không khớp -> phải nuốt, nếu không set -e giết cả script
    local running
    running=$(pgrep -af "scripts/(extract_cbd_dfb_basis|hf_forget_train|infer_deepseek)\.py" 2>/dev/null | head -3 || true)
    if [ -n "${running}" ]; then
        echo ""
        echo "  ⚠️  ĐANG CHẠY — đừng khởi động chồng lên:"
        echo "${running}" | sed 's/^/      /'
    fi
    echo "════════════════════════════════════════════════════════════════"
}

# Chọn giai đoạn chạy
STAGE="${1:-all}"

# ══════════════════════════════════════════════════════════════════════════════
# GIAI ĐOẠN ①: TRÍCH XUẤT CBD-DFB BASIS
# ══════════════════════════════════════════════════════════════════════════════
run_basis() {
    if [ -f "${BASIS_FILE}" ] && [ "${FORCE}" != "1" ]; then
        echo "⏭  ① basis đã có, bỏ qua → ${BASIS_FILE}   (FORCE=1 để chạy lại)"
        return 0
    fi
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  ① TRÍCH XUẤT CBD-DFB BASIS"
    echo "════════════════════════════════════════════════════════════════"
    echo "  Data:       ${DATA_PATH}"
    echo "  Model:      ${BASE_MODEL}"
    echo "  Output:     ${BASIS_OUTPUT_DIR}"
    echo "  max_forget: ${MAX_FORGET}, max_retain: ${MAX_RETAIN}, top_k: ${TOP_K}"
    echo ""

    python scripts/extract_cbd_dfb_basis.py \
        --data_path "${DATA_PATH}" \
        --base_model_name "${BASE_MODEL}" \
        --train_ratio ${TRAIN_RATIO} \
        --max_forget ${MAX_FORGET} \
        --max_retain ${MAX_RETAIN} \
        --top_k ${TOP_K} \
        --batch_size ${BASIS_BATCH_SIZE} \
        --max_len ${MAX_LEN} \
        --seed ${SEED} \
        --output_dir "${BASIS_OUTPUT_DIR}"

    echo ""
    echo "✅ Basis extraction hoàn tất → ${BASIS_FILE}"
}

# ══════════════════════════════════════════════════════════════════════════════
# GIAI ĐOẠN ②: TRAIN UNLEARN MODEL
# ══════════════════════════════════════════════════════════════════════════════
run_train() {
    if [ -n "$(find_latest_checkpoint)" ] && [ "${FORCE}" != "1" ]; then
        echo "⏭  ② đã có checkpoint, bỏ qua → $(find_latest_checkpoint)   (FORCE=1 để train lại)"
        return 0
    fi
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  ② TRAIN UNLEARN MODEL (CBD-DFB)"
    echo "════════════════════════════════════════════════════════════════"
    echo "  Config:     ${HYDRA_CONFIG}"
    echo "  Basis:      ${BASIS_FILE}"
    echo "  Seed:       ${SEED}"
    echo ""

    if [ ! -f "${BASIS_FILE}" ]; then
        echo "❌ Không tìm thấy basis file: ${BASIS_FILE}"
        echo "   Hãy chạy giai đoạn ① trước: bash run_cbd.sh basis"
        exit 1
    fi

    echo "  GPU:        ${TRAIN_GPU}"
    echo "  Epochs:     ${TRAIN_EPOCHS}  (~2417 bước mỗi epoch)"
    echo "  Ckpt giữ:   ${TRAIN_SAVE_TOTAL_LIMIT} (một checkpoint mỗi epoch)"
    echo ""

    # Không còn tập valid (train_ratio=1.0) nên phải tắt eval nội bộ của Trainer.
    CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" \
    SAVE_TOTAL_LIMIT="${TRAIN_SAVE_TOTAL_LIMIT}" \
    DISABLE_INTERNAL_EVAL=1 python scripts/hf_forget_train.py \
        --config-name "${HYDRA_CONFIG}" \
        enable_cbd_dfb=true \
        cbd_dfb_basis_path="${BASIS_FILE}" \
        trainer.max_epochs=${TRAIN_EPOCHS} \
        seed=${SEED}

    CKPT=$(find_latest_checkpoint)
    echo ""
    echo "✅ Training hoàn tất → ${CKPT}"
}

# ══════════════════════════════════════════════════════════════════════════════
# GIAI ĐOẠN ③: INFER + THỐNG KÊ WEIGHT
# ══════════════════════════════════════════════════════════════════════════════
run_infer() {
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  ③ INFER + THỐNG KÊ SCORE"
    echo "════════════════════════════════════════════════════════════════"

    CKPT=$(find_latest_checkpoint)
    if [ -z "${CKPT}" ]; then
        echo "❌ Không tìm thấy checkpoint."
        echo "   Hãy chạy giai đoạn ② trước: bash run_cbd.sh train"
        exit 1
    fi

    echo "  Original:   ${BASE_MODEL}"
    echo "  Finetuned:  ${CKPT}"
    echo "  Test dep:   ${TEST_DEP_PATH}"
    echo "  Test nondep:${TEST_NONDEP_PATH}"
    echo "  Output:     ${EVAL_OUTPUT_DIR}"
    echo ""

    python scripts/infer_deepseek.py \
        --original_model_path "${BASE_MODEL}" \
        --finetuned_model_path "${CKPT}" \
        --valid_data_path "${DATA_PATH}" \
        --test_dep_path "${TEST_DEP_PATH}" \
        --test_nondep_path "${TEST_NONDEP_PATH}" \
        --output_dir "${EVAL_OUTPUT_DIR}" \
        --prompt_field "${PROMPT_FIELD}" \
        --calib_dep_n ${CALIB_DEP_N} \
        --calib_nondep_n ${CALIB_NONDEP_N} \
        --test_nondep_n ${TEST_NONDEP_N} \
        --batch_size ${INFER_BATCH_SIZE} \
        --max_len ${MAX_LEN} \
        --seed ${SEED}

    echo ""
    echo "✅ Inference hoàn tất → ${EVAL_OUTPUT_DIR}/routing_statistics.json"
}

# ══════════════════════════════════════════════════════════════════════════════
# DISPATCHER
# ══════════════════════════════════════════════════════════════════════════════
case "$STAGE" in
    status) show_status; exit 0 ;;
    basis)  preflight; run_basis ;;
    train)  preflight; run_train ;;
    infer)  preflight; run_infer ;;
    all)
        preflight
        show_status
        run_basis
        run_train
        run_infer
        echo ""
        echo "════════════════════════════════════════════════════════════════"
        echo "  ✅ TOÀN BỘ PIPELINE HOÀN TẤT"
        echo "════════════════════════════════════════════════════════════════"
        ;;
    *)
        echo "Cách dùng: bash run_cbd.sh [status|basis|train|infer|all]"
        exit 1
        ;;
esac
