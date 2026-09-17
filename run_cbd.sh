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
#   bash run_cbd.sh              # Chạy cả 3 giai đoạn
#   bash run_cbd.sh basis        # Chỉ chạy giai đoạn ①
#   bash run_cbd.sh train        # Chỉ chạy giai đoạn ②
#   bash run_cbd.sh infer        # Chỉ chạy giai đoạn ③
# ==============================================================================

set -e  # Dừng nếu có lỗi

# ── Cấu hình ──────────────────────────────────────────────────────────────────
SEED=42
DATA_PATH="../Data-Collection/deepseek/D_forget.json"
TEST_DEP_PATH="../Data-Collection/deepseek/D_test_U_dep.json"
TEST_NONDEP_PATH="../Data-Collection/deepseek/D_test_U_nondep.json"
BASE_MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Giai đoạn ① — Basis extraction
BASIS_OUTPUT_DIR="artifacts/basis_cbd_dfb/deepseek_seed${SEED}"
MAX_FORGET=7733      # 80% của 9667
MAX_RETAIN=7733
TOP_K=192
BASIS_BATCH_SIZE=4
MAX_LEN=512

# Giai đoạn ② — Training
HYDRA_CONFIG="cbd_dfb_deepseek"

# Giai đoạn ③ — Inference
EVAL_OUTPUT_DIR="artifacts/eval_outputs/deepseek"
INFER_BATCH_SIZE=4
# ──────────────────────────────────────────────────────────────────────────────

# Thêm đường dẫn hiện tại vào PYTHONPATH để import được uld
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Tự động tìm basis file path
BASIS_FILE="${BASIS_OUTPUT_DIR}/cbd_dfb_basis_deepseek_forget_vs_deepseek_retain.pkl"

# Tự động tìm checkpoint mới nhất (sau train)
find_latest_checkpoint() {
    local ckpt_dir="artifacts/outputs_trained_models/cbd_dfb_deepseek"
    if [ ! -d "$ckpt_dir" ]; then
        echo ""
        return
    fi
    # Tìm thư mục checkpoint mới nhất chứa adapter_config.json
    find "$ckpt_dir" -name "adapter_config.json" -printf '%h\n' 2>/dev/null \
        | sort -r | head -1
}

# Chọn giai đoạn chạy
STAGE="${1:-all}"

# ══════════════════════════════════════════════════════════════════════════════
# GIAI ĐOẠN ①: TRÍCH XUẤT CBD-DFB BASIS
# ══════════════════════════════════════════════════════════════════════════════
run_basis() {
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

    python scripts/hf_forget_train.py \
        --config-name "${HYDRA_CONFIG}" \
        enable_cbd_dfb=true \
        cbd_dfb_basis_path="${BASIS_FILE}" \
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
        --batch_size ${INFER_BATCH_SIZE} \
        --max_len ${MAX_LEN} \
        --seed ${SEED}

    echo ""
    echo "✅ Inference hoàn tất → ${EVAL_OUTPUT_DIR}/weight_statistics.json"
}

# ══════════════════════════════════════════════════════════════════════════════
# DISPATCHER
# ══════════════════════════════════════════════════════════════════════════════
case "$STAGE" in
    basis)  run_basis ;;
    train)  run_train ;;
    infer)  run_infer ;;
    all)
        run_basis
        run_train
        run_infer
        echo ""
        echo "════════════════════════════════════════════════════════════════"
        echo "  ✅ TOÀN BỘ PIPELINE HOÀN TẤT"
        echo "════════════════════════════════════════════════════════════════"
        ;;
    *)
        echo "Cách dùng: bash run_cbd.sh [basis|train|infer|all]"
        exit 1
        ;;
esac
