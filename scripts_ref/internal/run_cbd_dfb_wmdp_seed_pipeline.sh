#!/usr/bin/env bash
set -euo pipefail

# Clean WMDP pipeline script:
# 1) extract basis
# 2) train A1
# 3) select threshold
# 4) optional full routed eval

SEED="${1:?seed required}"
GPU="${2:-0}"
RUN_SUFFIX="${3:-wmdp_cbd_dfb}"
UNLEARN_LOSS="${4:-gd+kl}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CONDA_SH="${CONDA_SH:-}"
if [[ -f "${CONDA_SH}" ]]; then
  source "${CONDA_SH}"
  conda activate "${REPRO_CONDA_ENV:-cbd}" || true
fi

DEFAULT_PY="python"
PY="${PYTHON:-${DEFAULT_PY}}"
if [[ ! -x "${PY}" ]]; then
  PY="python3"
fi

# Base/assist model selection
BASE_MODEL="${BASE_MODEL:-HuggingFaceH4/zephyr-7b-beta}"
ASSIST_MODEL="${ASSIST_MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
ASSIST_BASE_IF_LORA="${ASSIST_BASE_IF_LORA:-${ASSIST_MODEL}}"
MODEL_CONFIG="${MODEL_CONFIG:-tinyllama}"
ASSIST_TOKENIZER="${ASSIST_TOKENIZER:-${ASSIST_BASE_IF_LORA}}"
BASE_TOKENIZER="${BASE_TOKENIZER:-${BASE_MODEL}}"

# Environment setup
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH:-.}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export FORCE_SAVE_FINAL_CHECKPOINT="${FORCE_SAVE_FINAL_CHECKPOINT:-1}"

RUN_TAG="wmdp_seed${SEED}_${RUN_SUFFIX}"
LOGROOT="artifacts/seed_runs/${RUN_TAG}"
mkdir -p "${LOGROOT}"

OUTMODELDIR="artifacts/outputs_trained_models/cbd_dfb_tinyllama_wmdp/${RUN_TAG}"
BASIS_ROOT="artifacts/basis_cbd_dfb/${RUN_TAG}"
EVAL_ROOT="artifacts/eval_outputs/wmdp/${RUN_TAG}"
BASELOGDIR_VALUE="artifacts/outputs/cbd_dfb_log_wmdp/${RUN_TAG}"
mkdir -p "${OUTMODELDIR}" "${BASIS_ROOT}" "${EVAL_ROOT}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

basis_elapsed_sec=0
train_elapsed_sec=0
threshold_elapsed_sec=0
eval_elapsed_sec=0

DATA_ROOT="${CBD_DATA_ROOT:-data}"
MMLU_DIR="${DATA_ROOT}/eval-method/wmdp/data/mmlu"
MMLU_TRAIN_FILE="${MMLU_TRAIN_FILE:-${MMLU_DIR}/all_auxiliary_train.jsonl}"
MMLU_VAL_FILE="${MMLU_VAL_FILE:-${MMLU_DIR}/all_validation.jsonl}"
MMLU_TEST_FILE="${MMLU_TEST_FILE:-${MMLU_DIR}/all_test.jsonl}"

# Stage 1: Basis extraction
FORGET_DOMAINS="${FORGET_DOMAINS:-bio,cyber}"
TRAIN_SPLIT="${TRAIN_SPLIT:-${FORGET_DOMAINS//,/_}}"
MAX_FORGET="${MAX_FORGET:-600}"
MAX_RETAIN="${MAX_RETAIN:-1200}"
BASIS_MAX_FORGET="${BASIS_MAX_FORGET:-${MAX_FORGET}}"
BASIS_MAX_RETAIN="${BASIS_MAX_RETAIN:-${MAX_RETAIN}}"
TOP_K="${TOP_K:-128}"
TRAIN_RETAIN_NUM="${TRAIN_RETAIN_NUM:-${MAX_RETAIN}}"
TRAIN_MAX_LEN="${TRAIN_MAX_LEN:-512}"
TRAIN_WEIGHT_DECAY="${TRAIN_WEIGHT_DECAY:-0.01}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-3}"
TRAIN_LR="${TRAIN_LR:-2e-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
TRAIN_GRAD_ACC="${TRAIN_GRAD_ACC:-4}"
CBD_DFB_PROJECT_FORGET_ONLY="${CBD_DFB_PROJECT_FORGET_ONLY:-false}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-false}"
FORGET_WEIGHT="${FORGET_WEIGHT:-}"
RETAIN_WEIGHT="${RETAIN_WEIGHT:-}"
LORA_R="${LORA_R:-}"
LORA_ALPHA="${LORA_ALPHA:-}"
LORA_DROPOUT="${LORA_DROPOUT:-}"
BASIS_GRAD_STORE_DTYPE="${BASIS_GRAD_STORE_DTYPE:-}"
THRESH_MAX_FORGET="${THRESH_MAX_FORGET:-${MAX_FORGET}}"
THRESH_MAX_RETAIN="${THRESH_MAX_RETAIN:-${TRAIN_RETAIN_NUM}}"
THRESH_OPTIMIZE="${THRESH_OPTIMIZE:-gap}"
THRESH_MIN_TPR="${THRESH_MIN_TPR:-}"
THRESH_MAX_FPR="${THRESH_MAX_FPR:-}"
SCORE_SPACE="${SCORE_SPACE:-vocab}"
SCORE_POS="${SCORE_POS:-prompt_last}"
SCORE_PROBE_SUFFIX="${SCORE_PROBE_SUFFIX:-}"
SCORE_LAST_K="${SCORE_LAST_K:-1}"
SCORE_LAST_K_REDUCE="${SCORE_LAST_K_REDUCE:-mean}"
SCORE_REDUCER_ALPHA="${SCORE_REDUCER_ALPHA:-1.0}"
SCORE_REDUCER_BETA="${SCORE_REDUCER_BETA:-1.0}"
SCORE_K_MODE="${SCORE_K_MODE:-last}"
TRUNCATE_MODE="${TRUNCATE_MODE:-left}"
THRESH_TRUNCATE_MODE="${THRESH_TRUNCATE_MODE:-${TRUNCATE_MODE}}"
EVAL_TRUNCATE_MODE="${EVAL_TRUNCATE_MODE:-${TRUNCATE_MODE}}"
THRESH_BATCH_SIZE="${THRESH_BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
EVAL_MAX_WMDP="${EVAL_MAX_WMDP:-0}"
EVAL_MAX_MMLU="${EVAL_MAX_MMLU:-0}"
SKIP_FINAL_EVAL="${SKIP_FINAL_EVAL:-0}"
BASIS_PATH_OVERRIDE="${BASIS_PATH_OVERRIDE:-}"

basis_t0="$(date +%s)"
if [[ -n "${BASIS_PATH_OVERRIDE}" ]]; then
  BASIS_PATH="${BASIS_PATH_OVERRIDE}"
  printf '[basis] reuse existing basis: %s\n' "${BASIS_PATH}" >"${LOGROOT}/basis.log"
else
  basis_extra_args=()
  if [[ -n "${LORA_R}" ]]; then
    basis_extra_args+=(--lora_r "${LORA_R}")
  fi
  if [[ -n "${LORA_ALPHA}" ]]; then
    basis_extra_args+=(--lora_alpha "${LORA_ALPHA}")
  fi
  if [[ -n "${LORA_DROPOUT}" ]]; then
    basis_extra_args+=(--lora_dropout "${LORA_DROPOUT}")
  fi
  if [[ -n "${BASIS_GRAD_STORE_DTYPE}" ]]; then
    basis_extra_args+=(--grad_store_dtype "${BASIS_GRAD_STORE_DTYPE}")
  fi
  "${PY}" scripts/extract_cbd_dfb_basis.py \
    --base_model_name "${ASSIST_MODEL}" --dataset wmdp_mcq --wmdp_domains "${FORGET_DOMAINS}" \
    --mmlu_retain_file "${MMLU_TRAIN_FILE}" --seed "${SEED}" --max_forget "${BASIS_MAX_FORGET}" \
    --max_retain "${BASIS_MAX_RETAIN}" --max_len "${TRAIN_MAX_LEN}" --top_k "${TOP_K}" \
    "${basis_extra_args[@]}" \
    --output_dir "${BASIS_ROOT}/wmdp_basis" \
    >"${LOGROOT}/basis.log" 2>&1
  BASIS_PATH="${BASIS_ROOT}/wmdp_basis/cbd_dfb_basis_wmdp_${FORGET_DOMAINS//,/_}_vs_mmlu.pkl"
fi
basis_elapsed_sec="$(( $(date +%s) - basis_t0 ))"

if [[ ! -f "${BASIS_PATH}" ]]; then
  echo "ERROR: basis not found: ${BASIS_PATH}" >&2
  exit 1
fi

# Stage 2: Train
train_t0="$(date +%s)"
train_extra_args=()
if [[ -n "${LORA_R}" ]]; then
  train_extra_args+=("model_mode.Lora.r=${LORA_R}")
fi
if [[ -n "${LORA_ALPHA}" ]]; then
  train_extra_args+=("model_mode.Lora.alpha=${LORA_ALPHA}")
fi
if [[ -n "${LORA_DROPOUT}" ]]; then
  train_extra_args+=("model_mode.Lora.dropout=${LORA_DROPOUT}")
fi
train_cmd=(
  "${PY}" scripts/hf_forget_train.py
  --config-name cbd_dfb_tinyllama_wmdp
  "data.dataset.split=${TRAIN_SPLIT}"
  "data.dataset.mmlu_retain_file=${MMLU_TRAIN_FILE}"
  "data.dataset.max_forget=${MAX_FORGET}"
  "data.dataset.max_len=${TRAIN_MAX_LEN}"
  "data.conv_template.max_len=${TRAIN_MAX_LEN}"
  "data_mode=forget_retain"
  "data_mode.retain_num=${TRAIN_RETAIN_NUM}"
  "+trainer.max_steps=${TRAIN_MAX_STEPS:-450}"
  "trainer.learning_rate=${TRAIN_LR}"
  "trainer.batch_size=${TRAIN_BATCH_SIZE}"
  "trainer.gradient_accumulation_steps=${TRAIN_GRAD_ACC}"
  "trainer.max_epochs=${TRAIN_EPOCHS}"
  "trainer.weight_decay=${TRAIN_WEIGHT_DECAY}"
  "model=${MODEL_CONFIG}"
  "model.model_path=${ASSIST_MODEL}"
  "model.tokenizer_path=${ASSIST_TOKENIZER}"
  "model_mode=base_freeze_a"
  "unlearn_loss=${UNLEARN_LOSS}"
  "enable_cbd_dfb=true"
  "cbd_dfb_basis_path=${BASIS_PATH}"
  "cbd_dfb_project_forget_only=${CBD_DFB_PROJECT_FORGET_ONLY}"
  "oracle_on_cpu=false"
  "gradient_checkpointing=${GRADIENT_CHECKPOINTING}"
)
if [[ -n "${FORGET_WEIGHT}" ]]; then
  train_cmd+=("unlearn_loss.forget_weight=${FORGET_WEIGHT}")
fi
if [[ -n "${RETAIN_WEIGHT}" ]]; then
  train_cmd+=("unlearn_loss.retain_weight=${RETAIN_WEIGHT}")
fi
train_cmd+=("${train_extra_args[@]}")
train_cmd+=(
  "project=cbd_dfb_wmdp_seed${SEED}"
  "seed=${SEED}"
  "lora_seed=${SEED}"
  "OUTPUTMODELDIR=${OUTMODELDIR}"
  "BASELOGDIR=${BASELOGDIR_VALUE}"
)
"${train_cmd[@]}" >"${LOGROOT}/train.log" 2>&1
train_elapsed_sec="$(( $(date +%s) - train_t0 ))"

# Stage 3: Threshold
CKPT="$(find "${OUTMODELDIR}" -type d -name 'checkpoint-*' -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-)"
if [[ -z "${CKPT}" || ! -d "${CKPT}" ]]; then
  echo "ERROR: checkpoint not found under ${OUTMODELDIR}" >&2
  exit 1
fi
printf '%s\n' "${CKPT}" >"${LOGROOT}/ckpt_path.txt"
threshold_t0="$(date +%s)"
threshold_extra=()
if [[ -n "${THRESH_MIN_TPR}" ]]; then
  threshold_extra+=(--min_tpr "${THRESH_MIN_TPR}")
fi
if [[ -n "${THRESH_MAX_FPR}" ]]; then
  threshold_extra+=(--max_fpr "${THRESH_MAX_FPR}")
fi
"${PY}" scripts/select_wmdp_routing_threshold.py \
  --finetuned_assist_path "${CKPT}" --original_assist_path "${ASSIST_MODEL}" \
  --base_if_lora "${ASSIST_BASE_IF_LORA}" \
  --wmdp_domains "${FORGET_DOMAINS}" \
  --mmlu_retain_file "${MMLU_VAL_FILE}" \
  --max_forget "${THRESH_MAX_FORGET}" --max_retain "${THRESH_MAX_RETAIN}" \
  --seed "${SEED}" --batch_size "${THRESH_BATCH_SIZE}" --max_len "${TRAIN_MAX_LEN}" \
  --optimize "${THRESH_OPTIMIZE}" \
  --score_space "${SCORE_SPACE}" --score_pos "${SCORE_POS}" \
  --score_probe_suffix "${SCORE_PROBE_SUFFIX}" \
  --score_last_k_reduce "${SCORE_LAST_K_REDUCE}" --score_last_k "${SCORE_LAST_K}" \
  --score_reducer_alpha "${SCORE_REDUCER_ALPHA}" --score_reducer_beta "${SCORE_REDUCER_BETA}" \
  --score_k_mode "${SCORE_K_MODE}" --truncate_mode "${THRESH_TRUNCATE_MODE}" \
  --output_json "${EVAL_ROOT}/threshold.json" "${threshold_extra[@]}" \
  >"${LOGROOT}/threshold.log" 2>&1
threshold_elapsed_sec="$(( $(date +%s) - threshold_t0 ))"

if [[ "${SKIP_FINAL_EVAL}" != "1" ]]; then
  eval_t0="$(date +%s)"
  "${PY}" scripts/eval_wmdp_routing.py \
    --finetuned_assist_path "${CKPT}" \
    --threshold_json "${EVAL_ROOT}/threshold.json" \
    --base_model "${BASE_MODEL}" \
    --original_assist "${ASSIST_MODEL}" \
    --assist_base_if_lora "${ASSIST_BASE_IF_LORA}" \
    --base_tokenizer "${BASE_TOKENIZER}" \
    --assist_tokenizer "${ASSIST_TOKENIZER}" \
    --base_device "cuda:0" \
    --original_device "cuda:0" \
    --finetuned_device "cuda:0" \
    --mmlu_test_file "${MMLU_TEST_FILE}" \
    --batch_size "${EVAL_BATCH_SIZE}" --max_len "${TRAIN_MAX_LEN}" \
    --max_mmlu "${EVAL_MAX_MMLU}" --max_wmdp "${EVAL_MAX_WMDP}" \
    --seed "${SEED}" \
    --score_space "${SCORE_SPACE}" --score_pos "${SCORE_POS}" \
    --score_probe_suffix "${SCORE_PROBE_SUFFIX}" \
    --score_last_k "${SCORE_LAST_K}" --score_last_k_reduce "${SCORE_LAST_K_REDUCE}" \
    --score_reducer_alpha "${SCORE_REDUCER_ALPHA}" --score_reducer_beta "${SCORE_REDUCER_BETA}" \
    --score_k_mode "${SCORE_K_MODE}" --truncate_mode "${EVAL_TRUNCATE_MODE}" \
    --progress_every "${WMDP_EVAL_PROGRESS_EVERY:-200}" \
    --out_json "${EVAL_ROOT}/eval.json" \
    >"${LOGROOT}/eval.log" 2>&1
  eval_elapsed_sec="$(( $(date +%s) - eval_t0 ))"
fi

"${PY}" - <<PY
import json
from pathlib import Path
timing = {
    "run_tag": "${RUN_TAG}",
    "basis_elapsed_sec": ${basis_elapsed_sec},
    "train_elapsed_sec": ${train_elapsed_sec},
    "threshold_elapsed_sec": ${threshold_elapsed_sec},
    "eval_elapsed_sec": ${eval_elapsed_sec},
    "algorithm_total_sec": ${basis_elapsed_sec} + ${train_elapsed_sec} + ${threshold_elapsed_sec},
    "full_pipeline_total_sec": ${basis_elapsed_sec} + ${train_elapsed_sec} + ${threshold_elapsed_sec} + ${eval_elapsed_sec},
    "checkpoint_path": "${CKPT}",
    "score_last_k_reduce": "${SCORE_LAST_K_REDUCE}",
    "score_reducer_alpha": "${SCORE_REDUCER_ALPHA}",
    "score_reducer_beta": "${SCORE_REDUCER_BETA}",
    "threshold_json": "${EVAL_ROOT}/threshold.json",
    "eval_json": "${EVAL_ROOT}/eval.json" if "${SKIP_FINAL_EVAL}" != "1" else ""
}
Path("${LOGROOT}/timing_wmdp.json").write_text(json.dumps(timing, ensure_ascii=False, indent=2), encoding="utf-8")
PY

log "DONE. Threshold: ${EVAL_ROOT}/threshold.json"
if [[ "${SKIP_FINAL_EVAL}" != "1" ]]; then
  log "DONE. Eval     : ${EVAL_ROOT}/eval.json"
fi
