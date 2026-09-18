#!/usr/bin/env bash
# 内部实现脚本。正式复现请统一从 scripts/hf_forget_train.py repro 进入。
set -euo pipefail

SEED="${1:?seed required}"
GPU="${2:-0}"
RUN_SUFFIX="${3:-gpm_tofu_table3}"
UNLEARN_LOSS="${4:-gd+kl}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

CONDA_SH="${CONDA_SH:-}"
if [[ -f "${CONDA_SH}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${REPRO_CONDA_ENV:-cbd}" >/dev/null 2>&1 || true
fi
PY="${PYTHON:-python}"
if [[ ! -x "${PY}" ]]; then
  PY="python3"
fi

export CUDA_VISIBLE_DEVICES="${GPU}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED="${SEED}"
export PYTHONPATH="${PYTHONPATH:-.}"
export FORCE_SAVE_FINAL_CHECKPOINT="${FORCE_SAVE_FINAL_CHECKPOINT:-1}"

FORGET_SPLIT="${FORGET_SPLIT:-forget10}"
TOFU_DATA_NAME="${TOFU_DATA_NAME:-${CBD_DATA_ROOT:-data}/TOFU}"
ASSIST_MODEL="${ASSIST_MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
TOFU_BASE_MODEL="${TOFU_BASE_MODEL:-locuslab/tofu_ft_llama2-7b}"
TRAIN_RETAIN_NUM="${TRAIN_RETAIN_NUM:-2400}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-4}"
TRAIN_LR="${TRAIN_LR:-2e-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
TRAIN_GRAD_ACC="${TRAIN_GRAD_ACC:-4}"
TRAIN_WEIGHT_DECAY="${TRAIN_WEIGHT_DECAY:-0.01}"
TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-}"
FORGET_WEIGHT="${FORGET_WEIGHT:-}"
RETAIN_WEIGHT="${RETAIN_WEIGHT:-}"
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
GPM_PROJECT_FORGET_ONLY="${GPM_PROJECT_FORGET_ONLY:-1}"
GPM_TARGET_VARIANCE="${GPM_TARGET_VARIANCE:-0.9}"
GPM_MAX_SAMPLES="${GPM_MAX_SAMPLES:-2400}"
GPM_MAX_LEN="${GPM_MAX_LEN:-128}"
GPM_BASIS_PATH="${GPM_BASIS_PATH:-}"
CE_MAX_NEW_TOKENS="${CE_MAX_NEW_TOKENS:-32}"
CE_BATCH_SIZE="${CE_BATCH_SIZE:-8}"
CE_MAX_SAMPLES="${CE_MAX_SAMPLES:-300}"
THRESH_OPTIMIZE="${THRESH_OPTIMIZE:-accuracy}"
THRESH_MIN_TPR="${THRESH_MIN_TPR:-}"
THRESH_MAX_FPR="${THRESH_MAX_FPR:-}"
SKIP_ROUTING_EVAL="${SKIP_ROUTING_EVAL:-0}"
ROUTING_EVAL_BATCH_SIZE="${ROUTING_EVAL_BATCH_SIZE:-8}"

ANALYZE_CE_SCRIPT="${ANALYZE_CE_SCRIPT:-scripts/analyze_cross_entropy.py}"
if [[ ! -f "${ANALYZE_CE_SCRIPT}" ]]; then
  echo "[ERROR] analyze_cross_entropy.py not found" >&2
  exit 1
fi

case "${FORGET_SPLIT}" in
  forget01) RETAIN_SPLIT="retain99" ;;
  forget05) RETAIN_SPLIT="retain95" ;;
  forget10) RETAIN_SPLIT="retain90" ;;
  *)
    echo "[ERROR] unsupported FORGET_SPLIT=${FORGET_SPLIT}" >&2
    exit 1
    ;;
esac

RUN_TAG="seed${SEED}_${RUN_SUFFIX}"
LOGROOT="artifacts/seed_runs/${RUN_TAG}"
OUTMODELDIR="artifacts/outputs_trained_models/gpm_tinyllama_${RUN_TAG}"
BASIS_ROOT="artifacts/basis_gpm/${RUN_TAG}"
CE_ROOT="artifacts/ce_results_gpm/${RUN_TAG}"
EVAL_ROOT="artifacts/eval_outputs/tofu/double_assis_routing_gpm/${RUN_TAG}"

mkdir -p "${LOGROOT}" "${OUTMODELDIR}" "${BASIS_ROOT}" "${CE_ROOT}" "${EVAL_ROOT}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

find_latest_ckpt() {
  local search_root="$1"
  local pattern="$2"
  local ckpts
  ckpts=$(find "${search_root}" -type d -path "*${pattern}*" -name "checkpoint-*" -printf '%T@ %p\n' || true)
  if [[ -z "${ckpts}" ]]; then
    echo ""
    return 1
  fi
  echo "${ckpts}" | sort -n | tail -n 1 | cut -d' ' -f2-
}

log "==== GPM ToFU Table3 启动: seed=${SEED} loss=${UNLEARN_LOSS} pf_only=${GPM_PROJECT_FORGET_ONLY} ===="

BASIS_DIR="${BASIS_ROOT}/${FORGET_SPLIT}_vs_${RETAIN_SPLIT}"
mkdir -p "${BASIS_DIR}"
if [[ -n "${GPM_BASIS_PATH}" ]]; then
  BASIS_PATH="${GPM_BASIS_PATH}"
  log "Stage1 跳过，复用现成基底: ${BASIS_PATH}"
else
  log "Stage1 基底提取: retain=${RETAIN_SPLIT} max_samples=${GPM_MAX_SAMPLES} max_len=${GPM_MAX_LEN} tv=${GPM_TARGET_VARIANCE}"
  ${PY} scripts/extract_retain_basis_svd.py \
    --base_model_name "${ASSIST_MODEL}" \
    --dataset tofu \
    --retain_split "${RETAIN_SPLIT}" \
    --target_variance "${GPM_TARGET_VARIANCE}" \
    --max_samples "${GPM_MAX_SAMPLES}" \
    --max_len "${GPM_MAX_LEN}" \
    --lora_r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout "${LORA_DROPOUT}" \
    --output_dir "${BASIS_DIR}" \
    >"${LOGROOT}/basis_${FORGET_SPLIT}.log" 2>&1
  BASIS_PATH="${BASIS_DIR}/gpm_retain_basis.pkl"
fi
if [[ ! -f "${BASIS_PATH}" ]]; then
  echo "[ERROR] basis not found: ${BASIS_PATH}" >&2
  exit 1
fi

PROJECT_NAME="gpm_${FORGET_SPLIT}_seed${SEED}"
log "Stage2 训练A1: retain_num=${TRAIN_RETAIN_NUM} epochs=${TRAIN_EPOCHS} lr=${TRAIN_LR}"
TRAIN_STEP_ARGS=()
if [[ -n "${TRAIN_MAX_STEPS}" && "${TRAIN_MAX_STEPS}" != "none" && "${TRAIN_MAX_STEPS}" != "None" && "${TRAIN_MAX_STEPS}" != "NONE" && "${TRAIN_MAX_STEPS}" != "0" ]]; then
  TRAIN_STEP_ARGS+=(+trainer.max_steps="${TRAIN_MAX_STEPS}")
fi
${PY} scripts/hf_forget_train.py \
  --config-name cbd_dfb_tinyllama_tofu \
  data.dataset.split="${FORGET_SPLIT}" \
  data_mode=forget_retain \
  data_mode.retain_num="${TRAIN_RETAIN_NUM}" \
  enable_cbd_dfb=false \
  enable_gmp=true \
  gmp_basis_path="${BASIS_PATH}" \
  gmp_project_forget_only="${GPM_PROJECT_FORGET_ONLY}" \
  project="${PROJECT_NAME}" \
  seed="${SEED}" \
  lora_seed="${SEED}" \
  trainer.max_epochs="${TRAIN_EPOCHS}" \
  trainer.learning_rate="${TRAIN_LR}" \
  trainer.batch_size="${TRAIN_BATCH_SIZE}" \
  trainer.gradient_accumulation_steps="${TRAIN_GRAD_ACC}" \
  trainer.weight_decay="${TRAIN_WEIGHT_DECAY}" \
  "${TRAIN_STEP_ARGS[@]}" \
  model=tinyllama \
  model_mode=base_freeze_a \
  model_mode.Lora.r="${LORA_R}" \
  model_mode.Lora.alpha="${LORA_ALPHA}" \
  model_mode.Lora.dropout="${LORA_DROPOUT}" \
  unlearn_loss="${UNLEARN_LOSS}" \
  ${FORGET_WEIGHT:+unlearn_loss.forget_weight="${FORGET_WEIGHT}"} \
  ${RETAIN_WEIGHT:+unlearn_loss.retain_weight="${RETAIN_WEIGHT}"} \
  OUTPUTMODELDIR="${OUTMODELDIR}" \
  BASELOGDIR="artifacts/outputs/gpm_log_${RUN_TAG}" \
  >"${LOGROOT}/train_${FORGET_SPLIT}.log" 2>&1

CKPT=$(find_latest_ckpt "${OUTMODELDIR}" "project_${PROJECT_NAME}" || true)
if [[ -z "${CKPT}" ]]; then
  CKPT=$(find "${OUTMODELDIR}" -type d -name 'checkpoint-*' -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-)
fi
if [[ -z "${CKPT}" ]]; then
  echo "[ERROR] training checkpoint not found under ${OUTMODELDIR}" >&2
  exit 1
fi
echo "${CKPT}" > "${LOGROOT}/ckpt_path.txt"

CE_DIR="${CE_ROOT}/${PROJECT_NAME}"
mkdir -p "${CE_DIR}"
log "Stage3 CE+threshold: forget=${FORGET_SPLIT} retain=${RETAIN_SPLIT}"
${PY} scripts/assis_tinyllama_test_path.py \
  --model_path "${CKPT}" \
  --dataset_name "${TOFU_DATA_NAME}" \
  --dataset_split "${FORGET_SPLIT}" \
  --forget_split "${FORGET_SPLIT}" \
  --metric fixed_sym_kl \
  --use_weighted_ce False \
  --use_length_factor False \
  --max_new_tokens "${CE_MAX_NEW_TOKENS}" \
  --batch_size "${CE_BATCH_SIZE}" \
  --max_samples "${CE_MAX_SAMPLES}" \
  --output_dir "${CE_DIR}" \
  >"${LOGROOT}/ce_${FORGET_SPLIT}.log" 2>&1

${PY} scripts/assis_tinyllama_test_path.py \
  --model_path "${CKPT}" \
  --dataset_name "${TOFU_DATA_NAME}" \
  --dataset_split "${RETAIN_SPLIT}" \
  --forget_split "${FORGET_SPLIT}" \
  --metric fixed_sym_kl \
  --use_weighted_ce False \
  --use_length_factor False \
  --max_new_tokens "${CE_MAX_NEW_TOKENS}" \
  --batch_size "${CE_BATCH_SIZE}" \
  --max_samples "${CE_MAX_SAMPLES}" \
  --output_dir "${CE_DIR}" \
  >>"${LOGROOT}/ce_${RETAIN_SPLIT}.log" 2>&1

${PY} "${ANALYZE_CE_SCRIPT}" \
  --data-dir "${CE_DIR}" \
  --forget-split "${FORGET_SPLIT}" \
  --retain-split "${RETAIN_SPLIT}" \
  --optimize "${THRESH_OPTIMIZE}" \
  ${THRESH_MIN_TPR:+--min-tpr "${THRESH_MIN_TPR}"} \
  ${THRESH_MAX_FPR:+--max-fpr "${THRESH_MAX_FPR}"} \
  >"${LOGROOT}/threshold_${FORGET_SPLIT}.log" 2>&1

THRESHOLD_JSON="${CE_DIR}/threshold_analysis_results.json"
if [[ ! -f "${THRESHOLD_JSON}" ]]; then
  echo "[ERROR] threshold json not found: ${THRESHOLD_JSON}" >&2
  exit 1
fi

THRESHOLD=$(${PY} - "${THRESHOLD_JSON}" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)
print(data["optimal_threshold"]["best_threshold"])
PY
)

if [[ "${SKIP_ROUTING_EVAL}" != "1" ]]; then
  EVAL_SPLIT="${FORGET_SPLIT}_perturbed"
  EVAL_OUTDIR="${EVAL_ROOT}/${EVAL_SPLIT}"
  log "Stage4 eval_tofu: split=${EVAL_SPLIT} threshold=${THRESHOLD}"
  ${PY} scripts/eval_tofu.py \
    OUTDIRNAME="${EVAL_OUTDIR}" \
    ckpt_path="${CKPT}" \
    model=tofu-llama-2 \
    model.model_path="${TOFU_BASE_MODEL}" \
    model.tokenizer_path="${TOFU_BASE_MODEL}" \
    model_mode=double_assis \
    model_mode.original_assist_path="${ASSIST_MODEL}" \
    model_mode.finetuned_assist_path="${CKPT}" \
    model_mode.threshold="${THRESHOLD}" \
    model_mode.max_new_tokens=32 \
    data.dataset.name="${TOFU_DATA_NAME}" \
    data.dataset.split="${EVAL_SPLIT}" \
    data.dataset.eval.batch_size="${ROUTING_EVAL_BATCH_SIZE}" \
    data.dataset.eval.retain_result=null \
    "+data.dataset.eval.max_num=300" \
    >"${LOGROOT}/eval_forget10_perturbed.log" 2>&1
fi

cat > "${LOGROOT}/summary.json" <<EOF
{
  "run_tag": "${RUN_TAG}",
  "method": "gpm",
  "dataset": "tofu",
  "forget_split": "${FORGET_SPLIT}",
  "retain_split": "${RETAIN_SPLIT}",
  "loss": "${UNLEARN_LOSS}",
  "gpm_project_forget_only": ${GPM_PROJECT_FORGET_ONLY},
  "basis_path": "${BASIS_PATH}",
  "ckpt_path": "${CKPT}",
  "threshold_json": "${THRESHOLD_JSON}",
  "threshold": "${THRESHOLD}",
  "eval_outroot": "${EVAL_ROOT}",
  "skip_routing_eval": ${SKIP_ROUTING_EVAL}
}
EOF

log "完成: threshold_json=${THRESHOLD_JSON}"
