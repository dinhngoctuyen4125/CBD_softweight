#!/usr/bin/env bash
# 内部实现脚本。正式复现请统一从 scripts/hf_forget_train.py repro 进入。
set -euo pipefail

SEED="${1:?seed required}"
GPU="${2:-0}"
RUN_SUFFIX="${3:-gpm_wmdp_table3}"
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

WMDP_DOMAINS="${WMDP_DOMAINS:-bio,cyber,chem}"
WMDP_TRAIN_SPLIT="${WMDP_TRAIN_SPLIT:-bio_cyber_chem}"
ASSIST_MODEL="${ASSIST_MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
BASE_MODEL="${BASE_MODEL:-HuggingFaceH4/zephyr-7b-beta}"
MMLU_TRAIN="${MMLU_TRAIN:-${CBD_DATA_ROOT:-data}/eval-method/wmdp/data/mmlu/all_auxiliary_train.jsonl}"
MMLU_VAL="${MMLU_VAL:-${CBD_DATA_ROOT:-data}/eval-method/wmdp/data/mmlu/all_validation.jsonl}"
MMLU_TEST="${MMLU_TEST:-${CBD_DATA_ROOT:-data}/eval-method/wmdp/data/mmlu/all_test.jsonl}"
TRAIN_MAX_FORGET="${TRAIN_MAX_FORGET:-600}"
TRAIN_RETAIN_NUM="${TRAIN_RETAIN_NUM:-1200}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-2}"
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
TRAIN_MAX_LEN="${TRAIN_MAX_LEN:-512}"
GPM_PROJECT_FORGET_ONLY="${GPM_PROJECT_FORGET_ONLY:-1}"
GPM_TARGET_VARIANCE="${GPM_TARGET_VARIANCE:-0.9}"
GPM_MAX_SAMPLES="${GPM_MAX_SAMPLES:-1200}"
GPM_BASIS_PATH="${GPM_BASIS_PATH:-}"
THRESH_MAX_FORGET="${THRESH_MAX_FORGET:-600}"
THRESH_MAX_RETAIN="${THRESH_MAX_RETAIN:-1200}"
THRESH_BATCH_SIZE="${THRESH_BATCH_SIZE:-16}"
THRESH_MAX_LEN="${THRESH_MAX_LEN:-512}"
THRESH_OPTIMIZE="${THRESH_OPTIMIZE:-tpr}"
THRESH_MAX_FPR="${THRESH_MAX_FPR:-0.04}"
SCORE_SPACE="${SCORE_SPACE:-vocab}"
SCORE_POS="${SCORE_POS:-prompt_last}"
SCORE_LAST_K="${SCORE_LAST_K:-4}"
SCORE_LAST_K_REDUCE="${SCORE_LAST_K_REDUCE:-mean}"
SCORE_REDUCER_ALPHA="${SCORE_REDUCER_ALPHA:-1.0}"
SCORE_REDUCER_BETA="${SCORE_REDUCER_BETA:-1.0}"
SCORE_K_MODE="${SCORE_K_MODE:-last}"
TRUNCATE_MODE="${TRUNCATE_MODE:-head_tail}"
SKIP_FINAL_EVAL="${SKIP_FINAL_EVAL:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
EVAL_MAX_WMDP="${EVAL_MAX_WMDP:-0}"
EVAL_MAX_MMLU="${EVAL_MAX_MMLU:-0}"
EVAL_SCORE_LAST_K="${EVAL_SCORE_LAST_K:-1}"
EVAL_SCORE_LAST_K_REDUCE="${EVAL_SCORE_LAST_K_REDUCE:-mean}"
EVAL_SCORE_K_MODE="${EVAL_SCORE_K_MODE:-last}"
EVAL_TRUNCATE_MODE="${EVAL_TRUNCATE_MODE:-left}"

RUN_TAG="wmdp_bb_seed${SEED}_${RUN_SUFFIX}"
LOGROOT="artifacts/seed_runs/${RUN_TAG}"
OUTMODELDIR="artifacts/outputs_trained_models/gpm_wmdp_blackbox/${RUN_TAG}"
BASIS_ROOT="artifacts/basis_gpm/${RUN_TAG}"
EVAL_ROOT="artifacts/eval_outputs/wmdp_blackbox/${RUN_TAG}"

mkdir -p "${LOGROOT}" "${OUTMODELDIR}" "${BASIS_ROOT}" "${EVAL_ROOT}"

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

log "==== GPM WMDP Table3 启动: seed=${SEED} loss=${UNLEARN_LOSS} pf_only=${GPM_PROJECT_FORGET_ONLY} ===="

BASIS_DIR="${BASIS_ROOT}/wmdp_${WMDP_DOMAINS//,/_}_vs_mmlu"
mkdir -p "${BASIS_DIR}"
if [[ -n "${GPM_BASIS_PATH}" ]]; then
  BASIS_PATH="${GPM_BASIS_PATH}"
  log "Stage1 跳过，复用现成基底: ${BASIS_PATH}"
else
  log "Stage1 基底提取: mmlu retain max_samples=${GPM_MAX_SAMPLES} tv=${GPM_TARGET_VARIANCE}"
  ${PY} scripts/extract_retain_basis_svd.py \
    --base_model_name "${ASSIST_MODEL}" \
    --dataset wmdp_mmlu \
    --mmlu_retain_file "${MMLU_TRAIN}" \
    --target_variance "${GPM_TARGET_VARIANCE}" \
    --max_samples "${GPM_MAX_SAMPLES}" \
    --max_len "${TRAIN_MAX_LEN}" \
    --lora_r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout "${LORA_DROPOUT}" \
    --output_dir "${BASIS_DIR}" \
    >"${LOGROOT}/basis.log" 2>&1
  BASIS_PATH="${BASIS_DIR}/gpm_retain_basis.pkl"
fi
if [[ ! -f "${BASIS_PATH}" ]]; then
  echo "[ERROR] basis not found: ${BASIS_PATH}" >&2
  exit 1
fi

PROJECT_NAME="gpm_wmdp_blackbox"
log "Stage2 训练A1: retain_num=${TRAIN_RETAIN_NUM} epochs=${TRAIN_EPOCHS} lr=${TRAIN_LR}"
${PY} scripts/hf_forget_train.py \
  --config-name cbd_dfb_tinyllama_wmdp \
  hydra.run.dir="artifacts/outputs/gpm_log_wmdp_blackbox/${RUN_TAG}/hydra" \
  data.dataset.split="${WMDP_TRAIN_SPLIT}" \
  data.dataset.mmlu_retain_file="${MMLU_TRAIN}" \
  data.dataset.max_forget="${TRAIN_MAX_FORGET}" \
  data.dataset.max_len="${TRAIN_MAX_LEN}" \
  data.conv_template.max_len="${TRAIN_MAX_LEN}" \
  data_mode=forget_retain \
  data_mode.retain_num="${TRAIN_RETAIN_NUM}" \
  enable_cbd_dfb=false \
  enable_gmp=true \
  gmp_basis_path="${BASIS_PATH}" \
  gmp_project_forget_only="${GPM_PROJECT_FORGET_ONLY}" \
  project="${PROJECT_NAME}" \
  seed="${SEED}" \
  lora_seed="${SEED}" \
  oracle_on_cpu=false \
  gradient_checkpointing=false \
  trainer.max_epochs="${TRAIN_EPOCHS}" \
  trainer.learning_rate="${TRAIN_LR}" \
  trainer.batch_size="${TRAIN_BATCH_SIZE}" \
  trainer.gradient_accumulation_steps="${TRAIN_GRAD_ACC}" \
  trainer.weight_decay="${TRAIN_WEIGHT_DECAY}" \
  model_mode.Lora.r="${LORA_R}" \
  model_mode.Lora.alpha="${LORA_ALPHA}" \
  model_mode.Lora.dropout="${LORA_DROPOUT}" \
  ${TRAIN_MAX_STEPS:+++trainer.max_steps="${TRAIN_MAX_STEPS}"} \
  unlearn_loss="${UNLEARN_LOSS}" \
  ${FORGET_WEIGHT:+unlearn_loss.forget_weight="${FORGET_WEIGHT}"} \
  ${RETAIN_WEIGHT:+unlearn_loss.retain_weight="${RETAIN_WEIGHT}"} \
  OUTPUTMODELDIR="${OUTMODELDIR}" \
  BASELOGDIR="artifacts/outputs/gpm_log_wmdp_blackbox/${RUN_TAG}" \
  >"${LOGROOT}/train.log" 2>&1

CKPT=$(find_latest_ckpt "${OUTMODELDIR}" "project_${PROJECT_NAME}" || true)
if [[ -z "${CKPT}" ]]; then
  CKPT=$(find "${OUTMODELDIR}" -type d -name 'checkpoint-*' -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-)
fi
if [[ -z "${CKPT}" ]]; then
  echo "[ERROR] training checkpoint not found under ${OUTMODELDIR}" >&2
  exit 1
fi
echo "${CKPT}" > "${LOGROOT}/ckpt_path.txt"

log "Stage3 阈值选择: max_forget=${THRESH_MAX_FORGET} max_retain=${THRESH_MAX_RETAIN}"
${PY} scripts/select_wmdp_routing_threshold.py \
  --original_assist_path "${ASSIST_MODEL}" \
  --finetuned_assist_path "${CKPT}" \
  --base_if_lora "${ASSIST_MODEL}" \
  --wmdp_domains "${WMDP_DOMAINS}" \
  --mmlu_retain_file "${MMLU_VAL}" \
  --max_forget "${THRESH_MAX_FORGET}" \
  --max_retain "${THRESH_MAX_RETAIN}" \
  --batch_size "${THRESH_BATCH_SIZE}" \
  --max_len "${THRESH_MAX_LEN}" \
  --optimize "${THRESH_OPTIMIZE}" \
  --score_space "${SCORE_SPACE}" \
  --score_pos "${SCORE_POS}" \
  --score_last_k "${SCORE_LAST_K}" \
  --score_last_k_reduce "${SCORE_LAST_K_REDUCE}" \
  --score_reducer_alpha "${SCORE_REDUCER_ALPHA}" \
  --score_reducer_beta "${SCORE_REDUCER_BETA}" \
  --score_k_mode "${SCORE_K_MODE}" \
  --truncate_mode "${TRUNCATE_MODE}" \
  --max_fpr "${THRESH_MAX_FPR}" \
  --output_json "${EVAL_ROOT}/threshold.json" \
  >"${LOGROOT}/threshold.log" 2>&1

THRESHOLD_JSON="${EVAL_ROOT}/threshold.json"
if [[ ! -f "${THRESHOLD_JSON}" ]]; then
  echo "[ERROR] threshold json not found: ${THRESHOLD_JSON}" >&2
  exit 1
fi

if [[ "${SKIP_FINAL_EVAL}" != "1" ]]; then
  log "Stage4 routed eval: WMDP+MMLU"
  THRESHOLD_VALUE="$(${PY} - "${THRESHOLD_JSON}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
print(json.loads(path.read_text(encoding="utf-8"))["selection"]["best_threshold"])
PY
)"
  ${PY} scripts/eval_wmdp_routing.py \
    --finetuned_assist_path "${CKPT}" \
    --threshold "${THRESHOLD_VALUE}" \
    --base_model "${BASE_MODEL}" \
    --original_assist "${ASSIST_MODEL}" \
    --assist_base_if_lora "${ASSIST_MODEL}" \
    --base_tokenizer "${BASE_MODEL}" \
    --assist_tokenizer "${ASSIST_MODEL}" \
    --base_device cuda:0 \
    --original_device cuda:0 \
    --finetuned_device cuda:0 \
    --mmlu_test_file "${MMLU_TEST}" \
    --batch_size "${EVAL_BATCH_SIZE}" \
    --max_len "${TRAIN_MAX_LEN}" \
    --max_mmlu "${EVAL_MAX_MMLU}" \
    --max_wmdp "${EVAL_MAX_WMDP}" \
    --seed "${SEED}" \
    --score_space "${SCORE_SPACE}" \
    --score_pos "${SCORE_POS}" \
    --score_last_k "${EVAL_SCORE_LAST_K}" \
    --score_last_k_reduce "${EVAL_SCORE_LAST_K_REDUCE}" \
    --score_reducer_alpha "${SCORE_REDUCER_ALPHA}" \
    --score_reducer_beta "${SCORE_REDUCER_BETA}" \
    --score_k_mode "${EVAL_SCORE_K_MODE}" \
    --truncate_mode "${EVAL_TRUNCATE_MODE}" \
    --out_json "${EVAL_ROOT}/eval.json" \
    >"${LOGROOT}/eval.log" 2>&1
fi

cat > "${LOGROOT}/summary.json" <<EOF
{
  "run_tag": "${RUN_TAG}",
  "method": "gpm",
  "dataset": "wmdp",
  "loss": "${UNLEARN_LOSS}",
  "gpm_project_forget_only": ${GPM_PROJECT_FORGET_ONLY},
  "basis_path": "${BASIS_PATH}",
  "ckpt_path": "${CKPT}",
  "threshold_json": "${THRESHOLD_JSON}",
  "eval_json": "${EVAL_ROOT}/eval.json",
  "skip_final_eval": ${SKIP_FINAL_EVAL}
}
EOF

log "完成: threshold_json=${THRESHOLD_JSON}"
