#!/usr/bin/env bash
set -euo pipefail

# Full CBD-DFB pipeline for one seed:
# 1) Extract CBD-DFB basis (forget vs retain) using gradient statistics
# 2) Train A1 with CBD-DFB projected gradients
# 3) CE threshold + routing eval

SEED="${1:?seed required}"
GPU="${2:-0}"
RUN_SUFFIX="${3:-}"
UNLEARN_LOSS="${4:-gd+kl}"
LORA_SEED="${LORA_SEED:-${SEED}}"
SPLITS="${SPLITS:-forget01 forget05 forget10}"
DATA_MODE="${DATA_MODE:-forget_retain}"
MU_MODE="${MU_MODE:-auto}"
MU_SCALE="${MU_SCALE:-1e-2}"
TARGET_VARIANCE="${TARGET_VARIANCE:-0.9}"
TOP_K="${TOP_K:-32}"
MAX_FORGET="${MAX_FORGET:-400}"
MAX_RETAIN="${MAX_RETAIN:-400}"
BASIS_MAX_FORGET="${BASIS_MAX_FORGET:-${MAX_FORGET}}"
BASIS_MAX_RETAIN="${BASIS_MAX_RETAIN:-${MAX_RETAIN}}"
TRAIN_RETAIN_NUM="${TRAIN_RETAIN_NUM:-}"
BASIS_MAX_LEN="${BASIS_MAX_LEN:-128}"
BASIS_BATCH_SIZE="${BASIS_BATCH_SIZE:-4}"
BASIS_GRAD_STORE_DTYPE="${BASIS_GRAD_STORE_DTYPE:-float16}"
CBD_DFB_EIGVAL_WEIGHT="${CBD_DFB_EIGVAL_WEIGHT:-false}"
CBD_DFB_TRUST_REGION="${CBD_DFB_TRUST_REGION:-0}"
CBD_DFB_TRUST_EPS="${CBD_DFB_TRUST_EPS:-1e-3}"
CBD_DFB_TRUST_DELTA="${CBD_DFB_TRUST_DELTA:-1e-12}"
CBD_DFB_PROJECT_FORGET_ONLY="${CBD_DFB_PROJECT_FORGET_ONLY:-0}"
REFUSE_ANSWER="${REFUSE_ANSWER:-}"
if [[ -z "${REFUSE_ANSWER}" ]]; then
  REFUSE_ANSWER="I don't know."
fi
TRAIN_EPOCHS="${TRAIN_EPOCHS:-10}"
TRAIN_LR="${TRAIN_LR:-2e-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
TRAIN_GRAD_ACC="${TRAIN_GRAD_ACC:-4}"
TRAIN_WEIGHT_DECAY="${TRAIN_WEIGHT_DECAY:-0.01}"
TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-}"
MODEL_ATTN_IMPL="${MODEL_ATTN_IMPL:-}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-}"
CE_MAX_NEW_TOKENS="${CE_MAX_NEW_TOKENS:-32}"
CE_BATCH_SIZE="${CE_BATCH_SIZE:-8}"
CE_METRIC="${CE_METRIC:-fixed_sym_kl}"
ROUTING_SCORE_KEY="${ROUTING_SCORE_KEY:-cross_entropy}"
CE_MAX_SAMPLES="${CE_MAX_SAMPLES:-300}"
SKIP_ROUTING_EVAL="${SKIP_ROUTING_EVAL:-1}"
ROUTING_EVAL_BATCH_SIZE="${ROUTING_EVAL_BATCH_SIZE:-4}"
THRESH_OPTIMIZE="${THRESH_OPTIMIZE:-accuracy}"
THRESH_MIN_TPR="${THRESH_MIN_TPR:-}"
THRESH_MAX_FPR="${THRESH_MAX_FPR:-}"
SKIP_BASIS="${SKIP_BASIS:-0}"
RETAIN_WEIGHT="${RETAIN_WEIGHT:-}"
FORGET_WEIGHT="${FORGET_WEIGHT:-}"
BASIS_ROOT_OVERRIDE="${BASIS_ROOT_OVERRIDE:-}"
LORA_R="${LORA_R:-}"
LORA_ALPHA="${LORA_ALPHA:-}"
LORA_DROPOUT="${LORA_DROPOUT:-}"
ASSIST_MODEL="${ASSIST_MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
TOFU_BASE_MODEL="${TOFU_BASE_MODEL:-locuslab/tofu_ft_llama2-7b}"
TOFU_DATA_NAME="${TOFU_DATA_NAME:-${CBD_DATA_ROOT:-data}/TOFU}"
TOFU_DATASET_NAME="${TOFU_DATASET_NAME:-${TOFU_DATA_NAME}}"

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

ANALYZE_CE_SCRIPT="${ANALYZE_CE_SCRIPT:-scripts/analyze_cross_entropy.py}"
if [[ ! -f "${ANALYZE_CE_SCRIPT}" ]]; then
  echo "ERROR: analyze_cross_entropy.py not found" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED="${SEED}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONPATH="${PYTHONPATH:-.}"
export FORCE_SAVE_FINAL_CHECKPOINT="${FORCE_SAVE_FINAL_CHECKPOINT:-1}"

if [[ -n "${RUN_SUFFIX}" ]]; then
  RUN_TAG="seed${SEED}_${RUN_SUFFIX}"
else
  RUN_TAG="seed${SEED}"
fi

LOGROOT="artifacts/seed_runs/${RUN_TAG}"
mkdir -p "${LOGROOT}"

OUTMODELDIR="${OUTPUTMODELDIR_OVERRIDE:-artifacts/outputs_trained_models/cbd_dfb_tinyllama_${RUN_TAG}}"
BASIS_ROOT="${BASIS_ROOT_OVERRIDE:-artifacts/basis_cbd_dfb/${RUN_TAG}}"
CE_ROOT="artifacts/ce_results_cbd_dfb/${RUN_TAG}"
EVAL_OUTROOT="artifacts/eval_outputs/tofu/double_assis_routing_cbd_dfb_${RUN_TAG}"
BASELOGDIR_OVERRIDE="${BASELOGDIR_OVERRIDE:-}"
if [[ -n "${BASELOGDIR_OVERRIDE}" ]]; then
  BASELOGDIR_VALUE="${BASELOGDIR_OVERRIDE}"
else
  BASELOGDIR_VALUE="artifacts/outputs/cbd_dfb_log_${RUN_TAG}"
fi

mkdir -p "${OUTMODELDIR}" "${BASIS_ROOT}" "${CE_ROOT}" "${EVAL_OUTROOT}"

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

extract_basis() {
  local forget_split="$1"
  local retain_split="$2"
  local out_dir="$3"
  log "Extract CBD-DFB basis: forget=${forget_split} retain=${retain_split}"
  local -a extra_args=()
  if [[ "${DATA_MODE}" == *"refuse"* ]]; then
    extra_args+=(--refuse_forget --refuse_answer "${REFUSE_ANSWER}")
  fi
  if [[ -n "${LORA_R}" ]]; then
    extra_args+=(--lora_r "${LORA_R}")
  fi
  if [[ -n "${LORA_ALPHA}" ]]; then
    extra_args+=(--lora_alpha "${LORA_ALPHA}")
  fi
  if [[ -n "${LORA_DROPOUT}" ]]; then
    extra_args+=(--lora_dropout "${LORA_DROPOUT}")
  fi
  "${PY}" scripts/extract_cbd_dfb_basis.py \
    --base_model_name "${ASSIST_MODEL}" \
    --seed "${LORA_SEED}" \
    --forget_split "${forget_split}" \
    --retain_split "${retain_split}" \
    --max_forget "${BASIS_MAX_FORGET}" \
    --max_retain "${BASIS_MAX_RETAIN}" \
    --max_len "${BASIS_MAX_LEN}" \
    --batch_size "${BASIS_BATCH_SIZE}" \
    --grad_store_dtype "${BASIS_GRAD_STORE_DTYPE}" \
    --mu-mode "${MU_MODE}" \
    --mu-scale "${MU_SCALE}" \
    --target_variance "${TARGET_VARIANCE}" \
    --top_k "${TOP_K}" \
    "${extra_args[@]}" \
    --output_dir "${out_dir}" \
    >"${LOGROOT}/basis_${forget_split}.log" 2>&1
}

train_cbd_dfb() {
  local forget_split="$1"
  local basis_path="$2"
  local project="$3"
  log "Train CBD-DFB: split=${forget_split} project=${project}"
  local -a extra_args=()
  if [[ "${CBD_DFB_TRUST_REGION}" == "1" ]]; then
    extra_args+=(
      cbd_dfb_trust_region=true
      cbd_dfb_trust_region_epsilon="${CBD_DFB_TRUST_EPS}"
      cbd_dfb_trust_region_delta="${CBD_DFB_TRUST_DELTA}"
    )
  fi
  if [[ "${CBD_DFB_PROJECT_FORGET_ONLY}" == "1" ]]; then
    extra_args+=(cbd_dfb_project_forget_only=true)
  fi
  if [[ -n "${TRAIN_RETAIN_NUM}" ]]; then
    extra_args+=(data_mode.retain_num="${TRAIN_RETAIN_NUM}")
  fi
  if [[ -n "${LORA_R}" ]]; then
    extra_args+=(model_mode.Lora.r="${LORA_R}")
  fi
  if [[ -n "${LORA_ALPHA}" ]]; then
    extra_args+=(model_mode.Lora.alpha="${LORA_ALPHA}")
  fi
  if [[ -n "${LORA_DROPOUT}" ]]; then
    extra_args+=(model_mode.Lora.dropout="${LORA_DROPOUT}")
  fi
  if [[ -n "${TRAIN_MAX_STEPS}" ]]; then
    extra_args+=(++trainer.max_steps="${TRAIN_MAX_STEPS}")
  fi
  if [[ -n "${MODEL_ATTN_IMPL}" ]]; then
    extra_args+=(model.attn_implementation="${MODEL_ATTN_IMPL}")
  fi
  if [[ -n "${GRADIENT_CHECKPOINTING}" ]]; then
    extra_args+=(++gradient_checkpointing="${GRADIENT_CHECKPOINTING}")
  fi
  "${PY}" scripts/hf_forget_train.py \
    --config-name cbd_dfb_tinyllama_tofu \
    data.dataset.split="${forget_split}" \
    data.dataset.name="${TOFU_DATASET_NAME}" \
    data_mode="${DATA_MODE}" \
    enable_cbd_dfb=true \
    cbd_dfb_basis_path="${basis_path}" \
    cbd_dfb_eigval_weight="${CBD_DFB_EIGVAL_WEIGHT}" \
    project="${project}" \
    seed="${SEED}" \
    lora_seed="${LORA_SEED}" \
    trainer.max_epochs="${TRAIN_EPOCHS}" \
    trainer.learning_rate="${TRAIN_LR}" \
    trainer.batch_size="${TRAIN_BATCH_SIZE}" \
    trainer.gradient_accumulation_steps="${TRAIN_GRAD_ACC}" \
    trainer.weight_decay="${TRAIN_WEIGHT_DECAY}" \
    model=tinyllama \
    model.model_path="${ASSIST_MODEL}" \
    model.tokenizer_path="${ASSIST_MODEL}" \
    model_mode=base_freeze_a \
    unlearn_loss="${UNLEARN_LOSS}" \
    ${RETAIN_WEIGHT:+unlearn_loss.retain_weight="${RETAIN_WEIGHT}"} \
    ${FORGET_WEIGHT:+unlearn_loss.forget_weight="${FORGET_WEIGHT}"} \
    "${extra_args[@]}" \
    OUTPUTMODELDIR="${OUTMODELDIR}" \
    BASELOGDIR="${BASELOGDIR_VALUE}" \
    >"${LOGROOT}/${project}_train.log" 2>&1
}

run_ce_and_threshold() {
  local forget_split="$1"
  local retain_split="$2"
  local finetuned_ckpt="$3"
  local ce_outdir="$4"
  log "CE+threshold: forget=${forget_split} retain=${retain_split}"
  mkdir -p "${ce_outdir}"
  "${PY}" scripts/assis_tinyllama_test_path.py \
    --model_path "${finetuned_ckpt}" \
    --pretrained_model_name "${ASSIST_MODEL}" \
    --dataset_name "${TOFU_DATASET_NAME}" \
    --dataset_split "${forget_split}" \
    --forget_split "${forget_split}" \
    --metric "${CE_METRIC}" \
    --use_weighted_ce False \
    --use_length_factor False \
    --max_new_tokens "${CE_MAX_NEW_TOKENS}" \
    --batch_size "${CE_BATCH_SIZE}" \
    --max_samples "${CE_MAX_SAMPLES}" \
    --output_dir "${ce_outdir}" \
    >"${LOGROOT}/ce_${forget_split}.log" 2>&1

  "${PY}" scripts/assis_tinyllama_test_path.py \
    --model_path "${finetuned_ckpt}" \
    --pretrained_model_name "${ASSIST_MODEL}" \
    --dataset_name "${TOFU_DATASET_NAME}" \
    --dataset_split "${retain_split}" \
    --forget_split "${forget_split}" \
    --metric "${CE_METRIC}" \
    --use_weighted_ce False \
    --use_length_factor False \
    --max_new_tokens "${CE_MAX_NEW_TOKENS}" \
    --batch_size "${CE_BATCH_SIZE}" \
    --max_samples "${CE_MAX_SAMPLES}" \
    --output_dir "${ce_outdir}" \
    >>"${LOGROOT}/ce_${retain_split}.log" 2>&1

  "${PY}" "${ANALYZE_CE_SCRIPT}" \
    --data-dir "${ce_outdir}" \
    --forget-split "${forget_split}" \
    --retain-split "${retain_split}" \
    --metric-key "${ROUTING_SCORE_KEY}" \
    --optimize "${THRESH_OPTIMIZE}" \
    ${THRESH_MIN_TPR:+--min-tpr "${THRESH_MIN_TPR}"} \
    ${THRESH_MAX_FPR:+--max-fpr "${THRESH_MAX_FPR}"} \
    >"${LOGROOT}/threshold_${forget_split}.log" 2>&1
}

run_routing_eval() {
  local split="$1"
  local finetuned_ckpt="$2"
  local threshold="$3"
  local outdir="${EVAL_OUTROOT}/${split}"
  log "Routing eval: split=${split} threshold=${threshold}"
  "${PY}" scripts/eval_tofu.py \
    OUTDIRNAME="${outdir}" \
    ckpt_path="${finetuned_ckpt}" \
    model=tofu-llama-2 \
    model.model_path="${TOFU_BASE_MODEL}" \
    model.tokenizer_path="${TOFU_BASE_MODEL}" \
    model_mode=double_assis \
    model_mode.original_assist_path="${ASSIST_MODEL}" \
    model_mode.finetuned_assist_path="${finetuned_ckpt}" \
    model_mode.threshold="${threshold}" \
    model_mode.max_new_tokens="${CE_MAX_NEW_TOKENS}" \
    data.dataset.name="${TOFU_DATASET_NAME}" \
    data.dataset.split="${split}" \
    data.dataset.eval.batch_size="${ROUTING_EVAL_BATCH_SIZE}" \
    data.dataset.eval.retain_result=null \
    >"${LOGROOT}/eval_${split}.log" 2>&1
}

declare -A RETAIN_MAP=(
  ["forget01"]="retain99"
  ["forget05"]="retain95"
  ["forget10"]="retain90"
)

log "==== [Seed ${SEED}] Stage 1: basis extraction ===="
for forget_split in ${SPLITS}; do
  retain_split="${RETAIN_MAP[${forget_split}]}"
  basis_dir="${BASIS_ROOT}/${forget_split}_vs_${retain_split}"
  basis_path="${basis_dir}/cbd_dfb_basis_${forget_split}_vs_${retain_split}.pkl"
  basis_wall_start="$(date +%s)"
  if [[ "${SKIP_BASIS}" == "1" && -f "${basis_path}" ]]; then
    log "Skip basis extraction (SKIP_BASIS=1): ${basis_path}"
  else
    extract_basis "${forget_split}" "${retain_split}" "${basis_dir}"
  fi
  basis_wall_end="$(date +%s)"
  echo "${basis_wall_start} ${basis_wall_end}" >"${LOGROOT}/timing_basis_${forget_split}.txt"
done

log "==== [Seed ${SEED}] Stage 2: train + CE + eval ===="
for forget_split in ${SPLITS}; do
  retain_split="${RETAIN_MAP[${forget_split}]}"
  basis_path="${BASIS_ROOT}/${forget_split}_vs_${retain_split}/cbd_dfb_basis_${forget_split}_vs_${retain_split}.pkl"
  if [[ ! -f "${basis_path}" ]]; then
    echo "ERROR: basis not found ${basis_path}" | tee -a "${LOGROOT}/errors.log"
    exit 1
  fi

  proj="cbd_dfb_${forget_split}_seed${SEED}"
  train_wall_start="$(date +%s)"
  train_cbd_dfb "${forget_split}" "${basis_path}" "${proj}"
  train_wall_end="$(date +%s)"
  echo "${train_wall_start} ${train_wall_end}" >"${LOGROOT}/timing_train_${forget_split}.txt"

  ckpt=$(find_latest_ckpt "${OUTMODELDIR}" "project_${proj}" || true)
  if [[ -z "${ckpt}" ]]; then
    echo "ERROR: ckpt not found for ${forget_split}" | tee -a "${LOGROOT}/errors.log"
    exit 1
  fi
  printf '%s\n' "${ckpt}" >"${LOGROOT}/ckpt_path_${forget_split}.txt"

  ce_dir="${CE_ROOT}/${proj}"
  mkdir -p "${ce_dir}"

  ce_combined_start="$(date +%s)"
  "${PY}" scripts/assis_tinyllama_test_path.py \
    --model_path "${ckpt}" \
    --pretrained_model_name "${ASSIST_MODEL}" \
    --dataset_name "${TOFU_DATASET_NAME}" \
    --dataset_split "${forget_split},${retain_split}" \
    --forget_split "${forget_split}" \
    --metric "${CE_METRIC}" \
    --use_weighted_ce False \
    --use_length_factor False \
    --max_new_tokens "${CE_MAX_NEW_TOKENS}" \
    --batch_size "${CE_BATCH_SIZE}" \
    --max_samples "${CE_MAX_SAMPLES}" \
    --output_dir "${ce_dir}" \
    >"${LOGROOT}/ce_${forget_split}_${retain_split}.log" 2>&1
  ce_combined_end="$(date +%s)"

  thresh_start="$(date +%s)"
  "${PY}" "${ANALYZE_CE_SCRIPT}" \
    --data-dir "${ce_dir}" \
    --forget-split "${forget_split}" \
    --retain-split "${retain_split}" \
    --metric-key "${ROUTING_SCORE_KEY}" \
    --optimize "${THRESH_OPTIMIZE}" \
    ${THRESH_MIN_TPR:+--min-tpr "${THRESH_MIN_TPR}"} \
    ${THRESH_MAX_FPR:+--max-fpr "${THRESH_MAX_FPR}"} \
    >"${LOGROOT}/threshold_${forget_split}.log" 2>&1
  thresh_end="$(date +%s)"

  echo "${ce_combined_start} ${ce_combined_end}" >"${LOGROOT}/timing_ce_combined_${forget_split}.txt"
  echo "${thresh_start} ${thresh_end}" >"${LOGROOT}/timing_threshold_${forget_split}.txt"

  threshold=$("${PY}" - "${ce_dir}/threshold_analysis_results.json" <<'PY'
import json
import sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
print(data["optimal_threshold"]["best_threshold"])
PY
)

  # Print classification summary (no ToFU eval needed during iteration)
  "${PY}" - "${ce_dir}/threshold_analysis_results.json" <<'PY'
import json
import sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
opt = data.get("optimal_threshold", {})
fpr = opt.get("retained_misidentification_rate")
tnr = None
if fpr is not None:
    try:
        tnr = 100.0 - float(fpr)
    except Exception:
        tnr = None
print("CBD-DFB classification summary:")
print("  threshold:", opt.get("best_threshold"))
print("  accuracy :", opt.get("accuracy"))
print("  TPR      :", opt.get("forgotten_identification_rate"))
print("  FPR      :", opt.get("retained_misidentification_rate"))
if tnr is not None:
    print("  TNR      :", tnr)
print("  gap      :", opt.get("gap_score"))
if "constraints_satisfied" in opt:
    print("  constraints_satisfied:", opt.get("constraints_satisfied"))
PY

  if [[ "${SKIP_ROUTING_EVAL}" != "1" ]]; then
    eval_wall_start="$(date +%s)"
    run_routing_eval "${forget_split}_perturbed" "${ckpt}" "${threshold}"
    eval_wall_end="$(date +%s)"
    echo "${eval_wall_start} ${eval_wall_end}" >"${LOGROOT}/timing_eval_${forget_split}.txt"
  else
    log "Skip routing ToFU eval (SKIP_ROUTING_EVAL=1)"
  fi

  # Persist per-split timing summary for reproducibility
  "${PY}" - "${LOGROOT}" "${forget_split}" "${retain_split}" "${RUN_TAG}" "${OUTMODELDIR}" "${BASIS_ROOT}" "${CE_ROOT}" "${proj}" "${ckpt}" <<'PY'
import json
import re
import sys
from pathlib import Path

logroot = Path(sys.argv[1])
forget_split = sys.argv[2]
retain_split = sys.argv[3]
run_tag = sys.argv[4]
out_model_dir = sys.argv[5]
basis_root = sys.argv[6]
ce_root = sys.argv[7]
proj = sys.argv[8]
ckpt = sys.argv[9]

def read_pair(path: Path):
    if not path.exists():
        return None
    s = path.read_text(encoding="utf-8").strip().split()
    if len(s) != 2:
        return None
    a, b = int(s[0]), int(s[1])
    return {"start": a, "end": b, "sec": max(0, b - a)}

def parse_basis_timing(path: Path):
    if not path.exists():
        return None
    m = None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "[timing]" in line:
            m = line
    if not m:
        return None
    # [timing] forget_grads_sec=15.6 retain_grads_sec=100.8 basis_sec=24.9 total_sec=141.3
    out = {}
    for k in ["forget_grads_sec", "retain_grads_sec", "basis_sec", "total_sec"]:
        mm = re.search(rf"{k}=([0-9.]+)", m)
        if mm:
            out[k] = float(mm.group(1))
    return out or None

def parse_train_runtime(path: Path):
    if not path.exists():
        return None
    # find last occurrence of 'train_runtime': 1145.6473
    rt = None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "'train_runtime':" in line:
            mm = re.search(r"'train_runtime':\\s*([0-9.]+)", line)
            if mm:
                rt = float(mm.group(1))
    return rt

payload = {
    "run_tag": run_tag,
    "forget_split": forget_split,
    "retain_split": retain_split,
    "paths": {
        "logroot": str(logroot),
        "out_model_dir": out_model_dir,
        "basis_root": basis_root,
        "ce_root": ce_root,
        "project": proj,
        "ckpt": ckpt,
        "threshold_json": str(Path(ce_root) / proj / "threshold_analysis_results.json"),
    },
    "timings": {
        "basis_wall": read_pair(logroot / f"timing_basis_{forget_split}.txt"),
        "train_wall": read_pair(logroot / f"timing_train_{forget_split}.txt"),
        "ce_forget_wall": read_pair(logroot / f"timing_ce_forget_{forget_split}.txt"),
        "ce_retain_wall": read_pair(logroot / f"timing_ce_retain_{forget_split}.txt"),
        "ce_combined_wall": read_pair(logroot / f"timing_ce_combined_{forget_split}.txt"),
        "threshold_wall": read_pair(logroot / f"timing_threshold_{forget_split}.txt"),
        "eval_wall": read_pair(logroot / f"timing_eval_{forget_split}.txt"),
        "basis_detail": parse_basis_timing(logroot / f"basis_{forget_split}.log"),
        "train_runtime": parse_train_runtime(logroot / f"{proj}_train.log"),
    },
}

out_path = logroot / f"timing_{forget_split}.json"
out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"[timing] wrote {out_path}")
PY
done

log "==== [Seed ${SEED}] DONE ===="
log "Outputs:"
log "  Models: ${OUTMODELDIR}"
log "  Bases : ${BASIS_ROOT}"
log "  CE    : ${CE_ROOT}"
log "  Eval  : ${EVAL_OUTROOT}"
