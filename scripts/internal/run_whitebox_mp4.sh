#!/usr/bin/env bash
# 内部实现脚本。正式复现请统一从 scripts/hf_forget_train.py repro 进入。
# 白盒方法4卡模型并行训练脚本
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

CONDA_SH="${CONDA_SH:-}"
if [[ -f "${CONDA_SH}" ]]; then
  source "${CONDA_SH}"
  conda activate "${REPRO_CONDA_ENV:-cbd}" >/dev/null 2>&1 || true
fi

PY="${PYTHON:-python}"
if [[ ! -x "${PY}" ]]; then
  PY="python3"
fi

METHOD="${1:?method required: ga, ga+gd, ga+kl, dpo, dpo+gd, dpo+kl, npo, npo+gd, npo+kl}"
SPLIT="${2:?split required: forget01, forget05, forget10}"
SEED="${3:-42}"
TRAIN_SPLIT="${TRAIN_SPLIT:-${SPLIT}}"
EVAL_SPLIT_OVERRIDE="${EVAL_SPLIT_OVERRIDE:-}"
WHITEBOX_PROTOCOL="${WHITEBOX_PROTOCOL:-fair_tinyllama}"
TRAIN_LR_RAW="${TRAIN_LR-}"
TRAIN_BATCH_SIZE_RAW="${TRAIN_BATCH_SIZE-}"
TRAIN_GRAD_ACC_RAW="${TRAIN_GRAD_ACC-}"
TRAIN_WEIGHT_DECAY_RAW="${TRAIN_WEIGHT_DECAY-}"
LORA_R_RAW="${LORA_R-}"
LORA_ALPHA_RAW="${LORA_ALPHA-}"
LORA_DROPOUT_RAW="${LORA_DROPOUT-}"
EVAL_BATCH_SIZE_RAW="${EVAL_BATCH_SIZE-}"
MODEL_CFG_RAW="${MODEL_CFG_OVERRIDE-}"
MODEL_PATH_OVERRIDE_RAW="${MODEL_PATH_OVERRIDE-}"
TOKENIZER_PATH_OVERRIDE_RAW="${TOKENIZER_PATH_OVERRIDE-}"
CONV_TEMPLATE_STYLE_RAW="${CONV_TEMPLATE_STYLE-}"
TRAIN_OPTIM_RAW="${TRAIN_OPTIM-}"
TRAIN_LOGGING_STEPS_RAW="${TRAIN_LOGGING_STEPS-}"
TRAIN_DATALOADER_NUM_WORKERS_RAW="${TRAIN_DATALOADER_NUM_WORKERS-}"
WHITEBOX_RETAIN_MATCH_FORGET_RAW="${WHITEBOX_RETAIN_MATCH_FORGET-}"

# 固定超参数
TRAIN_LR="${TRAIN_LR:-1.5e-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
TRAIN_GRAD_ACC="${TRAIN_GRAD_ACC:-2}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-10}"
TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-none}"
TRAIN_WARMUP_RATIO="${TRAIN_WARMUP_RATIO:-0.1}"
TRAIN_WEIGHT_DECAY="${TRAIN_WEIGHT_DECAY:-0}"
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
RUN_EVAL="${RUN_EVAL:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
EVAL_MAX_NUM="${EVAL_MAX_NUM:-300}"
RUN_TAG_SUFFIX="${RUN_TAG_SUFFIX:-}"
MODEL_CFG="${MODEL_CFG_OVERRIDE:-tinyllama}"
TOFU_DATA_NAME="${TOFU_DATA_NAME:-${CBD_DATA_ROOT:-data}/TOFU}"
TOFU_AUG_ROOT="${TOFU_AUG_ROOT:-${CBD_DATA_ROOT:-data}/data/aug_data/tofu}"
MODEL_PATH_OVERRIDE="${MODEL_PATH_OVERRIDE:-}"
TOKENIZER_PATH_OVERRIDE="${TOKENIZER_PATH_OVERRIDE:-}"
CONV_TEMPLATE_STYLE="${CONV_TEMPLATE_STYLE:-default}"
DISABLE_INTERNAL_EVAL="${DISABLE_INTERNAL_EVAL:-1}"
FORCE_SAVE_FINAL_CHECKPOINT="${FORCE_SAVE_FINAL_CHECKPOINT:-0}"
TRAIN_OPTIM="${TRAIN_OPTIM:-adamw_torch}"
TRAIN_LOGGING_STEPS="${TRAIN_LOGGING_STEPS:-10}"
TRAIN_DATALOADER_NUM_WORKERS="${TRAIN_DATALOADER_NUM_WORKERS:-4}"
OFFICIAL_TRAINER_BEHAVIOR="${OFFICIAL_TRAINER_BEHAVIOR:-0}"
OFFICIAL_ULD_MODEL_UTILS="${OFFICIAL_ULD_MODEL_UTILS:-0}"
WHITEBOX_RETAIN_MATCH_FORGET="${WHITEBOX_RETAIN_MATCH_FORGET:-1}"

# 4卡模型并行环境变量。单卡 eval 阶段通过入口设置 SKIP_TRAIN=1,
# USE_MODEL_PARALLEL=0，避免 eval 继承训练期的 4 卡加载策略。
export USE_MODEL_PARALLEL="${USE_MODEL_PARALLEL:-1}"
export MP_REQUIRE_4GPU="${MP_REQUIRE_4GPU:-1}"
export MP_DEVICE_MAP="${MP_DEVICE_MAP:-balanced}"
export MP_DTYPE="${MP_DTYPE:-float16}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH:-.}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export DISABLE_INTERNAL_EVAL
export FORCE_SAVE_FINAL_CHECKPOINT
export TRAIN_OPTIM
export TRAIN_LOGGING_STEPS
export TRAIN_DATALOADER_NUM_WORKERS
export OFFICIAL_TRAINER_BEHAVIOR
export OFFICIAL_ULD_MODEL_UTILS

case "${WHITEBOX_PROTOCOL}" in
  official_uld)
    [[ -z "${MODEL_CFG_RAW}" ]] && MODEL_CFG="tofu-llama-2"
    [[ "${TRAIN_SPLIT}" == "${SPLIT}" ]] && TRAIN_SPLIT="${SPLIT}_perturbed"
    [[ -z "${TRAIN_LR_RAW}" ]] && TRAIN_LR="1e-5"
    [[ -z "${TRAIN_BATCH_SIZE_RAW}" ]] && TRAIN_BATCH_SIZE="4"
    [[ -z "${TRAIN_GRAD_ACC_RAW}" ]] && TRAIN_GRAD_ACC="4"
    [[ -z "${TRAIN_WEIGHT_DECAY_RAW}" ]] && TRAIN_WEIGHT_DECAY="0.01"
    [[ -z "${LORA_R_RAW}" ]] && LORA_R="0"
    [[ -z "${LORA_ALPHA_RAW}" ]] && LORA_ALPHA="32"
    [[ -z "${LORA_DROPOUT_RAW}" ]] && LORA_DROPOUT="0.05"
    [[ -z "${CONV_TEMPLATE_STYLE_RAW}" ]] && CONV_TEMPLATE_STYLE="llama_inst"
    [[ -z "${MODEL_PATH_OVERRIDE_RAW}" ]] && MODEL_PATH_OVERRIDE="locuslab/tofu_ft_llama2-7b"
    [[ -z "${TOKENIZER_PATH_OVERRIDE_RAW}" ]] && TOKENIZER_PATH_OVERRIDE="locuslab/tofu_ft_llama2-7b"
    [[ -z "${EVAL_BATCH_SIZE_RAW}" ]] && EVAL_BATCH_SIZE="4"
    [[ -z "${TRAIN_OPTIM_RAW}" ]] && TRAIN_OPTIM="paged_adamw_32bit"
    [[ -z "${TRAIN_LOGGING_STEPS_RAW}" ]] && TRAIN_LOGGING_STEPS="1"
    [[ -z "${TRAIN_DATALOADER_NUM_WORKERS_RAW}" ]] && TRAIN_DATALOADER_NUM_WORKERS="0"
    OFFICIAL_TRAINER_BEHAVIOR=1
    OFFICIAL_ULD_MODEL_UTILS=1
    ;;
  fair_tinyllama)
    ;;
  *)
    echo "ERROR: unsupported WHITEBOX_PROTOCOL=${WHITEBOX_PROTOCOL}" >&2
    exit 1
    ;;
esac

RUN_TAG="whitebox_mp4_${METHOD//+/p}_tofu_${SPLIT}_s${SEED}"
if [[ -n "${RUN_TAG_SUFFIX}" ]]; then
  RUN_TAG="${RUN_TAG}_${RUN_TAG_SUFFIX}"
fi
LOGROOT="artifacts/seed_runs/${RUN_TAG}"
OUTMODELDIR="artifacts/outputs_trained_models/whitebox_mp4/${RUN_TAG}"
BASELOGDIR="artifacts/outputs/whitebox_mp4/${RUN_TAG}"
mkdir -p "${LOGROOT}" "${OUTMODELDIR}" "${BASELOGDIR}"

CFG_NAME="cbd_dfb_tinyllama_tofu"
MODEL_MODE=""
UNLEARN_LOSS=""
DATA_MODE=""
RETAIN_NUM=""
TRAIN_ARGS_EXTRA=()
EVAL_ARGS_EXTRA=()
TRAIN_SCRIPT="${TRAIN_SCRIPT:-scripts/hf_forget_train.py}"

case "${METHOD}" in
  ga)
    MODEL_MODE="base"
    UNLEARN_LOSS="ga"
    DATA_MODE="forget"
    ;;
  ga+gd)
    MODEL_MODE="base"
    UNLEARN_LOSS="ga+gd"
    DATA_MODE="forget_retain"
    RETAIN_NUM="2400"
    ;;
  ga+kl)
    MODEL_MODE="base"
    UNLEARN_LOSS="ga+kl"
    DATA_MODE="forget_retain"
    RETAIN_NUM="2400"
    ;;
  dpo)
    MODEL_MODE="base"
    UNLEARN_LOSS="dpo"
    DATA_MODE="dpo"
    ;;
  dpo+gd)
    MODEL_MODE="base"
    UNLEARN_LOSS="dpo+gd"
    DATA_MODE="forget_retain"
    RETAIN_NUM="2400"
    TRAIN_ARGS_EXTRA+=("data_mode.with_dpo=true")
    ;;
  dpo+kl)
    MODEL_MODE="base"
    UNLEARN_LOSS="dpo+kl"
    DATA_MODE="forget_retain"
    RETAIN_NUM="2400"
    TRAIN_ARGS_EXTRA+=("data_mode.with_dpo=true")
    ;;
  npo)
    MODEL_MODE="base"
    UNLEARN_LOSS="npo"
    DATA_MODE="forget"
    ;;
  npo+gd)
    MODEL_MODE="base"
    UNLEARN_LOSS="npo+gd"
    DATA_MODE="forget_retain"
    RETAIN_NUM="2400"
    ;;
  npo+kl)
    MODEL_MODE="base"
    UNLEARN_LOSS="npo+kl"
    DATA_MODE="forget_retain"
    RETAIN_NUM="2400"
    ;;
  *)
    echo "ERROR: unsupported method ${METHOD}" >&2
    exit 1
    ;;
esac

if [[ "${WHITEBOX_PROTOCOL}" == "official_uld" ]]; then
  case "${METHOD}" in
    ga+gd|ga+kl|npo+gd|npo+kl)
      DATA_MODE="forget_retain"
      RETAIN_NUM="400"
      if [[ "${WHITEBOX_RETAIN_MATCH_FORGET}" == "1" || "${WHITEBOX_RETAIN_MATCH_FORGET}" == "true" || "${WHITEBOX_RETAIN_MATCH_FORGET}" == "TRUE" ]]; then
        TRAIN_ARGS_EXTRA+=("data_mode.retain_match_forget=true")
      elif [[ -n "${WHITEBOX_RETAIN_MATCH_FORGET_RAW}" ]]; then
        TRAIN_ARGS_EXTRA+=("data_mode.retain_match_forget=false")
      fi
      ;;
    dpo)
      DATA_MODE="dpo"
      RETAIN_NUM=""
      ;;
    dpo+gd|dpo+kl)
      DATA_MODE="dpo_retain"
      RETAIN_NUM="400"
      TRAIN_ARGS_EXTRA=()
      ;;
    ga|npo)
      RETAIN_NUM=""
      ;;
  esac
fi

TRAIN_LOG="${LOGROOT}/train.log"
TRAIN_CMD=(
  "${TRAIN_SCRIPT}"
  "--config-name" "${CFG_NAME}"
  "enable_cbd_dfb=false"
  "model=${MODEL_CFG}"
  "model_mode=${MODEL_MODE}"
  "unlearn_loss=${UNLEARN_LOSS}"
  "data.dataset.split=${TRAIN_SPLIT}"
  "data.dataset.name=${TOFU_DATA_NAME}"
  "data_mode=${DATA_MODE}"
  "trainer.max_epochs=${TRAIN_EPOCHS}"
  "trainer.learning_rate=${TRAIN_LR}"
  "trainer.batch_size=${TRAIN_BATCH_SIZE}"
  "trainer.gradient_accumulation_steps=${TRAIN_GRAD_ACC}"
  "trainer.warmup_ratio=${TRAIN_WARMUP_RATIO}"
  "trainer.weight_decay=${TRAIN_WEIGHT_DECAY}"
  "++gradient_checkpointing=false"
  "++oracle_on_cpu=false"
  "model_mode.Lora.r=${LORA_R}"
  "model_mode.Lora.alpha=${LORA_ALPHA}"
  "model_mode.Lora.dropout=${LORA_DROPOUT}"
  "hydra.run.dir=${BASELOGDIR}/hydra_run"
  "seed=${SEED}"
  "lora_seed=${SEED}"
  "project=whitebox_mp4_${METHOD//+/p}_tofu"
  "BASELOGDIR=${BASELOGDIR}"
  "OUTPUTMODELDIR=${OUTMODELDIR}"
  "trainer.strategy=none"
)

if [[ -n "${RETAIN_NUM}" ]]; then
  TRAIN_CMD+=("data_mode.retain_num=${RETAIN_NUM}")
fi
if [[ -n "${MODEL_PATH_OVERRIDE}" ]]; then
  TRAIN_CMD+=("model.model_path=${MODEL_PATH_OVERRIDE}")
fi
if [[ -n "${TOKENIZER_PATH_OVERRIDE}" ]]; then
  TRAIN_CMD+=("model.tokenizer_path=${TOKENIZER_PATH_OVERRIDE}")
fi
if [[ "${TRAIN_SPLIT}" == *_perturbed ]]; then
  TRAIN_CMD+=(
    "data.dataset.perturb_path=${TOFU_AUG_ROOT}/${TRAIN_SPLIT}/perturb_res.csv"
    "data.dataset.paraphrase_path=${TOFU_AUG_ROOT}/${TRAIN_SPLIT}/paraphrase_res.csv"
  )
fi
case "${CONV_TEMPLATE_STYLE}" in
  llama_inst)
    TRAIN_CMD+=(
      "data.conv_template.question_start_token='[INST] '"
      "data.conv_template.question_end_token=' [/INST]'"
      "data.conv_template.answer_token="
    )
    EVAL_ARGS_EXTRA+=(
      "data.conv_template.question_start_token='[INST] '"
      "data.conv_template.question_end_token=' [/INST]'"
      "data.conv_template.answer_token="
    )
    ;;
  default)
    ;;
  *)
    echo "ERROR: unsupported CONV_TEMPLATE_STYLE=${CONV_TEMPLATE_STYLE}" >&2
    exit 1
    ;;
esac
TRAIN_CMD+=("${TRAIN_ARGS_EXTRA[@]}")
if [[ -n "${TRAIN_MAX_STEPS}" && "${TRAIN_MAX_STEPS}" != "none" && "${TRAIN_MAX_STEPS}" != "None" && "${TRAIN_MAX_STEPS}" != "NONE" && "${TRAIN_MAX_STEPS}" != "0" ]]; then
  TRAIN_CMD+=("+trainer.max_steps=${TRAIN_MAX_STEPS}")
fi

if [[ "${SKIP_TRAIN}" != "1" ]]; then
  echo "=========================================="
  echo "Training: ${METHOD} on ${SPLIT} (seed=${SEED})"
  echo "4-GPU Model Parallel Mode"
  echo "Protocol: ${WHITEBOX_PROTOCOL}"
  echo "Model: ${MODEL_CFG}"
  echo "Batch size: ${TRAIN_BATCH_SIZE}, Grad acc: ${TRAIN_GRAD_ACC}"
  echo "Epochs: ${TRAIN_EPOCHS}, Max steps: ${TRAIN_MAX_STEPS}, LR: ${TRAIN_LR}"
  echo "Log: ${TRAIN_LOG}"
  echo "=========================================="

  "${PY}" "${TRAIN_CMD[@]}" >"${TRAIN_LOG}" 2>&1
else
  echo "=========================================="
  echo "Skip training: ${METHOD} on ${SPLIT} (seed=${SEED})"
  echo "Reusing checkpoint under ${OUTMODELDIR}"
  echo "=========================================="
fi

CKPT="$(find "${OUTMODELDIR}" -type d -name 'checkpoint-*' -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-)"
if [[ -z "${CKPT}" || ! -d "${CKPT}" ]]; then
  echo "ERROR: checkpoint not found under ${OUTMODELDIR}" >&2
  exit 1
fi
printf '%s\n' "${CKPT}" >"${LOGROOT}/ckpt_path.txt"

if [[ "${SKIP_TRAIN}" != "1" ]]; then
  echo "✓ Training completed!"
fi
echo "Checkpoint: ${CKPT}"

if [[ "${RUN_EVAL}" != "1" ]]; then
  exit 0
fi

EVAL_OUTDIR="artifacts/eval_outputs/tofu/${RUN_TAG}"
EVAL_LOG="${LOGROOT}/eval_tofu.log"
mkdir -p "${EVAL_OUTDIR}"
if [[ -n "${EVAL_SPLIT_OVERRIDE}" ]]; then
  EVAL_SPLIT="${EVAL_SPLIT_OVERRIDE}"
elif [[ "${SPLIT}" == *_perturbed ]]; then
  EVAL_SPLIT="${SPLIT}"
else
  EVAL_SPLIT="${SPLIT}_perturbed"
fi
case "${SPLIT}" in
  forget01)
    DEFAULT_RETAIN_RESULT="${CBD_DATA_ROOT:-data}/data/retain99_llama_wd0.01/eval_results/ds_size300/eval_log_aggregated.json"
    ;;
  forget05)
    DEFAULT_RETAIN_RESULT="${CBD_DATA_ROOT:-data}/data/retain95_llama_wd0.01/eval_results/ds_size300/eval_log_aggregated.json"
    ;;
  forget10)
    DEFAULT_RETAIN_RESULT="${CBD_DATA_ROOT:-data}/data/retain90_llama_wd0.01/eval_results/ds_size300/eval_log_aggregated.json"
    ;;
  *)
    DEFAULT_RETAIN_RESULT="${CBD_DATA_ROOT:-data}/data/retain90_llama_wd0.01/eval_results/ds_size300/eval_log_aggregated.json"
    ;;
esac
EVAL_RETAIN_RESULT="${EVAL_RETAIN_RESULT:-${DEFAULT_RETAIN_RESULT}}"

echo "=========================================="
echo "Eval: ${METHOD} on ${SPLIT} (seed=${SEED})"
echo "Eval split: ${EVAL_SPLIT}"
echo "Eval batch size: ${EVAL_BATCH_SIZE}"
echo "Eval max num: ${EVAL_MAX_NUM}"
echo "Retain result: ${EVAL_RETAIN_RESULT}"
echo "Eval log: ${EVAL_LOG}"
echo "=========================================="

	"${PY}" scripts/eval_tofu.py \
	  OUTDIRNAME="${EVAL_OUTDIR}" \
	  ckpt_path="${CKPT}" \
  model="${MODEL_CFG}" \
  model_mode="${MODEL_MODE}" \
  model_mode.Lora.r="${LORA_R}" \
  model_mode.Lora.alpha="${LORA_ALPHA}" \
  model_mode.Lora.dropout="${LORA_DROPOUT}" \
	  data.dataset.split="${EVAL_SPLIT}" \
	  data.dataset.name="${TOFU_DATA_NAME}" \
	  data.dataset.perturb_path="${TOFU_AUG_ROOT}/${EVAL_SPLIT}/perturb_res.csv" \
	  data.dataset.paraphrase_path="${TOFU_AUG_ROOT}/${EVAL_SPLIT}/paraphrase_res.csv" \
	  data.dataset.eval.batch_size="${EVAL_BATCH_SIZE}" \
  data.dataset.eval.retain_result="${EVAL_RETAIN_RESULT}" \
  "+data.dataset.eval.max_num=${EVAL_MAX_NUM}" \
  ${MODEL_PATH_OVERRIDE:+model.model_path="${MODEL_PATH_OVERRIDE}"} \
  ${TOKENIZER_PATH_OVERRIDE:+model.tokenizer_path="${TOKENIZER_PATH_OVERRIDE}"} \
  "${EVAL_ARGS_EXTRA[@]}" \
  >"${EVAL_LOG}" 2>&1

echo "✓ Eval completed!"
echo "Eval target: ${EVAL_OUTDIR}"
