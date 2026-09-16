#!/usr/bin/env bash
# 内部实现脚本。正式复现请统一从 scripts/hf_forget_train.py repro 进入。
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

METHOD_GROUP="${1:?method_group required: whitebox or graybox}"
METHOD="${2:?method required}"
DATASET="${3:?dataset required: tofu or wmdp}"
SPLIT="${4:?split required}"
SEED="${5:?seed required}"
RUN_SUFFIX="${6:-commonproto}"

if [[ "${METHOD_GROUP}" != "whitebox" && "${METHOD_GROUP}" != "graybox" ]]; then
  echo "ERROR: METHOD_GROUP must be whitebox or graybox" >&2
  exit 1
fi
if [[ "${DATASET}" != "tofu" && "${DATASET}" != "wmdp" ]]; then
  echo "ERROR: DATASET must be tofu or wmdp" >&2
  exit 1
fi

GPU_SET="${GPU_SET:-0}"
IFS=',' read -r -a GPU_ARR <<<"${GPU_SET}"
WORLD_SIZE="${WORLD_SIZE:-${#GPU_ARR[@]}}"
if [[ "${WORLD_SIZE}" -lt 1 ]]; then
  echo "ERROR: WORLD_SIZE must be >= 1" >&2
  exit 1
fi

TRAIN_LR="${TRAIN_LR:-1.5e-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
TRAIN_GRAD_ACC="${TRAIN_GRAD_ACC:-2}"
TRAIN_WEIGHT_DECAY="${TRAIN_WEIGHT_DECAY:-0}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-10}"
TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-180}"
TRAIN_WARMUP_RATIO="${TRAIN_WARMUP_RATIO:-0.1}"
TRAIN_STRATEGY="${TRAIN_STRATEGY:-none}"
TRAIN_RETAIN_NUM="${TRAIN_RETAIN_NUM:-}"
FORGET_WEIGHT="${FORGET_WEIGHT:-}"
RETAIN_WEIGHT="${RETAIN_WEIGHT:-}"
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-false}"
ORACLE_ON_CPU="${ORACLE_ON_CPU:-false}"
ORACLE_DEVICE="${ORACLE_DEVICE:-}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
CKPT_PATH_OVERRIDE="${CKPT_PATH_OVERRIDE:-}"
GRAYBOX_OFFSET_LOSS="${GRAYBOX_OFFSET_LOSS:-gd+kl}"
GRAYBOX_OFFSET_DATA_MODE="${GRAYBOX_OFFSET_DATA_MODE:-forget_retain}"
GRAYBOX_OFFSET_WEIGHT="${GRAYBOX_OFFSET_WEIGHT:-1.0}"
GRAYBOX_OFFSET_ASSIST_ONLY="${GRAYBOX_OFFSET_ASSIST_ONLY:-0}"
OFFSET_BASE_DEVICE="${OFFSET_BASE_DEVICE:-}"
OFFSET_BASE_ASSIST_DEVICE="${OFFSET_BASE_ASSIST_DEVICE:-}"
OFFSET_ASSIST_DEVICE="${OFFSET_ASSIST_DEVICE:-}"
TOFU_ULD_EVAL_DEVICES="${TOFU_ULD_EVAL_DEVICES:-}"
TOFU_OFFSET_EVAL_DEVICES="${TOFU_OFFSET_EVAL_DEVICES:-}"
TRAIN_SPLIT_OVERRIDE="${TRAIN_SPLIT_OVERRIDE:-}"
TOFU_CONV_TEMPLATE_STYLE="${TOFU_CONV_TEMPLATE_STYLE:-default}"
GRAYBOX_ULD_LOSS="${GRAYBOX_ULD_LOSS:-gd+kl}"
GRAYBOX_ULD_DATA_MODE="${GRAYBOX_ULD_DATA_MODE:-forget_retain}"
GRAYBOX_ULD_RETAIN_NUM="${GRAYBOX_ULD_RETAIN_NUM:-}"
GRAYBOX_ULD_RETAIN_WEIGHT="${GRAYBOX_ULD_RETAIN_WEIGHT:-}"
GRAYBOX_ULD_TRAIN_WEIGHT="${GRAYBOX_ULD_TRAIN_WEIGHT:-}"
GRAYBOX_ULD_TRAIN_TOP_LOGIT_FILTER="${GRAYBOX_ULD_TRAIN_TOP_LOGIT_FILTER:-}"
GRAYBOX_ULD_EVAL_WEIGHT="${GRAYBOX_ULD_EVAL_WEIGHT:-}"
GRAYBOX_ULD_EVAL_TOP_LOGIT_FILTER="${GRAYBOX_ULD_EVAL_TOP_LOGIT_FILTER:-}"
DATA_ROOT="${CBD_DATA_ROOT:-data}"

case "${SPLIT}" in
  forget01)
    DEFAULT_TOFU_RETAIN_RESULT="${DATA_ROOT}/data/retain99_llama_wd0.01/eval_results/ds_size300/eval_log_aggregated.json"
    ;;
  forget05)
    DEFAULT_TOFU_RETAIN_RESULT="${DATA_ROOT}/data/retain95_llama_wd0.01/eval_results/ds_size300/eval_log_aggregated.json"
    ;;
  *)
    DEFAULT_TOFU_RETAIN_RESULT="${DATA_ROOT}/data/retain90_llama_wd0.01/eval_results/ds_size300/eval_log_aggregated.json"
    ;;
esac
TOFU_RETAIN_RESULT="${TOFU_RETAIN_RESULT:-${DEFAULT_TOFU_RETAIN_RESULT}}"
TOFU_EVAL_MAX_NUM="${TOFU_EVAL_MAX_NUM:-300}"
TOFU_EVAL_BATCH_SIZE="${TOFU_EVAL_BATCH_SIZE:-4}"
TOFU_DATA_NAME="${TOFU_DATA_NAME:-${DATA_ROOT}/TOFU}"
TOFU_AUG_ROOT="${TOFU_AUG_ROOT:-${DATA_ROOT}/data/aug_data/tofu}"
TOFU_MODEL_PATH="${TOFU_MODEL_PATH:-locuslab/tofu_ft_llama2-7b}"
TOFU_TOKENIZER_PATH="${TOFU_TOKENIZER_PATH:-locuslab/tofu_ft_llama2-7b}"

WMDP_MAX_FORGET="${WMDP_MAX_FORGET:-400}"
WMDP_RETAIN_NUM="${WMDP_RETAIN_NUM:-2400}"
WMDP_MAX_LEN="${WMDP_MAX_LEN:-512}"
WMDP_MMLU_RETAIN_FILE="${WMDP_MMLU_RETAIN_FILE:-${DATA_ROOT}/eval-method/wmdp/data/mmlu/all_auxiliary_train.jsonl}"
WMDP_MMLU_TEST_FILE="${WMDP_MMLU_TEST_FILE:-${DATA_ROOT}/eval-method/wmdp/data/mmlu/all_test.jsonl}"
WMDP_EVAL_MAX_WMDP="${WMDP_EVAL_MAX_WMDP:-0}"
WMDP_EVAL_MAX_MMLU="${WMDP_EVAL_MAX_MMLU:-0}"
WMDP_EVAL_BATCH_SIZE="${WMDP_EVAL_BATCH_SIZE:-8}"
WMDP_EVAL_TRUNCATE_MODE="${WMDP_EVAL_TRUNCATE_MODE:-left}"
WMDP_EVAL_PROGRESS_EVERY="${WMDP_EVAL_PROGRESS_EVERY:-200}"
WMDP_MODEL_PATH="${WMDP_MODEL_PATH:-HuggingFaceH4/zephyr-7b-beta}"
WMDP_TOKENIZER_PATH="${WMDP_TOKENIZER_PATH:-HuggingFaceH4/zephyr-7b-beta}"
ASSIST_MODEL_PATH="${ASSIST_MODEL_PATH:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH:-.}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export CBD_FORCE_LOCAL_DATASETS_SHIM="${CBD_FORCE_LOCAL_DATASETS_SHIM:-1}"
export FORCE_SAVE_FINAL_CHECKPOINT="${FORCE_SAVE_FINAL_CHECKPOINT:-1}"
export EVAL_ATTN_IMPL="${EVAL_ATTN_IMPL:-eager}"
# Paper reproduction uses the post-training evaluators below. Keeping
# trainer-side validation disabled avoids long/stale validation stalls.
export DISABLE_INTERNAL_EVAL="${DISABLE_INTERNAL_EVAL:-1}"

RUN_TAG="${METHOD_GROUP}_${METHOD//+/p}_${DATASET}_${SPLIT}_s${SEED}_${RUN_SUFFIX}"
LOGROOT="artifacts/seed_runs/${RUN_TAG}"
OUTMODELDIR="artifacts/outputs_trained_models/baselines/${RUN_TAG}"
BASELOGDIR="artifacts/outputs/baselines/${RUN_TAG}"
mkdir -p "${LOGROOT}" "${OUTMODELDIR}" "${BASELOGDIR}"

CFG_NAME=""
MODEL_CFG=""
MODEL_MODE=""
EVAL_MODEL_MODE=""
UNLEARN_LOSS=""
DATA_MODE=""
RETAIN_NUM=""
TRAIN_ARGS_EXTRA=()
EVAL_ARGS_EXTRA=()

case "${DATASET}" in
  tofu)
    CFG_NAME="cbd_dfb_tinyllama_tofu"
    MODEL_CFG="tofu-llama-2"
    TRAIN_ARGS_EXTRA+=(
      "data.dataset.name=${TOFU_DATA_NAME}"
      "model.model_path=${TOFU_MODEL_PATH}"
      "model.tokenizer_path=${TOFU_TOKENIZER_PATH}"
    )
    ;;
	  wmdp)
	    CFG_NAME="cbd_dfb_tinyllama_wmdp"
	    MODEL_CFG="zephyr7b"
	    TRAIN_ARGS_EXTRA+=(
	      "data.dataset.mmlu_retain_file=${WMDP_MMLU_RETAIN_FILE}"
	      "data.dataset.max_len=${WMDP_MAX_LEN}"
	      "data.conv_template.max_len=${WMDP_MAX_LEN}"
	      "model.model_path=${WMDP_MODEL_PATH}"
	      "model.tokenizer_path=${WMDP_TOKENIZER_PATH}"
	    )
	    if [[ -n "${WMDP_MAX_FORGET}" && "${WMDP_MAX_FORGET}" != "none" ]]; then
	      TRAIN_ARGS_EXTRA+=("data.dataset.max_forget=${WMDP_MAX_FORGET}")
	    fi
	    ;;
esac

case "${METHOD_GROUP}:${METHOD}" in
  whitebox:ga)
    MODEL_MODE="base"
    UNLEARN_LOSS="ga"
    DATA_MODE="forget"
    ;;
  whitebox:ga+gd)
    MODEL_MODE="base"
    UNLEARN_LOSS="ga+gd"
    DATA_MODE="forget_retain"
    ;;
  whitebox:ga+kl)
    MODEL_MODE="base"
    UNLEARN_LOSS="ga+kl"
    DATA_MODE="forget_retain"
    ;;
  whitebox:dpo)
    MODEL_MODE="base"
    UNLEARN_LOSS="dpo"
    DATA_MODE="dpo"
    ;;
  whitebox:dpo+gd)
    MODEL_MODE="base"
    UNLEARN_LOSS="dpo+gd"
    DATA_MODE="forget_retain"
    TRAIN_ARGS_EXTRA+=("data_mode.with_dpo=true")
    ;;
  whitebox:dpo+kl)
    MODEL_MODE="base"
    UNLEARN_LOSS="dpo+kl"
    DATA_MODE="forget_retain"
    TRAIN_ARGS_EXTRA+=("data_mode.with_dpo=true")
    ;;
  whitebox:npo)
    MODEL_MODE="base"
    UNLEARN_LOSS="npo"
    DATA_MODE="forget"
    ;;
  whitebox:npo+gd)
    MODEL_MODE="base"
    UNLEARN_LOSS="npo+gd"
    DATA_MODE="forget_retain"
    ;;
  whitebox:npo+kl)
    MODEL_MODE="base"
    UNLEARN_LOSS="npo+kl"
    DATA_MODE="forget_retain"
    ;;
	  graybox:uld)
	    MODEL_MODE="uld"
	    UNLEARN_LOSS="${GRAYBOX_ULD_LOSS}"
	    DATA_MODE="${GRAYBOX_ULD_DATA_MODE}"
	    if [[ -n "${GRAYBOX_ULD_RETAIN_WEIGHT}" ]]; then
	      TRAIN_ARGS_EXTRA+=("unlearn_loss.retain_weight=${GRAYBOX_ULD_RETAIN_WEIGHT}")
	    fi
	    if [[ -n "${GRAYBOX_ULD_TRAIN_WEIGHT}" ]]; then
	      TRAIN_ARGS_EXTRA+=("model_mode.weight=${GRAYBOX_ULD_TRAIN_WEIGHT}")
	    fi
	    if [[ -n "${GRAYBOX_ULD_TRAIN_TOP_LOGIT_FILTER}" ]]; then
	      TRAIN_ARGS_EXTRA+=("model_mode.top_logit_filter=${GRAYBOX_ULD_TRAIN_TOP_LOGIT_FILTER}")
	    fi
	    if [[ -n "${GRAYBOX_ULD_EVAL_WEIGHT}" ]]; then
	      EVAL_ARGS_EXTRA+=("model_mode.weight=${GRAYBOX_ULD_EVAL_WEIGHT}")
	    fi
	    if [[ -n "${GRAYBOX_ULD_EVAL_TOP_LOGIT_FILTER}" ]]; then
	      EVAL_ARGS_EXTRA+=("model_mode.top_logit_filter=${GRAYBOX_ULD_EVAL_TOP_LOGIT_FILTER}")
	    fi
	    if [[ "${DATASET}" == "tofu" && -n "${TOFU_ULD_EVAL_DEVICES}" ]]; then
	      EVAL_ARGS_EXTRA+=("+model_mode.eval_devices=${TOFU_ULD_EVAL_DEVICES}")
	    fi
	    ;;
	  graybox:offset)
	    UNLEARN_LOSS="${GRAYBOX_OFFSET_LOSS}"
	    DATA_MODE="${GRAYBOX_OFFSET_DATA_MODE}"
	    if [[ "${DATASET}" == "wmdp" && "${GRAYBOX_OFFSET_ASSIST_ONLY}" == "1" ]]; then
	      MODEL_MODE="base"
	      EVAL_MODEL_MODE="offset"
	      MODEL_CFG="tinyllama"
	      TRAIN_ARGS_EXTRA+=(
	        "model.model_path=${ASSIST_MODEL_PATH}"
	        "model.tokenizer_path=${ASSIST_MODEL_PATH}"
	      )
	    else
	      MODEL_MODE="offset"
	      TRAIN_ARGS_EXTRA+=(
	        "model_mode.base_assist_path=${ASSIST_MODEL_PATH}"
	        "model_mode.weight=${GRAYBOX_OFFSET_WEIGHT}"
	      )
	      if [[ -n "${OFFSET_BASE_DEVICE}" ]]; then
	        TRAIN_ARGS_EXTRA+=("model_mode.base_device=${OFFSET_BASE_DEVICE}")
	      fi
	      if [[ -n "${OFFSET_BASE_ASSIST_DEVICE}" ]]; then
	        TRAIN_ARGS_EXTRA+=("model_mode.base_assist_device=${OFFSET_BASE_ASSIST_DEVICE}")
	      fi
	      if [[ -n "${OFFSET_ASSIST_DEVICE}" ]]; then
	        TRAIN_ARGS_EXTRA+=("model_mode.assist_device=${OFFSET_ASSIST_DEVICE}")
	      fi
	    fi
	    EVAL_ARGS_EXTRA+=(
	      "model_mode.base_assist_path=${ASSIST_MODEL_PATH}"
	      "model_mode.weight=${GRAYBOX_OFFSET_WEIGHT}"
	    )
	    if [[ "${DATASET}" == "tofu" && -n "${TOFU_OFFSET_EVAL_DEVICES}" ]]; then
	      EVAL_ARGS_EXTRA+=("+model_mode.eval_devices=${TOFU_OFFSET_EVAL_DEVICES}")
	    fi
	    ;;
  *)
    echo "ERROR: unsupported combination ${METHOD_GROUP}:${METHOD}" >&2
    exit 1
    ;;
esac

if [[ "${DATA_MODE}" == "forget_retain" ]]; then
  if [[ "${DATASET}" == "tofu" ]]; then
    RETAIN_NUM="2400"
  else
    RETAIN_NUM="${WMDP_RETAIN_NUM}"
  fi
fi
if [[ -n "${GRAYBOX_ULD_RETAIN_NUM}" ]]; then
  RETAIN_NUM="${GRAYBOX_ULD_RETAIN_NUM}"
fi
if [[ -n "${TRAIN_RETAIN_NUM}" ]]; then
  RETAIN_NUM="${TRAIN_RETAIN_NUM}"
fi
if [[ -n "${FORGET_WEIGHT}" ]]; then
  TRAIN_ARGS_EXTRA+=("++unlearn_loss.forget_weight=${FORGET_WEIGHT}")
fi
if [[ -n "${RETAIN_WEIGHT}" ]]; then
  TRAIN_ARGS_EXTRA+=("++unlearn_loss.retain_weight=${RETAIN_WEIGHT}")
fi
if [[ -n "${MODEL_ATTN_IMPL:-}" ]]; then
  TRAIN_ARGS_EXTRA+=("+model.attn_implementation=${MODEL_ATTN_IMPL}")
fi

case "${TOFU_CONV_TEMPLATE_STYLE}" in
  llama_inst)
    TRAIN_ARGS_EXTRA+=(
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
    echo "ERROR: unsupported TOFU_CONV_TEMPLATE_STYLE=${TOFU_CONV_TEMPLATE_STYLE}" >&2
    exit 1
    ;;
esac

TRAIN_SPLIT="${TRAIN_SPLIT_OVERRIDE:-${SPLIT}}"
if [[ "${DATASET}" == "tofu" ]]; then
  EVAL_SPLIT="${SPLIT}_perturbed"
  if [[ "${TRAIN_SPLIT}" == *_perturbed ]]; then
    TRAIN_ARGS_EXTRA+=(
      "data.dataset.perturb_path=${TOFU_AUG_ROOT}/${TRAIN_SPLIT}/perturb_res.csv"
      "data.dataset.paraphrase_path=${TOFU_AUG_ROOT}/${TRAIN_SPLIT}/paraphrase_res.csv"
    )
  fi
  EVAL_ARGS_EXTRA+=(
    "data.dataset.perturb_path=${TOFU_AUG_ROOT}/${EVAL_SPLIT}/perturb_res.csv"
    "data.dataset.paraphrase_path=${TOFU_AUG_ROOT}/${EVAL_SPLIT}/paraphrase_res.csv"
  )
fi

TRAIN_LOG="${LOGROOT}/train.log"
cat >"${LOGROOT}/run_config.env" <<EOF
METHOD_GROUP=${METHOD_GROUP}
METHOD=${METHOD}
DATASET=${DATASET}
SPLIT=${SPLIT}
SEED=${SEED}
REPRO_PROFILE=${REPRO_PROFILE:-default}
RUN_TAG=${RUN_TAG}
MODEL_MODE=${MODEL_MODE}
EVAL_MODEL_MODE=${EVAL_MODEL_MODE:-${MODEL_MODE}}
UNLEARN_LOSS=${UNLEARN_LOSS}
DATA_MODE=${DATA_MODE}
TRAIN_SPLIT=${TRAIN_SPLIT}
RETAIN_NUM=${RETAIN_NUM}
TRAIN_LR=${TRAIN_LR}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}
TRAIN_GRAD_ACC=${TRAIN_GRAD_ACC}
EFFECTIVE_BATCH=$(( TRAIN_BATCH_SIZE * TRAIN_GRAD_ACC ))
TRAIN_WEIGHT_DECAY=${TRAIN_WEIGHT_DECAY}
TRAIN_MAX_STEPS=${TRAIN_MAX_STEPS}
TRAIN_WARMUP_RATIO=${TRAIN_WARMUP_RATIO}
CBD_FORCE_LOCAL_DATASETS_SHIM=${CBD_FORCE_LOCAL_DATASETS_SHIM}
MODEL_ATTN_IMPL=${MODEL_ATTN_IMPL}
EVAL_ATTN_IMPL=${EVAL_ATTN_IMPL}
LORA_R=${LORA_R}
LORA_ALPHA=${LORA_ALPHA}
LORA_DROPOUT=${LORA_DROPOUT}
FORGET_WEIGHT=${FORGET_WEIGHT}
RETAIN_WEIGHT=${RETAIN_WEIGHT}
USE_MODEL_PARALLEL=${USE_MODEL_PARALLEL:-0}
GPU_SET=${GPU_SET}
SKIP_TRAIN=${SKIP_TRAIN}
SKIP_EVAL=${SKIP_EVAL}
WMDP_MMLU_TEST_FILE=${WMDP_MMLU_TEST_FILE}
WMDP_EVAL_MAX_WMDP=${WMDP_EVAL_MAX_WMDP}
WMDP_EVAL_MAX_MMLU=${WMDP_EVAL_MAX_MMLU}
WMDP_EVAL_BATCH_SIZE=${WMDP_EVAL_BATCH_SIZE}
WMDP_EVAL_TRUNCATE_MODE=${WMDP_EVAL_TRUNCATE_MODE}
WMDP_EVAL_PROGRESS_EVERY=${WMDP_EVAL_PROGRESS_EVERY}
ORACLE_ON_CPU=${ORACLE_ON_CPU}
ORACLE_DEVICE=${ORACLE_DEVICE}
OFFSET_BASE_DEVICE=${OFFSET_BASE_DEVICE}
OFFSET_BASE_ASSIST_DEVICE=${OFFSET_BASE_ASSIST_DEVICE}
OFFSET_ASSIST_DEVICE=${OFFSET_ASSIST_DEVICE}
TOFU_ULD_EVAL_DEVICES=${TOFU_ULD_EVAL_DEVICES}
TOFU_OFFSET_EVAL_DEVICES=${TOFU_OFFSET_EVAL_DEVICES}
OFFICIAL_ULD_MODEL_UTILS=${OFFICIAL_ULD_MODEL_UTILS:-0}
EOF
TRAIN_CMD=(
  "scripts/hf_forget_train.py"
  "--config-name" "${CFG_NAME}"
  "enable_cbd_dfb=false"
  "model=${MODEL_CFG}"
  "model_mode=${MODEL_MODE}"
  "unlearn_loss=${UNLEARN_LOSS}"
  "data.dataset.split=${TRAIN_SPLIT}"
  "data_mode=${DATA_MODE}"
  "trainer.max_epochs=${TRAIN_EPOCHS}"
  "trainer.learning_rate=${TRAIN_LR}"
  "trainer.batch_size=${TRAIN_BATCH_SIZE}"
  "trainer.gradient_accumulation_steps=${TRAIN_GRAD_ACC}"
  "trainer.warmup_ratio=${TRAIN_WARMUP_RATIO}"
  "trainer.weight_decay=${TRAIN_WEIGHT_DECAY}"
  "++gradient_checkpointing=${GRADIENT_CHECKPOINTING}"
  "++oracle_on_cpu=${ORACLE_ON_CPU}"
  "model_mode.Lora.r=${LORA_R}"
  "model_mode.Lora.alpha=${LORA_ALPHA}"
  "model_mode.Lora.dropout=${LORA_DROPOUT}"
  "hydra.run.dir=${BASELOGDIR}/hydra_run"
  "seed=${SEED}"
  "lora_seed=${SEED}"
  "project=baseline_${METHOD_GROUP}_${METHOD//+/p}_${DATASET}"
  "BASELOGDIR=${BASELOGDIR}"
  "OUTPUTMODELDIR=${OUTMODELDIR}"
)
if [[ -n "${TRAIN_MAX_STEPS}" && "${TRAIN_MAX_STEPS}" != "none" && "${TRAIN_MAX_STEPS}" != "None" && "${TRAIN_MAX_STEPS}" != "NONE" && "${TRAIN_MAX_STEPS}" != "0" ]]; then
  TRAIN_CMD+=("+trainer.max_steps=${TRAIN_MAX_STEPS}")
fi
if [[ -n "${RETAIN_NUM}" ]]; then
  TRAIN_CMD+=("data_mode.retain_num=${RETAIN_NUM}")
fi
TRAIN_CMD+=("${TRAIN_ARGS_EXTRA[@]}")

if [[ "${SKIP_TRAIN}" == "1" ]]; then
  echo "[stage=eval] skip training; locating checkpoint under ${OUTMODELDIR}" >"${LOGROOT}/stage_eval.log"
else
  if [[ "${USE_MODEL_PARALLEL:-0}" == "1" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU_SET}" "${PY}" "${TRAIN_CMD[@]}" "trainer.strategy=none" >"${TRAIN_LOG}" 2>&1
  elif [[ "${WORLD_SIZE}" -gt 1 ]]; then
    CUDA_VISIBLE_DEVICES="${GPU_SET}" torchrun --standalone --nproc_per_node="${WORLD_SIZE}" "${TRAIN_CMD[@]}" "trainer.strategy=ddp" >"${TRAIN_LOG}" 2>&1
  else
    CUDA_VISIBLE_DEVICES="${GPU_SET}" "${PY}" "${TRAIN_CMD[@]}" "trainer.strategy=${TRAIN_STRATEGY}" >"${TRAIN_LOG}" 2>&1
  fi
fi

if [[ -n "${CKPT_PATH_OVERRIDE}" ]]; then
  CKPT="${CKPT_PATH_OVERRIDE}"
else
  CKPT="$(find "${OUTMODELDIR}" -type d -name 'checkpoint-*' -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-)"
fi
if [[ -z "${CKPT}" || ! -d "${CKPT}" ]]; then
  echo "ERROR: checkpoint not found under ${OUTMODELDIR}" >&2
  exit 1
fi
printf '%s\n' "${CKPT}" >"${LOGROOT}/ckpt_path.txt"

if [[ "${SKIP_EVAL}" == "1" ]]; then
  echo "checkpoint=${CKPT}"
  exit 0
fi

case "${DATASET}" in
  tofu)
    EVAL_OUTDIR="artifacts/eval_outputs/tofu/baselines_commonproto/${RUN_TAG}"
    EVAL_LOG="${LOGROOT}/eval_tofu.log"
    "${PY}" scripts/eval_tofu.py \
      OUTDIRNAME="${EVAL_OUTDIR}" \
      ckpt_path="${CKPT}" \
      model=tofu-llama-2 \
      model.model_path="${TOFU_MODEL_PATH}" \
      model.tokenizer_path="${TOFU_TOKENIZER_PATH}" \
      model_mode="${MODEL_MODE}" \
      model_mode.Lora.r="${LORA_R}" \
      model_mode.Lora.alpha="${LORA_ALPHA}" \
      model_mode.Lora.dropout="${LORA_DROPOUT}" \
      data.dataset.name="${TOFU_DATA_NAME}" \
      data.dataset.split="${SPLIT}_perturbed" \
      data.dataset.eval.batch_size="${TOFU_EVAL_BATCH_SIZE}" \
      data.dataset.eval.retain_result="${TOFU_RETAIN_RESULT}" \
      "+data.dataset.eval.max_num=${TOFU_EVAL_MAX_NUM}" \
      "${EVAL_ARGS_EXTRA[@]}" \
      >"${EVAL_LOG}" 2>&1
    echo "eval_target=${EVAL_OUTDIR}"
    ;;
  wmdp)
    EVAL_JSON="artifacts/eval_outputs/wmdp_direct/baselines_commonproto/${RUN_TAG}.json"
    EVAL_LOG="${LOGROOT}/eval_wmdp.log"
    mkdir -p "$(dirname "${EVAL_JSON}")"
    WMDP_DIRECT_MODEL_MODE="direct"
    WMDP_DIRECT_EXTRA_ARGS=()
    WMDP_EVAL_GPU_SET_EFFECTIVE="${WMDP_EVAL_GPU_SET:-${GPU_SET}}"
    if [[ "${EVAL_MODEL_MODE:-${MODEL_MODE}}" == "uld" ]]; then
      WMDP_DIRECT_MODEL_MODE="uld"
      WMDP_EVAL_GPU_SET_EFFECTIVE="${WMDP_EVAL_GPU_SET:-${WMDP_ULD_EVAL_GPU_SET:-0,1}}"
      WMDP_DIRECT_EXTRA_ARGS+=(
        --uld_weight "${GRAYBOX_ULD_EVAL_WEIGHT:-${GRAYBOX_ULD_TRAIN_WEIGHT:--0.75}}"
        --uld_top_logit_filter "${GRAYBOX_ULD_EVAL_TOP_LOGIT_FILTER:-${GRAYBOX_ULD_TRAIN_TOP_LOGIT_FILTER:-0.01}}"
        --eval_devices "${WMDP_ULD_EVAL_DEVICES:-cuda:0,cuda:1}"
      )
    elif [[ "${EVAL_MODEL_MODE:-${MODEL_MODE}}" == "offset" ]]; then
      WMDP_DIRECT_MODEL_MODE="offset"
      WMDP_DIRECT_EXTRA_ARGS+=(
        --offset_base_assist_path "${ASSIST_MODEL_PATH}"
        --offset_weight "${GRAYBOX_OFFSET_WEIGHT}"
        --eval_devices "${WMDP_OFFSET_EVAL_DEVICES:-cuda:0}"
      )
    fi
    CUDA_VISIBLE_DEVICES="${WMDP_EVAL_GPU_SET_EFFECTIVE}" "${PY}" scripts/eval_wmdp_direct.py \
      --model_path "${CKPT}" \
      --model_base_if_lora "${WMDP_MODEL_PATH}" \
      --tokenizer_path "${WMDP_TOKENIZER_PATH}" \
      --model_mode "${WMDP_DIRECT_MODEL_MODE}" \
      --device cuda \
      --mmlu_test_file "${WMDP_MMLU_TEST_FILE}" \
      --seed "${SEED}" \
      --max_wmdp "${WMDP_EVAL_MAX_WMDP}" \
      --max_mmlu "${WMDP_EVAL_MAX_MMLU}" \
      --batch_size "${WMDP_EVAL_BATCH_SIZE}" \
      --max_len "${WMDP_MAX_LEN}" \
      --truncate_mode "${WMDP_EVAL_TRUNCATE_MODE}" \
      --progress_every "${WMDP_EVAL_PROGRESS_EVERY}" \
      "${WMDP_DIRECT_EXTRA_ARGS[@]}" \
      --out_json "${EVAL_JSON}" \
      >"${EVAL_LOG}" 2>&1
    echo "eval_target=${EVAL_JSON}"
    ;;
esac

echo "checkpoint=${CKPT}"
