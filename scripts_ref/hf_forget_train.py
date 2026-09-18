#! This script initializes a small LLM and finetune to remember some facts
import os
import sys
import argparse
import json
import shlex
import subprocess
from pathlib import Path


_EARLY_ROOT = Path(__file__).resolve().parents[1]
_EARLY_ENTRY = Path("scripts/hf_forget_train.py")
DEFAULT_DATA_ROOT = "data"
DEFAULT_PYTHON = "python"
DEFAULT_WMDP_MODEL = "HuggingFaceH4/zephyr-7b-beta"
DEFAULT_ASSIST_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_TOFU_MODEL = "locuslab/tofu_ft_llama2-7b"


def _early_common_env(seed, gpus):
    data_root = os.environ.get("CBD_DATA_ROOT", DEFAULT_DATA_ROOT)
    repro_env = os.environ.get("REPRO_CONDA_ENV", "cbd")
    repro_python = os.environ.get("PYTHON", DEFAULT_PYTHON)
    return {
        "PYTHONPATH": str(_EARLY_ROOT),
        "CBD_DATA_ROOT": data_root,
        "TOFU_DATA_NAME": os.environ.get("TOFU_DATA_NAME", f"{data_root}/TOFU"),
        "REPRO_CONDA_ENV": repro_env,
        "PYTHON": repro_python,
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1",
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", str(seed)),
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "1"),
        "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE", "1"),
        "HF_DATASETS_OFFLINE": os.environ.get("HF_DATASETS_OFFLINE", "1"),
        "CBD_FORCE_LOCAL_DATASETS_SHIM": os.environ.get("CBD_FORCE_LOCAL_DATASETS_SHIM", "1"),
        "CUDA_VISIBLE_DEVICES": str(gpus),
        "GPU_SET": str(gpus),
    }


def _early_format_cmd(cmd):
    return " ".join(shlex.quote(str(part)) for part in cmd)


def _early_run_or_print(cmd, env=None, dry_run=False):
    merged_env = os.environ.copy()
    if env:
        merged_env.update({k: str(v) for k, v in env.items() if v is not None})
    if dry_run and env:
        visible_env_keys = [
            "REPRO_CONDA_ENV",
            "REPRO_PROFILE",
            "PYTHON",
            "LD_PRELOAD",
            "CUDA_VISIBLE_DEVICES",
            "GPU_SET",
            "PYTHONHASHSEED",
            "REPRO_ARTIFACT_ROOT",
            "REPRO_CLEAN_LOG_ROOT",
            "WMDP_NUMERIC_MODE",
            "TRAIN_EXACT_DETERMINISTIC",
            "CUBLAS_WORKSPACE_CONFIG",
            "CBD_FORCE_LOCAL_DATASETS_SHIM",
            "MODEL_ATTN_IMPL",
            "EVAL_ATTN_IMPL",
            "RUN_SUFFIX",
            "RUN_TAG_SUFFIX",
            "WHITEBOX_PROTOCOL",
            "WHITEBOX_RETAIN_MATCH_FORGET",
            "USE_MODEL_PARALLEL",
            "MP_REQUIRE_4GPU",
            "MP_DEVICE_MAP",
            "MP_DTYPE",
            "TOP_K",
            "MAX_FORGET",
            "MAX_RETAIN",
            "BASIS_MAX_RETAIN",
            "BASIS_MAX_FORGET",
            "BASIS_ROOT_OVERRIDE",
            "BASIS_PATH_OVERRIDE",
            "SKIP_BASIS",
            "BASIS_GRAD_STORE_DTYPE",
            "LORA_R",
            "LORA_ALPHA",
            "LORA_DROPOUT",
            "TRAIN_MAX_STEPS",
            "SAVE_STEPS_OVERRIDE",
            "TRAIN_EPOCHS",
            "TRAIN_LR",
            "TRAIN_BATCH_SIZE",
            "TRAIN_GRAD_ACC",
            "TRAIN_WEIGHT_DECAY",
            "TRAIN_DATALOADER_NUM_WORKERS",
            "FORGET_WEIGHT",
            "RETAIN_WEIGHT",
            "TRAIN_STRATEGY",
            "CBD_DFB_PROJECT_FORGET_ONLY",
            "GPM_PROJECT_FORGET_ONLY",
            "GPM_MAX_SAMPLES",
            "THRESH_OPTIMIZE",
            "THRESH_MAX_FORGET",
            "THRESH_MAX_RETAIN",
            "THRESH_MAX_FPR",
            "THRESH_MIN_TPR",
            "THRESH_TRUNCATE_MODE",
            "SCORE_LAST_K",
            "SCORE_LAST_K_REDUCE",
            "SCORE_K_MODE",
            "EVAL_SCORE_LAST_K",
            "EVAL_SCORE_LAST_K_REDUCE",
            "EVAL_SCORE_K_MODE",
            "EVAL_TRUNCATE_MODE",
            "GRAYBOX_OFFSET_LOSS",
            "GRAYBOX_OFFSET_WEIGHT",
            "GRAYBOX_OFFSET_DATA_MODE",
            "OFFSET_BASE_DEVICE",
            "OFFSET_BASE_ASSIST_DEVICE",
            "OFFSET_ASSIST_DEVICE",
            "ORACLE_DEVICE",
            "GRAYBOX_ULD_LOSS",
            "GRAYBOX_ULD_DATA_MODE",
            "GRAYBOX_ULD_RETAIN_NUM",
            "GRAYBOX_ULD_RETAIN_WEIGHT",
            "GRAYBOX_ULD_TRAIN_TOP_LOGIT_FILTER",
            "GRAYBOX_ULD_TRAIN_WEIGHT",
            "GRAYBOX_ULD_EVAL_TOP_LOGIT_FILTER",
            "GRAYBOX_ULD_EVAL_WEIGHT",
            "OFFICIAL_ULD_MODEL_UTILS",
            "TRAIN_SPLIT_OVERRIDE",
            "TRAIN_RETAIN_NUM",
            "TOFU_CONV_TEMPLATE_STYLE",
            "TOFU_DATASET_LOAD_MODE",
            "DISABLE_INTERNAL_EVAL",
            "DISABLE_CONTRAST_CACHE",
            "CONTRAST_BATCH_FALLBACK",
            "CONTRAST_STRICT_TOP_MASK",
            "WMDP_MAX_FORGET",
            "WMDP_RETAIN_NUM",
            "WMDP_EVAL_GPU_SET",
            "WMDP_OFFSET_EVAL_DEVICES",
            "WMDP_ULD_EVAL_GPU_SET",
            "WMDP_ULD_EVAL_DEVICES",
            "TOFU_ULD_EVAL_DEVICES",
            "TOFU_OFFSET_EVAL_DEVICES",
            "RUN_EVAL",
            "SKIP_TRAIN",
            "SKIP_EVAL",
            "SKIP_ROUTING_EVAL",
            "SKIP_FINAL_EVAL",
        ]
        visible = {k: merged_env[k] for k in visible_env_keys if k in merged_env and merged_env[k] != ""}
        if visible:
            print("# env " + " ".join(f"{k}={shlex.quote(str(v))}" for k, v in visible.items()))
    print(_early_format_cmd(cmd))
    if dry_run:
        return 0
    return subprocess.call(cmd, cwd=str(_EARLY_ROOT), env=merged_env)


def _early_repro_usage():
    return """\
固定复现入口:

  python scripts/hf_forget_train.py repro whitebox tofu <method> <split> [seed] [--gpus 0,1,2,3] [--stage both|train|eval]
  python scripts/hf_forget_train.py repro whitebox wmdp <method> [seed] [--split bio_cyber_chem] [--gpus 0,1,2,3]
  python scripts/hf_forget_train.py repro graybox tofu <uld|offset> <split> [seed] [--gpus 0] [--profile default|official]
  python scripts/hf_forget_train.py repro graybox wmdp <uld|offset> [seed] [--split bio_cyber_chem] [--gpus 0]
  python scripts/hf_forget_train.py repro blackbox tofu <split> [seed] [--top-k paper-default] [--gpus 0]
  python scripts/hf_forget_train.py repro blackbox wmdp [seed] [--top-k 160] [--gpus 0]
  python scripts/hf_forget_train.py repro baseline tofu <vanilla|retain> <split> [seed] [--gpus 0]
  python scripts/hf_forget_train.py repro baseline wmdp vanilla [seed] [--gpus 0]
  python scripts/hf_forget_train.py repro eval tofu-route <run_tag> <split> <checkpoint_path> <threshold> [seed] [--gpus 0]
  python scripts/hf_forget_train.py repro eval wmdp-route <run_tag> <checkpoint_path> <threshold_json> [seed] [--gpus 0]
  python scripts/hf_forget_train.py repro eval wmdp <run_tag> <checkpoint_path> [seed] [--gpus 0]
  python scripts/hf_forget_train.py repro gpm tofu <split> [seed] [--gpus 0]
  python scripts/hf_forget_train.py repro gpm wmdp [seed] [--gpus 0]
  python scripts/hf_forget_train.py repro table <A1|A2|A3|A4|B1|B2|B3|B4|B5|B6|all> [--dry-run]
  python scripts/hf_forget_train.py repro sweep topk <tofu01|tofu05|tofu10|wmdp> --values 32,64,96,128,160
  python scripts/hf_forget_train.py repro sweep basis-retain <tofu01|tofu05|tofu10|wmdp> --values 400,800,1200,1600,2000
  python scripts/hf_forget_train.py repro sweep basis-forget <tofu01|tofu05|tofu10|wmdp> --values 100,200,300,400,500
  python scripts/hf_forget_train.py repro sweep lora-r <tofu01|tofu05|tofu10|wmdp> --values 16,32,48,64,80
  python scripts/hf_forget_train.py repro sweep forget-steps <tofu01|tofu05|tofu10|wmdp> --values 60,120,180,240,300

说明:
  这个文件是唯一公开入口。其它脚本只能作为内部实现被这里调用。
  加 --dry-run 只打印将要执行的命令，不启动训练。
"""


def _paper_slug(method):
    return str(method).replace("+", "p")


def _early_suffix_with_append(suffix):
    append = os.environ.get("REPRO_RUN_SUFFIX_APPEND", os.environ.get("REPRO_PAPER_PROBE_STAMP", "")).strip()
    if not append:
        return suffix
    return f"{suffix}_{append}"


def _wmdp_b6_cbddfb_suffix(step):
    step = str(step)
    if step == "150":
        return _early_suffix_with_append("wmdp_s150_topk_160_20260503i")
    if step in {"50", "100", "200"}:
        return _early_suffix_with_append(f"wmdp_cbddfb_topk160_step{step}_20260504a")
    return _early_suffix_with_append(f"stepgrid_20260516b_wmdp_cbddfb_s{step}")


def _wmdp_b6_whitebox_suffix(method, step):
    step = str(step)
    slug = _paper_slug(method)
    if step in {"50", "100", "150"}:
        return _early_suffix_with_append(f"wmdp_whitebox_lr2e5_stepgrid_20260505a_{slug}_s{step}")
    return _early_suffix_with_append(f"stepgrid_20260516b_wmdp_{slug}_s{step}")


def _wmdp_b6_uld_suffix(step):
    step = str(step)
    if step == "150":
        return _early_suffix_with_append("wmdp_graybox_unified_lr2e4_b2g4_20260514a")
    if step in {"50", "100"}:
        return _early_suffix_with_append(f"stepstab_20260516a_wmdp_uld_s{step}")
    return _early_suffix_with_append(f"stepgrid_20260516b_wmdp_uld_s{step}")


def _wmdp_cbddfb_suffix(sweep_kind="topk", value="160"):
    value = str(value)
    if sweep_kind == "topk":
        return _early_suffix_with_append(f"wmdp_s150_topk_{value}_20260503i")
    if sweep_kind == "basis-retain":
        return _early_suffix_with_append(f"wmdp_cbddfb_topk160_basisretain{value}_20260504a")
    if sweep_kind == "basis-forget":
        return _early_suffix_with_append(f"wmdp_cbddfb_topk160_basisforget{value}_20260504a")
    if sweep_kind == "lora-r":
        return _early_suffix_with_append(f"wmdp_cbddfb_topk160_lorar{value}_20260504a")
    if sweep_kind in {"forget-steps", "max-steps"}:
        if value == "150":
            return _early_suffix_with_append("wmdp_s150_topk_160_20260503i")
        return _early_suffix_with_append(f"wmdp_cbddfb_topk160_step{value}_20260504a")
    raise SystemExit(f"unsupported WMDP CBD-DFB paper sweep: {sweep_kind}")


def _table_value_csv(table_id, target):
    values = {
        ("B1", "tofu10"): "32,64,96,128,160,192,224,256,288,320,352,384,416,448,480",
        ("B1", "wmdp"): "32,64,96,128,160,192,224,256,320",
        ("B2", "tofu10"): "400,800,1200,1600,2000,2400,2800,3200",
        ("B2", "wmdp"): "300,600,900,1200,1500,1800",
        ("B3", "tofu10"): "100,200,300,400,500",
        ("B3", "wmdp"): "300,600,900,1200,1500,1800",
        ("B4", "tofu10"): "16,32,48,64,80",
        ("B4", "wmdp"): "16,32,48,64,80",
        ("B5", "tofu10"): "60,120,180,240,300",
        ("B5", "wmdp"): "50,100,150,200,250,300",
    }
    try:
        return values[(table_id, target)]
    except KeyError:
        raise SystemExit(f"unsupported paper table values: {table_id}/{target}")


def _tofu10_b6_cbddfb_suffix(step):
    step = str(step)
    if step == "180":
        return _early_suffix_with_append("tofu_cbddfb_forget10_retain400_projecttrue_basis400_20260514e")
    return _early_suffix_with_append(f"stepgrid_20260516b_tofu10_cbddfb_s{step}")


def _tofu10_b6_whitebox_suffix(method, step):
    step = str(step)
    slug = _paper_slug(method)
    if method == "ga" and step == "120":
        return _early_suffix_with_append("cbd_repro_whitebox_tofu10_ga_mp4_20260427g")
    if method == "npo+gd" and step == "120":
        return _early_suffix_with_append("cbd_full_tofu_whitebox_mp4_20260427l")
    if method == "npo+gd" and step == "180":
        return _early_suffix_with_append("stepstab_20260516a_tofu10_npopgd_s180")
    return _early_suffix_with_append(f"stepgrid_20260516b_tofu10_{slug}_s{step}")


def _tofu10_b6_uld_suffix(step):
    step = str(step)
    if step == "120":
        return _early_suffix_with_append("stepstab_20260516a_tofu10_uld_s120")
    if step == "180":
        return _early_suffix_with_append("unified_budget_uld_forget10_20260506a")
    return _early_suffix_with_append(f"stepgrid_20260516b_tofu10_uld_s{step}")


def _tofu_a_cbddfb_suffix(split):
    return _early_suffix_with_append({
        "forget01": "cbddfb_tofu01_trainretain400_20260503k",
        "forget05": "tofu_cbddfb_forget05_retain400_projecttrue_basisall_20260514e",
        "forget10": "tofu_cbddfb_forget10_retain400_projecttrue_basis400_20260514e",
    }[split])


def _tofu_a_cbddfb_top_k(split):
    return {"forget01": "40", "forget05": "192", "forget10": "192"}[split]


def _tofu_a_graybox_suffix(method, split):
    if method == "ULD":
        return _early_suffix_with_append(f"unified_budget_uld_{split}_20260506a")
    if method == "Offset":
        return _early_suffix_with_append(f"unified_budget_b4ga2_offset_{split}_20260506b")
    raise SystemExit(f"unsupported ToFU graybox paper row: {method}/{split}")


def _tofu_a_gpm_suffix(split):
    if split == "forget10":
        return _early_suffix_with_append("tofu_graybox_unified_cbddfb_20260516a_gpm_forget10_pf1")
    return _early_suffix_with_append(f"tofu_graybox_unified_cbddfb_20260514c_gpm_{split}")


def _tofu_a_whitebox_suffix(method, split):
    suffixes = {
        ("forget01", "ga"): "tofu_wb_step_search_20260515a_forget01_ga_s20",
        ("forget01", "ga+gd"): "tofu_wb_step_search_20260515b_forget01_gapgd_s50",
        ("forget01", "ga+kl"): "tofu_wb_step_search_20260515c_forget01_gapkl_s50",
        ("forget01", "npo"): "tofu_wb_step_search_20260515h_forget01_npo_s100",
        ("forget01", "npo+gd"): "tofu_wb_step_search_20260515f_forget01_npopgd_s100",
        ("forget01", "npo+kl"): "tofu_whitebox_epoch10_full_20260507d",
        ("forget01", "dpo"): "tofu01_remaining_wb_stepgrid_20260506c_s15",
        ("forget01", "dpo+gd"): "tofu01_remaining_wb_stepgrid_20260506c_s20",
        ("forget01", "dpo+kl"): "tofu01_remaining_wb_stepgrid_20260506c_s20",
        ("forget05", "ga"): "tofu_wb_fix_20260515s_forget05_ga_s80",
        ("forget05", "ga+gd"): "tofu_wb_fix_20260515s_forget05_gapgd_s120",
        ("forget05", "ga+kl"): "tofu_wb_fix_20260515s_forget05_gapkl_s120",
        ("forget05", "npo"): "tofu_wb_fix_20260515s_forget05_npo_s80",
        ("forget05", "npo+gd"): "tofu_whitebox_epoch10_full_20260507d",
        ("forget05", "npo+kl"): "tofu_whitebox_epoch10_full_20260507d",
        ("forget05", "dpo"): "tofu_wb_fix_20260515u_forget05_dpo_s25",
        ("forget05", "dpo+gd"): "tofu_wb_fix_20260515t_forget05_dpopgd_s30",
        ("forget05", "dpo+kl"): "tofu_wb_fix_20260515t_forget05_dpopkl_s30",
        ("forget10", "ga"): "cbd_repro_whitebox_tofu10_ga_mp4_20260427g",
        ("forget10", "ga+gd"): "tofu_wb_fix_20260515t_forget10_gapgd_s170",
        ("forget10", "ga+kl"): "tofu_wb_fix_20260515t_forget10_gapkl_s180",
        ("forget10", "npo"): "cbd_repro_whitebox_tofu10_npo_mp4_20260427g",
        ("forget10", "npo+gd"): "cbd_full_tofu_whitebox_mp4_20260427l",
        ("forget10", "npo+kl"): "cbd_full_tofu_whitebox_mp4_20260427l",
        ("forget10", "dpo"): "tofu_wb_fix_20260515t_forget10_dpo_s25",
        ("forget10", "dpo+gd"): "cbd_full_tofu_whitebox_mp4_20260427l",
        ("forget10", "dpo+kl"): "cbd_full_tofu_whitebox_mp4_20260427l",
    }
    try:
        return _early_suffix_with_append(suffixes[(split, method)])
    except KeyError:
        raise SystemExit(f"unsupported ToFU whitebox paper row: {split}/{method}")


def _early_dispatch_repro(argv):
    if not argv or argv[0] in {"-h", "--help"}:
        print(_early_repro_usage())
        return 0
    parser = argparse.ArgumentParser(prog="python scripts/hf_forget_train.py repro")
    parser.add_argument("family", choices=["whitebox", "graybox", "blackbox", "baseline", "eval", "gpm", "sweep", "table"])
    parser.add_argument("dataset")
    parser.add_argument("rest", nargs="*")
    parser.add_argument("--split", default=None)
    parser.add_argument("--seed", default=None)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--top-k", default=None)
    parser.add_argument("--values", default=None)
    parser.add_argument("--stage", choices=["both", "train", "eval"], default=os.environ.get("REPRO_STAGE", "both"))
    parser.add_argument("--profile", choices=["default", "official"], default=os.environ.get("REPRO_PROFILE", "default"))
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    family = args.family
    dataset = args.dataset
    rest = list(args.rest)
    dry_run = args.dry_run
    profile = args.profile

    tofu_targets = {"tofu01": "forget01", "tofu05": "forget05", "tofu10": "forget10"}
    tofu_whitebox_steps = {
        "forget01": {
            "ga": "20",
            "ga+gd": "50",
            "ga+kl": "50",
            "npo": "100",
            "npo+gd": "100",
            "npo+kl": "140",
            "dpo": "15",
            "dpo+gd": "20",
            "dpo+kl": "20",
        },
        "forget05": {
            "ga": "80",
            "ga+gd": "120",
            "ga+kl": "120",
            "npo": "80",
            "npo+gd": "190",
            "npo+kl": "190",
            "dpo": "25",
            "dpo+gd": "30",
            "dpo+kl": "30",
        },
        "forget10": {
            "ga": "120",
            "ga+gd": "170",
            "ga+kl": "180",
            "npo": "120",
            "npo+gd": "120",
            "npo+kl": "120",
            "dpo": "25",
            "dpo+gd": "30",
            "dpo+kl": "30",
        },
    }
    dpo_family = {"dpo", "dpo+gd", "dpo+kl"}
    retain_split_map = {"forget01": "retain99", "forget05": "retain95", "forget10": "retain90"}
    tofu_whitebox_epoch_full = {
        ("npo+kl", "forget01"),
        ("npo+gd", "forget05"),
        ("npo+kl", "forget05"),
    }

    def paper_tofu_whitebox_step(method, split):
        if split not in tofu_whitebox_steps:
            raise SystemExit(f"unsupported ToFU split: {split}")
        if method not in tofu_whitebox_steps[split]:
            raise SystemExit(f"unsupported ToFU whitebox method: {method}")
        return tofu_whitebox_steps[split][method]

    def paper_tofu_whitebox_is_epoch_full(method, split):
        return (method, split) in tofu_whitebox_epoch_full

    def paper_cbd_dfb_tofu_top_k(split):
        if split == "forget01":
            return "40"
        if split in {"forget05", "forget10"}:
            return "192"
        raise SystemExit(f"unsupported ToFU split: {split}")

    paper_wmdp_cbd_dfb_suffix = _wmdp_cbddfb_suffix

    def wmdp_route_eval_cmd(env, run_tag, ckpt_path, threshold_json, seed):
        base_model = os.environ.get("BASE_MODEL", DEFAULT_WMDP_MODEL)
        assist_model = os.environ.get("ASSIST_MODEL", DEFAULT_ASSIST_MODEL)
        assist_base = os.environ.get("ASSIST_BASE_IF_LORA", assist_model)
        out_json = os.environ.get("WMDP_ROUTE_EVAL_OUT_JSON", f"artifacts/eval_outputs/wmdp/{run_tag}/eval.json")
        return [
            env["PYTHON"],
            "scripts/eval_wmdp_routing.py",
            "--finetuned_assist_path",
            ckpt_path,
            "--threshold_json",
            threshold_json,
            "--base_model",
            base_model,
            "--original_assist",
            assist_model,
            "--assist_base_if_lora",
            assist_base,
            "--base_tokenizer",
            os.environ.get("BASE_TOKENIZER", base_model),
            "--assist_tokenizer",
            os.environ.get("ASSIST_TOKENIZER", assist_base),
            "--base_device",
            os.environ.get("WMDP_ROUTE_BASE_DEVICE", "cuda:0"),
            "--original_device",
            os.environ.get("WMDP_ROUTE_ORIGINAL_DEVICE", "cuda:0"),
            "--finetuned_device",
            os.environ.get("WMDP_ROUTE_FINETUNED_DEVICE", "cuda:0"),
            "--mmlu_test_file",
            os.environ.get("MMLU_TEST_FILE", f"{env['CBD_DATA_ROOT']}/eval-method/wmdp/data/mmlu/all_test.jsonl"),
            "--batch_size",
            os.environ.get("WMDP_ROUTE_EVAL_BATCH_SIZE", os.environ.get("EVAL_BATCH_SIZE", "4")),
            "--max_len",
            os.environ.get("WMDP_ROUTE_MAX_LEN", os.environ.get("TRAIN_MAX_LEN", "512")),
            "--max_mmlu",
            os.environ.get("EVAL_MAX_MMLU", "0"),
            "--max_wmdp",
            os.environ.get("EVAL_MAX_WMDP", "0"),
            "--seed",
            seed,
            "--score_space",
            os.environ.get("SCORE_SPACE", "vocab"),
            "--score_pos",
            os.environ.get("SCORE_POS", "prompt_last"),
            "--score_probe_suffix",
            os.environ.get("SCORE_PROBE_SUFFIX", ""),
            "--score_last_k",
            os.environ.get("SCORE_LAST_K", "4"),
            "--score_last_k_reduce",
            os.environ.get("SCORE_LAST_K_REDUCE", "max"),
            "--score_reducer_alpha",
            os.environ.get("SCORE_REDUCER_ALPHA", "1.0"),
            "--score_reducer_beta",
            os.environ.get("SCORE_REDUCER_BETA", "1.0"),
            "--score_k_mode",
            os.environ.get("SCORE_K_MODE", "last"),
            "--truncate_mode",
            os.environ.get("EVAL_TRUNCATE_MODE", "head_tail"),
            "--progress_every",
            os.environ.get("WMDP_EVAL_PROGRESS_EVERY", "200"),
            "--out_json",
            out_json,
        ]

    def apply_cbd_dfb_tofu_defaults(env, split):
        env.update({
            "TOP_K": os.environ.get("TOP_K", env.get("TOP_K", "192")),
            "MAX_RETAIN": os.environ.get("MAX_RETAIN", "2400"),
            "TRAIN_LR": os.environ.get("TRAIN_LR", "0.00015"),
            "TRAIN_BATCH_SIZE": os.environ.get("TRAIN_BATCH_SIZE", "4"),
            "TRAIN_GRAD_ACC": os.environ.get("TRAIN_GRAD_ACC", "2"),
            "TRAIN_WEIGHT_DECAY": os.environ.get("TRAIN_WEIGHT_DECAY", "0"),
            "TRAIN_MAX_STEPS": os.environ.get("TRAIN_MAX_STEPS", "180"),
            "FORGET_WEIGHT": os.environ.get("FORGET_WEIGHT", "4"),
            "RETAIN_WEIGHT": os.environ.get("RETAIN_WEIGHT", "1"),
            "LORA_R": os.environ.get("LORA_R", "32"),
            "LORA_ALPHA": os.environ.get("LORA_ALPHA", "64"),
            "LORA_DROPOUT": os.environ.get("LORA_DROPOUT", "0.05"),
            "BASIS_BATCH_SIZE": os.environ.get("BASIS_BATCH_SIZE", "4"),
            "BASIS_GRAD_STORE_DTYPE": os.environ.get("BASIS_GRAD_STORE_DTYPE", "float16"),
            "MU_MODE": os.environ.get("MU_MODE", "auto"),
            "MU_SCALE": os.environ.get("MU_SCALE", "1e-2"),
            "TARGET_VARIANCE": os.environ.get("TARGET_VARIANCE", "0.9"),
            "UNLEARN_LOSS": os.environ.get("UNLEARN_LOSS", "gd+kl"),
            "THRESH_OPTIMIZE": os.environ.get("THRESH_OPTIMIZE", "accuracy"),
            "THRESH_MAX_FPR": os.environ.get("THRESH_MAX_FPR", "0.04"),
            "SKIP_ROUTING_EVAL": os.environ.get("SKIP_ROUTING_EVAL", "0"),
            "ASSIST_MODEL": os.environ.get("ASSIST_MODEL", DEFAULT_ASSIST_MODEL),
            "TOFU_BASE_MODEL": os.environ.get("TOFU_BASE_MODEL", DEFAULT_TOFU_MODEL),
            "TOFU_DATASET_NAME": os.environ.get("TOFU_DATASET_NAME", "locuslab/TOFU"),
        })
        if split == "forget10":
            env.update({
                "MAX_FORGET": os.environ.get("MAX_FORGET", "400"),
                "TRAIN_RETAIN_NUM": os.environ.get("TRAIN_RETAIN_NUM", "400"),
                "TRAIN_EPOCHS": os.environ.get("TRAIN_EPOCHS", "10"),
                "CBD_DFB_PROJECT_FORGET_ONLY": os.environ.get("CBD_DFB_PROJECT_FORGET_ONLY", "1"),
            })
        elif split == "forget05":
            env.update({
                "MAX_FORGET": os.environ.get("MAX_FORGET", "200"),
                "TRAIN_RETAIN_NUM": os.environ.get("TRAIN_RETAIN_NUM", "400"),
                "TRAIN_EPOCHS": os.environ.get("TRAIN_EPOCHS", "10"),
                "CBD_DFB_PROJECT_FORGET_ONLY": os.environ.get("CBD_DFB_PROJECT_FORGET_ONLY", "1"),
            })
        elif split == "forget01":
            env.update({
                "MAX_FORGET": os.environ.get("MAX_FORGET", "40"),
                "TRAIN_RETAIN_NUM": os.environ.get("TRAIN_RETAIN_NUM", "400"),
                "TRAIN_EPOCHS": os.environ.get("TRAIN_EPOCHS", "10"),
                "CBD_DFB_PROJECT_FORGET_ONLY": os.environ.get("CBD_DFB_PROJECT_FORGET_ONLY", "1"),
            })
        else:
            raise SystemExit(f"unsupported ToFU split: {split}")

    def tofu_retain_result_path(env, split):
        retain_name = retain_split_map.get(split)
        if retain_name is None:
            raise SystemExit(f"unsupported ToFU split: {split}")
        return f"{env['CBD_DATA_ROOT']}/data/{retain_name}_llama_wd0.01/eval_results/ds_size300/eval_log_aggregated.json"

    def tofu_eval_cmd(run_tag, ckpt_path, split, env, batch_size="2"):
        eval_split = f"{split}_perturbed" if not split.endswith("_perturbed") else split
        plain_split = split[:-10] if split.endswith("_perturbed") else split
        data_root = env["CBD_DATA_ROOT"]
        tofu_model = os.environ.get("TOFU_BASE_MODEL", DEFAULT_TOFU_MODEL)
        return [
            env["PYTHON"],
            "scripts/eval_tofu.py",
            f"OUTDIRNAME=artifacts/eval_outputs/tofu/{run_tag}",
            f"ckpt_path={ckpt_path}",
            "model=tofu-llama-2",
            "model_mode=base",
            f"model.model_path={tofu_model}",
            f"model.tokenizer_path={os.environ.get('TOFU_TOKENIZER_PATH', tofu_model)}",
            f"data.dataset.name={env['TOFU_DATA_NAME']}",
            f"data.dataset.split={eval_split}",
            f"data.dataset.perturb_path={data_root}/data/aug_data/tofu/{eval_split}/perturb_res.csv",
            f"data.dataset.paraphrase_path={data_root}/data/aug_data/tofu/{eval_split}/paraphrase_res.csv",
            f"data.dataset.eval.batch_size={os.environ.get('TOFU_EVAL_BATCH_SIZE', batch_size)}",
            f"data.dataset.eval.retain_result={tofu_retain_result_path(env, plain_split)}",
            f"+data.dataset.eval.max_num={os.environ.get('TOFU_EVAL_MAX_NUM', '300')}",
        ]

    if family == "table":
        table_id = dataset.upper()
        seed = args.seed or (rest[0] if rest and rest[0].isdigit() else "42")
        all_gpus = args.gpus or "0,1,2,3"
        first_gpu = all_gpus.split(",")[0]
        requested = {item.lower() for item in rest if not item.isdigit()}
        target_tokens = {"tofu", "tofu01", "tofu05", "tofu10", "wmdp"}
        target_filters = requested & target_tokens
        method_filters = requested - target_filters
        whitebox_methods = ["ga", "ga+gd", "ga+kl", "npo", "npo+gd", "npo+kl", "dpo", "dpo+gd", "dpo+kl"]
        specs = []

        def _matches(name, choices):
            aliases = {name.lower()}
            if name.lower() in {"cbd-dfb", "cbddfb"}:
                aliases.update({"cbd-dfb", "cbddfb"})
            return bool(aliases & choices)

        def wants(name):
            return not method_filters or _matches(name, method_filters)

        def wants_target(name):
            return not target_filters or name.lower() in target_filters

        def add(label, child_args, gpus=None, env=None, allow_stage=True):
            child = list(child_args)
            if allow_stage and args.stage != "both" and child and child[0] in {"whitebox", "graybox", "blackbox"}:
                child.extend(["--stage", args.stage])
            child.extend(["--gpus", gpus or first_gpu])
            specs.append((label, child, gpus or first_gpu, env or {}))

        def add_tofu_main(section, split):
            if wants("baseline"):
                add(f"{section} ToFU {split} vanilla", ["baseline", "tofu", "vanilla", split, seed], first_gpu, allow_stage=False)
                add(f"{section} ToFU {split} retain", ["baseline", "tofu", "retain", split, seed], first_gpu, allow_stage=False)
            if wants("uld"):
                add(f"{section} ToFU {split} ULD", ["graybox", "tofu", "uld", split, seed], first_gpu, {"RUN_SUFFIX": _tofu_a_graybox_suffix("ULD", split)})
            if wants("offset"):
                add(f"{section} ToFU {split} Offset", ["graybox", "tofu", "offset", split, seed], all_gpus, {"RUN_SUFFIX": _tofu_a_graybox_suffix("Offset", split)})
            if wants("cbd-dfb") or wants("cbddfb"):
                add(f"{section} ToFU {split} CBD-DFB", ["blackbox", "tofu", split, seed, "--top-k", _tofu_a_cbddfb_top_k(split)], first_gpu, {"RUN_SUFFIX": _tofu_a_cbddfb_suffix(split)})
            if wants("gpm"):
                add(f"{section} ToFU {split} GPM", ["gpm", "tofu", split, seed], first_gpu, {"RUN_SUFFIX": _tofu_a_gpm_suffix(split)}, allow_stage=False)
            for method in whitebox_methods:
                if wants(method):
                    add(f"{section} ToFU {split} {method}", ["whitebox", "tofu", method, split, seed], all_gpus, {"RUN_SUFFIX": _tofu_a_whitebox_suffix(method, split)})

        def add_wmdp_main():
            if wants("baseline"):
                add("A4 WMDP vanilla", ["baseline", "wmdp", "vanilla", seed], first_gpu, allow_stage=False)
            whitebox_env = {
                "TRAIN_LR": "2e-5",
                "TRAIN_MAX_STEPS": "50",
                "SAVE_STEPS_OVERRIDE": "50",
                "TRAIN_BATCH_SIZE": "1",
                "TRAIN_GRAD_ACC": "2",
                "TRAIN_WEIGHT_DECAY": "0",
                "WMDP_RETAIN_NUM": "1200",
                "WMDP_MAX_FORGET": "none",
            }
            for method in whitebox_methods:
                if wants(method):
                    env = dict(whitebox_env)
                    env["RUN_SUFFIX"] = _wmdp_b6_whitebox_suffix(method, "50")
                    add(f"A4 WMDP {method}", ["whitebox", "wmdp", method, seed], all_gpus, env)
            if wants("uld"):
                add("A4 WMDP ULD", ["graybox", "wmdp", "uld", seed, "--split", "bio_cyber_chem"], all_gpus, {"RUN_SUFFIX": _wmdp_b6_uld_suffix("150")})
            if wants("offset"):
                add("A4 WMDP Offset", ["graybox", "wmdp", "offset", seed, "--split", "bio_cyber_chem"], all_gpus, {"RUN_SUFFIX": _wmdp_b6_uld_suffix("150")})
            if wants("cbd-dfb") or wants("cbddfb"):
                add("A4 WMDP CBD-DFB", ["blackbox", "wmdp", seed, "--top-k", "160"], first_gpu, {"RUN_SUFFIX": _wmdp_cbddfb_suffix("topk", "160")})
            if wants("gpm"):
                add("A4 WMDP GPM", ["gpm", "wmdp", seed], first_gpu, {"RUN_SUFFIX": _early_suffix_with_append("wmdp_gpm_match_cbddfb150_20260501a")}, allow_stage=False)

        def add_b_sweep(section):
            sweep_kind = {"B1": "topk", "B2": "basis-retain", "B3": "basis-forget", "B4": "lora-r", "B5": "forget-steps"}[section]
            for target in ["tofu10", "wmdp"]:
                if wants_target(target):
                    values = args.values or _table_value_csv(section, target)
                    child = ["sweep", sweep_kind, target, "--values", values, "--seed", seed]
                    if args.stage != "both":
                        child.extend(["--stage", args.stage])
                    add(f"{section} {target} {sweep_kind}", child, first_gpu, allow_stage=False)

        def add_b6():
            tofu_steps = ["80", "100", "120", "140", "160", "180", "200"]
            wmdp_steps = ["50", "75", "100", "125", "150", "175", "200"]
            if args.values:
                selected = {value.strip() for value in args.values.split(",") if value.strip()}
                tofu_steps = [step for step in tofu_steps if step in selected]
                wmdp_steps = [step for step in wmdp_steps if step in selected]
                if not tofu_steps and not wmdp_steps:
                    raise SystemExit(f"no B6 steps selected by --values {args.values!r}")
            if wants_target("tofu10"):
                tofu_cbddfb_steps = ["180", "80", "100", "120", "140", "160", "200"]
                if args.values:
                    tofu_cbddfb_steps = [step for step in tofu_cbddfb_steps if step in set(tofu_steps)]
                    if tofu_cbddfb_steps and "180" not in tofu_cbddfb_steps and args.stage != "eval":
                        tofu_cbddfb_steps.insert(0, "180")
                tofu_cbddfb_basis_suffix = _tofu10_b6_cbddfb_suffix("180")
                tofu_cbddfb_basis_root = f"artifacts/basis_cbd_dfb/seed{seed}_{tofu_cbddfb_basis_suffix}"
                if wants("cbd-dfb") or wants("cbddfb"):
                    for step in tofu_cbddfb_steps:
                        env = {
                            "RUN_SUFFIX": _tofu10_b6_cbddfb_suffix(step),
                            "TRAIN_MAX_STEPS": step,
                            "SAVE_STEPS_OVERRIDE": step,
                            "TOP_K": "192",
                            "MAX_FORGET": "400",
                            "MAX_RETAIN": "2400",
                            "BASIS_MAX_FORGET": "400",
                            "BASIS_MAX_RETAIN": "2400",
                            "TRAIN_RETAIN_NUM": "400",
                            "CBD_DFB_PROJECT_FORGET_ONLY": "1",
                            "TRAIN_LR": "0.00015",
                            "TRAIN_BATCH_SIZE": "4",
                            "TRAIN_GRAD_ACC": "2",
                            "TRAIN_WEIGHT_DECAY": "0",
                            "FORGET_WEIGHT": "4",
                            "RETAIN_WEIGHT": "1",
                            "LORA_R": "32",
                            "LORA_ALPHA": "64",
                            "LORA_DROPOUT": "0.05",
                        }
                        if step != "180":
                            env["BASIS_ROOT_OVERRIDE"] = os.environ.get("BASIS_ROOT_OVERRIDE", tofu_cbddfb_basis_root)
                            env["SKIP_BASIS"] = os.environ.get("SKIP_BASIS", "1")
                        add(f"B6 ToFU10 CBD-DFB step {step}", ["blackbox", "tofu", "forget10", seed, "--top-k", "192"], first_gpu, env)
                for method in ["ga", "npo+gd"]:
                    if not wants(method):
                        continue
                    for step in tofu_steps:
                        add(
                            f"B6 ToFU10 {method} step {step}",
                            ["whitebox", "tofu", method, "forget10", seed],
                            all_gpus,
                            {"RUN_SUFFIX": _tofu10_b6_whitebox_suffix(method, step), "TRAIN_MAX_STEPS": step, "SAVE_STEPS_OVERRIDE": step},
                        )
                if wants("uld"):
                    for step in tofu_steps:
                        add(
                            f"B6 ToFU10 ULD step {step}",
                            ["graybox", "tofu", "uld", "forget10", seed],
                            first_gpu,
                            {"RUN_SUFFIX": _tofu10_b6_uld_suffix(step), "TRAIN_MAX_STEPS": step, "SAVE_STEPS_OVERRIDE": step},
                        )
            if wants_target("wmdp"):
                cbddfb_env = {
                    "TOP_K": "160",
                    "MAX_FORGET": "600",
                    "MAX_RETAIN": "1200",
                    "TRAIN_RETAIN_NUM": "1200",
                    "TRAIN_LR": "2e-4",
                    "TRAIN_BATCH_SIZE": "2",
                    "TRAIN_GRAD_ACC": "4",
                    "TRAIN_WEIGHT_DECAY": "0.01",
                    "FORGET_WEIGHT": "1",
                    "RETAIN_WEIGHT": "4",
                    "LORA_R": "32",
                    "LORA_ALPHA": "64",
                    "LORA_DROPOUT": "0.05",
                    "CBD_DFB_PROJECT_FORGET_ONLY": "true",
                    "THRESH_OPTIMIZE": "tpr",
                    "THRESH_MAX_FPR": "0.215",
                    "SCORE_POS": "prompt_last",
                    "SCORE_LAST_K": "4",
                    "SCORE_LAST_K_REDUCE": "max",
                    "THRESH_TRUNCATE_MODE": "head_tail",
                    "EVAL_TRUNCATE_MODE": "head_tail",
                }
                if wants("cbd-dfb") or wants("cbddfb"):
                    wmdp_cbddfb_steps = ["150", "50", "75", "100", "125", "175", "200"]
                    if args.values:
                        wmdp_cbddfb_steps = [step for step in wmdp_cbddfb_steps if step in set(wmdp_steps)]
                        if wmdp_cbddfb_steps and "150" not in wmdp_cbddfb_steps and args.stage != "eval":
                            wmdp_cbddfb_steps.insert(0, "150")
                    wmdp_cbddfb_basis_suffix = _wmdp_b6_cbddfb_suffix("150")
                    wmdp_cbddfb_basis_path = f"artifacts/basis_cbd_dfb/wmdp_seed{seed}_{wmdp_cbddfb_basis_suffix}/wmdp_basis/cbd_dfb_basis_wmdp_bio_cyber_chem_vs_mmlu.pkl"
                    for step in wmdp_cbddfb_steps:
                        env = dict(cbddfb_env)
                        env.update({"RUN_SUFFIX": _wmdp_b6_cbddfb_suffix(step), "TRAIN_MAX_STEPS": step, "SAVE_STEPS_OVERRIDE": step})
                        if step != "150":
                            env["BASIS_PATH_OVERRIDE"] = os.environ.get("BASIS_PATH_OVERRIDE", wmdp_cbddfb_basis_path)
                        add(f"B6 WMDP CBD-DFB step {step}", ["blackbox", "wmdp", seed, "--top-k", "160"], first_gpu, env)
                whitebox_env = {
                    "TRAIN_LR": "2e-5",
                    "TRAIN_BATCH_SIZE": "1",
                    "TRAIN_GRAD_ACC": "2",
                    "TRAIN_WEIGHT_DECAY": "0",
                    "WMDP_RETAIN_NUM": "1200",
                    "WMDP_MAX_FORGET": "none",
                }
                for method in ["ga", "npo+gd"]:
                    if not wants(method):
                        continue
                    for step in wmdp_steps:
                        env = dict(whitebox_env)
                        env.update({"RUN_SUFFIX": _wmdp_b6_whitebox_suffix(method, step), "TRAIN_MAX_STEPS": step, "SAVE_STEPS_OVERRIDE": step})
                        add(f"B6 WMDP {method} step {step}", ["whitebox", "wmdp", method, seed], all_gpus, env)
                if wants("uld"):
                    for step in wmdp_steps:
                        add(
                            f"B6 WMDP ULD step {step}",
                            ["graybox", "wmdp", "uld", seed, "--split", "bio_cyber_chem"],
                            all_gpus,
                            {"RUN_SUFFIX": _wmdp_b6_uld_suffix(step), "TRAIN_MAX_STEPS": step, "SAVE_STEPS_OVERRIDE": step},
                        )

        tables = ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "B5", "B6"] if table_id == "ALL" else [table_id]
        for one in tables:
            if one == "A1":
                add_tofu_main(one, "forget01")
            elif one == "A2":
                add_tofu_main(one, "forget05")
            elif one == "A3":
                add_tofu_main(one, "forget10")
            elif one == "A4":
                add_wmdp_main()
            elif one in {"B1", "B2", "B3", "B4", "B5"}:
                add_b_sweep(one)
            elif one == "B6":
                add_b6()
            else:
                raise SystemExit(f"unsupported paper table: {one}")
        if not specs:
            raise SystemExit(f"no commands selected for table {table_id}; filters={sorted(requested)}")

        failures = 0
        for label, child, gpus, env_overrides in specs:
            env = os.environ.copy()
            env.update(_early_common_env(seed, gpus))
            env.update({key: str(value) for key, value in env_overrides.items()})
            print(f"# table {label}")
            rc = _early_run_or_print([env["PYTHON"], _EARLY_ENTRY, "repro", *child], env=env, dry_run=dry_run)
            if rc != 0:
                failures += 1
                if not args.continue_on_error:
                    return rc
        return 1 if failures else 0

    if family == "eval" and dataset == "tofu-route":
        if len(rest) < 4:
            raise SystemExit("usage: python scripts/hf_forget_train.py repro eval tofu-route <run_tag> <split> <checkpoint_path> <threshold> [seed]")
        run_tag = rest[0]
        split = rest[1]
        ckpt_path = rest[2]
        threshold = rest[3]
        seed = rest[4] if len(rest) > 4 else (args.seed or "42")
        gpus = args.gpus or "0"
        if split not in retain_split_map:
            raise SystemExit(f"unsupported ToFU split: {split}")
        env = _early_common_env(seed, gpus)
        eval_split = f"{split}_perturbed"
        assist_model = os.environ.get("ASSIST_MODEL", DEFAULT_ASSIST_MODEL)
        tofu_model = os.environ.get("TOFU_BASE_MODEL", DEFAULT_TOFU_MODEL)
        outdir = os.environ.get("TOFU_ROUTE_EVAL_OUTDIR", f"artifacts/eval_outputs/tofu/{run_tag}/{eval_split}")
        cmd = [
            env["PYTHON"],
            "scripts/eval_tofu.py",
            f"OUTDIRNAME={outdir}",
            f"ckpt_path={ckpt_path}",
            "model=tofu-llama-2",
            f"model.model_path={tofu_model}",
            f"model.tokenizer_path={os.environ.get('TOFU_TOKENIZER_PATH', tofu_model)}",
            "model_mode=double_assis",
            f"model_mode.original_assist_path={assist_model}",
            f"model_mode.finetuned_assist_path={ckpt_path}",
            f"model_mode.threshold={threshold}",
            f"model_mode.max_new_tokens={os.environ.get('CE_MAX_NEW_TOKENS', '32')}",
            f"data.dataset.name={os.environ.get('ROUTE_TOFU_DATA_NAME', env['TOFU_DATA_NAME'])}",
            f"data.dataset.split={eval_split}",
            f"data.dataset.eval.batch_size={os.environ.get('ROUTING_EVAL_BATCH_SIZE', '4')}",
            f"data.dataset.eval.retain_result={tofu_retain_result_path(env, split)}",
            f"+data.dataset.eval.max_num={os.environ.get('TOFU_EVAL_MAX_NUM', '300')}",
        ]
        return _early_run_or_print(cmd, env=env, dry_run=dry_run)

    if family == "eval" and dataset == "wmdp":
        if len(rest) < 2:
            raise SystemExit("usage: python scripts/hf_forget_train.py repro eval wmdp <run_tag> <checkpoint_path> [seed]")
        run_tag = rest[0]
        ckpt_path = rest[1]
        seed = rest[2] if len(rest) > 2 else (args.seed or "42")
        gpus = args.gpus or "0"
        env = _early_common_env(seed, gpus)
        base_model = os.environ.get("WMDP_MODEL_BASE_IF_LORA", os.environ.get("BASE_MODEL", DEFAULT_WMDP_MODEL))
        tokenizer_path = os.environ.get("WMDP_TOKENIZER_PATH", os.environ.get("BASE_TOKENIZER", base_model))
        out_json = os.environ.get("WMDP_EVAL_OUT_JSON", f"artifacts/eval_outputs/wmdp_direct/baselines_commonproto/{run_tag}.json")
        eval_model_mode = os.environ.get("WMDP_EVAL_MODEL_MODE", "auto")
        cmd = [
            env["PYTHON"],
            "scripts/eval_wmdp_direct.py",
            "--model_path",
            ckpt_path,
            "--model_base_if_lora",
            base_model,
            "--tokenizer_path",
            tokenizer_path,
            "--model_mode",
            eval_model_mode,
            "--device",
            "cuda:0",
            "--mmlu_test_file",
            os.environ.get("MMLU_TEST_FILE", f"{env['CBD_DATA_ROOT']}/eval-method/wmdp/data/mmlu/all_test.jsonl"),
            "--seed",
            seed,
            "--max_mmlu",
            os.environ.get("EVAL_MAX_MMLU", "0"),
            "--max_wmdp",
            os.environ.get("EVAL_MAX_WMDP", "0"),
            "--batch_size",
            os.environ.get("WMDP_EVAL_BATCH_SIZE", "8"),
            "--max_len",
            os.environ.get("WMDP_EVAL_MAX_LEN", "512"),
            "--truncate_mode",
            os.environ.get("WMDP_EVAL_TRUNCATE_MODE", "left"),
            "--progress_every",
            os.environ.get("WMDP_EVAL_PROGRESS_EVERY", "200"),
            "--out_json",
            out_json,
        ]
        if eval_model_mode == "uld":
            cmd.extend([
                "--uld_weight",
                os.environ.get("WMDP_ULD_EVAL_WEIGHT", os.environ.get("GRAYBOX_ULD_EVAL_WEIGHT", "-0.75")),
                "--uld_top_logit_filter",
                os.environ.get("WMDP_ULD_EVAL_TOP_LOGIT_FILTER", os.environ.get("GRAYBOX_ULD_EVAL_TOP_LOGIT_FILTER", "0.01")),
                "--eval_devices",
                os.environ.get("WMDP_ULD_EVAL_DEVICES", "cuda:0,cuda:1"),
            ])
        elif eval_model_mode == "offset":
            cmd.extend([
                "--offset_base_assist_path",
                os.environ.get("WMDP_OFFSET_BASE_ASSIST_PATH", os.environ.get("ASSIST_MODEL_PATH", DEFAULT_ASSIST_MODEL)),
                "--offset_weight",
                os.environ.get("WMDP_OFFSET_WEIGHT", os.environ.get("GRAYBOX_OFFSET_WEIGHT", "1.0")),
                "--eval_devices",
                os.environ.get("WMDP_OFFSET_EVAL_DEVICES", "cuda:0"),
            ])
        return _early_run_or_print(cmd, env=env, dry_run=dry_run)

    if family == "eval" and dataset == "wmdp-route":
        if len(rest) < 3:
            raise SystemExit("usage: python scripts/hf_forget_train.py repro eval wmdp-route <run_tag> <checkpoint_path> <threshold_json> [seed]")
        run_tag = rest[0]
        ckpt_path = rest[1]
        threshold_json = rest[2]
        seed = rest[3] if len(rest) > 3 else (args.seed or "42")
        gpus = args.gpus or "0"
        env = _early_common_env(seed, gpus)
        cmd = wmdp_route_eval_cmd(env, run_tag, ckpt_path, threshold_json, seed)
        return _early_run_or_print(cmd, env=env, dry_run=dry_run)

    if family == "whitebox" and dataset == "tofu":
        method = rest[0]
        split = rest[1]
        seed = rest[2] if len(rest) > 2 else (args.seed or "42")
        gpus = args.gpus or "0,1,2,3"
        env = _early_common_env(seed, gpus)
        epoch_full = paper_tofu_whitebox_is_epoch_full(method, split)
        paper_step = paper_tofu_whitebox_step(method, split)
        default_run_suffix = "paperreal_20260609a" if epoch_full else "fixedentry"
        env.update({
            "USE_MODEL_PARALLEL": "0" if args.stage == "eval" else "1",
            "MP_REQUIRE_4GPU": "0" if args.stage == "eval" else "1",
            "MP_DEVICE_MAP": "balanced",
            "MP_DTYPE": "float16",
            "RUN_EVAL": "0" if args.stage == "train" else os.environ.get("RUN_EVAL", "1"),
            "SKIP_TRAIN": "1" if args.stage == "eval" else os.environ.get("SKIP_TRAIN", "0"),
            "DISABLE_INTERNAL_EVAL": os.environ.get("DISABLE_INTERNAL_EVAL", "1"),
            "TOFU_DATASET_LOAD_MODE": os.environ.get("TOFU_DATASET_LOAD_MODE", "legacy"),
            "TRAIN_EPOCHS": os.environ.get("TRAIN_EPOCHS", "10" if epoch_full else "100"),
            "TRAIN_MAX_STEPS": os.environ.get("TRAIN_MAX_STEPS", "none" if epoch_full else paper_step),
            "SAVE_STEPS_OVERRIDE": os.environ.get("SAVE_STEPS_OVERRIDE", "none" if epoch_full else paper_step),
            "FORCE_SAVE_FINAL_CHECKPOINT": os.environ.get("FORCE_SAVE_FINAL_CHECKPOINT", "1"),
            "RUN_TAG_SUFFIX": os.environ.get("RUN_SUFFIX", os.environ.get("RUN_TAG_SUFFIX", default_run_suffix)),
            # The paper ToFU whitebox table uses the locked 4-GPU ULD-style protocol.
            "WHITEBOX_PROTOCOL": os.environ.get("WHITEBOX_PROTOCOL", "official_uld"),
            "WHITEBOX_RETAIN_MATCH_FORGET": os.environ.get("WHITEBOX_RETAIN_MATCH_FORGET", "0" if epoch_full else "1"),
            "TRAIN_LR": os.environ.get("TRAIN_LR", "1e-5"),
            "TRAIN_BATCH_SIZE": os.environ.get("TRAIN_BATCH_SIZE", "4"),
            "TRAIN_GRAD_ACC": os.environ.get("TRAIN_GRAD_ACC", "8"),
            "TRAIN_WEIGHT_DECAY": os.environ.get("TRAIN_WEIGHT_DECAY", "0.01"),
            "LORA_R": os.environ.get("LORA_R", "0"),
            "LORA_ALPHA": os.environ.get("LORA_ALPHA", "32"),
            "LORA_DROPOUT": os.environ.get("LORA_DROPOUT", "0.05"),
        })
        return _early_run_or_print(["bash", "scripts/internal/run_whitebox_mp4.sh", method, split, seed], env=env, dry_run=dry_run)

    if family == "baseline" and dataset == "tofu":
        if len(rest) < 2:
            raise SystemExit("usage: python scripts/hf_forget_train.py repro baseline tofu <vanilla|retain> <split> [seed]")
        method = rest[0]
        split = rest[1]
        seed = rest[2] if len(rest) > 2 else (args.seed or "42")
        gpus = args.gpus or "0"
        if split not in retain_split_map:
            raise SystemExit(f"unsupported ToFU split: {split}")
        if method not in {"vanilla", "retain"}:
            raise SystemExit(f"unsupported ToFU baseline method: {method}")
        env = _early_common_env(seed, gpus)
        run_suffix = os.environ.get("RUN_SUFFIX", "fixedentry")
        if method == "vanilla":
            ckpt_path = os.environ.get("TOFU_BASE_MODEL", DEFAULT_TOFU_MODEL)
            run_tag = f"baseline_vanilla_tofu_{split}_s{seed}_{run_suffix}"
        else:
            retain_name = retain_split_map[split]
            env_key = f"{retain_name.upper()}_MODEL_PATH"
            ckpt_path = os.environ.get(env_key, f"{env['CBD_DATA_ROOT']}/artifacts/hf_models/open_unlearning_{retain_name}")
            run_tag = f"baseline_{retain_name}_tofu_{split}_s{seed}_{run_suffix}"
        return _early_run_or_print(tofu_eval_cmd(run_tag, ckpt_path, split, env), env=env, dry_run=dry_run)

    if family == "baseline" and dataset == "wmdp":
        method = rest[0] if rest else "vanilla"
        seed = rest[1] if len(rest) > 1 else (args.seed or "42")
        gpus = args.gpus or "0"
        if method != "vanilla":
            raise SystemExit("usage: python scripts/hf_forget_train.py repro baseline wmdp vanilla [seed]")
        env = _early_common_env(seed, gpus)
        run_suffix = os.environ.get("RUN_SUFFIX", "fixedentry")
        base_model = os.environ.get("BASE_MODEL", DEFAULT_WMDP_MODEL)
        out_json = f"artifacts/eval_outputs/wmdp_direct/baseline_vanilla_wmdp_s{seed}_{run_suffix}.json"
        cmd = [
            env["PYTHON"],
            "scripts/eval_wmdp_direct.py",
            "--model_path",
            base_model,
            "--tokenizer_path",
            os.environ.get("BASE_TOKENIZER", base_model),
            "--device",
            "cuda:0",
            "--mmlu_test_file",
            os.environ.get("MMLU_TEST_FILE", f"{env['CBD_DATA_ROOT']}/eval-method/wmdp/data/mmlu/all_test.jsonl"),
            "--seed",
            seed,
            "--max_mmlu",
            os.environ.get("EVAL_MAX_MMLU", "0"),
            "--max_wmdp",
            os.environ.get("EVAL_MAX_WMDP", "0"),
            "--batch_size",
            os.environ.get("WMDP_EVAL_BATCH_SIZE", "8"),
            "--max_len",
            os.environ.get("WMDP_EVAL_MAX_LEN", "512"),
            "--truncate_mode",
            os.environ.get("WMDP_EVAL_TRUNCATE_MODE", "left"),
            "--progress_every",
            os.environ.get("WMDP_EVAL_PROGRESS_EVERY", "200"),
            "--out_json",
            out_json,
        ]
        return _early_run_or_print(cmd, env=env, dry_run=dry_run)

    if family in {"whitebox", "graybox"} and dataset in {"tofu", "wmdp"}:
        method = rest[0]
        split = args.split or (rest[1] if len(rest) > 1 and dataset == "tofu" else "bio_cyber_chem")
        seed = args.seed or (rest[2] if len(rest) > 2 and dataset == "tofu" else (rest[1] if len(rest) > 1 and dataset == "wmdp" else "42"))
        gpus = args.gpus or "0"
        env = _early_common_env(seed, gpus)
        env.update({
            "REPRO_PROFILE": profile,
            "SKIP_EVAL": "1" if args.stage == "train" else os.environ.get("SKIP_EVAL", "0"),
            "SKIP_TRAIN": "1" if args.stage == "eval" else os.environ.get("SKIP_TRAIN", "0"),
        })
        if family == "whitebox" and "," in str(gpus):
            env.update({
                "USE_MODEL_PARALLEL": "1",
                "MP_REQUIRE_4GPU": "1",
                "MP_DEVICE_MAP": os.environ.get("MP_DEVICE_MAP", "balanced"),
                "MP_DTYPE": os.environ.get("MP_DTYPE", "float16"),
            })
        if family == "whitebox" and dataset == "wmdp":
            env.update({
                "TRAIN_LR": os.environ.get("TRAIN_LR", "2e-5"),
                "TRAIN_BATCH_SIZE": os.environ.get("TRAIN_BATCH_SIZE", "1"),
                "TRAIN_GRAD_ACC": os.environ.get("TRAIN_GRAD_ACC", "2"),
                "TRAIN_WEIGHT_DECAY": os.environ.get("TRAIN_WEIGHT_DECAY", "0"),
                "TRAIN_MAX_STEPS": os.environ.get("TRAIN_MAX_STEPS", "50"),
                "SAVE_STEPS_OVERRIDE": os.environ.get("SAVE_STEPS_OVERRIDE", "50"),
                "TRAIN_EPOCHS": os.environ.get("TRAIN_EPOCHS", "2"),
                "WMDP_MAX_FORGET": os.environ.get("WMDP_MAX_FORGET", "none"),
                "WMDP_RETAIN_NUM": os.environ.get("WMDP_RETAIN_NUM", "1200"),
                "LORA_R": os.environ.get("LORA_R", "32"),
                "LORA_ALPHA": os.environ.get("LORA_ALPHA", "64"),
                "LORA_DROPOUT": os.environ.get("LORA_DROPOUT", "0.05"),
            })
        if family == "graybox" and method == "uld" and dataset == "tofu":
            repro_env = os.environ.get("REPRO_CONDA_ENV", "cbd")
            if profile == "official":
                env.update({
                    "REPRO_CONDA_ENV": repro_env,
                    "PYTHON": os.environ.get("PYTHON", DEFAULT_PYTHON),
                    "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", str(seed)),
                    "TRAIN_MAX_STEPS": os.environ.get("TRAIN_MAX_STEPS", "none"),
                    "SAVE_STEPS_OVERRIDE": os.environ.get("SAVE_STEPS_OVERRIDE", "none"),
                    "FORCE_SAVE_FINAL_CHECKPOINT": os.environ.get("FORCE_SAVE_FINAL_CHECKPOINT", "1"),
                    "USE_MODEL_PARALLEL": os.environ.get("USE_MODEL_PARALLEL", "1"),
                    "MP_REQUIRE_4GPU": os.environ.get("MP_REQUIRE_4GPU", "1"),
                    "MP_DEVICE_MAP": os.environ.get("MP_DEVICE_MAP", "balanced"),
                    "MP_DTYPE": os.environ.get("MP_DTYPE", "float16"),
                    "TRAIN_STRATEGY": os.environ.get("TRAIN_STRATEGY", "none"),
                    "TRAIN_SPLIT_OVERRIDE": os.environ.get("TRAIN_SPLIT_OVERRIDE", f"{split}_perturbed"),
                    "GRAYBOX_ULD_LOSS": os.environ.get("GRAYBOX_ULD_LOSS", "remember+uniform"),
                    "GRAYBOX_ULD_DATA_MODE": os.environ.get("GRAYBOX_ULD_DATA_MODE", "forget_more_retain_perturb"),
                    "GRAYBOX_ULD_TRAIN_WEIGHT": os.environ.get("GRAYBOX_ULD_TRAIN_WEIGHT", "-1.0"),
                    "GRAYBOX_ULD_TRAIN_TOP_LOGIT_FILTER": os.environ.get("GRAYBOX_ULD_TRAIN_TOP_LOGIT_FILTER", "0.1"),
                    "GRAYBOX_ULD_EVAL_WEIGHT": os.environ.get("GRAYBOX_ULD_EVAL_WEIGHT", "-0.8"),
                    "GRAYBOX_ULD_EVAL_TOP_LOGIT_FILTER": os.environ.get("GRAYBOX_ULD_EVAL_TOP_LOGIT_FILTER", "0.01"),
                    "TOFU_ULD_EVAL_DEVICES": os.environ.get("TOFU_ULD_EVAL_DEVICES", "cuda:0|cuda:1"),
                    "TRAIN_LR": os.environ.get("TRAIN_LR", "1e-3"),
                    "TRAIN_BATCH_SIZE": os.environ.get("TRAIN_BATCH_SIZE", "8"),
                    # Official script uses 2-way DDP: 8 * 2 grad * 2 ranks = 32.
                    # This repo uses 4-GPU model-parallel single-process training,
                    # so grad_acc=4 preserves the same effective batch.
                    "TRAIN_GRAD_ACC": os.environ.get("TRAIN_GRAD_ACC", "4"),
                    "TRAIN_WEIGHT_DECAY": os.environ.get("TRAIN_WEIGHT_DECAY", "0.01"),
                    "TRAIN_EPOCHS": os.environ.get("TRAIN_EPOCHS", "10"),
                    "GRADIENT_CHECKPOINTING": os.environ.get("GRADIENT_CHECKPOINTING", "true"),
                    "ORACLE_ON_CPU": os.environ.get("ORACLE_ON_CPU", "false"),
                    "LORA_R": os.environ.get("LORA_R", "16"),
                    "LORA_ALPHA": os.environ.get("LORA_ALPHA", "32"),
                    "LORA_DROPOUT": os.environ.get("LORA_DROPOUT", "0.05"),
                    "TOFU_EVAL_BATCH_SIZE": os.environ.get("TOFU_EVAL_BATCH_SIZE", "4"),
                    "TOFU_CONV_TEMPLATE_STYLE": os.environ.get("TOFU_CONV_TEMPLATE_STYLE", "default"),
                    "OFFICIAL_ULD_MODEL_UTILS": os.environ.get("OFFICIAL_ULD_MODEL_UTILS", "1"),
                    "DISABLE_CONTRAST_CACHE": os.environ.get("DISABLE_CONTRAST_CACHE", "0"),
                    "CONTRAST_BATCH_FALLBACK": os.environ.get("CONTRAST_BATCH_FALLBACK", "0"),
                    "CONTRAST_STRICT_TOP_MASK": os.environ.get("CONTRAST_STRICT_TOP_MASK", "1"),
                })
            else:
                # Current unified-paper setting after the 2026-05-03 retain sweep:
                # ULD is strongest and most stable with retain_num=400 on ToFU.
                tofu_retain_num = "400"
                env.update({
                    "REPRO_CONDA_ENV": repro_env,
                    "PYTHON": os.environ.get("PYTHON", DEFAULT_PYTHON),
                    "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", str(seed)),
                    "TRAIN_EPOCHS": os.environ.get("TRAIN_EPOCHS", "10"),
                    "TRAIN_MAX_STEPS": os.environ.get("TRAIN_MAX_STEPS", "180"),
                    "SAVE_STEPS_OVERRIDE": os.environ.get("SAVE_STEPS_OVERRIDE", "180"),
                    "FORCE_SAVE_FINAL_CHECKPOINT": os.environ.get("FORCE_SAVE_FINAL_CHECKPOINT", "1"),
                    "USE_MODEL_PARALLEL": os.environ.get("USE_MODEL_PARALLEL", "0"),
                    "MP_REQUIRE_4GPU": os.environ.get("MP_REQUIRE_4GPU", "0"),
                    "MP_DEVICE_MAP": os.environ.get("MP_DEVICE_MAP", "balanced"),
                    "MP_DTYPE": os.environ.get("MP_DTYPE", "float16"),
                    "TRAIN_STRATEGY": os.environ.get("TRAIN_STRATEGY", "none"),
                    "TRAIN_SPLIT_OVERRIDE": os.environ.get("TRAIN_SPLIT_OVERRIDE", split),
                    "TRAIN_RETAIN_NUM": os.environ.get("TRAIN_RETAIN_NUM", tofu_retain_num),
                    "GRAYBOX_ULD_LOSS": os.environ.get("GRAYBOX_ULD_LOSS", "gd+uniform"),
                    "GRAYBOX_ULD_DATA_MODE": os.environ.get("GRAYBOX_ULD_DATA_MODE", "forget_retain"),
                    "GRAYBOX_ULD_TRAIN_WEIGHT": os.environ.get("GRAYBOX_ULD_TRAIN_WEIGHT", "-1.0"),
                    "GRAYBOX_ULD_TRAIN_TOP_LOGIT_FILTER": os.environ.get("GRAYBOX_ULD_TRAIN_TOP_LOGIT_FILTER", "0.1"),
                    "GRAYBOX_ULD_EVAL_WEIGHT": os.environ.get("GRAYBOX_ULD_EVAL_WEIGHT", "-0.8"),
                    "GRAYBOX_ULD_EVAL_TOP_LOGIT_FILTER": os.environ.get("GRAYBOX_ULD_EVAL_TOP_LOGIT_FILTER", "0.01"),
                    "TOFU_ULD_EVAL_DEVICES": os.environ.get("TOFU_ULD_EVAL_DEVICES", f"cuda:{str(gpus).split(',')[0]}"),
                    "TRAIN_LR": os.environ.get("TRAIN_LR", "1.5e-4"),
                    "TRAIN_BATCH_SIZE": os.environ.get("TRAIN_BATCH_SIZE", "4"),
                    "TRAIN_GRAD_ACC": os.environ.get("TRAIN_GRAD_ACC", "2"),
                    "TRAIN_WEIGHT_DECAY": os.environ.get("TRAIN_WEIGHT_DECAY", "0"),
                    "FORGET_WEIGHT": os.environ.get("FORGET_WEIGHT", "4"),
                    "RETAIN_WEIGHT": os.environ.get("RETAIN_WEIGHT", "1"),
                    "GRADIENT_CHECKPOINTING": os.environ.get("GRADIENT_CHECKPOINTING", "true"),
                    "ORACLE_ON_CPU": os.environ.get("ORACLE_ON_CPU", "false"),
                    "LORA_R": os.environ.get("LORA_R", "32"),
                    "LORA_ALPHA": os.environ.get("LORA_ALPHA", "64"),
                    "LORA_DROPOUT": os.environ.get("LORA_DROPOUT", "0.05"),
                    "TOFU_EVAL_BATCH_SIZE": os.environ.get("TOFU_EVAL_BATCH_SIZE", "4"),
                    "TOFU_CONV_TEMPLATE_STYLE": os.environ.get("TOFU_CONV_TEMPLATE_STYLE", "default"),
                    "DISABLE_CONTRAST_CACHE": os.environ.get("DISABLE_CONTRAST_CACHE", "0"),
                    "CONTRAST_BATCH_FALLBACK": os.environ.get("CONTRAST_BATCH_FALLBACK", "0"),
                    "CONTRAST_STRICT_TOP_MASK": os.environ.get("CONTRAST_STRICT_TOP_MASK", "1"),
                })
        if family == "graybox" and method == "uld" and dataset == "wmdp":
            env.setdefault("RUN_SUFFIX", _early_suffix_with_append("wmdp_graybox_unified_lr2e4_b2g4_20260514a"))
            env.update({
                "USE_MODEL_PARALLEL": os.environ.get("USE_MODEL_PARALLEL", "1"),
                "MP_REQUIRE_4GPU": os.environ.get("MP_REQUIRE_4GPU", "1"),
                "MP_DEVICE_MAP": os.environ.get("MP_DEVICE_MAP", "balanced"),
                "MP_DTYPE": os.environ.get("MP_DTYPE", "float16"),
                "TRAIN_STRATEGY": os.environ.get("TRAIN_STRATEGY", "none"),
                "TRAIN_LR": os.environ.get("TRAIN_LR", "2e-4"),
                "TRAIN_BATCH_SIZE": os.environ.get("TRAIN_BATCH_SIZE", "2"),
                "TRAIN_GRAD_ACC": os.environ.get("TRAIN_GRAD_ACC", "4"),
                "TRAIN_WEIGHT_DECAY": os.environ.get("TRAIN_WEIGHT_DECAY", "0.01"),
                "TRAIN_MAX_STEPS": os.environ.get("TRAIN_MAX_STEPS", "150"),
                "SAVE_STEPS_OVERRIDE": os.environ.get("SAVE_STEPS_OVERRIDE", "150"),
                "TRAIN_EPOCHS": os.environ.get("TRAIN_EPOCHS", "2"),
                "GRADIENT_CHECKPOINTING": os.environ.get("GRADIENT_CHECKPOINTING", "true"),
                "ORACLE_ON_CPU": os.environ.get("ORACLE_ON_CPU", "false"),
                "WMDP_MAX_FORGET": os.environ.get("WMDP_MAX_FORGET", "600"),
                "WMDP_RETAIN_NUM": os.environ.get("WMDP_RETAIN_NUM", "1200"),
                "FORGET_WEIGHT": os.environ.get("FORGET_WEIGHT", "1"),
                "RETAIN_WEIGHT": os.environ.get("RETAIN_WEIGHT", "4"),
                "GRAYBOX_ULD_LOSS": os.environ.get("GRAYBOX_ULD_LOSS", "gd+uniform"),
                "GRAYBOX_ULD_DATA_MODE": os.environ.get("GRAYBOX_ULD_DATA_MODE", "forget_retain"),
                "GRAYBOX_ULD_TRAIN_WEIGHT": os.environ.get("GRAYBOX_ULD_TRAIN_WEIGHT", "-0.75"),
                "GRAYBOX_ULD_TRAIN_TOP_LOGIT_FILTER": os.environ.get("GRAYBOX_ULD_TRAIN_TOP_LOGIT_FILTER", "0.01"),
                "LORA_R": os.environ.get("LORA_R", "32"),
                "LORA_ALPHA": os.environ.get("LORA_ALPHA", "64"),
                "LORA_DROPOUT": os.environ.get("LORA_DROPOUT", "0.05"),
            })
        if family == "graybox" and method == "offset":
            tofu_retain_num = None
            if dataset == "tofu":
                tofu_retain_num = "400"
            offset_env = {
                "USE_MODEL_PARALLEL": os.environ.get("USE_MODEL_PARALLEL", "1"),
                "MP_REQUIRE_4GPU": os.environ.get("MP_REQUIRE_4GPU", "1"),
                "MP_DEVICE_MAP": os.environ.get("MP_DEVICE_MAP", "balanced"),
                "MP_DTYPE": os.environ.get("MP_DTYPE", "float16"),
                "TRAIN_BATCH_SIZE": os.environ.get("TRAIN_BATCH_SIZE", "1"),
                "TRAIN_GRAD_ACC": os.environ.get("TRAIN_GRAD_ACC", "1"),
                "TRAIN_LR": os.environ.get("TRAIN_LR", "1e-5"),
                "TRAIN_WEIGHT_DECAY": os.environ.get("TRAIN_WEIGHT_DECAY", "0.01"),
                "TRAIN_EPOCHS": os.environ.get("TRAIN_EPOCHS", "10"),
                "LORA_R": os.environ.get("LORA_R", "32"),
                "LORA_ALPHA": os.environ.get("LORA_ALPHA", "64"),
                "LORA_DROPOUT": os.environ.get("LORA_DROPOUT", "0.05"),
                "GRADIENT_CHECKPOINTING": os.environ.get("GRADIENT_CHECKPOINTING", "false"),
                "ORACLE_ON_CPU": os.environ.get("ORACLE_ON_CPU", "false"),
                "ORACLE_DEVICE": os.environ.get("ORACLE_DEVICE", "cuda:3"),
                "OFFSET_BASE_DEVICE": os.environ.get("OFFSET_BASE_DEVICE", "cuda:0"),
                "OFFSET_BASE_ASSIST_DEVICE": os.environ.get("OFFSET_BASE_ASSIST_DEVICE", "cuda:1"),
                "OFFSET_ASSIST_DEVICE": os.environ.get("OFFSET_ASSIST_DEVICE", "cuda:2"),
                "GRAYBOX_OFFSET_DATA_MODE": os.environ.get("GRAYBOX_OFFSET_DATA_MODE", "forget_retain"),
                "FORCE_SAVE_FINAL_CHECKPOINT": os.environ.get("FORCE_SAVE_FINAL_CHECKPOINT", "1"),
            }
            if dataset == "tofu":
                if profile == "official":
                    offset_env.update({
                        "TRAIN_SPLIT_OVERRIDE": os.environ.get("TRAIN_SPLIT_OVERRIDE", f"{split}_perturbed"),
                        "TRAIN_LR": os.environ.get("TRAIN_LR", "1e-5"),
                        "TRAIN_BATCH_SIZE": os.environ.get("TRAIN_BATCH_SIZE", "2"),
                        # Official script uses 2-way DDP: 2 * 8 grad * 2 ranks = 32.
                        # Model-parallel single-process training needs grad_acc=16
                        # to keep the same effective batch.
                        "TRAIN_GRAD_ACC": os.environ.get("TRAIN_GRAD_ACC", "16"),
                        "TRAIN_WEIGHT_DECAY": os.environ.get("TRAIN_WEIGHT_DECAY", "0.01"),
                        "TRAIN_EPOCHS": os.environ.get("TRAIN_EPOCHS", "10"),
                        "TRAIN_MAX_STEPS": os.environ.get("TRAIN_MAX_STEPS", "none"),
                        "SAVE_STEPS_OVERRIDE": os.environ.get("SAVE_STEPS_OVERRIDE", "none"),
                        "FORGET_WEIGHT": os.environ.get("FORGET_WEIGHT", "1"),
                        "RETAIN_WEIGHT": os.environ.get("RETAIN_WEIGHT", "1"),
                        "GRAYBOX_OFFSET_LOSS": os.environ.get("GRAYBOX_OFFSET_LOSS", "npo+kl"),
                        "GRAYBOX_OFFSET_DATA_MODE": os.environ.get("GRAYBOX_OFFSET_DATA_MODE", "forget_retain"),
                        "GRAYBOX_OFFSET_WEIGHT": os.environ.get("GRAYBOX_OFFSET_WEIGHT", "1.0"),
                        "LORA_R": os.environ.get("LORA_R", "0"),
                        "LORA_ALPHA": os.environ.get("LORA_ALPHA", "32"),
                        "LORA_DROPOUT": os.environ.get("LORA_DROPOUT", "0.05"),
                        "TOFU_OFFSET_EVAL_DEVICES": os.environ.get("TOFU_OFFSET_EVAL_DEVICES", "cuda:0|cuda:1|cuda:2"),
                        "TOFU_EVAL_BATCH_SIZE": os.environ.get("TOFU_EVAL_BATCH_SIZE", "1"),
                        "TOFU_CONV_TEMPLATE_STYLE": os.environ.get("TOFU_CONV_TEMPLATE_STYLE", "default"),
                    })
                else:
                    offset_env.update({
                        "TRAIN_SPLIT_OVERRIDE": os.environ.get("TRAIN_SPLIT_OVERRIDE", split),
                        "TRAIN_RETAIN_NUM": os.environ.get("TRAIN_RETAIN_NUM", tofu_retain_num),
                        "TRAIN_LR": os.environ.get("TRAIN_LR", "1.5e-4"),
                        "TRAIN_BATCH_SIZE": os.environ.get("TRAIN_BATCH_SIZE", "4"),
                        "TRAIN_GRAD_ACC": os.environ.get("TRAIN_GRAD_ACC", "2"),
                        "TRAIN_WEIGHT_DECAY": os.environ.get("TRAIN_WEIGHT_DECAY", "0"),
                        "TRAIN_EPOCHS": os.environ.get("TRAIN_EPOCHS", "10"),
                        "TRAIN_MAX_STEPS": os.environ.get("TRAIN_MAX_STEPS", "180"),
                        "SAVE_STEPS_OVERRIDE": os.environ.get("SAVE_STEPS_OVERRIDE", "180"),
                        "FORGET_WEIGHT": os.environ.get("FORGET_WEIGHT", "4"),
                        "RETAIN_WEIGHT": os.environ.get("RETAIN_WEIGHT", "1"),
                        "GRAYBOX_OFFSET_LOSS": os.environ.get("GRAYBOX_OFFSET_LOSS", "npo+kl"),
                        "GRAYBOX_OFFSET_WEIGHT": os.environ.get("GRAYBOX_OFFSET_WEIGHT", "96"),
                        "TOFU_OFFSET_EVAL_DEVICES": os.environ.get("TOFU_OFFSET_EVAL_DEVICES", "cuda:0|cuda:1|cuda:2"),
                        "TOFU_ULD_EVAL_DEVICES": os.environ.get("TOFU_ULD_EVAL_DEVICES", "cuda:0|cuda:1"),
                        "TOFU_EVAL_BATCH_SIZE": os.environ.get("TOFU_EVAL_BATCH_SIZE", "1"),
                        "TOFU_CONV_TEMPLATE_STYLE": os.environ.get("TOFU_CONV_TEMPLATE_STYLE", "default"),
                    })
            else:
                offset_env["RUN_SUFFIX"] = os.environ.get(
                    "RUN_SUFFIX",
                    _early_suffix_with_append("wmdp_graybox_unified_lr2e4_b2g4_20260514a"),
                )
                offset_env.update({
                    "TRAIN_LR": os.environ.get("TRAIN_LR", "2e-4"),
                    "TRAIN_BATCH_SIZE": os.environ.get("TRAIN_BATCH_SIZE", "2"),
                    "TRAIN_GRAD_ACC": os.environ.get("TRAIN_GRAD_ACC", "4"),
                    "TRAIN_WEIGHT_DECAY": os.environ.get("TRAIN_WEIGHT_DECAY", "0.01"),
                    "TRAIN_EPOCHS": os.environ.get("TRAIN_EPOCHS", "2"),
                    "TRAIN_MAX_STEPS": os.environ.get("TRAIN_MAX_STEPS", "150"),
                    "SAVE_STEPS_OVERRIDE": os.environ.get("SAVE_STEPS_OVERRIDE", "150"),
                    "WMDP_MAX_FORGET": os.environ.get("WMDP_MAX_FORGET", "600"),
                    "WMDP_RETAIN_NUM": os.environ.get("WMDP_RETAIN_NUM", "1200"),
                    "FORGET_WEIGHT": os.environ.get("FORGET_WEIGHT", "1"),
                    "RETAIN_WEIGHT": os.environ.get("RETAIN_WEIGHT", "4"),
                    "GRAYBOX_OFFSET_LOSS": os.environ.get("GRAYBOX_OFFSET_LOSS", "gd+kl"),
                    "GRAYBOX_OFFSET_WEIGHT": os.environ.get("GRAYBOX_OFFSET_WEIGHT", "4.0"),
                    "WMDP_EVAL_GPU_SET": os.environ.get("WMDP_EVAL_GPU_SET", "0,1,2"),
                    "WMDP_OFFSET_EVAL_DEVICES": os.environ.get("WMDP_OFFSET_EVAL_DEVICES", "cuda:0,cuda:1,cuda:2"),
                })
            env.update(offset_env)
        default_run_suffix = "official" if profile == "official" else "fixedentry"
        run_suffix_arg = env.get("RUN_SUFFIX", os.environ.get("RUN_SUFFIX", default_run_suffix))
        return _early_run_or_print(
            ["bash", "scripts/internal/run_baseline_commonproto_seed.sh", family, method, dataset, split, seed, run_suffix_arg],
            env=env,
            dry_run=dry_run,
        )

    if family == "blackbox" and dataset == "tofu":
        split = rest[0]
        seed = args.seed or (rest[1] if len(rest) > 1 else "42")
        top_k = args.top_k or paper_cbd_dfb_tofu_top_k(split)
        gpus = args.gpus or "0"
        env = _early_common_env(seed, gpus)
        env.update({"SPLITS": split, "TOFU_TARGET_SPLIT": split, "TOFU10_EXACT_RUN_SUFFIX": os.environ.get("RUN_SUFFIX", f"fixedentry_{split}_topk{top_k}")})
        apply_cbd_dfb_tofu_defaults(env, split)
        env["TOP_K"] = top_k
        return _early_run_or_print(["bash", "scripts/internal/run_cbd_dfb_seed_pipeline.sh", seed, gpus.split(",")[0], env["TOFU10_EXACT_RUN_SUFFIX"], env["UNLEARN_LOSS"]], env=env, dry_run=dry_run)

    if family == "blackbox" and dataset == "wmdp":
        seed = args.seed or (rest[0] if rest else "42")
        top_k = args.top_k or "160"
        gpus = args.gpus or "0"
        env = _early_common_env(seed, gpus)
        run_suffix = os.environ.get("RUN_SUFFIX", paper_wmdp_cbd_dfb_suffix("topk", top_k))
        wmdp_numeric_mode = os.environ.get("WMDP_NUMERIC_MODE", "paper")
        if wmdp_numeric_mode not in {"paper", "deterministic"}:
            raise SystemExit("WMDP_NUMERIC_MODE must be one of: paper, deterministic")
        train_exact_deterministic = os.environ.get(
            "TRAIN_EXACT_DETERMINISTIC",
            "1" if wmdp_numeric_mode == "deterministic" else "0",
        )
        env.update({
            "TOP_K": top_k,
            "WMDP_BLACKBOX_RUN_SUFFIX": run_suffix,
            "WMDP_NUMERIC_MODE": wmdp_numeric_mode,
            "TRAIN_EXACT_DETERMINISTIC": train_exact_deterministic,
            "BASE_MODEL": os.environ.get("BASE_MODEL", DEFAULT_WMDP_MODEL),
            "ASSIST_MODEL": os.environ.get("ASSIST_MODEL", DEFAULT_ASSIST_MODEL),
            "ASSIST_BASE_IF_LORA": os.environ.get("ASSIST_BASE_IF_LORA", DEFAULT_ASSIST_MODEL),
            "BASE_TOKENIZER": os.environ.get("BASE_TOKENIZER", DEFAULT_WMDP_MODEL),
            "ASSIST_TOKENIZER": os.environ.get("ASSIST_TOKENIZER", DEFAULT_ASSIST_MODEL),
            "FORGET_DOMAINS": os.environ.get("FORGET_DOMAINS", "bio,cyber,chem"),
            "TRAIN_SPLIT": os.environ.get("TRAIN_SPLIT", "bio_cyber_chem"),
            "MMLU_TRAIN_FILE": os.environ.get("MMLU_TRAIN_FILE", f"{env['CBD_DATA_ROOT']}/eval-method/wmdp/data/mmlu/all_validation.jsonl"),
            "MAX_FORGET": os.environ.get("MAX_FORGET", "600"),
            "MAX_RETAIN": os.environ.get("MAX_RETAIN", "1200"),
            "TRAIN_RETAIN_NUM": os.environ.get("TRAIN_RETAIN_NUM", os.environ.get("MAX_RETAIN", "1200")),
            "TRAIN_MAX_STEPS": os.environ.get("TRAIN_MAX_STEPS", "150"),
            "SAVE_STEPS_OVERRIDE": os.environ.get("SAVE_STEPS_OVERRIDE", os.environ.get("TRAIN_MAX_STEPS", "150")),
            "TRAIN_LR": os.environ.get("TRAIN_LR", "2e-4"),
            "TRAIN_BATCH_SIZE": os.environ.get("TRAIN_BATCH_SIZE", "2"),
            "TRAIN_GRAD_ACC": os.environ.get("TRAIN_GRAD_ACC", "4"),
            "TRAIN_WEIGHT_DECAY": os.environ.get("TRAIN_WEIGHT_DECAY", "0.01"),
            "CBD_DFB_PROJECT_FORGET_ONLY": os.environ.get("CBD_DFB_PROJECT_FORGET_ONLY", "true"),
            "LORA_R": os.environ.get("LORA_R", "32"),
            "LORA_ALPHA": os.environ.get("LORA_ALPHA", "64"),
            "LORA_DROPOUT": os.environ.get("LORA_DROPOUT", "0.05"),
            "BASIS_GRAD_STORE_DTYPE": os.environ.get("BASIS_GRAD_STORE_DTYPE", ""),
            "FORGET_WEIGHT": os.environ.get("FORGET_WEIGHT", "1"),
            "RETAIN_WEIGHT": os.environ.get("RETAIN_WEIGHT", "4"),
            "THRESH_OPTIMIZE": os.environ.get("THRESH_OPTIMIZE", "tpr"),
            "THRESH_MAX_FORGET": os.environ.get("THRESH_MAX_FORGET", os.environ.get("MAX_FORGET", "600")),
            "THRESH_MAX_RETAIN": os.environ.get("THRESH_MAX_RETAIN", os.environ.get("MAX_RETAIN", "1200")),
            "THRESH_MAX_FPR": os.environ.get("THRESH_MAX_FPR", "0.215"),
            "SCORE_POS": os.environ.get("SCORE_POS", "prompt_last"),
            "SCORE_LAST_K": os.environ.get("SCORE_LAST_K", "4"),
            "SCORE_LAST_K_REDUCE": os.environ.get("SCORE_LAST_K_REDUCE", "max"),
            "SCORE_K_MODE": os.environ.get("SCORE_K_MODE", "last"),
            "THRESH_TRUNCATE_MODE": os.environ.get("THRESH_TRUNCATE_MODE", "head_tail"),
            "EVAL_TRUNCATE_MODE": os.environ.get("EVAL_TRUNCATE_MODE", "head_tail"),
            "EVAL_SCORE_LAST_K": os.environ.get("EVAL_SCORE_LAST_K", os.environ.get("SCORE_LAST_K", "4")),
            "EVAL_SCORE_LAST_K_REDUCE": os.environ.get("EVAL_SCORE_LAST_K_REDUCE", os.environ.get("SCORE_LAST_K_REDUCE", "max")),
            "EVAL_SCORE_K_MODE": os.environ.get("EVAL_SCORE_K_MODE", os.environ.get("SCORE_K_MODE", "last")),
        })
        if train_exact_deterministic == "1":
            env["CUBLAS_WORKSPACE_CONFIG"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            env["TRAIN_DATALOADER_NUM_WORKERS"] = os.environ.get("TRAIN_DATALOADER_NUM_WORKERS", "0")
        elif os.environ.get("CUBLAS_WORKSPACE_CONFIG"):
            env["CUBLAS_WORKSPACE_CONFIG"] = os.environ["CUBLAS_WORKSPACE_CONFIG"]
        if args.stage == "eval":
            run_tag = f"wmdp_seed{seed}_{run_suffix}"
            ckpt_file = _EARLY_ROOT / "artifacts" / "seed_runs" / run_tag / "ckpt_path.txt"
            threshold_json = _EARLY_ROOT / "artifacts" / "eval_outputs" / "wmdp" / run_tag / "threshold.json"
            if not ckpt_file.exists():
                raise SystemExit(f"cannot eval-only; missing checkpoint pointer: {ckpt_file}")
            ckpt_path = ckpt_file.read_text(encoding="utf-8").strip()
            ckpt_check = Path(ckpt_path)
            if not ckpt_check.is_absolute():
                ckpt_check = _EARLY_ROOT / ckpt_check
            if not ckpt_check.exists():
                raise SystemExit(f"cannot eval-only; checkpoint from {ckpt_file} does not exist: {ckpt_path}")
            if not threshold_json.exists():
                raise SystemExit(f"cannot eval-only; missing threshold json: {threshold_json}")
            cmd = wmdp_route_eval_cmd(env, run_tag, ckpt_path, str(threshold_json.relative_to(_EARLY_ROOT)), seed)
            return _early_run_or_print(cmd, env=env, dry_run=dry_run)
        if args.stage == "train":
            env["SKIP_FINAL_EVAL"] = os.environ.get("SKIP_FINAL_EVAL", "1")
        return _early_run_or_print(["bash", "scripts/internal/run_cbd_dfb_wmdp_seed_pipeline.sh", seed, gpus.split(",")[0], env["WMDP_BLACKBOX_RUN_SUFFIX"], "gd+kl"], env=env, dry_run=dry_run)

    if family == "gpm" and dataset == "tofu":
        split = rest[0]
        seed = args.seed or (rest[1] if len(rest) > 1 else "42")
        gpus = args.gpus or "0"
        env = _early_common_env(seed, gpus)
        env.update({
            "FORGET_SPLIT": split,
            "TRAIN_LR": os.environ.get("TRAIN_LR", "1.5e-4"),
            "TRAIN_BATCH_SIZE": os.environ.get("TRAIN_BATCH_SIZE", "4"),
            "TRAIN_GRAD_ACC": os.environ.get("TRAIN_GRAD_ACC", "2"),
            "TRAIN_MAX_STEPS": os.environ.get("TRAIN_MAX_STEPS", "180"),
            "SAVE_STEPS_OVERRIDE": os.environ.get("SAVE_STEPS_OVERRIDE", "180"),
            "TRAIN_EPOCHS": os.environ.get("TRAIN_EPOCHS", "10"),
            "TRAIN_RETAIN_NUM": os.environ.get("TRAIN_RETAIN_NUM", "400"),
            "FORGET_WEIGHT": os.environ.get("FORGET_WEIGHT", "4"),
            "RETAIN_WEIGHT": os.environ.get("RETAIN_WEIGHT", "1"),
            "TRAIN_WEIGHT_DECAY": os.environ.get("TRAIN_WEIGHT_DECAY", "0"),
            "LORA_R": os.environ.get("LORA_R", "32"),
            "LORA_ALPHA": os.environ.get("LORA_ALPHA", "64"),
            "LORA_DROPOUT": os.environ.get("LORA_DROPOUT", "0.05"),
            "GPM_MAX_SAMPLES": os.environ.get("GPM_MAX_SAMPLES", "2400"),
            "GPM_PROJECT_FORGET_ONLY": os.environ.get("GPM_PROJECT_FORGET_ONLY", "1"),
            "THRESH_MAX_FPR": os.environ.get("THRESH_MAX_FPR", "0.04"),
            "ROUTING_EVAL_BATCH_SIZE": os.environ.get("ROUTING_EVAL_BATCH_SIZE", "4"),
        })
        if split not in {"forget01", "forget05", "forget10"}:
            raise SystemExit(f"unsupported GPM ToFU split: {split}")
        return _early_run_or_print(["bash", "scripts/internal/run_gpm_tofu_table3_seed.sh", seed, gpus.split(",")[0], os.environ.get("RUN_SUFFIX", f"fixedentry_gpm_{split}")], env=env, dry_run=dry_run)

    if family == "gpm" and dataset == "wmdp":
        seed = args.seed or (rest[0] if rest else "42")
        gpus = args.gpus or "0"
        env = _early_common_env(seed, gpus)
        env.update({
            "BASE_MODEL": os.environ.get("BASE_MODEL", DEFAULT_WMDP_MODEL),
            "ASSIST_MODEL": os.environ.get("ASSIST_MODEL", DEFAULT_ASSIST_MODEL),
            "ASSIST_BASE_IF_LORA": os.environ.get("ASSIST_BASE_IF_LORA", DEFAULT_ASSIST_MODEL),
            "BASE_TOKENIZER": os.environ.get("BASE_TOKENIZER", DEFAULT_WMDP_MODEL),
            "ASSIST_TOKENIZER": os.environ.get("ASSIST_TOKENIZER", DEFAULT_ASSIST_MODEL),
            "FORGET_DOMAINS": os.environ.get("FORGET_DOMAINS", "bio,cyber,chem"),
            "TRAIN_SPLIT": os.environ.get("TRAIN_SPLIT", "bio_cyber_chem"),
            "MMLU_TRAIN_FILE": os.environ.get("MMLU_TRAIN_FILE", f"{env['CBD_DATA_ROOT']}/eval-method/wmdp/data/mmlu/all_validation.jsonl"),
            "MAX_FORGET": os.environ.get("MAX_FORGET", "600"),
            "MAX_RETAIN": os.environ.get("MAX_RETAIN", "1200"),
            "TRAIN_RETAIN_NUM": os.environ.get("TRAIN_RETAIN_NUM", os.environ.get("MAX_RETAIN", "1200")),
            "TRAIN_MAX_STEPS": os.environ.get("TRAIN_MAX_STEPS", "150"),
            "SAVE_STEPS_OVERRIDE": os.environ.get("SAVE_STEPS_OVERRIDE", os.environ.get("TRAIN_MAX_STEPS", "150")),
            "TRAIN_LR": os.environ.get("TRAIN_LR", "2e-4"),
            "TRAIN_BATCH_SIZE": os.environ.get("TRAIN_BATCH_SIZE", "2"),
            "TRAIN_GRAD_ACC": os.environ.get("TRAIN_GRAD_ACC", "4"),
            "TRAIN_WEIGHT_DECAY": os.environ.get("TRAIN_WEIGHT_DECAY", "0.01"),
            "FORGET_WEIGHT": os.environ.get("FORGET_WEIGHT", "1"),
            "RETAIN_WEIGHT": os.environ.get("RETAIN_WEIGHT", "4"),
            "LORA_R": os.environ.get("LORA_R", "32"),
            "LORA_ALPHA": os.environ.get("LORA_ALPHA", "64"),
            "LORA_DROPOUT": os.environ.get("LORA_DROPOUT", "0.05"),
            "GPM_MAX_SAMPLES": os.environ.get("GPM_MAX_SAMPLES", "1200"),
            "GPM_PROJECT_FORGET_ONLY": os.environ.get("GPM_PROJECT_FORGET_ONLY", "1"),
            "THRESH_OPTIMIZE": os.environ.get("THRESH_OPTIMIZE", "tpr"),
            "THRESH_MAX_FORGET": os.environ.get("THRESH_MAX_FORGET", os.environ.get("MAX_FORGET", "600")),
            "THRESH_MAX_RETAIN": os.environ.get("THRESH_MAX_RETAIN", os.environ.get("MAX_RETAIN", "1200")),
            "THRESH_MAX_FPR": os.environ.get("THRESH_MAX_FPR", "0.215"),
            "SCORE_POS": os.environ.get("SCORE_POS", "prompt_last"),
            "SCORE_LAST_K": os.environ.get("SCORE_LAST_K", "4"),
            "SCORE_LAST_K_REDUCE": os.environ.get("SCORE_LAST_K_REDUCE", "max"),
            "SCORE_K_MODE": os.environ.get("SCORE_K_MODE", "last"),
            "THRESH_TRUNCATE_MODE": os.environ.get("THRESH_TRUNCATE_MODE", "head_tail"),
            "EVAL_TRUNCATE_MODE": os.environ.get("EVAL_TRUNCATE_MODE", "head_tail"),
            "EVAL_SCORE_LAST_K": os.environ.get("EVAL_SCORE_LAST_K", os.environ.get("SCORE_LAST_K", "4")),
            "EVAL_SCORE_LAST_K_REDUCE": os.environ.get("EVAL_SCORE_LAST_K_REDUCE", os.environ.get("SCORE_LAST_K_REDUCE", "max")),
            "EVAL_SCORE_K_MODE": os.environ.get("EVAL_SCORE_K_MODE", os.environ.get("SCORE_K_MODE", "last")),
        })
        return _early_run_or_print(["bash", "scripts/internal/run_gpm_wmdp_table3_seed.sh", seed, gpus.split(",")[0], os.environ.get("RUN_SUFFIX", "fixedentry_gpm_wmdp")], env=env, dry_run=dry_run)

    if family == "sweep":
        sweep_kind = dataset
        target = rest[0] if rest else ""
        if not args.values:
            raise SystemExit("--values is required for sweep")
        if sweep_kind == "basis-size":
            sweep_kind = "basis-retain"
        for value in [v.strip() for v in args.values.split(",") if v.strip()]:
            seed = args.seed or "42"
            env = os.environ.copy()
            common_env = _early_common_env(seed, args.gpus or "0")
            env.update(common_env)
            child_python = common_env["PYTHON"]
            if target in tofu_targets:
                env["RUN_SUFFIX"] = f"fixedentry_{sweep_kind}_{target}_{value}"
                cmd = [child_python, __file__, "repro", "blackbox", "tofu", tofu_targets[target], seed, "--gpus", args.gpus or "0"]
                if target == "tofu10":
                    # The B1-B5 ToFU10 paper sweeps used the source-index
                    # table settings: basis forget=300, basis retain=2400,
                    # train retain=2400, and full CBD-DFB projection.
                    # Keep this scoped to `repro sweep` so the main A3
                    # blackbox default can still use its retain-400 row.
                    env["BASIS_MAX_FORGET"] = os.environ.get("BASIS_MAX_FORGET", "300")
                    env["TRAIN_RETAIN_NUM"] = os.environ.get("TRAIN_RETAIN_NUM", "2400")
                    env["CBD_DFB_PROJECT_FORGET_ONLY"] = os.environ.get("CBD_DFB_PROJECT_FORGET_ONLY", "0")
                if sweep_kind == "topk":
                    cmd.extend(["--top-k", value])
                elif sweep_kind == "basis-retain":
                    env["BASIS_MAX_RETAIN"] = value
                elif sweep_kind == "basis-forget":
                    env["BASIS_MAX_FORGET"] = value
                elif sweep_kind == "lora-r":
                    env["LORA_R"] = value
                elif sweep_kind in {"forget-steps", "max-steps"}:
                    env["TRAIN_MAX_STEPS"] = value
                else:
                    raise SystemExit(f"unsupported sweep: {sweep_kind} {target}")
            elif sweep_kind == "topk" and target == "wmdp":
                env["RUN_SUFFIX"] = paper_wmdp_cbd_dfb_suffix(sweep_kind, value)
                cmd = [child_python, __file__, "repro", "blackbox", "wmdp", seed, "--top-k", value, "--gpus", args.gpus or "0"]
            elif target == "wmdp":
                env["RUN_SUFFIX"] = paper_wmdp_cbd_dfb_suffix(sweep_kind, value)
                cmd = [child_python, __file__, "repro", "blackbox", "wmdp", seed, "--gpus", args.gpus or "0"]
                if sweep_kind == "basis-retain":
                    env["BASIS_MAX_RETAIN"] = value
                elif sweep_kind == "basis-forget":
                    env["BASIS_MAX_FORGET"] = value
                elif sweep_kind == "lora-r":
                    env["LORA_R"] = value
                elif sweep_kind in {"forget-steps", "max-steps"}:
                    env["TRAIN_MAX_STEPS"] = value
                else:
                    raise SystemExit(f"unsupported sweep: {sweep_kind} {target}")
            else:
                raise SystemExit(f"unsupported sweep: {sweep_kind} {target}")
            if args.stage != "both":
                cmd.extend(["--stage", args.stage])
            rc = _early_run_or_print(cmd, env=env, dry_run=dry_run)
            if rc != 0:
                return rc
        return 0

    raise SystemExit(f"unsupported repro command: {' '.join(argv)}")


if len(sys.argv) >= 2 and sys.argv[1] == "repro":
    raise SystemExit(_early_dispatch_repro(sys.argv[2:]))

import hydra
import torch
import random
import numpy as np
import json
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import inspect
import math

from uld.utils import init_script, create_log_dir, NameTimer
from uld.data import create_datamod

from uld.model import TRAIN_INIT_FUNCS
from uld.model.forget_losses import create_unlearn_loss, loss_requries_oracle
from uld.model.utils import get_dtype
from uld.hfutil import ForgetTrainer, SimpleProfileCallback
os.environ['TOKENIZERS_PARALLELISM'] = 'False'

from uld.hfutil.gmp_trainer import GPMForgetTrainer
from uld.hfutil.cbd_dfb_trainer import CBDDFBForgetTrainer


def _display_configs(configs):
    data = OmegaConf.to_container(configs, resolve=False)
    trainer = data.get("trainer", {})
    try:
        max_steps = int(trainer.get("max_steps")) if trainer.get("max_steps") is not None else 0
    except Exception:
        max_steps = 0
    if max_steps > 0:
        trainer.pop("max_epochs", None)
        trainer["schedule_mode"] = "max_steps"
        trainer["schedule_steps"] = max_steps
    return data


def _ensure_valid_padding_idx(model, tokenizer=None, tag="model"):
    try:
        emb = model.get_input_embeddings()
        if emb is None:
            return
        num_embeddings = int(emb.weight.size(0))
        if num_embeddings <= 0:
            print(f"[pad_fix] {tag}: skip invalid num_embeddings={num_embeddings}")
            return
        model_pad = getattr(getattr(model, "config", None), "pad_token_id", None)
        tok_pad = getattr(tokenizer, "pad_token_id", None) if tokenizer is not None else None
        tok_eos = getattr(tokenizer, "eos_token_id", None) if tokenizer is not None else None

        target_pad = model_pad if isinstance(model_pad, int) and model_pad >= 0 else None
        if target_pad is None and isinstance(tok_pad, int) and tok_pad >= 0:
            target_pad = tok_pad
        if target_pad is None and isinstance(tok_eos, int) and tok_eos >= 0:
            target_pad = tok_eos

        if target_pad is not None and target_pad >= num_embeddings:
            resize_to = max(num_embeddings, target_pad + 1)
            model.resize_token_embeddings(resize_to)
            emb = model.get_input_embeddings()
            num_embeddings = int(emb.weight.size(0))
            print(f"[pad_fix] {tag}: resize_token_embeddings -> {num_embeddings}")

        safe_pad = None
        for candidate in (target_pad, tok_pad, tok_eos, 0):
            if isinstance(candidate, int) and 0 <= candidate < num_embeddings:
                safe_pad = candidate
                break
        if safe_pad is None:
            safe_pad = max(num_embeddings - 1, 0)

        if hasattr(model, "config") and getattr(model.config, "pad_token_id", None) != safe_pad:
            model.config.pad_token_id = safe_pad

        emb_pad = getattr(emb, "padding_idx", None)
        if emb_pad is None or emb_pad < 0 or emb_pad >= num_embeddings:
            emb.padding_idx = safe_pad
            print(f"[pad_fix] {tag}: set embedding.padding_idx={safe_pad}")
    except Exception as exc:
        print(f"[pad_fix] {tag}: skip ({exc})")


def _sanitize_resume_trainer_state(resume_from_checkpoint: str):
    if not resume_from_checkpoint:
        return resume_from_checkpoint
    state_path = Path(resume_from_checkpoint) / transformers.trainer.TRAINER_STATE_NAME
    if not state_path.exists():
        return resume_from_checkpoint
    try:
        with state_path.open("r", encoding="utf-8") as f:
            raw_state = json.load(f)
    except Exception as exc:
        print(f"[resume] skip trainer_state sanitize ({exc})")
        return resume_from_checkpoint

    allowed = set(inspect.signature(transformers.trainer_callback.TrainerState.__init__).parameters.keys())
    filtered_state = {k: v for k, v in raw_state.items() if k in allowed}
    dropped = sorted(set(raw_state.keys()) - set(filtered_state.keys()))
    if not dropped:
        return resume_from_checkpoint

    backup_path = state_path.with_suffix(state_path.suffix + ".bak")
    try:
        if not backup_path.exists():
            backup_path.write_text(json.dumps(raw_state, ensure_ascii=False, indent=2), encoding="utf-8")
        state_path.write_text(json.dumps(filtered_state, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[resume] sanitized trainer_state.json, dropped keys: {dropped}")
    except Exception as exc:
        print(f"[resume] failed to sanitize trainer_state.json ({exc})")
    return resume_from_checkpoint


def _load_oracle_model_unsharded(model_path, torch_dtype=torch.bfloat16):
    ds_obj = None
    ds_module = None
    try:
        from transformers.integrations import deepspeed as ds_module  # type: ignore
        ds_ref = getattr(ds_module, "_hf_deepspeed_config_weak_ref", None)
        ds_obj = ds_ref() if ds_ref is not None else None
        if ds_obj is not None and hasattr(ds_module, "unset_hf_deepspeed_config"):
            ds_module.unset_hf_deepspeed_config()
    except Exception:
        ds_obj = None
        ds_module = None

    try:
        try:
            return AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch_dtype,
                use_flash_attention_2=False, trust_remote_code=True,
            )
        except TypeError:
            return AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
    finally:
        try:
            if ds_obj is not None and ds_module is not None and hasattr(ds_module, "set_hf_deepspeed_config"):
                ds_module.set_hf_deepspeed_config(ds_obj)
        except Exception:
            pass


@hydra.main(version_base=None, config_path="../configs", config_name="tune_config")
def main(configs):
    local_rank = 0
    exact_deterministic = os.environ.get("TRAIN_EXACT_DETERMINISTIC", "0") == "1"

    if exact_deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as exc:
            print(f"[deterministic] torch.use_deterministic_algorithms skipped: {exc}")
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("highest")
        except Exception as exc:
            print(f"[deterministic] tf32 flags skipped: {exc}")
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception as exc:
            print(f"[deterministic] cudnn flags skipped: {exc}")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        print("[deterministic] TRAIN_EXACT_DETERMINISTIC=1")

    # 检测是否使用四卡模型并行（白盒方法）
    use_model_parallel = os.environ.get("USE_MODEL_PARALLEL", "0") == "1"

    if use_model_parallel:
        # 四卡模型并行模式
        gpu_count = torch.cuda.device_count()
        require_4gpu = os.environ.get("MP_REQUIRE_4GPU", "1") == "1"
        if require_4gpu and gpu_count < 4:
            raise RuntimeError(f"[MP] need 4 visible gpus, got {gpu_count}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
        if gpu_count <= 0:
            raise RuntimeError("[MP] no cuda device visible")
        # 强制 num_devices=1，因为四卡模型并行是一个逻辑设备
        num_devices = 1
        device_map_strategy = os.environ.get("MP_DEVICE_MAP", "balanced")
        print(f"[MP] Using 4-GPU model parallel mode")
        print(f"[MP] num_devices forced to 1 (logical device)")
        print(f"[MP] device_map strategy: {device_map_strategy}")
    else:
        # 单卡或DDP模式
        num_devices = int(os.environ.get('WORLD_SIZE', 1))
        device_map_strategy = None
        if os.environ.get('LOCAL_RANK') is not None:
            local_rank = int(os.environ.get('LOCAL_RANK', '0'))
            device_map = {'': local_rank}

    # ! Setup Logger
    BASELOGDIR = configs.BASELOGDIR
    output_dir = HydraConfig.get().runtime.output_dir
    configs.base_logdir = os.path.join(output_dir, "logs")
    LOGGER = init_script(configs)
    LOGGER.info("Config", configs=_display_configs(configs))
    LOGGER.info(f"num_devices: {num_devices}")

    OmegaConf.set_struct(configs, False)  # Disable struct mode temporarily
    all_choices = OmegaConf.to_container(HydraConfig.get().runtime.choices)
    configs.name = "|".join([
        "dataset:" + all_choices.get('data'),
        "loss:" + all_choices.get('unlearn_loss'), 
        "model:" + all_choices.get('model'),
        "datamode:" + all_choices.get('data_mode'), 
    ])
    print("RunName", configs.name)
    OmegaConf.set_struct(configs, True)

    now, nowname, logdir, ckptdir, cfgdir = create_log_dir(configs)
    os.makedirs(logdir, exist_ok=True)
    
    #! setup dataset
    model_config = configs.model
    tokenizer = AutoTokenizer.from_pretrained(model_config.tokenizer_path)
    tokenizer.padding_side = "right"
    if "mistral" in model_config.model_name.lower():
        tokenizer.padding_side = "left" #! no idea why this is needed for mistral
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data_module = create_datamod(
        dataset_config=configs.data.dataset,
        conv_template_config=configs.data.conv_template,
        data_mode_config=configs.data_mode,
        tokenizer=tokenizer,
    )
    data_module.prepare_data()
    data_module.setup('fit')

    trainer_config = configs.get("trainer", OmegaConf.create())    
    batch_size = configs.trainer.batch_size
    train_set = data_module.train_set()
    val_set = data_module.val_set()

    # NOTE: `data_module.train_dataloader()` uses its own default batch_size; the Trainer dataloader uses
    # `per_device_train_batch_size`. Compute steps from the dataset length to avoid under-training.
    train_data_size = len(train_set)
    global_batch_size = batch_size * num_devices
    num_batches_per_epoch = math.ceil(train_data_size / max(global_batch_size, 1))
    num_update_steps_per_epoch = math.ceil(num_batches_per_epoch / max(trainer_config.gradient_accumulation_steps, 1))
    num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
    epoch_budget = num_update_steps_per_epoch * trainer_config.max_epochs
    max_steps_override = trainer_config.get("max_steps", None)
    if max_steps_override is not None:
        try:
            max_steps_override = int(max_steps_override)
        except Exception:
            max_steps_override = None
    schedule_mode = "max_epochs"
    num_training_steps = epoch_budget
    if max_steps_override is not None and max_steps_override > 0:
        schedule_mode = "max_steps"
        num_training_steps = max_steps_override
    effective_save_eval_steps = num_update_steps_per_epoch
    if use_model_parallel:
        effective_save_eval_steps = min(num_update_steps_per_epoch, num_training_steps)
    effective_save_eval_steps = max(effective_save_eval_steps, 1)
    save_steps_override_env = os.environ.get("SAVE_STEPS_OVERRIDE")
    eval_steps_override_env = os.environ.get("EVAL_STEPS_OVERRIDE")
    disable_intermediate_saves = False
    if save_steps_override_env is not None and str(save_steps_override_env).strip().lower() in {
        "none",
        "no",
        "off",
        "false",
        "0",
    }:
        disable_intermediate_saves = True
        save_steps_override = None
    else:
        try:
            save_steps_override = int(save_steps_override_env) if save_steps_override_env else None
        except Exception:
            save_steps_override = None
    try:
        eval_steps_override = int(eval_steps_override_env) if eval_steps_override_env else None
    except Exception:
        eval_steps_override = None
    if disable_intermediate_saves:
        effective_save_steps = max(num_training_steps, 1)
    elif save_steps_override is not None and save_steps_override > 0:
        effective_save_steps = save_steps_override
    else:
        effective_save_steps = effective_save_eval_steps
    if eval_steps_override is not None and eval_steps_override > 0:
        effective_eval_steps = eval_steps_override
    else:
        effective_eval_steps = effective_save_eval_steps
    print("train_data_size", train_data_size)
    print("global_batch_size", global_batch_size)
    print("num_batches_per_epoch", num_batches_per_epoch)
    print("num_update_steps_per_epoch", num_update_steps_per_epoch)
    print("schedule_mode", schedule_mode)
    if schedule_mode == "max_steps":
        print("configured_max_steps", num_training_steps)
        print("actual_epoch_fraction", round(num_training_steps / num_update_steps_per_epoch, 6))
    else:
        print("configured_max_epochs", trainer_config.max_epochs)
        print("epoch_budget", epoch_budget)
    print("num_training_steps", num_training_steps)
    print("effective_save_eval_steps", effective_save_eval_steps)
    print("effective_save_steps", "disabled" if disable_intermediate_saves else effective_save_steps)
    print("effective_eval_steps", effective_eval_steps)
    #num_update_steps_per_epoch = 500

    #! change checkpoint foler at runtime
    tmpckptdir = ckptdir.split(BASELOGDIR)[-1]
    checkpoint_dir = os.path.join(
        configs.OUTPUTMODELDIR, "/".join(tmpckptdir.split("/")[1:-1]).replace(",", "|").replace("=","_")
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    model_config = configs.get('model')
    is_offset = 'offset' in model_config.get('mode', 'base')

    #! setup trainer
    os.makedirs(logdir, exist_ok=True)
    os.environ["WANDB_PROJECT"] = configs.project
    os.environ["WANDB_DIR"] = logdir
    force_deepspeed = os.environ.get("FORCE_DEEPSPEED", "0") == "1"
    is_deepspeed = force_deepspeed or ('deepspeed' in str(trainer_config.get('strategy', "")))
    if is_deepspeed:
        print("Loading deepspeed")
        deepspeed_configfile = os.environ.get("DEEPSPEED_CONFIG_PATH", "configs/ds_config.json")
    else:
        print("None deepspeed")
        deepspeed_configfile = None

    gradient_checkpointing = bool(configs.get('gradient_checkpointing', False))
    try:
        logging_steps = int(os.environ.get("TRAIN_LOGGING_STEPS", "10"))
    except Exception:
        logging_steps = 10
    logging_steps = max(1, logging_steps)
    train_optim = os.environ.get("TRAIN_OPTIM", "adamw_torch").strip() or "adamw_torch"
    print(f"train_optim={train_optim}")

    # TrainingArguments has breaking changes across transformers versions (e.g. eval_strategy vs evaluation_strategy).
    # Build kwargs and filter by the installed version's signature for robustness.
    ddp_timeout_env = os.environ.get("BASELINE_DDP_TIMEOUT_SEC") or os.environ.get("DDP_TIMEOUT_SEC")
    try:
        ddp_timeout_sec = int(ddp_timeout_env) if ddp_timeout_env is not None else 1800
    except Exception:
        ddp_timeout_sec = 1800

    ddp_find_unused_env = os.environ.get("DDP_FIND_UNUSED_PARAMETERS")
    if ddp_find_unused_env is None:
        ddp_find_unused_parameters = False
    else:
        ddp_find_unused_parameters = str(ddp_find_unused_env).strip().lower() in {"1", "true", "yes", "y", "on"}

    ddp_static_graph_env = os.environ.get("DDP_STATIC_GRAPH")
    if ddp_static_graph_env is None:
        ddp_static_graph = False
    else:
        ddp_static_graph = str(ddp_static_graph_env).strip().lower() in {"1", "true", "yes", "y", "on"}

    dataloader_workers_env = os.environ.get("TRAIN_DATALOADER_NUM_WORKERS", "4")
    try:
        dataloader_num_workers = max(0, int(dataloader_workers_env))
    except Exception:
        dataloader_num_workers = 0
    pin_memory_env = os.environ.get("TRAIN_DATALOADER_PIN_MEMORY", "1")
    dataloader_pin_memory = str(pin_memory_env).strip().lower() in {"1", "true", "yes", "y", "on"}

    # 在模型并行模式下，禁用 bf16 以避免与 accelerate hooks 冲突
    use_bf16 = True
    if use_model_parallel:
        use_bf16 = False
        print("[MP] Disabling bf16 in model parallel mode (conflicts with accelerate hooks)")

    disable_internal_eval = os.environ.get("DISABLE_INTERNAL_EVAL", "0") == "1"
    eval_strategy_value = "no" if disable_internal_eval else "steps"
    print(f"disable_internal_eval={disable_internal_eval}")

    hf_max_steps = num_training_steps if schedule_mode == "max_steps" else -1
    save_strategy_value = "no" if disable_intermediate_saves else "steps"
    training_args_kwargs = dict(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=trainer_config.gradient_accumulation_steps,
        warmup_steps=int(num_training_steps * trainer_config.warmup_ratio),
        learning_rate=trainer_config.learning_rate,
        weight_decay=trainer_config.weight_decay,
        max_steps=hf_max_steps,
        num_train_epochs=trainer_config.max_epochs,
        bf16=use_bf16,
        bf16_full_eval=use_bf16,
        logging_steps=logging_steps,
        logging_dir=logdir,
        output_dir=checkpoint_dir,
        optim=train_optim,
        save_only_model=True,
        save_total_limit=1,
        ddp_find_unused_parameters=ddp_find_unused_parameters,
        ddp_static_graph=ddp_static_graph,
        deepspeed=deepspeed_configfile,
        save_steps=effective_save_steps,
        eval_steps=effective_eval_steps,
        save_strategy=save_strategy_value,
        eval_strategy=eval_strategy_value,
        evaluation_strategy=eval_strategy_value,
        seed=configs.get('seed', 42),
        report_to='none',
        run_name=configs.name,
        remove_unused_columns=False,
        ddp_timeout=ddp_timeout_sec,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_pin_memory=dataloader_pin_memory,
    )
    if dataloader_num_workers > 0:
        training_args_kwargs["dataloader_persistent_workers"] = True
    if gradient_checkpointing:
        training_args_kwargs["gradient_checkpointing"] = True
    ta_params = inspect.signature(transformers.TrainingArguments.__init__).parameters
    training_args_kwargs = {k: v for k, v in training_args_kwargs.items() if k in ta_params}
    training_args = transformers.TrainingArguments(**training_args_kwargs)
    print(f"ddp_find_unused_parameters={ddp_find_unused_parameters}")
    print(f"ddp_static_graph={ddp_static_graph}")
    
    simpleprofilercallback = SimpleProfileCallback(
        logdir, "simpleprofile.txt"
    )

    #! Logging training mode
    # NOTE: For some benchmarks (e.g. WMDP), we must avoid printing raw prompts/questions in logs.
    batch = next(iter(data_module.train_dataloader()))

    data_cfg = configs.get("data", None)
    dataset_cfg = data_cfg.get("dataset", {}) if data_cfg else {}
    dataset_name = str(dataset_cfg.get("name", "") or "")
    dataset_class = str(dataset_cfg.get("class_name", "") or "")
    force_safe_skip = os.environ.get("SAFE_SKIP_TEXT_LOG", "0") == "1"
    safe_skip_text = force_safe_skip or ("wmdp" in dataset_name.lower()) or (dataset_class.lower() == "wmdp")

    sampledatas = {"train_sample_keys": list(batch.keys())}
    if safe_skip_text:
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -1
        sampledatas["safe_log"] = True
        sampledatas["train_sample_lengths"] = (
            (batch["input_ids"][:2] != pad_id).sum(dim=1).detach().cpu().tolist()
        )
        if "prefer_input_ids" in batch:
            sampledatas["prefer_sample_lengths"] = (
                (batch["prefer_input_ids"][:2] != pad_id).sum(dim=1).detach().cpu().tolist()
            )
    else:
        sampledatas["safe_log"] = False
        sampledatas["train_sample"] = tokenizer.batch_decode(batch["input_ids"][:2], skip_special_tokens=True)
        if "prefer_input_ids" in batch:
            sampledatas["prefer_sample"] = tokenizer.batch_decode(batch["prefer_input_ids"][:2], skip_special_tokens=True)

    if "retainlabel" in batch:
        sampledatas["retainlabel"] = batch["retainlabel"].tolist()

    LOGGER.info("Sample data", **sampledatas, shape=batch["input_ids"].shape)

    #! Setup model
    baseoutdir = checkpoint_dir
    model_mode = configs.get('model_mode', None)
    init_func = TRAIN_INIT_FUNCS.get(model_mode.get('mode', 'base'))

    # 如果使用四卡模型并行，传递 device_map
    if use_model_parallel:
        model_mode = dict(model_mode)  # 转换为可修改的字典
        model_mode['device_map'] = device_map_strategy
        print(f"[MP] Passing device_map={device_map_strategy} to model init")

    # Decouple LoRA (frozen) initialization from training randomness for more stable multi-seed runs.
    train_seed = int(configs.get('seed', 42))
    lora_seed = int(configs.get('lora_seed', train_seed))

    # 1) Seed for model init (LoRA-A/B init happens inside init_func)
    random.seed(lora_seed)
    np.random.seed(lora_seed)
    torch.manual_seed(lora_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(lora_seed)

    model = init_func(
        **model_config,
        **model_mode,
        baseoutdir=baseoutdir,
    )
    model_path = model_config.get('model_path')
    model = model.train()

    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        gc_use_reentrant_false = os.environ.get("GC_USE_REENTRANT_FALSE", "0") == "1"
        if gc_use_reentrant_false:
            try:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                print("gradient_checkpointing_kwargs.use_reentrant=False")
            except TypeError:
                model.gradient_checkpointing_enable()
        else:
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        try:
            model.config.use_cache = False
        except Exception:
            pass

    # 2) Reseed for training-time randomness (samplers/dropout/etc.)
    random.seed(train_seed)
    np.random.seed(train_seed)
    torch.manual_seed(train_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_seed)

    #! Setup loss function
    loss_config = configs.get('unlearn_loss')
    loss_function = create_unlearn_loss(loss_config)
    if loss_requries_oracle(loss_config):
        with NameTimer("Load oracle"):
            oracle_dtype = torch.bfloat16
            if use_model_parallel:
                oracle_dtype_name = os.environ.get("MP_DTYPE", "float16")
                resolved_oracle_dtype = get_dtype(oracle_dtype_name)
                if resolved_oracle_dtype is not None:
                    oracle_dtype = resolved_oracle_dtype
                print(f"[MP] Loading oracle with torch_dtype={oracle_dtype}")
            oracle_on_cpu_env = os.environ.get("ORACLE_ON_CPU", "0") == "1"
            oracle_device_env = os.environ.get("ORACLE_DEVICE", "").strip()
            if oracle_device_env and not oracle_on_cpu_env:
                print(f"[oracle] Loading oracle on explicit device={oracle_device_env}")
                oracle_kwargs = {
                    "torch_dtype": oracle_dtype,
                    "device_map": {"": oracle_device_env},
                    "trust_remote_code": True,
                    "attn_implementation": os.environ.get("MODEL_ATTN_IMPL", "sdpa"),
                    "low_cpu_mem_usage": True,
                }
                oracle_model = AutoModelForCausalLM.from_pretrained(model_path, **oracle_kwargs)
                if str(oracle_device_env).startswith("cuda"):
                    from uld.model.utils import _check_pure_gpu_device_map
                    _check_pure_gpu_device_map(oracle_model, "oracle_model")
            elif use_model_parallel and not oracle_on_cpu_env:
                # 四卡模型并行模式，oracle 也用 device_map
                print(f"[MP] Loading oracle with device_map={device_map_strategy}")
                oracle_kwargs = {
                    "torch_dtype": oracle_dtype,
                    "device_map": device_map_strategy,
                    "trust_remote_code": True,
                    "attn_implementation": os.environ.get("MODEL_ATTN_IMPL", "sdpa"),
                    "low_cpu_mem_usage": True,
                }
                oracle_model = AutoModelForCausalLM.from_pretrained(model_path, **oracle_kwargs)
                # 检查纯GPU
                from uld.model.utils import _check_pure_gpu_device_map
                _check_pure_gpu_device_map(oracle_model, "oracle_model")
            elif oracle_on_cpu_env:
                # Oracle 放在 CPU
                print("[MP] Loading oracle on CPU (ORACLE_ON_CPU=1)")
                oracle_model = _load_oracle_model_unsharded(model_path, torch_dtype=oracle_dtype)
                oracle_model = oracle_model.to("cpu")
            else:
                # 单卡模式，oracle 正常加载
                oracle_model = _load_oracle_model_unsharded(model_path, torch_dtype=oracle_dtype)
            _ensure_valid_padding_idx(oracle_model, tokenizer=tokenizer, tag="oracle_model")
            oracle_model.eval()
            oracle_model.requires_grad_(False)
    else:
        oracle_model = None

    requires_equal_sampler = (loss_function.retain_loss_func is not None)
    if os.environ.get("FORCE_DISABLE_EQUAL_SAMPLER", "0") == "1" and requires_equal_sampler:
        LOGGER.info("Disable equal sampler by env FORCE_DISABLE_EQUAL_SAMPLER=1")
        requires_equal_sampler = False
    LOGGER.info("Training with equal sampler: ", requires_equal_sampler=requires_equal_sampler)

    custom_callbacks = [simpleprofilercallback]

    enable_cbd_dfb = configs.get('enable_cbd_dfb', False)
    cbd_dfb_basis_path = configs.get('cbd_dfb_basis_path', None)
    cbd_dfb_eigval_weight = bool(configs.get('cbd_dfb_eigval_weight', False))
    cbd_dfb_trust_region = bool(configs.get('cbd_dfb_trust_region', False))
    cbd_dfb_trust_region_epsilon = float(configs.get('cbd_dfb_trust_region_epsilon', 1e-3))
    cbd_dfb_trust_region_delta = float(configs.get('cbd_dfb_trust_region_delta', 1e-12))
    cbd_dfb_project_forget_only = bool(configs.get('cbd_dfb_project_forget_only', False))
    oracle_on_cpu = bool(configs.get('oracle_on_cpu', False))
    enable_gmp = configs.get('enable_gmp', False)
    gmp_basis_path = configs.get('gmp_basis_path', './gmp_basis/retain99_pca_basis.pkl')
    gmp_project_forget_only = bool(configs.get('gmp_project_forget_only', False))

    if enable_cbd_dfb:
        if not cbd_dfb_basis_path:
            raise ValueError("enable_cbd_dfb=True 但未提供 cbd_dfb_basis_path")
        print(f"🚀 使用 CBD-DFB 训练器，基底路径: {cbd_dfb_basis_path}")
        trainer = CBDDFBForgetTrainer(
            model=model,
            train_loss_function=loss_function,
            oracle_model=oracle_model,
            equal_sampler=requires_equal_sampler,
            is_deepspeed=is_deepspeed,
            train_dataset=train_set,
            eval_dataset=None if disable_internal_eval else val_set,
            seed=configs.get('seed', 42),
            callbacks=custom_callbacks,
            args=training_args,
            is_offset=is_offset,
            cbd_dfb_basis_path=cbd_dfb_basis_path,
            enable_cbd_dfb=True,
            use_eigval_weight=cbd_dfb_eigval_weight,
            trust_region=cbd_dfb_trust_region,
            trust_region_epsilon=cbd_dfb_trust_region_epsilon,
            trust_region_delta=cbd_dfb_trust_region_delta,
            project_forget_only=cbd_dfb_project_forget_only,
            oracle_on_cpu=oracle_on_cpu,
        )
    elif enable_gmp:
        print(f"🚀 使用GPM训练器，基底路径: {gmp_basis_path}")
        trainer = GPMForgetTrainer(
            model=model,
            train_loss_function=loss_function,
            oracle_model=oracle_model,
            equal_sampler=requires_equal_sampler,
            is_deepspeed=is_deepspeed,
            train_dataset=train_set,
            eval_dataset=None if disable_internal_eval else val_set,
            seed=configs.get('seed', 42),
            callbacks=custom_callbacks,
            args=training_args,
            is_offset=is_offset,
            gmp_basis_path=gmp_basis_path,
            enable_gmp=True,
            project_forget_only=gmp_project_forget_only,
            oracle_on_cpu=oracle_on_cpu,
        )
    else:
        print("📝 使用标准ForgetTrainer")

        trainer = ForgetTrainer(
            model=model,
            train_loss_function=loss_function,
            oracle_model=oracle_model,
            equal_sampler=requires_equal_sampler,
            is_deepspeed=is_deepspeed,
            train_dataset=train_set,
            eval_dataset=None if disable_internal_eval else val_set,
            seed=configs.get('seed', 42),
            callbacks=custom_callbacks,
            args=training_args,
            is_offset=is_offset,
            oracle_on_cpu=oracle_on_cpu,
        )
        
    model.config.use_cache = False
    resume_from_checkpoint = os.environ.get("RESUME_FROM_CHECKPOINT", "").strip()
    if not resume_from_checkpoint:
        try:
            resume_from_checkpoint = configs.get("resume_from_checkpoint", "") or ""
        except Exception:
            resume_from_checkpoint = ""
    if resume_from_checkpoint:
        resume_from_checkpoint = _sanitize_resume_trainer_state(resume_from_checkpoint)
        print(f"[train] resume_from_checkpoint={resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        trainer.train()

    force_save_final_checkpoint = os.environ.get("FORCE_SAVE_FINAL_CHECKPOINT", "0") == "1"
    if force_save_final_checkpoint:
        actual_global_step = getattr(getattr(trainer, "state", None), "global_step", None)
        try:
            actual_global_step = int(actual_global_step)
        except Exception:
            actual_global_step = 0
        if actual_global_step <= 0:
            actual_global_step = num_training_steps
        final_ckpt_dir = os.path.join(checkpoint_dir, f"checkpoint-{actual_global_step}")
        if not os.path.isdir(final_ckpt_dir):
            print(f"[train] Saving final model to {final_ckpt_dir}")
            trainer.save_model(final_ckpt_dir)

    if local_rank == 0:
        os.symlink(output_dir, os.path.join(checkpoint_dir, "trainlogdir"))

if __name__ == "__main__":
    cleaned_argv = []
    skip_next = False
    for i, arg in enumerate(sys.argv):
        if skip_next:
            skip_next = False
            continue
        if arg in ("--local_rank", "--local-rank"):
            if i + 1 < len(sys.argv):
                skip_next = True
            continue
        if arg.startswith("--local_rank=") or arg.startswith("--local-rank="):
            continue
        cleaned_argv.append(arg)
    sys.argv = cleaned_argv
    main()
