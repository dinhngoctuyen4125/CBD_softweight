#! This script initializes a small LLM and finetune for unlearning (CBD-DFB)
import os
import sys

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
