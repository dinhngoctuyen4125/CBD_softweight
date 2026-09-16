import copy 
import os
from functools import partial

import torch
try:
    import deepspeed
except Exception:
    deepspeed = None
import datasets
from transformers.utils import is_datasets_available
from transformers.trainer_utils import seed_worker
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import Trainer
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.trainer_utils import has_length
from typing import Callable, Dict, Optional
import numpy as np
import inspect

from transformers.training_args import TrainingArguments
from ..data.datamodule import EqualForgetRetainSampler


class ForgetTrainer(Trainer):

    def __init__(self, model, train_loss_function: Callable, is_deepspeed=False, oracle_model=None, equal_sampler=False, seed=42, is_offset=False, oracle_on_cpu=False, **kwargs):
        # Check if using model parallelism before calling super().__init__
        self.use_model_parallel = os.environ.get("USE_MODEL_PARALLEL", "0") == "1"
        self.use_official_trainer_behavior = os.environ.get("OFFICIAL_TRAINER_BEHAVIOR", "0") == "1"

        super(ForgetTrainer, self).__init__(model=model, **kwargs)
        self.train_loss_function = train_loss_function
        self.equal_sampler = equal_sampler
        self.oracle_model = oracle_model
        self.oracle_on_cpu = bool(oracle_on_cpu)
        self.seed = seed
        wrap_oracle_with_deepspeed = os.getenv("WRAP_ORACLE_WITH_DEEPSPEED", "1") == "1"
        if oracle_model is not None and is_deepspeed and wrap_oracle_with_deepspeed and not self.oracle_on_cpu:
            self.oracle_model.requires_grad_(False)
            self.oracle_model = self.e_prepare_deepspeed(oracle_model)
        else:
            if self.oracle_model is not None:
                self.oracle_model.requires_grad_(False)
                if self.use_model_parallel:
                    print("[MP] Keeping oracle_model on its existing device_map")
                elif not self.oracle_on_cpu:
                    self._move_model_to_device(self.oracle_model, self.args.device)

    def _wrap_model(self, model, training=True, dataloader=None):
        """Override to skip wrapping in model parallel mode."""
        if self.use_model_parallel and not self.use_official_trainer_behavior:
            # In model parallel mode, model is already distributed via device_map
            # Don't let Trainer/Accelerator wrap it again
            print(f"[MP] Skipping _wrap_model (use_model_parallel=True)")
            return model
        else:
            # Normal mode: use parent's wrapping
            return super()._wrap_model(model, training, dataloader)

    def _should_log_trainloss(self) -> bool:
        state = getattr(self, "state", None)
        if state is None:
            return False
        step = int(getattr(state, "global_step", 0))
        log_steps = max(1, int(getattr(self.args, "logging_steps", 1) or 1))
        if step != 0 and (step % log_steps) != 0:
            return False
        last_logged = getattr(self, "_last_trainloss_logged_step", None)
        if last_logged == step:
            return False
        self._last_trainloss_logged_step = step
        return True
                
    def _get_train_sampler(self, generator=None) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.equal_sampler:
            print("Using EqualForgetRetainSampler")
            return EqualForgetRetainSampler(self.train_dataset.forget_length, self.train_dataset.retain_length, generator=generator)
        else:
            # Build the sampler.
            return RandomSampler(self.train_dataset, generator=generator)
    
    def get_train_dataloader(self) -> DataLoader:
        """
        Override the original get_train_dataloader function simply for debugging.
        This is identical to the get_train_dataloader function in transformer.Trainer.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }
        
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.state.global_step)
        print(f'Generator........Epoch-{self.state.global_step}')

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):

            if os.environ.get("TRAIN_EXACT_DETERMINISTIC", "0") == "1":
                dataloader_generator = torch.Generator()
                dataloader_generator.manual_seed(self.seed + self.state.global_step)
                dataloader_params["generator"] = dataloader_generator
            # dataloader_params["shuffle"] = True # set shuffle=True with specified generator.
            dataloader_params["sampler"] = self._get_train_sampler(generator=generator)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            # Fix: seed_worker in transformers 4.38.1 only accepts worker_id parameter
            dataloader_params["worker_init_fn"] = seed_worker

        dataloader = DataLoader(train_dataset, **dataloader_params)

        # In model parallel mode (device_map), skip accelerate.prepare() for dataloader
        # because the model is already distributed and accelerate.prepare() can cause device conflicts
        use_model_parallel = os.environ.get("USE_MODEL_PARALLEL", "0") == "1"
        if use_model_parallel:
            if self.use_official_trainer_behavior:
                print("[MP] Using official trainer behavior: prepare dataloader with accelerator")
                return self.accelerator.prepare(dataloader)
            return dataloader
        else:
            return self.accelerator.prepare(dataloader)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        losses = self.train_loss_function(model, inputs, self.oracle_model)
        # print("loss", losses)
        loss = losses['loss']
        forgetloss = losses['forget_loss']
        retainloss = losses['retain_loss']

        #! Notice that these are evaluated on mini-batch instead of total effective batch 
        if self._should_log_trainloss():
            logitems = {
                'trainloss/loss': loss.item(),
                'trainloss/forgetloss': forgetloss.item(),
                'trainloss/retainloss': retainloss.item()
            }
            self.log(logitems)

        return loss

    def prediction_step(self, model, inputs, prediction_loss_only=True, ignore_keys=None):
        import inspect
        signature = inspect.signature(model.forward)
        _signature_columns = list(signature.parameters.keys())
        _signature_columns += list(set(["label", "label_ids"]))
        inputs = {k:v for k, v in inputs.items() if k in _signature_columns}
        labels = inputs['labels']

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            loss = outputs.loss

        if prediction_loss_only:
            return (loss, None, None)
        else:
            if len(logits) == 1:
                logits = logits[0]
            if len(labels) == 1:
                labels = labels[0]
            return (loss, logits, labels)
    
    # Copied from ToFU repo
    def e_prepare_deepspeed(self, model, stage=None):
        if deepspeed is None:
            raise ImportError("deepspeed is required for deepspeed strategy but could not be imported.")
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = copy.deepcopy(deepspeed_plugin.deepspeed_config)
        if stage is None:
            stage = config_kwargs["zero_optimization"]["stage"] 

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and stage == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if stage != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        config_kwargs["optimizer"] = {"type": None}
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        #set the gradients to false for every parameter
        for param in model.parameters():
            param.requires_grad = False
        
        return model
