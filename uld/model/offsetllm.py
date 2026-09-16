import os
from typing import Optional, List
import torch
from torch.nn import CrossEntropyLoss
from omegaconf import OmegaConf
from transformers import (
    AutoConfig,
    AutoModelForCausalLM, 
    GenerationConfig, 
    PreTrainedModel,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from peft import LoraConfig, PeftModel, get_peft_model

from .peft_util import find_all_linear_names

def _mv(tensor, device):
    if tensor is None:
        return None
    if str(tensor.device) == str(device):
        return tensor
    return tensor.to(device)

def _resolve_offset_devices(device=None, device_map=None, base_device=None, base_assist_device=None, assist_device=None):
    if base_device is not None:
        base_assist_device = base_assist_device or base_device
        assist_device = assist_device or base_assist_device
        return str(base_device), str(base_assist_device), str(assist_device)

    if device_map is not None:
        gpu_count = max(torch.cuda.device_count(), 1)
        base_device = "cuda:0"
        base_assist_device = "cuda:1" if gpu_count > 1 else base_device
        assist_device = "cuda:2" if gpu_count > 2 else base_assist_device
        return base_device, base_assist_device, assist_device

    default_device = str(device or "cuda")
    return default_device, default_device, default_device

#! Unofficial implementation for the paper 'Offset Unlearning For Large Language Model' (https://arxiv.org/pdf/2404.11045)
def create_offset_model(model_path, data_type='bfloat16', **kwargs):
    baseconfig = AutoConfig.from_pretrained(model_path)
    base_device_override = kwargs.pop("base_device", None)
    base_assist_device_override = kwargs.pop("base_assist_device", None)
    assist_device_override = kwargs.pop("assist_device", None)
    base_device, base_assist_device, assist_device = _resolve_offset_devices(
        device=kwargs.get("device"),
        device_map=kwargs.get("device_map"),
        base_device=base_device_override,
        base_assist_device=base_assist_device_override,
        assist_device=assist_device_override,
    )
    model = OffsetAssitedModel(
        baseconfig,
        base_device=base_device,
        base_assist_device=base_assist_device,
        assist_device=assist_device,
        torch_dtype=torch.bfloat16,
        **kwargs,
    )
    if kwargs.get("device_map") is None and (device := kwargs.get("device", None)):
        model = model.to(device=device)
    return model

class OffsetAssitedModel(PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"assist_model.*",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"assist_model.*",
    ]

    def __init__(
        self,
        config,
        base_assist_path,
        new_assist_path=None,
        weight=1.0,
        is_lora=False,
        Lora=OmegaConf.create({"r":0, "alpha": 32, "dropout": 0.05}),
        base_device="cuda",
        base_assist_device="cuda",
        assist_device="cuda",
        **kwargs,
    ):
        tmplora = OmegaConf.to_container(Lora)
        config.Lora = tmplora
        config.base_model_name = config._name_or_path
        config.is_offset = True
        config.base_assist_path = base_assist_path
        config.new_assist_path = new_assist_path
        config.weight = weight
        config.new_assist_path = new_assist_path
        super().__init__(config, **kwargs)
        
        self.vocab_size = config.vocab_size
        self.base_device = base_device
        self.base_assist_device = base_assist_device
        self.assist_device = assist_device
        self.is_parallelizable = True
        self.model_parallel = len({self.base_device, self.base_assist_device, self.assist_device}) > 1
        if self.model_parallel:
            self.hf_device_map = {
                "basellm": self.base_device,
                "base_assist_llm": self.base_assist_device,
                "assist_llm": self.assist_device,
            }

        local_files_only = kwargs.get("local_files_only", False)
        common_kwargs = {
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
            "local_files_only": local_files_only,
        }

        self.basellm = AutoModelForCausalLM.from_pretrained(
            config.base_model_name, use_flash_attention_2=False, **common_kwargs
        ).to(self.base_device)
        self.basellm.eval()
        self.basellm.requires_grad_(False) #! Freeze

        self.base_assist_llm = AutoModelForCausalLM.from_pretrained(
            base_assist_path, use_flash_attention_2=False, **common_kwargs
        ).to(self.base_assist_device)
        self.base_assist_llm.eval()
        self.base_assist_llm.requires_grad_(False) #! Freeze
        
        if new_assist_path is None:
            assist_path = base_assist_path
        else:
            assist_path = new_assist_path

        assist_config_path = base_assist_path if new_assist_path is not None else assist_path
        assist_config = AutoConfig.from_pretrained(
            assist_config_path,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        if new_assist_path is not None and os.path.exists(os.path.join(str(new_assist_path), "adapter_config.json")):
            assist_base = AutoModelForCausalLM.from_pretrained(
                base_assist_path,
                config=assist_config,
                use_flash_attention_2=False,
                **common_kwargs,
            ).to(self.assist_device)
            self.assist_llm = PeftModel.from_pretrained(
                assist_base,
                new_assist_path,
                is_trainable=False,
                local_files_only=local_files_only,
            ).to(self.assist_device)
        else:
            self.assist_llm = AutoModelForCausalLM.from_pretrained(
                assist_path,
                config=assist_config,
                use_flash_attention_2=False,
                **common_kwargs,
            ).to(self.assist_device)
        if Lora.r != 0:
            peftconfig = LoraConfig(
                r=Lora.r,
                lora_alpha=Lora.alpha,
                target_modules=find_all_linear_names(self.assist_llm), 
                lora_dropout=Lora.dropout,
                bias=Lora.bias, 
                task_type="CAUSAL_LM",
            )
            self.assist_llm = get_peft_model(self.assist_llm, peftconfig)

        self.weight = weight
        self.generation_config = GenerationConfig.from_model_config(self.config)
    
    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.basellm.prepare_inputs_for_generation(
            *args, **kwargs
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.LongTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs,
    ):
        if input_ids is None:
            input_ids = inputs
        if input_ids is None:
            raise ValueError("input_ids is required")

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=input_ids.device)

        if max_new_tokens is not None:
            target_len = input_ids.shape[1] + int(max_new_tokens)
        else:
            target_len = int(max_length if max_length is not None else (input_ids.shape[1] + 20))

        if target_len <= input_ids.shape[1]:
            return input_ids

        if pad_token_id is None:
            pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else self.config.eos_token_id
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id

        if isinstance(eos_token_id, int):
            eos_tokens = {int(eos_token_id)}
        elif eos_token_id is None:
            eos_tokens = set()
        else:
            eos_tokens = {int(x) for x in eos_token_id}

        seq = input_ids.to(self.base_device)
        mask = attention_mask.to(self.base_device)
        finished = torch.zeros(seq.size(0), dtype=torch.bool, device=self.base_device)
        past_base = None
        past_base_assist = None
        past_assist = None
        cur_input = seq

        while seq.size(1) < target_len:
            with torch.no_grad():
                base_out = self.basellm(
                    input_ids=_mv(cur_input, self.base_device),
                    attention_mask=_mv(mask, self.base_device),
                    past_key_values=past_base,
                    use_cache=True,
                )
                base_assist_out = self.base_assist_llm(
                    input_ids=_mv(cur_input, self.base_assist_device),
                    attention_mask=_mv(mask, self.base_assist_device),
                    past_key_values=past_base_assist,
                    use_cache=True,
                )
            assist_out = self.assist_llm(
                input_ids=_mv(cur_input, self.assist_device),
                attention_mask=_mv(mask, self.assist_device),
                past_key_values=past_assist,
                use_cache=True,
            )
            past_base = base_out.past_key_values
            past_base_assist = base_assist_out.past_key_values
            past_assist = assist_out.past_key_values

            next_logits = (
                base_out.logits[:, -1, :]
                + self.weight * (
                    assist_out.logits[:, -1, :].to(self.base_device)
                    - base_assist_out.logits[:, -1, :].to(self.base_device)
                )
            )

            if do_sample:
                temp = max(float(temperature), 1e-5)
                probs = torch.softmax(next_logits / temp, dim=-1)
                if 0.0 < float(top_p) < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                    cdf = torch.cumsum(sorted_probs, dim=-1)
                    remove = cdf > float(top_p)
                    remove[..., 0] = False
                    filtered = torch.where(remove, torch.zeros_like(sorted_probs), sorted_probs)
                    denom = filtered.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                    filtered = filtered / denom
                    sampled = torch.multinomial(filtered, 1).squeeze(-1)
                    next_token = sorted_indices.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
                else:
                    next_token = torch.multinomial(probs, 1).squeeze(-1)
            else:
                next_token = torch.argmax(next_logits, dim=-1)

            if eos_tokens:
                is_eos = torch.zeros_like(finished)
                for eos_id in eos_tokens:
                    is_eos = is_eos | (next_token == eos_id)
                next_token = torch.where(finished, torch.full_like(next_token, int(pad_token_id)), next_token)
                finished = finished | is_eos

            seq = torch.cat([seq, next_token.unsqueeze(-1).to(self.base_device)], dim=-1)
            cur_input = next_token.unsqueeze(-1)
            mask = torch.cat(
                [mask, torch.ones((mask.size(0), 1), dtype=mask.dtype, device=self.base_device)],
                dim=-1,
            )
            if eos_tokens and bool(torch.all(finished)):
                break

        return seq

    def forward(
        self, 
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        with torch.no_grad():
            outputs = self.basellm(
                input_ids=_mv(input_ids, self.base_device),
                attention_mask=_mv(attention_mask, self.base_device),
                position_ids=_mv(position_ids, self.base_device),
                past_key_values=past_key_values,
                inputs_embeds=_mv(inputs_embeds, self.base_device),
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                # cache_position=cache_position,
            )
            base_logits = outputs.logits.detach()
            outputs = self.base_assist_llm(
                input_ids=_mv(input_ids, self.base_assist_device),
                attention_mask=_mv(attention_mask, self.base_assist_device),
                position_ids=_mv(position_ids, self.base_assist_device),
                past_key_values=past_key_values,
                inputs_embeds=_mv(inputs_embeds, self.base_assist_device),
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            base_assist_logits = outputs.logits.detach().to(self.base_device)

        assist_outputs = self.assist_llm(
            input_ids=_mv(input_ids, self.assist_device),
            attention_mask=_mv(attention_mask, self.assist_device),
            position_ids=_mv(position_ids, self.assist_device),
            past_key_values=past_key_values,
            inputs_embeds=_mv(inputs_embeds, self.assist_device),
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        assist_logits = assist_outputs.logits.to(self.base_device)

        logits = base_logits + self.weight * (assist_logits - base_assist_logits) #! ajust the final distribution
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def save_pretrained(self, path, **kwargs):
        self.assist_llm.save_pretrained(path)
        self.config.save_pretrained(path)
