import json
import os
import re
from typing import List, Optional, Union, Tuple
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from .gen_util import ContrastGenerationMixin


class DoubleAssisLLM(torch.nn.Module, ContrastGenerationMixin):
    
    def __init__(
        self,
        basellm: AutoModelForCausalLM,
        original_assist_llm: AutoModelForCausalLM,
        finetuned_assist_llm: AutoModelForCausalLM,
        routing_score_paths: Optional[List[str]] = None,
        threshold: float = 12.6943,
        max_new_tokens: int = 20
    ) -> None:
        super().__init__()
        self.basellm = basellm
        self.original_assist_llm = original_assist_llm
        self.finetuned_assist_llm = finetuned_assist_llm
        self.threshold = threshold
        self.max_new_tokens = max_new_tokens

        # 设备和配置
        self.device = self.basellm.device
        self.config = self.basellm.config
        self.generation_config = basellm.generation_config

        # 初始化tokenizer - 使用base模型的tokenizer以保持一致性
        self.tokenizer = AutoTokenizer.from_pretrained(self.config._name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 添加调试模式
        self.debug_mode = False

        # 路由缓存：key=question_only_ids(tuple), value=(is_forget(bool), score(float))
        # ToFU 评估里同一问题会在 generate/forward 多次出现（paraphrase/perturb），缓存可显著降耗
        self._routing_cache = {}
        self._routing_score_lookup = {}

        # 增强的统计功能 - 支持4×3统计矩阵
        self.model_selection_stats = {
            'forward_calls': 0,
            'forward_base_model': 0,
            'forward_original_assist': 0,
            'generate_calls': 0,
            'generate_base_model': 0,
            'generate_original_assist': 0,
            'cross_entropies': []
        }

        # 数据集级别统计 - 4个split × 3个任务
        self.dataset_stats = {}
        self.current_dataset = None
        self.current_task = None

        # 确保所有模型都在同一设备上
        self.original_assist_llm = self.original_assist_llm.to(self.device)
        self.finetuned_assist_llm = self.finetuned_assist_llm.to(self.device)

        # 设置模型为评估模式
        self.basellm.eval()
        self.original_assist_llm.eval()
        self.finetuned_assist_llm.eval()

        # 初始化4×3统计矩阵
        self.init_dataset_stats()
        self._load_routing_score_cache(routing_score_paths or [])

    def init_dataset_stats(self):
        """
        初始化4×3统计矩阵
        4个数据分割：forget01_perturbed, forget05_perturbed, forget10_perturbed, retain99
        3个评估任务：forget, retain, real_authors
        """
        splits = ['forget01_perturbed', 'forget05_perturbed', 'forget10_perturbed', 'retain99']
        tasks = ['forget', 'retain', 'real_authors']

        self.dataset_stats = {}
        for split in splits:
            self.dataset_stats[split] = {}
            for task in tasks:
                self.dataset_stats[split][task] = {
                    'base_model': 0,
                    'assist_model': 0,
                    'total_calls': 0,
                    'cross_entropies': []
                }

    def set_current_context(self, dataset: str, task: str = None):
        """
        设置当前数据集和任务上下文，用于统计

        Args:
            dataset: 数据集名称 (如 'forget01_perturbed')
            task: 任务名称 (如 'forget', 'retain', 'real_authors')
        """
        self.current_dataset = dataset
        self.current_task = task

        # 确保数据集存在于统计中
        if dataset not in self.dataset_stats:
            self.dataset_stats[dataset] = {}

        if task and task not in self.dataset_stats[dataset]:
            self.dataset_stats[dataset][task] = {
                'base_model': 0,
                'assist_model': 0,
                'total_calls': 0,
                'cross_entropies': []
            }

    def set_current_dataset(self, dataset_name):
        """设置当前正在评估的数据集"""
        if dataset_name in self.dataset_stats:
            self.current_dataset = dataset_name
            print(f"设置当前评估数据集: {dataset_name}")
        else:
            print(f"警告: 未知的数据集名称: {dataset_name}")

    def format_prompt(self, question):
        """格式化问题为模型输入格式"""
        return f"question: {question.strip()} answer:"

    def _normalize_question_text(self, text: str) -> str:
        text = (text or "").strip()
        text = re.sub(r"^\s*question:\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*answer:\s*$", "", text, flags=re.IGNORECASE)
        return " ".join(text.split())

    def _load_routing_score_cache(self, routing_score_paths: List[str]):
        loaded = 0
        for path in routing_score_paths:
            if not path or not os.path.exists(path):
                continue
            try:
                payload = json.load(open(path, "r"))
            except Exception as exc:
                print(f"[routing-cache] failed to load {path}: {exc}")
                continue

            if isinstance(payload, dict):
                if isinstance(payload.get("results"), list):
                    entries = payload["results"]
                else:
                    entries = []
            elif isinstance(payload, list):
                entries = payload
            else:
                entries = []

            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                question = self._normalize_question_text(entry.get("question", ""))
                score = entry.get("cross_entropy", None)
                if not question or score is None:
                    continue
                self._routing_score_lookup[question] = float(score)
                loaded += 1

        if loaded:
            print(
                f"[routing-cache] loaded {loaded} question scores "
                f"from {len(routing_score_paths)} file(s)"
            )
    
    def compute_cross_entropy(self, finetuned_logits, original_logits):
        """
        计算两个模型输出的交叉熵，参考assis_tinyllama_test.py的实现
        """
        try:
            # 检查输入是否为空
            if finetuned_logits.numel() == 0 or original_logits.numel() == 0:
                return 0.0

            # 确保两个logits在同一设备上
            if finetuned_logits.device != original_logits.device:
                original_logits = original_logits.to(finetuned_logits.device)

            # 处理长度不一致的情况
            min_len = min(finetuned_logits.shape[0], original_logits.shape[0])
            if min_len == 0:
                return 0.0

            a1_logits = finetuned_logits[:min_len]
            a2_logits = original_logits[:min_len]

            # 计算长度差异因子 - 与参考实现保持一致
            finetuned_length = finetuned_logits.size(0)
            original_length = original_logits.size(0)
            max_length = max(finetuned_length, original_length)
            length_diff = abs(finetuned_length - original_length)
            length_factor = 1.0 + 0.1 * (length_diff / max(max_length, 1))

            # 将logits转换为概率分布
            a1_probs = F.softmax(a1_logits, dim=-1)
            a2_log_probs = F.log_softmax(a2_logits, dim=-1)

            # 计算每个token位置的交叉熵
            token_cross_entropies = -(a1_probs * a2_log_probs).sum(dim=-1)

            # 使用CE(t)²/∑CE(t)的加权方案，增加长度差异因子
            ce_squared = token_cross_entropies ** 2
            sum_ce = token_cross_entropies.sum()

            # 确保sum_ce是标量
            if hasattr(sum_ce, 'numel') and sum_ce.numel() > 1:
                sum_ce = sum_ce.mean()

            if sum_ce > 0:  # 避免除以0
                ratio = ce_squared.sum() / sum_ce
                # 确保ratio是标量
                if hasattr(ratio, 'numel') and ratio.numel() > 1:
                    ratio = ratio.mean()

                # 确保ratio可以转换为标量
                if hasattr(ratio, 'item'):
                    ratio_value = ratio.item()
                else:
                    ratio_value = float(ratio)

                weighted_ce = ratio_value * length_factor
            else:
                weighted_ce = 0.0

            return weighted_ce

        except Exception as e:
            print(f"Error in compute_cross_entropy: {e}")
            return 0.0

    def _compute_fixed_path_symmetric_kl(self, prompt_input_ids: torch.LongTensor) -> float:
        """
        固定 original 路径的 token-wise 对称 KL（fixed_sym_kl）：
        1) 用 original_assist_llm greedy 生成 token 序列（并记录每步 logits）
        2) 用 finetuned_assist_llm 在同一前缀路径上前向一次，取对应位置 logits
        3) 计算每步对称 KL 并取均值作为打分

        返回:
            score(float): sym_kl.mean()
        """
        if prompt_input_ids is None:
            return 0.0
        if prompt_input_ids.dim() == 1:
            prompt_input_ids = prompt_input_ids.unsqueeze(0)

        prompt_len = int(prompt_input_ids.size(1))
        if prompt_len <= 0:
            return 0.0

        # 1) original 路径生成 + 记录 step logits
        _, full_tokens, orig_step_logits = self.generate_with_logits(
            self.original_assist_llm,
            prompt_input_ids,
            max_new_tokens=self.max_new_tokens,
        )

        gen_len = int(full_tokens.size(1) - prompt_len)
        if gen_len <= 0 or orig_step_logits.numel() == 0:
            return 0.0

        # 2) finetuned 在同一路径上一次前向
        with torch.no_grad():
            ft_out = self.finetuned_assist_llm(input_ids=full_tokens)
            ft_logits_full = ft_out.logits[0]  # [seq_len, vocab]

        start = max(prompt_len - 1, 0)
        end = start + gen_len
        ft_step_logits = ft_logits_full[start:end, :]
        orig_step_logits = orig_step_logits[:gen_len, :]

        if ft_step_logits.numel() == 0 or orig_step_logits.numel() == 0:
            return 0.0

        orig_logp = F.log_softmax(orig_step_logits, dim=-1)
        ft_logp = F.log_softmax(ft_step_logits, dim=-1)
        orig_p = orig_logp.exp()
        ft_p = ft_logp.exp()

        kl_of = (orig_p * (orig_logp - ft_logp)).sum(dim=-1)
        kl_fo = (ft_p * (ft_logp - orig_logp)).sum(dim=-1)
        sym_kl = 0.5 * (kl_of + kl_fo)
        return float(sym_kl.mean().item())
    
    def generate_with_logits(self, model, input_ids, max_new_tokens=None):
        """
        使用指定模型生成答案并返回logits
        """
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
        max_new_tokens = int(max_new_tokens)

        if input_ids is None:
            vocab_size = int(getattr(getattr(model, "config", None), "vocab_size", self.config.vocab_size))
            empty = torch.empty(0, vocab_size, device=self.device)
            return "", torch.empty(1, 0, dtype=torch.long, device=self.device), empty

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        # NOTE: The previous implementation performed full-seq forward for each generated token,
        # which is O(T * L^2) and extremely slow. We switch to `generate(..., use_cache=True,
        # output_scores=True)` to keep the exact greedy path while using KV-cache.
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
        with torch.no_grad():
            gen_out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        sequences = gen_out.sequences  # [bsz, prompt+steps]
        scores_list = list(getattr(gen_out, "scores", []) or [])
        if scores_list:
            # [bsz, steps, vocab] -> [steps, vocab] for bsz=1
            step_logits = torch.stack(scores_list, dim=1)
            if step_logits.size(0) == 1:
                stacked_logits = step_logits.squeeze(0).contiguous()
            else:
                # Fallback: average over batch to preserve old return shape.
                stacked_logits = step_logits.mean(dim=0).contiguous()
        else:
            vocab_size = int(getattr(getattr(model, "config", None), "vocab_size", self.config.vocab_size))
            stacked_logits = torch.empty(0, vocab_size, device=input_ids.device)

        generated_text = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)[0]
        return generated_text, sequences, stacked_logits

    def extract_question_only(self, input_ids):
        """
        从完整输入中提取只包含问题的部分
        用于Forward模式下的交叉熵计算
        标准化格式以确保一致性
        """
        try:
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            input_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

            # 检查是否包含答案（通过寻找answer:后面的内容）
            if " answer:" in input_text:
                # 找到answer:的位置
                answer_pos = input_text.find(" answer:")
                # 提取问题部分
                question_part = input_text[:answer_pos].strip()

                # 标准化格式：确保question:后面只有一个空格，answer:前面只有一个空格
                # 移除question:后面的多余空格
                if question_part.startswith("question:"):
                    question_content = question_part[9:].strip()  # 移除"question:"并去除空格
                    standardized_question = f"question: {question_content} answer:"
                else:
                    # 如果格式不标准，直接添加answer:
                    standardized_question = f"{question_part} answer:"

                # 重新编码为token
                question_ids = self.tokenizer(standardized_question, return_tensors="pt")['input_ids']
                question_ids = question_ids.to(input_ids.device)

                return question_ids
            else:
                # 如果没有答案部分，直接返回原输入
                return input_ids

        except Exception as e:
            print(f"Error in extract_question_only: {e}")
            return input_ids

    def _predict_forget_mask_and_score(self, input_ids: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对一个 batch 的 input_ids 逐样本计算路由分数与 forget mask。

        打分方式与 `scripts/assis_tinyllama_test_path.py --metric fixed_sym_kl` 保持一致：
        固定 original 路径，计算 token-wise 对称 KL 的均值。

        返回:
            forget_mask: BoolTensor[bsz]（True=走 assist）
            scores: FloatTensor[bsz]
        """
        if input_ids is None:
            return torch.zeros(1, dtype=torch.bool), torch.zeros(1, dtype=torch.float32)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        bsz = int(input_ids.size(0))
        forget_mask = torch.zeros(bsz, dtype=torch.bool)
        scores = torch.zeros(bsz, dtype=torch.float32)

        for i in range(bsz):
            single_ids = input_ids[i : i + 1]
            question_only_ids = self.extract_question_only(single_ids)

            # cache key: CPU tuple
            key = tuple(int(x) for x in question_only_ids[0].detach().to("cpu").tolist())
            cached = self._routing_cache.get(key)
            if cached is not None:
                is_forget_i, score_i = cached
            else:
                question_key = self._normalize_question_text(
                    self.tokenizer.decode(question_only_ids[0], skip_special_tokens=True)
                )
                score_i = self._routing_score_lookup.get(question_key, None)
                if score_i is None:
                    # ensure on correct device
                    if question_only_ids.device != self.original_assist_llm.device:
                        question_only_ids = question_only_ids.to(self.original_assist_llm.device)
                    score_i = self._compute_fixed_path_symmetric_kl(question_only_ids)
                is_forget_i = bool(score_i > self.threshold)
                self._routing_cache[key] = (is_forget_i, float(score_i))

            forget_mask[i] = bool(is_forget_i)
            scores[i] = float(score_i)

        return forget_mask, scores

    def is_forget_related(self, input_ids):
        """
        判断输入是否与遗忘数据相关（fixed_sym_kl）。

        兼容旧调用：
          - batch=1: 返回 (bool, float)
          - batch>1: 返回 (BoolTensor[bsz], FloatTensor[bsz])
        """
        try:
            forget_mask, scores = self._predict_forget_mask_and_score(input_ids)

            # batch=1 兼容旧接口
            if input_ids is None:
                return False, 0.0
            if input_ids.dim() == 1 or (input_ids.dim() == 2 and input_ids.size(0) == 1):
                is_forget = bool(forget_mask[0].item())
                score = float(scores[0].item())
                if self.debug_mode:
                    try:
                        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
                        print(f"Input: {text[:80]}...")
                    except Exception:
                        pass
                    print(f"Score(fixed_sym_kl)={score:.6f}, threshold={self.threshold}, is_forget={is_forget}")
                return is_forget, score

            return forget_mask, scores

        except Exception as e:
            print(f"Error in is_forget_related: {e}")
            import traceback
            traceback.print_exc()
            if input_ids is not None and input_ids.dim() == 2 and input_ids.size(0) > 1:
                bsz = int(input_ids.size(0))
                return torch.zeros(bsz, dtype=torch.bool), torch.zeros(bsz, dtype=torch.float32)
            return False, 0.0

    def get_loss(self, logits, labels=None, attention_mask=None, reduction='mean'):
        """计算损失函数"""
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            if reduction == 'batchmean':
                loss_fct = CrossEntropyLoss(reduction='none')
                # 使用实际logits的vocab_size而不是self.config.vocab_size
                vocab_size = shift_logits.size(-1)
                shift_logits = shift_logits.view(-1, vocab_size)
                shift_labels = shift_labels.view(-1)
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)
                if attention_mask is not None:
                    loss = loss.sum(dim=-1) / (attention_mask.sum(dim=-1))
            else:
                loss_fct = CrossEntropyLoss(reduction=reduction)
                # 使用实际logits的vocab_size而不是self.config.vocab_size
                vocab_size = shift_logits.size(-1)
                shift_logits = shift_logits.view(-1, vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)
        return loss

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
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        前向传播函数
        根据 fixed_sym_kl 路由到 base / assist 模型进行推理
        """
        output_attentions = False
        output_hidden_states = False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is None:
            raise ValueError("input_ids must be provided")
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        bsz = int(input_ids.size(0))
        forget_mask, scores = self._predict_forget_mask_and_score(input_ids)

        # 更新统计信息（按样本计数）
        self.model_selection_stats['forward_calls'] += bsz
        self.model_selection_stats['cross_entropies'].extend([float(x) for x in scores.tolist()])
        num_assist = int(forget_mask.sum().item())
        self.model_selection_stats['forward_original_assist'] += num_assist
        self.model_selection_stats['forward_base_model'] += (bsz - num_assist)
        self._update_dataset_stats('forward', forget_mask, scores)

        # 统一把输入移到同一 device（三个模型已强制放到 self.device）
        device = self.device
        if input_ids.device != device:
            input_ids = input_ids.to(device)
        if attention_mask is not None and attention_mask.device != device:
            attention_mask = attention_mask.to(device)
        if position_ids is not None and position_ids.device != device:
            position_ids = position_ids.to(device)
        if labels is not None and labels.device != device:
            labels = labels.to(device)

        base_indices = (~forget_mask).nonzero(as_tuple=False).view(-1).to(device)
        assist_indices = (forget_mask).nonzero(as_tuple=False).view(-1).to(device)

        base_logits = None
        assist_logits = None

        if base_indices.numel() > 0:
            base_out = self.basellm(
                input_ids=input_ids.index_select(0, base_indices),
                attention_mask=attention_mask.index_select(0, base_indices) if attention_mask is not None else None,
                position_ids=position_ids.index_select(0, base_indices) if position_ids is not None else None,
                past_key_values=None,
                inputs_embeds=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
            base_logits = base_out.logits

        if assist_indices.numel() > 0:
            assist_out = self.original_assist_llm(
                input_ids=input_ids.index_select(0, assist_indices),
                attention_mask=attention_mask.index_select(0, assist_indices) if attention_mask is not None else None,
                position_ids=position_ids.index_select(0, assist_indices) if position_ids is not None else None,
                past_key_values=None,
                inputs_embeds=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
            assist_logits = assist_out.logits

        logits_template = base_logits if base_logits is not None else assist_logits
        if logits_template is None:
            vocab_size = int(self.config.vocab_size)
            logits = torch.empty(bsz, input_ids.size(1), vocab_size, device=device, dtype=torch.bfloat16)
        else:
            logits = logits_template.new_empty((bsz,) + logits_template.shape[1:])

        if base_logits is not None:
            logits[base_indices] = base_logits
        if assist_logits is not None:
            logits[assist_indices] = assist_logits

        loss = self.get_loss(logits, labels, attention_mask)

        # past_key_values / hidden_states / attentions 无法可靠合并，ToFU eval 也不使用
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    def _update_dataset_stats(self, operation: str, forget_mask: torch.Tensor, scores: torch.Tensor):
        """
        更新按数据集聚合的路由统计。

        说明：
        - ToFU eval 会 batch>1；这里按 batch 聚合统计，避免每条样本都调用一次。
        - key 使用 operation（forward/generate），避免依赖 current_task 映射不一致的问题。

        Args:
            operation: 'forward' 或 'generate'
            forget_mask: BoolTensor[bsz]（True=走 assist）
            scores: FloatTensor[bsz]（fixed_sym_kl 分数）
        """
        current_dataset = self.current_dataset or "unknown_dataset"
        op_key = operation or "unknown_op"

        if current_dataset not in self.dataset_stats:
            self.dataset_stats[current_dataset] = {}

        if op_key not in self.dataset_stats[current_dataset]:
            self.dataset_stats[current_dataset][op_key] = {
                "base_model": 0,
                "assist_model": 0,
                "total_calls": 0,
                "cross_entropies": [],
            }

        stats = self.dataset_stats[current_dataset][op_key]
        bsz = int(scores.numel())
        assist = int(forget_mask.sum().item())
        base = bsz - assist
        stats["total_calls"] += bsz
        stats["assist_model"] += assist
        stats["base_model"] += base
        stats["cross_entropies"].extend([float(x) for x in scores.tolist()])

    def prepare_inputs_for_generation(self, *args, **kwargs):
        """为生成准备输入，委托给基础模型"""
        return self.basellm.prepare_inputs_for_generation(*args, **kwargs)

    def can_generate(self):
        """检查是否可以生成"""
        return True

    def _get_generation_model(self, input_ids):
        """兼容旧接口：仅支持 batch=1，返回用于生成的模型"""
        if input_ids is None:
            return self.basellm
        if input_ids.dim() == 2 and input_ids.size(0) > 1:
            raise ValueError("_get_generation_model only supports batch=1; use generate() for batched routing.")

        is_forget, score = self.is_forget_related(input_ids)
        self.model_selection_stats["generate_calls"] += 1
        self.model_selection_stats["cross_entropies"].append(float(score))

        if is_forget:
            self.model_selection_stats["generate_original_assist"] += 1
            self._update_dataset_stats("generate", torch.tensor([True]), torch.tensor([float(score)]))
            if self.debug_mode:
                print(f"Generation: assist (score={score:.6f} > {self.threshold})")
            return self.original_assist_llm

        self.model_selection_stats["generate_base_model"] += 1
        self._update_dataset_stats("generate", torch.tensor([False]), torch.tensor([float(score)]))
        if self.debug_mode:
            print(f"Generation: base (score={score:.6f} <= {self.threshold})")
        return self.basellm

    def generate(self, inputs=None, input_ids=None, **kwargs):
        """
        简化的生成方法
        根据 fixed_sym_kl 逐样本路由到 base / assist 进行生成
        """
        # 处理不同的输入格式
        if inputs is not None:
            input_ids = inputs
        elif input_ids is not None:
            pass  # 使用input_ids
        else:
            raise ValueError("Either inputs or input_ids must be provided")

        if input_ids is None:
            raise ValueError("Either inputs or input_ids must be provided")
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        device = self.device
        if input_ids.device != device:
            input_ids = input_ids.to(device)

        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is not None and attention_mask.device != device:
            attention_mask = attention_mask.to(device)
            kwargs["attention_mask"] = attention_mask

        bsz = int(input_ids.size(0))
        if bsz == 1:
            selected_model = self._get_generation_model(input_ids)
            if input_ids.device != selected_model.device:
                input_ids = input_ids.to(selected_model.device)
            with torch.no_grad():
                return selected_model.generate(input_ids=input_ids, **kwargs)

        # batch>1：逐样本路由，再分组生成并合并
        forget_mask, scores = self._predict_forget_mask_and_score(input_ids)
        self.model_selection_stats["generate_calls"] += bsz
        self.model_selection_stats["cross_entropies"].extend([float(x) for x in scores.tolist()])
        num_assist = int(forget_mask.sum().item())
        self.model_selection_stats["generate_original_assist"] += num_assist
        self.model_selection_stats["generate_base_model"] += (bsz - num_assist)
        self._update_dataset_stats("generate", forget_mask, scores)

        base_indices = (~forget_mask).nonzero(as_tuple=False).view(-1).to(device)
        assist_indices = (forget_mask).nonzero(as_tuple=False).view(-1).to(device)

        base_out = None
        assist_out = None
        with torch.no_grad():
            if base_indices.numel() > 0:
                base_kwargs = dict(kwargs)
                base_kwargs["input_ids"] = input_ids.index_select(0, base_indices)
                if attention_mask is not None:
                    base_kwargs["attention_mask"] = attention_mask.index_select(0, base_indices)
                base_out = self.basellm.generate(**base_kwargs)

            if assist_indices.numel() > 0:
                assist_kwargs = dict(kwargs)
                assist_kwargs["input_ids"] = input_ids.index_select(0, assist_indices)
                if attention_mask is not None:
                    assist_kwargs["attention_mask"] = attention_mask.index_select(0, assist_indices)
                assist_out = self.original_assist_llm.generate(**assist_kwargs)

        out_template = base_out if base_out is not None else assist_out
        if out_template is None:
            return torch.full((bsz, input_ids.size(1)), self.tokenizer.eos_token_id, dtype=torch.long, device=device)

        pad_token_id = kwargs.get("pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

        max_len = int(out_template.size(1))
        if base_out is not None:
            max_len = max(max_len, int(base_out.size(1)))
        if assist_out is not None:
            max_len = max(max_len, int(assist_out.size(1)))

        final = torch.full((bsz, max_len), pad_token_id, dtype=out_template.dtype, device=out_template.device)
        if base_out is not None:
            final[base_indices, : base_out.size(1)] = base_out
        if assist_out is not None:
            final[assist_indices, : assist_out.size(1)] = assist_out
        return final

    def generate_statistics_report(self) -> dict:
        """
        生成详细的4×3统计报告

        Returns:
            包含完整统计信息的字典
        """
        import numpy as np

        report = {
            'summary': {
                'total_forward_calls': self.model_selection_stats['forward_calls'],
                'total_generate_calls': self.model_selection_stats['generate_calls'],
                'forward_base_model': self.model_selection_stats['forward_base_model'],
                'forward_assist_model': self.model_selection_stats['forward_original_assist'],
                'generate_base_model': self.model_selection_stats['generate_base_model'],
                'generate_assist_model': self.model_selection_stats['generate_original_assist'],
            },
            'dataset_matrix': {},
            'validation': {}
        }

        # 生成4×3统计矩阵
        splits = ['forget01_perturbed', 'forget05_perturbed', 'forget10_perturbed', 'retain99']
        tasks = ['forget', 'retain', 'real_authors']

        for split in splits:
            if split in self.dataset_stats:
                report['dataset_matrix'][split] = {}
                for task in tasks:
                    if task in self.dataset_stats[split]:
                        stats = self.dataset_stats[split][task]
                        report['dataset_matrix'][split][task] = {
                            'base_model': stats['base_model'],
                            'assist_model': stats['assist_model'],
                            'total_calls': stats['total_calls'],
                            'base_model_ratio': stats['base_model'] / max(stats['total_calls'], 1),
                            'assist_model_ratio': stats['assist_model'] / max(stats['total_calls'], 1)
                        }

                        # 添加交叉熵统计
                        if stats['cross_entropies']:
                            ce_stats = {
                                'mean': float(np.mean(stats['cross_entropies'])),
                                'std': float(np.std(stats['cross_entropies'])),
                                'min': float(np.min(stats['cross_entropies'])),
                                'max': float(np.max(stats['cross_entropies'])),
                                'count': len(stats['cross_entropies'])
                            }
                            report['dataset_matrix'][split][task]['cross_entropy_stats'] = ce_stats

        # 验证数据一致性
        report['validation'] = self._validate_statistics()

        return report

    def _validate_statistics(self) -> dict:
        """
        验证统计数据的一致性

        Returns:
            验证结果字典
        """
        validation = {
            'consistent': True,
            'issues': [],
            'split_totals': {}
        }

        # 检查每个split在不同任务中的总调用次数是否一致
        for split in self.dataset_stats:
            task_totals = {}
            for task in self.dataset_stats[split]:
                total = self.dataset_stats[split][task]['total_calls']
                task_totals[task] = total

            validation['split_totals'][split] = task_totals

            # 检查一致性（允许小的差异，因为可能有不同的评估阶段）
            if len(set(task_totals.values())) > 1:
                max_diff = max(task_totals.values()) - min(task_totals.values())
                if max_diff > 10:  # 允许10个调用的差异
                    validation['consistent'] = False
                    validation['issues'].append(f"Split {split} has inconsistent task totals: {task_totals}")

        return validation

    def print_statistics_table(self):
        """
        打印4×3统计表格到控制台
        """
        print("\n" + "="*80)
        print("Double Assist LLM - 4×3 统计矩阵")
        print("="*80)

        splits = ['forget01_perturbed', 'forget05_perturbed', 'forget10_perturbed', 'retain99']
        tasks = ['forget', 'retain', 'real_authors']

        # 表头
        print(f"{'Split':<20} {'Task':<15} {'Base Model':<12} {'Assist Model':<12} {'Total':<8} {'Base %':<8} {'Assist %':<8}")
        print("-" * 80)

        for split in splits:
            if split in self.dataset_stats:
                for i, task in enumerate(tasks):
                    if task in self.dataset_stats[split]:
                        stats = self.dataset_stats[split][task]
                        base_count = stats['base_model']
                        assist_count = stats['assist_model']
                        total = stats['total_calls']

                        base_pct = (base_count / max(total, 1)) * 100
                        assist_pct = (assist_count / max(total, 1)) * 100

                        split_name = split if i == 0 else ""
                        print(f"{split_name:<20} {task:<15} {base_count:<12} {assist_count:<12} {total:<8} {base_pct:<7.1f}% {assist_pct:<7.1f}%")
                print("-" * 80)

        # 总结
        total_base = self.model_selection_stats['forward_base_model'] + self.model_selection_stats['generate_base_model']
        total_assist = self.model_selection_stats['forward_original_assist'] + self.model_selection_stats['generate_original_assist']
        grand_total = total_base + total_assist

        print(f"{'TOTAL':<20} {'ALL':<15} {total_base:<12} {total_assist:<12} {grand_total:<8} {(total_base/max(grand_total,1)*100):<7.1f}% {(total_assist/max(grand_total,1)*100):<7.1f}%")
        print("="*80)

    def get_model_selection_stats(self):
        """获取模型选择统计信息"""
        stats = self.model_selection_stats.copy()

        # 计算百分比
        if stats['forward_calls'] > 0:
            stats['forward_base_model_pct'] = (stats['forward_base_model'] / stats['forward_calls']) * 100
            stats['forward_original_assist_pct'] = (stats['forward_original_assist'] / stats['forward_calls']) * 100
        else:
            stats['forward_base_model_pct'] = 0
            stats['forward_original_assist_pct'] = 0

        if stats['generate_calls'] > 0:
            stats['generate_base_model_pct'] = (stats['generate_base_model'] / stats['generate_calls']) * 100
            stats['generate_original_assist_pct'] = (stats['generate_original_assist'] / stats['generate_calls']) * 100
        else:
            stats['generate_base_model_pct'] = 0
            stats['generate_original_assist_pct'] = 0

        # 交叉熵统计
        if stats['cross_entropies']:
            import numpy as np
            stats['cross_entropy_mean'] = np.mean(stats['cross_entropies'])
            stats['cross_entropy_std'] = np.std(stats['cross_entropies'])
            stats['cross_entropy_min'] = np.min(stats['cross_entropies'])
            stats['cross_entropy_max'] = np.max(stats['cross_entropies'])
        else:
            stats['cross_entropy_mean'] = 0
            stats['cross_entropy_std'] = 0
            stats['cross_entropy_min'] = 0
            stats['cross_entropy_max'] = 0

        # 添加数据集级别的统计 - 处理二维结构
        stats['dataset_stats'] = {}
        for split_name, split_data in self.dataset_stats.items():
            if isinstance(split_data, dict):
                stats['dataset_stats'][split_name] = {}
                for task_name, task_data in split_data.items():
                    if isinstance(task_data, dict) and 'base_model' in task_data:
                        total_calls = task_data.get('total_calls', 0)
                        base_calls = task_data.get('base_model', 0)
                        assist_calls = task_data.get('assist_model', 0)

                        stats['dataset_stats'][split_name][task_name] = {
                            'total_calls': total_calls,
                            'base_model': base_calls,
                            'assist_model': assist_calls,
                            'base_model_pct': (base_calls / total_calls * 100) if total_calls > 0 else 0,
                            'assist_model_pct': (assist_calls / total_calls * 100) if total_calls > 0 else 0,
                            'cross_entropies': task_data.get('cross_entropies', []).copy()
                        }

                        # 交叉熵统计
                        cross_entropies = task_data.get('cross_entropies', [])
                        if cross_entropies:
                            import numpy as np
                            stats['dataset_stats'][split_name][task_name]['cross_entropy_mean'] = np.mean(cross_entropies)
                            stats['dataset_stats'][split_name][task_name]['cross_entropy_std'] = np.std(cross_entropies)
                            stats['dataset_stats'][split_name][task_name]['cross_entropy_min'] = np.min(cross_entropies)
                            stats['dataset_stats'][split_name][task_name]['cross_entropy_max'] = np.max(cross_entropies)
                        else:
                            stats['dataset_stats'][split_name][task_name]['cross_entropy_mean'] = 0
                            stats['dataset_stats'][split_name][task_name]['cross_entropy_std'] = 0
                            stats['dataset_stats'][split_name][task_name]['cross_entropy_min'] = 0
                            stats['dataset_stats'][split_name][task_name]['cross_entropy_max'] = 0

        return stats

    def reset_model_selection_stats(self):
        """重置模型选择统计信息"""
        self.model_selection_stats = {
            'forward_calls': 0,
            'forward_base_model': 0,
            'forward_original_assist': 0,
            'generate_calls': 0,
            'generate_base_model': 0,
            'generate_original_assist': 0,
            'cross_entropies': []
        }

        # 重置数据集级别统计 - 保持二维结构
        for split in self.dataset_stats:
            if isinstance(self.dataset_stats[split], dict):
                for task in self.dataset_stats[split]:
                    if isinstance(self.dataset_stats[split][task], dict):
                        self.dataset_stats[split][task] = {
                            'base_model': 0,
                            'assist_model': 0,
                            'total_calls': 0,
                            'cross_entropies': []
                        }

    def print_model_selection_stats(self):
        """打印模型选择统计信息"""
        stats = self.get_model_selection_stats()

        print("\n" + "="*60)
        print("DoubleAssisLLM 模型选择统计")
        print("="*60)

        print(f"Forward调用统计:")
        print(f"  总调用次数: {stats['forward_calls']}")
        print(f"  基础模型: {stats['forward_base_model']} ({stats['forward_base_model_pct']:.1f}%)")
        print(f"  原始辅助模型: {stats['forward_original_assist']} ({stats['forward_original_assist_pct']:.1f}%)")

        print(f"\nGenerate调用统计:")
        print(f"  总调用次数: {stats['generate_calls']}")
        print(f"  基础模型: {stats['generate_base_model']} ({stats['generate_base_model_pct']:.1f}%)")
        print(f"  原始辅助模型: {stats['generate_original_assist']} ({stats['generate_original_assist_pct']:.1f}%)")

        print(f"\n交叉熵统计:")
        print(f"  平均值: {stats['cross_entropy_mean']:.4f}")
        print(f"  标准差: {stats['cross_entropy_std']:.4f}")
        print(f"  最小值: {stats['cross_entropy_min']:.4f}")
        print(f"  最大值: {stats['cross_entropy_max']:.4f}")
        print(f"  阈值: {self.threshold}")

        # 打印数据集级别统计
        print(f"\n按数据集统计:")
        print("-"*60)
        for dataset_name, dataset_stat in stats['dataset_stats'].items():
            if dataset_stat['forward_calls'] > 0 or dataset_stat['generate_calls'] > 0:
                print(f"\n{dataset_name}:")
                if dataset_stat['forward_calls'] > 0:
                    print(f"  Forward: {dataset_stat['forward_calls']} 次")
                    print(f"    基础模型: {dataset_stat['forward_base_model']} ({dataset_stat['forward_base_model_pct']:.1f}%)")
                    print(f"    辅助模型: {dataset_stat['forward_original_assist']} ({dataset_stat['forward_original_assist_pct']:.1f}%)")
                if dataset_stat['generate_calls'] > 0:
                    print(f"  Generate: {dataset_stat['generate_calls']} 次")
                    print(f"    基础模型: {dataset_stat['generate_base_model']} ({dataset_stat['generate_base_model_pct']:.1f}%)")
                    print(f"    辅助模型: {dataset_stat['generate_original_assist']} ({dataset_stat['generate_original_assist_pct']:.1f}%)")
                if dataset_stat['cross_entropies']:
                    print(f"  交叉熵: 均值={dataset_stat['cross_entropy_mean']:.4f}, 标准差={dataset_stat['cross_entropy_std']:.4f}")

        print("="*60)

    def __call__(self, *args, **kwargs):
        """
        调用接口，支持两种模式：
        1. 作为模型的forward调用: model(input_ids=..., attention_mask=...)
        2. 作为文本生成调用: model(["text1", "text2"], max_len=100)
        """
        # 如果第一个参数是字符串列表，则进行文本生成
        if args and isinstance(args[0], (list, tuple)) and all(isinstance(x, str) for x in args[0]):
            return self._call_text_generation(*args, **kwargs)
        else:
            # 否则调用forward方法
            return self.forward(*args, **kwargs)

    def _call_text_generation(self, inputs: List[str], max_len=100, top_p=1.0, temperature=1e-9) -> List[str]:
        """
        文本生成调用接口
        """
        # 将文本转换为token
        input_ids = self.tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True
        )['input_ids'].to(self.device)

        # 生成输出
        outputs = self.generate(
            inputs=input_ids,
            max_length=max_len,
            top_p=top_p,
            temperature=temperature,
            do_sample=(temperature > 1e-9),
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # 解码输出
        outstrs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        genstrs = []
        for inp, out in zip(inputs, outstrs):
            # 提取生成的部分（去除输入部分）
            if inp in out:
                generated_part = out.split(inp)[-1]
            else:
                generated_part = out
            genstrs.append(generated_part)

        return genstrs
