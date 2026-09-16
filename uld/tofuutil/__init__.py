import os
import csv
import json
import time
import torch
import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf

from .data_module import TextDatasetQA, custom_data_collator, get_batch_loss
from .evaluate_util import eval_rouge_recall
from .utils import get_model_utility, get_forget_quality, get_forget_quality_func, get_forget_prob
from ..utils import set_progress

def prepare_loader(dataset, batch_size):
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, collate_fn=custom_data_collator 
    )
    return loader


def prepare_dataset(dataname, tokenizer, conv_template, split, question_key, answer_key, max_num=-1):
    dataset = TextDatasetQA(
        data_path=dataname,
        tokenizer=tokenizer,
        conv_template=conv_template, 
        split=split,
        question_key=question_key,
        answer_key=answer_key,
        max_num=max_num
    )
    return dataset


def _to_plain_value(value):
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _resolve_eval_field(raw_value, eval_split, primary_split, default_value):
    value = _to_plain_value(raw_value)
    if value is None:
        return default_value

    if isinstance(value, dict):
        return value.get(eval_split, value.get(primary_split, default_value))

    if isinstance(value, (list, tuple)):
        ordered_splits = [
            primary_split,
            "retain_perturbed",
            "real_authors_perturbed",
            "world_facts_perturbed",
        ]
        if eval_split in ordered_splits:
            idx = ordered_splits.index(eval_split)
            if idx < len(value):
                return value[idx]
        return value[0] if value else default_value

    return value


def tofu_eval(OUTPUTDIR, LOGGER, configs, model, tokenizer, right_pad_tokenizer, conv_template, only_forget_quality=False):
    progress = set_progress(disable=os.getenv("POOR", False)) 
    no_text_log = os.getenv("TOFU_NO_TEXT_LOG", "0") == "1"
    profile_timing = os.getenv("TOFU_PROFILE_TIMING", "0") == "1"
    requested_splits_raw = os.getenv("TOFU_EVAL_SPLITS", "").strip()
    timing_profile = {
        "dataset_prepare_sec": 0.0,
        "generation_sec": 0.0,
        "rouge_sec": 0.0,
        "nexttoken_sec": 0.0,
        "perturb_ratio_sec": 0.0,
        "cache_clear_sec": 0.0,
        "cache_clear_calls": 0,
        "splits": {},
    }
    with progress:
        retain_result_path = getattr(getattr(configs.dataset, "eval", None), "retain_result", None)
        has_retain_result = bool(retain_result_path) and os.path.exists(retain_result_path)

        if not only_forget_quality:
            eval_tasks = [
                configs.dataset.split, 
                "retain_perturbed", 
                "real_authors_perturbed", "world_facts_perturbed", 
            ]
        else:
            eval_tasks = [
                configs.dataset.split, "retain_perturbed", 
            ]

        if requested_splits_raw:
            alias_to_split = {
                "forget": configs.dataset.split,
                "retain": "retain_perturbed",
                "real_authors": "real_authors_perturbed",
                "real_world": "world_facts_perturbed",
            }
            requested_splits = []
            for item in requested_splits_raw.split(","):
                key = item.strip()
                if not key:
                    continue
                requested_splits.append(alias_to_split.get(key, key))
            requested_set = set(requested_splits)
            filtered_eval_tasks = [split for split in eval_tasks if split in requested_set]
            if filtered_eval_tasks:
                eval_tasks = filtered_eval_tasks
            print(f"[tofu-eval] requested_splits={requested_splits_raw} -> {eval_tasks}")

        eval_task = progress.add_task(
            "evalbar",
            name="[green][Main Evaluate]",
            total=len(eval_tasks),
        )

        eval_logs_by_task = {}
        for eval_split in eval_tasks:
            split_timing = {
                "dataset_prepare_sec": 0.0,
                "generation_sec": 0.0,
                "rouge_sec": 0.0,
                "nexttoken_sec": 0.0,
                "perturb_ratio_sec": 0.0,
                "cache_clear_sec": 0.0,
                "cache_clear_calls": 0,
            }
            # 为 DoubleAssisLLM 等路由模型设置当前评估 split，上报路由统计用
            if hasattr(model, "set_current_context"):
                try:
                    model.set_current_context(eval_split, None)
                except Exception:
                    pass
            task_name = eval_split if eval_split != configs.dataset.split else "eval_log_forget"
            question_key = _resolve_eval_field(
                getattr(configs.dataset, "question_key", None),
                eval_split,
                configs.dataset.split,
                "question",
            )
            answer_key = _resolve_eval_field(
                getattr(configs.dataset, "answer_key", None),
                eval_split,
                configs.dataset.split,
                "answer",
            )
            base_default = "answer" if eval_split in ["real_authors_perturbed", "world_facts_perturbed"] else "paraphrased_answer"
            base_answer_key = _resolve_eval_field(
                getattr(configs.dataset, "base_answer_key", None),
                eval_split,
                configs.dataset.split,
                base_default,
            )
            perturbed_answer_key = _resolve_eval_field(
                getattr(configs.dataset, "perturbed_answer_key", None),
                eval_split,
                configs.dataset.split,
                "perturbed_answer",
            )
            batch_size = configs.dataset.eval.batch_size
            print("Batch size", batch_size)
            # ToFU official implementation uses the first 300 by default.
            # Allow overriding for screening/ablation (keeps default behavior unchanged).
            try:
                MAX_NUM = int(getattr(getattr(configs.dataset, "eval", None), "max_num", 300))
            except Exception:
                MAX_NUM = 300
            if MAX_NUM <= 0:
                MAX_NUM = 300

            eval_logs = {}
            #! evaluate generations
            gen_outputs = []
            ground_truths = []
            input_strings = []
            t0 = time.perf_counter()
            eval_dataset = prepare_dataset(
                configs.dataset.name, tokenizer, conv_template, eval_split, question_key, answer_key, max_num=MAX_NUM
            )
            eval_dataloader = prepare_loader(
                eval_dataset, batch_size,
            )
            dt_prepare = time.perf_counter() - t0
            timing_profile["dataset_prepare_sec"] += dt_prepare
            split_timing["dataset_prepare_sec"] += dt_prepare
            with torch.no_grad():
                gen_task = progress.add_task( #? build progress
                    f"[red][{task_name}-generation]", name=f"{task_name}", total=len(eval_dataloader),
                )
                t_generation = time.perf_counter()
                def batch_generator(tfdataset, batch_size): #! we only need question and answer for eval_dataset
                    for i in range(0, len(tfdataset), batch_size):
                        batchdata = tfdataset[i:min(i + batch_size, len(tfdataset))]
                        if isinstance(batchdata, dict):
                            yield batchdata[question_key], batchdata[answer_key]
                        else:
                            yield [item[question_key] for item in batchdata], [item[answer_key] for item in batchdata]

                for batch in batch_generator(eval_dataset.data, batch_size):
                    questions, answers = batch
                    gen_inputs = [
                        conv_template.prepare_gen_prompt(question, answer) for question, answer in zip(questions, answers)
                    ]
                    inputs = tokenizer(
                        gen_inputs, add_special_tokens=True, return_tensors="pt", padding=True, 
                    ).to(model.device)
                    outputs = model.generate(
                        **inputs,
                        max_length=configs.dataset.eval.generation.max_length,
                        max_new_tokens=configs.dataset.eval.generation.max_new_tokens, 
                        do_sample=False, 
                        use_cache=True, 
                        pad_token_id=tokenizer.eos_token_id,
                    )
                    out_strs = tokenizer.batch_decode(
                        outputs[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True)

                    # print(gen_inputs, "\n", out_strs)
                    gen_outputs.extend(out_strs)
                    input_strings.extend(gen_inputs)
                    ground_truths.extend(answers)

                    # LOGGER.info("Generation", input=gen_inputs, output=out_strs)
                    progress.advance(gen_task) #? update progress
                dt_generation = time.perf_counter() - t_generation
                timing_profile["generation_sec"] += dt_generation
                split_timing["generation_sec"] += dt_generation

            t_rouge = time.perf_counter()
            rougeL = eval_rouge_recall(gen_outputs, ground_truths)
            dt_rouge = time.perf_counter() - t_rouge
            timing_profile["rouge_sec"] += dt_rouge
            split_timing["rouge_sec"] += dt_rouge
            eval_logs.update(rougeL)
            if not no_text_log:
                eval_logs['generated_text'] = list(zip(input_strings, gen_outputs, ground_truths))

            LOGGER.info("GenerationResult", generationout=np.mean(rougeL['rougeL_recall']))

            #! evaluate next-token probs
            t0 = time.perf_counter()
            eval_dataset = prepare_dataset(
                configs.dataset.name, right_pad_tokenizer, conv_template, eval_split, question_key, answer_key, max_num=MAX_NUM
            )
            eval_dataloader = prepare_loader(
                eval_dataset, batch_size,
            )
            dt_prepare = time.perf_counter() - t0
            timing_profile["dataset_prepare_sec"] += dt_prepare
            split_timing["dataset_prepare_sec"] += dt_prepare
            with torch.no_grad():
                gen_task = progress.add_task( #? build progress
                    f"[red][{task_name}-nexttoken]", name=f"{task_name}", total=len(eval_dataloader),
                )
                t_nexttoken = time.perf_counter()
                for batch in eval_dataloader:
                    input_ids, labels, attention_mask = batch
                    batch = {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
                    for k, v in batch.items():
                        batch[k] = v.to(model.device)
                    outputs = model(**batch) #! forward to get logits
                    gt_loss = get_batch_loss(outputs.logits, batch['labels']).float()
                    num_token_gt = (batch['labels'] != -100).sum(-1)
                    eval_logs['avg_gt_loss'] = eval_logs.get('avg_gt_loss', []) + (gt_loss / num_token_gt).float().cpu().numpy().tolist()
                    eval_logs['gt_loss'] = eval_logs.get('gt_loss', []) + gt_loss.tolist()
                    eval_logs['num_token_gt'] = eval_logs.get('num_token_gt', []) + num_token_gt.tolist()
                    progress.advance(gen_task) #? update progress
                dt_nexttoken = time.perf_counter() - t_nexttoken
                timing_profile["nexttoken_sec"] += dt_nexttoken
                split_timing["nexttoken_sec"] += dt_nexttoken

            #! evaluate ratio
            t0 = time.perf_counter()
            base_eval_dataloader = prepare_loader(
                prepare_dataset(
                    configs.dataset.name, right_pad_tokenizer, conv_template, eval_split, question_key, base_answer_key, max_num=MAX_NUM
                ),
                max(1, batch_size // 4),
            )
            perturb_dataloader = prepare_loader(
                prepare_dataset(
                    configs.dataset.name, right_pad_tokenizer, conv_template, eval_split, question_key, perturbed_answer_key, max_num=MAX_NUM
                ),
                max(1, batch_size // 4),
            )
            dt_prepare = time.perf_counter() - t0
            timing_profile["dataset_prepare_sec"] += dt_prepare
            split_timing["dataset_prepare_sec"] += dt_prepare

            with torch.no_grad():
                tmp_logs = {}
                gen_task = progress.add_task( #? build progress
                    f"[red][{task_name}-perturb_ratio]", name=f"{task_name}", total=len(eval_dataloader),
                )
                t_ratio = time.perf_counter()
                for batch, perturb_batch in zip(base_eval_dataloader, perturb_dataloader):
                    input_ids, labels, attention_mask = batch
                    batch = {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
                    perturb_input_ids, perturb_labels, perturb_attention_mask = perturb_batch
                    if len(perturb_input_ids.shape) > 2:
                        bsz, seq_len = perturb_input_ids.shape[0:2]
                    else:
                        bsz = perturb_input_ids.shape[0]
                        seq_len = 1
                    perturb_batch = {
                        "input_ids": perturb_input_ids.view(bsz*seq_len, -1), 
                        "labels": perturb_labels.view(bsz*seq_len, -1), 
                        "attention_mask": perturb_attention_mask.view(bsz*seq_len, -1)
                    }

                    #send to device
                    for k, v in batch.items():
                        batch[k] = v.to(model.device)
                    for k, v in perturb_batch.items():
                        perturb_batch[k] = v.to(model.device)

                    # The perturbation-ratio pass uses the same model on the
                    # original answers and their perturbed variants. Run them
                    # in a single forward to avoid paying the routing/model
                    # setup cost twice for the same batch.
                    combined_batch = {
                        "input_ids": torch.cat([batch["input_ids"], perturb_batch["input_ids"]], dim=0),
                        "labels": torch.cat([batch["labels"], perturb_batch["labels"]], dim=0),
                        "attention_mask": torch.cat([batch["attention_mask"], perturb_batch["attention_mask"]], dim=0),
                    }

                    combined_outputs = model(**combined_batch, use_cache=False)
                    combined_logits = combined_outputs.logits.float()
                    gt_loss = get_batch_loss(combined_logits[:bsz], batch['labels']).detach()
                    perturb_loss = get_batch_loss(
                        combined_logits[bsz:], perturb_batch['labels']
                    ).detach().view(bsz, seq_len)
                    num_token_gt = (batch['labels']!=-100).sum(-1).detach()
                    num_token_perturb = (perturb_batch['labels']!=-100).view(bsz, seq_len, -1).sum(-1).detach()

                    tmp_logs['average_perturb_loss'] = tmp_logs.get('average_perturb_loss', []) + (
                        (perturb_loss / num_token_perturb.clamp_min(1)).detach().cpu().tolist()
                    )
                    tmp_logs['avg_paraphrased_loss'] = tmp_logs.get('avg_paraphrased_loss', []) + (
                        (gt_loss / num_token_gt.clamp_min(1)).detach().cpu().tolist()
                    )
                    tmp_logs['paraphrased_loss'] = tmp_logs.get('paraphrased_loss', []) + gt_loss.detach().cpu().tolist()
                    tmp_logs['perturb_loss'] = tmp_logs.get('perturb_loss', []) + perturb_loss.detach().cpu().tolist()
                    tmp_logs['num_token_paraphrased'] = tmp_logs.get('num_token_paraphrased', []) + num_token_gt.detach().cpu().tolist()
                    tmp_logs['num_token_perturb'] = tmp_logs.get('num_token_perturb', []) + num_token_perturb.detach().cpu().tolist()
                    del combined_outputs, combined_logits, combined_batch, batch, perturb_batch
                    progress.advance(gen_task) #? update progress
                dt_ratio = time.perf_counter() - t_ratio
                timing_profile["perturb_ratio_sec"] += dt_ratio
                split_timing["perturb_ratio_sec"] += dt_ratio

                eval_logs.update(tmp_logs)
                if eval_split == configs.dataset.split:
                    if has_retain_result:
                        retain_result = json.load(open(retain_result_path, 'r'))['eval_log_forget.json']
                        forget_quality = get_forget_quality_func(eval_logs, retain_result)
                        avg_gt_prob = get_forget_prob(eval_logs)
                        gt_probs = np.exp(-1 * np.array(eval_logs['avg_gt_loss']))
                        LOGGER.info("ForgetResult", 
                                    forget_quality=forget_quality['Forget Quality'], forget_proba=avg_gt_prob,
                        )
                        tmp_logs['forget truth ratio'] = forget_quality['Forget Truth Ratio']
                    else:
                        LOGGER.info("ForgetResult", note="skip_forget_quality(no retain_result provided)")

                #! Save intermediate results
                eval_logs.update(tmp_logs)
                eval_logs_by_task[f"{task_name}.json"] = eval_logs
                if not no_text_log:
                    save_name = os.path.join(OUTPUTDIR, f"{task_name}.json")
                    with open(save_name, "w") as f:
                        json.dump(eval_logs, f, indent=2)
                progress.advance(eval_task) #? update progress
            if torch.cuda.is_available() and os.getenv("TOFU_EMPTY_CACHE_PER_SPLIT", "1") == "1":
                t_cache = time.perf_counter()
                torch.cuda.empty_cache()
                dt_cache = time.perf_counter() - t_cache
                timing_profile["cache_clear_sec"] += dt_cache
                timing_profile["cache_clear_calls"] += 1
                split_timing["cache_clear_sec"] += dt_cache
                split_timing["cache_clear_calls"] += 1
            if profile_timing:
                timing_profile["splits"][eval_split] = split_timing
	 
        #! Final result
        if no_text_log:
            aggregated_logs = eval_logs_by_task
        else:
            aggregated_logs = {}
            for eval_split in eval_tasks:
                task_name = eval_split if eval_split != configs.dataset.split else "eval_log_forget"
                eval_logs = json.load(open(os.path.join(OUTPUTDIR, f"{task_name}.json"), 'r'))
                aggregated_logs[f"{task_name}.json"] = eval_logs

        model_utility = get_model_utility(aggregated_logs)
        if has_retain_result:
            retain_result = json.load(open(retain_result_path, 'r'))
            forget_quality = get_forget_quality(aggregated_logs, retain_result)
            forget_quality.pop('Forget Truth Ratio')
            aaggregate_stat = {**model_utility, **forget_quality}
        else:
            aaggregate_stat = model_utility

        #! Save final result 
        with open(os.path.join(OUTPUTDIR, "aggregate_stat.csv"), 'w') as csvfile:
            field_names = list(aaggregate_stat.keys())
            writer = csv.DictWriter(csvfile, fieldnames=field_names)
            writer.writeheader()
            writer.writerow(aaggregate_stat)

        if not no_text_log:
            # Save aggregated logs for convenient reuse as `retain_result`.
            with open(os.path.join(OUTPUTDIR, "eval_log_aggregated.json"), "w") as f:
                json.dump(aggregated_logs, f, indent=2)

        # Save routing summary if the model exposes routing stats.
        if hasattr(model, "dataset_stats") and isinstance(getattr(model, "dataset_stats", None), dict):
            try:
                routing = {
                    "metric": "fixed_sym_kl",
                    "threshold": float(getattr(model, "threshold", float("nan"))),
                    "max_new_tokens": int(getattr(model, "max_new_tokens", -1)),
                    "splits": {},
                }
                for s in eval_tasks:
                    s_stats = model.dataset_stats.get(s, {})
                    gen_stats = s_stats.get("generate", {})
                    total = int(gen_stats.get("total_calls", 0) or 0)
                    assist = int(gen_stats.get("assist_model", 0) or 0)
                    base = int(gen_stats.get("base_model", 0) or 0)
                    assist_rate = (assist / total) if total else None
                    base_rate = (base / total) if total else None

                    expected_forget = (s == configs.dataset.split)
                    entry = {
                        "expected_forget": expected_forget,
                        "generate_total": total,
                        "generate_assist": assist,
                        "generate_base": base,
                        "generate_assist_rate": assist_rate,
                        "generate_base_rate": base_rate,
                    }
                    if expected_forget and total:
                        entry["tpr"] = assist / total
                    if (not expected_forget) and total:
                        entry["fpr"] = assist / total
                        entry["tnr"] = base / total

                    # score stats (optional)
                    scores = gen_stats.get("cross_entropies", [])
                    if isinstance(scores, list) and scores:
                        entry["score_mean"] = float(np.mean(scores))
                        entry["score_std"] = float(np.std(scores))
                        entry["score_min"] = float(np.min(scores))
                        entry["score_max"] = float(np.max(scores))
                    routing["splits"][s] = entry

                with open(os.path.join(OUTPUTDIR, "routing_summary.json"), "w") as f:
                    json.dump(routing, f, indent=2)

                # Optional numeric-only dump for threshold stability bootstrap.
                # Enabled only when TOFU_DUMP_NPZ is set; default behavior unchanged.
                dump_npz = os.getenv("TOFU_DUMP_NPZ", "")
                if dump_npz:
                    if dump_npz in ("1", "true", "True", "yes", "YES"):
                        dump_npz_path = os.path.join(OUTPUTDIR, "routing_scores.npz")
                    else:
                        dump_npz_path = dump_npz

                    npz_dict = {}
                    split_names = []
                    for s in eval_tasks:
                        split_names.append(str(s))
                        s_stats = model.dataset_stats.get(s, {})
                        gen_stats = s_stats.get("generate", {})
                        scores = gen_stats.get("cross_entropies", [])
                        if isinstance(scores, list) and scores:
                            arr = np.asarray(scores, dtype=np.float32)
                        else:
                            arr = np.asarray([], dtype=np.float32)
                        key = f"scores_{str(s).replace('/', '_')}"
                        npz_dict[key] = arr

                    npz_dict["threshold"] = np.asarray([
                        float(getattr(model, "threshold", float("nan")))
                    ], dtype=np.float32)
                    npz_dict["split_names_json"] = np.asarray([
                        json.dumps(split_names, ensure_ascii=False)
                    ], dtype=np.str_)

                    dump_dir = os.path.dirname(dump_npz_path)
                    if dump_dir:
                        os.makedirs(dump_dir, exist_ok=True)
                    np.savez_compressed(dump_npz_path, **npz_dict)
                    print("tofu_dump_npz", dump_npz_path)
            except Exception:
                pass
        if profile_timing:
            with open(os.path.join(OUTPUTDIR, "timing_profile.json"), "w") as f:
                json.dump(timing_profile, f, indent=2)
