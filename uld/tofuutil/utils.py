import yaml
import copy
import numpy as np
from scipy.stats import sem, hmean, ks_2samp

_EXP_CLIP = 60.0

def _safe_exp_diff(lhs, rhs):
    diff = np.array(lhs, dtype=np.float64) - np.array(rhs, dtype=np.float64)
    diff = np.nan_to_num(diff, nan=0.0, posinf=_EXP_CLIP, neginf=-_EXP_CLIP)
    diff = np.clip(diff, -_EXP_CLIP, _EXP_CLIP)
    return np.exp(diff)

def _safe_prob(avg_gt_loss, average_perturb_loss=None):
    true_loss = np.array(avg_gt_loss, dtype=np.float64)
    true_loss = np.nan_to_num(true_loss, nan=_EXP_CLIP, posinf=_EXP_CLIP, neginf=-_EXP_CLIP)
    true_prob = np.exp(-np.clip(true_loss, -_EXP_CLIP, _EXP_CLIP))

    if average_perturb_loss is None:
        finite = true_prob[np.isfinite(true_prob)]
        return float(np.mean(finite)) if finite.size else 0.0

    false_loss = np.array(average_perturb_loss, dtype=np.float64)
    false_loss = np.nan_to_num(false_loss, nan=_EXP_CLIP, posinf=_EXP_CLIP, neginf=-_EXP_CLIP)
    false_prob = np.exp(-np.clip(false_loss, -_EXP_CLIP, _EXP_CLIP))
    all_prob = np.concatenate([np.expand_dims(true_prob, axis=-1), false_prob], axis=1).sum(-1)
    gt_ratio = np.divide(true_prob, all_prob, out=np.zeros_like(true_prob), where=all_prob > 0)
    finite = gt_ratio[np.isfinite(gt_ratio)]
    return float(np.mean(finite)) if finite.size else 0.0

def get_model_identifiers_from_yaml(model_family):
    #path is model_configs.yaml
    '''
    models:
        llama2-7b:
            hf_key: "NousResearch/Llama-2-7b-chat-hf"
            question_start_tag: "[INST] "
            question_end_tag: " [/INST] "
            answer_tag: ""
            start_of_sequence_token: "<s>"
    '''
    model_configs  = {}
    with open("tofuconfig/model_config.yaml", "r") as f:
        model_configs = yaml.load(f, Loader=yaml.FullLoader)
    return model_configs[model_family]

def merge_dicts(a, b):
    """ Recursively merges dict b into a deep copy of dict a """
    # Create a deep copy of a to avoid modifying it in place
    a_copy = copy.deepcopy(a)

    for key, value in b.items():
        if key in a_copy:
            if isinstance(a_copy[key], dict) and isinstance(value, dict):
                a_copy[key] = merge_dicts(a_copy[key], value)
            elif isinstance(a_copy[key], list) and isinstance(value, list):
                a_copy[key] = a_copy[key] + value  # or use other logic to merge lists
            else:
                a_copy[key] = value  # Overwrite value from b into a_copy
        else:
            a_copy[key] = value

    return a_copy

def get_total_len(name, forget_rate):
    if name == 'eval_real_author_wo_options.json':
        return 100
    elif name == 'eval_real_world_wo_options.json':
        return 117
    elif name == 'eval_log.json':
        return 300
    else:
        if forget_rate == 'forget01':
            return 40
        elif forget_rate == 'forget05':
            return 200
        else:
            return 300

def interleave(a, b, size):
    assert len(a) == len(b)
    assert size > 0
    c = []
    for i in range(0, len(a), size):
        c.extend(a[i:i+size])
        c.extend(b[i:i+size])
    return c

# PLEASE BE VERY VERY CAREFUL HERE
# This code, although takes num_processes as an argument, it in fact only supports num_processes=2
# Future improvement should support interleave for more than 2 processes
# also, small_bsz = large_bsz//4 is hardcoded, which is only true for our experiments
# because when we construct perturb and paraphrase data_loader, we set batch_size=large_bsz//4 specifically 
def interleave_eval_result_dict(eval_result_dict, forget_rate, large_bsz, num_processes=2):
    small_bsz = large_bsz//4
    for k, v in eval_result_dict.items():
        # each v corresponds to one ckpt
        for metric, value in v.items():
            bsz = small_bsz if 'perturb' in metric or 'paraphrase' in metric else large_bsz
            total_len = get_total_len(k, forget_rate)
            # split in two
            a = value[0:len(value)//2]
            b = value[len(value)//2:]
            eval_result_dict[k][metric] = interleave(a, b, bsz)[:total_len]
    return eval_result_dict

def get_model_utility(eval_result_dict):
    eval_task_dict = {
        'retain_perturbed.json': 'Retain',
        'real_authors_perturbed.json': 'Real Authors',
        'world_facts_perturbed.json': 'Real World',
        'eval_log_forget.json': 'Forget'
    }
    eval_tasks = list(eval_task_dict.keys())
    metrics = ['ROUGE', 'Probability', 'Truth Ratio']

    output_result = {}
    for eval_task in eval_tasks:
        for metric in metrics:
            output_result[eval_task_dict[eval_task] + ' ' + metric] = []

    # k is different files
    for k, v in eval_result_dict.items():
        # getting Probability
        if 'retain_perturbed.json' in k:
            avg_gt_prob = _safe_prob(eval_result_dict[k]['avg_gt_loss'])
        else:
            avg_gt_prob = _safe_prob(
                eval_result_dict[k]['avg_gt_loss'],
                eval_result_dict[k]['average_perturb_loss'],
            )
        output_result[f'{eval_task_dict[k]} Probability'] = avg_gt_prob

        # getting ROUGE
        avg_rouge = np.array(eval_result_dict[k]['rougeL_recall']).mean()
        output_result[f'{eval_task_dict[k]} ROUGE'] = avg_rouge

        # getting Truth Ratio
        avg_paraphrase_np_values = np.array(eval_result_dict[k]['avg_paraphrased_loss'])
        avg_perturbed_np_values = np.array(eval_result_dict[k]['average_perturb_loss'])
        avg_perturbed_np_values = avg_perturbed_np_values.mean(axis=-1)

        curr_stat_1 = _safe_exp_diff(avg_perturbed_np_values, avg_paraphrase_np_values)
        curr_stat_1 = np.where(np.isfinite(curr_stat_1) & (curr_stat_1 > 0), curr_stat_1, 1.0)
        if 'forget' in k:
            truth_vals = np.minimum(curr_stat_1, 1/curr_stat_1)
            finite = truth_vals[np.isfinite(truth_vals)]
            paraphrased_perturb_ratio = float(np.mean(finite)) if finite.size else 0.0
        else:
            truth_vals = np.maximum(0, 1 - 1/curr_stat_1)
            finite = truth_vals[np.isfinite(truth_vals)]
            paraphrased_perturb_ratio = float(np.mean(finite)) if finite.size else 0.0
        output_result[f'{eval_task_dict[k]} Truth Ratio'] = paraphrased_perturb_ratio

    model_utility_cands = []
    for k, v in output_result.items():
        if 'Forget' not in k:
            if v == []:
                model_utility_cands.append(0)
            else:
                model_utility_cands.append(v)
    output_result['Model Utility'] = hmean(model_utility_cands)
    return output_result

def get_forget_quality(unlearn_result, retain_result):
    unlearn_forget_result = unlearn_result['eval_log_forget.json']
    retain_forget_result = retain_result['eval_log_forget.json']
    
    unlearn_paraphrase_np_values = np.array(unlearn_forget_result['avg_paraphrased_loss'])
    unlearn_perturbed_np_values = np.array(unlearn_forget_result['average_perturb_loss'])
    unlearn_perturbed_np_values = unlearn_perturbed_np_values.mean(axis=-1)

    retain_paraphrase_np_values = np.array(retain_forget_result['avg_paraphrased_loss'])
    retain_perturbed_np_values = np.array(retain_forget_result['average_perturb_loss'])
    retain_perturbed_np_values = retain_perturbed_np_values.mean(axis=-1)

    unlearn_truth_ratio = _safe_exp_diff(unlearn_perturbed_np_values, unlearn_paraphrase_np_values)
    retain_truth_ratio = _safe_exp_diff(retain_perturbed_np_values, retain_paraphrase_np_values)

    unlearn_truth_ratio = unlearn_truth_ratio[np.isfinite(unlearn_truth_ratio)]
    retain_truth_ratio = retain_truth_ratio[np.isfinite(retain_truth_ratio)]
    if len(unlearn_truth_ratio) >= 2 and len(retain_truth_ratio) >= 2:
        test_res = ks_2samp(unlearn_truth_ratio, retain_truth_ratio)
        pvalue, stat = float(test_res.pvalue), float(test_res.statistic)
    else:
        pvalue, stat = 1.0, 0.0
    return {'Forget Quality': pvalue, 'KS Test PVal Forget': pvalue, 'KS Test Forget': stat, "Forget Truth Ratio": unlearn_truth_ratio.tolist()}

def get_forget_quality_func(unlearn_forget_result, retain_forget_result, para_key='avg_paraphrased_loss'):
    # import ipdb; ipdb.set_trace()
    unlearn_paraphrase_np_values = np.array(unlearn_forget_result[para_key])
    le = len(unlearn_paraphrase_np_values)
    unlearn_perturbed_np_values = np.array(unlearn_forget_result['average_perturb_loss'])[:le]
    unlearn_perturbed_np_values = unlearn_perturbed_np_values.mean(axis=-1)

    retain_paraphrase_np_values = np.array(retain_forget_result[para_key])
    le = len(retain_paraphrase_np_values)
    retain_perturbed_np_values = np.array(retain_forget_result['average_perturb_loss'])[:le]
    retain_perturbed_np_values = retain_perturbed_np_values.mean(axis=-1)

    unlearn_truth_ratio = _safe_exp_diff(unlearn_perturbed_np_values, unlearn_paraphrase_np_values)
    retain_truth_ratio = _safe_exp_diff(retain_perturbed_np_values, retain_paraphrase_np_values)
    unlearn_truth_ratio = unlearn_truth_ratio[np.isfinite(unlearn_truth_ratio)]
    retain_truth_ratio = retain_truth_ratio[np.isfinite(retain_truth_ratio)]
    if len(unlearn_truth_ratio) >= 2 and len(retain_truth_ratio) >= 2:
        test_res = ks_2samp(unlearn_truth_ratio, retain_truth_ratio)
        pvalue, stat = float(test_res.pvalue), float(test_res.statistic)
    else:
        pvalue, stat = 1.0, 0.0
    return {'Forget Quality': pvalue, 'KS Test PVal Forget': pvalue, 'KS Test Forget': stat, "Forget Truth Ratio": unlearn_truth_ratio.tolist()}


def get_truth_ratio(result):
    paraphrase_np_values = np.array(result['avg_paraphrased_loss'])
    perturbed_np_values = np.array(result['average_perturb_loss'])
    perturbed_np_values = perturbed_np_values.mean(axis=-1)
    truth_ratio = _safe_exp_diff(perturbed_np_values, paraphrase_np_values)
    return truth_ratio

def get_truth_ratio2(result):
    paraphrase_np_values = np.array(result['avg_gt_loss'])
    perturbed_np_values = np.array(result['average_perturb_loss'])[:len(paraphrase_np_values)]
    perturbed_np_values = perturbed_np_values.mean(axis=-1)
    truth_ratio = _safe_exp_diff(perturbed_np_values, paraphrase_np_values)
    return truth_ratio


def get_forget_prob(result):
    return _safe_prob(result['avg_gt_loss'], result['average_perturb_loss'])
