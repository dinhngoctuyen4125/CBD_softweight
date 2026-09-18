import json
import numpy as np
import os
import argparse
from collections import defaultdict

DEFAULT_METRIC_KEY = "cross_entropy"
AUTO_METRIC_PRIORITY = [
    "span_score",
    "sces_score",
    "escort_score",
    "feis_score",
    "cbd_weighted_kl",
    "max_token_ce",
    "cross_entropy",
]


def _available_metric_keys(data):
    keys = []
    for key in AUTO_METRIC_PRIORITY:
        if any(
            ((key in item) and (item.get(key) is not None))
            or (
                isinstance(item.get("cross_entropy_details"), dict)
                and item["cross_entropy_details"].get(key) is not None
            )
            for item in data
        ):
            keys.append(key)
    return keys


def _select_metric_key(data, metric_key):
    metric_key = str(metric_key or DEFAULT_METRIC_KEY).strip()
    available = _available_metric_keys(data)
    if metric_key.lower() == "auto":
        if not available:
            raise ValueError("No supported routing score keys found in input JSON")
        return available[0], available
    if metric_key not in available:
        raise ValueError(
            f"Metric key {metric_key!r} not found in input JSON. Available keys: {available if available else '[]'}"
        )
    return metric_key, available


def analyze_cross_entropy(file_path, metric_key=DEFAULT_METRIC_KEY):
    # 读取JSON文件
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    selected_metric_key, available_metric_keys = _select_metric_key(data, metric_key)
    cross_entropies = []
    for item in data:
        if item.get(selected_metric_key) is not None:
            cross_entropies.append(item[selected_metric_key])
            continue
        details = item.get("cross_entropy_details") or {}
        if details.get(selected_metric_key) is not None:
            cross_entropies.append(details[selected_metric_key])
    
    # 计算统计指标
    average = np.mean(cross_entropies)
    maximum = np.max(cross_entropies)
    minimum = np.min(cross_entropies)
    median = np.median(cross_entropies)
    
    # 计算区间分布（区间间隔为1）
    interval_counts = defaultdict(int)
    for ce in cross_entropies:
        interval = int(ce)
        interval_counts[interval] += 1
    
    # 输出结果
    print(f"交叉熵（Cross Entropy）统计分析")
    print(f"=========================")
    print(f"分数字段: {selected_metric_key}")
    print(f"平均值: {average:.4f}")
    print(f"最大值: {maximum:.4f}")
    print(f"最小值: {minimum:.4f}")
    print(f"中位数: {median:.4f}")
    # print(f"\n区间分布 (区间间隔为1):")
    # print(f"=========================")
    
    # # 对区间进行排序并显示
    # for interval in sorted(interval_counts.keys()):
    #     count = interval_counts[interval]
    #     percentage = (count / len(cross_entropies)) * 100
    #     print(f"[{interval}, {interval+1}): {count} 项 ({percentage:.2f}%)")
    
    return {
        "average": average,
        "maximum": maximum,
        "minimum": minimum,
        "median": median,
        "metric_key": selected_metric_key,
        "available_metric_keys": available_metric_keys,
        "interval_distribution": dict(interval_counts),
        "cross_entropies": cross_entropies
    }


def _candidate_thresholds(forget_data, retain_data):
    combined = np.concatenate([forget_data, retain_data]).astype(np.float64, copy=False)
    unique = np.unique(combined)
    unique.sort()
    if unique.size == 0:
        return unique
    # Add one value above max so that "predict none" is representable.
    above_max = np.nextafter(unique[-1], np.inf)
    return np.concatenate([unique, np.array([above_max], dtype=unique.dtype)])


def _metrics_at_threshold(forget_data, retain_data, threshold):
    forget_pred = forget_data >= threshold
    retain_pred = retain_data >= threshold

    tp = int(forget_pred.sum())
    fn = int((~forget_pred).sum())
    fp = int(retain_pred.sum())
    tn = int((~retain_pred).sum())

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tpr
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "threshold": float(threshold),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "accuracy": float(accuracy),
        "tpr": float(tpr),
        "fpr": float(fpr),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "gap": float(tpr - fpr),
        "forgotten_ratio": float(tpr),
        "retained_ratio": float(fpr),
    }


def find_optimal_threshold(forget_data, retain_data, optimize="accuracy", min_tpr=None, max_fpr=None):
    """
    寻找最佳阈值，区分遗忘数据和保留数据
    optimize:
      - gap: 最大化 (TPR - FPR)
      - accuracy: 最大化整体准确率
      - f1: 最大化 F1
    """
    candidate_thresholds = _candidate_thresholds(forget_data, retain_data)
    if candidate_thresholds.size == 0:
        raise ValueError("Empty threshold candidate set (no data?)")

    min_tpr = float(min_tpr) if min_tpr is not None else None
    max_fpr = float(max_fpr) if max_fpr is not None else None

    print("\n评估候选阈值...")
    best_any = None
    best_any_score = None
    best_constrained = None
    best_constrained_score = None

    for threshold in candidate_thresholds:
        m = _metrics_at_threshold(forget_data, retain_data, threshold)
        if optimize == "gap":
            score = m["gap"]
        elif optimize == "f1":
            score = m["f1"]
        else:
            score = m["accuracy"]

        # Tie-break: higher score, then higher TPR, then lower FPR.
        score_key = (float(score), float(m["tpr"]), -float(m["fpr"]))

        if best_any_score is None or score_key > best_any_score:
            best_any_score = score_key
            best_any = m

        ok = True
        if min_tpr is not None:
            ok = ok and (m["tpr"] >= min_tpr)
        if max_fpr is not None:
            ok = ok and (m["fpr"] <= max_fpr)
        if ok:
            if best_constrained_score is None or score_key > best_constrained_score:
                best_constrained_score = score_key
                best_constrained = m

    constraints_satisfied = best_constrained is not None
    best_details = best_constrained if constraints_satisfied else best_any
    best_threshold = float(best_details["threshold"])

    # 打印结果
    print("\n优化阈值分析结果:")
    print("============================")
    print(f"最佳阈值: {best_threshold:.4f} (optimize={optimize})")
    if min_tpr is not None or max_fpr is not None:
        print(
            "约束: "
            f"min_tpr={min_tpr if min_tpr is not None else 'None'}, "
            f"max_fpr={max_fpr if max_fpr is not None else 'None'} "
            f"-> satisfied={constraints_satisfied}"
        )
    print(f"在该阈值下:")
    print(f"  - 遗忘命中率(TPR): {best_details['tpr']*100:.2f}%")
    print(f"  - retain误报率(FPR): {best_details['fpr']*100:.2f}%")
    print(f"  - retain正确率(TNR): {(1.0-best_details['fpr'])*100:.2f}%")
    print(f"  - 准确率(Accuracy): {best_details['accuracy']*100:.2f}%")
    print(f"  - F1: {best_details['f1']*100:.2f}%")
    print(f"  - 差距得分(TPR-FPR): {best_details['gap']:.4f}")
    
    return {
        "best_threshold": float(best_threshold),
        "optimize": optimize,
        "constraints": {"min_tpr": min_tpr, "max_fpr": max_fpr},
        "constraints_satisfied": bool(constraints_satisfied),
        "tp": int(best_details["tp"]),
        "fn": int(best_details["fn"]),
        "fp": int(best_details["fp"]),
        "tn": int(best_details["tn"]),
        "accuracy": best_details["accuracy"] * 100,
        "f1": best_details["f1"] * 100,
        "forgotten_identification_rate": best_details['tpr'] * 100,
        "retained_misidentification_rate": best_details['fpr'] * 100,
        "retained_correct_rate": (1.0 - best_details["fpr"]) * 100,
        "gap_score": best_details["gap"],
    }

def analyze_threshold_performance(datasets, threshold):
    """分析给定阈值在各个数据集上的表现"""
    results = {}
    
    for name, data in datasets.items():
        # 计算大于阈值的比例
        above_threshold = np.mean(data >= threshold) * 100
        results[name] = above_threshold
        print(f"数据集 {name} 中大于阈值的比例: {above_threshold:.2f}%")
    
    return results

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='分析交叉熵数据')
    parser.add_argument('--data-dir', '-d', type=str, default='.',
                        help='数据文件所在的目录路径 (默认为当前目录)')
    parser.add_argument('--forget-split', '-f', type=str, default='forget05_perturbed_1',
                        help='要分析的forget数据分割 (默认为forget05_perturbed_1)')
    parser.add_argument('--retain-split', '-r', type=str, default='retain90',
                        help='要分析的retain数据分割 (默认为retain90)')
    parser.add_argument('--optimize', type=str, default='accuracy',
                        choices=['accuracy', 'gap', 'f1'],
                        help='阈值选择优化目标 (accuracy/gap/f1)')
    parser.add_argument('--min-tpr', type=float, default=None,
                        help='阈值约束：遗忘命中率下限 (0~1). 例如 0.9 表示 TPR>=90%%')
    parser.add_argument('--max-fpr', type=float, default=None,
                        help='阈值约束：retain 误报率上限 (0~1). 例如 0.1 表示 FPR<=10%%')
    parser.add_argument('--metric-key', type=str, default=DEFAULT_METRIC_KEY,
                        help="要分析的路由分数字段名。默认 cross_entropy；也可用 max_token_ce/feis_score/cbd_weighted_kl/escort_score/sces_score/span_score/auto")
    args = parser.parse_args()
    
    # 数据集文件列表
    base_files = [
        args.forget_split,
        args.retain_split,
    ]
    
    # 存储每个数据集的交叉熵
    datasets = {}
    
    print(f"正在从目录 '{args.data_dir}' 读取数据文件...")
    
    # 读取并分析每个数据集
    for file in base_files:
        file_path = os.path.join(args.data_dir, f"tinyllama_comparison_results_{file}.json")
        print(f'\n=========={file}==========')
        print(f"读取文件: {file_path}")
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"警告: 文件 {file_path} 不存在，跳过该数据集")
            continue
            
        results = analyze_cross_entropy(file_path, metric_key=args.metric_key)
        datasets[file] = np.array(results["cross_entropies"])
    
    # 检查是否有足够的数据集
    if args.forget_split not in datasets:
        print(f"错误: 未找到遗忘数据集 ({args.forget_split})，无法进行分析")
        exit(1)
    
    non_forgotten_datasets = [name for name in [args.retain_split] if name in datasets]
    if not non_forgotten_datasets:
        print("错误: 未找到任何非遗忘数据集，无法进行分析")
        exit(1)
    
    # 找到最佳阈值
    print("\n\n======== 寻找最佳阈值 ========")
    # 合并所有非遗忘数据
    non_forgotten_data = np.concatenate([datasets[name] for name in non_forgotten_datasets])
    
    optimal_results = find_optimal_threshold(
        forget_data=datasets[args.forget_split],
        retain_data=non_forgotten_data,
        optimize=args.optimize,
        min_tpr=args.min_tpr,
        max_fpr=args.max_fpr,
    )
    
    # 分析不同阈值在各个数据集上的表现
    print("\n\n======== 各个数据集在最佳阈值下的表现 ========")
    threshold_performance = analyze_threshold_performance(
        datasets=datasets,
        threshold=optimal_results["best_threshold"]
    )
    
    # 保存结果到指定目录
    output_file = os.path.join(args.data_dir, "threshold_analysis_results.json")
    with open(output_file, "w") as f:
        json.dump({
            "optimal_threshold": optimal_results,
            "dataset_performance": threshold_performance,
            "metric_key": args.metric_key,
            "available_metric_keys": results["available_metric_keys"],
        }, f, indent=2)
    
    print(f"\n分析结果已保存到: {output_file}")

    # for threshold in np.arange(5.0, 20.0, 0.1):
    #     print(f'========thereshold:{threshold}=======')
    #     analyze_threshold_performance(datasets, threshold)
