# -*- coding: utf-8 -*-
# python metric.py
import os
import json
from collections import defaultdict
from datasets import load_dataset


# ---------- 加载 dataset ----------
def load_triples_from_file(file_path: str):
    data_dir = os.path.dirname(file_path)
    data_file = os.path.basename(file_path)

    dct = load_dataset(
        path="json",
        data_dir=data_dir,
        data_files=data_file,
        split="train"
    )

    gold_samples, pred_samples, tasks = [], [], []

    error_output = 0
    for sample in dct:
        gold_str = sample["conversations"][1]["content"]
        pred_str = sample["pred"]
        task = sample["task"]
        id = sample["id"]

        gold_json = gold_str.strip() if isinstance(gold_str, str) else ""
        pred_json = pred_str.strip() if isinstance(pred_str, str) else ""

        gold_samples.append(gold_json)
        pred_samples.append(pred_json)
        tasks.append(task)
        
    return gold_samples, pred_samples, tasks


def compute_metrics(gold_samples, pred_samples):
    """
    二分类评估：
    label ∈ {正确, 错误}
    正类 = 正确
    """

    tp = fp = fn = tn = 0

    for gold, pred in zip(gold_samples, pred_samples):
        gold_label = str(gold).strip()
        pred_label = str(pred).strip()

        if gold_label == "正确" and pred_label == "正确":
            tp += 1
        elif gold_label == "错误" and pred_label == "正确":
            fp += 1
        elif gold_label == "正确" and pred_label == "错误":
            fn += 1
        elif gold_label == "错误" and pred_label == "错误":
            tn += 1
        else:
            # 非法标签，直接忽略
            pass

    total = tp + fp + fn + tn

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) \
                if (precision + recall) > 0 else 0.0
    accuracy  = (tp + tn) / total if total > 0 else 0.0

    return {
        "binary_classification": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "total": total
        }
    }


def format_metrics(metrics: dict) -> str:
    m = metrics["binary_classification"]

    output = []
    output.append("=" * 50)
    output.append("三元组正确性（二分类）评估结果")
    output.append("=" * 50)
    output.append(f"Precision : {m['precision']:.4f}")
    output.append(f"Recall    : {m['recall']:.4f}")
    output.append(f"F1-score  : {m['f1']:.4f}")
    output.append(f"Accuracy  : {m['accuracy']:.4f}")
    output.append("")
    output.append(f"TP: {m['tp']}, FP: {m['fp']}, FN: {m['fn']}, TN: {m['tn']}")
    output.append(f"Total samples: {m['total']}")
    output.append("=" * 50)

    return "\n".join(output)



def main():
    import argparse, os

    parser = argparse.ArgumentParser(description="Evaluation Metrics Script")
    parser.add_argument("--pred", type=str, required=True, help="prediction json file path")
    parser.add_argument("--metric", type=str, required=True, help="metric output json path")

    args = parser.parse_args()

    print("加载预测数据...")
    gold_samples, pred_samples, _ = load_triples_from_file(args.pred)
    metrics = compute_metrics(gold_samples, pred_samples)


    print(format_metrics(metrics))

    os.makedirs(os.path.dirname(args.metric), exist_ok=True)
    with open(args.metric, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\n详细评估指标已保存至: {args.metric}")



if __name__ == "__main__":
    main()