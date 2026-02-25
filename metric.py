# -*- coding: utf-8 -*-
# python metric.py
import os
import json
from collections import defaultdict
from datasets import load_dataset

# 解决pred被截断的问题
def extract_complete_json_objects(text):
    objs = []
    brace_count = 0
    start = None

    for i, ch in enumerate(text):
        if ch == "{":
            if brace_count == 0:
                start = i
            brace_count += 1
        elif ch == "}":
            brace_count -= 1
            if brace_count == 0 and start is not None:
                obj_str = text[start:i+1]
                try:
                    objs.append(json.loads(obj_str))
                except json.JSONDecodeError:
                    print(f"子对象有括号，但内部结构错误❌️：{obj_str}")
                    pass
                start = None
    # print(f"\n🧠   抢救成功：{objs}")
    return objs


def clean_output(text):
    # 去掉 ```json ``` 包裹
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()

    return cleaned


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
        pred_str = clean_output(sample["pred"])    # 去掉 ```json ``` 包裹
        task = sample["task"]
        id = sample["id"]
        if task == "topic":
            gold_json = gold_str.strip() if isinstance(gold_str, str) else ""
            pred_json = pred_str.strip() if isinstance(pred_str, str) else ""
        else:
            try:
                gold_json = json.loads(gold_str)    # 肯定正确
                pred_json = json.loads(pred_str)    # 可能解析错误
            except Exception as e:
                # print(f"\n🚀    {id}  需要抢救，原始 pred_str：{pred_str}")
                error_output += 1
                pred_json = extract_complete_json_objects(pred_str)

        gold_samples.append(gold_json)
        pred_samples.append(pred_json)
        tasks.append(task)
        
    print(f"\n不能解析的样本数量：{error_output}")
    return gold_samples, pred_samples, tasks



def parse_json_output(text: str):
    """解析JSON输出"""
    try:
        if text.strip():
            return json.loads(text)
        return []
    except Exception:
        return []


def compute_metrics(gold_samples, pred_samples, tasks):
    """计算评估指标"""
    
    results = {}

    # ========= validator =========
    validator_correct, validator_total = 0, 0
    validator_tp = validator_fp = validator_fn = validator_tn = 0

    # ========= topic =========
    topic_correct, topic_total = 0, 0
    topic_tp = topic_fp = topic_fn = topic_tn = 0

    # ========= ner =========
    ner_tp = ner_pred_total = ner_gold_total = 0
    ner_type_stats = defaultdict(lambda: {"tp": 0, "pred": 0, "gold": 0})

    # ========= re =========
    re_tp = re_pred_total = re_gold_total = 0
    re_type_stats = defaultdict(lambda: {"tp": 0, "pred": 0, "gold": 0})

    # ========= 逐条计算 =========
    for gold, pred, task in zip(gold_samples, pred_samples, tasks):
        if task == "validator":
            validator_total += 1
            pred_text = pred if isinstance(pred, str) else str(pred)
            gold_text = gold if isinstance(gold, str) else str(gold)
            if pred_text.strip() == gold_text.strip():
                validator_correct += 1
            
            if gold == "正确" and pred == "正确":
                validator_tp += 1
            elif gold == "错误" and pred == "正确":
                validator_fp += 1
            elif gold == "正确" and pred == "错误":
                validator_fn += 1
            elif gold == "错误" and pred == "错误":
                validator_tn += 1

        elif task == "topic":
            topic_total += 1
            pred_text = pred if isinstance(pred, str) else str(pred)
            gold_text = gold if isinstance(gold, str) else str(gold)
            if pred_text.strip() == gold_text.strip():
                topic_correct += 1
            
            if gold == "相关" and pred == "相关":
                topic_tp += 1
            elif gold == "不相关" and pred == "相关":
                topic_fp += 1
            elif gold == "相关" and pred == "不相关":
                topic_fn += 1
            elif gold == "不相关" and pred == "不相关":
                topic_tn += 1

        elif task == "ner":
            pred_json = pred if isinstance(pred, list) else []
            gold_json = gold if isinstance(gold, list) else []

            # 转换为实体集合
            pred_set = set()
            for e in pred_json:
                if isinstance(e, dict):
                    entity = e.get("entity", "").strip()
                    type = e.get("type", "").strip()
                    if entity and type:
                        pred_set.add(f"{entity}::{type}")
                        ner_type_stats[type]["pred"] += 1

            gold_set = set()
            for e in gold_json:
                if isinstance(e, dict):
                    entity = e.get("entity", "").strip()
                    type = e.get("type", "").strip()
                    if entity and type:
                        gold_set.add(f"{entity}::{type}")
                        ner_type_stats[type]["gold"] += 1

            # 计算总体指标
            correct_set = pred_set & gold_set
            ner_tp += len(correct_set)
            ner_pred_total += len(pred_set)
            ner_gold_total += len(gold_set)

            # 按类型统计正确数
            for entity_str in correct_set:
                type = entity_str.split("::")[1] if "::" in entity_str else ""
                if type:
                    ner_type_stats[type]["tp"] += 1

        elif task == "re":
            pred_json = pred if isinstance(pred, list) else []
            gold_json = gold if isinstance(gold, list) else []

            # 转换为关系集合
            pred_set = set()
            for r in pred_json:
                if isinstance(r, dict):
                    head = r.get("head", "").strip()
                    relation = r.get("relation", "").strip()
                    tail = r.get("tail", "").strip()
                    if head and relation and tail:
                        pred_set.add(f"{head}::{relation}::{tail}")
                        re_type_stats[relation]["pred"] += 1

            gold_set = set()
            for r in gold_json:
                if isinstance(r, dict):
                    head = r.get("head", "").strip()
                    relation = r.get("relation", "").strip()
                    tail = r.get("tail", "").strip()
                    if head and relation and tail:
                        gold_set.add(f"{head}::{relation}::{tail}")
                        re_type_stats[relation]["gold"] += 1

            # 计算总体指标
            correct_set = pred_set & gold_set
            re_tp += len(correct_set)
            re_pred_total += len(pred_set)
            re_gold_total += len(gold_set)

            # 按类型统计正确数
            for rel_str in correct_set:
                relation = rel_str.split("::")[1] if "::" in rel_str else ""
                if relation:
                    re_type_stats[relation]["tp"] += 1

    # ===== 汇总指标 =====
    
    # validator 准确率
    if validator_total > 0:
        precision = validator_tp / (validator_tp + validator_fp) if validator_tp + validator_fp > 0 else 0.0
        recall = validator_tp / (validator_tp + validator_fn) if validator_tp + validator_fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

        results["validator"] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": validator_correct / validator_total,
            "correct": validator_correct,
            "total": validator_total,
            "tp": validator_tp,
            "fp": validator_fp,
            "fn": validator_fn,
            "tn": validator_tn
        }


    # Topic 准确率
    if topic_total > 0:
        precision = topic_tp / (topic_tp + topic_fp) if topic_tp + topic_fp > 0 else 0.0
        recall = topic_tp / (topic_tp + topic_fn) if topic_tp + topic_fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

        results["topic"] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": topic_correct / topic_total,
            "correct": topic_correct,
            "total": topic_total,
            "tp": topic_tp,
            "fp": topic_fp,
            "fn": topic_fn,
            "tn": topic_tn
        }

    # NER 指标
    if ner_gold_total > 0:
        # Micro 指标
        ner_precision = ner_tp / ner_pred_total if ner_pred_total > 0 else 0.0
        ner_recall = ner_tp / ner_gold_total if ner_gold_total > 0 else 0.0
        ner_f1 = 2 * ner_precision * ner_recall / (ner_precision + ner_recall) if (ner_precision + ner_recall) > 0 else 0.0
        
        results["ner"] = {
            "micro": {
                "precision": ner_precision,
                "recall": ner_recall,
                "f1": ner_f1
            },
            "tp": ner_tp,
            "pred_total": ner_pred_total,
            "gold_total": ner_gold_total
        }
        
        # 每个实体类型的指标
        type_metrics = {}
        for type, stats in ner_type_stats.items():
            if stats["pred"] > 0 or stats["gold"] > 0:
                type_precision = stats["tp"] / stats["pred"] if stats["pred"] > 0 else 0.0
                type_recall = stats["tp"] / stats["gold"] if stats["gold"] > 0 else 0.0
                type_f1 = 2 * type_precision * type_recall / (type_precision + type_recall) if (type_precision + type_recall) > 0 else 0.0
                
                type_metrics[type] = {
                    "precision": type_precision,
                    "recall": type_recall,
                    "f1": type_f1,
                    "tp": stats["tp"],
                    "pred": stats["pred"],
                    "gold": stats["gold"]
                }
        
        results["ner"]["per_type"] = type_metrics

    # RE 指标
    if re_gold_total > 0:
        # Micro 指标
        re_precision = re_tp / re_pred_total if re_pred_total > 0 else 0.0
        re_recall = re_tp / re_gold_total if re_gold_total > 0 else 0.0
        re_f1 = 2 * re_precision * re_recall / (re_precision + re_recall) if (re_precision + re_recall) > 0 else 0.0
        
        results["re"] = {
            "micro": {
                "precision": re_precision,
                "recall": re_recall,
                "f1": re_f1
            },
            "tp": re_tp,
            "pred_total": re_pred_total,
            "gold_total": re_gold_total
        }
        
        # 每个关系类型的指标
        type_metrics = {}
        for relation, stats in re_type_stats.items():
            if stats["pred"] > 0 or stats["gold"] > 0:
                type_precision = stats["tp"] / stats["pred"] if stats["pred"] > 0 else 0.0
                type_recall = stats["tp"] / stats["gold"] if stats["gold"] > 0 else 0.0
                type_f1 = 2 * type_precision * type_recall / (type_precision + type_recall) if (type_precision + type_recall) > 0 else 0.0
                
                type_metrics[relation] = {
                    "precision": type_precision,
                    "recall": type_recall,
                    "f1": type_f1,
                    "tp": stats["tp"],
                    "pred": stats["pred"],
                    "gold": stats["gold"]
                }
        
        results["re"]["per_type"] = type_metrics

    # 综合指标
    results["total_samples"] = len(pred_samples)
    results["task_distribution"] = {
        "validator": validator_total,
        "topic": topic_total,
        "ner": sum(1 for t in tasks if t == "ner"),
        "re": sum(1 for t in tasks if t == "re")
    }

    return results


def format_metrics(metrics: dict) -> str:
    """格式化输出评估指标"""
    output = []
    output.append("=" * 50)
    output.append("评估结果汇总")
    output.append("=" * 50)

    # validator 结果
    if "validator" in metrics:
        validator = metrics["validator"]
        output.append(f"\n[validator 验证]")
        output.append(f"精确率: {validator['precision']:.4f}")
        output.append(f"召回率: {validator['recall']:.4f} ")
        output.append(f"F1分数: {validator['f1']:.4f} ")
        output.append(f"准确率: {validator['accuracy']:.4f} ({validator['correct']}/{validator['total']})")
        output.append(f"TP: {validator['tp']}, FP: {validator['fp']}, FN: {validator['fn']}, TN: {validator['tn']}")
    
    # Topic 结果
    if "topic" in metrics:
        topic = metrics["topic"]
        output.append(f"\n[Topic 分类]")
        output.append(f"精确率: {topic['precision']:.4f}")
        output.append(f"召回率: {topic['recall']:.4f} ")
        output.append(f"F1分数: {topic['f1']:.4f} ")
        output.append(f"准确率: {topic['accuracy']:.4f} ({topic['correct']}/{topic['total']})")
        output.append(f"TP: {topic['tp']}, FP: {topic['fp']}, FN: {topic['fn']}, TN: {topic['tn']}")
    
    # NER 结果
    if "ner" in metrics:
        ner = metrics["ner"]
        output.append(f"\n[NER 实体识别]")
        output.append(f"Micro-Precision: {ner['micro']['precision']:.4f}")
        output.append(f"Micro-Recall: {ner['micro']['recall']:.4f}")
        output.append(f"Micro-F1: {ner['micro']['f1']:.4f}")
        output.append(f"总预测数: {ner['pred_total']}, 总标签数: {ner['gold_total']}, 正确数: {ner['tp']}")
        
        if ner['per_type']:
            output.append("\n按实体类型统计:")
            for type, type_metrics in ner['per_type'].items():
                output.append(f"  {type}: P={type_metrics['precision']:.4f}, "
                            f"R={type_metrics['recall']:.4f}, F1={type_metrics['f1']:.4f} "
                            f"(TP={type_metrics['tp']}, P={type_metrics['pred']}, G={type_metrics['gold']})")
    
    # RE 结果
    if "re" in metrics:
        re = metrics["re"]
        output.append(f"\n[RE 关系抽取]")
        output.append(f"Micro-Precision: {re['micro']['precision']:.4f}")
        output.append(f"Micro-Recall: {re['micro']['recall']:.4f}")
        output.append(f"Micro-F1: {re['micro']['f1']:.4f}")
        output.append(f"总预测数: {re['pred_total']}, 总标签数: {re['gold_total']}, 正确数: {re['tp']}")
        
        if re['per_type']:
            output.append("\n按关系类型统计:")
            for relation, type_metrics in re['per_type'].items():
                output.append(f"  {relation}: P={type_metrics['precision']:.4f}, "
                            f"R={type_metrics['recall']:.4f}, F1={type_metrics['f1']:.4f} "
                            f"(TP={type_metrics['tp']}, P={type_metrics['pred']}, G={type_metrics['gold']})")
    
    output.append(f"\n总样本数: {metrics['total_samples']}")
    output.append(f"任务分布: {metrics['task_distribution']}")
    output.append("=" * 50)
    
    return "\n".join(output)


def main():
    import argparse, os

    parser = argparse.ArgumentParser(description="Evaluation Metrics Script")
    parser.add_argument("--pred", type=str, required=True, help="prediction json file path")
    parser.add_argument("--metric", type=str, required=True, help="metric output json path")

    args = parser.parse_args()

    print("加载预测数据...")
    gold_samples, pred_samples, tasks = load_triples_from_file(args.pred)

    assert len(gold_samples) == len(pred_samples), \
        f"样本数不一致：{len(gold_samples)} vs {len(pred_samples)}"


    metrics = compute_metrics(gold_samples, pred_samples, tasks)

    print(format_metrics(metrics))

    os.makedirs(os.path.dirname(args.metric), exist_ok=True)
    with open(args.metric, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\n详细评估指标已保存至: {args.metric}")



if __name__ == "__main__":
    main()