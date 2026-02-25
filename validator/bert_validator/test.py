# -*- coding:utf-8 -*-

import torch
from tqdm import tqdm
from utils import read_data, MyDataset
from config import parsers
from torch.utils.data import DataLoader
from model import MyModel
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import json
from datetime import datetime


# ===== id → label 映射（真实语义标签）=====
ID2LABEL = {
    0: "错误",
    1: "正确"
}


def write_log(log_file, data):
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def save_predictions(file_path, records):
    """
    保存预测结果为 jsonl
    """
    with open(file_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def test_data(log_file=None):
    args = parsers()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # ===== 数据 =====
    test_text, test_label = read_data(args.test_file)
    test_dataset = MyDataset(test_text, test_label, args.max_len)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    # ===== 模型 =====
    model = MyModel().to(device)
    model.load_state_dict(
        torch.load(args.save_model_best, map_location=device)
    )
    model.eval()

    all_pred, all_true = [], []
    pred_records = []

    with torch.no_grad():
        idx = 0
        for batch_text, batch_label in tqdm(test_dataloader, desc="Testing"):
            batch_label = batch_label.to(device)

            logits = model(batch_text)
            pred = torch.argmax(logits, dim=1)

            pred = pred.cpu().numpy().tolist()
            label = batch_label.cpu().numpy().tolist()

            for p, y in zip(pred, label):
                pred_records.append({
                    "text": test_text[idx],
                    "label_id": y,
                    "label": ID2LABEL[y],
                    "pred_id": p,
                    "pred": ID2LABEL[p],
                    "correct": int(p == y)
                })
                idx += 1

            all_pred.extend(pred)
            all_true.extend(label)

    # ===== 评估指标（基于 id）=====
    accuracy = accuracy_score(all_true, all_pred)
    precision = precision_score(all_true, all_pred, zero_division=0)
    recall = recall_score(all_true, all_pred, zero_division=0)
    f1 = f1_score(all_true, all_pred, zero_division=0)

    print("====== Test Metrics ======")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1       : {f1:.4f}")

    # ===== 保存预测结果 =====
    pred_file = args.save_model_best.replace(".pth", "_pred.jsonl")
    save_predictions(pred_file, pred_records)
    print(f"✅ 预测结果已保存至: {pred_file}")

    # ===== 写入日志 =====
    if log_file is not None:
        write_log(
            log_file,
            {
                "type": "test",
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "test_size": len(test_text),
                "model": args.save_model_best,
                "pred_file": pred_file,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
        )


if __name__ == "__main__":
    test_data()
