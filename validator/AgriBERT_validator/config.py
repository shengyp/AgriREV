# -*- coding:utf-8 -*-

import argparse
import os.path
import os


def parsers():
    # model_name = "bert-base-chinese"
    model_name = "chinese-bert-wwm"
    # model_name = "chinese-macbert-base"
    # model_name = "agriculture-bert-uncased"
    # model_name = "agriculture-bert-base-chinese"
    batch_size = 16
    lr = 1e-5
    parser = argparse.ArgumentParser(description="Bert model of argparse")
    parser.add_argument("--train_file", type=str, default=os.path.join("data", "train.txt"))
    parser.add_argument("--dev_file", type=str, default=os.path.join("data", "dev.txt"))
    parser.add_argument("--test_file", type=str, default=os.path.join("data", "test.txt"))
    parser.add_argument("--log_file", type=str, default=os.path.join(f"models/{model_name}-{batch_size}-{lr}", f"train_log.jsonl"))
    parser.add_argument("--classification", type=str, default=os.path.join("data", "class.txt"))
    parser.add_argument("--bert_pred", type=str, default=f"/root/autodl-tmp/models/{model_name}")
    parser.add_argument("--class_num", type=int, default=2)
    parser.add_argument("--max_len", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=batch_size)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learn_rate", type=float, default=lr)
    parser.add_argument("--warmup_ratio", type=int, default=0.1)  # 新增 逐步增加学习率
    # parser.add_argument("--dropout_rate", type=float, default=0.05)  # 新增 防止过拟合
    parser.add_argument("--num_filters", type=int, default=768)
    parser.add_argument("--save_model_best", type=str, default=os.path.join(f"models/{model_name}-{batch_size}-{lr}", "best_model.pth"))
    parser.add_argument("--save_model_last", type=str, default=os.path.join(f"models/{model_name}-{batch_size}-{lr}", "last_model.pth"))
    parser.add_argument("--log_interval", type=int, default=20)
    args = parser.parse_args()
    os.makedirs(f"models/{model_name}-{batch_size}-{lr}", exist_ok=True)
    return args
