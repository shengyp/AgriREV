# -*- coding:utf-8 -*-

import torch
from utils import read_data, MyDataset
from config import parsers
from torch.utils.data import DataLoader
from model import MyModel
from torch.optim import AdamW
import torch.nn as nn
from sklearn.metrics import accuracy_score
import time
from test import test_data
from tqdm import tqdm
import json
from datetime import datetime
import os
import random
import numpy as np
from transformers import get_linear_schedule_with_warmup

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 让 CUDA 算子确定性（速度会略慢）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def write_log(data):
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    
    set_seed(42)

    start = time.time()
    args = parsers()
    # os.makedirs("logs", exist_ok=True)

    LOG_INTERVAL = args.log_interval    # 每多少个 batch 记录一次
    log_file = args.log_file

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_text, train_label, max_len = read_data(args.train_file)
    dev_text, dev_label = read_data(args.dev_file)
    
    print(f"args.max_len:{args.max_len}")
    # args.max_len = max_len
    print(f"max_len:{max_len}")
    
    train_dataset = MyDataset(train_text, train_label, args.max_len)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    dev_dataset = MyDataset(dev_text, dev_label, args.max_len)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)

    model = MyModel().to(device)
    opt = AdamW(model.parameters(), lr=args.learn_rate)
    loss_fn = nn.CrossEntropyLoss()

    # 计算总训练步数
    total_steps = len(train_dataloader) * args.epochs
    print(f"Total training steps: {total_steps}")
    warmup_steps = int(args.warmup_ratio * total_steps)    # 预热步数为总步数的10%
    print(f"Warmup_steps: {warmup_steps}")
    # 创建调度器, 在训练初期，学习率从较小值逐渐增加，避免模型因较大学习率而不稳定
    scheduler = get_linear_schedule_with_warmup(
        opt,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )


    # ===== 记录训练配置（只写一次，作为第一条日志） =====
    write_log({
        "type": "config",
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "seed": 42,
        "device": device,
        "train_size": len(train_text),
        "dev_size": len(dev_text),
        "params": {
            "train_file": args.train_file,
            "dev_file": args.dev_file,
            "test_file": args.test_file,
            "bert_pred": args.bert_pred,
            "class_num": args.class_num,
            "max_len": args.max_len,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learn_rate": args.learn_rate,
            "log_interval": args.log_interval,
            "optimizer": "AdamW",
            "loss": "CrossEntropyLoss"
        }
    })
    
    acc_max = float("-inf")     # 历史最优 dev acc
    early_stop_patience = 6    # 连续多少个 epoch 无提升就停止
    early_stop_counter = 0     # 已连续无提升的 epoch 数

    for epoch in range(args.epochs):
        loss_sum, count = 0, 0
        model.train()
        for batch_index, (batch_text, batch_label) in enumerate(
            tqdm(train_dataloader, desc=f"Training-Epoch-{epoch}", total=len(train_dataloader))
        ):
            batch_label = batch_label.to(device)    # 每个 batch = 32 条文本 + 32 个标签
            pred = model(batch_text)    # 前向传播

            loss = loss_fn(pred, batch_label)    # 计算这一个batch的平均loss

            # 反向传播三连: 清梯度, 反向传播, 参数更新
            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()  # 更新学习率

            loss_sum += loss    # 累积 loss
            count += 1    # 累积 batch 数

            # ===== 是否需要打印/记录 =====
            is_log_step = (
                (batch_index + 1) % LOG_INTERVAL == 0
                or (batch_index + 1) == len(train_dataloader)
            )
    
            if is_log_step:
                avg_loss = (loss_sum / count).item()    # 计算这 LOG_INTERVAL 个 batch 的平均 loss
    
                msg = "[{0}/{1:5d}]\tTrain_Loss:{2:.4f}"
                # print(msg.format(epoch + 1, batch_index + 1, avg_loss))
    
                write_log({
                    "type": "train",
                    "epoch": epoch + 1,
                    "step": batch_index + 1,
                    "loss": avg_loss,
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })
    
                loss_sum, count = 0.0, 0

        # 每epoch验证1次
        model.eval()
        all_pred, all_true = [], []
        with torch.no_grad():
            for batch_text, batch_label in dev_dataloader:
                batch_label = batch_label.to(device)
                pred = model(batch_text)

                pred = torch.argmax(pred, dim=1).cpu().numpy().tolist()
                label = batch_label.cpu().numpy().tolist()

                all_pred.extend(pred)
                all_true.extend(label)

        acc = accuracy_score(all_true, all_pred)
        print(f"dev acc:{acc:.4f}")
        write_log({
            "type": "dev",
            "epoch": epoch + 1,
            "accuracy": acc,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # ===== Early Stopping & Best Model Save =====
        if acc > acc_max:
            print(f"Dev acc improved: {acc_max:.4f} → {acc:.4f}")
            acc_max = acc
            early_stop_counter = 0   # 🔥 重置计数器
        
            torch.save(model.state_dict(), args.save_model_best)
            print("✅ 已保存最佳模型")
        
            write_log({
                "type": "checkpoint",
                "epoch": epoch + 1,
                "best_acc": acc,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })
        else:
            early_stop_counter += 1
            print(
                f"⚠️ Dev acc 未提升（{early_stop_counter}/{early_stop_patience}）"
            )
        
            write_log({
                "type": "early_stop_wait",
                "epoch": epoch + 1,
                "wait": early_stop_counter,
                "best_acc": acc_max,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })
        
            if early_stop_counter >= early_stop_patience:
                print("🛑 触发 Early Stopping，提前终止训练")
                write_log({
                    "type": "early_stop",
                    "epoch": epoch + 1,
                    "best_acc": acc_max,
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })
                break

    torch.save(model.state_dict(), args.save_model_last)

    end = time.time()
    print(f"运行时间：{(end-start)/60%60:.4f} min")

    # ===== 测试集评估 + 写日志 =====
    test_data(log_file=log_file)
