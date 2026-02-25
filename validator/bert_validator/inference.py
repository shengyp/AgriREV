# -*- coding:utf-8 -*-

from model import MyModel
from config import parsers
import torch
from transformers import BertTokenizer
import time
import json
import os
from tqdm import tqdm

def load_model(device, model_path):
    myModel = MyModel().to(device)
    myModel.load_state_dict(torch.load(model_path))
    myModel.eval()
    return myModel


def process_text(text, bert_pred):
    tokenizer = BertTokenizer.from_pretrained(bert_pred)
    token_id = tokenizer.convert_tokens_to_ids(["[CLS]"] + tokenizer.tokenize(text))
    mask = [1] * len(token_id) + [0] * (args.max_len + 2 - len(token_id))
    token_ids = token_id + [0] * (args.max_len + 2 - len(token_id))
    token_ids = torch.tensor(token_ids).unsqueeze(0)
    mask = torch.tensor(mask).unsqueeze(0)
    x = torch.stack([token_ids, mask])
    return x


def text_class_name(pred):
    # 模型算出了每个类别的分数，选分数最高的那个类别，并把它变成普通 Python 数字
    result = torch.argmax(pred, dim=1)    
    result = result.cpu().numpy().tolist()   

    # 读取类别文件
    classification = open(args.classification, "r", encoding="utf-8").read().split("\n")    # classification: ['false', 'true']
    
    # 构建「索引 → 类别名」字典
    classification_dict = dict(zip(range(len(classification)), classification))    # classification_dict: {0: 'false', 1: 'true'}
    # print(f"文本：{text}\t预测的类别为：{classification_dict[result[0]]}")
    return classification_dict[result]

def read_validator_file(file_path):
    """
    逐行读取 jsonl 文件
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def write_jsonl(file_path, data_list):
    with open(file_path, "w", encoding="utf-8") as f:
        for data in data_list:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    

if __name__ == "__main__":
    start = time.time()
    args = parsers()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = load_model(device, args.save_model_best)

    # ===== 文件路径 =====
    
    validator_file = "../../output/output_7572.json"
    output_file = "output/pred.json"

    # ===== 读取原始数据 =====
    raw_data = read_validator_file(validator_file)
    output_data = []

    print("模型预测中...")

    for item in tqdm(raw_data):
        # 只处理 re 任务
        if item.get("task_type") != "re":
            output_data.append(item)
            continue

        conversations = item.get("conversations", [])
        for conv in conversations:
            if conv.get("role") == "assistant":
                try:
                    triples = json.loads(conv.get("content", "[]"))
                except json.JSONDecodeError:
                    continue

                # 逐个 triple 进行预测
                for triple in triples:
                    head = triple.get("head", "")
                    relation = triple.get("relation", "")
                    tail = triple.get("tail", "")

                    triple_text = f"{head}{relation}{tail}"

                    x = process_text(triple_text, args.bert_pred, args)
                    x = x.to(device)

                    with torch.no_grad():
                        pred = model(x)

                    label = text_class_name(pred, args.classification)
                    triple["bert_validator"] = label

                # 写回 conversations
                conv["content"] = json.dumps(triples, ensure_ascii=False)

        output_data.append(item)

    # ===== 写入新文件 =====
    write_jsonl(output_file, output_data)

    end = time.time()
    print(f"预测完成，结果已保存至 {output_file}")
    print(f"耗时为：{end - start:.2f} s")