# -*- coding: utf-8 -*-
# 多卡并行推理（数据并行）
# 启动方式示例：
# CUDA_VISIBLE_DEVICES=0,1,2,3, torchrun --nproc_per_node=4 inference.py

import os
import json
import argparse
from tqdm import tqdm
from pathlib import Path
from typing import Union

import torch
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM, PeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

# ======================
# 类型定义
# ======================
ModelType = Union[PreTrainedModel, PeftModelForCausalLM]
TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

# ======================
# 工具函数
# ======================
def _resolve_path(path: Union[str, Path]) -> Path:
    return Path(path).expanduser().resolve()

# ======================
# 加载模型和 tokenizer
# （单卡，不使用 device_map）
# ======================
def load_model_and_tokenizer(model_dir: Union[str, Path], device: torch.device):
    model_dir = _resolve_path(model_dir)

    if (model_dir / "adapter_config.json").exists():
        print(f"[Rank {os.environ['LOCAL_RANK']}] 使用 LoRA 合并模型推理")
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True
        ).to(device)
        tokenizer_dir = model.peft_config["default"].base_model_name_or_path
    else:
        print(f"[Rank {os.environ['LOCAL_RANK']}] 使用基础模型推理")
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True
        ).to(device)
        tokenizer_dir = model_dir

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir,
        trust_remote_code=True
    )
    return model, tokenizer

# ======================
# 主函数
# ======================
def main():
    # ------------------
    # 分布式信息
    # ------------------
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # ------------------
    # 参数解析
    # ------------------
    parser = argparse.ArgumentParser("多卡并行推理工具")
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="dataset")
    parser.add_argument("--data_file", type=str, default="test.json")
    parser.add_argument("--output_prefix", type=str, default="pred")
    parser.add_argument("--epoch", type=str, default=None)

    args = parser.parse_args()

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir, f"{args.epoch}-{args.output_prefix}-rank{local_rank}.json"
    )

    if local_rank == 0:
        print("=" * 60)
        print("多卡并行推理启动")
        print(f"World size : {world_size}")
        print(f"Model dir  : {args.model_dir}")
        print(f"Data file  : {os.path.join(args.data_dir, args.data_file)}")
        print("=" * 60)

    # ------------------
    # 加载数据（json / jsonl 都支持）
    # ------------------
    dataset_dict = load_dataset(
        path="json",
        data_dir=args.data_dir,
        data_files=args.data_file
    )
    dataset = dataset_dict["train"]

    # ------------------
    # 加载模型
    # ------------------
    model, tokenizer = load_model_and_tokenizer(args.model_dir, device)

    # ------------------
    # 推理参数
    # ------------------
    gen_kwargs = {
        "do_sample": False,
        "max_new_tokens":20,
        "max_length": None,
        "num_beams": 1,
        "temperature": None,  
        "top_p": None,  
        "use_cache": True
    }

    # ------------------
    # 推理（按 rank 切数据）
    # ------------------
    total, ok, empty = 0, 0, 0

    with open(output_path, "w", encoding="utf-8") as f:
        for idx, item in enumerate(tqdm(dataset, desc=f"Rank {local_rank}")):
            if idx % world_size != local_rank:
                continue

            total += 1
            prompt = item["conversations"][0]["content"]

            try:
                response, _ = model.chat(
                    tokenizer,
                    prompt,
                    history=[],
                    **gen_kwargs
                )
            except Exception as e:
                print(f"[Rank {local_rank}] WARN idx={idx} 推理失败：{e}")
                response = ""
                empty += 1

            # item["conversations"][1]["content"] 是label
            item["pred"] = response
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            ok += 1

    print(
        f"[Rank {local_rank}] 完成：样本数={total} 成功={ok} 失败={empty} 输出={output_path}"
    )

# ======================
# 程序入口
# ======================
if __name__ == "__main__":
    main()
