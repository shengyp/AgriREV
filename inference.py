# -*- coding: utf-8 -*-
# 多卡批量推理 + 中断重跑安全版
# 启动示例：
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 inference_batch_resume.py \
#     --model_dir lora/checkpoint-12190 \
#     --data_dir ./DPcorpus/dataset \
#     --data_file chatglm_topic.json \
#     --batch_size 3 \
#     --output_prefix pred

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
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    return model, tokenizer

# ======================
# 主函数
# ======================
def main():
    parser = argparse.ArgumentParser("多卡批量推理工具（可中断重跑）")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="dataset")
    parser.add_argument("--data_file", type=str, default="test.json")
    parser.add_argument("--output_prefix", type=str, default="pred")
    parser.add_argument("--batch_size", type=int, default=3, help="每卡 batch size")

    args = parser.parse_args()

    # ------------------
    # 分布式信息
    # ------------------
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    output_dir = Path("./output")
    # output_dir = Path("./DPcorpus/KG")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{args.output_prefix}-rank{local_rank}.json"

    if local_rank == 0:
        print("=" * 60)
        print("多卡批量推理（可中断重跑）启动")
        print(f"World size : {world_size}")
        print(f"Model dir  : {args.model_dir}")
        print(f"Data file  : {os.path.join(args.data_dir, args.data_file)}")
        print(f"每卡 batch_size : {args.batch_size}")
        print(f"输出路径 : {output_path}")
        print("=" * 60)

    # ------------------
    # 加载已完成的 id
    # ------------------
    completed_ids = set()
    if output_path.exists():
        print(f"[Rank {local_rank}] 检测到已有输出文件，加载已完成样本...")
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    completed_ids.add(item["id"])
                except:
                    continue
        print(f"[Rank {local_rank}] 已完成 {len(completed_ids)} 条样本，将跳过")

    # ------------------
    # 加载数据
    # ------------------
    num_workers = 16
    dataset_dict = load_dataset(
        path="json",
        data_dir=args.data_dir,
        data_files=args.data_file,
        num_proc=num_workers
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
        "do_sample": True,
        "max_new_tokens": 600,
        "use_cache": True,
        "temperature": 0.2,  # 控制随机性，越高越随机
        "top_p": 0.9       
    }

    # ------------------
    # 批量推理
    # ------------------
    total, ok, empty = 0, 0, 0
    batch_prompts = []
    batch_items = []

    # 打开文件模式改为追加
    with open(output_path, "a", encoding="utf-8") as f:
        for idx, item in enumerate(tqdm(dataset, desc=f"Rank {local_rank}")):
            if idx % world_size != local_rank:
                continue

            if item["id"] in completed_ids:
                continue  # 已完成样本跳过

            total += 1
            prompt = item["conversations"][0]["content"]
            batch_prompts.append(prompt)
            batch_items.append(item)

            if len(batch_prompts) == args.batch_size:
                # ===== batch 推理 =====
                try:
                    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
                    with torch.no_grad():
                        outputs = model.generate(**inputs, **gen_kwargs)
                    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                except Exception as e:
                    print(f"[Rank {local_rank}] WARN batch 推理失败：{e}")
                    responses = [""] * len(batch_items)
                    empty += len(batch_items)

                for it, resp, prompt_text in zip(batch_items, responses, batch_prompts):
                    pred_text = resp.split("<|assistant|>")[-1].strip()
                    it["pred"] = pred_text
                    f.write(json.dumps(it, ensure_ascii=False) + "\n")
                    ok += 1

                batch_prompts.clear()
                batch_items.clear()

        # ===== 处理剩余不足 batch 的样本 =====
        if batch_prompts:
            try:
                inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
                with torch.no_grad():
                    outputs = model.generate(**inputs, **gen_kwargs)
                responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            except Exception as e:
                print(f"[Rank {local_rank}] WARN 最后 batch 推理失败：{e}")
                responses = [""] * len(batch_items)
                empty += len(batch_items)

            for it, resp, prompt_text in zip(batch_items, responses, batch_prompts):
                pred_text = resp.split("<|assistant|>")[-1].strip()
                it["pred"] = pred_text
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
                ok += 1

    print(f"[Rank {local_rank}] 完成：样本数={total} 成功={ok} 失败={empty} 输出={output_path}")


if __name__ == "__main__":
    main()
