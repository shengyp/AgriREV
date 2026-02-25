# python ZhipuAi.py --model GLM-4-Flash  --api_key 82c13a96349b4ca3ae37264d66015727.Vet5kQjAzgr2tNPa --input ../dataset/test.json 
# python ZhipuAi.py --model GLM-4.5-Flash  --api_key 82c13a96349b4ca3ae37264d66015727.Vet5kQjAzgr2tNPa --input ../dataset/test.json 
# python ZhipuAi.py --model GLM-4.7-Flash  --api_key 82c13a96349b4ca3ae37264d66015727.Vet5kQjAzgr2tNPa --input ../dataset/test.json 


from zai import ZhipuAiClient
import json
import os
import argparse
import threading
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ========== 写锁，防止多线程写文件冲突 ==========
write_lock = threading.Lock()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--api_key", type=str, default="82c13a96349b4ca3ae37264d66015727.Vet5kQjAzgr2tNPa")
    parser.add_argument("--model", type=str, default="GLM-4-Flash")
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--input", type=str, default="../dataset/test.json")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_workers", type=int, default=16, help="并发线程数")

    return parser.parse_args()


def load_done_ids(output_file):
    """断点续跑：读取已完成样本 id"""
    done = set()
    if not os.path.exists(output_file):
        return done

    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                done.add(obj.get("id"))
            except Exception:
                continue
    return done


def safe_write(fout, sample):
    """线程安全写入"""
    with write_lock:
        fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
        fout.flush()


def call_llm(client, model, sample, temperature):
    """单条样本调用"""
    try:
        prompt = sample["conversations"][0]["content"]

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一个有用的AI助手。"},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            # extra_body={"thinking": {"type": "disabled"}}
        )

        reply = response.choices[0].message.content.strip()
        sample["pred"] = reply
        return sample

    except Exception as e:
        sample["pred"] = ""
        sample["error"] = str(e)
        return sample


def main():
    args = parse_args()

    client = ZhipuAiClient(
        api_key=args.api_key,
        base_url=args.base_url
    )

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"{args.model}.json")

    # ========= 断点续跑 =========
    done_ids = load_done_ids(output_file)

    # ========= 读取数据 =========
    with open(args.input, "r", encoding="utf-8") as f:
        samples = []
        for line in f:
            obj = json.loads(line)
            if obj.get("id") not in done_ids:
                samples.append(obj)

    if not samples:
        print("✅ 所有样本已完成，无需再跑")
        return

    print(f"🚀 待处理样本数: {len(samples)}（已跳过 {len(done_ids)} 条）")

    # ========= 并发 + 实时写入 =========
    with open(output_file, "a", encoding="utf-8") as fout, \
         ThreadPoolExecutor(max_workers=args.max_workers) as executor:

        futures = [
            executor.submit(
                call_llm,
                client,
                args.model,
                sample,
                args.temperature
            )
            for sample in samples
        ]

        for future in tqdm(as_completed(futures), total=len(futures),
                           desc=f"Running {args.model}"):
            result = future.result()
            safe_write(fout, result)

    print(f"\n✅ 并发推理完成，结果已安全写入: {output_file}")


if __name__ == "__main__":
    main()
