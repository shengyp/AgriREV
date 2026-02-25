import json
import pandas as pd

# -----------------------------
# 参数配置
# -----------------------------
model_files = {
    "deepseek-v3.2": "deepseek-v3.2.json",
    "GLM-4-Flash": "GLM-4-Flash.json",
    "gpt-3.5-turbo": "gpt-3.5-turbo.json"
}

NER_TYPES = [
    "虫害", "病害", "作物", "药剂", "品种", "肥料",
    "病原", "周期", "部位", "公司", "组织机构", "生物介元"
]

RE_TYPES = [
    "危害作物", "危害部位", "病原为", "传播病原", "防治对象",
    "适用作物", "施用时期", "抗性对象", "隶属作物",
    "分类归属", "生产研发"
]

TOPIC_TYPES = ["topic"]  # 如果你想保留 topic，可以按类似方式处理

# -----------------------------
# 工具函数：百分比化并保留两位
# -----------------------------
def to_percent(x):
    if x is None:
        return None
    return round(x * 100, 2)

# -----------------------------
# 函数：生成表格
# -----------------------------
def build_df(task_name, type_list):
    rows = []
    for model_name, file_path in model_files.items():
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        per_type = data.get(task_name, {}).get("per_type", {})

        # 每个模型生成 3 行：P / R / F
        for metric_key, metric_name in zip(["precision", "recall", "f1"], ["P", "R", "F"]):
            row = {"模型": model_name, "指标": metric_name}
            for t in type_list:
                metrics = per_type.get(t)
                if metrics:
                    row[t] = to_percent(metrics.get(metric_key))
                else:
                    row[t] = None
            rows.append(row)

    columns = ["模型", "指标"] + type_list
    return pd.DataFrame(rows, columns=columns)

# -----------------------------
# 生成各任务 DataFrame
# -----------------------------
ner_df = build_df("ner", NER_TYPES)
re_df = build_df("re", RE_TYPES)

# 如果需要 topic sheet，可以这样：
def build_topic_df():
    rows = []
    for model_name, file_path in model_files.items():
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        topic_data = data.get("topic", {})
        row = {"模型": model_name}
        for metric_key in ["precision", "recall", "f1"]:
            row[metric_key.upper()] = to_percent(topic_data.get(metric_key))
        rows.append(row)
    columns = ["模型", "P", "R", "F"]
    return pd.DataFrame(rows, columns=columns)

topic_df = build_topic_df()

# -----------------------------
# 保存到同一个 Excel，分别保存 sheet
# -----------------------------
with pd.ExcelWriter("model_metrics.xlsx", engine="openpyxl") as writer:
    ner_df.to_excel(writer, sheet_name="NER", index=False)
    re_df.to_excel(writer, sheet_name="RE", index=False)
    topic_df.to_excel(writer, sheet_name="Topic", index=False)

print("Excel 文件生成完成：model_metrics.xlsx")
