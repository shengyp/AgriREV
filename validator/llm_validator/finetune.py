# -*- coding: utf-8 -*-
# 使用方法: python finetune.py --model_dir /root/autodl-tmp/models/chatglm3-6b --data_dir dataset --config configs/lora.yaml

from transformers import Trainer

def _skip_load_rng_state(self, checkpoint):
    print("[INFO] Skip loading rng_state.pth (PyTorch 2.6 compatibility)")

Trainer._load_rng_state = _skip_load_rng_state

from transformers import BitsAndBytesConfig
import os
import json
import dataclasses as dc
import functools
import argparse
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Optional, Union
import numpy as np
import ruamel.yaml as yaml
import torch
from datasets import Dataset, DatasetDict, NamedSplit, Split, load_dataset
from peft import (
    PeftConfig,
    PeftModelForCausalLM,
    get_peft_config,
    get_peft_model
)
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoModel,
    AutoTokenizer,
    EvalPrediction,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    Seq2SeqTrainingArguments,
    AutoConfig,
)
from transformers import DataCollatorForSeq2Seq as _DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer as _Seq2SeqTrainer
from sklearn.metrics import accuracy_score, f1_score, classification_report

ModelType = Union[PreTrainedModel, PeftModelForCausalLM]
TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


class DataCollatorForSeq2Seq(_DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        # 提取非数值字段（如 task）
        non_tensor_fields = {}
        for key in list(features[0].keys()):
            if key not in ['input_ids', 'attention_mask', 'labels', 'output_ids']:
                non_tensor_fields[key] = [feature.pop(key) for feature in features]
        
        output_ids = (
            [feature['output_ids'] for feature in features]
            if 'output_ids' in features[0].keys()
            else None
        )
        if output_ids is not None:
            max_output_length = max(len(out) for out in output_ids)
            if self.pad_to_multiple_of is not None:
                max_output_length = (
                        (
                                max_output_length + self.pad_to_multiple_of - 1) //
                        self.pad_to_multiple_of * self.pad_to_multiple_of
                )
            for feature in features:
                remainder = [self.tokenizer.pad_token_id] * (
                        max_output_length - len(feature['output_ids'])
                )
                if isinstance(feature['output_ids'], list):
                    feature['output_ids'] = feature['output_ids'] + remainder
                else:
                    feature['output_ids'] = np.concatenate(
                        [feature['output_ids'], remainder]
                    ).astype(np.int64)
        
        # 调用父类的 __call__ 方法
        batch = super().__call__(features, return_tensors)
        
        # 将非数值字段添加回批次
        for key, value in non_tensor_fields.items():
            batch[key] = value
            
        return batch


class Seq2SeqTrainer(_Seq2SeqTrainer):
    def prediction_step(
            self,
            model: nn.Module,
            inputs: dict[str, Any],
            prediction_loss_only: bool,
            ignore_keys=None,
            **gen_kwargs,
    ) -> tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        # 提取非数值字段
        non_tensor_fields = {}
        for key in list(inputs.keys()):
            if key not in ['input_ids', 'attention_mask', 'labels', 'output_ids']:
                non_tensor_fields[key] = inputs.pop(key)
                
        if self.args.predict_with_generate:
            # 保存task以便后续评估使用
            output_ids = inputs.pop('output_ids', None)
        else:
            output_ids = None
            
        input_ids = inputs['input_ids']
        loss, generated_tokens, labels = super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys, **gen_kwargs
        )
        
        if generated_tokens is not None:
            generated_tokens = generated_tokens[:, input_ids.size()[1]:]
        
        if self.args.predict_with_generate:
            labels = output_ids
            
        return loss, generated_tokens, labels


def _resolve_path(path: Union[str, Path]) -> Path:
    return Path(path).expanduser().resolve()


def _sanity_check(
        input_ids: Sequence[int],
        output_ids: Sequence[int],
        tokenizer: PreTrainedTokenizer,
):
    print("====Sanity check：train_dataset[0][input_ids], train_dataset[0][labels], tokenizer ====")
    for in_id, out_id in zip(input_ids, output_ids):
        if in_id == 0:
            continue
        if in_id in tokenizer.tokenizer.index_special_tokens:
            in_text = tokenizer.tokenizer.index_special_tokens[in_id]
        else:
            in_text = tokenizer.decode([in_id])
        # print(f'{repr(in_text):>20}: {in_id} -> {out_id}')


@functools.cache
def _get_yaml_parser() -> yaml.YAML:
    parser = yaml.YAML(typ='safe', pure=True)
    parser.indent(mapping=2, offset=2, sequence=4)
    parser.default_flow_style = False
    return parser


@dc.dataclass
class DataConfig(object):
    train_file: str
    val_file: Optional[str] = None
    test_file: Optional[str] = None

    num_proc: Optional[int] = None

    @property
    def data_format(self) -> str:
        return Path(self.train_file).suffix

    @property
    def data_files(self) -> dict[NamedSplit, str]:
        return {
            split: data_file
            for split, data_file in zip(
                [Split.TRAIN, Split.VALIDATION, Split.TEST],
                [self.train_file, self.val_file, self.test_file],
            )
            if data_file is not None
        }


@dc.dataclass
class FinetuningConfig(object):
    data_config: DataConfig

    max_input_length: int
    max_output_length: int

    training_args: Seq2SeqTrainingArguments = dc.field(
        default_factory=lambda: Seq2SeqTrainingArguments(output_dir='./output')
    )
    peft_config: Optional[PeftConfig] = None

    def __post_init__(self):
        if not self.training_args.do_eval or self.data_config.val_file is None:
            self.training_args.do_eval = False
            self.training_args.evaluation_strategy = 'no'
            self.data_config.val_file = None
        else:
            self.training_args.per_device_eval_batch_size = (
                    self.training_args.per_device_eval_batch_size
                    or self.training_args.per_device_train_batch_size
            )

    @classmethod
    def from_dict(cls, **kwargs) -> 'FinetuningConfig':
        training_args = kwargs.get('training_args', None)
        if training_args is not None and not isinstance(
                training_args, Seq2SeqTrainingArguments
        ):
            gen_config = training_args.get('generation_config')
            if not isinstance(gen_config, GenerationConfig):
                training_args['generation_config'] = GenerationConfig(
                    **gen_config
                )
            kwargs['training_args'] = Seq2SeqTrainingArguments(**training_args)

        data_config = kwargs.get('data_config')
        if not isinstance(data_config, DataConfig):
            kwargs['data_config'] = DataConfig(**data_config)

        peft_config = kwargs.get('peft_config', None)
        if peft_config is not None and not isinstance(peft_config, PeftConfig):
            kwargs['peft_config'] = get_peft_config(peft_config)
        return cls(**kwargs)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> 'FinetuningConfig':
        path = _resolve_path(path)
        kwargs = _get_yaml_parser().load(path)
        return cls.from_dict(**kwargs)


def _load_datasets(
        data_dir: Path,
        data_format: str,
        data_files: dict[NamedSplit, str],
        num_proc: Optional[int],
) -> DatasetDict:
    if data_format in ('.csv', '.json', '.jsonl'):
        dataset_dct = load_dataset(
            data_format[1:],
            data_dir=data_dir,
            data_files=data_files,
            num_proc=num_proc,
        )
    else:
        err_msg = f"Cannot load dataset in the '{data_format}' format."
        raise NotImplementedError(err_msg)

    return dataset_dct


class DataManager(object):
    def __init__(self, data_dir: str, data_config: DataConfig):
        self._num_proc = data_config.num_proc

        self._dataset_dct = _load_datasets(
            _resolve_path(data_dir),
            data_config.data_format,
            data_config.data_files,
            self._num_proc,
        )

    def _get_dataset(self, split: NamedSplit) -> Optional[Dataset]:
        return self._dataset_dct.get(split, None)

    def get_dataset(
            self,
            split: NamedSplit,
            process_fn: Callable[[dict[str, Any]], dict[str, Any]],
            batched: bool = True,
            remove_orig_columns: bool = True,
    ) -> Optional[Dataset]:
        orig_dataset = self._get_dataset(split)
        if orig_dataset is None:
            return
        
        if remove_orig_columns:
            remove_columns = orig_dataset.column_names      # 原始列会被删除，只保留 process_fn 返回的列
        else:
            remove_columns = None   # 不删除任何列
        return orig_dataset.map(
            process_fn,
            batched=batched,
            remove_columns=remove_columns,
            num_proc=self._num_proc,
        )


def print_model_size(model: PreTrainedModel):
    print("==== Current print_model_size ====")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n--> model has {total_params / 1e6}M params\n")
    print("==================================")


"""
✅ 构建批次训练集
input_ids = '[gMASK]' + 'sop' + user + assistant + eos_token_id
"""
def process_batch(
    batch: dict[str, list],
    tokenizer,
    max_input_length: int,
    max_output_length: int,
) -> dict[str, list]:
    batched_input_ids = []
    batched_labels = []

    for conv in batch["conversations"]:
        input_ids = [tokenizer.get_command("[gMASK]"), tokenizer.get_command("sop")]
        loss_mask = [False, False]  # 前两位不计算损失

        for msg in conv:
            role = msg["role"]
            content = msg["content"]

            # 构建消息 token
            msg_ids = tokenizer.build_single_message(role, "", content)

            # 只对 assistant 输出计算 loss
            if role == "assistant":
                loss_mask.extend([True] * len(msg_ids))
            else:
                loss_mask.extend([False] * len(msg_ids))

            input_ids.extend(msg_ids)

        # EOS
        input_ids.append(tokenizer.eos_token_id)
        loss_mask.append(False)  # EOS 不计算 loss

        # 构建 labels：mask = True 的保留 token id，否则 -100
        labels = [
            token_id if mask else -100
            for token_id, mask in zip(input_ids, loss_mask)
        ]

        # 截断到最大长度
        max_len = max_input_length + max_output_length + 1
        batched_input_ids.append(input_ids[:max_len])
        batched_labels.append(labels[:max_len])

    return {"input_ids": batched_input_ids, "labels": batched_labels}



def process_batch_eval(
    batch: dict[str, list],
    tokenizer,
    max_input_length: int,
    max_output_length: int,
) -> dict[str, list]:
    batched_input_ids = []
    batched_output_ids = []

    for conv in batch["conversations"]:
        input_ids = [tokenizer.get_command("[gMASK]"), tokenizer.get_command("sop")]
        assistant_output_ids = []

        for msg in conv:
            role = msg["role"]
            content = msg["content"]

            msg_ids = tokenizer.build_single_message(role, "", content)

            if role == "assistant":
                # assistant 输出作为模型需要预测的 target
                output_ids = msg_ids[1:]  # 去掉 role token
                output_ids.append(tokenizer.eos_token_id)
                
                # 输入只保留前面的 context + role token
                batched_input_ids.append(input_ids[:max_input_length] + msg_ids[:1])
                batched_output_ids.append(output_ids[:max_output_length])

            # 所有消息都追加到 context
            input_ids.extend(msg_ids)

        # 安全截断
        input_ids = input_ids[:max_input_length]

    return {"input_ids": batched_input_ids, "output_ids": batched_output_ids}



# ✅ 4bit加载模型
def load_tokenizer_and_model(
        model_dir: str,
        peft_config: Optional[PeftConfig] = None,
) -> tuple[PreTrainedTokenizer, nn.Module]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    print("==== Current peft_config ====")
    print(peft_config)
    print("==================================")

    if peft_config is not None:
        if peft_config.peft_type.name == "PREFIX_TUNING":
            config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
            config.pre_seq_len = peft_config.num_virtual_tokens
            config.use_cache = False
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                trust_remote_code=True,
                config=config,
            )
        elif peft_config.peft_type.name == "LORA":




            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,  # 使用FP32计算更稳定
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModel.from_pretrained(
                model_dir,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                quantization_config=bnb_config
            )
            model = get_peft_model(model, peft_config)



    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            empty_init=False,
            use_cache=False
        )
    
    print("==== Current trainable_parameters ====")
    model.print_trainable_parameters()
    print("==================================")
    
    return tokenizer, model





def parse_json_output(output_str):
    """尝试解析JSON输出"""
    try:
        # 移除可能的空白字符
        output_str = output_str.strip()
        label = json.loads(output_str)
        # print("\n===========json解析label===========\n", label)
        # 尝试直接解析
        return label
    except json.JSONDecodeError:
        # 如果失败，尝试找到JSON部分
        # print("\n===========解析label失败===========")
        start_idx = output_str.find('[')
        end_idx = output_str.rfind(']')
        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
            try:
                json_str = output_str[start_idx:end_idx+1]
                return json.loads(json_str)
            except:
                return None
        return None


def infer_task_with_log(pred, gold):
    log = {
        "gold_raw": gold,
        "pred_raw": pred,
        "task": None,
        "task_source": None,
        "gold_json": [],
        "pred_json": [],
        "gold_keys": [],
        "pred_keys": []
    }

    gold_strip = gold.strip()

    # 1️⃣ topic：纯文本
    if gold_strip in {"相关", "不相关"}:
        log["task"] = "topic"
        log["task_source"] = "topic_text"
        return log

    # 2️⃣ 解析 gold
    gold_json = parse_json_output(gold) or []
    log["gold_json"] = gold_json

    if gold_json:
        gold_keys = list(gold_json[0].keys())
        log["gold_keys"] = gold_keys

        if {"entity", "type"} <= set(gold_keys):
            log["task"] = "ner"
            log["task_source"] = "gold_keys"
            return log

        if {"head", "relation", "tail"} <= set(gold_keys):
            log["task"] = "re"
            log["task_source"] = "gold_keys"
            return log

    # 3️⃣ gold = []，fallback 用 pred
    pred_json = parse_json_output(pred) or []
    log["pred_json"] = pred_json

    if pred_json:
        pred_keys = list(pred_json[0].keys())
        log["pred_keys"] = pred_keys

        if {"entity", "type"} <= set(pred_keys):
            log["task"] = "ner"
            log["task_source"] = "pred_keys"
            return log

        if {"head", "relation", "tail"} <= set(pred_keys):
            log["task"] = "re"
            log["task_source"] = "pred_keys"
            return log

    # 4️⃣ 无法判定
    log["task"] = None
    log["task_source"] = "undetermined"
    return log


    
"""
eval_preds:包含 preds_ids 和 labels_ids
tokenizer:
origin_val_dataset: 包含id,task,conversations的原始数据
"""
def compute_metrics(eval_preds: EvalPrediction, tokenizer):
    preds_ids, labels_ids = eval_preds.predictions, eval_preds.label_ids

    pred_str = tokenizer.batch_decode(preds_ids, skip_special_tokens=True)  # 预测文本列表
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)    # 真实标签文本列表
    # 多卡训练问题：此时传入的eval_preds顺序已经不再是原来origin_val_dataset的顺序了

    results = {}
    topic_correct, topic_total = 0, 0   # topic
    ner_tp = ner_pred_total = ner_gold_total = 0    # ner
    re_tp = re_pred_total = re_gold_total = 0   # re
    logs = []
    for pred, gold in zip(pred_str, label_str):
        info = infer_task_with_log(pred, gold)
        logs.append(info)
        task = info["task"]
        if  task == "topic":
            topic_total += 1
            if pred.strip() == gold.strip():
                topic_correct += 1
            # print("\n=======topic=========")
            # print("pred：",pred.strip())
            # print("gold：",gold.strip())

        elif task == "ner":
            pred_json = parse_json_output(pred) or []
            gold_json = parse_json_output(gold) or []

            # print("\n=======ner=========")
            # print("pred：",pred_json)
            # print("gold：",gold_json)

            # {'江苏::区域', '水浸状软化::症状'}
            pred_set = {
                f"{e.get('entity','')}::{e.get('type','')}"
                for e in pred_json if isinstance(e, dict)
            }
            gold_set = {
                f"{e.get('entity','')}::{e.get('type','')}"
                for e in gold_json if isinstance(e, dict)
            }

            ner_tp += len(pred_set & gold_set)    # 预测实体正确
            ner_pred_total += len(pred_set)    # 预测实体总数
            ner_gold_total += len(gold_set)    # 黄金实体总数
            
            # print("ner_pred_set：",pred_set)
            # print("ner_gold_set：",gold_set)
            print("ner_tp：",ner_tp)
            print("ner_pred_total：",ner_pred_total)
            print("ner_gold_total：",ner_gold_total)

        elif task == "re":
            pred_json = parse_json_output(pred) or []
            gold_json = parse_json_output(gold) or []
            # print("\n=======re=========")
            # print("pred：",pred_json)
            # print("gold：",gold_json)

            # {'豇豆茎枯病::防治药剂::琥胶肥酸铜', '苎麻青枯病::适宜发生条件::高温多雨'}
            pred_set = {
                f"{r.get('head','')}::{r.get('relation','')}::{r.get('tail','')}"
                for r in pred_json if isinstance(r, dict)
            }
            gold_set = {
                f"{r.get('head','')}::{r.get('relation','')}::{r.get('tail','')}"
                for r in gold_json if isinstance(r, dict)
            }

            re_tp += len(pred_set & gold_set)    # 预测三元组正确
            re_pred_total += len(pred_set)    # 预测三元组总数
            re_gold_total += len(gold_set)    # 黄金三元组总数
            
            print("re_pred_set：",pred_set)
            print("re_gold_set：",gold_set)
            print("re_tp：",re_tp)
            print("re_pred_total：",re_pred_total)
            print("re_gold_total：",re_gold_total)

    with open("./log/task_infer_log.jsonl", "a", encoding="utf-8") as f:
        for item in logs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    # ===== 汇总 =====
    if topic_total > 0:
        results["topic_accuracy"] = topic_correct / topic_total
        results["topic_samples"] = topic_total

    def safe_f1(tp, pred_n, gold_n):
        if pred_n == 0 or gold_n == 0:
            return 0.0
        p = tp / pred_n
        r = tp / gold_n
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    if ner_gold_total > 0:
        results["ner_micro_f1"] = safe_f1(ner_tp, ner_pred_total, ner_gold_total)
        results["ner_samples"] = ner_gold_total

    if re_gold_total > 0:
        results["re_micro_f1"] = safe_f1(re_tp, re_pred_total, re_gold_total)
        results["re_samples"] = re_gold_total

    results["total_samples"] = len(pred_str)
    
    results["eval_loss"] = 0.5 * results.get("ner_micro_f1", 0.0) + 0.5 * results.get("re_micro_f1", 0.0)

    print("=======final results=========")
    print(results)
    
    return results



def main():
    parser = argparse.ArgumentParser(description="ChatGLM3-6b LoRA微调")
    parser.add_argument("--model_dir", type=str, required=True, help="基础模型路径")
    parser.add_argument("--data_dir", type=str, required=True, help="训练数据集目录")
    parser.add_argument("--config_file", type=str, required=True, help="配置文件路径")
    parser.add_argument("--auto_resume_from_checkpoint", type=str, default=None, 
                       help="从检查点恢复训练，可以是'latest'或具体的检查点路径")

    args = parser.parse_args()
    model_dir = args.model_dir
    data_dir = args.data_dir
    config_file = args.config_file
    auto_resume_from_checkpoint = args.auto_resume_from_checkpoint
    
    ft_config = FinetuningConfig.from_file(config_file)
    print("==== TrainingArguments ====")
    print(ft_config.training_args)
    print("==================================")

    # 加载模型
    tokenizer, model = load_tokenizer_and_model(model_dir, peft_config=ft_config.peft_config)
    data_manager = DataManager(data_dir, ft_config.data_config)

    # 处理批次训练集
    train_dataset = data_manager.get_dataset(
        Split.TRAIN,
        functools.partial(
            process_batch,
            tokenizer=tokenizer,
            max_input_length=ft_config.max_input_length,
            max_output_length=ft_config.max_output_length,
        ),
        batched=True,
    )

    # 处理批次验证集,只保留input_id,label_id两个列名
    val_dataset = data_manager.get_dataset(
        Split.VALIDATION,
        functools.partial(
            process_batch_eval,
            tokenizer=tokenizer,
            max_input_length=ft_config.max_input_length,
            max_output_length=ft_config.max_output_length,
        ),
        batched=True,
        remove_orig_columns=True,
    )

    # 包含原始列名的验证集：id, task, conversations
    origin_val_dataset = data_manager._get_dataset(Split.VALIDATION)

    
    if val_dataset is not None:
        print(f"\n批次验证集列名: {val_dataset.column_names}")
    if origin_val_dataset is not None:
        print(f"\n原始验证集列名: {origin_val_dataset.column_names}")
    if train_dataset is not None:
        _sanity_check(train_dataset[0]["input_ids"], train_dataset[0]["labels"], tokenizer)
        print("\n批次训练集列名: : {train_dataset.column_names}")


    ft_config.training_args.generation_config.pad_token_id = (
        tokenizer.pad_token_id
    )
    ft_config.training_args.generation_config.eos_token_id = [
        tokenizer.eos_token_id,
        tokenizer.get_command('<|user|>'),
        tokenizer.get_command('<|observation|>'),
    ]
    
    # model.gradient_checkpointing_enable()
    # model.enable_input_require_grads()

    use_tokenizer = True
    if ft_config.peft_config is not None:
        use_tokenizer = False if ft_config.peft_config.peft_type == "LORA" else True

    # 创建计算评估指标函数，传入原始验证集 origin_val_dataset，包含task便于评估指标计算时区分任务
    compute_metrics_func = functools.partial(
        compute_metrics, 
        tokenizer=tokenizer
    )

    # 训练器
    trainer = Seq2SeqTrainer(
        model=model,
        args=ft_config.training_args,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding='longest',
            return_tensors='pt',
        ),
        train_dataset=train_dataset,
        # eval_dataset=val_dataset,
        eval_dataset= None,
        tokenizer=tokenizer if use_tokenizer else None,
        compute_metrics=compute_metrics_func
    )

    # 从头训练
    if auto_resume_from_checkpoint is None or auto_resume_from_checkpoint.upper() == "":
        print("\n==== Current restart training !!! ====")
        trainer.train()
        return

    # 加载检查点继续训练
    else:
        print("\n==== Current continue training !!! ====")
        def do_rf_checkpoint(sn):
            checkpoint_directory = os.path.join(output_dir, f"checkpoint-{sn}")
            print(f"Loading model from {checkpoint_directory}.")
            
            # model.gradient_checkpointing_enable()
            # model.enable_input_require_grads()

            trainer.args.ignore_data_skip = True  # ⭐关键
            trainer.train(resume_from_checkpoint=checkpoint_directory)
            return

        output_dir = ft_config.training_args.output_dir

        # resume from latest checkpoint
        if auto_resume_from_checkpoint.upper() == "YES":
            dirlist = os.listdir(output_dir)
            checkpoint_sn = 0
            print("get latest checkpoint")
            print(f"dirlist: {dirlist}")
            for checkpoint_str in dirlist:
                if checkpoint_str.find("eckpoint-") > 0 and checkpoint_str.find("tmp") == -1:
                    checkpoint = int(checkpoint_str.replace("checkpoint-", ""))
                    if checkpoint > checkpoint_sn:
                        checkpoint_sn = checkpoint
            if checkpoint_sn > 0:
                do_rf_checkpoint(str(checkpoint_sn))
            else:
                print("==== ERROR CHECKPOINT! PLEASE ATTENTION! ====")
                return
        else:
            # resume from specific checkpoint
            if auto_resume_from_checkpoint.isdigit() and int(auto_resume_from_checkpoint) > 0:
                do_rf_checkpoint(auto_resume_from_checkpoint)
            else:
                print(auto_resume_from_checkpoint,
                      "The specified checkpoint sn(" + auto_resume_from_checkpoint + ") has not been saved.")


if __name__ == '__main__':
    main()