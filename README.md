# AgriREV: An Agricultural Relation Extraction and Verification Framework by Integrating Large Language Models and Human-in-the-Loop

Large Language Models (LLMs) have shown strong general-purpose abilities for tasks like summarization, QA and information extraction, but off-the-shelf models struggle in vertical, terminology-dense domains such as agricultural pest and disease texts¡ªwhere entity names are nested or abbreviated, relation cues span sentences/paragraphs, and document styles vary (papers, manuals, field reports). To address domain drift, hallucination and sentence-level fragmentation, AgriREV implements a  **two-stage, instruction-tuned pipeline** : (1) an LLM fine-tuned with structured instruction templates to **filter relevant passages, perform NER, and generate candidate relation triples** under type/direction constraints; and (2) a multi-stage **verification workflow** (AgriBERT discriminator + LLM reasoning validator + optional human-in-the-loop) that systematically de-noises and verifies triples. This design improves cross-sentence/paragraph understanding, reduces false positives, and yields more explainable, high-confidence knowledge suitable for building agricultural knowledge graphs and decision-support systems.

## Overall Framework Diagram![alt text](./image/framework.png)

# Complete Directory Structure

```plaintext
AgriREV/
|-- API/                      # Model API interfaces and invocation utilities
|
|-- configs/                  # Experiment and runtime configuration files
|
|-- dataset/                  # Dataset definitions and preprocessing pipeline
|
|-- metric/                   # Evaluation outputs, logs, and metric artifacts
|
|-- output/                   # Model predictions and inference results
|
|-- validator/                # Triple validation framework
|   |
|   |-- AgriBERT_validator/   # Discriminative validator based on AgriBERT
|   |   |
|   |   |-- bert-base-chinese/  # Pretrained Chinese BERT backbone
|   |   |-- data/               # Validation datasets
|   |   |-- model/              # Saved checkpoints / trained weights
|   |   |-- config.py           # Validator configuration
|   |   |-- inference.py        # Validation / prediction script
|   |   |-- main.py             # Training entry point
|   |   |-- model.py            # Model architecture definition
|   |   |-- requirement.txt     # Dependency list
|   |   |-- test.py             # Testing / debugging utilities
|   |   |-- utils.py            # Data processing helpers
|   |
|   |-- LLM_validator/         # Reasoning-based validator using LLMs
|       |
|       |-- configs/           # Validator configuration files
|       |-- dataset/           # Validation datasets
|       |-- metric/            # Evaluation outputs
|       |-- output/            # Validator predictions
|       |-- finetune.py        # Instruction fine-tuning script
|       |-- inference.py       # Validation / inference script
|       |-- metric.py          # Evaluation utilities
|
|-- finetune.py               # Main instruction fine-tuning entry
|
|-- inference.py              # Main inference pipeline
|
|-- metric.py                 # Evaluation utilities / metrics computation
```

# Fine-tuning

## finetune.py

```bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
torchrun \
  --nproc_per_node=4 \
  --master_port=29601 \
  finetune.py \
  --model_dir /root/autodl-tmp/models/chatglm3-6b \
  --data_dir dataset \
  --config_file configs/lora.yaml
```

## inference.py

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 inference.py \
    --model_dir lora/checkpoint-10419 \
    --data_dir ./dataset \
    --data_file test.json \
    --batch_size 16 \
    --output_prefix pred
```

## inference results

```
output/pred.json
```

## metric results

```
metric/metric.json
```

# Triplet Verification

### Workflow-1 AgriBERT Validator

![workflow1](./image/workflow1.png)

### Workflow-2 LLM Validator

![workflow2](./image/workflow2.png)

### Workflow-3 LLM + Human

![workflow3](./image/workflow3.png)
