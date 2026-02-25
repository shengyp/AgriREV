# 单个运行
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 inference.py \
#   --model_dir  lora/checkpoint-1225 \
#   --epoch origin 
# checkpoint-0
# checkpoint-2254
# checkpoint-4508

#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL_DIR=/root/autodl-tmp/models/chatglm3-6b
LORA_DIR=lora

CHECKPOINTS=(
  checkpoint-0
)

for ckpt in "${CHECKPOINTS[@]}"; do
  echo "=============================="
  echo "🚀   Running inference for: $ckpt"
  echo "=============================="

  if [ "$ckpt" = "checkpoint-0" ]; then
    torchrun --nproc_per_node=4 inference.py \
      --model_dir $MODEL_DIR \
      --epoch $ckpt
  else
    torchrun --nproc_per_node=4 inference.py \
      --model_dir  $LORA_DIR/$ckpt \
      --epoch     $ckpt
  fi

done
