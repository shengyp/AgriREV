# 单个运行
# epoch_num=10
# python metric.py \
#   --pred output/epoch_${epoch_num}_pred.json \
#   --metric metric/epoch_${epoch_num}_metric.json


#!/bin/bash

OUTPUT_DIR=output
METRIC_DIR=metric
mkdir -p "${METRIC_DIR}"

EPOCHS=(
  checkpoint-0
)

for epoch in "${EPOCHS[@]}"; do
  echo "=============================="
  echo "📦   Metric epoch: $epoch"
  echo "=============================="

  pred="${OUTPUT_DIR}/${epoch}-pred.json"
  metric="${METRIC_DIR}/${epoch}-metric.json"
  
  python metric.py \
    --pred ${pred} \
    --metric $metric

done


