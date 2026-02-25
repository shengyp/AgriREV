# 单个运行
# epoch_num=10
# cat output/epoch_${epoch_num}_pred_rank*.json > output/epoch_${epoch_num}_pred.json \
# && rm -f output/epoch_${epoch_num}_pred_rank*.json

#!/bin/bash

OUTPUT_DIR=output

EPOCHS=(
  checkpoint-0
)

for epoch in "${EPOCHS[@]}"; do
  echo "=============================="
  echo "📦   Merging epoch: $epoch"
  echo "=============================="

  pattern="${OUTPUT_DIR}/${epoch}-pred-rank*.json"
  target="${OUTPUT_DIR}/${epoch}-pred.json"
  
  if ls $pattern >/dev/null 2>&1; then
    # 1️⃣ 合并
    if cat $pattern > "$target"; then
      echo "✅ Merged -> $target"

      # 2️⃣ 删除 rank 文件
      rm -f $pattern
      echo "🗑️  Removed rank files"
    else
      echo "❌ Merge failed for $epoch, skip delete"
    fi
  else
    echo "⚠️    No rank files found for $epoch"
  fi
done

