export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
torchrun \
  --nproc_per_node=4 \
  --master_port=29601 \
  finetune.py \
  --model_dir /root/autodl-tmp/models/chatglm3-6b \
  --data_dir dataset \
  --config_file configs/lora.yaml
