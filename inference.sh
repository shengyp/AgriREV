CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 inference.py \
    --model_dir lora/checkpoint-12190 \
    --data_dir ./dataset \
    --data_file test.json \
    --batch_size 16 \
    --output_prefix pred
