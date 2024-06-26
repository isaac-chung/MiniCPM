formatted_time=$(date +"%Y%m%d%H%M%S")
echo $formatted_time


deepspeed finetune.py \
    --model_name_or_path openbmb/MiniCPM-2B-sft-bf16 \
    --output_dir /home/ubuntu/work/experimental/isaac/llm/MiniCPM-2B-RAFT-lora-hotpotqa-dev \
    --train_data_path /home/ubuntu/work/gorilla/data/output/train.jsonl \
    --eval_data_path /home/ubuntu/work/gorilla/data/output/test.jsonl \
    --learning_rate 5e-5 --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8  --model_max_length 384 --bf16 --use_lora \
    --gradient_accumulation_steps 1 --warmup_steps 100 \
    --max_steps 3000 --weight_decay 0.01 \
    --evaluation_strategy steps --eval_steps 500 \
    --save_strategy steps --save_steps 500 --seed 42 \
    --log_level info --logging_strategy steps --logging_steps 10 \
    --deepspeed configs/ds_config_zero3_offload.json
