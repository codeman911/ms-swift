#!/bin/bash

# Optimized Higgs-Audio LoRA Training for 8x H200 GPUs + 128 CPU cores
# Uses MS-SWIFT native acceleration features for maximum performance

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=16  # 128/8 = 16 threads per GPU
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3

swift sft \
    --custom_register_path ./mswift_higgs_register_validating.py \
    --model_type higgs-audio-validating \
    --model ../train-higgs-audio/model_file/ \
    --template higgs-chatml-validating \
    --dataset higgs-chatml-validating \
    --val_dataset_sample 0.1 \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_target_modules ALL \
    --adalora_beta1 0.9 \
    --adalora_beta2 0.95 \
    \
    --num_train_epochs 1 \
    --max_length 2048 \
    --learning_rate 5e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-8 \
    \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --dataloader_num_workers 32 \
    --dataloader_pin_memory true \
    --dataloader_persistent_workers true \
    \
    --deepspeed zero3 \
    --bf16 true \
    --tf32 true \
    --gradient_checkpointing true \
    --ddp_backend nccl \
    --ddp_find_unused_parameters false \
    --ddp_broadcast_buffers false \
    \
    --max_grad_norm 1.0 \
    --logging_steps 10 \
    --save_steps 500 \
    --eval_steps 500 \
    --save_total_limit 2 \
    --save_steps 250 \
    --eval_steps 250 \
    --save_total_limit 3 \
    --evaluation_strategy steps \
    --eval_accumulation_steps 4 \
    \
    --output_dir ./output_higgs_lora_optimized \
    --logging_dir ./logs_higgs_lora_optimized \
    --report_to tensorboard \
    --run_name higgs_audio_lora_8xh200 \
    \
    --save_safetensors true \
    --resume_from_checkpoint auto \
    --ignore_data_skip false \
    --lazy_tokenize false \
    \
    --preprocess_num_proc 32 \
    --streaming false \
    --push_hub_strategy push_best \
    --hub_private_repo true \
    \
    --seed 42 \
    --data_seed 42 \
    --remove_unused_columns false \
    --label_smoothing_factor 0.0 \
    --optim adamw_torch_fused \
    --group_by_length true \
    --length_column_name length \
    --disable_tqdm false \
    --predict_with_generate false \
    --generation_max_length 512 \
    --generation_num_beams 1 \
    --include_inputs_for_metrics false
