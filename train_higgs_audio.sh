#!/bin/bash
# Higgs-Audio V2 Fine-tuning Script with MS-SWIFT
# For zero-shot voice cloning and TTS training

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # Use all 8 GPUs
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TOKENIZERS_PARALLELISM=false

# Training configuration
MODEL_TYPE="higgs-audio-v2-3b"
MODEL_PATH="bosonai/higgs-audio-v2-generation-3B-base"
AUDIO_TOKENIZER="bosonai/higgs-audio-v2-tokenizer"
OUTPUT_DIR="./output_higgs_audio_v2"
DATASET_PATH="./data/higgs_audio_train.jsonl"

# Training parameters
LEARNING_RATE=5e-5
AUDIO_LEARNING_RATE=1e-4
BATCH_SIZE=4
GRADIENT_ACCUMULATION=4
MAX_LENGTH=2048
MAX_AUDIO_LENGTH=1024
NUM_EPOCHS=3
SAVE_STEPS=500
EVAL_STEPS=500
WARMUP_RATIO=0.1

# LoRA configuration
LORA_RANK=32
LORA_ALPHA=64
LORA_DROPOUT=0.1

# Launch training with MS-SWIFT
python -u swift/cli/train.py \
    --model_type "$MODEL_TYPE" \
    --model_id_or_path "$MODEL_PATH" \
    --custom_register_path "./plugins/register.py" \
    --dataset "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --sft_type "lora" \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,audio_ffn.gate_proj,audio_ffn.up_proj,audio_ffn.down_proj" \
    --learning_rate $LEARNING_RATE \
    --audio_learning_rate $AUDIO_LEARNING_RATE \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
    --max_length $MAX_LENGTH \
    --max_audio_length $MAX_AUDIO_LENGTH \
    --warmup_ratio $WARMUP_RATIO \
    --save_strategy "steps" \
    --save_steps $SAVE_STEPS \
    --save_total_limit 3 \
    --evaluation_strategy "steps" \
    --eval_steps $EVAL_STEPS \
    --logging_steps 10 \
    --dataloader_num_workers 8 \
    --bf16 true \
    --gradient_checkpointing true \
    --deepspeed "configs/deepspeed_zero2.json" \
    --use_flash_attention true \
    --trainer_type "higgs_audio" \
    --data_collator "higgs_audio" \
    --audio_tokenizer "$AUDIO_TOKENIZER" \
    --audio_loss_weight 1.0 \
    --text_loss_weight 1.0 \
    --auxiliary_loss_weight 0.001 \
    --remove_unused_columns false \
    --report_to "tensorboard" \
    --resume_from_checkpoint "auto" \
    --seed 42 \
    2>&1 | tee "${OUTPUT_DIR}/training.log"

echo "Training completed! Model saved to ${OUTPUT_DIR}"
