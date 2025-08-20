#!/bin/bash

# Higgs-Audio V2 Training with Rewritten Plugins
# Uses original boson_multimodal components via MS-SWIFT plugins

set -e

echo "🎵 Starting Higgs-Audio V2 Training with Original Wrapper Plugins..."

# Environment setup
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=16
export PYTHONPATH="${PWD}:${PWD}/higgs-audio:${PYTHONPATH}"

# Force HuggingFace Hub
export USE_HF=1
export USE_MODELSCOPE_HUB=0

# Dataset path (update to your actual dataset)
DATASET_PATH="temp_chatml_samples.jsonl"

echo "📦 Using plugins: model_registration, collator, trainer, dataset_registration"
echo "📊 Dataset: $DATASET_PATH"
echo "🤖 Model: bosonai/higgs-audio-v2-generation-3B-base"

# Register plugins and start training
swift sft \
    --model_type higgs-audio-v2-generation-3b-base \
    --model bosonai/higgs-audio-v2-generation-3B-base \
    --template higgs-audio-template \
    --custom_register_path plugins/register.py \
    --dataset higgs-audio-tts \
    --dataset_path "$DATASET_PATH" \
    --train_type lora \
    --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --warmup_steps 500 \
    --max_steps 5000 \
    --bf16 true \
    --gradient_checkpointing true \
    --save_steps 1000 \
    --logging_steps 10 \
    --output_dir ./output_higgs_plugins \
    --logging_dir ./logs_higgs_plugins \
    --report_to tensorboard \
    --run_name higgs_audio_v2_plugins \
    --remove_unused_columns false \
    --dataloader_num_workers 8 \
    --seed 42

echo "✅ Training completed!"
