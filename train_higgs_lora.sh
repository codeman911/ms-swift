#!/bin/bash

# Robust Higgs-Audio LoRA Training Pipeline
# Uses plugin-based approach with on-the-fly tokenization

set -e  # Exit on any error

echo "[INFO] Starting Higgs-Audio LoRA training with plugin-based approach..."

# Environment setup
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=16
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

# Force HuggingFace Hub
export USE_HF=1
export USE_MODELSCOPE_HUB=0
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

# Add current directory to Python path
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Verify plugin files exist
for plugin in plugins/higgs_ms_swift_register.py plugins/loss.py plugins/higgs_dataset.py; do
    if [[ ! -f "$plugin" ]]; then
        echo "[ERROR] Required plugin file missing: $plugin"
        exit 1
    fi
done

# Create dataset path (update this to your actual dataset)
DATASET_PATH="../higgs-audio/lora_training_data_zr/chatml_fixed/val_chatml_samples.json"
if [[ ! -f "$DATASET_PATH" ]]; then
    echo "[WARNING] Dataset not found at $DATASET_PATH, using placeholder"
    DATASET_PATH="plugins/higgs_dataset.py"  # Will use the dataset loader
fi

echo "[INFO] Using dataset: $DATASET_PATH"
echo "[INFO] Using model: bosonai/higgs-audio-v2-generation-3B-base"

# Run training with plugin-based approach
swift sft \
    --model bosonai/higgs-audio-v2-generation-3B-base \
    --use_hf true \
    --template higgs_chatml \
    --custom_register_path plugins/higgs_ms_swift_register.py plugins/loss.py plugins/higgs_dataset.py \
    --dataset "$DATASET_PATH" \
    --remove_unused_columns false \
    --streaming true \
    --train_type lora \
    --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --warmup_steps 1000 \
    --max_steps 5000 \
    --bf16 true \
    --gradient_checkpointing true \
    --save_steps 1000 \
    --logging_steps 20 \
    --loss_type higgs_text_audio \
    --output_dir ./output_higgs_lora_plugins \
    --logging_dir ./logs_higgs_lora_plugins \
    --report_to tensorboard \
    --run_name higgs_audio_lora_plugins \
    --save_safetensors true \
    --ignore_data_skip false \
    --lazy_tokenize false \
    --preprocess_num_proc 8 \
    --dataloader_num_workers 16 \
    --dataloader_pin_memory true \
    --dataloader_persistent_workers true \
    --seed 42 \
    --data_seed 42 \
    --optim adamw_torch_fused \
    --group_by_length false \
    --disable_tqdm false \
    --predict_with_generate false \
    --generation_max_length 512 \
    --generation_num_beams 1 \
    --include_inputs_for_metrics false

echo "[INFO] Training completed successfully!"
