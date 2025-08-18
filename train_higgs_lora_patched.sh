#!/bin/bash
# train_higgs_lora_patched.sh
# Complete MS-SWIFT launcher for Higgs Audio LoRA training
# Usage: bash train_higgs_lora_patched.sh /abs/path/to/chatml_data.jsonl

set -e

# Check if dataset path is provided
if [ $# -eq 0 ]; then
    echo "Usage: bash train_higgs_lora_patched.sh /abs/path/to/chatml_data.jsonl"
    echo "Example: bash train_higgs_lora_patched.sh /mnt/data/chatml_raw.jsonl"
    exit 1
fi

DATASET_PATH="$1"

# Validate dataset file exists
if [ ! -f "$DATASET_PATH" ]; then
    echo "[ERROR] Dataset file not found: $DATASET_PATH"
    echo "Please provide a valid path to your ChatML JSONL file."
    exit 1
fi

echo "[INFO] Starting Higgs Audio LoRA training with dataset: $DATASET_PATH"

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # Use all 8 GPUs
export TRANSFORMERS_CACHE=/tmp/hf_cache
export HF_DATASETS_CACHE=/tmp/datasets_cache
export USE_MODELSCOPE_HUB=0

# Create output directory
OUTPUT_DIR="./output_higgs_lora_patched"
mkdir -p "$OUTPUT_DIR"

# Launch training with corrected CLI flags
swift sft \
  --model "bosonai/higgs-audio-v2-generation-3B-base" \
  --use_hf true \
  --template "higgs_chatml" \
  --custom_register_path "plugins/higgs_ms_swift_register.py plugins/loss.py plugins/higgs_dataset.py" \
  --dataset "higgs-chatml-custom#path=${DATASET_PATH}" \
  --remove_unused_columns false \
  --streaming true \
  --train_type lora \
  --target_modules "q_proj k_proj v_proj o_proj gate_proj up_proj down_proj audio_mlp.gate_proj audio_mlp.up_proj audio_mlp.down_proj" \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --warmup_steps 1000 \
  --max_steps 5000 \
  --bf16 true \
  --gradient_checkpointing true \
  --save_steps 1000 \
  --logging_steps 20 \
  --loss "higgs_text_audio" \
  --output_dir "$OUTPUT_DIR" \
  --logging_dir "${OUTPUT_DIR}/logs" \
  --report_to tensorboard \
  --run_name "higgs_audio_lora_patched" \
  --save_safetensors true \
  --ignore_data_skip false \
  --lazy_tokenize false \
  --dataset_num_proc 8 \
  --dataloader_num_workers 16 \
  --dataloader_pin_memory true \
  --dataloader_persistent_workers true \
  --seed 42 \
  --data_seed 42 \
  --optim "adamw_torch_fused" \
  --group_by_length false \
  --ddp_find_unused_parameters false \
  --save_only_model true

echo "[INFO] Training completed. Outputs saved to: $OUTPUT_DIR"
echo "[INFO] Logs available at: ${OUTPUT_DIR}/logs"
echo "[INFO] To resume training, add: --resume_from_checkpoint ${OUTPUT_DIR}/checkpoint-XXXX"
