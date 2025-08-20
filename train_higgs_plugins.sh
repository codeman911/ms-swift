#!/bin/bash

# Higgs-Audio Training Script following CUSTOM_TTS.md specifications
# This script implements the complete training pipeline as documented

set -e

echo "🎵 Starting Higgs-Audio Training Pipeline (CUSTOM_TTS.md compliant)..."
echo "================================================="

# Environment setup
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=16
export PYTHONPATH="${PWD}:${PWD}/higgs-audio:${PYTHONPATH}"

# Force HuggingFace Hub as per documentation
export USE_HF=1
export USE_MODELSCOPE_HUB=0

# Parse command line arguments
MODEL_PATH="../train-higgs-audio/model_file/"  # Default model path
DATASET_PATH="lora_training_data_zr/chatml_fixed/val_chatml_samples.json"  # Default dataset
OUTPUT_DIR="./output_higgs_final"  # Default output

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --dataset_path)
            DATASET_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--model MODEL_PATH] [--dataset_path DATASET_PATH] [--output_dir OUTPUT_DIR]"
            exit 1
            ;;
    esac
done

echo "📦 Configuration:"
echo "  - Model: Higgs-Audio V2 (from $MODEL_PATH)"
echo "  - Dataset: $DATASET_PATH"
echo "  - Output: $OUTPUT_DIR"
echo "  - Template: higgs-audio-chatml"
echo "  - Plugins: All components from CUSTOM_TTS.md"

# Main training command following CUSTOM_TTS.md specifications
swift sft \
    --model_type higgs-audio \
    --model "$MODEL_PATH" \
    --template higgs-audio-chatml \
    --custom_register_path plugins/register.py \
    --dataset higgs_audio:"$DATASET_PATH" \
    --train_type lora \
    --lora_target_modules ALL \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.1 \
    --num_train_epochs 3 \
    --bf16 true \
    --gradient_checkpointing true \
    --save_strategy steps \
    --save_steps 500 \
    --logging_steps 10 \
    --output_dir "$OUTPUT_DIR" \
    --logging_dir "${OUTPUT_DIR}/logs" \
    --report_to tensorboard \
    --run_name higgs_audio_custom_tts \
    --remove_unused_columns false \
    --dataloader_num_workers 4 \
    --seed 42 \
    --ddp_find_unused_parameters false \
    --logging_first_step true \
    --save_total_limit 3

echo "================================================="
echo "✅ Training completed successfully!"
echo "📁 Model saved to: $OUTPUT_DIR"
echo "📊 Logs available at: ${OUTPUT_DIR}/logs"
echo ""
echo "To run inference with the trained model:"
echo "  swift infer --model_type higgs-audio --ckpt_dir $OUTPUT_DIR"
echo "=================================================="
