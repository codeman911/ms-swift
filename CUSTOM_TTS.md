# Training a Custom Higgs Audio TTS Model with MS-SWIFT

This guide explains how to use the `mswift_higgs_register.py` script to train a Higgs Audio-style TTS model on your custom ChatML dataset with on-the-fly audio tokenization.

## Prerequisites

1.  **Environment**: Ensure your `ms-swift` environment is activated.
2.  **Dependencies**: Make sure the `higgs-audio` library is installed and accessible in your `PYTHONPATH`.
    ```bash
    pip install -e higgs-audio
    ```
3.  **Dataset**: Your custom ChatML dataset (e.g., `dataset.json`) should be ready, with correct relative paths to the audio files.

## Training Script

Create a shell script named `train_custom_tts.sh` to launch the fine-tuning job. This script will use the custom components we defined in `mswift_higgs_register.py`.

```bash
# train_custom_tts.sh

# --- Configuration ---
export CUDA_VISIBLE_DEVICES=0

MODEL_TYPE="higgs-audio-v2" # The base model to fine-tune
DATASET_PATH="/path/to/your/dataset.json" # IMPORTANT: Use absolute path
OUTPUT_DIR="./output/higgs-custom-tts"

# --- Training Command ---
swift sft \
    --model_type $MODEL_TYPE \
    --model_layer_cls_name LlamaDecoderLayer \
    --dataset higgs-chatml-custom \
    --template_type higgs-chatml-custom \
    --train_dataset_path $DATASET_PATH \
    --output_dir $OUTPUT_DIR \
    --custom_register_path mswift_higgs_register.py \
    --train_type lora \
    --lora_target_modules ALL \
    --lora_rank 8 \
    --lora_alpha 32 \
    --batch_size 1 \
    --num_train_epochs 3 \
    --learning_rate 1e-4 \
    --use_flash_attn true \
    --save_steps 500 \
    --logging_steps 10 \
    --gradient_accumulation_steps 16 \
    --eval_steps 500 \
    --deepspeed_config_path ds_config.json # Optional: for multi-GPU