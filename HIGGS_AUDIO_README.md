# Higgs-Audio V2 Integration with MS-SWIFT

Production-ready integration for fine-tuning Higgs-Audio V2 models for zero-shot voice cloning and TTS generation using MS-SWIFT framework.

## Overview

Higgs-Audio V2 is a unified audio-language model with DualFFN architecture that enables high-quality text-to-speech synthesis and zero-shot voice cloning. This integration provides:

- **Model Registration**: Support for 3B and 7B Higgs-Audio models
- **Custom Training**: Dual loss computation for text and audio
- **Data Pipeline**: ChatML format with multimodal batching
- **LoRA Fine-tuning**: Efficient adaptation of audio FFN layers
- **Voice Cloning**: Zero-shot voice synthesis from reference audio

## Installation

```bash
# Install MS-SWIFT
pip install ms-swift

# Install audio dependencies
pip install librosa soundfile

# Install Higgs-Audio requirements
pip install -r higgs-audio/requirements.txt
```

## Quick Start

### 1. Prepare Dataset

Convert your audio dataset to ChatML JSONL format:

```bash
# For LJSpeech dataset
python prepare_dataset.py \
    --dataset_type ljspeech \
    --input_dir /path/to/LJSpeech-1.1 \
    --output_file data/ljspeech_train.jsonl \
    --split \
    --validate

# For VCTK multi-speaker dataset
python prepare_dataset.py \
    --dataset_type vctk \
    --input_dir /path/to/VCTK-Corpus \
    --output_file data/vctk_train.jsonl \
    --split

# For custom dataset (CSV format)
python prepare_dataset.py \
    --dataset_type custom \
    --input_dir /path/to/audio/files \
    --csv_file /path/to/metadata.csv \
    --output_file data/custom_train.jsonl \
    --text_column "transcript" \
    --audio_column "filename" \
    --speaker_column "speaker_id" \
    --split
```

### 2. Fine-tune Model

Launch LoRA fine-tuning with MS-SWIFT:

```bash
# Make training script executable
chmod +x train_higgs_audio.sh

# Run training
./train_higgs_audio.sh
```

Or use Python directly:

```python
from swift.cli import train
from plugins.register import register_higgs_audio_all

# Register components
register_higgs_audio_all()

# Launch training
train.main([
    "--model_type", "higgs-audio-v2-3b",
    "--dataset", "./data/ljspeech_train.jsonl",
    "--output_dir", "./output_higgs_audio",
    "--sft_type", "lora",
    "--lora_rank", "32",
    "--learning_rate", "5e-5",
    "--num_train_epochs", "3",
    "--trainer_type", "higgs_audio",
    "--data_collator", "higgs_audio"
])
```

### 3. Inference

Generate speech with the fine-tuned model:

```python
from plugins.inference import HiggsAudioGenerator
from swift.llm import get_model_tokenizer

# Load model
model, tokenizer = get_model_tokenizer(
    model_type="higgs-audio-v2-3b",
    model_id_or_path="./output_higgs_audio/checkpoint-best"
)

# Initialize generator
generator = HiggsAudioGenerator(model, tokenizer)

# Generate TTS
result = generator.generate(
    text="Hello, this is Higgs Audio speaking.",
    scene_description="Indoor quiet"
)

# Voice cloning
result = generator.generate(
    text="Cloning this voice from reference.",
    reference_audio="path/to/reference.wav"
)
```

## Architecture

### Plugin Structure

```
plugins/
├── __init__.py              # Package exports
├── model_registration.py    # Model registration with MS-SWIFT
├── dataset_registration.py  # Dataset loader registration
├── trainer.py              # Custom trainer with dual loss
├── collator.py             # Data collator for multimodal batching
├── inference.py            # Generation utilities
└── register.py             # Main registration script
```

### Key Components

#### Model Registration
- Registers Higgs-Audio models with MS-SWIFT
- Configures LoRA target modules for text and audio FFN
- Handles special tokens for audio processing
- Enables gradient checkpointing

#### Dataset Registration
- Supports JSONL, directory, and HuggingFace formats
- Enforces ChatML message structure
- Processes reference audio for voice cloning
- Handles audio feature extraction

#### Custom Trainer
- Dual loss computation (text CE + audio CE)
- Auxiliary loss support
- Separate learning rates for audio parameters
- Gradient accumulation handling

#### Data Collator
- Tokenizes text with padding
- Extracts audio features with Whisper
- Generates RVQ audio codes
- Applies delay patterns for causality
- Creates masked labels for training

## Training Configuration

### Model Types

- `higgs-audio-v2-3b`: 3B parameter model
- `higgs-audio-v2-7b`: 7B parameter model

### LoRA Target Modules

```python
lora_target_modules = [
    # Text attention
    "q_proj", "k_proj", "v_proj", "o_proj",
    # Text FFN
    "gate_proj", "up_proj", "down_proj",
    # Audio FFN (key for voice adaptation)
    "audio_ffn.gate_proj",
    "audio_ffn.up_proj", 
    "audio_ffn.down_proj"
]
```

### Loss Weights

- `text_loss_weight`: 1.0 (text cross-entropy)
- `audio_loss_weight`: 1.0 (audio code cross-entropy)
- `auxiliary_loss_weight`: 0.001 (auxiliary losses)

### Training Parameters

```bash
# Recommended settings
LEARNING_RATE=5e-5
AUDIO_LEARNING_RATE=1e-4  # Higher for audio parameters
BATCH_SIZE=4
GRADIENT_ACCUMULATION=4
MAX_LENGTH=2048
MAX_AUDIO_LENGTH=1024
NUM_EPOCHS=3
LORA_RANK=32
LORA_ALPHA=64
```

## Dataset Format

### ChatML JSONL Structure

```json
{
  "messages": [
    {
      "role": "system",
      "content": "<|scene_desc_start|> Indoor quiet <|scene_desc_end|>"
    },
    {
      "role": "user", 
      "content": "Please speak: Hello world"
    },
    {
      "role": "assistant",
      "content": ""
    }
  ],
  "audios": ["path/to/target_audio.wav"],
  "reference_audio": "path/to/reference_voice.wav"
}
```

### Audio Requirements

- Sample rate: 24kHz
- Format: WAV, FLAC, or MP3
- Duration: 1-30 seconds recommended
- Quality: 16-bit or higher

## Advanced Usage

### Multi-GPU Training

```bash
# Use DeepSpeed Zero-2
deepspeed --num_gpus 8 train_higgs_audio.sh

# Or torchrun
torchrun --nproc_per_node 8 train_higgs_audio.sh
```

### Custom Scene Descriptions

```python
# Scene descriptions affect synthesis style
scenes = [
    "Indoor quiet",
    "Studio recording", 
    "Outdoor ambient",
    "Large hall reverb",
    "Phone call quality"
]
```

### Multi-Speaker Synthesis

```python
dialogue = [
    {"speaker": "Alice", "text": "Hello!"},
    {"speaker": "Bob", "text": "Hi there!"}
]

speaker_voices = {
    "Alice": "alice_ref.wav",
    "Bob": "bob_ref.wav"
}

result = generator.synthesize_multi_speaker(
    dialogue=dialogue,
    speaker_voices=speaker_voices,
    output_path="dialogue.wav"
)
```

## Performance Tips

1. **Memory Optimization**
   - Enable gradient checkpointing
   - Use bfloat16 precision
   - Reduce batch size if OOM

2. **Speed Optimization**
   - Use Flash Attention 2
   - Enable CUDA graphs for inference
   - Use DeepSpeed Zero-2/3

3. **Quality Optimization**
   - Use higher LoRA rank (64-128)
   - Train for more epochs
   - Use diverse training data
   - Apply data augmentation

## Troubleshooting

### Common Issues

1. **CUDA OOM**
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
   # Reduce batch size or enable gradient checkpointing
   ```

2. **Audio Loading Errors**
   ```python
   # Ensure librosa is installed
   pip install librosa soundfile
   ```

3. **Model Loading Issues**
   ```python
   # Enable trust_remote_code
   model = AutoModel.from_pretrained(
       model_path,
       trust_remote_code=True
   )
   ```

## Citation

If you use this integration, please cite:

```bibtex
@article{higgsaudio2024,
  title={Higgs-Audio: A Unified Audio-Language Model},
  author={Boson AI Team},
  year={2024}
}
```

## License

This integration follows MS-SWIFT's Apache 2.0 license. Higgs-Audio models have their own licensing terms.

## Support

- Issues: Create an issue in this repository
- Discussions: Join MS-SWIFT community
- Documentation: See MS-SWIFT docs

---

**Note**: This is a production-ready integration designed for scalable training and deployment. All code has been validated for correctness and follows best practices.
