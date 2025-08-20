# Higgs-Audio Training Pipeline Documentation

## Overview
This document explains how the current implementation strictly follows the CUSTOM_TTS.md specifications for integrating Higgs-Audio with MS-SWIFT.

## Component Alignment with CUSTOM_TTS.md

### 1. Model Registration (Section 1 of CUSTOM_TTS.md)
**File**: `plugins/model_registration.py`

✅ **Implemented as specified**:
- Registers model using `ModelMeta` with type `'higgs-audio'`
- Uses `AutoTokenizer` with `trust_remote_code=True`
- Adds special tokens: `<|AUDIO|>`, `<|AUDIO_OUT|>`, `<|audio_bos|>`, `<|audio_eos|>`, `<|DELAY|>`
- Configures model loading with proper device mapping
- Enables gradient checkpointing support via `enable_input_require_grads`

### 2. Template Registration (Section 2 of CUSTOM_TTS.md)
**File**: `plugins/template.py`

✅ **Implemented as specified**:
- Custom `HiggsAudioTemplate` class extending base Template
- ChatML format with `<|im_start|>` and `<|im_end|>` tokens
- Processes inline `<audio>` tags in messages
- Default system prompt for voice cloning capability
- Audio placeholder replacement logic in `_encode` method

### 3. Dataset Registration (Section 3 of CUSTOM_TTS.md)
**File**: `plugins/dataset_registration.py`

✅ **Implemented as specified**:
- Custom preprocessing function `higgs_preprocess`
- Converts multimodal JSON messages to conversation format
- Extracts audio URLs and creates inline placeholders
- Handles both TTS and voice cloning datasets
- Supports the exact JSON structure from documentation:
  ```json
  {
    "messages": [...],
    "speaker": "...",
    "misc": {...}
  }
  ```

### 4. Training Plugin (Section 4 of CUSTOM_TTS.md)
**File**: `plugins/trainer_plugin.py`

✅ **Implemented as specified**:
- Custom `HiggsAudioTrainerPlugin` extending `PeftTrainerPlugin`
- Computes combined loss: `total_loss = llm_loss + audio_loss`
- Logs text and audio losses separately
- Integrates with MS-SWIFT training loop

### 5. Data Collator (Original Higgs-Audio)
**File**: `plugins/collator.py`

✅ **Maintains compatibility**:
- Uses original `HiggsAudioDataCollator` from boson_multimodal
- Handles audio feature extraction with librosa
- Processes both reference and target audio
- Implements proper padding and batching
- Added strategic logging for debugging

## Data Flow

### Input Processing Pipeline:
1. **Dataset Loading**: JSONL file → `higgs_preprocess` function
2. **Message Conversion**: Multimodal messages → ChatML format with `<audio>` placeholders
3. **Template Encoding**: Text with placeholders → Token IDs with audio positions
4. **Collation**: Individual samples → Batched tensors with audio features
5. **Model Forward**: Batched inputs → DualFFN processing → Loss computation

### Loss Computation:
```python
# As per CUSTOM_TTS.md Section 4
total_loss = llm_loss + audio_loss
```

## Training Script Configuration

**File**: `train_higgs_plugins.sh`

The training script follows MS-SWIFT conventions while respecting Higgs-Audio requirements:

- **Model Type**: `higgs-audio` (registered via plugin)
- **Template**: `higgs-audio-chatml` (custom template)
- **Dataset**: `higgs_audio:path` (custom dataset)
- **LoRA Configuration**: Target all modules with rank 8
- **Batch Size**: 1 per device with gradient accumulation
- **Mixed Precision**: BF16 for stable training
- **Gradient Checkpointing**: Enabled for memory efficiency

## Key Design Decisions

1. **Plugin Architecture**: All components are registered as plugins to maintain modularity
2. **Original Code Reuse**: Leverages boson_multimodal components where possible
3. **MS-SWIFT Integration**: Follows framework patterns for seamless integration
4. **Error Handling**: Robust validation and logging throughout pipeline
5. **Performance**: Optimized data loading with workers and caching

## Validation Points

The pipeline validates:
- ✅ Model loads with correct architecture
- ✅ Tokenizer includes all special tokens
- ✅ Dataset preprocessing maintains audio-text alignment
- ✅ Template handles multimodal content
- ✅ Collator processes audio features correctly
- ✅ Loss computation combines text and audio losses

## Running the Pipeline

```bash
# Basic training
./train_higgs_plugins.sh

# With custom dataset and output
./train_higgs_plugins.sh my_dataset.jsonl ./my_output

# With local model path
./train_higgs_plugins.sh dataset.jsonl output/ /path/to/model
```

## Monitoring Training

Training progress can be monitored via:
- TensorBoard logs in `{output_dir}/logs`
- Separate text_loss and audio_loss metrics
- Model checkpoints every 500 steps
- Detailed logging every 10 steps

## Next Steps for Production

1. Scale dataset to full training corpus
2. Adjust hyperparameters based on validation metrics
3. Implement evaluation pipeline for voice quality
4. Add inference optimization for deployment
5. Create model card with performance metrics

## Compliance Summary

✅ All 4 sections of CUSTOM_TTS.md are fully implemented
✅ Data format matches specification exactly
✅ Loss computation follows documented formula
✅ Registration pattern aligns with MS-SWIFT conventions
✅ Training script uses correct parameters and plugins

This implementation provides a complete, production-ready training pipeline for Higgs-Audio within the MS-SWIFT framework, strictly following the CUSTOM_TTS.md specifications.
