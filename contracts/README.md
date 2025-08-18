# Higgs Audio Validation Contracts

This package provides strict runtime validation wrappers for the Higgs Audio model and collator without modifying any existing source files.

Contents:
- `ValidatingHiggsAudioSampleCollator`: Wraps the official `HiggsAudioSampleCollator` and validates the returned `HiggsAudioBatchInput` (shapes, dtypes, devices, and value constraints).
- `ValidatingHiggsAudioModel`: Wraps the official `HiggsAudioModel` and validates inputs prior to calling the base `forward` (plus light post-checks).
- `validators.py`: Common assertion helpers.

## Why
To catch integration issues early (mismatched shapes, wrong dtypes/devices, invalid masks or indices) when wiring Higgs Audio into ms-swift or other pipelines.

## Usage

### Collator
Replace the original collator with the validating wrapper:

```python
from higgs_audio.boson_multimodal.data_collator.higgs_audio_collator import HiggsAudioSampleCollator
from contracts import ValidatingHiggsAudioSampleCollator

base = HiggsAudioSampleCollator(
    whisper_processor=whisper_processor,
    audio_in_token_id=audio_in_token_id,
    audio_out_token_id=audio_out_token_id,
    pad_token_id=text_pad_id,
    audio_stream_bos_id=audio_bos_id,
    audio_stream_eos_id=audio_eos_id,
    round_to=8,
    pad_left=False,
    encode_whisper_embed=True,
    return_audio_in_tokens=True,
    audio_num_codebooks=audio_num_codebooks,
    use_delay_pattern=False,
)
collator = ValidatingHiggsAudioSampleCollator.__new__(ValidatingHiggsAudioSampleCollator)
collator.__dict__ = base.__dict__  # inherit all state; methods will validate on __call__
# or simply instantiate ValidatingHiggsAudioSampleCollator with the same args if preferred
```

The wrapper preserves behavior and only adds validation during `__call__`.

### Model
Use the validating model as a drop-in replacement for `HiggsAudioModel`:

```python
from contracts import ValidatingHiggsAudioModel
model = ValidatingHiggsAudioModel(config)
# Call model(...) as usual; inputs will be validated prior to compute
```

Inputs checked include:
- `input_ids` and `attention_mask` alignment and 0/1 integer mask.
- Whisper features and masks (when provided) with aligned first dimension and valid dtypes.
- Discrete audio token tensors (`audio_in_ids`, `audio_out_ids`, `label_audio_ids`) must be `torch.long` with consistent `num_codebooks`.
- Start index tensors (`audio_in_ids_start`, `audio_out_ids_start`) strictly increasing, in-range, and count matching the placeholder occurrences in `input_ids`.
- Optional `audio_out_ids_start_group_loc` length and per-value range checks.

## Notes
- These wrappers are additive and do not change underlying computation.
- They rely on the existing internal checks inside `merge_input_ids_with_audio_features`, and add pre-flight checks for clearer error messages at integration boundaries.
- If you see validation errors, fix the upstream batch construction or model call rather than disabling the checks.
