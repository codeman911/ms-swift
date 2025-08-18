from __future__ import annotations

import torch
from typing import List, Optional
import sys
import os

# Add higgs-audio to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'higgs-audio'))

from boson_multimodal.data_collator.higgs_audio_collator import (
    HiggsAudioSampleCollator,
    HiggsAudioBatchInput,
)

from .validators import (
    assert_is_long,
    assert_is_float,
    assert_ndim,
    assert_shape,
    assert_same_device,
    assert_mask01,
    assert_monotonic_starts,
    count_token_occurrences,
)


class ValidatingHiggsAudioSampleCollator(HiggsAudioSampleCollator):
    """
    Wrapper around `HiggsAudioSampleCollator` that enforces strict runtime
    validation of the returned `HiggsAudioBatchInput`.

    This class does not change behavior, only validates shapes/dtypes/devices/value constraints.
    """

    def __call__(self, batch) -> HiggsAudioBatchInput:
        ret = super().__call__(batch)
        self.validate_batch(ret)
        return ret

    def validate_batch(self, ret: HiggsAudioBatchInput) -> None:
        ctx = "ValidatingHiggsAudioSampleCollator"

        # Basic shapes for core text inputs
        assert_is_long(ctx, ret.input_ids, "input_ids")
        assert_ndim(ctx, ret.input_ids, "input_ids", 2)

        assert_ndim(ctx, ret.attention_mask, "attention_mask", 2)
        assert_shape(ctx, ret.attention_mask, "attention_mask", [ret.input_ids.shape[0], ret.input_ids.shape[1]])
        assert_mask01(ctx, ret.attention_mask, "attention_mask")

        if ret.label_ids is not None:
            assert_is_long(ctx, ret.label_ids, "label_ids")
            assert_shape(ctx, ret.label_ids, "label_ids", [ret.input_ids.shape[0], ret.input_ids.shape[1]])

        # Audio feature tensors
        if self.encode_whisper_embed:
            # When encoding whisper features, features and masks must be present and aligned
            assert_is_float(ctx, ret.audio_features, "audio_features")
            assert_ndim(ctx, ret.audio_features, "audio_features", 3)  # [num_audio_in, feat_dim, max_mel_T]

            assert_ndim(
                ctx, ret.audio_feature_attention_mask, "audio_feature_attention_mask", 2
            )  # [num_audio_in, max_mel_T]
            assert_mask01(ctx, ret.audio_feature_attention_mask, "audio_feature_attention_mask")

            # If we also return audio_in tokens, check num_audio_in alignment
            if self.return_audio_in_tokens and ret.audio_in_ids_start is not None:
                assert_shape(
                    ctx,
                    ret.audio_feature_attention_mask,
                    "audio_feature_attention_mask",
                    [ret.audio_in_ids_start.numel(), ret.audio_feature_attention_mask.shape[1]],
                )
                assert_shape(
                    ctx,
                    ret.audio_features,
                    "audio_features",
                    [ret.audio_in_ids_start.numel(), ret.audio_features.shape[1], ret.audio_features.shape[2]],
                )
        else:
            # Not using whisper features: both should be None
            if ret.audio_features is not None or ret.audio_feature_attention_mask is not None:
                raise ValueError(
                    f"[{ctx}] encode_whisper_embed=False but audio_features/audio_feature_attention_mask not None"
                )

        # Audio tokens (in/out) and starts
        # All ids must be torch.long if present
        if ret.audio_in_ids is not None:
            assert_is_long(ctx, ret.audio_in_ids, "audio_in_ids")
            assert_ndim(ctx, ret.audio_in_ids, "audio_in_ids", 2)
        if ret.audio_out_ids is not None:
            assert_is_long(ctx, ret.audio_out_ids, "audio_out_ids")
            assert_ndim(ctx, ret.audio_out_ids, "audio_out_ids", 2)
        if ret.label_audio_ids is not None:
            assert_is_long(ctx, ret.label_audio_ids, "label_audio_ids")
            assert_ndim(ctx, ret.label_audio_ids, "label_audio_ids", 2)

        # Starts arrays
        if ret.audio_in_ids_start is not None:
            assert_is_long(ctx, ret.audio_in_ids_start, "audio_in_ids_start")
            assert_ndim(ctx, ret.audio_in_ids_start, "audio_in_ids_start", 1)
            total_len_in = int(ret.audio_in_ids.shape[-1]) if ret.audio_in_ids is not None else 0
            assert_monotonic_starts(ctx, ret.audio_in_ids_start, total_len_in, "audio_in_ids_start")

        if ret.audio_out_ids_start is not None:
            assert_is_long(ctx, ret.audio_out_ids_start, "audio_out_ids_start")
            assert_ndim(ctx, ret.audio_out_ids_start, "audio_out_ids_start", 1)
            total_len_out = int(ret.audio_out_ids.shape[-1]) if ret.audio_out_ids is not None else 0
            assert_monotonic_starts(ctx, ret.audio_out_ids_start, total_len_out, "audio_out_ids_start")

        # Group loc should align with audio_out_ids_start
        if ret.audio_out_ids_start_group_loc is not None:
            assert_is_long(ctx, ret.audio_out_ids_start_group_loc, "audio_out_ids_start_group_loc")
            assert_ndim(ctx, ret.audio_out_ids_start_group_loc, "audio_out_ids_start_group_loc", 1)
            if ret.audio_out_ids_start is not None:
                assert_shape(
                    ctx,
                    ret.audio_out_ids_start_group_loc,
                    "audio_out_ids_start_group_loc",
                    [ret.audio_out_ids_start.numel()],
                )
            # values in [0, batch_size-1]
            if ret.audio_out_ids_start_group_loc.numel() > 0:
                bsz = int(ret.input_ids.shape[0])
                if not (ret.audio_out_ids_start_group_loc.min().item() >= 0 and ret.audio_out_ids_start_group_loc.max().item() < bsz):
                    raise ValueError(
                        f"[{ctx}] audio_out_ids_start_group_loc must be within [0, {bsz-1}]"
                    )

        # Placeholder counts must match number of starts
        audio_in_count = count_token_occurrences(ret.input_ids, self.audio_in_token_id)
        audio_out_count = count_token_occurrences(ret.input_ids, self.audio_out_token_id)

        if self.return_audio_in_tokens:
            expected_in = audio_in_count
            got_in = int(ret.audio_in_ids_start.numel()) if ret.audio_in_ids_start is not None else 0
            if expected_in != got_in:
                raise ValueError(
                    f"[{ctx}] Mismatch in <|AUDIO|> occurrences vs audio_in_ids_start. expected={expected_in}, got={got_in}"
                )
            if expected_in > 0 and ret.audio_in_ids is None:
                raise ValueError(f"[{ctx}] audio_in_ids is None but there are {expected_in} <|AUDIO|> placeholders")
        else:
            if ret.audio_in_ids is not None or ret.audio_in_ids_start is not None:
                raise ValueError(
                    f"[{ctx}] return_audio_in_tokens=False but audio_in_ids/audio_in_ids_start are not None"
                )

        expected_out = audio_out_count
        got_out = int(ret.audio_out_ids_start.numel()) if ret.audio_out_ids_start is not None else 0
        if expected_out != got_out:
            raise ValueError(
                f"[{ctx}] Mismatch in <|AUDIO_OUT|> occurrences vs audio_out_ids_start. expected={expected_out}, got={got_out}"
            )
        if expected_out > 0 and ret.audio_out_ids is None:
            raise ValueError(f"[{ctx}] audio_out_ids is None but there are {expected_out} <|AUDIO_OUT|> placeholders")

        # Ensure the first dimension (num_codebooks) matches across audio tensors
        codebooks = []
        for t in (ret.audio_in_ids, ret.audio_out_ids, ret.label_audio_ids):
            if t is not None and t.numel() > 0:
                codebooks.append(int(t.shape[0]))
        if len(codebooks) > 1 and len(set(codebooks)) != 1:
            raise ValueError(f"[{ctx}] Inconsistent num_codebooks across audio tensors: {codebooks}")

        # Device consistency (collator typically returns CPU tensors, just ensure all the same)
        tensors = [
            ret.input_ids,
            ret.attention_mask,
            ret.label_ids,
            ret.audio_features,
            ret.audio_feature_attention_mask,
            ret.audio_in_ids,
            ret.audio_in_ids_start,
            ret.audio_out_ids,
            ret.audio_out_ids_start,
            ret.audio_out_ids_start_group_loc,
            ret.label_audio_ids,
        ]
        assert_same_device(ctx, [t for t in tensors if t is not None])
