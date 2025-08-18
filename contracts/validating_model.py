from __future__ import annotations

import torch
from typing import Optional
import sys
import os

# Add higgs-audio to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'higgs-audio'))

from boson_multimodal.model.higgs_audio.modeling_higgs_audio import HiggsAudioModel

from .validators import (
    assert_is_long,
    assert_is_float,
    assert_ndim,
    assert_shape,
    assert_mask01,
    assert_monotonic_starts,
    count_token_occurrences,
)


class ValidatingHiggsAudioModel(HiggsAudioModel):
    """A drop-in subclass of `HiggsAudioModel` with strict input validation.

    This wrapper enforces shape/dtype/value contracts before delegating to the
    base model. It does not modify model behavior or existing source files.
    """

    # Keep the same signature for full compatibility
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        audio_features: Optional[torch.FloatTensor] = None,
        audio_feature_attention_mask: Optional[torch.Tensor] = None,
        audio_in_ids: Optional[torch.LongTensor] = None,
        audio_in_ids_start: Optional[torch.LongTensor] = None,
        audio_out_ids: Optional[torch.LongTensor] = None,
        audio_out_ids_start: Optional[torch.LongTensor] = None,
        audio_out_ids_start_group_loc: Optional[torch.LongTensor] = None,
        label_ids: Optional[torch.LongTensor] = None,
        label_audio_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[object] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_audio_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        cache_audio_discrete_codes_mask: Optional[torch.LongTensor] = None,
        past_key_values_buckets: Optional[object] = None,
        reward: Optional[torch.FloatTensor] = None,
        **kwargs
    ):
        ctx = "ValidatingHiggsAudioModel.forward"

        # MS-SWIFT/PEFT compatibility: Map 'labels' to HiggsAudio parameter names
        if 'labels' in kwargs:
            labels = kwargs.pop('labels')
            if labels is not None and label_ids is None:
                # For HiggsAudio, MS-SWIFT's 'labels' maps to 'label_ids' (text labels)
                # label_audio_ids should be handled separately by the data collator
                label_ids = labels

        # Core text inputs
        assert_is_long(ctx, input_ids, "input_ids")
        assert_ndim(ctx, input_ids, "input_ids", 2)

        if attention_mask is not None:
            assert_ndim(ctx, attention_mask, "attention_mask", 2)
            assert_shape(ctx, attention_mask, "attention_mask", [input_ids.shape[0], input_ids.shape[1]])
            assert_mask01(ctx, attention_mask, "attention_mask")

        # Labels (text)
        if label_ids is not None:
            assert_is_long(ctx, label_ids, "label_ids")
            assert_shape(ctx, label_ids, "label_ids", [input_ids.shape[0], input_ids.shape[1]])

        # Audio features (Whisper tower). Only validate when provided (skip_audio_tower can be True)
        if audio_features is not None:
            assert_is_float(ctx, audio_features, "audio_features")
            assert_ndim(ctx, audio_features, "audio_features", 3)  # [num_audio_in, feat_dim, max_mel_T]
            if audio_feature_attention_mask is None:
                raise ValueError(f"[{ctx}] audio_feature_attention_mask must be provided when audio_features is given")
            assert_ndim(
                ctx, audio_feature_attention_mask, "audio_feature_attention_mask", 2
            )  # [num_audio_in, max_mel_T]
            assert_mask01(ctx, audio_feature_attention_mask, "audio_feature_attention_mask")
            # first dim must match
            assert_shape(
                ctx,
                audio_feature_attention_mask,
                "audio_feature_attention_mask",
                [audio_features.shape[0], audio_feature_attention_mask.shape[1]],
            )

        # Audio token ids and starts
        num_in_ph = count_token_occurrences(input_ids, getattr(self, "audio_in_token_idx"))
        num_out_ph = count_token_occurrences(input_ids, getattr(self, "audio_out_token_idx"))

        # audio_in (only required if encode_audio_in_tokens and placeholders exist)
        if getattr(self.config, "encode_audio_in_tokens", False) and num_in_ph > 0:
            if audio_in_ids is None or audio_in_ids_start is None:
                raise ValueError(
                    f"[{ctx}] encode_audio_in_tokens=True and {num_in_ph} <|AUDIO|> placeholders exist, "
                    f"but audio_in_ids/audio_in_ids_start not provided"
                )
        # when provided, validate shapes and starts
        if audio_in_ids is not None:
            assert_is_long(ctx, audio_in_ids, "audio_in_ids")
            assert_ndim(ctx, audio_in_ids, "audio_in_ids", 2)  # [num_codebooks, total_len]
        if audio_in_ids_start is not None:
            assert_is_long(ctx, audio_in_ids_start, "audio_in_ids_start")
            assert_ndim(ctx, audio_in_ids_start, "audio_in_ids_start", 1)
            total_len_in = int(audio_in_ids.shape[-1]) if audio_in_ids is not None else 0
            assert_monotonic_starts(ctx, audio_in_ids_start, total_len_in, "audio_in_ids_start")
            # count match
            if num_in_ph != int(audio_in_ids_start.numel()):
                raise ValueError(
                    f"[{ctx}] <|AUDIO|> placeholder count ({num_in_ph}) must equal len(audio_in_ids_start) "
                    f"({int(audio_in_ids_start.numel())})"
                )

        # audio_out is required if placeholders exist
        if num_out_ph > 0 and (audio_out_ids is None or audio_out_ids_start is None):
            raise ValueError(
                f"[{ctx}] {num_out_ph} <|AUDIO_OUT|> placeholders exist, but audio_out_ids/audio_out_ids_start not provided"
            )
        if audio_out_ids is not None:
            assert_is_long(ctx, audio_out_ids, "audio_out_ids")
            assert_ndim(ctx, audio_out_ids, "audio_out_ids", 2)
        if audio_out_ids_start is not None:
            assert_is_long(ctx, audio_out_ids_start, "audio_out_ids_start")
            assert_ndim(ctx, audio_out_ids_start, "audio_out_ids_start", 1)
            total_len_out = int(audio_out_ids.shape[-1]) if audio_out_ids is not None else 0
            assert_monotonic_starts(ctx, audio_out_ids_start, total_len_out, "audio_out_ids_start")
            if num_out_ph != int(audio_out_ids_start.numel()):
                raise ValueError(
                    f"[{ctx}] <|AUDIO_OUT|> placeholder count ({num_out_ph}) must equal len(audio_out_ids_start) "
                    f"({int(audio_out_ids_start.numel())})"
                )

        if audio_out_ids_start_group_loc is not None:
            # must align with starts length and be within [0, batch_size-1]
            assert_is_long(ctx, audio_out_ids_start_group_loc, "audio_out_ids_start_group_loc")
            assert_ndim(ctx, audio_out_ids_start_group_loc, "audio_out_ids_start_group_loc", 1)
            if audio_out_ids_start is not None:
                assert_shape(
                    ctx,
                    audio_out_ids_start_group_loc,
                    "audio_out_ids_start_group_loc",
                    [audio_out_ids_start.numel()],
                )
            if audio_out_ids_start_group_loc.numel() > 0:
                bsz = int(input_ids.shape[0])
                min_v = int(audio_out_ids_start_group_loc.min().item())
                max_v = int(audio_out_ids_start_group_loc.max().item())
                if not (0 <= min_v and max_v < bsz):
                    raise ValueError(
                        f"[{ctx}] audio_out_ids_start_group_loc values must be within [0, {bsz-1}], got range [{min_v}, {max_v}]"
                    )

        # label_audio_ids alignment (when provided)
        if label_audio_ids is not None:
            assert_is_long(ctx, label_audio_ids, "label_audio_ids")
            assert_ndim(ctx, label_audio_ids, "label_audio_ids", 2)
            if audio_out_ids is not None:
                assert_shape(ctx, label_audio_ids, "label_audio_ids", [audio_out_ids.shape[0], audio_out_ids.shape[1]])

        # Ensure consistent num_codebooks across audio tensors when present
        codebooks = []
        for t in (audio_in_ids, audio_out_ids, label_audio_ids):
            if t is not None and t.numel() > 0:
                codebooks.append(int(t.shape[0]))
        if len(codebooks) > 1 and len(set(codebooks)) != 1:
            raise ValueError(f"[{ctx}] Inconsistent num_codebooks across audio tensors: {codebooks}")

        # Move start indices to the same device as input_ids to satisfy downstream merge validation
        target_device = input_ids.device
        if audio_in_ids_start is not None and audio_in_ids_start.device != target_device:
            audio_in_ids_start = audio_in_ids_start.to(target_device)
        if audio_out_ids_start is not None and audio_out_ids_start.device != target_device:
            audio_out_ids_start = audio_out_ids_start.to(target_device)

        # Delegate to base model
        ret = super().forward(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            audio_features=audio_features,
            audio_feature_attention_mask=audio_feature_attention_mask,
            audio_in_ids=audio_in_ids,
            audio_in_ids_start=audio_in_ids_start,
            audio_out_ids=audio_out_ids,
            audio_out_ids_start=audio_out_ids_start,
            audio_out_ids_start_group_loc=audio_out_ids_start_group_loc,
            label_ids=label_ids,
            label_audio_ids=label_audio_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_audio_hidden_states=output_audio_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            cache_audio_discrete_codes_mask=cache_audio_discrete_codes_mask,
            past_key_values_buckets=past_key_values_buckets,
            reward=reward,
        )

        # Light post-checks
        if hasattr(ret, "attention_mask") and ret.attention_mask is not None:
            assert_ndim(ctx, ret.attention_mask, "ret.attention_mask", 2)
            assert_mask01(ctx, ret.attention_mask, "ret.attention_mask")
        if hasattr(ret, "logits") and ret.logits is not None:
            if ret.logits.ndim != 3:
                raise ValueError(f"[{ctx}] ret.logits must be 3-D, got {ret.logits.ndim}")
        if hasattr(ret, "audio_logits") and ret.audio_logits is not None:
            if ret.audio_logits.ndim != 3:
                raise ValueError(f"[{ctx}] ret.audio_logits must be 3-D, got {ret.audio_logits.ndim}")

        return ret

    def get_input_embeddings(self):
        """Return input embeddings for gradient checkpointing compatibility."""
        return self.embed_tokens

    def set_input_embeddings(self, value):
        """Set input embeddings for gradient checkpointing compatibility."""
        self.embed_tokens = value

    def get_output_embeddings(self):
        """Return output embeddings for gradient checkpointing compatibility."""
        return self.audio_decoder_proj.text_lm_head

    def set_output_embeddings(self, new_embeddings):
        """Set output embeddings for gradient checkpointing compatibility."""
        self.audio_decoder_proj.text_lm_head = new_embeddings

    def tie_weights(self):
        """Tie input and output embeddings if needed."""
        # Call parent implementation if it exists
        if hasattr(super(), 'tie_weights'):
            super().tie_weights()
