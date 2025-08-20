"""Higgs-Audio model registration for MS-SWIFT.

This module registers the Higgs-Audio models with MS-SWIFT's model registry.
"""

import os
from typing import Any, Dict, Optional

from transformers import AutoTokenizer, AutoModelForCausalLM

from swift.utils import get_logger
from ..register import register_model, ModelMeta, ModelGroup, Model, get_model_tokenizer_with_flash_attn
from ..constant import ModelType
from ...template import TemplateType
from ..model_arch import ModelArch

logger = get_logger()


def get_model_tokenizer_higgs_audio(
    model_dir: str,
    model_info,
    model_kwargs: Dict[str, Any],
    load_model: bool = True,
    **kwargs
) -> tuple:
    """Load Higgs-Audio model and tokenizer.
    
    Args:
        model_dir: Path to model directory
        model_kwargs: Additional model arguments
        load_model: Whether to load the model
        **kwargs: Additional arguments
    
    Returns:
        Tuple of (model, tokenizer)
    """
    # Load via framework helper (handles dtype, device_map, attn_impl, etc.)
    model, tokenizer = get_model_tokenizer_with_flash_attn(
        model_dir, model_info, model_kwargs, load_model, **kwargs
    )

    # Add special tokens for audio
    special_tokens = {
        'additional_special_tokens': [
            '<|audio_start|>',
            '<|audio_end|>',
            '<|ref_audio_start|>',
            '<|ref_audio_end|>',
            '<|text_start|>',
            '<|text_end|>',
        ]
    }
    num_new_tokens = tokenizer.add_special_tokens(special_tokens)

    # Resize token embeddings if needed
    if load_model and model is not None and num_new_tokens > 0:
        try:
            if hasattr(model, 'get_input_embeddings'):
                model.resize_token_embeddings(len(tokenizer))
        except Exception:
            pass

    return model, tokenizer


# Define Higgs-Audio model types
class HiggsAudioModelType:
    higgs_audio_full = 'higgs-audio-full'
    higgs_audio_lora = 'higgs-audio-lora'
    higgs_audio_qlora = 'higgs-audio-qlora'


# Register model types
if not hasattr(ModelType, 'higgs_audio_full'):
    ModelType.higgs_audio_full = HiggsAudioModelType.higgs_audio_full
if not hasattr(ModelType, 'higgs_audio_lora'):
    ModelType.higgs_audio_lora = HiggsAudioModelType.higgs_audio_lora
if not hasattr(ModelType, 'higgs_audio_qlora'):
    ModelType.higgs_audio_qlora = HiggsAudioModelType.higgs_audio_qlora


# Register Higgs-Audio models
register_model(
    ModelMeta(
        model_groups=[
            ModelGroup([
                Model('bosonai/higgs-audio-v2-generation-3B-base', 'bosonai/higgs-audio-v2-generation-3B-base'),
                Model('bosonai/higgs-audio-v2-generation-7B-base', 'bosonai/higgs-audio-v2-generation-7B-base'),
            ])
        ],
        model_type=HiggsAudioModelType.higgs_audio_full,
        get_function=get_model_tokenizer_higgs_audio,
        template=TemplateType.higgs_chatml,
        architectures=['HiggsAudioModel', 'LlamaForCausalLM'],
        model_arch=ModelArch.llama,
        is_multimodal=True,
        requires=['transformers>=4.37.0'],
        tags=['audio', 'tts', 'voice-cloning', 'multi-modal'],
    ))

logger.info(f"Registered Higgs-Audio models: {HiggsAudioModelType.higgs_audio_full}")

register_model(
    ModelMeta(
        model_groups=[
            ModelGroup([
                Model('bosonai/higgs-audio-v2-generation-3B-lora', 'bosonai/higgs-audio-v2-generation-3B-lora'),
                Model('bosonai/higgs-audio-v2-generation-7B-lora', 'bosonai/higgs-audio-v2-generation-7B-lora'),
            ])
        ],
        model_type=HiggsAudioModelType.higgs_audio_lora,
        get_function=get_model_tokenizer_higgs_audio,
        template=TemplateType.higgs_chatml,
        architectures=['HiggsAudioForCausalLM'],
        model_arch=None,
        is_multimodal=True,
        requires=['transformers>=4.37.0'],
        tags=['audio', 'tts', 'voice-cloning', 'lora'],
    ))

logger.info(f"Registered Higgs-Audio models: {HiggsAudioModelType.higgs_audio_full}, {HiggsAudioModelType.higgs_audio_lora}")
