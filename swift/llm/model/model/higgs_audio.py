"""Higgs-Audio model registration for MS-SWIFT.

This module registers the Higgs-Audio models with MS-SWIFT's model registry.
"""

import os
from typing import Any, Dict, Optional

from transformers import AutoTokenizer, AutoModelForCausalLM

from swift.utils import get_logger
from ..register import register_model, ModelMeta, ModelGroup, Model
from ..constant import ModelType
from ...template import TemplateType
from ..model_arch import ModelArch

logger = get_logger()


def get_model_tokenizer_higgs_audio(
    model_dir: str,
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
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True,
        **kwargs
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
    tokenizer.add_special_tokens(special_tokens)
    
    model = None
    if load_model:
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            **model_kwargs
        )
        
        # Resize token embeddings if needed
        if len(tokenizer) > model.config.vocab_size:
            model.resize_token_embeddings(len(tokenizer))
    
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
        get_model_tokenizer=get_model_tokenizer_higgs_audio,
        template=TemplateType.higgs_chatml,
        architectures=['HiggsAudioModel', 'LlamaForCausalLM'],
        model_arch=ModelArch.llama,
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
        get_model_tokenizer=get_model_tokenizer_higgs_audio,
        template=TemplateType.higgs_chatml,
        architectures=['HiggsAudioForCausalLM'],
        model_arch=None,
        requires=['transformers>=4.37.0'],
        tags=['audio', 'tts', 'voice-cloning', 'lora'],
    ))

logger.info(f"Registered Higgs-Audio models: {HiggsAudioModelType.higgs_audio_full}, {HiggsAudioModelType.higgs_audio_lora}")
