"""Higgs-Audio model registration for MS-SWIFT.

This module registers the Higgs-Audio model with MS-SWIFT's model registry.
"""

import os
from typing import Any, Dict, Optional, Tuple

from swift.llm import TemplateType
from swift.utils import get_logger
from ..constant import ModelType
from ..register import Model, ModelGroup, ModelMeta, register_model

logger = get_logger()


def get_model_tokenizer_higgs_audio(
    model_dir: str,
    torch_dtype: Optional[Any] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    load_model: bool = True,
    **kwargs
) -> Tuple[Any, Any]:
    """Load Higgs-Audio model and tokenizer.
    
    Args:
        model_dir: Directory containing the model
        torch_dtype: Data type for model weights
        model_kwargs: Additional model arguments
        load_model: Whether to load the model
        **kwargs: Additional arguments
    
    Returns:
        Tuple of (model, tokenizer)
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True,
        use_fast=False
    )
    
    # Add special tokens for audio
    special_tokens = {
        'additional_special_tokens': [
            '<|audio_start|>',
            '<|audio_end|>',
            '<|audio_pad|>',
            '<|ref_audio|>',
            '<|target_audio|>'
        ]
    }
    tokenizer.add_special_tokens(special_tokens)
    
    if not load_model:
        return None, tokenizer
    
    # Initialize model
    if model_kwargs is None:
        model_kwargs = {}
    
    if torch_dtype is None:
        torch_dtype = torch.float16
    
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        **model_kwargs
    )
    
    # Resize embeddings if needed
    if len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer


# Create a custom ModelType for Higgs-Audio
class HiggsAudioModelType:
    higgs_audio_full = 'higgs-audio-full'
    higgs_audio_lora = 'higgs-audio-lora'


# Add to ModelType if possible
if hasattr(ModelType, '__dict__'):
    ModelType.higgs_audio_full = HiggsAudioModelType.higgs_audio_full
    ModelType.higgs_audio_lora = HiggsAudioModelType.higgs_audio_lora


# Register Higgs-Audio model
register_model(
    ModelMeta(
        HiggsAudioModelType.higgs_audio_full,
        [
            ModelGroup([
                Model('Higgs-Audio-Full', 'higgs-audio/higgs-audio-full'),
            ]),
        ],
        TemplateType.higgs_chatml,
        get_model_tokenizer_higgs_audio,
        architectures=['HiggsAudioForCausalLM'],
        model_arch=None,
        requires=['transformers>=4.37.0'],
        tags=['audio', 'tts', 'voice-cloning'],
    ))

register_model(
    ModelMeta(
        HiggsAudioModelType.higgs_audio_lora,
        [
            ModelGroup([
                Model('Higgs-Audio-LoRA', 'higgs-audio/higgs-audio-lora'),
            ]),
        ],
        TemplateType.higgs_chatml,
        get_model_tokenizer_higgs_audio,
        architectures=['HiggsAudioForCausalLM'],
        model_arch=None,
        requires=['transformers>=4.37.0'],
        tags=['audio', 'tts', 'voice-cloning', 'lora'],
    ))

logger.info(f"Registered Higgs-Audio models: {HiggsAudioModelType.higgs_audio_full}, {HiggsAudioModelType.higgs_audio_lora}")
