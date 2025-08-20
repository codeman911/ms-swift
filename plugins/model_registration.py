"""Higgs-Audio model registration for MS-SWIFT.

This module handles the registration of Higgs-Audio models with MS-SWIFT's model registry.
"""

import os
import sys
import types
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from swift.llm import TemplateType
from swift.llm.model import Model, ModelGroup, ModelMeta, register_model
from swift.llm.model.constant import ModelType
from swift.llm.model.model_arch import ModelArch
from swift.utils import get_logger

logger = get_logger()

# Add Higgs-Audio path to system path
higgs_audio_path = os.path.join(os.path.dirname(__file__), '..', 'higgs-audio')
if os.path.exists(higgs_audio_path):
    sys.path.insert(0, higgs_audio_path)

# Try to import Higgs-Audio components with compatibility handling
HIGGS_AUDIO_AVAILABLE = False
HiggsAudioModel = None
load_higgs_audio_tokenizer = None

try:
    # Patch for transformers compatibility
    import transformers.models.llama.modeling_llama as llama_module
    if not hasattr(llama_module, 'LLAMA_ATTENTION_CLASSES'):
        # Create a compatibility shim for older transformers versions
        llama_module.LLAMA_ATTENTION_CLASSES = {
            "eager": getattr(llama_module, 'LlamaAttention', None),
            "flash_attention_2": getattr(llama_module, 'LlamaFlashAttention2', None),
            "sdpa": getattr(llama_module, 'LlamaSdpaAttention', None),
        }
        # Remove None values
        llama_module.LLAMA_ATTENTION_CLASSES = {k: v for k, v in llama_module.LLAMA_ATTENTION_CLASSES.items() if v is not None}
    
    from boson_multimodal.model.higgs_audio.modeling_higgs_audio import HiggsAudioModel
    from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
    HIGGS_AUDIO_AVAILABLE = True
    logger.info("Successfully imported Higgs-Audio components")
except ImportError as e:
    logger.warning(f"Could not import Higgs-Audio components: {e}")
    logger.warning("Using fallback model loading without Higgs-Audio specific components")

# Model configuration
MODEL_TYPE = 'higgs-audio'
MODEL_ID = 'bosonai/higgs-audio-v2-generation-3B-base'
REVISION = None

# Register model type constant
if not hasattr(ModelType, 'higgs_audio'):
    ModelType.higgs_audio = MODEL_TYPE

def get_model_tokenizer(
        model_dir: str,
        model_info=None,
        torch_dtype: Optional[torch.dtype] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        load_model: bool = True,
        **kwargs,
    ) -> Tuple[Optional[HiggsAudioModel], Any]:
        """Load Higgs-Audio model and tokenizer as per CUSTOM_TTS.md specifications.
        
        Args:
            model_dir: Directory containing the model
            torch_dtype: Data type for model weights
            model_kwargs: Additional model loading arguments
            load_model: Whether to load the model
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (model, tokenizer)
        """
        
        # Handle torch_dtype parameter - ensure it's a valid torch dtype
        if torch_dtype is None:
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif hasattr(torch_dtype, 'torch_dtype'):  # If it's a model_info object
            torch_dtype = torch_dtype.torch_dtype
        elif not hasattr(torch_dtype, 'is_floating_point'):  # If it's not a torch dtype
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        # Load tokenizer using AutoTokenizer as per CUSTOM_TTS.md
        logger.info(f"Loading Higgs-Audio tokenizer from {model_dir}")
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        
        # Add Higgs-Audio special tokens
        special_tokens = [
            '<|audio_bos|>', '<|audio_eos|>',
            '<|audio_out_bos|>', '<|audio_out_eos|>',
            '<|AUDIO|>', '<|AUDIO_OUT|>',
            '<|DELAY|>'
        ]
        tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        
        # Set padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Fix pad_token_id issue - MS-SWIFT requires this for data collation
        # Use the same pad_token_id as in Higgs-Audio config (128001 = eos_token_id)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id  # 128001
            logger.info(f"Set pad_token_id to {tokenizer.pad_token_id} (same as eos_token_id)")
        
        model = None
        if load_model:
            # Prepare model kwargs - ensure it's a dictionary
            if model_kwargs is None or not isinstance(model_kwargs, dict):
                model_kwargs = {}
            
            # Create a copy to avoid modifying the original
            model_kwargs = model_kwargs.copy()
            
            # Filter out parameters that HiggsAudioModel doesn't accept
            # HiggsAudioModel only accepts config, not use_cache or trust_remote_code
            filtered_kwargs = {
                "torch_dtype": torch_dtype,
            }
            
            # Only add device_map if it's in the original kwargs
            if "device_map" in model_kwargs:
                filtered_kwargs["device_map"] = model_kwargs["device_map"]
            
            model_kwargs = filtered_kwargs
            
            # Add device map for multi-GPU
            if torch.cuda.device_count() > 1 and "device_map" not in model_kwargs:
                model_kwargs["device_map"] = "auto"
            
            # Ensure pad_token_id is set in model config to match tokenizer
            # Note: Don't pass pad_token_id to model constructor as it's not a valid parameter
            # The model config already has the correct pad_token_id from the pretrained config
            
            # Load Higgs-Audio model as per CUSTOM_TTS.md
            logger.info(f"Loading Higgs-Audio model from {model_dir}")
            
            # Set default model kwargs
            if model_kwargs is None:
                model_kwargs = {}
            
            try:
                # Load model with HiggsAudioForCausalLM
                # Remove duplicate parameters to avoid conflicts
                if 'torch_dtype' in model_kwargs:
                    del model_kwargs['torch_dtype']
                
                # Don't explicitly set device_map if it's already in model_kwargs
                if 'device_map' not in model_kwargs and torch.cuda.is_available():
                    model_kwargs['device_map'] = 'auto'
                
                model = HiggsAudioModel.from_pretrained(
                    model_dir,
                    torch_dtype=torch_dtype or torch.bfloat16,
                    trust_remote_code=True,
                    **(model_kwargs or {})
                )
                
                # Resize embeddings for new tokens
                model.resize_token_embeddings(len(tokenizer))
                
                # Add gradient checkpointing support
                def enable_input_require_grads(self):
                    def make_inputs_require_grads(module, input, output):
                        output.requires_grad_(True)
                    input_embeddings = self.get_input_embeddings()
                    if input_embeddings:
                        self._require_grads_hook = input_embeddings.register_forward_hook(make_inputs_require_grads)
                
                model.enable_input_require_grads = types.MethodType(enable_input_require_grads, model)
                
            except Exception as e:
                logger.warning(f"Failed to load model: {e}")
                model = None
    
        return model, tokenizer


def register_higgs_audio_models():
    """Register Higgs-Audio models with MS-SWIFT"""
    
    # Register the model with MS-SWIFT
    register_model(
        ModelMeta(
            model_type=MODEL_TYPE,
            model_groups=[
                ModelGroup(
                    models=[
                        Model(
                            hf_model_id=MODEL_ID,
                            hf_revision=REVISION,
                        ),
                    ],
                ),
            ],
            get_function=get_model_tokenizer,
            template="higgs-chatml",  # Custom template for Higgs-Audio
            requires=["transformers>=4.37.0", "librosa", "soundfile", "whisper"],
            torch_dtype=torch.bfloat16,
            tags=["audio", "tts", "multimodal", "llm", "voice-cloning"],
            additional_saved_files=["audio_tokenizer.json", "audio_config.json"],
            architectures=["HiggsAudioModel", "HiggsAudioForCausalLM"],
            is_multimodal=True,
            model_arch=ModelArch.llama,  # Base architecture is LLaMA
        ),
        exist_ok=True,
    )
    
    logger.info("✅ Higgs-Audio model registered successfully")
