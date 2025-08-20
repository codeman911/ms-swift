"""Higgs-Audio V2 model registration for MS-SWIFT using original wrapper classes.

This module uses the exact Higgs-Audio model classes and registration patterns
from the original boson_multimodal codebase.
"""

import os
import sys
import torch
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

# Add Higgs-Audio path to sys.path for imports
higgs_audio_path = Path(__file__).parent.parent / "higgs-audio"
if str(higgs_audio_path) not in sys.path:
    sys.path.insert(0, str(higgs_audio_path))

# Import original Higgs-Audio components
from boson_multimodal.model.higgs_audio import HiggsAudioModel, HiggsAudioConfig
from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
from transformers import AutoTokenizer, AutoConfig, AutoModel

# Import MS-SWIFT components
from swift.llm import ModelType, register_model, ModelMeta, ModelGroup, Model
from swift.llm.utils.model import ModelArch
from swift.llm.constant import LLMModelType
from swift.utils import get_logger

logger = get_logger()

# Register Higgs-Audio model with transformers Auto classes
try:
    AutoConfig.register("higgs_audio", HiggsAudioConfig)
    AutoModel.register(HiggsAudioConfig, HiggsAudioModel)
except Exception:
    pass  # Already registered

# LoRA target modules for Higgs-Audio V2 (from original model architecture)
HIGGS_LORA_TARGET_MODULES = [
    # Text decoder layers (LLaMA backbone)
    "layers.*.self_attn.q_proj",
    "layers.*.self_attn.k_proj", 
    "layers.*.self_attn.v_proj",
    "layers.*.self_attn.o_proj",
    "layers.*.mlp.gate_proj",
    "layers.*.mlp.up_proj",
    "layers.*.mlp.down_proj",
    # Audio-specific modules
    "layers.*.mlp.gate_proj_audio",
    "layers.*.mlp.up_proj_audio",
    "layers.*.mlp.down_proj_audio",
    # Audio projectors and heads
    "audio_projector.*",
    "audio_head.*",
    # Embeddings (optional, can be frozen)
    # "embed_tokens",
    # "lm_head",
]

def register_higgs_audio_models():
    """Register Higgs-Audio models with MS-SWIFT following CUSTOM_TTS.md specifications."""
    
    # Register main model as per CUSTOM_TTS.md
    register_model(
        ModelMeta(
            model_type='higgs-audio',  # Unique identifier as per doc
            model_groups=[Model('bosonai/higgs-audio-v2-generation-3B-base')],  # HF model name
            template='higgs-audio-chatml',  # Custom template we'll register
            get_model_tokenizer=get_higgs_model_tokenizer,  # Function to load model/tokenizer
            model_arch=ModelArch.llama,  # Base architecture
            is_multimodal=True,  # Handles audio
            requires_attention_mask=True,
            tags=['audio', 'tts', 'voice-cloning'],
            lora_target_modules=HIGGS_LORA_TARGET_MODULES,
        ),
        exist_ok=True
    )
    logger.info("Registered Higgs-Audio model with MS-SWIFT")

def get_higgs_model_tokenizer(
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
            
            # Add required configurations for Higgs-Audio
            model_kwargs.update({
                'trust_remote_code': True,
                'torch_dtype': torch_dtype or torch.bfloat16,
                'device_map': 'auto',
            })
            
            # Remove flash attention if not available
            if 'attn_implementation' not in model_kwargs:
                try:
                    import flash_attn
                    model_kwargs['attn_implementation'] = 'flash_attention_2'
                except ImportError:
                    logger.info("Flash attention not available, using default attention")
            
            # Load model using HiggsAudioModel from boson_multimodal
            model = HiggsAudioModel.from_pretrained(
                model_dir,
                **model_kwargs
            )
            
            # Follow original Higgs-Audio patterns - no custom modifications
            model.eval()
            
            # Fix gradient checkpointing compatibility
            # The issue is that PEFT wraps the model and the gradient checkpointing
            # is applied to the wrapped model, but the PEFT wrapper doesn't have
            # get_input_embeddings implemented. We need to access the base model.
            def enable_input_require_grads(self):
                def make_inputs_require_grads(module, input, output):
                    output.requires_grad_(True)
                
                # Access the actual HiggsAudioModel through PEFT wrapper
                # The model will be wrapped by PEFT later, so we need to handle both cases
                if hasattr(self, 'base_model') and hasattr(self.base_model, 'model'):
                    # PEFT wrapped model: PeftModel -> LoraModel -> HiggsAudioModel
                    base_model = self.base_model.model
                elif hasattr(self, 'model'):
                    # Direct PEFT model access
                    base_model = self.model
                else:
                    # Direct model access
                    base_model = self
                
                # Get input embeddings from the actual HiggsAudioModel
                input_embeddings = base_model.embed_tokens
                self._require_grads_hook = input_embeddings.register_forward_hook(make_inputs_require_grads)
            
            # Bind the method to the model instance to override the base class method
            import types
            model.enable_input_require_grads = types.MethodType(enable_input_require_grads, model)
            
            # Also add disable method for completeness
            def disable_input_require_grads(self):
                if hasattr(self, '_require_grads_hook'):
                    self._require_grads_hook.remove()
                    delattr(self, '_require_grads_hook')
            
            model.disable_input_require_grads = types.MethodType(disable_input_require_grads, model)
            
            # Override forward method to handle 'labels' argument
            original_forward = model.forward
            
            def forward_with_labels_mapping(
                input_ids=None,
                attention_mask=None,
                labels=None,  # MS-SWIFT passes 'labels'
                **kwargs
            ):
                # Log forward method inputs for debugging
                logger.info(f"[MODEL DEBUG] Forward called with:")
                logger.info(f"[MODEL DEBUG]   input_ids: {input_ids.shape if input_ids is not None else 'None'}")
                logger.info(f"[MODEL DEBUG]   attention_mask: {attention_mask.shape if attention_mask is not None else 'None'}")
                logger.info(f"[MODEL DEBUG]   labels: {labels.shape if labels is not None else 'None'}")
                
                # Log other kwargs
                audio_related_keys = [k for k in kwargs.keys() if 'audio' in k.lower()]
                for key in audio_related_keys:
                    value = kwargs[key]
                    if hasattr(value, 'shape'):
                        logger.info(f"[MODEL DEBUG]   {key}: shape={value.shape}, dtype={value.dtype}")
                    else:
                        logger.info(f"[MODEL DEBUG]   {key}: type={type(value)}")
                
                # Map 'labels' to 'label_ids' for HiggsAudioModel
                if labels is not None:
                    kwargs['label_ids'] = labels
                
                # Call original forward
                outputs = original_forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **kwargs
                )
                
                # Log output structure
                logger.info(f"[MODEL DEBUG] Model outputs:")
                if hasattr(outputs, 'logits') and outputs.logits is not None:
                    logger.info(f"[MODEL DEBUG]   logits shape: {outputs.logits.shape}")
                if hasattr(outputs, 'audio_logits') and outputs.audio_logits is not None:
                    logger.info(f"[MODEL DEBUG]   audio_logits shape: {outputs.audio_logits.shape}")
                if hasattr(outputs, 'loss'):
                    logger.info(f"[MODEL DEBUG]   loss: {outputs.loss}")
                if hasattr(outputs, 'llm_loss'):
                    logger.info(f"[MODEL DEBUG]   llm_loss: {outputs.llm_loss}")
                if hasattr(outputs, 'audio_loss'):
                    logger.info(f"[MODEL DEBUG]   audio_loss: {outputs.audio_loss}")
                
                # Compute loss if labels provided
                if labels is not None and outputs.logits is not None:
                    # Compute text loss (cross-entropy)
                    from torch.nn import CrossEntropyLoss
                    loss_fct = CrossEntropyLoss(ignore_index=-100)
                    
                    # Reshape for loss computation
                    logits = outputs.logits
                    batch_size, seq_len, vocab_size = logits.shape
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = labels[:, 1:].contiguous()
                    
                    # Compute loss
                    loss = loss_fct(
                        shift_logits.view(-1, vocab_size),
                        shift_labels.view(-1)
                    )
                    
                    logger.info(f"[MODEL DEBUG] Computed text loss: {loss.item()}")
                    
                    # Set loss fields for MS-SWIFT compatibility
                    outputs.loss = loss
                    outputs.llm_loss = loss
                
                return outputs
            
            model.forward = types.MethodType(forward_with_labels_mapping, model)
            
            logger.info("Model loaded and set to eval mode")
        
        return model, tokenizer
    
    # Register the model with MS-SWIFT
    register_model(
        ModelMeta(
            model_type=model_type,
            model_groups=[
                ModelGroup(
                    models=[
                        Model(
                            hf_model_id=model_id,
                            hf_revision=revision,
                        ),
                    ],
                ),
            ],
            get_function=get_model_tokenizer,
            template="chatml",  # Use standard chatml template with preprocessed data
            requires=["transformers>=4.37.0", "librosa", "soundfile"],
            torch_dtype=torch.bfloat16,
            tags=["audio", "tts", "multimodal", "llm", "voice-cloning"],
            additional_saved_files=["audio_tokenizer.json", "audio_config.json"],
            architectures=["HiggsAudioModel"],
            is_multimodal=True,
        ),
        exist_ok=True,
    )
