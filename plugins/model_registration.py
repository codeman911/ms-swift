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
from swift.utils import get_logger

logger = get_logger()

# Register Higgs-Audio model with transformers Auto classes
AutoConfig.register("higgs_audio", HiggsAudioConfig)
AutoModel.register(HiggsAudioConfig, HiggsAudioModel)

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
    """Register Higgs-Audio models with MS-SWIFT using original wrapper classes."""
    
    # Define model configurations
    model_configs = [
        {
            "model_type": "higgs-audio-v2",
            "model_id": "Boson-AI/Higgs-Audio-V2",
            "revision": "main",
        },
        {
            "model_type": "higgs-audio-v2-llama3",
            "model_id": "Boson-AI/Higgs-Audio-V2-LLaMA3",
            "revision": "main",
        },
        {
            "model_type": "higgs-audio-v2-generation-3b-base",
            "model_id": "bosonai/higgs-audio-v2-generation-3B-base",
            "revision": "main",
        },
    ]
    
    for config in model_configs:
        register_higgs_audio_model(
            model_type=config["model_type"],
            model_id=config["model_id"],
            revision=config.get("revision", "main"),
        )
        logger.info(f"Registered Higgs-Audio model: {config['model_type']}")

def register_higgs_audio_model(
    model_type: str,
    model_id: str,
    revision: str = "main",
):
    """Register a single Higgs-Audio model with MS-SWIFT.
    
    Args:
        model_type: The model type identifier
        model_id: The HuggingFace model ID
        revision: The model revision
    """
    
    def get_model_tokenizer(
        model_dir: str,
        model_info=None,
        torch_dtype: Optional[torch.dtype] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        load_model: bool = True,
        **kwargs,
    ) -> Tuple[Optional[HiggsAudioModel], Any]:
        """Load Higgs-Audio model and tokenizer using original wrapper classes.
        
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
        
        # Load text tokenizer (not the audio tokenizer)
        # The load_higgs_audio_tokenizer loads the audio tokenizer, we need the text tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_dir,
                trust_remote_code=True,
                use_fast=False,
            )
            logger.info(f"Loaded text tokenizer from {model_dir}")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise e
            
        # Load audio tokenizer separately for later use
        try:
            # Custom loading function that filters out unexpected parameters
            def load_audio_tokenizer_safe(tokenizer_name_or_path, device="cuda"):
                import json
                import os
                from huggingface_hub import snapshot_download
                
                is_local = os.path.exists(tokenizer_name_or_path)
                if not is_local:
                    tokenizer_path = snapshot_download(tokenizer_name_or_path)
                else:
                    tokenizer_path = tokenizer_name_or_path
                    
                config_path = os.path.join(tokenizer_path, "config.json")
                model_path = os.path.join(tokenizer_path, "model.pth")
                config = json.load(open(config_path))
                
                # Filter out parameters that HiggsAudioTokenizer doesn't accept
                valid_params = {
                    'n_filters', 'D', 'target_bandwidths', 'ratios', 'sample_rate',
                    'bins', 'n_q', 'codebook_dim', 'normalize', 'causal',
                    'semantic_techer', 'last_layer_semantic', 'merge_mode',
                    'downsample_mode', 'semantic_mode', 'vq_scale', 'semantic_sample_rate'
                }
                
                filtered_config = {k: v for k, v in config.items() if k in valid_params}
                
                model = HiggsAudioTokenizer(
                    **filtered_config,
                    device=device,
                )
                parameter_dict = torch.load(model_path, map_location=device)
                model.load_state_dict(parameter_dict, strict=False)
                model.to(device)
                model.eval()
                return model
            
            audio_tokenizer = load_audio_tokenizer_safe(model_dir)
            # Store audio tokenizer as an attribute
            tokenizer.audio_tokenizer = audio_tokenizer
            logger.info(f"Loaded Higgs-Audio audio tokenizer from {model_dir}")
        except Exception as e:
            logger.warning(f"Failed to load Higgs-Audio audio tokenizer: {e}")
            tokenizer.audio_tokenizer = None
        
        # Ensure special tokens are set
        special_tokens = [
            "<|AUDIO|>", "<|AUDIO_OUT|>", 
            "<|audio_bos|>", "<|audio_eos|>",
            "<|DELAY|>"
        ]
        
        tokens_to_add = [t for t in special_tokens if t not in tokenizer.get_vocab()]
        if tokens_to_add:
            tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})
            logger.info(f"Added special tokens: {tokens_to_add}")
        
        # Set special token IDs as attributes
        tokenizer.audio_in_token_id = tokenizer.convert_tokens_to_ids("<|AUDIO|>")
        tokenizer.audio_out_token_id = tokenizer.convert_tokens_to_ids("<|AUDIO_OUT|>")
        tokenizer.audio_bos_token_id = tokenizer.convert_tokens_to_ids("<|audio_bos|>")
        tokenizer.audio_eos_token_id = tokenizer.convert_tokens_to_ids("<|audio_eos|>")
        tokenizer.delay_token_id = tokenizer.convert_tokens_to_ids("<|DELAY|>")
        
        model = None
        if load_model:
            # Prepare model kwargs - ensure it's a dictionary
            if model_kwargs is None or not isinstance(model_kwargs, dict):
                model_kwargs = {}
            
            # Create a copy to avoid modifying the original
            model_kwargs = model_kwargs.copy()
            model_kwargs.update({
                "torch_dtype": torch_dtype,
                "trust_remote_code": True,
                "use_cache": True,
            })
            
            # Add device map for multi-GPU
            if torch.cuda.device_count() > 1 and "device_map" not in model_kwargs:
                model_kwargs["device_map"] = "auto"
            
            # Load model using Higgs-Audio model class
            model = HiggsAudioModel.from_pretrained(
                model_dir,
                **model_kwargs,
            )
            
            # Enable gradient checkpointing for memory efficiency
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
                logger.info("Enabled gradient checkpointing")
            
            # Resize token embeddings if needed
            if len(tokenizer) > model.config.vocab_size:
                model.resize_token_embeddings(len(tokenizer))
                logger.info(f"Resized token embeddings to {len(tokenizer)}")
        
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
            template="higgs-audio-template",  # Use custom Higgs-Audio template
            requires=["transformers>=4.37.0", "librosa", "soundfile"],
            torch_dtype=torch.bfloat16,
            tags=["audio", "tts", "multimodal", "llm", "voice-cloning"],
            additional_saved_files=["audio_tokenizer.json", "audio_config.json"],
            architectures=["HiggsAudioModel"],
            is_multimodal=True,
        ),
        exist_ok=True,
    )
