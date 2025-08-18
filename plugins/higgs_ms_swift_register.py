# plugins/higgs_ms_swift_register.py
# Purpose: register a custom template that hands raw dataset rows to your Higgs collator
# so you can tokenize on-the-fly. Optionally register a model_type.

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.auto import CONFIG_MAPPING, MODEL_FOR_CAUSAL_LM_MAPPING
import logging
import sys
import os
import importlib
from typing import List, Dict, Any

def register_higgs_audio():
    """Register HiggsAudio model and configuration with MS-SWIFT."""
    
    try:
        # Import model after patching
        from higgs_audio.boson_multimodal.model.higgs_audio.configuration_higgs_audio import HiggsAudioConfig
        from higgs_audio.boson_multimodal.model.higgs_audio.modeling_higgs_audio import HiggsAudioModel
        
        logger.info("Loading HiggsAudio model classes...")
        
        # Register with transformers
        AutoConfig.register("higgs_audio", HiggsAudioConfig, exist_ok=True)
        AutoModelForCausalLM.register(HiggsAudioConfig, HiggsAudioModel, exist_ok=True)
        
        # Force override for the specific model we're using
        CONFIG_MAPPING._extra_content["higgs_audio"] = HiggsAudioConfig
        MODEL_FOR_CAUSAL_LM_MAPPING._extra_content[HiggsAudioConfig] = HiggsAudioModel
        
        logger.info("✓ Successfully registered HiggsAudio model with transformers auto classes")
        logger.info(f"✓ HiggsAudioModel has get_input_embeddings: {hasattr(HiggsAudioModel, 'get_input_embeddings')}")
        logger.info(f"✓ HiggsAudioModel has set_input_embeddings: {hasattr(HiggsAudioModel, 'set_input_embeddings')}")
        logger.info(f"✓ HiggsAudioModel has enable_input_require_grads: {hasattr(HiggsAudioModel, 'enable_input_require_grads')}")
        
    except Exception as e:
        print(f"[WARNING] Failed to register HiggsAudio model: {e}")

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add the higgs-audio directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'higgs-audio'))

# Import the HiggsAudioConfig and HiggsAudioModel from the local higgs-audio directory
from boson_multimodal.model.higgs_audio.configuration_higgs_audio import HiggsAudioConfig
from boson_multimodal.model.higgs_audio.modeling_higgs_audio import HiggsAudioModel

# MS-SWIFT APIs
from swift.llm import register_template, TemplateMeta, Template

# ---- import Higgs Audio code with correct paths ----
try:
    from boson_multimodal.data_collator.higgs_audio_collator import HiggsAudioSampleCollator
    from boson_multimodal.audio_processing.higgs_audio_tokenizer import HiggsAudioTokenizer
    from boson_multimodal.dataset.chatml_dataset import ChatMLDatasetSample
    from boson_multimodal.constants import AUDIO_IN_TOKEN, AUDIO_OUT_TOKEN
    print("[INFO] Successfully imported Higgs Audio components")
    
    # Register HiggsAudio model with transformers auto classes
    register_higgs_audio()
    
    # Hook into model loading to patch any dynamically loaded models
    original_from_pretrained = AutoModelForCausalLM.from_pretrained
    
    def patched_from_pretrained(pretrained_model_name_or_path, *args, **kwargs):
        """Wrapper to ensure HiggsAudio models have required methods."""
        # CRITICAL: Patch HiggsAudioModel BEFORE loading
        if 'higgs' in str(pretrained_model_name_or_path).lower():
            logger.info(f"Pre-load patching for HiggsAudioModel from {pretrained_model_name_or_path}")
            # Patch HiggsAudioModel BEFORE loading
            def patch_higgs_audio_model():
                """Comprehensively patch HiggsAudioModel class with required methods."""
                
                def get_input_embeddings(self) -> nn.Module:
                    """Return the text embedding layer for gradient checkpointing."""
                    return self.embed_tokens
                
                def set_input_embeddings(self, value: nn.Module):
                    """Set the text embedding layer for gradient checkpointing."""
                    self.embed_tokens = value
                
                def enable_input_require_grads(self):
                    """Enable gradients for input embeddings (required for gradient checkpointing)."""
                    def make_inputs_require_grad(module, input, output):
                        if output is not None:
                            for out in output if isinstance(output, tuple) else [output]:
                                if torch.is_tensor(out) and out.dtype == torch.float:
                                    out.requires_grad_(True)
                    
                    # Register forward hook on embeddings
                    if hasattr(self, '_input_require_grads_hook'):
                        self._input_require_grads_hook.remove()
                    
                    embeddings = self.get_input_embeddings()
                    if embeddings is not None:
                        self._input_require_grads_hook = embeddings.register_forward_hook(make_inputs_require_grad)
                
                def disable_input_require_grads(self):
                    """Disable gradients for input embeddings."""
                    if hasattr(self, '_input_require_grads_hook'):
                        self._input_require_grads_hook.remove()
                        delattr(self, '_input_require_grads_hook')
                
                # Patch the imported HiggsAudioModel class
                if not hasattr(HiggsAudioModel, 'get_input_embeddings'):
                    logger.info("Patching HiggsAudioModel with get_input_embeddings method")
                    HiggsAudioModel.get_input_embeddings = get_input_embeddings
                
                if not hasattr(HiggsAudioModel, 'set_input_embeddings'):
                    logger.info("Patching HiggsAudioModel with set_input_embeddings method")
                    HiggsAudioModel.set_input_embeddings = set_input_embeddings
                
                if not hasattr(HiggsAudioModel, 'enable_input_require_grads'):
                    logger.info("Patching HiggsAudioModel with enable_input_require_grads method")
                    HiggsAudioModel.enable_input_require_grads = enable_input_require_grads
                
                if not hasattr(HiggsAudioModel, 'disable_input_require_grads'):
                    logger.info("Patching HiggsAudioModel with disable_input_require_grads method")
                    HiggsAudioModel.disable_input_require_grads = disable_input_require_grads
                
                # Also patch in sys.modules if the module is already imported
                for module_name in list(sys.modules.keys()):
                    if 'higgs_audio' in module_name and 'modeling' in module_name:
                        module = sys.modules[module_name]
                        if hasattr(module, 'HiggsAudioModel'):
                            model_class = getattr(module, 'HiggsAudioModel')
                            if not hasattr(model_class, 'get_input_embeddings'):
                                logger.info(f"Patching {module_name}.HiggsAudioModel")
                                model_class.get_input_embeddings = get_input_embeddings
                                model_class.set_input_embeddings = set_input_embeddings
                                model_class.enable_input_require_grads = enable_input_require_grads
                                model_class.disable_input_require_grads = disable_input_require_grads
                
                logger.info("✓ HiggsAudioModel patching complete for gradient checkpointing")
            
            patch_higgs_audio_model()
        
        model = original_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        
        # Double-check and patch the loaded instance if needed
        if model.__class__.__name__ == 'HiggsAudioModel' or 'higgs' in str(pretrained_model_name_or_path).lower():
            logger.info(f"Post-load verification for HiggsAudioModel")
            
            # Patch the class again to be sure
            patch_higgs_audio_model()
            
            # Resize embeddings if we have a tokenizer with new tokens
            if hasattr(model, 'config') and hasattr(model.config, 'vocab_size'):
                # Check if tokenizer was passed
                tokenizer = kwargs.get('tokenizer', None)
                if tokenizer and hasattr(tokenizer, 'vocab_size'):
                    if tokenizer.vocab_size > model.config.vocab_size:
                        logger.info(f"Resizing model embeddings from {model.config.vocab_size} to {tokenizer.vocab_size}")
                        model.resize_token_embeddings(tokenizer.vocab_size)
                        model.config.vocab_size = tokenizer.vocab_size
            
            # Also patch the instance directly if needed
            if not hasattr(model, 'get_input_embeddings'):
                logger.info("Patching model instance with embedding methods")
                model.get_input_embeddings = lambda: model.embed_tokens
                model.set_input_embeddings = lambda v: setattr(model, 'embed_tokens', v)
                
                # Add enable_input_require_grads to instance
                def instance_enable_input_require_grads():
                    def make_inputs_require_grad(module, input, output):
                        if output is not None:
                            for out in output if isinstance(output, tuple) else [output]:
                                if torch.is_tensor(out) and out.dtype == torch.float:
                                    out.requires_grad_(True)
                    
                    if hasattr(model, '_input_require_grads_hook'):
                        model._input_require_grads_hook.remove()
                    
                    embeddings = model.embed_tokens
                    if embeddings is not None:
                        model._input_require_grads_hook = embeddings.register_forward_hook(make_inputs_require_grad)
                
                model.enable_input_require_grads = instance_enable_input_require_grads
        
        return model
    
    # Replace the from_pretrained method
    AutoModelForCausalLM.from_pretrained = patched_from_pretrained
    logger.info("✓ Hooked into AutoModelForCausalLM.from_pretrained for dynamic patching")
    
    # === CRITICAL: Patch PEFT wrapper classes ===
    # When using LoRA, the model gets wrapped with PeftModelForCausalLM
    # We need to ensure the PEFT wrapper delegates embedding methods to base model
    try:
        from peft import PeftModel, PeftModelForCausalLM
        
        # Store original methods if they exist
        orig_peft_get_input = getattr(PeftModelForCausalLM, 'get_input_embeddings', None)
        orig_peft_set_input = getattr(PeftModelForCausalLM, 'set_input_embeddings', None)
        
        def peft_get_input_embeddings(self) -> nn.Module:
            """Get input embeddings from the base model."""
            # Try base_model first (PEFT structure)
            if hasattr(self, 'base_model'):
                base = self.base_model
                if hasattr(base, 'model'):
                    base = base.model
                
                # CRITICAL: Check for embed_tokens FIRST before trying get_input_embeddings
                # This avoids NotImplementedError from HiggsAudioModel
                if hasattr(base, 'embed_tokens'):
                    return base.embed_tokens
                
                # Try to find embed_tokens in model attributes
                for attr_name in ['model', 'language_model', 'transformer']:
                    if hasattr(base, attr_name):
                        sub_model = getattr(base, attr_name)
                        if hasattr(sub_model, 'embed_tokens'):
                            return sub_model.embed_tokens
                
                # Only try get_input_embeddings as last resort
                if hasattr(base, 'get_input_embeddings'):
                    try:
                        result = base.get_input_embeddings()
                        if result is not None:
                            return result
                    except NotImplementedError:
                        pass
            
            # Fallback to original if it exists
            if orig_peft_get_input:
                try:
                    return orig_peft_get_input(self)
                except:
                    pass
            
            raise NotImplementedError(f"Could not find input embeddings in PEFT model. Model type: {type(self)}, Base model type: {type(self.base_model) if hasattr(self, 'base_model') else 'No base_model'}")
        
        def peft_set_input_embeddings(self, value: nn.Module):
            """Set input embeddings in the base model."""
            # Try base_model first (PEFT structure)
            if hasattr(self, 'base_model'):
                base = self.base_model
                if hasattr(base, 'model'):
                    base = base.model
                
                # Try set_input_embeddings method first
                if hasattr(base, 'set_input_embeddings') and not isinstance(getattr(base, 'set_input_embeddings'), type(lambda: None)):
                    try:
                        base.set_input_embeddings(value)
                        return
                    except NotImplementedError:
                        pass
                
                # Fallback to direct assignment
                if hasattr(base, 'embed_tokens'):
                    base.embed_tokens = value
                    return
                
                # Try to find embed_tokens in model attributes
                for attr_name in ['model', 'language_model', 'transformer']:
                    if hasattr(base, attr_name):
                        sub_model = getattr(base, attr_name)
                        if hasattr(sub_model, 'embed_tokens'):
                            sub_model.embed_tokens = value
                            return
            
            # Fallback to original if it exists
            if orig_peft_set_input:
                orig_peft_set_input(self, value)
                return
            
            raise NotImplementedError("Could not set input embeddings in PEFT model")
        
        def peft_enable_input_require_grads(self):
            """Enable input require grads for PEFT model."""
            def make_inputs_require_grad(module, input, output):
                if output is not None:
                    for out in output if isinstance(output, tuple) else [output]:
                        if torch.is_tensor(out) and out.dtype == torch.float:
                            out.requires_grad_(True)
            
            if hasattr(self, '_input_require_grads_hook'):
                self._input_require_grads_hook.remove()
            
            embeddings = self.get_input_embeddings()
            if embeddings is not None:
                self._input_require_grads_hook = embeddings.register_forward_hook(make_inputs_require_grad)
        
        # Apply patches to PEFT classes
        PeftModelForCausalLM.get_input_embeddings = peft_get_input_embeddings
        PeftModelForCausalLM.set_input_embeddings = peft_set_input_embeddings
        PeftModelForCausalLM.enable_input_require_grads = peft_enable_input_require_grads
        
        # Also patch base PeftModel class
        PeftModel.get_input_embeddings = peft_get_input_embeddings
        PeftModel.set_input_embeddings = peft_set_input_embeddings
        PeftModel.enable_input_require_grads = peft_enable_input_require_grads
        
        logger.info("✓ Patched PEFT model classes for gradient checkpointing compatibility")
        
    except ImportError:
        logger.info("PEFT not installed, skipping PEFT model patches")
    except Exception as e:
        logger.warning(f"Failed to patch PEFT models: {e}")
    
    # Patch HiggsAudioModel forward directly to handle labels
    original_forward = HiggsAudioModel.forward
    
    # Define valid forward arguments for HiggsAudioModel
    VALID_FORWARD_ARGS = {
        'input_ids', 'inputs_embeds', 'attention_mask', 'audio_features',
        'audio_feature_attention_mask', 'audio_in_ids', 'audio_in_ids_start',
        'audio_out_ids', 'audio_out_ids_start', 'audio_out_ids_start_group_loc',
        'label_ids', 'label_audio_ids', 'past_key_values', 'use_cache',
        'output_attentions', 'output_hidden_states', 'output_audio_hidden_states',
        'return_dict', 'cache_position', 'cache_audio_discrete_codes_mask',
        'past_key_values_buckets', 'reward'
    }
    
    def wrapped_forward(self, labels=None, **kwargs):
        """Forward that maps 'labels' to 'label_ids' and filters invalid args."""
        # Map labels to label_ids if present
        if labels is not None and 'label_ids' not in kwargs:
            kwargs['label_ids'] = labels
        
        # Filter out any unexpected arguments
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in VALID_FORWARD_ARGS}
        
        # Log if any args were filtered
        filtered_out = set(kwargs.keys()) - set(filtered_kwargs.keys())
        if filtered_out:
            logger.debug(f"Filtered out unexpected forward args: {filtered_out}")
        
        # Call original forward with filtered arguments
        return original_forward(self, **filtered_kwargs)
    
    # Replace the forward method
    HiggsAudioModel.forward = wrapped_forward
    logger.info("✓ Patched HiggsAudioModel.forward to handle labels -> label_ids mapping")
    
except ImportError as e:
    print(f"[ERROR] Failed to import Higgs Audio components: {e}")
    # Fallback imports - create minimal stubs
    class HiggsAudioSampleCollator:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, batch):
            return {}
    
    class HiggsAudioTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()
        def encode(self, audio_path):
            return torch.zeros((8, 100), dtype=torch.long)  # Mock RVQ codes
    
    class ChatMLDatasetSample:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    AUDIO_IN_TOKEN = "<|AUDIO_IN|>"
    AUDIO_OUT_TOKEN = "<|AUDIO_OUT|>"

class HiggsAudioCollator:
    """Data collator for HiggsAudio that handles both text and audio."""
    
    def __init__(self, text_tokenizer, audio_tokenizer, mask_first_audio_step=True, 
                 mask_eos_in_audio=False, min_text_tokens=50, text_weight=1.0, **kwargs):
        self.text_tokenizer = text_tokenizer
        self.audio_tokenizer = audio_tokenizer
        self.mask_first_audio_step = mask_first_audio_step
        self.mask_eos_in_audio = mask_eos_in_audio
        self.min_text_tokens = min_text_tokens
        self.text_weight = text_weight
        
        # Add audio tokens to tokenizer if not present
        special_tokens_added = False
        if AUDIO_IN_TOKEN not in text_tokenizer.get_vocab():
            text_tokenizer.add_tokens([AUDIO_IN_TOKEN], special_tokens=True)
            special_tokens_added = True
        if AUDIO_OUT_TOKEN not in text_tokenizer.get_vocab():
            text_tokenizer.add_tokens([AUDIO_OUT_TOKEN], special_tokens=True)
            special_tokens_added = True
        
        if special_tokens_added:
            print(f"[INFO] Added audio special tokens to tokenizer")
            # Update tokenizer vocab size
            print(f"[INFO] Tokenizer vocab size after adding special tokens: {len(text_tokenizer)}")
        
        # Get audio token IDs
        self.audio_in_token_id = text_tokenizer.convert_tokens_to_ids(AUDIO_IN_TOKEN)
        self.audio_out_token_id = text_tokenizer.convert_tokens_to_ids(AUDIO_OUT_TOKEN)
        print(f"[INFO] Audio token IDs - IN: {self.audio_in_token_id}, OUT: {self.audio_out_token_id}")
        
        # Get special token IDs
        self.text_bos_id = text_tokenizer.bos_token_id or text_tokenizer.eos_token_id
        self.text_eos_id = text_tokenizer.eos_token_id
        self.text_pad_id = text_tokenizer.pad_token_id or text_tokenizer.eos_token_id
        
        # Audio special tokens
        self.audio_bos_id = 1024  # Start of sequence for audio codes
        self.audio_eos_id = 1025  # End of sequence for audio codes
        
        print(f"[INFO] HiggsAudioCollator initialized - Text PAD: {self.text_pad_id}, Audio BOS/EOS: {self.audio_bos_id}/{self.audio_eos_id}")

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Process batch of features for on-the-fly tokenization.
        Expected feature format:
        {
            "messages": [...],  # ChatML format
            "ref_audio_path": "path/to/ref.wav", 
            "tgt_audio_path": "path/to/tgt.wav"
        }
        """
        batch_size = len(features)
        
        # --- 1. On-the-fly Audio Tokenization ---
        ref_audio_codes = []
        tgt_audio_codes = []
        
        for feature in features:
            # Handle different possible keys for audio paths
            ref_path = feature.get("ref_audio_path") or feature.get("ref_wav") or ""
            tgt_path = feature.get("tgt_audio_path") or feature.get("tgt_wav") or ""
            
            if ref_path and os.path.exists(ref_path):
                ref_codes = self.audio_tokenizer.encode(ref_path)
                if not isinstance(ref_codes, torch.Tensor):
                    ref_codes = torch.tensor(ref_codes, dtype=torch.long)
            else:
                # Mock codes if audio not available
                ref_codes = torch.zeros((8, 100), dtype=torch.long)
            
            if tgt_path and os.path.exists(tgt_path):
                tgt_codes = self.audio_tokenizer.encode(tgt_path)
                if not isinstance(tgt_codes, torch.Tensor):
                    tgt_codes = torch.tensor(tgt_codes, dtype=torch.long)
            else:
                # Mock codes if audio not available  
                tgt_codes = torch.zeros((8, 150), dtype=torch.long)
            
            ref_audio_codes.append(ref_codes)
            tgt_audio_codes.append(tgt_codes)

        # --- 2. Prepare Text from ChatML Messages ---
        all_input_ids = []
        
        for feature in features:
            messages = feature.get("messages", [])
            
            # Build conversation text with audio placeholders
            text_sequence = ""
            
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                
                if role == "system":
                    text_sequence += f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>"
                elif role == "user":
                    # Extract text content, handle both string and list formats
                    if isinstance(content, list):
                        text_content = ""
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                text_content += item.get("text", "")
                        content = text_content
                    # Use audio token in text
                    text_sequence += f"<|start_header_id|>user<|end_header_id|>\n\n{AUDIO_IN_TOKEN} {content}<|eot_id|>"
                elif role == "assistant":
                    # Extract text content for TTS
                    if isinstance(content, list):
                        text_content = ""
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                text_content += item.get("text", "")
                        content = text_content
                    # Use audio token in text
                    text_sequence += f"<|start_header_id|>assistant<|end_header_id|>\n\n{AUDIO_OUT_TOKEN} {content}<|eot_id|>"
            
            # Debug: print text before tokenization
            if not text_sequence:
                print(f"[WARNING] Empty text sequence for sample")
                text_sequence = f"<|begin_of_text|>{AUDIO_IN_TOKEN} Hello {AUDIO_OUT_TOKEN} Hi<|eot_id|>"  # Default text
            
            # Tokenize the text sequence
            text_encoding = self.text_tokenizer(
                text_sequence,
                padding=False,
                truncation=True,
                max_length=2048,
                return_tensors="pt"
            )
            tokens = text_encoding['input_ids'].squeeze(0)
            
            # Debug token count
            if len(tokens) < 5:
                print(f"[WARNING] Very few tokens ({len(tokens)}) from text: {text_sequence[:100]}...")
                print(f"[DEBUG] Token IDs: {tokens}")
            
            all_input_ids.append(tokens)
        
        # --- 3. Text Tokenization & Padding ---
        # Pad all sequences to same length
        padded_input_ids = torch.nn.utils.rnn.pad_sequence(
            all_input_ids, batch_first=True, padding_value=self.text_pad_id
        )
        attention_mask = (padded_input_ids != self.text_pad_id).long()
        
        # Text labels with proper +1 shift for causal LM
        text_labels = padded_input_ids.clone()
        # Shift left by 1: labels[t] = input_ids[t+1], mask first position
        text_labels[:, :-1] = padded_input_ids[:, 1:]
        text_labels[:, -1] = -100
        text_labels[text_labels == self.text_pad_id] = -100
        
        # --- 4. Audio Padding ---
        padded_ref_codes, ref_lengths = self._pad_audio(ref_audio_codes)
        padded_tgt_codes, tgt_lengths = self._pad_audio(tgt_audio_codes)
        
        # --- 5. Audio Teacher Forcing ---
        audio_inputs, audio_labels = self._prepare_audio_teacher_forcing(padded_tgt_codes)
        
        # Mask padded positions in audio labels
        B, C, T = audio_labels.shape
        pad_mask = torch.arange(T).unsqueeze(0).unsqueeze(0).expand(B, C, T) >= tgt_lengths.view(B,1,1)
        audio_labels[pad_mask] = -100
        
        # --- 6. Logging for Debugging ---
        print(f"[DEBUG] Batch size: {batch_size}")
        print(f"[DEBUG] Text input shape: {padded_input_ids.shape}")
        print(f"[DEBUG] Ref audio shape: {padded_ref_codes.shape}")
        print(f"[DEBUG] Audio inputs shape: {audio_inputs.shape}")

        # Create start indices for audio sequences
        batch_size = len(features)
        # Each sample has one audio sequence starting at index 0
        audio_in_ids_start = torch.arange(batch_size, dtype=torch.long)
        audio_out_ids_start = torch.arange(batch_size, dtype=torch.long)
        
        # Group locations - for single sequences, it's just [0]
        audio_out_ids_start_group_loc = torch.zeros(1, dtype=torch.long)

        # Match HiggsAudioBatchInput format
        return {
            "input_ids": padded_input_ids,
            "attention_mask": attention_mask,
            "label_ids": text_labels,                  # text labels (shifted)
            "label_audio_ids": audio_labels,           # audio labels (masked)
            "audio_in_ids": audio_inputs,              # BOS + shifted target codes
            "audio_in_ids_start": audio_in_ids_start,  # start indices for each audio sequence
            "audio_out_ids": padded_tgt_codes,         # raw target codes (unshifted)
            "audio_out_ids_start": audio_out_ids_start,# start indices for output audio
            "audio_out_ids_start_group_loc": audio_out_ids_start_group_loc, # group locations
            "audio_features": padded_ref_codes,        # using ref codes as features for now
            "audio_feature_attention_mask": torch.ones(padded_ref_codes.shape[:2], dtype=torch.long),
            "reward": None
        }

    def _pad_audio(self, audio_codes_list: List[torch.Tensor]) -> tuple:
        """Pads a list of audio code tensors to the same length."""
        if not audio_codes_list:
            return torch.zeros((1, 8, 1), dtype=torch.long), torch.tensor([1], dtype=torch.long)
            
        max_len = max(c.shape[1] for c in audio_codes_list)
        num_quantizers = audio_codes_list[0].shape[0]
        pad_val = 0  # neutral pad value
        padded_batch = torch.full((len(audio_codes_list), num_quantizers, max_len), 
                                  pad_val, dtype=torch.long)
        lengths = []
        
        for i, codes in enumerate(audio_codes_list):
            T = codes.shape[1]
            padded_batch[i, :, :T] = codes
            lengths.append(T)
            
        return padded_batch, torch.tensor(lengths, dtype=torch.long)

    def _prepare_audio_teacher_forcing(self, audio_codes: torch.Tensor) -> tuple:
        """Prepare audio codes for teacher-forcing in the decoder."""
        # Input: BOS -> code_1 -> ... -> code_n-1
        inputs = audio_codes.clone()
        inputs = torch.cat([
            torch.full((inputs.shape[0], inputs.shape[1], 1), self.audio_bos_id, 
                      dtype=torch.long, device=inputs.device),
            inputs[..., :-1]
        ], dim=-1)
        
        # Labels: code_1 -> ... -> code_n -> EOS
        labels = audio_codes.clone()
        
        # Mask first step only (as requested)
        if self.mask_first_audio_step:
            labels[:, :, 0] = -100
            
        # No EOS masking (as requested)
        return inputs, labels


class HiggsChatMLTemplate(Template):
    """
    A minimal Template for MS-SWIFT that:
      - does not rewrite messages
      - uses your on-the-fly DataCollator for both text and audio
      - relies on remove_unused_columns=false so we see raw fields
    """

    def __init__(self, processor, template_meta, default_system=None, max_length=None, **kwargs):
        super().__init__(processor, template_meta, default_system, max_length, **kwargs)
        self._text_tok = None
        self._audio_tok = None
        self._collator = None

    # --------- required by SWIFT ----------
    def _encode(self, inputs) -> Dict[str, Any]:
        """
        MS-SWIFT expects encoded output with input_ids, attention_mask, etc.
        Handle both StdTemplateInputs objects and dictionary inputs.
        """
        # Initialize text tokenizer if needed
        self._init_text_tokenizer()
        
        # Handle StdTemplateInputs object or dictionary
        if hasattr(inputs, 'messages'):
            # StdTemplateInputs object
            messages = inputs.messages or []
            system = inputs.system
            original_item = {
                'messages': messages,
                'system': system,
            }
        else:
            # Dictionary input  
            messages = inputs.get('messages', [])
            system = inputs.get('system')
            original_item = inputs
        
        # Create ChatML format text
        text_content = ""
        if system:
            text_content += f"<|im_start|>system\n{system}<|im_end|>\n"
            
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get('role', '')
                content = msg.get('content', '')
                text_content += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        
        # Tokenize the text content
        encoded = self._text_tok(
            text_content,
            truncation=True,
            max_length=self.max_length or 2048,
            padding=False,
            return_tensors=None
        )
        
        # Return MS-SWIFT expected format
        result = {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded.get('attention_mask', [1] * len(encoded['input_ids'])),
            'labels': encoded['input_ids'].copy(),  # Will be properly shifted in collator
            # Preserve original data for collator
            '_original_item': original_item
        }
        
        return result

    def _post_encoding(self, encoded_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # No-op; we handle everything in collator.
        return encoded_list

    # --------- our wiring ----------
    def _init_text_tokenizer(self, text_tokenizer_id: str = None):
        if self._text_tok is None:
            from transformers import AutoTokenizer
            tid = text_tokenizer_id or getattr(self.tokenizer, "name_or_path", 
                                              "meta-llama/Meta-Llama-3.1-8B-Instruct")
            self._text_tok = AutoTokenizer.from_pretrained(tid, use_fast=True)
            self._text_tok.pad_token = self._text_tok.pad_token or self._text_tok.eos_token

    def _init_audio_tokenizer(self, audio_tokenizer_id: str = "bosonai/higgs-audio-v2-tokenizer"):
        if self._audio_tok is None:
            try:
                self._audio_tok = HiggsAudioTokenizer.from_pretrained(audio_tokenizer_id)
            except Exception as e:
                print(f"[WARNING] Failed to load audio tokenizer: {e}, using mock")
                self._audio_tok = HiggsAudioTokenizer()

    def _build_collator(self, min_text_tokens: int = 32, text_weight: float = 1.0):
        if self._collator is None:
            self._collator = HiggsAudioCollator(
                text_tokenizer=self._text_tok,
                audio_tokenizer=self._audio_tok,
                shift_teacher_forcing=True,
                mask_first_audio_step=True,    # labels[:,:,0] = -100
                mask_eos_in_audio=False,       # NO EOS masking
                min_text_tokens=min_text_tokens,
                text_weight=text_weight
            )
        return self._collator

    def _data_collator(self, features: List[Dict[str, Any]], padding_to=None) -> Dict[str, torch.Tensor]:
        """
        SWIFT calls this to collate a batch. We delegate to our collator.
        Accept padding_to parameter for compatibility with MS-SWIFT base template.
        """
        # Lazily init tokenizers & collator
        self._init_text_tokenizer()
        self._init_audio_tokenizer()
        collator = self._build_collator()
        # Note: Our custom collator doesn't use padding_to, but we accept it for compatibility
        return collator(features)


# ---- Register the template so `--template higgs_chatml` works ----
register_template(
    TemplateMeta(
        template_type="higgs_chatml",
        prefix=[],  # No prefix needed for ChatML
        prompt=['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'],
        chat_sep=['<|im_end|>\n'],
        template_cls=HiggsChatMLTemplate,
        default_system=None
    )
)

print("[INFO] Registered higgs_chatml template successfully")
