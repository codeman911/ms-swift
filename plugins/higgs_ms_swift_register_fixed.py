# plugins/higgs_ms_swift_register_fixed.py
# Purpose: Complete MS-SWIFT plugin with template, collator, and PEFT fixes

import torch
import torch.nn as nn
import logging
import sys
import os
from typing import List, Dict, Any
import numpy as np

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.auto import CONFIG_MAPPING, MODEL_FOR_CAUSAL_LM_MAPPING
from swift.utils import get_logger
from swift.llm import register_template, TemplateMeta

# Setup logging
logger = get_logger()

# Add the higgs-audio directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'higgs-audio'))

# === Import HiggsAudio components ===
try:
    from boson_multimodal.model.higgs_audio.configuration_higgs_audio import HiggsAudioConfig
    from boson_multimodal.model.higgs_audio.modeling_higgs_audio import HiggsAudioModel
    HIGGS_AUDIO_AVAILABLE = True
    logger.info("✓ HiggsAudio components imported successfully")
except ImportError as e:
    logger.warning(f"HiggsAudio components not available: {e}")
    HIGGS_AUDIO_AVAILABLE = False
    # Create dummy classes
    class HiggsAudioConfig:
        pass
    class HiggsAudioModel:
        pass

# === PEFT Gradient Checkpointing Fix ===
try:
    from peft import PeftModel, PeftModelForCausalLM
    
    def _find_embed_tokens(mod):
        """Robustly locate the text embedding module in a Higgs/Llama-style stack."""
        if mod is None:
            return None
        # Try get_input_embeddings() first
        if hasattr(mod, "get_input_embeddings"):
            try:
                emb = mod.get_input_embeddings()
                if emb is not None:
                    return emb
            except (NotImplementedError, AttributeError):
                pass
        # Walk common attribute paths
        paths = [
            "model.embed_tokens", "model.model.embed_tokens", "language_model.embed_tokens",
            "llm.embed_tokens", "text_model.embed_tokens", "embed_tokens", "tok_embeddings", "word_embeddings"
        ]
        for p in paths:
            obj = mod
            ok = True
            for attr in p.split("."):
                if not hasattr(obj, attr):
                    ok = False
                    break
                obj = getattr(obj, attr)
            if ok and obj is not None:
                return obj
        return None
    
    def _peft_get_input_embeddings(self):
        base = getattr(self, "get_base_model", lambda: getattr(self, "base_model", None))()
        emb = _find_embed_tokens(base) or _find_embed_tokens(self)
        if emb is None:
            raise RuntimeError(
                "Could not locate text embedding layer. "
                "Either implement get_input_embeddings() on the base model "
                "or run with --gradient_checkpointing false."
            )
        return emb
    
    def _peft_enable_input_require_grads(self):
        try:
            emb = self.get_input_embeddings()
        except Exception:
            emb = None
        if emb is not None and hasattr(emb, "weight"):
            emb.weight.requires_grad_(True)
    
    # Patch PEFT
    PeftModelForCausalLM.get_input_embeddings = _peft_get_input_embeddings
    PeftModelForCausalLM.enable_input_require_grads = _peft_enable_input_require_grads
    PeftModel.get_input_embeddings = _peft_get_input_embeddings
    PeftModel.enable_input_require_grads = _peft_enable_input_require_grads
    logger.info("✓ Patched PEFT for gradient checkpointing compatibility")
    
except ImportError:
    logger.warning("PEFT not available, skipping gradient checkpointing patches")

# === Template and Collator Implementation ===
def _higgs_data_collator(features: List[Dict], tokenizer, model=None):
    """On-the-fly collator for Higgs Audio training."""
    
    # Import audio tokenizer
    try:
        from boson_multimodal.audio_tokenizer import AudioTokenizer
        audio_tokenizer = AudioTokenizer.from_pretrained("bosonai/higgs-audio-v2-generation-3B-base")
    except ImportError:
        logger.error("boson_multimodal not available - install with: pip install boson-multimodal")
        raise
    
    batch_size = len(features)
    
    # === 1. Text Processing ===
    text_sequences = []
    for sample in features:
        messages = sample.get('messages', [])
        text_sequence = ""
        
        for msg in messages:
            role = msg.get('role', '')
            content = msg.get('content', '')
            
            if role == 'system':
                text_sequence += f"<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>"
            elif role == 'user':
                text_sequence += f"<|start_header_id|>user<|end_header_id|>\n\n<|AUDIO_IN|> {content}<|eot_id|>"
            elif role == 'assistant':
                text_sequence += f"<|start_header_id|>assistant<|end_header_id|>\n\n<|AUDIO_OUT|> {content}<|eot_id|>"
        
        if not text_sequence:
            text_sequence = "<|begin_of_text|><|AUDIO_IN|> Generate speech <|AUDIO_OUT|> Hello world<|eot_id|>"
        
        text_sequences.append(text_sequence)
    
    # Tokenize text
    text_encodings = tokenizer(
        text_sequences,
        padding=True,
        truncation=True,
        max_length=2048,
        return_tensors="pt"
    )
    
    input_ids = text_encodings['input_ids']
    attention_mask = text_encodings['attention_mask']
    
    # Text labels with next-token shift
    text_labels = input_ids.clone()
    text_labels[:, :-1] = input_ids[:, 1:]  # Shift left
    text_labels[:, -1] = -100  # Mask final position
    text_labels[text_labels == tokenizer.pad_token_id] = -100  # Mask padding
    
    # === 2. Reference Audio Features (Conditioning) ===
    audio_features_list = []
    audio_feature_masks = []
    
    for sample in features:
        ref_wav_path = sample.get('ref_wav', '')
        if ref_wav_path and os.path.exists(ref_wav_path):
            try:
                # Get conditioning features via Boson tokenizer
                features_dict = audio_tokenizer.get_regress_target(ref_wav_path)
                ref_features = features_dict['audio_features']  # [S, D]
                audio_features_list.append(torch.from_numpy(ref_features))
                audio_feature_masks.append(torch.ones(ref_features.shape[0], dtype=torch.long))
            except Exception as e:
                logger.warning(f"Failed to process ref audio {ref_wav_path}: {e}")
                # Fallback: dummy features
                dummy_features = torch.zeros(100, 768)  # Typical Whisper dimensions
                audio_features_list.append(dummy_features)
                audio_feature_masks.append(torch.ones(100, dtype=torch.long))
        else:
            # Dummy features for missing audio
            dummy_features = torch.zeros(100, 768)
            audio_features_list.append(dummy_features)
            audio_feature_masks.append(torch.ones(100, dtype=torch.long))
    
    # Pad audio features
    max_audio_len = max(f.shape[0] for f in audio_features_list)
    audio_features = torch.zeros(batch_size, max_audio_len, audio_features_list[0].shape[1])
    audio_feature_attention_mask = torch.zeros(batch_size, max_audio_len, dtype=torch.long)
    
    for i, (features, mask) in enumerate(zip(audio_features_list, audio_feature_masks)):
        seq_len = features.shape[0]
        audio_features[i, :seq_len] = features
        audio_feature_attention_mask[i, :seq_len] = mask
    
    # === 3. Target Audio Codes (Teacher Forcing) ===
    audio_codes_list = []
    
    for sample in features:
        tgt_wav_path = sample.get('tgt_wav', '')
        if tgt_wav_path and os.path.exists(tgt_wav_path):
            try:
                # Encode target audio to RVQ codes
                if hasattr(audio_tokenizer, 'encode_file'):
                    codes = audio_tokenizer.encode_file(tgt_wav_path)  # [8, T]
                else:
                    # Fallback methods
                    import soundfile as sf
                    audio_data, sr = sf.read(tgt_wav_path)
                    codes = audio_tokenizer.encode_wav(torch.from_numpy(audio_data).unsqueeze(0), sr)
                
                if isinstance(codes, tuple):
                    codes = codes[0]  # Take first element if tuple
                
                # Ensure shape [8, T]
                if codes.dim() == 3:
                    codes = codes.squeeze(0)
                
                audio_codes_list.append(codes)
            except Exception as e:
                logger.warning(f"Failed to encode target audio {tgt_wav_path}: {e}")
                # Dummy codes
                audio_codes_list.append(torch.randint(0, 1024, (8, 100)))
        else:
            # Dummy codes for missing audio
            audio_codes_list.append(torch.randint(0, 1024, (8, 100)))
    
    # Pad audio codes
    max_code_len = max(c.shape[1] for c in audio_codes_list)
    audio_out_ids = torch.zeros(batch_size, 8, max_code_len, dtype=torch.long)
    
    for i, codes in enumerate(audio_codes_list):
        seq_len = codes.shape[1]
        audio_out_ids[i, :, :seq_len] = codes
    
    # Teacher forcing: BOS + shifted codes for inputs
    audio_bos_id = 1024
    audio_in_ids = torch.full((batch_size, 8, max_code_len + 1), 0, dtype=torch.long)
    audio_in_ids[:, :, 0] = audio_bos_id  # BOS at position 0
    audio_in_ids[:, :, 1:] = audio_out_ids  # Shift codes right
    
    # Audio labels: mask only first step
    audio_labels = torch.full((batch_size, 8, max_code_len + 1), -100, dtype=torch.long)
    audio_labels[:, :, 0] = -100  # Mask BOS
    audio_labels[:, :, 1:] = audio_out_ids  # Targets
    
    # Mask padded positions
    for i, codes in enumerate(audio_codes_list):
        valid_len = codes.shape[1]
        audio_labels[i, :, valid_len + 1:] = -100
    
    # Return batch dict matching HiggsAudio expectations
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': text_labels,  # For HF trainer compatibility
        'text_labels': text_labels,  # For custom loss
        'audio_labels': audio_labels,  # For custom loss
        'audio_in_ids': audio_in_ids,  # Teacher forcing inputs
        'audio_out_ids': audio_out_ids,  # Raw target codes
        'audio_features': audio_features,  # Reference conditioning
        'audio_feature_attention_mask': audio_feature_attention_mask
    }

def register_higgs_audio():
    """Register HiggsAudio model, template and collator with MS-SWIFT."""
    
    # Register template
    from swift.llm import register_template, TemplateMeta
    
    register_template(
        TemplateMeta(
            template_type='higgs_chatml',
            prefix=[],
            prompt=[],
            chat_sep=None,
            suffix=[],
            system_prefix=[],
            _data_collator=_higgs_data_collator
        )
    )
    
    logger.info("✓ Registered higgs_chatml template with on-the-fly collator")
    
    # Register HiggsAudio model with transformers if available
    if HIGGS_AUDIO_AVAILABLE:
        try:
            AutoConfig.register("higgs_audio", HiggsAudioConfig, exist_ok=True)
            AutoModelForCausalLM.register(HiggsAudioConfig, HiggsAudioModel, exist_ok=True)
            
            # Force override for the specific model we're using
            if hasattr(CONFIG_MAPPING, '_extra_content'):
                CONFIG_MAPPING._extra_content["higgs_audio"] = HiggsAudioConfig
            if hasattr(MODEL_FOR_CAUSAL_LM_MAPPING, '_extra_content'):
                MODEL_FOR_CAUSAL_LM_MAPPING._extra_content[HiggsAudioConfig] = HiggsAudioModel
            
            logger.info("✓ Successfully registered HiggsAudio model with transformers auto classes")
        except Exception as e:
            logger.error(f"Failed to register HiggsAudio model: {e}")

# === Patch HiggsAudioModel if available ===
if HIGGS_AUDIO_AVAILABLE:
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
    
    original_forward = HiggsAudioModel.forward
    
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
    
    # Add embedding methods to HiggsAudioModel class if missing
    if not hasattr(HiggsAudioModel, 'get_input_embeddings'):
        def get_input_embeddings(self) -> nn.Module:
            """Return the text embedding layer for gradient checkpointing."""
            return self.embed_tokens
        HiggsAudioModel.get_input_embeddings = get_input_embeddings
    
    if not hasattr(HiggsAudioModel, 'set_input_embeddings'):
        def set_input_embeddings(self, value: nn.Module):
            """Set the text embedding layer for gradient checkpointing."""
            self.embed_tokens = value
        HiggsAudioModel.set_input_embeddings = set_input_embeddings
    
    if not hasattr(HiggsAudioModel, 'enable_input_require_grads'):
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
        HiggsAudioModel.enable_input_require_grads = enable_input_require_grads
    
    logger.info("✓ Added missing embedding methods to HiggsAudioModel")

# === Execute registration on module import ===
register_higgs_audio()

logger.info("✓ Higgs MS-SWIFT registration complete")
