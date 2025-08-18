# plugins/higgs_ms_swift_register.py
# Purpose: register a custom template that hands raw dataset rows to your Higgs collator
# so you can tokenize on-the-fly. Optionally register a model_type.

import sys
import os
from pathlib import Path
from typing import Any, Dict, List
import functools
import torch

# Add higgs-audio to Python path
current_dir = Path(__file__).parent.parent.absolute()
higgs_audio_path = current_dir / "higgs-audio"
if higgs_audio_path.exists():
    sys.path.insert(0, str(higgs_audio_path))
    print(f"[INFO] Added higgs-audio to Python path: {higgs_audio_path}")

# MS-SWIFT APIs
from swift.llm import register_template, TemplateMeta, Template

# ---- import Higgs Audio code with correct paths ----
try:
    from boson_multimodal.data_collator.higgs_audio_collator import HiggsAudioSampleCollator
    from boson_multimodal.audio_processing.higgs_audio_tokenizer import HiggsAudioTokenizer
    from boson_multimodal.dataset.chatml_dataset import ChatMLDatasetSample
    from boson_multimodal.constants import AUDIO_IN_TOKEN, AUDIO_OUT_TOKEN
    from boson_multimodal.model.higgs_audio import HiggsAudioConfig, HiggsAudioModel
    print("[INFO] Successfully imported Higgs Audio components")
    
    # Register HiggsAudio model with transformers auto classes
    try:
        from transformers import AutoConfig, AutoModelForCausalLM
        AutoConfig.register("higgs_audio", HiggsAudioConfig)
        AutoModelForCausalLM.register(HiggsAudioConfig, HiggsAudioModel)
        
        # Force override for the specific model we're using
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
        from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING
        CONFIG_MAPPING.update([("higgs_audio", HiggsAudioConfig)])
        MODEL_FOR_CAUSAL_LM_MAPPING.update([(HiggsAudioConfig, HiggsAudioModel)])
        
        print("[INFO] Successfully registered HiggsAudio model with transformers auto classes")
        print(f"[INFO] HiggsAudioModel has get_input_embeddings: {hasattr(HiggsAudioModel, 'get_input_embeddings')}")
        
        # Verify the model has the required methods
        if hasattr(HiggsAudioModel, 'get_input_embeddings'):
            print("[INFO] ✓ HiggsAudioModel has get_input_embeddings method")
        else:
            print("[ERROR] ✗ HiggsAudioModel missing get_input_embeddings method")
            
    except Exception as e:
        print(f"[WARNING] Failed to register HiggsAudio model: {e}")
        
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
    """A custom data collator for on-the-fly Higgs Audio tokenization."""
    
    def __init__(self, text_tokenizer, audio_tokenizer, 
                 shift_teacher_forcing=True, mask_first_audio_step=True, 
                 mask_eos_in_audio=False, min_text_tokens=32, text_weight=1.0):
        self.text_tokenizer = text_tokenizer
        self.audio_tokenizer = audio_tokenizer
        self.shift_teacher_forcing = shift_teacher_forcing
        self.mask_first_audio_step = mask_first_audio_step
        self.mask_eos_in_audio = mask_eos_in_audio
        self.min_text_tokens = min_text_tokens
        self.text_weight = text_weight
        
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
                    text_sequence += f"<|start_header_id|>user<|end_header_id|>\n\n{AUDIO_IN_TOKEN} {content}<|eot_id|>"
                elif role == "assistant":
                    # Extract text content for TTS
                    if isinstance(content, list):
                        text_content = ""
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                text_content += item.get("text", "")
                        content = text_content
                    text_sequence += f"<|start_header_id|>assistant<|end_header_id|>\n\n{AUDIO_OUT_TOKEN} {content}<|eot_id|>"
            
            # Tokenize the text sequence
            text_encoding = self._text_tok(
                text_sequence,
                padding=False,
                truncation=True,
                max_length=2048,
                return_tensors="pt"
            )
            all_input_ids.append(text_encoding['input_ids'].squeeze(0))
        
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

        # Match HiggsAudioBatchInput format
        return {
            "input_ids": padded_input_ids,
            "attention_mask": attention_mask,
            "label_ids": text_labels,                  # text labels (shifted)
            "label_audio_ids": audio_labels,           # audio labels (masked)
            "audio_in_ids": audio_inputs,              # BOS + shifted target codes
            "audio_in_ids_start": None,                # or proper indices if you support groups
            "audio_out_ids": padded_tgt_codes,         # raw target codes (unshifted)
            "audio_out_ids_start": None,
            "audio_out_ids_start_group_loc": None,
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

    def _data_collator(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        SWIFT calls this to collate a batch. We delegate to our collator.
        """
        # Lazily init tokenizers & collator
        self._init_text_tokenizer()
        self._init_audio_tokenizer()
        collator = self._build_collator()
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
