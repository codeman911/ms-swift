import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

import torch
import json
import librosa
from swift.llm import (register_dataset, register_template, DatasetMeta, 
                       TemplateMeta, Template, ModelType)
from torch.utils.data import Dataset

# --- Higgs Audio Imports ---
# Note: Ensure higgs-audio is installed or in your PYTHONPATH
from higgs_audio.boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer, HiggsAudioTokenizer

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
# These should match the special tokens in your LLM tokenizer
AUDIO_IN_TOKEN = "<|audio_in|>"
AUDIO_OUT_TOKEN = "<|audio_out|>"

# --- 1. Custom Dataset for your ChatML format ---

class HiggsChatMLDataset(Dataset):
    """Dataset to parse the specific JSON structure provided."""
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.data = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        logger.info(f"Loaded {len(self.data)} samples from {dataset_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, Any]:
        sample = self.data[idx]
        messages = sample['messages']
        
        ref_audio_path = None
        tgt_audio_path = None
        
        # Extract audio paths from the nested content
        # User message contains reference audio
        for item in messages[1]['content']:
            if item['type'] == 'audio':
                ref_audio_path = item['audio_url']
                break
        
        # Assistant message contains target audio
        for item in messages[2]['content']:
            if item['type'] == 'audio':
                tgt_audio_path = item['audio_url']
                break

        if not ref_audio_path or not tgt_audio_path:
            raise ValueError(f"Audio path not found in sample {idx}")

        # Resolve relative paths if necessary
        base_dir = os.path.dirname(self.dataset_path)
        ref_audio_path = os.path.join(base_dir, ref_audio_path)
        tgt_audio_path = os.path.join(base_dir, tgt_audio_path)

        return {
            "messages": messages,
            "ref_audio_path": ref_audio_path,
            "tgt_audio_path": tgt_audio_path,
        }

# Register the dataset with ms-swift
register_dataset(DatasetMeta(
    dataset_id="higgs-chatml-custom",
    get_function=HiggsChatMLDataset,
    is_custom=True
))


# --- 2. Custom Data Collator ---

@dataclass
class HiggsAudioCollator:
    """A custom data collator for on-the-fly Higgs Audio tokenization."""
    llm_tokenizer: Any  # The tokenizer for the text part (e.g., from Llama)
    audio_tokenizer: HiggsAudioTokenizer
    
    def __post_init__(self):
        # Get special token IDs from the LLM tokenizer
        self.text_bos_id = self.llm_tokenizer.bos_token_id
        self.text_eos_id = self.llm_tokenizer.eos_token_id
        self.text_pad_id = self.llm_tokenizer.pad_token_id
        
        # From Higgs Audio, for audio codes
        self.audio_bos_id = 1024  # Start of sequence for audio codes
        self.audio_eos_id = 1025  # End of sequence for audio codes
        logger.info(f"Collator initialized. Text PAD ID: {self.text_pad_id}, Audio BOS/EOS: {self.audio_bos_id}/{self.audio_eos_id}")

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # --- 1. On-the-fly Audio Tokenization ---
        ref_audio_codes = []
        tgt_audio_codes = []
        for feature in features:
            # Encode reference and target audio files
            ref_codes = self.audio_tokenizer.encode(feature["ref_audio_path"])
            tgt_codes = self.audio_tokenizer.encode(feature["tgt_audio_path"])
            ref_audio_codes.append(torch.tensor(ref_codes, dtype=torch.long))
            tgt_audio_codes.append(torch.tensor(tgt_codes, dtype=torch.long))

        # --- 2. Prepare Text and Labels ---
        all_input_ids = []
        for feature in features:
            # Reconstruct the conversation string, inserting special audio tokens
            # System prompt
            text_sequence = self.llm_tokenizer.bos_token + feature['messages'][0]['content']
            # User prompt with reference audio placeholder
            text_sequence += self.llm_tokenizer.eos_token + '\nUSER: ' + AUDIO_IN_TOKEN + feature['messages'][1]['content'][2]['text']
            # Assistant response with target audio placeholder
            text_sequence += self.llm_tokenizer.eos_token + '\nASSISTANT: ' + AUDIO_OUT_TOKEN + feature['messages'][2]['content'][0]['text'] + self.llm_tokenizer.eos_token
            
            tokenized_sequence = self.llm_tokenizer(text_sequence, return_tensors='pt').input_ids.squeeze(0)
            all_input_ids.append(tokenized_sequence)

        # --- 3. Padding --- 
        # Pad text inputs
        padded_input_ids = torch.nn.utils.rnn.pad_sequence(
            all_input_ids, batch_first=True, padding_value=self.text_pad_id
        )
        
        # Pad audio codes (num_quantizers, seq_len)
        padded_ref_codes = self._pad_audio(ref_audio_codes)
        padded_tgt_codes = self._pad_audio(tgt_audio_codes)
        
        # --- 4. Create Labels ---
        # Create text labels (standard next-token prediction, ignore padding)
        text_labels = padded_input_ids.clone()
        text_labels[text_labels == self.text_pad_id] = -100

        # Create audio labels (teacher-forcing)
        audio_inputs, audio_labels = self._prepare_audio_teacher_forcing(padded_tgt_codes)
        
        # --- 5. Create Attention Masks ---
        attention_mask = (padded_input_ids != self.text_pad_id).long()

        # --- Logging for Debugging ---
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Batch Size: {len(features)}")
            logger.debug(f"Padded Input IDs Shape: {padded_input_ids.shape}")
            logger.debug(f"Padded Ref Audio Codes Shape: {padded_ref_codes.shape}")
            logger.debug(f"Audio Inputs Shape: {audio_inputs.shape}")
            logger.debug(f"Audio Labels Shape: {audio_labels.shape}")
            logger.debug(f"First Input IDs: {padded_input_ids[0, :32]}")

        return {
            "input_ids": padded_input_ids,
            "attention_mask": attention_mask,
            "labels": text_labels,
            "ref_audio_codes": padded_ref_codes,
            "tgt_audio_codes_input": audio_inputs,
            "tgt_audio_codes_labels": audio_labels,
        }

    def _pad_audio(self, audio_codes_list: List[torch.Tensor]) -> torch.Tensor:
        """Pads a list of audio code tensors to the same length."""
        max_len = max(c.shape[1] for c in audio_codes_list)
        num_quantizers = audio_codes_list[0].shape[0]
        padded_batch = torch.full((len(audio_codes_list), num_quantizers, max_len), self.audio_eos_id, dtype=torch.long)

        for i, codes in enumerate(audio_codes_list):
            padded_batch[i, :, :codes.shape[1]] = codes
        return padded_batch

    def _prepare_audio_teacher_forcing(self, audio_codes: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """Prepare audio codes for teacher-forcing in the decoder."""
        # Input: BOS -> code_1 -> ... -> code_n-1
        inputs = audio_codes.clone()
        inputs = torch.cat([
            torch.full((inputs.shape[0], inputs.shape[1], 1), self.audio_bos_id, dtype=torch.long),
            inputs[..., :-1]
        ], dim=-1)
        
        # Labels: code_1 -> ... -> code_n -> EOS
        labels = audio_codes.clone()
        # Mask the BOS token in the labels, as it's not predicted
        labels[labels == self.audio_bos_id] = -100 # Should not happen if BOS is not in vocab, but good practice
        return inputs, labels

# --- 3. Custom Template to provide the Collator ---

class HiggsChatMLTemplate(Template):
    def __init__(self, tokenizer, **kwargs):
        super().__init__(template_type="higgs-chatml-custom", tokenizer=tokenizer, **kwargs)
        # Instantiate the audio tokenizer once, it will be passed to the collator
        logger.info("Initializing HiggsAudioTokenizer...")
        self.audio_tokenizer = load_higgs_audio_tokenizer("bosonai/higgs-audio-v2-tokenizer", device="cpu")
        logger.info("HiggsAudioTokenizer loaded successfully.")

    def _data_collator(self, **kwargs) -> Any:
        # This method is called by ms-swift to get the collator instance
        return HiggsAudioCollator(llm_tokenizer=self.tokenizer, audio_tokenizer=self.audio_tokenizer)

    def _encode(self, **kwargs) -> Any:
        # This template is only for training with the custom collator
        raise NotImplementedError("This template does not support synchronous encoding.")

# Register the template with ms-swift
register_template(TemplateMeta(
    template_type="higgs-chatml-custom",
    template_cls=HiggsChatMLTemplate,
    is_custom=True
))
