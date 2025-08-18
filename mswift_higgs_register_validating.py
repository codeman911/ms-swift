import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

# Force HuggingFace Hub download (set before any imports)
os.environ['USE_HF'] = '1'
os.environ['USE_MODELSCOPE_HUB'] = '0'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

import torch
import json
import librosa
from swift.llm import (register_dataset, register_template, register_model, 
                       DatasetMeta, TemplateMeta, Template, ModelMeta, Model, ModelGroup, ModelInfo)
from torch.utils.data import Dataset

# --- Higgs Audio Imports ---
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'higgs-audio'))

from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
from boson_multimodal.dataset.chatml_dataset import ChatMLDatasetSample
from boson_multimodal.model.higgs_audio.modeling_higgs_audio import HiggsAudioModel
from boson_multimodal.constants import AUDIO_IN_TOKEN, AUDIO_OUT_TOKEN
from transformers import AutoConfig, AutoTokenizer

# --- Validation Contract Imports ---
from contracts import ValidatingHiggsAudioModel, ValidatingHiggsAudioSampleCollator

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. Validating Dataset for ChatML format ---

class ValidatingHiggsChatMLDataset(Dataset):
    """
    Validating dataset that produces ChatMLDatasetSample objects compatible 
    with ValidatingHiggsAudioSampleCollator.
    
    This wraps the original dataset loading logic but produces samples in the
    format expected by the Higgs Audio collator.
    """
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.data = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Load the audio tokenizer for on-the-fly tokenization
        self.audio_tokenizer = load_higgs_audio_tokenizer("bosonai/higgs-audio-v2-tokenizer", device="cpu")
        logger.info(f"ValidatingHiggsChatMLDataset loaded {len(self.data)} samples from {dataset_path}")
        logger.info(f"Audio tokenizer loaded for on-the-fly tokenization")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> ChatMLDatasetSample:
        sample = self.data[idx]
        messages = sample['messages']
        
        ref_audio_path = None
        tgt_audio_path = None
        
        # Extract audio paths from the nested content
        for item in messages[1]['content']:
            if item['type'] == 'audio':
                ref_audio_path = item['audio_url']
                break
        
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

        # Load audio waveforms
        ref_wv, ref_sr = librosa.load(ref_audio_path, sr=None)
        tgt_wv, tgt_sr = librosa.load(tgt_audio_path, sr=None)
        
        # Create simple tokenized text sequence with audio placeholders
        # System prompt
        text_sequence = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{messages[0]['content']}<|eot_id|>"
        # User prompt with reference audio placeholder
        text_sequence += f"<|start_header_id|>user<|end_header_id|>\n\n<|audio_bos|><|AUDIO|><|audio_eos|>{messages[1]['content'][2]['text']}<|eot_id|>"
        # Assistant response with target audio placeholder
        text_sequence += f"<|start_header_id|>assistant<|end_header_id|>\n\n<|audio_out_bos|><|AUDIO_OUT|><|audio_eos|>{messages[2]['content'][0]['text']}<|eot_id|>"
        
        # Create mock tokenized sequence (for validation purposes - real tokenization happens in collator)
        # This creates a simple token sequence for testing
        mock_input_ids = torch.arange(len(text_sequence.split()), dtype=torch.long)
        mock_label_ids = mock_input_ids.clone()
        
        # Real audio tokenization using the loaded tokenizer
        ref_codes = self.audio_tokenizer.encode(ref_wv, ref_sr)
        tgt_codes = self.audio_tokenizer.encode(tgt_wv, tgt_sr)
        
        # Concatenate audio codes and track starts
        audio_ids_concat = torch.cat([ref_codes, tgt_codes], dim=1)
        audio_ids_start = torch.tensor([0, ref_codes.shape[1]], dtype=torch.long)
        
        # Concatenate waveforms and track starts
        ref_wv_tensor = torch.tensor(ref_wv, dtype=torch.float32)
        tgt_wv_tensor = torch.tensor(tgt_wv, dtype=torch.float32)
        audio_waveforms_concat = torch.cat([ref_wv_tensor, tgt_wv_tensor])
        audio_waveforms_start = torch.tensor([0, len(ref_wv_tensor)], dtype=torch.long)
        
        # Sample rates
        audio_sample_rate = torch.tensor([ref_sr, tgt_sr], dtype=torch.float32)
        
        # Speaker indices (unknown)
        audio_speaker_indices = torch.tensor([-1, -1], dtype=torch.long)
        
        return ChatMLDatasetSample(
            input_ids=mock_input_ids,
            label_ids=mock_label_ids,
            audio_ids_concat=audio_ids_concat,
            audio_ids_start=audio_ids_start,
            audio_waveforms_concat=audio_waveforms_concat,
            audio_waveforms_start=audio_waveforms_start,
            audio_sample_rate=audio_sample_rate,
            audio_speaker_indices=audio_speaker_indices,
            audio_label_ids_concat=None,  # Optional
            reward=None  # Optional
        )

# Register the validating dataset
register_dataset(DatasetMeta(
    ms_dataset_id="higgs-chatml-validating",
    load_function=ValidatingHiggsChatMLDataset,
))

# --- 2. Validating Template with ValidatingHiggsAudioSampleCollator ---

class ValidatingHiggsChatMLTemplate(Template):
    """
    Validating template that uses ValidatingHiggsAudioSampleCollator for strict validation.
    """
    def __init__(self, tokenizer, system_prefix=None, *args, **kwargs):
        # MS-SWIFT passes additional positional arguments like system_prefix
        super().__init__(template_type="higgs-chatml-validating", tokenizer=tokenizer, **kwargs)
        logger.info("Initializing ValidatingHiggsChatMLTemplate...")
        
        # Load the Higgs Audio tokenizer
        self.audio_tokenizer = load_higgs_audio_tokenizer("bosonai/higgs-audio-v2-tokenizer", device="cpu")
        logger.info("HiggsAudioTokenizer loaded successfully for validating template.")

    def _data_collator(self, **kwargs) -> Any:
        """Returns the validating collator instance."""
        from transformers.models.whisper.processing_whisper import WhisperProcessor
        
        # Initialize WhisperProcessor for audio features
        whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-base")
        
        # Get special token IDs from tokenizer
        audio_in_token_id = self.tokenizer.convert_tokens_to_ids("<|AUDIO|>") 
        audio_out_token_id = self.tokenizer.convert_tokens_to_ids("<|AUDIO_OUT|>")
        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        
        # Audio stream tokens (from Higgs Audio config)
        audio_stream_bos_id = 1024  # Start of audio sequence
        audio_stream_eos_id = 1025  # End of audio sequence
        
        return ValidatingHiggsAudioSampleCollator(
            whisper_processor=whisper_processor,
            audio_in_token_id=audio_in_token_id,
            audio_out_token_id=audio_out_token_id,
            pad_token_id=pad_token_id,
            audio_stream_bos_id=audio_stream_bos_id,
            audio_stream_eos_id=audio_stream_eos_id,
            round_to=8,
            pad_left=False,
            encode_whisper_embed=True,
            return_audio_in_tokens=True,
            audio_num_codebooks=4,
            use_delay_pattern=False,
            disable_audio_codes_transform=False,
            chunk_size_seconds=30,
            add_new_bos_eos_for_long_chunk=True,
            mask_audio_out_token_label=True,
        )

    def _encode(self, **kwargs) -> Any:
        """This template is only for training with the validating collator."""
        raise NotImplementedError("ValidatingHiggsChatMLTemplate does not support synchronous encoding.")

# Register the validating template
register_template(TemplateMeta(
    template_type="higgs-chatml-validating",
    prefix=[''],
    prompt=['{{QUERY}}'],
    chat_sep=None,
    template_cls=ValidatingHiggsChatMLTemplate,
))

# --- 3. Validating Model Registration ---

def get_validating_higgs_audio_model(model_dir: str,
                                     model_info: ModelInfo,
                                     model_kwargs: Dict[str, Any],
                                     load_model: bool = True,
                                     **kwargs):
    """Return (model, tokenizer) for the validating Higgs-Audio model.

    Matches ms-swift get_function signature.
    """
    logger.info("Loading ValidatingHiggsAudioModel and tokenizer...")
    
    # Force HuggingFace Hub download
    os.environ['USE_HF'] = '1'
    os.environ['USE_MODELSCOPE_HUB'] = '0'

    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = None
    if load_model:
        model = ValidatingHiggsAudioModel.from_pretrained(
            model_dir,
            config=config,
            torch_dtype=model_info.torch_dtype,
            trust_remote_code=True,
            **model_kwargs,
        )

    logger.info("ValidatingHiggsAudioModel and tokenizer prepared.")
    return model, tokenizer

# Register the validating model
register_model(ModelMeta(
    model_type="higgs-audio-validating",
    model_groups=[
        ModelGroup([Model(hf_model_id="bosonai/higgs-audio-v2-generation-3B-base")])
    ],
    template="higgs-chatml-validating",
    get_function=get_validating_higgs_audio_model,
    model_arch="higgs-audio",
    model_source="huggingface",
))

logger.info("Validating Higgs Audio registration completed successfully.")
