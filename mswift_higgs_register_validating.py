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
                       DatasetMeta, TemplateMeta, ModelMeta, Model, ModelGroup, ModelInfo)
from torch.utils.data import Dataset

# --- Higgs Audio Imports ---
import sys
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

# --- 1. Validating Dataset Load Function ---

def load_validating_higgs_chatml_dataset(path: str, *args, **kwargs):
    """
    Load function for validating Higgs ChatML dataset with on-the-fly audio tokenization.
    """
    logger.info(f"Loading ValidatingHiggsChatMLDataset from {path}...")
    
    # Load the dataset JSON
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Load the Higgs Audio tokenizer for on-the-fly tokenization
    audio_tokenizer = load_higgs_audio_tokenizer("bosonai/higgs-audio-v2-tokenizer", device="cpu")
    logger.info(f"Loaded {len(data)} samples from {path}")
    logger.info("HiggsAudioTokenizer loaded successfully for ValidatingHiggsChatMLDataset.")
    
    class ValidatingHiggsChatMLDataset(Dataset):
        def __init__(self):
            self.data = data
            self.audio_tokenizer = audio_tokenizer
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx) -> ChatMLDatasetSample:
            """Load sample with on-the-fly audio tokenization."""
            sample = self.data[idx]
            
            # Extract conversation data
            conversations = sample.get("conversations", [])
            
            # Process messages to extract audio paths
            messages = []
            audio_paths = []
            
            for conv in conversations:
                role = conv.get("from", "user")
                content = conv.get("value", "")
                
                # Extract audio paths from content
                if "ref_audio_path" in conv:
                    ref_audio_path = conv["ref_audio_path"]
                    audio_paths.append(("ref", ref_audio_path))
                
                if "tgt_audio_path" in conv:
                    tgt_audio_path = conv["tgt_audio_path"]
                    audio_paths.append(("tgt", tgt_audio_path))
                    
                messages.append({"role": role, "content": content})
            
            # Load and tokenize audio files
            audio_codes = []
            audio_waveforms = []
            
            for audio_type, audio_path in audio_paths:
                if os.path.exists(audio_path):
                    # Load audio waveform
                    waveform, sample_rate = librosa.load(audio_path, sr=None)
                    
                    # Perform on-the-fly audio tokenization
                    codes = self.audio_tokenizer.encode(waveform, sample_rate)
                    
                    audio_codes.extend(codes)
                    audio_waveforms.append(waveform)
                else:
                    logger.warning(f"Audio file not found: {audio_path}")
            
            # Concatenate all audio codes and waveforms
            if audio_codes:
                audio_codes = torch.cat(audio_codes, dim=0) if len(audio_codes) > 1 else audio_codes[0]
            else:
                audio_codes = torch.empty(0, dtype=torch.long)
                
            if audio_waveforms:
                audio_waveforms = torch.cat([torch.tensor(wv, dtype=torch.float32) for wv in audio_waveforms], dim=0)
            else:
                audio_waveforms = torch.empty(0, dtype=torch.float32)
            
            return ChatMLDatasetSample(
                messages=messages,
                audio_codes=audio_codes,
                audio_waveforms=audio_waveforms
            )
    
    return ValidatingHiggsChatMLDataset()

# Register the validating dataset
register_dataset(DatasetMeta(
    dataset_name="higgs-chatml-validating",
    dataset_path="../higgs-audio/lora_training_data_zr/chatml_fixed/train_chatml_samples.json",
    load_function=load_validating_higgs_chatml_dataset,
))

# --- 2. Validating Template Registration ---

# Function to create validating data collator
def get_validating_higgs_data_collator(tokenizer, **kwargs):
    """Returns the validating collator instance."""
    from transformers.models.whisper.processing_whisper import WhisperProcessor
    
    # Initialize WhisperProcessor for audio features
    whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    
    # Get special token IDs from tokenizer
    audio_in_token_id = tokenizer.convert_tokens_to_ids("<|AUDIO|>") 
    audio_out_token_id = tokenizer.convert_tokens_to_ids("<|AUDIO_OUT|>")
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    
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
    )

# Register the validating template using TemplateMeta
register_template(TemplateMeta(
    template_type="higgs-chatml-validating",
    prefix=['<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{{SYSTEM}}<|eot_id|>'],
    prompt=['<|start_header_id|>user<|end_header_id|>\n\n{{QUERY}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'],
    chat_sep=['<|eot_id|>'],
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
            **model_kwargs
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
))

logger.info("Validating Higgs Audio registration completed successfully.")
