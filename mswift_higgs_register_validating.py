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

def load_validating_higgs_chatml_dataset(dataset_syntax, dataset_meta, *args, **kwargs):
    """
    Load function for validating Higgs ChatML dataset with on-the-fly audio tokenization.
    """
    from datasets import Dataset as HFDataset
    
    # Extract path from DatasetSyntax object
    path = dataset_syntax.dataset if hasattr(dataset_syntax, 'dataset') else str(dataset_syntax)
    logger.info(f"Loading ValidatingHiggsChatMLDataset from {path}...")
    
    # Load the dataset JSON
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} samples from {path}")
    
    # Convert to HuggingFace Dataset format
    # Just keep the original conversations structure for now
    dataset_dict = {
        "conversations": [sample.get("conversations", []) for sample in data]
    }
    
    # Create HuggingFace Dataset
    hf_dataset = HFDataset.from_dict(dataset_dict)
    
    logger.info("ValidatingHiggsChatMLDataset created as HuggingFace Dataset.")
    return hf_dataset

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
