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
    Load function for validating Higgs ChatML dataset with robust Arrow schema.
    """
    from datasets import Dataset as HFDataset, Features, Value, Sequence
    
    # Extract path from DatasetSyntax object
    path = dataset_syntax.dataset if hasattr(dataset_syntax, 'dataset') else str(dataset_syntax)
    logger.info(f"Loading ValidatingHiggsChatMLDataset from {path}...")
    
    # Load the dataset JSON
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} samples from {path}")
    
    def normalize_content(content):
        """Normalize ALL content to bulletproof consistent structure."""
        if isinstance(content, str):
            return [{
                "type": "text",
                "text": content,
                "audio_url": "",
                "raw_audio": "",
                "duration": 0.0,
                "offset": 0.0
            }]
        elif isinstance(content, list):
            normalized_list = []
            for item in content:
                if isinstance(item, dict):
                    normalized_item = {
                        "type": str(item.get("type", "text")),
                        "text": str(item.get("text", "")),
                        "audio_url": str(item.get("audio_url") or ""),
                        "raw_audio": str(item.get("raw_audio", "")),
                        "duration": float(item.get("duration") or 0.0),
                        "offset": float(item.get("offset") or 0.0)
                    }
                    normalized_list.append(normalized_item)
                else:
                    normalized_list.append({
                        "type": "text",
                        "text": str(item),
                        "audio_url": "",
                        "raw_audio": "",
                        "duration": 0.0,
                        "offset": 0.0
                    })
            return normalized_list
        else:
            return [{
                "type": "text",
                "text": str(content),
                "audio_url": "",
                "raw_audio": "",
                "duration": 0.0,
                "offset": 0.0
            }]
    
    def extract_text_from_content(content):
        """Extract text for MS-SWIFT template processing."""
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            return " ".join(text_parts).strip()
        return str(content)
    
    # Dual processing: normalized for dataset + text extracted for templates
    normalized_data = []
    for sample in data:
        messages = sample.get("messages", [])
        
        # Process messages with dual format
        processed_messages = []
        for msg in messages:
            role = str(msg.get("role", "user"))
            content = msg.get("content", "")
            
            # Normalize content for Arrow schema
            normalized_content = normalize_content(content)
            
            # Add both normalized content and extracted text
            processed_msg = {
                "role": role,
                "content": normalized_content,
                "text_content": extract_text_from_content(normalized_content)  # For template processing
            }
            processed_messages.append(processed_msg)
        
        normalized_data.append({
            "messages": processed_messages,
            "speaker": str(sample.get("speaker", "")),
            "start_index": int(sample.get("start_index", 0))
        })
    
    # Define bulletproof Features schema
    features = Features({
        "messages": Sequence({
            "role": Value("string"),
            "content": Sequence({
                "type": Value("string"),
                "text": Value("string"),
                "audio_url": Value("string"),
                "raw_audio": Value("string"),
                "duration": Value("float64"),
                "offset": Value("float64")
            }),
            "text_content": Value("string")
        }),
        "speaker": Value("string"),
        "start_index": Value("int64")
    })
    
    # Create dataset with explicit schema
    hf_dataset = HFDataset.from_list(normalized_data, features=features)
    
    logger.info(f"ValidatingHiggsChatMLDataset: {len(hf_dataset)} samples with robust Arrow schema and dual content format.")
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

# Custom template class to handle dual content format
from swift.llm.template.base import Template

class ValidatingHiggsChatMLTemplate(Template):
    def _encode_messages(self, messages):
        """Override to use text_content field for encoding."""
        # Extract text from dual format messages
        processed_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            # Use extracted text_content instead of complex content structure
            text_content = msg.get("text_content", "")
            if not text_content and "content" in msg:
                # Fallback extraction if text_content not available
                content = msg["content"]
                if isinstance(content, list):
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                    text_content = " ".join(text_parts).strip()
                else:
                    text_content = str(content)
            
            processed_messages.append({
                "role": role,
                "content": text_content
            })
        
        # Use parent class encoding with processed messages
        return super()._encode_messages(processed_messages)

# Register template with custom class
register_template(TemplateMeta(
    template_type="higgs-chatml-validating",
    template_cls=ValidatingHiggsChatMLTemplate,
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
