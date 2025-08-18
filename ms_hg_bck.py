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
    
    # Process data for MS-SWIFT standard format with validation
    normalized_data = []
    skipped_count = 0
    
    for i, sample in enumerate(data):
        messages = sample.get("messages", [])
        
        # Validate messages structure
        if not messages or not isinstance(messages, list):
            logger.warning(f"Sample {i}: Invalid messages structure - skipping")
            skipped_count += 1
            continue
            
        # Create standard MS-SWIFT messages format
        processed_messages = []
        valid_sample = True
        
        for msg_idx, msg in enumerate(messages):
            if not isinstance(msg, dict):
                logger.warning(f"Sample {i}, message {msg_idx}: Invalid message format - skipping sample")
                valid_sample = False
                break
                
            role = str(msg.get("role", "user"))
            content = msg.get("content", "")
            
            # Validate role
            if not role or role not in ["system", "user", "assistant"]:
                logger.warning(f"Sample {i}, message {msg_idx}: Invalid role '{role}' - skipping sample")
                valid_sample = False
                break
            
            # Extract and validate content
            text_content = extract_text_from_content(content)
            if not text_content.strip():
                logger.warning(f"Sample {i}, message {msg_idx}: Empty content after extraction - skipping sample")
                valid_sample = False
                break
            
            normalized_content = normalize_content(content)
            
            processed_msg = {
                "role": role,
                "content": text_content.strip(),  # MS-SWIFT expects text content here
                "normalized_content": normalized_content  # Store multimodal data separately
            }
            processed_messages.append(processed_msg)
        
        if valid_sample and processed_messages:
            normalized_data.append({
                "messages": processed_messages,
                "speaker": str(sample.get("speaker", "")),
                "start_index": int(sample.get("start_index", 0))
            })
        else:
            skipped_count += 1
    
    logger.info(f"Processed {len(normalized_data)} valid samples, skipped {skipped_count} invalid samples")
    
    # Define Features schema - avoid Sequence to prevent dict conversion
    features = Features({
        "messages": [{"role": Value("string"), "content": Value("string")}],
        "speaker": Value("string"),
        "start_index": Value("int64")
    })
    
    # Simplify data for MS-SWIFT - only keep text content for messages
    simplified_data = []
    for item in normalized_data:
        simplified_messages = []
        for msg in item["messages"]:
            simplified_messages.append({
                "role": msg["role"],
                "content": msg["content"]  # Only text content for MS-SWIFT
            })
        simplified_data.append({
            "messages": simplified_messages,
            "speaker": item["speaker"],
            "start_index": item["start_index"]
        })
    
    # Create dataset with simplified data
    hf_dataset = HFDataset.from_list(simplified_data)
    
    logger.info(f"ValidatingHiggsChatMLDataset: {len(hf_dataset)} samples with robust Arrow schema and dual content format.")
    
    # Debug: Check first sample structure
    if len(hf_dataset) > 0:
        sample = hf_dataset[0]
        logger.info(f"DEBUG - First sample structure: {list(sample.keys())}")
        logger.info(f"DEBUG - Messages type: {type(sample['messages'])}")
        logger.info(f"DEBUG - Messages length: {len(sample['messages'])}")
        logger.info(f"DEBUG - Messages content: {sample['messages']}")
        # Skip trying to access [0] since HF converts to dict
    
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

# Custom template class with debugging
from swift.llm.template.base import Template

class DebuggingValidatingTemplate(Template):
    def encode(self, inputs, return_length=False, **kwargs):
        """Override encode to debug what MS-SWIFT receives."""
        logger.info(f"DEBUG - Template.encode received inputs type: {type(inputs)}")
        logger.info(f"DEBUG - Template.encode received inputs keys: {list(inputs.keys()) if isinstance(inputs, dict) else 'Not a dict'}")
        
        if isinstance(inputs, dict) and 'messages' in inputs:
            messages = inputs['messages']
            logger.info(f"DEBUG - Messages field type: {type(messages)}")
            logger.info(f"DEBUG - Messages length: {len(messages) if hasattr(messages, '__len__') else 'No length'}")
            if hasattr(messages, '__len__') and len(messages) > 0:
                logger.info(f"DEBUG - First message type: {type(messages[0])}")
                logger.info(f"DEBUG - First message: {messages[0]}")
        
        try:
            return super().encode(inputs, return_length=return_length, **kwargs)
        except Exception as e:
            logger.error(f"DEBUG - Template encoding failed: {e}")
            logger.error(f"DEBUG - Full inputs: {inputs}")
            raise

# Register custom debugging template
register_template(TemplateMeta(
    template_type="higgs-chatml-validating",
    template_cls=DebuggingValidatingTemplate,
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
