"""
Higgs-Audio Dataset Registration for MS-SWIFT
Handles dataset loading, ChatML formatting, and audio-text alignment
"""

import os
import json
import torch
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from datasets import Dataset, load_dataset, Audio
from swift.llm import register_dataset, DatasetMeta
from swift.llm.dataset.preprocessor import AutoPreprocessor
from preprocessor import HiggsAudioPreprocessor
from swift.utils import get_logger

logger = get_logger()


def register_higgs_audio_datasets():
    """Register Higgs-Audio datasets with MS-SWIFT"""
    
    # Register standard Higgs-Audio training dataset
    register_dataset(
        DatasetMeta(
            dataset_name='higgs-audio-tts',
            tags=['audio', 'tts', 'voice-cloning', 'multimodal'],
            preprocess_func=HiggsAudioPreprocessor(columns=['messages']),
            load_function=load_higgs_audio_dataset,
            split=['train']
        ),
        exist_ok=True
    )
    
    # Register voice cloning dataset
    register_dataset(
        DatasetMeta(
            dataset_name='higgs-voice-cloning',
            tags=['voice-cloning', 'zero-shot', 'tts'],
            preprocess_func=HiggsAudioPreprocessor(columns=['messages']),
            load_function=load_voice_cloning_dataset,
            split=['train']
        ),
        exist_ok=True
    )
    
    logger.info("✅ Higgs-Audio datasets registered successfully")


def load_higgs_audio_dataset(
    dataset_path: str,
    split: str = 'train',
    **kwargs
) -> Dataset:
    """Load Higgs-Audio dataset from JSONL file with proper ChatML format.
    
    Args:
        dataset_path: Path to the JSONL dataset file
        split: Dataset split (train/val/test)
        **kwargs: Additional arguments
        
    Returns:
        Dataset: Loaded dataset with proper multimodal ChatML format
    """
    
    logger.info(f"Loading Higgs-Audio dataset from {dataset_path}, split={split}")
    
    # Load JSONL file
    data = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                sample = json.loads(line.strip())
                # Ensure proper ChatML format with multimodal content
                if 'messages' in sample:
                    # Process each message to ensure proper content structure
                    processed_messages = []
                    for message in sample['messages']:
                        content = message.get('content', '')
                        
                        # Handle both string and list content
                        if isinstance(content, str):
                            processed_content = [{'type': 'text', 'text': content}]
                        elif isinstance(content, list):
                            processed_content = content
                        else:
                            processed_content = [{'type': 'text', 'text': str(content)}]
                        
                        processed_messages.append({
                            'role': message['role'],
                            'content': processed_content
                        })
                    
                    sample['messages'] = processed_messages
                
                data.append(sample)
    
    # Convert to HuggingFace Dataset
    dataset = Dataset.from_list(data)
    
    logger.info(f"Loaded {len(dataset)} samples from {dataset_path}")
    return dataset


def load_voice_cloning_dataset(
    dataset_path: str,
    split: str = 'train',
    **kwargs
) -> Dataset:
    """
    Load voice cloning dataset with reference audio support
    
    Args:
        dataset_path: Path to dataset
        split: Dataset split
        
    Returns:
        Dataset with messages, audios, and reference_audio columns
    """
    
    logger.info(f"Loading voice cloning dataset from {dataset_path}")
    
    # Load base dataset
    dataset = load_higgs_audio_dataset(dataset_path, split, **kwargs)
    
    # Process for voice cloning
    def add_reference_audio(example):
        """Add reference audio to messages for zero-shot cloning"""
        
        messages = example['messages']
        audios = example.get('audios', [])
        
        # Check if reference audio is specified
        reference_audio = example.get('reference_audio', None)
        
        if reference_audio:
            # Insert reference audio in system message
            system_msg = messages[0] if messages[0]['role'] == 'system' else None
            
            if system_msg:
                # Add scene description and reference audio marker
                system_msg['content'] = (
                    f"<|scene_desc_start|> Indoor quiet <|scene_desc_end|>\n"
                    f"Reference voice: <audio>"
                )
                
                # Ensure reference audio is in audios list
                if reference_audio not in audios:
                    audios.insert(0, reference_audio)
            
            example['audios'] = audios
            example['messages'] = messages
        
        return example
    
    # Apply voice cloning preprocessing
    dataset = dataset.map(add_reference_audio)
    
    return dataset


def load_jsonl_dataset(filepath: str) -> List[Dict]:
    """
    Load dataset from JSONL file
    
    Expected format:
    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "Text to speak: <audio>"},
            {"role": "assistant", "content": ""}
        ],
        "audios": ["/path/to/audio.wav"],
        "reference_audio": "/path/to/reference.wav"  # Optional
    }
    """
    
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                
                # Validate required fields
                if 'messages' not in item:
                    logger.warning(f"Skipping item without messages: {item}")
                    continue
                
                # Process ChatML messages
                messages = process_messages(item['messages'])
                
                # Process audio paths
                audios = item.get('audios', [])
                audios = [process_audio_path(p) for p in audios]
                
                data.append({
                    'messages': messages,
                    'audios': audios,
                    'reference_audio': item.get('reference_audio', None)
                })
                
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line: {e}")
                continue
    
    logger.info(f"Loaded {len(data)} samples from {filepath}")
    return data


def load_directory_dataset(directory: Path, split: str) -> List[Dict]:
    """
    Load dataset from directory structure
    
    Expected structure:
    dataset/
    ├── train/
    │   ├── metadata.jsonl
    │   └── audios/
    │       ├── sample1.wav
    │       └── sample2.wav
    └── validation/
        ├── metadata.jsonl
        └── audios/
    """
    
    split_dir = directory / split
    metadata_file = split_dir / 'metadata.jsonl'
    audio_dir = split_dir / 'audios'
    
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    # Load metadata
    data = load_jsonl_dataset(str(metadata_file))
    
    # Update audio paths to absolute paths
    for item in data:
        audios = item.get('audios', [])
        updated_audios = []
        
        for audio_path in audios:
            if not Path(audio_path).is_absolute():
                audio_path = str(audio_dir / audio_path)
            updated_audios.append(audio_path)
        
        item['audios'] = updated_audios
    
    return data


def process_messages(messages: List[Dict]) -> List[Dict]:
    """
    Process and validate ChatML messages
    
    Args:
        messages: List of message dictionaries
        
    Returns:
        Processed messages with proper formatting
    """
    
    processed = []
    
    for msg in messages:
        role = msg.get('role', '')
        content = msg.get('content', '')
        
        # Validate role
        if role not in ['system', 'user', 'assistant']:
            logger.warning(f"Invalid role: {role}, skipping message")
            continue
        
        # Process audio placeholders
        if '<audio>' in content:
            # Ensure proper audio token formatting
            content = content.replace('<audio>', '<|audio|>')
        
        processed.append({
            'role': role,
            'content': content
        })
    
    # Ensure conversation structure
    if not processed:
        raise ValueError("No valid messages found")
    
    # Add system message if missing
    if processed[0]['role'] != 'system':
        processed.insert(0, {
            'role': 'system',
            'content': 'You are a helpful text-to-speech assistant.'
        })
    
    # Ensure assistant response exists
    if processed[-1]['role'] != 'assistant':
        processed.append({
            'role': 'assistant',
            'content': ''
        })
    
    return processed


def process_audio_path(audio_path: str) -> str:
    """
    Process and validate audio file path
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Validated absolute path
    """
    
    path = Path(audio_path)
    
    # Convert to absolute path if needed
    if not path.is_absolute():
        path = path.resolve()
    
    # Validate file exists
    if not path.exists():
        logger.warning(f"Audio file not found: {path}")
        # Return path anyway for lazy loading
    
    # Validate audio format
    valid_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    if path.suffix.lower() not in valid_extensions:
        logger.warning(f"Unsupported audio format: {path.suffix}")
    
    return str(path)


def process_hf_dataset(dataset: Dataset) -> Dataset:
    """
    Process HuggingFace dataset to match expected format
    
    Args:
        dataset: Raw HuggingFace dataset
        
    Returns:
        Processed dataset with messages and audios columns
    """
    
    def transform_example(example):
        """Transform dataset example to ChatML format"""
        
        # Handle different column names
        text_column = None
        audio_column = None
        
        for col in ['text', 'transcription', 'sentence']:
            if col in example:
                text_column = col
                break
        
        for col in ['audio', 'speech', 'wav']:
            if col in example:
                audio_column = col
                break
        
        # Create ChatML messages
        text = example.get(text_column, '') if text_column else ''
        
        messages = [
            {'role': 'system', 'content': 'Convert text to natural speech.'},
            {'role': 'user', 'content': f"Please speak: {text} <|audio|>"},
            {'role': 'assistant', 'content': ''}
        ]
        
        # Handle audio
        audios = []
        if audio_column and audio_column in example:
            audio_data = example[audio_column]
            if isinstance(audio_data, dict) and 'path' in audio_data:
                audios.append(audio_data['path'])
            elif isinstance(audio_data, str):
                audios.append(audio_data)
        
        return {
            'messages': messages,
            'audios': audios
        }
    
    # Apply transformation
    dataset = dataset.map(transform_example)
    
    # Keep only required columns
    columns_to_keep = ['messages', 'audios']
    columns_to_remove = [col for col in dataset.column_names 
                         if col not in columns_to_keep]
    
    if columns_to_remove:
        dataset = dataset.remove_columns(columns_to_remove)
    
    return dataset
