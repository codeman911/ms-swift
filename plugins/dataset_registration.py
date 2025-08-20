"""Higgs-Audio Dataset Registration for MS-SWIFT following CUSTOM_TTS.md
Handles dataset loading, ChatML formatting, and audio-text alignment
Based on the specifications in CUSTOM_TTS.md Section 3
"""

import os
import json
import torch
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from datasets import Dataset, load_dataset, Audio
from swift.llm import register_dataset, DatasetMeta
from swift.llm.dataset.preprocessor import AutoPreprocessor
# from preprocessor import HiggsAudioPreprocessor  # Will use custom preprocessing
from swift.utils import get_logger

logger = get_logger()


def higgs_preprocess(ds):
    """Preprocess dataset as per CUSTOM_TTS.md Section 3"""
    def map_sample(sample):
        # Process messages according to CUSTOM_TTS.md format
        messages = sample.get('messages', [])
        if not messages:
            return sample
            
        # Extract system, user, and assistant messages
        sys_msg = messages[0].get('content', '') if messages[0].get('role') == 'system' else 'You are a helpful assistant.'
        
        # Process user message with multimodal content
        user_content_list = messages[1].get('content', []) if len(messages) > 1 else []
        user_text_parts = []
        audio_urls = []
        
        if isinstance(user_content_list, list):
            for item in user_content_list:
                if isinstance(item, dict):
                    if item.get('type') == 'text':
                        user_text_parts.append(item.get('text', ''))
                    elif item.get('type') == 'audio':
                        audio_url = item.get('audio_url', '')
                        if audio_url:
                            user_text_parts.append(f'<audio>{audio_url}</audio>')
                            audio_urls.append(audio_url)
        elif isinstance(user_content_list, str):
            user_text_parts.append(user_content_list)
        
        # Process assistant message
        assistant_content_list = messages[2].get('content', []) if len(messages) > 2 else []
        assistant_text_parts = []
        
        if isinstance(assistant_content_list, list):
            for item in assistant_content_list:
                if isinstance(item, dict):
                    if item.get('type') == 'text':
                        assistant_text_parts.append(item.get('text', ''))
                    elif item.get('type') == 'audio':
                        audio_url = item.get('audio_url', '')
                        if audio_url:
                            assistant_text_parts.append(f'<audio>{audio_url}</audio>')
                            audio_urls.append(audio_url)
        elif isinstance(assistant_content_list, str):
            assistant_text_parts.append(assistant_content_list)
        
        # Build conversation list as per CUSTOM_TTS.md
        conversations = [
            {"from": "system", "value": sys_msg},
            {"from": "user", "value": ' '.join(user_text_parts)},
            {"from": "assistant", "value": ' '.join(assistant_text_parts)}
        ]
        
        return {"conversations": conversations, "audios": audio_urls}
    
    return ds.map(map_sample)

def register_higgs_audio_datasets():
    """Register Higgs-Audio datasets with MS-SWIFT as per CUSTOM_TTS.md"""
    
    # Register as specified in CUSTOM_TTS.md Section 3
    register_dataset(
        DatasetMeta(
            dataset_name='higgs_audio',  # Name as per doc
            tags=['audio', 'tts', 'voice-cloning', 'multimodal'],
            preprocess_func=higgs_preprocess,  # Custom preprocessing
            load_function=load_higgs_audio_dataset,
            split=['train', 'validation']  # Support both splits
        ),
        exist_ok=True
    )
    
    # Register voice cloning dataset
    register_dataset(
        DatasetMeta(
            dataset_name='higgs-voice-cloning',
            tags=['voice-cloning', 'zero-shot', 'tts'],
            preprocess_func=higgs_preprocess,  # Use same preprocessing
            load_function=load_voice_cloning_dataset,
            split=['train']
        ),
        exist_ok=True
    )
    
    logger.info("✅ Higgs-Audio datasets registered successfully")


def normalize_dataset_item(item):
    """Normalize dataset item to ensure PyArrow compatibility"""
    normalized = {}
    
    for key, value in item.items():
        if key == 'messages':
            # Ensure messages is always a list of dicts with consistent structure
            if isinstance(value, list):
                normalized_messages = []
                for msg in value:
                    if isinstance(msg, dict):
                        # Ensure content is always a string (not mixed list/string)
                        content = msg.get('content', '')
                        if isinstance(content, list):
                            # Convert list content to string representation
                            content_parts = []
                            for part in content:
                                if isinstance(part, dict):
                                    if part.get('type') == 'text':
                                        content_parts.append(part.get('text', ''))
                                    elif part.get('type') == 'audio':
                                        audio_url = part.get('audio_url', '')
                                        if audio_url:
                                            content_parts.append(f'<audio>{audio_url}</audio>')
                                elif isinstance(part, str):
                                    content_parts.append(part)
                            content = ' '.join(content_parts)
                        
                        normalized_messages.append({
                            'role': msg.get('role', ''),
                            'content': str(content)
                        })
                normalized[key] = normalized_messages
            else:
                normalized[key] = []
        else:
            # Keep other fields as-is but ensure they're JSON serializable
            normalized[key] = value
    
    return normalized


def load_higgs_audio_dataset(
    dataset_syntax,
    dataset_meta,
    split: str = 'train',
    **kwargs
) -> Dataset:
    """Load Higgs-Audio dataset from a JSONL file using the HuggingFace loader.

    Args:
        dataset_syntax: The dataset syntax object from MS-SWIFT.
        dataset_meta: The dataset metadata.
        split: The dataset split to load (e.g., 'train').
        **kwargs: Additional arguments.

    Returns:
        A HuggingFace Dataset object.
    """
    if hasattr(dataset_syntax, 'subsets') and dataset_syntax.subsets:
        # MS-SWIFT splits the path by '/', so we need to reconstruct it
        # Filter out empty strings and join with '/'
        path_parts = [part for part in dataset_syntax.subsets if part]
        if path_parts:
            dataset_path = '/' + '/'.join(path_parts)
        else:
            dataset_path = None
        logger.info(f"Reconstructed dataset path from subsets: {dataset_path}")
    else:
        dataset_path = dataset_syntax.dataset
        logger.info(f"Using dataset path from dataset attribute: {dataset_path}")

    # Ensure dataset_path is not empty
    if not dataset_path:
        raise ValueError("Dataset path is empty. Please provide a valid dataset path.")
        
    logger.info(f"Loading Higgs-Audio dataset from: {dataset_path} (split: {split})")

    # Convert to absolute path if it's a relative path
    if not os.path.isabs(dataset_path):
        # Check if the path exists relative to the current directory
        if os.path.exists(dataset_path):
            dataset_path = os.path.abspath(dataset_path)
        else:
            # Try to find the dataset in common locations
            potential_paths = [
                os.path.join(os.getcwd(), dataset_path),
                os.path.join(os.path.dirname(os.path.dirname(__file__)), dataset_path),
                os.path.join('/vs', dataset_path)
            ]
            
            for path in potential_paths:
                if os.path.exists(path):
                    dataset_path = path
                    logger.info(f"Found dataset at: {dataset_path}")
                    break
    
    # Check if the dataset exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")
        
    # Load JSON data using the proper ChatML format from Higgs Audio
    import json
    import pandas as pd
    from datasets import Dataset
    
    data_list = []
    
    valid_samples = 0
    skipped_samples = 0
    
    # Load the entire JSON file (it's a JSON array, not JSONL)
    with open(dataset_path, 'r', encoding='utf-8') as f:
        try:
            json_data = json.load(f)
            if not isinstance(json_data, list):
                raise ValueError("Expected JSON array format")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON file: {e}")
            raise
    
    for item_idx, item in enumerate(json_data):
        try:
            # Skip items that don't have the expected structure
            if not isinstance(item, dict):
                skipped_samples += 1
                continue
            
            # MS-SWIFT expects 'messages' field, not 'conversations'
            # Keep the original messages format that MS-SWIFT expects
            if 'messages' in item:
                messages = item['messages']
                
                # Normalize multimodal content in messages
                normalized_messages = []
                for msg in messages:
                    if not isinstance(msg, dict):
                        continue
                        
                    role = msg.get('role', '')
                    content = msg.get('content', '')
                    
                    # Handle multimodal content (list format)
                    if isinstance(content, list):
                        content_text = ""
                        for part in content:
                            if isinstance(part, dict):
                                if part.get('type') == 'text':
                                    content_text += part.get('text', '')
                                elif part.get('type') == 'audio':
                                    audio_url = part.get('audio_url', '')
                                    if audio_url:
                                        content_text += f"<audio>{audio_url}</audio>"
                            elif isinstance(part, str):
                                content_text += part
                        content = content_text
                    
                    normalized_messages.append({
                        "role": role,
                        "content": str(content)
                    })
                
                # Add items with valid messages
                if normalized_messages and len(normalized_messages) >= 2:  # At least user and assistant
                    # Create the normalized item with 'messages' field for MS-SWIFT
                    normalized_item = {
                        "messages": normalized_messages
                    }
                    
                    # Add other fields if they exist
                    for key in ['speaker', 'misc', 'audios']:
                        if key in item:
                            normalized_item[key] = item[key]
                    
                    data_list.append(normalized_item)
                    valid_samples += 1
                else:
                    skipped_samples += 1
            elif 'conversations' in item:
                # Convert conversations format to messages format
                conversations = item['conversations']
                messages = []
                for conv in conversations:
                    messages.append({
                        'role': conv.get('from', ''),
                        'content': conv.get('value', '')
                    })
                
                if messages and len(messages) >= 2:
                    normalized_item = {
                        "messages": messages
                    }
                    
                    # Add other fields if they exist
                    for key in ['speaker', 'misc', 'audios']:
                        if key in item:
                            normalized_item[key] = item[key]
                    
                    data_list.append(normalized_item)
                    valid_samples += 1
                else:
                    skipped_samples += 1
            else:
                skipped_samples += 1
            
        except Exception:
            # Silently skip any other errors
            skipped_samples += 1
            continue
    
    logger.info(f"Processed {valid_samples} valid samples, skipped {skipped_samples} invalid samples")
    
    # Create dataset from normalized data
    dataset = Dataset.from_list(data_list)

    logger.info(f"Successfully loaded {len(dataset)} samples for split '{split}'.")
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
