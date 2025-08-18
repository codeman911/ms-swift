# plugins/higgs_dataset.py
# Purpose: Custom dataset loader for ChatML format with audio paths

import json
import os
from typing import Dict, Any, List
from datasets import Dataset

def load_higgs_chatml_dataset(dataset_syntax, dataset_meta, **kwargs) -> Dataset:
    """
    Load ChatML dataset with audio paths for on-the-fly processing.
    
    Expected JSON format:
    {
        "id": "sample_00000001",
        "lang": "ar", 
        "messages": [
            {"role": "system", "content": "You are a TTS model."},
            {"role": "user", "content": "Generate speech for the assistant text."},
            {"role": "assistant", "content": "النص الذي يجب نطقه هنا."}
        ],
        "ref_wav": "data/wavs/ref/sample_00000001_ref.wav",
        "tgt_wav": "data/wavs/tgt/sample_00000001_tgt.wav"
    }
    
    OR the more complex nested content format:
    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": [
                {"type": "text", "text": "..."},
                {"type": "audio", "audio_url": "path/to/ref.wav"}
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": "..."},
                {"type": "audio", "audio_url": "path/to/tgt.wav"}
            ]}
        ]
    }
    """
    # Extract dataset path from MS-SWIFT parameters
    dataset_name_or_path = "../higgs-audio/lora_training_data_zr/chatml_fixed/val_chatml_samples.json"
    print(f"[INFO] Loading Higgs ChatML dataset from: {dataset_name_or_path}")
    
    # Handle different input types
    if os.path.isfile(dataset_name_or_path):
        # Single JSON file
        with open(dataset_name_or_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif os.path.isdir(dataset_name_or_path):
        # Directory with JSON files
        data = []
        for filename in os.listdir(dataset_name_or_path):
            if filename.endswith('.json') or filename.endswith('.jsonl'):
                filepath = os.path.join(dataset_name_or_path, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    if filename.endswith('.jsonl'):
                        # JSONL format - one JSON per line
                        for line in f:
                            line = line.strip()
                            if line:
                                data.append(json.loads(line))
                    else:
                        # Regular JSON
                        file_data = json.load(f)
                        if isinstance(file_data, list):
                            data.extend(file_data)
                        else:
                            data.append(file_data)
    else:
        raise ValueError(f"Dataset path not found: {dataset_name_or_path}")
    
    print(f"[INFO] Loaded {len(data)} samples")
    
    # Normalize data format
    normalized_data = []
    base_dir = os.path.dirname(dataset_name_or_path) if os.path.isfile(dataset_name_or_path) else dataset_name_or_path
    
    for i, sample in enumerate(data):
        try:
            # Extract messages
            messages = sample.get("messages", [])
            if not messages:
                print(f"[WARNING] Sample {i}: No messages found, skipping")
                continue
            
            # Extract audio paths - handle different formats
            ref_audio_path = None
            tgt_audio_path = None
            
            # Format 1: Direct keys
            if "ref_wav" in sample:
                ref_audio_path = sample["ref_wav"]
            if "tgt_wav" in sample:
                tgt_audio_path = sample["tgt_wav"]
            
            # Format 2: Extract from nested content
            if not ref_audio_path or not tgt_audio_path:
                for msg in messages:
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "audio":
                                audio_url = item.get("audio_url", "")
                                if msg.get("role") == "user" and not ref_audio_path:
                                    ref_audio_path = audio_url
                                elif msg.get("role") == "assistant" and not tgt_audio_path:
                                    tgt_audio_path = audio_url
            
            # Resolve relative paths
            if ref_audio_path and not os.path.isabs(ref_audio_path):
                ref_audio_path = os.path.join(base_dir, ref_audio_path)
            if tgt_audio_path and not os.path.isabs(tgt_audio_path):
                tgt_audio_path = os.path.join(base_dir, tgt_audio_path)
            
            # Create normalized sample
            normalized_sample = {
                "id": sample.get("id", f"sample_{i:08d}"),
                "messages": messages,
                "ref_audio_path": ref_audio_path or "",
                "tgt_audio_path": tgt_audio_path or "",  
                "lang": sample.get("lang", "en"),
                "speaker": sample.get("speaker", ""),
            }
            
            # Validate that we have required fields
            if not normalized_sample["messages"]:
                print(f"[WARNING] Sample {i}: No valid messages, skipping")
                continue
                
            normalized_data.append(normalized_sample)
            
        except Exception as e:
            print(f"[ERROR] Failed to process sample {i}: {e}")
            continue
    
    print(f"[INFO] Normalized {len(normalized_data)} valid samples")
    
    # Ensure consistent data types for PyArrow compatibility
    for sample in normalized_data:
        # Ensure messages is always a list
        if not isinstance(sample["messages"], list):
            sample["messages"] = []
        
        # Normalize messages content field - this is the main issue
        for msg in sample["messages"]:
            if isinstance(msg, dict):
                content = msg.get("content", "")
                # Convert all content to strings to avoid mixed list/non-list types
                if isinstance(content, list):
                    # Extract text content from complex structures
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                text_parts.append(item.get("text", ""))
                            elif item.get("type") == "audio":
                                # Keep audio reference as text
                                text_parts.append(f"[AUDIO: {item.get('audio_url', '')}]")
                        else:
                            text_parts.append(str(item))
                    msg["content"] = " ".join(text_parts)
                elif not isinstance(content, str):
                    msg["content"] = str(content)
        
        # Ensure string fields are always strings (not None)
        for key in ["id", "ref_audio_path", "tgt_audio_path", "lang", "speaker"]:
            if sample[key] is None:
                sample[key] = ""
            elif not isinstance(sample[key], str):
                sample[key] = str(sample[key])
    
    # Convert to HuggingFace Dataset
    dataset = Dataset.from_list(normalized_data)
    
    # Log sample for debugging
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"[DEBUG] Sample format: {list(sample.keys())}")
        print(f"[DEBUG] First sample messages: {len(sample['messages'])} messages")
        print(f"[DEBUG] Audio paths exist: ref={os.path.exists(sample['ref_audio_path']) if sample['ref_audio_path'] else False}, tgt={os.path.exists(sample['tgt_audio_path']) if sample['tgt_audio_path'] else False}")
    
    return dataset

# Register the dataset loader
from swift.llm import register_dataset, DatasetMeta

register_dataset(
    DatasetMeta(
        dataset_name="higgs-chatml-custom",
        load_function=load_higgs_chatml_dataset,
    )
)

print("[INFO] Registered higgs-chatml-custom dataset loader")
