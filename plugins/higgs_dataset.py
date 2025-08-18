# plugins/higgs_dataset.py
# Purpose: Complete ChatML dataset loader for Higgs Audio training

import json
import os
from typing import Dict, Any, List
from datasets import Dataset

def load_higgs_chatml_dataset(dataset_syntax, dataset_meta, **kwargs) -> Dataset:
    """
    Load ChatML dataset with audio paths for on-the-fly processing.
    
    Expected JSONL format (one JSON per line):
    {
        "id": "sample_0001",
        "lang": "ar",
        "messages": [
            {"role": "system", "content": "You are a TTS model."},
            {"role": "user", "content": "Clone this voice."},
            {"role": "assistant", "content": "النص الذي يجب نطقه هنا."}
        ],
        "ref_wav": "/abs/path/ref.wav",
        "tgt_wav": "/abs/path/target.wav"
    }
    """
    # Extract dataset path from MS-SWIFT dataset syntax: "higgs-chatml-custom#path=/abs/path/file.jsonl"
    if '#path=' in dataset_syntax:
        dataset_path = dataset_syntax.split('#path=')[1]
    else:
        # Fallback for direct path usage
        dataset_path = kwargs.get('dataset_path', dataset_syntax)
    
    print(f"[INFO] Loading Higgs ChatML dataset from: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset file not found: {dataset_path}")
        print(f"[INFO] Use: --dataset 'higgs-chatml-custom#path=/abs/path/to/data.jsonl'")
        # Return empty dataset with proper structure
        return Dataset.from_dict({
            'id': [],
            'messages': [],
            'ref_wav': [],
            'tgt_wav': [],
            'lang': []
        })
    
    data = []
    try:
        if os.path.isfile(dataset_path):
            with open(dataset_path, 'r', encoding='utf-8') as f:
                if dataset_path.endswith('.jsonl'):
                    # JSONL format - one JSON per line
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:
                            try:
                                data.append(json.loads(line))
                            except json.JSONDecodeError as e:
                                print(f"[WARNING] Line {line_num}: Invalid JSON - {e}")
                else:
                    # Regular JSON format
                    file_data = json.load(f)
                    if isinstance(file_data, list):
                        data = file_data
                    else:
                        data = [file_data]
        elif os.path.isdir(dataset_path):
            # Directory with multiple files
            for filename in sorted(os.listdir(dataset_path)):
                if filename.endswith(('.json', '.jsonl')):
                    filepath = os.path.join(dataset_path, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        if filename.endswith('.jsonl'):
                            for line in f:
                                if line.strip():
                                    data.append(json.loads(line))
                        else:
                            file_data = json.load(f)
                            if isinstance(file_data, list):
                                data.extend(file_data)
                            else:
                                data.append(file_data)
        else:
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
            
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        return Dataset.from_dict({
            'id': [],
            'messages': [],
            'ref_wav': [],
            'tgt_wav': [],
            'lang': []
        })
    
    print(f"[INFO] Loaded {len(data)} raw samples")
    
    # Normalize samples to consistent format
    normalized_samples = []
    base_dir = os.path.dirname(os.path.abspath(dataset_path))
    
    for i, sample in enumerate(data):
        try:
            # Validate required fields
            messages = sample.get("messages", [])
            if not messages:
                print(f"[WARNING] Sample {i}: No messages, skipping")
                continue
                
            ref_wav = sample.get("ref_wav", "")
            tgt_wav = sample.get("tgt_wav", "")
            
            if not ref_wav or not tgt_wav:
                print(f"[WARNING] Sample {i}: Missing audio paths, skipping")
                continue
            
            # Resolve relative paths
            if not os.path.isabs(ref_wav):
                ref_wav = os.path.join(base_dir, ref_wav)
            if not os.path.isabs(tgt_wav):
                tgt_wav = os.path.join(base_dir, tgt_wav)
            
            # Check file existence
            if not os.path.exists(ref_wav):
                print(f"[WARNING] Sample {i}: ref_wav not found: {ref_wav}")
                continue
            if not os.path.exists(tgt_wav):
                print(f"[WARNING] Sample {i}: tgt_wav not found: {tgt_wav}")
                continue
            
            # Create normalized sample
            normalized_sample = {
                'id': sample.get('id', f'sample_{i:08d}'),
                'messages': messages,
                'ref_wav': ref_wav,
                'tgt_wav': tgt_wav,
                'lang': sample.get('lang', 'en')
            }
            
            normalized_samples.append(normalized_sample)
            
        except Exception as e:
            print(f"[WARNING] Sample {i}: Error processing - {e}")
            continue
    
    print(f"[INFO] Successfully processed {len(normalized_samples)} valid samples")
    
    if not normalized_samples:
        print(f"[WARNING] No valid samples found")
        return Dataset.from_dict({
            'id': [],
            'messages': [],
            'ref_wav': [],
            'tgt_wav': [],
            'lang': []
        })
    
    # Convert to HuggingFace Dataset
    dataset_dict = {
        'id': [s['id'] for s in normalized_samples],
        'messages': [s['messages'] for s in normalized_samples],
        'ref_wav': [s['ref_wav'] for s in normalized_samples],
        'tgt_wav': [s['tgt_wav'] for s in normalized_samples],
        'lang': [s['lang'] for s in normalized_samples]
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    
    # Log sample for debugging
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"[DEBUG] Sample format: {list(sample.keys())}")
        print(f"[DEBUG] First sample messages: {len(sample['messages'])} messages")
        print(f"[DEBUG] Audio paths exist: ref={os.path.exists(sample['ref_wav'])}, tgt={os.path.exists(sample['tgt_wav'])}")
    
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
