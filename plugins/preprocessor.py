"""
Custom preprocessor for Higgs-Audio ChatML format that integrates with MS-SWIFT.
Uses the original boson_multimodal preprocessing logic.
"""

import sys
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add Higgs-Audio path to sys.path for imports
higgs_audio_path = Path(__file__).parent.parent / "higgs-audio"
if str(higgs_audio_path) not in sys.path:
    sys.path.insert(0, str(higgs_audio_path))

# Import original Higgs-Audio components
from boson_multimodal.dataset.chatml_dataset import prepare_chatml_sample, ChatMLDatasetSample
from boson_multimodal.data_types import ChatMLSample, Message, AudioContent, TextContent
from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer

from swift.llm.dataset.preprocessor import PreprocessFunc
from swift.utils import get_logger

logger = get_logger()


class HiggsAudioPreprocessor(PreprocessFunc):
    """Custom preprocessor for Higgs-Audio that uses original boson_multimodal logic."""
    
    def __init__(self, columns: Optional[List[str]] = None):
        self.columns = columns or ['messages']
        
    def __call__(self, dataset, tokenizer, **kwargs):
        """Preprocess dataset using original Higgs-Audio logic."""
        
        def process_sample(example):
            """Process a single sample using original Higgs-Audio preprocessing."""
            
            try:
                # Convert to ChatMLSample format
                messages = []
                for msg in example['messages']:
                    role = msg['role']
                    content = msg['content']
                    
                    # Convert content to proper format
                    processed_content = []
                    if isinstance(content, str):
                        processed_content.append(TextContent(text=content))
                    elif isinstance(content, list):
                        for item in content:
                            if item['type'] == 'text':
                                processed_content.append(TextContent(text=item['text']))
                            elif item['type'] == 'audio':
                                processed_content.append(AudioContent(
                                    audio_url=item['audio_url'],
                                    raw_audio=item.get('raw_audio', ''),
                                    duration=item.get('duration'),
                                    offset=item.get('offset')
                                ))
                    
                    messages.append(Message(role=role, content=processed_content))
                
                # Create ChatMLSample
                chatml_sample = ChatMLSample(
                    messages=messages,
                    start_index=example.get('start_index', 0),
                    speaker=example.get('speaker'),
                    misc=example.get('misc', {})
                )
                
                # Use original preprocessing
                input_tokens, label_tokens, audio_contents, speaker_id = prepare_chatml_sample(
                    chatml_sample, tokenizer
                )
                
                if input_tokens is None:
                    logger.warning("Failed to preprocess sample, skipping")
                    return None
                
                # Convert to tensors
                input_ids = torch.tensor(input_tokens, dtype=torch.long)
                label_ids = torch.tensor(label_tokens, dtype=torch.long) if label_tokens else None
                
                # Process audio contents
                audio_waveforms_concat = torch.tensor([])
                audio_waveforms_start = torch.tensor([])
                audio_sample_rate = torch.tensor([])
                audio_speaker_indices = torch.tensor([])
                audio_ids_concat = torch.tensor([[]])
                audio_ids_start = torch.tensor([])
                
                # Create ChatMLDatasetSample
                processed_sample = ChatMLDatasetSample(
                    input_ids=input_ids,
                    label_ids=label_ids,
                    audio_ids_concat=audio_ids_concat,
                    audio_ids_start=audio_ids_start,
                    audio_waveforms_concat=audio_waveforms_concat,
                    audio_waveforms_start=audio_waveforms_start,
                    audio_sample_rate=audio_sample_rate,
                    audio_speaker_indices=audio_speaker_indices,
                    audio_label_ids_concat=None
                )
                
                return {
                    'input_ids': input_ids,
                    'labels': label_ids,
                    'chatml_sample': processed_sample,
                    'audio_contents': audio_contents,
                    'speaker_id': speaker_id
                }
                
            except Exception as e:
                logger.error(f"Error processing sample: {e}")
                return None
        
        # Apply preprocessing
        processed_dataset = dataset.map(
            process_sample,
            remove_columns=dataset.column_names,
            desc="Processing Higgs-Audio samples"
        )
        
        # Filter out None samples
        processed_dataset = processed_dataset.filter(lambda x: x is not None)
        
        return processed_dataset
