"""Higgs-Audio dataset registration for MS-SWIFT.

This module registers the Higgs-Audio datasets with MS-SWIFT's dataset registry.
"""

import json
import os
from typing import Any, Dict, List, Optional

from swift.utils import get_logger
from ..preprocessor import MessagesPreprocessor
from ..register import DatasetMeta, SubsetDataset, register_dataset

logger = get_logger()


def preprocess_higgs_audio(dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Preprocess Higgs-Audio dataset.
    
    Args:
        dataset: Raw dataset entries
    
    Returns:
        Processed dataset entries
    """
    preprocessor = MessagesPreprocessor()
    processed = []
    
    for entry in dataset:
        # Process messages format
        if 'messages' in entry:
            result = preprocessor(entry)
            if result:
                processed.append(result)
        else:
            # Handle other formats if needed
            processed.append(entry)
    
    return processed


# Register Higgs-Audio voice cloning dataset
register_dataset(
    DatasetMeta(
        dataset_name='higgs-audio-voice-cloning',
        dataset_path=None,  # Will be provided at runtime
        ms_dataset_id='higgs-audio/voice-cloning',
        hf_dataset_id='higgs-audio/voice-cloning',
        preprocess_func=preprocess_higgs_audio,
        subsets=[
            SubsetDataset(name='train', subset='train'),
            SubsetDataset(name='validation', subset='validation'),
            SubsetDataset(name='test', subset='test'),
        ],
        tags=['audio', 'tts', 'voice-cloning'],
    ))

# Register Higgs-Audio multi-speaker dataset
register_dataset(
    DatasetMeta(
        dataset_name='higgs-audio-multi-speaker',
        dataset_path=None,  # Will be provided at runtime
        ms_dataset_id='higgs-audio/multi-speaker',
        hf_dataset_id='higgs-audio/multi-speaker',
        preprocess_func=preprocess_higgs_audio,
        subsets=[
            SubsetDataset(name='train', subset='train'),
            SubsetDataset(name='validation', subset='validation'),
            SubsetDataset(name='test', subset='test'),
        ],
        tags=['audio', 'tts', 'multi-speaker'],
    ))

logger.info("Registered Higgs-Audio datasets: higgs-audio-voice-cloning, higgs-audio-multi-speaker")
