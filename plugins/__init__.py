"""
Higgs-Audio V2 MS-SWIFT Integration Plugin
For zero-shot voice cloning and TTS fine-tuning
"""

from .model_registration import register_higgs_audio_models
from .dataset_registration import register_higgs_audio_datasets
from .trainer import HiggsAudioTrainer
from .collator import HiggsAudioCollator

__all__ = [
    'register_higgs_audio_models',
    'register_higgs_audio_datasets', 
    'HiggsAudioTrainer',
    'HiggsAudioCollator'
]
