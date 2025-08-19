"""
Main registration script for Higgs-Audio V2 with MS-SWIFT
Call this to register all components before training
"""

import sys
import os
from pathlib import Path
from typing import Optional

# Add higgs-audio to path if needed
higgs_audio_path = Path(__file__).parent.parent / 'higgs-audio'
if higgs_audio_path.exists():
    sys.path.insert(0, str(higgs_audio_path))

from swift.utils import get_logger
# Allow import both as top-level module (via --custom_register_path) and as package module
try:
    # When imported as a standalone module (module name: 'register')
    from model_registration import register_higgs_audio_models
    from dataset_registration import register_higgs_audio_datasets
    from trainer import HiggsAudioTrainer
    from collator import HiggsAudioCollator
except ImportError:
    # When imported as part of the 'plugins' package (module name: 'plugins.register')
    from .model_registration import register_higgs_audio_models
    from .dataset_registration import register_higgs_audio_datasets
    from .trainer import HiggsAudioTrainer
    from .collator import HiggsAudioCollator

logger = get_logger()


def register_higgs_audio_all():
    """
    Register all Higgs-Audio components with MS-SWIFT
    This should be called before training or inference
    """
    
    logger.info("=" * 60)
    logger.info("Registering Higgs-Audio V2 with MS-SWIFT")
    logger.info("=" * 60)
    
    # Register models
    logger.info("📦 Registering Higgs-Audio models...")
    register_higgs_audio_models()
    
    # Register datasets
    logger.info("📊 Registering Higgs-Audio datasets...")
    register_higgs_audio_datasets()
    
    # Register trainer in Swift's trainer registry
    logger.info("🎯 Registering HiggsAudioTrainer...")
    from swift.trainers.trainer_factory import TrainerFactory
    TrainerFactory.TRAINER_MAPPING['higgs-audio-trainer'] = HiggsAudioTrainer

    # Register the custom template for the collator
    logger.info("📋 Registering HiggsAudioTemplate...")
    from swift.llm import TEMPLATE_MAPPING
    from .template import HiggsAudioTemplate
    TEMPLATE_MAPPING['higgs-audio-template'] = HiggsAudioTemplate
    
    logger.info("=" * 60)
    logger.info("✅ Higgs-Audio registration complete!")
    logger.info("=" * 60)
    
    return True


def validate_environment():
    """
    Validate that required dependencies are available
    """
    
    errors = []
    warnings = []
    
    # Check required packages
    try:
        import torch
        if not torch.cuda.is_available():
            warnings.append("CUDA not available - training will be slow")
    except ImportError:
        errors.append("PyTorch not installed")
    
    try:
        import transformers
        from packaging import version
        if version.parse(transformers.__version__) < version.parse("4.35.0"):
            warnings.append(f"Transformers version {transformers.__version__} < 4.35.0")
    except ImportError:
        errors.append("Transformers not installed")
    
    try:
        import swift
    except ImportError:
        errors.append("MS-SWIFT not installed")
    
    try:
        import librosa
    except ImportError:
        errors.append("librosa not installed (required for audio processing)")
    
    try:
        import soundfile
    except ImportError:
        warnings.append("soundfile not installed (optional for audio I/O)")
    
    # Check for Higgs-Audio model files
    try:
        from boson_multimodal.model.higgs_audio import HiggsAudioForCausalLM
    except ImportError:
        warnings.append("boson_multimodal not found - will use HuggingFace models")
    
    # Report results
    if errors:
        logger.error("❌ Environment validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        return False
    
    if warnings:
        logger.warning("⚠️ Environment warnings:")
        for warning in warnings:
            logger.warning(f"  - {warning}")
    
    logger.info("✅ Environment validation passed")
    return True


# Auto-register when imported
if __name__ != "__main__":
    try:
        if validate_environment():
            register_higgs_audio_all()
    except Exception as e:
        logger.error(f"Failed to register Higgs-Audio: {e}")
        logger.error("Please ensure all dependencies are installed")
        logger.error("Run: pip install -r requirements.txt")
