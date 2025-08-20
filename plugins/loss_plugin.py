"""Higgs-Audio custom training loss plugin for MS-SWIFT.

This plugin implements the dual loss calculation for Higgs-Audio model training,
combining text and audio cross-entropy losses.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

from swift.utils import get_logger

logger = get_logger()


@dataclass
class HiggsAudioLossConfig:
    """Configuration for Higgs-Audio loss calculation"""
    text_weight: float = 0.5
    audio_weight: float = 0.5
    label_smoothing: float = 0.0
    temperature: float = 1.0
    log_interval: int = 100


def higgs_audio_loss_func(outputs, labels, loss_scale=None, num_items_in_batch=None, **kwargs):
    """Custom loss function for Higgs-Audio training.
    
    This function calculates a combined loss from text and audio outputs,
    supporting weighted combination and label smoothing.
    
    Args:
        outputs: Model outputs containing logits
        labels: Target labels
        loss_scale: Optional loss scaling factor
        num_items_in_batch: Number of items in batch
        **kwargs: Additional arguments
    
    Returns:
        Combined loss tensor
    """
    
    # Get configuration from environment variables
    text_weight = float(os.environ.get('HIGGS_TEXT_WEIGHT', '0.5'))
    audio_weight = float(os.environ.get('HIGGS_AUDIO_WEIGHT', '0.5'))
    label_smoothing = float(os.environ.get('HIGGS_LABEL_SMOOTHING', '0.0'))
    temperature = float(os.environ.get('HIGGS_TEMPERATURE', '1.0'))
    
    # Initialize loss functions
    text_loss_fn = nn.CrossEntropyLoss(
        label_smoothing=label_smoothing,
        ignore_index=-100
    )
    audio_loss_fn = nn.CrossEntropyLoss(
        label_smoothing=label_smoothing,
        ignore_index=-100
    )
    
    # Handle different output formats
    if isinstance(outputs, dict):
        # Check for Higgs-specific outputs
        text_logits = outputs.get('text_logits')
        audio_logits = outputs.get('audio_logits')
        
        # Fallback to standard logits if specific ones not found
        if text_logits is None and audio_logits is None:
            logits = outputs.get('logits')
            if logits is not None:
                # Assume standard language model output
                text_logits = logits
    else:
        # Assume outputs are logits directly
        text_logits = outputs
        audio_logits = None
    
    # Apply temperature scaling if configured
    if temperature != 1.0:
        if text_logits is not None:
            text_logits = text_logits / temperature
        if audio_logits is not None:
            audio_logits = audio_logits / temperature
    
    # Calculate individual losses
    device = text_logits.device if text_logits is not None else 'cuda'
    text_loss = torch.tensor(0.0, device=device)
    audio_loss = torch.tensor(0.0, device=device)
    
    if text_logits is not None and labels is not None:
        # Handle text loss
        if len(text_logits.shape) == 3:
            # Reshape for loss calculation
            batch_size, seq_len, vocab_size = text_logits.shape
            text_logits_flat = text_logits.view(-1, vocab_size)
            labels_flat = labels.view(-1)
            
            # Calculate text loss
            text_loss = text_loss_fn(text_logits_flat, labels_flat)
        else:
            text_loss = text_loss_fn(text_logits, labels)
    
    if audio_logits is not None:
        # Handle audio loss if we have audio-specific labels
        audio_labels = kwargs.get('audio_labels')
        if audio_labels is not None:
            if len(audio_logits.shape) == 3:
                batch_size, seq_len, audio_vocab_size = audio_logits.shape
                audio_logits_flat = audio_logits.view(-1, audio_vocab_size)
                audio_labels_flat = audio_labels.view(-1)
                
                # Calculate audio loss
                audio_loss = audio_loss_fn(audio_logits_flat, audio_labels_flat)
            else:
                audio_loss = audio_loss_fn(audio_logits, audio_labels)
    
    # Combine losses with weights
    total_loss = text_weight * text_loss + audio_weight * audio_loss
    
    # Log loss components if in training mode
    if text_loss.requires_grad or audio_loss.requires_grad:
        logger.debug(f"Loss components - Text: {text_loss.item():.4f}, Audio: {audio_loss.item():.4f}, Total: {total_loss.item():.4f}")
    
    return total_loss


def register_higgs_audio_loss_plugin():
    """Register the Higgs-Audio loss function with MS-SWIFT.
    
    This function registers the custom loss function for use in training.
    """
    try:
        # Import the loss mapping from swift.plugin.loss
        from swift.plugin.loss import loss_mapping
        
        # Register our custom loss function
        loss_mapping['higgs_text_audio'] = higgs_audio_loss_func
        
        # Log configuration
        text_weight = float(os.environ.get('HIGGS_TEXT_WEIGHT', '0.5'))
        audio_weight = float(os.environ.get('HIGGS_AUDIO_WEIGHT', '0.5'))
        
        logger.info(f"Registered Higgs-Audio loss function 'higgs_text_audio' with weights - Text: {text_weight}, Audio: {audio_weight}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to register Higgs-Audio loss function: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Test registration
    success = register_higgs_audio_loss_plugin()
    if success:
        print(f"✓ Higgs-Audio loss plugin registered successfully")
    else:
        print(f"✗ Failed to register Higgs-Audio loss plugin")
        exit(1)
