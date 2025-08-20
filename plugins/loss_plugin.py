"""Higgs-Audio loss plugin for MS-SWIFT following CUSTOM_TTS.md

Custom training loss plugin that combines text and audio losses.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

from swift.utils import get_logger
from swift.llm.train.plugins import Plugin
from swift.llm.model.register import register_plugin

logger = get_logger()

# Loss weight environment variables
TEXT_WEIGHT_KEY = 'HIGGS_TEXT_WEIGHT'
AUDIO_WEIGHT_KEY = 'HIGGS_AUDIO_WEIGHT'


@dataclass
class HiggsAudioLossConfig:
    """Configuration for Higgs-Audio loss calculation"""
    text_weight: float = 1.0
    audio_weight: float = 1.0
    normalize_logits: bool = True
    ignore_index: int = -100
    reduction: str = 'mean'
    label_smoothing: float = 0.0
    temperature: float = 1.0


class HiggsAudioLossPlugin(Plugin):
    """Custom loss plugin for Higgs-Audio training.
    
    This plugin implements the dual loss calculation as specified in CUSTOM_TTS.md:
    - Text cross-entropy loss for language modeling
    - Audio cross-entropy loss for audio token prediction
    - Combined weighted loss for joint optimization
    """
    
    def __init__(self, config: Optional[HiggsAudioLossConfig] = None):
        """Initialize the loss plugin.
        
        Args:
            config: Loss configuration
        """
        super().__init__()
        
        self.config = config or HiggsAudioLossConfig()
        
        # Load weights from environment variables
        self.text_weight = float(os.getenv(TEXT_WEIGHT_KEY, str(self.config.text_weight)))
        self.audio_weight = float(os.getenv(AUDIO_WEIGHT_KEY, str(self.config.audio_weight)))
        
        logger.info(f"Initialized HiggsAudioLossPlugin with weights: "
                   f"text={self.text_weight}, audio={self.audio_weight}")
        
        # Loss functions
        self.text_loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.config.ignore_index,
            reduction=self.config.reduction,
            label_smoothing=self.config.label_smoothing
        )
        
        self.audio_loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.config.ignore_index,
            reduction=self.config.reduction,
            label_smoothing=self.config.label_smoothing
        )
    
    def compute_loss(
        self,
        model_output: Dict[str, Any],
        batch: Dict[str, torch.Tensor],
        model: nn.Module
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined text and audio loss.
        
        Args:
            model_output: Output from the model forward pass
            batch: Input batch with labels
            model: The model instance
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        
        # Extract logits and labels
        text_logits = model_output.get('logits', None)
        audio_logits = model_output.get('audio_logits', None)
        
        text_labels = batch.get('labels', None)
        audio_labels = batch.get('audio_labels', None)
        
        # Initialize losses
        text_loss = torch.tensor(0.0, device=model.device)
        audio_loss = torch.tensor(0.0, device=model.device)
        
        # Compute text loss
        if text_logits is not None and text_labels is not None:
            text_loss = self._compute_text_loss(text_logits, text_labels)
        
        # Compute audio loss
        if audio_logits is not None and audio_labels is not None:
            audio_loss = self._compute_audio_loss(audio_logits, audio_labels)
        
        # Combine losses with weights
        total_loss = self.text_weight * text_loss + self.audio_weight * audio_loss
        
        # Create loss dictionary for logging
        loss_dict = {
            'loss': total_loss.item(),
            'text_loss': text_loss.item(),
            'audio_loss': audio_loss.item(),
            'text_weight': self.text_weight,
            'audio_weight': self.audio_weight
        }
        
        return total_loss, loss_dict
    
    def _compute_text_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute text cross-entropy loss.
        
        Args:
            logits: Text logits [batch_size, seq_len, vocab_size]
            labels: Text labels [batch_size, seq_len]
            
        Returns:
            Text loss scalar
        """
        
        # Normalize logits if needed
        if self.config.normalize_logits:
            logits = self._normalize_logits(logits)
        
        # Apply temperature scaling
        if self.config.temperature != 1.0:
            logits = logits / self.config.temperature
        
        # Flatten for loss computation
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.view(-1, vocab_size)
        labels_flat = labels.view(-1)
        
        # Compute loss
        loss = self.text_loss_fn(logits_flat, labels_flat)
        
        return loss
    
    def _compute_audio_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute audio cross-entropy loss.
        
        Args:
            logits: Audio logits [batch_size, num_codebooks, seq_len, vocab_size]
                   or [batch_size, seq_len, vocab_size]
            labels: Audio labels [batch_size, num_codebooks, seq_len]
                   or [batch_size, seq_len]
            
        Returns:
            Audio loss scalar
        """
        
        # Handle different logit shapes
        if logits.dim() == 4:
            # Multi-codebook case
            batch_size, num_codebooks, seq_len, vocab_size = logits.shape
            
            # Normalize if needed
            if self.config.normalize_logits:
                logits = self._normalize_logits(logits, dim=-1)
            
            # Apply temperature
            if self.config.temperature != 1.0:
                logits = logits / self.config.temperature
            
            # Compute loss per codebook
            losses = []
            for cb in range(num_codebooks):
                cb_logits = logits[:, cb, :, :].reshape(-1, vocab_size)
                cb_labels = labels[:, cb, :].reshape(-1)
                cb_loss = self.audio_loss_fn(cb_logits, cb_labels)
                losses.append(cb_loss)
            
            # Average across codebooks
            loss = torch.stack(losses).mean()
            
        else:
            # Single codebook case
            if self.config.normalize_logits:
                logits = self._normalize_logits(logits)
            
            if self.config.temperature != 1.0:
                logits = logits / self.config.temperature
            
            # Flatten and compute
            batch_size, seq_len, vocab_size = logits.shape
            logits_flat = logits.view(-1, vocab_size)
            labels_flat = labels.view(-1)
            
            loss = self.audio_loss_fn(logits_flat, labels_flat)
        
        return loss
    
    def _normalize_logits(
        self,
        logits: torch.Tensor,
        dim: int = -1
    ) -> torch.Tensor:
        """Normalize logits to prevent overflow/underflow.
        
        Args:
            logits: Logits tensor
            dim: Dimension to normalize along
            
        Returns:
            Normalized logits
        """
        
        # Subtract max for numerical stability
        logits_max = logits.max(dim=dim, keepdim=True)[0]
        logits = logits - logits_max
        
        return logits
    
    def on_train_batch_end(
        self,
        trainer,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        loss: torch.Tensor,
        outputs: Dict[str, Any]
    ) -> None:
        """Hook called after each training batch.
        
        Log additional metrics and statistics.
        """
        
        # Log loss components if available
        if hasattr(self, '_last_loss_dict'):
            for key, value in self._last_loss_dict.items():
                if key != 'loss':  # Skip total loss (already logged)
                    trainer.log(f'train/{key}', value, prog_bar=True)
    
    def on_validation_batch_end(
        self,
        trainer,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        outputs: Dict[str, Any]
    ) -> None:
        """Hook called after each validation batch.
        
        Log validation metrics.
        """
        
        # Compute validation loss if needed
        if 'loss' not in outputs:
            loss, loss_dict = self.compute_loss(outputs, batch, model)
            outputs['loss'] = loss
            
            # Log validation metrics
            for key, value in loss_dict.items():
                trainer.log(f'val/{key}', value, prog_bar=True)


class HiggsAudioLossFunction(nn.Module):
    """Standalone loss function for Higgs-Audio.
    
    Can be used independently of the plugin system.
    """
    
    def __init__(
        self,
        text_weight: float = 1.0,
        audio_weight: float = 1.0,
        ignore_index: int = -100,
        reduction: str = 'mean',
        label_smoothing: float = 0.0
    ):
        """Initialize loss function.
        
        Args:
            text_weight: Weight for text loss
            audio_weight: Weight for audio loss
            ignore_index: Label index to ignore
            reduction: Loss reduction method
            label_smoothing: Label smoothing factor
        """
        super().__init__()
        
        self.text_weight = text_weight
        self.audio_weight = audio_weight
        
        self.text_criterion = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing=label_smoothing
        )
        
        self.audio_criterion = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing=label_smoothing
        )
    
    def forward(
        self,
        text_logits: Optional[torch.Tensor],
        text_labels: Optional[torch.Tensor],
        audio_logits: Optional[torch.Tensor],
        audio_labels: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute combined loss.
        
        Args:
            text_logits: Text model outputs
            text_labels: Text targets
            audio_logits: Audio model outputs
            audio_labels: Audio targets
            
        Returns:
            Total loss and component losses
        """
        
        losses = {}
        total_loss = 0
        
        # Text loss
        if text_logits is not None and text_labels is not None:
            # Reshape for loss computation
            vocab_size = text_logits.size(-1)
            text_loss = self.text_criterion(
                text_logits.view(-1, vocab_size),
                text_labels.view(-1)
            )
            losses['text_loss'] = text_loss
            total_loss = total_loss + self.text_weight * text_loss
        
        # Audio loss
        if audio_logits is not None and audio_labels is not None:
            if audio_logits.dim() == 4:
                # Multi-codebook: average across codebooks
                batch, codebooks, seq_len, vocab_size = audio_logits.shape
                audio_losses = []
                
                for cb in range(codebooks):
                    cb_loss = self.audio_criterion(
                        audio_logits[:, cb].reshape(-1, vocab_size),
                        audio_labels[:, cb].reshape(-1)
                    )
                    audio_losses.append(cb_loss)
                
                audio_loss = torch.stack(audio_losses).mean()
            else:
                # Single codebook
                vocab_size = audio_logits.size(-1)
                audio_loss = self.audio_criterion(
                    audio_logits.view(-1, vocab_size),
                    audio_labels.view(-1)
                )
            
            losses['audio_loss'] = audio_loss
            total_loss = total_loss + self.audio_weight * audio_loss
        
        losses['total_loss'] = total_loss
        
        return total_loss, losses


def register_higgs_audio_loss_plugin():
    """Register the Higgs-Audio loss plugin with MS-SWIFT"""
    
    # Register the plugin
    register_plugin(
        plugin_name='higgs-audio-loss',
        plugin_class=HiggsAudioLossPlugin,
        description='Dual loss plugin for Higgs-Audio text and audio generation',
        tags=['loss', 'audio', 'multimodal'],
        exist_ok=True
    )
    
    logger.info("✅ Higgs-Audio loss plugin registered successfully")


# Export main components
__all__ = [
    'HiggsAudioLossPlugin',
    'HiggsAudioLossFunction',
    'HiggsAudioLossConfig',
    'register_higgs_audio_loss_plugin'
]
