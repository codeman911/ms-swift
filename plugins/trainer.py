"""Higgs-Audio trainer for MS-SWIFT using original components.

This module wraps the original Higgs-Audio training logic from boson_multimodal
for seamless integration with MS-SWIFT training pipeline.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union

# Add Higgs-Audio path to sys.path for imports
higgs_audio_path = Path(__file__).parent.parent / "higgs-audio"
if str(higgs_audio_path) not in sys.path:
    sys.path.insert(0, str(higgs_audio_path))

# Import original Higgs-Audio components
from boson_multimodal.model.higgs_audio import HiggsAudioModel
from boson_multimodal.data_collator.higgs_audio_collator import HiggsAudioBatchInput
from transformers import Trainer
from transformers.modeling_utils import PreTrainedModel
from swift.utils import get_logger

logger = get_logger()

class HiggsAudioTrainer(Trainer):
    """MS-SWIFT trainer wrapper for Higgs-Audio model.
    
    This trainer extends the HuggingFace Trainer to properly handle
    Higgs-Audio's dual-modality training with the original model architecture.
    """
    
    def __init__(
        self,
        model: Union[PreTrainedModel, HiggsAudioModel],
        text_loss_weight: float = 1.0,
        audio_loss_weight: float = 1.0,
        use_delay_pattern_mask: bool = True,
        audio_kl_weight: float = 0.0,
        audio_commitment_weight: float = 0.25,
        **kwargs,
    ):
        """Initialize the Higgs-Audio trainer.
        
        Args:
            model: The Higgs-Audio model instance
            text_loss_weight: Weight for text prediction loss
            audio_loss_weight: Weight for audio prediction loss  
            use_delay_pattern_mask: Whether to apply delay pattern masking
            audio_kl_weight: Weight for audio KL divergence loss
            audio_commitment_weight: Weight for VQ commitment loss
            **kwargs: Additional arguments for base Trainer
        """
        super().__init__(model=model, **kwargs)
        
        self.text_loss_weight = text_loss_weight
        self.audio_loss_weight = audio_loss_weight
        self.use_delay_pattern_mask = use_delay_pattern_mask
        self.audio_kl_weight = audio_kl_weight
        self.audio_commitment_weight = audio_commitment_weight
        
        logger.info(
            f"Initialized HiggsAudioTrainer with text_weight={text_loss_weight}, "
            f"audio_weight={audio_loss_weight}, kl_weight={audio_kl_weight}"
        )
    
    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Any],
        return_outputs: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Compute combined loss for text and audio generation using original Higgs-Audio model.
        
        This method properly handles HiggsAudioBatchInput and computes the dual-modality loss.
        Args:
            model: The Higgs-Audio model being trained
            inputs: HiggsAudioBatchInput or dictionary of input tensors
            return_outputs: Whether to return model outputs
        
        Returns:
            Loss tensor or tuple of (loss, outputs)
        """
        
        # Ensure model is in training mode for loss computation
        model_mode = model.training
        
        # Handle HiggsAudioBatchInput dataclass
        if isinstance(inputs, HiggsAudioBatchInput):
            # Forward pass with original model
            outputs = model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                audio_features=inputs.audio_features,
                audio_feature_attention_mask=inputs.audio_feature_attention_mask,
                audio_out_ids=inputs.audio_out_ids,
                audio_out_ids_start=inputs.audio_out_ids_start,
                audio_out_ids_start_group_loc=inputs.audio_out_ids_start_group_loc,
                audio_in_ids=inputs.audio_in_ids,
                audio_in_ids_start=inputs.audio_in_ids_start,
                labels=inputs.label_ids,
                audio_labels=inputs.label_audio_ids,
            )
        else:
            # Direct forward with dictionary inputs
            outputs = model(**inputs)
        
        # Extract loss from outputs
        if hasattr(outputs, 'loss'):
            loss = outputs.loss
        elif isinstance(outputs, tuple):
            loss = outputs[0]
        else:
            loss = outputs.get('loss', outputs[0] if isinstance(outputs, (list, tuple)) else outputs)
        
        # Apply loss weighting if dual losses are available
        if hasattr(outputs, 'text_loss') and hasattr(outputs, 'audio_loss'):
            text_loss = outputs.text_loss
            audio_loss = outputs.audio_loss
            
            # Apply configured weights
            weighted_text_loss = self.text_loss_weight * text_loss
            weighted_audio_loss = self.audio_loss_weight * audio_loss
            
            # Add auxiliary losses if available
            auxiliary_loss = 0
            if hasattr(outputs, 'kl_loss') and self.audio_kl_weight > 0:
                auxiliary_loss += self.audio_kl_weight * outputs.kl_loss
            if hasattr(outputs, 'commitment_loss') and self.audio_commitment_weight > 0:
                auxiliary_loss += self.audio_commitment_weight * outputs.commitment_loss
            
            # Combine all losses
            loss = weighted_text_loss + weighted_audio_loss + auxiliary_loss
            
            # Log loss components
            if self.state.global_step % self.args.logging_steps == 0:
                logger.info(
                    f"Step {self.state.global_step}: "
                    f"text_loss={text_loss:.4f}, audio_loss={audio_loss:.4f}, "
                    f"auxiliary={auxiliary_loss:.4f}, total_loss={loss:.4f}"
                )
        
        if return_outputs:
            return loss, outputs
        return loss
    
    def _compute_loss_from_logits(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        outputs: CausalLMOutputWithPast
    ) -> torch.Tensor:
        """
        Manually compute loss from logits when not provided by model
        
        Args:
            model: The model
            inputs: Input batch
            outputs: Model outputs with logits
            
        Returns:
            Combined loss tensor
        """
        
        logits = outputs.logits
        labels = inputs.get('labels', inputs.get('label_ids'))
        
        if labels is None:
            raise ValueError("No labels found in inputs for loss computation")
        
        # Shift for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Compute text token loss
        text_vocab_size = model.config.vocab_size if hasattr(model.config, 'vocab_size') else 128000
        text_logits = shift_logits[..., :text_vocab_size]
        
        # Create text mask (tokens < text_vocab_size and not -100)
        text_mask = (shift_labels >= 0) & (shift_labels < text_vocab_size)
        text_labels = torch.where(text_mask, shift_labels, -100)
        
        text_loss = F.cross_entropy(
            text_logits.view(-1, text_vocab_size),
            text_labels.view(-1),
            ignore_index=-100,
            reduction='mean'
        )
        
        # Compute audio token loss if audio logits exist
        audio_loss = torch.tensor(0.0, device=model.device)
        
        if shift_logits.size(-1) > text_vocab_size:
            audio_logits = shift_logits[..., text_vocab_size:]
            audio_vocab_size = audio_logits.size(-1)
            
            # Create audio mask (tokens >= text_vocab_size)
            audio_mask = shift_labels >= text_vocab_size
            audio_labels = torch.where(
                audio_mask,
                shift_labels - text_vocab_size,
                -100
            )
            
            if (audio_labels >= 0).any():
                audio_loss = F.cross_entropy(
                    audio_logits.view(-1, audio_vocab_size),
                    audio_labels.view(-1),
                    ignore_index=-100,
                    reduction='mean'
                )
        
        # Combine losses
        total_loss = self.text_loss_weight * text_loss + self.audio_loss_weight * audio_loss
        
        return total_loss
    
    def training_step(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Perform a training step
        
        Args:
            model: The model being trained
            inputs: Batch inputs
            
        Returns:
            Loss tensor
        """
        
        model.train()
        
        # Add training context for Higgs-Audio
        with self._training_context(model):
            inputs = self._prepare_inputs(inputs)
            
            # Use automatic mixed precision if enabled
            if self.use_apex and self.fp16:
                with self.apex.amp.scale_loss() as scaled_loss:
                    loss = self.compute_loss(model, inputs)
                    scaled_loss.backward()
            else:
                loss = self.compute_loss(model, inputs)
                
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps
                
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
        
        return loss.detach()
    
    @contextmanager
    def _training_context(self, model: nn.Module):
        """
        Context manager for training-specific settings
        
        Args:
            model: The model being trained
        """
        
        # Store original settings
        original_gradient_checkpointing = getattr(model, 'gradient_checkpointing', False)
        
        try:
            # Enable gradient checkpointing if configured
            if self.args.gradient_checkpointing:
                if hasattr(model, 'gradient_checkpointing_enable'):
                    model.gradient_checkpointing_enable()
            
            # Set model to training mode
            model.train()
            
            yield
            
        finally:
            # Restore original settings
            if not original_gradient_checkpointing and hasattr(model, 'gradient_checkpointing_disable'):
                model.gradient_checkpointing_disable()
    
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform evaluation/prediction step
        
        Args:
            model: The model
            inputs: Batch inputs
            prediction_loss_only: Whether to only compute loss
            ignore_keys: Keys to ignore in outputs
            
        Returns:
            Tuple of (loss, logits, labels)
        """
        
        model.eval()
        
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        
        if prediction_loss_only:
            return (loss, None, None)
        
        # Extract logits and labels
        logits = outputs.logits
        labels = inputs.get('labels', inputs.get('label_ids'))
        
        # Handle audio outputs if present
        if hasattr(outputs, 'audio_logits'):
            # Concatenate text and audio logits
            audio_logits = outputs.audio_logits
            if audio_logits is not None:
                logits = torch.cat([logits, audio_logits], dim=-1)
        
        return (loss, logits, labels)
    
    def log(self, logs: Dict[str, float]) -> None:
        """
        Log training metrics including loss components
        
        Args:
            logs: Dictionary of metrics to log
        """
        
        # Add loss components to logs
        if self.loss_components['text_loss']:
            logs['text_loss'] = sum(self.loss_components['text_loss']) / len(self.loss_components['text_loss'])
            self.loss_components['text_loss'] = []
        
        if self.loss_components['audio_loss']:
            logs['audio_loss'] = sum(self.loss_components['audio_loss']) / len(self.loss_components['audio_loss'])
            self.loss_components['audio_loss'] = []
        
        if self.loss_components['auxiliary_loss']:
            logs['auxiliary_loss'] = sum(self.loss_components['auxiliary_loss']) / len(self.loss_components['auxiliary_loss'])
            self.loss_components['auxiliary_loss'] = []
        
        super().log(logs)
    
    def _save_checkpoint(self, model, trial, metrics=None):
        """
        Save model checkpoint with audio-specific components
        
        Args:
            model: The model to save
            trial: Optuna trial (if using hyperparameter search)
            metrics: Current metrics
        """
        
        # Save audio tokenizer if present
        if hasattr(model, 'audio_tokenizer'):
            audio_tokenizer_path = f"{self.args.output_dir}/audio_tokenizer"
            logger.info(f"Saving audio tokenizer to {audio_tokenizer_path}")
            model.audio_tokenizer.save_pretrained(audio_tokenizer_path)
        
        # Call parent save
        super()._save_checkpoint(model, trial, metrics)
    
    def create_optimizer(self):
        """
        Create optimizer with proper parameter groups for Higgs-Audio
        """
        
        if self.optimizer is not None:
            return self.optimizer
        
        # Get parameter groups
        decay_parameters = []
        no_decay_parameters = []
        audio_parameters = []
        
        # Separate parameters
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Audio-specific parameters get different learning rate
            if 'audio_ffn' in name or 'audio_output' in name:
                audio_parameters.append(param)
            # LayerNorm and bias terms don't get weight decay
            elif 'bias' in name or 'LayerNorm' in name or 'layernorm' in name:
                no_decay_parameters.append(param)
            else:
                decay_parameters.append(param)
        
        # Create parameter groups
        optimizer_grouped_parameters = [
            {
                'params': decay_parameters,
                'weight_decay': self.args.weight_decay,
                'lr': self.args.learning_rate
            },
            {
                'params': no_decay_parameters,
                'weight_decay': 0.0,
                'lr': self.args.learning_rate
            }
        ]
        
        # Add audio parameters with potentially different LR
        if audio_parameters:
            audio_lr = getattr(self.args, 'audio_learning_rate', self.args.learning_rate)
            optimizer_grouped_parameters.append({
                'params': audio_parameters,
                'weight_decay': self.args.weight_decay,
                'lr': audio_lr
            })
            logger.info(f"Using separate learning rate for audio parameters: {audio_lr}")
        
        # Create optimizer
        optimizer_cls = torch.optim.AdamW
        optimizer_kwargs = {
            'betas': (self.args.adam_beta1, self.args.adam_beta2),
            'eps': self.args.adam_epsilon
        }
        
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        
        return self.optimizer
