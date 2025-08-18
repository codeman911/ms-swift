# plugins/loss.py
# Purpose: compute joint loss from model outputs and collated labels
# Assumes your model forward returns both text and audio logits OR a single dict with these keys.

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Dict, Any

def higgs_text_audio_loss(outputs: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
    """
    Compute joint text and audio loss for HiggsAudio model.
    
    Expected inputs:
    - outputs['text_logits'] or outputs['logits']: [B, T_text, V_text] 
    - outputs['audio_logits']: [B, 8, T_audio, V_audio]
    - labels['text_labels']: [B, T_text] with -100 for masked positions
    - labels['audio_labels']: [B, 8, T_audio] with -100 for masked positions
    
    Returns:
        Combined loss tensor (text_ce + audio_ce)
    """
    device = next(iter(outputs.values())).device
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    # === Text Loss (Causal Language Modeling) ===
    text_logits = outputs.get('text_logits') or outputs.get('logits')
    text_labels = labels.get('text_labels')
    
    if text_logits is not None and text_labels is not None:
        # text_logits: [B, T, V], text_labels: [B, T]
        # Labels are already shifted in the collator
        text_loss = F.cross_entropy(
            text_logits.view(-1, text_logits.size(-1)),
            text_labels.view(-1),
            ignore_index=-100,
            reduction='mean'
        )
        total_loss = total_loss + text_loss
    
    # === Audio Loss (Multi-codebook RVQ) ===
    audio_logits = outputs.get('audio_logits')
    audio_labels = labels.get('audio_labels')
    
    if audio_logits is not None and audio_labels is not None:
        # audio_logits: [B, 8, T, V], audio_labels: [B, 8, T]
        B, num_codebooks, T, vocab_size = audio_logits.shape
        
        # Flatten for cross entropy computation
        audio_logits_flat = audio_logits.view(-1, vocab_size)  # [B*8*T, V]
        audio_labels_flat = audio_labels.view(-1)  # [B*8*T]
        
        audio_loss = F.cross_entropy(
            audio_logits_flat,
            audio_labels_flat,
            ignore_index=-100,
            reduction='mean'
        )
        total_loss = total_loss + audio_loss
    
    return total_loss

# Register the loss function with MS-SWIFT
from swift.llm import register_loss

register_loss(
    loss_name="higgs_text_audio",
    loss_function=higgs_text_audio_loss
)

print("[INFO] Registered higgs_text_audio loss function with MS-SWIFT")
