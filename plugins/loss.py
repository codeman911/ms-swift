# plugins/loss.py
# Purpose: compute joint loss from model outputs and collated labels
# Assumes your model forward returns both text and audio logits OR a single dict with these keys.

import torch
import torch.nn.functional as F

def higgs_text_audio_loss(outputs, labels, loss_scale=None, num_items_in_batch=None) -> torch.Tensor:
    """
    Compute joint text + audio cross-entropy loss.
    
    Expected inputs:
      labels["text_labels"]:    [B, Ttxt] - text token labels
      labels["audio_labels"]:   [B, C=8, Taud] - audio RVQ code labels
      
    Expected model outputs:
      outputs["text_logits"]:   [B, Ttxt, Vtxt] OR outputs["logits"]
      outputs["audio_logits"]:  [B, C=8, Taud, Vaud] OR outputs["audio"]["logits"]
      
    The collator should have already:
      - Applied teacher forcing shift for audio decoder inputs
      - Set labels[:, :, 0] = -100 for audio (ONLY first step masked)
      - NO EOS masking anywhere
    """
    device = next(iter(outputs.values())).device if isinstance(outputs, dict) else outputs.device
    
    # ---- Fetch logits (be permissive with different key formats) ----
    text_logits = None
    audio_logits = None
    
    if hasattr(outputs, 'logits'):
        text_logits = outputs.logits
    elif isinstance(outputs, dict):
        text_logits = outputs.get("text_logits", outputs.get("logits", None))
        audio_logits = outputs.get("audio_logits", None)
        
        # Try nested audio structure
        if audio_logits is None and "audio" in outputs:
            audio_dict = outputs["audio"]
            if isinstance(audio_dict, dict):
                audio_logits = audio_dict.get("logits", None)
            elif hasattr(audio_dict, 'logits'):
                audio_logits = audio_dict.logits
    
    # Fallback: try as attributes
    if text_logits is None and hasattr(outputs, 'text_logits'):
        text_logits = outputs.text_logits
    if audio_logits is None and hasattr(outputs, 'audio_logits'):
        audio_logits = outputs.audio_logits
    
    # ---- Validate we have the required tensors ----
    if text_logits is None:
        print(f"[ERROR] Missing text logits in model outputs. Available keys: {list(outputs.keys()) if isinstance(outputs, dict) else 'Not a dict'}")
        # Return a dummy loss to prevent training crash
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    if audio_logits is None:
        print(f"[WARNING] Missing audio logits in model outputs. Using text-only loss.")
        # Fall back to text-only loss
        audio_loss = torch.tensor(0.0, device=device)
    
    # ---- Extract labels ----
    text_labels = labels.get("text_labels", labels.get("labels", None))
    audio_labels = labels.get("audio_labels", None)
    
    if text_labels is None:
        print(f"[ERROR] Missing text_labels in labels. Available keys: {list(labels.keys())}")
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Move to correct device
    text_labels = text_labels.to(device)
    if audio_labels is not None:
        audio_labels = audio_labels.to(device)
    
    # ---- Compute Text Cross-Entropy Loss ----
    Vtxt = text_logits.size(-1)
    text_loss = F.cross_entropy(
        text_logits.reshape(-1, Vtxt),
        text_labels.reshape(-1),
        ignore_index=-100,
        reduction='mean'
    )
    
    # ---- Compute Audio Cross-Entropy Loss ----
    if audio_logits is not None and audio_labels is not None:
        # audio_logits: [B, C=8, Taud, Vaud]
        # audio_labels: [B, C=8, Taud]
        Vaud = audio_logits.size(-1)
        audio_loss = F.cross_entropy(
            audio_logits.reshape(-1, Vaud),
            audio_labels.reshape(-1),
            ignore_index=-100,
            reduction='mean'
        )
    else:
        audio_loss = torch.tensor(0.0, device=device)
    
    # ---- Combine losses ----
    total_loss = text_loss + audio_loss
    
    # ---- Store individual losses for logging ----
    if isinstance(outputs, dict):
        outputs["text_ce"] = text_loss.detach()
        outputs["audio_ce"] = audio_loss.detach()
        outputs["total_loss"] = total_loss.detach()
    
    # ---- Debug logging ----
    if torch.rand(1).item() < 0.01:  # Log ~1% of batches
        print(f"[DEBUG LOSS] Text: {text_loss.item():.4f}, Audio: {audio_loss.item():.4f}, Total: {total_loss.item():.4f}")
        print(f"[DEBUG SHAPES] Text logits: {text_logits.shape}, Audio logits: {audio_logits.shape if audio_logits is not None else 'None'}")
        print(f"[DEBUG LABELS] Text labels: {text_labels.shape}, Audio labels: {audio_labels.shape if audio_labels is not None else 'None'}")
    
    return total_loss

def higgs_text_only_loss(outputs, labels, loss_scale=None, num_items_in_batch=None) -> torch.Tensor:
    """
    Fallback text-only loss for debugging or when audio components fail.
    """
    device = next(iter(outputs.values())).device if isinstance(outputs, dict) else outputs.device
    
    # Get text logits
    if hasattr(outputs, 'logits'):
        text_logits = outputs.logits
    elif isinstance(outputs, dict):
        text_logits = outputs.get("text_logits", outputs.get("logits", None))
    else:
        text_logits = outputs
    
    # Get text labels
    text_labels = labels.get("text_labels", labels.get("labels", None))
    if text_labels is None:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    text_labels = text_labels.to(device)
    
    # Compute CE loss
    Vtxt = text_logits.size(-1)
    loss = F.cross_entropy(
        text_logits.reshape(-1, Vtxt),
        text_labels.reshape(-1),
        ignore_index=-100,
        reduction='mean'
    )
    
    return loss

# Register loss functions for MS-SWIFT
# MS-SWIFT will look for loss_mapping dictionary
loss_mapping = {
    "higgs_text_audio": higgs_text_audio_loss,
    "higgs_text_only": higgs_text_only_loss
}

print("[INFO] Registered custom loss functions: higgs_text_audio, higgs_text_only")
