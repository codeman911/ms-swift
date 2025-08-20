"""Higgs-Audio Training Plugin following CUSTOM_TTS.md Section 4

Implements custom loss computation that sums text and audio losses
as specified in the documentation.
"""

from swift.llm.plugin import register_plugin, PeftTrainerPlugin
from swift.utils import get_logger

logger = get_logger()


class HiggsAudioTrainerPlugin(PeftTrainerPlugin):
    """Training plugin for Higgs-Audio model as per CUSTOM_TTS.md.
    
    This plugin hooks into the training loop and computes the combined
    loss from both text (llm_loss) and audio (audio_loss) outputs.
    """
    
    def compute_loss(self, model, inputs):
        """Compute combined loss from text and audio outputs.
        
        As specified in CUSTOM_TTS.md Section 4:
        - Retrieves llm_loss and audio_loss from model output
        - Returns their sum as total loss
        - Logs them separately for monitoring
        
        Args:
            model: The Higgs-Audio model
            inputs: Input batch dictionary
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Get model outputs
        outputs = model(**inputs)
        
        # Extract losses from HiggsAudioModelOutput
        # outputs.llm_loss and outputs.audio_loss come from HiggsAudioModelOutput
        text_loss = outputs.llm_loss if hasattr(outputs, 'llm_loss') else None
        audio_loss = outputs.audio_loss if hasattr(outputs, 'audio_loss') else None
        
        # Handle cases where losses might not be computed
        if text_loss is None and audio_loss is None:
            # Fall back to standard loss if available
            if hasattr(outputs, 'loss') and outputs.loss is not None:
                return outputs.loss, {"loss": outputs.loss}
            else:
                raise ValueError("No loss computed by model")
        
        # Compute total loss as sum of text and audio losses
        total_loss = 0.0
        loss_dict = {}
        
        if text_loss is not None:
            total_loss = total_loss + text_loss
            loss_dict["text_loss"] = text_loss.item() if hasattr(text_loss, 'item') else text_loss
            
        if audio_loss is not None:
            total_loss = total_loss + audio_loss
            loss_dict["audio_loss"] = audio_loss.item() if hasattr(audio_loss, 'item') else audio_loss
        
        loss_dict["total_loss"] = total_loss.item() if hasattr(total_loss, 'item') else total_loss
        
        # Log losses for monitoring
        if len(loss_dict) > 1:  # Only log if we have separate losses
            logger.debug(f"Loss breakdown: {loss_dict}")
        
        return total_loss, loss_dict


# Register the plugin with MS-SWIFT
register_plugin(HiggsAudioTrainerPlugin())

logger.info("✅ Higgs-Audio training plugin registered")
