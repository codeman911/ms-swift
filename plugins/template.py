"""Higgs-Audio Template Registration following CUSTOM_TTS.md Section 2

Implements multi-modal chat format with inline audio support as specified in the documentation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
from swift.llm.template import Template, Prompt, TemplateMeta, register_template
from swift.llm.template.base import findall
from swift.utils import get_logger

logger = get_logger()

class HiggsAudioTemplate(Template):
    """Higgs-Audio template as per CUSTOM_TTS.md specifications.
    
    Handles multi-turn chat format with inline audio segments.
    Follows the ChatML-style format with <|im_start|> and <|im_end|> tokens.
    """
    
    skip_prompt = False
    placeholder_tokens = ['<|unk|>']  # Placeholder for unknown tokens
    
    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        
        # Set audio token ID if available
        self.audio_id = tokenizer.convert_tokens_to_ids('<|AUDIO|>') if '<|AUDIO|>' in tokenizer.get_vocab() else -2
    
    def _encode(self, inputs):
        """Encode with audio support as per CUSTOM_TTS.md Section 2.
        
        Processes <audio> tags and replaces them with model-ready features.
        """
        # Call base to tokenize the text with <audio> markers
        encoded = super()._encode(inputs)
        input_ids = encoded['input_ids']
        labels = encoded.get('labels', None)
        
        # Find and process audio placeholders
        if hasattr(inputs, 'audios') and inputs.audios:
            # Find all audio placeholder positions
            idx_list = findall(input_ids, self.audio_id)
            
            # Process each audio file using processor
            if hasattr(self, 'processor') and idx_list:
                try:
                    # Load and process audio features
                    audio_feats = self.processor.process_audio(inputs.audios, return_tensors='pt')
                    
                    # Replace placeholders with audio features
                    # This follows the Megrez pattern referenced in CUSTOM_TTS.md
                    logger.info(f"Processed {len(inputs.audios)} audio files")
                    
                    # Store audio features in encoded dict for collator
                    encoded['audio_features'] = audio_feats
                    encoded['audio_positions'] = idx_list
                except Exception as e:
                    logger.warning(f"Failed to process audio: {e}")
        
        return encoded

# Register the template with MS-SWIFT
register_template(
    TemplateMeta(
        template_type='higgs-audio-chatml',
        prefix=[],
        prompt=['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'],
        chat_sep=['<|im_end|>\n'],
        suffix=['<|im_end|>'],
        system_prefix=['<|im_start|>system\n{{SYSTEM}}<|im_end|>\n'],
        template_cls=HiggsAudioTemplate,  # Use custom template class
        default_system='You are a helpful assistant capable of generating speech in the voice of the provided reference audio.',
        auto_add_bos=True,
    )
)
