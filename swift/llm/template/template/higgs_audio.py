"""Higgs-Audio template registration for MS-SWIFT.

This module registers the Higgs-Audio chat template with MS-SWIFT's template registry.
"""

from typing import Any, Dict, List, Optional, Tuple

from swift.utils import get_logger
from ..base import Template
from ..constant import TemplateType
from ..register import TemplateMeta, register_template

logger = get_logger()


class HiggsChatMLTemplate(Template):
    """Custom ChatML template for Higgs-Audio with audio placeholders."""
    
    def __init__(self):
        super().__init__(
            prefix=['<|im_start|>system\nYou are a helpful assistant capable of generating speech in the voice of the provided reference audio.<|im_end|>\n'],
            prompt=['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'],
            chat_sep=['\n'],
            suffix=['<|im_end|>'],
            default_system='You are a helpful assistant capable of generating speech in the voice of the provided reference audio.'
        )
    
    def encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Encode messages with audio placeholders.
        
        Args:
            example: Input example with messages
        
        Returns:
            Tuple of (inputs, tokenizer_kwargs)
        """
        messages = example.get('messages', [])
        
        if not messages:
            return super().encode(example)
        
        # Process messages to add audio placeholders
        processed_messages = []
        for msg in messages:
            role = msg.get('role', '')
            content = msg.get('content', '')
            
            if isinstance(content, list):
                # Multi-modal content
                text_parts = []
                audio_parts = []
                
                for item in content:
                    if item.get('type') == 'text':
                        text_parts.append(item.get('text', ''))
                    elif item.get('type') == 'audio':
                        audio_parts.append('<|audio_start|>AUDIO_PLACEHOLDER<|audio_end|>')
                
                # Combine text and audio
                combined_content = ' '.join(text_parts)
                if audio_parts:
                    combined_content += ' ' + ' '.join(audio_parts)
                
                processed_messages.append({
                    'role': role,
                    'content': combined_content
                })
            else:
                processed_messages.append(msg)
        
        # Update example with processed messages
        example = dict(example)
        example['messages'] = processed_messages
        
        return super().encode(example)


# Create custom TemplateType for Higgs-Audio
class HiggsTemplateType:
    higgs_chatml = 'higgs-chatml'


# Add to TemplateType if possible
if hasattr(TemplateType, '__dict__'):
    TemplateType.higgs_chatml = HiggsTemplateType.higgs_chatml


# Register the template
register_template(
    TemplateMeta(
        template_type=HiggsTemplateType.higgs_chatml,
        template=HiggsChatMLTemplate(),
        infer_media_type='round',
    ))

logger.info(f"Registered Higgs-Audio template: {HiggsTemplateType.higgs_chatml}")
