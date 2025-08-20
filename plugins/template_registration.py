"""Higgs-Audio template registration for MS-SWIFT.

Custom template for handling multi-modal chat format with audio placeholders.
"""

import os
import re
import json
from typing import Dict, Any, List, Optional, Tuple, Union

from swift.llm.template import (
    Template,
    TemplateMeta,
    register_template
)
from swift.utils import get_logger

logger = get_logger()

# Template configuration
TEMPLATE_TYPE = 'higgs-chatml'


class HiggsChatMLTemplate(Template):
    """Custom template for Higgs-Audio with multi-modal chat support"""
    
    # Audio placeholders
    audio_placeholder = ['<|AUDIO|>']
    audio_out_placeholder = ['<|AUDIO_OUT|>']
    
    def _process_audio_content(self, content: Union[str, List[Dict]], is_assistant: bool = False) -> str:
        """Process content that may contain audio references.
        
        Args:
            content: The content to process
            is_assistant: Whether this is assistant content
            
        Returns:
            Processed content string with audio placeholders
        """
        if isinstance(content, str):
            return content
        
        if not isinstance(content, list):
            return str(content)
        
        # Process multi-modal content
        processed_parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get('type') == 'text':
                    processed_parts.append(item.get('text', ''))
                elif item.get('type') == 'audio':
                    # Replace audio with placeholder
                    if is_assistant:
                        # Assistant audio output
                        processed_parts.append('<|AUDIO_OUT|>')
                    else:
                        # User audio input (reference audio)
                        processed_parts.append('<|AUDIO|>')
            else:
                processed_parts.append(str(item))
        
        return ' '.join(processed_parts)
    
    def encode(self, inputs):
        """Encode messages using MS-Swift's standard encoding.

        This method processes messages with audio placeholders and delegates
        to the parent class for standard encoding.
        """
        # Process messages to replace audio content with placeholders
        if hasattr(inputs, 'messages') and inputs.messages:
            processed_messages = []
            for message in inputs.messages:
                role = message.get('role', '')
                content = message.get('content', '')

                # Process audio content
                processed_content = self._process_audio_content(
                    content,
                    is_assistant=(role == 'assistant')
                )

                processed_messages.append({
                    'role': role,
                    'content': processed_content
                })

            inputs.messages = processed_messages

        # Call parent encoding
        return super().encode(inputs)


def register_higgs_audio_template():
    """Register the Higgs-Audio template with MS-SWIFT.
    
    Returns:
        bool: True if registration successful
    """
    try:
        # Create template metadata with simple ChatML format
        template_meta = TemplateMeta(
            template_type=TEMPLATE_TYPE,
            prefix=['<|im_start|>'],
            prompt=['{{ROLE}}: {{QUERY}}<|im_end|>\n<|im_start|>assistant\n'],
            chat_sep=['<|im_end|>\n<|im_start|>'],
            suffix=['<|im_end|>'],
            system_prefix=['<|im_start|>system\n{{SYSTEM}}<|im_end|>\n'],
            default_system=(
                "You are a helpful assistant capable of generating speech "
                "in the voice of the provided reference audio."
            ),
            template_cls=HiggsChatMLTemplate
        )
        
        # Register the template (idempotent on re-runs)
        register_template(template_meta, exist_ok=True)
        
        logger.info(f"Successfully registered Higgs-Audio template: {TEMPLATE_TYPE}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to register Higgs-Audio template: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Test registration
    success = register_higgs_audio_template()
    if success:
        print(f"✓ Higgs-Audio template '{TEMPLATE_TYPE}' registered successfully")
    else:
        print(f"✗ Failed to register Higgs-Audio template")
        exit(1)
