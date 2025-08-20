"""Higgs-Audio template registration for MS-SWIFT following CUSTOM_TTS.md

Custom template for handling multi-modal chat format with audio placeholders.
"""

import os
import re
import json
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field

from swift.llm.template import (
    Template,
    TemplateMeta,
    register_template,
    Prompt,
    Messages,
    History,
    encode_context,
    StopWords
)
from swift.llm.template.utils import Context
from swift.utils import get_logger

logger = get_logger()

# Template configuration
TEMPLATE_TYPE = 'higgs-chatml'

@dataclass
class HiggsChatMLTemplateMeta(TemplateMeta):
    """Template metadata for Higgs-Audio ChatML format"""
    
    template_type: str = TEMPLATE_TYPE
    prefix: List[str] = field(default_factory=lambda: ['<|im_start|>system\n{{SYSTEM}}<|im_end|>\n'])
    prompt: List[str] = field(default_factory=lambda: [
        '<|im_start|>user\n{{QUERY}}<|im_end|>\n',
        '<|im_start|>assistant\n{{RESPONSE}}'
    ])
    chat_sep: Optional[str] = None
    suffix: List[str] = field(default_factory=lambda: ['<|im_end|>'])
    default_system: Optional[str] = 'You are a helpful assistant capable of generating speech in the voice of the provided reference audio.'
    stop_words: List[str] = field(default_factory=lambda: ['<|im_end|>'])
    audio_placeholder: str = '<|AUDIO|>'
    audio_out_placeholder: str = '<|AUDIO_OUT|>'


class HiggsChatMLTemplate(Template):
    """Custom template for Higgs-Audio with multi-modal chat support"""
    
    def __init__(self, tokenizer, default_system=None, max_length=None, truncation_strategy='delete', **kwargs):
        """Initialize Higgs-Audio ChatML template.
        
        Args:
            tokenizer: The tokenizer to use
            default_system: Default system message
            max_length: Maximum sequence length
            truncation_strategy: How to handle truncation
        """
        self.tokenizer = tokenizer
        self.default_system = default_system or HiggsChatMLTemplateMeta.default_system
        self.max_length = max_length
        self.truncation_strategy = truncation_strategy
        
        # Audio placeholders
        self.audio_placeholder = '<|AUDIO|>'
        self.audio_out_placeholder = '<|AUDIO_OUT|>'
        self.audio_bos = '<|audio_bos|>'
        self.audio_eos = '<|audio_eos|>'
        self.audio_out_bos = '<|audio_out_bos|>'
        self.audio_out_eos = '<|audio_out_eos|>'
        
        # Chat template tokens
        self.im_start = '<|im_start|>'
        self.im_end = '<|im_end|>'
        
        super().__init__(
            tokenizer=tokenizer,
            default_system=self.default_system,
            max_length=max_length,
            truncation_strategy=truncation_strategy,
            **kwargs
        )
    
    def _process_audio_content(self, content: Union[str, List[Dict]], is_assistant: bool = False) -> str:
        """Process content that may contain audio references.
        
        Args:
            content: The content to process (text or list of content items)
            is_assistant: Whether this is assistant content
            
        Returns:
            Processed content string with audio placeholders
        """
        if isinstance(content, str):
            return content
        
        if not isinstance(content, list):
            return str(content)
        
        processed_parts = []
        
        for item in content:
            if isinstance(item, dict):
                if item.get('type') == 'text':
                    processed_parts.append(item.get('text', ''))
                elif item.get('type') == 'audio':
                    # Handle audio content
                    if is_assistant:
                        # Assistant audio output
                        processed_parts.append(self.audio_out_placeholder)
                    else:
                        # User audio input (reference audio)
                        processed_parts.append(self.audio_placeholder)
            else:
                processed_parts.append(str(item))
        
        return ' '.join(processed_parts)
    
    def encode(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Encode an example for training.
        
        Args:
            example: The example to encode
            
        Returns:
            Encoded example with input_ids and labels
        """
        messages = example.get('messages', [])
        
        if not messages:
            return {'input_ids': [], 'labels': []}
        
        # Build the conversation
        input_ids = []
        labels = []
        
        for message in messages:
            role = message.get('role', '')
            content = message.get('content', '')
            
            if role == 'system':
                # System message
                system_text = self._process_audio_content(content)
                system_tokens = self.tokenizer.encode(
                    f"{self.im_start}system\n{system_text}{self.im_end}\n",
                    add_special_tokens=False
                )
                input_ids.extend(system_tokens)
                labels.extend([-100] * len(system_tokens))  # Don't compute loss on system
                
            elif role == 'user':
                # User message
                user_text = self._process_audio_content(content)
                user_tokens = self.tokenizer.encode(
                    f"{self.im_start}user\n{user_text}{self.im_end}\n",
                    add_special_tokens=False
                )
                input_ids.extend(user_tokens)
                labels.extend([-100] * len(user_tokens))  # Don't compute loss on user input
                
            elif role == 'assistant':
                # Assistant message
                assistant_text = self._process_audio_content(content, is_assistant=True)
                
                # Add prompt part (no loss)
                prompt_tokens = self.tokenizer.encode(
                    f"{self.im_start}assistant\n",
                    add_special_tokens=False
                )
                input_ids.extend(prompt_tokens)
                labels.extend([-100] * len(prompt_tokens))
                
                # Add response part (compute loss)
                response_tokens = self.tokenizer.encode(
                    assistant_text,
                    add_special_tokens=False
                )
                input_ids.extend(response_tokens)
                labels.extend(response_tokens)  # Compute loss on assistant response
                
                # Add end token
                end_tokens = self.tokenizer.encode(
                    self.im_end,
                    add_special_tokens=False
                )
                input_ids.extend(end_tokens)
                labels.extend(end_tokens)  # Include end token in loss
        
        # Add final tokens if needed
        if len(input_ids) > 0 and input_ids[-1] != self.tokenizer.eos_token_id:
            input_ids.append(self.tokenizer.eos_token_id)
            labels.append(self.tokenizer.eos_token_id)
        
        # Truncate if needed
        if self.max_length and len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': [1] * len(input_ids)
        }
    
    def decode(self, input_ids: List[int], **kwargs) -> str:
        """Decode token IDs to text.
        
        Args:
            input_ids: The token IDs to decode
            
        Returns:
            Decoded text
        """
        return self.tokenizer.decode(input_ids, skip_special_tokens=True, **kwargs)


def register_higgs_audio_template():
    """Register the Higgs-Audio template with MS-SWIFT"""
    
    # Register template metadata
    register_template(
        TemplateMeta(
            template_type=TEMPLATE_TYPE,
            prefix=['<|im_start|>system\n{{SYSTEM}}<|im_end|>\n'],
            prompt=[
                '<|im_start|>user\n{{QUERY}}<|im_end|>\n',
                '<|im_start|>assistant\n{{RESPONSE}}'
            ],
            chat_sep=None,
            suffix=['<|im_end|>'],
            default_system='You are a helpful assistant capable of generating speech in the voice of the provided reference audio.',
            stop_words=['<|im_end|>'],
            efficient_eos=True,
        ),
        exist_ok=True
    )
    
    # Also register the template class
    TemplateType.higgs_chatml = TEMPLATE_TYPE
    
    logger.info("✅ Higgs-Audio template registered successfully")
