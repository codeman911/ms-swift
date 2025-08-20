"""
Custom template for Higgs-Audio training with voice cloning support.
"""

from swift.llm.utils.template import Template
from collator import HiggsAudioDataCollator

class HiggsAudioTemplate(Template):
    """
    A custom template for Higgs-Audio that uses the HiggsAudioDataCollator.
    Supports both TTS and voice cloning with reference audio.
    """
    def __init__(self):
        # Define special tokens for Higgs-Audio
        special_tokens = [
            '<|AUDIO|>',      # Audio input placeholder
            '<|AUDIO_OUT|>',  # Audio output placeholder  
            '<|audio_bos|>',  # Audio beginning of sequence
            '<|audio_eos|>',  # Audio end of sequence
            '<|DELAY|>'       # Delay pattern token
        ]
        
        super().__init__(
            prefix=[],
            prompt=['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'],
            chat_ml=True,
            suffix=['<|im_end|>'],
            system_prefix=['<|im_start|>system\n{{SYSTEM}}<|im_end|>\n'],
            special_tokens=special_tokens,
            auto_add_bos=True,
        )
        
        # Set data collator class
        self.data_collator_class = HiggsAudioDataCollator
    
    def encode(self, example):
        """Encode example with audio support."""
        # Handle messages with audio content
        if 'messages' in example:
            messages = example['messages']
            audios = example.get('audios', [])
            reference_audio = example.get('reference_audio', None)
            
            # Process messages and insert audio placeholders
            processed_messages = []
            audio_idx = 0
            
            for message in messages:
                content = message.get('content', '')
                
                # Replace audio references with placeholders
                if '<audio>' in content.lower():
                    if audio_idx < len(audios):
                        content = content.replace('<audio>', '<|AUDIO|>')
                        audio_idx += 1
                
                # Handle voice cloning with reference audio
                if reference_audio and message.get('role') == 'user':
                    content = f"Reference: <|AUDIO|>\n{content}"
                
                processed_messages.append({
                    'role': message['role'],
                    'content': content
                })
            
            example['messages'] = processed_messages
        
        return super().encode(example)
