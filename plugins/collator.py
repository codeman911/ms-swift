"""Higgs-Audio data collator for MS-SWIFT using original wrapper classes.

This module wraps the original Higgs-Audio collator from boson_multimodal
for seamless integration with MS-SWIFT training.
"""

from __future__ import annotations

import os
import sys
import torch
import librosa
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Transformers types and processors
from transformers import PreTrainedTokenizer, WhisperProcessor

# Add Higgs-Audio path to sys.path for imports
higgs_audio_path = Path(__file__).parent.parent / "higgs-audio"
if str(higgs_audio_path) not in sys.path:
    sys.path.insert(0, str(higgs_audio_path))

# Import original Higgs-Audio collator and components
from boson_multimodal.data_collator.higgs_audio_collator import (
    HiggsAudioSampleCollator,
    HiggsAudioBatchInput,
)
from boson_multimodal.dataset.chatml_dataset import ChatMLDatasetSample
from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer

# Import MS-SWIFT utilities
from swift.utils import get_logger

logger = get_logger()

# Re-export the batch input class for compatibility
__all__ = ['HiggsAudioDataCollator', 'HiggsAudioBatchInput', 'ChatMLDatasetSample']


class HiggsAudioDataCollator:
    """MS-SWIFT wrapper for the original Higgs-Audio collator.
    
    This class wraps the original HiggsAudioSampleCollator from boson_multimodal
    to provide compatibility with MS-SWIFT's training pipeline.
    """
    
    def __init__(
        self,
        tokenizer,
        whisper_processor: Optional[WhisperProcessor] = None,
        audio_sample_rate: int = 24000,
        max_audio_length: int = 30,  # seconds
        audio_token_offset: int = 128000,  # Offset for audio tokens
        use_delay_pattern: bool = True,
        audio_num_codebooks: int = 4,
        pad_token_id: Optional[int] = None,
        ignore_index: int = -100,
        chunk_size_seconds: int = 30,
        add_new_bos_eos_for_long_chunk: bool = True,
        mask_audio_out_token_label: bool = True,
    ):
        """Initialize the wrapper collator.
        
        Args:
            tokenizer: The text tokenizer
            whisper_processor: Whisper processor for audio feature extraction
            audio_sample_rate: Sample rate for audio (24kHz for Higgs-Audio)
            max_audio_length: Maximum audio length in seconds
            audio_token_offset: Offset to apply to audio tokens
            use_delay_pattern: Whether to use delay pattern for RVQ codes
            audio_num_codebooks: Number of audio codebooks
            pad_token_id: Token ID for padding
            ignore_index: Label value to ignore in loss computation
            chunk_size_seconds: Size of audio chunks in seconds
            add_new_bos_eos_for_long_chunk: Add BOS/EOS for long chunks
            mask_audio_out_token_label: Mask audio out token in labels
        """
        self.tokenizer = tokenizer
        
        # Load Whisper processor if not provided
        if whisper_processor is None:
            whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-base")
        
        # Get special token IDs
        audio_in_token_id = tokenizer.convert_tokens_to_ids("<|AUDIO|>")
        audio_out_token_id = tokenizer.convert_tokens_to_ids("<|AUDIO_OUT|>")
        audio_bos_token_id = tokenizer.convert_tokens_to_ids("<|audio_bos|>")
        audio_eos_token_id = tokenizer.convert_tokens_to_ids("<|audio_eos|>")
        
        # Initialize the original Higgs-Audio collator
        self.collator = HiggsAudioSampleCollator(
            whisper_processor=whisper_processor,
            audio_in_token_id=audio_in_token_id,
            audio_out_token_id=audio_out_token_id,
            pad_token_id=pad_token_id or tokenizer.pad_token_id,
            audio_stream_bos_id=audio_bos_token_id,
            audio_stream_eos_id=audio_eos_token_id,
            round_to=8,
            pad_left=False,
            encode_whisper_embed=True,
            return_audio_in_tokens=True,
            audio_num_codebooks=audio_num_codebooks,
            use_delay_pattern=use_delay_pattern,
            disable_audio_codes_transform=False,
            chunk_size_seconds=chunk_size_seconds,
            add_new_bos_eos_for_long_chunk=add_new_bos_eos_for_long_chunk,
            mask_audio_out_token_label=mask_audio_out_token_label,
        )
        
        self.audio_sample_rate = audio_sample_rate
        self.ignore_index = ignore_index
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Process batch of samples using original collator.
        
        Args:
            features: List of sample dictionaries from MS-SWIFT with ChatMLDatasetSample
            
        Returns:
            Dictionary of batched tensors for model input
        """
        
        # Extract ChatMLDatasetSample objects from preprocessed features
        chatml_samples = []
        for feature in features:
            if 'chatml_sample' in feature:
                # Use the preprocessed ChatMLDatasetSample
                chatml_samples.append(feature['chatml_sample'])
            else:
                # Fallback: create from basic features
                input_ids = feature.get('input_ids', torch.tensor([]))
                labels = feature.get('labels', feature.get('label_ids', None))
                
                sample = ChatMLDatasetSample(
                    input_ids=input_ids,
                    label_ids=labels,
                    audio_ids_concat=torch.tensor([[]]),
                    audio_ids_start=torch.tensor([]),
                    audio_waveforms_concat=torch.tensor([]),
                    audio_waveforms_start=torch.tensor([]),
                    audio_sample_rate=torch.tensor([]),
                    audio_speaker_indices=torch.tensor([]),
                    audio_label_ids_concat=None
                )
                chatml_samples.append(sample)
        
        # Use original collator
        try:
            batch_input = self.collator(chatml_samples)
            
            # Convert HiggsAudioBatchInput to dictionary format expected by MS-SWIFT
            result = {
                'input_ids': batch_input.input_ids,
                'attention_mask': batch_input.attention_mask,
                'labels': batch_input.label_ids,
            }
            
            # Add audio-specific fields if they exist
            if batch_input.audio_features is not None:
                result['audio_features'] = batch_input.audio_features
            if batch_input.audio_feature_attention_mask is not None:
                result['audio_feature_attention_mask'] = batch_input.audio_feature_attention_mask
            if batch_input.audio_out_ids is not None:
                result['audio_out_ids'] = batch_input.audio_out_ids
            if batch_input.audio_out_ids_start is not None:
                result['audio_out_ids_start'] = batch_input.audio_out_ids_start
            if batch_input.audio_out_ids_start_group_loc is not None:
                result['audio_out_ids_start_group_loc'] = batch_input.audio_out_ids_start_group_loc
            if batch_input.audio_in_ids is not None:
                result['audio_in_ids'] = batch_input.audio_in_ids
            if batch_input.audio_in_ids_start is not None:
                result['audio_in_ids_start'] = batch_input.audio_in_ids_start
            if batch_input.label_audio_ids is not None:
                result['label_audio_ids'] = batch_input.label_audio_ids
                
            return result
            
        except Exception as e:
            logger.error(f"Error in collator: {e}")
            # Fallback to basic collation
            return self._fallback_collate(features)
    
    def _fallback_collate(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Convert MS-SWIFT dataset format to ChatMLDatasetSample format
        samples = []
        for feature in features:
            # Extract required fields from MS-SWIFT format
            input_ids = feature.get('input_ids')
            labels = feature.get('labels', input_ids)  # Use input_ids as labels if not provided
            
            # Handle audio data if present
            audio_ids = feature.get('audio_ids', None)
            audio_waveforms = feature.get('audio_waveforms', None)
            
            # Create ChatMLDatasetSample compatible with original collator
            if isinstance(input_ids, list):
                input_ids = torch.tensor(input_ids, dtype=torch.long)
            if isinstance(labels, list):
                labels = torch.tensor(labels, dtype=torch.long)
                
            # Prepare audio tensors if present
            if audio_ids is not None:
                if isinstance(audio_ids, list):
                    audio_ids = torch.tensor(audio_ids, dtype=torch.long)
                audio_ids_concat = audio_ids if audio_ids.dim() == 2 else audio_ids.unsqueeze(0)
                audio_ids_start = torch.tensor([0], dtype=torch.long)
            else:
                audio_ids_concat = torch.zeros((self.collator.audio_num_codebooks, 0), dtype=torch.long)
                audio_ids_start = torch.zeros(0, dtype=torch.long)
                
            if audio_waveforms is not None:
                if isinstance(audio_waveforms, list):
                    audio_waveforms = torch.tensor(audio_waveforms, dtype=torch.float32)
                audio_waveforms_concat = audio_waveforms.flatten()
                audio_waveforms_start = torch.tensor([0], dtype=torch.long)
                audio_sample_rate = torch.tensor([self.audio_sample_rate], dtype=torch.float32)
            else:
                audio_waveforms_concat = torch.zeros(0, dtype=torch.float32)
                audio_waveforms_start = torch.zeros(0, dtype=torch.long)
                audio_sample_rate = torch.zeros(0, dtype=torch.float32)
                
            # Create ChatMLDatasetSample
            sample = ChatMLDatasetSample(
                input_ids=input_ids,
                label_ids=labels,
                audio_ids_concat=audio_ids_concat,
                audio_ids_start=audio_ids_start,
                audio_waveforms_concat=audio_waveforms_concat,
                audio_waveforms_start=audio_waveforms_start,
                audio_sample_rate=audio_sample_rate,
                audio_speaker_indices=torch.tensor([-1] if audio_waveforms is not None else [], dtype=torch.long),
                audio_label_ids_concat=audio_ids_concat.clone() if audio_ids is not None else None,
            )
            samples.append(sample)
        
        # Use original collator to process samples
        batch = self.collator(samples)
        
        # Return the batch as-is (HiggsAudioBatchInput instance)
        return batch


class HiggsAudioCollator:
    """
    Data collator for Higgs-Audio training
    Handles ChatML to tensor conversion with validation
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        audio_tokenizer: Optional[Any] = None,
        whisper_processor: Optional[Any] = None,
        max_seq_length: int = 2048,
        max_audio_length: int = 1024,
        num_codebooks: int = 12,
        audio_sample_rate: int = 24000,
        use_delay_pattern: bool = True,
        delay_pattern_steps: int = 1,
        pad_to_multiple_of: Optional[int] = 8,
        audio_token_offset: int = 128000,
        validate_inputs: bool = True,
        device: Optional[torch.device] = None
    ):
        """
        Initialize collator
        
        Args:
            tokenizer: Text tokenizer
            audio_tokenizer: Audio RVQ tokenizer
            whisper_processor: Whisper feature extractor
            max_seq_length: Maximum sequence length
            max_audio_length: Maximum audio token length
            num_codebooks: Number of RVQ codebooks
            audio_sample_rate: Audio sampling rate
            use_delay_pattern: Apply delay pattern mask for causality
            delay_pattern_steps: Delay steps between codebooks
            pad_to_multiple_of: Pad sequence to multiple of this
            audio_token_offset: Offset for audio token IDs
            validate_inputs: Validate tensor shapes and values
            device: Target device for tensors
        """
        
        self.tokenizer = tokenizer
        self.audio_tokenizer = audio_tokenizer
        self.whisper_processor = whisper_processor
        self.max_seq_length = max_seq_length
        self.max_audio_length = max_audio_length
        self.num_codebooks = num_codebooks
        self.audio_sample_rate = audio_sample_rate
        self.use_delay_pattern = use_delay_pattern
        self.delay_pattern_steps = delay_pattern_steps
        self.pad_to_multiple_of = pad_to_multiple_of
        self.audio_token_offset = audio_token_offset
        self.validate_inputs = validate_inputs
        self.device = device or torch.device('cpu')
        
        # Special token IDs
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        self.audio_bos_token_id = tokenizer.convert_tokens_to_ids('<|audio_stream_bos|>')
        self.audio_eos_token_id = tokenizer.convert_tokens_to_ids('<|audio_stream_eos|>')
        self.audio_token_id = tokenizer.convert_tokens_to_ids('<|audio|>')
        
        logger.info(f"Initialized HiggsAudioCollator: "
                   f"max_seq={max_seq_length}, max_audio={max_audio_length}, "
                   f"codebooks={num_codebooks}, device={self.device}")
    
    def __call__(self, features: List[Dict[str, Any]]) -> HiggsAudioBatchInput:
        """
        Collate batch of samples
        
        Args:
            features: List of sample dictionaries with 'messages' and 'audios'
            
        Returns:
            HiggsAudioBatchInput with batched tensors
        """
        
        batch_size = len(features)
        
        # Process each sample
        processed_samples = []
        for feature in features:
            try:
                sample = self._process_sample(feature)
                processed_samples.append(sample)
            except Exception as e:
                logger.error(f"Failed to process sample: {e}")
                if self.validate_inputs:
                    raise
                # Skip invalid sample
                continue
        
        if not processed_samples:
            raise ValueError("No valid samples in batch")
        
        # Batch samples
        batch = self._batch_samples(processed_samples)
        
        # Validate if enabled
        if self.validate_inputs:
            self._validate_batch(batch)
        
        # Move to device
        batch = batch.to(self.device)
        
        return batch
    
    def _process_sample(self, feature: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Process single sample from ChatML to tensors
        
        Args:
            feature: Sample with 'messages' and optional 'audios'
            
        Returns:
            Dictionary of tensors for this sample
        """
        
        messages = feature.get('messages', [])
        audios = feature.get('audios', [])
        
        # Build conversation text
        text = self._build_conversation_text(messages)
        
        # Tokenize text
        text_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_length,
            padding=False,
            return_tensors='pt'
        )
        
        input_ids = text_encoding['input_ids'].squeeze(0)
        attention_mask = text_encoding['attention_mask'].squeeze(0)
        
        # Process audio if present
        audio_features = None
        audio_feature_mask = None
        audio_in_ids = None
        audio_out_ids = None
        
        if audios:
            audio_data = self._process_audio_files(audios)
            
            if audio_data:
                audio_features = audio_data.get('features')
                audio_feature_mask = audio_data.get('attention_mask')
                audio_in_ids = audio_data.get('input_codes')
                audio_out_ids = audio_data.get('output_codes')
        
        # Create labels
        label_ids = self._create_text_labels(input_ids, messages)
        label_audio_ids = self._create_audio_labels(audio_out_ids) if audio_out_ids is not None else None
        
        # Build sample dict
        sample = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label_ids': label_ids
        }
        
        if audio_features is not None:
            sample['audio_features'] = audio_features
            sample['audio_feature_attention_mask'] = audio_feature_mask
        
        if audio_in_ids is not None:
            sample['audio_in_ids'] = audio_in_ids
            sample['audio_in_ids_start'] = torch.tensor([0], dtype=torch.long)
        
        if audio_out_ids is not None:
            sample['audio_out_ids'] = audio_out_ids
            sample['audio_out_ids_start'] = torch.tensor([len(input_ids)], dtype=torch.long)
            sample['label_audio_ids'] = label_audio_ids
        
        return sample
    
    def _build_conversation_text(self, messages: List[Dict[str, str]]) -> str:
        """
        Build conversation text from ChatML messages
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            
        Returns:
            Formatted conversation string
        """
        
        conversation = []
        
        for msg in messages:
            role = msg['role']
            content = msg['content']
            
            if role == 'system':
                conversation.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == 'user':
                conversation.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == 'assistant':
                conversation.append(f"<|im_start|>assistant\n{content}")
                if content:  # Only add end token if response exists
                    conversation.append("<|im_end|>")
        
        return "\n".join(conversation)
    
    def _process_audio_files(self, audio_paths: List[str]) -> Optional[Dict[str, torch.Tensor]]:
        """
        Process audio files to features and codes
        
        Args:
            audio_paths: List of audio file paths
            
        Returns:
            Dictionary with audio tensors or None
        """
        
        if not self.audio_tokenizer or not audio_paths:
            return None
        
        try:
            import librosa
            
            # Load and concatenate audio
            waveforms = []
            for path in audio_paths:
                waveform, sr = librosa.load(path, sr=self.audio_sample_rate)
                waveforms.append(waveform)
            
            waveform = np.concatenate(waveforms)
            
            # Extract Whisper features if processor available
            features = None
            attention_mask = None
            
            if self.whisper_processor:
                inputs = self.whisper_processor(
                    waveform,
                    sampling_rate=self.audio_sample_rate,
                    return_tensors='pt'
                )
                features = inputs['input_features']
                attention_mask = torch.ones_like(features[..., 0], dtype=torch.bool)
            
            # Encode to RVQ codes
            with torch.no_grad():
                codes = self.audio_tokenizer.encode(
                    torch.tensor(waveform).unsqueeze(0)
                )
            
            # Apply delay pattern if enabled
            if self.use_delay_pattern:
                codes = self._apply_delay_pattern(codes)
            
            # Split into input/output for training
            split_point = codes.shape[-1] // 2
            input_codes = codes[..., :split_point]
            output_codes = codes[..., split_point:]
            
            # Add audio token offset
            input_codes = input_codes + self.audio_token_offset
            output_codes = output_codes + self.audio_token_offset
            
            return {
                'features': features,
                'attention_mask': attention_mask,
                'input_codes': input_codes.squeeze(0),
                'output_codes': output_codes.squeeze(0)
            }
            
        except Exception as e:
            logger.warning(f"Failed to process audio: {e}")
            return None
    
    def _apply_delay_pattern(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Apply delay pattern to RVQ codes for causality
        
        Args:
            codes: RVQ codes of shape (batch, codebooks, length)
            
        Returns:
            Delayed codes
        """
        
        batch_size, num_codebooks, seq_len = codes.shape
        delayed = torch.zeros(
            (batch_size, num_codebooks, seq_len + num_codebooks * self.delay_pattern_steps),
            dtype=codes.dtype
        )
        
        for cb in range(num_codebooks):
            delay = cb * self.delay_pattern_steps
            delayed[:, cb, delay:delay + seq_len] = codes[:, cb, :]
        
        # Trim to original length
        delayed = delayed[..., :seq_len]
        
        return delayed
    
    def _create_text_labels(
        self,
        input_ids: torch.Tensor,
        messages: List[Dict[str, str]]
    ) -> torch.Tensor:
        """
        Create labels for text tokens (mask non-assistant tokens)
        
        Args:
            input_ids: Token IDs
            messages: Original messages
            
        Returns:
            Label tensor with -100 for masked positions
        """
        
        labels = input_ids.clone()
        
        # Find assistant response boundaries
        text = self._build_conversation_text(messages)
        
        # Simple heuristic: mask everything before last assistant tag
        assistant_start = text.rfind('<|im_start|>assistant')
        
        if assistant_start > 0:
            # Tokenize prefix to find masking point
            prefix = text[:assistant_start]
            prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
            mask_length = len(prefix_ids)
            
            # Mask non-assistant tokens
            labels[:mask_length] = -100
        
        return labels
    
    def _create_audio_labels(
        self,
        audio_codes: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """
        Create labels for audio tokens
        
        Args:
            audio_codes: Audio codebook tokens
            
        Returns:
            Audio labels or None
        """
        
        if audio_codes is None:
            return None
        
        # Audio labels are just the codes themselves (teacher forcing)
        return audio_codes.clone()
    
    def _batch_samples(self, samples: List[Dict[str, torch.Tensor]]) -> HiggsAudioBatchInput:
        """
        Batch list of samples with padding
        
        Args:
            samples: List of sample dictionaries
            
        Returns:
            Batched HiggsAudioBatchInput
        """
        
        batch_size = len(samples)
        
        # Find max lengths
        max_text_len = max(s['input_ids'].shape[0] for s in samples)
        max_audio_in_len = 0
        max_audio_out_len = 0
        
        for s in samples:
            if 'audio_in_ids' in s:
                max_audio_in_len = max(max_audio_in_len, s['audio_in_ids'].shape[-1])
            if 'audio_out_ids' in s:
                max_audio_out_len = max(max_audio_out_len, s['audio_out_ids'].shape[-1])
        
        # Pad to multiple if specified
        if self.pad_to_multiple_of:
            max_text_len = ((max_text_len + self.pad_to_multiple_of - 1) // 
                           self.pad_to_multiple_of * self.pad_to_multiple_of)
        
        # Initialize batch tensors
        input_ids = torch.full((batch_size, max_text_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_text_len), dtype=torch.long)
        label_ids = torch.full((batch_size, max_text_len), -100, dtype=torch.long)
        
        audio_features = None
        audio_feature_mask = None
        audio_in_ids = None
        audio_in_ids_start = None
        audio_out_ids = None
        audio_out_ids_start = None
        label_audio_ids = None
        
        # Fill batch tensors
        for i, sample in enumerate(samples):
            seq_len = sample['input_ids'].shape[0]
            input_ids[i, :seq_len] = sample['input_ids']
            attention_mask[i, :seq_len] = sample['attention_mask']
            label_ids[i, :seq_len] = sample['label_ids']
            
            # Handle audio features
            if 'audio_features' in sample and sample['audio_features'] is not None:
                if audio_features is None:
                    feat_shape = sample['audio_features'].shape
                    audio_features = torch.zeros(
                        (batch_size,) + feat_shape[1:],
                        dtype=torch.float32
                    )
                    audio_feature_mask = torch.zeros(
                        (batch_size,) + feat_shape[1:-1],
                        dtype=torch.bool
                    )
                
                audio_features[i] = sample['audio_features']
                audio_feature_mask[i] = sample['audio_feature_attention_mask']
            
            # Handle audio codes
            if 'audio_in_ids' in sample and max_audio_in_len > 0:
                if audio_in_ids is None:
                    audio_in_ids = torch.zeros(
                        (batch_size, self.num_codebooks, max_audio_in_len),
                        dtype=torch.long
                    )
                    audio_in_ids_start = torch.zeros((batch_size,), dtype=torch.long)
                
                codes = sample['audio_in_ids']
                audio_in_ids[i, :, :codes.shape[-1]] = codes
                audio_in_ids_start[i] = sample['audio_in_ids_start'][0]
            
            if 'audio_out_ids' in sample and max_audio_out_len > 0:
                if audio_out_ids is None:
                    audio_out_ids = torch.zeros(
                        (batch_size, self.num_codebooks, max_audio_out_len),
                        dtype=torch.long
                    )
                    audio_out_ids_start = torch.zeros((batch_size,), dtype=torch.long)
                    label_audio_ids = torch.full(
                        (batch_size, self.num_codebooks, max_audio_out_len),
                        -100,
                        dtype=torch.long
                    )
                
                codes = sample['audio_out_ids']
                audio_out_ids[i, :, :codes.shape[-1]] = codes
                audio_out_ids_start[i] = sample['audio_out_ids_start'][0]
                
                if 'label_audio_ids' in sample:
                    labels = sample['label_audio_ids']
                    label_audio_ids[i, :, :labels.shape[-1]] = labels
        
        return HiggsAudioBatchInput(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_features=audio_features,
            audio_feature_attention_mask=audio_feature_mask,
            audio_in_ids=audio_in_ids,
            audio_in_ids_start=audio_in_ids_start,
            audio_out_ids=audio_out_ids,
            audio_out_ids_start=audio_out_ids_start,
            label_ids=label_ids,
            label_audio_ids=label_audio_ids
        )
    
    def _validate_batch(self, batch: HiggsAudioBatchInput) -> None:
        """
        Validate batch tensors for consistency
        
        Args:
            batch: Batch to validate
            
        Raises:
            ValueError: If validation fails
        """
        
        batch_size = batch.input_ids.shape[0]
        
        # Check batch size consistency
        assert batch.attention_mask.shape[0] == batch_size
        assert batch.label_ids.shape[0] == batch_size
        
        # Check sequence length consistency
        seq_len = batch.input_ids.shape[1]
        assert batch.attention_mask.shape[1] == seq_len
        assert batch.label_ids.shape[1] == seq_len
        
        # Check audio tensor consistency
        if batch.audio_in_ids is not None:
            assert batch.audio_in_ids.shape[0] == batch_size
            assert batch.audio_in_ids.shape[1] == self.num_codebooks
            assert batch.audio_in_ids_start.shape[0] == batch_size
        
        if batch.audio_out_ids is not None:
            assert batch.audio_out_ids.shape[0] == batch_size
            assert batch.audio_out_ids.shape[1] == self.num_codebooks
            assert batch.audio_out_ids_start.shape[0] == batch_size
            
            if batch.label_audio_ids is not None:
                assert batch.label_audio_ids.shape == batch.audio_out_ids.shape
        
        # Check value ranges
        assert (batch.input_ids >= 0).all()
        assert (batch.attention_mask >= 0).all()
        assert (batch.attention_mask <= 1).all()
        
        # Check label masking
        valid_labels = batch.label_ids != -100
        if valid_labels.any():
            assert (batch.label_ids[valid_labels] >= 0).all()
        
        logger.debug(f"Batch validation passed: batch_size={batch_size}, seq_len={seq_len}")
