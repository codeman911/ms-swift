"""Higgs-Audio inference utilities for MS-SWIFT using original components.

This module wraps the original Higgs-Audio serve engine from boson_multimodal
for seamless integration with MS-SWIFT inference pipeline.
"""

import os
import sys
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import asyncio

# Add Higgs-Audio path to sys.path for imports
higgs_audio_path = Path(__file__).parent.parent / "higgs-audio"
if str(higgs_audio_path) not in sys.path:
    sys.path.insert(0, str(higgs_audio_path))

# Import original Higgs-Audio components
from boson_multimodal.serve.serve_engine import (
    AsyncHiggsAudioStreamer,
    HiggsAudioStreamerDelta,
)
from boson_multimodal.model.higgs_audio import HiggsAudioModel
from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
from boson_multimodal.data_collator.higgs_audio_collator import HiggsAudioSampleCollator
from boson_multimodal.dataset.chatml_dataset import prepare_chatml_sample, ChatMLSample
from transformers import AutoTokenizer, PreTrainedModel, WhisperProcessor
from swift.utils import get_logger

logger = get_logger()


class HiggsAudioInference:
    """MS-SWIFT wrapper for Higgs-Audio inference using original serve engine.
    
    This class wraps the original HiggsAudioInferenceEngine and AsyncHiggsAudioStreamer
    to provide seamless integration with MS-SWIFT's inference pipeline.
    """
    
    def __init__(
        self,
        model: Union[PreTrainedModel, HiggsAudioModel],
        tokenizer,
        audio_tokenizer=None,
        whisper_processor=None,
        device: Optional[torch.device] = None,
        audio_sample_rate: int = 24000,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        use_delay_pattern: bool = True,
        stream: bool = False,
    ):
        """Initialize the inference wrapper.
        
        Args:
            model: The Higgs-Audio model instance
            tokenizer: Text tokenizer
            audio_tokenizer: Audio RVQ tokenizer
            whisper_processor: Whisper processor for audio features
            device: Device to run inference on
            audio_sample_rate: Sample rate for audio generation
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            top_k: Top-k sampling
            use_delay_pattern: Whether to use delay pattern for RVQ
            stream: Whether to use streaming generation
        """
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Store model and tokenizers
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.audio_tokenizer = audio_tokenizer or load_higgs_audio_tokenizer()
        self.whisper_processor = whisper_processor or WhisperProcessor.from_pretrained(
            "openai/whisper-small"
        )
        
        # Initialize the collator for processing inputs
        self.collator = HiggsAudioSampleCollator(
            tokenizer=tokenizer,
            whisper_processor=self.whisper_processor,
            audio_sample_rate=audio_sample_rate,
            use_delay_pattern=use_delay_pattern,
        )
        
        # Generation parameters
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        
        # Streaming support
        self.stream = stream
        if stream:
            self.streamer = AsyncHiggsAudioStreamer(
                tokenizer=tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )
        
        self.audio_sample_rate = audio_sample_rate
        self.use_delay_pattern = use_delay_pattern
        
        logger.info(
            f"Initialized HiggsAudioInference on {self.device} with "
            f"temperature={temperature}, top_p={top_p}, top_k={top_k}, stream={stream}"
        )

    def generate(
        self,
        text: str,
        reference_audio: Optional[Union[str, np.ndarray]] = None,
        scene_description: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        stream: bool = False,
    ) -> Union[Dict[str, Any], AsyncHiggsAudioStreamer]:
        """
        Generate speech from text using original Higgs-Audio model.
        
        Args:
            text: Text to synthesize
            reference_audio: Reference audio for voice cloning (path or waveform)
            scene_description: Scene description for prompt conditioning
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling probability
            do_sample: Whether to use sampling
            stream: Whether to stream the generation
        
        Returns:
            Dictionary with generated audio or streamer for async generation
        """
        # Use provided parameters or defaults
        max_new_tokens = max_new_tokens or self.max_new_tokens
        temperature = temperature or self.temperature
        top_k = top_k or self.top_k
        top_p = top_p or self.top_p
        
        # Prepare the input prompt
        messages = []
        if scene_description:
            messages.append({"role": "system", "content": f"Scene: {scene_description}"})
        messages.append({"role": "user", "content": text})
        
        # Prepare ChatML sample
        sample = ChatMLSample(
            conversations=messages,
            audio_paths=[reference_audio] if reference_audio else None,
        )
        
        # Process sample through collator
        dataset_sample = prepare_chatml_sample(
            sample=sample,
            tokenizer=self.tokenizer,
            audio_tokenizer=self.audio_tokenizer,
            whisper_processor=self.whisper_processor,
        )
        
        # Collate into batch
        batch = self.collator([dataset_sample])
        
        # Move batch to device
        batch = self._move_to_device(batch)
        
        # Setup generation kwargs
        generation_kwargs = {
            "input_ids": batch.input_ids,
            "attention_mask": batch.attention_mask,
            "audio_features": batch.audio_features,
            "audio_feature_attention_mask": batch.audio_feature_attention_mask,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "do_sample": do_sample,
        }
        
        # Add streamer if streaming
        if stream and self.streamer:
            generation_kwargs["streamer"] = self.streamer
            return self.streamer
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(**generation_kwargs)
        
        # Post-process outputs
        return self._post_process_outputs(outputs, batch)
    
    def _move_to_device(self, batch: Any) -> Any:
        """Move batch tensors to the target device."""
        if hasattr(batch, '__dataclass_fields__'):
            # Handle dataclass
            for field in batch.__dataclass_fields__:
                value = getattr(batch, field)
                if isinstance(value, torch.Tensor):
                    setattr(batch, field, value.to(self.device))
        return batch
    
    def _post_process_outputs(
        self,
        outputs: torch.Tensor,
        batch: Any,
    ) -> Dict[str, Any]:
        """Post-process model outputs to extract audio and text."""
        # Extract generated tokens
        generated_ids = outputs[0] if isinstance(outputs, tuple) else outputs
        
        # Decode text tokens
        text_tokens = []
        audio_tokens = []
        
        # Split tokens into text and audio
        for token_id in generated_ids[0].tolist():
            if token_id >= 128000:  # Audio token range
                audio_tokens.append(token_id - 128000)
            else:
                text_tokens.append(token_id)
        
        # Decode text
        generated_text = self.tokenizer.decode(text_tokens, skip_special_tokens=True)
        
        # Convert audio tokens to waveform if available
        audio_waveform = None
        if audio_tokens and self.audio_tokenizer:
            audio_codes = torch.tensor(audio_tokens).unsqueeze(0)
            audio_waveform = self.audio_tokenizer.decode(audio_codes)
            if isinstance(audio_waveform, torch.Tensor):
                audio_waveform = audio_waveform.cpu().numpy()
        
        return {
            "text": generated_text,
            "audio": audio_waveform,
            "sample_rate": self.audio_sample_rate,
            "text_tokens": text_tokens,
            "audio_tokens": audio_tokens,
        }
    
    def voice_clone(
        self,
        text: str,
        reference_audio: Union[str, np.ndarray],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform zero-shot voice cloning using Higgs-Audio.
        
        Args:
            text: Text to synthesize in the cloned voice
            reference_audio: Reference audio for voice cloning
            **kwargs: Additional generation parameters
        
        Returns:
            Dictionary with cloned audio
        """
        return self.generate(
            text=text,
            reference_audio=reference_audio,
            **kwargs
        )
            gen_config: Generation config
            
        Yields:
            Partial generation results
        """
        
        # Create streamer
        from transformers import TextIteratorStreamer
        
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_special_tokens=True,
            skip_prompt=True
        )
        
        gen_config['streamer'] = streamer
        
        # Start generation in background
        import threading
        
        def generate():
            with torch.no_grad():
                self.model.generate(**inputs, **gen_config)
        
        thread = threading.Thread(target=generate)
        thread.start()
        
        # Stream results
        for text_chunk in streamer:
            yield {'text': text_chunk, 'audio': None}
        
        thread.join()
    
    def _decode_audio(self, audio_codes: torch.Tensor) -> np.ndarray:
        """
        Decode audio codes to waveform
        
        Args:
            audio_codes: RVQ codes tensor
            
        Returns:
            Audio waveform
        """
        
        if audio_codes is None:
            return None
        
        # Remove delay pattern if present
        audio_codes = self._revert_delay_pattern(audio_codes)
        
        # Decode with audio tokenizer
        with torch.no_grad():
            waveform = self.audio_tokenizer.decode(audio_codes)
        
        # Convert to numpy
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.cpu().numpy()
        
        # Ensure 1D
        if waveform.ndim > 1:
            waveform = waveform.squeeze()
        
        return waveform
    
    def _revert_delay_pattern(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Revert delay pattern from RVQ codes
        
        Args:
            codes: Delayed codes
            
        Returns:
            Original codes
        """
        
        # Simple version - assumes no delay or handles it internally
        return codes
    
    def synthesize_multi_speaker(
        self,
        dialogue: List[Dict[str, str]],
        speaker_voices: Dict[str, Union[str, np.ndarray]],
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Synthesize multi-speaker dialogue
        
        Args:
            dialogue: List of {'speaker': name, 'text': text} dicts
            speaker_voices: Mapping of speaker names to reference audio
            output_path: Optional path to save output
            
        Returns:
            Combined audio and metadata
        """
        
        segments = []
        
        for turn in dialogue:
            speaker = turn['speaker']
            text = turn['text']
            
            # Get reference audio for speaker
            reference = speaker_voices.get(speaker)
            
            # Generate speech
            result = self.generate(
                text=text,
                reference_audio=reference,
                scene_description="Multi-speaker conversation"
            )
            
            if result['audio'] is not None:
                segments.append(result['audio'])
        
        # Concatenate segments
        if segments:
            combined = np.concatenate(segments)
            
            # Save if path provided
            if output_path:
                sf.write(output_path, combined, 24000)
                logger.info(f"Saved multi-speaker audio to {output_path}")
            
            return {
                'audio': combined,
                'sample_rate': 24000,
                'num_speakers': len(speaker_voices),
                'num_segments': len(segments)
            }
        
        return None
