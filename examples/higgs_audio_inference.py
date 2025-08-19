#!/usr/bin/env python
"""
Example script for Higgs-Audio inference and voice cloning
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from swift.llm import get_model_tokenizer
from plugins.inference import HiggsAudioGenerator
from plugins.register import register_higgs_audio_all

def main():
    # Register Higgs-Audio components
    print("Registering Higgs-Audio components...")
    register_higgs_audio_all()
    
    # Model configuration
    model_type = "higgs-audio-v2-3b"
    model_path = "bosonai/higgs-audio-v2-generation-3B-base"
    
    # Load model and tokenizer
    print(f"Loading model: {model_type}")
    model, tokenizer = get_model_tokenizer(
        model_type=model_type,
        model_id_or_path=model_path,
        dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Initialize generator
    generator = HiggsAudioGenerator(
        model=model,
        tokenizer=tokenizer,
        audio_tokenizer=None,  # Will be loaded from model
        use_flash_attention=True
    )
    
    # Example 1: Basic TTS
    print("\n" + "="*60)
    print("Example 1: Basic Text-to-Speech")
    print("="*60)
    
    text = "Hello, this is a test of the Higgs Audio text to speech system."
    result = generator.generate(
        text=text,
        scene_description="Indoor quiet",
        temperature=0.8,
        top_k=50,
        top_p=0.95
    )
    
    print(f"Generated text: {result['text']}")
    if result['audio'] is not None:
        # Save audio
        import soundfile as sf
        output_path = "output_basic_tts.wav"
        sf.write(output_path, result['audio'], result['sample_rate'])
        print(f"Audio saved to: {output_path}")
    
    # Example 2: Voice Cloning
    print("\n" + "="*60)
    print("Example 2: Zero-shot Voice Cloning")
    print("="*60)
    
    reference_audio = "path/to/reference_speaker.wav"  # Replace with actual path
    text = "This speech is generated using voice cloning from a reference audio sample."
    
    if Path(reference_audio).exists():
        result = generator.generate(
            text=text,
            reference_audio=reference_audio,
            scene_description="Studio recording",
            temperature=0.7,
            use_ras=True  # Use Repetition-Aware Sampling
        )
        
        print(f"Generated text: {result['text']}")
        if result['audio'] is not None:
            output_path = "output_voice_clone.wav"
            sf.write(output_path, result['audio'], result['sample_rate'])
            print(f"Cloned audio saved to: {output_path}")
    else:
        print(f"Reference audio not found: {reference_audio}")
    
    # Example 3: Multi-speaker Dialogue
    print("\n" + "="*60)
    print("Example 3: Multi-speaker Dialogue")
    print("="*60)
    
    dialogue = [
        {"speaker": "Alice", "text": "Hello Bob, how are you today?"},
        {"speaker": "Bob", "text": "I'm doing great, thanks for asking!"},
        {"speaker": "Alice", "text": "That's wonderful to hear."},
        {"speaker": "Bob", "text": "How about you? How has your day been?"}
    ]
    
    # Define speaker voices (using different reference audios)
    speaker_voices = {
        "Alice": "path/to/alice_reference.wav",  # Replace with actual paths
        "Bob": "path/to/bob_reference.wav"
    }
    
    # Check if reference files exist
    all_exist = all(Path(v).exists() for v in speaker_voices.values())
    
    if all_exist:
        result = generator.synthesize_multi_speaker(
            dialogue=dialogue,
            speaker_voices=speaker_voices,
            output_path="output_dialogue.wav"
        )
        
        if result:
            print(f"Generated dialogue with {result['num_speakers']} speakers")
            print(f"Total segments: {result['num_segments']}")
    else:
        print("Multi-speaker synthesis skipped - reference audios not found")
    
    # Example 4: Streaming Generation
    print("\n" + "="*60)
    print("Example 4: Streaming TTS Generation")
    print("="*60)
    
    text = "This is a longer text that will be generated in streaming mode, " \
           "allowing for real-time synthesis and playback."
    
    print("Streaming text generation:")
    for chunk in generator.generate(
        text=text,
        scene_description="Podcast recording",
        stream=True
    ):
        if chunk['text']:
            print(chunk['text'], end='', flush=True)
    print()  # New line after streaming
    
    print("\n" + "="*60)
    print("Inference examples completed!")
    print("="*60)


if __name__ == "__main__":
    main()
