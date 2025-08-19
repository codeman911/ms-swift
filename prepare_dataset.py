"""
Dataset Preparation Script for Higgs-Audio Training
Converts various audio-text formats to ChatML JSONL format
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
import pandas as pd
import os

def create_chattml_sample(
    text: str,
    audio_path: str,
    reference_audio: Optional[str] = None,
    scene_description: str = "Indoor quiet",
    speaker_id: Optional[str] = None
) -> Dict:
    """
    Create a ChatML format sample for Higgs-Audio training
    
    Args:
        text: Text transcript
        audio_path: Path to target audio file
        reference_audio: Optional reference audio for voice cloning
        scene_description: Scene description
        speaker_id: Optional speaker identifier
        
    Returns:
        ChatML formatted dictionary
    """
    
    # Build system message
    system_content = f"<|scene_desc_start|> {scene_description} <|scene_desc_end|>"
    if speaker_id:
        system_content += f"\nSpeaker: {speaker_id}"
    
    # Build user message
    user_content = f"Please speak the following text: {text}"
    if reference_audio:
        user_content = f"<|audio|> {user_content}"
    
    # Create ChatML structure
    sample = {
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": ""}  # Empty for training
        ],
        "audios": [audio_path]
    }
    
    if reference_audio:
        sample["reference_audio"] = reference_audio
        sample["audios"].insert(0, reference_audio)
    
    return sample


def process_ljspeech(data_dir: Path, output_file: Path, max_samples: Optional[int] = None):
    """
    Process LJSpeech dataset to ChatML format
    
    Args:
        data_dir: LJSpeech dataset directory
        output_file: Output JSONL file
        max_samples: Maximum samples to process
    """
    
    metadata_file = data_dir / "metadata.csv"
    wavs_dir = data_dir / "wavs"
    
    # Read metadata
    df = pd.read_csv(metadata_file, sep="|", header=None, 
                     names=["filename", "text", "normalized_text"])
    
    if max_samples:
        df = df.head(max_samples)
    
    samples = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing LJSpeech"):
        audio_path = str(wavs_dir / f"{row['filename']}.wav")
        
        if Path(audio_path).exists():
            sample = create_chattml_sample(
                text=row['normalized_text'],
                audio_path=audio_path,
                scene_description="Studio recording",
                speaker_id="ljspeech"
            )
            samples.append(sample)
    
    # Write to JSONL
    with open(output_file, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"Processed {len(samples)} samples to {output_file}")


def process_vctk(data_dir: Path, output_file: Path, max_speakers: int = 10):
    """
    Process VCTK dataset for multi-speaker training
    
    Args:
        data_dir: VCTK dataset directory
        output_file: Output JSONL file
        max_speakers: Maximum number of speakers to include
    """
    
    txt_dir = data_dir / "txt"
    wav_dir = data_dir / "wav48_silence_trimmed"
    
    samples = []
    speakers = sorted(list(txt_dir.glob("*")))[:max_speakers]
    
    for speaker_dir in tqdm(speakers, desc="Processing VCTK speakers"):
        speaker_id = speaker_dir.name
        
        # Use first audio as reference for this speaker
        wav_files = list((wav_dir / speaker_id).glob("*.flac"))
        if not wav_files:
            continue
            
        reference_audio = str(wav_files[0])
        
        # Process all utterances
        for txt_file in (txt_dir / speaker_id).glob("*.txt"):
            utterance_id = txt_file.stem
            audio_path = wav_dir / speaker_id / f"{utterance_id}.flac"
            
            if audio_path.exists():
                with open(txt_file, 'r') as f:
                    text = f.read().strip()
                
                sample = create_chattml_sample(
                    text=text,
                    audio_path=str(audio_path),
                    reference_audio=reference_audio,
                    scene_description="Studio recording",
                    speaker_id=f"vctk_{speaker_id}"
                )
                samples.append(sample)
    
    # Write to JSONL
    with open(output_file, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"Processed {len(samples)} samples from {max_speakers} speakers to {output_file}")


def process_custom_dataset(
    csv_file: Path,
    audio_dir: Path,
    output_file: Path,
    text_column: str = "text",
    audio_column: str = "audio_file",
    speaker_column: Optional[str] = "speaker"
):
    """
    Process custom CSV dataset to ChatML format
    
    Args:
        csv_file: CSV file with text and audio file mappings
        audio_dir: Directory containing audio files
        output_file: Output JSONL file
        text_column: Column name for text
        audio_column: Column name for audio file names
        speaker_column: Optional column for speaker IDs
    """
    
    df = pd.read_csv(csv_file)
    
    # Group by speaker for reference audio
    speaker_refs = {}
    if speaker_column and speaker_column in df.columns:
        for speaker in df[speaker_column].unique():
            speaker_data = df[df[speaker_column] == speaker]
            if len(speaker_data) > 0:
                ref_audio = audio_dir / speaker_data.iloc[0][audio_column]
                if ref_audio.exists():
                    speaker_refs[speaker] = str(ref_audio)
    
    samples = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing custom dataset"):
        audio_path = audio_dir / row[audio_column]
        
        if audio_path.exists():
            speaker = row.get(speaker_column) if speaker_column else None
            reference = speaker_refs.get(speaker) if speaker else None
            
            sample = create_chattml_sample(
                text=row[text_column],
                audio_path=str(audio_path),
                reference_audio=reference,
                speaker_id=speaker
            )
            samples.append(sample)
    
    # Write to JSONL
    with open(output_file, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"Processed {len(samples)} samples to {output_file}")


def split_dataset(
    input_file: Path,
    train_ratio: float = 0.9,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 42
):
    """
    Split JSONL dataset into train/val/test sets
    
    Args:
        input_file: Input JSONL file
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed
    """
    
    # Read all samples
    samples = []
    with open(input_file, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    
    # Shuffle
    np.random.seed(seed)
    np.random.shuffle(samples)
    
    # Split
    n = len(samples)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_samples = samples[:train_end]
    val_samples = samples[train_end:val_end]
    test_samples = samples[val_end:]
    
    # Write splits
    base_path = input_file.parent
    base_name = input_file.stem
    
    for split_name, split_samples in [
        ("train", train_samples),
        ("validation", val_samples),
        ("test", test_samples)
    ]:
        output_file = base_path / f"{base_name}_{split_name}.jsonl"
        with open(output_file, 'w') as f:
            for sample in split_samples:
                f.write(json.dumps(sample) + '\n')
        print(f"Wrote {len(split_samples)} samples to {output_file}")


def augment_audio(
    audio_path: str,
    output_dir: Path,
    augmentations: List[str] = ["speed", "pitch", "noise"]
) -> List[str]:
    """
    Apply audio augmentations for data diversity
    
    Args:
        audio_path: Input audio file
        output_dir: Directory to save augmented files
        augmentations: List of augmentation types
        
    Returns:
        List of augmented audio file paths
    """
    
    # Load audio
    y, sr = librosa.load(audio_path, sr=24000)
    base_name = Path(audio_path).stem
    augmented_paths = []
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Speed perturbation
    if "speed" in augmentations:
        for speed_factor in [0.9, 1.1]:
            y_speed = librosa.effects.time_stretch(y, rate=speed_factor)
            output_path = output_dir / f"{base_name}_speed_{speed_factor}.wav"
            sf.write(output_path, y_speed, sr)
            augmented_paths.append(str(output_path))
    
    # Pitch shift
    if "pitch" in augmentations:
        for n_steps in [-2, 2]:
            y_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
            output_path = output_dir / f"{base_name}_pitch_{n_steps}.wav"
            sf.write(output_path, y_pitch, sr)
            augmented_paths.append(str(output_path))
    
    # Add noise
    if "noise" in augmentations:
        noise = np.random.normal(0, 0.005, len(y))
        y_noise = y + noise
        output_path = output_dir / f"{base_name}_noise.wav"
        sf.write(output_path, y_noise, sr)
        augmented_paths.append(str(output_path))
    
    return augmented_paths


def validate_dataset(jsonl_file: Path):
    """
    Validate JSONL dataset for training
    
    Args:
        jsonl_file: JSONL file to validate
    """
    
    print(f"Validating {jsonl_file}")
    
    valid_samples = 0
    invalid_samples = 0
    missing_audio = 0
    
    with open(jsonl_file, 'r') as f:
        for i, line in enumerate(f):
            try:
                sample = json.loads(line)
                
                # Check required fields
                assert "messages" in sample
                assert "audios" in sample
                assert len(sample["messages"]) >= 2
                
                # Check audio files exist
                for audio_path in sample["audios"]:
                    if not Path(audio_path).exists():
                        missing_audio += 1
                        print(f"Missing audio: {audio_path}")
                
                valid_samples += 1
                
            except Exception as e:
                invalid_samples += 1
                print(f"Invalid sample at line {i+1}: {e}")
    
    print(f"\nValidation Results:")
    print(f"  Valid samples: {valid_samples}")
    print(f"  Invalid samples: {invalid_samples}")
    print(f"  Missing audio files: {missing_audio}")
    
    return valid_samples > 0 and invalid_samples == 0 and missing_audio == 0


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for Higgs-Audio training")
    parser.add_argument("--dataset_type", type=str, required=True,
                       choices=["ljspeech", "vctk", "custom"],
                       help="Type of dataset to process")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Input dataset directory")
    parser.add_argument("--output_file", type=str, required=True,
                       help="Output JSONL file")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum samples to process")
    parser.add_argument("--split", action="store_true",
                       help="Split into train/val/test sets")
    parser.add_argument("--validate", action="store_true",
                       help="Validate the output dataset")
    parser.add_argument("--augment", action="store_true",
                       help="Apply data augmentation")
    
    # Custom dataset args
    parser.add_argument("--csv_file", type=str,
                       help="CSV file for custom dataset")
    parser.add_argument("--text_column", type=str, default="text",
                       help="Text column name")
    parser.add_argument("--audio_column", type=str, default="audio_file",
                       help="Audio file column name")
    parser.add_argument("--speaker_column", type=str, default="speaker",
                       help="Speaker ID column name")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)
    
    # Process dataset
    if args.dataset_type == "ljspeech":
        process_ljspeech(input_dir, output_file, args.max_samples)
    elif args.dataset_type == "vctk":
        process_vctk(input_dir, output_file, max_speakers=10)
    elif args.dataset_type == "custom":
        if not args.csv_file:
            raise ValueError("--csv_file required for custom dataset")
        process_custom_dataset(
            Path(args.csv_file),
            input_dir,
            output_file,
            args.text_column,
            args.audio_column,
            args.speaker_column
        )
    
    # Split if requested
    if args.split:
        split_dataset(output_file)
    
    # Validate
    if args.validate:
        if validate_dataset(output_file):
            print("✅ Dataset validation passed!")
        else:
            print("❌ Dataset validation failed!")


if __name__ == "__main__":
    main()
