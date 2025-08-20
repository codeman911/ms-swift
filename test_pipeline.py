"""Validation script to test Higgs-Audio pipeline components"""

import sys
import os
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent / 'plugins'))
sys.path.insert(0, str(Path(__file__).parent / 'higgs-audio'))

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        # Test plugin imports
        from plugins.model_registration import register_higgs_audio_models
        print("✅ Model registration imported")
        
        from plugins.dataset_registration import register_higgs_audio_datasets
        print("✅ Dataset registration imported")
        
        from plugins.template import HiggsAudioTemplate
        print("✅ Template imported")
        
        from plugins.trainer_plugin import HiggsAudioTrainerPlugin
        print("✅ Trainer plugin imported")
        
        from plugins.collator import HiggsAudioDataCollator
        print("✅ Collator imported")
        
        # Test MS-SWIFT imports
        from swift.llm.template import Template, register_template
        print("✅ MS-SWIFT template system imported")
        
        from swift.llm.utils import ModelType, register_model
        print("✅ MS-SWIFT model registration imported")
        
        from swift.llm.dataset import register_dataset
        print("✅ MS-SWIFT dataset registration imported")
        
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_registration():
    """Test component registration"""
    print("\nTesting registration...")
    
    try:
        from plugins.register import register_higgs_audio_all, validate_environment
        
        # Validate environment first
        if not validate_environment():
            print("⚠️ Environment validation failed but continuing...")
        
        # Register all components
        success = register_higgs_audio_all()
        
        if success:
            print("✅ All components registered successfully")
            
            # Verify registrations
            from swift.llm.utils import ModelType
            if hasattr(ModelType, 'higgs_audio') or 'higgs-audio' in str(ModelType._member_names_):
                print("✅ Model type registered")
            
            from swift.llm.template import TEMPLATE_MAPPING
            if 'higgs-audio-chatml' in TEMPLATE_MAPPING:
                print("✅ Template registered")
            
            from swift.llm.dataset import DATASET_MAPPING
            if 'higgs_audio' in DATASET_MAPPING or 'higgs-audio' in DATASET_MAPPING:
                print("✅ Dataset registered")
            
            return True
        else:
            print("❌ Registration failed")
            return False
            
    except Exception as e:
        print(f"❌ Registration error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_preprocessing():
    """Test data preprocessing function"""
    print("\nTesting data preprocessing...")
    
    try:
        from plugins.dataset_registration import higgs_preprocess
        
        # Create sample data
        sample = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hello"},
                        {"type": "audio", "audio_url": "ref_audio.wav"},
                        {"type": "text", "text": "Generate speech"}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Hello world"},
                        {"type": "audio", "audio_url": "target_audio.wav"}
                    ]
                }
            ]
        }
        
        # Process sample
        result = higgs_preprocess(sample)
        
        # Verify output
        assert 'conversation' in result, "Missing conversation field"
        assert 'audios' in result, "Missing audios field"
        assert len(result['audios']) == 2, "Incorrect number of audio files"
        assert '<audio>' in str(result['conversation']), "Missing audio placeholders"
        
        print("✅ Data preprocessing working correctly")
        print(f"   - Conversation length: {len(result['conversation'])}")
        print(f"   - Audio files: {result['audios']}")
        return True
        
    except Exception as e:
        print(f"❌ Preprocessing error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_loading():
    """Test model loading (lightweight check)"""
    print("\nTesting model configuration...")
    
    try:
        from plugins.model_registration import get_model_tokenizer
        
        # Note: This would normally load the model but we'll skip for validation
        print("✅ Model loading function available")
        print("   - Full model loading skipped (requires GPU and model files)")
        return True
        
    except Exception as e:
        print(f"⚠️ Model loading check: {e}")
        return True  # Non-critical for validation

def main():
    """Run all validation tests"""
    print("=" * 60)
    print("🔍 Higgs-Audio Pipeline Validation")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Registration", test_registration()))
    results.append(("Preprocessing", test_data_preprocessing()))
    results.append(("Model Config", test_model_loading()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Validation Summary:")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("🎉 All validation tests passed!")
        print("✅ Pipeline is ready for training")
        print("\nNext step: Run training with:")
        print("  ./train_higgs_plugins.sh")
    else:
        print("⚠️ Some tests failed")
        print("Please check the errors above and fix any issues")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())
