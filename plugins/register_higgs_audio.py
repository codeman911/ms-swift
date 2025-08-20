"""Main registration script for Higgs-Audio integration with MS-SWIFT.

This module ties together all components for end-to-end training and inference:
- Model registration
- Template registration  
- Dataset registration
- Custom data collator
- Loss plugin
"""

import os
import sys
from pathlib import Path

# Add current directory to path for imports
plugin_dir = Path(__file__).parent
if str(plugin_dir) not in sys.path:
    sys.path.insert(0, str(plugin_dir))

# Import MS-SWIFT utilities
from swift.utils import get_logger

# Import all registration functions
from model_registration import register_higgs_audio_models
from template_registration import register_higgs_audio_template
from dataset_registration import register_higgs_audio_dataset
from loss_plugin import register_higgs_audio_loss_plugin

logger = get_logger()


def register_all_higgs_audio_components():
    """Register all Higgs-Audio components with MS-SWIFT.
    
    This function registers:
    1. Model and tokenizer loading functions
    2. Custom multi-modal chat template
    3. Dataset preprocessing and loading
    4. Training loss plugin
    
    Call this function before training to enable Higgs-Audio support.
    """
    
    logger.info("=" * 80)
    logger.info("Starting Higgs-Audio MS-SWIFT Integration")
    logger.info("=" * 80)
    
    try:
        # 1. Register model
        logger.info("📦 Registering Higgs-Audio model...")
        register_higgs_audio_models()
        logger.info("✅ Model registration complete")
        
        # 2. Register template
        logger.info("📝 Registering custom chat template...")
        register_higgs_audio_template()
        logger.info("✅ Template registration complete")
        
        # 3. Register dataset
        logger.info("📊 Registering dataset loaders...")
        register_higgs_audio_dataset()
        logger.info("✅ Dataset registration complete")
        
        # 4. Register loss plugin
        logger.info("📈 Registering loss plugin...")
        register_higgs_audio_loss_plugin()
        logger.info("✅ Loss plugin registration complete")
        
        # Log configuration
        logger.info("\n" + "=" * 80)
        logger.info("🎉 Higgs-Audio Integration Complete!")
        logger.info("=" * 80)
        
        logger.info("\n📋 Configuration Summary:")
        logger.info("  - Model Type: higgs-audio-full")
        logger.info("  - Template: higgs-chatml")
        logger.info("  - Dataset: higgs-audio-voice-cloning")
        logger.info("  - Loss Plugin: higgs-audio-loss")
        
        # Check environment variables
        text_weight = os.getenv('HIGGS_TEXT_WEIGHT', '1.0')
        audio_weight = os.getenv('HIGGS_AUDIO_WEIGHT', '1.0')
        
        logger.info("\n⚙️ Loss Weights (from environment):")
        logger.info(f"  - Text Weight: {text_weight}")
        logger.info(f"  - Audio Weight: {audio_weight}")
        
        logger.info("\n📝 Usage Instructions:")
        logger.info("1. Prepare your dataset in JSONL format following the schema")
        logger.info("2. Set environment variables for loss weights if needed:")
        logger.info("   export HIGGS_TEXT_WEIGHT=1.0")
        logger.info("   export HIGGS_AUDIO_WEIGHT=1.0")
        logger.info("3. Run training with MS-SWIFT CLI:")
        logger.info("   swift sft \\")
        logger.info("     --model_type higgs-audio-full \\")
        logger.info("     --dataset higgs-audio-voice-cloning \\")
        logger.info("     --dataset_path /path/to/your/data.jsonl \\")
        logger.info("     --output_dir ./output \\")
        logger.info("     --num_train_epochs 3 \\")
        logger.info("     --batch_size 2 \\")
        logger.info("     --gradient_accumulation_steps 4 \\")
        logger.info("     --learning_rate 1e-5 \\")
        logger.info("     --warmup_ratio 0.1 \\")
        logger.info("     --eval_steps 100 \\")
        logger.info("     --save_steps 500 \\")
        logger.info("     --logging_steps 10")
        
        logger.info("\n✨ All components registered successfully!")
        logger.info("=" * 80 + "\n")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to register Higgs-Audio components: {e}")
        logger.error("Please check the error messages above for details")
        raise


def verify_registration():
    """Verify that all components are properly registered.
    
    Returns:
        bool: True if all components are registered correctly
    """
    
    from swift.llm.model.register import MODEL_MAPPING
    from swift.llm.dataset.register import DATASET_MAPPING
    from swift.llm.template.register import TEMPLATE_MAPPING
    
    verification_passed = True
    
    logger.info("\n🔍 Verifying component registration...")
    
    # Check model registration
    if 'higgs-audio-full' in MODEL_MAPPING:
        logger.info("✅ Model 'higgs-audio-full' is registered")
    else:
        logger.error("❌ Model 'higgs-audio-full' is NOT registered")
        verification_passed = False
    
    # Check template registration
    if 'higgs-chatml' in TEMPLATE_MAPPING:
        logger.info("✅ Template 'higgs-chatml' is registered")
    else:
        logger.error("❌ Template 'higgs-chatml' is NOT registered")
        verification_passed = False
    
    # Check dataset registration
    if 'higgs-audio-voice-cloning' in DATASET_MAPPING:
        logger.info("✅ Dataset 'higgs-audio-voice-cloning' is registered")
    else:
        logger.error("❌ Dataset 'higgs-audio-voice-cloning' is NOT registered")
        verification_passed = False
    
    if verification_passed:
        logger.info("\n✅ All components verified successfully!")
    else:
        logger.error("\n❌ Some components failed verification")
    
    return verification_passed


def main():
    """Main entry point for registration."""
    
    # Register all components
    success = register_all_higgs_audio_components()
    
    if success:
        # Verify registration
        verified = verify_registration()
        
        if verified:
            logger.info("\n🚀 Higgs-Audio is ready for training with MS-SWIFT!")
            return 0
        else:
            logger.error("\n⚠️ Registration verification failed")
            return 1
    else:
        logger.error("\n❌ Registration failed")
        return 1


if __name__ == "__main__":
    exit(main())
