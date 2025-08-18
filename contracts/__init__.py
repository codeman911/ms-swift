"""Validation contracts for Higgs Audio.

This package provides wrapper classes that enforce strict input contracts
(shape, dtype, device, value constraints) for the Higgs Audio model and data collator.

Classes:
- ValidatingHiggsAudioModel: wraps HiggsAudioModel with pre/post-forward validation.
- ValidatingHiggsAudioSampleCollator: wraps HiggsAudioSampleCollator with batch validation.

Utilities:
- validators: common assertion helpers used by the wrappers.
"""

from .validating_model import ValidatingHiggsAudioModel
from .validating_collator import ValidatingHiggsAudioSampleCollator
