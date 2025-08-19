"""
Custom template for Higgs-Audio training.
"""

from swift.llm.utils.template import ChatMLTemplate, TemplateType
from .collator import HiggsAudioDataCollator

class HiggsAudioTemplate(ChatMLTemplate):
    """
    A custom template for Higgs-Audio that uses the HiggsAudioDataCollator.
    """
    def __init__(self):
        super().__init__(
            template_type=TemplateType.higgs_audio_template,
            data_collator_class=HiggsAudioDataCollator,
        )
