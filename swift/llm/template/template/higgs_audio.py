"""Higgs-Audio template registration for MS-SWIFT."""

from swift.utils import get_logger
from ..register import TemplateMeta, register_template

logger = get_logger()

# Create template meta for Higgs-Audio ChatML format
higgs_chatml_template = TemplateMeta(
    template_type='higgs-chatml',
    prefix=[],
    prompt=['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'],
    chat_sep=['<|im_end|>\n'],
    suffix=['<|im_end|>'],
    system_prefix=['<|im_start|>system\n{{SYSTEM}}<|im_end|>\n'],
    default_system='You are a helpful assistant capable of generating speech in the voice of the provided reference audio.',
    auto_add_bos=True,
    stop_words=['<|im_end|>', '<|im_start|>'],
)

# Register the template - only pass template_meta
register_template(higgs_chatml_template)

logger.info("Registered Higgs-Audio template: higgs-chatml")
