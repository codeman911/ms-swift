"""Higgs-Audio template registration for MS-SWIFT."""

from swift.utils import get_logger
from ..register import TemplateMeta, register_template

logger = get_logger()

# Create template meta for Higgs-Audio ChatML format
higgs_chatml_template = TemplateMeta(
    prefix=['<|im_start|>system\n{{SYSTEM}}<|im_end|>\n'],
    prompt=['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'],
    chat_sep=['\n'],
    suffix=['<|im_end|>'],
    default_system='You are a helpful assistant capable of generating speech in the voice of the provided reference audio.',
    system_prefix=['<|im_start|>system\n'],
    system_suffix=['<|im_end|>\n'],
    auto_add_bos=True,
    stop_words=['<|im_end|>', '<|im_start|>'],
    infer_media_type='round',
)

# Register the template
register_template(
    template_type='higgs-chatml',
    template_meta=higgs_chatml_template,
)

logger.info("Registered Higgs-Audio template: higgs-chatml")
