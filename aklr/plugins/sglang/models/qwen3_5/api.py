# Created by OpenAI

"""Compatibility exports from this family's plugin module."""

from .plugin import declare as declare
from .plugin import (
    gated_attention_decode,
    gated_attention_decode_group_fp8_quant,
    gated_attention_prefill,
    gated_attention_prefill_group_fp8_quant,
    gdn_decode,
    gdn_decode_group_fp8_quant,
    gdn_prefill,
    gdn_prefill_group_fp8_quant,
)

__all__ = [
    "gated_attention_decode",
    "gated_attention_decode_group_fp8_quant",
    "gated_attention_prefill",
    "gated_attention_prefill_group_fp8_quant",
    "gdn_decode",
    "gdn_decode_group_fp8_quant",
    "gdn_prefill",
    "gdn_prefill_group_fp8_quant",
]
