# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The MLA absorb products sglang runs with torch.bmm on ROCm for a bf16 kv_b (Kimi-K3's), from
``srt/models/deepseek_common/attention_forward_methods/forward_mla_rocm.py``: AKLR's ``bmm`` op.

``rocm_absorb_q_bmm`` multiplies each head's no-rope query by its ``w_kc``; ``rocm_absorb_v_bmm`` each head's
attention output by its ``w_vc``, then hands the (tokens, heads, v) product to o_proj, quantized when o_proj is.
The routes take only the bf16 path at unit scale; every quantized weight stays sglang's."""

from __future__ import annotations

import sys

import torch

from aklr.core.runtime.dispatch import FallbackReason
from aklr.ops.gemm.api import bmm
from aklr.plugins.sglang.hook import sglang_route

__all__: list[str] = []

_MODULE = "sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla_rocm"


def _bf16_at_unit_scale(attn) -> bool:
    """sglang's torch.bmm path, with the weight it would multiply by as it is (``_absorb_weight_bf16``)."""
    scale = getattr(attn, "w_scale", None)
    return (
        attn.w_kc.dtype == torch.bfloat16
        and attn.w_vc.dtype == torch.bfloat16
        and isinstance(scale, (int, float))
        and scale == 1.0
    )


@sglang_route(f"{_MODULE}.rocm_absorb_q_bmm", versions=("0.5.22",))
def _rocm_absorb_q_bmm(attn, q_nope, *, is_capture_mode):
    if not _bf16_at_unit_scale(attn):
        return FallbackReason.OUT_OF_CONTRACT
    # w_kc is a (heads, k, n) view of a (heads, n, k) tensor: one contiguous copy, made at the first (eager) call
    w_kc = getattr(attn, "_aklr_w_kc", None)
    if w_kc is None:
        w_kc = attn.w_kc.contiguous()
        attn._aklr_w_kc = w_kc
    return bmm(q_nope.to(torch.bfloat16).transpose(0, 1).contiguous(), w_kc, trans_b=False)


@sglang_route(f"{_MODULE}.rocm_absorb_v_bmm", versions=("0.5.22",))
def _rocm_absorb_v_bmm(attn, attn_output):
    module = sys.modules[_MODULE]
    if not _bf16_at_unit_scale(attn) or module.is_in_tc_piecewise_cuda_graph():
        return FallbackReason.OUT_OF_CONTRACT
    # w_vc is the (heads, k, n) view of a contiguous (heads, n, k) tensor
    product = bmm(attn_output.to(torch.bfloat16).transpose(0, 1).contiguous(), attn.w_vc.transpose(1, 2))
    if isinstance(product, FallbackReason):
        return product
    buf = product.transpose(0, 1).contiguous()  # (tokens, heads, v), sglang's _bmm_buf
    # what sglang hands o_proj from its _bmm_buf, as forward_mla_rocm.rocm_absorb_v_bmm does
    if attn.o_proj.weight.dtype == torch.uint8:
        return module.fused_flatten_mxfp4_quant(buf)
    if module._is_block_scale_fp8(attn.o_proj):
        emit_bpre = module.emit_transposed_bpreshuffle_scale(
            buf.shape[0], on_bpreshuffle_gfx95=module._use_aiter_bpreshuffle_gfx95
        )
        out = module.fused_flatten_fp8_group_quant(
            buf, group_size=128, dtype_quant=torch.float8_e4m3fn, transpose_scale=emit_bpre
        )
        if emit_bpre:
            return module.view_aiter_fused_rms_transposed_fp8_scale_tuple(out)
        if module._use_aiter_bpreshuffle_gfx95:
            return module.materialize_bpreshuffle_fp8_scale_tuple(out)
        return out
    return buf.flatten(1, 2)
