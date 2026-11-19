# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import sys

import torch

from aklr.core.runtime.dispatch import FallbackReason, api
from aklr.plugins.sglang.hook import sglang_route
from aklr.plugins.sglang.models.forward_mla_rocm.contract import MLA_V_ABSORB

__all__ = ["mla_v_absorb"]

_MODULE = "sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla_rocm"


@api(MLA_V_ABSORB)
def mla_v_absorb(attn_output, w_vc, output=None): ...


# Served in CUDA graph capture and eagerly alike: a capture records only a Point an eager call warmed.
@sglang_route(f"{_MODULE}.rocm_absorb_v_bmm", versions=("0.5.24.dev809",))
def _rocm_absorb_v_bmm(attn, attn_output):
    """rocm_absorb_v_bmm's buffered torch.bmm branch as one op call, then the function's own hand-off of the
    (tokens, heads, v) product to o_proj; every other branch stays sglang's."""
    module = sys.modules[_MODULE]
    if _bmm_falls_back(module, attn, attn_output):
        return FallbackReason.OUT_OF_CONTRACT
    # w_vc is the (heads, 512, v) view of a (heads, v, 512) tensor, which the op takes; it declines any other
    product = mla_v_absorb(attn_output=attn_output, w_vc=attn.w_vc.transpose(1, 2))
    if isinstance(product, FallbackReason):
        return product
    return _o_proj_input(module, attn, product)


def _bmm_falls_back(module, attn, attn_output) -> bool:
    """Whether rocm_absorb_v_bmm takes a branch before its buffered torch.bmm, or multiplies by other than
    w_vc as it is (_absorb_weight_bf16 of a bf16 w_vc at unit scale)."""
    w_vc, w_scale = attn.w_vc, getattr(attn, "w_scale", None)
    if (
        (module._use_aiter_gfx95 and (w_vc.dtype == torch.uint8 or attn.w_kc.dtype == torch.float8_e4m3fn))
        or module.is_in_tc_piecewise_cuda_graph()
        or w_vc.dtype != torch.bfloat16
        or not (isinstance(w_scale, (int, float)) and w_scale == 1.0)
        or (
            module._use_gfx1250_absorb_bmm
            and module._gfx1250_absorb_bmm_ok(x=attn_output, w=w_vc, w_scale=w_scale)
        )
    ):
        return True
    if not module.get_exec().kernel.enable_amd_monokernel:
        return False
    # the monokernel's triton v absorb, for a contiguous w_vc
    from sglang.kernels.ops.attention.dsa import mla_v_absorb as triton_v_absorb

    return triton_v_absorb.covered(attn_output, w_vc)


def _o_proj_input(module, attn, buf):
    """What rocm_absorb_v_bmm hands o_proj from its (tokens, heads, v) _bmm_buf: quantized as a quantized
    o_proj takes it, else flattened."""
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
