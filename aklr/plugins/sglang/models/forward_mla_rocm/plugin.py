# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The MLA absorb steps sglang runs on ROCm for DeepSeek-style models
(srt/models/deepseek_common/attention_forward_methods/forward_mla_rocm.py), in sglang's own layouts: a
scenario is a rank's heads and value head size, Kimi-K3.1-Pro's 12 and 128 at TP8."""

from __future__ import annotations

import sys

import torch

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    DotProductBound,
    DType,
    FiniteDistribution,
    LecunNormalDistribution,
)
from aklr.core.contract.spec import DTypeSpec, LayoutVariant, LayoutVariantsSpec, TensorSpec, axis
from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins import Plugin

__all__ = ["MLA_V_ABSORB", "declare", "family", "mla_v_absorb"]


_BF16 = DTypeSpec(any_of=(DType.BFLOAT16,))

# Not bmm@1.0: its row-major (batch, m, k) operands would take two transposing copies of sglang's (tokens,
# heads, ...) tensors, and its trans_b baseline is aiter's, not the torch.bmm sglang runs here.
MLA_V_ABSORB = OpContract(
    contract_id="mla_v_absorb@1.0",
    description=(
        "sglang's MLA value absorption on bf16 weights at unit scale (rocm_absorb_v_bmm's torch.bmm): "
        "per token and head, attn_output . w_vc^T, the latent attention output times the head's "
        "kv_b_proj value rows, with sglang's (tokens, heads, ...) layouts in and out"
    ),
    op_type="forward_mla_rocm",
    axes={
        "n_tokens": axis("Tokens in this call.", 1, 65536, is_var=True),
        "n_heads": axis("Query heads on this rank.", 1, 128),
        "v_head_dim": axis("Value head size: w_vc's rows per head.", 32, 512, multiple_of=32),
    },
    inputs={
        "attn_output": TensorSpec(
            description="The attention output per head: 512 latent values.",
            shape=("n_tokens", "n_heads", 512),
            dtype=_BF16,
            distribution=FiniteDistribution(),
        ),
        "w_vc": TensorSpec(
            description=(
                "Per head, kv_b_proj's value rows: v_head_dim rows of 512 latent values, the storage "
                "whose (heads, 512, v_head_dim) view sglang holds as w_vc."
            ),
            shape=("n_heads", "v_head_dim", 512),
            dtype=_BF16,
            distribution=LecunNormalDistribution(fan_in=512),
            lifetime="weight",
        ),
    },
    outputs={
        "output": TensorSpec(
            description="Per token and head, attn_output . w_vc^T: o_proj's input before its flatten.",
            shape=("n_tokens", "n_heads", "v_head_dim"),
            dtype=_BF16,
            # bmm@1.0's bound
            error_bounds=DotProductBound(length=512, accumulate_dtype=DType.FLOAT32, max_split_k=16),
        )
    },
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
)


family = Plugin("sglang")
family.define(MLA_V_ABSORB)
family.build()


_MODULE = "sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla_rocm"


@family.op
def mla_v_absorb(attn_output, w_vc, output=None): ...


# Served in CUDA graph capture and eagerly alike: a capture records only a Point an eager call warmed.
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


def declare(plugin: Plugin) -> None:
    plugin.wrap(
        f"{_MODULE}.rocm_absorb_v_bmm",
        versions=("0.5.24.dev809",),
        replace=_rocm_absorb_v_bmm,
    )
