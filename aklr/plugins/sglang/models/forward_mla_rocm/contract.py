# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The MLA absorb steps sglang runs on ROCm for DeepSeek-style models
(srt/models/deepseek_common/attention_forward_methods/forward_mla_rocm.py), in sglang's own layouts: a
scenario is a rank's heads and value head size, Kimi-K3.1-Pro's 12 and 128 at TP8."""

from __future__ import annotations

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    GEMM_BF16_SNR,
    DType,
    FiniteDistribution,
    LecunNormalDistribution,
)
from aklr.core.contract.spec import DTypeSpec, LayoutVariant, LayoutVariantsSpec, TensorSpec, axis
from aklr.plugins.sglang.models.forward_mla_rocm.baseline import MlaVAbsorbBaseline
from aklr.plugins.sglang.models.forward_mla_rocm.reference import MlaVAbsorbReference

__all__ = ["CONTRACTS", "MLA_V_ABSORB"]

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
            error_bounds=GEMM_BF16_SNR,
        )
    },
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
    reference=MlaVAbsorbReference(),
    baseline=MlaVAbsorbBaseline(),
)

CONTRACTS = (MLA_V_ABSORB,)
