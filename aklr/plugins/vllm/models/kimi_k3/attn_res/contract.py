# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3's attention residual as vLLM runs it on ROCm (``models/kimi_k3/amd/ops/attn_res.py``).

Each decoder layer reads its input as a softmax-weighted mix of the residual bank's written
rows and the running prefix, before attention and before the MLP, and the model once more
before the final norm. vLLM folds the pending layer output into the prefix in place, writes
the prefix as a fresh bank row every 12th layer, normalizes the mix for the layer (not at
the end), and scores each row with the product of two bf16 weights (the score norm's and
the score projection's), exact in float32."""

from __future__ import annotations

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    DTYPE_TRAITS,
    DType,
    FiniteDistribution,
    LecunNormalDistribution,
    ToleranceBound,
)
from aklr.core.contract.spec import (
    DTypeSpec,
    LayoutVariant,
    LayoutVariantsSpec,
    ScalarSpec,
    TensorSpec,
    attr,
    axis,
)
from aklr.plugins.vllm.models.kimi_k3.attn_res.baseline import KimiK3AttnResBaseline, KimiK3AttnResFp8Baseline
from aklr.plugins.vllm.models.kimi_k3.attn_res.reference import (
    KimiK3AttnResFp8Reference,
    KimiK3AttnResReference,
)

__all__ = ["CONTRACTS", "FP8_MAX", "KIMI_K3_ATTN_RES", "KIMI_K3_ATTN_RES_FP8"]

_FLOAT32_LARGEST = DTYPE_TRAITS["float32"].largest_finite_value
_BF16 = DTypeSpec(any_of=(DType.BFLOAT16,))
_SAME_AS_PREFIX = DTypeSpec(same_as_tensor="prefix")
_ROWS = ("n_tokens", "hidden_size")
_EXACT = ToleranceBound(absolute_tolerance=0, relative_tolerance=0)
# any finite RMS weight vLLM computes with, zero and negative ones too
_NORM_WEIGHT_DISTRIBUTION = FiniteDistribution()

KIMI_K3_ATTN_RES = OpContract(
    contract_id="kimi_k3_attn_res@1.0",
    description=(
        "p = prefix, or bf16(prefix + addend) with an addend, then stored back to prefix; "
        "rows = bank[:, :n_valid_rows] and p; each row r scores (r . (score_norm_weight * "
        "score_proj_weight)) * rsqrt(mean(r**2) + score_eps); the softmax of the scores "
        "weights the rows into mix; out = mix * rsqrt(mean(mix**2) + norm_eps) * norm_weight "
        "with a norm_weight, else mix; the means over the hidden axis. With write_bank, "
        "bank[:, n_valid_rows] = p. Bank rows from n_valid_rows on, but the one written, are "
        "unwritten memory in vLLM: a kernel reads none of them."
    ),
    op_type="attn_res",
    axes={
        # one workgroup per token: vLLM's batch sizes are matched by power-of-two range
        "n_tokens": axis("Tokens.", 1, 65536, is_var=True),
        "hidden_size": axis("Width of the residual stream.", 256, 8192, multiple_of=256),
        "n_bank_rows": axis("Rows the bank holds per token.", 1, 16),
    },
    attrs={
        "n_valid_rows": attr(
            "Bank rows written so far, all of which the mix reads (0 on the first layer, "
            "whose mix is the prefix alone).",
            8,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            default=8,
            param_name="n_valid_rows",
            abbrev="rows",
        ),
        "write_bank": attr(
            "Whether p also becomes bank row n_valid_rows.",
            False,
            True,
            default=False,
            param_name="write_bank",
            abbrev="write",
        ),
        "has_addend": attr("Whether an addend is passed.", True, False, default=True, abbrev="add"),
        "has_norm": attr("Whether norm_weight is passed.", True, False, default=True, abbrev="norm"),
    },
    constraints=(
        "n_valid_rows <= n_bank_rows",
        "(not write_bank) or n_valid_rows < n_bank_rows",
    ),
    inputs={
        "prefix": TensorSpec(
            description=(
                "The running prefix: this block's layer outputs so far; p is stored back to it "
                "with an addend."
            ),
            shape=_ROWS,
            dtype=_BF16,
            distribution=FiniteDistribution(),
            mutable=True,
            error_bounds=_EXACT,
        ),
        "addend": TensorSpec(
            description="The pending layer output, not yet in the prefix.",
            shape=_ROWS,
            # an optional tensor names its dtypes
            dtype=_BF16,
            present_when="has_addend",
            distribution=FiniteDistribution(),
        ),
        "bank": TensorSpec(
            description="Each token's bank: the embedding, then the sum of each block of layers' outputs.",
            shape=("n_tokens", "n_bank_rows", "hidden_size"),
            dtype=_SAME_AS_PREFIX,
            distribution=FiniteDistribution(),
            mutable=True,
            error_bounds=_EXACT,
        ),
        "score_norm_weight": TensorSpec(
            description="The score's RMSNorm weight.",
            shape=("hidden_size",),
            dtype=_SAME_AS_PREFIX,
            distribution=_NORM_WEIGHT_DISTRIBUTION,
            lifetime="weight",
        ),
        "score_proj_weight": TensorSpec(
            description="The score projection's weight (its one output row).",
            shape=("hidden_size",),
            dtype=_SAME_AS_PREFIX,
            # scores about N(0, 1) on N(0, 1) rows
            distribution=LecunNormalDistribution(fan_in="hidden_size"),
            lifetime="weight",
        ),
        "norm_weight": TensorSpec(
            description="The layer norm's weight, scaling each normalized column.",
            shape=("hidden_size",),
            dtype=_BF16,
            present_when="has_norm",
            distribution=_NORM_WEIGHT_DISTRIBUTION,
            lifetime="weight",
        ),
        "score_eps": ScalarSpec(
            description="Added to the mean square of each scored row.",
            dtype=DType.FLOAT32,
            interval=(2.0**-149, _FLOAT32_LARGEST),
            default=1e-5,
        ),
        "norm_eps": ScalarSpec(
            description="Added to the mean square of the mix; read with a norm_weight only.",
            dtype=DType.FLOAT32,
            interval=(0.0, _FLOAT32_LARGEST),
            default=1e-5,
        ),
    },
    outputs={
        "out": TensorSpec(
            description="The normalized mix scaled by norm_weight, or the mix without one.",
            shape=_ROWS,
            dtype=_SAME_AS_PREFIX.model_copy(update={"param_name": "out_dtype"}),
            # the mix cancels: an element near 0 carries the float32 rounding of its row's
            # terms, about 1e-6 at N(0, 1) rows, which a bound relative to the element alone
            # rejects
            error_bounds=ToleranceBound(absolute_tolerance=2**-12, relative_tolerance=2**-7),
        ),
    },
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
    reference=KimiK3AttnResReference(),
    baseline=KimiK3AttnResBaseline(),
)

FP8_MAX = 448.0  # float8_e4m3fn's largest finite value, vLLM's get_fp8_min_max() on gfx950


def _with_fp8_out(contract: OpContract) -> OpContract:
    """The contract whose out is vLLM's per-token fp8 one, for the linear after it (vLLM's quant_dtype)."""
    fields = {name: getattr(contract, name) for name in type(contract).model_fields}
    fields.update(
        contract_id="kimi_k3_attn_res_fp8@1.0",
        description=contract.description
        + " out is then quantized per token for the fp8 linear after it: o = out rounded to bf16, "
        "scale = max(max |o|, 1e-10) * (1 / 448) in float32, and quantized = float8_e4m3fn(clamp(o * (1 / scale), "
        "-448, 448)), the reciprocal and the product each rounded to float32.",
        outputs={
            "quantized": TensorSpec(
                description="The bf16 out of kimi_k3_attn_res@1.0 quantized per token to float8_e4m3fn.",
                shape=_ROWS,
                dtype=DTypeSpec(any_of=(DType.FLOAT8_E4M3FN,)),
                # an element off by one fp8 step where its bf16 out (last bit free under kimi_k3_attn_res@1.0) or its
                # row's scale crosses an fp8 rounding boundary: a step is 2**-3 relative for normals, 2**-9 below them;
                # plus one fp8 rounding (2**-3), the output's own, which the compare no longer adds
                error_bounds=ToleranceBound(absolute_tolerance=2**-9, relative_tolerance=2**-3 + 2**-3),
            ),
            "scale": TensorSpec(
                description="Each token's dequantization scale.",
                shape=("n_tokens", 1),
                dtype=DTypeSpec(any_of=(DType.FLOAT32,)),
                # its row's largest |o|, one bf16 rounding of a float32 mix: off by a bf16 step where that ties
                error_bounds=ToleranceBound(absolute_tolerance=0, relative_tolerance=2**-7),
            ),
        },
        reference=KimiK3AttnResFp8Reference(),
        baseline=KimiK3AttnResFp8Baseline(),
    )
    return OpContract(**fields)


KIMI_K3_ATTN_RES_FP8 = _with_fp8_out(KIMI_K3_ATTN_RES)

CONTRACTS = (KIMI_K3_ATTN_RES, KIMI_K3_ATTN_RES_FP8)
