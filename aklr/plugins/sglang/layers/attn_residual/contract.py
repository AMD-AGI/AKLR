# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3's attention-residual aggregation (sglang ``srt/layers/attn_residual.py``).

Each layer reads its input as a softmax-weighted mix of the residual bank's written rows
and the running prefix, then normalizes the mix, twice per layer (before attention and
before the MLP), and at the end before the final norm."""

from __future__ import annotations

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    DTYPE_TRAITS,
    DType,
    FiniteDistribution,
    LecunNormalDistribution,
    LogUniformDistribution,
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

__all__ = ["ADD_ATTN_RES", "ATTN_RES", "CONTRACTS"]

_FLOAT32_LARGEST = DTYPE_TRAITS["float32"].largest_finite_value
_BF16 = DTypeSpec(any_of=(DType.BFLOAT16,))
_SAME_AS_PREFIX = DTypeSpec(same_as_tensor="prefix")
_ROWS = ("n_tokens", "hidden_size")
_EXACT = ToleranceBound(absolute_tolerance=0, relative_tolerance=0)
_MATH = (
    "rows = bank[:, :n_valid_rows] and the prefix p; each row r scores (r . score_weight) * "
    "rsqrt(mean(r**2) + score_eps); the softmax of the scores weights the rows into mix; "
    "normalized = mix * rsqrt(mean(mix**2) + norm_eps) * norm_weight, the means over the hidden "
    "axis. With write_bank, bank[:, n_valid_rows] = p. Bank rows from n_valid_rows on are unwritten "
    "memory in sglang: a kernel reads none of them."
)
_AXES = {
    "n_tokens": axis("Tokens.", 1, 65536, is_var=True),
    # sglang's kernel keeps the scored rows of a token in registers: at most 8 x 8192
    "hidden_size": axis("Width of the residual stream.", 256, 8192, multiple_of=256),
    "n_bank_rows": axis("Rows the bank holds per token.", 2, 16),
}
_ATTRS = {
    "n_valid_rows": attr(
        "Bank rows written so far, all of which the mix reads.",
        8,
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
        "Whether the prefix also becomes bank row n_valid_rows.",
        False,
        True,
        default=False,
        param_name="write_bank",
        abbrev="write",
    ),
}
_CONSTRAINTS = ("n_valid_rows <= n_bank_rows", "(not write_bank) or n_valid_rows < n_bank_rows")
_BANK = TensorSpec(
    description="Each token's bank: the embedding, then the sum of each block of layers' outputs.",
    shape=("n_tokens", "n_bank_rows", "hidden_size"),
    dtype=_SAME_AS_PREFIX,
    distribution=FiniteDistribution(),
    mutable=True,
    error_bounds=_EXACT,
)
_WEIGHTS = {
    "score_weight": TensorSpec(
        description="The score projection's weight times its RMSNorm's weight.",
        shape=("hidden_size",),
        dtype=DTypeSpec(any_of=(DType.FLOAT32,)),
        # scores about N(0, 1) on N(0, 1) rows
        distribution=LecunNormalDistribution(fan_in="hidden_size"),
        lifetime="weight",
    ),
    "norm_weight": TensorSpec(
        description="The layer norm's weight, scaling each normalized column.",
        shape=("hidden_size",),
        dtype=_SAME_AS_PREFIX,
        # about [0.5, 1.5]
        distribution=LogUniformDistribution(log2_lo="-1", log2_hi="0.58"),
        lifetime="weight",
    ),
}
_EPSILONS = {
    name: ScalarSpec(
        description=f"Added to the mean square of {what}.",
        dtype=DType.FLOAT32,
        interval=(2.0**-149, _FLOAT32_LARGEST),
        default=1e-6,
    )
    for name, what in (("score_eps", "each scored row"), ("norm_eps", "the mix"))
}
_NORMALIZED = TensorSpec(
    description="The normalized mix, scaled by norm_weight.",
    shape=_ROWS,
    dtype=_SAME_AS_PREFIX,
    # the mix cancels: an element near 0 carries the float32 rounding of its row's terms, about
    # 1e-6 at N(0, 1) rows, which a bound relative to the element alone rejects
    error_bounds=ToleranceBound(absolute_tolerance=2**-12, relative_tolerance=0),
)
_LAYOUTS = LayoutVariantsSpec(by_name={"row_major": LayoutVariant()})

ATTN_RES = OpContract(
    contract_id="attn_res@1.0",
    description=f"p = prefix; {_MATH}",
    op_type="attn_residual",
    axes=_AXES,
    attrs=_ATTRS,
    constraints=_CONSTRAINTS,
    inputs={
        "prefix": TensorSpec(
            description="The running prefix: this block's layer outputs so far.",
            shape=_ROWS,
            dtype=_BF16,
            distribution=FiniteDistribution(),
        ),
        "bank": _BANK,
        **_WEIGHTS,
        **_EPSILONS,
    },
    outputs={"normalized": _NORMALIZED},
    layout_variants=_LAYOUTS,
)

ADD_ATTN_RES = OpContract(
    contract_id="add_attn_res@1.0",
    description=f"p = prefix + addend, rounded to their dtype before it is scored; prefix_out = p; {_MATH}",
    op_type="attn_residual",
    axes=_AXES,
    attrs=_ATTRS,
    constraints=_CONSTRAINTS,
    inputs={
        "prefix": TensorSpec(
            description="The running prefix before the pending layer output.",
            shape=_ROWS,
            dtype=_BF16,
            distribution=FiniteDistribution(),
        ),
        "addend": TensorSpec(
            description="The pending layer output.",
            shape=_ROWS,
            dtype=_SAME_AS_PREFIX,
            distribution=FiniteDistribution(),
        ),
        "bank": _BANK,
        **_WEIGHTS,
        **_EPSILONS,
    },
    outputs={
        "normalized": _NORMALIZED,
        "prefix_out": TensorSpec(
            description="prefix + addend, which the next layer adds on to.",
            shape=_ROWS,
            dtype=_SAME_AS_PREFIX,
            error_bounds=_EXACT,
        ),
    },
    layout_variants=_LAYOUTS,
)

CONTRACTS = (ADD_ATTN_RES, ATTN_RES)
