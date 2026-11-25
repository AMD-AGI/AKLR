# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""How a tensor the kernel writes is compared with the reference: its error bound, and the common
bounds contracts choose from."""

from __future__ import annotations

import math
from abc import abstractmethod
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import Field

from ._dtypes import DType
from ._frozen_model import FrozenModel

if TYPE_CHECKING:
    from torch import Tensor


class ErrorBound(FrozenModel):
    """One ``error_bounds`` item: how a tensor the kernel writes is compared with the reference."""

    kind: str

    @abstractmethod
    def compare(self, actual: Tensor, expected: Tensor) -> tuple[bool, str]:
        """Whether ``actual`` is within the bound of ``expected``, the reference's, both in the
        tensor's dtype, and the measures that decide it, in one line."""


class ToleranceBound(ErrorBound):
    """Every element within B = absolute_tolerance + relative_tolerance * |y| of y, the reference
    (both 0: equal); where y is not finite, the same non-finite value: torch's ``isclose`` with
    ``equal_nan``."""

    kind: Literal["tolerance"] = "tolerance"
    absolute_tolerance: Annotated[float, Field(ge=0.0)]
    relative_tolerance: Annotated[float, Field(ge=0.0)]

    def compare(self, actual, expected):
        import torch

        # integers as integers: float32 holds an int exactly only up to 2**24
        a, y = (t.float() if t.is_floating_point() else t.long() for t in (actual, expected))
        atol, rtol = self.absolute_tolerance, self.relative_tolerance
        is_close = torch.isclose(a, y, rtol=rtol, atol=atol, equal_nan=True)
        finite = a.isfinite() & y.isfinite()
        error = (a - y)[finite].abs()
        multiples = (error / (atol + rtol * y[finite].abs())).masked_fill(error == 0, 0.0)
        # a count over the size: a float32 mean misses the exact fraction
        violation_fraction = int((~is_close).sum()) / is_close.numel() if is_close.numel() else 0.0
        peak_bound_multiple = multiples.max().item() if multiples.numel() else 0.0
        return bool(is_close.all()), (
            f"violation_fraction {violation_fraction:.3g}, peak_bound_multiple "
            f"{peak_bound_multiple:.3g}, {int((~is_close & ~finite).sum())} non-finite mismatches"
        )


class SqnrBound(ErrorBound):
    """The whole tensor at an SQNR of at least min_sqnr_db: 20 * log10(||y|| / ||actual - y||), y
    the reference, over the elements where both are finite (equal: inf; an all-zero y allows no
    noise); where y is not finite, the same non-finite value."""

    kind: Literal["sqnr"] = "sqnr"
    min_sqnr_db: float

    def compare(self, actual, expected):
        # float64: a difference or a square of float32 values can pass float32's range
        a, y = actual.double(), expected.double()
        kept = a.isfinite() & y.isfinite()
        noise, signal = (a - y)[kept].norm().item(), y[kept].norm().item()
        # elsewhere both must hold the same value, NaN matching NaN
        mismatches = int((~kept & (a != y) & ~(a.isnan() & y.isnan())).sum())
        is_within = noise <= signal * 10 ** (-self.min_sqnr_db / 20) and not mismatches
        if not noise:
            sqnr_db = math.inf
        elif not signal:
            sqnr_db = -math.inf
        else:
            sqnr_db = 20 * math.log10(signal / noise)
        return (
            is_within,
            f"sqnr {sqnr_db:.3g} dB (min {self.min_sqnr_db:g} dB), {mismatches} non-finite mismatches",
        )


ErrorBoundUnion = Annotated[ToleranceBound | SqnrBound, Field(discriminator="kind")]


# The common bounds, Primus-Turbo's (tests/pytorch): an output that needs others writes its own.
# get_tolerances (test_utils.py): atol = rtol, by the output's dtype
_TOLERANCES = {
    DType.FLOAT32: 1e-4,
    DType.BFLOAT16: 1e-2,
    DType.FLOAT16: 1e-2,
    DType.FLOAT8_E4M3FN: 1e-1,
    DType.FLOAT8_E5M2: 1e-1,
}


def tolerance(*dtypes: DType) -> ToleranceBound:
    """The common atol and rtol of an output of any of ``dtypes``, which must share them."""
    values = {_TOLERANCES[DType(d)] for d in dtypes}
    if len(values) != 1:
        raise ValueError(f"{dtypes} have no common tolerance")
    (value,) = values
    return ToleranceBound(absolute_tolerance=value, relative_tolerance=value)


# the SNR floors of its op tests: test_grouped_gemm, test_gemm_fp8, test_gemm_fp4, test_attention,
# test_fused_mega_moe_fp8 and test_grouped_mlp_fp4
GEMM_BF16_SNR = SqnrBound(min_sqnr_db=45.0)
GEMM_FP16_SNR = SqnrBound(min_sqnr_db=50.0)
GEMM_FP8_E4M3_SNR = SqnrBound(min_sqnr_db=25.0)
GEMM_FP8_E5M2_SNR = SqnrBound(min_sqnr_db=20.0)
GEMM_FP4_SNR = SqnrBound(min_sqnr_db=10.0)
ATTENTION_FP8_SNR = SqnrBound(min_sqnr_db=20.0)
MOE_MXFP8_SNR = SqnrBound(min_sqnr_db=20.0)
MOE_MXFP4_SNR = SqnrBound(min_sqnr_db=10.0)
