# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import math
from abc import abstractmethod
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Any, Literal, NamedTuple

from pydantic import Field, TypeAdapter

from ._dtypes import DType
from ._frozen_model import FrozenModel

if TYPE_CHECKING:
    from torch import Tensor


class CompareResult(NamedTuple):
    is_passed: bool  # no element past its bound, non-finite values matched
    violation_fraction: float  # of the elements where the reference is finite, past their bound
    peak_bound_multiple: float  # max |error| / B
    n_nonfinite_mismatches: int

    @property
    def detail(self) -> str:
        """The measures as one line, for a finding's detail."""
        return (
            f"violation_fraction {self.violation_fraction:.3g}, peak_bound_multiple "
            f"{self.peak_bound_multiple:.3g}, {self.n_nonfinite_mismatches} non-finite mismatches"
        )

    @classmethod
    def of_measures(
        cls,
        a: Tensor,
        y: Tensor,
        *,
        violation_fraction: float,
        peak_bound_multiple: float,
    ) -> CompareResult:
        """A bound's measures of output ``a``, with its non-finite mismatches against the
        reference ``y`` (both float32)."""
        is_finite = y.isfinite()
        is_same = (a == y) | (a.isnan() & y.isnan())
        n_nonfinite_mismatches = int((~is_finite & ~is_same).sum())
        n_nonfinite_mismatches += int((is_finite & ~a.isfinite()).sum())
        return cls(
            is_passed=n_nonfinite_mismatches == 0 and violation_fraction == 0,
            violation_fraction=violation_fraction,
            peak_bound_multiple=peak_bound_multiple,
            n_nonfinite_mismatches=n_nonfinite_mismatches,
        )


def _scaled(x: Tensor) -> tuple[float, Tensor]:
    """(peak, x / peak), peak = max |x|: float32 squares and sums of x / peak cannot overflow.
    A zero or non-finite peak leaves x as it is (peak 1)."""
    peak = x.abs().max().item() if x.numel() else 0.0
    if peak == 0 or not math.isfinite(peak):
        return 1.0, x
    return peak, x / peak


def _norm(x: Tensor) -> float:
    peak, unit = _scaled(x)
    return peak * unit.norm().item()


def _fraction(mask: Tensor) -> float:
    """The fraction of ``mask`` set, as the count over the size: a float32 mean misses the
    exact ratio a threshold compares with."""
    return int(mask.sum()) / mask.numel() if mask.numel() else 0.0


def _byte_rows(tensor: Tensor) -> Tensor:
    import torch

    return tensor.contiguous().reshape(-1).view(torch.uint8).reshape(tensor.numel(), -1)


def run_to_run_fraction(a: Tensor, b: Tensor) -> float:
    """The fraction of elements whose bytes differ between two runs (NaN equals itself)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        raise ValueError(f"runs differ in shape or dtype: {a.shape}, {b.shape}")
    if a.numel() == 0:
        return 0.0
    return _fraction((_byte_rows(a) != _byte_rows(b)).any(-1))


def mutable_tensor_declarations(schema: Mapping[str, Any]) -> dict[str, Any]:
    """The schema json's mutable tensors (``OpContract.mutable_tensors``), by name: its
    outputs, then its mutable inputs."""
    mutable = {n: d for n, d in schema["inputs"].items() if d.get("mutable")}
    return {**schema["outputs"], **mutable}


def comparison_criteria(schema: Mapping[str, Any]) -> dict[str, Any]:
    """What ``compare_outputs`` judges by, from a schema json: its ``element_order`` and each
    mutable tensor's error bound, in ``mutable_tensor_declarations`` order."""
    return {
        "element_order": schema["compare"]["element_order"],
        "outputs": {
            name: {"error_bound": declaration["error_bounds"][0][1]}
            for name, declaration in mutable_tensor_declarations(schema).items()
        },
    }


def compare_outputs(
    criteria: Mapping[str, Any],
    actual_outputs: Mapping[str, Tensor],
    reference_outputs: Mapping[str, Tensor],
) -> dict[str, CompareResult]:
    """Each output of ``criteria`` (``comparison_criteria``) against the reference's, both in the
    output's dtype, on the tensors' device, under its error bound; keyed in that order."""
    element_order = ElementOrder(criteria["element_order"])
    results = {}
    for name, output in criteria["outputs"].items():
        actual, expected = element_order.arrange(actual_outputs[name], reference_outputs[name])
        error_bound = _ERROR_BOUND_ADAPTER.validate_python(output["error_bound"])
        results[name] = error_bound.compare(actual, expected)
    return results


class DeterminismLevel(str, Enum):
    """How the outputs of several runs must agree."""

    BITWISE = "bitwise"
    BOUNDED = "bounded"


def runs_agree(determinism: str, outputs: Sequence[Tensor]) -> bool:
    """``bitwise``: every run's output byte-identical to the first's; ``bounded``:
    always (each run passes compare on its own)."""
    if determinism == "bounded":
        return True
    if determinism != "bitwise":
        raise ValueError(f"unknown determinism {determinism!r}")
    return all(run_to_run_fraction(outputs[0], o) == 0 for o in outputs[1:])


class ErrorBound(FrozenModel):
    """One ``error_bounds`` item of a mutable tensor: how it is compared with the reference."""

    kind: str

    @abstractmethod
    def compare(self, actual: Tensor, expected: Tensor) -> CompareResult:
        """Output ``actual`` against ``expected``, the reference's, both in the output's dtype."""


class ToleranceBound(ErrorBound):
    """B = absolute_tolerance + relative_tolerance * |y|, the largest |actual - y| allowed at
    each element where y, the reference, is finite (both 0: equal)."""

    kind: Literal["tolerance"] = "tolerance"
    absolute_tolerance: Annotated[float, Field(ge=0.0)]
    relative_tolerance: Annotated[float, Field(ge=0.0)]

    def compare(self, actual, expected):
        # integers as integers: float32 holds an int exactly only up to 2**24
        a, y = (t.float() if t.is_floating_point() else t.long() for t in (actual, expected))
        is_finite = y.isfinite()
        error = (a - y)[is_finite].abs()
        limit = self.absolute_tolerance + self.relative_tolerance * y[is_finite].abs()
        violation_fraction = peak_bound_multiple = 0.0
        if error.numel():
            violation_fraction = _fraction(error > limit)
            peak_bound_multiple = (error / limit).masked_fill(error == 0, 0.0).max().item()
        return CompareResult.of_measures(
            a.float(),
            y.float(),
            violation_fraction=violation_fraction,
            peak_bound_multiple=peak_bound_multiple,
        )


class SqnrBound(ErrorBound):
    """The whole output passes when 20*log10(||y|| / ||actual - y||) >= min_sqnr_db (equal
    tensors: inf); no per-element B, so ``violation_fraction`` is 0 or 1 and
    ``peak_bound_multiple`` is the noise norm over the allowed one."""

    kind: Literal["sqnr"] = "sqnr"
    min_sqnr_db: float

    def compare(self, actual, expected):
        import torch

        a, y = actual.float(), expected.float()
        kept = y.isfinite() & a.isfinite()  # a non-finite one is a mismatch
        # both over one peak first: the difference of two large values can pass float32's range
        peak, _ = _scaled(torch.cat((a[kept], y[kept])))
        noise = peak * _norm((a[kept] / peak) - (y[kept] / peak))
        allowed = _norm(y[kept]) * 10 ** (-self.min_sqnr_db / 20)
        if not noise:
            peak_bound_multiple = 0.0
        else:
            # an all-zero reference allows no noise: -inf dB
            peak_bound_multiple = noise / allowed if allowed else math.inf
        return CompareResult.of_measures(
            a,
            y,
            violation_fraction=float(peak_bound_multiple > 1),
            peak_bound_multiple=peak_bound_multiple,
        )


ErrorBoundUnion = Annotated[ToleranceBound | SqnrBound, Field(discriminator="kind")]
_ERROR_BOUND_ADAPTER = TypeAdapter(ErrorBoundUnion)

# The common bounds, Primus-Turbo's (tests/pytorch): an output that needs others writes its own.
# get_tolerances (test_utils.py): atol = rtol, by the output's dtype
_TOLERANCES = {
    DType.FLOAT32: 1e-4,
    DType.BFLOAT16: 1e-2,
    DType.FLOAT16: 1e-2,
    DType.FLOAT8_E4M3FN: 1e-1,
    DType.FLOAT8_E5M2: 1e-1,
}
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


def tolerance(*dtypes: DType) -> ToleranceBound:
    """The common atol and rtol of an output of any of ``dtypes``, which must share them."""
    values = {_TOLERANCES[DType(d)] for d in dtypes}
    if len(values) != 1:
        raise ValueError(f"{dtypes} have no common tolerance")
    (value,) = values
    return ToleranceBound(absolute_tolerance=value, relative_tolerance=value)


class ElementOrder(str, Enum):
    """Which reference element each output element is compared with."""

    FIXED = "fixed"
    ANY_IN_FULL_ROWS = "any_in_full_rows"

    def arrange(self, actual: Tensor, reference_out: Tensor) -> tuple[Tensor, Tensor]:
        """``actual`` and ``reference_out`` arranged so that equal positions are
        compared."""
        match self:
            case ElementOrder.FIXED:
                return actual, reference_out
            case ElementOrder.ANY_IN_FULL_ROWS:
                import torch

                is_full = (reference_out >= 0).all(dim=-1, keepdim=True)
                return (
                    torch.where(is_full, actual.sort(dim=-1).values, actual),
                    torch.where(is_full, reference_out.sort(dim=-1).values, reference_out),
                )
            case _:
                raise ValueError(f"no rule for {self!r}")
