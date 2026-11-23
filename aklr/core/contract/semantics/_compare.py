# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import math
from abc import abstractmethod
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal, NamedTuple

from pydantic import Field, TypeAdapter

from ._dtypes import DTYPE_TRAITS, DType
from ._frozen_model import FrozenModel

if TYPE_CHECKING:
    from torch import Tensor


class CompareResult(NamedTuple):
    is_passed: bool  # violations within the limit, non-finite values matched
    is_bound_exceeded: bool  # violation_fraction over max_violation_fraction
    violation_fraction: float  # of the elements where the rounded reference is finite
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
        rounded: Tensor,
        *,
        is_bound_exceeded: bool,
        violation_fraction: float,
        peak_bound_multiple: float,
    ) -> CompareResult:
        """A bound's measures of output ``a``, with its non-finite mismatches against
        ``rounded`` (both float32)."""
        is_finite = rounded.isfinite()
        is_same = (a == rounded) | (a.isnan() & rounded.isnan())
        n_nonfinite_mismatches = int((~is_finite & ~is_same).sum())
        n_nonfinite_mismatches += int((is_finite & ~a.isfinite()).sum())
        return cls(
            is_passed=n_nonfinite_mismatches == 0 and not is_bound_exceeded,
            is_bound_exceeded=is_bound_exceeded,
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
    """What ``compare_outputs`` judges by, from a schema json: its ``compare`` settings and each
    mutable tensor's dtype and error bound (none for an integral one), in
    ``mutable_tensor_declarations`` order."""
    compare = schema["compare"]
    return {
        "element_order": compare["element_order"],
        "max_violation_fraction": compare["max_violation_fraction"],
        "outputs": {
            name: {
                "dtype": declaration["dtype"]["any_of"][0],
                "error_bound": declaration["error_bounds"][0][1] if declaration["error_bounds"] else None,
            }
            for name, declaration in mutable_tensor_declarations(schema).items()
        },
    }


def compare_outputs(
    criteria: Mapping[str, Any],
    actual_outputs: Mapping[str, Tensor],
    reference_outputs: Mapping[str, Tensor],
) -> dict[str, CompareResult]:
    """Each output of ``criteria`` (``comparison_criteria``) against its reference, on the
    tensors' device, under its error bound (an integer one must be exact); keyed in that order."""
    element_order = ElementOrder(criteria["element_order"])
    max_violation_fraction = criteria["max_violation_fraction"]
    results = {}
    for name, output in criteria["outputs"].items():
        out_dtype = output["dtype"]
        actual, reference_out = element_order.arrange(actual_outputs[name], reference_outputs[name])
        if DTYPE_TRAITS[out_dtype].is_integral:
            # integral unit_roundoff is 0: B = 0, i.e. exact
            assert DTYPE_TRAITS[out_dtype].unit_roundoff == 0
            results[name] = _exact_compare(
                actual,
                reference_out.to(actual.dtype),
                max_violation_fraction,
            )
            continue
        a = actual.float()
        rounded = reference_out.to(actual.dtype).float()
        error_bound = _ERROR_BOUND_ADAPTER.validate_python(output["error_bound"])
        results[name] = error_bound.compare(
            a,
            rounded,
            reference_out,
            max_violation_fraction=max_violation_fraction,
            out_unit_roundoff=DTYPE_TRAITS[out_dtype].unit_roundoff,
        )
    return results


def _exact_compare(actual: Tensor, rounded: Tensor, max_violation_fraction: float) -> CompareResult:
    """An integral output against its reference in the same dtype, element by element as
    integers (float32 holds an int exactly only up to 2**24)."""
    differs = actual != rounded
    violation_fraction = _fraction(differs)
    return CompareResult.of_measures(
        actual.float(),
        rounded.float(),
        is_bound_exceeded=violation_fraction > max_violation_fraction,
        violation_fraction=violation_fraction,
        peak_bound_multiple=math.inf if bool(differs.any()) else 0.0,
    )


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
    """One ``error_bounds`` item of a floating output: how it is compared with y, the
    reference."""

    kind: str
    expr_fields: ClassVar[tuple[str, ...]] = ()  # expression fields, in order

    @abstractmethod
    def compare(
        self,
        a: Tensor,
        rounded: Tensor,
        reference_out: Tensor,
        *,
        max_violation_fraction: float,
        out_unit_roundoff: float,
    ) -> CompareResult:
        """Output ``a`` against y (``reference_out``) and y rounded to the output dtype
        (``rounded``), all three in float32."""


class ElementwiseBound(ErrorBound):
    """B, the largest |actual - y| allowed per element: at most
    ``max_violation_fraction`` of the finite elements exceed it."""

    @abstractmethod
    def allowed_error(self, reference_out: Tensor, out_unit_roundoff: float) -> Tensor:
        """B at each element of the one output compared."""

    def compare(
        self,
        a,
        rounded,
        reference_out,
        *,
        max_violation_fraction,
        out_unit_roundoff,
    ):
        is_finite = rounded.isfinite()
        error = (a - reference_out)[is_finite]
        limit = self.allowed_error(reference_out, out_unit_roundoff)[is_finite]
        violation_fraction = peak_bound_multiple = 0.0
        if error.numel():
            violation_fraction = _fraction(error.abs() > limit)
            multiples = (error.abs() / limit).masked_fill(error == 0, 0.0)
            peak_bound_multiple = multiples.max().item()
        return CompareResult.of_measures(
            a,
            rounded,
            is_bound_exceeded=violation_fraction > max_violation_fraction,
            violation_fraction=violation_fraction,
            peak_bound_multiple=peak_bound_multiple,
        )


class ToleranceBound(ElementwiseBound):
    """B = absolute_tolerance + (relative_tolerance + out_unit_roundoff) * |y|."""

    kind: Literal["tolerance"] = "tolerance"
    absolute_tolerance: Annotated[float, Field(ge=0.0)]
    relative_tolerance: Annotated[float, Field(ge=0.0)]

    def allowed_error(self, reference_out, out_unit_roundoff):
        relative = self.relative_tolerance + out_unit_roundoff
        return self.absolute_tolerance + relative * reference_out.abs()


class SqnrBound(ErrorBound):
    """The whole output passes when 20*log10(||y_r|| / ||actual - y_r||) >= min_sqnr_db, y_r
    the reference rounded to the output dtype (equal tensors: inf); no per-element B, so
    ``violation_fraction`` is 0 or 1 and ``peak_bound_multiple`` is the noise norm over the
    allowed one."""

    kind: Literal["sqnr"] = "sqnr"
    min_sqnr_db: float

    def compare(
        self,
        a,
        rounded,
        reference_out,
        *,
        max_violation_fraction,
        out_unit_roundoff,
    ):
        import torch

        kept = rounded.isfinite() & a.isfinite()  # a non-finite one is a mismatch
        # both over one peak first: the difference of two large values can pass float32's range
        peak, _ = _scaled(torch.cat((a[kept], rounded[kept])))
        noise = peak * _norm((a[kept] / peak) - (rounded[kept] / peak))
        allowed = _norm(rounded[kept]) * 10 ** (-self.min_sqnr_db / 20)
        if not noise:
            peak_bound_multiple = 0.0
        else:
            # an all-zero reference allows no noise: main's -inf dB
            peak_bound_multiple = noise / allowed if allowed else math.inf
        is_bound_exceeded = peak_bound_multiple > 1
        return CompareResult.of_measures(
            a,
            rounded,
            is_bound_exceeded=is_bound_exceeded,
            violation_fraction=float(is_bound_exceeded),
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
