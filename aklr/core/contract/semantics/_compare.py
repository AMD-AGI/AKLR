# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import math
from abc import abstractmethod
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal, NamedTuple

from pydantic import Field, PositiveInt, TypeAdapter

from ._dtypes import DTYPE_TRAITS, DType
from ._frozen_model import FrozenModel

if TYPE_CHECKING:
    from torch import Tensor


class CompareResult(NamedTuple):
    is_passed: bool  # violations within the limit, non-finite values matched, unbiased
    is_bound_exceeded: bool  # violation_fraction over max_violation_fraction
    violation_fraction: float  # of the elements where the rounded reference is finite
    peak_bound_multiple: float  # max |error| / B
    bias_z_score: float  # |mean(error)| / rms(error)
    n_nonfinite_mismatches: int

    @property
    def detail(self) -> str:
        """The measures as one line, for a finding's detail."""
        return (
            f"violation_fraction {self.violation_fraction:.3g}, peak_bound_multiple "
            f"{self.peak_bound_multiple:.3g}, bias_z_score {self.bias_z_score:.3g}, "
            f"{self.n_nonfinite_mismatches} non-finite mismatches"
        )

    @classmethod
    def of_measures(
        cls,
        a: Tensor,
        rounded: Tensor,
        *,
        is_bound_exceeded: bool,
        is_unbiased: bool,
        violation_fraction: float,
        peak_bound_multiple: float,
        bias_z_score: float,
    ) -> CompareResult:
        """A bound's measures of output ``a``, with its non-finite mismatches against
        ``rounded`` (both float64)."""
        is_finite = rounded.isfinite()
        is_same = (a == rounded) | (a.isnan() & rounded.isnan())
        n_nonfinite_mismatches = int((~is_finite & ~is_same).sum())
        n_nonfinite_mismatches += int((is_finite & ~a.isfinite()).sum())
        return cls(
            is_passed=n_nonfinite_mismatches == 0 and not is_bound_exceeded and is_unbiased,
            is_bound_exceeded=is_bound_exceeded,
            violation_fraction=violation_fraction,
            peak_bound_multiple=peak_bound_multiple,
            bias_z_score=bias_z_score,
            n_nonfinite_mismatches=n_nonfinite_mismatches,
        )


def _baseline_error(baseline_out: Tensor | None, y: Tensor) -> tuple[Tensor, Tensor]:
    if baseline_out is None:
        raise ValueError("a baseline_relative bound needs baseline_out")
    baseline = baseline_out.double()
    return baseline - y, y.isfinite() & baseline.isfinite()


def _byte_rows(tensor: Tensor) -> Tensor:
    import torch

    return tensor.contiguous().reshape(-1).view(torch.uint8).reshape(tensor.numel(), -1)


def run_to_run_fraction(a: Tensor, b: Tensor) -> float:
    """The fraction of elements whose bytes differ between two runs (NaN equals itself)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        raise ValueError(f"runs differ in shape or dtype: {a.shape}, {b.shape}")
    if a.numel() == 0:
        return 0.0
    return (_byte_rows(a) != _byte_rows(b)).any(-1).double().mean().item()


def mutable_tensor_declarations(schema: Mapping[str, Any]) -> dict[str, Any]:
    """The schema json's mutable tensors (``OpContract.mutable_tensors``), by name: its
    outputs, then its mutable inputs."""
    mutable = {n: d for n, d in schema["inputs"].items() if d.get("mutable")}
    return {**schema["outputs"], **mutable}


def compare_outputs(
    schema: Mapping[str, Any],
    actual_outputs: Mapping[str, Tensor],
    reference_outputs: Mapping[str, Tensor],
    *,
    magnitude_outputs: Mapping[str, Tensor] | None = None,
    baseline_outputs: Mapping[str, Tensor] | None = None,
) -> dict[str, CompareResult]:
    """Each mutable tensor of the schema json (``mutable_tensor_declarations``) against its
    reference, on the tensors' device, under its one ``error_bounds`` item (an integer one
    must be exact); keyed in that order."""
    compare = schema["compare"]
    element_order = ElementOrder(compare["element_order"])
    magnitude_outputs = magnitude_outputs or {}
    baseline_outputs = baseline_outputs or {}
    results = {}
    for name, declaration in mutable_tensor_declarations(schema).items():
        out_dtype = declaration["dtype"]["any_of"][0]
        baseline_out = baseline_outputs.get(name)
        actual, reference_out = element_order.arrange(actual_outputs[name], reference_outputs[name])
        a = actual.double()
        rounded = reference_out.to(actual.dtype).double()
        if DTYPE_TRAITS[out_dtype].is_integral:
            # integral unit_roundoff is 0: B = 0 on the rounded reference, i.e. exact
            assert DTYPE_TRAITS[out_dtype].unit_roundoff == 0
            error_bound, reference_out = OutputRoundingBound(), rounded
        else:
            error_bound = _ERROR_BOUND_ADAPTER.validate_python(declaration["error_bounds"][0][1])
        results[name] = error_bound.compare(
            a,
            rounded,
            reference_out,
            max_violation_fraction=compare["max_violation_fraction"],
            out_unit_roundoff=DTYPE_TRAITS[out_dtype].unit_roundoff,
            magnitude_out=magnitude_outputs.get(name),
            baseline_out=baseline_out,
        )
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
    """One ``error_bounds`` item of a floating output: how it is compared with y, the
    float64 reference."""

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
        magnitude_out: Tensor | None,
        baseline_out: Tensor | None,
    ) -> CompareResult:
        """Output ``a`` against y (``reference_out``) and y rounded to the output dtype
        (``rounded``), all float64."""


class ElementwiseBound(ErrorBound):
    """B, the largest |actual - y| allowed per element: at most
    ``max_violation_fraction`` of the finite elements exceed it, and the error is
    unbiased. Collapses into ErrorBound once SqnrBound (FIXME(main-kernels)) goes."""

    @abstractmethod
    def allowed_error(
        self,
        reference_out: Tensor,
        out_unit_roundoff: float,
        magnitude_out: Tensor | None,
        baseline_out: Tensor | None,
    ) -> Tensor:
        """B at each element of the one output compared."""

    def compare(
        self,
        a,
        rounded,
        reference_out,
        *,
        max_violation_fraction,
        out_unit_roundoff,
        magnitude_out,
        baseline_out,
    ):
        is_finite = rounded.isfinite()
        errors = a - reference_out
        error = errors[is_finite]
        limit = self.allowed_error(reference_out, out_unit_roundoff, magnitude_out, baseline_out)[is_finite]
        violation_fraction = peak_bound_multiple = 0.0
        if error.numel():
            violation_fraction = (error.abs() > limit).double().mean().item()
            multiples = (error.abs() / limit).masked_fill(error == 0, 0.0)
            peak_bound_multiple = multiples.max().item()
        is_unbiased, bias_z_score = self.bias(
            errors, is_finite, violation_fraction, reference_out, baseline_out
        )
        return CompareResult.of_measures(
            a,
            rounded,
            is_bound_exceeded=violation_fraction > max_violation_fraction,
            is_unbiased=is_unbiased,
            violation_fraction=violation_fraction,
            peak_bound_multiple=peak_bound_multiple,
            bias_z_score=bias_z_score,
        )

    def bias(
        self,
        errors: Tensor,
        is_compared: Tensor,
        violation_fraction: float,
        reference_out: Tensor,
        baseline_out: Tensor | None,
    ) -> tuple[bool, float]:
        """Whether the compared errors are unbiased, |mean| / rms <= 5 / sqrt(n), and
        that bias_z_score."""
        error = errors[is_compared]
        n = error.numel()
        rms = error.square().mean().sqrt().item() if n else 0.0
        bias_z_score = abs(error.mean().item()) / rms if rms > 0 else 0.0
        return n == 0 or bias_z_score <= 5 / math.sqrt(n), bias_z_score


# The multiples and slack of the error bounds, the same for every output.
_SQRT_LENGTH_MULTIPLE = 6.0
_CONSTANT_MULTIPLE = 2.0
_SLACK = 2.0
_OUTPUT_ROUNDING_MULTIPLE = 2.0
_MAX_ERROR_MULTIPLE = 2.0


class DotProductBound(ElementwiseBound):
    """B = _SLACK * (_SQRT_LENGTH_MULTIPLE * sqrt(n_eff) + _CONSTANT_MULTIPLE) * u_acc *
    M + out_unit_roundoff * |y|."""

    kind: Literal["dot_product"] = "dot_product"
    length: str | int  # the reduction length, on point.symbols(); an int in a schema
    accumulate_dtype: DType
    max_split_k: PositiveInt = 1
    expr_fields = ("length",)

    def allowed_error(self, reference_out, out_unit_roundoff, magnitude_out, baseline_out):
        if magnitude_out is None:
            raise ValueError("a dot_product bound needs magnitude_out")
        effective_length = max(
            -(-self.length // s) + (s - 1).bit_length() for s in range(1, self.max_split_k + 1)
        )
        multiple = _SQRT_LENGTH_MULTIPLE * math.sqrt(effective_length) + _CONSTANT_MULTIPLE
        accumulated = _SLACK * multiple * DTYPE_TRAITS[self.accumulate_dtype].unit_roundoff
        return accumulated * magnitude_out + out_unit_roundoff * reference_out.abs()


class OutputRoundingBound(ElementwiseBound):
    """B = _OUTPUT_ROUNDING_MULTIPLE * out_unit_roundoff * |y|."""

    kind: Literal["output_rounding"] = "output_rounding"

    def allowed_error(self, reference_out, out_unit_roundoff, magnitude_out, baseline_out):
        return _OUTPUT_ROUNDING_MULTIPLE * out_unit_roundoff * reference_out.abs()


class BaselineRelativeBound(ElementwiseBound):
    """B = _MAX_ERROR_MULTIPLE * max(|e_b|, r) + out_unit_roundoff * |y|, relative to a
    baseline kernel's error e_b = baseline - y and its per-row rms r."""

    kind: Literal["baseline_relative"] = "baseline_relative"

    def allowed_error(self, reference_out, out_unit_roundoff, magnitude_out, baseline_out):
        import torch

        baseline_error, is_compared = _baseline_error(baseline_out, reference_out)
        squares = torch.where(is_compared, baseline_error.square(), 0)
        rms = (squares.sum(-1) / is_compared.sum(-1)).sqrt()  # NaN: no element compared
        relative = _MAX_ERROR_MULTIPLE * torch.maximum(baseline_error.abs(), rms[..., None])
        rounding = out_unit_roundoff * reference_out.abs()
        is_relative = is_compared & rms.isfinite()[..., None]
        return torch.where(is_relative, relative + rounding, _OUTPUT_ROUNDING_MULTIPLE * rounding)

    def bias(self, errors, is_compared, violation_fraction, reference_out, baseline_out):
        """Also unbiased when the mean error is within _MAX_ERROR_MULTIPLE of the
        baseline's, on the elements both compare."""
        baseline_error, is_baseline_compared = _baseline_error(baseline_out, reference_out)
        is_compared = is_compared & is_baseline_compared
        is_unbiased, bias_z_score = super().bias(
            errors, is_compared, violation_fraction, reference_out, baseline_out
        )
        if not is_unbiased:
            error = errors[is_compared]
            rms = error.square().mean().sqrt().item()
            baseline_bias = abs(baseline_error[is_compared].mean().item())
            is_unbiased = abs(error.mean().item()) <= (
                _MAX_ERROR_MULTIPLE * baseline_bias + 5 * rms / error.numel()
            )
        return is_unbiased, bias_z_score


class ToleranceBound(ElementwiseBound):
    """B = absolute_tolerance + (relative_tolerance + out_unit_roundoff) * |y|."""

    kind: Literal["tolerance"] = "tolerance"
    absolute_tolerance: Annotated[float, Field(ge=0.0)]
    relative_tolerance: Annotated[float, Field(ge=0.0)]

    def allowed_error(self, reference_out, out_unit_roundoff, magnitude_out, baseline_out):
        relative = self.relative_tolerance + out_unit_roundoff
        return self.absolute_tolerance + relative * reference_out.abs()

    def bias(self, errors, is_compared, violation_fraction, reference_out, baseline_out):
        """No bias test once no element exceeds B (main's compare has none)."""
        is_unbiased, bias_z_score = super().bias(
            errors, is_compared, violation_fraction, reference_out, baseline_out
        )
        return is_unbiased or violation_fraction == 0, bias_z_score


class SqnrBound(ErrorBound):
    """FIXME(main-kernels): main's SQNRCompare. The whole output passes when
    20*log10(||y_r|| / ||actual - y_r||) >= min_sqnr_db, y_r the reference rounded to
    the output dtype (equal tensors: inf); no per-element B and no bias test, so
    ``violation_fraction`` is 0 or 1 and ``peak_bound_multiple`` is the noise norm over
    the allowed one. bias_z_score is 0: main's SQNRCompare has no bias test."""

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
        magnitude_out,
        baseline_out,
    ):
        kept = rounded.isfinite() & a.isfinite()  # a non-finite one is a mismatch
        noise = (a - rounded)[kept].norm().item()
        allowed = rounded[kept].norm().item() * 10 ** (-self.min_sqnr_db / 20)
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
            is_unbiased=True,
            violation_fraction=float(is_bound_exceeded),
            peak_bound_multiple=peak_bound_multiple,
            bias_z_score=0.0,
        )


ErrorBoundUnion = Annotated[
    DotProductBound | OutputRoundingBound | BaselineRelativeBound | ToleranceBound | SqnrBound,
    Field(discriminator="kind"),
]
_ERROR_BOUND_ADAPTER = TypeAdapter(ErrorBoundUnion)


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
