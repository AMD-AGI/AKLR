# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import math

import pytest
import torch

from aklr.core.contract import semantics
from tests.op_case import OpCase


class TestRowMajorInputs:
    """``row_major_inputs``: seeded by the workload and the input set alone."""

    def test_same_input_set_draws_same_values(self) -> None:
        """Drawn twice for the fused_moe case's ``numeric/0``, every tensor is bitwise equal."""
        case = OpCase.of("fused_moe@1.0")
        first, second = case.inputs("numeric/0"), case.inputs("numeric/0")
        assert first.keys() == second.keys()
        for name, value in first.items():
            if torch.is_tensor(value):
                assert torch.equal(
                    value.contiguous().view(torch.uint8),
                    second[name].contiguous().view(torch.uint8),
                ), name

    def test_other_input_set_draws_other_values(self) -> None:
        """``numeric/0`` and ``numeric/1`` of the gemm case differ in some tensor."""
        case = OpCase.of("gemm@1.0")
        first, second = case.inputs("numeric/0"), case.inputs("numeric/1")
        assert any(
            not torch.equal(value, second[name]) for name, value in first.items() if torch.is_tensor(value)
        )


class TestPoolDistributions:
    """The multi-GB pools: drawn in their distribution, no two pages alike."""

    def test_kv_cache_pages_all_differ(self) -> None:
        """A dsv4 fp8 pool longer than its tiled payload pages: every page, the one a
        tile period away included, differs from every other, and all are normal."""
        distribution = semantics.Deepseekv4Fp8KvCacheDistribution()
        generator = torch.Generator().manual_seed(0)
        pool = distribution.sample((600, 4, 1, 584), "uint8", generator)
        assert distribution.contains(pool, "uint8")
        pages = pool.reshape(600, -1)
        assert len(torch.unique(pages, dim=0)) == 600

    def test_byte_pool_covers_its_range(self) -> None:
        """A uint8 int_range 0..63 pool takes every value in range and none outside."""
        distribution = semantics.IntRangeDistribution(lo=0, hi=63)
        pool = distribution.sample((1000, 132), "uint8", torch.Generator())
        assert pool.dtype == torch.uint8
        assert torch.equal(pool.unique(), torch.arange(64, dtype=torch.uint8))


class TestCompareOutputs:
    """``compare_outputs`` fails an output off its reference."""

    @pytest.mark.parametrize(
        "contract_id",
        [
            pytest.param("gemm@1.0", id="float_output"),
            pytest.param("gemm_a8w8_blockwise_scaled@1.0", id="tolerance_output"),
            pytest.param("fused_moe@1.0", id="sqnr_output"),
            pytest.param("biased_grouped_topk@1.0", id="integer_output"),
        ],
    )
    def test_rejects_corrupted_output(self, contract_id: str) -> None:
        """The reference with one element changed past its error bound."""
        case = OpCase.of(contract_id)
        args = case.inputs()
        reference = case.reference_outputs(args)
        actual = {
            o: reference[o].to(semantics.torch_dtypes(case.point.dtypes[o])[0]) for o in case.contract.outputs
        }
        corrupted = list(actual)[-1]
        flat = actual[corrupted].view(-1)
        flat[0] += 1e3 * (1 + flat.abs().max()) if flat.is_floating_point() else 1
        results = case.compare(actual, args)
        assert not results[corrupted].is_passed
        assert results[corrupted].is_bound_exceeded


def _schema(dtype: str, error_bound: dict | None = None) -> dict:
    """A one-output schema, ``max_violation_fraction`` 0."""
    out = {"dtype": {"any_of": [dtype]}, "error_bounds": [[None, error_bound]]}
    return {
        "compare": {"max_violation_fraction": 0.0, "element_order": "fixed"},
        "inputs": {},
        "outputs": {"out": out},
    }


class TestErrorBounds:
    """The integral path and the two ``bias`` overrides."""

    def test_integral_off_by_one_fails(self) -> None:
        """An integer output must be exact: B = 0."""
        y = torch.arange(16, dtype=torch.int32)
        actual = y.clone()
        actual[0] += 1
        result = semantics.compare_outputs(_schema("int32"), {"out": actual}, {"out": y})["out"]
        assert result.peak_bound_multiple == math.inf
        assert not result.is_passed

    @pytest.mark.parametrize(
        "error_bound",
        [
            {"kind": "tolerance", "absolute_tolerance": 1e-2, "relative_tolerance": 0},
            {"kind": "baseline_relative"},
        ],
        ids=["tolerance", "baseline_relative"],
    )
    def test_biased_within_bound_passes(self, error_bound: dict) -> None:
        """Every element off by the same 2**-8, within B, passes though fully biased
        (z ~ 1 > 5 / sqrt(1024))."""
        y = torch.randn(8, 128, generator=torch.manual_seed(0)).double()
        actual = (y + 2**-8).float()
        result = semantics.compare_outputs(
            _schema("float32", error_bound),
            {"out": actual},
            {"out": y},
            baseline_outputs={"out": actual},
        )["out"]
        assert result.bias_z_score > 0.9
        assert result.violation_fraction == 0
        assert result.is_passed


_SQNR_SCHEMA = {
    "compare": {"max_violation_fraction": 1e-3, "element_order": "fixed"},
    "inputs": {},
    "outputs": {
        "out": {
            "dtype": {"any_of": ["bfloat16"]},
            "error_bounds": [[None, {"kind": "sqnr", "min_sqnr_db": 13.0}]],
        }
    },
}


class TestSqnrBound:
    """``sqnr``: the whole output passes at an SQNR of at least ``min_sqnr_db``."""

    @pytest.mark.parametrize(
        ("noise_db", "max_violation_fraction", "is_passed"),
        [
            (None, 1e-3, True),
            (20.0, 1e-3, True),
            (6.0, 1e-3, False),
            (6.0, 1.0, False),
        ],
    )
    def test_passes_at_threshold(
        self, noise_db: float | None, max_violation_fraction: float, is_passed: bool
    ) -> None:
        """The rounded reference, ~20 dB of biased noise pass; ~6 dB fails, whatever
        ``max_violation_fraction`` allows."""
        schema = {
            **_SQNR_SCHEMA,
            "compare": {
                **_SQNR_SCHEMA["compare"],
                "max_violation_fraction": max_violation_fraction,
            },
        }
        y = torch.randn(64, 128, dtype=torch.float64, generator=torch.manual_seed(0))
        actual = y if noise_db is None else y + y.norm() / y.numel() ** 0.5 * 10 ** (-noise_db / 20)
        result = semantics.compare_outputs(schema, {"out": actual.bfloat16()}, {"out": y})["out"]
        assert result.is_passed is is_passed
        assert result.violation_fraction == float(not is_passed)
        assert result.bias_z_score == 0.0

    @pytest.mark.parametrize(
        ("reference", "fault", "n_nonfinite"),
        [
            pytest.param(1.0, math.nan, 1, id="nan"),
            pytest.param(0.0, 1.0, 0, id="zero_reference"),
        ],
    )
    def test_fails_fault(self, reference: float, fault: float, n_nonfinite: int) -> None:
        """One NaN fails as a non-finite mismatch; any noise on an all-zero reference
        fails."""
        y = torch.full((8, 8), reference, dtype=torch.float64)
        actual = y.bfloat16()
        actual[0, 0] = fault
        result = semantics.compare_outputs(_SQNR_SCHEMA, {"out": actual}, {"out": y})
        assert not result["out"].is_passed
        assert result["out"].n_nonfinite_mismatches == n_nonfinite
