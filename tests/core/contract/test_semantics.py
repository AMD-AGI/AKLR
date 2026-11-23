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

    def test_distinct_values_differ_within_a_row(self) -> None:
        """Distinct int_range rows shorter than the range: in range, no value twice."""
        distribution = semantics.IntRangeDistribution(lo=3, hi=202, distinct=True)
        values = distribution.sample((64, 100), "int32", torch.Generator().manual_seed(0))
        assert values.shape == (64, 100) and values.dtype == torch.int32
        assert distribution.contains(values, "int32")

    @pytest.mark.gpu
    def test_distinct_values_are_the_seeds_alone(self) -> None:
        """A page table of 128 rows of 4096 distinct pages, drawn 20 times on the GPU
        from one seed, is the same every time: each task draws its own inputs."""
        distribution = semantics.IntRangeDistribution(lo=0, hi=4095, distinct=True)

        def draw() -> torch.Tensor:
            with torch.device("cuda"):
                generator = torch.Generator("cuda").manual_seed(0)
                return distribution.sample((128, 4096), "int32", generator)

        first = draw()
        assert distribution.contains(first, "int32")
        assert all(torch.equal(draw(), first) for _ in range(20))


class TestReferenceCheck:
    """``OpReference.check`` fails an output off its reference."""

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
        actual = {o: reference[o] for o in case.contract.outputs}
        corrupted = list(actual)[-1]
        flat = actual[corrupted].view(-1)
        flat[0] += 1e3 * (1 + flat.abs().max()) if flat.is_floating_point() else 1
        assert corrupted in case.compare(actual, args)


_EXACT = semantics.ToleranceBound(absolute_tolerance=0, relative_tolerance=0)
_SQNR = semantics.SqnrBound(min_sqnr_db=13.0)


class TestErrorBounds:
    """The elementwise bounds, integers included."""

    def test_integral_off_by_one_fails(self) -> None:
        """An integer output under a zero tolerance must be exact."""
        y = torch.arange(16, dtype=torch.int32)
        actual = y.clone()
        actual[0] += 1
        is_passed, detail = _EXACT.compare(actual, y)
        assert not is_passed and "peak_bound_multiple inf," in detail

    def test_integral_compares_exactly_past_float32(self) -> None:
        """Ids past 2**24, one off: float32 would hold both as one value; compare must not."""
        y = torch.full((16,), 2**30 + 1, dtype=torch.int32)
        actual = y.clone()
        actual[0] -= 1
        is_passed, detail = _EXACT.compare(actual, y)
        assert not is_passed and detail.startswith("violation_fraction 0.0625,")

    def test_an_empty_output_passes(self) -> None:
        """An output with no element is within any bound."""
        assert _EXACT.compare(torch.zeros(0), torch.zeros(0))[0]
        assert _SQNR.compare(torch.zeros(0), torch.zeros(0))[0]

    def test_two_elements_past_the_bound_fail(self) -> None:
        """2 of 2000 elements past B: the output fails, its violation_fraction exactly 1e-3, which a
        float32 mean would miss."""
        y = torch.zeros(2000)
        actual = y.clone()
        actual[:2] = 1.0
        bound = semantics.ToleranceBound(absolute_tolerance=0.5, relative_tolerance=0)
        is_passed, detail = bound.compare(actual, y)
        assert not is_passed and detail.startswith("violation_fraction 0.001,")


class TestSqnrBound:
    """``sqnr``: the whole output passes at an SQNR of at least ``min_sqnr_db``."""

    @pytest.mark.parametrize(("noise_db", "is_passed"), [(None, True), (20.0, True), (6.0, False)])
    def test_passes_at_threshold(self, noise_db: float | None, is_passed: bool) -> None:
        """The reference rounded to bf16 and ~20 dB of biased noise pass; ~6 dB fails."""
        y = torch.randn(64, 128, generator=torch.manual_seed(0))
        actual = y if noise_db is None else y + y.norm() / y.numel() ** 0.5 * 10 ** (-noise_db / 20)
        assert _SQNR.compare(actual.bfloat16(), y)[0] is is_passed

    def test_noise_past_float32_keeps_its_sqnr(self) -> None:
        """4096 values of 2**127, one turned to -2**127: the difference 2**128 is past float32's
        range, the SQNR about 30 dB."""
        y = torch.full((4096,), 2.0**127)
        actual = y.bfloat16()
        actual[0] = -(2.0**127)
        assert _SQNR.compare(actual, y)[0]

    def test_large_values_keep_their_sqnr(self) -> None:
        """Values near 2**100, whose squares are past float32's range: an offset of 2**-10 of
        the peak passes (about 47 dB), one of the peak fails (below 0 dB)."""
        y = torch.randn(64, 128, generator=torch.manual_seed(0)) * 2.0**100
        peak = y.abs().max()
        for offset, is_passed in ((peak * 2.0**-10, True), (peak, False)):
            assert _SQNR.compare((y + offset).bfloat16(), y)[0] is is_passed

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
        y = torch.full((8, 8), reference)
        actual = y.bfloat16()
        actual[0, 0] = fault
        is_passed, detail = _SQNR.compare(actual, y)
        assert not is_passed and detail.endswith(f" {n_nonfinite} non-finite mismatches")


class TestRoundedToBfloat16:
    """``rounded_to_bfloat16``: torch's own float32 to bfloat16 cast, rounded once."""

    def test_float32_values_round_as_torch_casts_them(self) -> None:
        """Random float32 bit patterns, every exponent included: the helper's bfloat16 is the
        cast's, subnormals, overflow to infinity and NaN alike."""
        bits = torch.randint(-(2**31), 2**31, (1 << 20,), generator=torch.Generator().manual_seed(0))
        edges = torch.tensor([0.0, -0.0, 2.0**-133, 2.0**-134, 2.0**-149, 3.3961e38, -3.3961e38, math.inf])
        x = torch.cat((bits.int().view(torch.float32), edges))
        expected = x.bfloat16().float()
        torch.testing.assert_close(semantics.rounded_to_bfloat16(x), expected, rtol=0, atol=0, equal_nan=True)


class TestScaleLayouts:
    """The byte that the e8m0 scale layouts write into their padding."""

    @staticmethod
    def _stored_bytes(layout: semantics.Layout, shape: tuple[int, ...], fill: int) -> torch.Tensor:
        """``layout``'s stored bytes of row-major e8m0 scales that are all ``fill``."""
        scales = torch.full(shape, fill, dtype=torch.uint8).view(torch.float8_e8m0fnu)
        return layout.encode(scales).view(torch.uint8)

    @pytest.mark.parametrize(
        ("layout", "shape", "pad"),
        [
            # gemm_a4w4_mxfp4's b_shuffled a_scale at m = 33, k = 2880: rows and groups padded
            (semantics.E8M0Scales(block_size=32, pad_to=(256, 8), shuffled=True), (33, 90), 0x00),
            (semantics.E8M0Scales(block_size=32, pad_to=(32, 8), shuffled=True), (2, 40, 12), 0x00),
            (semantics.E8M0Scales(block_size=32, pad_to=(1, 8)), (3, 12), 0x00),
            # fused_moe's w1_scale and w2_scale at inter_dim 384, 12 groups padded to 16: 1.0, as
            # aiter's shuffle_scale(is_guinterleave=True) pads the scales sglang serves
            (semantics.GateUpInterleavedScales(gate_up=True), (2, 64, 12), 0x7F),
            (semantics.GateUpInterleavedScales(gate_up=False), (2, 64, 12), 0x7F),
        ],
    )
    def test_padding_value(self, layout: semantics.Layout, shape: tuple[int, ...], pad: int) -> None:
        """Every byte the layout pads is ``pad``: of the bytes stored from scales all 1 and all
        2, the ones both share, as many as the stored shape adds."""
        ones, twos = self._stored_bytes(layout, shape, 1), self._stored_bytes(layout, shape, 2)
        padding = ones == twos
        assert padding.sum().item() == math.prod(layout.storage_shape(shape)) - math.prod(shape)
        assert (ones[padding] == pad).all()
