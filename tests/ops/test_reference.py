# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from collections.abc import Callable
from types import SimpleNamespace

import pytest
import torch

from aklr.core.contract import semantics
from aklr.core.platform.device import DeviceTarget
from aklr.core.validation.stages._bench._evaluation._comparison import (
    is_reference_comparable,
)
from tests.op_case import OpCase


class TestReference:
    """The reference's outputs are as declared, admitted by the contract's compare."""

    def test_outputs_are_declared(self, op_case: OpCase) -> None:
        """On the CPU, ``reference_outputs`` holds every result, float32 or float64 (an
        exact one), finite."""
        reference = op_case.reference_outputs(op_case.inputs())
        assert list(reference) == list(op_case.contract.mutable_tensors)
        for name, tensor in reference.items():
            assert tensor.device.type == "cpu", name
            assert tensor.dtype in (torch.float32, torch.float64), name
            assert tensor.isfinite().all(), name
            assert tuple(tensor.shape) == op_case.contract.dtype_storage_shape_of(name, op_case.point), name

    def test_rounded_reference_passes_compare(self, op_case: OpCase) -> None:
        """The reference rounded to each output's dtype passes ``op_case.compare``."""
        args = op_case.inputs()
        assert is_reference_comparable(
            op_case.point.dtypes,
            op_case.reference_outputs(args),
            lambda rounded: op_case.compare(rounded, args, baseline_outputs=rounded),
        )

    @pytest.mark.gpu
    def test_backend_matches_reference(self, op_case: OpCase) -> None:
        if op_case.needs_gfx950 and DeviceTarget.current().arch != "gfx950":
            pytest.skip("the backend runs the Point only on gfx950")
        args = op_case.inputs()
        call = op_case.call(args, "cuda")
        outputs = op_case.contract.call_mutable_tensors(op_case.contract.baseline()(**call), call)
        results = op_case.compare(outputs, args, baseline_outputs=outputs)
        output_specs = semantics.mutable_tensor_declarations(op_case.request.schema_json)
        baseline_relative = {
            o
            for o, spec in output_specs.items()
            if spec["error_bounds"] and spec["error_bounds"][0][1]["kind"] == "baseline_relative"
        }
        failed = {o: r.detail for o, r in results.items() if not r.is_passed and o not in baseline_relative}
        assert failed == {}
        # self-baselined compare is vacuous: check the reference by relative L2 error
        # shortcut: 0.1 uncalibrated (MI300X skips), set to ~3-5x the gfx950 ratio once measured
        reference = op_case.reference_outputs(args)
        for o in baseline_relative:
            error = (outputs[o].cpu().double() - reference[o]).norm()
            assert error <= 0.1 * reference[o].norm(), (
                o,
                (error / reference[o].norm()).item(),
            )


def test_mqa_logits_reference_is_the_same_in_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One row a chunk gives fp8_paged_mqa_logits' reference bit for bit."""
    import aklr.ops.mla.reference as mla_reference

    case = OpCase.of("fp8_paged_mqa_logits@1.0")
    args = case.inputs()
    whole = case.reference_outputs(args)["logits"]
    monkeypatch.setattr(mla_reference, "_CHUNK_ELEMENTS", 1)
    assert torch.equal(case.reference_outputs(args)["logits"], whole)


def test_biased_grouped_topk_ranks_scores_float32_would_tie() -> None:
    """Two logits a float32 sigmoid maps to one score: the ids still follow the larger."""
    from aklr.ops.biased_grouped_topk.reference import biased_grouped_topk_reference

    point = SimpleNamespace(
        dtypes={"router_logits": "float32", "correction_bias": "float32"},
        axes={"topk": 1},
        attrs={"renormalize": False},
    )
    args = {
        "router_logits": torch.tensor([[0.0, 2.0**-25]]),
        "correction_bias": torch.zeros(2),
        "routed_scaling_factor": 1.0,
    }
    _, ids = biased_grouped_topk_reference(point, args, semantics.to_float32)
    assert ids.tolist() == [[1.0]]


def test_biased_grouped_topk_ranks_tied_experts_lower_first() -> None:
    """Equal biased scores: the lower expert ranks first, whatever torch.topk's algorithm does."""
    from aklr.ops.biased_grouped_topk.reference import biased_grouped_topk_reference

    point = SimpleNamespace(
        dtypes={"router_logits": "float32", "correction_bias": "float32"},
        axes={"topk": 3},
        attrs={"renormalize": False},
    )
    logits = torch.zeros(1, 1024)
    logits[0, [583, 571, 368, 31]] = 1.0
    args = {
        "router_logits": logits,
        "correction_bias": torch.zeros(1024),
        "routed_scaling_factor": 1.0,
    }
    _, ids = biased_grouped_topk_reference(point, args, semantics.to_float32)
    assert ids.tolist() == [[31.0, 368.0, 571.0]]


def test_rms_norm_reference_holds_large_finite_inputs() -> None:
    """A row of bfloat16 2**64 normalizes to 1, its square past float32's range."""
    from aklr.ops.rms_norm.reference import rms_norm_reference

    point = SimpleNamespace(dtypes={"x": "bfloat16", "weight": "bfloat16"})
    args = {
        "x": torch.full((1, 8), 2.0**64, dtype=torch.bfloat16),
        "weight": torch.ones(8, dtype=torch.bfloat16),
        "epsilon": 1e-6,
    }
    torch.testing.assert_close(rms_norm_reference(point, args, semantics.to_float32), torch.ones(1, 8))


def test_add_rms_norm_reference_holds_a_sum_past_float32() -> None:
    """bfloat16 x = residual = 2**127: the exact sum 2**128 is past float32's range, its
    normalized row still 1."""
    from aklr.ops.rms_norm.reference import add_rms_norm_reference

    point = SimpleNamespace(dtypes={"x": "bfloat16", "residual": "bfloat16", "weight": "bfloat16"})
    x = torch.full((1, 8), 2.0**127, dtype=torch.bfloat16)
    args = {"x": x, "residual": x, "weight": torch.ones(8, dtype=torch.bfloat16), "epsilon": 1e-6}
    normalized, total = add_rms_norm_reference(point, args, semantics.to_float32)
    torch.testing.assert_close(normalized, torch.ones(1, 8, dtype=normalized.dtype))
    assert total.tolist() == [[2.0**128] * 8]


@pytest.mark.parametrize("fused", [False, True], ids=["pre", "fused_post_pre"])
def test_mhc_pre_normalizes_large_finite_residuals(fused: bool) -> None:
    """Residual streams of bfloat16 2**64, a zero projection: both mean squares are past
    float32's range, the normalized layer input is still 1."""
    from aklr.ops.mhc.reference import mhc_fused_post_pre_reference, mhc_pre_reference

    tokens, streams, hidden = 1, 2, 4
    mixes = 2 * streams + streams * streams
    args = {
        "residual": torch.full((tokens, streams, hidden), 2.0**64, dtype=torch.bfloat16),
        "x": torch.zeros(tokens, hidden, dtype=torch.bfloat16),
        "post_mix": torch.zeros(tokens, streams),
        "comb_mix": torch.eye(streams).expand(tokens, streams, streams).contiguous(),
        "proj_weight": torch.zeros(mixes, streams * hidden),
        "mix_scale": torch.ones(3),
        "mix_bias": torch.zeros(mixes),
        "norm_weight": torch.ones(hidden, dtype=torch.bfloat16),
        "rms_eps": 1e-6,
        "pre_eps": 0.0,
        "post_multiplier": 2.0,
        "sinkhorn_eps": 1e-6,
        "sinkhorn_iters": 2,
        "norm_eps": 1e-6,
    }
    point = SimpleNamespace(
        dtypes={k: str(v.dtype).removeprefix("torch.") for k, v in args.items() if torch.is_tensor(v)},
        attrs={"has_norm_weight": True},
    )
    reference = mhc_fused_post_pre_reference if fused else mhc_pre_reference
    layer_input = reference(point, args, semantics.to_float32)[2]
    torch.testing.assert_close(layer_input, torch.ones(tokens, hidden, dtype=layer_input.dtype))


def test_float32_reference_fares_as_the_float64_one(op_case: OpCase) -> None:
    """Rounded to the output dtypes and compared against the float64 reference, the float32
    reference passes as the float64 one does: its own error leaves kernels their budget."""
    args = op_case.inputs()

    def worst_fraction(decode: Callable[[torch.Tensor, str], torch.Tensor]) -> float:
        rounded = {
            o: t.to(semantics.torch_dtypes(op_case.point.dtypes[o])[0])
            for o, t in op_case.reference_outputs(args, decode).items()
        }
        results = op_case.compare(rounded, args, baseline_outputs=rounded, decode=semantics.to_float64)
        assert all(r.is_passed for r in results.values()), {o: r.detail for o, r in results.items()}
        return max(r.violation_fraction for r in results.values())

    assert worst_fraction(semantics.to_float32) <= worst_fraction(semantics.to_float64) + 0.1
