# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import inspect
from types import SimpleNamespace

import pytest

import aklr.seam  # noqa: F401 -- declares the seam adapters
from aklr.core.runtime.dispatch import OpVariants
from aklr.core.runtime.seam import SeamRegistry
from aklr.core.schema import Schema

# the ops' own: tests add apis of their toy ops to the declared ones
_BASELINED = tuple(
    s
    for api in OpVariants.ops().values()
    for s in api.variants
    if s.standard.reference.__module__.startswith("aklr.ops.") and s.standard.baseline is not None
)


@pytest.mark.parametrize("template", _BASELINED, ids=lambda t: t.schema_name({}))
def test_a_baseline_is_timed_under_one_seam_backend(template: Schema) -> None:
    # the bench keys its speedups by this backend, the name dispatch looks them up by
    assert len({a.backend for a in SeamRegistry.default().adapters(template.op_type)}) == 1


@pytest.mark.parametrize(("rows", "act_quant"), [(255, None), (256, "mxfp4")])
def test_aiters_swiglu_call_binds_the_swiglu_template_with_aiters_activation_quant(
    rows: int, act_quant: str | None
) -> None:
    """aiter keeps swiglu's activations bf16 below 256 rows (GPTOSS_SWIGLU_MXFP4_BF16_BOUND), MXFP4 from there."""
    import torch

    (adapter,) = (
        a for a in SeamRegistry.default().adapters("fused_moe") if a.entry == "aiter.fused_moe.fused_moe"
    )
    w1 = torch.empty(8, 512, 256, dtype=torch.float4_e2m1fn_x2, device="meta")
    w1.is_shuffled = True
    bound = adapter.bind_fn(
        torch.empty(rows, 512, dtype=torch.bfloat16, device="meta"),
        w1,
        torch.empty(8, 512, 128, dtype=torch.float4_e2m1fn_x2, device="meta"),
        torch.empty(rows, 2, device="meta"),
        torch.empty(rows, 2, dtype=torch.int32, device="meta"),
        activation=SimpleNamespace(name="Swiglu"),
        quant_type=SimpleNamespace(name="per_1x32"),
        swiglu_limit=7.0,
    )
    assert bound is not None
    assert bound["act_quant"] == act_quant
    signature = inspect.signature(adapter.api)
    call = signature.bind_partial(**bound)
    call.apply_defaults()
    assert adapter.api.resolve(call.arguments).op_name == "fused_moe.swiglu_shuffled_16x16"


@pytest.mark.parametrize(
    ("extra", "served"),
    [({}, True), ({"bias": object()}, False), ({"check": True}, False)],
    ids=["plain", "bias", "check"],
)
def test_aiters_preshuffled_a8w8_gemm_binds_the_shuffled_variant(extra: dict, served: bool) -> None:
    """aiter raises on a bias and check is its debug flag: those calls stay aiter's."""
    import torch

    (adapter,) = (
        a
        for a in SeamRegistry.default().adapters("gemm_a8w8_nt")
        if a.entry == "aiter.ops.gemm_op_a8w8.gemm_a8w8_bpreshuffle"
    )
    bound = adapter.bind_fn(
        torch.empty(4, 64, dtype=torch.float8_e4m3fn, device="meta"),
        torch.empty(32, 64, dtype=torch.float8_e4m3fn, device="meta"),
        torch.empty(4, 1, device="meta"),
        torch.empty(32, 1, device="meta"),
        **extra,
    )
    assert (bound is not None) == served
    if served:
        call = inspect.signature(adapter.api).bind_partial(**bound)
        call.apply_defaults()
        assert adapter.api.resolve(call.arguments).op_name == "gemm_a8w8_nt.shuffled"
