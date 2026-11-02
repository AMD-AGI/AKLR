# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The GDN decode contracts and their committed schemas."""

from pathlib import Path

import pytest

from aklr.core.contract.registry import ContractRegistry
from aklr.core.validation.stages._audit._stage import check_kernel
from aklr.plugins.sglang.models.qwen3_5.contract import (
    GDN_DECODE,
    GDN_DECODE_GROUP_FP8_QUANT,
    GDN_PREFILL_GROUP_FP8_QUANT,
)
from tests.op_case import OpCase

IMPLS = Path(__file__).parents[3] / "aklr/plugins/sglang/models/qwen3_5/impls"


@pytest.mark.parametrize(
    ("contract", "outputs"),
    [
        (GDN_DECODE_GROUP_FP8_QUANT, ("values", "scales")),
        (GDN_DECODE, ("normalized",)),
    ],
)
def test_contract_compares_outputs_and_both_live_states(contract, outputs):
    assert contract.mutable_tensors == (*outputs, "conv_state", "delta_state")
    case = OpCase.of(contract.contract_id)
    args = case.inputs()
    # the small Point draws a -1 padding row and leaves a slot unindexed
    assert -1 in args["cache_indices"].tolist()
    assert list(case.reference_outputs(args)) == list(contract.mutable_tensors)


def test_decode_shares_the_prefill_layer_inputs():
    """One layer: decode reads prefill's projections, weights and state pools."""
    prefill = GDN_PREFILL_GROUP_FP8_QUANT.inputs
    for contract in (GDN_DECODE_GROUP_FP8_QUANT, GDN_DECODE):
        for name, spec in contract.inputs.items():
            if name not in ("cache_indices", "delta_state"):
                assert spec is prefill[name], name
        state = contract.inputs["delta_state"]
        assert state.model_copy(update={"error_bounds": ()}) == prefill["delta_state"].model_copy(
            update={"error_bounds": ()}
        )
        ((_, bound),) = state.error_bounds
        assert (bound.absolute_tolerance, bound.relative_tolerance) == (1e-2, 1e-2)
        ((_, distribution),) = contract.inputs["cache_indices"].distribution
        assert (distribution.lo, distribution.hi) == (-1, "n_slots - 1")
        assert distribution.distinct


@pytest.mark.parametrize(
    ("contract", "rows"),
    [
        (GDN_DECODE_GROUP_FP8_QUANT, [1, 2, 4, 8, 16, 32, 128, 256]),
        (GDN_DECODE, [64]),
    ],
)
def test_decode_kernels_are_self_contained_schemas(contract, rows):
    registry = ContractRegistry.default()
    name = contract.contract_id.split("@")[0]
    paths = sorted((IMPLS / "triton").glob(f"{name}_n_v_heads*/*.json"))
    served = []
    for path in paths:
        schema = registry.parse_schema(path.read_bytes(), str(path))
        assert schema.contract_id == contract.contract_id
        kernel = schema.kernel
        assert kernel.language == "triton" and kernel.author == "openai"
        assert kernel.requires == {"triton": ">=3.8"}
        assert kernel.entry_point == "kernel.py::run"
        assert not kernel.state_bytes and not kernel.peer_bytes
        assert [s.path for s in kernel.sources] == ["kernel.py"]
        assert kernel.sources[0].content.startswith("# Created by OpenAI\n")
        findings = check_kernel(
            str(path),
            kernel,
            call_keys=schema.kernel_call_keys,
            tensor_names=tuple(schema.tensors()),
            mutable_names=schema.mutable_tensors,
        )
        assert findings == ()
        # one Point a kernel, its pool a few slots larger than its rows
        (workload,) = schema.workloads
        n_tokens, n_slots = workload.axes["n_tokens"], workload.axes["n_slots"]
        assert n_tokens < n_slots <= n_tokens + 64
        served.append(n_tokens)
    assert sorted(served) == rows
