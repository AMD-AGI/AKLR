# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The TP8 expert width through the public schema builder and numerical oracle."""

import pytest
import torch

from aklr.core.contract.registry import ContractRegistry
from aklr.core.runtime.builder import BuilderRegistry
from aklr.plugins.aiter.fused_moe.plugin import FUSED_MOE
from tests.op_case import OpCase
from tests.plugins.aiter.operator_cases import SMALL_POINTS


@pytest.mark.gfx950
@pytest.mark.parametrize("rows", [1, 5, 17])
def test_minimax_tp8_kernel_matches_reference(rows, builder_registry):
    point = next(
        p
        for p in SMALL_POINTS
        if p.contract_id == "fused_moe@1.0"
        and p.attrs["activation"] == "swiglu"
        and not p.attrs["has_expert_mask"]
    )
    point = point.model_copy(
        update={
            "axes": point.axes
            | {"n_tokens": rows, "model_dim": 6144, "n_experts": 128, "topk": 4, "inter_dim": 384}
        }
    )
    schemas, errors = ContractRegistry.default().read_schemas("aklr.plugins.aiter.fused_moe")
    assert errors == []
    matching = [s for _, s in schemas if s.name == FUSED_MOE.name_at(point) and s.kernel.sources]
    assert matching, "MiniMax TP8 has no executable 384-wide expert kernel"
    case = OpCase(point)
    args = case.inputs()
    call = case.call(args, "cuda")
    call["out"] = torch.full_like(call["hidden_states"], float("nan"))
    baseline = FUSED_MOE.baseline()(**call)
    # The baseline may write the same caller-provided output. Preserve its
    # values, then poison the candidate output so an unwritten result fails.
    baseline = {name: value.clone() for name, value in FUSED_MOE.call_mutable_tensors(baseline, call).items()}
    call["out"].fill_(float("nan"))
    runnable = BuilderRegistry.get_instance().build(matching[0])
    runnable(**call)
    torch.cuda.synchronize()
    actual = {"out": call["out"]}
    compared = case.compare(actual, args, baseline_outputs=baseline)
    assert all(r.is_passed for r in compared.values()), compared
