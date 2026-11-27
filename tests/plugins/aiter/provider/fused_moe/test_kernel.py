# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The TP8 expert width through the public schema builder and numerical oracle."""

import pytest
import torch

from aklr.core.contract.registry import ContractRegistry
from aklr.core.runtime.builder import BuilderRegistry
from aklr.plugins.aiter.provider.fused_moe.contract import FUSED_MOE
from tests.op_case import OpCase
from tests.plugins.aiter.provider.cases import SMALL_POINTS


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
    schemas, errors = ContractRegistry.default().read_schemas("aklr.plugins.aiter.provider.fused_moe")
    assert errors == []
    matching = [s for _, s in schemas if s.name == FUSED_MOE.name_at(point) and s.kernel.sources]
    assert matching, "MiniMax TP8 has no executable 384-wide expert kernel"
    case = OpCase(point)
    args = case.inputs()
    call = case.call(args, "cuda")
    # poisoned: an unwritten result fails
    call["out"] = torch.full_like(call["hidden_states"], float("nan"))
    runnable = BuilderRegistry.get_instance().build(matching[0])
    runnable(**call)
    torch.cuda.synchronize()
    compared = case.compare({"out": call["out"]}, args)
    assert not compared, compared


@pytest.mark.gfx950
def test_the_expert_parallel_kernel_takes_aiters_enums(builder_registry):
    """aiter's route and the engines hand a kernel aiter's ActivationType and QuantType; validation hands it
    the contract's names, so a kernel reading only those passes there and fails every served call."""
    aiter = pytest.importorskip("aiter")
    schemas, errors = ContractRegistry.default().read_schemas("aklr.plugins.aiter.provider.fused_moe")
    assert errors == []
    name = (
        "fused_moe_interleave_has_expert_mask_has_num_local_tokens_has_swiglu_limit_gate_up_interleaved"
        "_model_dim7168_n_experts96_topk6_inter_dim3072_global_experts384"
    )
    for schema in [s for _, s in schemas if s.name == name and s.kernel.sources]:
        case = OpCase(schema.point_with_contract_id(schema.workloads[0].axes, FUSED_MOE.contract_id))
        args = case.inputs()
        call = case.call(args, "cuda") | {
            "activation": aiter.ActivationType.Silu,
            "quant_type": aiter.QuantType.per_1x32,
        }
        call["out"] = torch.full_like(call["hidden_states"], float("nan"))
        BuilderRegistry.get_instance().build(schema)(**call)
        torch.cuda.synchronize()
        compared = case.compare({"out": call["out"]}, args)
        assert not compared, compared
