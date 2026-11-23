# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""GLM FP8's TP4 medium batches require admitted workloads and native-close output."""

import pytest
import torch

from aklr.core.contract.registry import ContractRegistry
from aklr.core.runtime.builder import BuilderRegistry
from aklr.plugins.sglang.models.deepseek_v2.contract import DEEPSEEK_V2_FP8_MOE
from tests.op_case import OpCase
from tests.plugins.sglang.cases import SMALL_POINTS


@pytest.mark.gfx950
@pytest.mark.parametrize("rows", [8])
def test_glm_fp8_tp4_medium_rows_match_native(rows, builder_registry):
    point = next(
        p
        for p in SMALL_POINTS
        if p.contract_id == "deepseek_v2_fp8_moe@1.0" and p.attrs["num_expert_group"] == 1
    )
    point = point.model_copy(
        update={"axes": point.axes | {"n_tokens": rows, "model_dim": 6144, "n_routed": 256, "inter_dim": 512}}
    )
    schemas, errors = ContractRegistry.default().read_schemas("aklr.plugins.sglang.models.deepseek_v2")
    assert errors == []
    matching = [
        schema
        for _, schema in schemas
        if schema.name == DEEPSEEK_V2_FP8_MOE.name_at(point)
        and schema.kernel.sources
        and any(workload.axes["n_tokens"] == rows for workload in schema.workloads)
    ]
    assert matching, "GLM FP8 TP4 medium batch has no executable workload"
    case = OpCase(point)
    arguments = case.inputs()
    call = case.call(arguments, "cuda")
    call["out"] = torch.full_like(call["hidden_states"], float("nan"))
    native = DEEPSEEK_V2_FP8_MOE.baseline()(**call)
    expected = {
        name: value.clone() for name, value in DEEPSEEK_V2_FP8_MOE.call_mutable_tensors(native, call).items()
    }
    call["out"].fill_(float("nan"))
    BuilderRegistry.get_instance().build(matching[0])(**call)
    torch.cuda.synchronize()
    compared = case.compare({"out": call["out"]}, arguments, baseline_outputs=expected)
    assert all(result.is_passed for result in compared.values()), compared
