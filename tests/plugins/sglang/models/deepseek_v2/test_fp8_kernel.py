# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""GLM FP8 decode workloads keep both native routing-bias precisions."""

import pytest
import torch

from aklr.core.contract.registry import ContractRegistry
from aklr.core.runtime.builder import BuilderRegistry
from aklr.plugins.sglang.models.deepseek_v2.contract import DEEPSEEK_V2_FP8_MOE
from tests.op_case import OpCase
from tests.plugins.sglang.cases import SMALL_POINTS


@pytest.mark.gfx950
@pytest.mark.parametrize("rows", [1, 2, 4, 8])
@pytest.mark.parametrize("inter_dim", [256, 512], ids=["tp8", "tp4"])
@pytest.mark.parametrize("bias_dtype", ["bfloat16", "float32"])
def test_glm_fp8_decode_rows_match_reference(rows, inter_dim, bias_dtype, builder_registry):
    point = next(
        p
        for p in SMALL_POINTS
        if p.contract_id == "deepseek_v2_fp8_moe@1.0" and p.attrs["num_expert_group"] == 1
    )
    point = point.model_copy(
        update={
            "axes": point.axes
            | {"n_tokens": rows, "model_dim": 6144, "n_routed": 256, "inter_dim": inter_dim},
            "dtypes": point.dtypes | {"correction_bias": bias_dtype},
        }
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
    assert matching, "GLM FP8 decode shape has no executable workload"
    case = OpCase(point)
    arguments = case.inputs()
    if bias_dtype == "float32":
        # Cover a representative positive layer offset without truncating the
        # FP32 differences that determine expert ordering.
        arguments["correction_bias"] = torch.linspace(7.94, 8.06, point.axes["n_routed"], dtype=torch.float32)
    call = case.call(arguments, "cuda")
    call["out"] = torch.full_like(call["hidden_states"], float("nan"))
    BuilderRegistry.get_instance().build(matching[0])(**call)
    torch.cuda.synchronize()
    compared = case.compare({"out": call["out"]}, arguments)
    assert compared == {}, compared
