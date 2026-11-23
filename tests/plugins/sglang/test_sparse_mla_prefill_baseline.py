# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""dsa_sparse_mla_fp8_ds_prefill's baseline, AKLR's copy of sglang's sparse_mla_prefill_gfx950, against sglang's own
kernel (gfx950, where the installed sglang has it: Kimi-K3.1's image): the same output and LSE, bit for bit, at
12 heads (Kimi-K3.1's at TP8) and at 16, with a -1 entry among the indices."""

import pytest
import torch

from aklr.plugins.sglang.layers.dsa_backend_hip.baseline import dsa_sparse_mla_fp8_ds_prefill_baseline
from tests.op_case import OpCase


@pytest.mark.gfx950
@pytest.mark.parametrize("n_heads", [12, 16])
def test_the_baselines_kernel_is_sglangs(n_heads):
    pytest.importorskip("sglang.kernels.ops.attention.dsa.sparse_mla_prefill_gfx950")
    from sglang.kernels.ops.attention.dsa.sparse_mla_prefill_gfx950 import sparse_mla_prefill_gfx950

    point = OpCase.of("dsa_sparse_mla_fp8_ds_prefill@1.0").point
    case = OpCase(point.model_copy(update={"axes": {**point.axes, "n_heads": n_heads}}))
    call = dict(case.call(case.inputs(), "cuda"))
    call["indices"][0, -1] = -1
    output, lse = dsa_sparse_mla_fp8_ds_prefill_baseline(
        call["q_nope"], call["q_rope"], call["kv_cache"], call["indices"], call["sm_scale"]
    )
    expected, expected_lse = sparse_mla_prefill_gfx950(
        call["q_nope"],
        call["q_rope"],
        call["kv_cache"].unsqueeze(1),
        call["indices"].unsqueeze(1),
        call["sm_scale"],
        return_lse=True,
    )
    torch.cuda.synchronize()
    assert torch.equal(output, expected.squeeze(0)) and torch.equal(lse, expected_lse)
