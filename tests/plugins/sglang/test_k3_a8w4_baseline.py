# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""kimi_k3_routed_moe_mxfp8's baseline, AKLR's copy of sglang's k3_a8w4 kernels, against sglang's own
k3_a8w4_moe on the same MXFP8 input (gfx950, where the installed sglang has it: Kimi-K3.1's image): the
same output, bit for bit, at each sort block size the copy picks."""

import pytest
import torch

from aklr.plugins.sglang.models.kimi_k3.baseline import kimi_k3_routed_moe_mxfp8_baseline
from tests.op_case import OpCase


def _sglangs(call):
    """sglang's k3_a8w4_moe as Kimi-K3.1's row-sharded front calls it: a_mx, hidden_states a shape."""
    from sglang.kernels.ops.moe.k3_a8w4_gfx950 import k3_a8w4_moe

    a = call["a"]
    return k3_a8w4_moe(
        a.new_empty(1, 1, dtype=torch.bfloat16).expand(a.shape),
        call["topk_ids"],
        call["topk_weights"],
        call["w1"],
        call["w1_scale"],
        call["w2"],
        call["w2_scale"],
        beta=call["beta"],
        linear_beta=call["linear_beta"],
        a_mx=(a.view(torch.uint8), call["a_scale"].view(torch.uint8)),
    )


@pytest.mark.gfx950
# 2, 8 and 80 routes an expert: sort blocks of 16, 32 and 64 rows
@pytest.mark.parametrize("n_tokens", [4, 16, 160], ids=["block_m16", "block_m32", "block_m64"])
def test_the_baselines_kernels_are_sglangs(n_tokens):
    pytest.importorskip("sglang.kernels.ops.moe.k3_a8w4_gfx950")
    point = OpCase.of("kimi_k3_routed_moe_mxfp8@1.0").point
    case = OpCase(point.model_copy(update={"axes": {**point.axes, "n_tokens": n_tokens}}))
    call = dict(case.call(case.inputs(), "cuda"))
    call.pop("out")
    out = kimi_k3_routed_moe_mxfp8_baseline(**call)
    expected = _sglangs(call)
    torch.cuda.synchronize()
    assert torch.equal(out, expected)


@pytest.mark.parametrize(
    ("latent_dim", "inter_dim", "topk"),
    [(256, 384, 16), (512, 256, 16), (512, 384, 8)],
    ids=["hidden_256", "inter_256", "top_8"],
)
def test_a_shape_sglangs_experts_do_not_run_is_not_the_baselines(latent_dim, inter_dim, topk):
    """Kimi-K3.1's MoE takes sglang's A8W4 experts only at a hidden size a multiple of 512, an intermediate size
    of 384 or 512 and top-16; elsewhere it keeps aiter's fused_moe, so the baseline does not serve the call."""
    point = OpCase.of("kimi_k3_routed_moe_mxfp8@1.0").point
    axes = {**point.axes, "latent_dim": latent_dim, "inter_dim": inter_dim, "topk": topk}
    case = OpCase(point.model_copy(update={"axes": axes}))
    call = dict(case.call(case.inputs()))
    with pytest.raises(NotImplementedError, match="k3_a8w4_moe"):
        kimi_k3_routed_moe_mxfp8_baseline(**call)
