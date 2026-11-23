# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""kda_chain_verify_ring's baseline, AKLR's copy of sglang's kda_verify_gfx950, against sglang's own kernel
(gfx950, where the installed sglang has it: Kimi-K3.1's image): the same output, conv windows and ring rows,
bit for bit, at both batch sizes the copy picks a tile width for and both gates."""

import pytest
import torch

from aklr.plugins.sglang.layers.kda_backend_hip.baseline import kda_chain_verify_ring_baseline
from tests.op_case import OpCase

HEAD_DIM = 128


def _sglangs(call):
    """sglang's kda_verify_gfx950 on the call, as kda_chain_verify_gfx950 hands the pools over."""
    from sglang.kernels.ops.attention.linear.kda_verify_gfx950 import kda_verify_gfx950

    n_tokens, heads = call["b"].shape
    softplus = (
        {}
        if call["lower_bound"] is not None
        else {"softplus_beta": call["softplus_beta"], "softplus_threshold": call["softplus_threshold"]}
    )
    result = kda_verify_gfx950(
        mixed_qkv=call["mixed_qkv"],
        conv_weight=call["conv_weight"],
        conv_state=call["conv_state"].transpose(-1, -2),
        intermediate_conv_window=call["conv_window"].transpose(-1, -2),
        intermediate_state_indices=call["window_indices"],
        a=call["a"],
        b=call["b"],
        A_log=call["a_log"],
        dt_bias=call["dt_bias"],
        ssm_states=call["ssm_state"],
        cache_indices=call["cache_indices"],
        intermediate_states_buffer=None,
        scale=call["scale"],
        T=call["conv_window"].shape[1],
        num_q_heads=heads,
        num_v_heads=heads,
        head_k_dim=HEAD_DIM,
        head_v_dim=HEAD_DIM,
        lower_bound=call["lower_bound"],
        cache_ring=True,
        replayssm_rawv=call["ring_v"],
        replayssm_rawk=call["ring_k"],
        replayssm_g=call["ring_g"],
        replayssm_beta=call["ring_beta"],
        **softplus,
    )
    return result.view(n_tokens, heads, HEAD_DIM)


def _point(n_requests, has_lower_bound):
    """Kimi-K3.1's shape at TP8 (12 heads, chains of 5, a ring of 16), no padding request: sglang's kernel leaves
    a padding request's output rows unwritten."""
    from aklr.core.contract.spec import PointSpec

    tensors = ("mixed_qkv", "conv_state", "conv_window", "a", "b", "ring_v", "ring_k", "output")
    return OpCase(
        PointSpec(
            contract_id="kda_chain_verify_ring@1.0",
            axes={
                "n_requests": n_requests,
                "n_tokens": 5 * n_requests,
                "n_verify_tokens": 5,
                "n_heads": 12,
                "n_slots": 2 * n_requests,
                "n_lines": n_requests,
                "ring_len": 16,
            },
            dtypes={
                **{name: "bfloat16" for name in tensors},
                "conv_weight": "float32",
                "cache_indices": "int32",
                "window_indices": "int32",
                "a_log": "float32",
                "dt_bias": "float32",
                "ssm_state": "float32",
                "ring_g": "float32",
                "ring_beta": "float32",
            },
            attrs={"has_lower_bound": has_lower_bound},
            layout_variant_name="row_major",
        )
    )


@pytest.mark.gfx950
@pytest.mark.parametrize("n_requests", [2, 6], ids=["bv4", "bv8"])
@pytest.mark.parametrize("has_lower_bound", [True, False], ids=["lower_bound", "softplus"])
def test_the_baselines_kernel_is_sglangs(n_requests, has_lower_bound):
    pytest.importorskip("sglang.kernels.ops.attention.linear.kda_verify_gfx950")
    case = _point(n_requests, has_lower_bound)
    args = case.inputs()
    args["cache_indices"].copy_(torch.randperm(2 * n_requests, dtype=torch.int32)[:n_requests])
    ours, theirs = dict(case.call(args, "cuda")), dict(case.call(args, "cuda"))
    ours.pop("output"), theirs.pop("output")
    out = kda_chain_verify_ring_baseline(**ours)
    expected = _sglangs(theirs)
    torch.cuda.synchronize()
    assert torch.equal(out, expected)
    for name in ("conv_window", "ring_v", "ring_k", "ring_g", "ring_beta"):
        assert torch.equal(ours[name], theirs[name]), name
