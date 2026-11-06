# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

__all__ = ["moe_sort_mxfp8_quant_baseline"]


def moe_sort_mxfp8_quant_baseline(
    hidden_states,
    topk_ids,
    topk_weights,
    sorted_ids,
    sorted_weights,
    sorted_expert_ids,
    sorted_scales,
    n_experts,
    block_size=32,
    num_valid_ids=None,
    moe_buf=None,
    quantized=None,
):
    """What aiter's fused_moe runs for this region under sglang 0.5.23, writing the caller's
    buffers: the sort is sglang's small-batch sort (moe_sorting_small, which sglang's aiter
    runner patches into aiter's _moe_sorting_impl) where it takes the call, else aiter's opus
    moe sorting at sglang's dispatch policy; then aiter's fused_dynamic_mx_quant_moe_sort
    kernel, bounded by the call's experts, as fused_dynamic_mxfp8_quant_moe_sort launches it for
    stage 1."""
    import os

    import torch
    from aiter import dtypes, moe_sorting_opus_fwd, moe_sorting_opus_get_workspace_size
    from aiter.ops.quant import fused_dynamic_mx_quant_moe_sort_hip_bounded

    n_tokens, topk = topk_ids.shape
    hidden_size = hidden_states.shape[1]
    device = hidden_states.device
    if num_valid_ids is None:
        num_valid_ids = torch.empty(2, dtype=torch.int32, device=device)
    if moe_buf is None:
        moe_buf = torch.empty((n_tokens, hidden_size), dtype=hidden_states.dtype, device=device)
    if quantized is None:
        quantized = torch.empty((n_tokens, hidden_size), dtype=torch.float8_e4m3fn, device=device)
    sorted_args = (
        topk_ids,
        topk_weights,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        moe_buf,
    )

    try:
        from sglang.kernels.ops.moe import moe_sorting_small
    except ImportError:
        moe_sorting_small = None
    # sglang's patch hands it the call when the activations are not emitted with the sort
    # (hidden_size % 2048 != 0 or past 256 pairs)
    if moe_sorting_small is not None and moe_sorting_small._small_sort_supported(
        topk_ids, block_size, None, None, n_experts, emit_mx=False
    ):
        moe_sorting_small._run_small_sort(*sorted_args, block_size, None, n_experts)
    else:
        policy = int(os.environ.get("SGLANG_AITER_MOE_SORTING_DISPATCH_POLICY", "2"))
        workspace_bytes = moe_sorting_opus_get_workspace_size(
            n_tokens, n_experts, topk, policy, topk_ids.get_device()
        )
        workspace = (
            torch.empty(workspace_bytes, dtype=torch.uint8, device=device) if workspace_bytes > 0 else None
        )
        moe_sorting_opus_fwd(
            *sorted_args, n_experts, block_size, None, None, workspace, policy, None, None, None
        )
    fused_dynamic_mx_quant_moe_sort_hip_bounded(
        quantized,
        sorted_scales.view(dtypes.fp8_e8m0),
        hidden_states,
        sorted_ids,
        num_valid_ids,
        n_tokens,
        block_size,
        n_tokens * topk,
        n_experts,
        32,
        sorted_weights,
    )
    return num_valid_ids, moe_buf, quantized
