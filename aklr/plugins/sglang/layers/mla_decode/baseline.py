# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What sglang 0.5.22 runs on gfx950 for kimi_k3_mla_decode: TritonAttnBackend.forward_decode's
decode_attention_fwd as a decode CUDA graph bakes it: Lean from 8 rows on (lean_capture_policy),
else the grouped split-K kernel at the splits get_num_kv_splits picks, with the backend's
buffers (allocated once per device, as the backend holds them) and the split cap of a context
of at least 8192 tokens."""

from __future__ import annotations

import functools

__all__ = ["kimi_k3_mla_decode_baseline"]


@functools.cache
def _decode_buffers(device, n_heads, rows):
    """The backend's decode buffers: Lean's persistent partials and locks, split-K's logits."""
    import torch
    from sglang.kernels.ops.attention import decode_attention
    from sglang.srt.layers.attention.triton_backend import _mla_decode_kv_splits_cap

    cores = torch.cuda.get_device_properties(device).multi_processor_count
    # sglang's AMD default of 16 splits, capped as for MLA at a context of 8192 tokens or more
    max_splits = _mla_decode_kv_splits_cap(16, cores, 8192)
    programs = decode_attention._lean_decode_launch_params(1, n_heads)[0]
    block_m = decode_attention._LEAN_BLOCK_M

    def zeros(*shape, dtype=torch.float32):
        return torch.zeros(shape, dtype=dtype, device=device)

    lean = {
        "lean_Mp": zeros(programs, block_m),
        "lean_Lp": zeros(programs, block_m),
        "lean_Op": zeros(programs, block_m, 512),
        "lean_locks": zeros(programs, dtype=torch.int32),
    }
    return cores, max_splits, zeros(rows, n_heads, max_splits, 512), zeros(rows, n_heads, max_splits), lean


def kimi_k3_mla_decode_baseline(q, kv_cache, kv_indptr, kv_indices, sm_scale=192**-0.5, output=None):
    """decode_attention_fwd with forward_decode's arguments for a decode CUDA graph."""
    import torch
    import triton
    from sglang.kernels.ops.attention import decode_attention
    from sglang.kernels.ops.attention.metadata import get_num_kv_splits_triton

    rows, n_heads = q.shape[0], q.shape[1]
    if output is None:
        output = q.new_empty((rows, n_heads, 512))
    cores, max_splits, logits, lse, lean = _decode_buffers(q.device, n_heads, rows)
    # init_forward_metadata_replay_cuda_graph's splits, once per forward in sglang
    splits = torch.empty(rows, dtype=torch.int32, device=q.device)
    get_num_kv_splits_triton[(1,)](
        splits,
        torch.diff(kv_indptr),
        rows,
        1,
        n_heads,
        1,
        max_splits,
        cores,
        MAX_NUM_SEQ=max(256, triton.next_power_of_2(rows)),
    )
    decode_attention.decode_attention_fwd(
        q,
        kv_cache,
        kv_cache[..., :512],
        output,
        kv_indptr,
        kv_indices,
        logits,
        lse,
        splits,
        max_splits,
        sm_scale,
        1.0,
        1.0,
        has_mla=True,
        enable_lean=decode_attention.lean_capture_policy(n_heads, n_heads, rows, True),
        **lean,
    )
    return output
