###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""The Baselines of the mla_prefill op: the backend calls its schemas' standards time a kernel against."""

from __future__ import annotations

import torch

from aklr.ops.mla_prefill.ps_plan import ps_plan


def pa_sparse_prefill_fp8_opus_baseline(
    q_nope,
    q_rope,
    prefix_nope,
    prefix_rope,
    prefix_indices,
    prefix_indptr,
    extend_nope,
    extend_rope,
    extend_indices,
    extend_indptr,
    sinks,
    softmax_scale=512**-0.5,
    output=None,
    **_,
):
    from aiter.ops.pa_sparse_prefill_opus import pa_sparse_prefill_fp8_opus

    return lambda: pa_sparse_prefill_fp8_opus(
        q_nope.view(torch.float8_e4m3fn),
        q_rope,
        prefix_nope.view(torch.float8_e4m3fn),
        prefix_rope,
        prefix_indices,
        prefix_indptr,
        extend_nope.view(torch.float8_e4m3fn),
        extend_rope,
        extend_indices,
        extend_indptr,
        sinks,
        softmax_scale,
        out=output,
    )


def mla_prefill_varlen_baseline(
    q, k, v, cu_seqlens_q, qkv_fp8=True, cu_seqlens_k=None, sm_scale=None, output=None, **_
):
    """sglang's MLA prefill on aiter: with qkv_fp8 its fp8 path (the heads tiled up to one aiter's reduce
    instantiates, q, k and v cast to fp8, the ASM prefill over aiter's persistent plan, then mla_reduce_v1), else
    aiter's flash attention in bf16. The plan is made here, as sglang makes it once a forward."""
    from aiter import get_ps_metadata_info_v1, get_ps_metadata_v1, mla_prefill_ps_asm_fwd, mla_reduce_v1

    if output is None or sm_scale is None:
        return None
    heads, v_dim = q.shape[1], v.shape[-1]
    offsets_k = cu_seqlens_q if cu_seqlens_k is None else cu_seqlens_k
    if not qkv_fp8:
        from aiter import flash_attn_varlen_func

        # the row counts bound the longest sequence's without reading the offsets
        return lambda: flash_attn_varlen_func(
            q, k, v, cu_seqlens_q, offsets_k, q.shape[0], k.shape[0], softmax_scale=sm_scale, causal=True,
            out=output,
        )  # fmt: skip
    tile_q = 256  # mla_reduce_v1's query tile
    # the head counts mla_reduce_v1 instantiates at v_head_dim 128 (aiter's reduce.cu): sglang tiles up to the next
    padded = min(h for h in (1, 2, 4, 8, 10, 16, 32, 40, 64, 128) if h >= heads)
    qo_cpu, kv_cpu = cu_seqlens_q.to("cpu", torch.int32), offsets_k.to("cpu", torch.int32)
    max_q = int(qo_cpu.diff().max())
    sizes = get_ps_metadata_info_v1(
        batch_size=qo_cpu.numel() - 1, num_head_k=padded, max_qlen=max_q, qlen_granularity=tile_q
    )
    # the planner writes host buffers, copied to the device in stream order: on device buffers it writes with a
    # null-stream hipMemcpy that does not order with the current stream's queued work
    buffers = [torch.empty(size, dtype=dtype) for size, dtype in sizes]
    get_ps_metadata_v1(
        qo_cpu, kv_cpu, kv_cpu.diff(), 1, padded, *buffers, qhead_granularity=1, qlen_granularity=tile_q,
        kvlen_granularity=128, block_size=1, is_causal=True,
    )  # fmt: skip
    _, work_indptr, work_info, reduce_indptr, reduce_final_map, reduce_partial_map = (
        b.to(q.device) for b in buffers
    )
    indices = torch.arange(k.shape[0], dtype=torch.int32, device=q.device)
    one = torch.ones((), dtype=torch.float32, device=q.device)
    partial = max(reduce_partial_map.size(0), 1) * tile_q

    def prefill():
        qf, kf, vf = (
            (x.repeat(1, -(-padded // heads), 1)[:, :padded].contiguous() if padded != heads else x).to(
                torch.float8_e4m3fn
            )
            for x in (q, k, v)
        )
        logits = torch.empty((partial, padded, v_dim), dtype=torch.float32, device=q.device)
        lse = torch.empty((partial, padded), dtype=torch.float32, device=q.device)
        out = torch.empty((q.shape[0], padded, v_dim), dtype=q.dtype, device=q.device)
        mla_prefill_ps_asm_fwd(
            qf, kf, vf, cu_seqlens_q, offsets_k, indices, work_indptr, work_info, max_q, sm_scale, True,
            logits, lse, out, one, one, one,
        )  # fmt: skip
        mla_reduce_v1(logits, lse, reduce_indptr, reduce_final_map, reduce_partial_map, tile_q, 0, out, None)
        output.copy_(out[:, :heads])

    return prefill


def mla_prefill_varlen_lse_baseline(
    q,
    k,
    v,
    cu_seqlens_q,
    qkv_fp8=True,
    causal=True,
    cu_seqlens_k=None,
    sm_scale=None,
    output=None,
    lse=None,
    **_,
):
    """sglang 0.5.23's chunked-prefix passes on aiter (``_mla_fp8_prefill_attn_ps``): with qkv_fp8 its fp8 path
    (the heads tiled up to one aiter's reduce instantiates, q, k and v cast to fp8, the ASM prefill over aiter's
    persistent plan built with need_lse, then mla_reduce_v1 writing the lse), else aiter's flash attention in bf16
    with return_lse. The plan is made here as aiter's planner builds it (ps_plan): only aiter 0.1.23 on has
    need_lse. On an older aiter, whose reduce skips a tile with one partial, that partial (the tile's result and
    lse as the reduce would write them) is copied into the outputs."""
    from aiter import get_ps_metadata_info_v1, get_ps_metadata_v1, mla_prefill_ps_asm_fwd, mla_reduce_v1

    if output is None or lse is None or sm_scale is None:
        return None
    heads, v_dim = q.shape[1], v.shape[-1]
    offsets_k = cu_seqlens_q if cu_seqlens_k is None else cu_seqlens_k
    if not qkv_fp8:
        from aiter import flash_attn_varlen_func

        def flash():
            out, row_lse = flash_attn_varlen_func(
                q, k, v, cu_seqlens_q, offsets_k, q.shape[0], k.shape[0], softmax_scale=sm_scale, causal=causal,
                return_lse=True,
            )[:2]  # fmt: skip
            output.copy_(out)
            lse.copy_(row_lse.transpose(0, 1))

        return flash
    tile_q = 256  # mla_reduce_v1's query tile
    padded = min(h for h in (1, 2, 4, 8, 10, 16, 32, 40, 64, 128) if h >= heads)
    qo, kv = cu_seqlens_q.tolist(), offsets_k.tolist()
    max_q = max(b - a for a, b in zip(qo, qo[1:]))
    work_indptr, works, reduce_indptr, final, partials = ps_plan(
        qo,
        kv,
        [b - a for a, b in zip(kv, kv[1:])],
        padded,
        num_cus=torch.cuda.get_device_properties(q.device).multi_processor_count,
        qlen_granularity=tile_q,
        kvlen_granularity=128,
        block_size=1,
        is_causal=causal,
    )
    # whether aiter's reduce takes a query tile with one partial: it does where the planner takes need_lse
    rows = torch.tensor([0, 1], dtype=torch.int32)
    probe = [torch.zeros(size, dtype=dtype) for size, dtype in get_ps_metadata_info_v1(1, 1, 1, tile_q)]
    try:
        get_ps_metadata_v1(
            rows, rows, rows.diff(), 1, 1, *probe, qlen_granularity=tile_q, kvlen_granularity=128,
            block_size=1, need_lse=True,
        )  # fmt: skip
        whole = []
    except (TypeError, RuntimeError) as e:  # torch's op dispatch raises a RuntimeError on the keyword
        if "need_lse" not in str(e):
            raise
        whole = [
            (start, end, partials[reduce_indptr[t]])
            for t, (start, end) in enumerate(final)
            if reduce_indptr[t + 1] - reduce_indptr[t] == 1
        ]
    dst = src = None
    if whole:
        dst = torch.cat([torch.arange(start, end) for start, end, _ in whole]).to(q.device)
        src = torch.cat([torch.arange(row, row + end - start) for start, end, row in whole]).to(q.device)
    # the reduce over as many query tiles as aiter sizes it for, the unused ones empty
    tiles = (len(qo) - 1) * -(-max_q // tile_q)
    reduce_indptr += [reduce_indptr[-1]] * (tiles + 1 - len(reduce_indptr))
    final += [[0, 0]] * (tiles - len(final))
    work_indptr, work_info, reduce_indptr, reduce_final_map, reduce_partial_map = (
        torch.tensor(x, dtype=torch.int32).to(q.device)
        for x in (work_indptr, works, reduce_indptr, final, partials)
    )
    indices = torch.arange(k.shape[0], dtype=torch.int32, device=q.device)
    one = torch.ones((), dtype=torch.float32, device=q.device)
    partial = max(len(partials), 1) * tile_q

    def prefill():
        qf, kf, vf = (
            (x.repeat(1, -(-padded // heads), 1)[:, :padded].contiguous() if padded != heads else x).to(
                torch.float8_e4m3fn
            )
            for x in (q, k, v)
        )
        logits = torch.empty((partial, padded, v_dim), dtype=torch.float32, device=q.device)
        part_lse = torch.empty((partial, padded), dtype=torch.float32, device=q.device)
        out = torch.empty((q.shape[0], padded, v_dim), dtype=q.dtype, device=q.device)
        final_lse = torch.empty((q.shape[0], padded), dtype=torch.float32, device=q.device)
        mla_prefill_ps_asm_fwd(
            qf, kf, vf, cu_seqlens_q, offsets_k, indices, work_indptr, work_info, max_q, sm_scale, causal,
            logits, part_lse, out, one, one, one,
        )  # fmt: skip
        mla_reduce_v1(
            logits, part_lse, reduce_indptr, reduce_final_map, reduce_partial_map, tile_q, 0, out, final_lse
        )
        if dst is not None:
            out.index_copy_(0, dst, logits.index_select(0, src).to(out.dtype))
            final_lse.index_copy_(0, dst, part_lse.index_select(0, src))
        output.copy_(out[:, :heads])
        lse.copy_(final_lse[:, :heads])

    return prefill
