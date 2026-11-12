# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.


def sparse_prefill_mxfp8_baseline(
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
    *,
    softmax_scale=512**-0.5,
    output=None,
):
    import torch
    from aiter.ops.pa_sparse_prefill_opus import pa_sparse_prefill_fp8_opus

    return pa_sparse_prefill_fp8_opus(
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


# aiter's fp8 prefill plan for a call's offsets, by the tensors' ids while they live: sglang plans it
# once a forward for every MLA layer, not at each call
_PLANS: dict[tuple[int, int], tuple] = {}
_TILE_Q = 256  # mla_reduce_v1's query tile
# head counts mla_reduce_v1 instantiates at v_head_dim 128 (aiter's reduce.cu); sglang tiles the query
# heads up to the next one
_REDUCE_HEADS = (1, 2, 4, 8, 10, 16, 32, 40, 64, 128)


def mla_prefill_varlen_baseline(
    q, k, v, cu_seqlens_q, qkv_fp8=True, cu_seqlens_k=None, sm_scale=None, output=None
):
    """sglang's MLA prefill on aiter: with qkv_fp8 its fp8 path (the heads tiled up to one aiter's
    reduce instantiates, q, k and v cast to fp8, the ASM prefill over aiter's persistent plan, then
    mla_reduce_v1), else aiter's flash attention in bf16."""
    import weakref

    import torch
    from aiter import get_ps_metadata_info_v1, get_ps_metadata_v1, mla_prefill_ps_asm_fwd, mla_reduce_v1

    heads, v_dim = q.shape[1], v.shape[-1]
    offsets_k = cu_seqlens_q if cu_seqlens_k is None else cu_seqlens_k
    if not qkv_fp8:
        from aiter import flash_attn_varlen_func

        # the row counts bound the longest sequence's without reading the offsets
        return flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            offsets_k,
            q.shape[0],
            k.shape[0],
            softmax_scale=sm_scale,
            causal=True,
            out=output,
        )
    padded = min(h for h in _REDUCE_HEADS if h >= heads)
    key = (id(cu_seqlens_q), id(offsets_k))
    plan = _PLANS.get(key)
    if plan is None:
        qo_cpu, kv_cpu = cu_seqlens_q.to("cpu", torch.int32), offsets_k.to("cpu", torch.int32)
        sizes = get_ps_metadata_info_v1(
            batch_size=qo_cpu.numel() - 1,
            num_head_k=padded,
            max_qlen=int(qo_cpu.diff().max()),
            qlen_granularity=_TILE_Q,
        )
        buffers = [torch.empty(size, dtype=dtype, device=q.device) for size, dtype in sizes]
        work_metadata, work_indptr, work_info, reduce_indptr, reduce_final_map, reduce_partial_map = buffers
        get_ps_metadata_v1(
            qo_cpu,
            kv_cpu,
            kv_cpu.diff(),
            1,
            padded,
            work_metadata,
            work_indptr,
            work_info,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
            qhead_granularity=1,
            qlen_granularity=_TILE_Q,
            kvlen_granularity=128,
            block_size=1,
            is_causal=True,
        )
        indices = torch.arange(k.shape[0], dtype=torch.int32, device=q.device)
        plan = _PLANS[key] = (int(qo_cpu.diff().max()), indices, buffers)
        for tensor in (cu_seqlens_q, offsets_k):
            weakref.finalize(tensor, _PLANS.pop, key, None)
    max_q, indices, (_, work_indptr, work_info, reduce_indptr, reduce_final_map, reduce_partial_map) = plan

    def tiled(x):
        return x.repeat(1, -(-padded // heads), 1)[:, :padded].contiguous() if padded != heads else x

    fp8 = torch.float8_e4m3fn
    qf, kf, vf = (tiled(x).to(fp8) for x in (q, k, v))
    one = torch.ones((), dtype=torch.float32, device=q.device)
    partial = max(reduce_partial_map.size(0), 1) * _TILE_Q
    logits = torch.empty((partial, padded, v_dim), dtype=torch.float32, device=q.device)
    lse = torch.empty((partial, padded), dtype=torch.float32, device=q.device)
    out = torch.empty((q.shape[0], padded, v_dim), dtype=q.dtype, device=q.device)
    mla_prefill_ps_asm_fwd(
        qf, kf, vf, cu_seqlens_q, offsets_k, indices, work_indptr, work_info, max_q, sm_scale, True,
        logits, lse, out, one, one, one,
    )  # fmt: skip
    mla_reduce_v1(logits, lse, reduce_indptr, reduce_final_map, reduce_partial_map, _TILE_Q, 0, out, None)
    out = out[:, :heads]
    if output is not None:
        output.copy_(out)
        return output
    return out


# aiter's fp8 prefill plan with every query tile reduced (need_lse), per call's offsets and causal mask
_LSE_PLANS: dict[tuple[int, int, bool], tuple] = {}
_REDUCES_WHOLE_TILES: list[bool] = []


def _aiter_reduces_whole_tiles() -> bool:
    """Whether aiter's mla_reduce_v1 reduces a query tile with one partial. It came with the planner's need_lse
    (aiter 0.1.23 on); before, the reduce skips such a tile, whose output and lse are then never written."""
    if not _REDUCES_WHOLE_TILES:
        import torch
        from aiter import get_ps_metadata_info_v1, get_ps_metadata_v1

        rows = torch.tensor([0, 1], dtype=torch.int32)
        buffers = [
            torch.zeros(size, dtype=dtype) for size, dtype in get_ps_metadata_info_v1(1, 1, 1, _TILE_Q)
        ]
        try:
            get_ps_metadata_v1(
                rows, rows, rows.diff(), 1, 1, *buffers, qlen_granularity=_TILE_Q, kvlen_granularity=128,
                block_size=1, need_lse=True,
            )  # fmt: skip
            _REDUCES_WHOLE_TILES.append(True)
        except (TypeError, RuntimeError) as e:  # torch's op dispatch raises a RuntimeError on the keyword
            if "need_lse" not in str(e):
                raise
            _REDUCES_WHOLE_TILES.append(False)
    return _REDUCES_WHOLE_TILES[0]


def mla_prefill_varlen_lse_baseline(
    q, k, v, cu_seqlens_q, qkv_fp8=True, causal=True, cu_seqlens_k=None, sm_scale=None, output=None, lse=None
):
    """sglang 0.5.22's chunked-prefix passes on aiter (``_mla_fp8_prefill_attn_ps``): with qkv_fp8 its fp8 path
    (the heads tiled up to one aiter's reduce instantiates, q, k and v cast to fp8, the ASM prefill over aiter's
    persistent plan built with need_lse, then mla_reduce_v1 writing the lse), else aiter's flash attention in bf16
    with return_lse. The plan is built as aiter's planner builds it (``ps_plan``): only aiter 0.1.23 on has
    need_lse. On an older aiter, whose reduce skips a tile with one partial, that partial (the tile's result and
    lse as the reduce would write them) is copied into the outputs."""
    import weakref

    import torch
    from aiter import mla_prefill_ps_asm_fwd, mla_reduce_v1

    from aklr.ops.mla_prefill.ps_plan import ps_plan

    heads, v_dim = q.shape[1], v.shape[-1]
    offsets_k = cu_seqlens_q if cu_seqlens_k is None else cu_seqlens_k
    if not qkv_fp8:
        from aiter import flash_attn_varlen_func

        out, row_lse = flash_attn_varlen_func(
            q, k, v, cu_seqlens_q, offsets_k, q.shape[0], k.shape[0], softmax_scale=sm_scale, causal=causal,
            return_lse=True,
        )[:2]  # fmt: skip
        row_lse = row_lse.transpose(0, 1)
        return _into(out, row_lse, output, lse)
    padded = min(h for h in _REDUCE_HEADS if h >= heads)
    key = (id(cu_seqlens_q), id(offsets_k), causal)
    plan = _LSE_PLANS.get(key)
    if plan is None:
        qo, kv = cu_seqlens_q.tolist(), offsets_k.tolist()
        max_q = max(b - a for a, b in zip(qo, qo[1:]))
        work_indptr, works, reduce_indptr, final, partials = ps_plan(
            qo,
            kv,
            [b - a for a, b in zip(kv, kv[1:])],
            padded,
            num_cus=torch.cuda.get_device_properties(q.device).multi_processor_count,
            qlen_granularity=_TILE_Q,
            kvlen_granularity=128,
            block_size=1,
            is_causal=causal,
        )
        fill = None
        if not _aiter_reduces_whole_tiles():
            whole = [
                (start, end, partials[reduce_indptr[t]])
                for t, (start, end) in enumerate(final)
                if reduce_indptr[t + 1] - reduce_indptr[t] == 1
            ]
            if whole:
                dst = torch.cat([torch.arange(start, end) for start, end, _ in whole])
                src = torch.cat([torch.arange(row, row + end - start) for start, end, row in whole])
                fill = (dst.to(q.device), src.to(q.device))
        # the reduce over as many query tiles as aiter sizes it for, the unused ones empty
        tiles = (len(qo) - 1) * -(-max_q // _TILE_Q)
        reduce_indptr += [reduce_indptr[-1]] * (tiles + 1 - len(reduce_indptr))
        final += [[0, 0]] * (tiles - len(final))
        buffers = [
            torch.tensor(x, dtype=torch.int32).to(q.device)
            for x in (work_indptr, works, reduce_indptr, final, partials)
        ]
        indices = torch.arange(k.shape[0], dtype=torch.int32, device=q.device)
        plan = _LSE_PLANS[key] = (max_q, indices, len(partials), buffers, fill)
        for tensor in (cu_seqlens_q, offsets_k):
            weakref.finalize(tensor, _LSE_PLANS.pop, key, None)
    (
        max_q,
        indices,
        n_partial,
        (work_indptr, work_info, reduce_indptr, reduce_final_map, reduce_partial_map),
        fill,
    ) = plan

    def tiled(x):
        return x.repeat(1, -(-padded // heads), 1)[:, :padded].contiguous() if padded != heads else x

    fp8 = torch.float8_e4m3fn
    qf, kf, vf = (tiled(x).to(fp8) for x in (q, k, v))
    one = torch.ones((), dtype=torch.float32, device=q.device)
    partial = max(n_partial, 1) * _TILE_Q
    logits = torch.empty((partial, padded, v_dim), dtype=torch.float32, device=q.device)
    part_lse = torch.empty((partial, padded), dtype=torch.float32, device=q.device)
    out = torch.empty((q.shape[0], padded, v_dim), dtype=q.dtype, device=q.device)
    final_lse = torch.empty((q.shape[0], padded), dtype=torch.float32, device=q.device)
    mla_prefill_ps_asm_fwd(
        qf, kf, vf, cu_seqlens_q, offsets_k, indices, work_indptr, work_info, max_q, sm_scale, causal,
        logits, part_lse, out, one, one, one,
    )  # fmt: skip
    mla_reduce_v1(
        logits, part_lse, reduce_indptr, reduce_final_map, reduce_partial_map, _TILE_Q, 0, out, final_lse
    )
    if fill is not None:
        dst, src = fill
        out.index_copy_(0, dst, logits.index_select(0, src).to(out.dtype))
        final_lse.index_copy_(0, dst, part_lse.index_select(0, src))
    return _into(out[:, :heads], final_lse[:, :heads], output, lse)


def _into(out, row_lse, output, lse):
    """The results, copied into the outputs the caller passed."""
    if output is not None:
        output.copy_(out)
    if lse is not None:
        lse.copy_(row_lse)
    return (
        output if output is not None else out.contiguous(),
        lse if lse is not None else row_lse.contiguous(),
    )
