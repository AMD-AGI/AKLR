# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import weakref

import torch

from aklr.core.contract.baseline import OpBaseline

__all__ = [
    "FlashMlaWithKvcacheBaseline",
    "Fp8PagedMqaLogitsBaseline",
    "Fp8PagedMqaLogitsVerifyBaseline",
    "MlaDecodeFwdBaseline",
    "MlaDecodeFwdCprrBaseline",
]

# aiter's work schedule for a kv_indptr, by the tensor's id while it lives: a framework plans
# it once a step for every MLA layer, not at each call
_SCHEDULES = {}
_NUM_KV_SPLITS = 64  # sglang's decode without speculation


class FlashMlaWithKvcacheBaseline(OpBaseline):
    def baseline(
        self,
        *,
        q,
        kv_cache,
        sparse_indices=None,
        sparse_lens=None,
        extra_kv_cache=None,
        extra_sparse_indices=None,
        extra_sparse_lens=None,
        block_table=None,
        cache_seqlens=None,
        cu_seqlens_k=None,
        causal=False,
        head_dim_v=None,
        sm_scale=None,
        sinks=None,
        kv_cache_format="dsv4_fp8",
        kv_scale=1.0,
        return_lse=False,
        kv_cache_layout="row_major",
        output=None,
        lse=None,
    ):
        """Sparse: sglang's kernel, as the route calls it. Dense: aiter's Gluon MLA decode, the
        kernel sglang serves the DSPARK verify prefix with, over the 2-D block table read up to
        cache_seqlens or over the flat one cut by cu_seqlens_k; several causal queries per row
        take its multi-token path. It writes into the caller's output and lse when it passes
        them."""
        if kv_cache_format == "dsv4_fp8":
            from sglang.kernels.ops.attention.dsv4.tilelang_fp8_attention import (
                dpsk_v4_fp8_attention_fwd,
            )

            output, lse = dpsk_v4_fp8_attention_fwd(
                q,
                kv_cache,
                None,
                None,
                None,
                None,
                softmax_scale=sm_scale,
                indices=sparse_indices,
                attn_sink=sinks,
                extra_k_cache=extra_kv_cache,
                extra_indices_in_kvcache=extra_sparse_indices,
                topk_length=sparse_lens,
                extra_topk_length=extra_sparse_lens,
            )
            return {"output": output, "lse": lse}
        if block_table is None or sinks is not None:
            raise ValueError("dense baseline: a block table and no sinks")
        from aiter.ops.triton.gluon.mla_gluon import mla_gluon

        batch, query_length, n_heads, dim = q.shape
        if query_length != 1 and not causal:
            raise ValueError("dense baseline: several queries per row only causally")
        if output is None:
            output = q.new_empty(batch, query_length, n_heads, head_dim_v)
        q_nope, q_pe, out = q[..., :head_dim_v], q[..., head_dim_v:], output
        if query_length == 1:
            # one query per row runs the plain decode; several, the causal multi-token path
            q_nope, q_pe = q_nope[:, 0], q_pe[:, 0]
            out = output.view(batch, n_heads, head_dim_v)
        _, row_lse = mla_gluon(
            q_nope,
            q_pe,
            kv_cache.view(-1, dim),
            out,
            block_table,
            cache_seqlens if cu_seqlens_k is None else cu_seqlens_k,
            sm_scale,
            kv_pe_offset=head_dim_v,
            use_2d_view=cu_seqlens_k is None,
            kv_scale=kv_scale,
            return_lse=True,
        )
        row_lse = row_lse.reshape(batch, query_length, n_heads)
        if lse is None:
            lse = row_lse
        else:
            lse.copy_(row_lse)
        return {"output": output, "lse": lse}


def _aiter_paged_mqa_logits(q, kv_cache, weights, context_lens, block_tables, logits, page_size, preshuffled):
    """aiter's kernel into ``logits``, with the max_model_len the route drops: the logits' width."""
    from aiter.ops.triton.attention.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits

    deepgemm_fp8_paged_mqa_logits(
        q,
        kv_cache,
        weights,
        logits,
        context_lens,
        block_tables,
        logits.shape[1],
        Preshuffle=preshuffled,
        KVBlockSize=page_size,
    )


class Fp8PagedMqaLogitsBaseline(OpBaseline):
    def baseline(self, *, q, kv_cache, weights, context_lens, block_tables, logits, page_size, preshuffled):
        """aiter's kernel."""
        _aiter_paged_mqa_logits(
            q, kv_cache, weights, context_lens, block_tables, logits, page_size, preshuffled
        )
        return {"logits": logits}


class Fp8PagedMqaLogitsVerifyBaseline(OpBaseline):
    def baseline(self, *, q, kv_cache, weights, context_lens, block_tables, logits, page_size, preshuffled):
        """sglang's verify logits with its kernel off: aiter's kernel a query row at a time, over the
        block table row-expanded as sglang passes it. The expansion, a copy sglang makes once a
        step, is timed with each call here."""
        batch, next_n = q.shape[:2]
        _aiter_paged_mqa_logits(
            q.flatten(0, 1).unsqueeze(1),
            kv_cache,
            weights.flatten(0, 1),
            context_lens.flatten(),
            block_tables.repeat_interleave(next_n, dim=0),
            logits.view(batch * next_n, -1),
            page_size,
            preshuffled,
        )
        return {"logits": logits}


class MlaDecodeFwdBaseline(OpBaseline):
    def baseline(
        self, *, q, kv_buffer, kv_indptr, kv_indices, q_scale, kv_scale, sm_scale=None, o=None, lse=None
    ):
        """aiter's ASM MLA decode as sglang calls it under decode context parallelism: fp8 q and
        pool, its persistent schedule (fast mode, 64 splits a request) planned once per kv_indptr."""
        from aiter.mla import mla_decode_fwd
        from aiter.ops.attention import get_mla_metadata_info_v1, get_mla_metadata_v1

        batch, n_heads = q.shape[:2]
        if o is None:
            o = torch.empty((batch, n_heads, 512), dtype=torch.bfloat16, device=q.device)
        # planned again for an offsets tensor rewritten in place, other heads, dtypes or device
        key = (id(kv_indptr), kv_indptr._version, n_heads, q.dtype, kv_buffer.dtype, q.device)
        schedule = _SCHEDULES.get(key)
        if schedule is None:
            qo_indptr = torch.arange(batch + 1, dtype=torch.int32, device=q.device)
            last_page_lens = torch.ones(batch, dtype=torch.int32, device=q.device)
            sizes = get_mla_metadata_info_v1(
                batch,
                1,
                n_heads,
                q.dtype,
                kv_buffer.dtype,
                is_sparse=False,
                fast_mode=True,
                num_kv_splits=_NUM_KV_SPLITS,
                intra_batch_mode=False,
            )
            buffers = [torch.empty(size, dtype=dtype, device=q.device) for size, dtype in sizes]
            work_metadata, work_indptr, work_info_set, reduce_indptr, reduce_final_map, reduce_partial_map = (
                buffers
            )
            get_mla_metadata_v1(
                qo_indptr,
                kv_indptr,
                last_page_lens,
                n_heads,
                1,
                False,
                work_metadata,
                work_info_set,
                work_indptr,
                reduce_indptr,
                reduce_final_map,
                reduce_partial_map,
                kv_granularity=16,
                max_seqlen_qo=1,
                uni_seqlen_qo=1,
                fast_mode=True,
                max_split_per_batch=_NUM_KV_SPLITS,
                intra_batch_mode=False,
                dtype_q=q.dtype,
                dtype_kv=kv_buffer.dtype,
            )
            schedule = _SCHEDULES[key] = (qo_indptr, last_page_lens, buffers)
            weakref.finalize(kv_indptr, _SCHEDULES.pop, key, None)
        qo_indptr, last_page_lens, buffers = schedule
        work_metadata, work_indptr, work_info_set, reduce_indptr, reduce_final_map, reduce_partial_map = (
            buffers
        )
        _, final_lse = mla_decode_fwd(
            q,
            kv_buffer,
            o,
            qo_indptr,
            kv_indptr,
            kv_indices,
            last_page_lens,
            1,
            sm_scale=sm_scale,
            num_kv_splits=_NUM_KV_SPLITS,
            work_meta_data=work_metadata,
            work_indptr=work_indptr,
            work_info_set=work_info_set,
            reduce_indptr=reduce_indptr,
            reduce_final_map=reduce_final_map,
            reduce_partial_map=reduce_partial_map,
            q_scale=q_scale,
            kv_scale=kv_scale,
            intra_batch_mode=False,
            return_lse=True,
        )
        final_lse = final_lse.view(batch, n_heads)
        if lse is None:
            return {"o": o, "lse": final_lse}
        lse.copy_(final_lse)
        return {"o": o, "lse": lse}


class MlaDecodeFwdCprrBaseline(OpBaseline):
    def baseline(
        self,
        *,
        q,
        kv_buffer,
        kv_indptr,
        kv_indices,
        g_kv_indptr,
        q_scale,
        kv_scale,
        cp_world_size=None,
        sm_scale=None,
        cp_rank=0,
        o=None,
        lse=None,
    ):
        """aiter's ASM MLA decode as vLLM's aiter MLA backend verifies draft tokens under decode context
        parallelism (its cprr route): fp8 q and pool, causal over global positions, the persistent
        schedule vLLM's metadata builder plans once a step for every MLA layer (fast mode, round robin,
        as many splits a request as the GPU has CUs), planned here once per kv_indptr."""
        from aiter.mla import mla_decode_fwd
        from aiter.ops.attention import get_mla_metadata_info_v1, get_mla_metadata_v1

        batch, query_length, n_heads, dim = q.shape
        rows = batch * query_length
        if o is None:
            o = torch.empty((batch, query_length, n_heads, 512), dtype=torch.bfloat16, device=q.device)
        # planned again for an offsets tensor rewritten in place, other queries, heads, dtypes or device
        key = (
            id(kv_indptr),
            kv_indptr._version,
            "cprr",
            query_length,
            n_heads,
            q.dtype,
            kv_buffer.dtype,
            q.device,
        )
        schedule = _SCHEDULES.get(key)
        if schedule is None:
            # vLLM's split cap: the CU count (AiterMLAMetadataBuilder._mla_max_split_per_batch)
            splits = torch.cuda.get_device_properties(q.device).multi_processor_count
            qo_indptr = torch.arange(batch + 1, dtype=torch.int32, device=q.device) * query_length
            last_page_lens = torch.ones(batch, dtype=torch.int32, device=q.device)
            sizes = get_mla_metadata_info_v1(
                batch,
                query_length,
                n_heads,
                q.dtype,
                kv_buffer.dtype,
                is_sparse=False,
                fast_mode=True,
                max_split_per_batch=splits,
            )
            buffers = [torch.empty(size, dtype=dtype, device=q.device) for size, dtype in sizes]
            work_metadata, work_indptr, work_info_set, reduce_indptr, reduce_final_map, reduce_partial_map = (
                buffers
            )
            # the round-robin schedule is not causal: the kernel masks by global position
            get_mla_metadata_v1(
                qo_indptr,
                kv_indptr,
                last_page_lens,
                n_heads,
                1,
                False,
                work_metadata,
                work_info_set,
                work_indptr,
                reduce_indptr,
                reduce_final_map,
                reduce_partial_map,
                page_size=1,
                kv_granularity=16,
                max_seqlen_qo=query_length,
                uni_seqlen_qo=query_length,
                fast_mode=True,
                max_split_per_batch=splits,
                intra_batch_mode=False,
                is_cp_round_robin=True,
                dtype_q=q.dtype,
                dtype_kv=kv_buffer.dtype,
            )
            schedule = _SCHEDULES[key] = (qo_indptr, last_page_lens, splits, buffers)
            weakref.finalize(kv_indptr, _SCHEDULES.pop, key, None)
        qo_indptr, last_page_lens, splits, buffers = schedule
        work_metadata, work_indptr, work_info_set, reduce_indptr, reduce_final_map, reduce_partial_map = (
            buffers
        )
        _, final_lse = mla_decode_fwd(
            q.view(rows, n_heads, dim),
            kv_buffer,
            o.view(rows, n_heads, 512),
            qo_indptr,
            kv_indptr,
            kv_indices,
            last_page_lens,
            query_length,
            sm_scale=sm_scale,
            num_kv_splits=splits,
            work_meta_data=work_metadata,
            work_indptr=work_indptr,
            work_info_set=work_info_set,
            reduce_indptr=reduce_indptr,
            reduce_final_map=reduce_final_map,
            reduce_partial_map=reduce_partial_map,
            q_scale=q_scale,
            kv_scale=kv_scale,
            intra_batch_mode=False,
            return_lse=True,
            g_kv_indptr=g_kv_indptr,
            cp_world_size=cp_world_size,
            cp_rank=cp_rank,
            causal=True,
        )
        final_lse = final_lse.view(batch, query_length, n_heads)
        if lse is None:
            return {"o": o, "lse": final_lse}
        lse.copy_(final_lse)
        return {"o": o, "lse": lse}
