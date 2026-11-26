# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What vLLM 0.34.0 runs at the call the route takes: ``chunk_kda_prefill`` (``models/kimi_k3/amd/ops/kda_prefill``)
on the fused backend, as ``kda.py`` calls it with the paged state: vLLM's ``fused_kda_prologue`` and
``fused_kda_chunk`` kernels (its ``_C``), reading and writing each sequence's cache row in place. The operands are
handed over in vLLM's shapes: q, k, v, the gate and beta with its leading batch dim, the gate bias flat, the cache as
(slot, head, v, k). The chunk index tables kda.py takes from vLLM's metadata come from the same helpers inside
``chunk_kda_prefill``, which read cu_seqlens on the host once and keep them for that tensor (the first call runs outside
a CUDA graph, as the bench's does). Without an ``out``, vLLM writes the output into v, as in vLLM.

vLLM's fused kernels are built into its ``_C`` for gfx950: there is no copy that runs without vLLM, and the baseline
raises where vLLM, or its fused kernels, are missing (``chunk_kda_prefill`` would quietly run its Triton chunk path
instead, which no call the route takes runs)."""

from aklr.core.contract.baseline import OpBaseline
from aklr.plugins.vllm.models.kimi_k3.kda_prefill.reference import CHECKPOINT_CHUNK

__all__ = ["KimiK3KdaPrefillBaseline", "KimiK3KdaPrefillCheckpointBaseline"]


def _chunk_kda_prefill(q, k, v, raw_g, raw_beta, a_log, g_bias, state_cache, state_indices, has_initial_state,
                       cu_seqlens, scale, lower_bound, out, **checkpoint):  # fmt: skip
    """vLLM's chunk_kda_prefill on the fused backend with the paged state, its output [n_tokens, heads, dim]."""
    from vllm.models.kimi_k3.amd.ops.kda_chunk import can_use_fused_kda_chunk
    from vllm.models.kimi_k3.amd.ops.kda_prefill import chunk_kda_prefill
    from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE

    heads, dim = q.shape[1], q.shape[2]
    if not can_use_fused_kda_chunk(dim, v.shape[-1], k.dtype, FLA_CHUNK_SIZE):
        raise RuntimeError(
            "vLLM's fused KDA chunk kernels (torch.ops._C.fused_kda_chunk, gfx950) are not here"
        )
    o, _ = chunk_kda_prefill(
        q=q[None],
        k=k[None],
        v=v[None],
        raw_g=raw_g[None],
        raw_beta=raw_beta[None],
        A_log=a_log,
        g_bias=g_bias.flatten(),
        scale=scale,
        lower_bound=lower_bound,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
        backend="fused",
        out=None if out is None else out[None],
        state_cache=state_cache.unflatten(1, (heads, dim)),
        state_indices=state_indices,
        has_initial_state=has_initial_state,
        **checkpoint,
    )
    return o[0]


class KimiK3KdaPrefillBaseline(OpBaseline):
    def baseline(
        self,
        *,
        q,
        k,
        v,
        raw_g,
        raw_beta,
        a_log,
        g_bias,
        state_cache,
        state_indices,
        has_initial_state,
        cu_seqlens,
        state_storage="row_major",
        scale=128**-0.5,
        lower_bound=-5.0,
        out_dtype=None,
        out=None,
    ):
        """vLLM's output [n_tokens, n_heads, head_dim]; each sequence's last state is written to its cache row."""
        o = _chunk_kda_prefill(q, k, v, raw_g, raw_beta, a_log, g_bias, state_cache, state_indices,
                               has_initial_state, cu_seqlens, scale, lower_bound, out)  # fmt: skip
        return {"out": o}


class KimiK3KdaPrefillCheckpointBaseline(OpBaseline):
    def baseline(
        self,
        *,
        q,
        k,
        v,
        raw_g,
        raw_beta,
        a_log,
        g_bias,
        state_cache,
        state_indices,
        has_initial_state,
        cu_seqlens,
        checkpoint_chunks,
        checkpoint_state,
        state_storage="row_major",
        scale=128**-0.5,
        lower_bound=-5.0,
        out_dtype=None,
        out=None,
    ):
        """As vLLM takes a snapshot mid-prefill: the same call with checkpoint_state, one row a sequence (vLLM's
        staging form, with no checkpoint_state_indices), and the snapshot offsets in tokens, as vLLM's metadata holds
        them (one elementwise launch from the chunks); vLLM's kernel writes the snapshot rows it takes."""
        heads, dim = q.shape[1], q.shape[2]
        o = _chunk_kda_prefill(q, k, v, raw_g, raw_beta, a_log, g_bias, state_cache, state_indices,
                               has_initial_state, cu_seqlens, scale, lower_bound, out,
                               checkpoint_state=checkpoint_state.unflatten(1, (heads, dim)),
                               checkpoint_offsets=checkpoint_chunks * CHECKPOINT_CHUNK)  # fmt: skip
        return {"out": o}
