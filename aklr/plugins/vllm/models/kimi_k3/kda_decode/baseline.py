# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What vLLM runs: its Triton ``fused_recurrent_kda`` (``models/kimi_k3/amd/ops/third_party/kda``), on the operands as
``kda.py`` hands them over for a speculative batch: q, k and v as their [1, n, heads, dim] views of the packed q|k|v
rows, the gate and beta with vLLM's leading batch dim, the state cache as (slot, head, v, k), the windows' slots as
[n_seqs, window] rows; it writes vLLM's output, as for a speculative batch alone."""

from aklr.core.contract.baseline import OpBaseline

__all__ = ["KimiK3KdaDecodeSpecBaseline"]

# each windows' offsets (vLLM's spec_query_start_loc), by device, sequence count and window: made once, not in a timed
# call
_OFFSETS: dict = {}


class KimiK3KdaDecodeSpecBaseline(OpBaseline):
    def baseline(
        self,
        *,
        qkv,
        raw_g,
        raw_beta,
        a_log,
        g_bias,
        state_cache,
        state_indices,
        num_accepted_tokens,
        storage="row_major",
        scale=128**-0.5,
        lower_bound=-5.0,
        out_dtype=None,
        out=None,
    ):
        """vLLM's out [n_tokens, n_heads, head_dim] of a speculative batch with windows of one length (vLLM's
        uniform_sequence_length); state_cache is written in place, as in vLLM."""
        import torch
        from vllm.models.kimi_k3.amd.ops.third_party.kda import fused_recurrent_kda

        n_tokens, heads, dim = raw_g.shape
        n_seqs = num_accepted_tokens.shape[0]
        window = n_tokens // n_seqs
        key = (qkv.device, n_seqs, window)
        if key not in _OFFSETS:
            _OFFSETS[key] = torch.arange(n_seqs + 1, dtype=torch.int32, device=qkv.device) * window
        q, k, v = (x[None] for x in qkv.unflatten(1, (3, heads)).unbind(1))
        if out is None:
            out = torch.empty(raw_g.shape, dtype=qkv.dtype, device=qkv.device)
        fused_recurrent_kda(
            q=q,
            k=k,
            v=v,
            raw_g=raw_g[None],
            raw_beta=raw_beta[None, :, :, 0],
            A_log=a_log,
            dt_bias=g_bias.flatten(),
            lower_bound=lower_bound,
            initial_state=state_cache.unflatten(1, (heads, dim)),
            cu_seqlens=_OFFSETS[key],
            ssm_state_indices=state_indices.view(n_seqs, window),
            num_accepted_tokens=num_accepted_tokens,
            uniform_sequence_length=window,
            out=out[None],
        )
        return {"out": out}
