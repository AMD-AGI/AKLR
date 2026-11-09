# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.runtime.dispatch import FallbackReason, api
from aklr.plugins.sglang.hook import sglang_route
from aklr.plugins.sglang.models.moe_transport.contract import (
    DEEPSEEK_V4_DP_MOE_GATHER,
)

__all__ = ["deepseek_v4_dp_moe_gather"]


@api(DEEPSEEK_V4_DP_MOE_GATHER)
def deepseek_v4_dp_moe_gather(hidden_states, num_token_non_padded, out_dtype=None, out=None): ...


@sglang_route(
    "sglang.srt.models.deepseek_common.moe_transport.MoeRowTransport.__init__", versions=("0.5.23",)
)
def _moe_row_transport_init(
    self,
    moe,
    hidden_states,
    forward_batch,
    input_ids,
    input_ids_global,
    *,
    cp_return_contiguous=False,
    use_cp=None,
    moe_gather=None,
    attn_scatter=None,
):
    """The TP-MoE gather of a DP-attention rank's rows on ROCm through aiter's custom
    all-gather, with the padding-row mask and aiter's copy into its registered input buffer as
    one op call writing that buffer, then aiter's all_gather_reg from it; the transport's fields
    as sglang sets them. Every other path (CP, a2a backends, attention TP, the variable-length
    gather, a shared expert run locally, any gather but aiter's copy-in one) stays on sglang.

    Relies on aiter's non-public CustomAllreduce members _pool["input"] (the registered input
    buffer, here torch-backed), _IS_CAPTURING and all_gather_reg, as sglang 0.5.23's
    GroupCoordinator._all_gather_into_tensor does."""
    from sglang.srt.models.deepseek_common import moe_transport

    if _transport_falls_back(moe_transport, moe, forward_batch, use_cp, moe_gather, attn_scatter):
        return FallbackReason.OUT_OF_CONTRACT
    tp_group = moe_transport.get_parallel().tp_group
    gather = _aiter_copy_in_gather(tp_group, hidden_states)
    if gather is None:
        return FallbackReason.OUT_OF_CONTRACT
    ca_comm, pool_input, warmup = gather
    out = deepseek_v4_dp_moe_gather(
        hidden_states=hidden_states,
        num_token_non_padded=forward_batch.num_token_non_padded,
        out=pool_input,
    )
    if isinstance(out, FallbackReason):
        return out
    from sglang.srt.layers import dp_attention

    global_hidden_states = moe_transport.get_global_dp_buffer(tp_group)
    # _dp_gather (layers/dp_attention.py:1002) before its all-gather
    dp_attention._note_dp_gather_in_prefill_graph()
    if warmup:
        # GroupCoordinator._all_gather_into_tensor (distributed/parallel_state.py:1318-1320, graph warmup)
        global_hidden_states.zero_()
    else:
        ca_comm.all_gather_reg(pool_input, out=global_hidden_states, dim=0)
    _set_gather_fields(
        self,
        moe_transport,
        global_hidden_states,
        forward_batch,
        input_ids,
        input_ids_global,
        cp_return_contiguous,
    )
    return None


def _transport_falls_back(moe_transport, moe, forward_batch, use_cp, moe_gather, attn_scatter) -> bool:
    """Whether MoeRowTransport.__init__ (models/deepseek_common/moe_transport.py:100-217) takes
    another path than its plain TP-MoE gather: CP, an a2a backend or attention TP (a2a scatter),
    the variable-length gather, a shared expert run before the gather, no padding mask (off ROCm,
    or no num_token_non_padded), or a piecewise or breakable graph."""
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
        is_in_breakable_cuda_graph,
    )
    from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph.context_manager import (
        is_in_tc_piecewise_cuda_graph,
    )

    parallel = moe_transport.get_parallel()
    use_cp = moe_transport.is_cp_active(forward_batch) if use_cp is None else use_cp
    mode = forward_batch.dp_padding_mode
    return (
        not moe_transport._is_hip
        or use_cp
        # _use_tp_moe_gather and not _use_tp_attn_a2a_scatter (lines 114-131)
        or not (
            moe_gather
            if moe_gather is not None
            else parallel.attn_dp_size > 1 and moe_transport.get_moe_a2a_backend().is_none()
        )
        or (
            attn_scatter
            if attn_scatter is not None
            else parallel.attn_tp_size > 1 and not moe_transport.get_moe_a2a_backend().is_none()
        )
        # _dp_gather's all-gather of equal slots (layers/dp_attention.py:1003-1039), from
        # attention TP 1 (_dp_gather_via_all_gather, line 724) in the TP group (line 732)
        or mode is None
        or not mode.is_max_len()
        or parallel.attn_cp_size > 1
        or parallel.attn_tp_size != 1
        or _world_dp_gather_enabled()
        # line 160: the shared expert reads the rows before the mask
        or moe_transport.shared_expert_runs_locally(moe)
        # _mask_dp_pad_rows (line 66) masks nothing without them
        or forward_batch.num_token_non_padded is None
        or is_in_tc_piecewise_cuda_graph()
        or is_in_breakable_cuda_graph()
    )


def _world_dp_gather_enabled() -> bool:
    from sglang.srt.layers.dp_attention import world_dp_gather_enabled

    return world_dp_gather_enabled()


def _aiter_copy_in_gather(tp_group, hidden_states):
    """(ca_comm, its registered input buffer viewed as hidden_states, whether this is a graph
    warmup) when GroupCoordinator._all_gather_into_tensor (distributed/parallel_state.py:1289-1321)
    takes aiter's custom all-gather and copies the rows into that buffer (all_gather_unreg:
    eager, or captured with SGLANG_AITER_CAPTURE_COPY_IN or the memory saver), or zeroes the
    output in a graph warmup; else None."""
    import torch
    from sglang.srt.distributed import parallel_state
    from sglang.srt.environ import envs

    ca_comm = tp_group.ca_comm
    if not (
        envs.SGLANG_USE_AITER_AG.get()
        and tp_group._has_aiter_custom_all_gather()
        # the gathered buffer is a new contiguous tensor (get_global_dp_buffer)
        and hidden_states.is_contiguous()
        and hidden_states.dtype in (torch.float32, torch.float16, torch.bfloat16)
        and ca_comm.should_custom_ag(hidden_states)
    ):
        return None
    warmup = False
    if getattr(ca_comm, "_IS_CAPTURING", False):
        if not torch.cuda.is_current_stream_capturing():
            warmup = True
        elif not (envs.SGLANG_MEMORY_SAVER_CUDA_GRAPH.get() or parallel_state._AITER_CAPTURE_COPY_IN):
            return None  # all_gather_reg straight from the rows: no copy to fold in
    try:
        pool = ca_comm._pool["input"]
        buffer = pool.tensor
    except (AttributeError, KeyError, RuntimeError, TypeError):
        return None  # a raw (tensorless) input buffer
    n_bytes = hidden_states.numel() * hidden_states.element_size()
    if (
        n_bytes > pool.max_size
        or buffer.dtype != torch.uint8
        or buffer.device != hidden_states.device
        or buffer.data_ptr() != pool.data_ptr
    ):
        return None
    # aiter's all_gather_reg reads the buffer from its start
    return ca_comm, buffer[:n_bytes].view(hidden_states.dtype).view(hidden_states.shape), warmup


def _set_gather_fields(
    self,
    moe_transport,
    global_hidden_states,
    forward_batch,
    input_ids,
    input_ids_global,
    cp_return_contiguous,
):
    """The fields MoeRowTransport.__init__ sets (moe_transport.py:204-217) on its plain TP-MoE
    gather: no CP, no a2a scatter, no local shared expert, MAX_LEN padding (so no
    variable-length reduce-scatter)."""
    parallel = moe_transport.get_parallel()
    should_use_dp_reduce_scatterv = moe_transport.should_use_dp_reduce_scatterv()
    # lines 142-150, with _use_tp_moe_gather, not _use_reduce_scatterv and MAX_LEN
    use_reduce_scatter = (
        moe_transport.envs.SGLANG_DP_USE_REDUCE_SCATTER.get()
        and not should_use_dp_reduce_scatterv
        and parallel.tp_size == parallel.attn_dp_size
    )
    self.input = global_hidden_states
    self.input_ids, self.input_ids_global = input_ids, input_ids_global
    self.batch = forward_batch
    self.cp_return_contiguous = cp_return_contiguous
    self._use_cp = False
    self._use_tp_moe_gather = True
    self._use_tp_attn_a2a_scatter = False
    self._use_reduce_scatterv = False
    self._use_reduce_scatter = use_reduce_scatter
    self.mlp_reduce_scatter = use_reduce_scatter or should_use_dp_reduce_scatterv
    self._shared_local = None
    self.do_shared_local = False
    self._a2a_scatter_chunks = None
    self.gathered_rows = moe_transport.nullcontext()
