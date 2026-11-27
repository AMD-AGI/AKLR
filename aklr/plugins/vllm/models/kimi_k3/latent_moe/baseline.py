# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What vLLM runs: ``_shard_up_proj_tail``'s four ops on vLLM's own TP group (its custom all-reduce, as the
deployment's), vLLM's RMSNorm layer, the addmm into this rank's columns of the shared output (in place, as vLLM
writes it), and the all-reduce of the shared output. vLLM's TP group is set up over the caller's process group on the
first call."""

import contextlib
import os
import tempfile

import torch.distributed as dist

from aklr.core.contract.baseline import DistOpBaseline

__all__ = ["KimiK3LatentMoeTailBaseline"]

# vLLM's RMSNorm layers by (width, eps, device): made once, not in a timed call
_NORMS: dict = {}
# a Unix socket's path holds at most 107 bytes; vLLM's message queue binds one at <VLLM_RPC_BASE_PATH>/<uuid4>
_SOCKET_PATH_MAX, _UUID_LEN = 107, 36


@contextlib.contextmanager
def _short_rpc_base():
    """vLLM's RPC base directory (VLLM_RPC_BASE_PATH, by default the temp directory) short enough for the sockets
    its TP group's message queue binds there: the temp directory itself where it is, else /tmp, as vLLM's default is
    without TMPDIR (validation gives each task a TMPDIR nested deep enough to pass the limit). A base the caller set
    is kept, and the variable is put back afterwards."""
    if os.environ.get("VLLM_RPC_BASE_PATH") or len(tempfile.gettempdir()) + 1 + _UUID_LEN <= _SOCKET_PATH_MAX:
        yield
        return
    os.environ["VLLM_RPC_BASE_PATH"] = "/tmp"
    try:
        yield
    finally:
        del os.environ["VLLM_RPC_BASE_PATH"]


def _vllm_tp(tp_size, device):
    """(vLLM's TP all-reduce, this rank's TP rank); vLLM's TP group set up over torch.distributed's ranks if none is."""
    if tp_size == 1:
        return (lambda x: x), 0
    try:
        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm.distributed import parallel_state as ps
    except ImportError as e:
        raise NotImplementedError("vLLM's tail needs vLLM") from e
    if not ps.model_parallel_is_initialized():
        if not dist.is_initialized() or dist.get_world_size() != tp_size:
            raise NotImplementedError(f"vLLM's tail needs a group of {tp_size} ranks")
        with set_current_vllm_config(VllmConfig()), _short_rpc_base():
            ps.init_distributed_environment(
                world_size=tp_size, rank=dist.get_rank(), local_rank=device.index, backend="nccl"
            )
            ps.initialize_model_parallel(tensor_model_parallel_size=tp_size)
    tp = ps.get_tp_group()
    if tp.world_size != tp_size:
        raise NotImplementedError(f"vLLM's TP group has {tp.world_size} ranks, not {tp_size}")
    return tp.all_reduce, tp.rank_in_group


def _vllm_norm(width, eps, device):
    key = (width, eps, device)
    if key not in _NORMS:
        try:
            from vllm.config import VllmConfig, set_current_vllm_config
            from vllm.model_executor.layers.layernorm import RMSNorm
        except ImportError as e:
            raise NotImplementedError("vLLM's tail needs vLLM") from e
        with set_current_vllm_config(VllmConfig()):
            _NORMS[key] = RMSNorm(width, eps=eps).to(device)
    return _NORMS[key]


@contextlib.contextmanager
def _captures_in_vllm_graph_capture():
    """The timing's graph captures inside vLLM's ``graph_capture``, each on the timing's own stream, as vLLM captures
    its decode graphs. vLLM's TP all-reduce (aiter's custom all-reduce on ROCm) registers the buffers a graph reads as
    that context ends; captured outside it, a replay reads unregistered peer addresses (a GPU memory fault at a null
    address). Only the captures: the timing's eager calls run vLLM's eager all-reduce. Without vLLM's TP group (no
    baseline call yet) the captures are left as they are. The capture helper is put back afterwards."""
    from vllm.distributed.parallel_state import (
        GraphCaptureContext,
        get_tp_group,
        model_parallel_is_initialized,
    )

    from aklr.core.utils import testing

    if not model_parallel_is_initialized():
        yield
        return
    capture = testing.cuda_graph_without_gc

    @contextlib.contextmanager
    def in_vllm_graph_capture(graph, stream=None, **kwargs):
        with (
            get_tp_group().graph_capture(GraphCaptureContext(stream)),
            capture(graph, stream=stream, **kwargs) as g,
        ):
            yield g

    testing.cuda_graph_without_gc = in_vllm_graph_capture
    try:
        yield
    finally:
        testing.cuda_graph_without_gc = capture


class KimiK3LatentMoeTailBaseline(DistOpBaseline):
    """vLLM's tail needs no setup of its own: vLLM's TP group is set up at the first call (``_vllm_tp``)."""

    def profile(self, fn):
        """DistOpBaseline's, its graphs captured as vLLM captures them (``_captures_in_vllm_graph_capture``); the
        candidate shares this clock, and the context changes nothing for a graph without vLLM's all-reduce."""
        with _captures_in_vllm_graph_capture():
            return super().profile(fn)

    def baseline(
        self,
        *,
        routed,
        shared,
        up_weight,
        norm_weight,
        tp_size=8,
        eps=1e-5,
        out_dtype=None,
        out=None,
    ):
        """vLLM's out [n_tokens, hidden_size]; shared's own columns are written in place, as in vLLM."""
        all_reduce, rank = _vllm_tp(tp_size, routed.device)
        norm = _vllm_norm(routed.shape[1], eps, routed.device)
        norm.weight.data = norm_weight
        latent = norm(all_reduce(routed))
        shard = up_weight.shape[0]
        shared.narrow(-1, rank * shard, shard).addmm_(latent, up_weight.t())
        result = all_reduce(shared)
        return {"out": result if out is None else out.copy_(result)}
