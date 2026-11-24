# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What vLLM runs: ``_shard_up_proj_tail``'s four ops on vLLM's own TP group (its custom all-reduce, as the
deployment's), vLLM's RMSNorm layer, the addmm into this rank's columns of the shared output (in place, as vLLM
writes it), and the all-reduce of the shared output. vLLM's TP group is set up over the caller's process group on the
first call."""

import torch.distributed as dist

from aklr.core.contract.baseline import DistOpBaseline

__all__ = ["KimiK3LatentMoeTailBaseline"]

# vLLM's RMSNorm layers by (width, eps, device): made once, not in a timed call
_NORMS: dict = {}


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
        with set_current_vllm_config(VllmConfig()):
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


class KimiK3LatentMoeTailBaseline(DistOpBaseline):
    """vLLM's tail needs no setup of its own: vLLM's TP group is set up at the first call (``_vllm_tp``)."""

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
