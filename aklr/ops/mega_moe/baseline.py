# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import os

import torch
import torch.distributed as dist

from aklr.core.contract.baseline import DistOpBaseline
from aklr.ops.fused_moe.baseline import FusedMoeBaseline

__all__ = ["MegaMoeBaseline"]

# sglang's DeepSeek-V4 MegaMoE capacity (SGLANG_AMD_FLYDSL_MEGA_MOE_MTPR)
_MAX_TOKENS_PER_RANK = 8192


# one aiter MegaMoEV2 per device and shape, built at its first call, as sglang keeps one for every layer
_INSTANCES = {}


class MegaMoeBaseline(DistOpBaseline):
    def setup(self):
        """mori's symmetric heap over the group, as aiter's MoriAll2AllManager sets it up: the
        kernels and MegaMoEV2 allocate their buffers there."""
        # the bench holds MegaMoEV2's buffers and the kernel's at once: more than mori's 4 GiB
        os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "16G")
        import mori

        torch._C._distributed_c10d._register_process_group("mori", dist.group.WORLD)
        mori.shmem.shmem_torch_process_group_init("mori")

    def teardown(self):
        """Drops the MegaMoEV2 instances, whose buffers are on mori's heap, then the heap."""
        import mori

        _INSTANCES.clear()
        mori.shmem.shmem_finalize()

    def baseline(
        self,
        *,
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        w1_scale,
        w2_scale,
        ep_size=1,
        swiglu_limit=10.0,
        out_dtype=None,
        out=None,
    ):
        """aiter's MoE as sglang runs it: MegaMoEV2's main path over an expert-parallel group,
        this layer's weights bound before the call; with one rank, aiter's local fused_moe on
        the same gate-up-interleaved experts (MegaMoEV2 needs mori's group)."""
        if ep_size == 1:
            return FusedMoeBaseline().baseline(
                hidden_states=hidden_states,
                w1=w1,
                w2=w2,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                activation="silu",
                quant_type="per_1x32",
                doweight_stage1=False,
                gate_mode="interleave",
                act_quant=4,
                expert_mask=None,
                num_local_tokens=None,
                beta=None,
                linear_beta=None,
                swiglu_limit=swiglu_limit,
                out_dtype=out_dtype,
                out=None,
            )
        weights = tuple(t.contiguous().view(torch.uint8) for t in (w1, w1_scale, w2, w2_scale))
        model_dim, inter_dim, n_experts, topk = (
            hidden_states.shape[1],
            w1.shape[1] // 2,
            w1.shape[0],
            topk_ids.shape[1],
        )
        key = (hidden_states.device, model_dim, inter_dim, n_experts, topk, ep_size, float(swiglu_limit))
        if key not in _INSTANCES:
            from aiter.ops.flydsl.kernels.mega_moe import MegaMoEV2

            _INSTANCES[key] = MegaMoEV2(
                rank=dist.get_rank(),
                world_size=ep_size,
                model_dim=model_dim,
                inter_dim=inter_dim,
                experts=n_experts * ep_size,
                topk=topk,
                quant="a8w4",
                w1=weights[0],
                w1_scale=weights[1],
                w2=weights[2],
                w2_scale=weights[3],
                max_tok_per_rank=_MAX_TOKENS_PER_RANK,
                swiglu_limit=float(swiglu_limit),
            )
        mega = _INSTANCES[key]
        mega._s1_w1, mega._s1_w1_scale, mega.w2, mega.w2_scale = weights
        out = mega.forward(hidden_states, topk_weights, topk_ids, slice_output=True)
        return {"out": out.clone() if out_dtype is None else out.to(out_dtype)}
