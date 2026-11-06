# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.runtime.dispatch import FallbackReason, api
from aklr.plugins.vllm.hook import vllm_route
from aklr.plugins.vllm.models.muse_spark.contract import MUSE_SPARK_MOE

__all__ = ["muse_spark_moe"]

_VERSIONS = ("0.33.0rc1.dev910",)
# set on the pooled rows whose pre-expert norm the experts apply: (weight, eps, gain_center)
_DEFERRED_NORM = "_aklr_muse_deferred_norm"


@api(MUSE_SPARK_MOE)
def muse_spark_moe(
    hidden_states,
    norm_weight,
    topk_ids,
    expert_mask,
    w1,
    w1_scale,
    w2,
    w2_scale,
    eps=1e-6,
    gain_center=1.0,
    out_dtype=None,
    out=None,
): ...


@vllm_route(
    "vllm.models.muse_spark.amd.model.MuseSpark1xMoE._pool",
    "vllm.models.muse_spark.amd.model.fused_rmsnorm_expand",
    versions=_VERSIONS,
)
def _defer_norm(x, weight, eps, gain_center, input_scale, repeats):
    """The pooled rows as they are, for the experts (_experts_apply) to normalize as they
    gather them; only MuseSpark1xMoE._pool calls it, for experts that take one row per route."""
    import torch

    if weight is None or input_scale or x.dtype != torch.bfloat16 or x.shape[0] == 0:
        return FallbackReason.OUT_OF_CONTRACT
    setattr(x, _DEFERRED_NORM, (weight, eps, gain_center))
    return x


@vllm_route(
    "vllm.model_executor.layers.fused_moe.modular_kernel.FusedMoEKernelModularImpl._fused_experts",
    "vllm.models.muse_spark.amd.muse_moe_runner.MuseSparkAiterExperts.apply",
    versions=_VERSIONS,
)
def _experts_apply(
    self,
    output,
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    activation,
    global_num_experts,
    expert_map,
    a1q_scale,
    a2_scale,
    workspace13,
    workspace2,
    expert_tokens_meta,
    apply_router_weight_on_input,
):
    """The norm _defer_norm left and the experts as one op call; without one it is vLLM's
    apply, and where the op falls back, vLLM's norm, then its apply."""
    deferred = getattr(hidden_states, _DEFERRED_NORM, None)
    if deferred is None:
        return FallbackReason.OUT_OF_CONTRACT
    delattr(hidden_states, _DEFERRED_NORM)
    weight, eps, gain_center = deferred
    moe, quant = self.moe_config, self.quant_config
    route_output = FallbackReason.OUT_OF_CONTRACT
    # what vLLM's apply does other than the op: no expert parallelism, rows another rank
    # sent, router weights on the input, another activation or padded experts
    if not (
        expert_map is None
        or expert_tokens_meta is not None
        or apply_router_weight_on_input
        or a1q_scale is not None
        or activation.name != "SILU"
        or not quant.use_mxfp4_w4a4
        or quant.w1_bias is not None
        or quant.w2_bias is not None
        or moe.hidden_dim_unpadded != hidden_states.shape[-1]
        or moe.intermediate_size_per_partition != moe.intermediate_size_per_partition_unpadded
        or moe.intermediate_pad not in (None, 0)
        or not output.is_contiguous()
    ):
        route_output = muse_spark_moe(
            hidden_states,
            weight,
            topk_ids,
            expert_map,
            w1,
            quant.w1_scale,
            w2,
            quant.w2_scale,
            eps=float(eps),
            gain_center=float(gain_center),
            out=output,
        )
    if isinstance(route_output, FallbackReason):
        from vllm.models.muse_spark.amd.muse_moe_runner import MuseSparkAiterExperts
        from vllm.models.muse_spark.amd.ops.fused_norm_expand import fused_rmsnorm_expand

        normalized = fused_rmsnorm_expand(hidden_states, weight, eps, gain_center, False, topk_ids.shape[1])
        return MuseSparkAiterExperts.apply.__wrapped__(
            self,
            output,
            normalized,
            w1,
            w2,
            topk_weights,
            topk_ids,
            activation,
            global_num_experts,
            expert_map,
            a1q_scale,
            a2_scale,
            workspace13,
            workspace2,
            expert_tokens_meta,
            apply_router_weight_on_input,
        )
    self._park_route_output(route_output.view_as(output))
    return None
