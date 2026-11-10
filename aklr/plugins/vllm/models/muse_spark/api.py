# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.runtime.dispatch import FallbackReason, api
from aklr.ops.gemm.api import gemm
from aklr.plugins.vllm.hook import original, vllm_route
from aklr.plugins.vllm.models.muse_spark.contract import MUSE_SPARK_MOE

__all__ = ["muse_spark_moe"]

_VERSIONS = ("0.33.0rc1.dev910",)
_NORM_EXPERTS_GROUP = "muse_spark.norm_experts"
# set on the pooled rows whose pre-expert norm the experts apply: (weight, eps, gain_center)
_DEFERRED_NORM = "_aklr_muse_deferred_norm"
# _defer_norm defers only where the op serves: materializing the norm in _experts_apply when
# it falls back costs the eager steps host time vLLM's own path does not.
# A layer (the id of its norm weight) to the experts' tensors the op takes, None if it never
# serves the layer; absent until _experts_apply first runs for it.
_EXPERTS: dict[int, tuple | None] = {}
# (layer, n_tokens) to whether the op serves it
_SERVES: dict[tuple[int, int], bool] = {}
# zeros as topk_ids, sliced to n_tokens rows for fallback_reason
_TOPK_IDS: dict[tuple, object] = {}
# the layers whose norm _defer_norm left and _experts_apply has not applied yet
_PENDING: set[int] = set()


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
    "vllm.model_executor.layers.linear.ReplicatedLinear.forward",
    "vllm.models.muse_spark.amd.model.MuseSparkTorchLinearMethod.apply",
    versions=_VERSIONS,
)
def _torch_linear(self, layer, x, bias=None):
    """The MoE pooler and unpooler, which vLLM keeps off aiter's tuned GEMM (its split-K
    kernels for these shapes accumulate in bf16) and runs through F.linear."""
    return gemm(x, layer.weight, bias=bias)


@vllm_route(
    "vllm.models.muse_spark.amd.model.MuseSpark1xMoE._pool",
    "vllm.models.muse_spark.amd.model.fused_rmsnorm_expand",
    versions=_VERSIONS,
    group=_NORM_EXPERTS_GROUP,
)
def _defer_norm(x, weight, eps, gain_center, input_scale, repeats):
    """The pooled rows as they are, for the experts (_experts_apply) to normalize as they
    gather them; only MuseSpark1xMoE._pool calls it, for experts that take one row per route."""
    import torch

    if weight is None or input_scale or x.dtype != torch.bfloat16 or x.shape[0] == 0:
        return FallbackReason.OUT_OF_CONTRACT
    layer = id(weight)
    serves = _SERVES.get((layer, x.shape[0]))
    if serves is None and layer in _EXPERTS:
        serves = _would_serve(x, weight, eps, gain_center, layer)
    if serves is False:
        return FallbackReason.OUT_OF_CONTRACT
    setattr(x, _DEFERRED_NORM, (weight, eps, gain_center))
    _PENDING.add(layer)
    return x


def _would_serve(x, weight, eps, gain_center, layer) -> bool | None:
    """Whether the op serves the layer's experts on ``x``, remembered unless it depends on
    the moment (a graph capture before the first eager call)."""
    import torch

    experts = _EXPERTS[layer]
    if experts is None:
        return False
    topk, ids_dtype, expert_map, w1, w1_scale, w2, w2_scale = experts
    n_tokens = x.shape[0]
    spec = (x.device, ids_dtype, topk)
    ids = _TOPK_IDS.get(spec)
    if ids is None or ids.shape[0] < n_tokens:
        ids = _TOPK_IDS[spec] = torch.zeros((max(n_tokens, 8192), topk), dtype=ids_dtype, device=x.device)
    reason = muse_spark_moe.fallback_reason(
        x,
        weight,
        ids[:n_tokens],
        expert_map,
        w1,
        w1_scale,
        w2,
        w2_scale,
        eps=float(eps),
        gain_center=float(gain_center),
    )
    if reason is FallbackReason.NOT_WARMED:
        return None
    _SERVES[(layer, n_tokens)] = reason is None
    return reason is None


@vllm_route(
    "vllm.model_executor.layers.fused_moe.modular_kernel.FusedMoEKernelModularImpl._fused_experts",
    "vllm.models.muse_spark.amd.muse_moe_runner.MuseSparkAiterExperts.apply",
    versions=_VERSIONS,
    group=_NORM_EXPERTS_GROUP,
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
        if _PENDING:
            # vLLM's prepare handed the experts other rows than the pooled ones (an all-to-all,
            # a copy): their norm, left to here, is lost, and vLLM's apply would run unnormalized
            raise RuntimeError("aklr: Muse Spark's experts got rows other than the pooled ones")
        return FallbackReason.OUT_OF_CONTRACT
    delattr(hidden_states, _DEFERRED_NORM)
    weight, eps, gain_center = deferred
    layer = id(weight)
    _PENDING.discard(layer)
    moe, quant = self.moe_config, self.quant_config
    route_output = FallbackReason.OUT_OF_CONTRACT
    _EXPERTS[layer] = None
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
        _EXPERTS[layer] = (
            topk_ids.shape[1],
            topk_ids.dtype,
            expert_map,
            w1,
            quant.w1_scale,
            w2,
            quant.w2_scale,
        )
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
        if route_output is not FallbackReason.NOT_WARMED:
            _SERVES[(layer, hidden_states.shape[0])] = False
        # Saved before the batch was installed: a later wrapper or an imported alias
        # must not redirect fallback back into _defer_norm/_experts_apply.
        normalized = original(_defer_norm)(hidden_states, weight, eps, gain_center, False, topk_ids.shape[1])
        return original(_experts_apply)(
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
