# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    DeterminismLevel,
    DType,
    E8M0Scales,
    ExpertMaskDistribution,
    FiniteDistribution,
    IntRangeDistribution,
    LogUniformDistribution,
    Preshuffle,
    SqnrBound,
)
from aklr.core.contract.spec import (
    CompareSpec,
    DTypeSpec,
    LayoutFact,
    LayoutVariant,
    LayoutVariantsSpec,
    ScalarSpec,
    TensorSpec,
    axis,
    derived_axis,
)
from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins import Plugin
from aklr.plugins.aiter.gemm.plugin import gemm
from aklr.plugins.vllm.hook import _original

__all__ = ["MUSE_SPARK_MOE", "declare", "family", "muse_spark_moe"]


_BF16 = DTypeSpec(any_of=(DType.BFLOAT16,))
_FP4 = DTypeSpec(any_of=(DType.FLOAT4_E2M1,))
_E8M0 = DTypeSpec(any_of=(DType.FLOAT8_E8M0,))


def _e8m0_scales(fan_in: str) -> LogUniformDistribution:
    return LogUniformDistribution(
        log2_lo=f"-5 - bit_length({fan_in}) // 2", log2_hi=f"1 - bit_length({fan_in}) // 2"
    )


def _mxfp4_facts(shuffled: bool) -> tuple[LayoutFact, ...]:
    facts = {"w1.is_shuffled": shuffled, "w2.is_shuffled": shuffled, "w1_scale.ndim": 3}
    return tuple(LayoutFact(location="tensor_attr", name=n, expected_value=v) for n, v in facts.items())


MUSE_SPARK_MOE = OpContract(
    contract_id="muse_spark_moe@1.0",
    description=(
        "vLLM's MuseSpark1xMoE experts on one expert-parallel rank with MXFP4 experts, from "
        "the pooled rows to the unweighted output of each route: each row's pre-expert "
        "RMSNorm (fp32, times norm_weight + gain_center), then per route, the routes to the "
        "experts expert_mask gives this rank, down(silu(gate(x)) * up(x)) of its MXFP4 "
        "weights (fused_moe's: fp4 values at an e8m0 scale per 32); the routes to the others "
        "write zeros. The router weights, the post-expert norm and the sum over the routes "
        "are vLLM's, after it."
    ),
    op_type="muse_spark",
    axes={
        "n_tokens": axis("Pooled rows in this call.", 1, 65536, is_var=True),
        "model_dim": axis("The pooled width: the experts' hidden size.", 2048, 16384, multiple_of=2048),
        "topk": axis("Routes per row.", 1, 32),
        "n_experts": axis("Experts this rank holds: w1's and w2's.", 2, 1024),
        "inter_dim": axis("Expert intermediate size.", 256, 16384, multiple_of=256),
        "global_experts": axis(
            "expert_mask's length: the group's routed experts, then vLLM's always-masked slot.",
            3,
            1025,
        ),
        "gate_up_rows": derived_axis("w1 rows: gate and up concatenated.", "2 * inter_dim"),
        "model_blocks": derived_axis("w1 scale columns.", "model_dim // 32"),
        "inter_blocks": derived_axis("w2 scale columns.", "inter_dim // 32"),
    },
    # vLLM's expert map gives each rank an equal block of the routed experts
    constraints=("topk < global_experts", "(global_experts - 1) % n_experts == 0"),
    inputs={
        "hidden_states": TensorSpec(
            description="The pooled rows, not yet normalized.",
            shape=("n_tokens", "model_dim"),
            dtype=_BF16,
            distribution=FiniteDistribution(),
        ),
        "norm_weight": TensorSpec(
            description="The pre-expert RMSNorm's gain, before gain_center is added.",
            shape=("model_dim",),
            dtype=_BF16,
            distribution=FiniteDistribution(),
            lifetime="weight",
        ),
        "topk_ids": TensorSpec(
            description="Each row's routed experts, distinct within a row: global ids.",
            shape=("n_tokens", "topk"),
            dtype=DTypeSpec(any_of=(DType.INT32,)),
            distribution=IntRangeDistribution(lo="0", hi="global_experts - 2", distinct=True),
        ),
        "expert_mask": TensorSpec(
            description="1 for the experts this rank holds, in the order w1 holds them; 0 in the last slot.",
            shape=("global_experts",),
            dtype=DTypeSpec(any_of=(DType.INT32,)),
            distribution=ExpertMaskDistribution(local_experts="n_experts"),
            lifetime="weight",
        ),
        "w1": TensorSpec(
            description="Gate-up projection per expert: gate rows, then up rows.",
            shape=("n_experts", "gate_up_rows", "model_dim"),
            dtype=_FP4,
            distribution=FiniteDistribution(),
            lifetime="weight",
        ),
        "w1_scale": TensorSpec(
            description="Per 1 x 32 block of w1.",
            shape=("n_experts", "gate_up_rows", "model_blocks"),
            dtype=_E8M0,
            distribution=_e8m0_scales("model_dim"),
            lifetime="weight",
        ),
        "w2": TensorSpec(
            description="Down projection per expert.",
            shape=("n_experts", "model_dim", "inter_dim"),
            dtype=_FP4,
            distribution=FiniteDistribution(),
            lifetime="weight",
        ),
        "w2_scale": TensorSpec(
            description="Per 1 x 32 block of w2.",
            shape=("n_experts", "model_dim", "inter_blocks"),
            dtype=_E8M0,
            distribution=_e8m0_scales("inter_dim"),
            lifetime="weight",
        ),
        "eps": ScalarSpec(
            description="The RMSNorm's epsilon.",
            dtype=DType.FLOAT32,
            interval=(1e-12, 1e-3),
            default=1e-6,
        ),
        "gain_center": ScalarSpec(
            description="Added to norm_weight: 1 for a zero-centered gain.",
            dtype=DType.FLOAT32,
            values=(0.0, 1.0),
            default=1.0,
        ),
    },
    outputs={
        "out": TensorSpec(
            description="Each route's expert output, unweighted; zeros off this rank.",
            shape=("n_tokens", "topk", "model_dim"),
            dtype=DTypeSpec(same_as_tensor="hidden_states"),
            # fused_moe's bound: its MXFP4 activations
            error_bounds=SqnrBound(min_sqnr_db=13.0),
        )
    },
    # as vLLM holds the experts for aiter: shuffle_weight's 16 x 16 tiles, the scales shuffled
    layout_variants=LayoutVariantsSpec(
        by_name={
            "row_major": LayoutVariant(layouts={}, detect_facts=_mxfp4_facts(False)),
            "shuffled_16x16": LayoutVariant(
                layouts={
                    "w1": Preshuffle(tile=(16, 16)),
                    "w2": Preshuffle(tile=(16, 16)),
                    "w1_scale": E8M0Scales(block_size=32, pad_to=(256, 8), shuffled=True),
                    "w2_scale": E8M0Scales(block_size=32, pad_to=(256, 8), shuffled=True),
                },
                detect_facts=_mxfp4_facts(True),
            ),
        }
    ),
    # stage 2 combines with atomics
    compare=CompareSpec(max_violation_fraction=1e-3, determinism=DeterminismLevel.BOUNDED),
)


family = Plugin("vllm")
family.define(MUSE_SPARK_MOE)
family.build()


_VERSIONS = ("0.34.0",)
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


@family.op
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


def _torch_linear(self, layer, x, bias=None):
    """The MoE pooler and unpooler, which vLLM keeps off aiter's tuned GEMM (its split-K
    kernels for these shapes accumulate in bf16) and runs through F.linear."""
    return gemm(x, layer.weight, bias=bias)


def _defer_norm(x, weight, eps, gain_center, input_scale, repeats):
    """The pooled rows as they are, for the experts (_experts_apply) to normalize as they
    gather them; only MuseSparkMoE._pool calls it, for experts that take one row per route."""
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
        normalized = _original(_defer_norm)(hidden_states, weight, eps, gain_center, False, topk_ids.shape[1])
        return _original(_experts_apply)(
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


def declare(plugin: Plugin) -> None:
    plugin.intercept(
        "vllm.model_executor.layers.linear.ReplicatedLinear.forward",
        "vllm.models.muse_spark.amd.model.MuseSparkTorchLinearMethod.apply",
        versions=("0.33.0rc1.dev910", "0.33.0", "0.34.0"),
        replace=_torch_linear,
    )
    plugin.intercept(
        "vllm.models.muse_spark.amd.model.MuseSparkMoE._pool",
        "vllm.models.muse_spark.amd.model.fused_rmsnorm_expand",
        versions=_VERSIONS,
        group=_NORM_EXPERTS_GROUP,
        replace=_defer_norm,
    )
    plugin.intercept(
        "vllm.model_executor.layers.fused_moe.modular_kernel.FusedMoEKernelModularImpl._fused_experts",
        "vllm.models.muse_spark.amd.muse_moe_runner.MuseSparkAiterExperts.apply",
        versions=_VERSIONS,
        group=_NORM_EXPERTS_GROUP,
        replace=_experts_apply,
    )
