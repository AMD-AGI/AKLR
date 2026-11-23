# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""sglang's Qwen2MoeSparseMoeBlock with its shared expert fused into the experts (aiter's
path for Qwen3.5 and Qwen3-Next): the router, the fp8 experts, and both at once. With
MXFP4 experts and the shared expert kept apart in BF16 (sglang's path for Qwen3.5's MXFP4
checkpoint, which cannot fuse it), the whole block.

A model is a scenario of each contract: Qwen3.5-397B-A17B at TP4 is model_dim 4096,
n_routed 512, inter_dim 256 (and shared_inter_dim 256); Qwen3-Next-80B-A3B at TP4 is
model_dim 2048, n_routed 512, inter_dim 128."""

from __future__ import annotations

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    DeterminismLevel,
    DType,
    E8M0Scales,
    FiniteDistribution,
    IntRangeDistribution,
    LecunNormalDistribution,
    LogUniformDistribution,
    Preshuffle,
    SqnrBound,
    ToleranceBound,
)
from aklr.core.contract.spec import (
    CompareSpec,
    DTypeSpec,
    LayoutFact,
    LayoutVariant,
    LayoutVariantsSpec,
    TensorSpec,
    attr,
    axis,
    derived_axis,
)
from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins import Plugin
from aklr.plugins.sglang.models import moe_blocks

__all__ = [
    "QWEN2_MOE_FP8_EXPERTS",
    "QWEN2_MOE_FP8_ROUTER_EXPERTS",
    "QWEN2_MOE_MXFP4_MOE",
    "QWEN2_MOE_ROUTER",
    "declare",
    "family",
    "qwen2_moe_fp8_experts",
    "qwen2_moe_fp8_router_experts",
    "qwen2_moe_mxfp4_moe",
    "qwen2_moe_router",
]


_BF16 = DTypeSpec(any_of=(DType.BFLOAT16,))
_FP8 = DTypeSpec(any_of=(DType.FLOAT8_E4M3FN,))
_FLOAT32 = DTypeSpec(any_of=(DType.FLOAT32,))
_INT32 = DTypeSpec(any_of=(DType.INT32,))
# the 128 x 128 block scales of an fp8 linear, as gemm_a8w8_blockwise_scaled draws them
_BLOCK_SCALE = LogUniformDistribution(log2_lo="-3", log2_hi="0")
_COMPARE = CompareSpec(determinism=DeterminismLevel.BOUNDED)

_TOKENS = {
    "n_tokens": axis("Tokens in this call.", 1, 65536, is_var=True),
    "model_dim": axis("Hidden size.", 128, 16384, multiple_of=128),
}
_ROUTED = axis("Routed experts: the router's rows.", 1, 1024)
_TOPK = axis("Routed experts per token.", 1, 64)
# the experts as sglang's FusedMoE holds them on one rank of its TP group: every expert,
# its intermediate size split over the ranks, the shared expert in the slot after the
# routed ones
_EXPERT_AXES = {
    "inter_dim": axis("Expert intermediate size on this rank.", 128, 16384, multiple_of=128),
    "gate_up_rows": derived_axis("w1 rows: gate and up concatenated.", "2 * inter_dim"),
    "model_blocks": derived_axis("128-blocks of the hidden size.", "model_dim // 128"),
    "gate_up_blocks": derived_axis("128-blocks of w1's rows.", "gate_up_rows // 128"),
    "inter_blocks": derived_axis("128-blocks of the intermediate size.", "inter_dim // 128"),
}
_SLOTS = "Expert slots: the routed experts, then the shared one."

_HIDDEN_STATES = TensorSpec(
    description="The tokens' hidden states, the MoE block's input.",
    shape=("n_tokens", "model_dim"),
    dtype=_BF16,
    distribution=FiniteDistribution(),
)
_ROUTER_WEIGHT = TensorSpec(
    description="The router (the block's gate) projection.",
    shape=("n_routed", "model_dim"),
    dtype=_BF16,
    distribution=LecunNormalDistribution(fan_in="model_dim"),
    lifetime="weight",
)
_SHARED_GATE_WEIGHT = TensorSpec(
    description=(
        "shared_expert_gate's weight: the shared expert weighs sigmoid(hidden_states shared_gate_weight^T)."
    ),
    shape=(1, "model_dim"),
    dtype=_BF16,
    distribution=LecunNormalDistribution(fan_in="model_dim"),
    lifetime="weight",
)
_EXPERTS = {
    "w1": TensorSpec(
        description=("Gate-up projection per expert slot on this rank: gate rows, then up rows."),
        shape=("n_experts", "gate_up_rows", "model_dim"),
        dtype=_FP8,
        distribution=LecunNormalDistribution(fan_in="model_dim"),
        lifetime="weight",
    ),
    "w1_scale": TensorSpec(
        description="w1's dequant scale per 128 x 128 block.",
        shape=("n_experts", "gate_up_blocks", "model_blocks"),
        dtype=_FLOAT32,
        distribution=_BLOCK_SCALE,
        lifetime="weight",
    ),
    "w2": TensorSpec(
        description="Down projection per expert slot on this rank.",
        shape=("n_experts", "model_dim", "inter_dim"),
        dtype=_FP8,
        distribution=LecunNormalDistribution(fan_in="inter_dim"),
        lifetime="weight",
    ),
    "w2_scale": TensorSpec(
        description="w2's dequant scale per 128 x 128 block.",
        shape=("n_experts", "model_blocks", "inter_blocks"),
        dtype=_FLOAT32,
        distribution=_BLOCK_SCALE,
        lifetime="weight",
    ),
}
# sglang holds the experts for aiter as shuffle_weight(w, (16, 16)) leaves them (and marks
# them is_shuffled), the scales as the checkpoint stores them
_EXPERT_LAYOUTS = LayoutVariantsSpec(
    by_name={
        name: LayoutVariant(
            layouts=(dict.fromkeys(("w1", "w2"), Preshuffle(tile=(16, 16))) if shuffled else {}),
            detect_facts=tuple(
                LayoutFact(
                    location="tensor_attr",
                    name=f"{w}.is_shuffled",
                    expected_value=shuffled,
                )
                for w in ("w1", "w2")
            ),
        )
        for name, shuffled in (("row_major", False), ("shuffled_16x16", True))
    }
)

_ROUTING = (
    "the logits hidden_states router_weight^T, rounded to bf16, pick each token's topk "
    "experts by descending logit, a tie going to the lower expert id, weighed by the "
    "softmax of their logits (the softmax over every expert, renormalized over the chosen "
    "ones)"
)
_SHARED = "and the shared expert by sigmoid(hidden_states shared_gate_weight^T)"
_EXPERT = (
    "each expert is down(silu(gate(x)) * up(x)) of fp8 weights at a scale per 128 x 128 "
    "block, each of its two linears taking its input as the checkpoint's FP8 scheme "
    "quantizes it: in bf16, then each 1 x 128 group in fp8 at a scale of its amax / 448; "
    "out is this rank's share of the block's output, which the TP group sums after the "
    "call"
)


# Implementations quantize the second linear's input from bf16 values they round at other
# points than the reference does, which flips some fp8 codes: about 33 to 38 dB on the
# bench's inputs. sglang's own gfx950 small-M experts keep bf16 activations, about 27 dB.
def _moe_output(min_sqnr_db: float) -> TensorSpec:
    return TensorSpec(
        description="This rank's share of the MoE block's output.",
        shape=("n_tokens", "model_dim"),
        dtype=DTypeSpec(same_as_tensor="hidden_states"),
        error_bounds=SqnrBound(min_sqnr_db=min_sqnr_db),
    )


QWEN2_MOE_ROUTER = OpContract(
    contract_id="qwen2_moe_router@1.0",
    description=(f"sglang's Qwen2MoeSparseMoeBlock router, its gate and TopK with renormalize: {_ROUTING}."),
    op_type="qwen2_moe",
    axes={**_TOKENS, "n_routed": _ROUTED, "topk": _TOPK},
    constraints=("topk <= n_routed",),
    inputs={"hidden_states": _HIDDEN_STATES, "router_weight": _ROUTER_WEIGHT},
    outputs={
        "topk_weights": TensorSpec(
            description="The chosen experts' weights, in topk_ids' order.",
            shape=("n_tokens", "topk"),
            dtype=_FLOAT32,
            # a logit summed in float32 may round to the next bf16 (2**-4 below 8),
            # which moves the weights by up to that
            error_bounds=ToleranceBound(absolute_tolerance=1e-6, relative_tolerance=2**-4),
        ),
        # exact, but two experts whose logits are adjacent bf16 values may swap, for the
        # same rounding (qwen2_moe_router_id_scores)
        "topk_ids": TensorSpec(
            description="The chosen experts, by descending logit.",
            shape=("n_tokens", "topk"),
            dtype=_INT32,
        ),
    },
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
    compare=_COMPARE,
)

QWEN2_MOE_FP8_EXPERTS = OpContract(
    contract_id="qwen2_moe_fp8_experts@1.0",
    description=(
        "sglang's Qwen2MoeSparseMoeBlock experts on one rank of its TP group, the shared "
        "expert fused in after the routed ones: each token's topk_ids experts weighed by "
        f"topk_weights, {_SHARED}; {_EXPERT}."
    ),
    op_type="qwen2_moe",
    axes={
        **_TOKENS,
        "n_experts": axis(_SLOTS, 2, 1025),
        "n_routed": derived_axis("Routed experts.", "n_experts - 1"),
        **_EXPERT_AXES,
        "topk": _TOPK,
    },
    constraints=("topk <= n_routed",),
    inputs={
        "hidden_states": _HIDDEN_STATES,
        "topk_weights": TensorSpec(
            description="The routed experts' weights.",
            shape=("n_tokens", "topk"),
            dtype=_FLOAT32,
            # a renormalized top-k softmax's
            distribution=LogUniformDistribution(log2_lo="-6", log2_hi="-1"),
        ),
        "topk_ids": TensorSpec(
            description="Each token's routed experts, distinct within a row.",
            shape=("n_tokens", "topk"),
            dtype=_INT32,
            distribution=IntRangeDistribution(lo=0, hi="n_routed - 1", distinct=True),
        ),
        "shared_gate_weight": _SHARED_GATE_WEIGHT,
        **_EXPERTS,
    },
    outputs={"out": _moe_output(25.0)},
    layout_variants=_EXPERT_LAYOUTS,
    compare=_COMPARE,
)

QWEN2_MOE_FP8_ROUTER_EXPERTS = OpContract(
    contract_id="qwen2_moe_fp8_router_experts@1.0",
    description=(
        "sglang's Qwen2MoeSparseMoeBlock._forward_router_experts on one rank of its TP "
        "group, the shared expert fused into the experts after the routed ones: "
        f"{_ROUTING}, {_SHARED}; {_EXPERT}."
    ),
    op_type="qwen2_moe",
    axes={
        **_TOKENS,
        "n_routed": _ROUTED,
        "n_experts": derived_axis(_SLOTS, "n_routed + 1"),
        **_EXPERT_AXES,
    },
    attrs={"topk": attr("Routed experts per token.", 10, param_name="topk")},
    constraints=("topk <= n_routed",),
    inputs={
        "hidden_states": _HIDDEN_STATES,
        "router_weight": _ROUTER_WEIGHT,
        "shared_gate_weight": _SHARED_GATE_WEIGHT,
        **_EXPERTS,
    },
    outputs={"out": _moe_output(25.0)},
    layout_variants=_EXPERT_LAYOUTS,
    compare=_COMPARE,
)

_MXFP4 = DTypeSpec(any_of=(DType.FLOAT4_E2M1,))
_E8M0 = DTypeSpec(any_of=(DType.FLOAT8_E8M0,))


def _e8m0_scales(fan_in: str) -> LogUniformDistribution:
    """MXFP4 block scales log-uniform over 2**-5 to 2**1 times 1 / sqrt(fan_in), as
    deepseek_v2_mxfp4_moe and minimax_m2_mxfp4_moe draw them."""
    return LogUniformDistribution(
        log2_lo=f"-5 - bit_length({fan_in}) // 2", log2_hi=f"1 - bit_length({fan_in}) // 2"
    )


def _mxfp4_facts(shuffled: bool) -> tuple[LayoutFact, ...]:
    facts = {"w1.is_shuffled": shuffled, "w2.is_shuffled": shuffled, "w1_scale.ndim": 3}
    return tuple(LayoutFact(location="tensor_attr", name=n, expected_value=v) for n, v in facts.items())


# MXFP4 experts as aiter's fused_moe takes them: fp4 values at an e8m0 scale per 32;
# row-major, or as sglang holds them for aiter: shuffle_weight's 16 x 16 tiles, the scales
# e8m0-shuffled
_MXFP4_LAYOUTS = LayoutVariantsSpec(
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
)

# The routed experts' MXFP4 activations come to 13 to 16 dB on the bench's inputs (sglang's
# own small-M kernel, which keeps them bf16, to about 48 dB). A reference without the shared
# expert, with it ungated or with its gate and up rows swapped falls below the bound from 4
# tokens on (1 to 9 dB), and mostly at 1 and 2.
_MXFP4_MIN_SQNR_DB = 11.0

QWEN2_MOE_MXFP4_MOE = OpContract(
    contract_id="qwen2_moe_mxfp4_moe@1.0",
    description=(
        "sglang's Qwen2MoeSparseMoeBlock.forward on one rank of its TP group, with MXFP4 "
        "routed experts and the shared expert apart in BF16 (sglang's path for a checkpoint "
        f"that keeps its shared expert BF16, which cannot be fused): {_ROUTING}; each routed "
        "expert down(silu(gate(x)) * up(x)) of its MXFP4 weights (fused_moe's: fp4 values "
        "at an e8m0 scale per 32), the shared expert down(silu(gate(x)) * up(x)) of its BF16 "
        "weights, weighed by sigmoid(hidden_states shared_gate_weight^T) and added; out is "
        "this rank's share of the block's output, which the TP group sums after the call"
    ),
    op_type="qwen2_moe",
    axes={
        **_TOKENS,
        "n_routed": _ROUTED,
        "inter_dim": axis("Routed expert intermediate size on this rank.", 128, 16384, multiple_of=128),
        "shared_inter_dim": axis(
            "Shared expert intermediate size on this rank.", 128, 16384, multiple_of=128
        ),
        "gate_up_rows": derived_axis("w1 rows: gate and up concatenated.", "2 * inter_dim"),
        "model_blocks": derived_axis("w1 scale columns.", "model_dim // 32"),
        "inter_blocks": derived_axis("w2 scale columns.", "inter_dim // 32"),
        "shared_gate_up_rows": derived_axis(
            "shared_w1 rows: gate and up concatenated.", "2 * shared_inter_dim"
        ),
    },
    attrs={"topk": attr("Routed experts per token.", 10, param_name="topk")},
    constraints=(
        "topk <= n_routed",
        # makes the scales' padding a no-op
        "model_dim % 256 == 0 and inter_dim % 256 == 0",
    ),
    inputs={
        "hidden_states": _HIDDEN_STATES,
        "router_weight": _ROUTER_WEIGHT,
        "w1": TensorSpec(
            description="Gate-up projection per routed expert: gate rows, then up rows.",
            shape=("n_routed", "gate_up_rows", "model_dim"),
            dtype=_MXFP4,
            distribution=FiniteDistribution(),
            lifetime="weight",
        ),
        "w1_scale": TensorSpec(
            description="Per 1 x 32 block of w1.",
            shape=("n_routed", "gate_up_rows", "model_blocks"),
            dtype=_E8M0,
            distribution=_e8m0_scales("model_dim"),
            lifetime="weight",
        ),
        "w2": TensorSpec(
            description="Down projection per routed expert.",
            shape=("n_routed", "model_dim", "inter_dim"),
            dtype=_MXFP4,
            distribution=FiniteDistribution(),
            lifetime="weight",
        ),
        "w2_scale": TensorSpec(
            description="Per 1 x 32 block of w2.",
            shape=("n_routed", "model_dim", "inter_blocks"),
            dtype=_E8M0,
            distribution=_e8m0_scales("inter_dim"),
            lifetime="weight",
        ),
        "shared_w1": TensorSpec(
            description="The shared expert's gate-up projection on this rank: gate rows, then up rows.",
            shape=("shared_gate_up_rows", "model_dim"),
            dtype=_BF16,
            distribution=LecunNormalDistribution(fan_in="model_dim"),
            lifetime="weight",
        ),
        "shared_w2": TensorSpec(
            description="The shared expert's down projection on this rank.",
            shape=("model_dim", "shared_inter_dim"),
            dtype=_BF16,
            # 8 times a Lecun normal's spread: the shared expert then weighs about as much in
            # out as the routed experts (their MXFP4 weights are drawn about twice a Lecun
            # normal's per linear), so that out's bound also fails a kernel that drops the
            # shared expert or misgates it
            distribution=LecunNormalDistribution(fan_in="shared_inter_dim // 64"),
            lifetime="weight",
        ),
        "shared_gate_weight": _SHARED_GATE_WEIGHT,
    },
    outputs={"out": _moe_output(_MXFP4_MIN_SQNR_DB)},
    layout_variants=_MXFP4_LAYOUTS,
    compare=_COMPARE,
)


family = Plugin("sglang")
family.define(QWEN2_MOE_ROUTER)
family.define(QWEN2_MOE_FP8_EXPERTS)
family.define(QWEN2_MOE_FP8_ROUTER_EXPERTS)
family.define(QWEN2_MOE_MXFP4_MOE)
family.build()


@family.op
def qwen2_moe_router(hidden_states, router_weight, topk_weights, topk_ids): ...


@family.op
def qwen2_moe_fp8_experts(
    hidden_states,
    topk_weights,
    topk_ids,
    shared_gate_weight,
    w1,
    w1_scale,
    w2,
    w2_scale,
): ...


@family.op
def qwen2_moe_fp8_router_experts(
    hidden_states,
    router_weight,
    shared_gate_weight,
    w1,
    w1_scale,
    w2,
    w2_scale,
    topk=10,
): ...


@family.op
def qwen2_moe_mxfp4_moe(
    hidden_states,
    router_weight,
    w1,
    w1_scale,
    w2,
    w2_scale,
    shared_w1,
    shared_w2,
    shared_gate_weight,
    topk=10,
): ...


def _forward(self, hidden_states, forward_batch=None, defer_finalize=False):
    """The whole block, its BF16 shared expert apart from the MXFP4 experts (sglang's path for
    a checkpoint that keeps its shared expert BF16), as one op; then the sum over the TP group
    where sglang's forward does it. Every other block runs sglang's forward, whose router and
    experts reach the _forward_router_experts route."""
    import torch

    if (
        defer_finalize
        or not torch.is_tensor(hidden_states)
        or hidden_states.dim() != 2
        or moe_blocks.runtime_falls_back()
        or _unfused_block_falls_back(self)
    ):
        return FallbackReason.OUT_OF_CONTRACT
    experts, shared = self.experts, self.shared_expert
    out = qwen2_moe_mxfp4_moe(
        hidden_states=hidden_states,
        router_weight=self.gate.weight,
        w1=experts.w13_weight,
        w1_scale=experts.w13_weight_scale,
        w2=experts.w2_weight,
        w2_scale=experts.w2_weight_scale,
        shared_w1=shared.gate_up_proj.weight,
        shared_w2=shared.down_proj.weight,
        shared_gate_weight=self.shared_expert_gate.weight,
        topk=self.topk.topk_config.top_k,
    )
    if isinstance(out, FallbackReason) or not self.reduce_results:
        return out
    from sglang.srt.layers.moe import reduce_moe_output

    return reduce_moe_output(out)


def _forward_router_experts(self, hidden_states, *, defer_finalize=False):
    """The router and experts, the shared expert last, as one op; where it does not serve, the
    router op, then the experts op or sglang's experts on its routing. The output is this
    rank's share, as sglang's."""
    import torch

    if (
        defer_finalize
        # a quantized tuple
        or not torch.is_tensor(hidden_states)
        or moe_blocks.runtime_falls_back()
        or _block_falls_back(self)
    ):
        return FallbackReason.OUT_OF_CONTRACT
    experts, topk = self.experts, self.topk.topk_config.top_k
    expert_args = {
        "hidden_states": hidden_states,
        "shared_gate_weight": self.shared_expert_gate.weight,
        "w1": experts.w13_weight,
        "w1_scale": experts.w13_weight_scale_inv,
        "w2": experts.w2_weight,
        "w2_scale": experts.w2_weight_scale_inv,
    }
    out = qwen2_moe_fp8_router_experts(**expert_args, router_weight=self.gate.weight, topk=topk)
    if not isinstance(out, FallbackReason):
        return out
    shape, device = (hidden_states.shape[0], topk), hidden_states.device
    routed = qwen2_moe_router(
        hidden_states=hidden_states,
        router_weight=self.gate.weight,
        topk_weights=torch.empty(shape, dtype=torch.float32, device=device),
        topk_ids=torch.empty(shape, dtype=torch.int32, device=device),
    )
    if isinstance(routed, FallbackReason):
        return routed
    topk_weights, topk_ids = routed
    out = qwen2_moe_fp8_experts(**expert_args, topk_weights=topk_weights, topk_ids=topk_ids)
    if not isinstance(out, FallbackReason):
        return out
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    # as sglang's small-M router leaves it: no logits
    topk_output = self._append_shared_to_topk_output(
        StandardTopKOutput(topk_weights=topk_weights, topk_ids=topk_ids, router_logits=None), hidden_states
    )
    return self.experts(hidden_states, topk_output)


def _block_falls_back(block) -> bool:
    from sglang.srt.runtime_context import get_parallel

    return (
        # the experts op indexes w1 by the router's ids over every expert, not this rank's
        get_parallel().moe_ep_size != 1
        or block.num_fused_shared_experts != 1
        or moe_blocks.topk_falls_back(block.topk, scoring_func="softmax", biased=False)
        or moe_blocks.experts_fall_back(block.experts, ("Fp8MoEMethod",))
    )


def _unfused_block_falls_back(block) -> bool:
    """Whether the block runs other than as qwen2_moe_mxfp4_moe takes it: its shared expert a
    BF16 Qwen2MoeMLP apart from the experts, gated by sigmoid(hidden_states gate^T) as
    sglang's fused gate adds it, its experts Quark's MXFP4 on aiter over every routed expert."""
    import torch
    from sglang.srt.models import qwen2_moe
    from sglang.srt.runtime_context import get_forward, get_parallel

    gate = block.shared_expert_gate
    return (
        # w1 holds every routed expert, the router's ids index it
        get_parallel().moe_ep_size != 1
        # sequence parallelism gathers the shared expert's input and scatters its output
        or get_forward().sp_active
        or block.num_fused_shared_experts != 0
        or block.shared_expert is None
        or type(gate) is not torch.nn.Linear
        or gate.bias is not None
        # sglang hands the shared expert's add to its LoRA MoE dispatch
        or qwen2_moe._SGLANG_EXPERIMENTAL_LORA_OPTI
        or _bf16_linear_falls_back(block.gate, "ReplicatedLinear")
        or _shared_expert_falls_back(block.shared_expert)
        or moe_blocks.topk_falls_back(block.topk, scoring_func="softmax", biased=False)
        or moe_blocks.experts_fall_back(block.experts, ("QuarkW4A4MXFp4MoE",))
    )


def _shared_expert_falls_back(shared) -> bool:
    """Whether ``shared`` runs other than as down(silu(gate(x)) * up(x)) of its BF16 weights,
    its output this rank's share (summed over the TP group with the routed experts')."""
    up, down = shared.gate_up_proj, shared.down_proj
    return (
        type(shared).__name__ != "Qwen2MoeMLP"
        or type(shared.act_fn).__name__ != "SiluAndMul"
        or shared._enable_silu_fp4_quant_fusion
        or _bf16_linear_falls_back(up, "MergedColumnParallelLinear")
        or _bf16_linear_falls_back(down, "RowParallelLinear")
        or up.gather_output
        or not down.input_is_parallel
        # the down projection would sum the shared expert over the TP group itself
        or down.reduce_results
        or down.use_decode_attn_tp
    )


def _bf16_linear_falls_back(layer, kind: str) -> bool:
    """Whether ``layer`` is other than a plain ``kind`` without a bias on the unquantized
    method, which multiplies by its weight as stored (a LoRA layer wraps it in another
    class)."""
    return (
        type(layer).__name__ != kind
        or type(layer.quant_method).__name__ != "UnquantizedLinearMethod"
        or layer.bias is not None
    )


def declare(plugin: Plugin) -> None:
    plugin.wrap(
        "sglang.srt.models.qwen2_moe.Qwen2MoeSparseMoeBlock.forward",
        versions=("0.5.22", "0.5.23"),
        replace=_forward,
    )
    plugin.wrap(
        "sglang.srt.models.qwen2_moe.Qwen2MoeSparseMoeBlock._forward_router_experts",
        versions=("0.5.22", "0.5.23"),
        replace=_forward_router_experts,
    )
