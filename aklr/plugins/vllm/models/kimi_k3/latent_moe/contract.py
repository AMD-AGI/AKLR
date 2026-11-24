# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3's latent-MoE tail as vLLM runs it on ROCm under tensor parallelism (``ROCmLatentMoERunner.
_shard_up_proj_tail``, ``models/kimi_k3/amd/latent_moe_runner.py``).

Each rank holds its partial of the routed experts' latent output (its experts' columns) and of the shared experts'
output (its intermediate columns). vLLM all-reduces the latent, normalizes it (the latent RMSNorm), adds this rank's
up-projection shard into its own columns of the shared output (one addmm in place), and all-reduces the shared output:
every rank ends with the layer's MoE output. The shared partial is written in place, as vLLM's addmm writes it (vLLM
reads it no more), so that vLLM's tail as vLLM calls it is the op."""

from __future__ import annotations

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    DType,
    FiniteDistribution,
    LecunNormalDistribution,
    ToleranceBound,
)
from aklr.core.contract.spec import (
    CompareSpec,
    DeterminismLevel,
    DTypeSpec,
    LayoutVariant,
    LayoutVariantsSpec,
    ScalarSpec,
    TensorSpec,
    attr,
    axis,
    derived_axis,
)
from aklr.plugins.vllm.models.kimi_k3.latent_moe.baseline import KimiK3LatentMoeTailBaseline
from aklr.plugins.vllm.models.kimi_k3.latent_moe.reference import KimiK3LatentMoeTailReference

__all__ = ["CONTRACTS", "KIMI_K3_LATENT_MOE_TAIL"]

_SAME_AS_ROUTED = DTypeSpec(same_as_tensor="routed")
_FINITE = FiniteDistribution()
# a kernel's bf16 rounding where the reference's differs (see each use), plus one bf16 rounding (2**-7): the
# reference rounds its outputs to bf16 too, and the compare no longer adds the output's own
_ONE_ROUNDING = ToleranceBound(absolute_tolerance=2.0**-5, relative_tolerance=2.0**-8 + 2.0**-7)

KIMI_K3_LATENT_MOE_TAIL = OpContract(
    contract_id="kimi_k3_latent_moe_tail@1.0",
    description=(
        "Every rank r of the tensor-parallel group calls the op once, with its own routed, shared and up_weight. "
        "latent = bf16(sum over ranks of routed); n_r = bf16(latent * rsqrt(mean(latent**2) + eps) * norm_weight_r), "
        "the mean over the latent axis; rank r's hidden columns are r * hidden_shard .. (r + 1) * hidden_shard - 1, "
        "and its shared partial becomes s_r = shared_r with those columns bf16(shared_r + n_r @ up_weight_r.T), in "
        "place; out = the sum over ranks of s_r, the same on every rank (vLLM's all-reduce, RMSNorm, addmm in place "
        "and all-reduce). The sums over ranks are in float32 before their bf16 rounding."
    ),
    op_type="latent_moe",
    axes={
        "n_tokens": axis("Rows of the step, the same on every rank.", 1, 65536, is_var=True),
        "hidden_size": axis("Width of the model's hidden rows.", 256, 16384, multiple_of=256),
        "latent_size": axis("Width of the routed experts' latent rows.", 128, 8192, multiple_of=128),
        "hidden_shard": derived_axis("Hidden columns a rank up-projects.", "hidden_size // tp_size"),
    },
    attrs={
        "tp_size": attr(
            "Ranks of the tensor-parallel group.", 8, 1, default=8, param_name="tp_size", abbrev="tp"
        ),
    },
    constraints=("hidden_size % tp_size == 0", "latent_size % tp_size == 0"),
    inputs={
        "routed": TensorSpec(
            description="This rank's partial of the routed experts' latent output (vLLM's fused_output).",
            shape=("n_tokens", "latent_size"),
            dtype=DTypeSpec(any_of=(DType.BFLOAT16,)),
            distribution=_FINITE,
        ),
        "shared": TensorSpec(
            description="This rank's partial of the shared experts' output (vLLM's shared_output); its own columns "
            "become s_r's, in place.",
            shape=("n_tokens", "hidden_size"),
            dtype=_SAME_AS_ROUTED,
            distribution=_FINITE,
            mutable=True,
            # one bf16 rounding of shared_r + n_r @ up_weight_r.T in its own columns, float32 sums in another order:
            # one bf16 unit apart at a tie, below 2**-7 |s_r|; the rest unchanged
            error_bounds=_ONE_ROUNDING,
        ),
        "up_weight": TensorSpec(
            description="This rank's rows of the routed up-projection: hidden columns r * hidden_shard .. on.",
            shape=("hidden_shard", "latent_size"),
            dtype=_SAME_AS_ROUTED,
            distribution=LecunNormalDistribution(fan_in="latent_size"),
            lifetime="weight",
        ),
        "norm_weight": TensorSpec(
            description="The latent RMSNorm's weight (vLLM's routed_expert_norm), the same on every rank in vLLM.",
            shape=("latent_size",),
            dtype=_SAME_AS_ROUTED,
            distribution=_FINITE,
            lifetime="weight",
        ),
        "eps": ScalarSpec(
            description="The latent RMSNorm's epsilon (vLLM's variance_epsilon).",
            dtype=DType.FLOAT32,
            interval=(2.0**-126, 1.0),
            default=1e-5,
        ),
    },
    outputs={
        "out": TensorSpec(
            description="The layer's MoE output, bitwise the same on every rank.",
            shape=("n_tokens", "hidden_size"),
            dtype=_SAME_AS_ROUTED,
            # vLLM rounds a rank's own columns twice (its addmm, then the all-reduce); a kernel may round once: half
            # a bf16 unit of shared_r + n_r @ up_weight_r.T apart, below 2**-5 for |values| < 16 (the draws: ~N(0, 2))
            error_bounds=_ONE_ROUNDING,
        ),
    },
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
    compare=CompareSpec(determinism=DeterminismLevel.BITWISE),
    ranks="tp_size",
    reference=KimiK3LatentMoeTailReference(),
    baseline=KimiK3LatentMoeTailBaseline(),
)

CONTRACTS = (KIMI_K3_LATENT_MOE_TAIL,)
