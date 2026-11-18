# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""GLM-5.3 (FP8)'s monokernel spec (RFC sgl-project/sglang#43949), as sgl-amd-mono's glm53: a routed
decode or MTP verify step calls each decoder layer's op (``api.py``), then the head's; SGLang's
forward runs all other steps."""

from __future__ import annotations

from sglang.kernels.plugin_kernels.amd.spec import (
    DeviceLimits,
    Feature,
    MonoSpec,
    Numerics,
    NumericsKind,
    Requirements,
)

from aklr.plugins.sglang.monokernels.glm53 import HF_CONFIG

# the rows a routed step may have; a step whose rows no kernel serves runs SGLang's forward
ROWS = tuple(range(1, 9))
TP_SIZES = (4, 8)
# one workgroup a CU on MI355X
RESIDENT_BLOCKS = 256

# SGLang reads the kind alone; TODO: each launch's bounds, from its contract's
NUMERICS = Numerics(NumericsKind.TOLERANCE, {"layer.streams.cosine": 0.95})


def _config_disable_reasons(cfg, hf_config) -> list[str]:
    """Before load: the checks ``Requirements`` cannot state."""
    reasons = [
        f"needs GLM-5.3's {key} = {want!r}, got {getattr(hf_config, key, None)!r}"
        for key, want in HF_CONFIG.items()
        if getattr(hf_config, key, None) != want
    ]
    quant = getattr(hf_config, "quantization_config", None) or {}
    if quant.get("quant_method") != "fp8" or list(quant.get("weight_block_size") or ()) != [128, 128]:
        reasons.append("needs the FP8 block-128 checkpoint")
    return reasons


def _model_disable_reasons(model, hf_config, copies: dict) -> list[str]:
    """After load: the weights' layouts, and a validated kernel for every launch at the server's
    widths and TP size (TODO)."""
    return ["the GLM-5.3 launches have no kernels yet"]


def _build(model, hf_config, ctx):
    """Each layer's weights and caches bound to its op, and the forward SGLang routes to (TODO)."""
    raise NotImplementedError


SPEC = MonoSpec(
    name="glm53_aklr",
    devices={"gfx950": DeviceLimits(widths=ROWS, tp_sizes=TP_SIZES, resident_blocks=RESIDENT_BLOCKS)},
    features=frozenset({Feature.SPECULATIVE_DECODING}),
    requires=Requirements(
        dtypes=frozenset({"bfloat16"}),
        attention_backends=frozenset({"dsa"}),
        server_args={
            "kv_cache_dtype": frozenset({"auto", "bfloat16", "fp8_e4m3"}),
            # the experts in AITER's shuffled layout
            "moe_runner_backend": frozenset({"auto", "aiter"}),
            # MTP: the NextN draft's verify steps (NEXTN resolves to EAGLE)
            "speculative_algorithm": frozenset({None, "EAGLE"}),
            # HiSparse maps the logical KV slots to device rows; the launches write raw slots
            "enable_hisparse": frozenset({False}),
            # the launches record no expert distribution
            "expert_distribution_recorder_mode": frozenset({None}),
        },
        env={
            # a layer reading the previous top-k reads KV slots; unfused top-k hands out positions
            "SGLANG_DSA_FUSE_TOPK": True,
            # the launches route with the model's own router
            "SGLANG_SIMULATE_UNIFORM_EXPERTS": False,
            "SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS": False,
        },
    ),
    numerics=NUMERICS,
    model_disable_reasons=_model_disable_reasons,
    build=_build,
    api_version=1,  # SGLang's MONOKERNEL_API_VERSION
    config_disable_reasons=_config_disable_reasons,
)
