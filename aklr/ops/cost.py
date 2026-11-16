# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The work of one call of each contract a roofline can rank (``OpContract.cost_at``)."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from aklr.core.contract.cost import Cost

__all__ = ["COSTS", "fused_moe_cost", "gemm_cost", "memory_cost"]

Symbols = Mapping[str, int | str | bool]


def memory_cost(symbols: Symbols, tensor_bytes: Mapping[str, int]) -> Cost:
    """A call that moves every tensor once and does no matrix math."""
    return Cost(0, sum(tensor_bytes.values()), None)


def gemm_cost(symbols: Symbols, tensor_bytes: Mapping[str, int]) -> Cost:
    """``2 * batch * m * n * k`` operations at the rate of ``a``'s dtype; every operand,
    scale and output moved once."""
    flops = 2 * int(symbols.get("batch", 1)) * int(symbols["m"]) * int(symbols["n"]) * int(symbols["k"])
    return Cost(flops, sum(tensor_bytes.values()), str(symbols["a.dtype"]))


def fused_moe_cost(symbols: Symbols, tensor_bytes: Mapping[str, int]) -> Cost:
    """Each route to one of this rank's experts runs that expert's gate-up and down
    projections, and only the experts some route reaches move their weights: routes are
    ``topk_ids``'s, uniform and distinct within a token. The math runs at the rate of the
    least precision the contract's bound admits: fp8 for 8-bit activations and for
    gate_mode 'interleave' (aiter's own fp8 past its row bound), bf16 for swiglu below
    256 rows, else fp4."""
    n_tokens, topk, n_experts = (int(symbols[a]) for a in ("n_tokens", "topk", "n_experts"))
    model_dim, inter_dim, gate_up_rows = (int(symbols[a]) for a in ("model_dim", "inter_dim", "gate_up_rows"))
    pool = int(symbols["global_experts"]) if symbols["has_expert_mask"] else n_experts
    local_routes = n_tokens * topk * n_experts / pool
    reached = 1 - (1 - topk / pool) ** n_tokens  # the share of experts at least one token routes to
    weights = sum(tensor_bytes[w] for w in ("w1", "w2", "w1_scale", "w2_scale"))
    moved = sum(tensor_bytes.values()) - weights + round(reached * weights)
    flops = round(2 * local_routes * model_dim * (gate_up_rows + inter_dim))
    if symbols["gate_mode"] == "interleave" or symbols["act_quant"] == 8:
        dtype = "float8_e4m3fn"
    else:
        dtype = "bfloat16" if symbols["activation"] == "swiglu" and n_tokens < 256 else "float4_e2m1"
    return Cost(flops, moved, dtype)


# by contract name; a contract absent here has no cost, and nothing guesses it
COSTS: Mapping[str, Callable[[Symbols, Mapping[str, int]], Cost]] = {
    **dict.fromkeys(("gemm", "gemm_a8w8", "gemm_a8w8_blockwise_scaled", "gemm_a4w4_mxfp4", "bmm"), gemm_cost),
    "fused_moe": fused_moe_cost,
    **dict.fromkeys(
        ("rms_norm", "add_rms_norm", "per_group_quant_fp8", "per_group_quant_mxfp8", "biased_grouped_topk"),
        memory_cost,
    ),
}
