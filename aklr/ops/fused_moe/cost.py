# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The work of one fused MoE call (``OpContract.cost_at``)."""

from __future__ import annotations

from collections.abc import Mapping

from aklr.core.contract.cost import Cost


def fused_moe_cost(symbols: Mapping[str, int | str | bool], tensor_bytes: Mapping[str, int]) -> Cost:
    """Each route to one of this rank's experts runs that expert's gate-up and down
    projections, and only the experts some route reaches move their weights: routes are
    ``topk_ids``'s, uniform and distinct within a token. Activations kept in 16 bits
    bound the math at the bf16 rate, 8-bit ones at the fp8 rate, else the fp4 rate."""
    n_tokens, topk, n_experts = (int(symbols[a]) for a in ("n_tokens", "topk", "n_experts"))
    model_dim, inter_dim, gate_up_rows = (int(symbols[a]) for a in ("model_dim", "inter_dim", "gate_up_rows"))
    pool = int(symbols["global_experts"]) if symbols["has_expert_mask"] else n_experts
    local_routes = n_tokens * topk * n_experts / pool
    reached = 1 - (1 - topk / pool) ** n_tokens  # the share of experts at least one token routes to
    weights = sum(tensor_bytes[w] for w in ("w1", "w2", "w1_scale", "w2_scale"))
    moved = sum(tensor_bytes.values()) - weights + round(reached * weights)
    flops = round(2 * local_routes * model_dim * (gate_up_rows + inter_dim))
    is_16_bit = symbols["gate_mode"] == "interleave" or (symbols["activation"] == "swiglu" and n_tokens < 256)
    dtype = "bfloat16" if is_16_bit else "float8_e4m3fn" if symbols["act_quant"] == 8 else "float4_e2m1"
    return Cost(flops, moved, dtype)
