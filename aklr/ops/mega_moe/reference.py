###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

from typing import TYPE_CHECKING

from aklr.ops.fused_moe.reference import fused_moe_gate_up_interleaved_reference

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["mega_moe_reference"]


def mega_moe_reference(hidden_states, topk_weights, topk_ids, swiglu_limit, group_calls) -> Tensor:
    """This rank's tokens through their top-k experts of the whole group, weighted, summed.

    Global expert e is local expert e % n_experts of rank e // n_experts. Each rank's experts are a fused_moe
    masked to them, which adds nothing for another rank's routes, so the group's output is their sum. Rounding
    points: the input of each linear in MXFP8, the gated activation in the input's dtype before it.
    """
    import torch

    n_experts = group_calls[0]["w1"].shape[0]
    owners = torch.arange(n_experts * len(group_calls), device=hidden_states.device) // n_experts
    return sum(
        fused_moe_gate_up_interleaved_reference(
            hidden_states,
            call["w1"],
            call["w2"],
            topk_weights,
            topk_ids,
            expert_mask=(owners == owner).int(),
            activation="silu",
            w1_scale=call["w1_scale"],
            w2_scale=call["w2_scale"],
            swiglu_limit=swiglu_limit,
            act_quant="mxfp8",
        )
        for owner, call in enumerate(group_calls)
    )
