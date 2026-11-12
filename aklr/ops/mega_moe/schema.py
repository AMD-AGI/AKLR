###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

import torch

from aklr.core.schema import Const, Scalar, Schema, Standard, Tensor, Var
from aklr.ops.fused_moe.schema import _fused_moe_gate_up_interleaved_init
from aklr.ops.mega_moe.reference import mega_moe_reference

__all__ = ["MEGA_MOE"]


def _mega_moe_check(reference_outputs, actual_outputs, **_unused):
    # flashinfer's DeepGEMM mega MoE test: every element within atol 0.15, rtol 0.05, relative L2 error < 0.02
    reference, actual = (
        o[0] if isinstance(o, (list, tuple)) else o for o in (reference_outputs, actual_outputs)
    )
    if reference.shape != actual.shape:
        return False
    reference, actual = reference.double(), actual.double()
    if not torch.isclose(actual, reference, rtol=0.05, atol=0.15).all():
        return False
    return (actual - reference).norm().item() < 0.02 * reference.norm().item()


def _mega_moe_init(*, n_tokens, model_dim, n_experts, topk, inter_dim, ep_size, device="cuda", seed=0):
    # one rank's call, the rank told apart by its seed: fused_moe's gate-up interleaved draw over the group's experts
    call = _fused_moe_gate_up_interleaved_init(
        n_tokens=n_tokens,
        model_dim=model_dim,
        n_experts=n_experts,
        topk=topk,
        inter_dim=inter_dim,
        global_experts=n_experts * ep_size,
        device=device,
        seed=seed,
    )
    names = ("hidden_states", "w1", "w2", "topk_weights", "topk_ids", "w1_scale", "w2_scale", "swiglu_limit")
    return {name: call[name] for name in names} | {"ep_size": ep_size}


MEGA_MOE = Schema(
    name_prefix="mega_moe_gate_up_interleaved",
    op_type="mega_moe",
    description=(
        "Expert-parallel MoE in one call on every rank of the expert-parallel group: each token of this rank goes"
        " to its top-k experts, held across the group's ranks (rank r holds global experts r * n_experts to"
        " (r + 1) * n_experts - 1), and out is the routing-weighted sum of their outputs, back on this rank. An"
        " expert is down(silu(min(gate, limit)) * clamp(up, -limit, limit)) of its mxfp4 weights with e8m0 scales"
        " per 1 x 32, each linear taking its input in MXFP8. Gate and up tiles interleaved in w1, the scales 2-D"
        " with columns padded to 8 and shuffled. Every rank of the group calls the op once per step, in the same"
        " order, with the same n_tokens."
    ),
    axes={
        "n_tokens": Var("Tokens of this rank in this call, the same on every rank of the group."),
        "ep_size": Const(
            "Ranks of the expert-parallel group: the ranks that call the op together.", abbrev="ep"
        ),
        "model_dim": Const("Hidden size."),
        "n_experts": Const("Experts on each rank."),
        "topk": Const("Experts per token."),
        "inter_dim": Const("Expert intermediate size.", extract=lambda call: call["w2"].shape[-1] * 2),
        "gate_up_rows": Const("w1 rows: 2 * inter_dim.", abbrev=""),
        "packed_model_dim": Const("Bytes per w1 row: model_dim // 2.", abbrev=""),
        "packed_inter_dim": Const("Bytes per w2 row: inter_dim // 2.", abbrev=""),
        "model_blocks": Const("w1 scale columns: model_dim // 32.", abbrev=""),
        "inter_blocks": Const("w2 scale columns: inter_dim // 32.", abbrev=""),
        "w1_scale_rows": Const("w1 scale rows, every expert's: n_experts * gate_up_rows.", abbrev=""),
        "w1_scale_columns": Const("w1 scale columns, padded: ceil_div(model_blocks, 8) * 8.", abbrev=""),
        "w2_scale_rows": Const("w2 scale rows, every expert's: n_experts * model_dim.", abbrev=""),
        "w2_scale_columns": Const("w2 scale columns, padded: ceil_div(inter_blocks, 8) * 8.", abbrev=""),
    },
    constraints=[
        "gate_up_rows == 2 * inter_dim",
        "packed_model_dim == model_dim // 2",
        "packed_inter_dim == inter_dim // 2",
        "model_blocks == model_dim // 32",
        "inter_blocks == inter_dim // 32",
        "w1_scale_rows == n_experts * gate_up_rows",
        "w1_scale_columns == ceil_div(model_blocks, 8) * 8",
        "w2_scale_rows == n_experts * model_dim",
        "w2_scale_columns == ceil_div(inter_blocks, 8) * 8",
        "model_dim % 128 == 0 and inter_dim % 128 == 0",
        "topk <= n_experts * ep_size",
        "hidden_states.dtype == 'bfloat16'",
        "topk_weights.dtype == 'float32'",
        "topk_ids.dtype == 'int32'",
    ],
    inputs={
        "hidden_states": Tensor(["n_tokens", "model_dim"], description="Token activations of this rank."),
        "w1": Tensor(
            ["n_experts", "gate_up_rows", "packed_model_dim"],
            dtype="float4_e2m1fn_x2",
            description="Gate-up projection of this rank's experts, two values a byte.",
        ),
        "w2": Tensor(
            ["n_experts", "model_dim", "packed_inter_dim"],
            dtype="float4_e2m1fn_x2",
            description="Down projection of this rank's experts, two values a byte.",
        ),
        "topk_weights": Tensor(["n_tokens", "topk"], description="Routing weight of each chosen expert."),
        "topk_ids": Tensor(
            ["n_tokens", "topk"], description="Chosen global experts per token, distinct within a row."
        ),
        "w1_scale": Tensor(
            ["w1_scale_rows", "w1_scale_columns"],
            dtype="float8_e8m0fnu",
            description="Per 1 x 32 block of w1.",
        ),
        "w2_scale": Tensor(
            ["w2_scale_rows", "w2_scale_columns"],
            dtype="float8_e8m0fnu",
            description="Per 1 x 32 block of w2.",
        ),
        "ep_size": Scalar("int32", description="Ranks of the expert-parallel group."),
        "swiglu_limit": Scalar(
            "float32", description="Clamped SwiGLU bound: gate at most this, up within +/- this."
        ),
    },
    outputs={
        "out": Tensor(
            ["n_tokens", "model_dim"],
            description="Routing-weighted sum of the chosen experts' outputs.",
            dtype_from="hidden_states",
        )
    },
    standard=Standard(
        reference=mega_moe_reference,
        # the Reference is fused_moe's over the group's experts: its outputs carry the same MXFP4 error
        check=_mega_moe_check,
        init=_mega_moe_init,
    ),
)
