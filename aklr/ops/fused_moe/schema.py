###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

from typing import Any

import torch

from aklr.core.schema import Const, Scalar, Schema, Tensor, Var
from aklr.ops.fused_moe.reference import (
    fused_moe_gate_up_interleaved_reference,
    fused_moe_reference,
    fused_moe_shuffled_16x16_reference,
)

__all__ = [
    "FUSED_MOE",
    "FUSED_MOE_GATE_UP_INTERLEAVED",
    "FUSED_MOE_SHUFFLED_16X16",
    "FUSED_MOE_SITUV2",
    "fused_moe_op_schema",
]


def _fused_moe_check(reference_outputs, actual_outputs, **_unused):
    # SQNR >= 13 dB, not a tolerance: the e33/e65 kernels keep fp8/MXFP4 partials,
    # far off aiter's error
    reference, actual = (
        o[0] if isinstance(o, (list, tuple)) else o for o in (reference_outputs, actual_outputs)
    )
    if reference.shape != actual.shape:
        return False
    reference, actual = reference.double(), actual.double()
    error = (actual - reference).norm().item()
    return error <= reference.norm().item() * 10 ** (-13.0 / 20)


def _e8m0_scales(shape, fan_in, device):
    lo = 122 - fan_in.bit_length() // 2
    codes = torch.randint(lo, lo + 7, shape, device=device)
    return codes.to(torch.uint8).view(torch.float8_e8m0fnu)


def _fused_moe_inputs(n_tokens, model_dim, n_experts, topk, inter_dim, global_experts, device, seed):
    torch.manual_seed(seed)
    id_range = n_experts if global_experts is None else global_experts
    w1 = torch.randint(0, 256, (n_experts, 2 * inter_dim, model_dim // 2), device=device).to(torch.uint8)
    w2 = torch.randint(0, 256, (n_experts, model_dim, inter_dim // 2), device=device).to(torch.uint8)
    call = {
        "hidden_states": torch.randn(n_tokens, model_dim, device=device).to(torch.bfloat16),
        "w1": w1.view(torch.float4_e2m1fn_x2),
        "w2": w2.view(torch.float4_e2m1fn_x2),
        "topk_weights": torch.randn(n_tokens, topk, device=device),
        "topk_ids": torch.randn(n_tokens, id_range, device=device).topk(topk, dim=-1).indices.to(torch.int32),
        "expert_mask": None,
        "activation": "silu",
        "use_shuffled_weight": True,
    }
    if global_experts is not None:
        rank = int(torch.randint(global_experts // n_experts, ()))
        mask = torch.zeros(global_experts, dtype=torch.int32, device=device)
        mask[rank * n_experts : (rank + 1) * n_experts] = 1
        call["expert_mask"] = mask
    return call


def _fused_moe_shuffled_16x16_init(
    *,
    n_tokens,
    model_dim,
    n_experts,
    topk,
    inter_dim,
    global_experts=None,
    device="cuda",
    seed=0,
):
    call = _fused_moe_inputs(n_tokens, model_dim, n_experts, topk, inter_dim, global_experts, device, seed)
    call["w1_scale"] = _e8m0_scales((n_experts, 2 * inter_dim, model_dim // 32), model_dim, device)
    call["w2_scale"] = _e8m0_scales((n_experts, model_dim, inter_dim // 32), inter_dim, device)
    call["gate_mode"] = "separated"
    call["act_quant"] = "mxfp4"
    return call


def _fused_moe_gate_up_interleaved_init(
    *,
    n_tokens,
    model_dim,
    n_experts,
    topk,
    inter_dim,
    global_experts=None,
    device="cuda",
    seed=0,
):
    call = _fused_moe_inputs(n_tokens, model_dim, n_experts, topk, inter_dim, global_experts, device, seed)
    w1_columns = (model_dim // 32 + 7) // 8 * 8
    w2_columns = (inter_dim // 32 + 7) // 8 * 8
    call["w1_scale"] = _e8m0_scales((n_experts * 2 * inter_dim, w1_columns), model_dim, device)
    call["w2_scale"] = _e8m0_scales((n_experts * model_dim, w2_columns), inter_dim, device)
    call["swiglu_limit"] = 10.0
    call["gate_mode"] = "interleave"
    call["act_quant"] = None
    return call


def _fused_moe_situv2_init(
    *,
    n_tokens,
    model_dim,
    n_experts,
    topk,
    inter_dim,
    global_experts=None,
    device="cuda",
    seed=0,
):
    call = _fused_moe_gate_up_interleaved_init(
        n_tokens=n_tokens,
        model_dim=model_dim,
        n_experts=n_experts,
        topk=topk,
        inter_dim=inter_dim,
        global_experts=global_experts,
        device=device,
        seed=seed,
    )
    del call["swiglu_limit"]
    # Kimi K3's SiTUv2
    call |= {"activation": "situv2", "beta": 4.0, "linear_beta": 25.0}
    call["gate_mode"] = "separated"
    call["act_quant"] = "mxfp8"
    return call


def _fused_moe(
    name: str,
    layout: str,
    reference: Any,
    shuffled: bool,
    gate_up_interleaved: bool,
    init: Any = None,
    situv2: bool = False,
) -> Schema:
    axes = {
        "n_tokens": Var("Tokens routed in this call."),
        "model_dim": Const("Hidden size."),
        "n_experts": Const("Local experts on this rank."),
        "topk": Const("Experts per token."),
        "inter_dim": Const(
            "Expert intermediate size.",
            extract=lambda call: call["w2"].shape[-1] * 2,
        ),
        "global_experts": Const("Experts of the whole expert-parallel group."),
        "gate_up_rows": Const("w1 rows: 2 * inter_dim.", abbrev=""),
        "packed_model_dim": Const("Bytes per w1 row: model_dim // 2.", abbrev=""),
        "packed_inter_dim": Const("Bytes per w2 row: inter_dim // 2.", abbrev=""),
        "model_blocks": Const("w1 scale columns: model_dim // 32.", abbrev=""),
        "inter_blocks": Const("w2 scale columns: inter_dim // 32.", abbrev=""),
        "one": Const("A single count.", abbrev="", value=1),
    }
    w1_scale = ["n_experts", "gate_up_rows", "model_blocks"]
    w2_scale = ["n_experts", "model_dim", "inter_blocks"]
    constraints = [
        "gate_up_rows == 2 * inter_dim",
        "packed_model_dim == model_dim // 2",
        "packed_inter_dim == inter_dim // 2",
        "model_blocks == model_dim // 32",
        "inter_blocks == inter_dim // 32",
        "expert_mask is not None or topk <= n_experts",
        "expert_mask is None or topk <= global_experts",
        "expert_mask is None or n_experts <= global_experts",
        "model_dim % 128 == 0 and inter_dim % 128 == 0",
        "hidden_states.dtype in ('bfloat16', 'float16')",
        "topk_weights.dtype in ('float32', 'bfloat16', 'float16')",
        "topk_ids.dtype in ('int32', 'int64')",
        "num_local_tokens is None or expert_mask is not None",
        "activation in ('silu', 'gelu', 'situv2')",
        "(activation == 'situv2') == (beta is not None)",
        "(beta is None) == (linear_beta is None)",
        "act_quant is None or act_quant in ('mxfp4', 'mxfp8')",
        "use_shuffled_weight" if shuffled else "not use_shuffled_weight",
        f"gate_mode == '{'interleave' if gate_up_interleaved and not situv2 else 'separated'}'",
    ]
    if situv2:
        constraints += ["activation == 'situv2'"]
    if gate_up_interleaved:
        axes |= {
            "w1_scale_rows": Const("w1 scale rows, every expert's: n_experts * gate_up_rows.", abbrev=""),
            "w1_scale_columns": Const("w1 scale columns, padded: ceil_div(model_blocks, 8) * 8.", abbrev=""),
            "w2_scale_rows": Const("w2 scale rows, every expert's: n_experts * model_dim.", abbrev=""),
            "w2_scale_columns": Const("w2 scale columns, padded: ceil_div(inter_blocks, 8) * 8.", abbrev=""),
        }
        w1_scale = ["w1_scale_rows", "w1_scale_columns"]
        w2_scale = ["w2_scale_rows", "w2_scale_columns"]
        constraints += [
            "w1_scale_rows == n_experts * gate_up_rows",
            "w1_scale_columns == ceil_div(model_blocks, 8) * 8",
            "w2_scale_rows == n_experts * model_dim",
            "w2_scale_columns == ceil_div(inter_blocks, 8) * 8",
        ]
    else:
        # the (256, 8) scale padding is a no-op: the scales keep their row-major shapes
        constraints += ["model_dim % 256 == 0 and inter_dim % 256 == 0"]
    return Schema(
        name_prefix=name,
        op_type="fused_moe",
        description=(
            "out = sum over the top-k experts e of topk_weights[e] * (act(gate) * up)"
            " @ w2[e].T, [gate | up] = hidden_states @ w1[e].T; mxfp4 weights with e8m0"
            f" scales per 1 x 32. {layout}"
        ),
        axes=axes,
        constraints=constraints,
        inputs={
            "hidden_states": Tensor(["n_tokens", "model_dim"], description="Token activations."),
            "w1": Tensor(
                ["n_experts", "gate_up_rows", "packed_model_dim"],
                dtype="float4_e2m1fn_x2",
                description="Gate-up projection per expert, two values a byte.",
            ),
            "w2": Tensor(
                ["n_experts", "model_dim", "packed_inter_dim"],
                dtype="float4_e2m1fn_x2",
                description="Down projection per expert, two values a byte.",
            ),
            "topk_weights": Tensor(
                ["n_tokens", "topk"],
                description="Routing weight of each chosen expert.",
            ),
            "topk_ids": Tensor(
                ["n_tokens", "topk"],
                description=(
                    "Chosen experts per token, distinct within a row; global ids under expert parallelism."
                ),
            ),
            "expert_mask": Tensor(
                ["global_experts"],
                dtype="int32",
                optional=True,
                description=("1 for the experts this rank holds, in the order w1 holds them."),
            ),
            "activation": Scalar("string", optional=True, description="Gate activation."),
            "act_quant": Scalar(
                "string",
                optional=True,
                description=(
                    "Quantization of each expert gemm's activations, per 1 x 32 with"
                    " e8m0 scales: 'mxfp4', 'mxfp8' (e4m3), or None for none."
                ),
            ),
            "w1_scale": Tensor(w1_scale, dtype="float8_e8m0fnu", description="Per 1 x 32 block of w1."),
            "w2_scale": Tensor(w2_scale, dtype="float8_e8m0fnu", description="Per 1 x 32 block of w2."),
            "swiglu_limit": Scalar(
                "float32",
                optional=True,
                description=("Clamped SwiGLU bound: gate at most this, up within +/- this."),
            ),
            "beta": Scalar("float32", optional=True, description="SiTUv2 gate saturation."),
            "linear_beta": Scalar("float32", optional=True, description="SiTUv2 up saturation."),
            "num_local_tokens": Tensor(
                ["one"],
                dtype="int32",
                optional=True,
                description="Rows this rank received; later rows are unused.",
            ),
        },
        outputs={
            "out": Tensor(
                ["n_tokens", "model_dim"],
                description=(
                    "Routing-weighted sum of the chosen experts' outputs;"
                    " under expert parallelism, this rank's share."
                ),
                dtype_from="hidden_states",
            )
        },
        reference=reference,
        check=_fused_moe_check,
        init=init,
    )


FUSED_MOE = _fused_moe(
    "fused_moe",
    "Row-major weights, (n_experts, rows, blocks) scales.",
    fused_moe_reference,
    shuffled=False,
    gate_up_interleaved=False,
)
FUSED_MOE_SHUFFLED_16X16 = _fused_moe(
    "fused_moe_shuffled_16x16",
    "Weights in aiter's shuffle_weight 16 x 16 tiles, scales in its shuffle_scale order.",
    fused_moe_shuffled_16x16_reference,
    shuffled=True,
    gate_up_interleaved=False,
    init=_fused_moe_shuffled_16x16_init,
)
FUSED_MOE_GATE_UP_INTERLEAVED = _fused_moe(
    "fused_moe_gate_up_interleaved",
    "Gate and up tiles interleaved in w1, the scales 2-D with columns padded to 8 and shuffled.",
    fused_moe_gate_up_interleaved_reference,
    shuffled=True,
    gate_up_interleaved=True,
    init=_fused_moe_gate_up_interleaved_init,
)
FUSED_MOE_SITUV2 = _fused_moe(
    "fused_moe_situv2_gate_up_interleaved",
    "SiTUv2 on the gate-up interleaved storage: gate and up tiles interleaved in w1, the scales"
    " 2-D with columns padded to 8 and shuffled.",
    fused_moe_gate_up_interleaved_reference,
    shuffled=True,
    gate_up_interleaved=True,
    init=_fused_moe_situv2_init,
    situv2=True,
)


def fused_moe_op_schema(
    *_: Any,
    use_shuffled_weight: bool = False,
    gate_mode: str = "separated",
    activation: str | None = None,
    **__: Any,
) -> Schema | None:
    if gate_mode == "interleave":
        return FUSED_MOE_GATE_UP_INTERLEAVED if use_shuffled_weight else None
    if activation == "situv2":  # aiter's SiTUv2 reads the gate-up interleaved storage only
        return FUSED_MOE_SITUV2 if use_shuffled_weight else None
    return FUSED_MOE_SHUFFLED_16X16 if use_shuffled_weight else FUSED_MOE


fused_moe_op_schema.schemas = (
    FUSED_MOE,
    FUSED_MOE_SHUFFLED_16X16,
    FUSED_MOE_GATE_UP_INTERLEAVED,
    FUSED_MOE_SITUV2,
)
