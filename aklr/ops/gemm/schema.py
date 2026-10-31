# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from aklr.core.schema import Const, Schema, Tensor, Var, default_check
from aklr.ops.gemm.reference import (
    bmm_bf16_nn_reference,
    bmm_bf16_nn_shuffled_reference,
    bmm_bf16_nt_reference,
    bmm_bf16_nt_shuffled_reference,
    gemm_a8w8_nt_reference,
    gemm_a8w8_nt_shuffled_reference,
    gemm_bf16_nt_reference,
    gemm_bf16_nt_shuffled_reference,
    gemm_fp8_nt_blockscaled_mn_reference,
    gemm_fp8_nt_blockscaled_reference,
    gemm_fp8_nt_blockscaled_shuffled_k_reference,
    gemm_fp8_nt_blockscaled_shuffled_mn_reference,
    gemm_mxfp4_nt_reference,
    gemm_mxfp4_nt_shuffled_reference,
)

__all__ = [
    "BMM_BF16_NN",
    "BMM_BF16_NN_SHUFFLED",
    "BMM_BF16_NT",
    "BMM_BF16_NT_SHUFFLED",
    "GEMM_A8W8_NT",
    "GEMM_A8W8_NT_SHUFFLED",
    "GEMM_BF16_NT",
    "GEMM_BF16_NT_SHUFFLED",
    "GEMM_FP8_NT_BLOCKSCALED",
    "GEMM_FP8_NT_BLOCKSCALED_MN",
    "GEMM_FP8_NT_BLOCKSCALED_SHUFFLED_K",
    "GEMM_FP8_NT_BLOCKSCALED_SHUFFLED_MN",
    "GEMM_MXFP4_NT",
    "GEMM_MXFP4_NT_SHUFFLED",
    "bmm_bf16_nn_op_schema",
    "bmm_bf16_nt_op_schema",
    "gemm_a8w8_nt_op_schema",
    "gemm_bf16_nt_op_schema",
    "gemm_fp8_nt_blockscaled_op_schema",
    "gemm_mxfp4_nt_op_schema",
]

# nt: b is stored (N, K). scale_major_mode and use_shuffled_weight are fixed per
# schema, so they are constraints, not inputs; the api's dispatch function picks
# the schema by them.


def _scaled_out(description: str) -> Tensor:
    return Tensor(
        ["M", "N"],
        description=description,
        dtype="bfloat16",
        dtype_from="out",
        dtype_from_scalar="out_dtype",
    )


def _gemm_check(
    reference_outputs,
    actual_outputs,
    *,
    rtol=None,
    atol=None,
    max_mismatch_pct=0.0,
    min_cos_sim=None,
):
    # main's compare_gemm_outputs: atol = rtol 1e-3 fp16, 1e-2 otherwise, every element
    out = actual_outputs
    if isinstance(out, (list, tuple)):
        out = out[0]
    tolerance = 1e-3 if out.dtype == torch.float16 else 1e-2
    return default_check(
        reference_outputs,
        actual_outputs,
        rtol=tolerance if rtol is None else rtol,
        atol=tolerance if atol is None else atol,
        max_mismatch_pct=max_mismatch_pct,
        min_cos_sim=min_cos_sim,
    )


def _shuffled(schema: Schema, reference: Any, init: Any = None) -> Schema:
    return replace(
        schema,
        name_prefix=f"{schema.name_prefix}_shuffled",
        description=f"{schema.description} b is aiter's 16 x 16 shuffle_weight.",
        constraints=[
            "use_shuffled_weight" if c == "not use_shuffled_weight" else c for c in schema.constraints
        ],
        reference=reference,
        init=init,
    )


def _gemm_bf16_nt_init(*, M, N, K, device="cuda", seed=0):
    torch.manual_seed(seed)
    return {
        "a": torch.randn(M, K, device=device).to(torch.bfloat16),
        "b": torch.randn(N, K, device=device).to(torch.bfloat16),
        "out_dtype": torch.bfloat16,
    }


def _gemm_bf16_nt_shuffled_init(*, M, N, K, device="cuda", seed=0):
    return _gemm_bf16_nt_init(M=M, N=N, K=K, device=device, seed=seed) | {"use_shuffled_weight": True}


def _gemm_fp8_nt_blockscaled_inputs(M, N, K, shuffled, mn_major, device, seed):
    torch.manual_seed(seed)
    k_blocks, n_blocks = (K + 127) // 128, (N + 127) // 128
    a_scale_shape = (k_blocks, M) if mn_major else (M, k_blocks)
    return {
        "a": torch.randn(M, K, device=device).to(torch.float8_e4m3fn),
        "b": torch.randn(N, K, device=device).to(torch.float8_e4m3fn),
        "a_scale": torch.exp2(torch.empty(a_scale_shape, device=device).uniform_(-3, 0)),
        "b_scale": torch.exp2(torch.empty(n_blocks, k_blocks, device=device).uniform_(-3, 0)),
        "scale_major_mode": "MN" if mn_major else "K",
        "use_shuffled_weight": shuffled,
        "out_dtype": torch.bfloat16,
    }


def _gemm_fp8_nt_blockscaled_init(*, M, N, K, device="cuda", seed=0):
    return _gemm_fp8_nt_blockscaled_inputs(M, N, K, False, False, device, seed)


def _gemm_fp8_nt_blockscaled_mn_init(*, M, N, K, device="cuda", seed=0):
    return _gemm_fp8_nt_blockscaled_inputs(M, N, K, False, True, device, seed)


def _gemm_fp8_nt_blockscaled_shuffled_mn_init(*, M, N, K, device="cuda", seed=0):
    return _gemm_fp8_nt_blockscaled_inputs(M, N, K, True, True, device, seed)


def _gemm_fp8_nt_blockscaled_shuffled_k_init(*, M, N, K, device="cuda", seed=0):
    return _gemm_fp8_nt_blockscaled_inputs(M, N, K, True, False, device, seed)


def _by_b_storage(plain: Schema, shuffled: Schema) -> Any:
    def op_schema(*_: Any, use_shuffled_weight: bool = False, **__: Any) -> Schema:
        return shuffled if use_shuffled_weight else plain

    op_schema.schemas = (plain, shuffled)
    return op_schema


GEMM_BF16_NT = Schema(
    name_prefix="gemm_bf16_nt",
    op_type="gemm_bf16_nt",
    description="bf16 gemm: out = a @ b.T + bias, b stored (N, K).",
    axes={
        "M": Var("Tokens."),
        "N": Const("Output columns.", abbrev="n"),
        "K": Const("Reduction length.", abbrev="k"),
    },
    constraints=[
        "not use_shuffled_weight",
        "bias is None or bias.dtype in ('bfloat16', 'float16', 'float32')",
        "out.dtype in ('bfloat16', 'float16', 'float32')",
    ],
    inputs={
        "a": Tensor(["M", "K"], dtype="bfloat16", description="Activations."),
        "b": Tensor(["N", "K"], dtype="bfloat16", description="Weights."),
        "bias": Tensor(["N"], optional=True, description="Added to every output row."),
    },
    outputs={
        "out": Tensor(
            ["M", "N"],
            description="a @ b.T + bias.",
            dtype="bfloat16",
            dtype_from="out",
            dtype_from_scalar="out_dtype",
        )
    },
    reference=gemm_bf16_nt_reference,
    check=_gemm_check,
    init=_gemm_bf16_nt_init,
)

GEMM_A8W8_NT = Schema(
    name_prefix="gemm_a8w8_nt",
    op_type="gemm_a8w8_nt",
    description="Per-token 8-bit gemm: out = (a * a_scale) @ (b * b_scale).T + bias.",
    axes={
        "M": Var("Tokens."),
        "N": Const("Output columns.", abbrev="n"),
        "K": Const("Reduction length.", abbrev="k"),
        "one": Const("A scale per row.", abbrev="", value=1),
    },
    constraints=[
        "not use_shuffled_weight",
        "a.dtype in ('int8', 'float8_e4m3fn', 'float8_e5m2')",
        "b.dtype == a.dtype",
        "out.dtype in ('bfloat16', 'float16')",
    ],
    inputs={
        "a": Tensor(["M", "K"], description="Activations."),
        "b": Tensor(["N", "K"], description="Weights."),
        "a_scale": Tensor(["M", "one"], dtype="float32", description="Per row of a."),
        "b_scale": Tensor(["N", "one"], dtype="float32", description="Per output column."),
        "bias": Tensor(
            ["N"],
            dtype="float32",
            optional=True,
            description="Added to every output row.",
        ),
    },
    outputs={"out": _scaled_out("a @ b.T, scaled.")},
    reference=gemm_a8w8_nt_reference,
    check=_gemm_check,
)
GEMM_BF16_NT_SHUFFLED = _shuffled(GEMM_BF16_NT, gemm_bf16_nt_shuffled_reference, _gemm_bf16_nt_shuffled_init)
GEMM_A8W8_NT_SHUFFLED = _shuffled(GEMM_A8W8_NT, gemm_a8w8_nt_shuffled_reference)
gemm_bf16_nt_op_schema = _by_b_storage(GEMM_BF16_NT, GEMM_BF16_NT_SHUFFLED)
gemm_a8w8_nt_op_schema = _by_b_storage(GEMM_A8W8_NT, GEMM_A8W8_NT_SHUFFLED)


# Fixed 128 x 128 blocks: all aiter passes.
def _blockscaled(
    name: str,
    shuffled: bool,
    scale_major_mode: str,
    reference: Any,
    init: Any,
    constraints: tuple[str, ...] = (),
) -> Schema:
    return Schema(
        name_prefix=name,
        op_type="gemm_fp8_nt_blockscaled",
        description=(
            "fp8 gemm dequantized per block: a fp32 scale per 128 x 128 of b, per 1 x"
            " 128 of a. A shuffled b is aiter's 16 x 16 shuffle_weight."
            " scale_major_mode 'K' stores a_scale as (M, K_div_block), 'MN' as"
            " (K_div_block, M); b_scale is (N_div_block, K_div_block) in both."
        ),
        axes={
            "M": Var("Tokens."),
            "N": Const("Output columns.", abbrev="n"),
            "K": Const("Reduction length.", abbrev="k"),
            "N_div_block": Const("Weight-scale rows: ceil_div(N, 128).", abbrev=""),
            "K_div_block": Const("Scale columns: ceil_div(K, 128).", abbrev=""),
        },
        constraints=[
            "use_shuffled_weight" if shuffled else "not use_shuffled_weight",
            f"scale_major_mode == '{scale_major_mode}'",
            "N_div_block == ceil_div(N, 128)",
            "K_div_block == ceil_div(K, 128)",
            "a.dtype in ('float8_e4m3fn', 'float8_e5m2')",
            "b.dtype == a.dtype",
            "out.dtype in ('bfloat16', 'float16')",
            *constraints,
        ],
        inputs={
            "a": Tensor(["M", "K"], description="Activations."),
            "b": Tensor(["N", "K"], description="Weights."),
            "a_scale": Tensor(
                (["M", "K_div_block"] if scale_major_mode == "K" else ["K_div_block", "M"]),
                dtype="float32",
                description="Per 1 x 128 block of a.",
            ),
            "b_scale": Tensor(
                ["N_div_block", "K_div_block"],
                dtype="float32",
                description="Per 128 x 128 block of b.",
            ),
        },
        outputs={"out": _scaled_out("a @ b.T, scaled.")},
        reference=reference,
        check=_gemm_check,
        init=init,
    )


GEMM_FP8_NT_BLOCKSCALED = _blockscaled(
    "gemm_fp8_nt_blockscaled",
    False,
    "K",
    gemm_fp8_nt_blockscaled_reference,
    _gemm_fp8_nt_blockscaled_init,
)
GEMM_FP8_NT_BLOCKSCALED_MN = _blockscaled(
    "gemm_fp8_nt_blockscaled_mn",
    False,
    "MN",
    gemm_fp8_nt_blockscaled_mn_reference,
    _gemm_fp8_nt_blockscaled_mn_init,
)
# aiter's shuffled kernel: e4m3 only.
GEMM_FP8_NT_BLOCKSCALED_SHUFFLED_MN = _blockscaled(
    "gemm_fp8_nt_blockscaled_shuffled_mn",
    True,
    "MN",
    gemm_fp8_nt_blockscaled_shuffled_mn_reference,
    _gemm_fp8_nt_blockscaled_shuffled_mn_init,
    constraints=("a.dtype == 'float8_e4m3fn'",),
)
GEMM_FP8_NT_BLOCKSCALED_SHUFFLED_K = _blockscaled(
    "gemm_fp8_nt_blockscaled_shuffled_k",
    True,
    "K",
    gemm_fp8_nt_blockscaled_shuffled_k_reference,
    _gemm_fp8_nt_blockscaled_shuffled_k_init,
    constraints=("a.dtype == 'float8_e4m3fn'",),
)

_BLOCKSCALED = {
    (False, "K"): GEMM_FP8_NT_BLOCKSCALED,
    (False, "MN"): GEMM_FP8_NT_BLOCKSCALED_MN,
    (True, "MN"): GEMM_FP8_NT_BLOCKSCALED_SHUFFLED_MN,
    (True, "K"): GEMM_FP8_NT_BLOCKSCALED_SHUFFLED_K,
}


def gemm_fp8_nt_blockscaled_op_schema(
    *_: Any,
    use_shuffled_weight: bool = False,
    scale_major_mode: str = "MN",
    **__: Any,
) -> Schema:
    """The schema of a call by its storage."""
    return _BLOCKSCALED[(use_shuffled_weight, scale_major_mode)]


gemm_fp8_nt_blockscaled_op_schema.schemas = tuple(_BLOCKSCALED.values())


def _mxfp4(name: str, shuffled: bool, reference: Any) -> Schema:
    # Shapes are stored ones: fp4 packs two values a byte, so a and b hold K // 2.
    # aiter's scales come from dynamic_mxfp4_quant(shuffle=True): padded, then shuffled.
    axes = {
        "M": Var("Tokens."),
        "N": Const("Output columns.", abbrev="n"),
        "K": Const(
            "Reduction length.",
            abbrev="k",
            extract=lambda call: call["a"].shape[-1] * 2,
        ),
        "K_div_2": Const("Bytes per row of a and b: K // 2.", abbrev=""),
        "K_div_block_size": Const("Scale columns: K // 32.", abbrev=""),
    }
    a_descale, b_descale = ["M", "K_div_block_size"], ["N", "K_div_block_size"]
    constraints = ["not use_shuffled_weight"]
    if shuffled:
        axes |= {
            "M_padded": Var("a_descale rows, padded."),
            "N_padded": Const("b_descale rows, padded.", abbrev=""),
            "K_div_block_size_padded": Const("Scale columns, padded.", abbrev=""),
        }
        a_descale = ["M_padded", "K_div_block_size_padded"]
        b_descale = ["N_padded", "K_div_block_size_padded"]
        constraints = [
            "use_shuffled_weight",
            "M_padded == ceil_div(M, 256) * 256",
            "N_padded == ceil_div(N, 256) * 256",
            "K_div_block_size_padded == ceil_div(K_div_block_size, 8) * 8",
        ]
    return Schema(
        name_prefix=name,
        op_type="gemm_mxfp4_nt",
        description=(
            "mxfp4 gemm: (a * a_descale) @ (b * b_descale).T + bias, e8m0 scales"
            " per 32; a shuffled b is stored 16 x 16 and both scales"
            " padded to (256, 8) and shuffled."
        ),
        axes=axes,
        constraints=[
            "K_div_2 == K // 2",
            "K_div_block_size == K // 32",
            "out.dtype in ('bfloat16', 'float16')",
            *constraints,
        ],
        inputs={
            "a": Tensor(["M", "K_div_2"], dtype="float4_e2m1fn_x2", description="Activations."),
            "b": Tensor(["N", "K_div_2"], dtype="float4_e2m1fn_x2", description="Weights."),
            "a_descale": Tensor(a_descale, description="Per 1 x 32 block of a."),
            "b_descale": Tensor(b_descale, description="Per 1 x 32 block of b."),
            "bias": Tensor(
                ["N"],
                dtype="float32",
                optional=True,
                description="Added to every output row.",
            ),
        },
        outputs={"out": _scaled_out("a @ b.T, scaled.")},
        reference=reference,
        check=_gemm_check,
    )


GEMM_MXFP4_NT = _mxfp4("gemm_mxfp4_nt", False, gemm_mxfp4_nt_reference)
GEMM_MXFP4_NT_SHUFFLED = _mxfp4("gemm_mxfp4_nt_shuffled", True, gemm_mxfp4_nt_shuffled_reference)


gemm_mxfp4_nt_op_schema = _by_b_storage(GEMM_MXFP4_NT, GEMM_MXFP4_NT_SHUFFLED)


def _bmm(name: str, nt: bool, reference: Any) -> Schema:
    return Schema(
        name_prefix=name,
        op_type=name,
        description=(
            f"Batched bf16 gemm: out[i] = A[i] @ B[i]{'.T' if nt else ''}, B stored"
            f" {'(batch_size, N, K)' if nt else '(batch_size, K, N)'}."
        ),
        axes={
            "batch_size": Var("Independent matmuls."),
            "M": Var("Tokens."),
            "N": Const("Output columns.", abbrev="n"),
            "K": Const("Reduction length.", abbrev="k"),
        },
        constraints=[
            "not use_shuffled_weight",
            "out.dtype in ('bfloat16', 'float16', 'float32')",
        ],
        inputs={
            "A": Tensor(["batch_size", "M", "K"], dtype="bfloat16", description="Activations."),
            "B": Tensor(
                ["batch_size", "N", "K"] if nt else ["batch_size", "K", "N"],
                dtype="bfloat16",
                description="Weights.",
            ),
        },
        outputs={
            "out": Tensor(
                ["batch_size", "M", "N"],
                description="A[i] @ B[i] in the stored orientation.",
                dtype="bfloat16",
                dtype_from="out",
                dtype_from_scalar="out_dtype",
            )
        },
        reference=reference,
        check=_gemm_check,
    )


BMM_BF16_NT = _bmm("bmm_bf16_nt", True, bmm_bf16_nt_reference)
BMM_BF16_NN = _bmm("bmm_bf16_nn", False, bmm_bf16_nn_reference)
BMM_BF16_NT_SHUFFLED = _shuffled(BMM_BF16_NT, bmm_bf16_nt_shuffled_reference)
BMM_BF16_NN_SHUFFLED = _shuffled(BMM_BF16_NN, bmm_bf16_nn_shuffled_reference)
bmm_bf16_nt_op_schema = _by_b_storage(BMM_BF16_NT, BMM_BF16_NT_SHUFFLED)
bmm_bf16_nn_op_schema = _by_b_storage(BMM_BF16_NN, BMM_BF16_NN_SHUFFLED)
