# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import torch

from aklr.core.schema import Const, Scalar, Schema, Tensor, Var, default_check
from aklr.ops.mla_prefill.reference import sparse_prefill_mxfp8_reference

__all__ = ["SPARSE_PREFILL_MXFP8"]


def _sparse_prefill_check(
    reference_outputs,
    actual_outputs,
    *,
    rtol=None,
    atol=None,
    max_mismatch_pct=0.0,
    min_cos_sim=None,
):
    return default_check(
        reference_outputs,
        actual_outputs,
        rtol=1e-2 if rtol is None else rtol,
        atol=1e-2 if atol is None else atol,
        max_mismatch_pct=max_mismatch_pct,
        min_cos_sim=min_cos_sim,
    )


def _sparse_prefill_init(
    *,
    n_tokens,
    n_heads,
    prefix_rows,
    extend_rows,
    prefix_nnz,
    extend_nnz,
    n_offsets,
    device="cuda",
    seed=0,
):
    torch.manual_seed(seed)

    def nope_rows(*shape):
        # 448 E4M3FN codes, 14 E8M0 scales, padding to 512 bytes; the codes repeat past 251 rows
        rows = 1
        for n in shape:
            rows *= n
        distinct = min(rows, 251)
        codes = torch.randn(distinct, 448, device=device).to(torch.float8_e4m3fn).view(torch.uint8)
        raw = torch.zeros(rows, 512, dtype=torch.uint8, device=device)
        raw[:, :448] = codes.repeat(-(-rows // distinct), 1)[:rows]
        raw[:, 448:462] = torch.randint(124, 128, (rows, 14), dtype=torch.uint8, device=device)
        return raw.reshape(*shape, 512)

    def indptr(total):
        # 0..total, split by weights exp(N(0, 1) clipped to +-2); the last offset pinned
        shares = torch.randn(n_offsets - 1, dtype=torch.float64).clamp(-2, 2).exp().cumsum(0)
        ends = (shares / shares[-1:] * total).floor().long()
        ends[-1:] = total
        return torch.cat((ends.new_zeros(1), ends)).to(torch.int32).to(device)

    def bf16(*shape):
        return torch.randn(*shape, device=device).to(torch.bfloat16)

    call = {"q_nope": nope_rows(n_tokens, n_heads), "q_rope": bf16(n_tokens, n_heads, 64)}
    for pool, rows, nnz in (("prefix", prefix_rows, prefix_nnz), ("extend", extend_rows, extend_nnz)):
        call[f"{pool}_nope"] = nope_rows(rows)
        call[f"{pool}_rope"] = bf16(rows, 64)
        call[f"{pool}_indices"] = torch.randint(0, rows, (nnz,), device=device).to(torch.int32)
        call[f"{pool}_indptr"] = indptr(nnz)
    call["sinks"] = torch.randn(n_heads, device=device)
    call["softmax_scale"] = 512**-0.5
    return call


def _pool(prefix: str) -> dict[str, Tensor]:
    return {
        f"{prefix}_nope": Tensor(
            [f"{prefix}_rows", "nope_bytes"],
            description="Packed NoPE rows: 448 FP8 codes, 14 E8M0 scales, padding to 512 bytes.",
        ),
        f"{prefix}_rope": Tensor(
            [f"{prefix}_rows", "rope_dim"], description="RoPE keys and values corresponding to the NoPE rows."
        ),
        f"{prefix}_indices": Tensor([f"{prefix}_nnz"], description="CSR entries selecting rows of the pool."),
        f"{prefix}_indptr": Tensor(
            ["n_offsets"], description="Monotone CSR offsets for each query, beginning with zero."
        ),
    }


SPARSE_PREFILL_MXFP8 = Schema(
    op_type="sparse_prefill_mxfp8",
    description=(
        "Sparse attention over two CSR ranges. NoPE uses block-scaled E4M3FN and "
        "RoPE BF16; each head's sink contributes only to the softmax denominator."
    ),
    axes={
        "n_tokens": Var("Query rows."),
        "n_heads": Const("Query heads."),
        "prefix_rows": Var("Rows of the prefix pool."),
        "extend_rows": Var("Rows of the extend pool."),
        "prefix_nnz": Var("Prefix CSR entries."),
        "extend_nnz": Var("Extend CSR entries."),
        "n_offsets": Var("CSR offsets: queries plus one."),
        "nope_bytes": Const("Bytes of a packed NoPE row.", abbrev="", value=512),
        "rope_dim": Const("RoPE values a row.", abbrev="", value=64),
    },
    constraints=[
        "n_offsets == n_tokens + 1",
        "n_heads % 16 == 0",
        "q_nope.dtype == 'uint8' and q_rope.dtype == 'bfloat16'",
    ],
    inputs={
        "q_nope": Tensor(["n_tokens", "n_heads", "nope_bytes"], description="Packed NoPE query rows."),
        "q_rope": Tensor(["n_tokens", "n_heads", "rope_dim"], description="Query RoPE components."),
        **_pool("prefix"),
        **_pool("extend"),
        "sinks": Tensor(["n_heads"], description="Per-head softmax sink logits."),
        "softmax_scale": Scalar("float32", description="Multiplier of the combined QK score."),
    },
    outputs={
        "output": Tensor(
            ["n_tokens", "n_heads", "nope_bytes"],
            description="Weighted sum of 448 dequantized NoPE and 64 RoPE values.",
            dtype="bfloat16",
        ),
    },
    reference=sparse_prefill_mxfp8_reference,
    check=_sparse_prefill_check,
    init=_sparse_prefill_init,
)
