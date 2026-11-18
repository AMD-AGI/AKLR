# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import torch

from aklr.core.schema import Const, Scalar, Schema, Standard, Tensor, Var, default_check
from aklr.plugins.aiter.mla_prefill.baseline import (
    mla_prefill_varlen_baseline,
    mla_prefill_varlen_lse_baseline,
    pa_sparse_prefill_fp8_opus_baseline,
)
from aklr.plugins.aiter.mla_prefill.reference import (
    mla_prefill_varlen_lse_reference,
    mla_prefill_varlen_reference,
    sparse_prefill_mxfp8_reference,
)

__all__ = ["MLA_PREFILL_VARLEN_LSE_VARIANTS", "MLA_PREFILL_VARLEN_VARIANTS", "SPARSE_PREFILL_MXFP8"]


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
    standard=Standard(
        reference=sparse_prefill_mxfp8_reference,
        check=_sparse_prefill_check,
        init=_sparse_prefill_init,
        baseline=pa_sparse_prefill_fp8_opus_baseline,
    ),
)


def _mla_prefill_varlen_check(reference_outputs, actual_outputs, **_unused):
    # shortcut: the bounds of a qkv_fp8 call, the only one the inits draw; a bf16 call's are 1e-2 on the output
    # and 1e-3 on the lse, once a variant draws one.
    # The output to a 20 dB SQNR: the reference attends over q, k and v rounded to fp8 too, as the kernels may,
    # so the bound is on the attention itself. The lse to 0.25: q . k on fp8-rounded operands moves an N(0, 1)
    # logit by up to ~0.1, and a row reading few keys carries that into its lse.
    references, actuals = (
        list(o) if isinstance(o, (list, tuple)) else [o] for o in (reference_outputs, actual_outputs)
    )
    if [r.shape for r in references] != [a.shape for a in actuals]:
        return False
    output, actual = references[0].double(), actuals[0].double()
    if not (actual - output).norm().item() <= output.norm().item() * 10 ** (-20.0 / 20):
        return False
    return all(
        bool(torch.isclose(a.float(), r.float(), rtol=1e-3, atol=0.25).all())
        for r, a in zip(references[1:], actuals[1:])
    )


def _mla_prefill_varlen_call(n_q, n_kv, n_offsets, n_heads, qk_head_dim, v_head_dim, min_keys, device, seed):
    """A qkv_fp8 call of n_offsets - 1 sequences; with min_keys, over a prefix of at least that many keys each."""
    torch.manual_seed(seed)

    def cu_seqlens(min_seqlen, total):
        # 0..total, each sequence at least min_seqlen rows; the spare ones split by weights exp(N(0, 1) clipped
        # to +-2)
        parts = n_offsets - 1
        shares = torch.randn(parts, dtype=torch.float64).clamp(-2, 2).exp().cumsum(0)
        ends = (shares / shares[-1:] * (total - parts * min_seqlen)).floor().long()
        ends[-1:] = total - parts * min_seqlen
        ends += min_seqlen * torch.arange(1, parts + 1)
        return torch.cat((ends.new_zeros(1), ends)).to(torch.int32).to(device)

    def bf16(*shape):
        return torch.randn(*shape, device=device).to(torch.bfloat16)

    call = {
        "q": bf16(n_q, n_heads, qk_head_dim),
        "k": bf16(n_kv, n_heads, qk_head_dim),
        "v": bf16(n_kv, n_heads, v_head_dim),
        "cu_seqlens_q": cu_seqlens(1, n_q),
        "qkv_fp8": True,
        # MLA's, 1/sqrt(128 + 64): at 1, N(0, 1) logits over 192 values reach +-60, past aiter's fp8 prefill
        "sm_scale": 192**-0.5,
    }
    if min_keys is not None:
        call["cu_seqlens_k"] = cu_seqlens(min_keys, n_kv)
    return call


def _mla_prefill_varlen_init(
    *, n_q, n_kv, n_offsets, n_heads, qk_head_dim, v_head_dim, device="cuda", seed=0
):
    return _mla_prefill_varlen_call(
        n_q, n_kv, n_offsets, n_heads, qk_head_dim, v_head_dim, None, device, seed
    )


def _mla_prefill_varlen_prefix_init(
    *, n_q, n_kv, n_offsets, n_heads, qk_head_dim, v_head_dim, device="cuda", seed=0
):
    # each sequence's keys are at least all the queries: none of its queries reads no key
    return _mla_prefill_varlen_call(n_q, n_kv, n_offsets, n_heads, qk_head_dim, v_head_dim, n_q, device, seed)


def _mla_prefill_varlen_lse_init(
    *, n_q, n_kv, n_offsets, n_heads, qk_head_dim, v_head_dim, device="cuda", seed=0
):
    call = _mla_prefill_varlen_call(
        n_q, n_kv, n_offsets, n_heads, qk_head_dim, v_head_dim, None, device, seed
    )
    return {**call, "causal": True}


def _mla_prefill_varlen_lse_no_causal_init(
    *, n_q, n_kv, n_offsets, n_heads, qk_head_dim, v_head_dim, device="cuda", seed=0
):
    call = _mla_prefill_varlen_call(
        n_q, n_kv, n_offsets, n_heads, qk_head_dim, v_head_dim, None, device, seed
    )
    return {**call, "causal": False}


def _mla_prefill_varlen_lse_prefix_init(
    *, n_q, n_kv, n_offsets, n_heads, qk_head_dim, v_head_dim, device="cuda", seed=0
):
    call = _mla_prefill_varlen_call(n_q, n_kv, n_offsets, n_heads, qk_head_dim, v_head_dim, 1, device, seed)
    return {**call, "causal": True}


def _mla_prefill_varlen_lse_prefix_no_causal_init(
    *, n_q, n_kv, n_offsets, n_heads, qk_head_dim, v_head_dim, device="cuda", seed=0
):
    call = _mla_prefill_varlen_call(n_q, n_kv, n_offsets, n_heads, qk_head_dim, v_head_dim, 1, device, seed)
    return {**call, "causal": False}


def _mla_prefill_varlen(lse: bool, variant: str, when: str, constraints: tuple[str, ...], init) -> Schema:
    has_prefix = "prefix" in variant
    reads = (
        "With causal, aligned at the end: query i of a sequence of Lq queries over Lk keys reads keys "
        "0 .. Lk - Lq + i; without, every key of its sequence (a prefix chunk, each of whose keys precedes "
        "every query)."
        if lse
        else "Causal, aligned at the end: query i of a sequence of Lq queries over Lk keys reads keys "
        "0 .. Lk - Lq + i."
    )
    return Schema(
        op_type="mla_prefill_varlen_lse" if lse else "mla_prefill_varlen",
        # a call without sm_scale is the backend's
        variant={variant: f"sm_scale is not None and {when}"},
        description=(
            "MLA's prefill attention once kv_b has expanded the latent keys, every head on its own: sequence "
            "s's queries are q's rows cu_seqlens_q[s]:cu_seqlens_q[s + 1], its keys and values the rows "
            "cu_seqlens_k[s]:cu_seqlens_k[s + 1] of k and v (with no prefix, the queries' own rows). "
            f"{reads} Logits q . k * sm_scale over qk_head_dim; output the softmax-weighted sum of v"
            + (
                ", lse the natural log of the sum of exp(logit) over the keys read, per query row and head"
                if lse
                else ""
            )
            + ". With qkv_fp8 a kernel may round q, k and v to float8_e4m3fn first, as aiter's fp8 prefill does."
        ),
        axes={
            "n_q": Var("Query rows, every sequence's back to back."),
            "n_kv": Var("Key and value rows, every sequence's back to back."),
            # a kernel serves the sequence counts its workloads measure, no other
            "n_offsets": Var("Sequence offsets: sequences + 1."),
            "n_heads": Const("Heads on this rank; k and v carry q's."),
            "qk_head_dim": Const("Values per q and k head: the no-rope and rope parts."),
            "v_head_dim": Const("Values per v and output head."),
        },
        constraints=[
            "n_offsets - 1 <= n_q",
            "cu_seqlens_k is not None" if has_prefix else "cu_seqlens_k is None and n_kv == n_q",
            *constraints,
        ],
        inputs={
            "q": Tensor(["n_q", "n_heads", "qk_head_dim"], dtype="bfloat16", description="Queries."),
            "k": Tensor(["n_kv", "n_heads", "qk_head_dim"], dtype="bfloat16", description="Keys."),
            "v": Tensor(["n_kv", "n_heads", "v_head_dim"], dtype="bfloat16", description="Values."),
            "cu_seqlens_q": Tensor(
                ["n_offsets"],
                dtype="int32",
                description="Where each sequence's queries start in q, then where the last ends.",
            ),
            "qkv_fp8": Scalar(
                "bool",
                optional=True,
                description="Whether a kernel may round q, k and v to float8_e4m3fn, scale 1, before it attends.",
            ),
            **(
                {
                    "causal": Scalar(
                        "bool",
                        optional=True,
                        description=(
                            "Whether the queries attend causally, aligned at the end of their keys, else to "
                            "every key."
                        ),
                    )
                }
                if lse
                else {}
            ),
            "cu_seqlens_k": Tensor(
                ["n_offsets"],
                dtype="int32",
                optional=not has_prefix,
                description="Where each sequence's keys start in k and v, then where the last ends.",
            ),
            "sm_scale": Scalar("float32", description="Multiplies every logit."),
        },
        outputs={
            "output": Tensor(
                ["n_q", "n_heads", "v_head_dim"],
                dtype="bfloat16",
                description="The softmax-weighted sum of the values each query reads.",
            ),
            **(
                {
                    "lse": Tensor(
                        ["n_q", "n_heads"],
                        dtype="float32",
                        description=(
                            "Natural log-sum-exp of the scaled logits each query reads, [query row, head]."
                        ),
                    )
                }
                if lse
                else {}
            ),
        },
        standard=Standard(
            reference=mla_prefill_varlen_lse_reference if lse else mla_prefill_varlen_reference,
            check=_mla_prefill_varlen_check,
            init=init,
            baseline=mla_prefill_varlen_lse_baseline if lse else mla_prefill_varlen_baseline,
        ),
    )


# each op's variants, in resolution order: its api takes the first whose condition matches the call
MLA_PREFILL_VARLEN_VARIANTS = (
    _mla_prefill_varlen(False, "default", "cu_seqlens_k is None", (), _mla_prefill_varlen_init),
    _mla_prefill_varlen(
        False,
        "prefix",
        "cu_seqlens_k is not None",
        # each sequence's keys are at least all the queries: none of its queries reads no key
        ("n_kv >= (n_offsets - 1) * n_q",),
        _mla_prefill_varlen_prefix_init,
    ),
)
MLA_PREFILL_VARLEN_LSE_VARIANTS = (
    _mla_prefill_varlen(True, "default", "cu_seqlens_k is None and causal", (), _mla_prefill_varlen_lse_init),
    _mla_prefill_varlen(
        True, "no_causal", "cu_seqlens_k is None and not causal", (), _mla_prefill_varlen_lse_no_causal_init
    ),
    _mla_prefill_varlen(
        True,
        "prefix",
        "cu_seqlens_k is not None and causal",
        # one sequence, whose keys are at least its queries, so none of them reads no key (the keys' split over
        # several sequences need not leave each its own queries' worth)
        ("n_offsets == 2 and n_kv >= n_q",),
        _mla_prefill_varlen_lse_prefix_init,
    ),
    _mla_prefill_varlen(
        True,
        "prefix_no_causal",
        "cu_seqlens_k is not None and not causal",
        ("n_kv >= n_offsets - 1",),
        _mla_prefill_varlen_lse_prefix_no_causal_init,
    ),
)
