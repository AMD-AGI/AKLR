# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Split-precision MXFP8/BF16 CSR attention over prefix and extend pools."""

from __future__ import annotations

import torch

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    DTYPE_TRAITS,
    CuSeqlensDistribution,
    DeterminismLevel,
    DType,
    FiniteDistribution,
    IntRangeDistribution,
    Mxfp8NopeRowsDistribution,
    SqnrBound,
    ToleranceBound,
)
from aklr.core.contract.spec import (
    CompareSpec,
    DTypeSpec,
    LayoutVariant,
    LayoutVariantsSpec,
    ScalarSpec,
    TensorSpec,
    attr,
    axis,
)
from aklr.plugins import Plugin
from aklr.plugins.aiter._candidate import offer, served

__all__ = [
    "MLA_PREFILL_VARLEN",
    "MLA_PREFILL_VARLEN_LSE",
    "SPARSE_PREFILL_MXFP8",
    "declare",
    "family",
    "mla_prefill_varlen",
    "mla_prefill_varlen_lse",
    "sparse_prefill_mxfp8",
]


def _tensor(description, shape, dtype, distribution, *, lifetime="per_call"):
    return TensorSpec(
        description=description,
        shape=shape,
        dtype=DTypeSpec(any_of=(dtype,)),
        distribution=distribution,
        lifetime=lifetime,
    )


def _pool(prefix):
    return {
        f"{prefix}_nope": _tensor(
            "Packed NoPE rows: 448 FP8 codes, 14 E8M0 scales, padding to 512 bytes.",
            (f"{prefix}_rows", 512),
            DType.UINT8,
            Mxfp8NopeRowsDistribution(),
            lifetime="weight",
        ),
        f"{prefix}_rope": _tensor(
            "RoPE keys and values corresponding to the NoPE rows.",
            (f"{prefix}_rows", 64),
            DType.BFLOAT16,
            FiniteDistribution(),
            lifetime="weight",
        ),
        f"{prefix}_indices": _tensor(
            "CSR entries selecting rows of the pool.",
            (f"{prefix}_nnz",),
            DType.INT32,
            IntRangeDistribution(lo=0, hi=f"{prefix}_rows - 1"),
        ),
        f"{prefix}_indptr": _tensor(
            "Monotone CSR offsets for each query, beginning with zero.",
            ("n_offsets",),
            DType.INT32,
            CuSeqlensDistribution(min_seqlen=0, total=f"{prefix}_nnz"),
        ),
    }


SPARSE_PREFILL_MXFP8 = OpContract(
    contract_id="sparse_prefill_mxfp8@1.0",
    description=(
        "Sparse attention over two CSR ranges. NoPE uses block-scaled E4M3FN and "
        "RoPE BF16; each head's sink contributes only to the softmax denominator."
    ),
    op_type="mla_prefill",
    axes={
        "n_tokens": axis("Query rows.", 1, 16384, is_var=True),
        "n_heads": axis("Query heads.", 16, 128, multiple_of=16),
        "prefix_rows": axis("Rows of the prefix pool.", 1, 8388608, is_var=True),
        "extend_rows": axis("Rows of the extend pool.", 1, 131072, is_var=True),
        "prefix_nnz": axis("Prefix CSR entries.", 0, 268435456, is_var=True),
        "extend_nnz": axis("Extend CSR entries.", 0, 268435456, is_var=True),
        "n_offsets": axis("CSR offsets, queries plus one.", 2, 16385, is_var=True),
    },
    constraints=("n_offsets == n_tokens + 1",),
    inputs={
        "q_nope": _tensor(
            "Packed NoPE query rows.",
            ("n_tokens", "n_heads", 512),
            DType.UINT8,
            Mxfp8NopeRowsDistribution(),
        ),
        "q_rope": _tensor(
            "Query RoPE components.",
            ("n_tokens", "n_heads", 64),
            DType.BFLOAT16,
            FiniteDistribution(),
        ),
        **_pool("prefix"),
        **_pool("extend"),
        "sinks": _tensor("Per-head softmax sink logits.", ("n_heads",), DType.FLOAT32, FiniteDistribution()),
        "softmax_scale": ScalarSpec(
            description="Multiplier of the combined QK score.",
            dtype=DType.FLOAT32,
            interval=(1e-10, 1.0),
            default=512**-0.5,
        ),
    },
    outputs={
        "output": TensorSpec(
            description="Weighted sum of 448 dequantized NoPE and 64 RoPE values.",
            shape=("n_tokens", "n_heads", 512),
            dtype=DTypeSpec(any_of=(DType.BFLOAT16,)),
            error_bounds=ToleranceBound(absolute_tolerance=1e-2, relative_tolerance=1e-2),
        ),
    },
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
)

_ROWS_LIMIT = 2**20  # one prefill's rows: sglang attends at most 131072 keys at once
_BF16_OR_SAME = DTypeSpec(same_as_tensor="q")

MLA_PREFILL_VARLEN = OpContract(
    contract_id="mla_prefill_varlen@1.0",
    description=(
        "MLA's prefill attention once kv_b has expanded the latent keys, every head on its own: sequence s's "
        "queries are q's rows cu_seqlens_q[s]:cu_seqlens_q[s + 1], its keys and values the rows "
        "cu_seqlens_k[s]:cu_seqlens_k[s + 1] of k and v (with no prefix, the queries' own rows). Causal, aligned "
        "at the end: query i of a sequence of Lq queries over Lk keys reads keys 0 .. Lk - Lq + i. Logits "
        "q . k * sm_scale over qk_head_dim; output the softmax-weighted sum of v. With qkv_fp8 a kernel may "
        "round q, k and v to float8_e4m3fn first, as aiter's fp8 prefill does."
    ),
    op_type="mla_prefill",
    axes={
        "n_q": axis("Query rows, every sequence's back to back.", 1, _ROWS_LIMIT, is_var=True, any_size=True),
        "n_kv": axis(
            "Key and value rows, every sequence's back to back.", 1, _ROWS_LIMIT, is_var=True, any_size=True
        ),
        # a kernel serves the sequence counts its workloads measure, no other
        "n_offsets": axis("Sequence offsets: sequences + 1.", 2, 4097, is_var=True),
        "n_heads": axis("Heads on this rank; k and v carry q's.", 1, 128),
        "qk_head_dim": axis("Values per q and k head: the no-rope and rope parts.", 64, 256, multiple_of=64),
        "v_head_dim": axis("Values per v and output head.", 64, 256, multiple_of=64),
    },
    attrs={
        "has_prefix": attr(
            "Whether cu_seqlens_k is passed: the sequences' keys start with a cached prefix.",
            False,
            True,
            default=False,
            abbrev="prefix",
        ),
        "qkv_fp8": attr(
            "Whether a kernel may round q, k and v to float8_e4m3fn, scale 1, before it attends.",
            True,
            False,
            default=True,
            param_name="qkv_fp8",
            abbrev="fp8",
        ),
    },
    constraints=(
        "n_offsets - 1 <= n_q",
        "has_prefix or n_kv == n_q",
        # each sequence's keys are at least all the queries: none of its queries reads no key
        "(not has_prefix) or n_kv >= (n_offsets - 1) * n_q",
    ),
    inputs={
        "q": TensorSpec(
            description="Queries.",
            shape=("n_q", "n_heads", "qk_head_dim"),
            dtype=DTypeSpec(any_of=(DType.BFLOAT16,)),
            distribution=FiniteDistribution(),
        ),
        "k": TensorSpec(
            description="Keys.",
            shape=("n_kv", "n_heads", "qk_head_dim"),
            dtype=_BF16_OR_SAME,
            distribution=FiniteDistribution(),
        ),
        "v": TensorSpec(
            description="Values.",
            shape=("n_kv", "n_heads", "v_head_dim"),
            dtype=_BF16_OR_SAME,
            distribution=FiniteDistribution(),
        ),
        "cu_seqlens_q": TensorSpec(
            description="Where each sequence's queries start in q, then where the last ends.",
            shape=("n_offsets",),
            dtype=DTypeSpec(any_of=(DType.INT32,)),
            distribution=CuSeqlensDistribution(min_seqlen=1, total="n_q"),
        ),
        "cu_seqlens_k": TensorSpec(
            description="Where each sequence's keys start in k and v, then where the last ends.",
            shape=("n_offsets",),
            dtype=DTypeSpec(any_of=(DType.INT32,)),
            present_when="has_prefix",
            distribution=CuSeqlensDistribution(min_seqlen="n_q", total="n_kv"),
        ),
        # always passed: present_when gives the api default None, which declines
        "sm_scale": ScalarSpec(
            description="Multiplies every logit.",
            dtype=DType.FLOAT32,
            interval=(2.0**-149, DTYPE_TRAITS["float32"].largest_finite_value),
            # MLA's, 1/sqrt(128 + 64): at 1, N(0, 1) logits over 192 values reach +-60, past aiter's fp8 prefill
            default=192**-0.5,
            present_when="True",
        ),
    },
    outputs={
        "output": TensorSpec(
            description="The softmax-weighted sum of the values each query reads.",
            shape=("n_q", "n_heads", "v_head_dim"),
            dtype=_BF16_OR_SAME,
            # with qkv_fp8 the reference attends over q, k and v rounded to fp8 too, as the kernels may: whatever
            # finite values they hold, the bound is on the attention itself
            error_bounds=(
                ("qkv_fp8", SqnrBound(min_sqnr_db=20.0)),
                (None, ToleranceBound(absolute_tolerance=1e-2, relative_tolerance=1e-2)),
            ),
        ),
    },
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
    compare=CompareSpec(determinism=DeterminismLevel.BOUNDED),
)

MLA_PREFILL_VARLEN_LSE = OpContract(
    contract_id="mla_prefill_varlen_lse@1.0",
    description=(
        "mla_prefill_varlen that also returns each query's log-sum-exp, the passes of a chunked prefix that "
        "sglang merges: sequence s's queries are q's rows cu_seqlens_q[s]:cu_seqlens_q[s + 1], its keys and "
        "values the rows cu_seqlens_k[s]:cu_seqlens_k[s + 1] of k and v (with no prefix, the queries' own rows). "
        "With causal, aligned at the end: query i of a sequence of Lq queries over Lk keys reads keys "
        "0 .. Lk - Lq + i; without, every key of its sequence (a prefix chunk, each of whose keys precedes every "
        "query). Logits q . k * sm_scale over "
        "qk_head_dim; output the softmax-weighted sum of v, lse the natural log of the sum of exp(logit) over the "
        "keys read, per query row and head. With qkv_fp8 a kernel may round q, k and v to float8_e4m3fn first, "
        "as aiter's fp8 prefill does."
    ),
    op_type="mla_prefill",
    axes={
        "n_q": axis("Query rows, every sequence's back to back.", 1, _ROWS_LIMIT, is_var=True, any_size=True),
        "n_kv": axis(
            "Key and value rows, every sequence's back to back.", 1, _ROWS_LIMIT, is_var=True, any_size=True
        ),
        # a kernel serves the sequence counts its workloads measure, no other
        "n_offsets": axis("Sequence offsets: sequences + 1.", 2, 4097, is_var=True),
        "n_heads": axis("Heads on this rank; k and v carry q's.", 1, 128),
        "qk_head_dim": axis("Values per q and k head: the no-rope and rope parts.", 64, 256, multiple_of=64),
        "v_head_dim": axis("Values per v and output head.", 64, 256, multiple_of=64),
    },
    attrs={
        "has_prefix": attr(
            "Whether cu_seqlens_k is passed: the sequences' keys are not the queries' rows (a cached prefix and, "
            "with causal, the queries' own keys after it).",
            False,
            True,
            default=False,
            abbrev="prefix",
        ),
        "qkv_fp8": attr(
            "Whether a kernel may round q, k and v to float8_e4m3fn, scale 1, before it attends.",
            True,
            False,
            default=True,
            param_name="qkv_fp8",
            abbrev="fp8",
        ),
        "causal": attr(
            "Whether the queries attend causally, aligned at the end of their keys, else to every key.",
            True,
            False,
            default=True,
            param_name="causal",
        ),
    },
    constraints=(
        "n_offsets - 1 <= n_q",
        "has_prefix or n_kv == n_q",
        # causal over a prefix: one sequence, whose keys are at least its queries, so none of them reads no key (the
        # keys' split over several sequences need not leave each its own queries' worth)
        "(not has_prefix) or (not causal) or (n_offsets == 2 and n_kv >= n_q)",
        "(not has_prefix) or n_kv >= n_offsets - 1",
    ),
    inputs={
        "q": TensorSpec(
            description="Queries.",
            shape=("n_q", "n_heads", "qk_head_dim"),
            dtype=DTypeSpec(any_of=(DType.BFLOAT16,)),
            distribution=FiniteDistribution(),
        ),
        "k": TensorSpec(
            description="Keys.",
            shape=("n_kv", "n_heads", "qk_head_dim"),
            dtype=_BF16_OR_SAME,
            distribution=FiniteDistribution(),
        ),
        "v": TensorSpec(
            description="Values.",
            shape=("n_kv", "n_heads", "v_head_dim"),
            dtype=_BF16_OR_SAME,
            distribution=FiniteDistribution(),
        ),
        "cu_seqlens_q": TensorSpec(
            description="Where each sequence's queries start in q, then where the last ends.",
            shape=("n_offsets",),
            dtype=DTypeSpec(any_of=(DType.INT32,)),
            distribution=CuSeqlensDistribution(min_seqlen=1, total="n_q"),
        ),
        "cu_seqlens_k": TensorSpec(
            description="Where each sequence's keys start in k and v, then where the last ends.",
            shape=("n_offsets",),
            dtype=DTypeSpec(any_of=(DType.INT32,)),
            present_when="has_prefix",
            distribution=CuSeqlensDistribution(min_seqlen=1, total="n_kv"),
        ),
        "sm_scale": ScalarSpec(
            description="Multiplies every logit.",
            dtype=DType.FLOAT32,
            interval=(2.0**-149, DTYPE_TRAITS["float32"].largest_finite_value),
            default=192**-0.5,
            present_when="True",
        ),
    },
    outputs={
        "output": TensorSpec(
            description="The softmax-weighted sum of the values each query reads.",
            shape=("n_q", "n_heads", "v_head_dim"),
            dtype=_BF16_OR_SAME,
            error_bounds=(
                ("qkv_fp8", SqnrBound(min_sqnr_db=20.0)),
                (None, ToleranceBound(absolute_tolerance=1e-2, relative_tolerance=1e-2)),
            ),
        ),
        "lse": TensorSpec(
            description="Natural log-sum-exp of the scaled logits each query reads, [query row, head].",
            shape=("n_q", "n_heads"),
            dtype=DTypeSpec(any_of=(DType.FLOAT32,)),
            # q . k on fp8-rounded operands moves an N(0, 1) logit by up to ~0.1, and a row reading few keys
            # carries that into its lse (aiter's fp8 prefill: up to 0.08 on 16384 causal rows)
            error_bounds=(
                ("qkv_fp8", ToleranceBound(absolute_tolerance=0.25, relative_tolerance=1e-3)),
                (None, ToleranceBound(absolute_tolerance=1e-3, relative_tolerance=1e-3)),
            ),
        ),
    },
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
    compare=CompareSpec(determinism=DeterminismLevel.BOUNDED),
)


family = Plugin("aiter")
family.define(MLA_PREFILL_VARLEN)
family.define(MLA_PREFILL_VARLEN_LSE)
family.define(SPARSE_PREFILL_MXFP8)
family.build()


@family.op
def sparse_prefill_mxfp8(
    q_nope,
    q_rope,
    prefix_nope,
    prefix_rope,
    prefix_indices,
    prefix_indptr,
    extend_nope,
    extend_rope,
    extend_indices,
    extend_indptr,
    sinks,
    *,
    softmax_scale=512**-0.5,
): ...


@family.op
def mla_prefill_varlen(
    q,
    k,
    v,
    cu_seqlens_q,
    qkv_fp8=True,
    cu_seqlens_k=None,
    sm_scale=None,
    output=None,
): ...


@family.op
def mla_prefill_varlen_lse(
    q,
    k,
    v,
    cu_seqlens_q,
    qkv_fp8=True,
    causal=True,
    cu_seqlens_k=None,
    sm_scale=None,
    output=None,
    lse=None,
): ...


# Fewer query rows stay on aiter: a decode step's extends (a few tokens over the whole context) reach
# flash_attn_varlen too, and AKLR's fp8 kernel, built for prefill chunks, casts the whole context and
# plans per shape on each call. Its workloads start at 257 rows.
_MIN_ROWS = 256


def _sparse_prefill_call(
    q_nope,
    q_rope,
    unified_kv_nope,
    unified_kv_rope,
    kv_indices_prefix,
    kv_indptr_prefix,
    kv_nope,
    kv_rope,
    kv_indices_extend,
    kv_indptr_extend,
    attn_sink,
    softmax_scale,
):
    """AKLR's call, as (args, kwargs); ``None`` outside it. AKLR reads the packed NoPE rows (448 e4m3fn
    codes, their E8M0 scales, padding) as bytes, and always takes a sink."""
    nope = (q_nope, unified_kv_nope, kv_nope)
    if attn_sink is None or any(t.dtype != torch.float8_e4m3fn for t in nope):
        return None
    q_nope, unified_kv_nope, kv_nope = (t.view(torch.uint8) for t in nope)
    return (
        q_nope,
        q_rope,
        unified_kv_nope,
        unified_kv_rope,
        kv_indices_prefix,
        kv_indptr_prefix,
        kv_nope,
        kv_rope,
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
    ), {"softmax_scale": softmax_scale}


def _select_mla_sparse_prefill_v4(args, kwargs):
    # its second tensor is q_rope (mla_sparse_prefill_v41 takes q_scale there); num_kv_splits only
    # splits aiter's own kernel's work
    (
        q_nope,
        q_rope,
        unified_kv_nope,
        unified_kv_rope,
        kv_indices_prefix,
        kv_indptr_prefix,
        kv_nope,
        kv_rope,
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
        softmax_scale,
        out,
        _num_kv_splits,
    ) = args
    # AKLR's output is copied into out: one aiter takes, [T, H, 512] bf16, else aiter raises
    if out is not None and (out.shape != q_nope.shape or out.dtype != torch.bfloat16):
        return None
    call = _sparse_prefill_call(
        q_nope,
        q_rope,
        unified_kv_nope,
        unified_kv_rope,
        kv_indices_prefix,
        kv_indptr_prefix,
        kv_nope,
        kv_rope,
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
        softmax_scale,
    )
    if call is None:
        return None
    call_args, call_kwargs = call
    return offer(sparse_prefill_mxfp8.serving(*call_args, **call_kwargs), _run_mla_sparse_prefill_v4)


def _run_mla_sparse_prefill_v4(
    q_nope,
    q_rope,
    unified_kv_nope,
    unified_kv_rope,
    kv_indices_prefix,
    kv_indptr_prefix,
    kv_nope,
    kv_rope,
    kv_indices_extend,
    kv_indptr_extend,
    attn_sink,
    softmax_scale,
    out,
    num_kv_splits,
):
    call_args, call_kwargs = _sparse_prefill_call(
        q_nope,
        q_rope,
        unified_kv_nope,
        unified_kv_rope,
        kv_indices_prefix,
        kv_indptr_prefix,
        kv_nope,
        kv_rope,
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
        softmax_scale,
    )
    output = served(sparse_prefill_mxfp8(*call_args, **call_kwargs))
    # AKLR's api takes no output: aiter writes the caller's out when given and returns it either way
    if out is None:
        return output
    out.copy_(output)
    return out


def _aligned(offsets, copy):
    """``offsets``, or where its data does not start on the 16 bytes AKLR's dispatch requires, a copy
    (``copy``) or an aligned tensor of its shape to ask with, which enqueues nothing."""
    if offsets is None or offsets.data_ptr() % 16 == 0:
        return offsets
    return offsets.clone() if copy else torch.empty_like(offsets)


def _varlen_call(q, cu_seqlens_q, cu_seqlens_k, softmax_scale, causal, copy):
    """mla_prefill_varlen_lse's offsets and keywords for aiter's call, unaligned offsets copied when
    ``copy`` (to run) or stood in for (to ask). Both causal masks are aligned at the end of the keys;
    keys that are the queries' own rows are AKLR's default."""
    keys = None if cu_seqlens_k is cu_seqlens_q else _aligned(cu_seqlens_k, copy)
    scale = q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
    return _aligned(cu_seqlens_q, copy), {
        "qkv_fp8": True,
        "causal": bool(causal),
        "cu_seqlens_k": keys,
        "sm_scale": scale,
    }


def _select_flash_attn_varlen(args, kwargs):
    # min_seqlen_q and how_v3_bf16_cvt tune aiter's kernels, max_seqlen_* size its grid, and the forward
    # pass is deterministic either way
    (
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        _max_seqlen_q,
        _max_seqlen_k,
        _min_seqlen_q,
        dropout_p,
        softmax_scale,
        logits_soft_cap,
        causal,
        window_size,
        bias,
        alibi_slopes,
        _deterministic,
        return_lse,
        return_attn_probs,
        _how_v3_bf16_cvt,
        block_table,
        out,
        cu_seqlens_q_padded,
        cu_seqlens_k_padded,
        sink_ptr,
        q_descale,
        k_descale,
        v_descale,
    ) = args
    # AKLR takes the pass vLLM merges over cached context: (output, lse), bf16 rows, no mask but the
    # causal one, and an output of its own
    if not return_lse or return_attn_probs or dropout_p or logits_soft_cap:
        return None
    window = tuple(window_size)
    if window[:2] != (-1, -1) or any(window[2:]):
        return None
    extras = (bias, alibi_slopes, block_table, out, cu_seqlens_q_padded, cu_seqlens_k_padded, sink_ptr)
    if any(t is not None for t in (*extras, q_descale, k_descale, v_descale)):
        return None
    if any(t.dtype != torch.bfloat16 for t in (q, k, v)) or q.shape[0] < _MIN_ROWS:
        return None
    # AKLR's result has no autograd graph; aiter's own path keeps the backward
    if torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)):
        return None
    offsets, call = _varlen_call(q, cu_seqlens_q, cu_seqlens_k, softmax_scale, causal, copy=False)
    # v may be a strided view (vLLM's is a slice of kv_b's output): asked with a contiguous stand-in
    stand_in = v if v.is_contiguous() else torch.empty(v.shape, dtype=v.dtype, device=v.device)
    return offer(mla_prefill_varlen_lse.serving(q, k, stand_in, offsets, **call), _run_flash_attn_varlen)


def _run_flash_attn_varlen(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    min_seqlen_q,
    dropout_p,
    softmax_scale,
    logits_soft_cap,
    causal,
    window_size,
    bias,
    alibi_slopes,
    deterministic,
    return_lse,
    return_attn_probs,
    how_v3_bf16_cvt,
    block_table,
    out,
    cu_seqlens_q_padded,
    cu_seqlens_k_padded,
    sink_ptr,
    q_descale,
    k_descale,
    v_descale,
):
    offsets, call = _varlen_call(q, cu_seqlens_q, cu_seqlens_k, softmax_scale, causal, copy=True)
    output, lse = served(mla_prefill_varlen_lse(q, k, v.contiguous(), offsets, **call))
    # aiter's lse is [heads, tokens], AKLR's [tokens, heads]
    return output, lse.t().contiguous()


def declare(plugin: Plugin) -> None:
    plugin.provide("mla_sparse_prefill_v4", version=1, select=_select_mla_sparse_prefill_v4)
    plugin.provide("flash_attn_varlen", version=1, select=_select_flash_attn_varlen)
