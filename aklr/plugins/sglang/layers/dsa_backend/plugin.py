# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    DTYPE_TRAITS,
    CausalTopkDistribution,
    CuSeqlensDistribution,
    DeterminismLevel,
    DType,
    ElementOrder,
    FiniteDistribution,
    IntRangeDistribution,
    PermutationDistribution,
    SqnrBound,
)
from aklr.core.contract.spec import (
    CompareSpec,
    DTypeSpec,
    LayoutVariant,
    LayoutVariantsSpec,
    ScalarSpec,
    TensorSpec,
    axis,
    derived_axis,
)
from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins import Plugin

__all__ = [
    "DSA_SPARSE_MLA_DECODE",
    "DSA_SPARSE_MLA_PREFILL",
    "DSA_TOPK_TRANSFORM",
    "ROPE_DIM",
    "declare",
    "dsa_sparse_mla_decode",
    "dsa_sparse_mla_prefill",
    "dsa_topk_transform",
    "family",
]


_FP8 = DTypeSpec(any_of=(DType.FLOAT8_E4M3FN,))
_INT32 = DTypeSpec(any_of=(DType.INT32,))
_FINITE = FiniteDistribution()
_POSITIVE_FLOAT32 = (2.0**-149, DTYPE_TRAITS["float32"].largest_finite_value)
# The KV pool's bytes that 32-bit buffer offsets address: rows past them stay on sglang.
_POOL_BYTES = 2**31 - 2
# The DSA models' context, 2^20 tokens, and room for the draft tokens sglang's tables add past it
_CONTEXT_LIMIT = 2**20 + 4096
# The RoPE part of a query head and of a KV row: the DSA models' (sglang's kernel takes a
# power of 2)
ROPE_DIM = 64
# GLM-5's softmax scale, 1 / sqrt(nope_dim + rope_dim): the default bench's inputs take
_SM_SCALE = 0.0625

DSA_SPARSE_MLA_DECODE = OpContract(
    contract_id="dsa_sparse_mla_decode@1.0",
    description=(
        "sglang's DSA backend's sparse MLA decode on one rank, kv_b_proj absorbed "
        "(DeepseekSparseAttnBackend._forward_triton_decode): each head of each token attends "
        "the KV pool rows its top-k slots name (-1: none) with scores q . row * sm_scale, q "
        "and the rows the fp8 values sglang holds, unscaled; the output is the "
        "softmax-weighted sum of the rows' latent parts, 0 for a token that reads no row."
    ),
    op_type="dsa_backend",
    axes={
        "n_tokens": axis("Tokens in this call, one query each.", 1, 65536, is_var=True),
        "n_heads": axis("Query heads on this rank.", 1, 128),
        "kv_dim": axis("A query head and a KV row: the latent, then 64 RoPE values.", 192, 576),
        "topk": axis("Top-k entries per token.", 1, 65536),
        "n_slots": axis("Rows of the KV pool.", 1, 2**31 - 1, is_var=True, any_size=True),
        "kv_lora_rank": derived_axis(
            "The latent's width: a row's first part, and the output's.",
            f"kv_dim - {ROPE_DIM}",
        ),
    },
    constraints=(
        "kv_lora_rank % 128 == 0",
        # distinct top-k entries, -1 among them
        "topk <= n_slots + 1",
        f"n_slots * kv_dim <= {_POOL_BYTES}",
    ),
    inputs={
        "q": TensorSpec(
            description=(
                "Each token's query per head: q_nope times W_UK (the latent's basis), then the RoPE'd q_pe."
            ),
            shape=("n_tokens", "n_heads", "kv_dim"),
            dtype=_FP8,
            distribution=_FINITE,
        ),
        "kv_cache": TensorSpec(
            description=("This layer's KV pool: a row per slot, the latent then the RoPE part."),
            shape=("n_slots", "kv_dim"),
            dtype=_FP8,
            distribution=_FINITE,
        ),
        "topk_indices": TensorSpec(
            description="The slots each token attends; -1: none.",
            shape=("n_tokens", "topk"),
            dtype=DTypeSpec(any_of=(DType.INT32,)),
            distribution=IntRangeDistribution(lo=-1, hi="n_slots - 1", distinct=True),
        ),
        "sm_scale": ScalarSpec(
            description="Multiplies every score.",
            dtype=DType.FLOAT32,
            interval=_POSITIVE_FLOAT32,
            default=_SM_SCALE,
        ),
    },
    outputs={
        "output": TensorSpec(
            description=(
                "Each head's output in the latent's basis: the rows' latent parts "
                "weighed by the softmax of their scores."
            ),
            shape=("n_tokens", "n_heads", "kv_lora_rank"),
            dtype=DTypeSpec(any_of=(DType.BFLOAT16,)),
            # sglang's own kernel, which rounds the softmax weights to fp8, comes to about
            # 33 dB on these inputs
            error_bounds=SqnrBound(min_sqnr_db=20.0),
        ),
    },
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
)

DSA_SPARSE_MLA_PREFILL = OpContract(
    contract_id="dsa_sparse_mla_prefill@1.0",
    description=(
        "sglang's DSA backend's sparse MLA prefill on one rank, kv_b_proj absorbed "
        "(DeepseekSparseAttnBackend.forward_extend's triton_sparse_mla_fwd): each head of each "
        "token attends the KV pool rows its top-k slots name (-1: none) with scores q . row * "
        "sm_scale, q and the rows the fp8 values sglang holds, unscaled; the output is the "
        "softmax-weighted sum of the rows' latent parts, 0 for a token that reads no row. Benched "
        "as a new sequence's tokens: token i names min(i + 1, topk) slots, then -1."
    ),
    op_type="dsa_backend",
    axes=DSA_SPARSE_MLA_DECODE.axes,
    constraints=DSA_SPARSE_MLA_DECODE.constraints,
    inputs={
        **DSA_SPARSE_MLA_DECODE.inputs,
        "topk_indices": TensorSpec(
            description=(
                "The slots each token attends; -1: none. Drawn as sglang's indexer picks them for "
                "a new sequence: a token's positions while it has at most topk, else topk of them."
            ),
            shape=("n_tokens", "topk"),
            dtype=DTypeSpec(any_of=(DType.INT32,)),
            distribution=CausalTopkDistribution(hi="n_slots - 1"),
        ),
    },
    outputs=DSA_SPARSE_MLA_DECODE.outputs,
    layout_variants=DSA_SPARSE_MLA_DECODE.layout_variants,
)

DSA_TOPK_TRANSFORM = OpContract(
    contract_id="dsa_topk_transform@1.0",
    description=(
        "sglang's DSA indexer top-k on HIP (sgl_kernel.fast_topk_transform_fused): each row "
        "selects the topk highest of its first lengths scores and writes their slots in the "
        "page_size=1 page table of its sequence, the cu_seqlens_q range holding the row; a row "
        "of at most topk scores writes all of its slots in order, then -1."
    ),
    op_type="dsa_backend",
    axes={
        "n_tokens": axis("Rows, one per token.", 1, 65536, is_var=True),
        "n_candidates": axis("Scores per row.", 1, _CONTEXT_LIMIT, is_var=True, any_size=True),
        "n_seqs": axis("Sequences.", 1, 65536, is_var=True, any_size=True),
        "n_starts": axis("Sequence boundaries: n_seqs + 1.", 2, 65537, is_var=True, any_size=True),
        "n_slots": axis("Page table columns.", 1, _CONTEXT_LIMIT, is_var=True, any_size=True),
        "topk": axis("Slots per row, 2048: all sglang's kernel takes.", 2048, 2048),
    },
    constraints=("n_starts == n_seqs + 1", "n_seqs <= n_tokens"),
    inputs={
        "score": TensorSpec(
            description="Candidate scores per row.",
            shape=("n_tokens", "n_candidates"),
            dtype=DTypeSpec(any_of=(DType.FLOAT32,)),
            distribution=PermutationDistribution(),
        ),
        "lengths": TensorSpec(
            description="Valid candidates per row, a prefix its page table covers.",
            shape=("n_tokens",),
            dtype=_INT32,
            distribution=IntRangeDistribution(lo="0", hi="min(n_candidates, n_slots)"),
        ),
        "page_table_size_1": TensorSpec(
            description="Each sequence's slot of each candidate position, distinct within a row.",
            shape=("n_seqs", "n_slots"),
            dtype=_INT32,
            distribution=IntRangeDistribution(lo="0", hi="n_slots - 1", distinct=True),
        ),
        "cu_seqlens_q": TensorSpec(
            description="Each sequence's first row, then n_tokens.",
            shape=("n_starts",),
            dtype=_INT32,
            distribution=CuSeqlensDistribution(min_seqlen=1, total="n_tokens"),
        ),
    },
    outputs={
        "out_page_indices": TensorSpec(
            description="Selected slots; -1 after a short row's entries.",
            shape=("n_tokens", "topk"),
            dtype=_INT32,
        )
    },
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
    # Ties and a full row's order are the kernel's choice.
    compare=CompareSpec(
        determinism=DeterminismLevel.BOUNDED,
        element_order=ElementOrder.ANY_IN_FULL_ROWS,
    ),
)


family = Plugin("sglang")
family.define(DSA_SPARSE_MLA_DECODE)
family.define(DSA_SPARSE_MLA_PREFILL)
family.define(DSA_TOPK_TRANSFORM)
family.build()


# sglang's short-sequence path in triton_sparse_mla_fwd
_SGLANG_SHORT_TOPK = 512


@family.op
def dsa_sparse_mla_decode(q, kv_cache, topk_indices, sm_scale): ...


@family.op
def dsa_sparse_mla_prefill(q, kv_cache, topk_indices, sm_scale): ...


@family.op
def dsa_topk_transform(score, lengths, page_table_size_1, cu_seqlens_q, out_page_indices): ...


# _forward_triton_decode, not forward_decode: sglang still picks the DSA decode
# implementation (triton, HIP's default), and this is the one the baseline measures.
def _forward_triton_decode(self, q_nope, q_rope, kv_cache, v_head_dim, page_table_1, sm_scale):
    """The decode's attention as one op call on the whole q, returned as sglang returns it,
    [tokens, heads, v_head_dim]; v_head_dim is q_nope's width, as sglang's kernel asserts."""
    from sglang.srt.runtime_context import get_spec

    q = _whole_query(q_nope, q_rope)
    # speculative decoding's draft steps stay on sglang
    if q is None or get_spec().speculative_algorithm is not None:
        return FallbackReason.OUT_OF_CONTRACT
    return dsa_sparse_mla_decode(
        q=q,
        kv_cache=kv_cache.view(-1, kv_cache.shape[-1]),
        topk_indices=page_table_1,
        sm_scale=sm_scale,
    )


def _with_target_verify(orig, *args, **kwargs):
    from aklr.plugins.aiter._verify import target_verify

    with target_verify(_verify_hint(*args, **kwargs)):
        return orig(*args, **kwargs)


def _verify_hint(self, forward_batch, layer_id, q_fp8, *args, **kwargs):
    rows, draft_tokens = q_fp8.shape[0], 0
    if forward_batch.forward_mode.is_target_verify():
        try:
            from sglang.srt.runtime_context import get_parallel, get_spec
        except ImportError:
            get_parallel = None
        if get_parallel is not None and get_parallel().attn_dcp_size > 1:
            draft_tokens = get_spec().speculative_num_draft_tokens or 0
    return (rows, draft_tokens) if draft_tokens else None


# HIP's DSA top-k: sglang turns the v2 paged top-k off there for DSA models
def _fast_topk_transform_fused(score, lengths, page_table_size_1, cu_seqlens_q, topk, row_starts=None):
    """The DSA indexer's paged top-k as one op call, into the [rows, topk] int32 tensor
    sgl_kernel allocates; a ragged row_starts stays on sgl_kernel."""
    import torch

    if row_starts is not None:
        return FallbackReason.OUT_OF_CONTRACT
    return dsa_topk_transform(
        score=score,
        lengths=lengths,
        page_table_size_1=page_table_size_1,
        cu_seqlens_q=cu_seqlens_q,
        out_page_indices=score.new_empty((score.shape[0], topk), dtype=torch.int32),
    )


# HIP's DSA prefill; forward_extend imports it at each call.
def _triton_sparse_mla_fwd(
    q_nope, q_rope, kv, indices, sm_scale, d_v=512, topk_length=None, max_topk_length=None
):
    """New sequences' prefill as one op call; tokens after a cached prefix stay on sglang."""
    import torch
    from sglang.srt.runtime_context import get_spec

    q = _whole_query(q_nope, q_rope)
    if (
        q is None
        or d_v != q_nope.shape[-1]
        or get_spec().speculative_algorithm is not None
        or topk_length is not None
        or (max_topk_length is not None and max_topk_length <= _SGLANG_SHORT_TOPK)
        or torch.cuda.is_current_stream_capturing()
        or not _is_new_sequences(q.shape[0])
    ):
        return FallbackReason.OUT_OF_CONTRACT
    output = dsa_sparse_mla_prefill(
        q=q,
        kv_cache=kv.view(-1, kv.shape[-1]),
        topk_indices=indices.squeeze(1),
        sm_scale=sm_scale,
    )
    if isinstance(output, FallbackReason):
        return output
    return output.unsqueeze(0)


def _is_new_sequences(tokens):
    """Whether this forward's sequences have no cached prefix (host lengths, no device sync)."""
    from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
    from sglang.srt.model_executor.forward_context import get_attn_backend, has_forward_context

    if not has_forward_context():
        return False
    backend = get_attn_backend()
    if not isinstance(backend, DeepseekSparseAttnBackend):
        return False
    metadata = backend.forward_metadata
    return metadata.seq_lens_sum == sum(metadata.dsa_extend_seq_lens_list) == tokens


def _whole_query(q_nope, q_rope):
    """The [tokens, heads, latent + rope] q whose two views forward_decode passes; None for two
    tensors apart."""
    tokens, heads, latent = q_nope.shape
    width = latent + q_rope.shape[-1]
    if not (
        q_rope.dtype == q_nope.dtype
        and q_rope.shape[:2] == q_nope.shape[:2]
        and q_nope.stride() == q_rope.stride() == (heads * width, width, 1)
        and q_rope.untyped_storage().data_ptr() == q_nope.untyped_storage().data_ptr()
        and q_rope.storage_offset() == q_nope.storage_offset() + latent
    ):
        return None
    return q_nope.as_strided((tokens, heads, width), q_nope.stride())


def declare(plugin: Plugin) -> None:
    plugin.wrap(
        "sglang.srt.layers.attention.dsa_backend.DeepseekSparseAttnBackend._forward_triton_decode",
        versions=("0.5.23",),
        replace=_forward_triton_decode,
    )
    plugin.wrap(
        "sglang.srt.layers.attention.dsa.dsa_indexer.Indexer._get_topk_paged",
        versions=("0.5.24.dev809",),
        around=_with_target_verify,
    )
    plugin.wrap(
        "sgl_kernel.top_k.fast_topk_transform_fused",
        versions=("0.5.22", "0.5.23"),
        replace=_fast_topk_transform_fused,
    )
    plugin.wrap(
        "sglang.kernels.ops.attention.dsa.triton_sparse_mla.triton_sparse_mla_fwd",
        versions=("0.5.22", "0.5.23"),
        replace=_triton_sparse_mla_fwd,
    )
