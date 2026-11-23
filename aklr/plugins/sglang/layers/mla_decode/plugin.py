# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""sglang's TritonAttnBackend.forward_decode on one rank: Kimi-K3's absorbed MLA decode."""

from __future__ import annotations

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    DTYPE_TRAITS,
    CuSeqlensDistribution,
    DeterminismLevel,
    DType,
    FiniteDistribution,
    IntRangeDistribution,
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
)
from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins import Plugin

__all__ = ["KIMI_K3_MLA_DECODE", "declare", "family", "kimi_k3_mla_decode"]


_BF16 = DTypeSpec(any_of=(DType.BFLOAT16,))
_POSITIVE_FLOAT32 = (2.0**-149, DTYPE_TRAITS["float32"].largest_finite_value)

KIMI_K3_MLA_DECODE = OpContract(
    contract_id="kimi_k3_mla_decode@1.0",
    description=(
        "sglang's TritonAttnBackend.forward_decode for Kimi-K3's MLA layers past its KV write, "
        "one query per sequence: each query head attends the slots "
        "kv_indices[kv_indptr[r]:kv_indptr[r + 1]] of the latent cache (page size 1, no RoPE), "
        "scores q . row * sm_scale over all 576 values, and output is the softmax-weighted sum "
        "of the rows' first 512 values"
    ),
    op_type="mla_decode",
    axes={
        "n_seqs": axis("Decode rows: one query each, padding rows included.", 1, 2048, is_var=True),
        "n_heads": axis("Query heads on this rank.", 1, 128),
        "n_slots": axis("Rows of the latent cache.", 1, 2**31 - 1, is_var=True, any_size=True),
        "n_offsets": axis("Offsets in kv_indptr: n_seqs + 1.", 2, 2049, is_var=True),
        "n_kv_tokens": axis("Entries of kv_indices.", 1, 2**31 - 1, is_var=True, any_size=True),
    },
    constraints=("n_offsets == n_seqs + 1", "n_kv_tokens >= n_seqs"),
    inputs={
        "q": TensorSpec(
            description="Each row's absorbed query per head: 512 latent values, then 64 rope ones.",
            shape=("n_seqs", "n_heads", 576),
            dtype=_BF16,
            distribution=FiniteDistribution(),
        ),
        "kv_cache": TensorSpec(
            description="This layer's latent cache: 512 latent values then 64 rope ones a slot.",
            shape=("n_slots", 1, 576),
            dtype=DTypeSpec(any_of=(DType.FLOAT8_E4M3FN,)),
            distribution=FiniteDistribution(),
            lifetime="weight",
        ),
        "kv_indptr": TensorSpec(
            description="Where each row's slots start in kv_indices, then where the last row's end.",
            shape=("n_offsets",),
            dtype=DTypeSpec(any_of=(DType.INT32,)),
            distribution=CuSeqlensDistribution(min_seqlen=1, total="n_kv_tokens"),
        ),
        "kv_indices": TensorSpec(
            description="The cache slots each row attends, row after row.",
            shape=("n_kv_tokens",),
            dtype=DTypeSpec(any_of=(DType.INT64,)),
            distribution=IntRangeDistribution(lo=0, hi="n_slots - 1"),
        ),
        "sm_scale": ScalarSpec(
            description="Multiplies every score: Kimi-K3's 192**-0.5.",
            dtype=DType.FLOAT32,
            interval=_POSITIVE_FLOAT32,
            default=192**-0.5,
        ),
    },
    outputs={
        "output": TensorSpec(
            description="Each head's softmax-weighted sum of the slots' first 512 values.",
            shape=("n_seqs", "n_heads", 512),
            dtype=DTypeSpec(same_as_tensor="q"),
            error_bounds=SqnrBound(min_sqnr_db=30.0),
        )
    },
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
    compare=CompareSpec(determinism=DeterminismLevel.BOUNDED),
)


family = Plugin("sglang")
family.define(KIMI_K3_MLA_DECODE)
family.build()


@family.op
def kimi_k3_mla_decode(q, kv_cache, kv_indptr, kv_indices, sm_scale=192**-0.5, output=None): ...


def _forward_decode(
    self, q, k, v, layer, forward_batch, save_kv_cache=True, sinks=None, score_mod=None, aux_tensors=None
):
    """MLA decode with a page-size-1 latent cache and no extras as one op, after sglang's own KV
    write; every other decode runs sglang's forward_decode."""
    if (
        not self.use_mla
        or self.dcp_size > 1
        or self.page_size != 1
        or self.enable_deterministic
        or sinks is not None
        or score_mod is not None
        or aux_tensors is not None
        or layer.k_scale is not None
        or layer.tp_k_head_num != 1
        or layer.logit_cap
        or (layer.sliding_window_size is not None and layer.sliding_window_size > -1)
        or (layer.xai_temperature_len or -1) > 0
    ):
        return FallbackReason.OUT_OF_CONTRACT
    metadata = self.forward_metadata
    arguments = {
        "q": q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim),
        "kv_cache": self.token_to_kv_pool.get_key_buffer(layer.layer_id),
        "kv_indptr": metadata.kv_indptr,
        "kv_indices": metadata.kv_indices,
        "sm_scale": layer.scaling,
    }
    reason = kimi_k3_mla_decode.fallback_reason(**arguments)
    if reason is not None:
        return reason
    if save_kv_cache:
        from sglang.srt.layers.attention.triton_backend import KVWriteLoc

        loc = KVWriteLoc.for_batch(forward_batch, swa_loc=metadata.swa_out_cache_loc)
        self.token_to_kv_pool.set_kv_buffer(layer, loc, k, v)
    output = kimi_k3_mla_decode(**arguments)
    if isinstance(output, FallbackReason):
        raise RuntimeError(f"kimi_k3_mla_decode declined after its KV write: {output}")
    return output.view(-1, layer.tp_q_head_num * layer.v_head_dim)


def declare(plugin: Plugin) -> None:
    plugin.wrap(
        "sglang.srt.layers.attention.triton_backend.TritonAttnBackend.forward_decode",
        versions=("0.5.22", "0.5.23"),
        replace=_forward_decode,
    )
