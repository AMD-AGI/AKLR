# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The DSA indexer's prefill logits on HIP (aiter's fp8_mqa_logits, as sglang's
Indexer._get_topk_ragged calls it): every query row against the batch's contiguous fp8 keys,
within the row's window. The top-k that reads them stays on sglang."""

from __future__ import annotations

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    DType,
    FiniteDistribution,
    IntRangeDistribution,
    LogUniformDistribution,
    SqnrBound,
)
from aklr.core.contract.spec import (
    CompareSpec,
    DTypeSpec,
    LayoutVariant,
    LayoutVariantsSpec,
    TensorSpec,
    axis,
)

__all__ = ["CONTRACTS", "FP8_MQA_LOGITS", "HEAD_DIM", "ROW_ALIGN"]

HEAD_DIM = 128
# aiter's logits rows: the keys rounded up to a multiple of 256 columns
ROW_ALIGN = 256

_FP8 = DTypeSpec(any_of=(DType.FLOAT8_E4M3FN,))
_FLOAT32 = DTypeSpec(any_of=(DType.FLOAT32,))
_INT32 = DTypeSpec(any_of=(DType.INT32,))
_FINITE = FiniteDistribution()
# The var axes' bounds: query rows of one prefill and keys of its requests
_MAX_ROWS = 2**20
_MAX_KEYS = 2**24

FP8_MQA_LOGITS = OpContract(
    contract_id="fp8_mqa_logits@1.0",
    description=(
        "The DSA indexer's prefill logits: for query row r and key c in [cu_starts[r], "
        "cu_ends[r]), logits[r, c] = kv_scales[c] * sum over heads h of weights[r, h] * "
        "relu(q[r, h] . kv[c]), written in place; logits outside a row's window keep their "
        "values."
    ),
    op_type="mqa_logits",
    axes={
        "n_q": axis("Query rows, every request's back to back.", 1, _MAX_ROWS, is_var=True, any_size=True),
        "n_k": axis("Keys, every request's back to back.", 1, _MAX_KEYS, is_var=True, any_size=True),
        "n_cols": axis(
            "Logits per row: the keys rounded up as aiter allocates them.",
            ROW_ALIGN,
            _MAX_KEYS,
            multiple_of=ROW_ALIGN,
            is_var=True,
            any_size=True,
        ),
        "n_heads": axis("Indexer heads.", 1, 128),
    },
    # a prefill's keys include its rows' own
    constraints=("n_q <= n_k", "n_cols == ceil_div(n_k, 256) * 256"),
    inputs={
        "q": TensorSpec(
            description="Each row's query heads.",
            shape=("n_q", "n_heads", HEAD_DIM),
            dtype=_FP8,
            distribution=_FINITE,
        ),
        "kv": TensorSpec(
            description="The keys, one fp8 row each.",
            shape=("n_k", HEAD_DIM),
            dtype=_FP8,
            distribution=_FINITE,
        ),
        "kv_scales": TensorSpec(
            description="Each key's fp32 scale, positive: aiter applies it before relu.",
            shape=("n_k",),
            dtype=_FLOAT32,
            distribution=LogUniformDistribution(log2_lo=-10, log2_hi=-3),
        ),
        "weights": TensorSpec(
            description="Each row's per-head weights.",
            shape=("n_q", "n_heads"),
            dtype=_FLOAT32,
            distribution=_FINITE,
        ),
        "cu_starts": TensorSpec(
            description="Each row's first key.",
            shape=("n_q",),
            dtype=_INT32,
            # shortcut: every row's window holds every key
            distribution=IntRangeDistribution(lo="0", hi="0"),
        ),
        "cu_ends": TensorSpec(
            description="Each row's end: its first key past the window.",
            shape=("n_q",),
            dtype=_INT32,
            distribution=IntRangeDistribution(lo="n_k", hi="n_k"),
        ),
        "logits": TensorSpec(
            description="The rows' logits, written in place inside each row's window.",
            shape=("n_q", "n_cols"),
            dtype=_FLOAT32,
            distribution=_FINITE,
            mutable=True,
            # the logits only rank keys for the top-k, from fp8 queries and keys
            error_bounds=SqnrBound(min_sqnr_db=40.0),
        ),
    },
    outputs={
        "no_values": TensorSpec(
            description="No values: the op's one result is logits, and a contract declares an output.",
            shape=(0,),
            dtype=_FLOAT32,
            # no element to compare
            error_bounds=SqnrBound(min_sqnr_db=0.0),
        ),
    },
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
    compare=CompareSpec(),
)

CONTRACTS = (FP8_MQA_LOGITS,)
