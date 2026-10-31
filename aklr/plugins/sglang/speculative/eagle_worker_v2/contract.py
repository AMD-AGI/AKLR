# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import DType, FiniteDistribution, OutputRoundingBound
from aklr.core.contract.spec import (
    DTypeSpec,
    LayoutVariant,
    LayoutVariantsSpec,
    TensorSpec,
    axis,
)

__all__ = ["CONTRACTS", "EAGLE_WORKER_V2_DRAFT_TOP1"]

EAGLE_WORKER_V2_DRAFT_TOP1 = OpContract(
    contract_id="eagle_worker_v2_draft_top1@1.0",
    description=(
        "sglang's EAGLE draft token at topk 1 without rejection sampling: each row's highest "
        "logit's index, as torch.argmax (NaN highest, the first of equal values), and its "
        "probability as sglang's topk 1 path records it, 1."
    ),
    op_type="eagle_worker_v2",
    axes={
        "n_rows": axis("Rows, one per request.", 1, 4096, is_var=True),
        "vocab_size": axis("Logits per row.", 4, 2**20, multiple_of=4),
    },
    inputs={
        "logits": TensorSpec(
            description="The draft's next-token logits.",
            shape=("n_rows", "vocab_size"),
            dtype=DTypeSpec(any_of=(DType.FLOAT32,)),
            distribution=FiniteDistribution(),
        ),
    },
    outputs={
        "topk_p": TensorSpec(
            description="1.",
            shape=("n_rows", 1),
            dtype=DTypeSpec(any_of=(DType.FLOAT32,)),
            error_bounds=OutputRoundingBound(),
        ),
        "topk_index": TensorSpec(
            description="The highest logit's index.",
            shape=("n_rows", 1),
            dtype=DTypeSpec(any_of=(DType.INT64,)),
        ),
    },
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
)

CONTRACTS = (EAGLE_WORKER_V2_DRAFT_TOP1,)
