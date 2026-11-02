###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

from aklr.core.schema import Const, Scalar, Schema, Tensor, Var, default_check
from aklr.ops.topk.reference import topk_transform_paged_reference

__all__ = ["TOPK_TRANSFORM_PAGED"]


def _topk_transform_paged_check(reference_outputs, actual_outputs, **_unused):
    # a full row's selections may come in any order: compare each row sorted
    reference, actual = (
        o[0] if isinstance(o, (list, tuple)) else o for o in (reference_outputs, actual_outputs)
    )
    if reference.shape != actual.shape:
        return False
    return default_check(
        reference.sort(dim=-1).values,
        actual.sort(dim=-1).values,
        min_cos_sim=None,
    )


TOPK_TRANSFORM_PAGED = Schema(
    op_type="topk_transform_paged",
    description=(
        "Select each row's k highest scores among its first seq_lens "
        "candidates; write their positions, or their page slots, padded with -1."
    ),
    axes={
        "n_tokens": Var("Rows, one per token."),
        "n_candidates": Var("Candidate scores per row."),
        "k": Const("Candidates selected per row.", abbrev="k"),
        "n_pages": Var("Page table columns."),
        "page_size": Const("Slots per page.", abbrev="ps"),
    },
    constraints=[
        "page_size >= 1",
        # every slot fits in int32
        "page_tables is None or n_pages * page_size <= 2147483648",
    ],
    inputs={
        "scores": Tensor(
            ["n_tokens", "n_candidates"],
            dtype="float32",
            description="Candidate scores per row.",
        ),
        "seq_lens": Tensor(
            ["n_tokens"],
            dtype="int32",
            description="Valid candidates per row, a prefix the page table covers.",
        ),
        "page_tables": Tensor(
            ["n_tokens", "n_pages"],
            dtype="int32",
            optional=True,
            description="Page of each page_size candidates, distinct within a row.",
        ),
        "k": Scalar("int32", description="Candidates selected per row."),
        "page_size": Scalar("int32", optional=True, description="Slots per page."),
    },
    outputs={
        "out_page_indices": Tensor(
            ["n_tokens", "k"],
            dtype="int32",
            description=(
                "Selected positions, or page_tables slots when passed; -1 after a short row's entries."
            ),
        )
    },
    reference=topk_transform_paged_reference,
    check=_topk_transform_paged_check,
)
