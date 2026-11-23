# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax TP8's 384-wide experts require padded, shuffled E8M0 scales."""

from aklr.ops.fused_moe.contract import FUSED_MOE
from tests.ops.cases import SMALL_POINTS


def test_swiglu_accepts_384_wide_experts():
    point = next(
        p
        for p in SMALL_POINTS
        if p.contract_id == "fused_moe@1.0"
        and p.attrs["activation"] == "swiglu"
        and not p.attrs["has_expert_mask"]
    )
    point = point.model_copy(update={"axes": point.axes | {"inter_dim": 384}})
    assert FUSED_MOE.domain.contains(point)
