# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.runtime.dispatch import api
from aklr.ops.kda_conv.contract import KIMI_K3_KDA_CONV

__all__ = ["kimi_k3_kda_conv"]


@api(KIMI_K3_KDA_CONV)
def kimi_k3_kda_conv(
    x,
    weight,
    conv_state,
    state_indices,
    has_initial_state,
    cu_seqlens,
    storage="row_major",
    out_dtype=None,
    out=None,
): ...
