###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

from aklr.core.runtime.dispatch import aklr_api
from aklr.plugins.aiter.kda_conv.schema import KIMI_K3_KDA_CONV_VARIANTS

__all__ = ["kimi_k3_kda_conv"]


@aklr_api(KIMI_K3_KDA_CONV_VARIANTS)
def kimi_k3_kda_conv(
    x,
    weight,
    conv_state,
    state_indices,
    has_initial_state,
    cu_seqlens,
    storage="row_major",
    q=None,
    k=None,
    v=None,
): ...
