# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.runtime.dispatch import api
from aklr.ops.mla_absorb.contract import (
    MLA_KC_ROPE_CACHE,
    MLA_VC,
)

__all__ = ["mla_kc_rope_cache", "mla_vc"]


@api(MLA_KC_ROPE_CACHE)
def mla_kc_rope_cache(
    q,
    qkv_latent,
    kv_latent,
    w_kc,
    w_scale,
    cos_cache,
    sin_cache,
    positions,
    out_cache_loc,
    kv_cache,
): ...


@api(MLA_VC)
def mla_vc(attn_output, w_vc, w_scale): ...
