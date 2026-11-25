# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The DSA sparse MLA decode's reference: the same computation, whichever kernel sglang runs."""

from __future__ import annotations

from aklr.plugins.sglang.layers.dsa_backend.reference import (
    dsa_sparse_mla_decode_reference as sparse_mla_verify_reference,
)

__all__ = ["sparse_mla_verify_reference"]
