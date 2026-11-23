# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3.1's DSA sparse MLA prefill on its fp8_ds_mla pool, as sglang runs it on gfx950."""

from aklr.plugins.sglang.layers.dsa_backend_hip.api import dsa_sparse_mla_fp8_ds_prefill

__all__ = ["dsa_sparse_mla_fp8_ds_prefill"]
