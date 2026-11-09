# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""No op of its own: the route in api.py serves sglang's hip_flash_mla through
aklr.ops.mla's flash_mla_with_kvcache. This module makes the package one the plugin finds
(locate.op_packages lists the plugin packages holding a contract.py) and imports api.py of."""

from __future__ import annotations

__all__ = ["CONTRACTS"]

CONTRACTS = ()
