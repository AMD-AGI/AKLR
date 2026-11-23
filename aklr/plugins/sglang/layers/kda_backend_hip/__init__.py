# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3's KDA chain verify in ReplaySSM ring mode, as sglang runs it on gfx950."""

from aklr.plugins.sglang.layers.kda_backend_hip.api import kda_chain_verify_ring

__all__ = ["kda_chain_verify_ring"]
