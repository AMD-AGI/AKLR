###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 by FlashInfer-Bench Contributors
#
# Adapted from FlashInfer-Bench (https://github.com/flashinfer-ai/flashinfer-bench)
# Modified by the AKLR team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of AKLR (see LICENSE).

"""Dispatch configs: how an op's variant is served."""

from __future__ import annotations

from functools import cache
from typing import Literal

from pydantic import BaseModel, Field

from aklr.core.runtime._config_registry import ConfigRegistry


class DispatchConfig(BaseModel):
    aot_ratio: float = Field(default=1.0, ge=0, le=1)
    """The share of each Const-axes point's key-winning kernels built when the dispatcher starts, most wins first."""
    on_miss_policy: Literal["fallback_only", "use_best_kernel"] = "fallback_only"
    """On a key the entry lacks: run the fallback, or the kernel ranked first on the most keys."""


class DispatchConfigRegistry(ConfigRegistry[DispatchConfig]):
    """Dispatch configs by op name (``gemm_bf16_nt.shuffled``); one with no config and no default is never dispatched."""


@cache
def get_default_registry() -> DispatchConfigRegistry:
    return DispatchConfigRegistry(default=DispatchConfig())
