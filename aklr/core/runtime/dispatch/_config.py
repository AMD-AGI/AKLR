###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 by FlashInfer-Bench Contributors
#
#


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


class DispatchConfigRegistry(ConfigRegistry[DispatchConfig]): ...


@cache
def get_default_registry() -> DispatchConfigRegistry:
    return DispatchConfigRegistry(default=DispatchConfig())
