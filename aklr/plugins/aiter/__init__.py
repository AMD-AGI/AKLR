# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""aiter's fused_moe call turned into AKLR's fused_moe op, for the aiter provider.

The provider (``aklr.plugins.aiter.provider``) serves aiter's routed ops; sglang and vLLM reach AKLR
through it once they enable aiter's kernel plugin interface, with no route of their own on an aiter
function. :func:`fused_moe` turns a routed call's arguments into the op call and returns the op's
result, or the :class:`~aklr.core.runtime.dispatch.FallbackReason` that sends the call back to aiter."""

from aklr.plugins.aiter.fused_moe import fused_moe

__all__ = ["fused_moe"]
