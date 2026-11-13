# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""aiter's public functions served by AKLR ops, for the sglang and vLLM plugins.

Both frameworks call some of the same aiter functions. One function here per aiter function turns a
call's arguments into an AKLR op call and returns the op's result, or the
:class:`~aklr.core.runtime.dispatch.FallbackReason` that sends the call back to aiter. Each framework's
plugin declares its route on the releases it was tested on, with the parameters of the aiter in that
release's image, and passes every argument by name."""

from aklr.plugins.aiter.fused_moe import fused_moe

__all__ = ["fused_moe"]
