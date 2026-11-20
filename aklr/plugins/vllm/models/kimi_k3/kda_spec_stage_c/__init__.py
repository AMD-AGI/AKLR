# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3's KDA layer between in_proj and o_proj in a speculative verify step: f_b_proj, the short convolution, the
recurrence and the gated norm in one launch, routed at the decoder layer's self-attention call."""

from aklr.plugins.vllm.models.kimi_k3.kda_spec_stage_c.api import kimi_k3_kda_spec_stage_c

__all__ = ["kimi_k3_kda_spec_stage_c"]
