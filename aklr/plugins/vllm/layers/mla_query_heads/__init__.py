# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The MLA decode query's heads repeated to the head count aiter's decode kernels take (vLLM's
``AiterMLAHelper.get_mla_padded_q``), in one launch."""

from aklr.plugins.vllm.layers.mla_query_heads.api import mla_query_heads

__all__ = ["mla_query_heads"]
