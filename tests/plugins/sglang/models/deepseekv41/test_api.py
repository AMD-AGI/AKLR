# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What sglang's DeepSeek-V4.1 routes hand the ops, on tensors shaped as sglang's."""

import torch

from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins.sglang.models.deepseekv41 import api


def test_a_steps_positions_reach_sparse_mla_on_the_16_bytes_dispatch_requires(monkeypatch):
    asked = []
    monkeypatch.setattr(
        api, "deepseekv41_sparse_mla", lambda **call: asked.append(call) or FallbackReason.NO_WORKLOAD
    )
    # sglang passes a step's positions as a slice of the batch's: int64, so 8 bytes past a boundary at an odd row
    positions = torch.arange(5)[1:]
    reason = api._aiter_sparse_decode_fwd(
        torch.zeros(4, 1, 16, 512, dtype=torch.bfloat16),
        torch.zeros(2, 64, 1, 584, dtype=torch.uint8),
        torch.zeros(4, 1, 64, dtype=torch.int32),
        torch.zeros(16),
        0.044,
        inv_rope=(torch.zeros(16, 64), positions),
    )
    assert reason is FallbackReason.NO_WORKLOAD
    (call,) = asked
    assert call["positions"].tolist() == [1, 2, 3, 4] and call["positions"].data_ptr() % 16 == 0
