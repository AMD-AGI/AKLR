# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What the model routes leave to sglang before asking an op, on stand-ins of sglang 0.5.23."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins.sglang.models.deepseek_v2 import api as deepseek_v2
from aklr.plugins.sglang.models.qwen3_5 import api as qwen3_5
from aklr.plugins.sglang.models.qwen3_moe import api as qwen3_moe
from aklr.plugins.sglang.models.qwen3_next import api as qwen3_next


@pytest.fixture
def moe_checks(monkeypatch):
    """The checks after DeepSeek-V2's own, recorded when reached, which send the call back to sglang."""
    reached = []
    monkeypatch.setattr(deepseek_v2.moe_blocks, "runtime_falls_back", lambda: reached.append(True) or True)
    monkeypatch.setattr(deepseek_v2, "_moe_call", lambda *args: pytest.fail("an op was asked"))
    return reached


def forward_context(monkeypatch, bound):
    """sglang's forward context, its MoE padding bound ``bound``."""
    module = ModuleType("sglang.srt.model_executor.forward_context")
    module.has_forward_context = lambda: True
    module.get_forward_context = lambda: SimpleNamespace(moe_num_token_non_padded=lambda: bound)
    monkeypatch.setitem(sys.modules, module.__name__, module)


@pytest.mark.parametrize(
    ("bound", "masked"), [(None, False), (torch.tensor([3]), True)], ids=["no_padding", "padded_rows"]
)
def test_a_deepseek_v2_moe_whose_padded_rows_sglang_masks_stays_on_sglang(
    monkeypatch, moe_checks, bound, masked
):
    """sglang 0.5.23's top-k masks padded rows by the forward context's bound, no longer an argument."""
    forward_context(monkeypatch, bound)
    assert (
        deepseek_v2._moe_forward_normal(SimpleNamespace(), torch.zeros(4, 8))
        is FallbackReason.OUT_OF_CONTRACT
    )
    assert moe_checks == ([] if masked else [True])


def test_a_deepseek_v2_moe_whose_output_sum_is_left_stays_on_sglang(moe_checks):
    """sglang 0.5.23's leave_output_sum: a later step sums the experts' output, the op's sum."""
    assert (
        deepseek_v2._moe_forward_normal(SimpleNamespace(), torch.zeros(4, 8), leave_output_sum=True)
        is FallbackReason.OUT_OF_CONTRACT
    )
    assert moe_checks == []


@pytest.mark.parametrize(
    ("bound", "masked"), [(None, False), (torch.tensor([3]), True)], ids=["no_padding", "padded_rows"]
)
def test_a_qwen3_moe_whose_padded_rows_sglang_masks_stays_on_sglang(monkeypatch, bound, masked):
    """sglang 0.5.23's top-k masks padded rows by the forward context's bound, a bound 0.5.22's
    Qwen3-MoE forward_normal never passed it."""
    reached = []
    monkeypatch.setattr(qwen3_moe.moe_blocks, "runtime_falls_back", lambda: reached.append(True) or True)
    monkeypatch.setattr(qwen3_moe, "qwen3_moe_fp8_moe", lambda **_: pytest.fail("an op was asked"))
    forward_context(monkeypatch, bound)
    assert (
        qwen3_moe._moe_forward_normal(SimpleNamespace(), torch.zeros(4, 8)) is FallbackReason.OUT_OF_CONTRACT
    )
    assert reached == ([] if masked else [True])


@pytest.mark.parametrize("api", [qwen3_5, qwen3_next], ids=["qwen3_5", "qwen3_next"])
def test_a_dp_relabeled_decode_stays_on_sglang(api):
    """sglang 0.5.23 relabels a decode batch a one-token extend under DP attention."""
    batch = SimpleNamespace(forward_mode="extend", logical_forward_mode="decode")
    assert api._gdn_forward(SimpleNamespace(), torch.zeros(4, 8), batch) is FallbackReason.OUT_OF_CONTRACT
