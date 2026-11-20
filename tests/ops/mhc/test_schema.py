# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch

from aklr.core.validation.stages.bench import InputSampler
from aklr.ops.mhc.schema import MHC_PRE


def test_mhc_pre_is_the_calls_without_a_norm_weight() -> None:
    # its committed kernel was written for them and reads no norm_weight
    call = InputSampler(MHC_PRE, {"n_tokens": 2, "hidden_size": 64}).draw(0, 0, "cpu")
    assert call.get("norm_weight") is None
    assert MHC_PRE.matches({"norm_weight": None, **call})
    assert not MHC_PRE.matches({**call, "norm_weight": torch.ones(64, dtype=torch.bfloat16)})
