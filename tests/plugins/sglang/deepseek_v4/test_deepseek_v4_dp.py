# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The DP gather's pad-row zeroing in one launch: the rows past the device's count of real
rows are zero, the others as they were, as sglang's ``_mask_dp_pad_rows`` leaves them."""

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.gpu


def _sglang(x, batch):
    pad = torch.arange(x.shape[0], device=x.device) >= batch.num_token_non_padded
    x.masked_fill_(pad.view(-1, 1), 0)


@pytest.mark.parametrize("rows, real", [(6, 6), (48, 30), (156, 1), (8, 0)])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_pad_rows_zeroed(rows, real, dtype):
    from aklr.plugins.sglang.models.deepseek_v4 import dp_routes

    x = torch.randn(rows, 5120, device="cuda").bfloat16()
    want = x.clone()
    batch = SimpleNamespace(num_token_non_padded=torch.tensor(real, dtype=dtype, device="cuda"))
    _sglang(want, batch)
    assert dp_routes._mask_dp_pad_rows(lambda *_: pytest.fail("sglang's"), x, batch) is None
    assert torch.equal(x, want)
