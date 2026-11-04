# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aklr.core.contract.semantics import DType, Mxfp8NopeRowsDistribution


@pytest.mark.parametrize("shape", [(2, 512), (2, 3, 512), (129, 512)])
def test_packed_rows_sample_and_domain(shape):
    distribution = Mxfp8NopeRowsDistribution()
    result = distribution.sample(shape, DType.UINT8, torch.Generator().manual_seed(17))
    assert tuple(result.shape) == shape
    assert distribution.contains(result, DType.UINT8)
    assert (result[..., 462:] == 0).all()
    raw = result.reshape(-1, 512)
    raw[0, 448] = 255
    assert not distribution.contains(result, DType.UINT8)
    raw[0, 448] = 127
    raw[0, 0] = 0x7F
    assert not distribution.contains(result, DType.UINT8)


def test_packed_rows_reject_wrong_width():
    with pytest.raises(ValueError, match="512 bytes"):
        Mxfp8NopeRowsDistribution().sample((2, 511), DType.UINT8, torch.Generator())
