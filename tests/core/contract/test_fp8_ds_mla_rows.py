# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aklr.core.contract.semantics import DType, Fp8DsMlaRowsDistribution


@pytest.mark.parametrize("shape", [(2, 656), (3, 1, 656), (300, 656)])
def test_fp8_ds_mla_rows_sample_and_domain(shape):
    distribution = Fp8DsMlaRowsDistribution()
    result = distribution.sample(shape, DType.UINT8, torch.Generator().manual_seed(17))
    assert tuple(result.shape) == shape and result.dtype == torch.uint8
    assert distribution.contains(result, DType.UINT8)
    raw = result.reshape(-1, 656)
    scales = raw[:, 512:528].contiguous().view(torch.float32)
    assert ((scales >= 2.0**-10) & (scales <= 2.0**-7)).all()
    raw[0, 512] = 1  # a scale off a power of two
    assert not distribution.contains(result, DType.UINT8)
    raw[0, 512] = 0
    raw[0, 0] = 0x7F  # a NaN code
    assert not distribution.contains(result, DType.UINT8)


@pytest.mark.parametrize(("scale", "is_in"), [(2.0**7, True), (2.0**8, False)], ids=["2**7", "2**8"])
def test_fp8_ds_mla_rows_scales_go_up_to_2_to_the_7(scale, is_in):
    distribution = Fp8DsMlaRowsDistribution()
    rows = distribution.sample((2, 656), DType.UINT8, torch.Generator().manual_seed(17))
    rows[1, 524:528] = torch.tensor([scale]).view(torch.uint8)
    assert distribution.contains(rows, DType.UINT8) == is_in


def test_fp8_ds_mla_rows_reject_wrong_width():
    with pytest.raises(ValueError, match="656 bytes"):
        Fp8DsMlaRowsDistribution().sample((2, 576), DType.UINT8, torch.Generator())
