# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest

from aklr.core.contract.cost import Cost
from aklr.core.platform.roofline import Roofline


class TestRoofline:
    """``Roofline.seconds``: the longer of a call's math and its bytes, never under the floor."""

    def test_seconds(self) -> None:
        roofline = Roofline(flops_per_second={"bfloat16": 1e15}, bytes_per_second=1e12, floor_seconds=1e-6)
        assert roofline.seconds(Cost(4 * 10**9, 10**6, "bfloat16")) == pytest.approx(4e-6)
        assert roofline.seconds(Cost(10**9, 3 * 10**6, "bfloat16")) == pytest.approx(3e-6)
        assert roofline.seconds(Cost(0, 10**5, None)) == pytest.approx(1e-6)
        with pytest.raises(ValueError, match="no matrix-core rate for float4_e2m1"):
            roofline.seconds(Cost(1, 1, "float4_e2m1"))
