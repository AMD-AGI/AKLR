# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest

from tests.op_case import OpCase
from tests.plugins.vllm import cases


@pytest.mark.gpu
@pytest.mark.parametrize("op_case", OpCase.params(cases.SMALL_POINTS))
def test_backend_matches_reference(op_case: OpCase) -> None:
    """``check_backend_matches_reference`` in vllm-test: vLLM's call on the aiter its image ships."""
    op_case.check_backend_matches_reference()
