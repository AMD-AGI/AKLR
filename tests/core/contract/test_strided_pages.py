# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch

from aklr.core.contract.semantics import StridedPages


def test_strided_pages_keep_each_page_dense_and_apart():
    pages = torch.arange(3 * 4 * 6, dtype=torch.uint8).reshape(3, 4, 1, 6)
    stored = StridedPages().encode(pages)
    assert torch.equal(stored, pages)
    assert stored.stride()[1:] == (6, 6, 1) and stored.stride(0) > 4 * 6
    assert eval(StridedPages().storage_test(), {"T": stored})
    assert eval(StridedPages().storage_test(), {"T": pages})
    assert not eval(StridedPages().storage_test(), {"T": pages[:, ::2]})
