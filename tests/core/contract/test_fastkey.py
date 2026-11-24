# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aklr.core.contract.fastkey import FastKeyCodec, FastKeyFnGenerator
from aklr.core.contract.semantics import DTYPE_TRAITS
from aklr.core.contract.spec import ScalarSpec
from tests.op_case import OpCase

# layouts whose pages view a bigger buffer: padded rows, or one half of each row
_PAGE_KINDS = ("deepseekv4_kv_pages", "kv_split_pages")


class TestFastKey:
    """The hot-path ``fast_key`` agrees with the offline ``FastKeyCodec``."""

    @pytest.mark.parametrize("form", ["as_is", "retyped", "strided", "unpadded"])
    def test_call_key_is_codec_key(self, op_case: OpCase, form: str) -> None:
        codec = FastKeyCodec.from_contract(op_case.contract)
        fast_key = FastKeyFnGenerator(codec, torch.device("cpu")).generate()
        call = op_case.call(op_case.inputs())
        stored = {n: t for n, t in call.items() if isinstance(t, torch.Tensor)}
        if form == "strided":
            # the stored bytes as a zero-copy view with transposed strides, as sglang
            # passes a column-major a_scale; fast_key hands on the stored tensor
            variant = op_case.contract.layout_variants.by_name[op_case.point.layout_variant_name]
            names = [n for n in stored if variant.layout_of(n).kind == "column_major"]
            if not names:
                pytest.skip("no column-major tensor")
            for n in names:
                call[n] = stored[n].view(*stored[n].shape[::-1]).mT
                assert not call[n].is_contiguous()
        if form == "unpadded":
            # a pool's pages copied out of their buffer are not its bytes
            variant = op_case.contract.layout_variants.by_name[op_case.point.layout_variant_name]
            names = [n for n in stored if variant.layout_of(n).kind in _PAGE_KINDS]
            if not names:
                pytest.skip("no padded kv pages")
            for n in names:
                call[n] = stored[n].contiguous()
            assert fast_key(call) is None
            return
        if form == "retyped":
            # a value of the wrong Python kind: 20 -> 20.0 for an int, True for a float
            scalars = {
                n: s
                for n, s in op_case.contract.inputs.items()
                if isinstance(s, ScalarSpec) and call.get(n) is not None
            }
            if not scalars:
                pytest.skip("no scalar input")
            for n, s in scalars.items():
                call[n] = float(call[n]) if DTYPE_TRAITS[s.dtype].is_integral else True
            assert fast_key(call) is None
            return
        key = fast_key(call)
        assert key == codec.to_fast_key(op_case.point)
        assert codec.to_point(key) == op_case.point
        for n, t in stored.items():  # the stored tensors, never a copy
            assert call[n].stride() == t.stride()
            assert call[n].data_ptr() == t.data_ptr()
            assert torch.equal(call[n].view(torch.uint8), t.view(torch.uint8))
