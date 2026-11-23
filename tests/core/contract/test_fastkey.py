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


def test_rows_of_one_page_grid_apart_in_every_page_share_no_byte() -> None:
    """Two views of one buffer's pages, each a dense part of every page (vLLM's Mamba pages: a slot's convolution
    state, then its recurrent state), are apart; rows that meet, or of another stride, are not."""
    from aklr.core.contract.fastkey._fast_key_fn import _apart_in_pages

    pages = torch.zeros(6, 64 + 128 + 16)  # 6 pages of a 64-float part, a 128-float part, padding
    conv = pages[:, :64].view(torch.bfloat16).view(6, 4, 32)
    state = pages[:, 64:192].view(6, 8, 16)
    assert _apart_in_pages(conv, state, conv.data_ptr(), state.data_ptr())
    assert _apart_in_pages(state, conv, state.data_ptr(), conv.data_ptr())
    meeting = pages.as_strided((6, 8, 16), (pages.stride(0), 16, 1), 60)
    assert not _apart_in_pages(conv, meeting, conv.data_ptr(), meeting.data_ptr())
    other_stride = torch.zeros(6, 300)[:, :128].view(6, 8, 16)
    assert not _apart_in_pages(conv, other_stride, conv.data_ptr(), other_stride.data_ptr())


def _paged_states(state_page: int) -> tuple[torch.Tensor, torch.Tensor]:
    """KDA's convolution state and recurrent state of 9 slots as vLLM's Mamba pages hold them: one buffer of pages,
    each a slot's 5 x 768 bf16 convolution row then its 256 x 128 float32 state row. The state view starts at page
    ``state_page``: 0 keeps every page's two rows apart; 1 puts each state row on the next slot's convolution row."""
    conv_bytes, state_bytes = 5 * 768 * 2, 256 * 128 * 4
    page = conv_bytes + state_bytes
    pages = torch.zeros(10, page, dtype=torch.uint8)
    conv = pages[:9, :conv_bytes].view(torch.bfloat16).view(9, 5, 768)
    start = conv_bytes if state_page == 0 else 0
    state = (
        pages[state_page : state_page + 9, start : start + state_bytes].view(torch.float32).view(9, 256, 128)
    )
    return conv, state


@pytest.mark.parametrize("state_page", [0, 1], ids=["apart", "aliased"])
def test_the_paged_states_are_served_only_when_apart(state_page: int) -> None:
    """The fast key serves stage C on vLLM's Mamba pages, whose convolution and recurrent state rows interleave, and
    declines when the two share bytes: a state view one page after the convolution view lands on the convolution rows
    of the later pages, though their logical sizes end before it."""
    op_case = OpCase.of("kimi_k3_kda_spec_stage_c@1.0")
    fast_key = FastKeyFnGenerator(
        FastKeyCodec.from_contract(op_case.contract), torch.device("cpu")
    ).generate()
    call = op_case.call(op_case.inputs())
    call["conv_state"], call["state_cache"] = _paged_states(state_page)
    conv, state = call["conv_state"], call["state_cache"]
    starts = sorted([conv.data_ptr(), state.data_ptr()])
    assert state_page == 0 or starts[1] - starts[0] >= conv.nbytes  # apart by their logical sizes
    assert (fast_key(call) is None) == (state_page == 1)
