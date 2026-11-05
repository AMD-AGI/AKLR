# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aklr.core.contract import semantics
from aklr.core.contract.fastkey import FastKeyCodec, FastKeyFnGenerator
from tests.op_case import OpCase


@pytest.mark.parametrize("decode", [semantics.to_float32, semantics.to_float64])
@pytest.mark.parametrize("has_bias", [False, True])
def test_mxfp8_reference_uses_each_rows_e8m0_blocks(decode, has_bias: bool) -> None:
    """Hand-computed sparse dots cross 32-value boundaries with distinct row scales."""
    case = OpCase.of("gemm_a8w8_mxfp8@1.0")
    point = case.point.model_copy(
        update={
            "dtypes": {**case.point.dtypes, "bias": "float32" if has_bias else None},
            "attrs": {**case.point.attrs, "has_bias": has_bias},
        }
    )
    a = torch.zeros(2, 128)
    a[:, [31, 32, 64, 96]] = torch.tensor([[1.0, 2.0, -1.0, 0.5], [-2.0, 1.0, 0.5, 4.0]])
    b = torch.zeros(16, 128)
    b[:2, [31, 32, 64, 96]] = torch.tensor([[1.0, 1.0, 1.0, 1.0], [2.0, -1.0, 0.5, 1.0]])
    b_scale = torch.full((16, 4), 127, dtype=torch.uint8)
    b_scale[:2] = torch.tensor([[127, 126, 128, 125], [128, 127, 126, 129]], dtype=torch.uint8)
    bias = torch.zeros(16) if has_bias else None
    if has_bias:
        bias[:2] = torch.tensor([0.25, -0.5])
    args = {
        "a": a.to(torch.float8_e4m3fn),
        "b": b.to(torch.float8_e4m3fn),
        "a_scale": torch.tensor([[127, 128, 126, 129], [126, 127, 129, 128]], dtype=torch.uint8),
        "b_scale": b_scale,
        "bias": bias,
    }
    actual = OpCase(point).reference_outputs(args, decode)["out"]
    dtype = torch.float64 if decode is semantics.to_float64 else torch.float32
    expected = torch.zeros(2, 16, dtype=dtype)
    expected[:, :2] = torch.tensor(
        [[2.75, 7.375], [5.75, 27.0]] if has_bias else [[2.5, 7.875], [5.5, 27.5]],
        dtype=dtype,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("k", [768, 3072, 6144])
@pytest.mark.parametrize("out_dtype", ["bfloat16", "float16"])
def test_mxfp8_gemm_accepts_minimax_reduction_lengths(k: int, out_dtype: str) -> None:
    """MiniMax's K values admit full row-major scale matrices and either 16-bit output."""
    base = OpCase.of("gemm_a8w8_mxfp8@1.0").point
    point = base.model_copy(
        update={"axes": {**base.axes, "k": k}, "dtypes": {**base.dtypes, "out": out_dtype}}
    )
    case = OpCase(point)
    call = case.call(case.inputs())
    assert tuple(call["a_scale"].shape) == (2, k // 32)
    assert tuple(call["b_scale"].shape) == (16, k // 32)
    codec = FastKeyCodec.from_contract(case.contract)
    fast_key = FastKeyFnGenerator(codec, torch.device("cpu")).generate()
    assert fast_key(call) == codec.to_fast_key(point)


@pytest.mark.parametrize(
    "invalid",
    ["transposed", "per_token", "fp32_scales", "block128", "strided_scales", "shuffled", "e5m2", "k", "n"],
)
def test_mxfp8_gemm_declines_incompatible_formats(invalid: str) -> None:
    """Different FP8 formats fail storage checks; unaligned sizes fall outside the domain."""
    base = OpCase.of("gemm_a8w8_mxfp8@1.0").point
    case = OpCase(base.model_copy(update={"axes": {"m": 2, "n": 32, "k": 256}}))
    call = case.call(case.inputs())
    codec = FastKeyCodec.from_contract(case.contract)
    fast_key = FastKeyFnGenerator(codec, torch.device("cpu")).generate()
    assert fast_key(call) is not None
    match invalid:
        case "transposed":
            call["trans_b"] = False
        case "per_token":
            call["a_scale"] = call["a_scale"][:, :1].contiguous()
        case "fp32_scales":
            call["a_scale"] = torch.ones_like(call["a_scale"], dtype=torch.float32)
        case "block128":
            call["b_scale"] = call["b_scale"][:1, :2].contiguous()
        case "strided_scales":
            call["a_scale"] = call["a_scale"].mT.contiguous().mT
        case "shuffled":
            call["b"].is_shuffled = True
        case "e5m2":
            call["a"] = call["a"].float().to(torch.float8_e5m2)
            call["b"] = call["b"].float().to(torch.float8_e5m2)
        case "k":
            call["a"] = call["a"][:, :192].contiguous()
            call["b"] = call["b"][:, :192].contiguous()
            call["a_scale"] = call["a_scale"][:, :6].contiguous()
            call["b_scale"] = call["b_scale"][:, :6].contiguous()
        case "n":
            call["b"] = call["b"][:24]
            call["b_scale"] = call["b_scale"][:24]
    key = fast_key(call)
    if invalid in ("k", "n"):
        # fast_key reads physical shapes; only domain-valid Points can serve kernels.
        assert key is not None
        with pytest.raises(ValueError, match="outside"):
            codec.to_fast_key(codec.to_point(key))
    else:
        assert key is None
