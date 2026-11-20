# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

import aklr.ops
from tests.op_sample import OpSample

_AXES = {"M": 4, "N": 32, "K": 64}
_SAMPLES = [
    (OpSample(aklr.ops.gemm_bf16_nt, "gemm_bf16_nt", _AXES), "bfloat16"),
    (OpSample(aklr.ops.gemm_bf16_nt, "gemm_bf16_nt_out_float32", _AXES), "float32"),
    (OpSample(aklr.ops.gemm_a8w8_nt, "gemm_a8w8_nt", _AXES), "bfloat16"),
    (OpSample(aklr.ops.gemm_a8w8_nt, "gemm_a8w8_nt_shuffled", _AXES), "bfloat16"),
]


@pytest.mark.parametrize(("sample", "out_dtype"), _SAMPLES, ids=lambda v: getattr(v, "name", v))
def test_the_init_draws_a_call_of_its_variant_at_the_output_dtype_its_schema_records(
    sample: OpSample, out_dtype: str
) -> None:
    call = sample.call()
    assert sample.api.resolve({"out": None, "use_shuffled_weight": False, **call}) is sample.template
    schema = sample.schema()
    assert schema.name == sample.name + "_n32_k64"
    assert schema.outputs["out"].dtype == out_dtype
    reference = schema.standard.reference
    out = reference(**{n: v for n, v in call.items() if n not in ("out_dtype", "use_shuffled_weight")})
    assert list(out.shape) == [4, 32]
    assert schema.standard.check(out, out.to(getattr(torch, out_dtype)))


def test_a_float32_output_buffer_makes_a_call_the_float32_variants() -> None:
    call = OpSample(aklr.ops.gemm_bf16_nt, "gemm_bf16_nt", _AXES).call()
    call |= {"out": torch.empty(4, 32), "use_shuffled_weight": False}
    assert aklr.ops.gemm_bf16_nt.resolve(call).op_name == "gemm_bf16_nt.out_float32"
