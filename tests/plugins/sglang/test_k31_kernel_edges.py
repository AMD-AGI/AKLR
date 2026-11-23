# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The committed Kimi-K3.1 KDA chain-verify kernel on finite inputs at the edges of float32 (gfx950): each case is
a small Point drawn as validation draws it, with the few values that once broke the kernel set by hand, and the
kernel's outputs pass the contract's compare against the reference."""

import json
from pathlib import Path

import pytest
import torch

from aklr.core.contract.spec import PointSpec
from tests.op_case import OpCase

_LAYERS = Path(__file__).resolve().parents[3] / "aklr/plugins/sglang/layers"
_KDA_VERIFY = "kda_backend_hip/impls/hip/kda_chain_verify_ring_n_verify_tokens5_n_heads12"


def _committed_kernel(directory: str) -> dict:
    (path,) = (_LAYERS / directory).glob("*.json")
    return json.loads(path.read_text())["kernel"]


def _run(builder_registry, case: OpCase, kernel: dict, args: dict) -> None:
    """The kernel on ``args`` passes ``case``'s compare against the reference."""
    schema = case.contract.narrow(
        case.point, workloads=[case.contract.workload_for(case.point, {})], kernel=kernel
    )
    runnable = builder_registry.build(schema)
    call = dict(case.call(args, "cuda"))
    for name in case.contract.outputs:
        shape = case.contract.dtype_storage_shape_of(name, case.point)
        call[name] = torch.empty(shape, dtype=getattr(torch, case.point.dtypes[name]), device="cuda")
    runnable(**call)
    torch.cuda.synchronize()
    outputs = {n: call[n] for n in (*case.contract.outputs, *case.contract.mutable_inputs)}
    for name, tensor in outputs.items():
        assert tensor.float().isfinite().all(), name
    failed = {o: r.detail for o, r in case.compare(outputs, args).items() if not r.is_passed}
    assert failed == {}


def _kda_case() -> OpCase:
    return OpCase(
        PointSpec(
            contract_id="kda_chain_verify_ring@1.0",
            axes={
                "n_requests": 1,
                "n_tokens": 5,
                "n_verify_tokens": 5,
                "n_heads": 12,
                "n_slots": 2,
                "n_lines": 1,
                "ring_len": 8,
            },
            dtypes={
                "mixed_qkv": "bfloat16",
                "conv_weight": "float32",
                "conv_state": "bfloat16",
                "cache_indices": "int32",
                "conv_window": "bfloat16",
                "window_indices": "int32",
                "a": "bfloat16",
                "b": "bfloat16",
                "a_log": "float32",
                "dt_bias": "float32",
                "ssm_state": "float32",
                "ring_v": "bfloat16",
                "ring_k": "bfloat16",
                "ring_g": "float32",
                "ring_beta": "float32",
                "output": "bfloat16",
            },
            attrs={"has_lower_bound": True},
            layout_variant_name="row_major",
        )
    )


def _kda_args(case: OpCase) -> dict:
    args = case.inputs()
    args["cache_indices"].fill_(1)  # a live request
    return args


@pytest.mark.gfx950
def test_kda_verify_normalizes_query_and_key_channels_past_float32s_squares(builder_registry):
    """Query and key channels of 2**64 (taps 0, 0, 0, 1): their squares pass float32, their norms do not."""
    case = _kda_case()
    args = _kda_args(case)
    qk = 2 * 12 * 128
    args["mixed_qkv"][:, :qk] = 2.0**64
    args["mixed_qkv"][:, qk:] = 0
    args["conv_state"].zero_()
    args["conv_weight"].copy_(torch.tensor([0.0, 0.0, 0.0, 1.0]).expand_as(args["conv_weight"]))
    args["a"].fill_(-100)
    args["a_log"].zero_()
    args["dt_bias"].zero_()
    args["b"].zero_()
    args["ssm_state"].copy_(torch.eye(128).expand_as(args["ssm_state"]))
    _run(builder_registry, case, _committed_kernel(_KDA_VERIFY), args)


@pytest.mark.gfx950
def test_kda_verify_gates_a_scale_past_float32(builder_registry):
    """A_log = 100 (exp(A_log) past float32) and a = dt_bias = 0: the gate is lower_bound / 2, not inf * 0."""
    case = _kda_case()
    args = _kda_args(case)
    args["a_log"].fill_(100)
    args["a"].zero_()
    args["dt_bias"].zero_()
    _run(builder_registry, case, _committed_kernel(_KDA_VERIFY), args)


@pytest.mark.gfx950
def test_kda_verify_convolves_taps_that_cancel(builder_registry):
    """Inputs of 1 and taps 2**26, 1, -2**26, 0: the convolution is 1, which a float32 sum loses."""
    case = _kda_case()
    args = _kda_args(case)
    args["mixed_qkv"].fill_(1)
    args["conv_state"].fill_(1)
    args["conv_weight"].copy_(torch.tensor([2.0**26, 1.0, -(2.0**26), 0.0]).expand_as(args["conv_weight"]))
    _run(builder_registry, case, _committed_kernel(_KDA_VERIFY), args)
