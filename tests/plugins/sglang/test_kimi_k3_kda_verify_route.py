# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The Kimi-K3 KDA verify route on stand-ins of sglang 0.5.22: a verify of dense chains with
ReplaySSM rings goes to kimi_k3_kda_verify with sglang's projections and pools, and o_proj
takes its rows; other modes and verify layouts stay on sglang before any projection runs."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins.sglang.models.kimi_k3 import api

STEPS, REQUESTS, HEADS = 8, 2, 12
ROWS, WIDTH = STEPS * REQUESTS, HEADS * 128


class KDAAttnBackend:
    pass


class TritonKDAKernel:
    pass


class Mode:
    def __init__(self, verify):
        self.verify = verify

    def is_target_verify(self):
        return self.verify


class Op:
    """kimi_k3_kda_verify: takes every call, records what it was handed."""

    def __init__(self, ask=None):
        self.ask, self.asked, self.calls = ask, [], []

    def fallback_reason(self, **kwargs):
        self.asked.append(kwargs)
        return self.ask

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return torch.zeros(REQUESTS, STEPS, HEADS, 128, dtype=torch.bfloat16)


class Linear:
    def __init__(self, width):
        self.width, self.inputs = width, []

    def __call__(self, x):
        self.inputs.append(x)
        return torch.zeros(x.shape[0], self.width, dtype=torch.bfloat16), None


class Layer:
    """A KDA layer of the MXFP4 checkpoint at TP8: the full-rank gate's separate [f_a | b]
    projection, no one-GEMM in-projection."""

    def __init__(self):
        self.use_full_rank_gate, self._qkvgbfa_sizes = True, None
        self._bfa_w, self._bfa_f_b_w = torch.zeros(144, 7168), torch.zeros(WIDTH, 128)
        self._kda_fused_decode_ready, self.all_reduce_fusion = False, False
        self.fused_qkvg_proj, self.o_proj = Linear(4 * WIDTH), Linear(7168)
        self.attn = SimpleNamespace(
            layer_id=3,
            bias=None,
            lower_bound=-5.0,
            activation="silu",
            head_k_dim=128,
            v_dim=WIDTH,
            conv_weights=torch.zeros(3 * WIDTH, 4),
        )
        self.o_norm = SimpleNamespace(activation="sigmoid", bias=None, eps=1e-5, weight=torch.zeros(128))
        self.A_log, self.dt_bias = torch.zeros(1, 1, HEADS, 1), torch.zeros(WIDTH)


@pytest.fixture
def sglang(monkeypatch):
    """sglang 0.5.22's modules the route imports, as stand-ins; returns the KDA backend."""
    cache = SimpleNamespace(
        conv=[torch.zeros(9, 3, 3 * WIDTH)],
        temporal=torch.zeros(9, HEADS, 128, 128),
        intermediate_conv_window=[torch.zeros(9, STEPS, 3, 3 * WIDTH)],
        intermediate_ssm=None,
        replayssm_rawv=torch.zeros(9, HEADS, 16, 128),
        replayssm_rawk=torch.zeros(9, HEADS, 16, 128),
        replayssm_g=torch.zeros(9, HEADS, 16, 128),
        replayssm_beta=torch.zeros(9, HEADS, 16),
    )
    backend = KDAAttnBackend()
    backend.forward_metadata = SimpleNamespace(
        mamba_cache_indices=torch.tensor([5, 2, -1, -1], dtype=torch.int32),
        retrieve_next_token=None,
        retrieve_parent_token=None,
    )
    backend.req_to_token_pool = SimpleNamespace(mamba2_layer_cache=lambda layer_id: cache)
    backend.verify_intermediate_state_indices = torch.arange(9, dtype=torch.int32)
    backend.accept_lens_pool = None
    backend._fused_chain_verify_fn = None
    backend.kernel_dispatcher = SimpleNamespace(verify_kernel=TritonKDAKernel())
    backend.pp_stable_rows = False
    backend.gemm_calls = []

    def tiny_gemm(x, w):
        backend.gemm_calls.append((x, w))
        return torch.zeros(x.shape[0], w.shape[0], dtype=torch.bfloat16)

    modules = {
        "sglang.srt.model_executor.forward_context": {
            "get_attn_backend": lambda: SimpleNamespace(linear_attn_backend=backend)
        },
        "sglang.srt.runtime_context": {"get_lora": lambda: SimpleNamespace(enable_lora=False)},
        "sglang.srt.layers.attention.linear.kda_backend": {"KDAAttnBackend": KDAAttnBackend},
        "sglang.srt.layers.attention.linear.kernels.kda_triton": {"TritonKDAKernel": TritonKDAKernel},
        "sglang.srt.layers.attention.linear.utils": {
            "pp_spec_stable_rows_enabled": lambda: backend.pp_stable_rows
        },
        "sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph": {
            "get_tc_piecewise_forward_context": lambda: None
        },
        "sglang.kernels.ops.gemm": {"kimi_k3_tiny_gemm": tiny_gemm},
    }
    for name, attrs in modules.items():
        stand_in = ModuleType(name)
        stand_in.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, stand_in)
    backend.cache = cache
    return backend


@pytest.fixture
def op(monkeypatch):
    stand_in = Op()
    monkeypatch.setattr(api, "kimi_k3_kda_verify", stand_in)
    return stand_in


def forward(layer, verify=True, ragged=None):
    batch = SimpleNamespace(
        forward_mode=Mode(verify),
        spec_info=SimpleNamespace(draft_token_num=STEPS, ragged_verify_layout=ragged),
    )
    return api._kda_forward(layer, torch.zeros(ROWS, 7168, dtype=torch.bfloat16), None, batch, None)


def test_a_dense_verify_runs_the_op_between_the_projections_and_o_proj(sglang, op):
    """sglang's two projections as (requests, steps, columns) views, the layer's weights and
    pools, each request's slot and scratch row; o_proj takes the op's rows."""
    layer = Layer()
    out = forward(layer)
    assert out.shape == (ROWS, 7168)
    (call,) = op.calls
    assert call["projected_qkvg"].shape == (REQUESTS, STEPS, 4 * WIDTH)
    assert call["projected_bfa"].shape == (REQUESTS, STEPS, 144)
    (gemm,) = sglang.gemm_calls
    assert gemm[1] is layer._bfa_w and call["f_b_weight"] is layer._bfa_f_b_w
    cache = sglang.cache
    assert call["conv_state"] is cache.conv[0] and call["kda_state"] is cache.temporal
    assert call["conv_windows"] is cache.intermediate_conv_window[0]
    assert call["ring_raw_v"] is cache.replayssm_rawv and call["ring_beta"] is cache.replayssm_beta
    assert call["cache_indices"].tolist() == [5, 2] and call["window_indices"].tolist() == [0, 1]
    assert call["a_log"].shape == (HEADS,) and call["lower_bound"] == -5.0 and call["eps"] == 1e-5
    (asked,) = op.asked
    assert asked["projected_qkvg"].shape == call["projected_qkvg"].shape
    (projected,) = layer.o_proj.inputs
    assert projected.shape == (ROWS, WIDTH)


def test_an_op_that_declines_leaves_the_layer_to_sglang(sglang, op):
    op.ask = FallbackReason.NO_WORKLOAD
    layer = Layer()
    assert forward(layer) is FallbackReason.NO_WORKLOAD
    assert not op.calls and not layer.fused_qkvg_proj.inputs and not sglang.gemm_calls


@pytest.mark.parametrize(
    "case",
    ["decode", "ragged", "snapshot_scratch", "pp_stable_rows", "fused_accept", "fused_in_projection"],
)
def test_other_calls_stay_on_sglang(sglang, op, case):
    """Other modes and verify layouts: sglang's own path, before any op or projection runs."""
    layer, verify, ragged = Layer(), True, None
    if case == "decode":
        verify = False
    elif case == "ragged":
        ragged = object()
    elif case == "snapshot_scratch":
        sglang.cache.replayssm_rawv = None
    elif case == "pp_stable_rows":
        sglang.pp_stable_rows = True
    elif case == "fused_accept":
        sglang.accept_lens_pool = torch.ones(9, dtype=torch.int32)
    else:
        layer._qkvgbfa_sizes, layer._qkvgbfa_bs_limit = [4608, 1536, 128, 12, 4], 64
    assert forward(layer, verify, ragged) is FallbackReason.OUT_OF_CONTRACT
    assert not op.asked and not op.calls and not layer.fused_qkvg_proj.inputs
