# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The route of the DSA backend's tilelang sparse MLA (DeepseekSparseAttnBackend._forward_tilelang, sglang 0.5.23):
which op each call goes to (a decode's to dsa_sparse_mla_decode, an extend's to dsa_sparse_mla_prefill, a speculative
step's to sparse_mla_verify), each against tilelang, on stand-ins of sglang's modules and of its backend."""

import pkgutil
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins.sglang import hook
from aklr.plugins.sglang.layers.dsa_backend import api as dsa_api
from aklr.plugins.sglang.layers.sparse_mla import api

HEADS, WIDTH, LATENT, TOPK, SLOTS = 2, 576, 512, 64, 128
TARGET = "sglang.srt.layers.attention.dsa_backend.DeepseekSparseAttnBackend._forward_tilelang"


class Op:
    """An op stand-in that records its calls and returns ``result``, else a [tokens, heads, latent] of each
    token's index."""

    def __init__(self, result=None):
        self.result, self.calls = result, []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.result is not None:
            return self.result
        tokens = kwargs["q"].shape[0]
        return torch.arange(tokens, dtype=torch.float32)[:, None, None].expand(tokens, HEADS, LATENT).clone()


class Backend:
    """sglang 0.5.23.dev20261120's DeepseekSparseAttnBackend as the routes read it: no dcp_size, an
    attn_dcp_size, and forward_metadata's host lengths."""

    def __init__(self, extend_lens, seq_lens_sum, attn_dcp_size=1):
        self.attn_dcp_size = attn_dcp_size
        self.forward_metadata = SimpleNamespace(
            dsa_extend_seq_lens_list=extend_lens, seq_lens_sum=seq_lens_sum
        )


@pytest.fixture
def sglang(monkeypatch):
    """sglang's runtime context, forward context and DSA backend; returns a setter of this forward's backend
    and speculative algorithm."""
    state = SimpleNamespace(backend=None, algorithm=None)
    modules = {
        "sglang.srt.runtime_context": dict(
            get_spec=lambda: SimpleNamespace(speculative_algorithm=state.algorithm)
        ),
        "sglang.srt.model_executor.forward_context": dict(
            has_forward_context=lambda: state.backend is not None, get_attn_backend=lambda: state.backend
        ),
        "sglang.srt.layers.attention.dsa_backend": dict(DeepseekSparseAttnBackend=Backend),
    }
    for name, attrs in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    return state


@pytest.fixture
def ops(monkeypatch):
    decode, prefill, verify = Op(), Op(), Op()
    monkeypatch.setattr(api, "dsa_sparse_mla_decode", decode)
    monkeypatch.setattr(api, "dsa_sparse_mla_prefill", prefill)
    monkeypatch.setattr(api, "sparse_mla_verify", verify)
    return SimpleNamespace(decode=decode, prefill=prefill, verify=verify)


def call(backend, tokens):
    q_all = torch.randn(tokens, HEADS, WIDTH, dtype=torch.bfloat16)
    kv_cache = torch.zeros(SLOTS, 1, WIDTH, dtype=torch.float8_e4m3fn)
    page_table_1 = torch.randint(0, SLOTS, (tokens, TOPK), dtype=torch.int32)
    output = api._forward_tilelang(backend, q_all, kv_cache, LATENT, page_table_1, 0.0625)
    return output, q_all, kv_cache, page_table_1


def test_the_route_admits_the_dev_build_the_image_runs(monkeypatch):
    """sglang 0.5.23.dev20261120, a pre-release of 0.5.23, which an exact "0.5.23" does not admit."""
    registered = []
    registry = ModuleType("sglang.srt.plugins.hook_registry")
    registry.HookType = SimpleNamespace(AROUND="around")
    registry.HookRegistry = SimpleNamespace(register=lambda *args: registered.append(args))
    monkeypatch.setitem(sys.modules, registry.__name__, registry)
    # external distribution metadata and Python resolution, not AKLR routing internals
    monkeypatch.setattr(hook, "version", lambda dist: "0.5.23.dev20261120+g28e5763574")
    monkeypatch.setattr(pkgutil, "resolve_name", lambda target: None)
    hook.install()
    assert TARGET in [name for name, _, _ in registered]


def test_a_decode_runs_the_decode_op_against_tilelang(sglang, ops):
    sglang.backend = Backend([1, 1, 1], seq_lens_sum=9000)
    output, q_all, kv_cache, page_table_1 = call(sglang.backend, 3)
    ((kwargs,),) = [ops.decode.calls]
    assert not ops.prefill.calls and not ops.verify.calls
    assert kwargs["native"] == "tilelang" and kwargs["sm_scale"] == 0.0625
    assert kwargs["q"].dtype == torch.float8_e4m3fn
    assert torch.equal(kwargs["q"].float(), q_all.to(torch.float8_e4m3fn).float())
    assert kwargs["kv_cache"].shape == (SLOTS, WIDTH) and kwargs["topk_indices"] is page_table_1
    # [tokens, heads, v_head_dim], as _forward_tilelang returns it on 0.5.23
    assert output.shape == (3, HEADS, LATENT)


@pytest.mark.parametrize(
    "extend_lens, seq_lens_sum",
    [([512, 512], 1024), ([1024], 9216), ([300, 724], 5000)],
    ids=["new_sequences", "onto_a_prefix", "new_and_onto_a_prefix"],
)
def test_an_extend_runs_the_prefill_op_against_tilelang(sglang, ops, extend_lens, seq_lens_sum):
    sglang.backend = Backend(extend_lens, seq_lens_sum)
    output, _, _, page_table_1 = call(sglang.backend, 1024)
    ((kwargs,),) = [ops.prefill.calls]
    assert not ops.decode.calls and not ops.verify.calls
    assert kwargs["native"] == "tilelang" and kwargs["kv_cache"].shape == (SLOTS, WIDTH)
    assert kwargs["topk_indices"] is page_table_1
    assert torch.equal(output, torch.arange(1024.0)[:, None, None].expand(1024, HEADS, LATENT))


def test_an_extend_off_the_measured_sizes_runs_padded_to_the_next(sglang, ops):
    """Prefill is measured every 1024 tokens: 1500 tokens run as 2048, the 548 more reading no slot."""
    sglang.backend = Backend([1500], 1500)
    output, q_all, _, page_table_1 = call(sglang.backend, 1500)
    ((kwargs,),) = [ops.prefill.calls]
    assert kwargs["q"].shape == (2048, HEADS, WIDTH) and kwargs["topk_indices"].shape == (2048, TOPK)
    assert torch.equal(kwargs["q"][:1500].float(), q_all.to(torch.float8_e4m3fn).float())
    assert torch.equal(kwargs["topk_indices"][:1500], page_table_1)
    assert (kwargs["topk_indices"][1500:] == -1).all()
    assert torch.equal(output, torch.arange(1500.0)[:, None, None].expand(1500, HEADS, LATENT))


def test_a_speculative_step_runs_sparse_mla_verify(sglang, ops):
    sglang.backend, sglang.algorithm = Backend([1, 1, 1], 9000), "EAGLE"
    call(sglang.backend, 3)
    assert not ops.decode.calls and not ops.prefill.calls
    assert ops.verify.calls[0]["native"] == "tilelang"


def test_a_decode_context_parallel_rank_stays_with_sglang(sglang, ops):
    """Its pool holds every attn_dcp_size-th slot under global indices."""
    sglang.backend = Backend([1, 1], seq_lens_sum=9000, attn_dcp_size=2)
    output, *_ = call(sglang.backend, 2)
    assert output is FallbackReason.OUT_OF_CONTRACT
    assert not ops.decode.calls and not ops.prefill.calls and not ops.verify.calls


def test_a_declined_decode_goes_back_to_sglang(sglang, ops):
    sglang.backend = Backend([1, 1], seq_lens_sum=9000)
    ops.decode.result = FallbackReason.NO_WORKLOAD
    output, *_ = call(sglang.backend, 2)
    assert output is FallbackReason.NO_WORKLOAD


def test_the_triton_decode_route_reads_attn_dcp_size(sglang, monkeypatch):
    """_forward_triton_decode's route on a backend with attn_dcp_size and no dcp_size."""
    decode = Op()
    monkeypatch.setattr(dsa_api, "dsa_sparse_mla_decode", decode)
    q = torch.zeros(2, HEADS, WIDTH, dtype=torch.float8_e4m3fn)
    kv_cache = torch.zeros(SLOTS, 1, WIDTH, dtype=torch.float8_e4m3fn)
    table = torch.zeros(2, TOPK, dtype=torch.int32)
    args = (q[..., :LATENT], q[..., LATENT:], kv_cache, LATENT, table, 0.0625)
    assert dsa_api._forward_triton_decode(Backend([1, 1], 9000), *args).shape == (2, HEADS, LATENT)
    assert dsa_api._forward_triton_decode(Backend([1, 1], 9000, attn_dcp_size=2), *args) is (
        FallbackReason.OUT_OF_CONTRACT
    )
    assert len(decode.calls) == 1
