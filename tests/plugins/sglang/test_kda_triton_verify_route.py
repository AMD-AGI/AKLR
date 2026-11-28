# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The route of Kimi-K3's KDA chain verify in ReplaySSM ring mode on sglang 0.5.23's KDA backend
(KDAAttnBackend._forward_target_verify): how the backend's projections and pools map onto the op, which calls
stay with sglang and the reason logged for each, the baseline's calls of sglang's own kernels, and the
contract and its reference beside kda_chain_verify_ring's, on stand-ins of sglang 0.5.23."""

import functools
import inspect
import logging
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from aklr.core.contract.fastkey import FastKeyCodec, FastKeyFnGenerator
from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins.sglang.layers.kda_backend_hip.contract import KDA_CHAIN_VERIFY_RING
from aklr.plugins.sglang.layers.kda_backend_hip.reference import KdaChainVerifyRingReference
from aklr.plugins.sglang.layers.kda_triton import api, baseline
from aklr.plugins.sglang.layers.kda_triton.contract import KIMI_K3_SGL_KDA_VERIFY
from tests.op_case import OpCase
from tests.plugins.sglang import cases

HEADS, DIM = 2, 128
CONV = 3 * HEADS * DIM
# two requests' chains of four draft tokens
REQUESTS, STEPS = 2, 4
TOKENS = REQUESTS * STEPS
# sglang's pools: the state slots, the request pool (a window line a request, then the padding requests' line)
# and the rows of a slot's ring
SLOTS, POOL, RING = 7, 5, 8
# a row of sglang's fused in-projection at this head count, [q, k, v | g | f_a | b | pad to 8]: the
# convolution input and beta are column slices of it
SPLIT = (CONV, HEADS * DIM, DIM, HEADS, 6)
ROW = sum(SPLIT)

# sglang 0.5.23's KDAAttnBackend._forward_target_verify (srt/layers/attention/linear/kda_backend.py)
SGLANG_TARGET_VERIFY = [
    (n, inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty)
    for n in ("self", "layer", "forward_batch", "mixed_qkv", "a", "b")
]
# what it passes causal_conv1d_update by keyword, and TritonKDAKernel.target_verify in ring mode
SGLANG_CONV_KEYWORDS = {
    "activation",
    "conv_state_indices",
    "intermediate_conv_window",
    "intermediate_state_indices",
    "retrieve_next_token",
    "retrieve_next_sibling",
    "retrieve_parent_token",
}
SGLANG_VERIFY_KEYWORDS = {
    *("A_log", "dt_bias", "q", "k", "v", "a", "b", "ssm_states", "cache_indices", "query_start_loc"),
    *("intermediate_states_buffer", "intermediate_state_indices", "cache_steps", "retrieve_parent_token"),
    *("lower_bound", "cache_ring", "replayssm_rawv", "replayssm_rawk", "replayssm_g", "replayssm_beta"),
}

FALLS_BACK = "aklr kda verify route falls back: "


def test_the_route_takes_sglangs_parameters():
    parameters = inspect.signature(api._forward_target_verify).parameters.values()
    assert [(p.name, p.kind, p.default) for p in parameters] == SGLANG_TARGET_VERIFY


class TritonKDAKernel:
    """sglang's Triton KDA kernel as the backend's dispatcher holds it for verify: the route knows it by
    name."""


class FlashInferKDAKernel:
    """Another of sglang's verify kernels."""


class Op:
    """The op's api as the route sees it: ``ask(call)`` answers fallback_reason (None: a kernel takes the
    call) and ``run(call)`` is the call's result; both record what they were handed."""

    def __init__(self):
        self.ask = lambda call: None
        self.run = lambda call: torch.zeros(TOKENS, HEADS, DIM, dtype=torch.bfloat16)
        self.asked, self.calls, self.results = [], [], []

    def fallback_reason(self, **call):
        self.asked.append(call)
        return self.ask(call)

    def __call__(self, **call):
        self.calls.append(call)
        self.results.append(self.run(call))
        return self.results[-1]


@functools.cache
def _fast_key():
    codec = FastKeyCodec.from_contract(KIMI_K3_SGL_KDA_VERIFY)
    return codec, FastKeyFnGenerator(codec, torch.device("cpu")).generate()


def point_of(call):
    """The contract's Point of a call the route hands the op, the api's defaults filled in; None outside the
    contract."""
    codec, fast_key = _fast_key()
    key = fast_key({p.name: call.get(p.name, p.default) for p in KIMI_K3_SGL_KDA_VERIFY.api_params})
    return None if key is None else codec.to_point(key)


def randn(gen, *shape):
    return torch.randn(*shape, generator=gen).to(torch.bfloat16)


def messages(caplog):
    return [r.getMessage() for r in caplog.records if r.name == api.__name__]


@pytest.fixture(autouse=True)
def _no_reason_logged_yet(monkeypatch):
    """The route logs each reason once per process: every test starts with none seen."""
    monkeypatch.setattr(api, "_SEEN", set())


@pytest.fixture
def sglang(monkeypatch):
    """sglang 0.5.23's modules the route imports, as stand-ins: off CUDA, PP spec's stable rows off. Returns
    both switches and the calls of select_verify_intermediate_state_indices."""
    state = SimpleNamespace(is_cuda=False, stable_rows=False, selections=[])

    def select_verify_intermediate_state_indices(default_indices, req_pool_indices, valid, pool_size):
        # srt/layers/attention/linear/utils.py
        state.selections.append((default_indices, req_pool_indices, valid, pool_size))
        if not state.stable_rows:
            return default_indices
        req_rows = req_pool_indices[: valid.shape[0]]
        return torch.where(valid, req_rows, torch.full_like(req_rows, pool_size)).to(torch.int32)

    linear = ModuleType("sglang.srt.layers.attention.linear")
    linear.kda_backend = SimpleNamespace(is_cuda=lambda: state.is_cuda)
    linear.utils = SimpleNamespace(
        pp_spec_stable_rows_enabled=lambda: state.stable_rows,
        select_verify_intermediate_state_indices=select_verify_intermediate_state_indices,
    )
    monkeypatch.setitem(sys.modules, linear.__name__, linear)
    return state


@pytest.fixture
def op(monkeypatch):
    """The verify op, a kernel taking every call."""
    op = Op()
    monkeypatch.setattr(api, "kimi_k3_sgl_kda_verify", op)
    return op


def dispatcher(kernel, is_nv_cutedsl=False):
    """The backend's kernel dispatcher, as far as the route reads it."""
    return SimpleNamespace(
        verify_kernel=kernel, verify_backend=SimpleNamespace(is_nv_cutedsl=lambda: is_nv_cutedsl)
    )


def spec_info(draft_token_num=STEPS, ragged_verify_layout=None):
    return SimpleNamespace(draft_token_num=draft_token_num, ragged_verify_layout=ragged_verify_layout)


@pytest.fixture
def verify(sglang):
    """A dense chain verify as the KDA backend's forward_extend hands it over under
    --enable-linear-replayssm-spec: the layer, its cache with the rings and no per-step snapshots, the
    metadata of chains, and the projections, the convolution input and beta column slices of the fused
    in-projection's output, the gate dense. ``run`` calls the route on them as they are then."""
    gen = torch.Generator().manual_seed(3)
    mixed_qkv, _, _, beta, _ = randn(gen, TOKENS, ROW).split(SPLIT, dim=-1)
    layer = SimpleNamespace(
        layer_id=3,
        num_q_heads=HEADS,
        num_k_heads=HEADS,
        num_v_heads=HEADS,
        head_q_dim=DIM,
        head_k_dim=DIM,
        head_v_dim=DIM,
        conv_weights=torch.randn(CONV, 4, generator=gen),
        bias=None,
        A_log=torch.zeros(1, 1, HEADS, 1),
        dt_bias=torch.zeros(HEADS * DIM),
        lower_bound=-5.0,
    )
    cache = SimpleNamespace(
        conv=[torch.zeros(SLOTS, 3, CONV, dtype=torch.bfloat16)],
        temporal=torch.zeros(SLOTS, HEADS, DIM, DIM),
        intermediate_ssm=None,
        intermediate_conv_window=[torch.zeros(POOL + 1, STEPS, 3, CONV, dtype=torch.bfloat16)],
        replayssm_rawv=torch.zeros(SLOTS, HEADS, RING, DIM, dtype=torch.bfloat16),
        replayssm_rawk=torch.zeros(SLOTS, HEADS, RING, DIM, dtype=torch.bfloat16),
        replayssm_g=torch.zeros(SLOTS, HEADS, RING, DIM),
        replayssm_beta=torch.zeros(SLOTS, HEADS, RING),
    )
    metadata = SimpleNamespace(
        mamba_cache_indices=torch.tensor([4, 1], dtype=torch.int32),
        query_start_loc=torch.arange(REQUESTS + 1, dtype=torch.int32) * STEPS,
        retrieve_next_token=None,
        retrieve_next_sibling=None,
        retrieve_parent_token=None,
    )
    backend = SimpleNamespace(
        forward_metadata=metadata,
        kernel_dispatcher=dispatcher(TritonKDAKernel()),
        req_to_token_pool=SimpleNamespace(mamba2_layer_cache={layer.layer_id: cache}.__getitem__, size=POOL),
        verify_intermediate_state_indices=torch.arange(POOL, dtype=torch.int32),
        accept_lens_pool=None,
    )
    batch = SimpleNamespace(
        spec_info=spec_info(),
        req_pool_indices=torch.tensor([3, 0]),
    )
    call = SimpleNamespace(
        sglang=sglang,
        backend=backend,
        layer=layer,
        cache=cache,
        metadata=metadata,
        batch=batch,
        mixed_qkv=mixed_qkv,
        a=randn(gen, TOKENS, HEADS * DIM),
        b=beta,
    )
    call.run = lambda: api._forward_target_verify(
        call.backend, call.layer, call.batch, call.mixed_qkv, call.a, call.b
    )
    return call


# ---- the calls the op takes


def test_a_chain_verify_runs_the_op_on_views_of_sglangs_projections_and_pools(verify, op):
    """The projections' rows apart, as the fused in-projection leaves them: the op reads them where they are
    (strided_rows), and the weights, pools, windows and rings as the layer and its cache hold them; its output
    goes back as sglang's [1, tokens, heads, dim]."""
    out = verify.run()
    (call,) = op.calls
    assert not op.asked
    assert out.shape == (1, TOKENS, HEADS, DIM) and out.data_ptr() == op.results[0].data_ptr()

    qkv, gate, beta = call["mixed_qkv"], call["a"], call["b"]
    assert qkv.shape == (TOKENS, 3 * HEADS, DIM) and qkv.stride() == (ROW, DIM, 1)
    assert gate.shape == (TOKENS, HEADS, DIM) and gate.is_contiguous()
    assert beta.shape == (TOKENS, HEADS, 1) and beta.stride() == (ROW, 1, 1)
    assert (qkv.data_ptr(), gate.data_ptr(), beta.data_ptr()) == (
        verify.mixed_qkv.data_ptr(),
        verify.a.data_ptr(),
        verify.b.data_ptr(),
    )
    layer, cache = verify.layer, verify.cache
    held = {
        "conv_weight": layer.conv_weights,
        "conv_state": cache.conv[0],
        "conv_window": cache.intermediate_conv_window[0],
        "ssm_state": cache.temporal,
        "ring_v": cache.replayssm_rawv,
        "ring_k": cache.replayssm_rawk,
        "ring_g": cache.replayssm_g,
        "ring_beta": cache.replayssm_beta,
    }
    assert all(call[name] is tensor for name, tensor in held.items())
    assert call["a_log"].shape == (HEADS,) and call["a_log"].data_ptr() == layer.A_log.data_ptr()
    assert call["dt_bias"].shape == (HEADS, DIM) and call["dt_bias"].data_ptr() == layer.dt_bias.data_ptr()
    slots, lines = call["cache_indices"], call["window_indices"]
    assert slots.tolist() == [4, 1] and slots.data_ptr() == verify.metadata.mamba_cache_indices.data_ptr()
    assert lines.tolist() == [0, 1]
    assert lines.data_ptr() == verify.backend.verify_intermediate_state_indices.data_ptr()
    # the mask of live slots, which only PP spec's stable rows read, is not computed
    assert verify.sglang.selections == []
    scalars = {n: v for n, v in call.items() if not isinstance(v, torch.Tensor)}
    assert scalars == {"storage": "strided_rows", "scale": DIM**-0.5, "lower_bound": -5.0}

    point = point_of(call)
    assert point.layout_variant_name == "strided_rows" and point.attrs == {"has_lower_bound": True}
    assert point.axes == {
        "n_requests": REQUESTS,
        "n_verify_tokens": STEPS,
        "n_tokens": TOKENS,
        "n_heads": HEADS,
        "n_slots": SLOTS,
        "n_lines": POOL + 1,
        "ring_len": RING,
    }


def test_a_layer_without_a_gate_floor_verifies_on_the_softplus_gate(verify, op):
    """sglang's Triton verify kernel takes the softplus gate at beta 1 and threshold 20
    (TritonKDAKernel.target_verify)."""
    verify.layer.lower_bound = None
    verify.run()
    (call,) = op.calls
    scalars = {n: v for n, v in call.items() if not isinstance(v, torch.Tensor)}
    assert scalars == {
        "storage": "strided_rows",
        "scale": DIM**-0.5,
        "softplus_beta": 1.0,
        "softplus_threshold": 20.0,
    }
    assert point_of(call).attrs == {"has_lower_bound": False}


@pytest.mark.parametrize("extra", [0, 3], ids=["dense", "rows_off_eight_channels"])
def test_dense_projections_go_to_the_op_as_rows_too(verify, op, extra):
    """A dense tensor is rows a row apart: the same layout, the same kernels; so are rows apart by a distance that is
    no multiple of eight channels (the kernel copies those dense)."""
    rows = torch.zeros(TOKENS, CONV + extra, dtype=torch.bfloat16)[:, :CONV]
    verify.mixed_qkv, verify.b = rows.copy_(verify.mixed_qkv), verify.b.contiguous()
    verify.run()
    (call,) = op.calls
    assert not op.asked and call["storage"] == "strided_rows"
    assert (call["mixed_qkv"].data_ptr(), call["b"].data_ptr()) == (
        verify.mixed_qkv.data_ptr(),
        verify.b.data_ptr(),
    )
    assert point_of(call).layout_variant_name == "strided_rows"


def test_under_pp_spec_each_request_takes_its_own_window_line(verify, op):
    """SGLANG_ENABLE_PP_SPEC: sglang's select_verify_intermediate_state_indices, on the mask of live slots,
    gives a live request the line of its row in the request pool and a padding request the line past the
    pool."""
    verify.sglang.stable_rows = True
    verify.metadata.mamba_cache_indices = torch.tensor([4, -1], dtype=torch.int32)
    verify.run()
    (call,) = op.calls
    assert call["window_indices"].tolist() == [3, POOL] and call["window_indices"].dtype == torch.int32
    ((default, rows, live, pool_size),) = verify.sglang.selections
    assert default is verify.backend.verify_intermediate_state_indices
    assert rows is verify.batch.req_pool_indices and live.tolist() == [True, False] and pool_size == POOL
    assert point_of(call) is not None


def test_the_nv_cutedsl_backend_off_cuda_verifies_on_the_triton_kernel(verify, op):
    """sglang's dispatcher holds the Triton kernel for verify under the nv_cutedsl backend, whose CuTe kernel
    takes the call on CUDA only (_can_run_dspark_cutedsl_mtp)."""
    verify.backend.kernel_dispatcher = dispatcher(TritonKDAKernel(), is_nv_cutedsl=True)
    assert verify.run().shape == (1, TOKENS, HEADS, DIM)
    assert len(op.calls) == 1


# ---- the calls that stay with sglang


def changed(**fields):
    """A change of a verify call: each ``holder__field``, or field of the call itself, set to its value."""

    def change(call):
        for path, value in fields.items():
            holder, separator, field = path.partition("__")
            setattr(getattr(call, holder) if separator else call, field if separator else holder, value)

    return change


STAYS_WITH_SGLANG = {
    "no_ring": (changed(cache__replayssm_rawv=None), "no ReplaySSM ring"),
    "snapshots": (changed(cache__intermediate_ssm=torch.zeros(1)), "per-step state snapshots"),
    # sglang without --enable-linear-replayssm-spec
    "snapshot_mode": (
        changed(cache__replayssm_rawv=None, cache__intermediate_ssm=torch.zeros(1)),
        "no ReplaySSM ring, per-step state snapshots",
    ),
    "ragged": (changed(batch__spec_info=spec_info(ragged_verify_layout=object())), "ragged verify layout"),
    "tree_next_token": (changed(metadata__retrieve_next_token=torch.zeros(1)), "tree verify"),
    "tree_next_sibling": (changed(metadata__retrieve_next_sibling=torch.zeros(1)), "tree verify"),
    "tree_parent_token": (changed(metadata__retrieve_parent_token=torch.zeros(1)), "tree verify"),
    "conv_bias": (changed(layer__bias=torch.zeros(CONV)), "convolution bias"),
    "fused_accept_state": (changed(backend__accept_lens_pool=torch.ones(POOL + 1)), "fused accept state"),
    "other_verify_kernel": (
        changed(backend__kernel_dispatcher=dispatcher(FlashInferKDAKernel())),
        "verify kernel",
    ),
    "nv_cutedsl_on_cuda": (
        changed(sglang__is_cuda=True, backend__kernel_dispatcher=dispatcher(TritonKDAKernel(), True)),
        "verify kernel",
    ),
    "broken_chains": (changed(batch__spec_info=spec_info(draft_token_num=3)), "8 tokens in chains of 3"),
    "more_query_heads": (changed(layer__num_q_heads=2 * HEADS), "head counts"),
    "more_key_heads": (changed(layer__num_k_heads=2 * HEADS), "head counts"),
    "narrow_query_heads": (changed(layer__head_q_dim=DIM // 2), "head sizes"),
    "narrow_value_heads": (changed(layer__head_v_dim=DIM // 2), "head sizes"),
    "narrow_convolution_input": (
        changed(mixed_qkv=torch.zeros(TOKENS, CONV - DIM, dtype=torch.bfloat16)),
        f"convolution input of ({TOKENS}, {CONV - DIM})",
    ),
    "short_gate": (
        changed(a=torch.zeros(TOKENS, HEADS * DIM - 1, dtype=torch.bfloat16)),
        f"gate of ({TOKENS}, {HEADS * DIM - 1})",
    ),
    "wide_beta": (
        changed(b=torch.zeros(TOKENS, HEADS + 1, dtype=torch.bfloat16)),
        f"beta of ({TOKENS}, {HEADS + 1})",
    ),
    "convolution_channels_apart": (
        changed(mixed_qkv=torch.zeros(TOKENS, 2 * CONV, dtype=torch.bfloat16)[:, ::2]),
        "convolution channels not dense",
    ),
    "gate_rows_apart": (
        changed(a=torch.zeros(TOKENS, HEADS * DIM + 8, dtype=torch.bfloat16)[:, : HEADS * DIM]),
        "gate not dense",
    ),
    "beta_heads_apart": (
        changed(b=torch.zeros(TOKENS, 2 * HEADS, dtype=torch.bfloat16)[:, ::2]),
        "beta's heads not dense",
    ),
    "wide_a_log": (changed(layer__A_log=torch.zeros(1, 1, HEADS + 1, 1)), "A_log"),
    "short_dt_bias": (changed(layer__dt_bias=torch.zeros(HEADS * DIM - 1)), "dt_bias"),
}


@pytest.mark.parametrize(("change", "reason"), STAYS_WITH_SGLANG.values(), ids=list(STAYS_WITH_SGLANG))
def test_other_verifies_stay_with_sglang_and_log_why_once(verify, op, caplog, change, reason):
    change(verify)
    with caplog.at_level(logging.WARNING, logger=api.__name__):
        results = [verify.run() for _ in range(2)]
    assert results == [FallbackReason.OUT_OF_CONTRACT] * 2
    assert not op.asked and not op.calls
    assert messages(caplog) == [FALLS_BACK + reason]


def test_a_verify_the_op_declines_goes_back_to_sglang_with_its_operands_logged_once(verify, op, caplog):
    """The op's own answer to the call (no kernel at its Point, or a Point first met inside a graph capture)
    goes back to sglang as it is, logged once with every tensor's shape and dtype."""
    op.run = lambda call: FallbackReason.NOT_WARMED
    with caplog.at_level(logging.WARNING, logger=api.__name__):
        results = [verify.run() for _ in range(2)]
    assert results == [FallbackReason.NOT_WARMED] * 2
    (message,) = messages(caplog)
    assert message.startswith(
        f"{FALLS_BACK}not_warmed: mixed_qkv ({TOKENS}, {3 * HEADS}, {DIM}) bfloat16, "
        f"a ({TOKENS}, {HEADS}, {DIM}) bfloat16, b ({TOKENS}, {HEADS}, 1) bfloat16, "
        f"conv_weight ({CONV}, 4) float32, "
    )
    assert message.endswith(f"ring_beta ({SLOTS}, {HEADS}, {RING}) float32")


# ---- the baseline calls sglang's own kernels as its backend does


@pytest.fixture
def sglang_kernels(monkeypatch):
    """Stand-ins of sglang 0.5.23's causal_conv1d_update and TritonKDAKernel.target_verify, recording their
    calls: each element of the convolution's output [requests, channels, steps] is its head's index (the query
    heads, then the key heads, then the value heads) plus a sixteenth of its token's, the verify kernel's
    output [1, tokens, heads, dim] sevens."""
    calls = {}

    def causal_conv1d_update(x, conv_state, weight, bias=None, **kwargs):
        calls["conv"] = (x, conv_state, weight, bias, kwargs)
        requests, channels, steps = x.shape
        heads = (torch.arange(channels) // DIM).view(1, channels, 1)
        tokens = torch.arange(requests * steps).view(requests, 1, steps)
        return (heads + tokens / 16).to(x.dtype)

    class TritonKDAKernel:
        def target_verify(self, **kwargs):
            calls["verify"] = kwargs
            return torch.full(kwargs["v"].shape, 7.0, dtype=kwargs["v"].dtype)

    modules = {
        "sglang.kernels.ops.mamba.causal_conv1d_triton": {"causal_conv1d_update": causal_conv1d_update},
        "sglang.srt.layers.attention.linear.kernels.kda_triton": {"TritonKDAKernel": TritonKDAKernel},
    }
    for name, attrs in modules.items():
        stand_in = ModuleType(name)
        stand_in.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, stand_in)
    # the baseline keeps one kernel object a process
    baseline._triton_kernel.cache_clear()
    yield calls
    baseline._triton_kernel.cache_clear()


def same_view(passed, sglangs):
    """Whether ``passed`` is the view sglang's backend passes: its bytes, shape and strides."""
    return (passed.data_ptr(), passed.shape, passed.stride()) == (
        sglangs.data_ptr(),
        sglangs.shape,
        sglangs.stride(),
    )


@pytest.mark.parametrize("lower_bound", [-5.0, None], ids=["lower_bound", "softplus"])
def test_the_verify_baseline_calls_sglangs_kernels_as_its_backend_does(
    verify, op, sglang_kernels, lower_bound
):
    """On the call the route hands the op, the baseline's two calls are _forward_target_verify's of sglang
    0.5.23 on the backend's own operands: the convolution of the chains with the windows, then the Triton
    verify kernel on the convolved rows' q, k and v, from the slots' states, writing the rings."""
    verify.layer.lower_bound = lower_bound
    verify.run()
    (args,) = op.calls
    out = baseline.KimiK3SglKdaVerifyBaseline().baseline(**args)["output"]
    layer, cache, metadata = verify.layer, verify.cache, verify.metadata
    lines = verify.backend.verify_intermediate_state_indices[:REQUESTS]

    x, conv_state, weight, bias, options = sglang_kernels["conv"]
    assert same_view(x, verify.mixed_qkv.view(REQUESTS, STEPS, -1).transpose(1, 2))
    assert same_view(conv_state, cache.conv[0].transpose(-1, -2))
    assert weight is layer.conv_weights and bias is None
    assert options.keys() == SGLANG_CONV_KEYWORDS
    assert same_view(options["intermediate_conv_window"], cache.intermediate_conv_window[0].transpose(-1, -2))
    assert same_view(options["conv_state_indices"], metadata.mamba_cache_indices[:REQUESTS])
    assert same_view(options["intermediate_state_indices"], lines)
    assert {n: v for n, v in options.items() if not isinstance(v, torch.Tensor)} == {
        "activation": "silu",
        "retrieve_next_token": None,
        "retrieve_next_sibling": None,
        "retrieve_parent_token": None,
    }

    call = sglang_kernels["verify"]
    assert call.keys() == SGLANG_VERIFY_KEYWORDS
    # q, k and v: the [1, tokens, heads, dim] views of the convolved rows, split in three
    convolved = torch.arange(3 * HEADS).view(1, -1, 1) + torch.arange(TOKENS).view(-1, 1, 1) / 16
    for name, heads in zip("qkv", convolved.split(HEADS, dim=1), strict=True):
        assert call[name].shape == (1, TOKENS, HEADS, DIM)
        assert torch.equal(call[name][0].float(), heads.expand(TOKENS, HEADS, DIM))
    step = HEADS * DIM * call["q"].element_size()
    assert (call["k"].data_ptr() - call["q"].data_ptr(), call["v"].data_ptr() - call["q"].data_ptr()) == (
        step,
        2 * step,
    )
    # the gate and beta where sglang's are, a token's a row apart as the kernel steps them
    assert call["a"].data_ptr() == verify.a.data_ptr() and call["a"].stride(1) == verify.a.stride(0)
    assert call["b"].data_ptr() == verify.b.data_ptr() and call["b"].stride()[-2:] == verify.b.stride()
    assert same_view(call["A_log"], layer.A_log) and same_view(call["dt_bias"], layer.dt_bias)
    assert call["ssm_states"] is cache.temporal
    assert same_view(call["cache_indices"], metadata.mamba_cache_indices)
    assert same_view(call["intermediate_state_indices"], lines)
    assert torch.equal(call["query_start_loc"], metadata.query_start_loc)
    assert call["query_start_loc"].dtype == metadata.query_start_loc.dtype
    rings = {
        "replayssm_rawv": cache.replayssm_rawv,
        "replayssm_rawk": cache.replayssm_rawk,
        "replayssm_g": cache.replayssm_g,
        "replayssm_beta": cache.replayssm_beta,
    }
    assert all(call[name] is ring for name, ring in rings.items())
    assert {n: v for n, v in call.items() if not isinstance(v, torch.Tensor)} == {
        "intermediate_states_buffer": None,
        "cache_steps": STEPS,
        "retrieve_parent_token": None,
        "lower_bound": lower_bound,
        "cache_ring": True,
    }
    assert out.shape == (TOKENS, HEADS, DIM) and bool((out == 7.0).all())


# ---- the contract, the reference and the small Points


def test_the_contract_is_the_ring_verifys_on_sglangs_projection_rows():
    """Derived from kda_chain_verify_ring: its axes, attrs, constraints, bounds and its other operands, with
    the convolution input a row of 128 channels a head and beta a row a token and head, which alone may lie
    apart (strided_rows), and a description that says so."""
    ring, ours = KDA_CHAIN_VERIFY_RING, KIMI_K3_SGL_KDA_VERIFY
    assert ours.inputs["mixed_qkv"].shape == ("n_tokens", "conv_heads", 128)
    assert ours.inputs["b"].shape == ("n_tokens", "n_heads", 1)
    assert list(ours.inputs) == list(ring.inputs)
    assert all(ours.inputs[n] == ring.inputs[n] for n in ring.inputs if n not in ("mixed_qkv", "b"))
    assert (ours.outputs, ours.attrs, ours.constraints) == (ring.outputs, ring.attrs, ring.constraints)
    assert {n: a for n, a in ours.axes.items() if n != "conv_heads"} == dict(ring.axes)
    assert ours.layout_variants.names() == ("row_major", "strided_rows")
    strided = ours.layout_variants.by_name["strided_rows"]
    assert [n for n in ours.tensors() if strided.layout_of(n).kind == "strided_rows"] == ["mixed_qkv", "b"]
    assert ours.description.startswith(
        "Kimi-K3's KDA chain verify in ReplaySSM ring mode, as sglang 0.5.23's KDA backend runs a target "
        "verify of dense chains on its Triton verify kernel."
    )
    assert "mixed_qkv[t]'s heads end to end, its beta input b[t, :, 0]" in ours.description


VERIFY_POINTS = [p for p in cases.SMALL_POINTS if p.contract_id == KIMI_K3_SGL_KDA_VERIFY.contract_id]
GATES = ["lower_bound" if p.attrs["has_lower_bound"] else "softplus" for p in VERIFY_POINTS]


@pytest.mark.parametrize("point", VERIFY_POINTS, ids=GATES)
def test_the_reference_is_the_ring_verifys_on_the_projections_rows(point):
    """kda_chain_verify_ring's reference on the convolution input as (tokens, 3 heads x dim) rows and beta as
    (tokens, heads): the same output, windows and rings, bit for bit."""
    case = OpCase(point)
    args = case.inputs()
    assert args["mixed_qkv"].dim() == 3 and args["b"].shape[2:] == (1,)
    outputs = case.reference_outputs(args)
    rings = KdaChainVerifyRingReference().reference(
        **args | {"mixed_qkv": args["mixed_qkv"].flatten(1), "b": args["b"].flatten(1)}
    )
    assert outputs.keys() == rings.keys() == {"output", *KIMI_K3_SGL_KDA_VERIFY.mutable_inputs}
    for name, expected in rings.items():
        assert torch.equal(outputs[name], expected), name


@pytest.mark.parametrize("point", VERIFY_POINTS, ids=GATES)
def test_the_small_points_draw_no_padding_request(point):
    """sglang's verify kernel leaves a padding request's output rows unwritten, which the GPU baseline test
    (test_backend_matches_reference) would compare with the reference's zeros."""
    assert bool((OpCase(point).inputs()["cache_indices"] >= 0).all())
