# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The routes of Kimi-K3's KDA prefill and decode on sglang's Triton backend (TritonKDAKernel.extend and .decode):
how the KDA backend's calls map onto the ops, which calls stay with sglang, and the baselines' calls of sglang's own
kernels, on stand-ins of sglang 0.5.22 and 0.5.23 (whose two methods take the same parameters)."""

import inspect
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins.sglang.layers.kda_triton import api, baseline
from aklr.plugins.sglang.layers.kda_triton.reference import (
    KimiK3SglKdaDecodeReference,
    KimiK3SglKdaPrefillReference,
    KimiK3SglKdaPrefillTrackReference,
)

HEADS, DIM, SLOTS = 2, 128, 5
# a row of sglang's [f_a | b] projection at this head count (f_a, b, the pad to 8): beta is a column slice of it
BFA = DIM + HEADS + 6

# sglang 0.5.22's and 0.5.23's TritonKDAKernel.extend and .decode (srt/layers/attention/linear/kernels/kda_triton.py)
P, K, V = (
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
    inspect.Parameter.KEYWORD_ONLY,
    inspect.Parameter.VAR_KEYWORD,
)
EMPTY = inspect.Parameter.empty
SGLANG_EXTEND = [
    *[(n, P, EMPTY) for n in ("self", "q", "k", "v", "g", "beta")],
    *[(n, K, EMPTY) for n in ("ssm_states", "cache_indices", "query_start_loc")],
    ("A_log", K, None),
    ("dt_bias", K, None),
    ("lower_bound", K, None),
    ("beta_is_raw", K, False),
    ("return_intermediate_states", K, False),
    ("kwargs", V, EMPTY),
]
SGLANG_DECODE = [
    *[(n, P, EMPTY) for n in ("self", "q", "k", "v", "a", "b")],
    *[(n, K, EMPTY) for n in ("A_log", "dt_bias", "ssm_states", "cache_indices", "query_start_loc")],
    ("lower_bound", K, None),
    ("kwargs", V, EMPTY),
]


# the route's check of sglang's own HIP prefill, before the autouse fixture stands it in
SGLANG_HIP_PREFILL = api._sglang_hip_prefill


def parameters(fn):
    return [(p.name, p.kind, p.default) for p in inspect.signature(fn).parameters.values()]


@pytest.mark.parametrize(("route", "sglangs"), [(api._extend, SGLANG_EXTEND), (api._decode, SGLANG_DECODE)])
def test_the_routes_take_sglangs_parameters(route, sglangs):
    assert parameters(route) == sglangs


@pytest.fixture(autouse=True)
def _no_capture_and_no_hip_prefill(monkeypatch):
    """A CPU torch: no graph capture; and sglang without its own HIP prefill (0.5.22)."""
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(api, "_sglang_hip_prefill", lambda: False)


def randn(gen, *shape, scale=1.0):
    return (torch.randn(*shape, generator=gen) * scale).to(torch.bfloat16)


def layer_operands(gen, n):
    """What the KDA backend hands the kernel for n tokens: q, k and v [1, n, heads, dim] split from the convolution's
    output rows, beta [1, n, heads] a column slice of the [f_a | b] projection, the layer's A_log [1, 1, heads, 1] and
    dt_bias, and the state pool [slots, heads, v, k]."""
    conv = randn(gen, n, 3 * HEADS * DIM)
    q, k, v = (x.unflatten(-1, (HEADS, DIM)).unsqueeze(0) for x in conv.split(HEADS * DIM, dim=-1))
    bfa = randn(gen, n, BFA)
    return conv, dict(
        q=q,
        k=k,
        v=v,
        b=bfa[:, DIM : DIM + HEADS].unsqueeze(0),
        A_log=torch.rand(1, 1, HEADS, 1, generator=gen).add(1).log(),
        dt_bias=torch.randn(HEADS * DIM, generator=gen),
        ssm_states=torch.randn(SLOTS, HEADS, DIM, DIM, generator=gen),
        lower_bound=-5.0,
    )


def sglang_steps(call, gate, spans, slots):
    """sglang's recurrent kernel, written from fused_sigmoid_gating_recurrent.py apart from the contracts' reference:
    per sequence and head the state h[k, v] (the pool holds it v by v), from the slot (zero where it is -1); per token
    q and k times 1 / sqrt(sum x**2 + 1e-6), q times dim ** -0.5, g = lower_bound * sigmoid(exp(A_log) * (a +
    dt_bias)), h *= exp(g)[:, None], v -= sum_k h k, v *= sigmoid(b), h += k v, o = sum_k h q; the last h back to
    the slot. Returns (out, pool, {(sequence, tokens done): h} of every state the walk passes)."""
    q, k, v = (call[n][0].float() for n in "qkv")
    a_log = call["A_log"].reshape(HEADS).float()
    dt_bias = call["dt_bias"].view(HEADS, DIM).float()
    beta = call["b"][0].float()
    pool = call["ssm_states"].clone()
    out = torch.empty_like(v)
    seen = {}
    for s, ((start, end), slot) in enumerate(zip(spans, slots, strict=True)):
        h = [pool[slot, i].T.clone() if slot >= 0 else torch.zeros(DIM, DIM) for i in range(HEADS)]
        seen[s, 0] = torch.stack([x.T for x in h])
        for t in range(start, end):
            for i in range(HEADS):
                qt = q[t, i] * (1 / torch.sqrt((q[t, i] ** 2).sum() + 1e-6)) * DIM**-0.5
                kt = k[t, i] * (1 / torch.sqrt((k[t, i] ** 2).sum() + 1e-6))
                g = call["lower_bound"] * torch.sigmoid(a_log[i].exp() * (gate[t, i] + dt_bias[i]))
                h[i] = h[i] * g.exp()[:, None]
                vt = (v[t, i] - (h[i] * kt[:, None]).sum(0)) * torch.sigmoid(beta[t, i])
                h[i] = h[i] + kt[:, None] * vt[None, :]
                out[t, i] = (h[i] * qt[:, None]).sum(0)
            seen[s, t + 1 - start] = torch.stack([x.T for x in h])
        if slot >= 0:
            pool[slot] = torch.stack([x.T for x in h])
    return out, pool, seen


class Op:
    """An op as the route sees it: takes every call (``reason`` None) or declines it, and runs ``run``."""

    def __init__(self, run=None, reason=None):
        self.run, self.reason, self.calls = run, reason, []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.reason is not None:
            return self.reason
        return self.run(*args, **kwargs)


def reference_op(reference, names):
    """The op as its reference computes it, writing the mutable inputs in place as a kernel does."""

    def run(*args, storage="row_major", scale=DIM**-0.5, lower_bound=-5.0, out_dtype=None, out=None):
        call = dict(zip(names, args, strict=True))
        results = reference.reference(**call, scale=scale, lower_bound=lower_bound)
        for name in ("state_cache", "track_state"):
            if name in results:
                call[name].copy_(results[name])
        return results["out"]

    return run


COMMON = ("qkv", "raw_g", "raw_beta", "a_log", "g_bias", "state_cache", "state_indices")


# ---- decode


def decode_call(indices):
    gen = torch.Generator().manual_seed(0)
    n = len(indices)
    conv, call = layer_operands(gen, n)
    call.update(
        a=randn(gen, n, HEADS * DIM, scale=DIM**-0.5),  # the decode gate, flat
        cache_indices=torch.tensor(indices, dtype=torch.int32),
        query_start_loc=torch.arange(n + 1, dtype=torch.int32),
    )
    return conv, call


@pytest.fixture
def decode_op(monkeypatch):
    op = Op(reference_op(KimiK3SglKdaDecodeReference(), COMMON))
    monkeypatch.setattr(api, "kimi_k3_sgl_kda_decode", op)
    return op


@pytest.mark.parametrize(
    "indices", [[3, 1, 4], [2, 0, -1, -1]], ids=["slots", "padding_slot_and_padded_graph_rows"]
)
def test_a_decode_runs_the_op_on_the_views_of_sglangs_operands(decode_op, indices):
    conv, call = decode_call(indices)
    pool = call["ssm_states"]
    gate = call["a"].float().view(len(indices), HEADS, DIM)
    spans = [(t, t + 1) for t in range(len(indices))]
    want_out, want_pool, _ = sglang_steps(call, gate, spans, indices)
    out = api._decode(None, **call)
    assert out.shape == (1, len(indices), HEADS, DIM) and out.dtype == torch.bfloat16
    torch.testing.assert_close(out[0].float(), want_out, atol=1e-3, rtol=1e-2)
    # every slot as sglang leaves it: the named ones stepped, -1 none, the others unchanged
    torch.testing.assert_close(pool, want_pool, atol=1e-4, rtol=1e-4)
    ((args, kwargs),) = decode_op.calls
    qkv, _, raw_beta, _, _, state_cache, _ = args
    # views, not copies: the convolution rows, beta's column of the projection, the pool
    assert qkv.data_ptr() == conv.data_ptr() and qkv.shape == (len(indices), 3 * HEADS, DIM)
    assert raw_beta.shape == (len(indices), HEADS, 1) and raw_beta.stride(0) == BFA
    assert state_cache.data_ptr() == pool.data_ptr() and state_cache.shape == (SLOTS, HEADS * DIM, DIM)
    assert kwargs == {"storage": "strided_beta", "scale": DIM**-0.5, "lower_bound": -5.0}


def test_a_dense_beta_is_row_major(decode_op):
    _, call = decode_call([1, 2])
    call["b"] = call["b"].contiguous()
    api._decode(None, **call)
    assert decode_op.calls[0][1]["storage"] == "row_major"


def unpacked_qkv(call):
    call.update(q=call["q"].clone(), k=call["k"].clone(), v=call["v"].clone())


@pytest.mark.parametrize(
    "change",
    [
        lambda c: c.update(lower_bound=None),  # the softplus gate (Kimi-Linear)
        lambda c: c.update(extra=True),  # an option sglang's KDA backend does not pass
        unpacked_qkv,
        lambda c: c.update(q=c["q"].half(), k=c["k"].half(), v=c["v"].half()),
        lambda c: c.update(a=c["a"].t().contiguous().t()),
        lambda c: c.update(b=c["b"].float()),
        lambda c: c.update(ssm_states=c["ssm_states"].to(torch.bfloat16)),
        lambda c: c.update(ssm_states=c["ssm_states"].transpose(-1, -2)),
        lambda c: c.update(cache_indices=c["cache_indices"].long()),
        lambda c: c.update(query_start_loc=torch.tensor([0, 2], dtype=torch.int32)),  # not a token a sequence
    ],
    ids=[
        "no_lower_bound",
        "option",
        "unpacked",
        "fp16",
        "gate_strided",
        "beta_fp32",
        "bf16_state",
        "state_k_major",
        "int64_slots",
        "not_one_token_each",
    ],  # fmt: skip
)
def test_other_decodes_stay_with_sglang(monkeypatch, change):
    monkeypatch.setattr(api, "kimi_k3_sgl_kda_decode", Op(lambda *a, **kw: pytest.fail("the op ran")))
    _, call = decode_call([1, 2])
    change(call)
    assert api._decode(None, **call) is FallbackReason.OUT_OF_CONTRACT


def test_a_declined_decode_goes_back_to_sglang(monkeypatch):
    monkeypatch.setattr(api, "kimi_k3_sgl_kda_decode", Op(reason=FallbackReason.NO_WORKLOAD))
    _, call = decode_call([1, 2])
    assert api._decode(None, **call) is FallbackReason.NO_WORKLOAD


# ---- prefill


def prefill_call(seqlens, indices, **kwargs):
    """A forward_extend call: the gate [1, n, heads, dim], raw beta, the lengths on the host."""
    gen = torch.Generator().manual_seed(1)
    n = sum(seqlens)
    conv, call = layer_operands(gen, n)
    bounds = [0]
    for length in seqlens:
        bounds.append(bounds[-1] + length)
    call.update(
        g=randn(gen, 1, n, HEADS, DIM, scale=DIM**-0.5),
        beta=call.pop("b"),
        cache_indices=torch.tensor(indices, dtype=torch.int32),
        query_start_loc=torch.tensor(bounds, dtype=torch.int32),
        beta_is_raw=True,
        extend_seq_lens_cpu=list(seqlens),
        extend_prefix_lens=torch.zeros(len(seqlens), dtype=torch.int64),
        layer_id=3,
        is_spec_decode=False,
        return_intermediate_states=False,
        track_ssm_h_src=None,
        track_state=None,
        track_chunk_idx=None,
        state_checkpoint_cu_starts=None,
        num_state_checkpoints=None,
        state_checkpoint_every_n_tokens=None,
        state_checkpoint_indices=None,
        track_ssm_h_batch_src=None,
    )
    call.update(kwargs)
    spans = list(zip(bounds[:-1], bounds[1:], strict=True))
    return conv, call, spans


def expected_prefill(call, spans):
    sglangs = dict(call, b=call["beta"])
    return sglang_steps(sglangs, call["g"][0].float(), spans, call["cache_indices"].tolist())


@pytest.fixture
def prefill_ops(monkeypatch):
    plain = Op(reference_op(KimiK3SglKdaPrefillReference(), (*COMMON, "cu_seqlens")))
    track = Op(
        reference_op(
            KimiK3SglKdaPrefillTrackReference(), (*COMMON, "cu_seqlens", "track_chunk_idx", "track_state")
        )
    )
    monkeypatch.setattr(api, "kimi_k3_sgl_kda_prefill", plain)
    monkeypatch.setattr(api, "kimi_k3_sgl_kda_prefill_track", track)
    return plain, track


@pytest.mark.parametrize("indices", [[2, 0], [4, -1]], ids=["slots", "padded_row"])
def test_a_prefill_runs_the_op_on_the_views_of_sglangs_operands(prefill_ops, indices):
    conv, call, spans = prefill_call((5, 3), indices)
    pool = call["ssm_states"]
    want_out, want_pool, _ = expected_prefill(call, spans)
    out = api._extend(None, **call)
    assert out.shape == (1, 8, HEADS, DIM) and out.dtype == torch.bfloat16
    torch.testing.assert_close(out[0].float(), want_out, atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(pool, want_pool, atol=1e-4, rtol=1e-4)
    plain, track = prefill_ops
    ((args, kwargs),) = plain.calls
    assert not track.calls
    qkv, raw_g, raw_beta, a_log, g_bias, state_cache, state_indices, cu_seqlens = args
    assert qkv.data_ptr() == conv.data_ptr() and raw_g.data_ptr() == call["g"].data_ptr()
    assert raw_beta.stride(0) == BFA and state_cache.data_ptr() == pool.data_ptr()
    assert a_log.shape == (HEADS,) and g_bias.shape == (HEADS, DIM)
    assert state_indices is call["cache_indices"] and cu_seqlens is call["query_start_loc"]
    assert kwargs == {"storage": "strided_beta", "scale": DIM**-0.5, "lower_bound": -5.0}


# the snapshot's chunk, as the backend sizes it: 64 tokens
LONG = (70, 66)


@pytest.mark.parametrize(
    "chunks",
    [[1, -1], [-1, 1], [1, 1], [0, -1], [2, 5]],
    ids=["first", "second", "both", "at_the_start", "past_the_end"],
)
def test_a_prefill_with_snapshots_writes_them_into_track_state(prefill_ops, chunks):
    sentinel = torch.full((2, HEADS, DIM, DIM), 7.0)
    _, call, spans = prefill_call(
        LONG,
        [2, 3],
        return_intermediate_states=True,
        track_state=sentinel.clone(),
        track_chunk_idx=torch.tensor(chunks, dtype=torch.int32),
        track_ssm_h_src=torch.tensor([0]),
    )
    want_out, want_pool, seen = expected_prefill(call, spans)
    out, h = api._extend(None, **call)
    assert h is None  # the backend reads track_state, never h, when it passes one
    torch.testing.assert_close(out[0].float(), want_out, atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(call["ssm_states"], want_pool, atol=1e-4, rtol=1e-4)
    for s, c in enumerate(chunks):
        if 0 <= c and 64 * c < LONG[s]:
            torch.testing.assert_close(call["track_state"][s], seen[s, 64 * c], atol=1e-4, rtol=1e-4)
        else:
            assert torch.equal(call["track_state"][s], sentinel[s])
    plain, track = prefill_ops
    assert not plain.calls and len(track.calls) == 1


def test_aligned_only_tracking_serves_the_plain_prefill(prefill_ops):
    """Every tracked row chunk-aligned: the backend reads the final state from the pool, no snapshot, no h."""
    _, call, _ = prefill_call(
        (5, 3), [2, 0], return_intermediate_states=True, track_ssm_h_src=torch.tensor([])
    )
    out, h = api._extend(None, **call)
    assert h is None and out.shape == (1, 8, HEADS, DIM)
    plain, track = prefill_ops
    assert len(plain.calls) == 1 and not track.calls


@pytest.mark.parametrize(
    "change",
    [
        {"lower_bound": None},
        {"A_log": None},
        {"beta_is_raw": False},
        {"is_spec_decode": True},
        {"chunk_indices": torch.zeros(1, 2, dtype=torch.int32)},  # a breakable prefill graph's padded tables
        {"return_intermediate_states": True, "track_ssm_h_src": torch.tensor([3])},  # wants h
        {"return_intermediate_states": True},  # wants h: sglang's track_ssm_h_src unknown
        {"track_state": torch.zeros(2, HEADS, DIM, DIM)},  # a snapshot without its chunks
        {"extend_seq_lens_cpu": None},
        {"extend_seq_lens_cpu": [5, 4]},  # rows past the sequences
        {"cache_indices": torch.tensor([2, 0])},  # int64
        {"query_start_loc": torch.tensor([0, 8], dtype=torch.int32)},
    ],
    ids=[
        "no_lower_bound",
        "no_A_log",
        "beta_activated",
        "spec_decode",
        "graph_tables",
        "wants_h",
        "wants_h_unknown",
        "snapshot_without_chunks",
        "no_lengths",
        "padded_rows",
        "int64_slots",
        "offsets",
    ],  # fmt: skip
)
def test_other_prefills_stay_with_sglang(monkeypatch, change):
    monkeypatch.setattr(api, "kimi_k3_sgl_kda_prefill", Op(lambda *a, **kw: pytest.fail("the op ran")))
    monkeypatch.setattr(api, "kimi_k3_sgl_kda_prefill_track", Op(lambda *a, **kw: pytest.fail("the op ran")))
    _, call, _ = prefill_call((5, 3), [2, 0])
    call.update(change)
    assert api._extend(None, **call) is FallbackReason.OUT_OF_CONTRACT


def test_a_prefill_in_a_graph_capture_stays_with_sglang(monkeypatch):
    monkeypatch.setattr(api, "kimi_k3_sgl_kda_prefill", Op(lambda *a, **kw: pytest.fail("the op ran")))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    _, call, _ = prefill_call((5, 3), [2, 0])
    call["q"] = SimpleNamespace(device=torch.device("cuda"))  # a capture happens on the device
    assert api._extend(None, **call) is FallbackReason.OUT_OF_CONTRACT


def test_a_declined_prefill_goes_back_to_sglang(monkeypatch):
    monkeypatch.setattr(api, "kimi_k3_sgl_kda_prefill", Op(reason=FallbackReason.NO_WORKLOAD))
    _, call, _ = prefill_call((5, 3), [2, 0])
    assert api._extend(None, **call) is FallbackReason.NO_WORKLOAD


@pytest.mark.parametrize(
    ("hip", "takes"),
    [(None, False), (False, False), (True, True)],
    ids=["0.5.22_none", "0.5.23_off", "0.5.23_on_gfx950"],
)
def test_sglangs_own_hip_prefill_keeps_its_calls(monkeypatch, hip, takes):
    """sglang 0.5.23 tries its HIP prefill (kda_gfx950_prefill) before chunk_kda on gfx950: those calls stay with it;
    0.5.22 has none."""
    module = ModuleType("sglang.srt.layers.attention.linear.kernels.kda_triton")
    if hip is not None:
        module.kda_gfx950_prefill = SimpleNamespace(available=lambda: hip)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(api, "_sglang_hip_prefill", SGLANG_HIP_PREFILL)  # the route's own check, afresh
    monkeypatch.setattr(api, "kimi_k3_sgl_kda_prefill", Op(lambda *a, **kw: torch.zeros(8, HEADS, DIM)))
    SGLANG_HIP_PREFILL.cache_clear()
    try:
        _, call, _ = prefill_call((5, 3), [2, 0])
        assert (api._extend(None, **call) is FallbackReason.OUT_OF_CONTRACT) is takes
    finally:
        SGLANG_HIP_PREFILL.cache_clear()


# ---- the baselines call sglang's own kernels as its backend does


@pytest.fixture
def sglang_kernels(monkeypatch):
    """Stand-ins of sglang's chunk_kda and fused_sigmoid_gating_delta_rule_update (the recurrent kernel at both its
    0.5.22 and 0.5.23 paths: the baseline tries 0.5.23's first, so with sglang installed a stand-in at 0.5.22's path
    alone leaves the real Triton kernel to run on these CPU tensors), each recording its call and returning an output
    of sglang's shape."""
    calls = {}

    def chunk_kda(**kwargs):
        calls["chunk_kda"] = kwargs
        o = torch.zeros(kwargs["v"].shape)
        return (o, None) if kwargs["output_intermediate_states"] else o

    def fused_sigmoid_gating_delta_rule_update(**kwargs):
        calls["recurrent"] = kwargs
        return torch.zeros(kwargs["v"].shape)

    for name, attrs in {
        "sglang.kernels.ops.attention.fla.kda": {"chunk_kda": chunk_kda},
        "sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent": {
            "fused_sigmoid_gating_delta_rule_update": fused_sigmoid_gating_delta_rule_update
        },
        "sglang.kernels.ops.attention.linear.fused_sigmoid_gating_recurrent": {
            "fused_sigmoid_gating_delta_rule_update": fused_sigmoid_gating_delta_rule_update
        },
    }.items():
        stand_in = ModuleType(name)
        stand_in.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, stand_in)
    return calls


def op_operands(n, n_seqs=None):
    gen = torch.Generator().manual_seed(2)
    bfa = randn(gen, n, BFA)
    return dict(
        qkv=randn(gen, n, 3 * HEADS, DIM),
        raw_g=randn(gen, n, HEADS, DIM),
        raw_beta=bfa[:, DIM : DIM + HEADS].unsqueeze(-1),
        a_log=torch.zeros(HEADS),
        g_bias=torch.zeros(HEADS, DIM),
        state_cache=torch.zeros(SLOTS, HEADS * DIM, DIM),
        state_indices=torch.arange(n if n_seqs is None else n_seqs, dtype=torch.int32),
    )


def sglang_views(kwargs, args):
    """The call's q, k, v, beta and pool are views of the op's operands, as the KDA backend's are of its rows."""
    heads_dim = HEADS * DIM * 2
    assert kwargs["q"].data_ptr() == args["qkv"].data_ptr()
    assert (kwargs["k"].data_ptr(), kwargs["v"].data_ptr()) == (
        args["qkv"].data_ptr() + heads_dim,
        args["qkv"].data_ptr() + 2 * heads_dim,
    )
    assert kwargs["q"].shape[:1] == (1,) and kwargs["q"].stride()[1:] == (3 * HEADS * DIM, DIM, 1)
    assert kwargs["initial_state" if "initial_state" in kwargs else "initial_state_source"].shape == (
        SLOTS,
        HEADS,
        DIM,
        DIM,
    )


def test_the_prefill_baseline_calls_chunk_kda_as_sglang_does(sglang_kernels):
    args = op_operands(8, n_seqs=2) | {"cu_seqlens": torch.tensor([0, 5, 8], dtype=torch.int32)}
    out = baseline.KimiK3SglKdaPrefillBaseline().baseline(**args)["out"]
    call = sglang_kernels["chunk_kda"]
    sglang_views(call, args)
    assert out.shape == (8, HEADS, DIM)
    assert call["initial_state"].data_ptr() == args["state_cache"].data_ptr()
    assert call["g"].shape == (1, 8, HEADS, DIM) and call["beta"].shape == (1, 8, HEADS)
    assert call["beta"].stride(1) == BFA  # raw and strided, as beta_is_raw takes it
    assert call["A_log"].numel() == HEADS and call["dt_bias"].shape == (HEADS * DIM,)
    assert call["initial_state_indices"] is args["state_indices"] and call["cu_seqlens"] is args["cu_seqlens"]
    flags = {n: call[n] for n in ("use_qk_l2norm_in_kernel", "beta_is_raw", "output_intermediate_states",
                                  "lower_bound", "track_state", "track_chunk_idx", "chunk_indices", "chunk_offsets")}  # fmt: skip
    assert flags == {
        "use_qk_l2norm_in_kernel": True,
        "beta_is_raw": True,
        "output_intermediate_states": False,
        "lower_bound": -5.0,
        "track_state": None,
        "track_chunk_idx": None,
        "chunk_indices": None,
        "chunk_offsets": None,
    }


def test_the_track_baseline_asks_chunk_kda_for_the_snapshots(sglang_kernels):
    snapshot = torch.zeros(2, HEADS * DIM, DIM)
    chunks = torch.tensor([1, -1], dtype=torch.int32)
    args = op_operands(8, n_seqs=2) | {
        "cu_seqlens": torch.tensor([0, 5, 8], dtype=torch.int32),
        "track_chunk_idx": chunks,
        "track_state": snapshot,
    }
    out = baseline.KimiK3SglKdaPrefillTrackBaseline().baseline(**args)["out"]
    call = sglang_kernels["chunk_kda"]
    assert out.shape == (8, HEADS, DIM) and call["output_intermediate_states"] is True
    assert (
        call["track_state"].shape == (2, HEADS, DIM, DIM)
        and call["track_state"].data_ptr() == snapshot.data_ptr()
    )
    assert call["track_chunk_idx"] is chunks


def test_the_decode_baseline_calls_the_recurrent_kernel_as_sglang_does(sglang_kernels):
    args = op_operands(3)
    out = baseline.KimiK3SglKdaDecodeBaseline().baseline(**args)["out"]
    call = sglang_kernels["recurrent"]
    sglang_views(call, args)
    assert out.shape == (3, HEADS, DIM)
    assert (
        call["a"].shape == (3, HEADS * DIM)
        and call["b"].shape == (1, 3, HEADS)
        and call["b"].stride(1) == BFA
    )
    assert call["initial_state_indices"] is args["state_indices"]
    assert call["cu_seqlens"].tolist() == [0, 1, 2, 3]
    flags = {n: call[n] for n in ("use_qk_l2norm_in_kernel", "is_kda", "lower_bound", "softplus_beta",
                                  "softplus_threshold")}  # fmt: skip
    assert flags == {
        "use_qk_l2norm_in_kernel": True,
        "is_kda": True,
        "lower_bound": -5.0,
        "softplus_beta": 1.0,
        "softplus_threshold": 20.0,
    }
