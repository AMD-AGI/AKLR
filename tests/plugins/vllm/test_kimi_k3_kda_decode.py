# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3's KDA decode on vLLM to AKLR's kimi_k3_kda_decode (packed, one token a sequence) and
kimi_k3_kda_decode_spec (DSPARK's verify windows): the references against vLLM's Triton kernels transcribed op by op
(CPU, float32, bit for bit), what the routes take and hand over, which calls stay with vLLM, the committed FlyDSL
schemas, and their kernel modules as FlyDSL reads them."""

import importlib.util
import inspect
import json
import sys
from types import SimpleNamespace

import pytest
import torch

from aklr.core.contract.fastkey import FastKeyCodec, FastKeyFnGenerator
from aklr.core.contract.locate import committed_schema_paths, op_dir
from aklr.core.contract.registry import ContractRegistry
from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins.vllm.models.kimi_k3.kda_decode import api
from aklr.plugins.vllm.models.kimi_k3.kda_decode.reference import (
    kimi_k3_kda_decode_reference,
    kimi_k3_kda_decode_spec_reference,
)

HEADS, DIM, SLOTS, CONV = 2, 128, 7, 96
N = 5
# K3's row of the fused projection on a rank of 12 heads: q, k, v, g (4 * 12 * 128), f_a (128), beta (12), padding (4)
PROJ_ROW, BETA_AT = 6288, 4 * 12 * 128 + 128
SCHEMA_TOKENS = (1, 2, 4, 8, 16, 24, 48, 96)

# vLLM 0.34.0's and 0.32.1rc1.dev276's models/kimi_k3/amd/ops/third_party/kda/fused_recurrent.py
VLLM_PARAMETERS = (
    "mixed_qkv raw_g raw_beta A_log dt_bias lower_bound initial_state state_indices scale".split()
)


def test_the_route_takes_vllms_parameters():
    assert list(inspect.signature(api._fused_recurrent_kda_packed_decode).parameters) == VLLM_PARAMETERS


def vllm_call(slots=(3, 0, 5, 1, -1), heads=HEADS):
    """A packed decode call as kda.py makes it: the convolution's packed q|k|v rows, the gate projection [1, n, heads,
    dim], beta a column of the fused projection, the recurrent state in pages shared with the convolution state."""
    gen = torch.Generator().manual_seed(0)
    n = len(slots)

    def randn(*shape, scale=1.0, dtype=torch.bfloat16):
        return (torch.randn(*shape, generator=gen) * scale).to(dtype)

    projection = randn(n, PROJ_ROW)
    pages = torch.randn(SLOTS, heads * DIM * DIM + CONV, generator=gen)
    return dict(
        mixed_qkv=randn(n, 3 * heads * DIM),
        raw_g=randn(n, heads * DIM, scale=DIM**-0.5).view(1, n, heads, DIM),
        raw_beta=projection[:, BETA_AT : BETA_AT + heads].unsqueeze(0),
        A_log=torch.rand(heads, generator=gen).mul(15).add(1).log(),
        dt_bias=torch.randn(heads * DIM, generator=gen),
        lower_bound=-5.0,
        initial_state=pages[:, : heads * DIM * DIM].view(SLOTS, heads, DIM, DIM),
        state_indices=torch.tensor(slots, dtype=torch.int32),
    ), pages


def triton_packed_decode(call, scale=None):
    """vLLM's fused_recurrent_kda_packed_decode_kernel, op by op in float32, one (token, head) at a time (its value
    blocks only split the rows): (out as float32, the state after the call)."""
    mixed, raw_g, raw_beta = call["mixed_qkv"], call["raw_g"], call["raw_beta"]
    state = call["initial_state"].clone()
    _, H, V, K = state.shape
    scale = K**-0.5 if scale is None else scale
    out = torch.empty(mixed.shape[0], H, V)
    for i_n, state_idx in enumerate(call["state_indices"].tolist()):
        for i_h in range(H):
            if state_idx <= 0:
                out[i_n, i_h] = 0
                continue
            b_state = state[state_idx, i_h].float()
            b_q = mixed[i_n, i_h * K : (i_h + 1) * K].float()
            b_k = mixed[i_n, H * K + i_h * K : H * K + (i_h + 1) * K].float()
            b_v = mixed[i_n, 2 * H * K + i_h * V : 2 * H * K + (i_h + 1) * V].float()
            b_q = b_q / torch.sqrt((b_q * b_q).sum() + 1e-6)
            b_k = b_k / torch.sqrt((b_k * b_k).sum() + 1e-6)
            b_q = b_q * scale
            b_g = raw_g[0, i_n, i_h].float()
            b_bias = call["dt_bias"][i_h * K : (i_h + 1) * K].float()
            b_a = torch.exp(call["A_log"][i_h].float())
            b_g = b_g + b_bias
            b_gate = call["lower_bound"] * torch.sigmoid(b_a * b_g)
            b_state = b_state * torch.exp(b_gate[None, :])
            b_v = b_v - (b_state * b_k[None, :]).sum(1)
            b_beta = torch.sigmoid(raw_beta[0, i_n, i_h].float())
            b_v = b_v * b_beta
            b_state = b_state + b_v[:, None] * b_k[None, :]
            out[i_n, i_h] = (b_state * b_q[None, :]).sum(1)
            state[state_idx, i_h] = b_state
    return out, state


def reference_args(call, scale=None):
    """The contract's row-major operands of a vLLM call, and the Point the reference reads."""
    heads = call["initial_state"].shape[1]
    args = dict(
        qkv=call["mixed_qkv"].unflatten(1, (3 * heads, DIM)).contiguous(),
        raw_g=call["raw_g"][0].contiguous(),
        raw_beta=call["raw_beta"][0].unsqueeze(-1).contiguous(),
        a_log=call["A_log"],
        g_bias=call["dt_bias"].view(heads, DIM),
        state_cache=call["initial_state"].flatten(1, 2).contiguous(),
        state_indices=call["state_indices"],
        scale=DIM**-0.5 if scale is None else scale,
        lower_bound=call["lower_bound"],
    )
    point = SimpleNamespace(axes={"n_heads": heads, "head_dim": DIM}, dtypes=dict.fromkeys(args))
    return args, point


@pytest.mark.parametrize("scale", [None, 0.25], ids=["default_scale", "scale"])
def test_the_reference_is_vllms_kernel_bit_for_bit(scale):
    """In float32 the reference computes what vLLM's Triton kernel does, op for op: q and k divided by their norms,
    q scaled after; the gate from g + bias; the state decayed, the delta, the output from the new state; a slot <= 0
    gives a zero output and touches nothing."""
    call, _ = vllm_call()
    want_out, want_state = triton_packed_decode(call, scale)
    args, point = reference_args(call, scale)
    out, state_cache = kimi_k3_kda_decode_reference(point, args, lambda x, _: x.float())
    assert torch.equal(out, want_out)
    assert torch.equal(state_cache.view(SLOTS, HEADS, DIM, DIM), want_state)
    assert torch.equal(out[1], torch.zeros(HEADS, DIM)) and torch.equal(out[4], torch.zeros(HEADS, DIM))


def test_the_reference_in_float64_is_the_recurrence():
    """In float64 against the delta rule written as matrices: S = S diag(exp g), S += beta (v - S k) k^T, out = S q."""
    call, _ = vllm_call()
    args, point = reference_args(call)
    out, state_cache = kimi_k3_kda_decode_reference(point, args, lambda x, _: x.double())
    states = call["initial_state"].double().clone()
    for t, slot in enumerate(call["state_indices"].tolist()):
        if slot <= 0:
            assert not out[t].any()
            continue
        q, k, v = args["qkv"][t].double().unflatten(0, (3, HEADS)).unbind(0)
        q = torch.nn.functional.normalize(q, dim=-1, eps=0) * DIM**-0.5
        k = torch.nn.functional.normalize(k, dim=-1, eps=0)
        g = -5.0 * torch.sigmoid(
            call["A_log"].double().exp()[:, None] * (args["raw_g"][t] + args["g_bias"]).double()
        )
        beta = torch.sigmoid(args["raw_beta"][t, :, 0].double())
        S = states[slot] @ torch.diag_embed(g.exp())
        S = S + beta[:, None, None] * (v - (S @ k[..., None])[..., 0])[..., None] @ k[:, None, :]
        torch.testing.assert_close(out[t], (S @ q[..., None])[..., 0], rtol=1e-6, atol=1e-8)
        states[slot] = S
    torch.testing.assert_close(state_cache.view_as(states), states, rtol=1e-6, atol=1e-8)


class _Op:
    """The op as the route sees it, recording its calls."""

    def __init__(self, fn):
        self.fn, self.calls = fn, []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.fn(*args, **kwargs)


@pytest.fixture(autouse=True)
def _fresh_route(monkeypatch):
    """Each test with no output rows kept and every fallback logged afresh."""
    monkeypatch.setattr(api, "_OUT", {})
    monkeypatch.setattr(api, "_RETIRED", [])
    monkeypatch.setattr(api, "_SEEN", set())


def _reference_op(qkv, raw_g, raw_beta, a_log, g_bias, state_cache, state_indices, storage="row_major",
                  scale=DIM**-0.5, lower_bound=-5.0, out_dtype=None, out=None):  # fmt: skip
    """The op as its reference computes it, writing state_cache and out as a kernel does."""
    heads = raw_g.shape[1]
    args = dict(qkv=qkv, raw_g=raw_g, raw_beta=raw_beta, a_log=a_log, g_bias=g_bias, state_cache=state_cache,
                state_indices=state_indices, scale=scale, lower_bound=lower_bound)  # fmt: skip
    point = SimpleNamespace(axes={"n_heads": heads, "head_dim": DIM}, dtypes=dict.fromkeys(args))
    o, states = kimi_k3_kda_decode_reference(point, args, lambda x, _: x.float())
    state_cache.copy_(states)
    return out.copy_(o)


@pytest.fixture
def op(monkeypatch):
    fake = _Op(_reference_op)
    monkeypatch.setattr(api, "kimi_k3_kda_decode", fake)
    return fake


def test_a_packed_decode_call_runs_the_op_on_vllms_paged_state(op):
    call, pages = vllm_call()
    before = pages.clone()
    want_out, want_state = triton_packed_decode(call)
    out, state = api._fused_recurrent_kda_packed_decode(**call)
    assert state is call["initial_state"] and out.shape == (1, N, HEADS, DIM) and out.dtype == torch.bfloat16
    torch.testing.assert_close(out[0].float(), want_out, atol=1e-3, rtol=1e-2)
    for slot in (1, 3, 5):
        torch.testing.assert_close(call["initial_state"][slot], want_state[slot], atol=1e-5, rtol=1e-5)
    for slot in (0, 2, 4, 6):  # the null block and slots no token names
        assert torch.equal(pages[slot], before[slot])
    assert torch.equal(pages[:, HEADS * DIM * DIM :], before[:, HEADS * DIM * DIM :])  # the convolution state


def test_the_route_hands_over_views_of_vllms_tensors(op):
    """No copy: the packed rows as (n, 3 heads, dim), the gate's [0], beta's column as (n, heads, 1), the state's slots
    as (slots, heads * dim, dim) rows of the pages, in the strided_rows layout; the scale vLLM's default."""
    call, _ = vllm_call()
    api._fused_recurrent_kda_packed_decode(**call)
    ((args, kwargs),) = op.calls
    qkv, raw_g, raw_beta, a_log, g_bias, state_cache, state_indices = args
    assert qkv.shape == (N, 3 * HEADS, DIM) and qkv.data_ptr() == call["mixed_qkv"].data_ptr()
    assert raw_g.shape == (N, HEADS, DIM) and raw_g.data_ptr() == call["raw_g"].data_ptr()
    assert raw_beta.shape == (N, HEADS, 1) and raw_beta.data_ptr() == call["raw_beta"].data_ptr()
    assert raw_beta.stride() == (PROJ_ROW, 1, 1)
    assert (
        a_log is call["A_log"]
        and g_bias.shape == (HEADS, DIM)
        and g_bias.data_ptr() == call["dt_bias"].data_ptr()
    )
    assert (
        state_cache.shape == (SLOTS, HEADS * DIM, DIM)
        and state_cache.data_ptr() == call["initial_state"].data_ptr()
    )
    assert state_cache.stride(0) == HEADS * DIM * DIM + CONV
    assert state_indices is call["state_indices"]
    assert (
        kwargs["storage"] == "strided_rows" and kwargs["scale"] == DIM**-0.5 and kwargs["lower_bound"] == -5.0
    )
    assert kwargs["out"].shape == (N, HEADS, DIM) and kwargs["out"].is_contiguous()


def test_the_handed_over_operands_are_a_point_of_the_contract(op):
    """The contract's own key closure reads the call the route makes as the Point of vLLM's shapes (no device
    check on the CPU): nothing it hands over is outside the contract's strided_rows layout."""
    call, _ = vllm_call()
    api._fused_recurrent_kda_packed_decode(**call)
    ((args, kwargs),) = op.calls
    contract = ContractRegistry.default().get("kimi_k3_kda_decode@1.0")
    bound = inspect.signature(_reference_op).bind(*args, **kwargs)  # the api's parameters
    bound.apply_defaults()
    codec = FastKeyCodec.from_contract(contract)
    key = FastKeyFnGenerator(codec, None).generate()(dict(bound.arguments))
    assert key is not None
    point = codec.to_point(key)
    assert point.axes == {"n_tokens": N, "n_slots": SLOTS, "n_heads": HEADS, "head_dim": DIM}
    assert point.layout_variant_name == "strided_rows"


def test_an_explicit_scale_is_handed_over(op):
    call, _ = vllm_call()
    api._fused_recurrent_kda_packed_decode(**call, scale=0.25)
    assert op.calls[0][1]["scale"] == 0.25


@pytest.mark.parametrize(
    "change",
    [
        {"lower_bound": None},
        {"mixed_qkv": "half"},
        {"initial_state": "half"},
        {"state_indices": "long"},
        {"state_indices": "table"},
        {"raw_g": "transposed"},
        {"initial_state": "transposed"},
        {"raw_beta": "short"},
        {"dt_bias": "strided"},
    ],
    ids=[
        "softplus_gate",
        "fp16_qkv",
        "fp16_state",
        "int64_indices",
        "spec_indices",
        "gate_not_dense",
        "state_not_dense",
        "beta_shape",
        "bias_not_dense",
    ],
)
def test_what_no_contract_states_goes_back_to_vllm(monkeypatch, change):
    monkeypatch.setattr(api, "kimi_k3_kda_decode", lambda *a, **kw: pytest.fail("the op ran"))
    call, _ = vllm_call()
    ((name, how),) = change.items()
    changed = {
        None: lambda x: None,
        "half": lambda x: x.half(),
        "long": lambda x: x.long(),
        "table": lambda x: x[:, None].expand(-1, 4),  # the speculative path's (token, draft) slots
        "transposed": lambda x: x.transpose(-1, -2),
        "short": lambda x: x[:, :, :-1],
        "strided": lambda x: torch.zeros(2 * x.numel())[::2],
    }
    call[name] = changed[how](call[name])
    assert api._fused_recurrent_kda_packed_decode(**call) is FallbackReason.OUT_OF_CONTRACT


@pytest.mark.parametrize(
    "reason", [FallbackReason.NO_WORKLOAD, FallbackReason.NOT_QUALIFIED, FallbackReason.NO_SCHEMA]
)
def test_a_declined_op_call_goes_back_to_vllm(monkeypatch, reason):
    monkeypatch.setattr(api, "kimi_k3_kda_decode", lambda *a, **kw: reason)
    call, _ = vllm_call()
    assert api._fused_recurrent_kda_packed_decode(**call) is reason


def test_the_op_without_a_table_declines():
    """The api as dispatch builds it, no schema loaded (the CPU): it declines, and vLLM runs its own kernel."""
    call, _ = vllm_call()
    assert isinstance(api._fused_recurrent_kda_packed_decode(**call), FallbackReason)


def test_a_block_table_column_is_handed_over_dense(op):
    """vLLM's eager batches name their slots by a column of the block table: the contract's are dense."""
    call, _ = vllm_call()
    table = torch.zeros(N, 8, dtype=torch.int32)
    table[:, 0] = call["state_indices"]
    call["state_indices"] = table[:, 0]
    api._fused_recurrent_kda_packed_decode(**call)
    handed = op.calls[0][0][6]
    assert handed.stride() == (1,) and torch.equal(handed, table[:, 0])


def test_a_block_table_column_inside_a_capture_goes_back_to_vllm(monkeypatch):
    """Inside a graph capture the route copies nothing: vLLM's graphs pass a dense vector anyway."""
    monkeypatch.setattr(api, "kimi_k3_kda_decode", lambda *a, **kw: pytest.fail("the op ran"))
    monkeypatch.setattr(api, "_capturing", lambda device: True)
    call, _ = vllm_call()
    call["state_indices"] = torch.zeros(N, 8, dtype=torch.int32)[:, 0]
    assert api._fused_recurrent_kda_packed_decode(**call) is FallbackReason.OUT_OF_CONTRACT


def test_the_output_rows_are_kept_and_never_freed(monkeypatch, op):
    """One buffer a stream, its first n rows a call's; a larger batch outside a capture grows it (doubling) and keeps
    the old one (a captured graph may still write it); inside a capture a buffer too small hands the call back to
    vLLM (as dispatch does for a Point never run outside one) and nothing is allocated."""
    call, _ = vllm_call()
    first, _ = api._fused_recurrent_kda_packed_decode(**call)
    small, _ = api._fused_recurrent_kda_packed_decode(**vllm_call(slots=(2, 4))[0])
    assert small.data_ptr() == first.data_ptr() and len(api._OUT) == 1 and not api._RETIRED
    big_call, _ = vllm_call(slots=(1, 2, 3, 4, 5, 6))
    big, _ = api._fused_recurrent_kda_packed_decode(**big_call)
    (buf,) = api._OUT.values()
    assert buf.shape[0] == 2 * N and big.data_ptr() == buf.data_ptr()
    assert len(api._RETIRED) == 1 and api._RETIRED[0].data_ptr() == first.data_ptr()
    monkeypatch.setattr(api, "_capturing", lambda device: True)
    calls = len(op.calls)
    huge_call, _ = vllm_call(slots=tuple(range(1, SLOTS)) + (0,) * 5)
    assert api._fused_recurrent_kda_packed_decode(**huge_call) is FallbackReason.NOT_WARMED
    assert len(op.calls) == calls and list(api._OUT.values()) == [buf]
    inside, _ = api._fused_recurrent_kda_packed_decode(**call)  # fits: the kept rows, nothing allocated
    assert inside.data_ptr() == buf.data_ptr()


# ---------- the speculative path: vLLM's fused_recurrent_kda on DSPARK's verify windows ----------

# vLLM 0.34.0's and 0.32.1rc1.dev276's fused_recurrent_kda
VLLM_SPEC_PARAMETERS = (
    "q k v raw_g raw_beta A_log dt_bias lower_bound initial_state cu_seqlens ssm_state_indices num_accepted_tokens "
    "uniform_sequence_length out fuse_gate"
).split()
SPEC_SLOTS = 16


def test_the_speculative_route_takes_vllms_parameters():
    assert list(inspect.signature(api._fused_recurrent_kda).parameters) == VLLM_SPEC_PARAMETERS


def vllm_spec_call(accepted=(2, 1), window=3, heads=HEADS, table_width=None, padding=False):
    """A speculative batch as kda.py makes it: q, k and v split from the convolution's packed rows, the gate [1, n,
    heads, dim], beta a column of the fused projection, the state pages, each window's slots a row of a dense buffer
    (the graph's; a block table of table_width columns, else), the accepted counts, the windows' offsets and length,
    out the batch's rows of core_attn_out. ``padding``: the last window's slots are the null block, as vLLM pads."""
    gen = torch.Generator().manual_seed(1)
    n_seqs = len(accepted)
    n = n_seqs * window

    def randn(*shape, scale=1.0, dtype=torch.bfloat16):
        return (torch.randn(*shape, generator=gen) * scale).to(dtype)

    mixed = randn(n, 3 * heads * DIM)
    q, k, v = (x.unflatten(1, (heads, DIM)).unsqueeze(0) for x in mixed.split(heads * DIM, dim=-1))
    projection = randn(n, PROJ_ROW)
    pages = torch.randn(SPEC_SLOTS, heads * DIM * DIM + CONV, generator=gen)
    slots = (torch.randperm(SPEC_SLOTS - 1, generator=gen)[:n] + 1).to(torch.int32).view(n_seqs, window)
    if padding:
        slots[-1] = 0
    table = torch.zeros(n_seqs, table_width or window, dtype=torch.int32)
    table[:, :window] = slots
    return dict(
        q=q,
        k=k,
        v=v,
        raw_g=randn(n, heads * DIM, scale=DIM**-0.5).view(1, n, heads, DIM),
        raw_beta=projection[:, BETA_AT : BETA_AT + heads].unsqueeze(0),
        A_log=torch.rand(heads, generator=gen).mul(15).add(1).log(),
        dt_bias=torch.randn(heads * DIM, generator=gen),
        lower_bound=-5.0,
        initial_state=pages[:, : heads * DIM * DIM].view(SPEC_SLOTS, heads, DIM, DIM),
        cu_seqlens=torch.arange(n_seqs + 1, dtype=torch.int32) * window,
        ssm_state_indices=table if table_width is None else table[:, :window],
        num_accepted_tokens=torch.tensor(accepted, dtype=torch.int32),
        uniform_sequence_length=window,
        out=torch.empty(1, n + 2, heads, DIM, dtype=torch.bfloat16)[:, :n],
    ), pages


def triton_spec_decode(call):
    """vLLM's fused_recurrent_kda_fwd_kernel on a speculative batch of uniform windows, op by op in float32, one
    (sequence, head) at a time: (out as float32, NaN where the kernel writes nothing; the state after the call)."""
    q, k, v = (call[x][0] for x in "qkv")
    state = call["initial_state"].clone()
    _, H, V, K = state.shape
    L = call["uniform_sequence_length"]
    out = torch.full((q.shape[0], H, V), float("nan"))
    for i_n, n_acc in enumerate(call["num_accepted_tokens"].tolist()):
        for i_h in range(H):
            bos = i_n * L
            state_index = int(call["ssm_state_indices"][i_n, n_acc - 1])
            if state_index <= 0:
                out[bos, i_h] = 0
                continue
            b_state = state[state_index, i_h].float()
            b_a = torch.exp(call["A_log"][i_h].float())
            b_bias = call["dt_bias"][i_h * K : (i_h + 1) * K].float()
            for i_t in range(L):
                b_q = q[bos + i_t, i_h].float()
                b_k = k[bos + i_t, i_h].float()
                b_v = v[bos + i_t, i_h].float()
                b_q = b_q * torch.rsqrt((b_q * b_q).sum() + 1e-6)
                b_k = b_k * torch.rsqrt((b_k * b_k).sum() + 1e-6)
                b_q = b_q * K**-0.5
                b_gate = call["raw_g"][0, bos + i_t, i_h].float()
                b_gate = b_gate + b_bias
                b_gate = call["lower_bound"] * torch.sigmoid(b_a * b_gate)
                b_state = b_state * torch.exp(b_gate[None, :])
                b_v = b_v - (b_state * b_k[None, :]).sum(1)
                b_beta = torch.sigmoid(call["raw_beta"][0, bos + i_t, i_h].float())
                b_v = b_v * b_beta
                b_state = b_state + b_v[:, None] * b_k[None, :]
                out[bos + i_t, i_h] = (b_state * b_q[None, :]).sum(1)
                final_state_index = int(call["ssm_state_indices"][i_n, i_t])
                if final_state_index > 0:
                    state[final_state_index, i_h] = b_state
    return out, state


def spec_reference_args(call):
    heads = call["initial_state"].shape[1]
    n = call["q"].shape[1]
    qkv = torch.cat([call[x][0] for x in "qkv"], dim=1)  # (n, 3 heads, dim)
    args = dict(
        qkv=qkv,
        raw_g=call["raw_g"][0].contiguous(),
        raw_beta=call["raw_beta"][0].unsqueeze(-1).contiguous(),
        a_log=call["A_log"],
        g_bias=call["dt_bias"].view(heads, DIM),
        state_cache=call["initial_state"].flatten(1, 2).contiguous(),
        state_indices=call["ssm_state_indices"].reshape(n),
        num_accepted_tokens=call["num_accepted_tokens"],
        scale=DIM**-0.5,
        lower_bound=call["lower_bound"],
    )
    return args, SimpleNamespace(axes={"n_heads": heads, "head_dim": DIM}, dtypes=dict.fromkeys(args))


@pytest.mark.parametrize("accepted", [(2, 1), (3, 3), (1, 2)], ids=["mid", "last", "first"])
def test_the_speculative_reference_is_vllms_kernel_bit_for_bit(accepted):
    """In float32 the reference computes what vLLM's recurrent kernel does on a speculative batch, op for op: each
    window from its accepted token's slot, every token's state to its own slot."""
    call, _ = vllm_spec_call(accepted=accepted)
    want_out, want_state = triton_spec_decode(call)
    args, point = spec_reference_args(call)
    out, state_cache = kimi_k3_kda_decode_spec_reference(point, args, lambda x, _: x.float())
    assert torch.equal(out, want_out)
    assert torch.equal(state_cache.view(SPEC_SLOTS, HEADS, DIM, DIM), want_state)


def test_a_padding_window_gives_zeros_and_writes_no_slot():
    """vLLM's padding windows start from its null block: its kernel writes the window's first output, zero, and no
    slot; the contract's outputs are all zero there (vLLM reads none of them)."""
    call, _ = vllm_spec_call(padding=True)
    want_out, want_state = triton_spec_decode(call)
    args, point = spec_reference_args(call)
    out, state_cache = kimi_k3_kda_decode_spec_reference(point, args, lambda x, _: x.float())
    assert not out[3:].any() and torch.equal(out[3], want_out[3]) and want_out[4:].isnan().all()
    assert torch.equal(out[:3], want_out[:3])
    assert torch.equal(state_cache.view(SPEC_SLOTS, HEADS, DIM, DIM), want_state)


def _spec_reference_op(qkv, raw_g, raw_beta, a_log, g_bias, state_cache, state_indices, num_accepted_tokens,
                       storage="row_major", scale=DIM**-0.5, lower_bound=-5.0, out_dtype=None, out=None):  # fmt: skip
    """The speculative op as its reference computes it, writing state_cache and out as a kernel does."""
    args = dict(qkv=qkv, raw_g=raw_g, raw_beta=raw_beta, a_log=a_log, g_bias=g_bias, state_cache=state_cache,
                state_indices=state_indices, num_accepted_tokens=num_accepted_tokens, scale=scale,
                lower_bound=lower_bound)  # fmt: skip
    point = SimpleNamespace(axes={"n_heads": raw_g.shape[1], "head_dim": DIM}, dtypes=dict.fromkeys(args))
    o, states = kimi_k3_kda_decode_spec_reference(point, args, lambda x, _: x.float())
    state_cache.copy_(states)
    return out.copy_(o)


@pytest.fixture
def spec_op(monkeypatch):
    fake = _Op(_spec_reference_op)
    monkeypatch.setattr(api, "kimi_k3_kda_decode_spec", fake)
    return fake


def test_a_speculative_call_runs_the_op_on_vllms_paged_state(spec_op):
    call, pages = vllm_spec_call()
    before = pages.clone()
    want_out, want_state = triton_spec_decode(call)
    out, state = api._fused_recurrent_kda(**call)
    assert out is call["out"] and state is call["initial_state"]
    torch.testing.assert_close(out[0].float(), want_out, atol=1e-3, rtol=1e-2)
    written = set(call["ssm_state_indices"].flatten().tolist())
    for slot in range(SPEC_SLOTS):
        if slot in written:
            torch.testing.assert_close(call["initial_state"][slot], want_state[slot], atol=1e-5, rtol=1e-5)
        else:
            assert torch.equal(pages[slot], before[slot])
    assert torch.equal(pages[:, HEADS * DIM * DIM :], before[:, HEADS * DIM * DIM :])


def test_the_speculative_route_hands_over_views_of_vllms_tensors(spec_op):
    """No copy: q, k and v as the (n, 3 heads, dim) rows they split, beta's column as (n, heads, 1), the state's slots
    as rows of the pages, the windows' slots flat, out's rows [0]."""
    call, _ = vllm_spec_call()
    api._fused_recurrent_kda(**call)
    ((args, kwargs),) = spec_op.calls
    qkv, raw_g, raw_beta, a_log, g_bias, state_cache, state_indices, accepted = args
    assert qkv.shape == (6, 3 * HEADS, DIM) and qkv.stride() == (3 * HEADS * DIM, DIM, 1)
    assert qkv.data_ptr() == call["q"].data_ptr()
    assert torch.equal(qkv[:, HEADS : 2 * HEADS], call["k"][0]) and torch.equal(
        qkv[:, 2 * HEADS :], call["v"][0]
    )
    assert raw_g.data_ptr() == call["raw_g"].data_ptr() and raw_beta.stride() == (PROJ_ROW, 1, 1)
    assert state_cache.shape == (SPEC_SLOTS, HEADS * DIM, DIM)
    assert state_cache.data_ptr() == call["initial_state"].data_ptr()
    assert state_indices.shape == (6,) and state_indices.data_ptr() == call["ssm_state_indices"].data_ptr()
    assert accepted.data_ptr() == call["num_accepted_tokens"].data_ptr()
    assert kwargs["out"].data_ptr() == call["out"].data_ptr() and kwargs["out"].shape == (6, HEADS, DIM)
    assert kwargs["storage"] == "strided_rows" and kwargs["scale"] == DIM**-0.5


def test_the_speculative_operands_are_a_point_of_the_contract(spec_op):
    call, _ = vllm_spec_call(accepted=(1, 2, 3), window=4)
    api._fused_recurrent_kda(**call)
    ((args, kwargs),) = spec_op.calls
    contract = ContractRegistry.default().get("kimi_k3_kda_decode_spec@1.0")
    bound = inspect.signature(_spec_reference_op).bind(*args, **kwargs)  # the api's parameters
    bound.apply_defaults()
    codec = FastKeyCodec.from_contract(contract)
    key = FastKeyFnGenerator(codec, None).generate()(dict(bound.arguments))
    assert key is not None
    point = codec.to_point(key)
    assert point.axes == {
        "n_seqs": 3,
        "n_tokens": 12,
        "n_slots": SPEC_SLOTS,
        "n_heads": HEADS,
        "head_dim": DIM,
    }


def test_without_out_the_kept_rows_are_returned(spec_op):
    """A batch mixing speculative and other sequences passes no out: the kept rows, which vLLM scatters right away."""
    call, _ = vllm_spec_call()
    call["out"] = None
    out, _ = api._fused_recurrent_kda(**call)
    ((buf,),) = [list(api._OUT.values())]
    assert out.shape == (1, 6, HEADS, DIM) and out.data_ptr() == buf.data_ptr()


def test_block_table_rows_are_handed_over_dense(spec_op):
    """vLLM's eager batches name the windows' slots by the first window columns of block table rows: copied dense."""
    call, _ = vllm_spec_call(table_width=8)
    api._fused_recurrent_kda(**call)
    handed = spec_op.calls[0][0][6]
    assert handed.is_contiguous() and torch.equal(handed, call["ssm_state_indices"].reshape(-1))


def test_block_table_rows_inside_a_capture_go_back_to_vllm(monkeypatch):
    monkeypatch.setattr(api, "kimi_k3_kda_decode_spec", lambda *a, **kw: pytest.fail("the op ran"))
    monkeypatch.setattr(api, "_capturing", lambda device: True)
    call, _ = vllm_spec_call(table_width=8)
    assert api._fused_recurrent_kda(**call) is FallbackReason.OUT_OF_CONTRACT


def _unpacked(call):
    call["q"], call["k"], call["v"] = (x.clone() for x in (call["q"], call["k"], call["v"]))


def _mixed_batch_decodes(call):
    """kda.py's other call: the plain decodes of a batch with prefills, one slot a token, no accepted counts."""
    call["ssm_state_indices"] = call["ssm_state_indices"][:, 0].contiguous()
    call["num_accepted_tokens"] = None
    call["uniform_sequence_length"] = None


@pytest.mark.parametrize(
    "change",
    [
        _mixed_batch_decodes,
        lambda c: c.update(uniform_sequence_length=None),
        lambda c: c.update(fuse_gate=False),
        lambda c: c.update(lower_bound=None),
        _unpacked,
        lambda c: c.update(uniform_sequence_length=2),
        lambda c: c.update(ssm_state_indices=c["ssm_state_indices"][:, :2], uniform_sequence_length=3),
        lambda c: c.update(ssm_state_indices=c["ssm_state_indices"].long()),
        lambda c: c.update(num_accepted_tokens=c["num_accepted_tokens"][:1]),
        lambda c: c.update(out=torch.empty(1, 6, HEADS, DIM, dtype=torch.bfloat16).transpose(2, 3)),
    ],
    ids=[
        "mixed_batch_decodes",
        "uneven_windows",
        "unfused_gate",
        "softplus_gate",
        "unpacked_qkv",
        "windows_mismatch",
        "slots_narrower_than_window",
        "int64_slots",
        "fewer_accepted_counts",
        "out_not_dense",
    ],
)
def test_what_no_speculative_contract_states_goes_back_to_vllm(monkeypatch, change):
    monkeypatch.setattr(api, "kimi_k3_kda_decode_spec", lambda *a, **kw: pytest.fail("the op ran"))
    call, _ = vllm_spec_call()
    change(call)
    assert api._fused_recurrent_kda(**call) is FallbackReason.OUT_OF_CONTRACT


# ---------- the committed schemas and their kernels ----------

SCHEMAS = {
    "kimi_k3_kda_decode@1.0": (
        "kimi_k3_kda_decode_strided_rows_n_heads12_head_dim128",
        "n_tokens",
        SCHEMA_TOKENS,
    ),
    "kimi_k3_kda_decode_spec@1.0": (
        "kimi_k3_kda_decode_spec_strided_rows_n_heads12_head_dim128",
        "n_seqs",
        (1, 2, 2, 3, 4, 4, 6, 6, 8, 8, 10, 12, 12, 14, 16, 20, 24, 32, 48, 64, 96, 120),
    ),
}
# the speculative schema's sequences at DSPARK's windows of 8 (7 drafts); the rest are windows of 5 (K=4, vLLM's
# c10-c60 recipes), an even number of sequences each
WINDOWS_OF_8 = (1, 2, 3, 4, 6, 8, 12, 14)


def _schema_path(contract_id):
    contract = ContractRegistry.default().get(contract_id)
    name = SCHEMAS[contract_id][0]
    (path,) = committed_schema_paths(op_dir(contract.op_package), name)
    return contract, path


@pytest.mark.parametrize("contract_id", list(SCHEMAS))
def test_the_committed_schema_is_its_canonical_text_at_its_path(contract_id):
    """One FlyDSL schema each, vLLM's layout at 12 heads: its file is the registry's schema text at the path its
    content names; the workloads K3's batches (windows of 8 and 5), scale and lower bound K3's, unmeasured
    (validate fills them)."""
    contract, path = _schema_path(contract_id)
    name, batch_axis, batches = SCHEMAS[contract_id]
    text = path.read_text()
    schema = ContractRegistry.default().parse_schema(path.read_bytes(), str(path))
    assert schema.schema_text() == text
    assert path.relative_to(op_dir(contract.op_package)).as_posix() == schema.relative_path().as_posix()
    assert schema.name == name
    assert schema.kernel.language == "flydsl" and schema.kernel.entry_point == "kernel.py::run"
    assert [w.axes[batch_axis] for w in schema.workloads] == list(batches)
    for w in schema.workloads:
        assert w.axes["n_slots"] > w.axes["n_tokens"]
        if batch_axis == "n_seqs":
            window, rest = divmod(w.axes["n_tokens"], w.axes["n_seqs"])
            assert rest == 0 and window in (5, 8)
        assert w.scalars == {
            "scale": float(torch.tensor(DIM**-0.5, dtype=torch.float32)),
            "lower_bound": -5.0,
        }
        assert w.accuracy is None and w.speedup is None
    if batch_axis == "n_seqs":
        windows = [(w.axes["n_seqs"], w.axes["n_tokens"] // w.axes["n_seqs"]) for w in schema.workloads]
        assert tuple(n for n, window in windows if window == 8) == WINDOWS_OF_8
        assert all(n % 2 == 0 for n, window in windows if window == 5)
    assert contract.schema_errors(path, schema) == []
    raw = json.loads(text)
    assert all("speedup" not in w and "accuracy" not in w for w in raw["workloads"])


@pytest.fixture(scope="module", params=list(SCHEMAS))
def kernel_module(request, tmp_path_factory):
    """A committed kernel source, written out and imported under the builders' package prefix, as the FlyDSL
    builder imports it; needs the kernel's FlyDSL (0.3.4 on)."""
    pytest.importorskip("flydsl")
    typing = pytest.importorskip("flydsl.expr.typing")
    if not hasattr(typing.Constexpr, "value_signature"):
        pytest.skip("a FlyDSL older than the kernel's")
    from aklr.core.runtime.builder import KERNEL_PACKAGE_PREFIX

    contract_id = request.param
    _, path = _schema_path(contract_id)
    (source,) = json.loads(path.read_text())["kernel"]["sources"]
    file = tmp_path_factory.mktemp("kda_decode_kernel") / source["path"]
    file.write_text(source["content"])
    name = f"{KERNEL_PACKAGE_PREFIX}test_{contract_id.split('@')[0]}.kernel"
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield contract_id, module
    finally:
        sys.modules.pop(name, None)


def test_the_kernel_module_serves_the_contract(kernel_module):
    contract_id, module = kernel_module
    contract, _ = _schema_path(contract_id)
    inspect.signature(module.run).bind(**dict.fromkeys(contract.kernel_call_keys))
    assert module.RANGE_SAFE is True and callable(module.warm_up) and module.WARM_HEADS == 12
    if contract_id == "kimi_k3_kda_decode@1.0":
        assert module.WARM_TOKENS == SCHEMA_TOKENS
        # few tokens take the narrow band (16 one-wave workgroups a head), more the wide one: two builds, none a size
        assert {n: module._band(n, 12) for n in SCHEMA_TOKENS} == {
            **dict.fromkeys((1, 2, 4, 8), module._NARROW),
            **dict.fromkeys((16, 24, 48, 96), module._WIDE),
        }
    else:
        assert module.WARM_SEQS == WINDOWS_OF_8 and module.WARM_WINDOW == 8 and module.MAX_WINDOW == 16
        # DSPARK's batches all take the narrow band: the walk is serial in the window
        assert {module._band(n, 12) for n in module.WARM_SEQS} == {module._NARROW}
