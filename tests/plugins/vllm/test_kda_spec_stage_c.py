# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3's KDA layer after in_proj in a speculative verify step (kimi_k3_kda_spec_stage_c): the reference against
vLLM's ops transcribed one by one (CPU, float32), its padding windows, rows and untouched slots, what the route at
the decoder layer's self-attention takes and hands over, which calls stay with vLLM and when, the baseline's scale
and staged convolution weight, the committed FlyDSL schema and its kernel module (on gfx950: its padding windows and
rows against the reference)."""

import gc
import importlib.util
import inspect
import json
import sys
import weakref
from types import ModuleType, SimpleNamespace

import pytest
import torch

from aklr.core.contract.fastkey import FastKeyCodec, FastKeyFnGenerator
from aklr.core.contract.locate import committed_schema_paths, op_dir
from aklr.core.contract.registry import ContractRegistry
from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins.vllm.models.kimi_k3.kda_spec_stage_c import api
from aklr.plugins.vllm.models.kimi_k3.kda_spec_stage_c.baseline import kimi_k3_kda_spec_stage_c_baseline
from aklr.plugins.vllm.models.kimi_k3.kda_spec_stage_c.reference import kimi_k3_kda_spec_stage_c_reference
from tests.op_case import OpCase

HEADS, DIM, TAPS, WINDOW = 2, 128, 4, 5
HD = HEADS * DIM
CONV_LEN = TAPS - 2 + WINDOW
SLOTS = 32
PREFIX = "model.layers.1.self_attn"
CONTRACT_ID = "kimi_k3_kda_spec_stage_c@1.0"
SWITCH = "AKLR_VLLM_KDA_STAGE_C"
SCHEMA_NAME = "kimi_k3_kda_spec_stage_c_strided_rows_n_heads12_head_dim128_width4_conv_len7"
SCHEMA_SEQS = tuple(range(1, 33))
# a speculative step as vLLM's GDN metadata names it: each window's slots (its block table row) and accepted count
# (DSpark's 4 drafts); the third sequence starts from its fourth token, the second from its first
WINDOW_SLOTS = ((3, 9, 4, 17, 21), (5, 11, 2, 30, 8), (14, 26, 7, 19, 12))
ACCEPTED = (2, 1, 4)
API = inspect.signature(api.kimi_k3_kda_spec_stage_c)  # the op's api, before a test replaces it


def _proj_width(heads=HEADS):
    return 4 * heads * DIM + DIM + heads + (-(4 * heads * DIM + DIM + heads) % 16)


class FakeKDA:
    """KimiK3DeltaAttention as the route reads it: the layer's weights and caches (the convolution state, SD, and the
    recurrent state in shared pages) and in_proj / o_proj, which record their calls."""

    def __init__(self, n_tokens, seed=0, heads=HEADS):
        gen = torch.Generator().manual_seed(seed)
        hd = heads * DIM

        def randn(*shape, scale=1.0, dtype=torch.float32):
            return (torch.randn(*shape, generator=gen) * scale).to(dtype)

        self.prefix, self.local_num_heads, self.head_dim, self.conv_size = PREFIX, heads, DIM, TAPS
        self.in_proj_padding = _proj_width(heads) - (4 * hd + DIM + heads)
        self.conv1d = SimpleNamespace(weight=randn(3 * hd, 1, TAPS, scale=0.5), bias=None)
        self.f_b_proj = SimpleNamespace(
            weight=randn(hd, DIM, scale=DIM**-0.5, dtype=torch.bfloat16), bias=None
        )
        self.A_log = torch.rand(heads, generator=gen).mul(15).add(1).log()
        self.dt_bias = randn(hd)
        self.gate_lower_bound = -5.0
        self.o_norm = SimpleNamespace(weight=randn(DIM, dtype=torch.bfloat16), eps=1e-5, activation="sigmoid")
        conv_floats = (
            CONV_LEN * 3 * hd // 2
        )  # the page's bf16 convolution state, then its float32 recurrent state
        self.pages = randn(SLOTS, conv_floats + hd * DIM)
        conv = self.pages[:, :conv_floats].view(torch.bfloat16).view(SLOTS, CONV_LEN, 3 * hd)
        conv.copy_(randn(SLOTS, CONV_LEN, 3 * hd))  # finite bf16
        self.kv_cache = (conv, self.pages[:, conv_floats:].view(SLOTS, heads, DIM, DIM))
        self.proj = randn(n_tokens, _proj_width(heads), dtype=torch.bfloat16)
        self.in_proj_calls, self.o_proj_calls = [], []

    def in_proj_qkvgfab(self, hidden_states):
        self.in_proj_calls.append(hidden_states)
        return (self.proj,)

    def o_proj(self, x):
        self.o_proj_calls.append(x)
        return (x * 2,)


def metadata(slots=WINDOW_SLOTS, accepted=ACCEPTED, **changes):
    """vLLM's GDNAttentionMetadata of a pure verify step (the fields the route reads)."""
    m = SimpleNamespace(
        num_prefills=0,
        num_decodes=0,
        num_spec_decodes=len(accepted),
        spec_state_indices_tensor=torch.tensor(slots, dtype=torch.int32),
        num_accepted_tokens=torch.tensor(accepted, dtype=torch.int32),
        uniform_spec_sequence_length=WINDOW,
    )
    for name, value in changes.items():
        setattr(m, name, value)
    return m


# ---------- vLLM's ops, transcribed one by one (float32) ----------


def vllm_f_b(f_a, weight):
    """f_b_proj: an unquantized GEMM writing bf16."""
    return (f_a.float() @ weight.float().T).bfloat16()


def vllm_conv_update(x, conv_state, weight, slots, accepted):
    """vLLM's _causal_conv1d_update_kernel with IS_SPEC_DECODING and SiLU, sequence by sequence: (its bf16 output,
    the state after)."""
    state = conv_state.clone()
    n_seqs = len(accepted)
    window = x.shape[0] // n_seqs
    state_len, width = state.shape[1], weight.shape[1]
    out = torch.empty(x.shape, dtype=torch.bfloat16)
    for s in range(n_seqs):
        coord, offset = slots[s][0], accepted[s] - 1
        cols = [state[coord, offset + j].float() for j in range(width - 1)]
        new = torch.stack([
            state[coord, offset + 1 + i] if i < state_len - window else x[s * window + i - (state_len - window)]
            for i in range(state_len)
        ])  # fmt: skip
        for t in range(window):
            taps = [*cols, x[s * window + t].float()]
            acc = torch.zeros(x.shape[1])
            for j in range(width):
                acc = acc + taps[j] * weight[:, j]
            cols = cols[1:] + [x[s * window + t].float()]
            out[s * window + t] = (acc / (1 + torch.exp(-acc))).bfloat16()
        state[coord] = new
    return out, state


def vllm_recurrent(qkv, raw_g, raw_beta, a_log, dt_bias, lower_bound, state, slots, accepted, scale):
    """vLLM's fused_recurrent_kda_fwd_kernel (speculative, gate and beta in the kernel), a (sequence, head) at a time:
    (its bf16 output, the state after)."""
    state = state.clone()
    n_seqs, heads = len(accepted), state.shape[1]
    window = qkv.shape[0] // n_seqs
    out = torch.zeros(qkv.shape[0], heads, DIM, dtype=torch.bfloat16)
    for s in range(n_seqs):
        start = slots[s][accepted[s] - 1]
        if start <= 0:
            continue
        for h in range(heads):
            b_state = state[start, h].float()
            b_a = torch.exp(a_log[h].float())
            for t in range(s * window, (s + 1) * window):
                b_q = qkv[t, h * DIM : (h + 1) * DIM].float()
                b_k = qkv[t, HD + h * DIM : HD + (h + 1) * DIM].float()
                b_v = qkv[t, 2 * HD + h * DIM : 2 * HD + (h + 1) * DIM].float()
                b_q = b_q * torch.rsqrt((b_q * b_q).sum() + 1e-6)
                b_k = b_k * torch.rsqrt((b_k * b_k).sum() + 1e-6)
                b_q = b_q * scale
                b_gate = raw_g[t, h].float() + dt_bias[h * DIM : (h + 1) * DIM]
                b_gate = lower_bound * torch.sigmoid(b_a * b_gate)
                b_state = b_state * torch.exp(b_gate[None, :])
                b_v = b_v - (b_state * b_k[None, :]).sum(1)
                b_v = b_v * torch.sigmoid(raw_beta[t, h].float())
                b_state = b_state + b_v[:, None] * b_k[None, :]
                out[t, h] = (b_state * b_q[None, :]).sum(1).bfloat16()
                slot = slots[s][t - s * window]
                if slot > 0:
                    state[slot, h] = b_state
    return out, state


def vllm_gated_norm(x, gate, weight, eps):
    """FLA's layer_norm_gated forward, RMS with the sigmoid gate, a head a row."""
    x = x.float()
    rstd = 1 / torch.sqrt((x * x).sum(-1, keepdim=True) / x.shape[-1] + eps)
    return (x * rstd * weight.float() * torch.sigmoid(gate.float())).bfloat16()


def vllm_chain(attn, slots=WINDOW_SLOTS, accepted=ACCEPTED):
    """vLLM's general path for the stage on the layer's in-projection rows, on copies of its caches: (out, conv
    state, recurrent state)."""
    proj, (conv, state) = attn.proj, attn.kv_cache
    rows = len(accepted) * WINDOW
    raw_g = vllm_f_b(proj[:rows, 4 * HD : 4 * HD + DIM], attn.f_b_proj.weight).view(rows, HEADS, DIM)
    y, conv = vllm_conv_update(
        proj[:rows, : 3 * HD], conv, attn.conv1d.weight.view(3 * HD, TAPS), slots, accepted
    )
    o, state = vllm_recurrent(y, raw_g, proj[:rows, 4 * HD + DIM : 4 * HD + DIM + HEADS], attn.A_log, attn.dt_bias,
                              attn.gate_lower_bound, state, slots, accepted, DIM**-0.5)  # fmt: skip
    out = vllm_gated_norm(
        o, proj[:rows, 3 * HD : 4 * HD].view(rows, HEADS, DIM), attn.o_norm.weight, attn.o_norm.eps
    )
    return out, conv, state


def logical_dtypes(args):
    """Each tensor operand's dtype as the contract names it (as torch does)."""
    return {
        name: str(value.dtype).removeprefix("torch.")
        for name, value in args.items()
        if torch.is_tensor(value)
    }


def stage_args(attn, slots=WINDOW_SLOTS, accepted=ACCEPTED, pad_slot=31):
    """The op's keyword arguments as the route hands them over (views of vLLM's tensors); each padding row past the
    windows names ``pad_slot`` (unread)."""
    flat = torch.tensor(slots, dtype=torch.int32).view(-1)
    n_pad = attn.proj.shape[0] - flat.numel()
    flat = torch.cat((flat, torch.full((n_pad,), pad_slot, dtype=torch.int32)))
    names = ("qkv out_gate f_a raw_beta f_b_weight conv_weight conv_state a_log g_bias norm_weight state_cache "
             "state_indices num_accepted_tokens").split()  # fmt: skip
    args = dict(zip(names, api._op_args(attn, attn.proj, flat, torch.tensor(accepted, dtype=torch.int32))))
    return args | {"scale": DIM**-0.5, "lower_bound": attn.gate_lower_bound, "eps": attn.o_norm.eps}


def reference_call(attn, slots, accepted, pad_slot=31):
    """The reference on the op's operands as the route hands them over (views of vLLM's tensors)."""
    args = stage_args(attn, slots, accepted, pad_slot)
    n_seqs = len(accepted)
    axes = {"n_heads": HEADS, "head_dim": DIM, "width": TAPS, "conv_len": CONV_LEN, "n_seqs": n_seqs,
            "n_tokens": attn.proj.shape[0], "n_slots": SLOTS}  # fmt: skip
    point = SimpleNamespace(axes=axes, dtypes=logical_dtypes(args))
    return kimi_k3_kda_spec_stage_c_reference(point, args)


def test_the_reference_is_vllms_ops_in_their_order():
    """In float32 the reference computes what vLLM's general path does, op by op, for windows from their first,
    second and fourth tokens: the bf16 GEMM, the convolution update and its SiLU output, the recurrence, every
    token's state to its slot, the gated norm of the bf16 core output."""
    attn = FakeKDA(len(ACCEPTED) * WINDOW)
    want_out, want_conv, want_state = vllm_chain(attn)
    out, conv, state = reference_call(attn, WINDOW_SLOTS, ACCEPTED)
    assert torch.equal(out.bfloat16(), want_out)
    assert torch.equal(conv, want_conv.float())
    torch.testing.assert_close(state.view_as(want_state), want_state, rtol=1e-6, atol=1e-6)
    written = {slot for row in WINDOW_SLOTS for slot in row}
    for slot in set(range(SLOTS)) - written:  # no slot of the step: unchanged
        assert torch.equal(state.view_as(want_state)[slot], attn.kv_cache[1][slot])
        assert torch.equal(conv[slot], attn.kv_cache[0][slot].float())


@pytest.mark.parametrize("sentinel", [0, -1], ids=["null_block", "pad_slot_id"])
def test_padding_windows_and_rows_are_zero_and_write_nothing(sentinel):
    """A window whose start slot is <= 0 (vLLM's graph padding: every slot its null block 0, or -1, repeated) gives
    zero rows and writes no slot, its convolution slot's included; rows past the windows are zero."""
    slots = (*WINDOW_SLOTS[:2], (sentinel,) * WINDOW)
    accepted = (*ACCEPTED[:2], 1)
    attn = FakeKDA(len(accepted) * WINDOW + 2)
    out, conv, state = reference_call(attn, slots, accepted, pad_slot=sentinel)
    assert not out[2 * WINDOW :].any()
    want_out, _, _ = vllm_chain(attn, slots[:2], accepted[:2])
    assert torch.equal(out[: 2 * WINDOW].bfloat16(), want_out)
    for slot in (0, *(s for s in range(SLOTS) if s not in {x for row in slots[:2] for x in row})):
        assert torch.equal(state.view(SLOTS, HEADS, DIM, DIM)[slot], attn.kv_cache[1][slot])
        assert torch.equal(conv[slot], attn.kv_cache[0][slot].float())


def test_a_window_reads_its_convolution_history_from_its_accepted_token():
    """Accepting 1 .. 5 tokens moves the history read and the history kept by as many places: the slot keeps the
    entries after the accepted token's, then the window's inputs."""
    attn = FakeKDA(WINDOW)
    conv_before = attn.kv_cache[0].float().clone()
    for n_acc in range(1, WINDOW + 1):
        _, conv, _ = reference_call(attn, WINDOW_SLOTS[:1], (n_acc,))
        slot = WINDOW_SLOTS[0][0]
        assert torch.equal(conv[slot, : TAPS - 2], conv_before[slot, n_acc : n_acc + TAPS - 2])
        assert torch.equal(conv[slot, TAPS - 2 :], attn.proj[:WINDOW, : 3 * HD].float())


# ---------- the route ----------


class _Op:
    """The op as the route sees it: the reference writing out and the caches as a kernel does, its calls recorded,
    and the dispatch's answer for a shape (with ``key_of``, OUT_OF_CONTRACT where that fast key declines the call)."""

    def __init__(self, answer=None, result=None):
        self.calls, self.asked, self.answer, self.result = [], [], answer, result
        self.key_of = None

    def fallback_reason(self, *args, **kwargs):
        self.asked.append((args, kwargs))
        if self.key_of is not None:
            bound = API.bind(*args, **kwargs)
            bound.apply_defaults()
            if self.key_of(dict(bound.arguments)) is None:
                return FallbackReason.OUT_OF_CONTRACT
        return self.answer

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.result is not None:
            return self.result
        bound = dict(zip(API.parameters, args)) | kwargs
        n_seqs = bound["num_accepted_tokens"].shape[0]
        axes = {"n_heads": HEADS, "head_dim": DIM, "width": TAPS, "conv_len": bound["conv_state"].shape[1],
                "n_seqs": n_seqs, "n_tokens": bound["qkv"].shape[0], "n_slots": SLOTS}  # fmt: skip
        point = SimpleNamespace(axes=axes, dtypes=logical_dtypes(bound))
        out, conv, state = kimi_k3_kda_spec_stage_c_reference(point, bound)
        bound["conv_state"].copy_(conv)
        bound["state_cache"].copy_(state)
        return bound["out"].copy_(out)


@pytest.fixture
def step(monkeypatch):
    """A KDA layer of a 3-sequence verify step in vLLM, the route switched on and its view of vLLM patched in, and
    the op."""
    monkeypatch.setattr(api, "_SEEN", set())
    monkeypatch.setattr(api, "_SERVES", {})
    monkeypatch.setenv(SWITCH, "1")
    monkeypatch.setattr(api, "_kda_class", lambda: FakeKDA)
    monkeypatch.setattr(api, "_breakable_graphs", lambda: False)
    op = _Op()
    monkeypatch.setattr(api, "kimi_k3_kda_spec_stage_c", op)
    attn = FakeKDA(len(ACCEPTED) * WINDOW)
    m = metadata()
    monkeypatch.setattr(api, "_forward_metadata", lambda: {PREFIX: m})
    hidden = torch.empty(len(ACCEPTED) * WINDOW, 16)
    return SimpleNamespace(attn=attn, layer=SimpleNamespace(self_attn=attn), m=m, op=op, hidden=hidden)


def test_the_route_takes_vllms_parameters():
    # KimiDecoderLayer._run_self_attn of vLLM 0.34.0 and 0.32.1rc1.dev276
    assert list(inspect.signature(api._run_self_attn).parameters) == ["self", "positions", "hidden_states"]


@pytest.mark.parametrize("value", [None, "0", "", "true"], ids=["unset", "zero", "empty", "true"])
def test_the_route_stays_off_unless_switched_on(step, monkeypatch, value):
    """Off until a deployment opts in with AKLR_VLLM_KDA_STAGE_C=1, read at each call: over chained steps the op and
    vLLM's served aiter chain are not yet shown to agree. Unset, or any other value, hands every call back to vLLM
    before anything runs."""
    if value is None:
        monkeypatch.delenv(SWITCH)
    else:
        monkeypatch.setenv(SWITCH, value)
    assert api._run_self_attn(step.layer, None, step.hidden) is FallbackReason.OUT_OF_CONTRACT
    assert step.attn.in_proj_calls == [] and step.op.calls == [] and step.op.asked == []


def test_a_verify_step_runs_in_proj_the_op_and_o_proj(step):
    """vLLM's in-projection once, the op on its rows and the layer's caches (what vLLM's path computes), o_proj on
    the op's output, its result returned."""
    attn = step.attn
    want_out, want_conv, want_state = vllm_chain(attn)
    result = api._run_self_attn(step.layer, None, step.hidden)
    assert attn.in_proj_calls == [step.hidden] and len(step.op.calls) == 1
    (core,) = attn.o_proj_calls
    assert core.shape == (len(ACCEPTED) * WINDOW, HD) and torch.equal(result, core * 2)
    assert torch.equal(core.view_as(want_out), want_out)
    assert torch.equal(attn.kv_cache[0], want_conv)
    torch.testing.assert_close(attn.kv_cache[1], want_state, rtol=1e-6, atol=1e-6)


def test_the_route_hands_over_views_of_vllms_tensors(step):
    """No copy: the in-projection's q|k|v, gate, f_a and beta columns as (rows, heads, dim) views, the weights as the
    layer holds them, the caches' slots as rows of their pages (strided_rows), the window slots flat; vLLM's scale,
    the layer's lower bound and norm eps."""
    attn = step.attn
    api._run_self_attn(step.layer, None, step.hidden)
    ((args, kwargs),) = step.op.calls
    (qkv, gate, f_a, beta, f_b, conv_weight, conv, a_log, g_bias, norm_weight, state, slots, accepted) = args
    base, row = attn.proj.data_ptr(), attn.proj.stride(0)
    for view, column, shape in (
        (qkv, 0, (15, 3 * HEADS, DIM)),
        (gate, 3 * HD, (15, HEADS, DIM)),
        (f_a, 4 * HD, (15, 1, DIM)),
        (beta, 4 * HD + DIM, (15, HEADS, 1)),
    ):
        assert view.shape == shape and view.stride(0) == row and view.data_ptr() == base + 2 * column
    assert f_b is attn.f_b_proj.weight and a_log is attn.A_log and norm_weight is attn.o_norm.weight
    assert conv_weight.shape == (3 * HD, TAPS) and conv_weight.data_ptr() == attn.conv1d.weight.data_ptr()
    assert g_bias.shape == (HEADS, DIM) and g_bias.data_ptr() == attn.dt_bias.data_ptr()
    assert conv is attn.kv_cache[0] and conv.stride(0) == 2 * attn.pages.stride(0)
    assert state.shape == (SLOTS, HD, DIM) and state.data_ptr() == attn.kv_cache[1].data_ptr()
    assert slots.tolist() == [s for row in WINDOW_SLOTS for s in row] and accepted.tolist() == list(ACCEPTED)
    assert kwargs["storage"] == "strided_rows" and kwargs["scale"] == DIM**-0.5
    assert kwargs["lower_bound"] == -5.0 and kwargs["eps"] == 1e-5
    assert kwargs["out"].shape == (15, HEADS, DIM) and kwargs["out"].is_contiguous()


def test_the_handed_over_operands_are_a_point_of_the_contract(step):
    """The contract's own key closure reads the call the route makes as a Point of vLLM's layout (no device check on
    the CPU), and the stand-in the route asks about first is the same Point."""
    api._run_self_attn(step.layer, None, step.hidden)
    contract = ContractRegistry.default().get(CONTRACT_ID)
    codec = FastKeyCodec.from_contract(contract)
    key_of = FastKeyFnGenerator(codec, None).generate()
    keys = []
    for args, kwargs in (*step.op.calls, *step.op.asked):
        bound = API.bind(*args, **kwargs)
        bound.apply_defaults()
        keys.append(key_of(dict(bound.arguments)))
    assert keys[0] is not None and keys[0] == keys[1]


def test_the_caches_of_one_page_grid_key_as_separate_buffers():
    """vLLM's two caches, rows of one buffer's pages (a slot's convolution state, then its recurrent state), are a
    Point of the contract as separate buffers are (aklr's fast key: written inputs apart in every page); state rows
    reaching into the convolution state's are not."""
    from tests.op_case import OpCase

    case = OpCase.of(CONTRACT_ID)
    fast_key = FastKeyFnGenerator(FastKeyCodec.from_contract(case.contract), torch.device("cpu")).generate()
    call = case.call(case.inputs())
    conv, state = call["conv_state"], call["state_cache"]
    conv_floats, state_floats = conv[0].numel() // 2, state[0].numel()
    pages = torch.zeros(conv.shape[0], conv_floats + state_floats)
    shared_conv = pages[:, :conv_floats].view(torch.bfloat16).view(conv.shape)
    shared_conv.copy_(conv)
    key = fast_key(call)
    assert key is not None
    for offset, served in ((conv_floats, True), (conv_floats - 4, False)):  # 4 floats: still 16-byte aligned
        shared_state = pages.as_strided(state.shape, (pages.stride(0), *state.stride()[1:]), offset)
        shared = fast_key({**call, "conv_state": shared_conv, "state_cache": shared_state})
        assert shared == (key if served else None)


def _quantized(step):
    """vLLM's fp8 attention input: QuantizedActivation(data, scale, dtype, orig_shape, key)."""
    step.hidden = SimpleNamespace(data=None, orig_shape=(len(ACCEPTED) * WINDOW, 7168))


def test_a_quantized_input_is_counted_by_its_original_rows(step):
    _quantized(step)
    api._run_self_attn(step.layer, None, step.hidden)
    assert len(step.op.calls) == 1 and step.attn.in_proj_calls == [step.hidden]


def _set(**changes):
    return lambda step: [setattr(step.m, k, v) for k, v in changes.items()]


def _layer(change):
    return lambda step: change(step.attn)


@pytest.mark.parametrize(
    "change",
    [
        lambda step: setattr(step.layer, "self_attn", SimpleNamespace()),  # an MLA layer
        lambda step: setattr(api, "_forward_metadata", lambda: {}),
        _set(num_prefills=1),
        _set(num_decodes=2),
        _set(num_spec_decodes=0),
        _set(uniform_spec_sequence_length=None),
        _set(num_spec_decodes=2),  # rows past the windows: graph padding
        _set(spec_state_indices_tensor=torch.tensor(WINDOW_SLOTS, dtype=torch.int64)),
        _set(num_accepted_tokens=torch.tensor(ACCEPTED[:2], dtype=torch.int32)),
        lambda step: setattr(api, "_breakable_graphs", lambda: True),
        _layer(lambda a: setattr(a.f_b_proj, "weight", a.f_b_proj.weight.to(torch.float8_e4m3fn))),
        _layer(lambda a: setattr(a, "kv_cache", (a.kv_cache[0].transpose(1, 2), a.kv_cache[1]))),
        # vLLM 0.34's mamba_ssm_cache_dtype can make the recurrent state bf16
        _layer(lambda a: setattr(a, "kv_cache", (a.kv_cache[0], a.kv_cache[1].to(torch.bfloat16)))),
        _layer(lambda a: setattr(a.o_norm, "activation", "swish")),
        _layer(lambda a: setattr(a, "gate_lower_bound", None)),
        _layer(lambda a: setattr(a.conv1d, "bias", torch.zeros(3 * HD))),
    ],
    ids=[
        "not_kda",
        "no_metadata",
        "prefills",
        "plain_decodes",
        "no_spec_decodes",
        "ragged_windows",
        "padding_rows",
        "int64_slots",
        "fewer_accepted_counts",
        "breakable_graphs",
        "fp8_f_b",
        "conv_state_ds_layout",
        "bf16_recurrent_state",
        "swish_norm",
        "softplus_gate",
        "conv_bias",
    ],
)
def test_what_the_contract_does_not_state_goes_back_before_anything_runs(step, change):
    change(step)
    assert api._run_self_attn(step.layer, None, step.hidden) is FallbackReason.OUT_OF_CONTRACT
    assert step.attn.in_proj_calls == [] and step.op.calls == [] and step.op.asked == []


def test_a_shape_the_op_does_not_serve_goes_back_before_the_in_projection(step):
    """The dispatch's answer for a step's shape is asked once, on a stand-in for the in-projection's rows."""
    step.op.answer = FallbackReason.NO_WORKLOAD
    for _ in range(2):
        assert api._run_self_attn(step.layer, None, step.hidden) is FallbackReason.NO_WORKLOAD
    assert step.attn.in_proj_calls == [] and step.op.calls == [] and len(step.op.asked) == 1
    (stand_in, *_), _ = step.op.asked[0]
    assert stand_in.shape == (15, 3 * HEADS, DIM) and stand_in.stride(0) == _proj_width()


def test_a_decline_from_the_steps_own_tensors_is_asked_again(step):
    """OUT_OF_CONTRACT can come from a step's tensors, not its shape: a dense slot table starting one int32 into its
    allocation misses the 16-byte alignment aklr's fast key asks of every operand. That step goes back to vLLM; a
    later step of the same shape on an aligned table is served."""
    contract = ContractRegistry.default().get(CONTRACT_ID)
    step.op.key_of = FastKeyFnGenerator(FastKeyCodec.from_contract(contract), None).generate()
    table = torch.zeros(len(ACCEPTED) * WINDOW + 1, dtype=torch.int32)
    table[1:] = torch.tensor(WINDOW_SLOTS).flatten()
    step.m.spec_state_indices_tensor = table[1:].view(len(ACCEPTED), WINDOW)
    assert step.m.spec_state_indices_tensor.is_contiguous()
    assert api._run_self_attn(step.layer, None, step.hidden) is FallbackReason.OUT_OF_CONTRACT
    assert step.attn.in_proj_calls == [] and step.op.calls == []
    step.m.spec_state_indices_tensor = torch.tensor(WINDOW_SLOTS, dtype=torch.int32)
    api._run_self_attn(step.layer, None, step.hidden)
    assert len(step.op.asked) == 2 and len(step.op.calls) == 1


def test_a_capture_before_any_eager_call_is_asked_again(step):
    step.op.answer = FallbackReason.NOT_WARMED
    assert api._run_self_attn(step.layer, None, step.hidden) is FallbackReason.NOT_WARMED
    step.op.answer = None
    api._run_self_attn(step.layer, None, step.hidden)
    assert len(step.op.asked) == 2 and len(step.op.calls) == 1


def test_a_decline_after_the_in_projection_goes_back_to_vllm(step):
    """The op said it serves and then declined: vLLM's own path runs (its in-projection again)."""
    step.op.result = FallbackReason.NO_SCHEMA
    assert api._run_self_attn(step.layer, None, step.hidden) is FallbackReason.NO_SCHEMA
    assert len(step.attn.in_proj_calls) == 1 and step.attn.o_proj_calls == []


def test_eager_block_table_slots_are_copied_and_a_capture_declines(step, monkeypatch):
    """vLLM's eager steps view the windows' columns of its block table (rows apart by more than a window): copied
    dense outside a capture; inside one (vLLM's graphs pass dense rows) nothing is copied and vLLM runs the step."""
    table = torch.zeros(len(ACCEPTED), 8, dtype=torch.int32)
    table[:, :WINDOW] = torch.tensor(WINDOW_SLOTS)
    step.m.spec_state_indices_tensor = table[:, :WINDOW]
    api._run_self_attn(step.layer, None, step.hidden)
    ((args, _),) = step.op.calls
    assert args[11].is_contiguous() and args[11].tolist() == table[:, :WINDOW].flatten().tolist()
    monkeypatch.setattr(api, "_capturing", lambda device: True)
    assert api._run_self_attn(step.layer, None, step.hidden) is FallbackReason.OUT_OF_CONTRACT
    assert len(step.op.calls) == 1


# ---------- the baseline ----------


def _vllm_gemm(layer, x, weight, bias):
    """vLLM's unquantized GEMM as dispatch_unquantized_gemm returns it: bf16 out."""
    return (x.float() @ weight.float().T).to(x.dtype)


@pytest.fixture
def vllm_calls(monkeypatch):
    """The two vLLM functions the baseline calls, as stand-ins: the unquantized GEMM dispatch, and aiter's fused
    kernel through rocm_aiter_ops, whose keyword arguments are recorded (the list returned)."""
    calls = []

    def module(name, **attrs):
        stand_in = ModuleType(name)
        stand_in.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, stand_in)

    module("vllm")
    module(
        "vllm._aiter_ops",
        rocm_aiter_ops=SimpleNamespace(kda_fused_recurrent_spec_decode=lambda **kw: calls.append(kw)),
    )
    module("vllm.model_executor")
    module("vllm.model_executor.layers")
    module("vllm.model_executor.layers.utils", dispatch_unquantized_gemm=lambda: _vllm_gemm)
    return calls


def _width_major(conv_weight):
    """vLLM's decode_conv1d_weight of a (3 * heads * dim, width) convolution weight: (3, width, heads * dim)."""
    return conv_weight.view(3, -1, conv_weight.shape[1]).transpose(1, 2)


def test_the_baseline_takes_the_float32_scale_the_workloads_record(vllm_calls):
    """The workloads record scale as the float32 aiter's kernel receives, 0.0883883461356163 (head_dim ** -0.5 is
    0.08838834764831845 as a double): the baseline serves it; a scale that is another float32 is a call vLLM does not
    make."""
    _, path = _schema_path()
    (scale,) = {w["scalars"]["scale"] for w in json.loads(path.read_text())["workloads"]}
    assert scale != DIM**-0.5
    args = stage_args(FakeKDA(len(ACCEPTED) * WINDOW)) | {"storage": "strided_rows"}
    kimi_k3_kda_spec_stage_c_baseline(**args | {"scale": scale})
    assert len(vllm_calls) == 1
    with pytest.raises(NotImplementedError, match="head_dim"):
        kimi_k3_kda_spec_stage_c_baseline(**args | {"scale": 0.0883883})


def test_the_baseline_restages_a_convolution_weight_reloaded_in_place(vllm_calls):
    """Validation reloads a weight through .data.copy_, which keeps its address and its version: the next call
    outside a graph capture hands aiter the width-major copy of the new values (what vLLM's
    process_weights_after_loading stages), not of the old."""
    args = stage_args(FakeKDA(len(ACCEPTED) * WINDOW)) | {"storage": "strided_rows"}
    weight = args["conv_weight"]
    kimi_k3_kda_spec_stage_c_baseline(**args)
    assert torch.equal(vllm_calls[0]["conv_weight"], _width_major(weight))
    weight.data.copy_(torch.randn(weight.shape, generator=torch.Generator().manual_seed(7)))
    kimi_k3_kda_spec_stage_c_baseline(**args)
    staged = vllm_calls[1]["conv_weight"]
    assert staged.is_contiguous() and torch.equal(staged, _width_major(weight))


def test_the_staged_convolution_weight_goes_with_its_weight(vllm_calls):
    """The width-major copy belongs to the weight tensor it was made from and is freed with it: an allocation that
    reuses the weight's address never finds it."""
    args = stage_args(FakeKDA(len(ACCEPTED) * WINDOW)) | {"storage": "strided_rows"}
    kimi_k3_kda_spec_stage_c_baseline(**args)
    staged = weakref.ref(vllm_calls.pop()["conv_weight"])
    assert staged() is not None
    del args
    gc.collect()
    assert staged() is None


@pytest.mark.gpu  # a CUDA graph capture
def test_a_capture_reads_the_weight_the_last_eager_call_staged(vllm_calls):
    """The timed calls are CUDA graph replays: a capture hands aiter the copy the eager call before it staged and
    records no copy of its own (as vLLM's graphs read the weight staged at load time); the next eager call stages
    the weight's values again."""
    args = stage_args(FakeKDA(len(ACCEPTED) * WINDOW)) | {"storage": "strided_rows"}
    args = {n: v.cuda() if torch.is_tensor(v) else v for n, v in args.items()}
    weight = args["conv_weight"]
    kimi_k3_kda_spec_stage_c_baseline(**args)
    staged = vllm_calls[-1]["conv_weight"]
    eager = staged.clone()
    weight.data.copy_(torch.randn_like(weight))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        kimi_k3_kda_spec_stage_c_baseline(**args)
    graph.replay()
    torch.cuda.synchronize()
    assert vllm_calls[-1]["conv_weight"] is staged and torch.equal(staged, eager)
    kimi_k3_kda_spec_stage_c_baseline(**args)
    assert vllm_calls[-1]["conv_weight"] is staged and torch.equal(staged, _width_major(weight))


# ---------- the committed schema and its kernel ----------


def _schema_path():
    contract = ContractRegistry.default().get(CONTRACT_ID)
    (path,) = committed_schema_paths(op_dir(contract.op_package), SCHEMA_NAME)
    return contract, path


def test_the_committed_schema_is_its_canonical_text_at_its_path():
    """One FlyDSL schema, vLLM's layout at 12 heads and DSpark's window of 5: its file is the registry's schema text
    at the path its content names; the workloads K3's verify steps, vLLM's scale, K3's lower bound and o_norm eps,
    unmeasured (validate fills them)."""
    contract, path = _schema_path()
    text = path.read_text()
    schema = ContractRegistry.default().parse_schema(path.read_bytes(), str(path))
    assert schema.schema_text() == text
    assert path.relative_to(op_dir(contract.op_package)).as_posix() == schema.relative_path().as_posix()
    assert schema.kernel.language == "flydsl" and schema.kernel.entry_point == "kernel.py::run"
    assert [w.axes["n_seqs"] for w in schema.workloads] == list(SCHEMA_SEQS)
    for w in schema.workloads:
        assert w.axes["n_tokens"] == WINDOW * w.axes["n_seqs"] < w.axes["n_slots"]
        assert w.scalars == {"scale": float(torch.tensor(DIM**-0.5)), "lower_bound": -5.0,
                             "eps": float(torch.tensor(1e-5))}  # fmt: skip
        assert w.accuracy is None and w.speedup is None
    assert contract.schema_errors(path, schema) == []
    assert "Apache-2.0" in json.loads(text)["kernel"]["sources"][0]["content"]  # K1's notice


@pytest.fixture(scope="module")
def kernel_module(tmp_path_factory):
    """The committed kernel source, written out and imported under the builder's package prefix; needs a FlyDSL with
    structs in shared memory (0.3.2 on)."""
    fx = pytest.importorskip("flydsl.expr")
    if not hasattr(fx, "SharedAllocator"):
        pytest.skip("a FlyDSL older than the kernel's")
    from aklr.core.runtime.builder import KERNEL_PACKAGE_PREFIX

    _, path = _schema_path()
    (source,) = json.loads(path.read_text())["kernel"]["sources"]
    file = tmp_path_factory.mktemp("kda_spec_stage_c_kernel") / source["path"]
    file.write_text(source["content"])
    name = f"{KERNEL_PACKAGE_PREFIX}test_kimi_k3_kda_spec_stage_c.kernel"
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(name, None)


def test_the_kernel_module_serves_the_contract(kernel_module):
    """run takes the contract's call; one build per band count, one band at every sequence count; warm_up builds
    each."""
    contract, _ = _schema_path()
    inspect.signature(kernel_module.run).bind(**dict.fromkeys(contract.kernel_call_keys))
    assert callable(kernel_module.warm_up)
    assert (
        kernel_module.MAX_WINDOW == 8
        and kernel_module.WARM_HEADS == 12
        and kernel_module.WARM_WINDOW == WINDOW
    )
    bands = {n: kernel_module.bands_for(n) for n in range(1, 33)}
    assert bands == dict.fromkeys(range(1, 33), 1)  # one band: split bands measured slower at every count
    assert {kernel_module.bands_for(n) for n in kernel_module.WARM_SEQS} == {1}
    for b in (8, 2, 1):
        rows_per_lane, waves = kernel_module._SHAPES[b]
        assert 8 * rows_per_lane * waves * b == DIM and callable(kernel_module._build(12, b, WINDOW))


@pytest.mark.gfx950
@pytest.mark.parametrize("sentinel", [0, -1], ids=["null_block", "pad_slot_id"])
def test_the_kernel_meets_the_reference_on_padding_windows_and_rows(kernel_module, sentinel):
    """The committed kernel at the small Point (two heads, two windows of three tokens and a padding row, vLLM's
    layout), its second window padding as vLLM's graphs pad it (every slot the sentinel, one accepted token) and the
    padding row's slot the sentinel too: out and both caches pass the contract's compare against the reference, the
    padding window's rows and the padding row are zero, and no slot but the live window's changes, slot 0 included."""
    case = OpCase.of(CONTRACT_ID)
    axes = case.point.axes
    window = axes["conv_len"] - axes["width"] + 2
    args = {n: v.cuda() if torch.is_tensor(v) else v for n, v in case.inputs().items()}
    args["state_indices"][window:] = sentinel
    args["num_accepted_tokens"][1] = 1
    call = case.call(args, "cuda")
    before = {n: call[n].clone() for n in case.contract.mutable_inputs}
    out = torch.empty(
        (axes["n_tokens"], axes["n_heads"], axes["head_dim"]), dtype=torch.bfloat16, device="cuda"
    )
    kernel_module.run(**{k: call[k] for k in case.contract.kernel_call_keys} | {"out": out})
    torch.cuda.synchronize()
    outputs = case.contract.call_mutable_tensors(out, call)
    results = case.compare(outputs, args)
    assert {o: r for o, r in results.items() if not r.is_passed} == {}
    assert not out[window:].any()
    live = set(args["state_indices"][:window].tolist())
    for name, cache in before.items():
        for slot in sorted(set(range(cache.shape[0])) - live):
            assert torch.equal(outputs[name][slot], cache[slot]), (name, slot)
