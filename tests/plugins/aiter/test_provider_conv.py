# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""AKLR serving aiter's prefill causal conv1d with the q/k/v split (``causal_conv1d_split_qkv``, the
Triton tile kernel Kimi-K3's KDA prefill calls on vLLM) as aiter's kernel plugin: what the selector
takes, what it declines, the AKLR call its run makes and that it computes aiter's result, and, with
aiter's plugin interface, the route aiter takes."""

import importlib
import inspect
from types import SimpleNamespace

import pytest
import torch

from aklr.core.runtime.builder import RunnableMetadata
from aklr.core.runtime.dispatch import FallbackReason
from aklr.ops.kda_conv.reference import KimiK3KdaConvReference

provider = importlib.import_module("aklr.plugins.aiter.provider")
base = importlib.import_module("aklr.plugins.aiter.provider._base")
conv = importlib.import_module("aklr.plugins.aiter.provider.conv")

BF16, F32, I32 = torch.bfloat16, torch.float32, torch.int32
KERNEL = RunnableMetadata(
    build_type="flydsl", contract_id="any@1.0", schema_name="any_schema", content_hash="0123456789abcdef0123"
)
PROJ, WIDTH, SLOTS, TOKENS = 128, 4, 4, 10
DIM = 3 * PROJ

# aiter's contract (tests/plugin_contracts.json at a6c759b1f): version, positional parameters
AITER_CONTRACTS = {
    "causal_conv1d_split_qkv": (
        1,
        (
            "x",
            "weight",
            "bias",
            "conv_states",
            "query_start_loc",
            "k_dim",
            "v_dim",
            "cache_indices",
            "has_initial_state",
            "activation",
            "pad_slot_id",
            "block_m",
            "block_n",
            "num_warps",
            "metadata",
            "preserve_input_dtype",
        ),
    ),
}


def _state(state_len=WIDTH - 1, sd=True):
    """vLLM's conv state: stored (slots, state_len, dim) and handed as its (slots, dim, state_len) view."""
    if sd:
        return torch.randn(SLOTS, state_len, DIM, dtype=BF16).transpose(1, 2)
    return torch.randn(SLOTS, DIM, state_len, dtype=BF16)


def _call(**changes):
    """A call AKLR's contract takes, by parameter name, defaults applied: x as vLLM hands it, the q|k|v
    columns of wider projection rows as [dim, n_tokens]."""
    wide = torch.randn(TOKENS, DIM + 64, dtype=BF16)
    call = {
        "x": wide[:, :DIM].t(),
        "weight": torch.randn(DIM, WIDTH, dtype=F32),
        "bias": None,
        "conv_states": _state(),
        "query_start_loc": torch.tensor([0, 4, TOKENS], dtype=I32),
        "k_dim": PROJ,
        "v_dim": PROJ,
        "cache_indices": torch.tensor([2, 0], dtype=I32),
        "has_initial_state": torch.tensor([True, False]),
        "activation": "silu",
        "pad_slot_id": -1,
        "block_m": 64,
        "block_n": 32,
        "num_warps": 4,
        "metadata": None,
        "preserve_input_dtype": False,
    }
    return {**call, **changes}


def _args(call):
    return tuple(call[name] for name in AITER_CONTRACTS["causal_conv1d_split_qkv"][1])


def _select(call):
    return base.SELECTORS["causal_conv1d_split_qkv"][1](_args(call), {})


def _answer(monkeypatch, serving):
    monkeypatch.setattr(conv.kimi_k3_kda_conv, "serving", lambda *a, **k: serving)


@pytest.fixture
def aklr_serves(monkeypatch):
    """AKLR's op answers with KERNEL; a candidate is (kernel, run)."""
    monkeypatch.setattr(
        conv, "offer", lambda serving, run: None if isinstance(serving, FallbackReason) else (serving, run)
    )
    _answer(monkeypatch, KERNEL)


class _Op:
    """A stand-in for AKLR's kimi_k3_kda_conv: its reference on the operands it is handed, writing the
    state into conv_state as the kernel does; ``serving`` answers KERNEL. Every call is kept."""

    def __init__(self):
        self.calls = []

    def serving(self, *args, **kwargs):
        return KERNEL

    def __call__(self, *args, storage):
        names = ("x", "weight", "conv_state", "state_indices", "has_initial_state", "cu_seqlens")
        call = {**dict(zip(names, args, strict=True)), "storage": storage}
        self.calls.append(call)
        out = KimiK3KdaConvReference().reference(**call)
        call["conv_state"].copy_(out["conv_state"])
        return out["q"], out["k"], out["v"]


def _aiters_result(call):
    """What aiter's kernel computes, from its operands as it reads them: per sequence, its slot's
    first width - 1 state rows (or zeros) then its tokens, convolved per channel, with silu;
    q, k and v in conv_states' dtype, and each slot it names left holding the window's last inputs."""
    x, weight, states = call["x"].t().float(), call["weight"].float(), call["conv_states"].clone()
    offsets, slots, resumes = (
        call[n].tolist() for n in ("query_start_loc", "cache_indices", "has_initial_state")
    )
    y = torch.empty_like(x)
    for seq, (slot, resume) in enumerate(zip(slots, resumes, strict=True)):
        start, end = offsets[seq], offsets[seq + 1]
        head = states[slot, :, : WIDTH - 1].t().float() if resume else torch.zeros(WIDTH - 1, DIM)
        window = torch.cat((head, x[start:end]))
        z = torch.nn.functional.conv1d(window.t()[None], weight[:, None, :], groups=DIM)[0].t()
        y[start:end] = torch.nn.functional.silu(z)
        states[slot, :, : WIDTH - 1] = window[-(WIDTH - 1) :].t().to(states.dtype)
    y = y.to(states.dtype)
    return (y[:, :PROJ], y[:, PROJ : 2 * PROJ], y[:, 2 * PROJ :]), states


def test_the_selector_serves_aiters_contract_at_its_version():
    version, _ = AITER_CONTRACTS["causal_conv1d_split_qkv"]
    assert base.SELECTORS["causal_conv1d_split_qkv"][0] == version
    assert provider.Provider.op_versions["causal_conv1d_split_qkv"] == {version}


def test_a_served_call_offers_a_run_taking_aiters_contract(aklr_serves):
    """aiter hands run every positional parameter: run has aiter's parameters, by name and order."""
    kernel, run = _select(_call())
    assert kernel is KERNEL
    params = inspect.signature(run).parameters
    assert tuple(params) == AITER_CONTRACTS["causal_conv1d_split_qkv"][1]
    assert all(p.kind is p.POSITIONAL_OR_KEYWORD for p in params.values())


DECLINED = {
    "bias": {"bias": torch.zeros(DIM, dtype=BF16)},
    # aiter applies no activation but silu (or its name swish)
    "no_activation": {"activation": None},
    "other_activation": {"activation": "gelu"},
    "k_dim_not_v_dim": {"k_dim": 64, "v_dim": 256},
    "rows_not_3_k_dim": {"k_dim": 64, "v_dim": 64},
    # aiter makes them up (slot i for sequence i, no state read); AKLR takes them from the caller
    "no_cache_indices": {"cache_indices": None},
    "no_has_initial_state": {"has_initial_state": None},
    # x channel-strided: no rows AKLR reads
    "x_token_major": {"x": torch.randn(DIM, TOKENS, dtype=BF16)},
    # a state stored (slots, dim, state_len): AKLR's contract takes vLLM's SD layout only
    "state_ds": {"conv_states": _state(sd=False)},
    "state_shorter_than_width": {"conv_states": _state(state_len=WIDTH - 2)},
    "state_dtype_not_xs": {"conv_states": _state().float()},
    "bf16_taps": {"weight": torch.randn(DIM, WIDTH, dtype=BF16)},
    "width_3": {"weight": torch.randn(DIM, 3, dtype=F32)},
    "int64_cache_indices": {"cache_indices": torch.tensor([2, 0])},
    "cache_indices_2d": {"cache_indices": torch.tensor([[2], [0]], dtype=I32)},
    "int_has_initial_state": {"has_initial_state": torch.tensor([1, 0], dtype=I32)},
    "int64_query_start_loc": {"query_start_loc": torch.tensor([0, 4, TOKENS])},
    # a pad_slot_id a slot could have: aiter skips the sequence in it
    "pad_slot_id_a_slot": {"pad_slot_id": 0},
}


@pytest.mark.parametrize("changes", list(DECLINED.values()), ids=list(DECLINED))
def test_a_call_outside_aklrs_contract_is_declined_before_aklr_is_asked(monkeypatch, aklr_serves, changes):
    asked = []
    monkeypatch.setattr(conv.kimi_k3_kda_conv, "serving", lambda *a, **k: asked.append(a) or KERNEL)
    assert _select(_call(**changes)) is None
    assert not asked


# each a function of no argument: the changes are built per test, which may write the state
SERVED = {
    # aiter's schedule: its chunk sizes, its memoized chunk schedule; a provider reads query_start_loc
    "aiters_schedule": lambda: {"block_m": 8, "block_n": 256, "num_warps": 8, "metadata": SimpleNamespace()},
    "swish": lambda: {"activation": "swish"},
    # x and conv_states of one dtype: the same result either way
    "preserve_input_dtype": lambda: {"preserve_input_dtype": True},
    "no_pad_slot": lambda: {"pad_slot_id": None},
    # with speculative decoding vLLM's slot keeps more rows (10 for Kimi-K3 with 7 drafts): the prefill's are
    # the first width - 1
    "speculative_rows": lambda: {"conv_states": _state(state_len=10)},
    # a resumed sequence shorter than the state: its slot keeps the state's newest inputs, then its own
    "short_sequence": lambda: {"query_start_loc": torch.tensor([0, 2, TOKENS], dtype=I32)},
}


@pytest.mark.parametrize("changes", list(SERVED.values()), ids=list(SERVED))
def test_a_call_within_aklrs_contract_is_offered(aklr_serves, changes):
    call = _call(**changes())
    kernel, run = _select(call)
    assert kernel is KERNEL
    inspect.signature(run).bind(*_args(call))


def test_a_call_aklr_falls_back_on_is_declined(monkeypatch, aklr_serves):
    _answer(monkeypatch, FallbackReason.NO_WORKLOAD)
    assert _select(_call()) is None


@pytest.mark.parametrize("changes", [dict, *SERVED.values()], ids=["vllms_call", *SERVED])
def test_the_run_computes_aiters_q_k_v_and_state(monkeypatch, aklr_serves, changes):
    """AKLR's op on the operands run hands it gives aiter's q, k and v, and leaves aiter's conv_states
    as aiter's kernel does (the speculative rows and the slots no sequence names unchanged)."""
    torch.manual_seed(0)
    op = _Op()
    monkeypatch.setattr(conv, "kimi_k3_kda_conv", op)
    call = _call(**changes())
    expected, expected_states = _aiters_result(call)
    _, run = _select(call)
    result = run(*_args(call))
    assert isinstance(result, tuple)
    for got, want in zip(result, expected, strict=True):
        torch.testing.assert_close(got, want)
    assert torch.equal(call["conv_states"], expected_states)
    (handed,) = op.calls
    assert handed["storage"] == "strided_rows" and handed["weight"] is call["weight"]
    assert handed["x"].shape == (TOKENS, 3, PROJ) and handed["x"].data_ptr() == call["x"].data_ptr()
    assert handed["conv_state"].shape == (SLOTS, WIDTH - 1, DIM)
    assert handed["conv_state"].data_ptr() == call["conv_states"].data_ptr()


def test_index_vectors_vllm_hands_as_a_column_go_to_aklr_dense(monkeypatch, aklr_serves):
    """vLLM's cache_indices can be a column of a wider table (stride 8 seen on Kimi-K3), a one-sequence
    one too, which ``.contiguous()`` leaves as it is: AKLR's kernel reads dense index vectors."""
    for n_seqs in (2, 1):
        op = _Op()
        monkeypatch.setattr(conv, "kimi_k3_kda_conv", op)
        table = torch.zeros(n_seqs, 8, dtype=I32)
        table[:, 0] = torch.tensor([2, 0][:n_seqs], dtype=I32)
        flags = torch.ones(n_seqs, 8, dtype=torch.bool)
        offsets = torch.zeros(n_seqs + 1, 2, dtype=I32)
        offsets[:, 0] = torch.tensor([0, 4, TOKENS][:n_seqs] + [TOKENS], dtype=I32)
        call = _call(cache_indices=table[:, 0], has_initial_state=flags[:, 0], query_start_loc=offsets[:, 0])
        _, run = _select(call)
        run(*_args(call))
        (handed,) = op.calls
        given = {"state_indices": table[:, 0], "has_initial_state": flags[:, 0], "cu_seqlens": offsets[:, 0]}
        for name, vector in given.items():
            assert handed[name].stride() == (1,) and torch.equal(handed[name], vector), name


def test_a_fallback_after_select_accepted_raises(monkeypatch, aklr_serves):
    """aiter does not fall back once run started; a fallback there is a bug, not a decline."""
    call = _call()
    _, run = _select(call)
    monkeypatch.setattr(conv, "kimi_k3_kda_conv", lambda *a, **k: FallbackReason.NOT_WARMED)
    with pytest.raises(RuntimeError, match="not_warmed"):
        run(*_args(call))


class _EntryPoint:
    def __init__(self, name, target):
        self.name, self.target = name, target

    def load(self):
        return self.target


def _causal_conv1d_split_qkv(
    x,
    weight,
    bias,
    conv_states,
    query_start_loc,
    k_dim,
    v_dim,
    cache_indices=None,
    has_initial_state=None,
    activation="silu",
    pad_slot_id=-1,
    block_m=64,
    block_n=32,
    num_warps=4,
    metadata=None,
    preserve_input_dtype=False,
):
    """aiter's routed function as it declares itself; its body must not run for a call AKLR serves."""
    raise AssertionError("aiter's kernel ran for a call AKLR serves")


@pytest.fixture
def aiter_plugin(monkeypatch):
    """aiter's plugin interface with AKLR installed and enabled; the conv routed to the stand-in above
    where this aiter does not route it itself (the plugin interface alone, or its module not imported)."""
    plugin = pytest.importorskip("aiter.plugin")
    plugin._reset_for_testing()
    monkeypatch.setattr(plugin, "_entry_points", lambda: (_EntryPoint("aklr", provider.provider),))
    monkeypatch.setattr(provider, "ready", lambda: True)
    route = None
    if "causal_conv1d_split_qkv" not in plugin._routed_ops:
        route = plugin.route(op="causal_conv1d_split_qkv")(_causal_conv1d_split_qkv)
    provider.enable("test")
    yield SimpleNamespace(plugin=plugin, route=route)
    plugin._reset_for_testing()


def test_aiter_routes_the_conv_aklr_serves_to_aklrs_run(monkeypatch, aiter_plugin):
    """The real dispatch, called as vLLM's KDA prefill calls it: defaults applied, the run checked
    against the contract, then called where the route is the test's own (aiter's would build its
    kernel); what comes back is aiter's result."""
    torch.manual_seed(0)
    monkeypatch.setattr(conv, "kimi_k3_kda_conv", _Op())
    call = _call()
    expected, expected_states = _aiters_result(call)
    names = AITER_CONTRACTS["causal_conv1d_split_qkv"][1]
    required = [call[name] for name in names[:7]]
    by_name = {name: call[name] for name in ("cache_indices", "has_initial_state", "activation", "block_m")}
    plugin = aiter_plugin.plugin
    decision = plugin.explain("causal_conv1d_split_qkv", *required, **by_name).decision
    assert (decision.route, decision.provider, decision.kernel_id) == ("provider", "aklr", KERNEL.kernel_ref)
    if aiter_plugin.route is not None:
        result = aiter_plugin.route(*required, **by_name)
        assert plugin.last_decision().route == "provider"
        for got, want in zip(result, expected, strict=True):
            torch.testing.assert_close(got, want)
        assert torch.equal(call["conv_states"], expected_states)
    monkeypatch.setattr(conv.kimi_k3_kda_conv, "serving", lambda *a, **k: FallbackReason.NO_WORKLOAD)
    assert plugin.explain("causal_conv1d_split_qkv", *required, **by_name).decision.route == "native"
