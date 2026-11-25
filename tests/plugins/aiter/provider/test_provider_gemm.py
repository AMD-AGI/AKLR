# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The GEMM selectors of AKLR's aiter provider for aiter's ``gemm_a16w16`` (tuned_gemm),
``gemm_a16w16_activation`` (Triton ``gemm_a16w16``), ``gemm_a8w8_blockscale_out`` (Triton
``gemm_a8w8_blockscale``) and ``gemm_a8w8_bpreshuffle`` (the per-token fp8 GEMM on a preshuffled
weight): what each takes, what it declines, the AKLR call its run makes, and, with an aiter that has
the plugin interface, the route aiter takes."""

import importlib
import inspect
from types import SimpleNamespace

import pytest
import torch

from aklr.core.runtime.builder import RunnableMetadata
from aklr.core.runtime.dispatch import FallbackReason

provider = importlib.import_module("aklr.plugins.aiter.provider")
base = importlib.import_module("aklr.plugins.aiter.provider._base")
gemm = importlib.import_module("aklr.plugins.aiter.provider.gemm")

FP8, BF16, FP16, F32, I8 = torch.float8_e4m3fn, torch.bfloat16, torch.float16, torch.float32, torch.int8
E5M2 = torch.float8_e5m2
E8M0 = getattr(torch, "float8_e8m0fnu", torch.uint8)
KERNEL = RunnableMetadata(
    build_type="flydsl", contract_id="any@1.0", schema_name="any_schema", content_hash="0123456789abcdef0123"
)

# aiter's contracts (tests/plugin_contracts.json of the aiter change that routes them): version,
# positional parameters
AITER_CONTRACTS = {
    "gemm_a16w16": (1, ("A", "B", "bias", "otype", "scale_a", "scale_b", "scale_c", "out")),
    "gemm_a16w16_activation": (
        1,
        (
            "x",
            "w",
            "bias",
            "dtype",
            "y",
            "config",
            "activation",
            "skip_reduce",
            "kernel_type",
            "persistent",
        ),
    ),
    "gemm_a8w8_blockscale_out": (
        1,
        ("x", "w", "x_scale", "w_scale", "dtype", "y", "config", "skip_reduce", "kernel_type"),
    ),
    "gemm_a8w8_bpreshuffle": (1, ("XQ", "WQ", "x_scale", "w_scale", "bias", "dtype", "check")),
}
# the AKLR apis the GEMM selectors ask
APIS = ("gemm", "gemm_a8w8", "gemm_a8w8_blockwise_scaled")


def _served_calls():
    """A call AKLR's contract takes, per aiter op, by parameter name, defaults applied."""
    z = torch.zeros
    return {
        "gemm_a16w16": {
            "A": z(4, 128, dtype=BF16),
            "B": z(256, 128, dtype=BF16),
            "bias": None,
            "otype": None,
            "scale_a": None,
            "scale_b": None,
            "scale_c": None,
            "out": None,
        },
        "gemm_a16w16_activation": {
            "x": z(4, 128, dtype=BF16),
            "w": z(256, 128, dtype=BF16),
            "bias": None,
            "dtype": BF16,
            "y": None,
            "config": None,
            "activation": None,
            "skip_reduce": False,
            "kernel_type": "bandwidth_bound",
            "persistent": False,
        },
        "gemm_a8w8_blockscale_out": {
            "x": z(4, 256, dtype=FP8),
            "w": z(256, 256, dtype=FP8),
            "x_scale": z(4, 2, dtype=F32),
            "w_scale": z(2, 2, dtype=F32),
            "dtype": BF16,
            "y": None,
            "config": None,
            "skip_reduce": False,
            "kernel_type": "bandwidth_bound",
        },
        # WQ as vLLM hands it: a Parameter re-wrapping the shuffled weight, without aiter's is_shuffled
        "gemm_a8w8_bpreshuffle": {
            "XQ": z(4, 256, dtype=FP8),
            "WQ": torch.nn.Parameter(z(256, 256, dtype=FP8), requires_grad=False),
            "x_scale": z(4, 1, dtype=F32),
            "w_scale": z(256, 1, dtype=F32),
            "bias": None,
            "dtype": BF16,
            "check": False,
        },
    }


def _args(op: str, **changes):
    call = {**_served_calls()[op], **changes}
    return tuple(call[name] for name in AITER_CONTRACTS[op][1])


def _select(op: str, **changes):
    return base.SELECTORS[op][1](_args(op, **changes), {})


@pytest.fixture
def aklr_serves(monkeypatch):
    """Every AKLR api the GEMM selectors ask answers with KERNEL; a candidate is (kernel, run)."""
    monkeypatch.setattr(
        gemm, "offer", lambda serving, run: None if isinstance(serving, FallbackReason) else (serving, run)
    )
    _answer(monkeypatch, KERNEL)


def _answer(monkeypatch, serving):
    for name in APIS:
        monkeypatch.setattr(getattr(gemm, name), "serving", lambda *a, **k: serving)


def _api(monkeypatch, name, result=None):
    """Replace the AKLR api ``name`` the GEMM selectors call; the calls its ``serving`` and the call
    itself get are kept as (args, kwargs), and the call returns ``result`` (else its ``out``)."""
    calls = SimpleNamespace(serving=[], run=[])

    def api(*args, **kwargs):
        calls.run.append((args, kwargs))
        return kwargs["out"] if result is None else result

    def serving(*args, **kwargs):
        calls.serving.append((args, kwargs))
        return KERNEL

    api.serving = serving
    monkeypatch.setattr(gemm, name, api)
    return calls


def test_each_gemm_selector_serves_its_aiter_contract_at_its_version():
    for op, (version, _) in AITER_CONTRACTS.items():
        assert base.SELECTORS[op][0] == version, op
        assert provider.Provider.op_versions[op] == {version}


@pytest.mark.parametrize("op", list(AITER_CONTRACTS))
def test_a_served_call_offers_a_run_taking_aiters_contract(aklr_serves, op):
    """aiter hands run every positional parameter: run has aiter's parameters, by name and order."""
    kernel, run = _select(op)
    assert kernel is KERNEL
    params = inspect.signature(run).parameters
    assert tuple(params) == AITER_CONTRACTS[op][1]
    assert all(p.kind is p.POSITIONAL_OR_KEYWORD for p in params.values())
    inspect.signature(run).bind(*_args(op))


DECLINED = [
    ("gemm_a16w16", {"scale_a": torch.ones(1)}),
    ("gemm_a16w16", {"scale_b": torch.ones(1)}),
    ("gemm_a16w16", {"scale_c": torch.ones(1)}),
    ("gemm_a16w16", {"B": SimpleNamespace(is_shuffled=True, shape=(256, 128))}),
    # aiter raises on an out of another dtype than otype
    ("gemm_a16w16", {"out": torch.zeros(4, 256, dtype=F32)}),
    ("gemm_a16w16", {"otype": F32, "out": torch.zeros(4, 256, dtype=BF16)}),
    # leading dims that cannot be viewed as rows: aiter runs F.linear
    ("gemm_a16w16", {"A": torch.zeros(4, 2, 128, dtype=BF16).transpose(0, 1)}),
    ("gemm_a16w16_activation", {"activation": "silu"}),
    ("gemm_a16w16_activation", {"skip_reduce": True}),
    ("gemm_a16w16_activation", {"dtype": None}),
    ("gemm_a8w8_blockscale_out", {"skip_reduce": True}),
    ("gemm_a8w8_blockscale_out", {"dtype": None}),
    ("gemm_a8w8_blockscale_out", {"x_scale": torch.zeros(4, 8, dtype=E8M0)}),
    ("gemm_a8w8_blockscale_out", {"w_scale": torch.zeros(2, 8, dtype=torch.uint8)}),
    # a transposed view of the scales, which aiter's kernel reads through its strides; AKLR reads
    # row-major bytes only
    ("gemm_a8w8_blockscale_out", {"x_scale": torch.zeros(2, 4, dtype=F32).t()}),
    # aiter's debug flag, and a bias aiter rejects
    ("gemm_a8w8_bpreshuffle", {"check": True}),
    ("gemm_a8w8_bpreshuffle", {"bias": torch.zeros(256, dtype=F32)}),
    # scales other than fp32 per token and per output column, per-tensor ones included
    ("gemm_a8w8_bpreshuffle", {"x_scale": torch.zeros(1, dtype=F32)}),
    ("gemm_a8w8_bpreshuffle", {"w_scale": torch.zeros(1, dtype=F32)}),
    ("gemm_a8w8_bpreshuffle", {"x_scale": torch.zeros(4, 2, dtype=F32)}),
    ("gemm_a8w8_bpreshuffle", {"x_scale": torch.zeros(4, 1, dtype=BF16)}),
    ("gemm_a8w8_bpreshuffle", {"w_scale": torch.zeros(256, 1, dtype=BF16)}),
    # a weight wider than K, which aiter reads as K zero-padded
    ("gemm_a8w8_bpreshuffle", {"WQ": torch.zeros(256, 384, dtype=FP8)}),
    # operand dtypes aiter's kernel does not take: e5m2, or a weight of another dtype than XQ
    (
        "gemm_a8w8_bpreshuffle",
        {"XQ": torch.zeros(4, 256, dtype=E5M2), "WQ": torch.zeros(256, 256, dtype=E5M2)},
    ),
    ("gemm_a8w8_bpreshuffle", {"XQ": torch.zeros(4, 256, dtype=I8)}),
    # aiter's int8 path, whose own rules (bf16 out only, K from 256) AKLR's contract does not state
    ("gemm_a8w8_bpreshuffle", {"XQ": torch.zeros(4, 256, dtype=I8), "WQ": torch.zeros(256, 256, dtype=I8)}),
    ("gemm_a8w8_bpreshuffle", {"XQ": torch.zeros(2, 2, 256, dtype=FP8)}),
]


@pytest.mark.parametrize(("op", "changes"), DECLINED)
def test_a_call_outside_aklrs_contract_is_declined_before_aklr_is_asked(
    monkeypatch, aklr_serves, op, changes
):
    asked = []
    for name in APIS:
        monkeypatch.setattr(getattr(gemm, name), "serving", lambda *a, **k: asked.append(a) or KERNEL)
    assert _select(op, **changes) is None
    assert not asked


SERVED = [
    ("gemm_a16w16", {"otype": BF16, "out": torch.zeros(4, 256, dtype=BF16)}),
    ("gemm_a16w16", {"A": torch.zeros(2, 3, 128, dtype=BF16)}),
    ("gemm_a16w16_activation", {"config": '{"BLOCK_SIZE_M": 64}', "kernel_type": "compute_bound"}),
    ("gemm_a16w16_activation", {"persistent": True, "dtype": None, "y": torch.zeros(4, 256, dtype=FP16)}),
    ("gemm_a8w8_blockscale_out", {"config": {"BLOCK_SIZE_M": 64}, "y": torch.zeros(4, 256, dtype=BF16)}),
    # a (1, k) transposed view is the row-major bytes
    ("gemm_a8w8_blockscale_out", {"x": torch.zeros(1, 256, dtype=FP8), "x_scale": torch.zeros(2, 1).t()}),
    ("gemm_a8w8_bpreshuffle", {"dtype": FP16}),
    # the scales squeezed, which AKLR's contract takes too; one token's per-tensor scale is per token
    ("gemm_a8w8_bpreshuffle", {"x_scale": torch.zeros(4, dtype=F32), "w_scale": torch.zeros(256, dtype=F32)}),
    ("gemm_a8w8_bpreshuffle", {"XQ": torch.zeros(1, 256, dtype=FP8), "x_scale": torch.zeros(1, dtype=F32)}),
]


@pytest.mark.parametrize(("op", "changes"), SERVED)
def test_a_call_within_aklrs_contract_is_offered(aklr_serves, op, changes):
    kernel, run = _select(op, **changes)
    assert kernel is KERNEL
    inspect.signature(run).bind(*_args(op, **changes))


@pytest.mark.parametrize("op", list(AITER_CONTRACTS))
def test_a_call_aklr_falls_back_on_is_declined(monkeypatch, aklr_serves, op):
    _answer(monkeypatch, FallbackReason.NO_WORKLOAD)
    assert _select(op) is None


def test_the_tuned_gemm_hands_aklr_rows_and_returns_aiters_shape(monkeypatch, aklr_serves):
    """aiter's tuned GEMM computes on a's leading dims as rows and returns them; ``out`` is
    viewed (n columns) and comes back as the result, which ``gemm_a16w16_out`` checks."""
    calls = _api(monkeypatch, "gemm")
    a = torch.zeros(2, 3, 128, dtype=BF16)
    b = torch.zeros(256, 128, dtype=BF16)
    bias = torch.zeros(256, dtype=BF16)
    out = torch.zeros(2, 3, 256, dtype=BF16)
    args = _args("gemm_a16w16", A=a, B=b, bias=bias, out=out)
    _, run = base.SELECTORS["gemm_a16w16"][1](args, {})
    result = run(*args)
    assert result.shape == (2, 3, 256) and result.data_ptr() == out.data_ptr()
    for got_args, got_kwargs in (calls.serving[-1], calls.run[-1]):
        rows, weight = got_args
        assert rows.shape == (6, 128) and rows.data_ptr() == a.data_ptr() and weight is b
        assert got_kwargs["bias"] is bias and got_kwargs["out_dtype"] is None
        assert got_kwargs["out"].shape == (6, 256) and got_kwargs["out"].data_ptr() == out.data_ptr()

    calls = _api(monkeypatch, "gemm", result=torch.ones(4, 256, dtype=F32))
    args = _args("gemm_a16w16", otype=F32)
    _, run = base.SELECTORS["gemm_a16w16"][1](args, {})
    assert run(*args).shape == (4, 256)
    assert calls.run[-1][1] == {"bias": None, "out_dtype": F32, "out": None}


def test_the_triton_gemm_hands_aklr_y_and_its_dtype(monkeypatch, aklr_serves):
    """``y`` is the output; its dtype is the result's (aiter's ``dtype`` then only sizes a new y)."""
    calls = _api(monkeypatch, "gemm")
    y = torch.zeros(4, 256, dtype=FP16)
    args = _args("gemm_a16w16_activation", dtype=BF16, y=y, config='{"NUM_KSPLIT": 2}')
    _, run = base.SELECTORS["gemm_a16w16_activation"][1](args, {})
    assert run(*args) is y
    x, w = args[:2]
    for got_args, got_kwargs in (calls.serving[-1], calls.run[-1]):
        assert got_args[0] is x and got_args[1] is w
        assert got_kwargs == {"bias": None, "out_dtype": FP16, "out": y}


def test_the_triton_blockscale_gemm_hands_aklr_row_major_operands(monkeypatch, aklr_serves):
    calls = _api(monkeypatch, "gemm_a8w8_blockwise_scaled")
    y = torch.zeros(4, 256, dtype=BF16)
    args = _args("gemm_a8w8_blockscale_out", y=y)
    _, run = base.SELECTORS["gemm_a8w8_blockscale_out"][1](args, {})
    assert run(*args) is y
    for got_args, got_kwargs in (calls.serving[-1], calls.run[-1]):
        assert all(got is given for got, given in zip(got_args, args[:4]))
        assert got_kwargs == {"out_dtype": BF16, "out": y, "operand_format": "none:none"}

    calls = _api(monkeypatch, "gemm_a8w8_blockwise_scaled", result="new")
    args = _args("gemm_a8w8_blockscale_out", dtype=FP16)
    _, run = base.SELECTORS["gemm_a8w8_blockscale_out"][1](args, {})
    assert run(*args) == "new"
    assert calls.run[-1][1] == {"out_dtype": FP16, "out": None, "operand_format": "none:none"}


def _marked_shuffled(w):
    w.is_shuffled = True
    return w


@pytest.mark.parametrize(
    "weight",
    [
        torch.nn.Parameter(torch.zeros(256, 256, dtype=FP8), requires_grad=False),
        torch.zeros(256, 256, dtype=FP8),
    ],
    ids=["vllms_parameter", "plain"],
)
def test_the_preshuffled_gemm_hands_aklr_its_weight_marked_shuffled(monkeypatch, aklr_serves, weight):
    """aiter's op reads WQ as shuffled whatever it carries; AKLR's contract tells a shuffled b by its
    is_shuffled, which vLLM's Parameter drops. AKLR gets a view of WQ that carries it; the caller's
    tensor is left as it was."""
    calls = _api(monkeypatch, "gemm_a8w8", result="new")
    args = _args("gemm_a8w8_bpreshuffle", WQ=weight, dtype=FP16)
    _, run = base.SELECTORS["gemm_a8w8_bpreshuffle"][1](args, {})
    assert run(*args) == "new"
    xq, _, x_scale, w_scale = args[:4]
    for got_args, got_kwargs in (calls.serving[-1], calls.run[-1]):
        a, b, a_scale, b_scale = got_args
        assert a is xq and a_scale is x_scale and b_scale is w_scale
        assert b.is_shuffled is True and b.data_ptr() == weight.data_ptr() and b.shape == weight.shape
        assert got_kwargs == {"out_dtype": FP16}
    assert not hasattr(weight, "is_shuffled")


def test_a_weight_still_marked_shuffled_is_handed_over_as_it_is(monkeypatch, aklr_serves):
    calls = _api(monkeypatch, "gemm_a8w8", result="new")
    weight = _marked_shuffled(torch.zeros(256, 256, dtype=FP8))
    args = _args("gemm_a8w8_bpreshuffle", WQ=weight)
    _, run = base.SELECTORS["gemm_a8w8_bpreshuffle"][1](args, {})
    run(*args)
    assert calls.serving[-1][0][1] is weight and calls.run[-1][0][1] is weight


def test_a_fallback_after_select_accepted_raises(monkeypatch, aklr_serves):
    """aiter does not fall back once run started; a fallback there is a bug, not a decline."""
    _api(monkeypatch, "gemm", result=FallbackReason.NOT_WARMED)
    _, run = _select("gemm_a16w16")
    with pytest.raises(RuntimeError, match="not_warmed"):
        run(*_args("gemm_a16w16"))


class _EntryPoint:
    def __init__(self, name, target):
        self.name, self.target = name, target

    def load(self):
        return self.target


# aiter's routed functions, signatures as aiter declares them (the bodies never run: explain only asks)
def _gemm_a16w16(A, B, bias=None, otype=None, scale_a=None, scale_b=None, scale_c=None, out=None):
    raise AssertionError("explain runs nothing")


def _gemm_a16w16_routed(
    x,
    w,
    bias=None,
    dtype=torch.bfloat16,
    y=None,
    config=None,
    activation=None,
    skip_reduce=False,
    kernel_type="bandwidth_bound",
    persistent=False,
):
    raise AssertionError("explain runs nothing")


def _gemm_a8w8_blockscale_routed(
    x,
    w,
    x_scale,
    w_scale,
    dtype=torch.bfloat16,
    y=None,
    config=None,
    skip_reduce=False,
    kernel_type="bandwidth_bound",
):
    raise AssertionError("explain runs nothing")


def _gemm_a8w8_bpreshuffle(XQ, WQ, x_scale, w_scale, bias=None, dtype=torch.bfloat16, check=False):
    raise AssertionError("explain runs nothing")


_STAND_INS = {
    "gemm_a16w16": _gemm_a16w16,
    "gemm_a16w16_activation": _gemm_a16w16_routed,
    "gemm_a8w8_blockscale_out": _gemm_a8w8_blockscale_routed,
    "gemm_a8w8_bpreshuffle": _gemm_a8w8_bpreshuffle,
}


@pytest.fixture
def aiter_plugin(monkeypatch):
    """aiter's plugin interface with AKLR installed and the GEMM ops routed."""
    plugin = pytest.importorskip("aiter.plugin")
    plugin._reset_for_testing()
    monkeypatch.setattr(plugin, "_entry_points", lambda: (_EntryPoint("aklr", provider.provider),))
    monkeypatch.setattr(provider, "ready", lambda: True)
    for op, fn in _STAND_INS.items():
        if op not in plugin._routed_ops:  # an aiter whose module routing it is not imported
            plugin.route(op=op, version=AITER_CONTRACTS[op][0])(fn)
    yield plugin
    plugin._reset_for_testing()


@pytest.mark.parametrize(
    ("op", "call_args", "call_kwargs"),
    [
        # tgemm.mm's spelling: keywords after the two operands
        (
            "gemm_a16w16",
            (torch.zeros(4, 128, dtype=BF16), torch.zeros(256, 128, dtype=BF16)),
            {"otype": BF16},
        ),
        # gemm_a16w16_'s spelling: every parameter positional
        ("gemm_a16w16_activation", _args("gemm_a16w16_activation"), {}),
        (
            "gemm_a8w8_blockscale_out",
            _args("gemm_a8w8_blockscale_out")[:4],
            {"y": torch.zeros(4, 256, dtype=BF16)},
        ),
        # vLLM's spelling: the four operands, the dtype by keyword
        ("gemm_a8w8_bpreshuffle", _args("gemm_a8w8_bpreshuffle")[:4], {"dtype": BF16}),
    ],
)
def test_aiter_routes_a_gemm_aklr_serves_to_aklrs_kernel(
    monkeypatch, aiter_plugin, op, call_args, call_kwargs
):
    """aiter fills in the defaults and hands the selector every parameter; AKLR's answer picks the route."""
    provider.enable("test")
    _answer(monkeypatch, KERNEL)
    decision = aiter_plugin.explain(op, *call_args, **call_kwargs).decision
    assert (decision.route, decision.provider, decision.kernel_id) == ("provider", "aklr", KERNEL.kernel_ref)
    _answer(monkeypatch, FallbackReason.NO_WORKLOAD)
    assert aiter_plugin.explain(op, *call_args, **call_kwargs).decision.route == "native"
