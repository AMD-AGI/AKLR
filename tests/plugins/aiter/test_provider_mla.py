# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""AKLR as aiter's kernel plugin for aiter's MLA and mHC routes: mla_decode_fwd, mla_paged_decode,
fp8_paged_mqa_logits, mla_sparse_prefill_v4 and mhc_fused_post_pre. What each selector takes, what it
declines, the AKLR call its run makes, and, with an aiter that has the plugin interface, that aiter
hands such a call to AKLR."""

import importlib
import inspect
from types import SimpleNamespace

import pytest
import torch

from aklr.core.runtime.builder import RunnableMetadata
from aklr.core.runtime.dispatch import FallbackReason

provider = importlib.import_module("aklr.plugins.aiter.provider")
base = importlib.import_module("aklr.plugins.aiter.provider._base")
mla = importlib.import_module("aklr.plugins.aiter.provider.mla")
mla_prefill = importlib.import_module("aklr.plugins.aiter.provider.mla_prefill")
mhc = importlib.import_module("aklr.plugins.aiter.provider.mhc")

FP8, BF16, F32, I32 = torch.float8_e4m3fn, torch.bfloat16, torch.float32, torch.int32
KERNEL = RunnableMetadata(
    build_type="flydsl", contract_id="any@1.0", schema_name="any_schema", content_hash="0123456789abcdef0123"
)
_REQUIRED = inspect.Parameter.empty

# aiter's contracts (ROCm/aiter tests/plugin_contracts.json, kyle/aklr-routes-mla): version, then each
# positional parameter with its default
AITER_SIGNATURES = {
    "mla_decode_fwd": (
        1,
        (
            ("q", _REQUIRED),
            ("kv_buffer", _REQUIRED),
            ("o", _REQUIRED),
            ("qo_indptr", _REQUIRED),
            ("kv_indptr", _REQUIRED),
            ("kv_indices", _REQUIRED),
            ("kv_last_page_lens", _REQUIRED),
            ("max_seqlen_q", _REQUIRED),
            ("page_size", 1),
            ("nhead_kv", 1),
            ("sm_scale", None),
            ("logit_cap", 0.0),
            ("num_kv_splits", None),
            ("num_kv_splits_indptr", None),
            ("work_meta_data", None),
            ("work_indptr", None),
            ("work_info_set", None),
            ("reduce_indptr", None),
            ("reduce_final_map", None),
            ("reduce_partial_map", None),
            ("q_scale", None),
            ("kv_scale", None),
            ("intra_batch_mode", False),
            ("return_logits", False),
            ("return_lse", False),
            ("g_kv_indptr", None),
            ("cp_world_size", 1),
            ("cp_rank", 0),
            ("causal", True),
            ("kv_layout", "token_major"),
            ("seq_lens", None),
            ("page_table", None),
            ("kv_csr_layout", "page"),
            ("fast_mode", True),
        ),
    ),
    "mla_paged_decode": (
        1,
        (
            ("q_nope", _REQUIRED),
            ("q_pe", _REQUIRED),
            ("kv_c", _REQUIRED),
            ("o", _REQUIRED),
            ("page_table", _REQUIRED),
            ("seq_info", _REQUIRED),
            ("sm_scale", _REQUIRED),
            ("k_pe", None),
            ("kv_pe_offset", 512),
            ("use_2d_view", True),
            ("kv_scale", 1.0),
            ("min_kv_seq_len", 1),
            ("return_lse", False),
        ),
    ),
    "fp8_paged_mqa_logits": (
        1,
        (
            ("q_fp8", _REQUIRED),
            ("kv_cache", _REQUIRED),
            ("weights", _REQUIRED),
            ("out_logits", _REQUIRED),
            ("context_lens", _REQUIRED),
            ("kv_indices", _REQUIRED),
            ("max_model_len", _REQUIRED),
            ("Preshuffle", False),
            ("KVBlockSize", 1),
            ("ChunkK", 256),
            ("TotalCuCount", None),
            ("WavePerEU", 2),
            ("VarCtxSchedule", None),
            ("SplitKV", None),
            ("PersistentSchedule", None),
        ),
    ),
    "mla_sparse_prefill_v4": (
        1,
        (
            ("q_nope", _REQUIRED),
            ("q_rope", _REQUIRED),
            ("unified_kv_nope", _REQUIRED),
            ("unified_kv_rope", _REQUIRED),
            ("kv_indices_prefix", _REQUIRED),
            ("kv_indptr_prefix", _REQUIRED),
            ("kv_nope", _REQUIRED),
            ("kv_rope", _REQUIRED),
            ("kv_indices_extend", _REQUIRED),
            ("kv_indptr_extend", _REQUIRED),
            ("attn_sink", _REQUIRED),
            ("softmax_scale", _REQUIRED),
            ("out", None),
            ("num_kv_splits", None),
        ),
    ),
    "mhc_fused_post_pre": (
        1,
        (
            ("layer_input", _REQUIRED),
            ("residual_in", _REQUIRED),
            ("post_layer_mix", _REQUIRED),
            ("comb_res_mix", _REQUIRED),
            ("fn", _REQUIRED),
            ("hc_scale", _REQUIRED),
            ("hc_base", _REQUIRED),
            ("rms_eps", 1e-6),
            ("hc_pre_eps", 1e-6),
            ("hc_sinkhorn_eps", 1e-6),
            ("hc_post_mult_value", 1.0),
            ("sinkhorn_repeat", 20),
            ("norm_weight", None),
            ("norm_eps", 1e-6),
            ("w_preshuffle_bf16", False),
            ("res_preshuffle", False),
        ),
    ),
}
# the entries test_provider.AITER_CONTRACTS takes for these ops
AITER_CONTRACTS = {
    op: (version, tuple(name for name, _ in params)) for op, (version, params) in AITER_SIGNATURES.items()
}


def _split_q(*lead):
    """q_nope and q_pe as the frameworks pass them: two adjacent views of one q of 576-value rows."""
    q = torch.zeros(*lead, 576, dtype=BF16)
    return q[..., :512], q[..., 512:]


def _calls():
    """A call AKLR's contract takes, per case, by aiter's parameter names, every default given."""
    z = torch.zeros
    q_nope, q_pe = _split_q(2, 4)
    mtp_nope, mtp_pe = _split_q(2, 3, 4)
    paged = {
        "q_nope": q_nope,
        "q_pe": q_pe,
        "kv_c": z(8, 576, dtype=FP8),
        "o": z(2, 4, 512, dtype=BF16),
        "page_table": z(2, 8, dtype=I32),
        "seq_info": torch.full((2,), 8, dtype=I32),
        "sm_scale": 0.1,
        "k_pe": None,
        "kv_pe_offset": 512,
        "use_2d_view": True,
        "kv_scale": 1.0,
        "min_kv_seq_len": 1,
        "return_lse": False,
    }
    return {
        "mla_decode_fwd": {
            **{name: default for name, default in AITER_SIGNATURES["mla_decode_fwd"][1]},
            "q": z(2, 16, 576, dtype=FP8),
            "kv_buffer": z(8, 1, 1, 576, dtype=FP8),
            "o": z(2, 16, 512, dtype=BF16),
            "qo_indptr": torch.arange(3, dtype=I32),
            "kv_indptr": torch.tensor([0, 3, 8], dtype=I32),
            "kv_indices": z(8, dtype=I32),
            "kv_last_page_lens": torch.ones(2, dtype=I32),
            "max_seqlen_q": 1,
            "num_kv_splits": 64,
            "work_meta_data": z(4, dtype=I32),
            "q_scale": torch.ones(1),
            "kv_scale": torch.ones(1),
            "return_lse": True,
        },
        "mla_paged_decode": paged,
        "mla_paged_decode_mtp": {
            **paged,
            "q_nope": mtp_nope,
            "q_pe": mtp_pe,
            "kv_c": z(8, 576, dtype=BF16),
            "o": z(2, 3, 4, 512, dtype=BF16),
            "page_table": z(16, dtype=I32),
            "seq_info": torch.tensor([0, 8, 16], dtype=I32),
            "use_2d_view": False,
            "return_lse": True,
        },
        "fp8_paged_mqa_logits": {
            "q_fp8": z(2, 1, 4, 128, dtype=FP8),
            "kv_cache": z(4, 64, 1, 132, dtype=FP8),
            "weights": z(2, 4),
            "out_logits": z(2, 128),
            "context_lens": torch.ones(2, 1, dtype=I32),
            "kv_indices": z(2, 2, dtype=I32),
            "max_model_len": 128,
            "Preshuffle": True,
            "KVBlockSize": 64,
            "ChunkK": 256,
            "TotalCuCount": None,
            "WavePerEU": 2,
            "VarCtxSchedule": None,
            "SplitKV": None,
            "PersistentSchedule": None,
        },
        "mla_sparse_prefill_v4": {
            "q_nope": z(2, 16, 512, dtype=FP8),
            "q_rope": z(2, 16, 64, dtype=BF16),
            "unified_kv_nope": z(4, 512, dtype=FP8),
            "unified_kv_rope": z(4, 64, dtype=BF16),
            "kv_indices_prefix": z(4, dtype=I32),
            "kv_indptr_prefix": torch.tensor([0, 2, 4], dtype=I32),
            "kv_nope": z(4, 512, dtype=FP8),
            "kv_rope": z(4, 64, dtype=BF16),
            "kv_indices_extend": z(4, dtype=I32),
            "kv_indptr_extend": torch.tensor([0, 2, 4], dtype=I32),
            "attn_sink": z(16),
            "softmax_scale": 512**-0.5,
            "out": z(2, 16, 512, dtype=BF16),
            "num_kv_splits": 4,
        },
        "mhc_fused_post_pre": {
            "layer_input": z(4, 8, dtype=BF16),
            "residual_in": z(4, 4, 8, dtype=BF16),
            "post_layer_mix": z(4, 4, 1),
            "comb_res_mix": z(4, 4, 4),
            "fn": z(24, 32),
            "hc_scale": z(3),
            "hc_base": z(24),
            "rms_eps": 1e-6,
            "hc_pre_eps": 1e-6,
            "hc_sinkhorn_eps": 1e-6,
            "hc_post_mult_value": 2.0,
            "sinkhorn_repeat": 20,
            "norm_weight": None,
            "norm_eps": 1e-5,
            "w_preshuffle_bf16": False,
            "res_preshuffle": False,
        },
    }


# aiter's keyword-only parameters, by op: aiter hands them to select and run by name
KEYWORD_ONLY = {"mla_decode_fwd": ("q_aux", "kv_aux", "nvfp4_params", "nvfp4_lse")}


def _kwargs(case: str, call=None, **changes):
    call = _call(case, **changes) if call is None else call
    return {name: call.get(name) for name in KEYWORD_ONLY.get(_op(case), ())}


def _op(case: str) -> str:
    return "mla_paged_decode" if case.startswith("mla_paged_decode") else case


def _call(case: str, **changes):
    return {**_calls()[case], **changes}


def _args(case: str, call=None, **changes):
    """aiter's positional arguments for ``call``, a fresh one by default."""
    call = _call(case, **changes) if call is None else call
    return tuple(call[name] for name in AITER_CONTRACTS[_op(case)][1])


def _select(case: str, call=None, **changes):
    return base.SELECTORS[_op(case)][1](_args(case, call, **changes), _kwargs(case, call, **changes))


def _selector_modules():
    return [
        module
        for _, module in inspect.getmembers(provider, inspect.ismodule)
        if module.__name__.startswith(f"{provider.__name__}.") and module is not base
    ]


def _answer(monkeypatch, serving):
    """Every AKLR api a selector asks answers ``serving``."""
    for module in _selector_modules():
        for api_fn in vars(module).values():
            if callable(getattr(api_fn, "serving", None)):
                monkeypatch.setattr(api_fn, "serving", lambda *a, **k: serving)


@pytest.fixture
def aklr_serves(monkeypatch):
    """Every AKLR api the selectors ask answers with KERNEL; a candidate is (kernel, run)."""
    for module in _selector_modules():
        monkeypatch.setattr(
            module,
            "offer",
            lambda serving, run: None if isinstance(serving, FallbackReason) else (serving, run),
        )
    _answer(monkeypatch, KERNEL)


class _Api:
    """A stand-in AKLR api: records each call, answers ``result``, and ``serving`` answers KERNEL."""

    def __init__(self, result=None):
        self.result, self.calls = result, []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result(*args, **kwargs) if callable(self.result) else self.result

    def serving(self, *args, **kwargs):
        return KERNEL


def test_each_selector_serves_an_aiter_contract_at_its_version():
    for op, (version, _) in AITER_CONTRACTS.items():
        assert base.SELECTORS[op][0] == version
        assert provider.provider.op_versions[op] == {version}


@pytest.mark.parametrize("case", list(_calls()))
def test_a_served_call_offers_a_run_taking_aiters_contract(aklr_serves, case):
    """aiter hands run every positional parameter; one it cannot bind is a contract mismatch."""
    kernel, run = _select(case)
    assert kernel is KERNEL
    inspect.signature(run).bind(*_args(case), **_kwargs(case))
    assert tuple(inspect.signature(run).parameters) == (
        *AITER_CONTRACTS[_op(case)][1],
        *KEYWORD_ONLY.get(_op(case), ()),
    )


_PE = torch.zeros(2, 4, 64, dtype=BF16)
_WIDE = torch.zeros(2, 4, 640, dtype=BF16)

DECLINED = [
    ("mla_decode_fwd", {"max_seqlen_q": 2}),
    ("mla_decode_fwd", {"q_aux": torch.zeros(2, 16, 8, dtype=torch.uint8)}),
    ("mla_decode_fwd", {"nvfp4_params": object()}),
    ("mla_decode_fwd", {"page_size": 64}),
    ("mla_decode_fwd", {"nhead_kv": 2}),
    ("mla_decode_fwd", {"logit_cap": 30.0}),
    ("mla_decode_fwd", {"num_kv_splits_indptr": torch.zeros(3, dtype=I32)}),
    ("mla_decode_fwd", {"return_logits": True}),
    ("mla_decode_fwd", {"return_lse": False}),
    ("mla_decode_fwd", {"q_scale": None}),
    ("mla_decode_fwd", {"kv_scale": None}),
    ("mla_decode_fwd", {"g_kv_indptr": torch.zeros(3, dtype=I32)}),
    ("mla_decode_fwd", {"cp_world_size": 2}),
    ("mla_decode_fwd", {"kv_layout": "kv_shuffled"}),
    ("mla_decode_fwd", {"seq_lens": torch.ones(2, dtype=I32)}),
    ("mla_decode_fwd", {"page_table": torch.zeros(2, 1, dtype=I32)}),
    ("mla_decode_fwd", {"kv_csr_layout": "token"}),
    ("mla_decode_fwd", {"q": torch.zeros(2, 1, 16, 576, dtype=FP8)}),
    ("mla_decode_fwd", {"kv_indptr": torch.tensor([0, 3, 5, 8], dtype=I32)}),  # 3 requests, 2 queries
    ("mla_paged_decode", {"k_pe": torch.zeros(8, 64, dtype=FP8)}),
    ("mla_paged_decode", {"kv_pe_offset": 0}),
    ("mla_paged_decode", {"q_pe": _PE}),  # not q_nope's neighbour
    ("mla_paged_decode", {"q_nope": torch.zeros(2, 4, 512, dtype=BF16), "q_pe": _PE}),
    ("mla_paged_decode", {"q_nope": _WIDE[..., :512], "q_pe": _WIDE[..., 576:]}),  # one row, a gap
    ("mla_paged_decode", {"kv_c": torch.zeros(8, 576, dtype=torch.float16)}),
    ("mla_paged_decode", {"kv_c": torch.zeros(8, 576, dtype=torch.float8_e4m3fnuz)}),
    ("mla_paged_decode", {"kv_c": torch.zeros(8, 1, 576, dtype=FP8)}),
    ("mla_paged_decode", {"kv_c": torch.zeros(8, 640, dtype=FP8)}),
    ("mla_paged_decode", {"o": torch.zeros(2, 4, 512)}),
    ("mla_paged_decode", {"o": torch.zeros(2, 1, 4, 512, dtype=BF16)}),
    ("fp8_paged_mqa_logits", {"q_fp8": torch.zeros(1, 2, 4, 128, dtype=FP8)}),  # next_n 2
    ("fp8_paged_mqa_logits", {"Preshuffle": False}),
    ("fp8_paged_mqa_logits", {"VarCtxSchedule": torch.zeros(4, dtype=I32)}),
    ("fp8_paged_mqa_logits", {"PersistentSchedule": torch.zeros(2, 4, dtype=I32)}),
    ("fp8_paged_mqa_logits", {"context_lens": torch.ones(2, 2, dtype=I32)}),
    ("fp8_paged_mqa_logits", {"out_logits": torch.zeros(2, 96)}),  # not whole pages
    ("fp8_paged_mqa_logits", {"max_model_len": 96}),  # aiter would leave columns 96.. alone
    ("fp8_paged_mqa_logits", {"kv_cache": torch.zeros(4, 64, 1, 132)}),
    ("mla_sparse_prefill_v4", {"attn_sink": None}),
    ("mla_sparse_prefill_v4", {"q_nope": torch.zeros(2, 16, 512, dtype=torch.float8_e4m3fnuz)}),
    ("mla_sparse_prefill_v4", {"kv_nope": torch.zeros(4, 512, dtype=BF16)}),
    ("mla_sparse_prefill_v4", {"out": torch.zeros(2, 16, 512)}),
    ("mla_sparse_prefill_v4", {"out": torch.zeros(2, 8, 512, dtype=BF16)}),
    ("mhc_fused_post_pre", {"w_preshuffle_bf16": True}),
    ("mhc_fused_post_pre", {"res_preshuffle": True}),
    ("mhc_fused_post_pre", {"post_layer_mix": torch.zeros(4, 4, 2)}),
]


@pytest.mark.parametrize(("case", "changes"), DECLINED)
def test_a_call_outside_aklrs_contract_is_declined_before_aklr_is_asked(aklr_serves, case, changes):
    assert _select(case, **changes) is None


@pytest.mark.parametrize("case", list(_calls()))
def test_a_call_aklr_falls_back_on_is_declined(monkeypatch, aklr_serves, case):
    _answer(monkeypatch, FallbackReason.NO_WORKLOAD)
    assert _select(case) is None


def test_mla_decode_fwd_hands_aklr_the_pool_and_aiters_o(monkeypatch, aklr_serves):
    call = _call("mla_decode_fwd")
    api = _Api(lambda *a, o, **k: (o, "lse"))
    monkeypatch.setattr(mla, "mla_decode_fwd", api)
    _, run = _select("mla_decode_fwd", call)
    o, lse = run(*_args("mla_decode_fwd", call), **_kwargs("mla_decode_fwd", call))
    assert o is call["o"] and lse == "lse"
    (args, kwargs), *_ = api.calls
    expected = [call[name] for name in ("q", "kv_buffer", "kv_indptr", "kv_indices", "q_scale", "kv_scale")]
    assert all(a is e for a, e in zip(args, expected, strict=True))
    assert kwargs == {"sm_scale": 576**-0.5, "o": call["o"]}


def test_mla_paged_decode_hands_aklr_one_q_and_writes_aiters_o(monkeypatch, aklr_serves):
    """q_nope and q_pe go as the one q they are views of; AKLR's output lands in aiter's o."""
    output = torch.full((2, 1, 4, 512), 3.0, dtype=BF16)
    api = _Api((output, torch.ones(2, 1, 4)))
    monkeypatch.setattr(mla, "flash_mla_with_kvcache", api)
    call = _call("mla_paged_decode")
    _, run = _select("mla_paged_decode", call)
    o, lse = run(*_args("mla_paged_decode", call))
    assert o is call["o"] and lse is None and torch.equal(o, output[:, 0])
    (q, kv_cache), kwargs = api.calls[0]
    assert q.shape == (2, 1, 4, 576) and q.data_ptr() == call["q_nope"].data_ptr()
    assert kv_cache.shape == (8, 1, 1, 576) and kv_cache.data_ptr() == call["kv_c"].data_ptr()
    assert kwargs == {
        "block_table": call["page_table"],
        "cache_seqlens": call["seq_info"],
        "cu_seqlens_k": None,
        "causal": False,
        "head_dim_v": 512,
        "sm_scale": 0.1,
        "kv_cache_format": "fp8_e4m3",
        "kv_scale": 1.0,
        "return_lse": True,
    }


def test_mla_paged_decode_takes_several_causal_queries_over_a_flat_table(monkeypatch, aklr_serves):
    lse = torch.ones(2, 3, 4)
    api = _Api((torch.full((2, 3, 4, 512), 5.0, dtype=BF16), lse))
    monkeypatch.setattr(mla, "flash_mla_with_kvcache", api)
    call = _call("mla_paged_decode_mtp")
    _, run = _select("mla_paged_decode_mtp", call)
    o, got_lse = run(*_args("mla_paged_decode_mtp", call))
    assert o is call["o"] and got_lse is lse and bool((o == 5).all())
    (q, _), kwargs = api.calls[0]
    assert q.shape == (2, 3, 4, 576) and q.data_ptr() == call["q_nope"].data_ptr()
    assert kwargs["cu_seqlens_k"] is call["seq_info"] and kwargs["cache_seqlens"] is None
    assert kwargs["block_table"] is call["page_table"]
    assert kwargs["causal"] is True and kwargs["kv_cache_format"] == "bf16"


def test_fp8_paged_mqa_logits_writes_aiters_out_logits(monkeypatch, aklr_serves):
    call = _call("fp8_paged_mqa_logits")
    api = _Api(call["out_logits"])
    monkeypatch.setattr(mla, "fp8_paged_mqa_logits", api)
    _, run = _select("fp8_paged_mqa_logits", call)
    assert run(*_args("fp8_paged_mqa_logits", call)) is None
    (q, kv_cache, weights, lens, block_tables, logits), kwargs = api.calls[0]
    assert q is call["q_fp8"] and weights is call["weights"] and block_tables is call["kv_indices"]
    assert logits is call["out_logits"]
    assert kv_cache.dtype == torch.uint8 and kv_cache.data_ptr() == call["kv_cache"].data_ptr()
    assert lens.shape == (2,) and lens.data_ptr() == call["context_lens"].data_ptr()
    assert kwargs == {"page_size": 64, "preshuffled": True}


@pytest.mark.parametrize("has_out", [True, False])
def test_mla_sparse_prefill_v4_hands_aklr_its_tensors_in_order(monkeypatch, aklr_serves, has_out):
    output = torch.full((2, 16, 512), 2.0, dtype=BF16)
    api = _Api(output)
    monkeypatch.setattr(mla_prefill, "sparse_prefill_mxfp8", api)
    call = _call("mla_sparse_prefill_v4", **({} if has_out else {"out": None}))
    _, run = _select("mla_sparse_prefill_v4", call)
    result = run(*_args("mla_sparse_prefill_v4", call))
    if has_out:
        assert result is call["out"] and torch.equal(result, output)
    else:
        assert result is output
    args, kwargs = api.calls[0]
    names = AITER_CONTRACTS["mla_sparse_prefill_v4"][1][:11]
    assert [a.data_ptr() for a in args] == [call[name].data_ptr() for name in names]
    assert [a.dtype for a in args[:3:2]] == [torch.uint8, torch.uint8] and args[6].dtype == torch.uint8
    assert args[1] is call["q_rope"] and args[10] is call["attn_sink"]
    assert kwargs == {"softmax_scale": 512**-0.5}


def test_mhc_fused_post_pre_hands_aklr_its_post_mix_per_stream(monkeypatch, aklr_serves):
    api = _Api(("post_mix", "comb_mix", "layer_input", "next_residual"))
    monkeypatch.setattr(mhc, "mhc_fused_post_pre", api)
    call = _call("mhc_fused_post_pre")
    _, run = _select("mhc_fused_post_pre", call)
    assert run(*_args("mhc_fused_post_pre", call)) == api.result
    args, kwargs = api.calls[0]
    assert args[2].shape == (4, 4) and args[2].data_ptr() == call["post_layer_mix"].data_ptr()
    assert args[0] is call["layer_input"] and args[1] is call["residual_in"] and args[6] is call["hc_base"]
    assert kwargs == {
        "rms_eps": 1e-6,
        "pre_eps": 1e-6,
        "sinkhorn_eps": 1e-6,
        "post_multiplier": 2.0,
        "sinkhorn_iters": 20,
        "norm_weight": None,
        "norm_eps": 1e-5,
    }


def test_a_fallback_after_select_accepted_raises(monkeypatch, aklr_serves):
    """aiter does not fall back once run started; a fallback there is a bug, not a decline."""
    _, run = _select("mla_sparse_prefill_v4")
    monkeypatch.setattr(mla_prefill, "sparse_prefill_mxfp8", _Api(FallbackReason.NOT_WARMED))
    with pytest.raises(RuntimeError, match="not_warmed"):
        run(*_args("mla_sparse_prefill_v4"))


class _EntryPoint:
    def __init__(self, name, target):
        self.name, self.target = name, target

    def load(self):
        return self.target


@pytest.fixture
def aiter_plugin(monkeypatch):
    """aiter's plugin interface with AKLR installed and enabled, and each op of this file routed
    to a native that must not run."""
    plugin = pytest.importorskip("aiter.plugin")
    plugin._reset_for_testing()
    monkeypatch.setattr(plugin, "_entry_points", lambda: (_EntryPoint("aklr", provider.provider),))
    monkeypatch.setattr(provider, "ready", lambda: True)
    routes = {}
    for op, (version, params) in AITER_SIGNATURES.items():

        def native(*args, **kwargs):
            raise AssertionError("aiter's kernel ran for a call AKLR serves")

        native.__signature__ = inspect.Signature(
            [
                *(
                    inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=d)
                    for name, d in params
                ),
                *(
                    inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=None)
                    for name in KEYWORD_ONLY.get(op, ())
                ),
            ]
        )
        routes[op] = plugin.route(op=op, version=version)(native)
    provider.enable("test")
    yield SimpleNamespace(plugin=plugin, routes=routes)
    plugin._reset_for_testing()


@pytest.mark.parametrize("case", list(_calls()))
def test_aiter_hands_a_call_aklr_serves_to_aklrs_run(monkeypatch, aiter_plugin, case):
    """The real dispatch: defaults applied, the run checked against the contract, then called."""
    _answer(monkeypatch, KERNEL)
    for module, name in (
        (mla, "mla_decode_fwd"),
        (mla, "flash_mla_with_kvcache"),
        (mla, "fp8_paged_mqa_logits"),
        (mla_prefill, "sparse_prefill_mxfp8"),
        (mhc, "mhc_fused_post_pre"),
    ):
        monkeypatch.setattr(module, name, _Api(_fake_result(name)))
    plugin, op = aiter_plugin.plugin, _op(case)
    # positional parameters with a default left to aiter, one passed by name
    names = AITER_CONTRACTS[op][1]
    call = _call(case)
    required = [n for n, d in AITER_SIGNATURES[op][1] if d is _REQUIRED]
    by_name = {n: call[n] for n in names[len(required) :]}
    decision = plugin.explain(op, *(call[n] for n in required), **by_name).decision
    assert (decision.route, decision.provider) == ("provider", "aklr")
    aiter_plugin.routes[op](*(call[n] for n in required), **by_name)
    assert plugin.last_decision().route == "provider"


def _fake_result(api_name):
    if api_name == "flash_mla_with_kvcache":
        return lambda q, *a, **k: (torch.zeros(*q.shape[:-1], 512, dtype=BF16), torch.zeros(q.shape[:-1]))
    if api_name == "sparse_prefill_mxfp8":
        return lambda q_nope, *a, **k: torch.zeros(*q_nope.shape, dtype=BF16)
    return ("aklr",)
