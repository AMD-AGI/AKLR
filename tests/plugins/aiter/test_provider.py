# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""AKLR as aiter's kernel plugin: what each selector takes, what it declines, the AKLR call its run
makes, and, with an aiter that has the plugin interface, the route aiter takes."""

import importlib
import inspect
from types import SimpleNamespace

import pytest
import torch

from aklr.core.platform.env import ENV_AITER_PLUGIN, RuntimeEnv
from aklr.core.runtime.builder import RunnableMetadata
from aklr.core.runtime.dispatch import FallbackReason

provider = importlib.import_module("aklr.plugins.aiter.provider")
base = importlib.import_module("aklr.plugins.aiter.provider._base")

FP8, BF16, F32 = torch.float8_e4m3fn, torch.bfloat16, torch.float32
E8M0 = getattr(torch, "float8_e8m0fnu", None)
KERNEL = RunnableMetadata(
    build_type="flydsl", contract_id="any@1.0", schema_name="any_schema", content_hash="0123456789abcdef0123"
)

# aiter's contracts (ROCm/aiter tests/plugin_contracts.json at 78a14ae5d): version, positional parameters
AITER_CONTRACTS = {
    "rmsnorm": (2, ("input", "weight", "epsilon", "use_model_sensitive_rmsnorm")),
    "rmsnorm_with_add": (
        1,
        (
            "out",
            "input",
            "residual_in",
            "residual_out",
            "weight",
            "epsilon",
            "gemma_norm",
            "use_model_sensitive_rmsnorm",
        ),
    ),
    "gemm_a8w8_blockscale": (1, ("XQ", "WQ", "x_scale", "w_scale", "dtype", "isBpreshuffled", "split_k")),
    "gemm_a8w8_blockscale_bpreshuffle": (1, ("XQ", "WQ", "x_scale", "w_scale", "dtype", "out")),
    "gemm_a8w8_bpreshuffle": (1, ("XQ", "WQ", "x_scale", "w_scale", "bias", "dtype", "check")),
    "per_group_quant": (
        1,
        (
            "x",
            "scale",
            "quant_dtype",
            "group_size",
            "transpose_scale",
            "num_rows",
            "num_rows_factor",
            "scale_type",
            "scale_layout_m32k4",
        ),
    ),
    "mhc_pre": (
        1,
        (
            "residual",
            "fn",
            "hc_scale",
            "hc_base",
            "rms_eps",
            "hc_pre_eps",
            "hc_sinkhorn_eps",
            "hc_post_mult_value",
            "sinkhorn_repeat",
            "norm_weight",
            "norm_eps",
            "large_m_splitk",
            "w_preshuffle_bf16",
            "res_preshuffle",
        ),
    ),
    "topk_gating": (
        1,
        (
            "topk_weights",
            "topk_indices",
            "gating_output",
            "correction_bias",
            "need_renorm",
            "routed_scaling_factor",
            "score_func",
            "num_shared_experts",
            "num_valid_rows",
        ),
    ),
    "gemm_a16w16": (1, ("A", "B", "bias", "otype", "scale_a", "scale_b", "scale_c", "out")),
    "gemm_a16w16_activation": (
        1,
        ("x", "w", "bias", "dtype", "y", "config", "activation", "skip_reduce", "kernel_type", "persistent"),
    ),
    "gemm_a8w8_blockscale_out": (
        1,
        ("x", "w", "x_scale", "w_scale", "dtype", "y", "config", "skip_reduce", "kernel_type"),
    ),
    "fused_moe": (
        1,
        (
            "hidden_states",
            "w1",
            "w2",
            "topk_weight",
            "topk_ids",
            "expert_mask",
            "activation",
            "quant_type",
            "doweight_stage1",
            "w1_scale",
            "w2_scale",
            "a1_scale",
            "a2_scale",
            "block_size_M",
            "num_local_tokens",
            "moe_sorting_dispatch_policy",
            "dtype",
            "hidden_pad",
            "intermediate_pad",
            "bias1",
            "bias2",
            "splitk",
            "swiglu_limit",
            "beta",
            "linear_beta",
            "gate_mode",
            "ep_arena_handle",
            "ep_combine_input_offset",
            "ep_slot_stride_bytes",
            "ep_max_tokens_per_rank",
            "ep_world_size",
            "ep_combine_quant",
            "ep_source_token_map",
            "output",
            "quant_type_a",
            "quant_dtype_a",
            "quant_dtype_a2",
            "herd_topk",
            "herd_routed_scaling_factor",
            "deterministic",
        ),
    ),
    "biased_grouped_topk": (
        1,
        (
            "gating_output",
            "correction_bias",
            "topk_weights",
            "topk_ids",
            "num_expert_group",
            "topk_group",
            "need_renorm",
            "routed_scaling_factor",
            "num_valid_rows",
        ),
    ),
    "mla_decode_fwd": (
        1,
        (
            "q",
            "kv_buffer",
            "o",
            "qo_indptr",
            "kv_indptr",
            "kv_indices",
            "kv_last_page_lens",
            "max_seqlen_q",
            "page_size",
            "nhead_kv",
            "sm_scale",
            "logit_cap",
            "num_kv_splits",
            "num_kv_splits_indptr",
            "work_meta_data",
            "work_indptr",
            "work_info_set",
            "reduce_indptr",
            "reduce_final_map",
            "reduce_partial_map",
            "q_scale",
            "kv_scale",
            "intra_batch_mode",
            "return_logits",
            "return_lse",
            "g_kv_indptr",
            "cp_world_size",
            "cp_rank",
            "causal",
            "kv_layout",
            "seq_lens",
            "page_table",
            "kv_csr_layout",
            "fast_mode",
        ),
    ),
    "mla_paged_decode": (
        1,
        (
            "q_nope",
            "q_pe",
            "kv_c",
            "o",
            "page_table",
            "seq_info",
            "sm_scale",
            "k_pe",
            "kv_pe_offset",
            "use_2d_view",
            "kv_scale",
            "min_kv_seq_len",
            "return_lse",
        ),
    ),
    "fp8_paged_mqa_logits": (
        1,
        (
            "q_fp8",
            "kv_cache",
            "weights",
            "out_logits",
            "context_lens",
            "kv_indices",
            "max_model_len",
            "Preshuffle",
            "KVBlockSize",
            "ChunkK",
            "TotalCuCount",
            "WavePerEU",
            "VarCtxSchedule",
            "SplitKV",
            "PersistentSchedule",
        ),
    ),
    "mla_sparse_prefill_v4": (
        1,
        (
            "q_nope",
            "q_rope",
            "unified_kv_nope",
            "unified_kv_rope",
            "kv_indices_prefix",
            "kv_indptr_prefix",
            "kv_nope",
            "kv_rope",
            "kv_indices_extend",
            "kv_indptr_extend",
            "attn_sink",
            "softmax_scale",
            "out",
            "num_kv_splits",
        ),
    ),
    "mhc_fused_post_pre": (
        1,
        (
            "layer_input",
            "residual_in",
            "post_layer_mix",
            "comb_res_mix",
            "fn",
            "hc_scale",
            "hc_base",
            "rms_eps",
            "hc_pre_eps",
            "hc_sinkhorn_eps",
            "hc_post_mult_value",
            "sinkhorn_repeat",
            "norm_weight",
            "norm_eps",
            "w_preshuffle_bf16",
            "res_preshuffle",
        ),
    ),
    "flash_attn_varlen": (
        1,
        (
            "q",
            "k",
            "v",
            "cu_seqlens_q",
            "cu_seqlens_k",
            "max_seqlen_q",
            "max_seqlen_k",
            "min_seqlen_q",
            "dropout_p",
            "softmax_scale",
            "logits_soft_cap",
            "causal",
            "window_size",
            "bias",
            "alibi_slopes",
            "deterministic",
            "return_lse",
            "return_attn_probs",
            "how_v3_bf16_cvt",
            "block_table",
            "out",
            "cu_seqlens_q_padded",
            "cu_seqlens_k_padded",
            "sink_ptr",
            "q_descale",
            "k_descale",
            "v_descale",
        ),
    ),
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


def _served_calls():
    """A call AKLR's contract takes, per aiter op, by parameter name, defaults applied."""
    z = torch.zeros
    blockscale = {
        "XQ": z(4, 128, dtype=FP8),
        "WQ": z(128, 128, dtype=FP8),
        "x_scale": z(4, 1, dtype=F32),
        "w_scale": z(1, 1, dtype=F32),
        "dtype": BF16,
    }
    quant = {
        "x": z(4, 128, dtype=BF16),
        "scale": None,
        "quant_dtype": FP8,
        "num_rows": None,
        "num_rows_factor": 1,
        "scale_layout_m32k4": False,
    }
    return {
        "rmsnorm": {
            "input": z(4, 8, dtype=BF16),
            "weight": z(8, dtype=BF16),
            "epsilon": 1e-6,
            "use_model_sensitive_rmsnorm": 0,
        },
        "rmsnorm_with_add": {
            "out": z(4, 8, dtype=BF16),
            "input": z(4, 8, dtype=BF16),
            "residual_in": z(4, 8, dtype=BF16),
            "residual_out": z(4, 8, dtype=BF16),
            "weight": z(8, dtype=BF16),
            "epsilon": 1e-6,
            "gemma_norm": False,
            "use_model_sensitive_rmsnorm": 0,
        },
        "gemm_a8w8_blockscale": {**blockscale, "isBpreshuffled": False, "split_k": None},
        "gemm_a8w8_blockscale_bpreshuffle": {**blockscale, "out": None},
        "per_group_quant_fp8": {**quant, "group_size": 128, "transpose_scale": True, "scale_type": F32},
        "per_group_quant_mxfp8": {**quant, "group_size": 32, "transpose_scale": False, "scale_type": E8M0},
        "mhc_pre": {
            "residual": z(4, 4, 8, dtype=BF16),
            "fn": z(24, 32),
            "hc_scale": z(3),
            "hc_base": z(24),
            "rms_eps": 1e-6,
            "hc_pre_eps": 1e-6,
            "hc_sinkhorn_eps": 1e-6,
            "hc_post_mult_value": 1.0,
            "sinkhorn_repeat": 20,
            "norm_weight": None,
            "norm_eps": 1e-6,
            "large_m_splitk": False,
            "w_preshuffle_bf16": 0,
            "res_preshuffle": False,
        },
        "topk_gating": {
            "topk_weights": z(4, 2),
            "topk_indices": z(4, 2, dtype=torch.int32),
            "gating_output": z(4, 8, dtype=BF16),
            "correction_bias": z(8, dtype=BF16),
            "need_renorm": True,
            "routed_scaling_factor": 1.0,
            "score_func": "sigmoid",
            "num_shared_experts": 0,
            "num_valid_rows": None,
        },
    }


def _served_cases():
    """The served cases this torch can express: mxfp8 needs its e8m0 dtype."""
    return [case for case in _served_calls() if E8M0 is not None or case != "per_group_quant_mxfp8"]


def _op(case: str) -> str:
    return "per_group_quant" if case.startswith("per_group_quant") else case


def _args(case: str, **changes):
    call = {**_served_calls()[case], **changes}
    return tuple(call[name] for name in AITER_CONTRACTS[_op(case)][1])


def _select(case: str, **changes):
    return base.SELECTORS[_op(case)][1](_args(case, **changes), {})


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


def test_each_selector_serves_an_aiter_contract_at_its_version():
    assert {op: version for op, (version, _) in base.SELECTORS.items()} == {
        op: version for op, (version, _) in AITER_CONTRACTS.items()
    }
    assert provider.provider.op_versions == {op: {v} for op, (v, _) in AITER_CONTRACTS.items()}


@pytest.mark.parametrize("case", _served_cases())
def test_a_served_call_offers_a_run_taking_aiters_contract(aklr_serves, case):
    """aiter hands run every positional parameter; one it cannot bind is a contract mismatch."""
    kernel, run = _select(case)
    assert kernel is KERNEL
    inspect.signature(run).bind(*_args(case))


DECLINED = [
    ("rmsnorm", {"use_model_sensitive_rmsnorm": 1}),
    ("rmsnorm_with_add", {"gemma_norm": True}),
    ("rmsnorm_with_add", {"use_model_sensitive_rmsnorm": 1}),
    ("gemm_a8w8_blockscale", {"isBpreshuffled": True}),
    ("gemm_a8w8_blockscale", {"split_k": 2}),
    ("gemm_a8w8_blockscale", {"x_scale": torch.zeros(4, 4, dtype=torch.uint8)}),
    ("gemm_a8w8_blockscale_bpreshuffle", {"w_scale": torch.zeros(1, 4, dtype=torch.uint8)}),
    ("per_group_quant_fp8", {"scale": torch.zeros(4, 1)}),
    ("per_group_quant_fp8", {"num_rows": torch.ones(1, dtype=torch.int32)}),
    ("per_group_quant_fp8", {"quant_dtype": torch.int8}),
    ("per_group_quant_fp8", {"x": torch.zeros(1, 4, 128, dtype=BF16)}),
    ("per_group_quant_fp8", {"transpose_scale": False}),
    ("per_group_quant_fp8", {"group_size": 64}),
    ("per_group_quant_mxfp8", {"scale_layout_m32k4": True}),
    ("per_group_quant_mxfp8", {"transpose_scale": True}),
    ("mhc_pre", {"w_preshuffle_bf16": 1}),
    ("mhc_pre", {"res_preshuffle": True}),
    ("topk_gating", {"score_func": "sqrtsoftplus"}),
    ("topk_gating", {"score_func": "softmax"}),
    ("topk_gating", {"correction_bias": None}),
    ("topk_gating", {"num_shared_experts": 1}),
    ("topk_gating", {"num_valid_rows": torch.ones(1, dtype=torch.int32)}),
]


def _row_major(scale):
    """``scale`` carrying AKLR's row-major marker."""
    scale.is_row_major = True
    return scale


DECLINED.append(("gemm_a8w8_blockscale_bpreshuffle", {"x_scale": _row_major(torch.zeros(4, 1))}))


@pytest.mark.parametrize(("case", "changes"), DECLINED)
def test_a_call_outside_aklrs_contract_is_declined_before_aklr_is_asked(aklr_serves, case, changes):
    assert _select(case, **changes) is None


def test_topk_gating_is_declined_while_aiter_routes_by_herd(monkeypatch, aklr_serves):
    """With HERD, aiter's sigmoid topk_gating drops experts by batch popularity, not AKLR's exact top-k."""
    assert _select("topk_gating") is not None
    monkeypatch.setenv("AITER_FLYDSL_USE_HERD", "1")
    assert _select("topk_gating") is None


@pytest.mark.parametrize("case", _served_cases())
def test_a_call_aklr_falls_back_on_is_declined(monkeypatch, aklr_serves, case):
    _answer(monkeypatch, FallbackReason.NO_WORKLOAD)
    assert _select(case) is None


def _api(fn):
    """A stand-in AKLR api calling ``fn``, whose ``serving`` answers KERNEL."""
    fn.serving = lambda *a, **k: KERNEL
    return fn


def test_runs_hand_aklr_aiters_layouts(monkeypatch, aklr_serves):
    """The transposed fp8 scales come back as aiter shapes them, (rows, groups) over AKLR's
    group-major bytes; the shuffled-weight GEMM asks for AKLR's preshuffled operand format."""
    quant = importlib.import_module("aklr.plugins.aiter.provider.quant")
    scales = torch.arange(4.0).view(1, 4)  # AKLR's (groups, rows) for 4 rows of 128
    monkeypatch.setattr(quant, "per_group_quant_fp8", _api(lambda x: ("q", scales)))
    _, run = _select("per_group_quant_fp8")
    quantized, aiter_scales = run(*_args("per_group_quant_fp8"))
    assert quantized == "q" and aiter_scales.shape == (4, 1) and aiter_scales.data_ptr() == scales.data_ptr()

    gemm = importlib.import_module("aklr.plugins.aiter.provider.gemm")
    calls = []
    monkeypatch.setattr(gemm, "gemm_a8w8_blockwise_scaled", _api(lambda *a, **k: calls.append(k) or "y"))
    _, run = _select("gemm_a8w8_blockscale_bpreshuffle")
    assert run(*_args("gemm_a8w8_blockscale_bpreshuffle")) == "y"
    assert calls == [{"out_dtype": BF16, "out": None, "operand_format": "none:preshuffled"}]


def test_mxfp8_is_declined_on_a_torch_without_e8m0(monkeypatch, aklr_serves):
    quant = importlib.import_module("aklr.plugins.aiter.provider.quant")
    monkeypatch.setattr(quant, "_E8M0", None)
    assert _select("per_group_quant_fp8", group_size=32, transpose_scale=False, scale_type=None) is None


def test_a_fallback_after_select_accepted_raises(monkeypatch, aklr_serves):
    """aiter does not fall back once run started; a fallback there is a bug, not a decline."""
    norm = importlib.import_module("aklr.plugins.aiter.provider.norm")
    _, run = _select("rmsnorm")
    monkeypatch.setattr(norm, "rms_norm", _api(lambda *a: FallbackReason.NOT_WARMED))
    with pytest.raises(RuntimeError, match="not_warmed"):
        run(*_args("rmsnorm"))


def test_a_process_aklr_cannot_start_in_declines_every_call(monkeypatch, aklr_serves):
    monkeypatch.setattr(provider, "ready", lambda: False)
    op = SimpleNamespace(name="rmsnorm", version=2)
    assert provider.provider.select(op, _args("rmsnorm"), {}) is None
    monkeypatch.setattr(provider, "ready", lambda: True)
    assert provider.provider.select(op, _args("rmsnorm"), {})[0] is KERNEL
    assert provider.provider.select(SimpleNamespace(name="no_such_op", version=1), (), {}) is None


@pytest.mark.parametrize(("value", "enabled"), [("", False), ("0", False), ("1", True)])
def test_aklr_aiter_plugin_switch(monkeypatch, value, enabled):
    monkeypatch.setenv(ENV_AITER_PLUGIN, value)
    assert RuntimeEnv().is_aiter_plugin_enabled is enabled


def test_a_bad_aklr_aiter_plugin_value_raises(monkeypatch):
    monkeypatch.setenv(ENV_AITER_PLUGIN, "yes")
    with pytest.raises(ValueError, match=ENV_AITER_PLUGIN):
        RuntimeEnv()


class _EntryPoint:
    def __init__(self, name, target):
        self.name, self.target = name, target

    def load(self):
        return self.target


@pytest.fixture
def aiter_plugin(monkeypatch):
    """aiter's plugin interface with AKLR and one other provider installed."""
    plugin = pytest.importorskip("aiter.plugin")
    plugin._reset_for_testing()
    other = SimpleNamespace(
        name="other", api_versions={plugin.API_VERSION}, op_versions={}, select=lambda *a: None
    )
    entry_points = (_EntryPoint("aklr", provider.provider), _EntryPoint("other", other))
    monkeypatch.setattr(plugin, "_entry_points", lambda: entry_points)
    monkeypatch.setattr(provider, "ready", lambda: True)
    yield plugin
    plugin._reset_for_testing()


def test_enable_asks_aklr_first_and_keeps_the_providers_enabled(aiter_plugin):
    aiter_plugin.configure(source="framework", enable=["other"])
    assert provider.enable("test") is True
    assert aiter_plugin.status()["enabled"] == ("aklr", "other")


def test_aiter_routes_a_call_aklr_serves_to_aklrs_kernel(monkeypatch, aiter_plugin):
    if "rmsnorm" not in aiter_plugin._routed_ops:  # an aiter whose rmsnorm module is not imported

        @aiter_plugin.route(op="rmsnorm", version=2)
        def rmsnorm(input, weight, epsilon, use_model_sensitive_rmsnorm):
            raise AssertionError("explain runs nothing")

    provider.enable("test")
    norm = importlib.import_module("aklr.plugins.aiter.provider.norm")
    monkeypatch.setattr(norm.rms_norm, "serving", lambda *a: KERNEL)
    decision = aiter_plugin.explain("rmsnorm", *_args("rmsnorm")).decision
    assert (decision.route, decision.provider, decision.kernel_id) == ("provider", "aklr", KERNEL.kernel_ref)
    monkeypatch.setattr(norm.rms_norm, "serving", lambda *a: FallbackReason.NO_WORKLOAD)
    assert aiter_plugin.explain("rmsnorm", *_args("rmsnorm")).decision.route == "native"


def test_a_call_in_declining_is_never_offered_to_aklr(monkeypatch):
    """A framework route that asked AKLR and fell back runs aiter's own kernel inside ``declining()``."""
    asked = []
    monkeypatch.setitem(
        base.SELECTORS, "rmsnorm", (2, lambda args, kwargs: asked.append(args) or "candidate")
    )
    monkeypatch.setattr(provider, "ready", lambda: True)
    op = SimpleNamespace(name="rmsnorm")
    with provider.declining():
        assert provider.provider.select(op, (), {}) is None
    assert asked == []
    assert provider.provider.select(op, (), {}) == "candidate"
