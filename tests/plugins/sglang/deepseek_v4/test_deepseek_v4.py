# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""DeepSeek-V4.1's backend: its registration, the backend it selects, and the steps it
leaves to sglang, each with its reason."""

import importlib
import sys
from types import SimpleNamespace

import pytest

from aklr.plugins.sglang import hook, stats

dsv4 = importlib.import_module("aklr.plugins.sglang.models.deepseek_v4")


FORWARDS = {
    "sglang.srt.models.deepseek_v4.DeepseekV4ForCausalLM.forward",
    "sglang.srt.models.deepseek_v4_dspark.DeepseekV4ForCausalLMDSpark.forward",
}
_HIP = "sglang.srt.layers.attention.deepseek_v4_backend_hip_radix"
# around routes on the steps of a served forward
STEP_ROUTES = {
    f"{_HIP}.DeepseekV4HipRadixBackend.__init__",
    f"{_HIP}.DeepseekV4HipRadixBackend.make_forward_metadata_from_raw_verify",
    f"{_HIP}.DSV4AttnMetadata.init_compression_metadata",
    f"{_HIP}.DeepseekV4HipRadixBackend.expand_extend_with_same_length",
    f"{_HIP}.DeepseekV4HipRadixBackend.get_swa_page_indices",
    f"{_HIP}.DeepseekV4HipRadixBackend.init_forward_metadata_in_graph",
    f"{_HIP}.DeepseekV4HipRadixBackend.init_forward_metadata_dspark_draft_block",
    "sglang.srt.mem_cache.deepseek_v4_memory_pool.DeepSeekV4TokenToKVPool.translate_loc_from_full_to_swa",
    "sglang.srt.layers.attention.dsv4.dsv41_sparse.token_req_indices",
    "sglang.srt.models.deepseek_v4_dspark.DeepseekV4ForCausalLMDSpark.attach_shared_modules",
    "sglang.srt.models.deepseek_v4_dspark.DeepseekV4ForCausalLMDSpark._logits_from_x_post_hc",
    "sglang.srt.models.deepseek_v4_dspark.DeepseekV4ForCausalLMDSpark.write_target_hidden_kv",
    "sglang.srt.speculative.dspark_components.dspark_draft.draft_markov_block",
    "sglang.srt.speculative.dspark_components.dspark_draft.resolve_draft_params",
    "sglang.srt.speculative.dspark_components.dspark_kv_inject.TargetHiddenKvInjector._inject_mla",
    "sglang.srt.layers.engram.EngramEmbedding._reduce_owned_rows",
    "sglang.srt.layers.attention.dsv4.dsv41_sparse.DeepseekV41Compressor.project_fused",
    "sglang.srt.layers.logits_processor.LogitsProcessor._get_logits",
    "sglang.srt.speculative.dspark_components.dspark_verify.DsparkVerifyEpilogue.__init__",
    "sglang.srt.speculative.dspark_components.dspark_verify.DsparkVerifyEpilogue.capture_hook",
    "sglang.srt.speculative.dspark_components.dspark_verify.DsparkVerifyEpilogue._accept",
    "sglang.srt.speculative.dspark_components.dspark_verify.DsparkVerifyEpilogue._static_epilogue",
    "sglang.srt.speculative.dspark_components.dspark_verify.DsparkVerifyEpilogue.begin_static_step",
    "sglang.srt.speculative.dspark_components.dspark_verify.TargetVerifyExecutor._forward_prepared_verify",
    "sglang.srt.speculative.dspark_components.dspark_verify.TargetVerifyExecutor.accept_and_finalize",
    "sglang.srt.speculative.dspark_components.dspark_verify.TargetVerifyExecutor.run_non_compact",
    "sglang.srt.speculative.dspark_components.dspark_verify.TargetVerifyExecutor.commit_hidden",
    "sglang.srt.mem_cache.unified_cache.unified_tree_core.UnifiedTreeCore._advance_insert",
    "sglang.srt.models.deepseek_v4.DeepseekV4ForCausalLM.__init__",
    "sglang.srt.managers.tokenizer_manager.TokenizerManager.__init__",
    "sglang.srt.models.deepseek_common.moe_transport._mask_dp_pad_rows",
    "sglang.kernels.ops.quantization.fp8_kernel._run_per_token_group_quant_8bit_kernel",
}


RUNTIME_INSTALL = "sglang.srt.monokernels.runtime.install_monokernel_forward"


def _register(monkeypatch, running="0.5.22"):
    monkeypatch.delenv(dsv4.ENV_MONO, raising=False)
    monkeypatch.setattr(hook, "_ROUTES", [])
    monkeypatch.setattr(dsv4, "version", lambda _: running)
    # register() imports these afresh: the routes declare into this test's _ROUTES, and
    # compat sees this test's sglang
    for name in (
        "compat",
        "routes",
        "metadata_routes",
        "proposal",
        "engram_routes",
        "compressor_routes",
        "dp_routes",
        "runtime_routes",
        "accept",
        "inject",
        "radix_routes",
        "jit_routes",
        "tokenize_routes",
    ):
        monkeypatch.delitem(sys.modules, f"{dsv4.__name__}.{name}", raising=False)
        monkeypatch.delattr(dsv4, name, raising=False)
    dsv4.register()


@pytest.fixture
def registered(monkeypatch, sglang):
    _register(monkeypatch)
    return sglang


def test_both_specs_register_as_backend_aklr(registered):
    assert {(s["backend"], s["architecture"]) for s in registered.specs} == {
        ("aklr", "DeepseekV4ForCausalLM"),
        ("aklr", "DeepseekV4ForCausalLMDSpark"),
    }
    assert {target for target, *_ in hook._ROUTES} == FORWARDS | STEP_ROUTES | {RUNTIME_INSTALL}
    assert all(around for target, _, _, around in hook._ROUTES if target not in FORWARDS)


def test_mono_0_registers_nothing(monkeypatch, sglang):
    monkeypatch.setenv(dsv4.ENV_MONO, "0")
    dsv4.register()
    assert sglang.specs == [] and sglang.resolution_hooks == {}


@pytest.mark.parametrize(
    "model_type, not_served, serves",
    [
        ("deepseek_v4", [], False),
        ("deepseek_v41", ["gfx942, not gfx950"], False),
        ("deepseek_v41", ["TP 8, not one of [2, 4]"], False),
        ("deepseek_v41", [], True),
    ],
)
def test_only_a_deepseek_v41_target_on_gfx950_at_tp_2_or_4_is_served(
    monkeypatch, registered, model_type, not_served, serves
):
    routes = sys.modules[f"{dsv4.__name__}.routes"]
    jit_routes = sys.modules[f"{dsv4.__name__}.jit_routes"]
    swaps, inits = [], []
    monkeypatch.setattr(dsv4, "_SERVING", False)
    monkeypatch.setattr(routes, "_not_served", lambda: not_served)
    monkeypatch.setattr(jit_routes, "activate", lambda: swaps.append(1))

    def init(self, config, quant_config=None, prefix=""):
        inits.append(config.model_type)

    for _ in range(2):  # the draft builds a target too
        routes._target_init(init, object(), SimpleNamespace(model_type=model_type))
    assert inits == [model_type] * 2
    assert dsv4.serving() == serves and swaps == ([1] if serves else [])
    info = stats.snapshot()["info"].get("step_routes")
    assert info == ("on" if serves else not_served or None)


def test_an_untested_sglang_registers_nothing(monkeypatch, sglang):
    _register(monkeypatch, running="0.5.23.dev20261101")
    assert sglang.specs == [] and sglang.resolution_hooks == {} and hook._ROUTES == []


def test_a_disabled_target_spec_is_not_selected(monkeypatch, registered):
    monkeypatch.setenv("AKLR_DISABLE", "aklr_deepseek_v41")
    select(registered)
    assert registered.declared == [] and registered.checked == []
    assert stats.snapshot()["info"]["backend_refused"] == ["AKLR_DISABLE names aklr_deepseek_v41"]


def test_a_raising_startup_check_is_a_reason(registered):
    registered.disable_reasons = None  # list(None) raises in the stand-in check
    select(registered)
    assert registered.declared == []
    (why,) = stats.snapshot()["info"]["backend_refused"]
    assert why.startswith("checking the setup: TypeError")


def select(state):
    """Run the handle_monokernel step as sglang would; what the built-in step then saw."""
    seen = []
    state.resolution_hooks["handle_monokernel"]("server_args", lambda server_args: seen.append(server_args))
    assert seen == ["server_args"]  # sglang's own step always runs after


def test_a_server_without_a_backend_selects_aklr(registered):
    select(registered)
    assert registered.checked == ["aklr"]
    assert registered.declared == [("aklr.plugins.sglang", {"monokernel_backend": "aklr"})]


def test_a_refused_setup_keeps_sglang_and_says_why(registered):
    registered.disable_reasons = ["tp_size 8 is not one of ['2', '4']"]
    select(registered)
    assert registered.declared == []
    assert stats.snapshot()["info"]["backend_refused"] == registered.disable_reasons


def test_amd_is_stood_in_for_where_the_setup_passes(registered):
    registered.view.monokernel_backend = "amd"
    select(registered)
    assert registered.checked == ["aklr"]
    assert registered.declared == [("aklr.plugins.sglang", {"monokernel_backend": "aklr"})]


@pytest.mark.parametrize(
    "backend, architectures, takeover",
    [
        ("amd", ["DeepseekV4ForCausalLM"], "0"),
        ("other", ["DeepseekV4ForCausalLM"], "1"),
        (None, ["Glm5ForCausalLM"], "1"),
    ],
)
def test_another_backend_or_model_is_left_alone(monkeypatch, registered, backend, architectures, takeover):
    monkeypatch.setenv(dsv4.ENV_TAKEOVER, takeover)
    registered.view.monokernel_backend = backend
    registered.architectures = architectures
    select(registered)
    assert registered.declared == [] and registered.checked == []


class Mode:
    def __init__(self, name):
        self.name = name

    def is_decode(self):
        return self.name == "DECODE"

    def is_target_verify(self):
        return self.name == "TARGET_VERIFY"


def batch(mode, bs=1, rows=6):
    return SimpleNamespace(
        forward_mode=Mode(mode),
        batch_size=bs,
        input_ids=SimpleNamespace(shape=(rows,), device="cpu"),
    )


@pytest.mark.parametrize(
    "backend, mode, served, reason",
    [
        (None, "TARGET_VERIFY", (), "monokernel backend None"),
        ("amd", "TARGET_VERIFY", (), "monokernel backend amd"),
        ("aklr", "EXTEND", (), "extend"),
        ("aklr", "TARGET_VERIFY", {("target_verify", 1)}, "shape not served"),
        ("aklr", "TARGET_VERIFY", (), "no spec built"),
    ],
)
def test_a_fallback_names_its_reason(monkeypatch, registered, backend, mode, served, reason):
    routes = sys.modules["aklr.plugins.sglang.models.deepseek_v4.routes"]
    monkeypatch.setattr(stats, "device_counter", lambda name, device: None)
    registered.backend = backend
    model = SimpleNamespace(_monokernel_unread_metadata=(served, ()))
    why = routes._target_forward(model, None, None, batch(mode, bs=2, rows=12))
    assert isinstance(why, hook.RouteFallback) and why.reason == reason


def Model(name):
    """An instance of a class named ``name``: the specs go by the model's class name."""
    return type(name, (), {})()


@pytest.fixture
def install(monkeypatch, registered):
    """The native install route on backend aklr, TP rank 0, and what sglang's install got."""
    runtime = sys.modules[f"{dsv4.__name__}.runtime_routes"]
    monkeypatch.setattr(runtime, "_BUILT", set())
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.runtime_context",
        SimpleNamespace(get_parallel=lambda: SimpleNamespace(tp_rank=0)),
    )
    registered.backend = "aklr"
    calls = []

    def run(model, sglang_install):
        def orig(**kwargs):
            calls.append(type(kwargs["model"]).__name__)
            return sglang_install(kwargs["model"])

        return runtime._install(orig, model=model, hf_config=None, capture_tail_hooks=[])

    run.calls, run.runtime = calls, runtime
    return run


def _built(model):
    model.__dict__["_monokernel_unread_metadata"] = ({("target_verify", 1)}, ())


def test_a_refusal_of_the_native_runtime_keeps_sglang(install):
    def refuse(model):
        raise ValueError("--monokernel-backend aklr: rank 0: tp_size 8; rank 1: tp_size 8")

    assert install(Model(dsv4.TARGET), refuse) is None
    info = stats.snapshot()["info"]
    assert info["aklr_deepseek_v41:refused"] == [
        "rank 0: tp_size 8",
        "rank 1: tp_size 8",
    ]
    # the draft never builds over a target that runs sglang's forward
    install(Model(dsv4.DRAFT), _built)
    assert install.calls == [dsv4.TARGET]
    info = stats.snapshot()["info"]
    assert info["aklr_deepseek_v41_dspark_draft:refused"] == ["the target model runs sglang's forward"]


def test_a_failed_build_still_raises(install):
    def fail(model):
        raise RuntimeError("HIP out of memory")

    with pytest.raises(RuntimeError, match="out of memory"):
        install(Model(dsv4.TARGET), fail)


def test_the_draft_builds_over_a_built_target(install):
    install(Model(dsv4.TARGET), _built)
    install(Model(dsv4.DRAFT), _built)
    assert install.calls == [dsv4.TARGET, dsv4.DRAFT]
    assert install.runtime.built_models() == {dsv4.TARGET, dsv4.DRAFT}


def test_aklr_disable_names_one_spec(monkeypatch, install):
    monkeypatch.setenv("AKLR_DISABLE", "aklr_deepseek_v41_dspark_draft")
    install(Model(dsv4.TARGET), _built)
    install(Model(dsv4.DRAFT), _built)
    assert install.calls == [dsv4.TARGET]


def test_another_backend_or_model_goes_to_sglang_as_is(registered, install):
    registered.backend = "amd"
    install(Model(dsv4.TARGET), lambda m: None)
    registered.backend = "aklr"
    install(Model("Glm5ForCausalLM"), lambda m: None)
    assert install.calls == [dsv4.TARGET, "Glm5ForCausalLM"]
    assert install.runtime.built_models() == frozenset()
