# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from aklr.core.runtime._vllm_hook_registry import HookRegistry
from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins.vllm import hook
from aklr.plugins.vllm.models.muse_spark import api


def _experts_apply(hidden_states):
    return api._experts_apply(None, None, hidden_states, *[None] * 13)


def test_rows_without_a_deferred_norm_run_vllm(monkeypatch):
    monkeypatch.setattr(api, "_PENDING", set())
    assert _experts_apply(torch.zeros(2, 4)) is FallbackReason.OUT_OF_CONTRACT


def test_a_deferred_norm_lost_on_the_way_ends_the_step(monkeypatch):
    """Rows other than the pooled ones reach the experts while a norm is left to them: vLLM's
    apply would run on unnormalized rows."""
    monkeypatch.setattr(api, "_PENDING", {1})
    with pytest.raises(RuntimeError, match="other than the pooled"):
        _experts_apply(torch.zeros(2, 4))


@pytest.fixture
def muse_routes(monkeypatch):
    """The real paired routes, with CPU stand-ins for their vLLM callers and callees."""
    registry = HookRegistry()
    monkeypatch.setattr(HookRegistry, "default", classmethod(lambda cls: registry))
    monkeypatch.setattr(
        hook, "_ROUTES", {route: hook._ROUTES[route] for route in (api._defer_norm, api._experts_apply)}
    )
    monkeypatch.setattr(hook, "_vllm_version", lambda: api._VERSIONS[0])
    for name in ("_EXPERTS", "_SERVES", "_TOPK_IDS"):
        monkeypatch.setattr(api, name, {})
    monkeypatch.setattr(api, "_PENDING", set())

    modules = {}

    def module(name):
        if name not in modules:
            parent, _, child = name.rpartition(".")
            owner = module(parent) if parent else None
            modules[name] = ModuleType(name)
            modules[name].__path__ = []
            monkeypatch.setitem(sys.modules, name, modules[name])
            if owner is not None:
                monkeypatch.setattr(owner, child, modules[name], raising=False)
        return modules[name]

    class MuseSpark1xMoE:
        def _pool(self, *args):
            from vllm.models.muse_spark.amd.model import fused_rmsnorm_expand

            return fused_rmsnorm_expand(*args)

    class MuseSparkAiterExperts:
        pass

    class FusedMoEKernelModularImpl:
        def _fused_experts(self, experts, *args):
            return experts.apply(*args)

    model = module("vllm.models.muse_spark.amd.model")
    model.MuseSpark1xMoE = MuseSpark1xMoE
    module("vllm.models.muse_spark.amd.muse_moe_runner").MuseSparkAiterExperts = MuseSparkAiterExperts
    module(
        "vllm.model_executor.layers.fused_moe.modular_kernel"
    ).FusedMoEKernelModularImpl = FusedMoEKernelModularImpl
    try:
        yield SimpleNamespace(
            registry=registry,
            model=model,
            pool=MuseSpark1xMoE(),
            experts=MuseSparkAiterExperts(),
            runner=FusedMoEKernelModularImpl(),
        )
    finally:
        registry.restore()


@pytest.mark.parametrize("phase", [None, "device_ready"], ids=["all", "routes_only"])
@pytest.mark.parametrize("reason", [FallbackReason.NOT_MEASURED, FallbackReason.NOT_WARMED])
def test_compound_fallback_uses_saved_natives_after_reinstallation(monkeypatch, muse_routes, phase, reason):
    """A declined fused op normalizes once, then passes those rows to the saved experts."""
    muse = muse_routes
    calls = []
    hidden_states = torch.zeros(2, 4, dtype=torch.bfloat16)
    output = torch.empty(4, 4, dtype=torch.bfloat16)
    topk_ids = torch.zeros(2, 2, dtype=torch.int32)
    muse.experts.moe_config = SimpleNamespace(
        hidden_dim_unpadded=4,
        intermediate_size_per_partition=8,
        intermediate_size_per_partition_unpadded=8,
        intermediate_pad=None,
    )
    muse.experts.quant_config = SimpleNamespace(
        use_mxfp4_w4a4=True,
        w1_bias=None,
        w2_bias=None,
        w1_scale=object(),
        w2_scale=object(),
    )

    def fused(*args, **kwargs):
        calls.append("fused")
        assert args[0] is hidden_states
        assert kwargs["out"] is output
        return reason

    monkeypatch.setattr(api, "muse_spark_moe", fused)

    def natives(label, norm_weight, normalized, result):
        def norm(x, weight, eps, gain_center, input_scale, repeats):
            calls.append(f"{label} norm")
            assert x is hidden_states
            assert weight is norm_weight
            assert (eps, gain_center, input_scale, repeats) == (1e-6, 1.0, False, 2)
            return normalized

        def experts(
            self,
            output,
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            activation,
            global_num_experts,
            expert_map,
            a1q_scale,
            a2_scale,
            workspace13,
            workspace2,
            expert_tokens_meta,
            apply_router_weight_on_input,
        ):
            calls.append(f"{label} experts")
            assert self is muse.experts
            assert hidden_states is normalized
            return result

        return norm, experts

    # Fresh layers keep a previous permanent decline from bypassing the deferred route.
    weights = [torch.ones(4, dtype=torch.bfloat16) for _ in range(2)]
    for label, weight in zip(("initial", "replacement"), weights):
        normalized = torch.full_like(output, 2)
        result = object()
        native_norm, native_experts = natives(label, weight, normalized, result)
        muse.model.fused_rmsnorm_expand = native_norm
        type(muse.experts).apply = native_experts
        assert set(hook._install()) == {
            "vllm.models.muse_spark.amd.model.fused_rmsnorm_expand",
            "vllm.models.muse_spark.amd.muse_moe_runner.MuseSparkAiterExperts.apply",
        }
        assert hook._original(api._defer_norm) is native_norm
        assert hook._original(api._experts_apply) is native_experts

        calls.clear()
        pooled = muse.pool._pool(hidden_states, weight, 1e-6, 1.0, False, topk_ids.shape[1])
        assert pooled is hidden_states
        assert calls == []
        assert api._PENDING == {id(weight)}
        assert (
            muse.runner._fused_experts(
                muse.experts,
                output,
                pooled,
                object(),
                object(),
                torch.ones_like(topk_ids, dtype=torch.float32),
                topk_ids,
                SimpleNamespace(name="SILU"),
                2,
                torch.arange(2),
                None,
                None,
                object(),
                object(),
                None,
                False,
            )
            is result
        )
        assert calls == ["fused", f"{label} norm", f"{label} experts"]
        assert not api._PENDING
        assert not hasattr(hidden_states, api._DEFERRED_NORM)
        if reason is FallbackReason.NOT_WARMED:
            assert (id(weight), hidden_states.shape[0]) not in api._SERVES
        else:
            assert api._SERVES[id(weight), hidden_states.shape[0]] is False

        muse.registry.restore(phase)
        assert muse.model.fused_rmsnorm_expand is native_norm
        assert type(muse.experts).apply is native_experts
        for route in (api._defer_norm, api._experts_apply):
            with pytest.raises(RuntimeError, match="has not been applied"):
                hook._original(route)


@pytest.mark.gpu
@pytest.mark.parametrize("n_tokens", [1, 7, 256])
@pytest.mark.parametrize("gain_center", [0.0, 1.0])
def test_the_baselines_norm_is_vllms(n_tokens, gain_center):
    """The copy of vLLM's norm kernel the baseline runs writes vLLM's bits."""
    fused_norm_expand = pytest.importorskip("vllm.models.muse_spark.amd.ops.fused_norm_expand")
    from aklr.plugins.vllm.models.muse_spark._norm_expand import rmsnorm_expand

    torch.manual_seed(0)
    x = torch.randn(n_tokens, 4096, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(4096, dtype=torch.bfloat16, device="cuda")
    first = fused_norm_expand.fused_rmsnorm_expand(x, weight, 1e-6, gain_center, False, 8)
    expected = fused_norm_expand.take_expanded(first, 8)
    assert torch.equal(rmsnorm_expand(x, weight, 1e-6, gain_center, False, 8), expected)
