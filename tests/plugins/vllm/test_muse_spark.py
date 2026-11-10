# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from aklr.core.runtime._hook_registry import HookRegistry
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


def _pool_call(x, weight, repeats):
    from fake_muse_model import fused_rmsnorm_expand

    return fused_rmsnorm_expand(x, weight, 1e-6, 1.0, False, repeats)


def _expert_call(expert, output, hidden_states, topk_weights, topk_ids, expert_map):
    return expert.apply(
        output,
        hidden_states,
        None,
        None,
        topk_weights,
        topk_ids,
        SimpleNamespace(name="SILU"),
        2,
        expert_map,
        None,
        None,
        None,
        None,
        None,
        False,
    )


@pytest.fixture
def native_muse(monkeypatch):
    """CPU host doubles around the actual Muse routes, without importing vLLM/AITER."""
    registry = HookRegistry()
    monkeypatch.setattr(HookRegistry, "default", staticmethod(lambda: registry))
    monkeypatch.setattr(hook, "_ROUTES", {})
    monkeypatch.setattr(hook, "_vllm_version", lambda: "0.33.0rc1.dev910")
    for name, value in (("_EXPERTS", {}), ("_SERVES", {}), ("_TOPK_IDS", {}), ("_PENDING", set())):
        monkeypatch.setattr(api, name, value)
    events = []

    def norm(x, weight, eps, gain_center, input_scale, repeats):
        events.append("native norm")
        assert not input_scale
        return (
            x.float()
            * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
            * (weight.float() + gain_center)
        ).to(x.dtype)

    def apply(
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
        events.append("native experts")
        assert not hasattr(hidden_states, api._DEFERRED_NORM)
        self.native_rows = hidden_states
        self.native_topk = topk_ids, topk_weights
        output.copy_(hidden_states[:, None, :].expand_as(output))

    class Experts:
        def _park_route_output(self, output):
            self.parked = output

    Experts.apply = apply
    model, native, experts_module = (
        ModuleType(name) for name in ("fake_muse_model", "fake_muse_native", "fake_muse_experts")
    )
    model.fused_rmsnorm_expand = native.fused_rmsnorm_expand = norm
    experts_module.Experts = Experts
    for module in (model, native, experts_module):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    for caller, target, route in (
        ("_pool_call", "fake_muse_model.fused_rmsnorm_expand", api._defer_norm),
        ("_expert_call", "fake_muse_experts.Experts.apply", api._experts_apply),
    ):
        hook.vllm_route(
            f"{__name__}.{caller}", target, versions=("0.33.0rc1.dev910",), group="muse.norm_experts"
        )(route)
    expert = Experts()
    expert.moe_config = SimpleNamespace(
        hidden_dim_unpadded=4,
        intermediate_size_per_partition=8,
        intermediate_size_per_partition_unpadded=8,
        intermediate_pad=0,
    )
    expert.quant_config = SimpleNamespace(
        use_mxfp4_w4a4=True, w1_bias=None, w2_bias=None, w1_scale=None, w2_scale=None
    )
    state = SimpleNamespace(
        registry=registry,
        model=model,
        native=native,
        experts_class=Experts,
        expert=expert,
        norm=norm,
        apply=apply,
        events=events,
        x=torch.tensor([[1, 2, 3, 4], [2, 4, 1, 3]], dtype=torch.bfloat16),
        weight=torch.ones(4, dtype=torch.bfloat16),
        topk_ids=torch.tensor([[0, 1], [1, 0]], dtype=torch.int32),
        topk_weights=torch.tensor([[0.4, 0.6], [0.7, 0.3]]),
        expert_map=torch.tensor([1, 1, 0], dtype=torch.int32),
        output=torch.empty(2, 2, 4, dtype=torch.bfloat16),
    )
    yield state
    registry.restore()


@pytest.mark.parametrize("reason", [FallbackReason.NOT_WARMED, FallbackReason.NOT_QUALIFIED])
def test_muse_fallback_uses_saved_norm_and_experts(native_muse, monkeypatch, reason):
    state = native_muse

    def miss(*args, **kwargs):
        state.events.append("aklr miss")
        return reason

    miss.fallback_reason = lambda *args, **kwargs: reason
    monkeypatch.setattr(api, "muse_spark_moe", miss)
    hook._install()
    assert hook._original(api._defer_norm) is state.norm
    assert hook._original(api._experts_apply) is state.apply

    def trap(*args, **kwargs):
        raise AssertionError("fallback followed a mutable global binding")

    # Neither an altered canonical alias nor the outer wrapper's __wrapped__ is a
    # reliable original. In particular NOT_WARMED must still materialize norm once.
    monkeypatch.setattr(state.native, "fused_rmsnorm_expand", trap)
    monkeypatch.setattr(state.experts_class.apply, "__wrapped__", trap)
    pooled = _pool_call(state.x, state.weight, 2)
    assert pooled is state.x
    assert api._PENDING == {id(state.weight)}
    assert (
        _expert_call(state.expert, state.output, pooled, state.topk_weights, state.topk_ids, state.expert_map)
        is None
    )
    assert state.events == ["aklr miss", "native norm", "native experts"]
    assert not api._PENDING
    assert not hasattr(pooled, api._DEFERRED_NORM)
    expected = state.norm(state.x, state.weight, 1e-6, 1.0, False, 2)
    assert torch.equal(state.expert.native_rows, expected)
    assert torch.equal(state.output, expected[:, None, :].expand_as(state.output))
    assert state.expert.native_topk[0] is state.topk_ids
    assert state.expert.native_topk[1] is state.topk_weights
    assert api._SERVES.get((id(state.weight), 2)) is (None if reason is FallbackReason.NOT_WARMED else False)


def test_muse_hit_hands_unweighted_routes_to_native_finalize(native_muse, monkeypatch):
    state = native_muse

    def hit(*args, out, **kwargs):
        state.events.append("aklr hit")
        return out.fill_(3)

    monkeypatch.setattr(api, "muse_spark_moe", hit)
    hook._install()
    pooled = _pool_call(state.x, state.weight, 2)
    assert (
        _expert_call(state.expert, state.output, pooled, state.topk_weights, state.topk_ids, state.expert_map)
        is None
    )
    assert state.events == ["aklr hit"]
    assert state.expert.parked.shape == (2, 2, 4)
    assert state.expert.parked.data_ptr() == state.output.data_ptr()
    assert torch.equal(state.expert.parked, torch.full_like(state.output, 3))
    assert not api._PENDING
    assert not hasattr(pooled, api._DEFERRED_NORM)


def test_muse_missing_experts_does_not_install_norm_alone(native_muse, monkeypatch):
    state = native_muse
    monkeypatch.delattr(state.experts_class, "apply")
    with pytest.raises(RuntimeError, match="apply"):
        hook._install()
    assert state.model.fused_rmsnorm_expand is state.norm
    assert state.native.fused_rmsnorm_expand is state.norm
    assert all(row["state"] == "registered" for row in state.registry.report())


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
