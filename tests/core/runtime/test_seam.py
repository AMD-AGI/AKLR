# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import dataclasses
import inspect
import sys
import textwrap
import types

import pytest
import torch

import aklr.ops
import aklr.ops.gemm.api as gemm_api
from aklr.core.runtime import seam
from aklr.core.runtime.dispatch import DispatcherRegistry
from aklr.core.runtime.seam import (
    RouteError,
    aklr_route,
    backend_bind,
    none_if_falsy,
    route_for,
    unused,
)
from aklr.ops.gemm.api import gemm
from tests.op_case import OpCase


def _opaque(*args, **kwargs): ...


def _same(out, x, residual, post_layer_mix, comb_res_mix, store_nt=-1): ...


def _renamed(out, x, residual, post_mix, comb_res_mix, store_nt=-1): ...


class TestRoute:
    """``aklr_route`` refuses a malformed route at declaration, before registering it."""

    @pytest.mark.parametrize(
        "body",
        [
            pytest.param("return gemm_api.gemm(a=a, b=b)", id="attribute_call"),
            pytest.param("c = a\n    return gemm(a=c, b=b)", id="body_computes"),
        ],
    )
    def test_rejects_malformed_body(self, body: str) -> None:
        """A route whose body calls the api other than by its plain name, or computes before the call."""

        def attribute_call(a, b):
            return gemm_api.gemm(a=a, b=b)

        def body_computes(a, b):
            c = a
            return gemm(a=c, b=b)

        reasons = {
            attribute_call: "call the op by a plain name",
            body_computes: "Anything that computes belongs in a kernel",
        }
        route = next(fn for fn in reasons if body in textwrap.dedent(inspect.getsource(fn)))
        api = f"{__name__}.{route.__name__}"
        with pytest.raises(RouteError, match=reasons[route]):
            aklr_route(api)(route)
        assert route_for(api) is None

    @pytest.mark.parametrize(
        "entry, raises",
        [
            pytest.param(_opaque, False, id="opaque_entry_binds_route_names"),
            pytest.param(_same, False, id="same_names"),
            pytest.param(_renamed, True, id="renamed_param_rejected"),
        ],
    )
    def test_backend_bind_entry(self, monkeypatch: pytest.MonkeyPatch, entry, raises: bool) -> None:
        """An opaque entry binds by the route's names; a visible entry must match the route."""
        aklr.ops.load_routes()
        api = "aiter.ops.mhc.mhc_post"
        monkeypatch.setattr(seam, "import_entry", lambda _: entry)
        case = OpCase.of("mhc_post@1.0")
        call = case.call(case.inputs())
        # a missing out needs cuda to allocate
        call["out"] = call["residual"].clone()
        route = route_for(api)
        if raises:
            with pytest.raises(RouteError, match="no longer describes the entry point"):
                route.backend_bind(case.contract, call)
            return
        bound = route.backend_bind(case.contract, call)
        assert bound.backend_fn is entry and not bound.args
        assert set(bound.kwargs) == {
            "out",
            "x",
            "residual",
            "post_layer_mix",
            "comb_res_mix",
        }

    def test_rejects_none_if_falsy_unforwarded(self) -> None:
        """``none_if_falsy`` names a parameter the op receives, not a dropped one."""

        def falsy_dropped(a, b, scale=0.0):
            unused(scale)
            none_if_falsy(scale)
            return gemm(a=a, b=b)

        api = f"{__name__}.falsy_dropped"
        with pytest.raises(RouteError, match="only a parameter forwarded"):
            aklr_route(api)(falsy_dropped)
        assert route_for(api) is None


class TestProxy:
    """A backend entry point proxied by ``aklr_proxy`` falls back to its own body."""

    def test_another_version_keeps_its_body_unless_strict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An entry point the route no longer describes raises under ``strict``; otherwise it
        keeps its own body and is named, while one the route describes is proxied."""
        aklr.ops.load_routes()
        stale, current = "aiter.ops.mhc.mhc_post", "aiter.ops.mhc.mhc_pre"
        declared = route_for(current).declared
        described = types.FunctionType(declared.__code__, declared.__globals__, "mhc_pre")
        entries = {stale: _renamed, current: described}

        def import_entry(api: str):
            if api not in entries:
                raise ImportError(api)
            return entries[api]

        monkeypatch.setattr(seam, "import_entry", import_entry)
        with pytest.raises(RouteError, match="no longer describes the entry point"):
            seam.aklr_proxy("aiter")
        caller = types.ModuleType("aiter_caller")
        caller.mhc_post, caller.mhc_pre = _renamed, described
        monkeypatch.setitem(sys.modules, caller.__name__, caller)
        assert seam.aklr_proxy("aiter", strict=False) == [stale]
        assert caller.mhc_post is _renamed
        assert caller.mhc_pre.aklr_proxy.backend_fn is described

    @pytest.mark.gpu
    @pytest.mark.parametrize("declined", [False, True], ids=["closed", "declined"])
    def test_falls_back_to_backend_body(
        self,
        default_dispatchers: DispatcherRegistry,
        monkeypatch: pytest.MonkeyPatch,
        declined: bool,
    ) -> None:
        """With every dispatcher closed (``NO_SCHEMA``), or when the route declines
        the call (``must_default``)."""
        aklr.ops.load_routes()
        from aiter.tuned_gemm import gemm_a16w16

        case = OpCase.of("gemm@1.0")
        bound = backend_bind(case.contract, case.call(case.inputs(), "cuda"))
        assert bound.backend_fn is not gemm_a16w16
        if declined:
            # any table opens the proxy; a decline never reaches it
            dispatcher = default_dispatchers.dispatchers()[case.contract_id]
            monkeypatch.setattr(dispatcher, "table", object())
            bound.kwargs["scale_a"] = torch.ones(1, device="cuda")
        proxied = gemm_a16w16(*bound.args, **bound.kwargs)
        backend = bound.backend_fn(*bound.args, **bound.kwargs)
        assert torch.equal(proxied, backend)

    @pytest.mark.parametrize(
        "limit, expected",
        [(None, None), (0.0, None), (0, None), (10.0, 10.0)],
        ids=["none", "zero_float", "zero_int", "set"],
    )
    def test_none_if_falsy(self, limit, expected) -> None:
        """aiter's falsy ``swiglu_limit`` means no clamp; the op receives None."""
        aklr.ops.load_routes()
        route = route_for("aiter.fused_moe.fused_moe")
        proxy = seam.Proxy(route, route.declared)
        api_kwargs = proxy._to_api_kwargs((1, 2, 3, 4, 5), {"swiglu_limit": limit})
        assert api_kwargs["swiglu_limit"] == expected
        assert api_kwargs["hidden_states"] == 1


def gemm_baseline(a, b, trans_b=True, bias=None, out_dtype=None, out=None):
    """A custom baseline, as an op's ``baseline.py`` would hold it."""
    return a @ (b.T if trans_b else b)


class TestBaseline:
    """The defined baseline: the op's ``baseline.py`` function, else its first route."""

    def test_topk_prefers_current_sglang_entry(self) -> None:
        contract = OpCase.of("topk_transform_paged@1.0").contract
        prefix = "sglang.kernels.ops.attention.dsv4.topk."
        assert seam.baseline_of(contract) == prefix + "topk_transform_paged_v2"

    def test_custom_baseline_wins_over_route(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A ``baseline.py`` function is bound with the api's params, route or not."""
        contract = OpCase.of("gemm@1.0").contract
        baseline_module = type(inspect)("baseline")
        baseline_module.gemm_baseline = gemm_baseline
        real_op_module = seam.op_module
        monkeypatch.setattr(
            seam,
            "op_module",
            lambda pkg, sub: baseline_module if sub == "baseline" else real_op_module(pkg, sub),
        )
        call = {"a": 1, "b": 2}
        bound = backend_bind(contract, call)
        assert bound.backend_fn is gemm_baseline
        assert bound.kwargs == call and bound.out_mode == "value"
        assert bound.name == seam.baseline_of(contract)
        assert bound.name.endswith(".gemm_baseline")

    def test_no_route_no_baseline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Neither: ``baseline_of`` is None and ``backend_bind`` raises
        ``NoBaselineError``, a ``RouteError``."""
        contract = OpCase.of("gemm@1.0").contract
        assert seam.baseline_of(contract) is not None  # gemm has routes
        monkeypatch.setattr(seam, "_ROUTES", {})
        assert seam.baseline_of(contract) is None
        with pytest.raises(seam.NoBaselineError) as raised:
            backend_bind(contract, {})
        assert isinstance(raised.value, RouteError)

    def test_route_of_another_version(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A route whose entry point the installed backend lacks, as one sglang version's
        on another, is passed over for the next; with none left, its error is raised."""
        case = OpCase.of("topk_transform_paged@1.0")
        (route,) = seam.contract_routes(case.contract)
        other = dataclasses.replace(route, api=f"{route.api}_of_another_sglang")
        monkeypatch.setattr(seam, "contract_routes", lambda contract: (other, route))
        missing, present = other.api, route.api

        def import_entry(api: str):
            if api == missing:
                raise AttributeError(api)
            return _opaque

        monkeypatch.setattr(seam, "import_entry", import_entry)
        call = case.call(case.inputs())
        assert backend_bind(case.contract, call).name == present
        monkeypatch.setattr(seam, "import_entry", lambda _: import_entry(missing))
        with pytest.raises(seam.BackendNotLoadedError, match=missing):
            backend_bind(case.contract, call)
