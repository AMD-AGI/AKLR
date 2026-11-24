# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""AKLR's kernel provider on a fake aiter: ``aiter.plugin`` as far as a provider meets it, routed ops
that exist only in ``sys.modules``, and families written under a temporary ``aklr/plugins/aiter/``."""

import functools
import inspect
import logging
import sys
import types
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch

from aklr.core.platform.env import ENV_AITER_PLUGIN, ENV_DISABLE, ENV_TRACE
from aklr.core.runtime import LoadError
from aklr.core.runtime.dispatch import DispatcherRegistry, FallbackReason
from aklr.plugins.aiter.provider import Provider
from aklr.plugins.sglang.kimi_k3_tiny_gemm.contract import KIMI_K3_TINY_GEMM
from tests.fake_host import SECOND, FakeHost

# a target in aiter is an op's name
_DECLARE_TINY_GEMM = 'plugin.declare("tiny_gemm", FIRST, versions=(2,))'


@dataclass(frozen=True)
class _Op:
    name: str
    version: int


@dataclass(frozen=True)
class _KernelCandidate:
    provider: str
    kernel_id: str
    artifact_digest: str
    qualification: str | None = None
    run: Any = None
    latency_us: float | None = None


class Aiter(FakeHost):
    """A fake aiter: ``aiter.plugin``'s ``route``, routed signatures and ``configure``, a dispatch that
    asks one provider as aiter's does, and the routed ops of ``aiter.ops.gemm``."""

    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.routed_ops: dict[str, Any] = {}
        self.configured: list[tuple[str, tuple[str, ...]]] = []
        self.candidates: list[_KernelCandidate] = []  # each one aiter was offered and ran
        self.provider: Provider | None = None  # the one aiter asks; None: nothing is enabled
        route = self.route

        @route(op="tiny_gemm", version=2)
        def _tiny_gemm_dispatch(x, w) -> torch.Tensor:
            return torch.nn.functional.linear(x, w)

        @route(op="gemm_with_mode")
        def gemm_with_mode(x, w, mode="exact", *, scale=1.0) -> torch.Tensor:
            return torch.nn.functional.linear(x, w)

        @route(op="gemm_into")
        def gemm_into(out, x, w) -> None:
            out.copy_(torch.nn.functional.linear(x, w))

        @route(op="gemm_of_one")
        def gemm_of_one(x) -> torch.Tensor:
            return x

        members_by_module = {
            "aiter.plugin": dict(
                Op=_Op,
                KernelCandidate=_KernelCandidate,
                route=route,
                configure=self.configure,
                _routed_ops=self.routed_ops,
            ),
            "aiter.ops.gemm": dict(
                _tiny_gemm_dispatch=_tiny_gemm_dispatch,
                gemm_with_mode=gemm_with_mode,
                gemm_into=gemm_into,
                gemm_of_one=gemm_of_one,
            ),
        }
        super().__init__("aiter", root, monkeypatch, members_by_module)
        self.ops = self.modules["aiter.ops.gemm"]

    def route(self, *, op: str, version: int = 1) -> Callable[[Callable], Callable]:
        operation = _Op(op, version)

        def decorate(native: Callable) -> Callable:
            contract = inspect.signature(native)
            self.routed_ops[op] = types.SimpleNamespace(op=operation, contract=contract)

            @functools.wraps(native)
            def routed(*args: Any, **kwargs: Any) -> Any:
                if self.provider is None:
                    return native(*args, **kwargs)
                bound = contract.bind(*args, **kwargs)
                bound.apply_defaults()
                candidate = self.provider.select(operation, bound.args, types.MappingProxyType(bound.kwargs))
                if candidate is None:
                    return native(*args, **kwargs)
                self.candidates.append(candidate)
                return candidate.run(*bound.args, **bound.kwargs)

            routed.__aiter_plugin_op__ = operation
            return routed

        return decorate

    def configure(self, *, source: str, enable: tuple[str, ...] = (), exclude: tuple[str, ...] = ()) -> None:
        self.configured.append((source, tuple(enable)))

    def discover(self) -> Provider:
        """aiter discovering its providers: it instantiates the entry point's class, and asks it from now on."""
        self.provider = Provider()
        return self.provider


@pytest.fixture
def aiter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, family_modules: None, dispatch_starts: list[str]
) -> Aiter:
    return Aiter(tmp_path, monkeypatch)


def _errors(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == "aklr.plugins.aiter.provider" and r.levelno == logging.ERROR
    ]


class TestDiscovery:
    """``Provider()``: what aiter learns of the declarations when it discovers the provider."""

    def test_op_versions_are_the_versions_declared_for_each_op(self, aiter: Aiter) -> None:
        """Every declared op, routed or not: aiter asks only for an op and version listed here."""
        aiter.family(
            "fam",
            'plugin.declare("tiny_gemm", FIRST, versions=(1, 2))\n'
            'plugin.declare("gemm_into", FIRST, versions=(1,))\n'
            'plugin.declare("an_op_aiter_does_not_route", FIRST, versions=(3,))',
        )
        provider = aiter.discover()
        assert provider.op_versions == {
            "tiny_gemm": (1, 2),
            "gemm_into": (1,),
            "an_op_aiter_does_not_route": (3,),
        }
        assert (provider.name, provider.api_versions) == ("aklr", (1,))

    def test_an_op_routed_after_the_discovery_is_served(
        self, aiter: Aiter, serve: Callable[..., dict[str, Any]]
    ) -> None:
        """aiter imports some ops' modules late: an op's signature is read when aiter first asks for it."""
        aiter.family("fam", 'plugin.declare("late_gemm", FIRST, versions=(1,))')
        aiter.discover()

        @aiter.route(op="late_gemm")
        def late_gemm(x, w) -> torch.Tensor:
            return torch.nn.functional.linear(x, w)

        call = serve(KIMI_K3_TINY_GEMM, "out.fill_(7.0)")
        assert (late_gemm(call["x"], call["w"]) == 7.0).all()

    def test_an_op_of_other_params_is_left_to_aiter_alone(
        self, aiter: Aiter, serve: Callable[..., dict[str, Any]], caplog: pytest.LogCaptureFixture
    ) -> None:
        """Found when aiter first asks for the op, logged once; the other ops are served."""
        aiter.family("fam", f'{_DECLARE_TINY_GEMM}\nplugin.declare("gemm_of_one", FIRST, versions=(1,))')
        aiter.discover()
        call = serve(KIMI_K3_TINY_GEMM, "out.fill_(7.0)")
        for _ in range(2):
            assert aiter.ops.gemm_of_one(call["x"]) is call["x"]
        (message,) = _errors(caplog)
        assert "gemm_of_one" in message and "missing 'w'" in message
        assert (aiter.ops._tiny_gemm_dispatch(call["x"], call["w"]) == 7.0).all()

    def test_an_op_version_no_declaration_lists_is_not_served(
        self, aiter: Aiter, serve: Callable[..., dict[str, Any]], caplog: pytest.LogCaptureFixture
    ) -> None:
        """aiter reads the declared versions and does not ask; asked all the same, the provider declines."""
        aiter.family("fam", 'plugin.declare("tiny_gemm", FIRST, versions=(1,))')
        provider = aiter.discover()
        assert provider.op_versions == {"tiny_gemm": (1,)}
        call = serve(KIMI_K3_TINY_GEMM, "out.fill_(7.0)")
        assert provider.select(_Op("tiny_gemm", 2), (call["x"], call["w"]), {}) is None
        assert _errors(caplog) == []


class TestSelect:
    """``select`` and the candidate's ``run``, as aiter's dispatch calls them."""

    def test_a_served_call_runs_aklrs_kernel(
        self, aiter: Aiter, serve: Callable[..., dict[str, Any]]
    ) -> None:
        aiter.family("fam", _DECLARE_TINY_GEMM)
        aiter.discover()
        call = serve(KIMI_K3_TINY_GEMM, "out.fill_(7.0)")
        result = aiter.ops._tiny_gemm_dispatch(call["x"], w=call["w"])
        assert result.dtype == torch.bfloat16 and result.shape == (4, 16) and (result == 7.0).all()
        (candidate,) = aiter.candidates
        kernel = DispatcherRegistry.default().dispatcher(KIMI_K3_TINY_GEMM).serving(call)
        assert (candidate.provider, candidate.kernel_id, candidate.artifact_digest) == (
            "aklr",
            kernel.kernel_ref,
            kernel.content_hash,
        )
        assert candidate.latency_us is None

    def test_one_run_serves_every_call_of_an_op_by_a_contract(
        self, aiter: Aiter, serve: Callable[..., dict[str, Any]]
    ) -> None:
        """aiter checks a ``run`` against the op's signature once, by its identity."""
        aiter.family(
            "fam",
            'plugin.declare("tiny_gemm", SECOND, versions=(2,))\nplugin.declare("tiny_gemm", FIRST, versions=(2,))',
        )
        aiter.discover()
        call = serve(KIMI_K3_TINY_GEMM, "out.fill_(7.0)")
        for _ in range(2):
            aiter.ops._tiny_gemm_dispatch(call["x"], call["w"])
        first_contract, again = aiter.candidates
        assert first_contract.run is again.run
        serve(SECOND, "out.fill_(9.0)")
        assert (aiter.ops._tiny_gemm_dispatch(call["x"], call["w"]) == 9.0).all()
        assert aiter.candidates[-1].run is not again.run

    def test_a_missed_call_runs_aiters_kernel(
        self, aiter: Aiter, serve: Callable[..., dict[str, Any]]
    ) -> None:
        aiter.family("fam", _DECLARE_TINY_GEMM)
        aiter.discover()
        call = serve(KIMI_K3_TINY_GEMM, "out.fill_(7.0)")
        other_rows = call["x"][:2]
        expected = torch.nn.functional.linear(other_rows, call["w"])
        assert torch.equal(aiter.ops._tiny_gemm_dispatch(other_rows, call["w"]), expected)
        assert aiter.candidates == []

    def test_a_param_the_contract_lacks_serves_only_at_its_default(
        self, aiter: Aiter, serve: Callable[..., dict[str, Any]]
    ) -> None:
        aiter.family("fam", 'plugin.declare("gemm_with_mode", FIRST, versions=(1,))')
        aiter.discover()
        call = serve(KIMI_K3_TINY_GEMM, "out.fill_(7.0)")
        x, w = call["x"], call["w"]
        native = torch.nn.functional.linear(x, w)
        assert (aiter.ops.gemm_with_mode(x, w) == 7.0).all()
        assert torch.equal(aiter.ops.gemm_with_mode(x, w, "fast"), native)
        assert torch.equal(aiter.ops.gemm_with_mode(x, w, scale=2.0), native)

    def test_an_op_annotated_to_return_none_returns_none(
        self, aiter: Aiter, serve: Callable[..., dict[str, Any]]
    ) -> None:
        aiter.family("fam", 'plugin.declare("gemm_into", FIRST, versions=(1,))')
        aiter.discover()
        call = serve(KIMI_K3_TINY_GEMM, "out.fill_(7.0)")
        out = torch.zeros(4, 16, dtype=torch.bfloat16)
        assert aiter.ops.gemm_into(out, call["x"], call["w"]) is None
        assert (out == 7.0).all() and len(aiter.candidates) == 1

    def test_run_raises_when_its_call_is_no_longer_served(
        self, aiter: Aiter, serve: Callable[..., dict[str, Any]]
    ) -> None:
        """aiter does not fall back once ``run`` has started."""
        aiter.family("fam", _DECLARE_TINY_GEMM)
        provider = aiter.discover()
        call = serve(KIMI_K3_TINY_GEMM, "out.fill_(7.0)")
        candidate = provider.select(_Op("tiny_gemm", 2), (call["x"], call["w"]), {})
        DispatcherRegistry.default().dispatcher(KIMI_K3_TINY_GEMM).close(FallbackReason.DISABLED)
        with pytest.raises(RuntimeError, match="disabled"):
            candidate.run(call["x"], call["w"])

    def test_a_failing_kernel_raises_to_the_caller(
        self, aiter: Aiter, serve: Callable[..., dict[str, Any]]
    ) -> None:
        aiter.family("fam", _DECLARE_TINY_GEMM)
        aiter.discover()
        call = serve(KIMI_K3_TINY_GEMM, "raise RuntimeError('broken')")
        with pytest.raises(RuntimeError, match="broken"):
            aiter.ops._tiny_gemm_dispatch(call["x"], call["w"])

    def test_a_call_traced_by_torch_compile_is_declined(
        self, aiter: Aiter, serve: Callable[..., dict[str, Any]], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        aiter.family("fam", _DECLARE_TINY_GEMM)
        provider = aiter.discover()
        call = serve(KIMI_K3_TINY_GEMM, "out.fill_(7.0)")
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
        assert provider.select(_Op("tiny_gemm", 2), (call["x"], call["w"]), {}) is None

    def test_a_call_select_cannot_read_is_declined_without_raising(
        self, aiter: Aiter, serve: Callable[..., dict[str, Any]], caplog: pytest.LogCaptureFixture
    ) -> None:
        aiter.family("fam", _DECLARE_TINY_GEMM)
        provider = aiter.discover()
        call = serve(KIMI_K3_TINY_GEMM, "out.fill_(7.0)")
        assert provider.select(_Op("tiny_gemm", 2), (call["x"],), {}) is None
        assert len(_errors(caplog)) == 1

    def test_the_first_call_starts_dispatch(self, aiter: Aiter, dispatch_starts: list[str]) -> None:
        """Nothing else starts it where only aiter's switch is on."""
        aiter.family("fam", _DECLARE_TINY_GEMM)
        aiter.discover()
        assert dispatch_starts == []
        x, w = torch.zeros(4, 128, dtype=torch.bfloat16), torch.zeros(16, 128, dtype=torch.bfloat16)
        for _ in range(2):
            aiter.ops._tiny_gemm_dispatch(x, w)
        assert dispatch_starts == ["init_once"]

    def test_a_dispatch_that_fails_to_start_declines_every_call(
        self,
        aiter: Aiter,
        dispatch_starts: list[str],
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        def init_once(registry: DispatcherRegistry) -> None:
            dispatch_starts.append("init_once")
            raise LoadError("no GPU visible")

        monkeypatch.setattr(DispatcherRegistry, "init_once", init_once)
        aiter.family("fam", _DECLARE_TINY_GEMM)
        aiter.discover()
        x, w = torch.zeros(4, 128, dtype=torch.bfloat16), torch.zeros(16, 128, dtype=torch.bfloat16)
        for _ in range(2):
            assert torch.equal(aiter.ops._tiny_gemm_dispatch(x, w), torch.nn.functional.linear(x, w))
        assert dispatch_starts == ["init_once"]
        (message,) = _errors(caplog)
        assert "no GPU visible" in message

    def test_trace_names_each_contract_asked_and_why_it_declined(
        self,
        aiter: Aiter,
        serve: Callable[..., dict[str, Any]],
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setenv(ENV_TRACE, "plugin")
        aiter.family(
            "fam",
            'plugin.declare("tiny_gemm", SECOND, versions=(2,))\nplugin.declare("tiny_gemm", FIRST, versions=(2,))',
        )
        aiter.discover()
        call = serve(KIMI_K3_TINY_GEMM, "out.fill_(7.0)")
        for _ in range(2):  # a distinct call is logged once
            aiter.ops._tiny_gemm_dispatch(call["x"], call["w"])
        lines = [
            r.getMessage().partition(" framework=")[2]
            for r in caplog.records
            if r.name == "aklr.trace.plugin"
        ]
        assert lines == [
            "aiter target=tiny_gemm route=second_gemm@1.0 result=fallback reason=no_schema",
            "aiter target=tiny_gemm route=kimi_k3_tiny_gemm@1.0 result=served reason=-",
        ]


class TestEnable:
    """``Provider.enable``: sglang's and vLLM's entry points enabling AKLR in aiter."""

    def test_the_switch_enables_aklr_in_aiter(self, aiter: Aiter) -> None:
        Provider.enable()
        assert aiter.configured == [("aklr: AKLR_AITER_PLUGIN=1", ("aklr",))]

    @pytest.mark.parametrize("switch", ["unset", "disable_all"])
    def test_a_switch_turned_off_leaves_aiter_alone(
        self, aiter: Aiter, monkeypatch: pytest.MonkeyPatch, switch: str
    ) -> None:
        if switch == "unset":
            monkeypatch.delenv(ENV_AITER_PLUGIN)
        else:
            monkeypatch.setenv(ENV_DISABLE, "1")
        Provider.enable()
        assert aiter.configured == []

    @pytest.mark.parametrize("failure", ["wrong_switch_value", "aiter_does_not_import"])
    def test_a_failure_leaves_aiter_alone_and_does_not_raise(
        self, aiter: Aiter, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, failure: str
    ) -> None:
        if failure == "wrong_switch_value":
            monkeypatch.setenv(ENV_AITER_PLUGIN, "yes")
        else:
            monkeypatch.setitem(sys.modules, "aiter", None)  # importing it raises
        Provider.enable()
        assert aiter.configured == [] and len(_errors(caplog)) == 1
