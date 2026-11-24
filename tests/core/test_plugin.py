# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""``Plugin`` on a fake host: families written under a temporary ``aklr/plugins/<host>/``, and host modules
that exist only in ``sys.modules``."""

import functools
import importlib
import importlib.metadata
import logging
import pkgutil
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import Any

import pytest
import torch

from aklr.core.platform.env import ENV_DISABLE, ENV_TRACE
from aklr.core.plugin import Plugin
from aklr.core.runtime import LoadError
from aklr.core.runtime.dispatch import DispatcherRegistry
from aklr.plugins.sglang.kimi_k3_tiny_gemm.contract import KIMI_K3_TINY_GEMM
from tests.fake_host import FIRST_STATEMENT_LINE, SECOND, FakeHost
from tests.op_case import OpCase

TARGET = "hostlib.gemm.tiny_gemm"
_VERSIONS = {"sglang": "0.5.22+rocm10", "vllm": "0.34.0"}
_SGLANG_DEVICE_READY = "sglang.srt.model_executor.model_runner.ModelRunner.init_torch_distributed"


def tiny_gemm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.linear(x, w)


def gemm_with_mode(x: torch.Tensor, w: torch.Tensor, mode: str = "exact") -> torch.Tensor:
    return torch.nn.functional.linear(x, w)


def gemm_into(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor) -> None:
    out.copy_(torch.nn.functional.linear(x, w))


def gemm_into_quoted(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor) -> "None":  # noqa: UP037
    out.copy_(torch.nn.functional.linear(x, w))


def gemm_of_one(x: torch.Tensor) -> torch.Tensor:
    return x


@functools.wraps(tiny_gemm)
def decorated_gemm(*args: Any, **kwargs: Any) -> torch.Tensor:
    return tiny_gemm(*args, **kwargs)


class _HookType(Enum):
    AFTER = "after"
    AROUND = "around"


class _HookRegistry:
    """sglang's ``HookRegistry`` as far as AKLR uses it: AFTER and AROUND hooks, applied at start-up."""

    def __init__(self) -> None:
        self.hooks: list[tuple[str, Any, _HookType]] = []
        self.n_applied = 0

    def register(self, target: str, hook: Any, hook_type: _HookType) -> None:
        self.hooks.append((target, hook, hook_type))

    def apply_hooks(self) -> None:
        pending, self.n_applied = self.hooks[self.n_applied :], len(self.hooks)
        for target, hook, hook_type in pending:
            owner_name, _, attr = target.rpartition(".")
            owner = pkgutil.resolve_name(owner_name)
            setattr(owner, attr, self._wrapped(getattr(owner, attr), hook, hook_type))

    @staticmethod
    def _wrapped(original: Any, hook: Any, hook_type: _HookType) -> Any:  # structure: allow S4 a closure per hook, made in apply_hooks' loop  # fmt: skip
        """A function, as sglang's wrappers are: on a class it binds ``self`` like the method it replaces."""

        def around(*args: Any, **kwargs: Any) -> Any:
            return hook(original, *args, **kwargs)

        def after(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            modified = hook(result, *args, **kwargs)
            return result if modified is None else modified

        return around if hook_type is _HookType.AROUND else after


class _ModelRunner:
    def init_torch_distributed(self) -> str:
        return "distributed"


class _Worker:
    def init_device(self) -> None:
        pass

    def compile_or_warm_up_model(self) -> str:
        return "warm"


class _WorkerWrapperBase:
    """vLLM's worker wrapper: it reaches the worker's own methods through ``__getattr__``."""

    def __init__(self) -> None:
        self.worker = _Worker()

    def init_device(self) -> None:
        self.worker.init_device()

    def __getattr__(self, attr: str) -> Any:
        return getattr(self.worker, attr)


class Host(FakeHost):
    """A fake sglang or vllm: the functions of ``hostlib.gemm`` its targets name, the modules AKLR hooks,
    and its start-up."""

    def __init__(self, name: str, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.hook_registry = _HookRegistry()
        members_by_module = {
            "hostlib.gemm": dict(
                tiny_gemm=tiny_gemm,
                gemm_with_mode=gemm_with_mode,
                gemm_into=gemm_into,
                gemm_into_quoted=gemm_into_quoted,
                gemm_of_one=gemm_of_one,
                decorated_gemm=decorated_gemm,
            ),
            "sglang.srt.plugins.hook_registry": dict(HookRegistry=self.hook_registry, HookType=_HookType),
            "sglang.srt.model_executor.model_runner": dict(
                ModelRunner=type("ModelRunner", (_ModelRunner,), {})
            ),
            "vllm.v1.worker.worker_base": dict(
                WorkerWrapperBase=type("WorkerWrapperBase", (_WorkerWrapperBase,), {})
            ),
        }
        super().__init__(name, root, monkeypatch, members_by_module)
        installed = importlib.metadata.version
        monkeypatch.setattr(
            importlib.metadata, "version", lambda package: _VERSIONS.get(package) or installed(package)
        )

    def start(self) -> Any:
        """The host's start-up as far as its device: it calls the register function and binds the device.
        Returns what warms the model up afterwards, for vllm."""
        self.plugin.register()
        if self.name == "sglang":
            self.hook_registry.apply_hooks()
            assert self.modules[
                "sglang.srt.model_executor.model_runner"
            ].ModelRunner().init_torch_distributed() == ("distributed")
            return None
        wrapper = self.modules["vllm.v1.worker.worker_base"].WorkerWrapperBase()
        wrapper.init_device()
        return wrapper

    def call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """The host calling its own function ``hostlib.gemm.<name>``, by the module attribute."""
        return getattr(self.modules["hostlib.gemm"], name)(*args, **kwargs)


@pytest.fixture(params=["sglang", "vllm"])
def host(
    request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, family_modules: None
) -> Host:
    """A fake host of each kind."""
    return Host(request.param, tmp_path, monkeypatch)


def _errors(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage() for r in caplog.records if r.name == "aklr.core.plugin" and r.levelno == logging.ERROR
    ]


class TestDeclarations:
    """``declare`` statements, and the queries over them."""

    def test_targets_lists_each_target_with_the_host_versions_of_its_declarations(self, host: Host) -> None:
        host.family(
            "fam",
            f'plugin.declare("{TARGET}", FIRST, versions=("0.5.22", "0.5.23"))\n'
            f'plugin.declare("{TARGET}", SECOND, versions=("0.5.21", "0.5.22"))\n'
            'plugin.declare("hostlib.gemm.gemm_into", FIRST, versions=("0.5.22",))',
        )
        assert dict(host.plugin.targets()) == {
            TARGET: ("0.5.22", "0.5.23", "0.5.21"),
            "hostlib.gemm.gemm_into": ("0.5.22",),
        }

    def test_dispatchers_are_the_versions_contracts_in_declaration_order(
        self, host: Host, dispatch_starts: list[str]
    ) -> None:
        host.family(
            "fam",
            f'plugin.declare("{TARGET}", SECOND, versions=("0.5.22", "0.5.23"))\n'
            f'plugin.declare("{TARGET}", FIRST, versions=("0.5.22",))',
        )
        registry = DispatcherRegistry.default()
        assert host.plugin.dispatchers(TARGET, "0.5.22") == (
            registry.dispatcher(SECOND),
            registry.dispatcher(KIMI_K3_TINY_GEMM),
        )
        assert host.plugin.dispatchers(TARGET, "0.5.23") == (registry.dispatcher(SECOND),)
        assert host.plugin.dispatchers(TARGET, "0.5.24") == ()
        with pytest.raises(KeyError, match="hostlib.gemm.unknown"):
            host.plugin.dispatchers("hostlib.gemm.unknown", "0.5.22")

    def test_integer_host_versions_are_compared_as_they_are(
        self, host: Host, dispatch_starts: list[str]
    ) -> None:
        """aiter's op versions."""
        host.family("fam", f'plugin.declare("{TARGET}", FIRST, versions=(2,))')
        assert dict(host.plugin.targets()) == {TARGET: (2,)}
        assert len(host.plugin.dispatchers(TARGET, 2)) == 1
        assert host.plugin.dispatchers(TARGET, 1) == ()

    @pytest.mark.parametrize(
        ("declarations", "error"),
        [
            ('plugin.declare("tiny gemm", FIRST, versions=("0.5.22",))', "not a name"),
            (f'plugin.declare("{TARGET}", FIRST, versions=())', "versions"),
            (f'plugin.declare("{TARGET}", FIRST, versions=("0.5.22", 2))', "versions"),
            (
                f'plugin.declare("{TARGET}", FIRST, versions=("0.5.22",))\n'
                f'plugin.declare("{TARGET}", FIRST, versions=("0.5.23",))',
                "already declared",
            ),
            (f'plugin.declare("{TARGET}", FIRST, "kernels", versions=("0.5.22",))', "not a directory"),
        ],
        ids=[
            "target_not_a_name",
            "no_versions",
            "mixed_versions",
            "declared_twice",
            "schemas_in_no_directory",
        ],
    )
    def test_a_wrong_statement_drops_its_family(
        self, host: Host, caplog: pytest.LogCaptureFixture, declarations: str, error: str
    ) -> None:
        """The statement raises ``ValueError``: its family's declarations are dropped, and one error names
        the statement's file and line."""
        host.family("good", 'plugin.declare("hostlib.gemm.gemm_into", FIRST, versions=("0.5.22",))')
        contract_py = host.family(
            "wrong", declarations, schemas={"impls/flydsl/a/0123456789abcdef.json": "x@1.0"}
        )
        assert dict(host.plugin.targets()) == {"hostlib.gemm.gemm_into": ("0.5.22",)}
        (message,) = _errors(caplog)
        line = FIRST_STATEMENT_LINE - 1 + len(declarations.split("\n"))
        assert f"{contract_py}:{line}: ValueError: " in message and error in message

    def test_a_contract_outside_its_familys_contracts_drops_the_family(
        self, host: Host, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Known once the file is imported: the error names the file and the statement."""
        host.family("good", 'plugin.declare("hostlib.gemm.gemm_into", FIRST, versions=("0.5.22",))')
        contract_py = host.family(
            "wrong",
            f'plugin.declare("{TARGET}", FIRST.model_copy(update={{"contract_id": "other@1.0"}}), '
            'versions=("0.5.22",))',
        )
        assert dict(host.plugin.targets()) == {"hostlib.gemm.gemm_into": ("0.5.22",)}
        (message,) = _errors(caplog)
        assert f"{contract_py}: ValueError: declare('{TARGET}', other@1.0)" in message
        assert "not in its family's CONTRACTS" in message

    def test_a_statement_is_heard_when_the_plugin_imports_its_contract_py(self, host: Host) -> None:
        """A family's statements stand at the bottom of its contract.py, which the registry imports too:
        outside the plugin's own collection a statement does nothing, and a file imported before it is
        collected all the same."""
        host.family("fam", f'plugin.declare("{TARGET}", FIRST, versions=("0.5.22",))')
        importlib.import_module(f"aklr.plugins.{host.name}.fam.contract")
        assert dict(host.plugin.targets()) == {TARGET: ("0.5.22",)}
        host.plugin.declare("hostlib.gemm.gemm_into", KIMI_K3_TINY_GEMM, "impls", versions=("0.5.22",))
        assert dict(host.plugin.targets()) == {TARGET: ("0.5.22",)}

    def test_a_target_is_declared_in_one_family(self, host: Host, caplog: pytest.LogCaptureFixture) -> None:
        host.family("a", f'plugin.declare("{TARGET}", FIRST, versions=("0.5.22",))')
        contract_py = host.family("b", f'plugin.declare("{TARGET}", SECOND, versions=("0.5.22",))')
        assert dict(host.plugin.targets()) == {TARGET: ("0.5.22",)}
        (message,) = _errors(caplog)
        assert f"{contract_py}:{FIRST_STATEMENT_LINE}: ValueError: " in message
        assert "declared in another family" in message

    def test_a_contract_py_that_fails_to_import_drops_its_family_alone(
        self, host: Host, caplog: pytest.LogCaptureFixture
    ) -> None:
        host.family("good", f'plugin.declare("{TARGET}", FIRST, versions=("0.5.22",))')
        contract_py = host.family(
            "broken",
            'plugin.declare("hostlib.gemm.gemm_into", FIRST, versions=("0.5.22",))\nraise RuntimeError("no luck")',
        )
        assert dict(host.plugin.targets()) == {TARGET: ("0.5.22",)}
        (message,) = _errors(caplog)
        assert f"{contract_py}:{FIRST_STATEMENT_LINE + 1}: RuntimeError: no luck" in message

    @pytest.mark.parametrize(
        ("declarations", "contract_id", "is_kept"),
        [
            (
                f'plugin.declare("{TARGET}", FIRST, "impls", versions=("0.5.22",))',
                "kimi_k3_tiny_gemm@1.0",
                True,
            ),
            (f'plugin.declare("{TARGET}", FIRST, "impls", versions=("0.5.22",))', "second_gemm@1.0", False),
            (f'plugin.declare("{TARGET}", FIRST, versions=("0.5.22",))', "kimi_k3_tiny_gemm@1.0", False),
        ],
        ids=["declared_with_its_directory", "of_an_undeclared_contract", "declared_without_a_directory"],
    )
    def test_each_schema_of_a_family_is_under_the_directory_of_a_declaration_of_its_contract(
        self, host: Host, caplog: pytest.LogCaptureFixture, declarations: str, contract_id: str, is_kept: bool
    ) -> None:
        schema = "impls/flydsl/gemm_n16_k128/0123456789abcdef.json"
        contract_py = host.family("fam", declarations, schemas={schema: contract_id})
        assert dict(host.plugin.targets()) == ({TARGET: ("0.5.22",)} if is_kept else {})
        if is_kept:
            assert _errors(caplog) == []
        else:
            (message,) = _errors(caplog)
            assert str(contract_py.parent / schema) in message

    def test_a_declaration_without_schemas_stands(self, host: Host, caplog: pytest.LogCaptureFixture) -> None:
        """A target intercepted and never served, as calls are recorded before a kernel exists."""
        host.family("fam", f'plugin.declare("{TARGET}", FIRST, versions=("0.5.22",))')
        assert dict(host.plugin.targets()) == {TARGET: ("0.5.22",)}
        assert _errors(caplog) == []


_DECLARE_TINY_GEMM = f'plugin.declare("{TARGET}", FIRST, versions=("0.5.22", "0.34.0"))'


class TestInterceptor:
    """A declared target once the host has started: served by AKLR's kernel, else the host's own function."""

    def test_a_served_call_returns_the_kernels_result(
        self, host: Host, dispatch_starts: list[str], serve: Callable[..., dict[str, Any]]
    ) -> None:
        host.family("fam", _DECLARE_TINY_GEMM)
        host.start()
        call = serve(KIMI_K3_TINY_GEMM, "out.fill_(7.0)")
        result = host.call("tiny_gemm", call["x"], call["w"])
        assert result.dtype == torch.bfloat16 and result.shape == (4, 16) and (result == 7.0).all()

    def test_a_missed_call_runs_the_host_function(
        self, host: Host, dispatch_starts: list[str], serve: Callable[..., dict[str, Any]]
    ) -> None:
        """Before any kernel is loaded, and at a call no kernel serves."""
        host.family("fam", _DECLARE_TINY_GEMM)
        host.start()
        call = OpCase.of("kimi_k3_tiny_gemm@1.0").call(OpCase.of("kimi_k3_tiny_gemm@1.0").inputs())
        expected = tiny_gemm(call["x"], call["w"])
        assert torch.equal(host.call("tiny_gemm", call["x"], call["w"]), expected)
        serve(KIMI_K3_TINY_GEMM, "out.fill_(7.0)")
        other_rows = call["x"][:2]
        assert torch.equal(host.call("tiny_gemm", other_rows, call["w"]), tiny_gemm(other_rows, call["w"]))

    def test_dispatch_starts_once_the_host_has_its_device(
        self, host: Host, dispatch_starts: list[str]
    ) -> None:
        host.family("fam", _DECLARE_TINY_GEMM)
        host.plugin.register()
        assert dispatch_starts == []
        host.start()
        assert dispatch_starts == ["init_once"]

    def test_a_param_the_contract_lacks_serves_only_at_its_default(
        self, host: Host, dispatch_starts: list[str], serve: Callable[..., dict[str, Any]]
    ) -> None:
        host.family(
            "fam", 'plugin.declare("hostlib.gemm.gemm_with_mode", FIRST, versions=("0.5.22", "0.34.0"))'
        )
        host.start()
        call = serve(KIMI_K3_TINY_GEMM, "out.fill_(7.0)")
        x, w = call["x"], call["w"]
        assert (host.call("gemm_with_mode", x, w) == 7.0).all()
        assert (host.call("gemm_with_mode", x, w=w, mode="exact") == 7.0).all()
        assert torch.equal(host.call("gemm_with_mode", x, w, mode="fast"), tiny_gemm(x, w))

    def test_a_decorated_target_is_checked_by_the_function_it_wraps(
        self, host: Host, dispatch_starts: list[str], serve: Callable[..., dict[str, Any]]
    ) -> None:
        """The decorator's ``*args, **kwargs`` tell no params: the contract is held against the function under it."""
        host.family(
            "fam", 'plugin.declare("hostlib.gemm.decorated_gemm", FIRST, versions=("0.5.22", "0.34.0"))'
        )
        host.start()
        call = serve(KIMI_K3_TINY_GEMM, "out.fill_(7.0)")
        assert (host.call("decorated_gemm", call["x"], w=call["w"]) == 7.0).all()
        other_rows = call["x"][:2]
        assert torch.equal(
            host.call("decorated_gemm", other_rows, call["w"]), tiny_gemm(other_rows, call["w"])
        )

    def test_the_first_declared_contract_that_serves_wins(
        self, host: Host, dispatch_starts: list[str], serve: Callable[..., dict[str, Any]]
    ) -> None:
        host.family(
            "fam",
            f'plugin.declare("{TARGET}", SECOND, versions=("0.5.22", "0.34.0"))\n'
            f'plugin.declare("{TARGET}", FIRST, versions=("0.5.22", "0.34.0"))',
        )
        host.start()
        call = serve(KIMI_K3_TINY_GEMM, "out.fill_(7.0)")
        assert (host.call("tiny_gemm", call["x"], call["w"]) == 7.0).all()
        serve(SECOND, "out.fill_(9.0)")
        assert (host.call("tiny_gemm", call["x"], call["w"]) == 9.0).all()

    @pytest.mark.parametrize("name", ["gemm_into", "gemm_into_quoted"])
    def test_a_target_annotated_to_return_none_returns_none(
        self, host: Host, dispatch_starts: list[str], serve: Callable[..., dict[str, Any]], name: str
    ) -> None:
        host.family("fam", f'plugin.declare("hostlib.gemm.{name}", FIRST, versions=("0.5.22", "0.34.0"))')
        host.start()
        call = serve(KIMI_K3_TINY_GEMM, "out.fill_(7.0)")
        out = torch.zeros(4, 16, dtype=torch.bfloat16)
        assert host.call(name, call["x"], call["w"], out) is None
        assert (out == 7.0).all()

    def test_a_call_traced_by_torch_compile_runs_the_host_function(
        self,
        host: Host,
        dispatch_starts: list[str],
        serve: Callable[..., dict[str, Any]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        host.family("fam", _DECLARE_TINY_GEMM)
        host.start()
        call = serve(KIMI_K3_TINY_GEMM, "out.fill_(7.0)")
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
        assert torch.equal(host.call("tiny_gemm", call["x"], call["w"]), tiny_gemm(call["x"], call["w"]))

    def test_a_failing_kernel_raises_to_the_caller(
        self, host: Host, dispatch_starts: list[str], serve: Callable[..., dict[str, Any]]
    ) -> None:
        host.family("fam", _DECLARE_TINY_GEMM)
        host.start()
        call = serve(KIMI_K3_TINY_GEMM, "raise RuntimeError('broken')")
        with pytest.raises(RuntimeError, match="broken"):
            host.call("tiny_gemm", call["x"], call["w"])

    def test_trace_names_each_contract_asked_and_why_it_fell_back(
        self,
        host: Host,
        dispatch_starts: list[str],
        serve: Callable[..., dict[str, Any]],
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setenv(ENV_TRACE, "plugin")
        host.family(
            "fam",
            f'plugin.declare("{TARGET}", SECOND, versions=("0.5.22", "0.34.0"))\n'
            f'plugin.declare("{TARGET}", FIRST, versions=("0.5.22", "0.34.0"))',
        )
        host.start()
        call = serve(KIMI_K3_TINY_GEMM, "out.fill_(7.0)")
        for _ in range(2):  # a distinct call is logged once
            host.call("tiny_gemm", call["x"], call["w"])
        lines = [
            r.getMessage().partition(" framework=")[2]
            for r in caplog.records
            if r.name == "aklr.trace.plugin"
        ]
        assert lines == [
            f"{host.name} target={TARGET} route=second_gemm@1.0 result=fallback reason=no_schema",
            f"{host.name} target={TARGET} route=kimi_k3_tiny_gemm@1.0 result=served reason=-",
        ]


class TestRegister:
    """What the register function takes over, and what it leaves to the host."""

    @pytest.mark.parametrize("switch", ["unset", "disable_all"])
    def test_a_switch_turned_off_leaves_the_host_alone(
        self, host: Host, dispatch_starts: list[str], monkeypatch: pytest.MonkeyPatch, switch: str
    ) -> None:
        if switch == "unset":
            monkeypatch.delenv(f"AKLR_{host.name.upper()}_PLUGIN")
        else:
            monkeypatch.setenv(ENV_DISABLE, "1")
        host.family("fam", _DECLARE_TINY_GEMM)
        host.start()
        assert host.modules["hostlib.gemm"].tiny_gemm is tiny_gemm
        assert host.hook_registry.hooks == [] and dispatch_starts == []

    def test_a_second_call_registers_nothing_more(
        self, host: Host, dispatch_starts: list[str], serve: Callable[..., dict[str, Any]]
    ) -> None:
        host.family("fam", _DECLARE_TINY_GEMM)
        host.plugin.register()
        n_hooks, replaced = len(host.hook_registry.hooks), host.modules["hostlib.gemm"].tiny_gemm
        host.plugin.register()
        assert len(host.hook_registry.hooks) == n_hooks and host.modules["hostlib.gemm"].tiny_gemm is replaced
        host.start()
        assert dispatch_starts == ["init_once"]

    def test_a_target_not_declared_for_the_running_version_keeps_the_host_function(
        self,
        host: Host,
        dispatch_starts: list[str],
        serve: Callable[..., dict[str, Any]],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The installed version is compared without its local label; an undeclared one is logged, not an error."""
        host.family(
            "fam",
            f"{_DECLARE_TINY_GEMM}\n"
            'plugin.declare("hostlib.gemm.gemm_with_mode", FIRST, versions=("0.5.21", "0.33.0"))',
        )
        with caplog.at_level(logging.INFO, logger="aklr.core.plugin"):
            host.start()
        call = serve(KIMI_K3_TINY_GEMM, "out.fill_(7.0)")
        assert (host.call("tiny_gemm", call["x"], call["w"]) == 7.0).all()
        assert host.modules["hostlib.gemm"].gemm_with_mode is gemm_with_mode
        assert _errors(caplog) == []
        version = _VERSIONS[host.name].partition("+")[0]
        (line,) = [r.getMessage() for r in caplog.records if r.name == "aklr.core.plugin"]
        assert f"intercepted ['{TARGET}']" in line
        assert f"not declared for {host.name} {version}: ['hostlib.gemm.gemm_with_mode']" in line

    def test_a_wrong_switch_value_drops_every_declaration(
        self,
        host: Host,
        dispatch_starts: list[str],
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setenv(f"AKLR_{host.name.upper()}_PLUGIN", "yes")
        host.family("fam", _DECLARE_TINY_GEMM)
        host.start()
        assert host.modules["hostlib.gemm"].tiny_gemm is tiny_gemm and dispatch_starts == []
        (message,) = _errors(caplog)
        assert f"AKLR_{host.name.upper()}_PLUGIN='yes'" in message

    @pytest.mark.parametrize(
        ("target", "error"),
        [
            ("hostlib.gemm.no_such_function", "no_such_function"),
            ("hostlib.gemm.gemm_of_one", "missing 'w'"),
        ],
        ids=["unresolved", "other_params"],
    )
    def test_a_target_that_fails_its_check_keeps_the_host_function_alone(
        self,
        host: Host,
        dispatch_starts: list[str],
        serve: Callable[..., dict[str, Any]],
        caplog: pytest.LogCaptureFixture,
        target: str,
        error: str,
    ) -> None:
        contract_py = host.family(
            "fam",
            f'plugin.declare("{target}", FIRST, versions=("0.5.22", "0.34.0"))\n{_DECLARE_TINY_GEMM}',
        )
        host.start()
        call = serve(KIMI_K3_TINY_GEMM, "out.fill_(7.0)")
        assert (host.call("tiny_gemm", call["x"], call["w"]) == 7.0).all()
        assert host.modules["hostlib.gemm"].gemm_of_one is gemm_of_one
        (message,) = _errors(caplog)
        assert f"{contract_py}: declare('{target}', ...): " in message and error in message

    def test_a_dispatch_that_fails_to_start_leaves_every_target_to_the_host(
        self,
        host: Host,
        dispatch_starts: list[str],
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The host dispatch_starts all the same, and its next device does not try again."""

        def init_once(registry: DispatcherRegistry) -> None:
            dispatch_starts.append("init_once")
            raise LoadError("no GPU visible")

        monkeypatch.setattr(DispatcherRegistry, "init_once", init_once)
        host.family("fam", _DECLARE_TINY_GEMM)
        host.start()
        host.start()
        assert dispatch_starts == ["init_once"]
        x, w = torch.zeros(4, 128, dtype=torch.bfloat16), torch.zeros(16, 128, dtype=torch.bfloat16)
        assert torch.equal(host.call("tiny_gemm", x, w), tiny_gemm(x, w))
        (message,) = _errors(caplog)
        assert "no GPU visible" in message

    def test_aiter_is_not_registered_through_the_register_function(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("AKLR_AITER_PLUGIN", "1")
        Plugin("aiter").register()
        (message,) = _errors(caplog)
        assert "kernel provider" in message


@pytest.mark.parametrize("host", ["vllm"], indirect=True)
class TestVllm:
    """What vLLM alone needs: no hook mechanism, and calls that may never reach the interceptor."""

    def test_the_target_is_replaced_when_the_register_function_is_called(
        self, host: Host, dispatch_starts: list[str]
    ) -> None:
        host.family("fam", _DECLARE_TINY_GEMM)
        host.plugin.register()
        replaced = host.modules["hostlib.gemm"].tiny_gemm
        assert replaced is not tiny_gemm and replaced.__name__ == "tiny_gemm"

    def test_targets_never_reaching_dispatch_by_the_end_of_warm_up_are_warned_of(
        self,
        host: Host,
        dispatch_starts: list[str],
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A call made only while torch.compile traces does not count: its function body is inlined."""
        host.family(
            "fam",
            f"{_DECLARE_TINY_GEMM}\n"
            'plugin.declare("hostlib.gemm.gemm_with_mode", FIRST, versions=("0.34.0",))\n'
            'plugin.declare("hostlib.gemm.gemm_into", FIRST, versions=("0.34.0",))',
        )
        wrapper = host.start()
        x, w = torch.zeros(4, 128, dtype=torch.bfloat16), torch.zeros(16, 128, dtype=torch.bfloat16)
        host.call("tiny_gemm", x, w)
        with monkeypatch.context() as tracing:
            tracing.setattr(torch.compiler, "is_compiling", lambda: True)
            host.call("gemm_with_mode", x, w)
        assert wrapper.compile_or_warm_up_model() == "warm"
        (warning,) = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert "['hostlib.gemm.gemm_into', 'hostlib.gemm.gemm_with_mode']" in warning

    def test_no_warning_when_every_target_reached_dispatch(
        self, host: Host, dispatch_starts: list[str], caplog: pytest.LogCaptureFixture
    ) -> None:
        host.family("fam", _DECLARE_TINY_GEMM)
        wrapper = host.start()
        host.call(
            "tiny_gemm", torch.zeros(4, 128, dtype=torch.bfloat16), torch.zeros(16, 128, dtype=torch.bfloat16)
        )
        wrapper.compile_or_warm_up_model()
        assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


def test_a_plugin_has_one_statement() -> None:
    """A target, its contract and where its schema files are, are one ``declare``."""
    assert not hasattr(Plugin("sglang"), "provide")
