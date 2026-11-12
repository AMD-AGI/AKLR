# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import runpy
import sys
import types
import warnings
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from aklr.core.platform.env import RuntimeEnv
from aklr.core.runtime.dispatch import KernelDispatcher
from aklr.core.runtime.seam import Adapter, SeamRegistry
from aklr.core.runtime.tracing import TracingConfig, TracingRuntime, WorkloadEntry
from aklr.core.schema import SchemaCollection, Workload
from aklr.ops import gemm_bf16_nt
from tests.op_case import GEMM_KERNEL, OpCase, dispatcher

_PACKAGE = "_aklr_fake_backend"
_ENTRY = f"{_PACKAGE}.ops.gemm"


def _bind(A, B, scale=None):
    return {"a": A, "b": B}


def _declining_bind(A, B, scale=None):
    return None if scale is not None else _bind(A, B)


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    module = types.ModuleType(f"{_PACKAGE}.ops")
    module.gemm = lambda A, B, scale=None: "backend"
    caller = types.ModuleType(f"{_PACKAGE}_caller")
    caller.gemm = module.gemm
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setitem(sys.modules, caller.__name__, caller)
    return caller


@pytest.fixture
def seam(request: pytest.FixtureRequest, backend: types.ModuleType) -> Iterator[types.ModuleType]:
    registry = SeamRegistry()
    registry.add(Adapter(_ENTRY, gemm_bf16_nt, getattr(request, "param", _bind)))
    registry.install()
    yield backend
    registry.uninstall()


class TestSeam:
    def test_backend_runs_without_a_dispatcher(self, seam: types.ModuleType) -> None:
        call = OpCase.from_id("gemm_bf16_nt").call()
        assert seam.gemm(call["a"], B=call["b"]) == "backend"

    def test_compiled_call_traces_the_backend(
        self, backend: types.ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Under torch.compile (as sglang compiles its DSA indexer's head gate) an active dispatcher
        still traces the backend, in one graph: dispatch reads data_ptr and keys on each call's sizes."""
        backend.gemm = sys.modules[f"{_PACKAGE}.ops"].gemm = lambda A, B, scale=None: A @ B.T
        registry = SeamRegistry()
        registry.add(Adapter(_ENTRY, gemm_bf16_nt, _bind))
        registry.install()
        # any instance makes the stand-in dispatch; tracing dispatch would fail on it
        monkeypatch.setattr(KernelDispatcher, "get_instance", classmethod(lambda cls: object()))
        try:
            assert hasattr(backend.gemm, "aklr_adapter")
            compiled = torch.compile(
                lambda a, b: backend.gemm(a, b), backend="eager", dynamic=True, fullgraph=True
            )
            for m in (1, 4, 6):
                a, b = torch.randn(m, 64), torch.randn(32, 64)
                torch.testing.assert_close(compiled(a, b), a @ b.T)
        finally:
            registry.uninstall()

    @pytest.mark.gpu
    @pytest.mark.usefixtures("builder_registry")
    def test_active_dispatcher_serves_the_backend_call(self, seam: types.ModuleType) -> None:
        case = OpCase.from_id("gemm_bf16_nt")
        call = case.call()
        with dispatcher(case.served(GEMM_KERNEL)):
            out = seam.gemm(call["a"], B=call["b"])
        assert torch.equal(out, call["a"] @ call["b"].mT)

    @pytest.mark.gpu
    def test_backend_runs_on_a_miss(self, seam: types.ModuleType) -> None:
        call = OpCase.from_id("gemm_bf16_nt").call()
        with dispatcher():
            assert seam.gemm(call["a"], B=call["b"]) == "backend"

    @pytest.mark.gpu
    @pytest.mark.usefixtures("builder_registry")
    @pytest.mark.parametrize("seam", [_declining_bind], indirect=True)
    def test_declined_call_runs_the_backend(self, seam: types.ModuleType) -> None:
        case = OpCase.from_id("gemm_bf16_nt")
        call = case.call()
        with dispatcher(case.served(GEMM_KERNEL)):
            assert seam.gemm(call["a"], B=call["b"], scale=2) == "backend"

    def test_uninstall_restores_the_entry_point(self, backend: types.ModuleType) -> None:
        original = backend.gemm
        registry = SeamRegistry()
        registry.add(Adapter(_ENTRY, gemm_bf16_nt, _bind))
        registry.install()
        assert backend.gemm is not original
        registry.uninstall()
        assert backend.gemm is original
        assert sys.modules[f"{_PACKAGE}.ops"].gemm is original

    @pytest.mark.parametrize(
        ("declared", "entry", "match"),
        [
            pytest.param([_ENTRY], _ENTRY, "two adapters", id="duplicate"),
            pytest.param([], "gemm", "not a dotted", id="undotted"),
        ],
    )
    def test_add_rejects(self, declared: list[str], entry: str, match: str) -> None:
        registry = SeamRegistry()
        for name in declared:
            registry.add(Adapter(name, gemm_bf16_nt, _bind))
        with pytest.raises(ValueError, match=match):
            registry.add(Adapter(entry, gemm_bf16_nt, _bind))

    @pytest.mark.parametrize(
        ("source", "warns"),
        [
            pytest.param(None, False, id="absent_module"),
            pytest.param("", False, id="absent_function"),
            pytest.param("import _aklr_absent_dependency\n", True, id="failing_import"),
            pytest.param("raise RuntimeError('build failed')\n", True, id="failing_build"),
        ],
    )
    def test_unavailable_entry_is_skipped(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        source: str | None,
        warns: bool,
    ) -> None:
        module = f"_aklr_{tmp_path.name}"
        if source is not None:
            (tmp_path / f"{module}.py").write_text(source)
        monkeypatch.syspath_prepend(tmp_path)
        registry = SeamRegistry()
        registry.add(Adapter(f"{module}.gemm", gemm_bf16_nt, _bind))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            registry.install()
        assert any(module in str(w.message) for w in caught) is warns


class TestTracing:
    @staticmethod
    def _traced() -> dict[str, list[list[Workload]]]:
        collection = SchemaCollection.from_path(str(RuntimeEnv().schema_path))
        return {name: [s.workloads for s in schemas] for name, schemas in collection.schemas.items()}

    def test_records_each_workload_the_backend_runs(self, seam: types.ModuleType) -> None:
        call = OpCase.from_id("gemm_bf16_nt").call()
        with TracingRuntime(TracingConfig()):
            for m in (4, 4, 8):
                assert seam.gemm(call["a"][:1].expand(m, -1), B=call["b"]) == "backend"
        assert self._traced() == {
            "gemm_bf16_nt_n32_k64": [[Workload(axes={"M": 4}), Workload(axes={"M": 8})]]
        }

    def test_writes_the_standard_beside_the_schemas(self, seam: types.ModuleType) -> None:
        case = OpCase.from_id("gemm_bf16_nt")
        call = case.call()
        with TracingRuntime(TracingConfig()):
            seam.gemm(call["a"], B=call["b"])
        directory = RuntimeEnv().schema_path / "standards" / "gemm_bf16_nt_n32_k64"
        assert sorted(p.name for p in directory.iterdir()) == [
            "baseline.py",
            "check.py",
            "init.py",
            "reference.py",
        ]
        # each runs on its own, without AKLR: as the template's functions do
        init, reference, check = (
            runpy.run_path(str(directory / f"{n}.py"))["run"] for n in ("init", "reference", "check")
        )
        drawn = init(M=4, N=32, K=64, device="cpu", seed=0)
        standard = case.template.standard
        assert drawn.keys() == standard.init(M=4, N=32, K=64, device="cpu", seed=0).keys()
        assert torch.equal(drawn["a"], standard.init(M=4, N=32, K=64, device="cpu", seed=0)["a"])
        expected = reference(a=drawn["a"], b=drawn["b"])
        assert torch.equal(expected, standard.reference(a=drawn["a"], b=drawn["b"]))
        assert check([expected], [expected]) and not check([expected], [expected + 1])

    def test_keeps_the_first_entry_of_each_workload(self) -> None:
        policy = TracingConfig().create_filter_policy()
        for order, m in enumerate((4, 4, 8)):
            policy.submit(WorkloadEntry("gemm", {"M": m}, {}, order))
        assert [entry.axes for entry in policy.drain()] == [{"M": 4}, {"M": 8}]

    def test_adds_to_the_recorded_workloads(self, seam: types.ModuleType) -> None:
        case = OpCase.from_id("gemm_bf16_nt")
        SchemaCollection.from_path(str(RuntimeEnv().schema_path)).add_workloads(
            [(case.schema(), Workload(axes={"M": 2}))]
        )
        call = case.call()
        with TracingRuntime(TracingConfig()):
            seam.gemm(call["a"], B=call["b"])
        assert self._traced() == {
            "gemm_bf16_nt_n32_k64": [[Workload(axes={"M": 2}), Workload(axes={"M": 4})]]
        }

    @pytest.mark.gpu
    @pytest.mark.usefixtures("builder_registry")
    def test_skips_a_served_call(self, seam: types.ModuleType) -> None:
        case = OpCase.from_id("gemm_bf16_nt")
        call = case.call()
        with dispatcher(case.served(GEMM_KERNEL)), TracingRuntime(TracingConfig()):
            seam.gemm(call["a"], B=call["b"])
        assert self._traced() == {}

    def test_skips_a_call_of_an_unnamed_dtype(self, seam: types.ModuleType) -> None:
        call = OpCase.from_id("gemm_bf16_nt").call()
        with TracingRuntime(TracingConfig()):
            assert seam.gemm(call["a"].double(), B=call["b"].double()) == "backend"
        assert self._traced() == {}

    def test_failing_trace_warns_once_and_runs_the_backend(self, seam: types.ModuleType) -> None:
        call = OpCase.from_id("gemm_bf16_nt").call()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with TracingRuntime(TracingConfig(input_dump_policy=["x"])):
                for _ in range(2):
                    assert seam.gemm(call["a"], B=call["b"]) == "backend"
        assert [w.filename for w in caught if "tracing failed" in str(w.message)] == [__file__]
        assert self._traced() == {}
