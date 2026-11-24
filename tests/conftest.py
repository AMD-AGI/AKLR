# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import os
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import torch

from aklr.core.contract.op import OpContract
from aklr.core.platform.device import DeviceTarget
from aklr.core.platform.env import ENV_KERNEL_CACHE_DIR
from aklr.core.runtime.builder import (
    BuilderRegistry,
    FlyDSLBuilder,
    HipBuilder,
    HsacoBuilder,
    TritonBuilder,
)
from aklr.core.runtime.dispatch import DispatcherRegistry, DispatchTable, FallbackReason, trace
from tests.op_case import KERNEL_TARGET, OpCase


def pytest_configure(config: pytest.Config) -> None:
    """Register the markers: ``gpu``, the test needs a GPU and the backends (aiter, sglang); ``gfx950``."""
    config.addinivalue_line("markers", "gpu: needs a GPU and the backends (aiter, sglang)")
    config.addinivalue_line("markers", "gfx950: needs a gfx950 GPU (the committed schemas' target)")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip every ``gpu`` test unless ``torch.cuda.is_available()``."""
    # the probe below initializes the GPU: a kernel tracer imported with a test module would
    # register in pytest's own process, and a failure there would read as "no GPU"
    assert "aklr.core.utils.testing" not in sys.modules, (
        "a test module imports aklr.core.utils.testing at collection; import it in the process that times"
    )
    has_gpu = torch.cuda.is_available()
    has_gfx950 = has_gpu and DeviceTarget.current().arch == "gfx950"
    for item in items:
        if item.get_closest_marker("gpu") and not has_gpu:
            item.add_marker(pytest.mark.skip(reason="needs a GPU"))
        if item.get_closest_marker("gfx950") and not has_gfx950:
            item.add_marker(pytest.mark.skip(reason="needs a gfx950 GPU"))


@pytest.fixture(autouse=True)
def aklr_env(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory) -> None:
    """Every ``AKLR_*`` variable unset, ``AKLR_KERNEL_CACHE_DIR`` a fresh temporary directory."""
    for name in [n for n in os.environ if n.startswith("AKLR_")]:
        monkeypatch.delenv(name)
    monkeypatch.setenv(ENV_KERNEL_CACHE_DIR, str(tmp_path_factory.mktemp("kernel_cache")))
    trace.cache_clear()


@pytest.fixture(params=OpCase.all(), ids=lambda case: case.contract_id)
def op_case(request: pytest.FixtureRequest) -> OpCase:
    """Each registered contract's ``OpCase`` in turn."""
    return request.param


@pytest.fixture
def builder_registry() -> Iterator[BuilderRegistry]:
    """A fresh process ``BuilderRegistry`` of the committed schemas' builders, installed with ``set_instance``."""
    registry = BuilderRegistry([TritonBuilder(), FlyDSLBuilder(), HipBuilder(), HsacoBuilder()])
    BuilderRegistry.set_instance(registry)
    try:
        yield registry
    finally:
        registry.cleanup()
        BuilderRegistry.set_instance(None)


@pytest.fixture(scope="session")
def committed_kernel_cache(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One kernel cache for every init over the committed schemas: kernels are cached
    by content hash, so each is built once per session, not once per test."""
    return tmp_path_factory.mktemp("committed_kernel_cache")


@pytest.fixture
def default_dispatchers(
    builder_registry: BuilderRegistry,
    committed_kernel_cache: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[DispatcherRegistry]:
    """``DispatcherRegistry.default()`` with every dispatcher closed (``NO_SCHEMA``), no
    recorder, building into ``committed_kernel_cache``."""
    monkeypatch.setenv(ENV_KERNEL_CACHE_DIR, str(committed_kernel_cache))
    registry = DispatcherRegistry.default()
    for dispatcher in registry.dispatchers().values():
        dispatcher.close(FallbackReason.NO_SCHEMA)
    registry.recorder, registry.is_initialized = None, False
    try:
        yield registry
    finally:
        for dispatcher in registry.dispatchers().values():
            dispatcher.close(FallbackReason.NO_SCHEMA)
        registry.recorder, registry.is_initialized = None, False


@pytest.fixture
def family_modules() -> Iterator[None]:
    """Forgets, after the test, the family modules it imported from a ``FakeHost``'s temporary
    ``aklr/plugins/<ns>/``."""
    before = set(sys.modules)
    yield
    for name in [n for n in sys.modules if n.startswith("aklr.plugins.") and n not in before]:
        del sys.modules[name]
        parent, _, child = name.rpartition(".")
        if hasattr(sys.modules.get(parent), child):
            delattr(sys.modules[parent], child)


@pytest.fixture
def dispatch_starts(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """A fresh process ``DispatcherRegistry``, nothing loaded; each ``init_once`` that would load is
    recorded here instead of run."""
    registry = DispatcherRegistry()
    monkeypatch.setattr(DispatcherRegistry, "default", classmethod(lambda cls: registry))
    starts: list[str] = []

    def init_once(registry: DispatcherRegistry) -> None:
        if not registry.is_initialized:
            starts.append("init_once")
            registry.is_initialized = True

    monkeypatch.setattr(DispatcherRegistry, "init_once", init_once)
    return starts


@pytest.fixture
def serve(
    builder_registry: BuilderRegistry, dispatch_starts: list[str]
) -> Callable[[OpContract, str], dict[str, Any]]:
    """Serves a contract of Kimi-K3's tiny gemm calls at its small Point, by a kernel of the given body
    over ``x``, ``w`` and ``out``; returns a call at that Point."""

    def serve(contract: OpContract, kernel_body: str) -> dict[str, Any]:
        case = OpCase.of("kimi_k3_tiny_gemm@1.0")
        point = case.point.model_copy(update={"contract_id": contract.contract_id})
        source = f"def run(*, x, w, out, **_):\n    {kernel_body}\n"
        schema = contract.narrow(
            point,
            workloads=[contract.workload_for(point, {})],
            kernel={
                "language": "flydsl",
                "targets": [KERNEL_TARGET],
                "entry_point": "kernel.py::run",
                "sources": [{"path": "kernel.py", "content": source}],
            },
        )
        table = DispatchTable.from_schemas(
            contract,
            torch.device("cpu"),
            [(schema, builder_registry.build(schema), schema.workloads)],
            min_speedup=0.0,
        )
        table.warmed_keys.update(table.runnables)
        DispatcherRegistry.default().dispatcher(contract).set_table(table)
        return case.call(case.inputs())

    return serve
