# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from aklr.core.platform.device import DeviceTarget
from aklr.core.platform.env import ENV_KERNEL_CACHE_DIR
from aklr.core.runtime.builder import (
    BuilderRegistry,
    FlyDSLBuilder,
    HipBuilder,
    TritonBuilder,
)
from aklr.core.runtime.dispatch import DispatcherRegistry, FallbackReason
from aklr.core.runtime.dispatch._trace import trace
from tests.op_case import OpCase


def pytest_configure(config: pytest.Config) -> None:
    """Register the markers: ``gpu``, the test needs a GPU and the backends (aiter, sglang); ``gfx950``."""
    config.addinivalue_line("markers", "gpu: needs a GPU and the backends (aiter, sglang)")
    config.addinivalue_line("markers", "gfx950: needs a gfx950 GPU (the committed schemas' target)")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip every ``gpu`` test unless ``torch.cuda.is_available()``."""
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
    registry = BuilderRegistry([TritonBuilder(), FlyDSLBuilder(), HipBuilder()])
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
