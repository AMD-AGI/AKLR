# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import os
import sys
from collections.abc import Iterator

import pytest
import torch

from aklr.core.platform.device import DeviceTarget
from aklr.core.platform.env import ENV_BUILD_CACHE_DIR, ENV_DISABLE, ENV_SCHEMA_PATH
from aklr.core.runtime.builder import (
    BuilderRegistry,
    FlyDSLBuilder,
    TorchBuilder,
    TritonBuilder,
)
from tests.op_case import OpCase


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "gpu(n=1): needs n GPUs and the backends (aiter, sglang)")
    config.addinivalue_line("markers", "gfx950: needs a gfx950 GPU (the committed schemas' target)")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    # the probe below initializes the GPU: a kernel tracer imported with a test module would
    # register in pytest's own process, and a failure there would read as "no GPU"
    assert "aklr.core.utils.testing" not in sys.modules, (
        "a test module imports aklr.core.utils.testing at collection; import it in the process that times"
    )
    gpu_count = torch.cuda.device_count()
    has_gfx950 = gpu_count > 0 and DeviceTarget.current().arch == "gfx950"
    for item in items:
        if (gpu := item.get_closest_marker("gpu")) and gpu_count < (needed := gpu.args[0] if gpu.args else 1):
            item.add_marker(pytest.mark.skip(reason=f"needs {needed} GPU(s)"))
        if item.get_closest_marker("gfx950") and not has_gfx950:
            item.add_marker(pytest.mark.skip(reason="needs a gfx950 GPU"))


@pytest.fixture(autouse=True)
def aklr_env(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory) -> None:
    for name in [n for n in os.environ if n.startswith("AKLR_")]:
        monkeypatch.delenv(name)
    monkeypatch.setenv(ENV_BUILD_CACHE_DIR, str(tmp_path_factory.mktemp("kernel_cache")))
    monkeypatch.setenv(ENV_SCHEMA_PATH, str(tmp_path_factory.mktemp("dataset")))
    monkeypatch.setenv(ENV_DISABLE, "1")


@pytest.fixture(params=OpCase.all(), ids=lambda case: case.id)
def op_case(request: pytest.FixtureRequest) -> OpCase:
    return request.param


@pytest.fixture
def builder_registry() -> Iterator[BuilderRegistry]:
    registry = BuilderRegistry([TritonBuilder(), FlyDSLBuilder(), TorchBuilder()])
    BuilderRegistry.set_instance(registry)
    try:
        yield registry
    finally:
        registry.cleanup()
        BuilderRegistry.set_instance(None)
