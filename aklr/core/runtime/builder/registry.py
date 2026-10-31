###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 FlashInfer-Bench Contributors
#
# Adapted from FlashInfer-Bench (https://github.com/flashinfer-ai/flashinfer-bench)
# Source: flashinfer_bench/compile/registry.py
# Modified by Advanced Micro Devices, Inc. for the AKLR project.
# Modified for aklr/core/runtime/builder
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of AKLR (see LICENSE).
###############################################################################


from __future__ import annotations

from typing import ClassVar

import filelock

from aklr.core.platform.env import RuntimeEnv
from aklr.core.runtime.builder.builder import (
    Builder,
    BuildError,
    FlyDSLBuilder,
    PythonBuilder,
    TorchBuilder,
    TritonBuilder,
)
from aklr.core.runtime.builder.runnable import Runnable
from aklr.core.schema import Schema

__all__ = ["BuilderRegistry"]

# the order of get_instance's builders; each claims only its own language
_BUILDER_PRIORITY: list[type[Builder]] = [
    TritonBuilder,
    FlyDSLBuilder,
    TorchBuilder,
    PythonBuilder,
]


class BuilderRegistry:
    """The available builders, one kernel built per content hash, and its
    :class:`Runnable` per set of call keys."""

    _instance: ClassVar[BuilderRegistry | None] = None
    """Singleton instance of the BuilderRegistry."""

    _builders: list[Builder]
    """List of available builders in priority order."""

    _cache: dict[str, Runnable | BuildError]
    """Cache mapping solution hashes to built runnables or build errors."""

    def __init__(self, builders: list[Builder]) -> None:
        if len(builders) == 0:
            raise ValueError("BuilderRegistry requires at least one builder")
        self._builders = list(builders)
        self._cache = {}

    @classmethod
    def get_instance(cls) -> BuilderRegistry:
        """The process registry, of every ``_BUILDER_PRIORITY`` builder whose ``is_available()``."""
        if cls._instance is None:
            builders = []
            for builder_type in _BUILDER_PRIORITY:
                if builder_type.is_available():
                    builders.append(builder_type())
            cls._instance = BuilderRegistry(builders)
        return cls._instance

    @classmethod
    def set_instance(cls, instance: BuilderRegistry | None) -> None:
        """Put ``instance`` in place of the process registry; ``None``: made afresh on next use."""
        cls._instance = instance

    def build(self, schema: Schema) -> Runnable:
        """The runnable of ``schema.kernel``, called with a call's api params: the
        kernel built once by the first builder that ``can_build`` it, checked to take
        ``schema.kernel_call_keys``."""
        assert schema.kernel is not None
        hash = schema.kernel.hash()
        if hash in self._cache:
            cached = self._cache[hash]
            if isinstance(cached, BuildError):
                raise cached
            return cached

        # Use file lock to ensure process-safe build
        lock_dir = RuntimeEnv().build_cache_path / "build_locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_file = lock_dir / f"{hash}.lock"

        with filelock.FileLock(lock_file):
            # Double-check after acquiring lock
            if hash in self._cache:
                cached = self._cache[hash]
                if isinstance(cached, BuildError):
                    raise cached
                return cached

            builder = next((b for b in self._builders if b.can_build(schema)), None)
            if builder is None:
                error = BuildError(f"No registered builder can build kernel '{schema.kernel.name}'")
                self._cache[hash] = error
                raise error

            try:
                runnable = builder.build(schema)
            except BuildError as e:
                self._cache[hash] = e
                raise
            else:
                self._cache[hash] = runnable

        return runnable

    def cleanup(self) -> None:
        """``cleanup()`` every built kernel, its errors ignored; then forget all."""
        for cached in self._cache.values():
            if isinstance(cached, Runnable):
                try:
                    cached.cleanup()
                except Exception:  # noqa: BLE001, S110
                    pass
        self._cache.clear()
