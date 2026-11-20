###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 by FlashInfer-Bench Contributors
#
#
###############################################################################

from __future__ import annotations

from typing import ClassVar

import filelock

from aklr.core.platform.env import RuntimeEnv
from aklr.core.runtime.builder.builder import Builder, BuildError
from aklr.core.runtime.builder.runnable import Runnable
from aklr.core.schema import Schema

__all__ = ["BuilderRegistry"]


class BuilderRegistry:
    _instance: ClassVar[BuilderRegistry | None] = None

    _cache: dict[str, Runnable | BuildError]

    def __init__(self) -> None:
        self._cache = {}

    @classmethod
    def get_instance(cls) -> BuilderRegistry:
        if cls._instance is None:
            cls._instance = BuilderRegistry()
        return cls._instance

    @classmethod
    def set_instance(cls, instance: BuilderRegistry | None) -> None:
        cls._instance = instance

    def build(self, schema: Schema) -> Runnable:
        assert schema.kernel is not None
        hash = schema.kernel.hash()
        if hash in self._cache:
            cached = self._cache[hash]
            if isinstance(cached, BuildError):
                raise cached
            return cached

        lock_dir = RuntimeEnv().build_cache_path / "build_locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_file = lock_dir / f"{hash}.lock"

        with filelock.FileLock(lock_file):
            if hash in self._cache:
                cached = self._cache[hash]
                if isinstance(cached, BuildError):
                    raise cached
                return cached

            language = schema.kernel.build.language
            try:
                try:
                    builder_type = Builder.resolve(language)
                except ValueError as e:
                    raise BuildError(f"kernel '{schema.kernel.name}': {e}") from e
                if not builder_type.is_available():
                    raise BuildError(
                        f"kernel '{schema.kernel.name}': the {language} toolchain is not available"
                    )
                runnable = builder_type().build(schema)
            except BuildError as e:
                self._cache[hash] = e
                raise
            else:
                self._cache[hash] = runnable

        return runnable

    def cleanup(self) -> None:
        for cached in self._cache.values():
            if isinstance(cached, Runnable):
                try:
                    cached.cleanup()
                except Exception:  # noqa: BLE001, S110
                    pass
        self._cache.clear()
