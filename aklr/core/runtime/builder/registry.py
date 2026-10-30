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
# not the MIT license that covers the rest of aiter (see LICENSE).
###############################################################################


from __future__ import annotations

from contextlib import ExitStack
from typing import TYPE_CHECKING, ClassVar

import filelock

from aklr.core.runtime.builder.builder import (
    _CACHE_HINT,
    Builder,
    BuildError,
    FlyDSLBuilder,
    HipBuilder,
    PythonBuilder,
    TritonBuilder,
    get_build_cache_path,
)
from aklr.core.runtime.builder.runnable import Runnable

if TYPE_CHECKING:
    from aklr.core.contract.op import OpContract

__all__ = ["BuilderRegistry"]

# the order of get_instance's builders; each claims only its own language
_BUILDER_PRIORITY: list[type[Builder]] = [
    TritonBuilder,
    FlyDSLBuilder,
    HipBuilder,
    PythonBuilder,
]


class BuilderRegistry:
    """The available builders, and one built :class:`Runnable` per kernel content hash."""

    _instance: ClassVar[BuilderRegistry | None] = None

    _builders: list[Builder]
    _hash_to_built: dict[str, Runnable | BuildError]

    def __init__(self, builders: list[Builder]) -> None:
        if len(builders) == 0:
            raise ValueError("BuilderRegistry requires at least one builder")
        self._builders = list(builders)
        self._hash_to_built = {}

    @classmethod
    def get_instance(cls) -> BuilderRegistry:
        """The process registry, of every ``_BUILDER_PRIORITY`` builder whose ``is_available()``."""
        if cls._instance is None:
            cls._instance = BuilderRegistry(
                [builder_type() for builder_type in _BUILDER_PRIORITY if builder_type.is_available()]
            )
        return cls._instance

    @classmethod
    def set_instance(cls, instance: BuilderRegistry | None) -> None:
        """Put ``instance`` in place of the process registry; ``None``: made afresh on next use."""
        cls._instance = instance

    def build(self, schema: OpContract) -> Runnable:
        """The runnable of ``schema.kernel``: built by the first builder that ``can_build`` it."""
        assert schema.kernel is not None
        content_hash = schema.kernel.content_hash
        built = self._hash_to_built.get(content_hash)
        if isinstance(built, BuildError):
            raise built
        if built is not None:
            return built
        try:
            root = get_build_cache_path()
        except BuildError as e:
            if not isinstance(e.__cause__, OSError):
                self._hash_to_built[content_hash] = e
            raise
        lock_path = root / "build_locks" / f"{content_hash}.lock"
        lock_stack = ExitStack()
        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            lock_stack.enter_context(filelock.FileLock(lock_path))
        except OSError as e:
            raise BuildError(f"cannot lock {lock_path}: {e}; {_CACHE_HINT}") from e
        # only taking the lock is wrapped above: the build's own errors pass through
        with lock_stack:
            built = self._hash_to_built.get(content_hash)
            if isinstance(built, BuildError):
                raise built
            if built is not None:
                return built
            builder = next((b for b in self._builders if b.can_build(schema)), None)
            if builder is None:
                error = BuildError(f"no builder for language {schema.kernel.language!r}")
                self._hash_to_built[content_hash] = error
                raise error
            try:
                runnable = builder.build(schema)
            except BuildError as e:
                if not isinstance(e.__cause__, OSError):
                    self._hash_to_built[content_hash] = e
                raise
            self._hash_to_built[content_hash] = runnable
            return runnable

    def cleanup(self) -> None:
        """``cleanup()`` every built runnable, its errors ignored; then forget all."""
        for built in self._hash_to_built.values():
            if isinstance(built, Runnable):
                try:
                    built.cleanup()
                except Exception:  # noqa: BLE001, S110
                    # Continue cleaning other runnables.
                    pass
        self._hash_to_built.clear()
