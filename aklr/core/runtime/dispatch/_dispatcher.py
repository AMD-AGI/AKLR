###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 by FlashInfer-Bench Contributors
#
#


from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any, ClassVar

from typing_extensions import Self

from aklr.core.platform.device import DeviceTarget
from aklr.core.platform.env import RuntimeEnv
from aklr.core.runtime.dispatch._config import DispatchConfig, DispatchConfigRegistry, get_default_registry
from aklr.core.runtime.dispatch._dispatch_key import DispatchKey
from aklr.core.runtime.dispatch._kernel_tuner import KernelTuner
from aklr.core.runtime.dispatch._operator_entry import OperatorEntry
from aklr.core.schema import Schema, SchemaCollection

logger = logging.getLogger(__name__)


class Dispatcher:
    _instance: ClassVar[Dispatcher | None] = None
    _env_initialized: ClassVar[bool] = False
    _env_error: ClassVar[Exception | None] = None
    _env_lock: ClassVar[threading.RLock] = threading.RLock()

    @classmethod
    def get_instance(cls) -> Dispatcher | None:
        if cls._instance is None:
            with cls._env_lock:
                if not cls._env_initialized:
                    cls._env_initialized = True
                    try:
                        if not RuntimeEnv().is_all_disabled:
                            cls._instance = cls(SchemaCollection.from_path())
                    except Exception as e:
                        cls._env_error = e
            if cls._instance is None and cls._env_error is not None:
                raise RuntimeError(
                    f"AKLR's dispatcher could not start from the environment: {cls._env_error}"
                ) from cls._env_error
        return cls._instance

    def __init__(
        self,
        schema_collection: SchemaCollection,
        dispatch_config: DispatchConfig | DispatchConfigRegistry | None = None,
        tuner: KernelTuner | None = None,
    ) -> None:
        self._tuner = tuner
        if dispatch_config is None:
            configs = get_default_registry()
        elif isinstance(dispatch_config, DispatchConfig):
            configs = DispatchConfigRegistry(default=dispatch_config)
        else:
            configs = dispatch_config

        target = DeviceTarget.current()
        disabled = RuntimeEnv().disabled_op_types
        by_op: dict[str, list[Schema]] = {}
        for schemas in schema_collection.schemas.values():
            for schema in schemas:
                if schema.op_type not in disabled:
                    by_op.setdefault(schema.op_name, []).append(schema)
        baselines = {
            baseline
            for schemas in by_op.values()
            for schema in schemas
            for workload in schema.workloads
            if (evaluation := workload.evaluation(target)) is not None
            for baseline in evaluation.speedup
        }
        self._entries: dict[str, OperatorEntry] = {}
        for op_name, schemas in by_op.items():
            config = configs.get(op_name)
            if config is not None:
                entry = OperatorEntry(schemas, config, target, RuntimeEnv().min_speedup, baselines)
                entry.prewarm()
                self._entries[op_name] = entry

        from aklr.core.runtime.seam import SeamRegistry

        SeamRegistry.default().install()

    def start(self) -> None:
        if Dispatcher._instance is self:
            return
        if Dispatcher._instance is not None:
            raise RuntimeError("Another Dispatcher is already active.")
        Dispatcher._instance = self

    def stop(self) -> None:
        if Dispatcher._instance is not self:
            raise RuntimeError("Cannot stop a Dispatcher that is not the current active instance.")
        Dispatcher._instance = None

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.stop()
        return False

    def call(
        self,
        schema: Schema,
        call: dict[str, Any],
        *,
        fallback: Callable[[], Any] | None = None,
        baseline: str | None = None,
        is_inplace: bool = False,
    ) -> Any:
        entry = self._entries.get(schema.op_name)
        kernel = None
        try:
            key = None if entry is None else DispatchKey.from_call(schema, call)
        except ValueError as e:
            if fallback is None:
                raise RuntimeError(f"Dispatch miss for '{schema.op_name}' and no fallback provided") from e
            key = None
        if key is not None:
            axes = dict(key.consts + key.vars)
            kernel = entry.lookup(key, baseline)
            if (
                self._tuner is not None
                and entry.slot(baseline) == OperatorEntry.UNMEASURED
                and len(candidates := entry.candidates(key, baseline)) > 1
            ):
                by_name = {candidate.name: candidate for candidate in candidates}
                kernel = by_name[
                    self._tuner.choose(
                        schema.op_name, key, list(by_name), lambda name: by_name[name](call, axes, is_inplace)
                    )
                ]
        if kernel is None:
            if fallback is None:
                raise RuntimeError(f"Dispatch miss for '{schema.op_name}' and no fallback provided")
            return fallback()
        return kernel(call, axes, is_inplace)
