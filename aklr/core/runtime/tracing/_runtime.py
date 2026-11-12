###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 by FlashInfer-Bench Contributors
#
# Adapted from FlashInfer-Bench (https://github.com/flashinfer-ai/flashinfer-bench)
# Modified by the AKLR team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of AKLR (see LICENSE).
###############################################################################

from __future__ import annotations

import atexit
import logging
import os
import signal
import threading
from typing import Any, ClassVar

import torch

from aklr.core.platform.env import RuntimeEnv
from aklr.core.runtime.tracing._config import TracingConfig, TracingConfigRegistry
from aklr.core.runtime.tracing._policy import FilterPolicy
from aklr.core.runtime.tracing._presets import get_default_configs
from aklr.core.runtime.tracing._workload_entry import WorkloadEntry
from aklr.core.schema import Schema
from aklr.core.schema._collection import SchemaCollection, TraceDumpLayout
from aklr.core.schema._workload import Workload
from aklr.core.utils import dtype_str_to_torch_dtype

logger = logging.getLogger(__name__)


class TracingRuntime:
    """Collects traced calls and flushes them as workloads into the dataset's kernel-less schemas, each one's
    standard beside them in a dump."""

    _instance: ClassVar[TracingRuntime | None] = None
    _cleanup_registered: ClassVar[bool] = False
    _env_initialized: ClassVar[bool] = False

    @classmethod
    def get_instance(cls) -> TracingRuntime | None:
        """The active runtime; the first call starts one if AKLR_ENABLE_TRACING=1."""
        if cls._instance is None and not cls._env_initialized:
            cls._env_initialized = True
            if RuntimeEnv().is_tracing_enabled:
                cls().start()
        return cls._instance

    def __init__(
        self,
        tracing_config: (TracingConfig | dict[str, TracingConfig] | TracingConfigRegistry | None) = None,
    ):
        if tracing_config is None:
            self._config_registry = get_default_configs()
        elif isinstance(tracing_config, TracingConfig):
            self._config_registry = TracingConfigRegistry(default=tracing_config)
        elif isinstance(tracing_config, dict):
            self._config_registry = TracingConfigRegistry(per_schema=tracing_config)
        else:
            self._config_registry = tracing_config

        self.order_counter = 0

        self._lock = threading.Lock()

        self._cuda_graph_entries: list[WorkloadEntry] = []
        self._in_cuda_graph = False

        self._filter_policies: dict[str, FilterPolicy] = {}
        self._schemas: dict[str, Schema | None] = {}
        self._schema_collection = SchemaCollection.from_path()

        for schema_name, config in self._config_registry.per_schema.items():
            self._filter_policies[schema_name] = config.create_filter_policy()

        from aklr.core.runtime.seam import SeamRegistry  # seam imports this module

        SeamRegistry.default().install()

    def start(self):
        if TracingRuntime._instance is self:
            return
        if TracingRuntime._instance is not None:
            raise RuntimeError("Another TracingRuntime is already active.")
        TracingRuntime._register_cleanup()
        TracingRuntime._instance = self

        logger.info("TracingRuntime started")
        default_status = "enabled" if self._config_registry.default is not None else "disabled"
        logger.info(
            f"  Tracing configs: default={default_status}, "
            f"{len(self._config_registry.per_schema)} schemas with custom config"
        )

    def stop(self):
        if TracingRuntime._instance is not self:
            raise RuntimeError("Cannot stop a TracingRuntime that is not the current active instance.")

        self.flush()
        TracingRuntime._instance = None
        logger.info("TracingRuntime stopped")

    def collect(
        self,
        schema: Schema,
        schema_name: str,
        namespace: dict[str, Any],
    ):
        """Record one call (its bound api arguments) as a workload of ``schema_name``."""
        logger.info(f"Tracing '{schema_name}'")

        tracing_config = self._config_registry.get(schema_name)
        if tracing_config is None:
            logger.debug(f"Tracing config not configured for {schema_name}, skipping")
            return

        if schema_name not in self._schemas:
            try:
                definition = schema.for_call(namespace)
                # a name the layout cannot place fails here, once, so flush never raises
                self._schema_collection.layout.path(definition)
            except ValueError as e:
                logger.error(f"Not tracing {schema_name}: {e}")
                definition = None
            self._schemas.setdefault(schema_name, definition)
        definition = self._schemas[schema_name]
        if definition is None:
            return

        from aklr.core.runtime.dispatch._adapt import ordered_input_values  # dispatch imports tracing

        input_names = list(definition.inputs.keys())
        input_values = ordered_input_values(schema, namespace)

        try:
            axes = definition.get_axes_values_from_inputs(input_values)
        except ValueError as e:
            logger.error(f"Error getting axis values for {schema_name}: {e}")
            return

        inputs_to_dump = tracing_config.get_inputs_to_dump(input_names, input_values)
        inputs_to_dump_tensors: dict[str, torch.Tensor] = {}
        for name, val in inputs_to_dump.items():
            try:
                inputs_to_dump_tensors[name] = self._convert_arg_to_tensor(val, definition.inputs[name].dtype)
            except ValueError as e:
                logger.error(f"Error converting argument '{name}' to tensor for {schema_name}: {e}")
                return

        entry = WorkloadEntry(
            schema_name=schema_name,
            axes=axes,
            inputs_to_dump=inputs_to_dump_tensors,
            order=self.order_counter,
        )

        with self._lock:
            if schema_name not in self._filter_policies:
                self._filter_policies[schema_name] = tracing_config.create_filter_policy()
            if self._in_cuda_graph:
                self._cuda_graph_entries.append(entry)
            else:
                self._filter_policies[schema_name].submit(entry)

            self.order_counter += 1

    def _convert_arg_to_tensor(
        self, val: float | bool | list | tuple | torch.Tensor, dtype: str
    ) -> torch.Tensor | None:
        if isinstance(val, (int, float, bool, list, tuple)):
            val = torch.tensor(val, dtype=dtype_str_to_torch_dtype(dtype))
            return val
        elif isinstance(val, torch.Tensor):
            return val.detach().cpu().clone()
        else:
            raise ValueError(f"Unsupported argument type: {type(val)}")

    def _snapshot_graph_tensors(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        for entry in self._cuda_graph_entries:
            snapshot = {}
            for name, tensor in entry.inputs_to_dump.items():
                if isinstance(tensor, torch.Tensor):
                    snapshot[name] = tensor.detach().cpu().clone()
                else:
                    snapshot[name] = tensor
            entry.cuda_graph_snapshot = snapshot
            entry.inputs_to_dump = snapshot

            filter_policy = self._filter_policies.get(entry.schema_name)
            if filter_policy is not None:
                filter_policy.submit(entry)

        self._cuda_graph_entries.clear()

    def flush(self):
        """Write the entries the filter policies kept into the dataset, as workloads."""
        num_selected_entries = 0

        if self._in_cuda_graph:
            raise RuntimeError("Cannot flush during CUDA Graph replay")

        with self._lock:
            drained = {
                schema_name: filter_policy.drain()
                for schema_name, filter_policy in self._filter_policies.items()
            }

        for schema_name, selected_entries in drained.items():
            if len(selected_entries) == 0:
                continue

            definition = self._schemas[schema_name]
            self._schema_collection.add_workloads(
                [(definition, Workload(axes=entry.axes)) for entry in selected_entries]
            )
            layout = self._schema_collection.layout
            # the json holds no standard: the dump keeps it beside, as modules that run without AKLR
            if isinstance(layout, TraceDumpLayout) and definition.standard is not None:
                definition.standard.save(layout.root / "standards" / schema_name)
            num_selected_entries += len(selected_entries)

        logger.info(f"Flush done. {num_selected_entries} entries selected")

    def __enter__(self) -> TracingRuntime:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.stop()
        return False

    @classmethod
    def _register_cleanup(cls) -> None:
        if cls._cleanup_registered:
            return
        cls._cleanup_registered = True
        atexit.register(cls._flush_all)
        # signal.signal raises off the main thread: the first start() must run there
        cls._orig_sigterm = signal.signal(signal.SIGTERM, cls._signal_cleanup)
        cls._orig_sigint = signal.signal(signal.SIGINT, cls._signal_cleanup)

    @classmethod
    def _flush_all(cls) -> None:
        if cls._instance is None:
            return
        try:
            cls._instance.flush()
        except Exception as e:
            logger.error(f"Flush failed during cleanup: {e}")

    @classmethod
    def _signal_cleanup(cls, signum: int, frame) -> None:
        cls._flush_all()

        orig_handler = cls._orig_sigterm if signum == signal.SIGTERM else cls._orig_sigint
        signal.signal(signum, orig_handler)

        if callable(orig_handler):
            orig_handler(signum, frame)
        elif orig_handler == signal.SIG_DFL:
            # the default action is back: resend, so the process still dies of this signal
            os.kill(os.getpid(), signum)
