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

import logging
from functools import cache
from typing import Any, Callable, ClassVar, Dict, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field
from typing_extensions import Self

from aklr.core.platform.device import DeviceTarget
from aklr.core.platform.env import RuntimeEnv
from aklr.core.runtime._config_registry import ConfigRegistry
from aklr.core.runtime.builder.registry import BuilderRegistry
from aklr.core.runtime.dispatch._adapt import adapt_and_call, ordered_input_values
from aklr.core.runtime.dispatch._dispatch_key import (
    DispatchKeyBuilder,
    DispatchKeyFactory,
)
from aklr.core.runtime.dispatch._dispatch_table import DispatchTable
from aklr.core.runtime.dispatch._kernel_tuner import KernelTuner
from aklr.core.schema._collection import SchemaCollection

logger = logging.getLogger(__name__)


class DispatchConfig(BaseModel):
    aot_ratio: float = Field(default=1.0, ge=0, le=1)
    """The share of each name's key-winning kernels built when the table loads, most wins first."""
    on_miss_policy: Literal["fallback_only", "use_best_kernel"] = "fallback_only"
    """On a key the table lacks: run the fallback, or the kernel ranked first on the most keys."""


class DispatchConfigRegistry(ConfigRegistry[DispatchConfig]):
    """Dispatch configs by schema name; a name with no config and no default is never dispatched."""


@cache
def get_default_registry() -> DispatchConfigRegistry:
    return DispatchConfigRegistry(default=DispatchConfig())


class KernelDispatcher:
    """Sends each api call to the best kernel the dispatch table has for it, else to its fallback."""

    _instance: ClassVar[Optional[KernelDispatcher]] = None
    _env_initialized: ClassVar[bool] = False

    @classmethod
    def get_instance(cls) -> Optional[KernelDispatcher]:
        """The active dispatcher; the first call starts one from the env unless AKLR_DISABLE=1."""
        if cls._instance is None and not cls._env_initialized:
            cls._env_initialized = True
            if not RuntimeEnv().is_all_disabled:
                cls._instance = cls(SchemaCollection.from_path())
        return cls._instance

    def __init__(
        self,
        schema_collection: SchemaCollection,
        dispatch_config: Union[DispatchConfig, DispatchConfigRegistry, None] = None,
        tuner: Optional[KernelTuner] = None,
    ) -> None:
        self._schema_collection = schema_collection
        self._tuner = tuner

        if dispatch_config is None:
            self._config_registry = get_default_registry()
        elif isinstance(dispatch_config, DispatchConfig):
            self._config_registry = DispatchConfigRegistry(default=dispatch_config)
        else:
            self._config_registry = dispatch_config

        self._table = DispatchTable.load_or_build(
            self._schema_collection,
            self._config_registry,
            DeviceTarget.current(),
            RuntimeEnv().min_speedup,
        )

        self._key_builders: Dict[str, DispatchKeyBuilder] = {}

        from aklr.core.runtime.seam import SeamRegistry  # seam imports this module

        SeamRegistry.default().install()

    def start(self) -> None:
        if KernelDispatcher._instance is self:
            return
        if KernelDispatcher._instance is not None:
            raise RuntimeError("Another KernelDispatcher is already active.")
        KernelDispatcher._instance = self

    def stop(self) -> None:
        if KernelDispatcher._instance is not self:
            raise RuntimeError("Cannot stop a KernelDispatcher that is not the current active instance.")
        KernelDispatcher._instance = None

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.stop()
        return False

    def dispatch(
        self,
        schema_name: str,
        args: Tuple[Any, ...],
        kwargs: dict[str, Any],
        fallback: Optional[Callable[..., Any]],
        baseline: Optional[str] = None,
        is_inplace: bool = False,
    ) -> Any:
        """Run the call on the best kernel for its key and ``baseline``, else on ``fallback``."""
        dispatch_config = self._config_registry.get(schema_name)
        if dispatch_config is None:
            if fallback is None:
                raise RuntimeError(
                    f"Dispatch config not configured for '{schema_name}' and no fallback provided"
                )
            return fallback(*args, **kwargs)

        schemas = self._schema_collection.schemas.get(schema_name)
        if not schemas:
            if fallback is None:
                raise RuntimeError(f"Schema '{schema_name}' not found and no fallback provided")
            return fallback(*args, **kwargs)
        schema = schemas[0]  # a name's files agree outside their kernel and workloads

        names = [*schema.inputs, *schema.outputs]
        if len(args) > len(names):
            raise TypeError(
                f"Too many positional arguments for '{schema_name}': got {len(args)}, "
                f"expected at most {len(names)}"
            )
        call = {**dict(zip(names, args)), **kwargs}
        dests = {name: name for name in schema.outputs}

        builder = self._key_builders.get(schema.name)
        if builder is None:
            builder = DispatchKeyFactory.specialize(schema)
            self._key_builders[schema.name] = builder
        key = builder.build_from_args(ordered_input_values(schema, call))

        kernel_name = self._table.match_kernel(schema_name, key, baseline)
        candidates = self._table.candidates(schema_name, key, baseline)
        if (
            self._tuner is not None
            and len(candidates) > 1
            and self._table.slot(baseline) == DispatchTable.UNMEASURED
        ):

            def run(name: str) -> Any:
                candidate = BuilderRegistry.get_instance().build(self._schema_collection.get_schema(name))
                return adapt_and_call(
                    template=schema,
                    fn=candidate,
                    namespace=call,
                    dps=candidate.metadata.destination_passing_style,
                    is_inplace=is_inplace,
                    dests=dests,
                    axes=dict(key.axes),
                )

            kernel_name = self._tuner.choose(schema_name, key, candidates, run)
        runnable = None
        if kernel_name:
            kernel_schema = self._schema_collection.get_schema(kernel_name)
            if kernel_schema:
                runnable = BuilderRegistry.get_instance().build(kernel_schema)

        if runnable is None:
            if dispatch_config.on_miss_policy == "use_best_kernel":
                best_kernel_name = self._table.match_best_kernel(schema_name, baseline)
                kernel_schema = (
                    self._schema_collection.get_schema(best_kernel_name) if best_kernel_name else None
                )
                if kernel_schema:
                    runnable = BuilderRegistry.get_instance().build(kernel_schema)

        if runnable is None:
            if fallback is None:
                raise RuntimeError(f"Dispatch miss for '{schema_name}' and no fallback provided")
            return fallback(*args, **kwargs)

        return adapt_and_call(
            template=schema,
            fn=runnable,
            namespace=call,
            dps=runnable.metadata.destination_passing_style,
            is_inplace=is_inplace,
            dests=dests,
            axes=dict(key.axes),
        )
