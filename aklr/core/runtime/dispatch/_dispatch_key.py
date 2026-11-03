###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 by FlashInfer-Bench Contributors
#
# Adapted from FlashInfer-Bench (https://github.com/flashinfer-ai/flashinfer-bench)
# Modified by the AKLR team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of AKLR (see LICENSE).
###############################################################################

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Type, Union

from aklr.core.schema import Schema
from aklr.core.schema._frozen_model import frozen_dataclass
from aklr.core.schema._workload import Workload


@frozen_dataclass
class DispatchKey:
    """A dispatch-table key: a call's or workload's axes as sorted (name, value) pairs, plus features."""

    axes: Tuple[Tuple[str, int], ...] = ()
    feats: Tuple[Tuple[str, Union[int, float, bool]], ...] = ()


class DispatchKeyBuilder(ABC):
    def __init__(self, schema: Schema) -> None:
        self.schema = schema

    @abstractmethod
    def build_from_args(self, args: Tuple[Any, ...]) -> DispatchKey: ...

    @abstractmethod
    def build_from_workload(self, workload: Workload) -> DispatchKey: ...

    @abstractmethod
    def features(self, args: Tuple[Any, ...]) -> Tuple[Tuple[str, Any], ...]: ...


class AxesOnlyKeyBuilder(DispatchKeyBuilder):
    def build_from_args(self, args: Tuple[Any, ...]) -> DispatchKey:
        axes = self.schema.get_axes_values_from_inputs(args)
        return DispatchKey(axes=tuple(sorted(axes.items())))

    def build_from_workload(self, workload: Workload) -> DispatchKey:
        axes = workload.axes
        return DispatchKey(axes=tuple(sorted(axes.items())))

    def features(self, args: Tuple[Any, ...]) -> Tuple[Tuple[str, Any], ...]:
        return ()


class DispatchKeyFactory:
    _REGISTRY: Dict[str, Type[DispatchKeyBuilder]] = {}

    @classmethod
    def register(cls, type_name: str, builder_cls: Type[DispatchKeyBuilder]) -> None:
        cls._REGISTRY[type_name] = builder_cls

    @classmethod
    def for_type(cls, type_name: str) -> Type[DispatchKeyBuilder]:
        return cls._REGISTRY.get(type_name, AxesOnlyKeyBuilder)

    @classmethod
    def specialize(cls, schema: Schema) -> DispatchKeyBuilder:
        builder_cls = cls.for_type(schema.op_type)
        return builder_cls(schema)
