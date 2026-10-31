"""Key builders for matching workloads to kernels."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Type, Union

from aklr.core._frozen_model import frozen_dataclass
from aklr.core.schema import Schema
from aklr.core.schema._workload import Workload


@frozen_dataclass
class DispatchKey:
    """Key for matching workloads to kernels in Dispatch runtime.

    This is an immutable (frozen) model that can be used as dict keys or in sets.
    """

    axes: Tuple[Tuple[str, int], ...] = ()
    """Variable axis values as sorted (name, value) tuples."""
    feats: Tuple[Tuple[str, Union[int, float, bool]], ...] = ()
    """Additional features extracted from input tensors."""


class DispatchKeyBuilder(ABC):
    def __init__(self, schema: Schema) -> None:
        self.schema = schema

    @abstractmethod
    def build_from_args(self, args: Tuple[Any, ...]) -> DispatchKey:
        """Build a key from positional runtime arguments (inputs only)"""
        ...

    @abstractmethod
    def build_from_workload(self, workload: Workload) -> DispatchKey:
        """Build a key from offline workload trace"""
        ...

    @abstractmethod
    def features(self, args: Tuple[Any, ...]) -> Tuple[Tuple[str, Any], ...]:
        """Lightweight feature extraction from input args"""
        ...


# Key Builders


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
        # Default to AxesOnlyKeyBuilder if not registered
        return cls._REGISTRY.get(type_name, AxesOnlyKeyBuilder)

    @classmethod
    def specialize(cls, schema: Schema) -> DispatchKeyBuilder:
        builder_cls = cls.for_type(schema.op_type)
        return builder_cls(schema)
