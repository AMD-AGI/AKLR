# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import inspect
from typing import Any, ClassVar, Generic, Self, TypeVar, cast

T = TypeVar("T")


class ClassRegistry(Generic[T]):
    """Named implementation classes; construction and lifetime belong to the caller.

    ``T`` can also be a structural Protocol: registration does not require nominal
    inheritance. Each registry instance provides an independent namespace.
    """

    def __init__(self, namespace: str) -> None:
        self._namespace = namespace
        self._classes: dict[str, type[T]] = {}

    def register(self, name: str, implementation: type[T]) -> type[T]:
        if not isinstance(name, str) or not name or name.strip() != name:
            raise ValueError("a registration name must be a nonempty string without surrounding whitespace")
        if not inspect.isclass(implementation) or inspect.isabstract(implementation):
            raise TypeError(f"{self._namespace} implementations must be concrete classes")
        if name in self._classes:
            raise ValueError(f"{self._namespace} name {name!r} is already registered")
        self._classes[name] = implementation
        return implementation

    def resolve(self, name: str) -> type[T]:
        try:
            return self._classes[name]
        except KeyError:
            available = ", ".join(self.names()) or "(none)"
            raise ValueError(f"unknown {self._namespace}: {name!r}; available: {available}") from None

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._classes))


class Registrable:
    """Each direct subclass owns a registry shared by its descendants.

    Declare implementations with ``class Concrete(Interface, name="choice")``.
    Descendants without an explicit name do not register or overwrite their parent.
    Registration happens when the defining module is imported; no modules are discovered.
    """

    _registry: ClassVar[ClassRegistry[Any]]

    def __init_subclass__(cls, *, name: str | None = None, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        registries = {base._registry for base in cls.__bases__ if hasattr(base, "_registry")}
        if len(registries) > 1:
            raise TypeError("an implementation cannot belong to multiple registry families")
        cls._registry = registries.pop() if registries else ClassRegistry(cls.__name__)
        if name is not None:
            cls._registry.register(name, cls)

    @classmethod
    def resolve(cls, name: str) -> type[Self]:
        implementation = cls._registry.resolve(name)
        if not issubclass(implementation, cls):
            raise ValueError(f"{name!r} is not a {cls.__name__} implementation")
        return cast(type[Self], implementation)
