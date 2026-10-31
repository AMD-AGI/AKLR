from __future__ import annotations

from collections.abc import Callable
from typing import (
    Any,
    ClassVar,
    Protocol,
    overload,
)

from aklr.core.runtime.tracing._workload_entry import WorkloadEntry


class InputDumpPolicy(Protocol):
    def dump(self, inputs: dict[str, Any]) -> list[str]: ...


class FilterPolicy(Protocol):
    def submit(self, entry: WorkloadEntry) -> None: ...

    def drain(self) -> list[WorkloadEntry]: ...

    def reset(self) -> None: ...


class PolicyRegistry:
    _input_dump_policies: ClassVar[dict[str, type[InputDumpPolicy]]] = {}
    _filter_policies: ClassVar[dict[str, type[FilterPolicy]]] = {}

    @overload
    @classmethod
    def register_input_dump_policy(
        cls, name: str, policy_cls: type[InputDumpPolicy], *, override: bool = False
    ) -> type[InputDumpPolicy]: ...

    @overload
    @classmethod
    def register_input_dump_policy(
        cls, name: str, *, override: bool = False
    ) -> Callable[[type[InputDumpPolicy]], type[InputDumpPolicy]]: ...

    @classmethod
    def register_input_dump_policy(
        cls,
        name: str,
        policy_cls: type[InputDumpPolicy] | None = None,
        *,
        override: bool = False,
    ) -> type[InputDumpPolicy] | Callable[[type[InputDumpPolicy]], type[InputDumpPolicy]]:

        def decorator(cls_to_register: type[InputDumpPolicy]) -> type[InputDumpPolicy]:
            if not override and name in cls._input_dump_policies:
                raise ValueError(f"Input dump policy '{name}' already registered")
            cls._input_dump_policies[name] = cls_to_register
            return cls_to_register

        if policy_cls is not None:
            return decorator(policy_cls)
        return decorator

    @classmethod
    def get_input_dump_policy(cls, name: str) -> type[InputDumpPolicy] | None:
        return cls._input_dump_policies.get(name)

    @classmethod
    def list_input_dump_policies(cls) -> list[str]:
        return list(cls._input_dump_policies.keys())

    @overload
    @classmethod
    def register_filter_policy(
        cls, name: str, policy_cls: type[FilterPolicy], *, override: bool = False
    ) -> type[FilterPolicy]: ...

    @overload
    @classmethod
    def register_filter_policy(
        cls, name: str, *, override: bool = False
    ) -> Callable[[type[FilterPolicy]], type[FilterPolicy]]: ...

    @classmethod
    def register_filter_policy(
        cls,
        name: str,
        policy_cls: type[FilterPolicy] | None = None,
        *,
        override: bool = False,
    ) -> type[FilterPolicy] | Callable[[type[FilterPolicy]], type[FilterPolicy]]:

        def decorator(cls_to_register: type[FilterPolicy]) -> type[FilterPolicy]:
            if not override and name in cls._filter_policies:
                raise ValueError(f"Filter policy '{name}' already registered")
            cls._filter_policies[name] = cls_to_register
            return cls_to_register

        if policy_cls is not None:
            return decorator(policy_cls)
        return decorator

    @classmethod
    def get_filter_policy(cls, name: str) -> type[FilterPolicy] | None:
        return cls._filter_policies.get(name)

    @classmethod
    def list_filter_policies(cls) -> list[str]:
        return list(cls._filter_policies.keys())
