# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from threading import RLock
from typing import TYPE_CHECKING, Concatenate, Self, overload

from packaging.version import Version

from aklr.core.contract.op import OpContract
from aklr.core.contract.registry import ContractRegistry

if TYPE_CHECKING:
    from aiter.plugin import KernelCandidate

_Selector = Callable[[tuple[object, ...], Mapping[str, object]], "KernelCandidate | None"]
_Callback = Callable[..., object]
_MISSING = object()


@dataclass(frozen=True)
class _Wrap:
    target: str
    versions: tuple[str, ...]
    callback: _Callback
    around: bool
    eager: bool


@dataclass(frozen=True)
class _Intercept:
    caller: str
    callees: tuple[str, ...]
    versions: tuple[str, ...]
    replace: _Callback
    group: str | None


def _site(label: str) -> str:
    caller = sys._getframe(2)
    return f"{label} at {caller.f_code.co_filename}:{caller.f_lineno}"


def _label(callback: Callable) -> str:
    return getattr(callback, "__qualname__", type(callback).__qualname__)


class Plugin:
    """Own the declarations for a plugin package or one of its contract families."""

    def __init__(self, name: str) -> None:
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError(f"invalid plugin name: {name!r}")
        self._name = name
        self._contracts: dict[str, OpContract] = {}
        self._selectors: dict[tuple[str, int], _Selector] = {}
        self._claims: dict[object, str] = {}
        self._wraps: list[_Wrap] = []
        self._intercepts: list[_Intercept] = []
        self._lock = RLock()
        self._building = False
        self._fragments: tuple[Callable[[Plugin], None], ...] | None = None

    @property
    def name(self) -> str:
        """Package namespace that owns these declarations."""
        return self._name

    @property
    def contracts(self) -> tuple[OpContract, ...]:
        """Contracts in declaration order, available after the first successful build."""
        with self._lock:
            if self._fragments is None:
                raise RuntimeError("build the plugin before reading its contracts")
            return tuple(self._contracts.values())

    @cached_property
    def _by_api(self) -> ContractRegistry:
        return ContractRegistry(self.contracts)

    def op(self, fn: _Callback) -> _Callback:
        """Bind an operator signature to the matching contract of this frozen family."""
        from aklr.core.runtime.dispatch import DispatcherRegistry

        contract = self._by_api.for_api(fn)
        return DispatcherRegistry.default().dispatcher(contract).make_api_fn(fn)

    def define(self, contract: OpContract) -> None:
        """Publish an existing operator contract in this namespace."""
        with self._lock:
            self._editable()
            if not isinstance(contract, OpContract):
                raise TypeError("define expects an OpContract")
            if not contract.op_package.startswith(f"aklr.plugins.{self.name}."):
                raise ValueError(f"{contract.contract_id} belongs to {contract.op_package}, not {self.name}")
            self._claim((contract.contract_id,), _site(contract.api_name))
            self._contracts[contract.contract_id] = contract

    def _editable(self, host: str | None = None) -> None:
        if self._fragments is not None:
            raise RuntimeError(f"plugin {self.name!r} is frozen")
        if host is not None and self.name != host:
            raise ValueError(f"expected {host} plugin, got {self.name}")

    def provide(self, op: str, *, version: int, select: _Selector) -> None:
        """Offer an AITER candidate for a versioned op without executing it."""
        with self._lock:
            self._editable("aiter")
            if not isinstance(op, str) or not op.isidentifier() or type(version) is not int or version < 1:
                raise ValueError(f"invalid AITER op or ABI version: {op!r}, {version!r}")
            if not callable(select):
                raise TypeError("select must be callable")
            key = (op, version)
            self._claim((key,), _site(_label(select)))
            self._selectors[key] = select

    @staticmethod
    def _host_keys(targets: tuple[str, ...], versions: tuple[str, ...], callback: _Callback):
        if not callable(callback):
            raise TypeError("callback must be callable")
        if (
            not isinstance(versions, tuple)
            or not versions
            or any(not isinstance(v, str) or Version(v).public != v for v in versions)
        ):
            raise ValueError(f"invalid host versions: {versions!r}")
        if any(not isinstance(t, str) or not all(p.isidentifier() for p in t.split(".")) for t in targets):
            raise ValueError(f"invalid target paths: {targets!r}")
        keys = [(version, target) for version in versions for target in targets]
        if len(set(keys)) != len(keys):
            raise ValueError(f"duplicate version or target in {keys}")
        return keys

    def _claim(self, keys: Sequence[object], source: str) -> None:
        for key in keys:
            if key in self._claims:
                raise ValueError(f"{key}: {self._claims[key]} conflicts with {source}")
        self._claims.update(dict.fromkeys(keys, source))

    @overload
    def wrap(
        self, target: str, *, versions: tuple[str, ...], replace: _Callback, eager: bool = False
    ) -> None: ...

    @overload
    def wrap(
        self,
        target: str,
        *,
        versions: tuple[str, ...],
        around: Callable[Concatenate[_Callback, ...], object],
        eager: bool = False,
    ) -> None: ...

    def wrap(
        self, target: str, *, versions: tuple[str, ...], replace=_MISSING, around=_MISSING, eager=False
    ) -> None:
        """Replace an SGLang target, or surround its next callable with an around callback."""
        with self._lock:
            self._editable("sglang")
            if (replace is _MISSING) == (around is _MISSING):
                raise TypeError("wrap expects exactly one callback: replace or around")
            callback = replace if around is _MISSING else around
            self._claim(self._host_keys((target,), versions, callback), _site(_label(callback)))
            self._wraps.append(_Wrap(target, versions, callback, around is not _MISSING, eager))

    def intercept(
        self,
        caller: str,
        callee: str,
        *other_callees: str,
        versions: tuple[str, ...],
        replace: _Callback,
        group: str | None = None,
    ) -> None:
        """Replace one or more callee bindings looked up by a vLLM caller."""
        with self._lock:
            self._editable("vllm")
            if not isinstance(caller, str) or not all(p.isidentifier() for p in caller.split(".")):
                raise ValueError(f"invalid caller: {caller!r}")
            if group is not None and (not isinstance(group, str) or not group):
                raise ValueError(f"invalid group: {group!r}")
            callees = (callee, *other_callees)
            self._claim(self._host_keys(callees, versions, replace), _site(_label(replace)))
            self._intercepts.append(_Intercept(caller, callees, versions, replace, group))

    def build(self, *declarations: Callable[[Plugin], None]) -> Self:
        """Run the fragments once and freeze. A failure rolls back this build for retry."""
        with self._lock:
            if self._fragments is not None:
                if declarations and not (
                    len(declarations) == len(self._fragments)
                    and all(a is b for a, b in zip(declarations, self._fragments))
                ):
                    raise RuntimeError(f"plugin {self.name!r} is frozen")
                return self
            if self._building:
                raise RuntimeError(f"recursive build of plugin {self.name!r}")
            before = self._contracts.copy(), self._selectors.copy(), self._claims.copy()
            wraps, intercepts = len(self._wraps), len(self._intercepts)
            self._building = True
            try:
                for declare in declarations:
                    declare(self)
                self._fragments = declarations
            except BaseException:
                self._contracts, self._selectors, self._claims = before
                del self._wraps[wraps:]
                del self._intercepts[intercepts:]
                raise
            finally:
                self._building = False
            return self
