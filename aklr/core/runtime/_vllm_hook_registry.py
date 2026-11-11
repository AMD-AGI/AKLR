# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Private vLLM startup hook engine with saved callables and reversible attribute writes.

This module has no framework dependencies. Callers own compatibility checks and
computation/fallback policy; the registry only installs synchronous Python hooks.
Registration and application must happen before the affected code starts running.
vLLM extensions declare routes through ``aklr.plugins.vllm.hook.vllm_route``;
this module is an implementation detail, not a separate registration API.
"""

from __future__ import annotations

import functools
import inspect
import logging
import os
import pkgutil
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from types import ModuleType
from typing import Any

__all__: list[str] = []

_LOGGER = logging.getLogger(__name__)
KEEP_RESULT = object()


class HookType(Enum):
    BEFORE = "before"
    AFTER = "after"
    AROUND = "around"
    REPLACE = "replace"


@dataclass(frozen=True)
class HookSpec:
    hook_id: str
    target: str
    hook: Callable
    hook_type: HookType = HookType.AROUND
    phase: str = "device_ready"
    group: str | None = None
    order: int = 0
    source: str = ""
    aliases: tuple[str, ...] = ()
    exclusive: bool = False
    validate: Callable[[Callable], None] | None = None


@dataclass(frozen=True)
class _Binding:
    owner: Any
    attr: str
    before: Any
    had_own: bool
    after: Any
    path: str

    @property
    def key(self) -> tuple[int, str]:
        return id(self.owner), self.attr

    def matches(self, value: Any, had_own: bool) -> bool:
        try:
            return (self.attr in vars(self.owner)) == had_own and inspect.getattr_static(
                self.owner, self.attr
            ) is value
        except AttributeError:
            return False


@dataclass(frozen=True)
class _Installation:
    target: str
    specs: tuple[HookSpec, ...]
    original: Callable
    bindings: tuple[_Binding, ...]


def _resolve(path: str) -> tuple[Any, str, Any]:
    owner_path, sep, attr = path.rpartition(".")
    if not sep or not owner_path or not attr:
        raise ValueError(f"Expected a qualified attribute path: {path!r}")
    # Some AITER compatibility imports replace their own sys.modules entry.
    pkgutil.resolve_name(owner_path)
    owner = pkgutil.resolve_name(owner_path)
    try:
        vars(owner)
    except TypeError:
        raise TypeError(f"Hook owner needs writable attributes: {path}") from None
    if isinstance(owner, type) and not any(attr in vars(cls) for cls in owner.__mro__):
        # A metaclass method binds the class, but is absent on that class's
        # instances. Installing it on the class would change both lookup rules.
        if any(attr in vars(meta) for meta in type(owner).__mro__):
            raise TypeError(f"Metaclass-only hook targets are not supported: {path}")
    return owner, attr, inspect.getattr_static(owner, attr)


def _callable(raw: Any, path: str) -> tuple[Callable, type | None]:
    descriptor = type(raw) if isinstance(raw, (classmethod, staticmethod)) else None
    fn = raw.__func__ if descriptor else raw
    if (
        not callable(fn)
        or isinstance(fn, type)
        or inspect.iscoroutinefunction(fn)
        or inspect.isgeneratorfunction(fn)
        or inspect.isasyncgenfunction(fn)
        or (
            hasattr(fn, "__get__")
            and not inspect.isfunction(fn)
            and not inspect.ismethod(fn)
            and not inspect.isbuiltin(fn)
        )
    ):
        raise TypeError(f"Unsupported hook target (expected a synchronous function/method): {path}")
    return fn, descriptor


def _wrap(inner: Callable, spec: HookSpec) -> Callable:
    hook = spec.hook
    if spec.hook_type is HookType.AROUND:

        @functools.wraps(inner)
        def wrapped(*args, **kwargs):
            return hook(inner, *args, **kwargs)

    elif spec.hook_type is HookType.REPLACE:

        @functools.wraps(inner)
        def wrapped(*args, **kwargs):
            return hook(*args, **kwargs)

    elif spec.hook_type is HookType.BEFORE:

        @functools.wraps(inner)
        def wrapped(*args, **kwargs):
            changed = hook(*args, **kwargs)
            if changed is not None:
                args, kwargs = changed
            return inner(*args, **kwargs)

    else:

        @functools.wraps(inner)
        def wrapped(*args, **kwargs):
            result = inner(*args, **kwargs)
            changed = hook(result, *args, **kwargs)
            return result if changed is KEEP_RESULT else changed

    return wrapped


def _lookup_depth(owner: Any) -> int:
    # Write inherited targets before their ancestors so every pre-write check still
    # observes the original lookup. Reversing this order also restores inheritance.
    if isinstance(owner, type):
        return len(owner.__mro__)
    return len(type(owner).__mro__) + 1


class HookRegistry:
    """One process's startup hook declarations and installed attribute bindings.

    A phase is committed atomically and then frozen. Hooks of a target are ordered
    by (order, hook_id), from inner to outer, with a sole REPLACE at the center.
    An exclusive hook permits no other hook on its target. Each target wraps the
    callable resolved before this batch, including inherited class/instance targets.
    ``restore`` is for startup failure/test cleanup, never live graph hot-swapping.
    """

    def __init__(self) -> None:
        self._specs: dict[str, HookSpec] = {}
        self._applied: dict[str, tuple[_Installation, ...]] = {}
        self._originals: dict[str, Callable] = {}

    @classmethod
    @functools.cache
    def default(cls) -> HookRegistry:
        return cls()

    def register(self, spec: HookSpec) -> None:
        if not spec.hook_id or not spec.phase or not isinstance(spec.hook_type, HookType):
            raise ValueError("A hook needs a stable ID, phase and HookType")
        _callable(spec.hook, spec.hook_id)
        if spec.validate is not None:
            _callable(spec.validate, f"{spec.hook_id}.validate")
        previous = self._specs.get(spec.hook_id)
        if previous is not None:
            if previous != spec:
                raise ValueError(f"Conflicting definition for hook {spec.hook_id}")
            return
        if spec.phase in self._applied:
            raise RuntimeError(f"Hook phase {spec.phase!r} is frozen; cannot add {spec.hook_id}")
        self._specs[spec.hook_id] = spec

    def _plan(self, phase: str) -> tuple[_Installation, ...]:
        groups: dict[str, str] = {}
        targets: dict[str, list[HookSpec]] = defaultdict(list)
        for spec in self._specs.values():
            if spec.group:
                previous_phase = groups.setdefault(spec.group, spec.phase)
                if previous_phase != spec.phase:
                    raise ValueError(f"Hook group {spec.group!r} crosses phases")
            if spec.phase == phase:
                targets[spec.target].append(spec)

        occupied = {
            binding.key: binding.path
            for installations in self._applied.values()
            for installation in installations
            for binding in installation.bindings
        }
        plan = []
        for target, specs in sorted(targets.items()):
            if sum(s.hook_type is HookType.REPLACE for s in specs) > 1:
                raise ValueError(f"Multiple REPLACE hooks for {target}")
            if sum(s.exclusive for s in specs) > 1:
                raise ValueError(f"Multiple exclusive hooks for {target}")
            if len(specs) > 1 and any(s.exclusive for s in specs):
                raise ValueError(f"Exclusive hook cannot be combined with other hooks for {target}")
            ordered = tuple(
                sorted(specs, key=lambda s: (s.hook_type is not HookType.REPLACE, s.order, s.hook_id))
            )
            owner, attr, raw = _resolve(target)
            original, descriptor = _callable(raw, target)
            if not isinstance(owner, type):
                # AITER's tgemm is a singleton instance. Its route receives no self:
                # save the bound method, then install a plain instance attribute.
                # This also covers methods on ModuleType subclasses; normal module
                # functions are unchanged. Keep raw separately for restoring lookup.
                resolved = getattr(owner, attr)
                if not callable(resolved):
                    raise TypeError(f"Hook target does not resolve to a callable: {target}")
                original, _ = _callable(resolved, target)
                descriptor = None
            for spec in ordered:
                if spec.validate is not None:
                    spec.validate(original)
            wrapped = original
            for spec in ordered:
                wrapped = _wrap(wrapped, spec)
            replacement = descriptor(wrapped) if descriptor else wrapped
            if isinstance(owner, type) and descriptor is None and not hasattr(type(raw), "__get__"):
                # Builtins, partials, bound methods and callable objects stored on
                # a class do not bind its instances. A plain function wrapper would.
                replacement = staticmethod(wrapped)
            bindings = [_Binding(owner, attr, raw, attr in vars(owner), replacement, target)]
            aliases = sorted({alias for spec in specs for alias in spec.aliases})
            if descriptor and aliases:
                raise ValueError(f"Descriptor aliases are not supported: {target}")
            for alias in aliases:
                alias_owner, alias_attr, alias_raw = _resolve(alias)
                if not isinstance(alias_owner, ModuleType) or alias_raw is not original:
                    raise ValueError(f"Alias {alias} must be a module binding to the original {target}")
                bindings.append(_Binding(alias_owner, alias_attr, alias_raw, True, wrapped, alias))
            for binding in bindings:
                if binding.key in occupied:
                    raise ValueError(
                        f"Binding {binding.path} conflicts with {occupied[binding.key]} "
                        "(duplicate target/alias or cross-phase write)"
                    )
                occupied[binding.key] = binding.path
            plan.append(_Installation(target, ordered, original, tuple(bindings)))
        return tuple(sorted(plan, key=lambda item: -_lookup_depth(item.bindings[0].owner)))

    @staticmethod
    def _restore_bindings(bindings: list[_Binding]) -> None:
        # Check the entire batch before writing: never overwrite another owner's patch.
        for binding in bindings:
            if not binding.matches(binding.after, True) and not binding.matches(
                binding.before, binding.had_own
            ):
                raise RuntimeError(f"Cannot restore {binding.path}: binding changed outside this registry")
        for binding in reversed(bindings):
            if binding.matches(binding.before, binding.had_own):
                continue
            if binding.had_own:
                setattr(binding.owner, binding.attr, binding.before)
            else:
                delattr(binding.owner, binding.attr)

    def apply(self, phase: str) -> list[str]:
        if phase in self._applied:
            for installation in self._applied[phase]:
                for binding in installation.bindings:
                    if not binding.matches(binding.after, True):
                        raise RuntimeError(f"Installed binding changed outside this registry: {binding.path}")
            return [installation.target for installation in self._applied[phase]]
        # Resolution, descriptors, aliases and conflicts are checked before any writes.
        try:
            plan = self._plan(phase)
        except Exception as exc:
            raise RuntimeError(f"Cannot prepare hook phase {phase!r}: {exc}") from exc
        written: list[_Binding] = []
        try:
            for installation in plan:
                for binding in installation.bindings:
                    if not binding.matches(binding.before, binding.had_own):
                        raise RuntimeError(f"Binding changed during hook preparation: {binding.path}")
                    # Include an assignment that mutates then raises in the rollback.
                    written.append(binding)
                    setattr(binding.owner, binding.attr, binding.after)
                    if not binding.matches(binding.after, True):
                        raise RuntimeError(f"Hook assignment did not take effect: {binding.path}")
        except Exception as exc:
            try:
                self._restore_bindings(written)
            except Exception as rollback_exc:
                raise RuntimeError(f"Hook phase {phase!r} failed; rollback failed: {rollback_exc}") from exc
            raise RuntimeError(f"Hook phase {phase!r} failed and was rolled back: {exc}") from exc
        self._applied[phase] = plan
        for installation in plan:
            for spec in installation.specs:
                self._originals[spec.hook_id] = installation.original
            _LOGGER.info(
                "Applied hooks %s to %s (phase=%s, aliases=%d)",
                ", ".join(s.hook_id for s in installation.specs),
                installation.target,
                phase,
                len(installation.bindings) - 1,
            )
        return [installation.target for installation in plan]

    def original(self, hook_id: str) -> Callable:
        try:
            return self._originals[hook_id]
        except KeyError:
            raise RuntimeError(f"Hook {hook_id!r} has not been applied") from None

    def restore(self, phase: str | None = None) -> None:
        phases = list(self._applied) if phase is None else [phase]
        installations = [item for name in phases for item in self._applied.get(name, ())]
        self._restore_bindings([binding for item in installations for binding in item.bindings])
        for installation in installations:
            for spec in installation.specs:
                self._originals.pop(spec.hook_id, None)
        for name in phases:
            self._applied.pop(name, None)

    def report(self) -> tuple[dict[str, Any], ...]:
        """Serializable declarations and installation state; not kernel hit statistics."""
        return tuple(
            {
                "hook_id": spec.hook_id,
                "target": spec.target,
                "type": spec.hook_type.value,
                "phase": spec.phase,
                "group": spec.group,
                "order": spec.order,
                "source": spec.source,
                "aliases": list(spec.aliases),
                "exclusive": spec.exclusive,
                "state": "installed" if spec.hook_id in self._originals else "registered",
                "pid": os.getpid(),
            }
            for spec in sorted(self._specs.values(), key=lambda s: (s.phase, s.target, s.order, s.hook_id))
        )
