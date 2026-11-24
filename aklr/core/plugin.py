# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""A host library's plugin: the targets AKLR serves in it, each declared with the contract describing it."""

from __future__ import annotations

import functools
import importlib
import importlib.metadata
import inspect
import json
import logging
import pkgutil
import re
import threading
import traceback
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, NamedTuple

import torch

from aklr.core.contract import locate
from aklr.core.contract.op import OpContract
from aklr.core.platform.env import RuntimeEnv
from aklr.core.runtime.dispatch import DispatcherRegistry, FallbackReason, KernelDispatcher, trace

__all__ = ["Plugin"]

_LOGGER = logging.getLogger(__name__)
# a dotted full name in sglang and vllm, an op's name in aiter
_TARGET_PATTERN = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*")
# sglang has its device when this returns, and loads the model after it
_SGLANG_DEVICE_READY = "sglang.srt.model_executor.model_runner.ModelRunner.init_torch_distributed"
# every vLLM executor binds a worker's device through it, whatever the worker class
_VLLM_WORKER_WRAPPER = "vllm.v1.worker.worker_base.WorkerWrapperBase"


class _Statement(NamedTuple):
    """One ``declare`` statement, as it was made."""

    target: str
    contract: OpContract
    path: str | None
    versions: tuple[str, ...] | tuple[int, ...]


class _Declaration(NamedTuple):
    """One ``declare`` statement of a family."""

    contract: OpContract
    versions: tuple[str, ...] | tuple[int, ...]
    path: Path  # the contract.py holding the statement: its directory is the family
    schemas: Path | None  # the directory the statement says the contract's schema files are under


class Plugin:
    """One host library's registrations: the targets AKLR serves in it and the contracts describing them."""

    def __init__(self, host: str) -> None:
        """``host`` names the host library and its directory ``aklr/plugins/<host>/``."""
        self.host = host
        self._declarations: dict[str, list[_Declaration]] = {}
        self._statements: list[_Statement] = []  # as made, before the collection sorts them into families
        self._reached: set[str] = set()  # the intercepted targets a call has taken to dispatch
        self._lock = threading.RLock()
        self._is_collected = False
        self._is_registered = False
        self._has_dispatch_failed = False

    def declare(
        self,
        target: str,
        contract: OpContract,
        path: str | None = None,
        *,
        versions: tuple[str, ...] | tuple[int, ...],
    ) -> None:
        """``target``, a host function's dotted full name (an op's name in aiter), is described by
        ``contract`` on the host versions ``versions``, and the contract's schema JSON, its kernels, are
        under ``path``, relative to the family's directory; ``None``: it has none yet. A statement at
        the bottom of a family's ``contract.py``, the one whose ``CONTRACTS`` hold ``contract``: it is
        heard whoever imports that file."""
        what = f"declare({target!r}, {contract.contract_id})"
        if not _TARGET_PATTERN.fullmatch(target):
            raise ValueError(f"{what}: the target is not a name")
        if not versions or {type(v) for v in versions} not in ({str}, {int}):
            raise ValueError(f"{what}: versions lists host versions, all strings or all integers")
        if any(
            s.target == target and s.contract.contract_id == contract.contract_id for s in self._statements
        ):
            raise ValueError(f"{what}: already declared")
        self._statements.append(_Statement(target, contract, path, tuple(versions)))

    def register(self) -> None:
        """The host's entry point calls it: collect the declarations, check them, intercept each target."""
        try:
            env = RuntimeEnv()
        except ValueError as e:
            _LOGGER.error("%s; no target of %s is intercepted", e, self.host)
            return
        if env.is_all_disabled or not getattr(env, f"is_{self.host}_plugin_enabled"):
            return
        if self.host == "aiter":
            _LOGGER.error(
                "aiter asks AKLR's kernel provider (aklr.plugins.aiter.provider): nothing registers here"
            )
            return
        with self._lock:
            if self._is_registered:
                return
            self._is_registered = True
        try:
            version = importlib.metadata.version(self.host).partition("+")[0]
            interceptors, undeclared = {}, []
            for target in self.targets():
                dispatchers = self.dispatchers(target, version)
                if not dispatchers:
                    undeclared.append(target)
                    continue
                try:
                    fn = inspect.unwrap(pkgutil.resolve_name(target))
                    interceptors[target] = self._interceptor(target, fn, dispatchers)
                except Exception as e:  # noqa: BLE001 -- a declaration's error never fails the host
                    _LOGGER.error(
                        "%s: declare(%r, ...): %s: %s; the target runs %s's own code",
                        self._declarations[target][0].path,
                        target,
                        type(e).__name__,
                        e,
                        self.host,
                    )
            if self.host == "sglang":
                from sglang.srt.plugins.hook_registry import HookRegistry, HookType

                HookRegistry.register(
                    _SGLANG_DEVICE_READY,
                    lambda result, *args, **kwargs: self._start_dispatch(),
                    HookType.AFTER,
                )
                for target, intercept in interceptors.items():
                    HookRegistry.register(target, intercept, HookType.AROUND)
            else:
                self._replace_vllm_targets(interceptors)
            _LOGGER.info(
                "AKLR intercepted %s; not declared for %s %s: %s",
                sorted(interceptors),
                self.host,
                version,
                sorted(undeclared),
            )
        except Exception as e:  # noqa: BLE001 -- AKLR's failure never fails the host
            _LOGGER.error(
                "AKLR did not register in %s, which runs its own code: %s: %s", self.host, type(e).__name__, e
            )

    def targets(self) -> Mapping[str, tuple[str, ...] | tuple[int, ...]]:
        """Every declared target and the host versions it is declared for."""
        self._collect()
        return {
            target: tuple(dict.fromkeys(v for d in declarations for v in d.versions))
            for target, declarations in self._declarations.items()
        }

    def dispatchers(self, target: str, version: str | int) -> tuple[KernelDispatcher, ...]:
        """The dispatchers of the contracts declared for ``target`` at host version ``version``, in
        declaration order; empty when no declaration lists ``version``."""
        self._collect()
        registry = DispatcherRegistry.default()
        return tuple(
            registry.dispatcher(d.contract) for d in self._declarations[target] if version in d.versions
        )

    def _collect(self) -> None:
        """Import the ``contract.py`` of every family under ``aklr/plugins/<host>/``, once, so that its
        statements are made; sort the statements into their families and check each family's."""
        with self._lock:
            if self._is_collected:
                return
            self._is_collected = True
            package = importlib.import_module(f"aklr.plugins.{self.host}")
            families: list[tuple[Path, tuple[OpContract, ...]]] = []
            for root in map(Path, package.__path__):
                for file in sorted(root.rglob("contract.py")):
                    name = ".".join((package.__name__, *file.parent.relative_to(root).parts, "contract"))
                    heard = len(self._statements)
                    try:
                        families.append((file, importlib.import_module(name).CONTRACTS))
                    except Exception as e:  # noqa: BLE001 -- a family's error never fails the host
                        del self._statements[heard:]  # what the file stated before it failed
                        lines = [
                            f.lineno for f in traceback.extract_tb(e.__traceback__) if f.filename == str(file)
                        ]
                        where = f"{file}:{lines[-1]}" if lines else file
                        self._drop(file, f"{where}: {type(e).__name__}: {e}")
            wrong: dict[Path, str] = {}  # each family with a wrong statement, and the first one
            for target, contract, path, versions in self._statements:
                what = f"declare({target!r}, {contract.contract_id})"
                # the registry may have imported the file first: the contract is the object it made
                file = next((f for f, contracts in families if any(contract is c for c in contracts)), None)
                if file is None:
                    _LOGGER.error(
                        "%s: the contract is in no family's CONTRACTS under aklr/plugins/%s/; the "
                        "statement is dropped",
                        what,
                        self.host,
                    )
                    continue
                declared = self._declarations.get(target, [])
                schemas = None if path is None else (file.parent / path).resolve()
                if declared and declared[0].path != file:
                    wrong.setdefault(
                        file, f"{what}: the target is declared in another family, {declared[0].path}"
                    )
                elif schemas is not None and not schemas.is_dir():
                    wrong.setdefault(file, f"{what}: {schemas} is not a directory")
                else:
                    self._declarations.setdefault(target, []).append(
                        _Declaration(contract, versions, file, schemas)
                    )
            for file, reason in wrong.items():
                self._drop(file, f"{file}: {reason}")
            for file in sorted({declarations[0].path for declarations in self._declarations.values()}):
                declared = [
                    d for declarations in self._declarations.values() for d in declarations if d.path == file
                ]
                for schema in locate.committed_schema_paths(file.parent):
                    try:
                        contract_id = json.loads(schema.read_bytes())["contract_id"]
                    except (OSError, ValueError, KeyError, TypeError) as e:
                        self._drop(file, f"{schema}: not a schema file: {e!r}")
                        break
                    if not any(
                        d.contract.contract_id == contract_id
                        and d.schemas is not None
                        and schema.resolve().is_relative_to(d.schemas)
                        for d in declared
                    ):
                        self._drop(file, f"{schema}: no declaration of {contract_id!r} names its directory")
                        break

    def _drop(self, file: Path, reason: str) -> None:
        """Forget the declarations of the family whose contract.py is ``file``, for ``reason``."""
        for target in [t for t, declarations in self._declarations.items() if declarations[0].path == file]:
            del self._declarations[target]
        _LOGGER.error(
            "%s; the declarations of %s are dropped, their targets run %s's own code", reason, file, self.host
        )

    def _interceptor(self, target: str, fn: Callable, dispatchers: tuple[KernelDispatcher, ...]) -> Callable:  # structure: allow S4 a closure per target, made in register's loop  # fmt: skip
        """What stands in for ``target``, the host function ``fn``, called as
        ``intercept(original, *args, **kwargs)``: the first of ``dispatchers`` serving the call returns
        its kernel's result, else ``original`` runs. ``TypeError`` unless ``fn``'s params are each
        dispatcher's contract's."""
        dispatch_fns = [(d.contract.contract_id, d.make_api_fn(fn)) for d in dispatchers]
        returns_none = inspect.signature(fn).return_annotation in (None, "None")
        tracer, host, reached = trace(), self.host, self._reached

        def intercept(original: Callable, *args: Any, **kwargs: Any) -> Any:
            if torch.compiler.is_compiling():  # a trace runs no kernel
                return original(*args, **kwargs)
            reached.add(target)
            for contract_id, dispatch_fn in dispatch_fns:
                result = dispatch_fn(*args, **kwargs)
                is_served = not isinstance(result, FallbackReason)
                if tracer is not None and tracer.is_plugin:
                    tracer.plugin(host, target, contract_id, None if is_served else result)
                if is_served:
                    return None if returns_none else result
            return original(*args, **kwargs)

        return intercept

    def _replace_vllm_targets(self, interceptors: Mapping[str, Callable]) -> None:  # structure: allow S4 register is long, and this is vLLM's lack of a hook mechanism  # fmt: skip
        """Put each interceptor where its target's module holds the target, start dispatch once a worker
        has its device, and warn after warm-up of the targets no call took to dispatch."""
        wrapper_class = pkgutil.resolve_name(_VLLM_WORKER_WRAPPER)
        init_device = wrapper_class.init_device

        @functools.wraps(init_device)
        def init_device_then_start(wrapper: Any, *args: Any, **kwargs: Any) -> Any:
            result = init_device(wrapper, *args, **kwargs)
            self._start_dispatch()
            try:
                warm_up = wrapper.worker.compile_or_warm_up_model

                @functools.wraps(warm_up)
                def warm_up_then_report(*args: Any, **kwargs: Any) -> Any:
                    times = warm_up(*args, **kwargs)
                    unreached = sorted(interceptors.keys() - self._reached)
                    if unreached:
                        _LOGGER.warning(
                            "no call reached AKLR for %s by the end of warm-up: vLLM holds another "
                            "reference to them, or calls them only from compiled code, so AKLR never serves them",
                            unreached,
                        )
                    return times

                wrapper.worker.compile_or_warm_up_model = warm_up_then_report
            except AttributeError as e:
                _LOGGER.error("AKLR cannot tell which targets vLLM's warm-up reached: %s", e)
            return result

        wrapper_class.init_device = init_device_then_start
        for target, intercept in interceptors.items():
            owner_name, _, attr = target.rpartition(".")
            owner = pkgutil.resolve_name(owner_name)
            original = getattr(owner, attr)
            setattr(owner, attr, functools.wraps(original)(functools.partial(intercept, original)))

    def _start_dispatch(self) -> None:
        """Load AKLR's kernels on the device the host has just bound; a failure is not tried again."""
        if self._has_dispatch_failed:
            return
        try:
            DispatcherRegistry.default().init_once()
        except Exception as e:  # noqa: BLE001 -- AKLR's failure never fails the host
            self._has_dispatch_failed = True
            _LOGGER.error("AKLR's dispatch did not start, every target runs %s's own code: %s", self.host, e)
