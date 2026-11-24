# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""AKLR's kernel provider in aiter: it answers aiter's routed ops from the declarations on ``Plugin("aiter")``."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable, Mapping
from typing import Any

import torch

from aklr.core.platform.env import RuntimeEnv
from aklr.core.runtime.dispatch import DispatcherRegistry, FallbackReason, KernelDispatcher, trace

__all__ = ["Provider"]

_LOGGER = logging.getLogger(__name__)


class Provider:
    """AKLR as an AITER kernel provider: it answers AITER from the declarations on Plugin("aiter")."""

    name = "aklr"
    api_versions = (1,)  # the aiter.plugin.API_VERSION this is written against
    op_versions: Mapping[str, tuple[int, ...]]  # AITER op name -> the op versions declared

    def __init__(self) -> None:
        """AITER instantiates the entry point's class when it discovers providers."""
        from aiter import plugin as aiter_plugin

        from aklr.plugins.aiter import plugin

        self.op_versions = dict(plugin.targets())  # a target in aiter is an op's name
        self._plugin = plugin
        # aiter's own record of each routed op, read for its signature: an op registered as a torch
        # custom op tells none, and aiter has no public way to ask
        self._routed_ops = aiter_plugin._routed_ops
        self._candidate = aiter_plugin.KernelCandidate
        # op name -> per contract declared for the running op version, in declaration order: the
        # dispatcher, the contract's call of aiter's arguments, and its candidates' run. Made when aiter
        # first asks for the op: aiter may import an op's module after it has discovered the provider.
        self._bindings: dict[str, list[tuple[KernelDispatcher, Callable, Callable]]] = {}
        self._has_dispatch_failed = False
        self._failed_ops: set[str] = set()  # those whose select has logged its failure
        try:
            self._tracer = trace()
        except ValueError as e:
            self._tracer = None
            _LOGGER.error("%s; nothing is traced", e)

    def select(self, op: Any, args: tuple, kwargs: Mapping) -> Any:
        """The kernel serving this call, else None; nothing runs."""
        try:
            if torch.compiler.is_compiling() or self._has_dispatch_failed:
                return None
            registry = DispatcherRegistry.default()
            if not registry.is_initialized:  # nothing else starts it where only aiter's switch is on
                try:
                    registry.init_once()
                except Exception as e:  # noqa: BLE001 -- AKLR's failure never fails aiter
                    self._has_dispatch_failed = True
                    _LOGGER.error("AKLR's dispatch did not start, aiter runs its own kernels: %s", e)
                    return None
            if op.name not in self._bindings:
                self._bindings[op.name] = []  # a failure below is not tried again
                signature = self._routed_ops[op.name].contract
                self._bindings[op.name] = [
                    self._binding(op.name, signature, d)
                    for d in self._plugin.dispatchers(op.name, op.version)
                ]
            for dispatcher, call_of, run in self._bindings[op.name]:
                call = call_of(args, kwargs)
                served = FallbackReason.OUT_OF_CONTRACT if call is None else dispatcher.serving(call)
                is_served = not isinstance(served, FallbackReason)
                if self._tracer is not None and self._tracer.is_plugin:
                    contract_id = dispatcher.contract.contract_id
                    self._tracer.plugin("aiter", op.name, contract_id, None if is_served else served)
                if is_served:
                    return self._candidate(
                        provider=self.name,
                        kernel_id=served.kernel_ref,
                        artifact_digest=served.content_hash,
                        run=run,
                    )
        except Exception as e:  # noqa: BLE001 -- AKLR's failure never fails aiter
            if op.name not in self._failed_ops:
                self._failed_ops.add(op.name)
                _LOGGER.error(
                    "AKLR cannot answer for %s, aiter runs its own kernel: %s: %s",
                    op.name,
                    type(e).__name__,
                    e,
                )
        return None

    @staticmethod
    def enable() -> None:
        """SGLang's and vLLM's entry points call it: enable AKLR in AITER when AKLR_AITER_PLUGIN=1."""
        try:
            env = RuntimeEnv()
            if env.is_all_disabled or not env.is_aiter_plugin_enabled:
                return
            from aiter import plugin as aiter_plugin

            aiter_plugin.configure(source="aklr: AKLR_AITER_PLUGIN=1", enable=[Provider.name])
        except Exception as e:  # noqa: BLE001 -- AKLR's failure never fails the host
            _LOGGER.error(
                "AKLR is not enabled in aiter, which runs its own kernels: %s: %s", type(e).__name__, e
            )

    @staticmethod
    def _binding(  # structure: allow S4 the closures of one op and contract, made per contract in select
        op_name: str, signature: inspect.Signature, dispatcher: KernelDispatcher
    ) -> tuple[KernelDispatcher, Callable, Callable]:
        """How ``dispatcher``'s contract serves the op of ``signature``: the dispatcher, the contract's
        call of the arguments aiter passes (``None`` where a param the contract lacks is off its default),
        and the one ``run`` of its candidates. ``TypeError`` unless the op's params are the contract's."""
        contract = dispatcher.contract

        def op_signature() -> None: ...  # check_api_signature reads a function's: aiter holds a Signature

        op_signature.__signature__, op_signature.__name__ = signature, op_name
        contract.check_api_signature(op_signature)
        params = signature.parameters
        positional = [n for n, p in params.items() if p.kind is not p.KEYWORD_ONLY]
        api_names = [p.name for p in contract.api_params]
        op_only = {n: p.default for n, p in params.items() if n not in api_names}
        contract_only = {p.name: p.default for p in contract.api_params if p.name not in params}
        returns_none = signature.return_annotation in (None, "None")

        def call_of(args: tuple, kwargs: Mapping) -> dict[str, Any] | None:
            values = {**dict(zip(positional, args, strict=True)), **kwargs}
            for name, default in op_only.items():
                if values[name] is not default and (default is None or values[name] != default):
                    return None
            return {n: values[n] if n in values else contract_only[n] for n in api_names}

        def run(*args: Any, **kwargs: Any) -> Any:
            call = call_of(args, kwargs)
            result = FallbackReason.OUT_OF_CONTRACT if call is None else dispatcher.dispatch(call)
            if isinstance(result, FallbackReason):  # select has just asked the same table: a bug
                raise RuntimeError(
                    f"{contract.contract_id} was offered for {op_name} and does not serve it: {result.value}"
                )
            return None if returns_none else result

        return dispatcher, call_of, run
