# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import atexit
import functools
import logging
from collections import Counter
from collections.abc import Callable, Mapping
from types import MappingProxyType

import torch

from aklr.core.contract.op import OpContract
from aklr.core.contract.registry import ContractRegistry
from aklr.core.contract.spec import CONTRACT_ID_PATTERN, ContractId
from aklr.core.platform.device import DeviceTarget
from aklr.core.platform.env import ENV_DISABLE, NAMESPACES, RuntimeEnv
from aklr.core.platform.toolchain import unmet_requirement
from aklr.core.runtime import LoadError
from aklr.core.runtime.builder import BuilderRegistry, BuildError
from aklr.core.runtime.dispatch._dispatcher import (
    DispatchTable,
    FallbackReason,
    KernelDispatcher,
    kernel_failure_hint,
)
from aklr.core.runtime.dispatch._recorder import CallRecorder

__all__ = ["DispatcherRegistry", "api"]

_LOGGER = logging.getLogger(__name__)


class DispatcherRegistry:
    """The dispatchers of this process by contract id, the recorder they count into while exporting."""

    __slots__ = ("_dispatchers", "is_initialized", "recorder")
    _dispatchers: dict[ContractId, KernelDispatcher]
    recorder: CallRecorder | None  # set by init; None while export is off
    is_initialized: bool

    def __init__(self) -> None:
        self._dispatchers = {}
        self.recorder = None
        self.is_initialized = False

    @classmethod
    @functools.cache
    def default(cls) -> DispatcherRegistry:
        """The process registry, made on first use and never replaced."""
        return cls()

    def dispatcher(self, contract: OpContract) -> KernelDispatcher:
        """The registered dispatcher of ``contract``, made and registered on first use."""
        contract_id = contract.contract_id
        dispatcher = self._dispatchers.get(contract_id)
        if dispatcher is None:
            dispatcher = self._dispatchers[contract_id] = KernelDispatcher(contract)
        elif dispatcher.contract != contract:
            raise ValueError(f"{contract_id}: another contract already has this id")
        return dispatcher

    def dispatchers(self) -> Mapping[ContractId, KernelDispatcher]:
        """The registered dispatchers by contract id, read-only (a ``MappingProxyType`` over the live dict)."""
        return MappingProxyType(self._dispatchers)

    def init(self) -> None:
        """Set every dispatcher from the committed schemas (``ContractRegistry.default().read_schemas``)."""
        try:
            runtime_env = RuntimeEnv()
        except ValueError as e:
            raise LoadError(str(e)) from e
        malformed_ids = sorted(
            contract_id
            for contract_id in runtime_env.disabled_contracts
            if not CONTRACT_ID_PATTERN.fullmatch(contract_id)
        )
        if malformed_ids:  # every other AKLR_DISABLE item is a contract id
            raise LoadError(
                f"{ENV_DISABLE}: expected '1', '0', contract ids (name@major.minor), "
                f"namespaces {list(NAMESPACES)} or backend names, got {malformed_ids}"
            )
        for contract in ContractRegistry.default().contracts():
            self.dispatcher(contract)
        registered_contract_ids = frozenset(self._dispatchers)
        disabled_contract_ids = (
            registered_contract_ids
            if runtime_env.is_all_disabled
            else (runtime_env.disabled_contracts & registered_contract_ids)
            | {
                contract_id
                for contract_id, dispatcher in self._dispatchers.items()
                if runtime_env.is_package_disabled(dispatcher.contract.op_package)
            }
        )
        live_contracts = {
            contract_id: dispatcher.contract
            for contract_id, dispatcher in self._dispatchers.items()
            if contract_id not in disabled_contract_ids
        }
        contract_registry = ContractRegistry.default()
        schemas, errors = [], []
        for op_package in sorted({c.op_package for c in live_contracts.values()}):
            op_schemas, op_errors = contract_registry.read_schemas(
                op_package,
                skipped_contract_ids=disabled_contract_ids,
                measured_dir=runtime_env.measured_dir,
            )
            # a definition (no kernel sources) declares a scenario; it serves nothing
            schemas += [(p, s) for p, s in op_schemas if s.kernel.sources]
            errors += op_errors
        if errors:
            raise LoadError("\n".join(errors))
        from aklr.core.runtime import seam  # seam imports this module

        # nothing to beat without a defined baseline (the bench's rule): speedup 1.0
        # qualifies; an op's baseline.py that fails to import fails the init
        min_speedups = {}
        for contract_id, contract in live_contracts.items():
            try:
                has_baseline = seam.baseline_of(contract) is not None
            except (ImportError, seam.RouteError) as e:
                raise LoadError(f"{contract_id}: its baseline: {e}") from e
            min_speedups[contract_id] = runtime_env.min_speedup_of(contract.op_package) if has_baseline else 0
        device = target = None
        if schemas or (runtime_env.is_export_enabled and live_contracts):
            device, target = self._probe_device()
        schema_runnables = {contract_id: [] for contract_id in live_contracts}
        n_skipped = 0
        n_unmet: Counter[str] = Counter()
        for path, schema in schemas:
            if not DeviceTarget.matches_target(target, schema.kernel.targets):
                n_skipped += 1
                continue
            if (unmet := unmet_requirement(schema.kernel.requires)) is not None:
                n_unmet[unmet] += 1
                continue
            try:
                runnable = BuilderRegistry.get_instance().build(schema)
            except BuildError as e:
                hint = kernel_failure_hint(schema.contract_id, schema.kernel.language)
                raise BuildError(f"{path}: {e} ({hint})") from e.__cause__
            schema_runnables[schema.contract_id].append((schema, runnable, schema.workloads))
        tables = {
            contract_id: DispatchTable.from_schemas(
                contract,
                device,
                schema_runnables[contract_id],
                min_speedup=min_speedups[contract_id],
            )
            for contract_id, contract in live_contracts.items()
            # an empty table while exporting: it records the calls no kernel serves yet
            if schema_runnables[contract_id] or runtime_env.is_export_enabled
        }
        recorder = None
        if runtime_env.is_export_enabled:
            recorder = CallRecorder(
                runtime_env.export_dir,
                target,
                previous_recorder=self.recorder,
            )
            try:
                recorder.save()
            except OSError as e:
                raise LoadError(f"cannot write call records: {e}") from e
        # commit: nothing below raises
        for contract_id, dispatcher in self._dispatchers.items():
            if contract_id in tables:
                table = tables[contract_id]
                if table.runnables or recorder is not None:
                    dispatcher.set_table(table, recorder)
                else:
                    dispatcher.close(table.closed_reason)
            elif contract_id in disabled_contract_ids:
                dispatcher.close(FallbackReason.DISABLED)
            else:
                dispatcher.close(FallbackReason.NO_SCHEMA)
        self.recorder, self.is_initialized = recorder, True
        if recorder is not None:
            atexit.unregister(self._save_calls)  # registered once per process
            atexit.register(self._save_calls)
        for unmet, n in n_unmet.items():
            _LOGGER.info("skipped %d schemas: %s", n, unmet)
        # every schema skipped: a wrong target or a hardware id normalized otherwise
        level = logging.WARNING if schemas and n_skipped == len(schemas) else logging.INFO
        _LOGGER.log(
            level,
            "loaded %d schemas, %d points; skipped %d schemas of other targets; device target %s",
            len(schemas) - n_skipped - n_unmet.total(),
            sum(len(table.runnables) for table in tables.values()),
            n_skipped,
            "not probed" if target is None else target,
        )
        for contract_id, contract in sorted(live_contracts.items()):
            if contract.op_package.startswith("aklr.plugins."):
                table = tables.get(contract_id)
                n = 0 if table is None else len(table.runnables)
                why = "" if n else f" ({FallbackReason.NO_SCHEMA.value})"
                _LOGGER.info("plugin op %s: %d points served%s", contract_id, n, why)

    def init_once(self) -> None:
        """``init`` unless ``is_initialized``: for hosts that may start twice (the sglang plugin)."""
        if not self.is_initialized:
            self.init()

    @staticmethod
    def _probe_device() -> tuple[torch.device, DeviceTarget]:
        hint = f"schemas and call records need the device target; {ENV_DISABLE}=1 starts without AKLR"
        if not torch.cuda.is_available():
            raise LoadError(f"no GPU visible: {hint}")
        device = torch.device("cuda", torch.cuda.current_device())
        target = DeviceTarget.current(device)
        if not target.is_complete:
            raise LoadError(f"device {device.index}: unknown arch or hardware ({target}): {hint}")
        return device, target

    def _save_calls(self) -> None:
        if self.recorder is not None:
            self.recorder.save_or_warn()


def api(contract: OpContract) -> Callable[[Callable], Callable]:
    """The decorator of an op's api function."""
    return DispatcherRegistry.default().dispatcher(contract).make_api_fn
