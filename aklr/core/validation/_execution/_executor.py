###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

import functools
import signal
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from aklr.core.validation._execution._backend import (
    ExecutionBackend,
    RunnerExecutionBackend,
)
from aklr.core.validation._execution._errors import ExecutorClosed
from aklr.core.validation._execution._future import ExecutionFuture
from aklr.core.validation._execution._model import (
    ExecutionTask,
    ExecutorState,
    ResourceRequest,
)
from aklr.core.validation._execution._resources import (
    LocalAcceleratorCatalog,
)
from aklr.core.validation._execution._runner import (
    DEFAULT_STARTUP_TIMEOUT_SECS,
    ExecutionRunner,
)
from aklr.core.validation._execution._scheduler import SchedulingStrategy

if TYPE_CHECKING:
    from concurrent.futures import Future
    from typing import Self


class Executor:
    def __init__(self, backend: ExecutionBackend, *, resources_per_runner: int = 1) -> None:
        self._resource_request = ResourceRequest(count=resources_per_runner)
        self._backend = backend
        self._state = ExecutorState.OPEN
        self._state_lock = threading.Lock()
        self._previous_signal_handlers: dict[int, Any] = {}
        self._backend.shutdown_result.add_done_callback(self._on_backend_closed)

    @classmethod
    def for_devices(
        cls,
        devices: Sequence[str],
        *,
        runner: str = "disposable",
        strategy: str | SchedulingStrategy = "round_robin",
        resources_per_runner: int = 1,
        startup_timeout_secs: float = DEFAULT_STARTUP_TIMEOUT_SECS,
        initializer: Callable[..., None] | None = None,
        initargs: Sequence[Any] = (),
    ) -> Executor:
        """Run tasks on these devices; like multiprocessing.Pool, each process runs initializer(*initargs) first."""
        ResourceRequest(count=resources_per_runner)
        # the catalog first: a bad device raises before the job directory exists
        resources = LocalAcceleratorCatalog(devices)
        if resources_per_runner > len(resources):
            raise ValueError("resources_per_runner exceeds the available device count")
        if isinstance(strategy, str):
            strategy = SchedulingStrategy.resolve(strategy)()
        return cls(
            RunnerExecutionBackend(
                resources=resources,
                strategy=strategy,
                runner=runner,
                startup_timeout_secs=startup_timeout_secs,
                initializer=None if initializer is None else functools.partial(initializer, *initargs),
            ),
            resources_per_runner=resources_per_runner,
        )

    @property
    def state(self) -> ExecutorState:
        with self._state_lock:
            return self._state

    def submit(
        self,
        fn: Callable[..., Any],
        /,
        *,
        timeout_secs: float,
        resource_request: ResourceRequest | None = None,
        metadata: Mapping[str, str] | None = None,
        runner: str | None = None,
        serial_group: str | None = None,
        **kwargs: Any,
    ) -> ExecutionFuture[Any]:
        """Run fn(context, **kwargs); an explicit resource request overrides the executor default."""
        if runner is not None:
            ExecutionRunner.resolve(runner)
        task = ExecutionTask(
            call=fn,
            kwargs=kwargs,
            timeout_secs=timeout_secs,
            task_id=uuid.uuid4().hex,
            resource_request=resource_request or self._resource_request,
            metadata=metadata or {},
            runner=runner,
            serial_group=serial_group,
        )
        with self._state_lock:
            if self._state != ExecutorState.OPEN:
                future: ExecutionFuture[Any] = ExecutionFuture(task.task_id)
                future.set_exception(ExecutorClosed(f"execution executor is {self._state.value}"))
                return future
            return self._backend.submit(task)

    def cancel_pending(self) -> int:
        return self._backend.cancel_pending()

    def shutdown(self, wait: bool = True) -> None:
        with self._state_lock:
            if self._state == ExecutorState.OPEN:
                self._state = ExecutorState.CLOSING
        try:
            self._backend.shutdown(wait=wait)
        finally:
            # Future.result() can wake a waiter before completion callbacks run.
            if self._backend.shutdown_result.done():
                self._on_backend_closed(self._backend.shutdown_result)

    def close(self) -> None:
        self.shutdown(wait=True)

    def __enter__(self) -> Self:
        # the children lead their own sessions: a signal must unwind through __exit__ to kill them
        if threading.current_thread() is threading.main_thread():
            for signal_number in (signal.SIGTERM, signal.SIGINT):
                self._previous_signal_handlers[signal_number] = signal.signal(
                    signal_number, self._exit_on_signal
                )
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        try:
            self.close()
        finally:
            for signal_number, handler in self._previous_signal_handlers.items():
                signal.signal(signal_number, handler)
            self._previous_signal_handlers.clear()

    def _on_backend_closed(self, result: Future[None]) -> None:
        with self._state_lock:
            self._state = ExecutorState.CLOSED

    @staticmethod
    def _exit_on_signal(signal_number: int, frame: object) -> None:
        raise SystemExit(128 + signal_number)
