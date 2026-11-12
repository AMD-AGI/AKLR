###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

import contextlib
import logging
import queue
import threading
from collections.abc import Callable
from concurrent.futures import CancelledError, Future, InvalidStateError
from pathlib import Path
from typing import ClassVar, Generic, TypeVar

from aklr.core.validation._execution._model import TaskState

T = TypeVar("T")


class _CallbackDispatcher:
    _instance: ClassVar[_CallbackDispatcher | None] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self) -> None:
        self._queue: queue.SimpleQueue[tuple[Callable[[ExecutionFuture], None], ExecutionFuture]] = (
            queue.SimpleQueue()
        )
        for index in range(4):
            threading.Thread(
                target=self._run,
                name=f"aklr-execution-callback-{index}",
                daemon=True,
            ).start()

    @classmethod
    def default(cls) -> _CallbackDispatcher:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def submit(
        self,
        callback: Callable[[ExecutionFuture], None],
        future: ExecutionFuture,
    ) -> None:
        self._queue.put((callback, future))

    def _run(self) -> None:
        while True:
            callback, future = self._queue.get()
            try:
                callback(future)
            except BaseException:
                logging.getLogger(__name__).exception("execution Future callback failed")


class ExecutionFuture(Future, Generic[T]):
    """A Future that also records where its task runs, and runs done-callbacks off the resolving thread."""

    def __init__(self, task_id: str) -> None:
        super().__init__()
        self._task_id = task_id
        self._task_state = TaskState.PENDING
        self._resource_ids: tuple[str, ...] = ()
        self._devices: tuple[str, ...] = ()
        self._directory: Path | None = None
        self._metadata_lock = threading.RLock()

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def state(self) -> TaskState:
        if self.cancelled():
            return TaskState.CANCELLED
        if self.done():
            try:
                return TaskState.FAILED if super().exception(timeout=0) is not None else TaskState.SUCCEEDED
            except CancelledError:
                return TaskState.CANCELLED
        with self._metadata_lock:
            return self._task_state

    @property
    def resource_id(self) -> str | None:
        with self._metadata_lock:
            return self._resource_ids[0] if self._resource_ids else None

    @property
    def device(self) -> str | None:
        with self._metadata_lock:
            return self._devices[0] if self._devices else None

    @property
    def resource_ids(self) -> tuple[str, ...]:
        with self._metadata_lock:
            return self._resource_ids

    @property
    def devices(self) -> tuple[str, ...]:
        with self._metadata_lock:
            return self._devices

    @property
    def directory(self) -> Path | None:
        with self._metadata_lock:
            return self._directory

    def assign(self, resource_ids: tuple[str, ...] | str, devices: tuple[str, ...] | str) -> bool:
        with self._metadata_lock:
            if self.done():
                return False
            self._resource_ids = (resource_ids,) if isinstance(resource_ids, str) else resource_ids
            self._devices = (devices,) if isinstance(devices, str) else devices
            self._task_state = TaskState.ASSIGNED
        return True

    def mark_running(self, directory: Path | None) -> bool:
        with self._metadata_lock:
            if self.cancelled():
                return False
            self.set_running_or_notify_cancel()
            self._directory = directory
            self._task_state = TaskState.RUNNING
        return True

    def cancel(self) -> bool:
        with self._metadata_lock:
            if self.done():
                return self.cancelled()
            if not super().cancel():
                return False
            # wait() and as_completed() need notification even if this task is never dispatched.
            self.set_running_or_notify_cancel()
            return True

    def add_done_callback(
        self,
        callback: Callable[[ExecutionFuture[T]], None],
    ) -> None:
        """Dispatch off-thread, including callbacks registered after completion."""
        super().add_done_callback(lambda future: _CallbackDispatcher.default().submit(callback, future))

    def publish_result(self, result: T) -> None:
        with contextlib.suppress(InvalidStateError):
            self.set_result(result)

    def publish_exception(self, error: BaseException) -> None:
        with contextlib.suppress(InvalidStateError):
            self.set_exception(error)
