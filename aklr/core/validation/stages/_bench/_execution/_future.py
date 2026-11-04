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

from aklr.core.validation.stages._bench._execution._model import TaskState

T = TypeVar("T")


class _CallbackDispatcher:
    _instance: ClassVar[_CallbackDispatcher | None] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self) -> None:
        self._queue: queue.Queue[tuple[Callable[[ExecutionFuture], None], ExecutionFuture]] = queue.Queue(
            maxsize=1024
        )
        self._threads = tuple(
            threading.Thread(
                target=self._run,
                name=f"aklr-execution-callback-{index}",
                daemon=True,
            )
            for index in range(4)
        )
        for thread in self._threads:
            thread.start()

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
        self._resource_id: str | None = None
        self._device: str | None = None
        self._directory: Path | None = None
        self._metadata_lock = threading.Lock()
        self._callbacks: list[Callable[[ExecutionFuture[T]], None]] = []
        self._callback_lock = threading.Lock()
        self._transition_lock = threading.Lock()
        self._cancellation_notified = False

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
            return self._resource_id

    @property
    def device(self) -> str | None:
        with self._metadata_lock:
            return self._device

    @property
    def directory(self) -> Path | None:
        with self._metadata_lock:
            return self._directory

    def assign(self, resource_id: str, device: str | None = None) -> bool:
        if self.cancelled():
            return False
        with self._metadata_lock:
            self._resource_id = resource_id
            self._device = device
            self._task_state = TaskState.ASSIGNED
        return True

    def mark_running(self, directory: Path | None) -> bool:
        with self._transition_lock:
            if not self.set_running_or_notify_cancel():
                self._cancellation_notified = True
                return False
        with self._metadata_lock:
            self._directory = directory
            self._task_state = TaskState.RUNNING
        return True

    def set_result(self, result: T) -> None:
        super().set_result(result)
        self._dispatch_callbacks()

    def set_exception(self, exception: BaseException) -> None:
        super().set_exception(exception)
        self._dispatch_callbacks()

    def cancel(self) -> bool:
        with self._transition_lock:
            is_cancelled = super().cancel()
            if is_cancelled and not self._cancellation_notified:
                # wait() and as_completed() see a cancel only once notified; notifying twice raises
                self.set_running_or_notify_cancel()
                self._cancellation_notified = True
        if is_cancelled:
            self._dispatch_callbacks()
        return is_cancelled

    def add_done_callback(
        self,
        callback: Callable[[ExecutionFuture[T]], None],
    ) -> None:
        with self._callback_lock:
            if not self.done():
                self._callbacks.append(callback)
                return
        try:
            callback(self)
        except BaseException:
            logging.getLogger(__name__).exception("execution Future callback failed")

    def publish_result(self, result: T) -> None:
        with contextlib.suppress(InvalidStateError):
            self.set_result(result)

    def publish_exception(self, error: BaseException) -> None:
        with contextlib.suppress(InvalidStateError):
            self.set_exception(error)

    def _dispatch_callbacks(self) -> None:
        with self._callback_lock:
            callbacks = tuple(self._callbacks)
            self._callbacks.clear()
        for callback in callbacks:
            _CallbackDispatcher.default().submit(callback, self)
