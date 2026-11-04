###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

import dataclasses
import threading
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Sequence
from typing import Any, Final

from aklr.core.validation.stages._bench._execution._errors import (
    NoResourceAvailable,
    SubmissionRejected,
)
from aklr.core.validation.stages._bench._execution._future import ExecutionFuture
from aklr.core.validation.stages._bench._execution._model import (
    ExecutionTask,
    ResourceHealth,
    ResourceSnapshot,
)
from aklr.core.validation.stages._bench._execution._resources import (
    ResourceCatalog,
    ResourceLease,
)

DRAIN_TIMEOUT_SECONDS: Final = 300.0  # how long shutdown waits for in-flight tasks to report back

Dispatch = Callable[[ExecutionTask, ExecutionFuture[Any], ResourceLease], None]
Assignment = tuple[ExecutionTask, ExecutionFuture[Any], ResourceLease]
Rejection = tuple[ExecutionFuture[Any], Exception]


class SchedulingStrategy(ABC):
    @abstractmethod
    def choose(
        self,
        task: ExecutionTask,
        candidates: Sequence[ResourceSnapshot],
    ) -> ResourceSnapshot | None: ...


class RoundRobinStrategy(SchedulingStrategy):
    def __init__(self) -> None:
        self._next_index = 0
        self._lock = threading.Lock()

    def choose(
        self,
        task: ExecutionTask,
        candidates: Sequence[ResourceSnapshot],
    ) -> ResourceSnapshot | None:
        del task
        if not candidates:
            return None
        with self._lock:
            ordered = sorted(candidates, key=lambda candidate: candidate.resource_id)
            choice = ordered[self._next_index % len(ordered)]
            self._next_index += 1
            return choice


class ExecutionScheduler:
    """Hands pending tasks to free resources on its own thread, so submitting never blocks."""

    def __init__(
        self,
        resources: ResourceCatalog,
        strategy: SchedulingStrategy,
        dispatch: Dispatch,
    ) -> None:
        self._resources = resources
        self._strategy = strategy
        self._dispatch = dispatch
        self._pending: deque[tuple[ExecutionTask, ExecutionFuture[Any]]] = deque()
        self._active: dict[str, ResourceLease] = {}
        self._known_task_ids: set[str] = set()
        self._accepting = True
        self._condition = threading.Condition()
        self._coordinator = threading.Thread(
            target=self._coordinate,
            name="aklr-execution-scheduler",
            daemon=True,
        )
        self._coordinator.start()

    def submit(self, task: ExecutionTask) -> ExecutionFuture[Any]:
        future: ExecutionFuture[Any] = ExecutionFuture(task.task_id)
        with self._condition:
            if not self._accepting:
                future.set_exception(SubmissionRejected("execution backend is closing"))
                return future
            if task.task_id in self._known_task_ids:
                future.set_exception(SubmissionRejected(f"task id '{task.task_id}' is already active"))
                return future
            if not self._resources.matching(task.resource_request):
                future.set_exception(
                    NoResourceAvailable(f"no execution resource satisfies task {task.task_id}")
                )
                return future
            self._known_task_ids.add(task.task_id)
            self._pending.append((task, future))
            self._condition.notify_all()
        return future

    def retry(self, task: ExecutionTask, future: ExecutionFuture[Any]) -> bool:
        if task.n_start_attempts > 0 or future.done():
            return False
        with self._condition:
            if (
                not self._accepting
                or task.task_id in self._known_task_ids
                or not self._resources.matching(task.resource_request)
            ):
                return False
            self._known_task_ids.add(task.task_id)
            self._pending.appendleft(
                (
                    dataclasses.replace(task, n_start_attempts=task.n_start_attempts + 1),
                    future,
                )
            )
            self._condition.notify_all()
        return True

    def complete(self, lease: ResourceLease, health: ResourceHealth) -> None:
        try:
            self._resources.release(lease, health)
        finally:
            with self._condition:
                self._active.pop(lease.task_id, None)
                self._known_task_ids.discard(lease.task_id)
                self._condition.notify_all()

    def cancel_pending(self) -> int:
        cancelled = 0
        with self._condition:
            pending = tuple(self._pending)
            self._pending.clear()
            for task, _ in pending:
                self._known_task_ids.discard(task.task_id)
            self._condition.notify_all()
        for _, future in pending:
            if future.cancel():
                cancelled += 1
        return cancelled

    def shutdown(self, wait: bool = True, timeout: float = DRAIN_TIMEOUT_SECONDS) -> None:
        with self._condition:
            self._accepting = False
            pending = tuple(self._pending)
            self._pending.clear()
            for task, _ in pending:
                self._known_task_ids.discard(task.task_id)
            self._condition.notify_all()
        for _, future in pending:
            future.cancel()
        if wait:
            with self._condition:
                is_drained = self._condition.wait_for(lambda: not self._active, timeout)
            if not is_drained:
                with self._condition:
                    stranded = tuple(self._active.values())
                    self._active.clear()
                    for lease in stranded:
                        self._known_task_ids.discard(lease.task_id)
                    self._condition.notify_all()
                for lease in stranded:
                    try:
                        self._resources.release(lease, ResourceHealth.UNHEALTHY)
                    except Exception:  # noqa: BLE001, S110
                        pass
        with self._condition:
            self._condition.notify_all()
        self._coordinator.join(timeout)

    def _coordinate(self) -> None:
        try:
            while True:
                assignment = None
                rejection = None
                with self._condition:
                    while assignment is None and rejection is None:
                        for pending_task, pending_future in self._pending:
                            if pending_future.cancelled():
                                self._known_task_ids.discard(pending_task.task_id)
                        self._pending = deque(
                            (pending_task, pending_future)
                            for pending_task, pending_future in self._pending
                            if not pending_future.cancelled()
                        )
                        if not self._accepting and not self._pending:
                            return
                        assignment, rejection = self._next_decision()
                        if assignment is None and rejection is None:
                            self._condition.wait()

                if rejection is not None:
                    future, error = rejection
                    future.publish_exception(error)
                    continue

                task, future, lease = assignment
                try:
                    self._dispatch(task, future, lease)
                except BaseException as error:  # noqa: BLE001
                    self.complete(lease, ResourceHealth.HEALTHY)
                    future.publish_exception(SubmissionRejected(str(error)))
        finally:
            with self._condition:
                self._condition.notify_all()

    def _next_decision(
        self,
    ) -> tuple[Assignment | None, Rejection | None]:
        for task, future in tuple(self._pending):
            lease = None
            try:
                if not self._resources.matching(task.resource_request):
                    self._discard(future)
                    self._known_task_ids.discard(task.task_id)
                    return None, (
                        future,
                        NoResourceAvailable(f"no healthy execution resource satisfies task {task.task_id}"),
                    )
                candidates = self._resources.available(task.resource_request)
                selected = self._strategy.choose(task, candidates)
                if selected is None:
                    continue
                lease = self._resources.acquire(selected.resource_id, task.task_id)
                if lease is None:
                    continue
                self._discard(future)
                if not future.assign(selected.resource_id, selected.device):
                    self._resources.release(lease, ResourceHealth.HEALTHY)
                    self._known_task_ids.discard(task.task_id)
                    continue
                self._active[task.task_id] = lease
                return (task, future, lease), None
            except Exception as error:  # noqa: BLE001
                if lease is not None:
                    try:
                        self._resources.release(lease, ResourceHealth.UNHEALTHY)
                    except Exception:  # noqa: BLE001, S110
                        pass
                self._discard(future)
                self._known_task_ids.discard(task.task_id)
                return None, (
                    future,
                    SubmissionRejected(
                        f"could not schedule task {task.task_id}: {type(error).__name__}: {error}"
                    ),
                )
        return None, None

    def _discard(self, discarded: ExecutionFuture[Any]) -> None:
        self._pending = deque((task, future) for task, future in self._pending if future is not discarded)
