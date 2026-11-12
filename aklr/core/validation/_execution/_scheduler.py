###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

import dataclasses
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import Any

from aklr.core.common import Registrable
from aklr.core.validation._execution._errors import (
    ExecutionError,
    ExecutorClosed,
    InsufficientResources,
    SubmissionRejected,
)
from aklr.core.validation._execution._future import ExecutionFuture
from aklr.core.validation._execution._model import (
    ExecutionTask,
    ResourceHealth,
    ResourceSnapshot,
    ResourceState,
)
from aklr.core.validation._execution._resources import (
    ResourceCatalog,
    ResourceLease,
)

Dispatch = Callable[[ExecutionTask, ExecutionFuture[Any], ResourceLease], None]
Assignment = tuple[ExecutionTask, ExecutionFuture[Any], ResourceLease]
Rejection = tuple[ExecutionFuture[Any], Exception]


class SchedulingStrategy(Registrable, ABC):
    @abstractmethod
    def choose(
        self,
        task: ExecutionTask,
        candidates: Sequence[ResourceSnapshot],
    ) -> tuple[ResourceSnapshot, ...] | None: ...


class RoundRobinStrategy(SchedulingStrategy, name="round_robin"):
    def __init__(self) -> None:
        self._next_index = 0
        self._lock = threading.Lock()

    def choose(
        self,
        task: ExecutionTask,
        candidates: Sequence[ResourceSnapshot],
    ) -> tuple[ResourceSnapshot, ...] | None:
        count = task.resource_request.count
        if len(candidates) < count:
            return None
        with self._lock:
            ordered = sorted(candidates, key=lambda candidate: candidate.resource_id)
            choice = tuple(ordered[(self._next_index + i) % len(ordered)] for i in range(count))
            self._next_index += count
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
        self._pending: OrderedDict[str, tuple[ExecutionTask, ExecutionFuture[Any]]] = OrderedDict()
        self._known_task_ids: set[str] = set()
        self._serial_groups: dict[str, str] = {}
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
                future.set_exception(ExecutorClosed("execution backend is closing"))
                return future
            if task.task_id in self._known_task_ids:
                future.set_exception(SubmissionRejected(f"task id '{task.task_id}' is already active"))
                return future
            if len(self._resources.matching(task.resource_request)) < task.resource_request.count:
                future.set_exception(
                    InsufficientResources(
                        f"cannot supply {task.resource_request.count} resources for task {task.task_id}"
                    )
                )
                return future
            self._known_task_ids.add(task.task_id)
            self._pending[task.task_id] = (task, future)
            self._condition.notify_all()
        return future

    def retry(self, task: ExecutionTask, future: ExecutionFuture[Any]) -> bool:
        if task.n_start_attempts > 0 or future.done():
            return False
        with self._condition:
            if (
                not self._accepting
                or task.task_id in self._known_task_ids
                or len(self._resources.matching(task.resource_request)) < task.resource_request.count
            ):
                return False
            self._known_task_ids.add(task.task_id)
            self._pending[task.task_id] = (
                dataclasses.replace(task, n_start_attempts=task.n_start_attempts + 1),
                future,
            )
            self._pending.move_to_end(task.task_id, last=False)
            self._condition.notify_all()
        return True

    def complete(self, lease: ResourceLease, health: ResourceHealth) -> None:
        try:
            self._resources.release(lease, health)
        finally:
            with self._condition:
                self._known_task_ids.discard(lease.task_id)
                self._serial_groups.pop(lease.task_id, None)
                self._condition.notify_all()

    def cancel_pending(self) -> int:
        cancelled = 0
        with self._condition:
            pending = tuple(self._pending.values())
            self._pending.clear()
            for task, _ in pending:
                self._known_task_ids.discard(task.task_id)
            self._condition.notify_all()
        for _, future in pending:
            if future.cancel():
                cancelled += 1
        return cancelled

    def shutdown(self) -> None:
        """Stop dispatching and cancel queued work; the supervisor owns running tasks."""
        with self._condition:
            self._accepting = False
            self._condition.notify_all()
        self.cancel_pending()
        self._coordinator.join()

    def _coordinate(self) -> None:
        while True:
            assignment = None
            rejection = None
            with self._condition:
                while assignment is None and rejection is None:
                    if not self._accepting:
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
                future.publish_exception(ExecutionError(f"could not dispatch task {task.task_id}: {error}"))

    def _next_decision(
        self,
    ) -> tuple[Assignment | None, Rejection | None]:
        for task, future in tuple(self._pending.values()):
            if future.cancelled():
                self._pending.pop(task.task_id)
                self._known_task_ids.discard(task.task_id)
                continue
            if task.serial_group is not None and task.serial_group in self._serial_groups.values():
                continue
            lease = None
            try:
                matching = self._resources.matching(task.resource_request)
                if len(matching) < task.resource_request.count:
                    self._pending.pop(task.task_id)
                    self._known_task_ids.discard(task.task_id)
                    return None, (
                        future,
                        InsufficientResources(f"not enough healthy resources for task {task.task_id}"),
                    )
                candidates = [resource for resource in matching if resource.state == ResourceState.AVAILABLE]
                selected = self._strategy.choose(task, candidates)
                if selected is None:
                    continue
                selected_ids = tuple(resource.resource_id for resource in selected)
                allowed_ids = {resource.resource_id for resource in candidates}
                if (
                    len(selected_ids) != task.resource_request.count
                    or len(set(selected_ids)) != len(selected_ids)
                    or not set(selected_ids) <= allowed_ids
                ):
                    raise ValueError("scheduling strategy returned an invalid resource group")
                lease = self._resources.acquire(selected_ids, task.task_id)
                if lease is None:
                    continue
                if not future.assign(selected_ids, lease.devices):
                    self._resources.release(lease, ResourceHealth.HEALTHY)
                    self._pending.pop(task.task_id)
                    self._known_task_ids.discard(task.task_id)
                    continue
                self._pending.pop(task.task_id)
                if task.serial_group is not None:
                    self._serial_groups[task.task_id] = task.serial_group
                return (task, future, lease), None
            except Exception as error:  # noqa: BLE001
                if lease is not None:
                    try:
                        self._resources.release(lease, ResourceHealth.UNHEALTHY)
                    except Exception:  # noqa: BLE001, S110
                        pass
                self._pending.pop(task.task_id)
                self._known_task_ids.discard(task.task_id)
                return None, (
                    future,
                    ExecutionError(
                        f"could not schedule task {task.task_id}: {type(error).__name__}: {error}"
                    ),
                )
        return None, None
