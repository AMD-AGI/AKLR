###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

import shutil
import tempfile
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from aklr.core.validation._execution._errors import (
    ExecutionError,
    RunnerStartupFailed,
    TaskCancelled,
)
from aklr.core.validation._execution._future import ExecutionFuture
from aklr.core.validation._execution._model import (
    ExecutionTask,
    ResourceHealth,
    RunnerHealth,
    RunnerState,
)
from aklr.core.validation._execution._resources import (
    ResourceCatalog,
    ResourceLease,
)
from aklr.core.validation._execution._runner import (
    DEFAULT_STARTUP_TIMEOUT_SECS,
    ExecutionRunner,
)
from aklr.core.validation._execution._scheduler import (
    ExecutionScheduler,
    SchedulingStrategy,
)


class ExecutionBackend(ABC):
    @property
    @abstractmethod
    def shutdown_result(self) -> Future[None]:
        """The same non-cancellable Future for every caller; completed after all cleanup ends."""

    @abstractmethod
    def submit(self, task: ExecutionTask) -> ExecutionFuture[Any]: ...

    @abstractmethod
    def cancel_pending(self) -> int: ...

    @abstractmethod
    def shutdown(self, wait: bool = True) -> None:
        """Stop accepting work, initiate cleanup once, and optionally wait for shutdown_result.

        Cleanup failures belong to shutdown_result; a cleanup callback reentering
        shutdown must not wait on its own completion.
        """


class RunnerSupervisor:
    """Owns runner lifetimes and resolves tasks only after cleanup and resource release."""

    def __init__(
        self,
        on_finished: Callable[[ResourceLease, ResourceHealth], None],
        retry: Callable[[ExecutionTask, ExecutionFuture[Any]], bool],
        max_concurrency: int,
        *,
        runner: str = "disposable",
        startup_timeout_secs: float = DEFAULT_STARTUP_TIMEOUT_SECS,
        initializer: Callable[[], None] | None = None,
    ) -> None:
        self._runner_type = ExecutionRunner.resolve(runner)
        self._startup_timeout_secs = startup_timeout_secs
        self._initializer = initializer
        self._job_dir = Path(tempfile.mkdtemp(prefix="aklr-validation-"))
        self._on_finished = on_finished
        self._retry = retry
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, max_concurrency),
            thread_name_prefix="aklr-execution-runner",
        )
        self._active: dict[str, ExecutionRunner] = {}
        self._idle: dict[tuple[str, ...], ExecutionRunner] = {}  # reusable runners, by their resource IDs
        self._is_closing = False
        self._lock = threading.Lock()

    def dispatch(
        self,
        task: ExecutionTask,
        future: ExecutionFuture[Any],
        lease: ResourceLease,
    ) -> None:
        self._pool.submit(self._run, task, future, lease)

    def shutdown(self) -> None:
        with self._lock:
            self._is_closing = True
            runners = (*self._active.values(), *self._idle.values())
            self._idle.clear()
        for runner in runners:
            try:
                runner.close()
            except Exception:  # noqa: BLE001, S110
                # the runner's own pool thread reads its health and reports it
                pass
        self._pool.shutdown(wait=True)
        shutil.rmtree(self._job_dir, ignore_errors=True)

    def _run(
        self,
        task: ExecutionTask,
        future: ExecutionFuture[Any],
        lease: ResourceLease,
    ) -> None:
        if future.cancelled():
            self._on_finished(lease, ResourceHealth.HEALTHY)
            return

        runner = None
        directory = None
        result = None
        is_started = False
        is_returned = False
        failure = None
        runner_health = RunnerHealth.HEALTHY
        resource_ids = tuple(resource.resource_id for resource in lease.resources)
        try:
            runner_type = self._runner_type if task.runner is None else ExecutionRunner.resolve(task.runner)
            with self._lock:
                if self._is_closing:
                    raise TaskCancelled(f"task {task.task_id} was cancelled: the backend is closing")
                idle = self._idle.pop(resource_ids, None)
                # a leased device is free, so an idle runner on it only holds its context: it goes
                stale = [ids for ids in self._idle if set(ids) & set(resource_ids)]
                evicted = [self._idle.pop(ids) for ids in stale]
                if idle is not None and type(idle) is runner_type:
                    runner = idle
                elif idle is not None:
                    evicted.append(idle)
            for old in evicted:
                old.close()
            if runner is None:
                directory = self._job_dir / lease.lease_id
                directory.mkdir()
                runner = runner_type.create(
                    lease.devices,
                    directory,
                    startup_timeout_secs=self._startup_timeout_secs,
                    initializer=self._initializer,
                )
            directory = runner.directory
            with self._lock:
                if self._is_closing:
                    raise TaskCancelled(f"task {task.task_id} was cancelled: the backend is closing")
                self._active[task.task_id] = runner
            if runner.state == RunnerState.CREATED:
                runner.start()
            is_started = True
        except BaseException as error:  # noqa: BLE001
            failure = self._as_execution_error(error, RunnerStartupFailed, "could not start execution runner")
        else:
            # marked running only once started: a start failure may retry the task elsewhere
            if future.mark_running(directory):
                try:
                    result = runner.execute(task)
                    is_returned = True
                except BaseException as error:  # noqa: BLE001
                    failure = self._as_execution_error(
                        error, ExecutionError, f"could not execute task {task.task_id}"
                    )
        finally:
            is_kept = False
            if runner is not None and is_returned and runner.reusable:
                # a runner whose task succeeded waits for the next task on the same resources
                with self._lock:
                    is_kept = (
                        not self._is_closing
                        and runner.health() == RunnerHealth.HEALTHY
                        and runner.state == RunnerState.READY
                    )
                    if is_kept:
                        self._idle[resource_ids] = runner
                        self._active.pop(task.task_id, None)
            if runner is not None and not is_kept:
                try:
                    runner.close()
                    runner_health = runner.health()
                except BaseException as error:  # noqa: BLE001
                    runner_health = RunnerHealth.RESOURCE_FAILED
                    failure = self._append_cleanup_failure(failure, "runner cleanup failed", error)
                finally:
                    with self._lock:
                        self._active.pop(task.task_id, None)

                try:
                    diagnostics = runner.diagnostics()
                except BaseException as error:  # noqa: BLE001
                    diagnostics = None
                    failure = self._append_cleanup_failure(failure, "runner diagnostics failed", error)
                if diagnostics and failure is not None:
                    # the log first, so the failure's own text stays the last line
                    message = f"Runner log:\n{diagnostics}\n\n{failure}"
                    failure = (
                        type(failure)(message)
                        if isinstance(failure, ExecutionError)
                        else ExecutionError(message)
                    )

            resource_health = (
                ResourceHealth.UNHEALTHY
                if runner_health == RunnerHealth.RESOURCE_FAILED
                else ResourceHealth.HEALTHY
            )
            try:
                self._on_finished(lease, resource_health)
            except BaseException as error:  # noqa: BLE001
                failure = self._append_cleanup_failure(failure, "resource release failed", error)

        if failure is None and is_returned:
            future.publish_result(result)
            return
        # a failed or cancelled task's directory goes now; a succeeded one's is the caller's
        if directory is not None:
            shutil.rmtree(directory, ignore_errors=True)
        if failure is None:
            return
        # the runner never started and its resource left: the task tries once more elsewhere
        if not is_started and resource_health == ResourceHealth.UNHEALTHY and self._retry(task, future):
            return
        future.publish_exception(failure)

    @staticmethod
    def _as_execution_error(
        error: BaseException,
        step_failure_type: type[ExecutionError],
        description: str,
    ) -> ExecutionError:
        if isinstance(error, ExecutionError):
            return error
        return step_failure_type(f"{description}: {type(error).__name__}: {error}")

    @staticmethod
    def _append_cleanup_failure(
        failure: BaseException | None,
        message: str,
        error: BaseException,
    ) -> ExecutionError:
        detail = f"{message}: {type(error).__name__}: {error}"
        if failure is not None:
            detail = f"{failure}\n\n{detail}"
            if isinstance(failure, ExecutionError):
                return type(failure)(detail)
        return ExecutionError(detail)


class RunnerExecutionBackend(ExecutionBackend):
    """Connects the scheduler to runner supervision and closes them in dependency order."""

    def __init__(
        self,
        resources: ResourceCatalog,
        strategy: str | SchedulingStrategy | None = None,
        max_concurrency: int | None = None,
        *,
        runner: str = "disposable",
        startup_timeout_secs: float = DEFAULT_STARTUP_TIMEOUT_SECS,
        initializer: Callable[[], None] | None = None,
    ) -> None:
        if strategy is None:
            strategy = "round_robin"
        if isinstance(strategy, str):
            strategy = SchedulingStrategy.resolve(strategy)()
        self._scheduler: ExecutionScheduler
        self._supervisor = RunnerSupervisor(
            runner=runner,
            startup_timeout_secs=startup_timeout_secs,
            initializer=initializer,
            on_finished=lambda lease, health: self._scheduler.complete(lease, health),
            retry=lambda task, future: self._scheduler.retry(task, future),
            max_concurrency=max_concurrency or len(resources),
        )
        self._scheduler = ExecutionScheduler(
            resources=resources,
            strategy=strategy,
            dispatch=self._supervisor.dispatch,
        )
        self._shutdown_lock = threading.Lock()
        self._shutdown_owner: threading.Thread | None = None
        self._shutdown_result: Future[None] = Future()
        self._shutdown_result.set_running_or_notify_cancel()

    @property
    def shutdown_result(self) -> Future[None]:
        return self._shutdown_result

    def submit(self, task: ExecutionTask) -> ExecutionFuture[Any]:
        return self._scheduler.submit(task)

    def cancel_pending(self) -> int:
        return self._scheduler.cancel_pending()

    def shutdown(self, wait: bool = True) -> None:
        # Stop accepting work before returning, including when cleanup runs in the background.
        self._scheduler.shutdown()
        with self._shutdown_lock:
            initiate = self._shutdown_owner is None
            if initiate:
                self._shutdown_owner = (
                    threading.current_thread()
                    if wait
                    else threading.Thread(
                        target=self._close_supervisor, name="aklr-execution-backend-shutdown", daemon=True
                    )
                )
                if not wait:
                    try:
                        self._shutdown_owner.start()
                    except BaseException:
                        self._shutdown_owner = None
                        raise
        if initiate and wait:
            self._close_supervisor()
        if wait and (self._shutdown_owner is not threading.current_thread() or self._shutdown_result.done()):
            self._shutdown_result.result()

    def _close_supervisor(self) -> None:
        try:
            self._supervisor.shutdown()
        except BaseException as error:  # noqa: BLE001
            self._shutdown_result.set_exception(error)
        else:
            self._shutdown_result.set_result(None)
