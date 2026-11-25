###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

import contextlib
import logging
import multiprocessing
import multiprocessing.connection
import os
import signal
import threading
import time
import traceback
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import Future
from pathlib import Path
from typing import Any, ClassVar, Final, Self

from aklr.core.common import Registrable
from aklr.core.validation._execution._errors import (
    ChannelError,
    ExecutionError,
    InvalidRunnerState,
    RunnerCrashed,
    RunnerStartupFailed,
    TaskCancelled,
    TaskFailed,
    TaskTimeout,
)
from aklr.core.validation._execution._model import (
    ExecutionContext,
    ExecutionTask,
    RunnerHealth,
    RunnerState,
)
from aklr.core.validation._execution._task_log import TaskLog
from aklr.core.validation._execution._transport import (
    ChannelReceiveTimeout,
    ChildRunnerEndpoint,
    ExecuteTask,
    MultiprocessingRunnerChannel,
    RunnerEvent,
    RunnerExited,
    RunnerFailed,
    RunnerStarted,
    ShutdownRunner,
    TaskCompleted,
    TaskRaised,
)

DEFAULT_STARTUP_TIMEOUT_SECS: Final = 30.0


class _RunnerExited(RuntimeError): ...


class _RunnerResponseTimeout(TimeoutError): ...


class _UnexpectedEvent(Exception):  # noqa: N818
    def __init__(self, rank: int, event: RunnerEvent) -> None:
        super().__init__(rank, event)
        self.rank = rank
        self.event = event


class ExecutionRunner(Registrable, ABC):
    reusable: ClassVar[bool] = False

    @classmethod
    @abstractmethod
    def create(
        cls,
        devices: tuple[str, ...],
        directory: Path,
        *,
        startup_timeout_secs: float,
        initializer: Callable[[], None] | None = None,
    ) -> Self: ...

    @property
    @abstractmethod
    def runner_id(self) -> str: ...

    @property
    @abstractmethod
    def state(self) -> RunnerState: ...

    @property
    @abstractmethod
    def directory(self) -> Path: ...

    @property
    @abstractmethod
    def devices(self) -> tuple[str, ...]: ...

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def execute(self, task: ExecutionTask) -> Any: ...

    @abstractmethod
    def health(self) -> RunnerHealth: ...

    @abstractmethod
    def close(self) -> None: ...

    def diagnostics(self) -> str | None:
        return None


class ProcessExecutionRunner(ExecutionRunner):
    _POLL_SECS: Final = 0.25
    _JOIN_SECS: Final = 2.0
    _KILL_WAIT_SECS: Final = 5.0

    @classmethod
    def create(
        cls,
        devices: tuple[str, ...],
        directory: Path,
        *,
        startup_timeout_secs: float,
        initializer: Callable[[], None] | None = None,
    ) -> Self:
        return cls(devices, directory, startup_timeout_secs=startup_timeout_secs, initializer=initializer)

    def __init__(
        self,
        devices: tuple[str, ...],
        directory: Path,
        startup_timeout_secs: float = DEFAULT_STARTUP_TIMEOUT_SECS,
        initializer: Callable[[], None] | None = None,
    ) -> None:
        self._devices = devices
        self._initializer = initializer
        self._directory = directory
        self._startup_timeout_secs = startup_timeout_secs
        self._runner_id = uuid.uuid4().hex
        self._state = RunnerState.CREATED
        self._health = RunnerHealth.HEALTHY
        self._state_lock = threading.Lock()
        self._processes: list[Any] = []
        self._channels: list[MultiprocessingRunnerChannel] = []
        self._close_lock = threading.RLock()
        self._close_result: Future[None] = Future()

    @property
    def runner_id(self) -> str:
        return self._runner_id

    @property
    def state(self) -> RunnerState:
        with self._state_lock:
            return self._state

    @property
    def directory(self) -> Path:
        return self._directory

    @property
    def devices(self) -> tuple[str, ...]:
        return self._devices

    def start(self) -> None:
        with self._state_lock:
            if self._state != RunnerState.CREATED:
                raise InvalidRunnerState(f"runner {self._runner_id} is {self._state.value}, expected created")
            self._state = RunnerState.STARTING

        context = multiprocessing.get_context("spawn")
        pairs = [MultiprocessingRunnerChannel.create(context) for _ in self._devices]
        processes = [
            context.Process(
                target=ProcessExecutionRunner._run_child,
                args=(endpoint, self._runner_id, self._devices, rank, self._directory, self._initializer),
                daemon=False,
            )
            for rank, (_, endpoint) in enumerate(pairs)
        ]
        try:
            with self._state_lock:
                if self._state != RunnerState.STARTING:
                    for channel, _ in pairs:
                        channel.close()
                    raise TaskCancelled(f"runner {self._runner_id} was closed while starting")
                self._channels = [channel for channel, _ in pairs]
                self._processes = processes
                for process in processes:
                    process.start()
            for _, endpoint in pairs:
                endpoint.close()
            try:
                self._receive_from_every_rank(self._startup_timeout_secs, RunnerStarted)
            except _UnexpectedEvent as unexpected:
                rank, event = unexpected.rank, unexpected.event
                if isinstance(event, RunnerFailed) and event.runner_id == self._runner_id:
                    self._set_health(
                        RunnerHealth.RESOURCE_FAILED if event.resource_failed else RunnerHealth.FAILED
                    )
                    raise RunnerStartupFailed(self._of_rank(rank, event.reason)) from None
                message = f"runner {self._runner_id} sent {type(event).__name__} while starting"
                raise RunnerStartupFailed(self._of_rank(rank, message)) from None
            with self._state_lock:
                if self._state != RunnerState.STARTING:
                    raise InvalidRunnerState(
                        f"runner {self._runner_id} became {self._state.value} while starting"
                    )
                self._state = RunnerState.READY
        except Exception as error:
            is_closed = self.state == RunnerState.CLOSED
            if not is_closed:
                self._set_state(RunnerState.FAILED)
                if self.health() == RunnerHealth.HEALTHY:
                    self._set_health(RunnerHealth.RESOURCE_FAILED)
            try:
                self.close()
            except ExecutionError as close_error:
                logging.getLogger(__name__).warning(
                    "runner %s did not close cleanly: %s", self._runner_id, close_error
                )
            if is_closed:
                raise TaskCancelled(f"runner {self._runner_id} was closed while starting") from error
            if isinstance(error, _RunnerResponseTimeout):
                raise RunnerStartupFailed(
                    f"runner {self._runner_id} did not start within {self._startup_timeout_secs} seconds"
                ) from error
            if isinstance(error, _RunnerExited):
                raise RunnerStartupFailed(str(error)) from error
            if isinstance(error, ExecutionError):
                raise
            raise RunnerStartupFailed(
                f"runner {self._runner_id} could not start: {type(error).__name__}: {error}"
            ) from error
        finally:
            for _, endpoint in pairs:
                endpoint.close()

    def execute(self, task: ExecutionTask) -> Any:
        task_id = task.task_id
        with self._state_lock:
            if self._state == RunnerState.CLOSED:
                raise TaskCancelled(f"task {task_id} was cancelled")
            if self._state != RunnerState.READY:
                raise InvalidRunnerState(f"runner {self._runner_id} is {self._state.value}, expected ready")
            for channel in self._require_channels():
                channel.send(ExecuteTask(runner_id=self._runner_id, task=task))
            self._state = RunnerState.RUNNING

        try:
            events = self._receive_from_every_rank(task.timeout_secs, TaskCompleted, task_id)
        except _UnexpectedEvent as unexpected:
            rank, event = unexpected.rank, unexpected.event
        except (_RunnerResponseTimeout, _RunnerExited, ChannelError) as error:
            if self.state == RunnerState.CLOSED:
                raise TaskCancelled(f"task {task_id} was cancelled") from error
            self._set_state(RunnerState.FAILED)
            self._set_health(RunnerHealth.FAILED)
            if isinstance(error, _RunnerResponseTimeout):
                raise TaskTimeout(str(error)) from error
            if isinstance(error, _RunnerExited):
                raise RunnerCrashed(str(error)) from error
            raise TaskFailed(
                f"runner {self._runner_id} sent an invalid message for task {task_id}: {error}"
            ) from error

        else:
            self._set_state(RunnerState.READY)
            return events[0].result
        self._set_state(RunnerState.FAILED)
        self._set_health(RunnerHealth.FAILED)
        if isinstance(event, TaskRaised) and event.runner_id == self._runner_id and event.task_id == task_id:
            raise TaskFailed(self._of_rank(rank, event.reason))
        if isinstance(event, RunnerExited) and event.runner_id == self._runner_id:
            raise RunnerCrashed(
                self._of_rank(rank, f"runner {self._runner_id} exited before task {task_id} completed")
            )
        raise TaskFailed(
            self._of_rank(rank, f"runner {self._runner_id} sent an invalid event for task {task_id}: {event}")
        )

    def health(self) -> RunnerHealth:
        with self._state_lock:
            return self._health

    def close(self) -> None:
        with self._close_lock:
            if self._close_result.running():
                return
            if not self._close_result.done():
                self._close_result.set_running_or_notify_cancel()
                try:
                    self._close()
                except BaseException as error:  # noqa: BLE001
                    self._close_result.set_exception(error)
                else:
                    self._close_result.set_result(None)
            self._close_result.result()

    def _close(self) -> None:
        with self._state_lock:
            previous_state = self._state
            self._state = RunnerState.CLOSED

        has_cleanup_failed = False
        is_shutdown_sent = False
        if previous_state in {RunnerState.STARTING, RunnerState.READY, RunnerState.RUNNING}:
            for channel in self._channels:
                try:
                    channel.send(ShutdownRunner(runner_id=self._runner_id))
                    is_shutdown_sent = True
                except Exception:  # noqa: BLE001
                    has_cleanup_failed = True

        is_orphaned = False
        if is_shutdown_sent and previous_state == RunnerState.READY:
            for process in self._processes:
                if process.pid is not None:
                    process.join(timeout=self._JOIN_SECS)
        for process in self._processes:
            if process.pid is None:
                continue
            is_gone = False
            try:
                process.kill()
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(process.pid, signal.SIGKILL)
                deadline = time.monotonic() + self._KILL_WAIT_SECS
                while True:
                    is_alive = False
                    for stat_path in Path("/proc").glob("[0-9]*/stat"):
                        try:
                            state, _, group_id = stat_path.read_bytes().rpartition(b")")[2].split()[:3]
                        except (OSError, ValueError):
                            continue
                        if int(group_id) == process.pid and state != b"Z":
                            is_alive = True
                            break
                    if not is_alive or time.monotonic() >= deadline:
                        break
                    time.sleep(0.01)
                process.join(timeout=self._JOIN_SECS)
                is_gone = not is_alive and bool(multiprocessing.connection.wait([process.sentinel], 0))
            except Exception:  # noqa: BLE001
                has_cleanup_failed = True
            is_orphaned = is_orphaned or not is_gone
        if is_orphaned:
            self._set_health(RunnerHealth.RESOURCE_FAILED)
        elif has_cleanup_failed and self.health() == RunnerHealth.HEALTHY:
            self._set_health(RunnerHealth.FAILED)

        for channel in self._channels:
            channel.close()
        if is_orphaned:
            raise ExecutionError(f"runner {self._runner_id}: the process group outlived its kill")

    def diagnostics(self) -> str | None:
        tails = [TaskLog(self._log_path(self._directory, rank)).tail() for rank in range(len(self._devices))]
        if len(tails) == 1:
            return tails[0] or None
        return "\n\n".join(f"rank {rank}:\n{tail}" for rank, tail in enumerate(tails) if tail) or None

    def _receive_from_every_rank(
        self, timeout: float | None, expected: type[RunnerEvent], task_id: str | None = None
    ) -> list[RunnerEvent]:
        if not self._processes:
            raise InvalidRunnerState(f"runner {self._runner_id} has no process")
        deadline = None if timeout is None else time.monotonic() + timeout
        pending = dict(enumerate(zip(self._processes, self._require_channels(), strict=True)))
        events: dict[int, RunnerEvent] = {}
        while pending:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is None or remaining > 0:
                handles = [
                    h
                    for process, channel in pending.values()
                    for h in (channel.event_connection, process.sentinel)
                ]
                multiprocessing.connection.wait(handles, min(self._POLL_SECS, remaining or self._POLL_SECS))
            for rank, (process, channel) in tuple(pending.items()):
                try:
                    event = channel.receive(timeout=0)
                except (ChannelReceiveTimeout, EOFError) as error:
                    event = None
                    if isinstance(error, ChannelReceiveTimeout) and process.is_alive():
                        continue
                    if isinstance(error, ChannelReceiveTimeout):
                        with contextlib.suppress(ChannelReceiveTimeout, EOFError):
                            event = channel.receive(timeout=0)
                    if event is None:
                        process.join(self._JOIN_SECS)
                        message = f"runner {self._runner_id} exited with code {process.exitcode}"
                        raise _RunnerExited(self._of_rank(rank, message)) from None
                if (
                    not isinstance(event, expected)
                    or event.runner_id != self._runner_id
                    or (task_id is not None and getattr(event, "task_id", None) != task_id)
                ):
                    raise _UnexpectedEvent(rank, event)
                events[rank] = event
                del pending[rank]
            if pending and remaining is not None and remaining <= 0:
                raise _RunnerResponseTimeout(
                    self._of_rank(
                        min(pending), f"runner {self._runner_id} did not answer within {timeout} seconds"
                    )
                )
        return [events[rank] for rank in sorted(events)]

    def _of_rank(self, rank: int, message: str) -> str:
        return message if len(self._devices) == 1 else f"rank {rank}: {message}"

    def _set_state(self, state: RunnerState) -> None:
        with self._state_lock:
            if self._state != RunnerState.CLOSED:
                self._state = state

    def _set_health(self, health: RunnerHealth) -> None:
        with self._state_lock:
            self._health = health

    def _require_channels(self) -> list[MultiprocessingRunnerChannel]:
        if not self._channels:
            raise InvalidRunnerState(f"runner {self._runner_id} has no channel")
        return self._channels

    @staticmethod
    def _log_path(directory: Path, rank: int) -> Path:
        return directory / ("task.log" if rank == 0 else f"task.rank{rank}.log")

    @staticmethod
    def _run_child(
        endpoint: ChildRunnerEndpoint,
        runner_id: str,
        devices: tuple[str, ...],
        rank: int,
        directory: Path,
        initializer: Callable[[], None] | None,
    ) -> None:
        task_id: str | None = None
        device = devices[rank]
        is_device_ready = False
        os.environ.update(RANK=str(rank), WORLD_SIZE=str(len(devices)), LOCAL_RANK=str(rank))
        try:
            os.setsid()

            def die_with_parent() -> None:
                parent = multiprocessing.parent_process()
                while os.getppid() == parent.pid and not multiprocessing.connection.wait(
                    [parent.sentinel], timeout=1.0
                ):
                    pass
                os.killpg(0, signal.SIGKILL)

            threading.Thread(target=die_with_parent, name="aklr-runner-parent-watch", daemon=True).start()
            os.chdir(directory)
            task_log = TaskLog(ProcessExecutionRunner._log_path(directory, rank))
            try:
                with task_log.redirected_stdio():
                    import torch

                    import aklr.core.utils.testing  # noqa: F401

                    torch.cuda.set_device(device)
                    torch.cuda.init()
                    is_device_ready = True
                    if initializer is not None:
                        initializer()
                endpoint.send(RunnerStarted(runner_id=runner_id))
                context = ExecutionContext(devices=devices, log_path=task_log.path, directory=directory)
                while not isinstance(command := endpoint.receive(), ShutdownRunner):
                    if not isinstance(command, ExecuteTask) or command.runner_id != runner_id:
                        raise RuntimeError(f"runner received invalid command: {command}")
                    task_id = command.task.task_id
                    with task_log.redirected_stdio():
                        result = command.task.call(context, **command.task.kwargs)
                        torch.cuda.synchronize()
                    endpoint.send(
                        TaskCompleted(
                            runner_id=runner_id, task_id=task_id, result=result if rank == 0 else None
                        )
                    )
                    task_id = None
            except BaseException as error:  # noqa: BLE001
                reason = f"{type(error).__name__}: {error}\n\n{traceback.format_exc()}"
                try:
                    endpoint.send(
                        TaskRaised(runner_id=runner_id, task_id=task_id, reason=reason)
                        if task_id is not None
                        else RunnerFailed(
                            runner_id=runner_id,
                            reason=reason,
                            resource_failed=not is_device_ready,
                        )
                    )
                except Exception:  # noqa: BLE001, S110
                    pass
        finally:
            try:
                endpoint.send(RunnerExited(runner_id=runner_id, task_id=task_id, exit_code=0))
            except Exception:  # noqa: BLE001, S110
                pass


class DisposableExecutionRunner(ProcessExecutionRunner, name="disposable"): ...


class ReusableExecutionRunner(ProcessExecutionRunner, name="reuse"):
    reusable: ClassVar[bool] = True
