###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

import ctypes
import itertools
import logging
import multiprocessing
import os
import queue
import shutil
import signal
import sys
import tempfile
import threading
import time
import traceback
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Final, NoReturn

from aklr.core.platform.device import physical_device
from aklr.core.validation.stages._bench._execution._engine import ExecutionEngine
from aklr.core.validation.stages._bench._execution._errors import (
    ChannelFailure,
    ExecutionError,
    InvalidRunnerState,
    RunnerCommandFailed,
    RunnerCrashed,
    RunnerExitedEarly,
    RunnerStartupFailed,
    TaskCancelled,
    TaskExecutionFailed,
    TaskExecutionTimeout,
)
from aklr.core.validation.stages._bench._execution._model import (
    ExecutionTask,
    RunnerHealth,
    RunnerState,
)
from aklr.core.validation.stages._bench._execution._process_group import ProcessGroup
from aklr.core.validation.stages._bench._execution._resources import ResourceLease
from aklr.core.validation.stages._bench._execution._task_log import TaskLog
from aklr.core.validation.stages._bench._execution._transport import (
    CancelTask,
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
    TaskFailed,
    TaskStarted,
)

DEFAULT_STARTUP_TIMEOUT_SECONDS: Final = 30.0
_PR_SET_PDEATHSIG: Final = 1
_PR_SET_CHILD_SUBREAPER: Final = 36
RUNNER_POLL_SECONDS: Final = 0.25
RUNNER_JOIN_SECONDS: Final = 2.0


class _OrphanProcessReaper:
    _instance: ClassVar[_OrphanProcessReaper | None] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self) -> None:
        self._processes: queue.Queue[Any] = queue.Queue()
        self._thread = threading.Thread(
            target=self._run,
            name="aklr-orphan-runner-reaper",
            daemon=True,
        )
        self._thread.start()

    @classmethod
    def default(cls) -> _OrphanProcessReaper:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def adopt(self, process: Any) -> None:
        self._processes.put(process)

    def _run(self) -> None:
        while True:
            process = self._processes.get()
            try:
                is_gone = ProcessGroup(process.pid).kill()
                process.kill()
                process.join(timeout=RUNNER_JOIN_SECONDS)
                if not is_gone or process.is_alive():
                    time.sleep(1.0)
                    self._processes.put(process)
            except Exception:
                # Retry orphan cleanup after any failure.
                logging.getLogger(__name__).exception("orphan runner cleanup failed")
                time.sleep(1.0)
                self._processes.put(process)


@dataclass(frozen=True)
class RunnerInvocation:
    runner: ExecutionRunner
    task_id: str

    def result(self, timeout: float | None = None) -> Any:
        return self.runner.result(self.task_id, timeout)


class ExecutionRunner(ABC):
    """Executes tasks on one leased resource, whatever the engine makes of them."""

    @property
    @abstractmethod
    def runner_id(self) -> str: ...

    @property
    @abstractmethod
    def state(self) -> RunnerState: ...

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def submit(self, task: ExecutionTask) -> RunnerInvocation: ...

    @abstractmethod
    def result(self, task_id: str, timeout: float | None = None) -> Any: ...

    @abstractmethod
    def cancel(self, task_id: str) -> bool: ...

    @abstractmethod
    def health(self) -> RunnerHealth: ...

    @abstractmethod
    def close(self) -> None: ...

    def diagnostics(self) -> str | None:
        return None


class DisposableExecutionRunner(ExecutionRunner):
    """A fresh spawned process that accepts exactly one execution task."""

    def __init__(
        self,
        lease: ResourceLease,
        engine: ExecutionEngine,
        directory: Path,
        n_resources: int,
        startup_timeout_seconds: float = DEFAULT_STARTUP_TIMEOUT_SECONDS,
    ) -> None:
        self._lease = lease
        self._engine = engine
        self._directory = directory
        self._n_resources = n_resources
        self._startup_timeout_seconds = startup_timeout_seconds
        self._runner_id = uuid.uuid4().hex
        self._state = RunnerState.CREATED
        self._health = RunnerHealth.HEALTHY
        self._state_lock = threading.Lock()
        self._process: Any | None = None
        self._channel: MultiprocessingRunnerChannel | None = None
        self._child_endpoint: ChildRunnerEndpoint | None = None
        self._submitted_task_id: str | None = None
        self._is_task_started = False
        self._cancelled_task_id: str | None = None
        self._has_read_diagnostics = False

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

    def start(self) -> None:
        self._require_state(RunnerState.CREATED)
        self._set_state(RunnerState.STARTING)

        context = multiprocessing.get_context("spawn")
        channel, child_endpoint = MultiprocessingRunnerChannel.create(context)
        process = context.Process(
            target=DisposableExecutionRunner._run_child,
            args=(
                child_endpoint,
                self._runner_id,
                self._lease.resource.devices,
                self._engine,
                self._directory,
                self._n_resources,
                os.getpid(),
            ),
            daemon=False,
        )
        self._channel = channel
        self._child_endpoint = child_endpoint
        self._process = process
        try:
            process.start()
            event = self._receive_event(self._startup_timeout_seconds)
            if isinstance(event, RunnerFailed) and event.runner_id == self._runner_id:
                self._raise_runner_failed(event, RunnerStartupFailed)
            if not isinstance(event, RunnerStarted) or event.runner_id != self._runner_id:
                raise RunnerStartupFailed(
                    f"runner {self._runner_id} sent {type(event).__name__} while starting"
                )
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
                    # E7: a resource whose child did not start leaves.
                    self._set_health(RunnerHealth.RESOURCE_FAILED)
            # Close while the failure is on its way up, and keep that failure.
            try:
                self.close()
            except ExecutionError as close_error:
                logging.getLogger(__name__).warning(
                    "runner %s did not close cleanly: %s", self._runner_id, close_error
                )
            if is_closed:
                raise TaskCancelled(f"runner {self._runner_id} was closed while starting") from error
            if isinstance(error, TaskExecutionTimeout):
                raise RunnerStartupFailed(
                    f"runner {self._runner_id} did not start within {self._startup_timeout_seconds} seconds"
                ) from error
            if isinstance(error, RunnerExitedEarly):
                raise RunnerStartupFailed(str(error)) from error
            if isinstance(error, ExecutionError):
                raise
            raise RunnerStartupFailed(
                f"runner {self._runner_id} could not start: {type(error).__name__}: {error}"
            ) from error

    def submit(self, task: ExecutionTask) -> RunnerInvocation:
        self._require_state(RunnerState.READY)
        if task.task_id != self._lease.task_id:
            raise InvalidRunnerState(f"task {task.task_id} does not own lease {self._lease.lease_id}")
        if self._submitted_task_id is not None:
            raise InvalidRunnerState(f"runner {self._runner_id} accepts only one task")

        self._submitted_task_id = task.task_id
        self._require_channel().send(
            ExecuteTask(
                runner_id=self._runner_id,
                task_id=task.task_id,
                task=task,
            )
        )
        self._set_state(RunnerState.RUNNING)
        return RunnerInvocation(runner=self, task_id=task.task_id)

    def result(self, task_id: str, timeout: float | None = None) -> Any:
        if task_id != self._submitted_task_id:
            raise InvalidRunnerState(f"runner {self._runner_id} does not own task {task_id}")
        if task_id == self._cancelled_task_id:
            raise TaskCancelled(f"task {task_id} was cancelled")
        self._require_state(RunnerState.RUNNING)

        deadline = None if timeout is None else time.monotonic() + timeout
        try:
            event = self._receive_event(timeout)
            if (
                isinstance(event, TaskStarted)
                and event.runner_id == self._runner_id
                and event.task_id == task_id
            ):
                self._is_task_started = True
                event = self._receive_event(None if deadline is None else deadline - time.monotonic())
        except (TaskExecutionTimeout, RunnerExitedEarly, ChannelFailure) as error:
            if self.state == RunnerState.CLOSED or task_id == self._cancelled_task_id:
                raise TaskCancelled(f"task {task_id} was cancelled") from error
            self._set_state(RunnerState.FAILED)
            self._set_health(RunnerHealth.FAILED)
            if isinstance(error, TaskExecutionTimeout):
                raise
            if isinstance(error, RunnerExitedEarly):
                raise RunnerCrashed(str(error)) from error
            # what the task's process wrote on the pipe is the task's (E1)
            raise TaskExecutionFailed(
                f"runner {self._runner_id} sent an invalid message for task {task_id}: {error}"
            ) from error

        if (
            isinstance(event, TaskCompleted)
            and event.runner_id == self._runner_id
            and event.task_id == task_id
        ):
            return event.result
        self._set_state(RunnerState.FAILED)
        self._set_health(RunnerHealth.FAILED)
        if isinstance(event, TaskFailed) and event.runner_id == self._runner_id and event.task_id == task_id:
            raise TaskExecutionFailed(event.reason)
        if isinstance(event, RunnerFailed) and event.runner_id == self._runner_id:
            if self._is_task_started:
                # only the task's code, sharing the child's pipe, sends one after TaskStarted
                raise TaskExecutionFailed(event.reason)
            # before the task ran: receiving its command (importing lent tensors) is the platform's
            self._raise_runner_failed(event, RunnerCommandFailed)
        if isinstance(event, RunnerExited) and event.runner_id == self._runner_id:
            raise RunnerCrashed(f"runner {self._runner_id} exited before task {task_id} completed")
        raise RunnerCrashed(f"runner {self._runner_id} sent an invalid event for task {task_id}: {event}")

    def cancel(self, task_id: str) -> bool:
        if task_id != self._submitted_task_id or self.state != RunnerState.RUNNING:
            return False
        self._cancelled_task_id = task_id
        self._set_health(RunnerHealth.FAILED)
        try:
            self._require_channel().send(CancelTask(runner_id=self._runner_id, task_id=task_id))
        except Exception:  # noqa: BLE001, S110
            # Termination below is the fallback.
            pass
        process = self._process
        if process is not None and process.pid is not None:
            try:
                ProcessGroup(process.pid).signal(signal.SIGTERM)
                process.terminate()
            except Exception:  # noqa: BLE001, S110
                # Close kills the group and records health.
                pass
        return True

    def health(self) -> RunnerHealth:
        with self._state_lock:
            return self._health

    def close(self) -> None:
        with self._state_lock:
            if self._state == RunnerState.CLOSED:
                return
            previous_state = self._state
            self._state = RunnerState.CLOSED

        process = self._process
        channel = self._channel
        has_cleanup_failed = False
        is_shutdown_sent = False
        if channel is not None and previous_state in {
            RunnerState.STARTING,
            RunnerState.READY,
            RunnerState.RUNNING,
        }:
            try:
                channel.send(
                    ShutdownRunner(
                        runner_id=self._runner_id,
                        reason="runner is closing",
                    )
                )
                is_shutdown_sent = True
            except Exception:  # noqa: BLE001
                # Failed shutdown must downgrade health.
                has_cleanup_failed = True

        is_orphaned = False
        if process is not None and process.pid is not None:
            is_gone = False
            try:
                if is_shutdown_sent and previous_state != RunnerState.STARTING:
                    # a ready child, or one whose task answered, exits by itself
                    process.join(timeout=RUNNER_JOIN_SECONDS)
                is_gone = ProcessGroup(process.pid).kill()
                process.kill()
                process.join(timeout=RUNNER_JOIN_SECONDS)
                is_gone = is_gone and not process.is_alive()
            except Exception:  # noqa: BLE001
                # A process that cannot be reaped is assumed alive.
                has_cleanup_failed = True
            if not is_gone:
                self._set_health(RunnerHealth.RESOURCE_FAILED)
                is_orphaned = True
            elif has_cleanup_failed and self.health() == RunnerHealth.HEALTHY:
                self._set_health(RunnerHealth.FAILED)

        if channel is not None:
            channel.close()
        self._child_endpoint = None
        if is_orphaned:
            _OrphanProcessReaper.default().adopt(process)
            raise RunnerCrashed(f"runner {self._runner_id}: the process group outlived its kill")

    def diagnostics(self) -> str | None:
        if self._has_read_diagnostics:
            return None
        self._has_read_diagnostics = True
        return TaskLog(self._directory / "task.log").tail() or None

    def _receive_event(self, timeout: float | None) -> RunnerEvent:
        process = self._process
        if process is None:
            raise InvalidRunnerState(f"runner {self._runner_id} has no process")
        channel = self._require_channel()
        deadline = None if timeout is None else time.monotonic() + timeout

        while True:
            wait_seconds = RUNNER_POLL_SECONDS
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    try:
                        return channel.receive_nowait()
                    except ChannelReceiveTimeout:
                        raise TaskExecutionTimeout(
                            f"runner {self._runner_id} did not answer within {timeout} seconds"
                        ) from None
                wait_seconds = min(wait_seconds, remaining)
            try:
                return channel.receive(wait_seconds)
            except ChannelReceiveTimeout:
                if process.is_alive():
                    continue
                try:
                    return channel.receive_nowait()
                except ChannelReceiveTimeout:
                    # Neutral: only the caller knows startup from mid-task.
                    raise RunnerExitedEarly(
                        f"runner {self._runner_id} exited with code {process.exitcode}"
                    ) from None

    def _require_state(self, expected: RunnerState) -> None:
        actual = self.state
        if actual != expected:
            raise InvalidRunnerState(f"runner {self._runner_id} is {actual.value}, expected {expected.value}")

    def _set_state(self, state: RunnerState) -> None:
        with self._state_lock:
            self._state = state

    def _set_health(self, health: RunnerHealth) -> None:
        with self._state_lock:
            self._health = health

    def _raise_runner_failed(self, event: RunnerFailed, error_type: type[ExecutionError]) -> NoReturn:
        self._set_health(RunnerHealth.RESOURCE_FAILED if event.resource_failed else RunnerHealth.FAILED)
        raise error_type(event.reason)

    def _require_channel(self) -> MultiprocessingRunnerChannel:
        if self._channel is None:
            raise InvalidRunnerState(f"runner {self._runner_id} has no channel")
        return self._channel

    @staticmethod
    def _run_child(
        endpoint: ChildRunnerEndpoint,
        runner_id: str,
        devices: tuple[str, ...],
        engine: ExecutionEngine,
        directory: Path,
        n_resources: int,
        parent_pid: int,
    ) -> None:
        task_id: str | None = None
        is_gpu = devices[0] != "cpu"
        try:
            os.setsid()
            libc = ctypes.CDLL(None, use_errno=True)
            # killed with the parent: a task's own children (the bench trial's candidate) end with it;
            # the signal follows the parent's thread that started this process, which the runner's
            # pool keeps until close
            libc.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL)
            # orphans of a task's processes come back here, not to init: under this group while it
            # lives, then to this runner's own runner (ProcessGroup.kill_adopted)
            libc.prctl(_PR_SET_CHILD_SUBREAPER, 1)
            if os.getppid() != parent_pid:  # the parent died before the signal was set
                return
            parent_cwd = os.getcwd()
            os.chdir(directory)
            sys.path[:] = [entry or parent_cwd for entry in sys.path]
            task_log = TaskLog(directory / "task.log")
            with task_log.redirected_stdio():
                try:
                    os.environ.update(
                        TMPDIR=str(directory),
                        OMP_NUM_THREADS=str(max(1, (os.cpu_count() or 1) // max(1, n_resources))),
                    )
                    if is_gpu:
                        # shortcut: an in-tree amdgpu has no version file, the ROCm DKMS one does;
                        # an ioctl probe of KFD's IPC would be exact
                        if not Path("/sys/module/amdgpu/version").exists():
                            raise RuntimeError(
                                "lending tensors takes KFD's legacy IPC (HSA_ENABLE_IPC_MODE_LEGACY), "
                                "which only the ROCm DKMS amdgpu driver has; this host's amdgpu is in-tree"
                            )
                        visible = ",".join(map(physical_device, devices))
                        os.environ.update(
                            HIP_VISIBLE_DEVICES=visible,
                            CUDA_VISIBLE_DEVICES=visible,
                            ROC_SIGNAL_POOL_SIZE="65536",
                            # tensors lent over KFD's own IPC handles: a dma-buf imported on the GPU it
                            # came from shares the exporter's GEM handle, which both frees delete, the
                            # second another buffer's that took the number since
                            # (amdgpu_amdkfd_gpuvm_import_dmabuf_fd); drop once KFD's import takes
                            # its own handle
                            HSA_ENABLE_IPC_MODE_LEGACY="1",
                        )
                        import torch

                        torch.cuda.init()  # a device that cannot start fails the runner, not a task
                    endpoint.send(RunnerStarted(runner_id=runner_id))

                    command = endpoint.receive()
                    if isinstance(command, ShutdownRunner):
                        return
                    if not isinstance(command, ExecuteTask) or command.runner_id != runner_id:
                        raise RuntimeError(f"runner received invalid command: {command}")

                    task_id = command.task_id
                    endpoint.send(TaskStarted(runner_id=runner_id, task_id=task_id))
                    result = engine.execute(
                        command.task,
                        engine.context_type(
                            device="cuda:0" if is_gpu else "cpu",
                            log_path=task_log.path,
                            n_devices=len(devices),
                        ),
                    )
                    # the log is complete when the parent reads its tail
                    sys.stdout.flush()
                    sys.stderr.flush()
                    endpoint.send(TaskCompleted(runner_id=runner_id, task_id=task_id, result=result))
                except BaseException as error:  # noqa: BLE001
                    # The task's, whatever it is; before started, the device's.
                    traceback.print_exc()
                    sys.stdout.flush()
                    sys.stderr.flush()
                    reason = f"{type(error).__name__}: {error}\n\n{traceback.format_exc()}"
                    try:
                        endpoint.send(
                            TaskFailed(runner_id=runner_id, task_id=task_id, reason=reason)
                            if task_id is not None
                            else RunnerFailed(
                                runner_id=runner_id,
                                reason=reason,
                                resource_failed=is_gpu,
                            )
                        )
                    except Exception:  # noqa: BLE001, S110
                        # Parent also observes process exit.
                        pass
        finally:
            try:
                endpoint.send(RunnerExited(runner_id=runner_id, task_id=task_id, exit_code=0))
            except Exception:  # noqa: BLE001, S110
                # Exit notification is best effort.
                pass


class RunnerFactory(ABC):
    """Creates a runner for an exclusive resource lease."""

    @abstractmethod
    def create(
        self,
        lease: ResourceLease,
        engine: ExecutionEngine,
    ) -> ExecutionRunner: ...

    @abstractmethod
    def close(self) -> None: ...


class DisposableRunnerFactory(RunnerFactory):
    def __init__(
        self,
        n_resources: int,
        startup_timeout_seconds: float = DEFAULT_STARTUP_TIMEOUT_SECONDS,
    ) -> None:
        self._n_resources = n_resources
        self._startup_timeout_seconds = startup_timeout_seconds
        self._job_dir = Path(tempfile.mkdtemp(prefix="aklr-validation-"))
        self._task_numbers = itertools.count()
        self._lock = threading.Lock()

    def create(
        self,
        lease: ResourceLease,
        engine: ExecutionEngine,
    ) -> ExecutionRunner:
        with self._lock:
            task_number = next(self._task_numbers)
        directory = self._job_dir / f"{task_number}-{lease.resource.resource_id.replace(':', '-')}"
        directory.mkdir()
        return DisposableExecutionRunner(
            lease=lease,
            engine=engine,
            directory=directory,
            n_resources=self._n_resources,
            startup_timeout_seconds=self._startup_timeout_seconds,
        )

    def close(self) -> None:
        shutil.rmtree(self._job_dir, ignore_errors=True)
