###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

import contextlib
import dataclasses
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from multiprocessing.reduction import ForkingPickler
from typing import Any, Final

from aklr.core.validation.stages._bench._execution._errors import ChannelFailure
from aklr.core.validation.stages._bench._execution._model import ExecutionTask

MAX_MESSAGE_BYTES: Final = 16 << 20


@dataclass(frozen=True)
class ExecuteTask:
    runner_id: str
    task_id: str
    task: ExecutionTask


@dataclass(frozen=True)
class CancelTask:
    runner_id: str
    task_id: str


@dataclass(frozen=True)
class ShutdownRunner:
    runner_id: str
    reason: str


RunnerCommand = ExecuteTask | CancelTask | ShutdownRunner


@dataclass(frozen=True)
class RunnerStarted:
    runner_id: str


@dataclass(frozen=True)
class RunnerFailed:
    runner_id: str
    reason: str
    resource_failed: bool


@dataclass(frozen=True)
class TaskCompleted:
    runner_id: str
    task_id: str
    result: Any


@dataclass(frozen=True)
class TaskFailed:
    runner_id: str
    task_id: str
    reason: str


@dataclass(frozen=True)
class RunnerExited:
    runner_id: str
    task_id: str | None
    exit_code: int | None


RunnerEvent = RunnerStarted | RunnerFailed | TaskCompleted | TaskFailed | RunnerExited


class ChannelReceiveTimeout(TimeoutError):
    pass


class RunnerChannel(ABC):
    @abstractmethod
    def send(self, command: RunnerCommand) -> None: ...

    @abstractmethod
    def receive(self, timeout: float | None = None) -> RunnerEvent: ...

    @abstractmethod
    def receive_nowait(self) -> RunnerEvent: ...

    @abstractmethod
    def close(self) -> None: ...


class ChildRunnerEndpoint:
    """The child's end: pickled commands in, JSON events out, so the parent never unpickles its bytes."""

    def __init__(self, command_queue: Any, event_connection: Any) -> None:
        self._command_queue = command_queue
        self._event_connection = event_connection

    def receive(self) -> RunnerCommand:
        try:
            return self._command_queue.get()
        except Exception as error:
            raise ChannelFailure(f"could not receive runner command: {error}") from error

    def send(self, event: RunnerEvent) -> None:
        try:
            message = json.dumps(
                {
                    "kind": type(event).__name__,
                    **{
                        event_field.name: getattr(event, event_field.name)
                        for event_field in dataclasses.fields(event)
                    },
                }
            ).encode()
        except (TypeError, ValueError, RecursionError) as error:
            raise ChannelFailure(f"could not send runner event: {error}") from error
        if len(message) > MAX_MESSAGE_BYTES:
            raise ChannelFailure(
                f"could not send runner event: {len(message)} bytes, more than {MAX_MESSAGE_BYTES}"
            )
        try:
            self._event_connection.send_bytes(message)
        except Exception as error:
            raise ChannelFailure(f"could not send runner event: {error}") from error

    def close(self) -> None:
        self._event_connection.close()


class MultiprocessingRunnerChannel(RunnerChannel):
    def __init__(self, command_queue: Any, event_connection: Any) -> None:
        self._command_queue = command_queue
        self._event_connection = event_connection
        self._is_closed = False

    @classmethod
    def create(cls, context: Any) -> tuple[MultiprocessingRunnerChannel, ChildRunnerEndpoint]:
        command_queue = context.Queue()
        event_reader, event_writer = context.Pipe(duplex=False)
        return cls(command_queue, event_reader), ChildRunnerEndpoint(command_queue, event_writer)

    def send(self, command: RunnerCommand) -> None:
        if self._is_closed:
            raise ChannelFailure("runner channel is closed")
        try:
            ForkingPickler.dumps(command)  # put() pickles on a feeder thread, which only prints its failure
            self._command_queue.put(command)
        except Exception as error:
            raise ChannelFailure(f"could not send runner command: {error}") from error

    def receive(self, timeout: float | None = None) -> RunnerEvent:
        if self._is_closed:
            raise ChannelFailure("runner channel is closed")
        try:
            if not self._event_connection.poll(timeout):
                raise ChannelReceiveTimeout
            message = json.loads(self._event_connection.recv_bytes(MAX_MESSAGE_BYTES))
            if not isinstance(message, dict):
                raise ValueError(  # noqa: TRY004
                    f"not a json object: {type(message).__name__}"
                )
            kind = message.pop("kind", None)
            event_type = (
                {
                    known_type.__name__: known_type
                    for known_type in (
                        RunnerStarted,
                        RunnerFailed,
                        TaskCompleted,
                        TaskFailed,
                        RunnerExited,
                    )
                }.get(kind)
                if isinstance(kind, str)
                else None
            )
            if event_type is None:
                raise ValueError(f"not an event kind: {kind!r}")
            field_names = {event_field.name for event_field in dataclasses.fields(event_type)}
            if message.keys() != field_names:
                raise ValueError(f"{kind} has fields {sorted(field_names)}, not {sorted(message)}")
            return event_type(**message)
        except ChannelReceiveTimeout:
            raise
        except (OSError, ValueError, RecursionError) as error:
            raise ChannelFailure(f"could not receive runner event: {error}") from error

    def receive_nowait(self) -> RunnerEvent:
        return self.receive(timeout=0)

    def close(self) -> None:
        if self._is_closed:
            return
        self._is_closed = True
        try:
            # a dead child never drains the pipe: joining the queue's feeder thread at exit would hang
            self._command_queue.cancel_join_thread()
            self._command_queue.close()
        except Exception:  # noqa: BLE001, S110
            pass
        with contextlib.suppress(OSError):
            self._event_connection.close()
