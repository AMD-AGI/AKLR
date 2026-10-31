# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import os
from multiprocessing.connection import Connection

import pytest

from aklr.core.validation.stages._bench._execution import (
    ExecutionError,
    Executor,
    RunnerCrashed,
    TaskExecutionFailed,
)


def _exit(context: object) -> None:
    os._exit(3)


def _exit_while_answering(context: object) -> str:
    send = Connection._send

    def send_half(self: Connection, buf: bytes, *args: object) -> None:
        send(self, bytes(buf[: len(buf) // 2]), *args)
        os._exit(3)

    Connection._send = send_half
    return "answer"


class TestExecutor:
    @pytest.mark.parametrize(
        ("task", "error", "match"),
        [
            pytest.param(_exit, RunnerCrashed, "exited with code 3", id="exit"),
            pytest.param(
                _exit_while_answering,
                TaskExecutionFailed,
                "end of file",
                id="exit_while_answering",
            ),
        ],
    )
    def test_a_task_whose_process_exits_fails(
        self, task: object, error: type[ExecutionError], match: str
    ) -> None:
        with Executor.for_devices(["cpu"]) as executor:
            future = executor.submit(task, timeout_seconds=60)
            with pytest.raises(error, match=match):
                future.result(timeout=120)
