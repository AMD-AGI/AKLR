###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

import contextlib
import os
import sys
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

MAX_LOG_BYTES: Final = 5 << 20
LOG_TAIL_BYTES: Final = 8 << 10


@dataclass(frozen=True)
class TaskLog:
    path: Path

    @contextlib.contextmanager
    def redirected_stdio(self) -> Iterator[None]:
        sys.stdout.flush()
        sys.stderr.flush()
        redirected_fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        saved_stdout_fd, saved_stderr_fd = os.dup(1), os.dup(2)
        try:
            os.dup2(redirected_fd, 1)
            os.dup2(redirected_fd, 2)
            os.close(redirected_fd)
            # the cap thread truncates fd 1, so it runs only while fd 1 is the log
            stop = threading.Event()
            cap_thread = threading.Thread(target=self._cap, args=(stop,), name="aklr-log-cap", daemon=True)
            cap_thread.start()
            try:
                yield
            finally:
                stop.set()
                cap_thread.join(timeout=0.2)
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved_stdout_fd, 1)
            os.dup2(saved_stderr_fd, 2)
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)

    def tail(self, n_bytes: int = LOG_TAIL_BYTES) -> str:
        try:
            with open(self.path, "rb") as file:
                file.seek(max(0, file.seek(0, os.SEEK_END) - n_bytes))
                return file.read().decode(errors="replace")
        except OSError:
            return ""

    @staticmethod
    def _cap(stop: threading.Event) -> None:
        while not stop.wait(0.05):
            try:
                if os.fstat(1).st_size <= MAX_LOG_BYTES:
                    continue
                os.ftruncate(1, MAX_LOG_BYTES)
                os.lseek(1, MAX_LOG_BYTES, os.SEEK_SET)  # else the next write leaves a hole of NUL bytes
            except OSError:
                return
