# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Print and start server commands, wait for readiness, and clean up their process groups."""

from __future__ import annotations

import contextlib
import http.client
import os
import shlex
import signal
import subprocess
import time
import urllib.request
from pathlib import Path


class RunNotKilledError(RuntimeError):
    """A process group outlived its ``SIGKILL`` and still holds the devices."""


class NodeProcess:
    """One process on one node, in a process group of its own, logging to ``log_path``."""

    def __init__(self, name: str, argv: tuple[str, ...], log_path: Path) -> None:
        self.name = name
        self.argv = argv
        self.log_path = log_path
        self._process: subprocess.Popen[bytes] | None = None
        self._kill_wait_seconds = 5.0
        self._log_tail_bytes = 2 << 10

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "wb") as log:
            self._process = subprocess.Popen(
                self.argv, start_new_session=True, stdin=subprocess.DEVNULL, stdout=log, stderr=log
            )

    @property
    def exit_code(self) -> int | None:
        return None if self._process is None else self._process.poll()

    def stop(self) -> str:
        """Kill the process group and return the log's tail; a group that outlives its kill
        raises ``RunNotKilledError``."""
        if self._process is None:
            return ""
        group = self._process.pid
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(group, signal.SIGKILL)
        deadline = time.monotonic() + self._kill_wait_seconds
        while True:
            # a zombie (an orphan no init reaps, as in a container) holds nothing
            is_alive = False
            for stat_path in Path("/proc").glob("[0-9]*/stat"):
                try:
                    state, _, process_group = stat_path.read_text().rpartition(")")[2].split()[:3]
                except (OSError, ValueError):
                    continue
                if int(process_group) == group and state != "Z":
                    is_alive = True
                    break
            if not is_alive or time.monotonic() >= deadline:
                break
            time.sleep(0.01)
        if not is_alive:  # a leader stuck past its SIGKILL never exits
            self._process.wait()
        try:
            with open(self.log_path, "rb") as log:
                log.seek(max(0, log.seek(0, os.SEEK_END) - self._log_tail_bytes))
                log_tail = log.read().decode(errors="replace")
        except OSError:
            log_tail = ""
        if is_alive:
            raise RunNotKilledError(f"{self.name}: the process group outlived its kill\n{log_tail}")
        return log_tail


class Deployment:
    """Run the commands for one scenario and AKLR mode."""

    def __init__(self, commands: dict[str, tuple[str, ...]], entry_url: str, run_dir: Path) -> None:
        self.processes = tuple(
            NodeProcess(name, argv, run_dir / f"{name}.log") for name, argv in commands.items()
        )
        self.entry_url = entry_url
        self._ready_poll_seconds = 5.0
        self._health_timeout_seconds = 60.0

    @property
    def argvs(self) -> tuple[str, ...]:
        """Every command line, for the report."""
        return tuple(shlex.join(p.argv) for p in self.processes)

    def start(self, timeout: float) -> str:
        """Start every process and return the entry URL once ``/health_generate`` answers;
        an exit or ``timeout`` seconds without an answer raises ``RuntimeError``."""
        deadline = time.monotonic() + timeout
        for process in self.processes:
            print(f"Command ({process.name}):", flush=True)
            print(shlex.join(process.argv), flush=True)
            process.start()
        while (remaining := deadline - time.monotonic()) > 0:
            if exited := self.exited():
                raise RuntimeError(f"{', '.join(exited)} before it was ready")
            with contextlib.suppress(OSError, http.client.HTTPException):
                url = f"{self.entry_url}/health_generate"
                with urllib.request.urlopen(
                    url, timeout=min(self._health_timeout_seconds, remaining)
                ) as response:
                    # a server that answers only after the deadline is not ready in time
                    if response.status == 200 and time.monotonic() < deadline:
                        return self.entry_url
            time.sleep(min(self._ready_poll_seconds, max(0.0, deadline - time.monotonic())))
        raise RuntimeError(f"{self.entry_url} not ready after {timeout:g} s")

    def exited(self) -> list[str]:
        """Each process that has exited, and its exit code."""
        return [f"{p.name} exited with {p.exit_code}" for p in self.processes if p.exit_code is not None]

    def stop(self) -> str:
        """Stop every process and return their log tails; ``RunNotKilledError`` once all
        are stopped, if one outlived its kill."""
        tails: list[str] = []
        not_killed: list[RunNotKilledError] = []
        for process in self.processes:
            try:
                tails.append(f"== {process.name}\n{process.stop()}")
            except RunNotKilledError as error:
                not_killed.append(error)
        if not_killed:
            raise RunNotKilledError("\n".join(map(str, not_killed)))
        return "\n".join(tails)
