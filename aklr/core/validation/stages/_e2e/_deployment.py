# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Print and start server commands, wait for readiness, and clean up their process groups."""

from __future__ import annotations

import contextlib
import http.client
import os
import shlex
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path

from aklr.core.validation.stages._e2e._worker import stop_group


class RunNotKilledError(RuntimeError):
    """A process group outlived its ``SIGKILL`` and still holds the devices."""


class NodeProcess:
    """One process on one node, in a process group of its own, logging to ``log_path``."""

    def __init__(self, name: str, argv: tuple[str, ...], log_path: Path) -> None:
        self.name = name
        self.log_path = log_path
        self._remote_host = argv[2] if argv[0] == "ssh" else None
        self._worker = (sys.executable, str(Path(__file__).with_name("_worker.py").resolve()))
        self._remote_state = log_path.with_name(f"{name}-{uuid.uuid4().hex}.pid")
        command = shlex.split(argv[3]) if self._remote_host else list(argv)
        self._host = command[command.index("--host") + 1]
        self._port = command[command.index("--port") + 1]
        self.argv = (
            self._on_host((*self._worker, "start", str(self._remote_state), *command))
            if self._remote_host
            else argv
        )
        self._process: subprocess.Popen[bytes] | None = None
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
        failures = []
        if self._remote_host:
            try:
                # Shared state prevents a delayed start even if SSH cannot reach the host.
                self._remote_state.with_suffix(".stopped").touch()
                self._control("stop", str(self._remote_state))
            except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                failures.append(f"remote cleanup was not verified: {error}")
        try:
            stop_group(self._process.pid)
            self._process.wait()
        except (OSError, RuntimeError) as error:
            failures.append(str(error))
        try:
            with open(self.log_path, "rb") as log:
                log.seek(max(0, log.seek(0, os.SEEK_END) - self._log_tail_bytes))
                log_tail = log.read().decode(errors="replace")
        except OSError:
            log_tail = ""
        if failures:
            raise RunNotKilledError(f"{self.name}: {'; '.join(failures)}\n{log_tail}")
        return log_tail

    def _on_host(self, argv: tuple[str, ...]) -> tuple[str, ...]:
        return (
            (
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=10",
                "--",
                self._remote_host,
                shlex.join(argv),
            )
            if self._remote_host
            else argv
        )

    def _control(self, *args: str) -> None:
        try:
            done = subprocess.run(
                self._on_host((*self._worker, *args)),
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise RuntimeError(f"{self.name}: {args[0]} could not be verified: {error}") from error
        if done.returncode:
            raise RuntimeError(f"{self.name}: {' '.join(args)}: {done.stderr.strip()}")

    def check_available(self) -> None:
        self._control("available", self._host, self._port)

    def check_owner(self) -> None:
        assert self._process is not None
        group = str(self._remote_state) if self._remote_host else str(self._process.pid)
        self._control("owns-port", group, self._port)


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
            process.check_available()
        for process in self.processes:
            if time.monotonic() >= deadline:
                raise RuntimeError(f"{self.entry_url} startup deadline exceeded")
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
                        for process in self.processes:
                            if process.name == "router" or process.name.endswith("-0"):
                                process.check_owner()
                        if exited := self.exited():
                            raise RuntimeError(f"{', '.join(exited)} during readiness check")
                        if time.monotonic() >= deadline:
                            raise RuntimeError(f"{self.entry_url} startup deadline exceeded")
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
