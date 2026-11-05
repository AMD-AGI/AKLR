# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""One deployment of a scenario's side: its server processes over nodes, and a PD router."""

from __future__ import annotations

import contextlib
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from aklr.core.validation.stages._e2e._config import UNIFIED_ROLE, Scenario
from aklr.core.validation.stages._e2e._cookbook import Cookbook

__all__ = ["Deployment", "NodeProcess", "RunNotKilledError"]

_SERVER_PORT: Final = 30000
_ROUTER_PORT: Final = 30080
# the cookbook's multi-node convention: --dist-init-addr <node 0>:20000
_DIST_INIT_PORT: Final = 20000
_DEFAULT_BOOTSTRAP_PORT: Final = "8998"
# the flags the launcher sets on every process itself
_LAUNCHER_FLAGS: Final = frozenset({"--host", "--port", "--nnodes", "--node-rank", "--dist-init-addr"})
_KNOWN_PLACEHOLDER: Final = re.compile(r"\{\{(\w+)\}\}")
_PLACEHOLDER: Final = re.compile(r"\{\{\w+\}\}|<[a-z0-9][a-z0-9-]*>")
_READY_POLL_SECONDS: Final = 5.0
_LOG_TAIL_BYTES: Final = 2 << 10
_KILL_WAIT_SECONDS: Final = 5.0


class RunNotKilledError(RuntimeError):
    """A process group outlived its ``SIGKILL`` and still holds the devices."""


@dataclass
class NodeProcess:
    """One process on one node, in a process group of its own, logging to ``log_path``."""

    name: str
    argv: tuple[str, ...]
    log_path: Path
    _process: subprocess.Popen[bytes] | None = field(default=None, repr=False)

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
        deadline = time.monotonic() + _KILL_WAIT_SECONDS
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
                log.seek(max(0, log.seek(0, os.SEEK_END) - _LOG_TAIL_BYTES))
                log_tail = log.read().decode(errors="replace")
        except OSError:
            log_tail = ""
        if is_alive:
            raise RunNotKilledError(f"{self.name}: the process group outlived its kill\n{log_tail}")
        return log_tail


@dataclass(frozen=True)
class Deployment:
    """The processes of one side (AKLR on or off) of a scenario, and where to send requests."""

    processes: tuple[NodeProcess, ...]
    entry_url: str

    @classmethod
    def plan(
        cls,
        scenario: Scenario,
        cookbook: Cookbook,
        *,
        hosts: Sequence[str],
        launch: str | None,
        env: Mapping[str, str],
        side_env: Mapping[str, str],
        run_dir: Path,
    ) -> Deployment:
        """Each role's processes from its cell with the scenario's overrides, over ``hosts``
        in order, the router on node 0 (the cell's env, then the scenario's, ``env`` and
        ``side_env``); ``ValueError`` when nodes are short or a placeholder is left."""
        cells = {role: cookbook.cell(match) for role, match in scenario.roles.items()}
        if (n_nodes := sum(c.nnodes for c in cells.values())) > len(hosts):
            raise ValueError(f"{scenario.name}: needs {n_nodes} nodes, --e2e-hosts gives {len(hosts)}")
        local_names = {"localhost", "127.0.0.1", socket.gethostname(), socket.getfqdn()}
        role_hosts: dict[str, Sequence[str]] = {}
        for role, cell in cells.items():
            n_assigned = sum(map(len, role_hosts.values()))
            role_hosts[role] = hosts[n_assigned : n_assigned + cell.nnodes]
        processes: list[NodeProcess] = []
        bootstrap_port = _DEFAULT_BOOTSTRAP_PORT
        for role, cell in cells.items():
            nodes = role_hosts[role]
            # the stage's and the side's last: a cookbook commit cannot redirect the devices,
            # PYTHONPATH or AKLR's switches
            role_env = (
                dict(e.split("=", 1) for e in cell.env) | scenario.env[role] | dict(env) | dict(side_env)
            )
            flags = [shlex.split(f) for f in cell.flags]
            overrides = {"--revision": scenario.revision} if scenario.revision else {}
            for name, value in (overrides | scenario.args[role]).items():
                flags = [f for f in flags if f[0].split("=", 1)[0] != name]
                if value is True:
                    flags.append([name])
                elif value is not None and value is not False:
                    flags.append([name, str(value)])
            flags = [f for f in flags if f[0].split("=", 1)[0] not in _LAUNCHER_FLAGS]
            for f in flags:
                if f[0].split("=", 1)[0] == "--disaggregation-bootstrap-port":
                    bootstrap_port = f[0].split("=", 1)[1] if "=" in f[0] else f[1]
            for rank, host in enumerate(nodes):
                node_flags = [*flags, ["--host", host], ["--port", str(_SERVER_PORT)]]
                if cell.nnodes > 1:
                    node_flags += [
                        ["--nnodes", str(cell.nnodes)],
                        ["--node-rank", str(rank)],
                        ["--dist-init-addr", f"{nodes[0]}:{_DIST_INIT_PORT}"],
                    ]
                argv = [
                    *(f"{k}={v}" for k, v in role_env.items()),
                    sys.executable, "-m", "sglang.launch_server", *(t for f in node_flags for t in f),
                ]  # fmt: skip
                values = {
                    "MODEL_NAME": scenario.model_path, "HOST_IP": host, "LOCAL_IP": host,
                    "PORT": str(_SERVER_PORT), "NODE0_IP": nodes[0], "NODE_RANK": str(rank),
                    "PREFILL_HEAD_IP": role_hosts.get("prefill", nodes)[0],
                }  # fmt: skip
                argv = [
                    _KNOWN_PLACEHOLDER.sub(lambda m, values=values: values.get(m[1], m[0]), t) for t in argv
                ]
                processes.append(
                    cls._node_process(f"{role}-{rank}", host, argv, launch, local_names, run_dir)
                )
        if UNIFIED_ROLE in role_hosts:
            entry_url = f"http://{role_hosts[UNIFIED_ROLE][0]}:{_SERVER_PORT}"
        else:
            prefill, decode = role_hosts["prefill"][0], role_hosts["decode"][0]
            router_env = dict(env) | dict(side_env)
            argv = [
                *(f"{k}={v}" for k, v in router_env.items()),
                sys.executable, "-m", "sglang_router.launch_router", "--pd-disaggregation",
                "--prefill", f"http://{prefill}:{_SERVER_PORT}", bootstrap_port,
                "--decode", f"http://{decode}:{_SERVER_PORT}",
                "--host", hosts[0], "--port", str(_ROUTER_PORT),
            ]  # fmt: skip
            processes.append(cls._node_process("router", hosts[0], argv, launch, local_names, run_dir))
            entry_url = f"http://{hosts[0]}:{_ROUTER_PORT}"
        for process in processes:
            if left := sorted({m for token in process.argv for m in _PLACEHOLDER.findall(token)}):
                raise ValueError(
                    f"{scenario.name}: {process.name}: placeholders {left}; set them in env or args"
                )
        return cls(processes=tuple(processes), entry_url=entry_url)

    @staticmethod
    def _node_process(
        name: str, host: str, argv: list[str], launch: str | None, local_names: set[str], run_dir: Path
    ) -> NodeProcess:
        """``env K=V ... argv`` on ``host``: a node other than this one through ``launch``."""
        prefix: list[str] = []
        if host not in local_names:
            if launch is None:
                raise ValueError(f"{name}: {host} is not this node, and no --e2e-launch reaches it")
            # shortcut: ssh joins its argv with spaces, so a value with a space splits there;
            # quote per launcher once a recipe needs one
            prefix = shlex.split(launch.format(host=host))
        return NodeProcess(name=name, argv=(*prefix, "env", *argv), log_path=run_dir / f"{name}.log")

    @property
    def argvs(self) -> tuple[str, ...]:
        """Every command line, for the report."""
        return tuple(shlex.join(p.argv) for p in self.processes)

    def start(self, timeout: float) -> str:
        """Start every process and return the entry URL once ``/health_generate`` answers;
        an exit or ``timeout`` seconds without an answer raises ``RuntimeError``."""
        deadline = time.monotonic() + timeout
        for process in self.processes:
            process.start()
        while True:
            for process in self.processes:
                if process.exit_code is not None:
                    raise RuntimeError(f"{process.name} exited with {process.exit_code} before it was ready")
            with contextlib.suppress(OSError):
                with urllib.request.urlopen(f"{self.entry_url}/health_generate", timeout=60) as response:
                    if response.status == 200:
                        return self.entry_url
            if time.monotonic() >= deadline:
                raise RuntimeError(f"{self.entry_url} not ready after {timeout:g} s")
            time.sleep(_READY_POLL_SECONDS)

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
