# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""One deployment of a scenario's side: its server processes over nodes, and a PD router."""

from __future__ import annotations

import contextlib
import http.client
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

from aklr.core.validation.stages._e2e._config import UNIFIED_ROLE, ArgValue, Scenario
from aklr.core.validation.stages._e2e._cookbook import Cookbook, CookbookCell

__all__ = ["Deployment", "NodeProcess", "Role", "RunNotKilledError"]

_SERVER_PORT: Final = 30000
_ROUTER_PORT: Final = 30080
# the cookbook's multi-node convention: --dist-init-addr <node 0>:20000
_DIST_INIT_PORT: Final = 20000
_DEFAULT_BOOTSTRAP_PORT: Final = "8998"
# the flags the launcher sets on every process itself
_LAUNCHER_FLAGS: Final = frozenset({"--host", "--port", "--nnodes", "--node-rank", "--dist-init-addr"})
_ENV_NAME: Final = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# what a cookbook's env may not set: the loader's, the interpreter's and the shell's own hooks
# run code before sglang does (LD_PRELOAD, PYTHONWARNINGS naming a module to import, BASH_ENV,
# PATH, a profiler library), Hugging Face's, ModelScope's, the caches', the certificates' and
# the proxies' settings redirect what is downloaded or loaded, and AKLR's are the stage's
_UNSAFE_ENV_NAME: Final = re.compile(
    r"LD_\w*|DYLD_\w*|PYTHON\w*|PATH|HOME|ENV|BASH_ENV|SHELL|IFS|GCONV_PATH|LOCPATH|NLSPATH|TMPDIR"
    r"|HF_\w*|HUGGINGFACE_\w*|\w*MODELSCOPE\w*|TRANSFORMERS_\w*|TORCH_HOME|XDG_\w*|TRITON_\w*"
    r"|AKLR_\w*|HSA_TOOLS_LIB|ROCP_\w*|OPENSSL_CONF|SSL_CERT_\w*|REQUESTS_CA_BUNDLE|CURL_CA_BUNDLE"
    r"|(?i:\w*_proxy)"
)
_KNOWN_PLACEHOLDER: Final = re.compile(r"\{\{\s*(\w+)\s*\}\}")
_PLACEHOLDER: Final = re.compile(r"\{\{.*?\}\}|<[A-Za-z][\w-]*>")
_READY_POLL_SECONDS: Final = 5.0
_HEALTH_TIMEOUT_SECONDS: Final = 60.0
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
class Role:
    """One role of a deployment: its cookbook cell with the scenario's overrides."""

    name: str
    cell: CookbookCell
    env: Mapping[str, str]  # the scenario's, over the cell's
    args: Mapping[str, ArgValue]  # the scenario's, over the cell's flags
    nnodes: int

    @classmethod
    def of(cls, scenario: Scenario, cookbook: Cookbook) -> tuple[Role, ...]:
        """The scenario's roles, in order; the launcher owns the weights (``--model-path``
        ``--tokenizer-path`` and ``--revision`` from the scenario, whatever the cell says), which
        only the scenario's ``args`` override."""
        # set, not only removed, and after the cell's flags: argparse takes the last value, and
        # also reads a prefix (--tokenizer-p) as the flag; a tokenizer of another repository
        # would run that repository's remote code
        weights = {
            "--model-path": scenario.model_path, "--tokenizer-path": scenario.model_path,
            "--model": None, "--tokenizer": None,
        } | ({"--revision": scenario.revision} if scenario.revision else {})  # fmt: skip
        roles = []
        for name, match in scenario.roles.items():
            cell = cookbook.cell(match)
            roles.append(
                cls(
                    name=name,
                    cell=cell,
                    env=scenario.env[name],
                    args=weights | scenario.args[name],
                    nnodes=scenario.nnodes.get(name, cell.nnodes),
                )
            )
        return tuple(roles)

    def server_env(self) -> dict[str, str]:
        """The cell's env, then the scenario's; ``ValueError`` for a name ``env`` would read
        as an option (-S runs a command line) or a name the cookbook may not set."""
        env: dict[str, str] = {}
        for entry in self.cell.env:
            # shell syntax: GLOO_SOCKET_IFNAME="<network-interface>"
            (token,) = shlex.split(entry) or [""]
            name, has_value, value = token.partition("=")
            if not has_value or _UNSAFE_ENV_NAME.fullmatch(name):
                raise ValueError(f"{self.name}: the cookbook's env {entry!r} is not allowed")
            env[name] = value
        env |= self.env
        if bad := [name for name in env if not _ENV_NAME.fullmatch(name)]:
            raise ValueError(f"{self.name}: not environment variable names: {bad}")
        return env

    def flags(self) -> list[list[str]]:
        """The cell's flags, each a name and its values, with ``args`` applied (replaced by
        name, null or false removes, true is a switch) and the launcher's own removed."""
        flags = [shlex.split(f) for f in self.cell.flags]
        if not all(flags):
            raise ValueError(f"{self.name}: an empty flag in the cookbook")
        for name, value in self.args.items():
            flags = [f for f in flags if f[0].split("=", 1)[0] != name]
            if value is True:
                flags.append([name])
            elif value is not None and value is not False:
                flags.append([name, str(value)])
        return [f for f in flags if f[0].split("=", 1)[0] not in _LAUNCHER_FLAGS]

    @property
    def bootstrap_port(self) -> str:
        """The ``--disaggregation-bootstrap-port`` a prefill role serves its KV on."""
        for name, *values in self.flags():
            flag, has_value, value = name.partition("=")
            if flag == "--disaggregation-bootstrap-port":
                if not (has_value or values):
                    raise ValueError(f"{self.name}: --disaggregation-bootstrap-port has no value")
                return value if has_value else values[0]
        return _DEFAULT_BOOTSTRAP_PORT


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
        """Each role's processes over ``hosts`` in order, the router on node 0; ``env`` and
        ``side_env`` over each role's own, so a cookbook cannot redirect the devices,
        PYTHONPATH or AKLR's switches. ``ValueError`` when nodes are short, a cookbook entry is
        not allowed or a placeholder is left."""
        roles = Role.of(scenario, cookbook)
        if (n_nodes := sum(r.nnodes for r in roles)) > len(hosts):
            raise ValueError(f"{scenario.name}: needs {n_nodes} nodes, --e2e-hosts gives {len(hosts)}")
        local_names = {"localhost", "127.0.0.1", socket.gethostname(), socket.getfqdn()}
        role_hosts: dict[str, Sequence[str]] = {}
        for role in roles:
            n_assigned = sum(map(len, role_hosts.values()))
            role_hosts[role.name] = hosts[n_assigned : n_assigned + role.nnodes]
        processes: list[NodeProcess] = []
        for role in roles:
            nodes = role_hosts[role.name]
            role_env = role.server_env() | dict(env) | dict(side_env)
            flags = role.flags()
            for rank, host in enumerate(nodes):
                node_flags = [*flags, ["--host", host], ["--port", str(_SERVER_PORT)]]
                if role.nnodes > 1:
                    node_flags += [
                        ["--nnodes", str(role.nnodes)],
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
                    cls._node_process(f"{role.name}-{rank}", host, argv, launch, local_names, run_dir)
                )
        if UNIFIED_ROLE in role_hosts:
            entry_url = f"http://{role_hosts[UNIFIED_ROLE][0]}:{_SERVER_PORT}"
        else:
            prefill, decode = role_hosts["prefill"][0], role_hosts["decode"][0]
            argv = [
                *(f"{k}={v}" for k, v in (dict(env) | dict(side_env)).items()),
                sys.executable, "-m", "sglang_router.launch_router", "--pd-disaggregation",
                "--prefill", f"http://{prefill}:{_SERVER_PORT}", roles[0].bootstrap_port,
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
        """``env K=V ... argv`` on ``host``; on another node, through ``launch`` with ``{cmd}``
        replaced by the whole command as one shell-quoted word (``ssh {host} {cmd}``, or
        ``srun -w {host} --overlap sh -c {cmd}``), so a remote shell parses it back unchanged."""
        command = ["env", *argv]
        if host in local_names:
            return NodeProcess(name=name, argv=tuple(command), log_path=run_dir / f"{name}.log")
        if launch is None:
            raise ValueError(f"{name}: {host} is not this node, and no --e2e-launch reaches it")
        prefix = [shlex.join(command) if t == "{cmd}" else t.format(host=host) for t in shlex.split(launch)]
        if prefix.count(shlex.join(command)) != 1:
            raise ValueError(f"--e2e-launch {launch!r}: needs one {{cmd}} word")
        return NodeProcess(name=name, argv=tuple(prefix), log_path=run_dir / f"{name}.log")

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
        while (remaining := deadline - time.monotonic()) > 0:
            if exited := self.exited():
                raise RuntimeError(f"{', '.join(exited)} before it was ready")
            with contextlib.suppress(OSError, http.client.HTTPException):
                url = f"{self.entry_url}/health_generate"
                with urllib.request.urlopen(url, timeout=min(_HEALTH_TIMEOUT_SECONDS, remaining)) as response:
                    if response.status == 200:
                        return self.entry_url
            time.sleep(min(_READY_POLL_SECONDS, max(0.0, deadline - time.monotonic())))
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
