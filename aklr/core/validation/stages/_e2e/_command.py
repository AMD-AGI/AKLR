# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Build cookbook commands for direct execution by the e2e stage."""

from __future__ import annotations

import re
import shlex
import socket
import sys
from collections.abc import Mapping, Sequence

from aklr.core.validation.stages._e2e._config import Scenario
from aklr.core.validation.stages._e2e._cookbook import Cookbook, CookbookCell


class LaunchArgumentRules:
    """Validate cookbook environment entries and resolve launch placeholders."""

    def __init__(self) -> None:
        self._env_name = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
        # Require scenario overrides for execution hooks, download settings and AKLR/device controls.
        self._unsafe_env_name = re.compile(
            r"LD_\w*|DYLD_\w*|PYTHON\w*|PATH|HOME|ENV|BASH_ENV|SHELL|IFS|GCONV_PATH|LOCPATH|NLSPATH|TMPDIR"
            r"|HF_\w*|HUGGINGFACE_\w*|\w*MODELSCOPE\w*|TRANSFORMERS_\w*|TORCH_HOME|XDG_\w*|TRITON_\w*"
            r"|ROCR_VISIBLE_DEVICES|HIP_VISIBLE_DEVICES|CUDA_VISIBLE_DEVICES|GPU_DEVICE_ORDINAL"
            r"|AKLR_\w*|HSA_TOOLS_LIB|ROCP_\w*|OPENSSL_CONF|SSL_CERT_\w*|REQUESTS_CA_BUNDLE|CURL_CA_BUNDLE"
            r"|(?i:\w*_proxy)"
        )
        self._known_placeholder = re.compile(r"\{\{\s*(\w+)\s*\}\}")
        self._placeholder = re.compile(r"\{\{.*?\}\}|<[A-Za-z][\w.:-]*>")

    def environment(self, role: str, entries: Sequence[str], overrides: Mapping[str, str]) -> dict[str, str]:
        """Apply explicit scenario overrides and reject unsafe cookbook environment settings."""
        env: dict[str, str] = {}
        for entry in entries:
            (token,) = shlex.split(entry) or [""]
            name, has_value, value = token.partition("=")
            if not has_value or (self._unsafe_env_name.fullmatch(name) and name not in overrides):
                raise ValueError(
                    f"{role}: the cookbook's env {entry!r} is not allowed; set {name} in the scenario's env"
                )
            env[name] = value
        env |= overrides
        if bad := [name for name in env if not self._env_name.fullmatch(name)]:
            raise ValueError(f"{role}: not environment variable names: {bad}")
        return env

    def substitute(self, argv: Sequence[str], values: Mapping[str, str]) -> list[str]:
        """Replace known cookbook placeholders with this node's launch values."""
        return [
            self._known_placeholder.sub(lambda match: values.get(match[1], match[0]), arg) for arg in argv
        ]

    def check_resolved(self, argv: Sequence[str], context: str) -> None:
        """Reject unresolved placeholders before any process starts."""
        if left := sorted({match for arg in argv for match in self._placeholder.findall(arg)}):
            raise ValueError(f"{context}: placeholders {left}; set them in env or args")


class ServerRecipe:
    """A unified, prefill or decode server recipe with scenario overrides applied."""

    def __init__(
        self,
        name: str,
        cell: CookbookCell,
        env: Mapping[str, str],
        args: Mapping[str, str | int | float | bool | None],
        nnodes: int,
    ) -> None:
        self.name = name
        self.cell = cell
        self.env = env
        self.args = args
        self.nnodes = nnodes
        self._rules = LaunchArgumentRules()

    @classmethod
    def of(cls, scenario: Scenario, cookbook: Cookbook) -> tuple[ServerRecipe, ...]:
        """Build each role from its cookbook cell, model path and scenario overrides."""
        # set, not only removed, and after the cell's flags: argparse takes the last value, and
        # also reads a prefix (--tokenizer-p) as the flag; a tokenizer of another repository
        # would run that repository's remote code
        weights = {
            "--model-path": scenario.model_path, "--tokenizer-path": scenario.model_path,
            "--model": None, "--tokenizer": None,
            "--revision": None,
        }  # fmt: skip
        roles = []
        for name, match in scenario.roles.items():
            cell = cookbook.cell(match)
            roles.append(
                cls(
                    name=name,
                    cell=cell,
                    env=scenario.env[name],
                    args=cookbook.parser_args(scenario.features[name].parsers)
                    | weights
                    | scenario.args[name],
                    nnodes=scenario.nnodes.get(name, cell.nnodes),
                )
            )
        return tuple(roles)

    def server_env(self) -> dict[str, str]:
        """The cell's env, then the scenario's; ``ValueError`` for a name ``env`` would read
        as an option (-S runs a command line), or one the cookbook may not set unless the
        scenario's env sets it too."""
        return self._rules.environment(self.name, self.cell.env, self.env)

    def flags(self) -> list[list[str]]:
        """The cell's flags, each a name and its values, with ``args`` applied (replaced by
        name, null or false removes, true is a switch) and the launcher's own removed."""
        # Keep placeholders in one token, including unknown names that must be rejected.
        flags = [shlex.split(re.sub(r"\{\{\s*(\w+)\s*\}\}", r"{{\1}}", f)) for f in self.cell.flags]
        if not all(flags):
            raise ValueError(f"{self.name}: an empty flag in the cookbook")
        for name, value in self.args.items():
            flags = [f for f in flags if f[0].split("=", 1)[0] != name]
            if value is True:
                flags.append([name])
            elif value is not None and value is not False:
                flags.append([name, str(value)])
        launcher_flags = {"--host", "--port", "--nnodes", "--node-rank", "--dist-init-addr"}
        return [f for f in flags if f[0].split("=", 1)[0] not in launcher_flags]

    @property
    def bootstrap_port(self) -> str:
        """The ``--disaggregation-bootstrap-port`` a prefill role serves its KV on."""
        for name, *values in self.flags():
            flag, has_value, value = name.partition("=")
            if flag == "--disaggregation-bootstrap-port":
                if not (has_value or values):
                    raise ValueError(f"{self.name}: --disaggregation-bootstrap-port has no value")
                return value if has_value else values[0]
        return "8998"


def export_commands(
    scenario: Scenario,
    cookbook: Cookbook,
    *,
    hosts: Sequence[str],
    env: Mapping[str, str],
    side_env: Mapping[str, str],
    startup_timeout: float,
) -> tuple[dict[str, tuple[str, ...]], str]:
    """Build server commands for the scenario using its cookbook recipe and launch settings."""
    server_port, router_port = 30000, 30080
    # The cookbook uses node 0 port 20000 for distributed initialization.
    dist_init_port = 20000
    rules = LaunchArgumentRules()
    roles = ServerRecipe.of(scenario, cookbook)
    if (n_nodes := sum(r.nnodes for r in roles)) > len(hosts):
        raise ValueError(f"{scenario.name}: needs {n_nodes} nodes, scenarios.yaml hosts gives {len(hosts)}")
    local_names = {"localhost", "127.0.0.1", socket.gethostname(), socket.getfqdn()}
    role_hosts: dict[str, Sequence[str]] = {}
    for role in roles:
        n_assigned = sum(map(len, role_hosts.values()))
        role_hosts[role.name] = hosts[n_assigned : n_assigned + role.nnodes]
    commands: dict[str, tuple[str, ...]] = {}
    for role in roles:
        nodes = role_hosts[role.name]
        inherited_hf = {k: v for k, v in env.items() if k in {"HF_HOME", "HF_HUB_CACHE", "HF_HUB_OFFLINE"}}
        authoritative = {k: v for k, v in env.items() if k not in inherited_hf}
        role_env = inherited_hf | role.server_env() | authoritative | dict(side_env)
        flags = role.flags()
        for rank, host in enumerate(nodes):
            node_flags = [*flags, ["--host", host], ["--port", str(server_port)]]
            if role.nnodes > 1:
                node_flags += [
                    ["--nnodes", str(role.nnodes)],
                    ["--node-rank", str(rank)],
                    ["--dist-init-addr", f"{nodes[0]}:{dist_init_port}"],
                ]
            argv = [
                *(f"{k}={v}" for k, v in role_env.items()),
                sys.executable, "-m", "sglang.launch_server", *(t for f in node_flags for t in f),
            ]  # fmt: skip
            values = {
                "MODEL_NAME": scenario.model_path, "HOST_IP": host, "LOCAL_IP": host,
                "PORT": str(server_port), "NODE0_IP": nodes[0], "NODE_RANK": str(rank),
                "PREFILL_HEAD_IP": role_hosts.get("prefill", nodes)[0],
            }  # fmt: skip
            argv = rules.substitute(argv, values)
            commands[f"{role.name}-{rank}"] = _host_command(host, argv, local_names)
    if "unified" in role_hosts:
        entry_url = f"http://{role_hosts['unified'][0]}:{server_port}"
    else:
        prefill, decode = role_hosts["prefill"][0], role_hosts["decode"][0]
        argv = [
            *(f"{k}={v}" for k, v in (dict(env) | dict(side_env)).items()),
            sys.executable, "-m", "sglang_router.launch_router", "--pd-disaggregation",
            "--prefill", f"http://{prefill}:{server_port}", roles[0].bootstrap_port,
            "--decode", f"http://{decode}:{server_port}",
            "--host", hosts[0], "--port", str(router_port),
            "--worker-startup-timeout-secs", str(int(startup_timeout)),
        ]  # fmt: skip
        commands["router"] = _host_command(hosts[0], argv, local_names)
        entry_url = f"http://{hosts[0]}:{router_port}"
    for name, argv in commands.items():
        rules.check_resolved(argv, f"{scenario.name}: {name}")
    return commands, entry_url


def _host_command(host: str, argv: list[str], local_names: set[str]) -> tuple[str, ...]:
    command = ["env", *argv]
    return tuple(command if host in local_names else ["ssh", "--", host, shlex.join(command)])
