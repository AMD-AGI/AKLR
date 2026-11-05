# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from aklr.core.validation.stages._e2e import _deployment
from aklr.core.validation.stages._e2e._config import E2eConfig, Scenario
from aklr.core.validation.stages._e2e._cookbook import Cookbook, CookbookCell
from aklr.core.validation.stages._e2e._deployment import Deployment, NodeProcess, RunNotKilledError
from aklr.core.validation.stages._e2e._report import SideResult
from aklr.core.validation.stages._e2e._stage import EXPECTED_ANSWER, E2eStage
from aklr.core.validation.stages._report._finding import FindingCode
from aklr.core.validation.stages.validation_scope import ValidationScope

_UNIFIED = {"hw": "mi355x", "strategy": "balanced"}
_PREFILL = {"hw": "a3", "pdMode": "prefill"}
_DECODE = {"hw": "a3", "pdMode": "decode"}
_COOKBOOK = Cookbook(
    source="main",
    sha="0" * 40,
    config="vendor/model",
    cells=(
        CookbookCell(
            match=_UNIFIED,
            env=("A=1", "B=2"),
            flags=(
                "--model-path {{MODEL_NAME}}",
                "--tp 8",
                "--mem-fraction-static 0.8",
                "--host {{HOST_IP}}",
                "--port {{PORT}}",
            ),
        ),
        CookbookCell(
            match={"hw": "mi355x", "strategy": "multi"}, nnodes=2, flags=("--model-path {{MODEL_NAME}}",)
        ),
        CookbookCell(
            match=_PREFILL,
            env=("SGLANG_HOST_IP={{LOCAL_IP}}",),
            flags=(
                "--model-path {{MODEL_NAME}}",
                "--disaggregation-mode prefill",
                "--disaggregation-bootstrap-port=9000",
            ),
        ),
        CookbookCell(match=_DECODE, flags=("--model-path {{MODEL_NAME}}", "--disaggregation-mode decode")),
    ),
    docker_images={},
)


def _scenario(
    roles: dict[str, Any], env: dict[str, Any] | None = None, args: dict[str, Any] | None = None
) -> Scenario:
    names = tuple(roles)
    return Scenario(
        name="model-x",
        model="model",
        display_name="Model",
        enabled=True,
        cookbook="vendor/model",
        roles=roles,
        model_path="org/Model",
        revision=None,
        env={r: (env or {}).get(r, {}) for r in names},
        args={r: (args or {}).get(r, {}) for r in names},
    )


class TestConfig:
    def test_reads_the_shipped_file(self) -> None:
        config = E2eConfig.read()
        assert config.cookbook_repo == "sgl-project/sglang"
        assert config.selected([]) == config.selected(["glm52-fp4"])
        (scenario,) = config.selected(["glm52-fp4-mi355x-default-mxfp4-low-latency-single"])
        assert scenario.roles == {"unified": {"hw": "mi355x", "variant": "default", "quant": "mxfp4",
            "strategy": "low-latency", "nodes": "single"}}  # fmt: skip
        with pytest.raises(ValueError, match="no e2e scenario or model"):
            config.selected(["glm"])

    def test_expands_matches_and_per_role_settings(self, tmp_path: Path) -> None:
        path = tmp_path / "scenarios.yaml"
        path.write_text(
            """
models:
  kimi-k3:
    cookbook:
      config: moonshotai/kimi-k3
      matches:
        - {hw: mi355x, pdMode: unified, strategy: balanced}
        - prefill: {hw: a3, pdMode: prefill, strategy: default}
          decode: {hw: a3, pdMode: decode, strategy: balanced}
    weights: {repo: moonshotai/Kimi-K3}
    env: {X: 1}
  disabled:
    enabled: false
    cookbook: {config: a/b, matches: [{hw: mi355x}]}
    weights: {repo: a/B}
    args: {prefill: {--tp: 4}}
""",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="per-role settings"):
            E2eConfig.read(path)
        path.write_text(path.read_text().replace("args: {prefill: {--tp: 4}}", "args: {--tp: 4, --x: null}"))
        config = E2eConfig.read(path)
        unified, pd, off = config.scenarios
        assert (unified.name, pd.name) == (
            "kimi-k3-mi355x-unified-balanced",
            "kimi-k3-a3-prefill-default-a3-decode-balanced",
        )
        assert list(pd.roles) == ["prefill", "decode"]
        assert pd.env == {"prefill": {"X": "1"}, "decode": {"X": "1"}}
        assert off.args == {"unified": {"--tp": 4, "--x": None}}
        assert config.selected([]) == (unified, pd)
        assert config.selected(["disabled"]) == (off,)

    @pytest.mark.parametrize(
        "text",
        [
            pytest.param("models: {}", id="no-model"),
            pytest.param(
                "models: {m: {cookbook: {config: a/b, matches: []}, weights: {repo: a/b}}}", id="no-match"
            ),
            pytest.param("models: {m: {cookbook: {config: a/b, matches: [{hw: x}]}}}", id="no-weights"),
            pytest.param("scenarios: []", id="old-format"),
            pytest.param(
                "models: {m: {cookbook: {config: a/b, matches: [{hw: x}, {hw: x}]}, weights: {repo: a/b}}}",
                id="repeated-name",
            ),
        ],
    )
    def test_invalid_files_fail(self, tmp_path: Path, text: str) -> None:
        path = tmp_path / "scenarios.yaml"
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match=str(path)):
            E2eConfig.read(path)


class TestDeploymentPlan:
    @staticmethod
    def _plan(
        scenario: Scenario, hosts: tuple[str, ...] = ("127.0.0.1",), launch: str | None = None
    ) -> Deployment:
        return Deployment.plan(
            scenario, _COOKBOOK, hosts=hosts, launch=launch, env={"S": "stage"}, side_env={"AKLR_DISABLE": "1"},
            run_dir=Path("/run"),
        )  # fmt: skip

    def test_unified_on_one_node(self) -> None:
        deployment = self._plan(_scenario({"unified": _UNIFIED}, env={"unified": {"B": "3"}}))
        (process,) = deployment.processes
        assert deployment.entry_url == "http://127.0.0.1:30000"
        assert process.argv == (
            "env", "A=1", "B=3", "S=stage", "AKLR_DISABLE=1", sys.executable, "-m", "sglang.launch_server",
            "--model-path", "org/Model", "--tp", "8", "--mem-fraction-static", "0.8",
            "--host", "127.0.0.1", "--port", "30000",
        )  # fmt: skip
        assert process.log_path == Path("/run/unified-0.log")

    def test_args_replace_remove_and_switch(self) -> None:
        args = {"--tp": 4, "--mem-fraction-static": None, "--enable-x": True, "--revision": "abc"}
        (process,) = self._plan(_scenario({"unified": _UNIFIED}, args={"unified": args})).processes
        flags = process.argv[process.argv.index("sglang.launch_server") + 1 :]
        assert flags == ("--model-path", "org/Model", "--tp", "4", "--enable-x", "--revision", "abc",
            "--host", "127.0.0.1", "--port", "30000")  # fmt: skip

    def test_multi_node_flags_and_launch(self) -> None:
        deployment = self._plan(
            _scenario({"unified": {"hw": "mi355x", "strategy": "multi"}}),
            hosts=("n0", "n1"),
            launch="ssh {host}",
        )
        node0, node1 = deployment.processes
        assert node1.argv[:3] == ("ssh", "n1", "env")
        assert node1.argv[-10:] == ("--host", "n1", "--port", "30000", "--nnodes", "2", "--node-rank", "1",
            "--dist-init-addr", "n0:20000")  # fmt: skip
        assert deployment.entry_url == "http://n0:30000"

    def test_pd_router_and_bootstrap_port(self) -> None:
        deployment = self._plan(
            _scenario({"prefill": _PREFILL, "decode": _DECODE}), hosts=("n0", "n1"), launch="ssh {host}"
        )
        prefill, decode, router = deployment.processes
        assert "SGLANG_HOST_IP=n0" in prefill.argv
        assert decode.argv[:2] == ("ssh", "n1")
        assert router.argv[-11:] == ("sglang_router.launch_router", "--pd-disaggregation",
            "--prefill", "http://n0:30000", "9000", "--decode", "http://n1:30000",
            "--host", "n0", "--port", "30080")  # fmt: skip
        assert deployment.entry_url == "http://n0:30080"

    @pytest.mark.parametrize(
        ("hosts", "launch", "message"),
        [
            pytest.param(("n0",), "ssh {host}", "needs 2 nodes, --e2e-hosts gives 1", id="few-nodes"),
            pytest.param(("n0", "n1"), None, "no --e2e-launch", id="no-launch"),
        ],
    )
    def test_nodes_must_suffice(self, hosts: tuple[str, ...], launch: str | None, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            self._plan(_scenario({"prefill": _PREFILL, "decode": _DECODE}), hosts=hosts, launch=launch)

    def test_a_placeholder_left_fails(self) -> None:
        args = {"unified": {"--nic": "<network-interface>"}}
        with pytest.raises(ValueError, match=r"placeholders \['<network-interface>'\]"):
            self._plan(_scenario({"unified": _UNIFIED}, args=args))
        with pytest.raises(ValueError, match="not environment variable names"):
            self._plan(_scenario({"unified": _UNIFIED}, env={"unified": {"-S": "sh -c id"}}))
        env = {"unified": {"NCCL_SOCKET_IFNAME": "{{NETWORK_IFACE}}"}}
        with pytest.raises(ValueError, match="NETWORK_IFACE"):
            self._plan(_scenario({"unified": _UNIFIED}, env=env))


def test_unkillable_process_raises_instead_of_hanging(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A group leader that outlives its SIGKILL (a GPU hang in D state) raises
    RunNotKilledError rather than blocking on its exit."""
    monkeypatch.setattr(_deployment.os, "killpg", lambda *args: None)
    monkeypatch.setattr(_deployment, "_KILL_WAIT_SECONDS", 0.2)
    process = NodeProcess(
        "unified-0", (sys.executable, "-c", "import time; time.sleep(5)"), tmp_path / "p.log"
    )
    process.start()
    raised: list[RunNotKilledError] = []

    def stop() -> None:
        try:
            process.stop()
        except RunNotKilledError as e:
            raised.append(e)

    thread = threading.Thread(target=stop, daemon=True)
    thread.start()
    thread.join(3)
    assert not thread.is_alive()
    assert len(raised) == 1


def test_stop_kills_the_group_and_returns_the_log(tmp_path: Path) -> None:
    code = "import subprocess, sys, time; print('up', flush=True); subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); time.sleep(60)"
    process = NodeProcess("unified-0", (sys.executable, "-c", code), tmp_path / "p.log")
    process.start()
    while not (tmp_path / "p.log").read_text():
        pass
    assert process.stop() == "up\n"
    assert process.exit_code is not None


class TestStageVerdicts:
    @pytest.mark.parametrize(
        ("off", "on", "sglang", "codes"),
        [
            pytest.param(
                SideResult(gsm8k_score=0.95),
                SideResult(gsm8k_score=0.94, answer_matches=True, prompt_answer="Paris.", n_hit_calls=3),
                "0.5.22",
                [],
                id="pass",
            ),  # fmt: skip
            pytest.param(
                SideResult(failure="no server"),
                None,
                "0.5.22",
                [FindingCode.PLATFORM_FAILURE],
                id="off-fails",
            ),
            pytest.param(SideResult(failure="no server"), None, "0.5.21", [], id="off-fails-other-sglang"),
            pytest.param(
                SideResult(gsm8k_score=0.95),
                SideResult(failure="crash", answer_matches=True),
                "0.5.22",
                [FindingCode.E2E_CRASH],
                id="on-crashes",
            ),  # fmt: skip
            pytest.param(
                SideResult(gsm8k_score=0.95),
                SideResult(gsm8k_score=0.92, answer_matches=True),
                "0.5.22",
                [FindingCode.E2E_ACCURACY_DROP],
                id="accuracy-drops",
            ),  # fmt: skip
            pytest.param(
                SideResult(gsm8k_score=0.95),
                SideResult(gsm8k_score=0.95, answer_matches=False, prompt_answer="Lyon"),
                "0.5.22",
                [FindingCode.E2E_WRONG_ANSWER],
                id="wrong-answer",
            ),  # fmt: skip
        ],
    )
    def test_verdicts(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        off: SideResult,
        on: SideResult | None,
        sglang: str,
        codes: list[FindingCode],
    ) -> None:
        cookbook = _COOKBOOK.model_copy(
            update={"docker_images": {"mi355x": "lmsysorg/sglang-rocm:v0.5.22-rocm720"}}
        )
        monkeypatch.setattr(Cookbook, "fetch", classmethod(lambda cls, *args: cookbook))
        monkeypatch.setattr("importlib.metadata.version", lambda name: sglang)
        sides = {"off": off, "on": on}
        ran: list[str] = []

        def run_side(
            self: E2eStage, scenario: Scenario, cookbook: Cookbook, side: str, *args: Any
        ) -> SideResult:
            ran.append(side)
            return sides[side]

        monkeypatch.setattr(E2eStage, "_run_side", run_side)
        stage = E2eStage(
            [_scenario({"unified": _UNIFIED})], ["cuda:0"], cookbook_repo="r", cookbook_source="main"
        )
        report = stage.run(ValidationScope(tmp_path, "", None, (), ()), Mock())
        assert [f.code for f in report.findings] == codes
        assert ran == (["off"] if on is None else ["off", "on"])
        (record,) = report.scenario_records
        assert record.cookbook_sha == "0" * 40
        assert report.expected_answer == EXPECTED_ANSWER
        assert "installed sglang" in record.notes
