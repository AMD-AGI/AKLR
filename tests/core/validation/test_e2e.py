# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import io
import json
import shlex
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from aklr.core.validation.stages._e2e import _deployment, _stage
from aklr.core.validation.stages._e2e._config import E2eConfig, Scenario
from aklr.core.validation.stages._e2e._cookbook import Cookbook, CookbookCell
from aklr.core.validation.stages._e2e._deployment import Deployment, NodeProcess, Role, RunNotKilledError
from aklr.core.validation.stages._e2e._report import E2eRecord, SideResult
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
    roles: dict[str, Any],
    env: dict[str, Any] | None = None,
    args: dict[str, Any] | None = None,
    nnodes: dict[str, int] | None = None,
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
        revision="r0",
        nnodes=nnodes or {},
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
    weights: {repo: moonshotai/Kimi-K3, revision: r0}
    nnodes: 2
    env: {X: 1}
  disabled:
    enabled: false
    cookbook: {config: a/b, matches: [{hw: mi355x}]}
    weights: {repo: a/B, revision: r0}
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
        assert pd.nnodes == {"prefill": 2, "decode": 2}
        assert off.nnodes == {}
        assert off.args == {"unified": {"--tp": 4, "--x": None}}
        assert config.selected([]) == (unified, pd)
        assert config.selected(["disabled"]) == (off,)

    @pytest.mark.parametrize(
        "text",
        [
            pytest.param("models: {}", id="no-model"),
            pytest.param(
                "models: {m: {cookbook: {config: a/b, matches: []}, weights: {repo: a/b, revision: r0}}}",
                id="no-match",
            ),
            pytest.param("models: {m: {cookbook: {config: a/b, matches: [{hw: x}]}}}", id="no-weights"),
            pytest.param(
                "models: {m: {cookbook: {config: a/b, matches: [{hw: x}]}, weights: {repo: a/b}}}",
                id="no-revision",
            ),
            pytest.param("scenarios: []", id="old-format"),
            pytest.param(
                "models: {m: {cookbook: {config: a/b, matches: [{hw: x}]}, weights: {repo: a/b, revision: r0}, nnodes: {decode: 2}}}",
                id="nnodes-of-no-role",
            ),
            pytest.param(
                "models: {m: {cookbook: {config: a/b, matches: [{hw: x}, {hw: x}]}, weights: {repo: a/b, revision: r0}}}",
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
        deployment = self._plan(
            _scenario({"unified": _UNIFIED}, env={"unified": {"B": "3", "S": "scenario"}})
        )
        (process,) = deployment.processes
        assert deployment.entry_url == "http://127.0.0.1:30000"
        # the stage's and the side's env last; the launcher's --model-path
        assert process.argv == (
            "env", "A=1", "B=3", "S=stage", "AKLR_DISABLE=1", sys.executable, "-m", "sglang.launch_server",
            "--tp", "8", "--mem-fraction-static", "0.8", "--model-path", "org/Model",
            "--tokenizer-path", "org/Model", "--revision", "r0", "--host", "127.0.0.1", "--port", "30000",
        )  # fmt: skip
        assert process.log_path == Path("/run/unified-0.log")

    def test_args_replace_remove_and_switch(self) -> None:
        args = {"--tp": 4, "--mem-fraction-static": None, "--enable-x": True, "--model-path": "local/dir"}
        (process,) = self._plan(_scenario({"unified": _UNIFIED}, args={"unified": args})).processes
        flags = process.argv[process.argv.index("sglang.launch_server") + 1 :]
        assert flags == ("--model-path", "local/dir", "--tokenizer-path", "org/Model", "--revision", "r0", "--tp", "4",
            "--enable-x", "--host", "127.0.0.1", "--port", "30000")  # fmt: skip

    def test_multi_node_flags_and_shell_quoted_launch(self) -> None:
        scenario = _scenario(
            {"unified": {"hw": "mi355x", "strategy": "multi"}}, env={"unified": {"V": "$(id)"}}
        )
        deployment = self._plan(scenario, hosts=("n0", "n1"), launch="ssh {host} {cmd}")
        node0, node1 = deployment.processes
        assert node1.argv[:2] == ("ssh", "n1")
        (command,) = node1.argv[2:]
        # the remote shell reads back the very argv, $(id) not run
        assert shlex.split(command)[:3] == ["env", "V=$(id)", "S=stage"]
        assert command.endswith("--host n1 --port 30000 --nnodes 2 --node-rank 1 --dist-init-addr n0:20000")
        assert deployment.entry_url == "http://n0:30000"

    def test_launch_needs_a_cmd_word(self) -> None:
        scenario = _scenario({"unified": {"hw": "mi355x", "strategy": "multi"}})
        with pytest.raises(ValueError, match=r"needs one \{cmd\} word"):
            self._plan(scenario, hosts=("n0", "n1"), launch="ssh {host}")
        for launch in ("ssh {host}", 'bash -lc "cd /x && {cmd}"'):
            with pytest.raises(ValueError, match=r"needs one \{cmd\} word"):
                E2eStage([scenario], ["cuda:0"], cookbook_repo="r", cookbook_source="main", launch=launch)
        # braces other than {host} and {cmd} are the launcher's own
        deployment = self._plan(
            scenario, hosts=("n0", "n1"), launch='ssh -o ProxyCommand="nc {proxy} 22" {host} {cmd}'
        )
        assert deployment.processes[1].argv[:4] == ("ssh", "-o", "ProxyCommand=nc {proxy} 22", "n1")

    def test_nnodes_overrides_the_cell(self) -> None:
        deployment = self._plan(
            _scenario({"unified": _UNIFIED}, nnodes={"unified": 2}),
            hosts=("n0", "n1"),
            launch="ssh {host} {cmd}",
        )
        assert len(deployment.processes) == 2
        assert "--nnodes 2" in deployment.processes[1].argv[-1]

    def test_pd_router_takes_the_prefill_bootstrap_port(self) -> None:
        scenario = _scenario(
            {"prefill": _PREFILL, "decode": _DECODE},
            args={"decode": {"--disaggregation-bootstrap-port": 9001}},
        )
        deployment = self._plan(scenario, hosts=("127.0.0.1", "localhost"))
        prefill, decode, router = deployment.processes
        assert "SGLANG_HOST_IP=127.0.0.1" in prefill.argv
        assert router.argv[-11:] == ("sglang_router.launch_router", "--pd-disaggregation",
            "--prefill", "http://127.0.0.1:30000", "9000", "--decode", "http://localhost:30000",
            "--host", "127.0.0.1", "--port", "30080")  # fmt: skip
        assert deployment.entry_url == "http://127.0.0.1:30080"

    @pytest.mark.parametrize(
        ("hosts", "launch", "message"),
        [
            pytest.param(("n0",), "ssh {host} {cmd}", "needs 2 nodes, --e2e-hosts gives 1", id="few-nodes"),
            pytest.param(("n0", "n1"), None, "no --e2e-launch", id="no-launch"),
        ],
    )
    def test_nodes_must_suffice(self, hosts: tuple[str, ...], launch: str | None, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            self._plan(_scenario({"prefill": _PREFILL, "decode": _DECODE}), hosts=hosts, launch=launch)

    @pytest.mark.parametrize(
        ("kind", "value"),
        [
            pytest.param("args", {"--nic": "<network-interface>"}, id="angle"),
            pytest.param("args", {"--nic": "<NETWORK_INTERFACE>"}, id="angle-upper"),
            pytest.param("env", {"IFNAME": "{{NETWORK_IFACE}}"}, id="braces"),
            pytest.param("env", {"IFNAME": "{{ NETWORK-IFACE }}"}, id="braces-odd"),
            pytest.param("env", {"ADDR": "<node0-ip:port>"}, id="angle-colon"),
            pytest.param("env", {"ADDR": "<node_0.ip>"}, id="angle-dot"),
        ],
    )
    def test_a_placeholder_left_fails(self, kind: str, value: dict[str, str]) -> None:
        with pytest.raises(ValueError, match="placeholders"):
            self._plan(_scenario({"unified": _UNIFIED}, **{kind: {"unified": value}}))


class TestRole:
    @staticmethod
    def _role(cell: CookbookCell, env: dict[str, str] | None = None) -> Role:
        return Role(name="unified", cell=cell, env=env or {}, args={}, nnodes=1)

    @pytest.mark.parametrize(
        "entry",
        [
            pytest.param("LD_PRELOAD=/tmp/x.so", id="loader"),
            pytest.param("PYTHONWARNINGS=ignore::antigravity.Geohash", id="interpreter"),
            pytest.param("BASH_ENV=/tmp/x", id="shell"),
            pytest.param("HF_ENDPOINT=https://example.com", id="hugging-face"),
            pytest.param("https_proxy=http://example.com", id="proxy"),
            pytest.param("SGLANG_USE_MODELSCOPE=1", id="modelscope"),
            pytest.param("AKLR_SGLANG_PLUGIN=0", id="aklr"),
            pytest.param("ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7", id="device-mask"),
            pytest.param("NOVALUE", id="no-value"),
        ],
    )
    def test_cookbook_env_is_limited(self, entry: str) -> None:
        with pytest.raises(ValueError, match="not allowed"):
            self._role(CookbookCell(match={}, env=(entry,))).server_env()

    def test_env_names_and_shell_quotes(self) -> None:
        cell = CookbookCell(match={}, env=('GLOO_SOCKET_IFNAME="eth 0"',))
        assert self._role(cell).server_env() == {"GLOO_SOCKET_IFNAME": "eth 0"}
        with pytest.raises(ValueError, match="not environment variable names"):
            self._role(cell, env={"-S": "sh -c id"}).server_env()

    def test_bad_flags_fail(self) -> None:
        with pytest.raises(ValueError, match="empty flag"):
            self._role(CookbookCell(match={}, flags=("",))).flags()
        with pytest.raises(ValueError, match="has no value"):
            _ = self._role(CookbookCell(match={}, flags=("--disaggregation-bootstrap-port",))).bootstrap_port


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
    code = (
        "import subprocess, sys, time; print('up', flush=True);"
        " subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); time.sleep(60)"
    )
    process = NodeProcess("unified-0", (sys.executable, "-c", code), tmp_path / "p.log")
    process.start()
    while not (tmp_path / "p.log").read_text():
        pass
    assert process.stop() == "up\n"
    assert process.exit_code is not None


class TestVerdicts:
    _OK = SideResult(gsm8k_score=0.95, answer_matches=True)

    @pytest.mark.parametrize(
        ("off", "on", "images", "codes"),
        [
            pytest.param(SideResult(gsm8k_score=0.95), _OK, (), [], id="pass"),
            pytest.param(
                SideResult(gsm8k_score=0.96),
                SideResult(gsm8k_score=0.94, answer_matches=True),
                (),
                [],
                id="drop-of-exactly-the-threshold",
            ),  # fmt: skip
            pytest.param(
                SideResult(gsm8k_score=1073 / 1319),
                SideResult(gsm8k_score=1046 / 1319, answer_matches=True),
                (),
                [FindingCode.E2E_ACCURACY_DROP],
                id="drop-just-past-the-threshold",
            ),  # fmt: skip
            pytest.param(
                SideResult(failure="no server"), None, (), [FindingCode.PLATFORM_FAILURE], id="off-fails"
            ),
            pytest.param(
                SideResult(failure="no server"),
                None,
                ("lmsysorg/sglang-rocm:v0.5.22-rocm720",),
                [FindingCode.PLATFORM_FAILURE],
                id="off-fails-same-release",
            ),  # fmt: skip
            pytest.param(
                SideResult(failure="no server"),
                None,
                ("lmsysorg/sglang-rocm:v0.5.13.post1-x",),
                [],
                id="off-fails-other-sglang",
            ),  # fmt: skip
            pytest.param(
                SideResult(gsm8k_score=0.0), None, (), [FindingCode.PLATFORM_FAILURE], id="off-scores-nothing"
            ),
            pytest.param(
                SideResult(gsm8k_score=0.95),
                SideResult(failure="crash", answer_matches=True),
                (),
                [FindingCode.E2E_CRASH],
                id="on-crashes",
            ),  # fmt: skip
            pytest.param(
                SideResult(gsm8k_score=0.95),
                SideResult(gsm8k_score=0.92, answer_matches=True),
                (),
                [FindingCode.E2E_ACCURACY_DROP],
                id="accuracy-drops",
            ),  # fmt: skip
            pytest.param(
                SideResult(gsm8k_score=0.95),
                SideResult(gsm8k_score=0.95, answer_matches=False),
                (),
                [FindingCode.E2E_WRONG_ANSWER],
                id="wrong-answer",
            ),  # fmt: skip
        ],
    )
    def test_findings(
        self, off: SideResult, on: SideResult | None, images: tuple[str, ...], codes: list[FindingCode]
    ) -> None:
        record = E2eRecord(
            scenario="s", cookbook="v/m", cookbook_sha="0" * 40, matches={}, cookbook_images=images,
            installed_sglang="0.5.22.post1", off=off, on=on,
        )  # fmt: skip
        findings = record.findings(max_drop=0.02, min_off_score=0.5)
        assert [f.code for f in findings] == codes


def test_an_off_failure_under_another_sglang_is_a_warning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cookbook = _COOKBOOK.model_copy(
        update={"docker_images": {"mi355x": "lmsysorg/sglang-rocm:v0.5.19-rocm720"}}
    )
    monkeypatch.setattr(Cookbook, "pinned", staticmethod(lambda source, repo: "0" * 40))
    monkeypatch.setattr(Cookbook, "fetch", classmethod(lambda cls, *args: cookbook))
    monkeypatch.setattr("importlib.metadata.version", lambda name: "0.5.22")
    monkeypatch.setattr(E2eStage, "_run_side", lambda self, *args: SideResult(gsm8k_score=0.1))
    stage = E2eStage(
        [_scenario({"unified": _UNIFIED})], ["cuda:0"], cookbook_repo="r", cookbook_source="main"
    )
    report = stage.run(ValidationScope(tmp_path, "", None, (), ()), Mock())
    (record,) = report.scenario_records
    assert report.findings == ()
    assert record.warning.startswith("off failed, on not compared")
    assert "warning: off failed" in record.table_row()[-1]


class TestStage:
    def _stage(self) -> E2eStage:
        return E2eStage(
            [_scenario({"unified": _UNIFIED})], ["cuda:0"], cookbook_repo="r", cookbook_source="main"
        )

    @pytest.mark.parametrize(
        ("off", "sides"),
        [
            pytest.param(SideResult(gsm8k_score=0.9), ["off", "on"], id="sound-off"),
            pytest.param(SideResult(failure="x"), ["off"], id="failed-off"),
            pytest.param(SideResult(gsm8k_score=0.1), ["off"], id="off-below-the-floor"),
        ],
    )
    def test_on_runs_only_after_a_sound_off(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, off: SideResult, sides: list[str]
    ) -> None:
        monkeypatch.setattr(Cookbook, "pinned", staticmethod(lambda source, repo: "0" * 40))
        monkeypatch.setattr(Cookbook, "fetch", classmethod(lambda cls, *args: _COOKBOOK))
        ran: list[str] = []

        def run_side(
            self: E2eStage, scenario: Scenario, cookbook: Cookbook, side: str, *args: Any
        ) -> SideResult:
            ran.append(side)
            return off if side == "off" else SideResult(gsm8k_score=0.9, answer_matches=True)

        monkeypatch.setattr(E2eStage, "_run_side", run_side)
        report = self._stage().run(ValidationScope(tmp_path, "", None, (), ()), Mock())
        assert ran == sides
        (record,) = report.scenario_records
        assert (record.cookbook, record.cookbook_sha, record.matches) == (
            "vendor/model", "0" * 40, {"unified": _UNIFIED},
        )  # fmt: skip
        assert report.expected_answer == EXPECTED_ANSWER

    def test_run_side_grades_the_server(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """GSM8K's full-precision score and its failed requests, the answer as the RFC
        compares it, and a server that died during the phases."""
        monkeypatch.setattr(Deployment, "start", lambda self, timeout: "http://h:1")
        monkeypatch.setattr(Deployment, "stop", lambda self: "log tail")
        exited: list[str] = []
        monkeypatch.setattr(Deployment, "exited", lambda self: exited)
        stdout = "Score: 0.814\n{'score': np.float64(0.81365), 'latency': 1.0}\n"
        monkeypatch.setattr(
            _stage.subprocess,
            "run",
            lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=stdout, stderr=""),
        )
        answer = " **Paris**.\n"
        monkeypatch.setattr(
            _stage.urllib.request,
            "urlopen",
            lambda *a, **k: io.BytesIO(json.dumps({"choices": [{"message": {"content": answer}}]}).encode()),
        )
        scenario = _scenario({"unified": _UNIFIED})
        on = self._stage()._run_side(scenario, _COOKBOOK, "on", {}, tmp_path)
        assert (on.gsm8k_score, on.answer_matches, on.n_hit_calls, on.failure) == (0.81365, False, 0, "")
        answer = "Paris."
        assert self._stage()._run_side(scenario, _COOKBOOK, "on", {}, tmp_path).answer_matches
        stdout += "All retry attempts exhausted for request. Returning empty response.\n"
        exited.append("unified-0 exited with -9")
        off = self._stage()._run_side(scenario, _COOKBOOK, "off", {}, tmp_path)
        assert off.gsm8k_score is None
        assert "1 GSM8K requests failed" in off.failure
        assert "during the phases: unified-0 exited with -9" in off.failure
        monkeypatch.setattr(
            Deployment, "start", lambda self, timeout: (_ for _ in ()).throw(RuntimeError("no"))
        )
        assert self._stage()._run_side(scenario, _COOKBOOK, "on", {}, tmp_path).n_hit_calls is None
        monkeypatch.setattr(Deployment, "start", lambda self, timeout: "http://h:1")
        answer = 42  # type: ignore[assignment]
        on = self._stage()._run_side(scenario, _COOKBOOK, "on", {}, tmp_path)
        assert "content is not a string" in on.failure
