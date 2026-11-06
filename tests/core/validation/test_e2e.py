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

from aklr.core.validation.stages._e2e import E2eStage, _deployment, _stage, export_commands
from aklr.core.validation.stages._e2e._command import ServerRecipe
from aklr.core.validation.stages._e2e._config import E2eConfig, Scenario
from aklr.core.validation.stages._e2e._cookbook import Cookbook, CookbookCell
from aklr.core.validation.stages._e2e._deployment import Deployment, NodeProcess, RunNotKilledError
from aklr.core.validation.stages._e2e._report import E2eRecord, ModelEvaluationResult
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
        assert config.cookbook_source == "main"
        assert config.hosts == ("127.0.0.1",)
        assert config.work_dir is None
        (scenario,) = config.enabled_scenarios
        assert scenario.roles == {"unified": {"hw": "mi355x", "variant": "default", "quant": "mxfp4",
            "strategy": "low-latency", "nodes": "single"}}  # fmt: skip

    def test_deployment_settings_come_from_yaml(self, tmp_path: Path) -> None:
        path = tmp_path / "scenarios.yaml"
        path.write_text(
            "cookbook: {repo: custom/sglang, source: /shared/sglang}\n"
            "hosts: [gpu0, gpu1]\n"
            "work_dir: /shared/e2e\n"
            "models: {m: {cookbook: {config: a/b, matches: [{hw: x}]}, "
            "weights: {repo: a/b, revision: r0}}}\n"
        )
        config = E2eConfig.read(path)
        assert config.cookbook_repo == "custom/sglang"
        assert config.cookbook_source == "/shared/sglang"
        assert config.hosts == ("gpu0", "gpu1")
        assert config.work_dir == Path("/shared/e2e")
        path.write_text(path.read_text().replace("hosts: [gpu0, gpu1]", "hosts: []"))
        with pytest.raises(ValueError, match="hosts"):
            E2eConfig.read(path)

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
        assert config.enabled_scenarios == (unified, pd)

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


class TestCommandBuilder:
    @staticmethod
    def _plan(
        scenario: Scenario, hosts: tuple[str, ...] = ("127.0.0.1",)
    ) -> tuple[dict[str, tuple[str, ...]], str]:
        return export_commands(
            scenario, _COOKBOOK, hosts=hosts, env={"S": "stage"}, side_env={"AKLR_DISABLE": "1"},
        )  # fmt: skip

    def test_unified_on_one_node(self) -> None:
        commands, entry_url = self._plan(
            _scenario({"unified": _UNIFIED}, env={"unified": {"B": "3", "S": "scenario"}})
        )
        (argv,) = commands.values()
        assert entry_url == "http://127.0.0.1:30000"
        # the stage's and the side's env last; the launcher's --model-path
        assert argv == (
            "env", "A=1", "B=3", "S=stage", "AKLR_DISABLE=1", sys.executable, "-m", "sglang.launch_server",
            "--tp", "8", "--mem-fraction-static", "0.8", "--model-path", "org/Model",
            "--tokenizer-path", "org/Model", "--revision", "r0", "--host", "127.0.0.1", "--port", "30000",
        )  # fmt: skip

    def test_args_replace_remove_and_switch(self) -> None:
        args = {"--tp": 4, "--mem-fraction-static": None, "--enable-x": True, "--model-path": "local/dir"}
        commands, _ = self._plan(_scenario({"unified": _UNIFIED}, args={"unified": args}))
        (argv,) = commands.values()
        flags = argv[argv.index("sglang.launch_server") + 1 :]
        assert flags == ("--model-path", "local/dir", "--tokenizer-path", "org/Model", "--revision", "r0", "--tp", "4",
            "--enable-x", "--host", "127.0.0.1", "--port", "30000")  # fmt: skip

    def test_multi_node_flags_and_shell_quoted_launch(self) -> None:
        scenario = _scenario(
            {"unified": {"hw": "mi355x", "strategy": "multi"}}, env={"unified": {"V": "$(id)"}}
        )
        commands, entry_url = self._plan(scenario, hosts=("n0", "n1"))
        node0, node1 = commands.values()
        assert node1[:3] == ("ssh", "--", "n1")
        (command,) = node1[3:]
        # the remote shell reads back the very argv, $(id) not run
        assert shlex.split(command)[:3] == ["env", "V=$(id)", "S=stage"]
        assert command.endswith("--host n1 --port 30000 --nnodes 2 --node-rank 1 --dist-init-addr n0:20000")
        assert entry_url == "http://n0:30000"

    def test_nnodes_overrides_the_cell(self) -> None:
        commands, entry_url = self._plan(
            _scenario({"unified": _UNIFIED}, nnodes={"unified": 2}),
            hosts=("n0", "n1"),
        )
        assert len(commands) == 2
        assert "--nnodes 2" in commands["unified-1"][-1]

    def test_pd_router_takes_the_prefill_bootstrap_port(self) -> None:
        scenario = _scenario(
            {"prefill": _PREFILL, "decode": _DECODE},
            args={"decode": {"--disaggregation-bootstrap-port": 9001}},
        )
        commands, entry_url = self._plan(scenario, hosts=("127.0.0.1", "localhost"))
        prefill, decode, router = commands.values()
        assert "SGLANG_HOST_IP=127.0.0.1" in prefill
        assert router[-11:] == ("sglang_router.launch_router", "--pd-disaggregation",
            "--prefill", "http://127.0.0.1:30000", "9000", "--decode", "http://localhost:30000",
            "--host", "127.0.0.1", "--port", "30080")  # fmt: skip
        assert entry_url == "http://127.0.0.1:30080"

    def test_nodes_must_suffice(self) -> None:
        with pytest.raises(ValueError, match="needs 2 nodes, scenarios.yaml hosts gives 1"):
            self._plan(_scenario({"prefill": _PREFILL, "decode": _DECODE}), hosts=("n0",))

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


class TestServerRecipe:
    @staticmethod
    def _recipe(cell: CookbookCell, env: dict[str, str] | None = None) -> ServerRecipe:
        return ServerRecipe(name="unified", cell=cell, env=env or {}, args={}, nnodes=1)

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
            self._recipe(CookbookCell(match={}, env=(entry,))).server_env()

    def test_the_scenario_may_replace_a_denied_name(self) -> None:
        cell = CookbookCell(match={}, env=("SGLANG_USE_MODELSCOPE=1",))
        assert self._recipe(cell, env={"SGLANG_USE_MODELSCOPE": "0"}).server_env() == {
            "SGLANG_USE_MODELSCOPE": "0"
        }

    def test_env_names_and_shell_quotes(self) -> None:
        cell = CookbookCell(match={}, env=('GLOO_SOCKET_IFNAME="eth 0"',))
        assert self._recipe(cell).server_env() == {"GLOO_SOCKET_IFNAME": "eth 0"}
        with pytest.raises(ValueError, match="not environment variable names"):
            self._recipe(cell, env={"-S": "sh -c id"}).server_env()

    def test_bad_flags_fail(self) -> None:
        with pytest.raises(ValueError, match="empty flag"):
            self._recipe(CookbookCell(match={}, flags=("",))).flags()
        with pytest.raises(ValueError, match="has no value"):
            _ = self._recipe(
                CookbookCell(match={}, flags=("--disaggregation-bootstrap-port",))
            ).bootstrap_port


def test_unkillable_process_raises_instead_of_hanging(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A group leader that outlives its SIGKILL (a GPU hang in D state) raises
    RunNotKilledError rather than blocking on its exit."""
    monkeypatch.setattr(_deployment.os, "killpg", lambda *args: None)
    process = NodeProcess(
        "unified-0", (sys.executable, "-c", "import time; time.sleep(5)"), tmp_path / "p.log"
    )
    monkeypatch.setattr(process, "_kill_wait_seconds", 0.2)
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
    _OK = ModelEvaluationResult(gsm8k_score=0.95, answer_matches=True)

    @pytest.mark.parametrize(
        ("off", "on", "images", "codes"),
        [
            pytest.param(ModelEvaluationResult(gsm8k_score=0.95), _OK, (), [], id="pass"),
            pytest.param(
                ModelEvaluationResult(gsm8k_score=0.96),
                ModelEvaluationResult(gsm8k_score=0.94, answer_matches=True),
                (),
                [],
                id="drop-of-exactly-the-threshold",
            ),  # fmt: skip
            pytest.param(
                ModelEvaluationResult(gsm8k_score=1073 / 1319),
                ModelEvaluationResult(gsm8k_score=1046 / 1319, answer_matches=True),
                (),
                [FindingCode.E2E_ACCURACY_DROP],
                id="drop-just-past-the-threshold",
            ),  # fmt: skip
            pytest.param(
                ModelEvaluationResult(failure="no server"),
                None,
                (),
                [FindingCode.PLATFORM_FAILURE],
                id="off-fails",
            ),
            pytest.param(
                ModelEvaluationResult(failure="no server"),
                None,
                ("lmsysorg/sglang-rocm:v0.5.22-rocm720",),
                [FindingCode.PLATFORM_FAILURE],
                id="off-fails-same-release",
            ),  # fmt: skip
            pytest.param(
                ModelEvaluationResult(failure="no server"),
                None,
                ("lmsysorg/sglang-rocm:v0.5.13.post1-x",),
                [],
                id="off-fails-other-sglang",
            ),  # fmt: skip
            pytest.param(
                ModelEvaluationResult(gsm8k_score=0.0),
                None,
                (),
                [FindingCode.PLATFORM_FAILURE],
                id="off-scores-nothing",
            ),
            pytest.param(
                ModelEvaluationResult(gsm8k_score=0.95),
                ModelEvaluationResult(failure="crash", answer_matches=True),
                (),
                [FindingCode.E2E_CRASH],
                id="on-crashes",
            ),  # fmt: skip
            pytest.param(
                ModelEvaluationResult(gsm8k_score=0.95),
                ModelEvaluationResult(gsm8k_score=0.92, answer_matches=True),
                (),
                [FindingCode.E2E_ACCURACY_DROP],
                id="accuracy-drops",
            ),  # fmt: skip
            pytest.param(
                ModelEvaluationResult(gsm8k_score=0.95),
                ModelEvaluationResult(gsm8k_score=0.95, answer_matches=False),
                (),
                [FindingCode.E2E_WRONG_ANSWER],
                id="wrong-answer",
            ),  # fmt: skip
        ],
    )
    def test_findings(
        self,
        off: ModelEvaluationResult,
        on: ModelEvaluationResult | None,
        images: tuple[str, ...],
        codes: list[FindingCode],
    ) -> None:
        record = E2eRecord(
            scenario="s", cookbook="v/m", cookbook_sha="0" * 40, matches={}, cookbook_images=images,
            installed_sglang="0.5.22.post1", off=off, on=on,
        )  # fmt: skip
        findings = record.findings(max_drop=0.02, min_off_score=0.5)
        assert [f.code for f in findings] == codes


@pytest.fixture
def stage_config(monkeypatch: pytest.MonkeyPatch) -> None:
    config = E2eConfig(
        cookbook_repo="r", cookbook_source="main", scenarios=(_scenario({"unified": _UNIFIED}),)
    )
    monkeypatch.setattr(E2eConfig, "read", classmethod(lambda cls: config))
    monkeypatch.setattr(Cookbook, "pinned", staticmethod(lambda source, repo: "0" * 40))
    monkeypatch.setattr(Cookbook, "fetch", classmethod(lambda cls, *args: _COOKBOOK))


@pytest.mark.usefixtures("stage_config")
def test_an_off_failure_under_another_sglang_is_a_warning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cookbook = _COOKBOOK.model_copy(
        update={"docker_images": {"mi355x": "lmsysorg/sglang-rocm:v0.5.19-rocm720"}}
    )
    monkeypatch.setattr(Cookbook, "pinned", staticmethod(lambda source, repo: "0" * 40))
    monkeypatch.setattr(Cookbook, "fetch", classmethod(lambda cls, *args: cookbook))
    monkeypatch.setattr("importlib.metadata.version", lambda name: "0.5.22")
    monkeypatch.setattr(
        E2eStage, "_evaluate_model", lambda self, *args, **kwargs: ModelEvaluationResult(gsm8k_score=0.1)
    )
    stage = E2eStage(["cuda:0"])
    report = stage.run(ValidationScope(tmp_path, "", None, (), ()), Mock())
    (record,) = report.scenario_records
    assert report.findings == ()
    assert record.warning.startswith("off failed, on not compared")
    assert "warning: off failed" in record.table_row()[-1]


def test_no_enabled_scenario_skips_evaluation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = E2eConfig(cookbook_repo="r", cookbook_source="main", scenarios=())
    monkeypatch.setattr(E2eConfig, "read", classmethod(lambda cls: config))
    assert E2eStage([]).run(ValidationScope(tmp_path, "", None, (), ()), Mock()).status == "skipped"


@pytest.mark.usefixtures("stage_config")
class TestEvaluator:
    def _evaluator(self) -> E2eStage:
        return E2eStage(["cuda:0"])

    @pytest.mark.parametrize(
        ("off", "sides"),
        [
            pytest.param(ModelEvaluationResult(gsm8k_score=0.9), [False, True], id="sound-off"),
            pytest.param(ModelEvaluationResult(failure="x"), [False], id="failed-off"),
            pytest.param(ModelEvaluationResult(gsm8k_score=0.1), [False], id="off-below-the-floor"),
        ],
    )
    def test_on_runs_only_after_a_sound_off(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        off: ModelEvaluationResult,
        sides: list[bool],
    ) -> None:
        monkeypatch.setattr(Cookbook, "pinned", staticmethod(lambda source, repo: "0" * 40))
        monkeypatch.setattr(Cookbook, "fetch", classmethod(lambda cls, *args: _COOKBOOK))
        ran: list[bool] = []

        def evaluate_model(
            self: E2eStage, scenario: Scenario, *args: Any, aklr_enabled: bool
        ) -> ModelEvaluationResult:
            ran.append(aklr_enabled)
            return off if not aklr_enabled else ModelEvaluationResult(gsm8k_score=0.9, answer_matches=True)

        monkeypatch.setattr(E2eStage, "_evaluate_model", evaluate_model)
        report = self._evaluator().run(ValidationScope(tmp_path, "", None, (), ()), Mock())
        assert ran == sides
        (record,) = report.scenario_records
        assert (record.cookbook, record.cookbook_sha, record.matches) == (
            "vendor/model", "0" * 40, {"unified": _UNIFIED},
        )  # fmt: skip
        assert report.expected_answer == "Paris"

    def test_evaluate_model_grades_the_server(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
        on = self._evaluator()._evaluate_model(scenario, _COOKBOOK, {}, tmp_path, aklr_enabled=True)
        assert "AKLR never started on the on side" in on.failure
        (tmp_path / "on-calls").mkdir()
        (tmp_path / "on-calls" / "calls.pid1.x.jsonl").write_text("")
        on = self._evaluator()._evaluate_model(scenario, _COOKBOOK, {}, tmp_path, aklr_enabled=True)
        assert (on.gsm8k_score, on.answer_matches, on.n_hit_calls) == (0.81365, False, 0)
        assert "AKLR hooked no sglang call" in on.failure
        record = Mock(fallback_reason="unknown_shape", n_calls=3)
        monkeypatch.setattr(_stage, "read_calls", lambda directory: [record])
        on = self._evaluator()._evaluate_model(scenario, _COOKBOOK, {}, tmp_path, aklr_enabled=True)
        assert (on.n_hit_calls, on.n_fallback_calls, on.failure) == (0, 3, "")
        answer = "Paris_."
        assert (
            self._evaluator()
            ._evaluate_model(scenario, _COOKBOOK, {}, tmp_path, aklr_enabled=True)
            .answer_matches
        )
        stdout += "Bad Request Error\n" * 2
        assert (
            self._evaluator()
            ._evaluate_model(scenario, _COOKBOOK, {}, tmp_path, aklr_enabled=False)
            .n_failed_requests
            == 2
        )
        stdout += "All retry attempts exhausted for request. Returning empty response.\n"
        exited.append("unified-0 exited with -9")
        off = self._evaluator()._evaluate_model(scenario, _COOKBOOK, {}, tmp_path, aklr_enabled=False)
        assert off.gsm8k_score is None
        assert "3 GSM8K requests failed" in off.failure
        assert off.n_failed_requests == 3
        assert "during the phases: unified-0 exited with -9" in off.failure
        monkeypatch.setattr(
            Deployment, "start", lambda self, timeout: (_ for _ in ()).throw(RuntimeError("no"))
        )
        assert (
            self._evaluator()
            ._evaluate_model(scenario, _COOKBOOK, {}, tmp_path, aklr_enabled=True)
            .n_hit_calls
            is None
        )
        monkeypatch.setattr(Deployment, "start", lambda self, timeout: "http://h:1")
        answer = 42  # type: ignore[assignment]
        on = self._evaluator()._evaluate_model(scenario, _COOKBOOK, {}, tmp_path, aklr_enabled=True)
        assert "content is not a string" in on.failure


def test_export_returns_argv_without_starting_processes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "Popen", Mock(side_effect=AssertionError("export started a process")))
    commands, url = export_commands(
        _scenario({"unified": _UNIFIED}),
        _COOKBOOK,
        hosts=("127.0.0.1",),
        env={},
        side_env={"AKLR_DISABLE": "1"},
    )
    assert url == "http://127.0.0.1:30000"
    argv = commands["unified-0"]
    assert "AKLR_DISABLE=1" in argv
    assert argv[argv.index("--revision") + 1] == "r0"


def test_prints_command_before_executing_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = (sys.executable, "-c", "import time; time.sleep(60)")
    deployment = Deployment({"unified-0": argv}, "http://localhost:1", tmp_path)
    start = NodeProcess.start

    def checked_start(process: NodeProcess) -> None:
        output = capsys.readouterr().out
        assert "Command (unified-0):" in output
        assert shlex.join(argv) in output
        start(process)

    monkeypatch.setattr(NodeProcess, "start", checked_start)
    response = Mock()
    response.__enter__ = Mock(return_value=Mock(status=200))
    response.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(_deployment.urllib.request, "urlopen", lambda *a, **kw: response)
    try:
        assert deployment.start(timeout=1) == "http://localhost:1"
        assert deployment.processes[0].exit_code is None
    finally:
        deployment.stop()
    assert deployment.processes[0].exit_code is not None


@pytest.mark.usefixtures("stage_config")
def test_cleanup_on_start_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    scenario = _scenario({"unified": _UNIFIED})
    argv = (sys.executable, "-c", "import time; time.sleep(60)")
    monkeypatch.setattr(
        _stage, "export_commands", lambda *a, **kw: ({"unified-0": argv}, "http://localhost:1")
    )
    started = []
    original = NodeProcess.start

    def start(process: NodeProcess) -> None:
        started.append(process)
        original(process)

    monkeypatch.setattr(NodeProcess, "start", start)
    monkeypatch.setattr(_deployment.urllib.request, "urlopen", Mock(side_effect=OSError("not ready")))
    stage = E2eStage(["cuda:0"])
    stage._startup_timeout = 0.05
    result = stage._evaluate_model(scenario, _COOKBOOK, {}, tmp_path, aklr_enabled=False)
    assert "not ready" in result.failure
    assert started and all(p.exit_code is not None for p in started)


def test_direct_subprocess_preserves_literal_arguments(tmp_path: Path) -> None:
    marker = tmp_path / "not-executed"
    value = f"spaces ' quotes $(touch {marker}) `touch {marker}`"
    process = NodeProcess(
        "literal", (sys.executable, "-c", "import sys; print(sys.argv[1])", value), tmp_path / "literal.log"
    )
    process.start()
    try:
        process._process.wait(timeout=5)
    finally:
        log = process.stop()
    assert log == value + "\n"
    assert not marker.exists()


@pytest.mark.usefixtures("stage_config")
def test_validate_reports_e2e_server_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from aklr.core.validation import command
    from aklr.core.validation.pipeline import PlatformRecord
    from aklr.core.validation.stages import AuditStage, BenchReport, BenchStage
    from aklr.core.validation.stages._audit._report import AuditReport

    scope = ValidationScope(tmp_path, "head", None, (tmp_path / "schema.json",), ())
    monkeypatch.setattr(command.ValidationScope, "of_working_tree", Mock(return_value=scope))
    platform = PlatformRecord(
        **(dict.fromkeys(PlatformRecord.model_fields, "unknown") | {"n_visible_devices": 1})
    )
    monkeypatch.setattr(PlatformRecord, "current", Mock(return_value=platform))
    monkeypatch.setattr(AuditStage, "run", lambda *args: AuditReport(status="passed"))
    monkeypatch.setattr(BenchStage, "run", lambda *args: BenchReport(status="passed"))
    argv = (sys.executable, "-c", "print('startup-failed', flush=True); raise SystemExit(7)")
    monkeypatch.setattr(
        _stage, "export_commands", lambda *a, **kw: ({"unified-0": argv}, "http://localhost:1")
    )
    monkeypatch.setattr(_deployment.urllib.request, "urlopen", Mock(side_effect=OSError("not ready")))
    output = tmp_path / "report"
    assert command.main(["--stages", "audit,bench,e2e", "--devices", "0", "--out", str(output)]) == 3
    captured = capsys.readouterr()
    assert shlex.join(argv) in captured.out
    assert "model-x" in captured.err
    report = json.loads((output / "report.json").read_text())
    assert [stage["stage_name"] for stage in report["stages"]] == ["audit", "bench", "e2e"]
    e2e = report["stages"][-1]
    assert e2e["status"] == "aborted"
    assert "exited with 7" in e2e["scenario_records"][0]["off"]["failure"]
    assert "startup-failed" in e2e["scenario_records"][0]["off"]["failure"]
    assert e2e["scenario_records"][0]["on"] is None
    assert "e2e" in (output / "report.md").read_text()
