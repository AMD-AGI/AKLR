# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Exercise the e2e package through its stage interface and external I/O boundaries."""

from __future__ import annotations

import io
import json
import os
import shlex
import subprocess
import urllib.request
from pathlib import Path
from unittest.mock import Mock

import pytest

from aklr.core.validation import PipelineReport, ValidationScope
from aklr.core.validation.pipeline import PlatformRecord
from aklr.core.validation.stages._e2e import E2eReport, E2eStage


@pytest.fixture
def serving(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> dict:
    cookbook = tmp_path / "sglang/docs/src/snippets/configs/vendor"
    cookbook.mkdir(parents=True)
    (cookbook / "model.jsx").write_text("""export default {
  cells: [{match: {hw: "mi355x"}, env: ["COOKBOOK_SETTING=1"],
    flags: ["--model-path {{MODEL_NAME}}", "--tp 8", "--host {{HOST_IP}}", "--port {{PORT}}"]}],
};
""")
    config = {
        "cookbook": {"source": str(tmp_path / "sglang")},
        "models": {
            "test": {
                "cookbook": {"config": "vendor/model", "matches": [{"hw": "mi355x"}]},
                "weights": {"repo": "vendor/Model", "revision": "weights-revision"},
                "args": {"--reasoning-parser": "glm45"},
            }
        },
    }
    state = {
        "config": config,
        "scores": [0.9, 0.9],
        "answer": "Paris_.",
        "commands": [],
        "killed": [],
        "evaluations": [],
        "console": [],
        "write_calls": True,
        "git_failure": False,
        "server_failure": None,
    }
    read_text = Path.read_text

    def read(path: Path, *args, **kwargs):
        if path.name == "scenarios.yaml" and path.parent.name == "_e2e":
            return json.dumps(config)
        return read_text(path, *args, **kwargs)

    def start(argv, **kwargs):
        # This is the OS process boundary, not an internal e2e class or method.
        output = capsys.readouterr()
        state["console"].append(output.out + output.err)
        assert shlex.join(argv) in output.out
        assert kwargs["start_new_session"] is True
        state["commands"].append(argv)
        state["env"] = dict(arg.split("=", 1) for arg in argv[1:] if not arg.startswith("--") and "=" in arg)
        if state["env"]["AKLR_DISABLE"] == "0" and state["write_calls"]:
            calls = Path(state["env"]["AKLR_EXPORT_DIR"])
            calls.mkdir(parents=True)
            record = {
                "contract_id": "gemm@1.0",
                "point": None,
                "scalars": {},
                "fallback_reason": "unknown_shape",
                "target": {"arch": "gfx950", "hardware": "MI355X"},
                "n_calls": 3,
            }
            (calls / "calls.pid1.jsonl").write_text(json.dumps(record) + "\n")
        return Mock(pid=2**30 + len(state["commands"]), poll=Mock(return_value=state["server_failure"]))

    def run(argv, **kwargs):
        if argv[0] == "git":
            assert 0 < kwargs.get("timeout", 0) <= 60, "Git lookup must be bounded"
            if state["git_failure"]:
                return subprocess.CompletedProcess(argv, 128, "not a ref record", "authentication failed")
            return subprocess.CompletedProcess(argv, 0, "a" * 40 + "\n", "")
        assert "sglang.test.run_eval" in argv
        state["evaluations"].append(argv)
        score = state["scores"][int(state["env"]["AKLR_DISABLE"] == "0")]
        return subprocess.CompletedProcess(argv, 0, f"{{'score': {score}}}\n", "")

    def urlopen(request, **kwargs):
        if isinstance(request, str):
            assert request.endswith("/health_generate")
            response = io.BytesIO(b"ok")
            response.status = 200
            return response
        assert request.full_url.endswith("/v1/chat/completions")
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": state["answer"]}}]}).encode())

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(subprocess, "Popen", start)
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(os, "killpg", lambda pid, signum: state["killed"].append(pid))
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    return state


def evaluate(tmp_path: Path, devices=("cuda:0",)) -> E2eReport:
    scope = ValidationScope(tmp_path, "head", None, (), ())
    # The earlier pipeline report is the stage's other public input.
    earlier = Mock(spec=PipelineReport)
    return E2eStage(devices).execute(scope, earlier)


def test_stage_prints_and_serves_both_modes(serving: dict, tmp_path: Path) -> None:
    report = evaluate(tmp_path, devices=("cuda:0", "cuda:2"))
    assert report.status == "passed"
    assert len(serving["commands"]) == len(serving["killed"]) == 2
    off, on = serving["commands"]
    assert "AKLR_DISABLE=1" in off and "AKLR_EXPORT=0" in off
    assert "AKLR_DISABLE=0" in on and "AKLR_EXPORT=1" in on
    for argv in (off, on):
        assert "COOKBOOK_SETTING=1" in argv
        assert "HIP_VISIBLE_DEVICES=0,2" in argv
        assert argv[argv.index("--model-path") + 1] == "vendor/Model"
        assert argv[argv.index("--revision") + 1] == "weights-revision"
        assert argv[argv.index("--reasoning-parser") + 1] == "glm45"
    (record,) = report.scenario_records
    assert record.cookbook_sha == "a" * 40
    assert record.on.answer_matches is True
    assert record.on.n_fallback_calls == 3
    assert "test-mi355x" in "".join(serving["console"])
    platform = PlatformRecord(
        **(dict.fromkeys(PlatformRecord.model_fields, "unknown") | {"n_visible_devices": 0})
    )
    pipeline = PipelineReport(
        head_sha="head",
        base_sha=None,
        measurement_profile={},
        platform=platform,
        schema_paths=(),
        bench_paths=(),
        stages=(report,),
    )
    assert pipeline.exit_code == 0
    assert "GSM8K" in pipeline.markdown_text()
    assert pipeline.model_dump()["stages"][0]["stage_name"] == "e2e"


@pytest.mark.parametrize(
    ("change", "value", "expected"),
    [
        ("scores", [0.9, 0.85], "e2e_accuracy_drop"),
        ("answer", "London", "e2e_wrong_answer"),
        ("write_calls", False, "e2e_crash"),
    ],
)
def test_stage_reports_evaluation_failures(serving: dict, tmp_path: Path, change, value, expected) -> None:
    serving[change] = value
    report = evaluate(tmp_path)
    assert report.status == "failed"
    assert expected in [finding.code.value for finding in report.findings]
    assert len(serving["killed"]) == 2


def test_stage_stops_after_bad_baseline(serving: dict, tmp_path: Path) -> None:
    serving["scores"] = [0.1, 0.9]
    report = evaluate(tmp_path)
    assert report.status == "aborted"
    assert len(serving["commands"]) == len(serving["killed"]) == 1
    assert report.scenario_records[0].on is None


def test_stage_reports_server_exit_and_cleans_up(serving: dict, tmp_path: Path) -> None:
    serving["server_failure"] = 7
    report = evaluate(tmp_path)
    assert report.status == "aborted"
    assert "exited with 7" in report.scenario_records[0].off.failure
    assert len(serving["killed"]) == 1
    assert not serving["evaluations"]


def test_disabled_scenarios_need_no_devices_or_servers(serving: dict, tmp_path: Path) -> None:
    serving["config"]["models"]["test"]["enabled"] = False
    assert evaluate(tmp_path, devices=()).status == "skipped"
    assert serving["commands"] == []


def test_stage_reports_remote_git_failure(serving: dict, tmp_path: Path) -> None:
    serving["config"]["cookbook"]["source"] = "main"
    serving["git_failure"] = True
    report = evaluate(tmp_path)
    assert report.status == "aborted"
    assert "git ls-remote failed" in report.findings[0].detail
    assert "authentication failed" in report.findings[0].detail
    assert serving["commands"] == []
