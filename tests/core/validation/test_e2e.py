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
from copy import deepcopy
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
  playgroundFeatures: {
    parsers: {
      showWhen: (s) => s.hw !== "a3",
      items: [
        {id: "reasoning", flag: "--reasoning-parser cookbook-reasoning"},
        {id: "toolCall", flag: "--tool-call-parser=cookbook-tool"},
      ],
    },
  },
  cells: [{match: {hw: "mi355x"}, env: ["COOKBOOK_SETTING=1"],
    flags: ["--model-path {{MODEL_NAME}}", "--tp 8", "--host {{HOST_IP}}", "--port {{PORT}}",
      "--reasoning-parser=inherited-reasoning", "--tool-call-parser inherited-tool"]}],
};
""")
    config = {
        "cookbook": {"source": str(tmp_path / "sglang")},
        "scenarios": [
            {
                "cookbook": "vendor/model",
                "model_path": "vendor/Model",
                "matches": [{"hw": "mi355x"}],
                "args": {"--revision": "weights-revision", "--reasoning-parser": "glm45"},
            }
        ],
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
            calls.mkdir(parents=True, exist_ok=True)
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
    assert record.scenario == "model-mi355x"
    assert "model-mi355x" in "".join(serving["console"])
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
    serving["config"]["scenarios"][0]["enabled"] = False
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


def test_entry_expands_matches_in_yaml_order(serving: dict, tmp_path: Path) -> None:
    entry = serving["config"]["scenarios"][0]
    matches = [
        {"hw": "mi355x", "strategy": "balanced"},
        {"strategy": "low-latency", "hw": "mi355x"},
    ]
    entry["matches"] = matches
    cookbook = tmp_path / "sglang/docs/src/snippets/configs/vendor/model.jsx"
    cells = [{"match": match, "flags": ["--model-path {{MODEL_NAME}}"]} for match in matches]
    cookbook.write_text("export default {\n  cells: " + json.dumps(cells) + "\n};\n")

    report = evaluate(tmp_path)

    assert report.status == "passed"
    assert [record.scenario for record in report.scenario_records] == [
        "model-mi355x-balanced",
        "model-low-latency-mi355x",
    ]
    assert len(serving["commands"]) == len(serving["killed"]) == 4
    for argv in serving["commands"]:
        assert argv[argv.index("--model-path") + 1] == "vendor/Model"
        assert argv[argv.index("--revision") + 1] == "weights-revision"


def test_pd_match_uses_role_overrides_and_prefill_then_decode_name(serving: dict, tmp_path: Path) -> None:
    entry = serving["config"]["scenarios"][0]
    prefill = {"hw": "mi355x", "pdMode": "prefill"}
    decode = {"hw": "mi355x", "pdMode": "decode"}
    entry["matches"] = [{"decode": decode, "prefill": prefill}]
    entry["env"] = {"prefill": {"ROLE": "prefill"}, "decode": {"ROLE": "decode"}}
    entry["features"] = {
        "prefill": {"parsers": {"reasoning": True}},
        "decode": {"parsers": {"reasoning": False}},
    }
    entry["args"] = {
        "prefill": {"--revision": "weights-revision", "--mem-fraction-static": 0.7},
        "decode": {"--revision": "weights-revision", "--mem-fraction-static": 0.8},
    }
    serving["config"]["hosts"] = ["127.0.0.1", "localhost"]
    cookbook = tmp_path / "sglang/docs/src/snippets/configs/vendor/model.jsx"
    cells = [
        {"match": match, "flags": ["--model-path {{MODEL_NAME}}", "--reasoning-parser inherited"]}
        for match in (prefill, decode)
    ]
    cookbook.write_text(
        "export default {\n  playgroundFeatures: {parsers: {items: "
        '[{id: "reasoning", flag: "--reasoning-parser cookbook-reasoning"}]}},\n  cells: '
        + json.dumps(cells)
        + "\n};\n"
    )

    report = evaluate(tmp_path)

    assert report.status == "passed"
    assert report.scenario_records[0].scenario == "model-mi355x-prefill-mi355x-decode"
    assert len(serving["commands"]) == len(serving["killed"]) == 6
    for prefill_argv, decode_argv, router_argv in (serving["commands"][:3], serving["commands"][3:]):
        assert "ROLE=prefill" in prefill_argv
        assert "ROLE=decode" in decode_argv
        assert prefill_argv[prefill_argv.index("--mem-fraction-static") + 1] == "0.7"
        assert decode_argv[decode_argv.index("--mem-fraction-static") + 1] == "0.8"
        assert prefill_argv[prefill_argv.index("--reasoning-parser") + 1] == "cookbook-reasoning"
        assert "--reasoning-parser" not in decode_argv
        assert "sglang_router.launch_router" in router_argv


def test_duplicate_scenario_names_fail_before_starting_servers(serving: dict, tmp_path: Path) -> None:
    serving["config"]["scenarios"].append(deepcopy(serving["config"]["scenarios"][0]))
    with pytest.raises(ValueError, match="the scenario name model-mi355x is repeated"):
        evaluate(tmp_path)
    assert serving["commands"] == []


@pytest.mark.parametrize("field", ["cookbook", "model_path", "matches"])
def test_missing_scenario_fields_fail_before_starting_servers(serving: dict, tmp_path: Path, field) -> None:
    del serving["config"]["scenarios"][0][field]
    with pytest.raises(ValueError, match=field):
        evaluate(tmp_path)
    assert serving["commands"] == []


def test_empty_scenarios_skip_without_devices(serving: dict, tmp_path: Path) -> None:
    serving["config"]["scenarios"] = []
    assert evaluate(tmp_path, devices=()).status == "skipped"
    assert serving["commands"] == []


@pytest.mark.parametrize(
    ("features", "args", "reasoning", "tool_call"),
    [
        ({}, {}, "inherited-reasoning", "inherited-tool"),
        ({"parsers": {"reasoning": True}}, {}, "cookbook-reasoning", "inherited-tool"),
        ({"parsers": {"reasoning": False}}, {}, None, "inherited-tool"),
        ({"parsers": {"toolCall": True}}, {}, "inherited-reasoning", "cookbook-tool"),
        ({"parsers": {"reasoning": True, "toolCall": False}}, {}, "cookbook-reasoning", None),
        ({"parsers": {"reasoning": True}}, {"--reasoning-parser": "override"}, "override", "inherited-tool"),
        ({"parsers": {"reasoning": True}}, {"--reasoning-parser": None}, None, "inherited-tool"),
        ({"parsers": {"reasoning": False}}, {"--reasoning-parser": "override"}, "override", "inherited-tool"),
    ],
)
def test_parser_features_inherit_toggle_and_allow_final_args(
    serving: dict, tmp_path: Path, features, args, reasoning, tool_call
) -> None:
    entry = serving["config"]["scenarios"][0]
    entry["features"], entry["args"] = features, args
    report = evaluate(tmp_path)
    assert report.status == "passed"
    for argv in serving["commands"]:
        for flag, expected in (("--reasoning-parser", reasoning), ("--tool-call-parser", tool_call)):
            values = [
                token.split("=", 1)[1] if "=" in token else argv[index + 1]
                for index, token in enumerate(argv)
                if token.split("=", 1)[0] == flag
            ]
            assert values == ([] if expected is None else [expected])


@pytest.mark.parametrize("enabled", [True, False])
def test_unknown_parser_fails_before_starting_servers(serving: dict, tmp_path: Path, enabled) -> None:
    serving["config"]["scenarios"][0]["features"] = {"parsers": {"missing": enabled}}
    report = evaluate(tmp_path)
    assert report.status == "aborted"
    assert "features.parsers.missing: no such parser" in report.findings[0].detail
    assert serving["commands"] == []


@pytest.mark.parametrize("features", [{"attention": {"tp": 4}}, {"parsers": {"reasoning": "false"}}])
def test_invalid_features_fail_configuration(serving: dict, tmp_path: Path, features) -> None:
    serving["config"]["scenarios"][0]["features"] = features
    with pytest.raises(ValueError):
        evaluate(tmp_path)
    assert serving["commands"] == []


def test_parser_value_tracks_the_cookbook(serving: dict, tmp_path: Path) -> None:
    entry = serving["config"]["scenarios"][0]
    entry["features"], entry["args"] = {"parsers": {"reasoning": True}}, {}
    path = tmp_path / "sglang/docs/src/snippets/configs/vendor/model.jsx"
    path.write_text(path.read_text().replace("cookbook-reasoning", "new-parser"))
    report = evaluate(tmp_path)
    assert report.status == "passed"
    for argv in serving["commands"]:
        assert argv[argv.index("--reasoning-parser") + 1] == "new-parser"


def test_duplicate_parser_ids_fail_before_starting_servers(serving: dict, tmp_path: Path) -> None:
    path = tmp_path / "sglang/docs/src/snippets/configs/vendor/model.jsx"
    path.write_text(path.read_text().replace('id: "toolCall"', 'id: "reasoning"'))
    report = evaluate(tmp_path)
    assert report.status == "aborted"
    assert "duplicate playground parser IDs" in report.findings[0].detail
    assert serving["commands"] == []
