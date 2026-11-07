# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Functional tests through the stage API, with process, network and host-probing I/O mocked."""

from __future__ import annotations

import html
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
import yaml

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
        "prompt": "What is the capital of France? Answer with one word.",
        "expected_answer": "Paris",
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
        "answer": "Paris",
        "requests": [],
        "commands": [],
        "killed": [],
        "evaluations": [],
        "console": [],
        "write_calls": True,
        "git_failure": False,
        "server_failure": None,
        "controls": [],
        "control_failures": {},
    }
    read_text = Path.read_text
    glob = Path.glob

    def paths(path: Path, pattern: str):
        # Fake process groups have no host processes; do not scan the real /proc tree.
        if path == Path("/proc"):
            return iter(())
        return glob(path, pattern)

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
        command = shlex.split(argv[-1]) if argv[0] == "ssh" else argv
        state["env"] = dict(
            arg.split("=", 1) for arg in command[1:] if not arg.startswith("--") and "=" in arg
        )
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
        command = shlex.split(argv[-1]) if argv[0] == "ssh" else argv
        if len(command) > 1 and command[1].endswith("/_worker.py"):
            action = command[2]
            state["controls"].append((argv, action))
            failure = state["control_failures"].get(action, "")
            return subprocess.CompletedProcess(argv, int(bool(failure)), "", failure)
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
        state["requests"].append(json.loads(request.data))
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": state["answer"]}}]}).encode())

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(Path, "glob", paths)
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


def pipeline_report(report: E2eReport) -> PipelineReport:
    platform = PlatformRecord(
        **(dict.fromkeys(PlatformRecord.model_fields, "unknown") | {"n_visible_devices": 0})
    )
    return PipelineReport(
        head_sha="head",
        base_sha=None,
        measurement_profile={},
        platform=platform,
        schema_paths=(),
        bench_paths=(),
        stages=(report,),
    )


def test_stage_sends_committed_prompt_with_one_key_at_100k_midpoint(serving: dict, tmp_path: Path) -> None:
    path = Path(__file__).resolve().parents[3] / "aklr/core/validation/stages/_e2e/scenarios.yaml"
    # Use the shipped prompt, with the fixture's local recipe and mocked server I/O.
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    serving["config"].update({key: config[key] for key in ("prompt", "expected_answer")})
    serving["answer"] = config["expected_answer"]

    report = evaluate(tmp_path)

    assert report.status == "passed"
    (request,) = serving["requests"]
    prompt = request["messages"][0]["content"]
    assert len(prompt) == 100000
    assert prompt.count(serving["answer"]) == 1
    assert abs(prompt.index(serving["answer"]) - 50000) < 100
    assert report.prompt == prompt
    assert report.scenario_records[0].on.answer_matches is True


@pytest.mark.parametrize("answer", [" kEy_9!\n", "key_9!", "kEy_9", "kEy_9! extra"])
def test_configured_long_prompt_is_sent_and_key_is_graded_exactly(
    serving: dict, tmp_path: Path, answer: str
) -> None:
    serving["config"]["prompt"] = {
        "length_chars": 100000,
        "prefix": "Read the archive.\n",
        "filler": "Ordinary background text.\n",
        "needle": "\nSecret key: kEy_9!\n",
        "suffix": "\nReturn only the secret key.",
    }
    serving["config"]["expected_answer"] = "kEy_9!"
    serving["answer"] = answer
    report = evaluate(tmp_path)
    (request,) = serving["requests"]
    prompt = request["messages"][0]["content"]
    assert len(prompt) == 100000
    assert prompt.count("kEy_9!") == 1
    assert abs(prompt.index("kEy_9!") - 50000) < 20
    assert report.prompt == prompt
    assert report.expected_answer == "kEy_9!"
    assert report.status == ("passed" if answer.strip() == "kEy_9!" else "failed")
    assert report.scenario_records[0].on.answer_matches == (answer.strip() == "kEy_9!")
    # JSON preserves the full input; Markdown stays readable.
    pipeline = pipeline_report(report)
    assert pipeline.model_dump()["stages"][0]["prompt"] == prompt
    markdown = pipeline.markdown_text()
    assert "100,000 characters" in markdown
    assert "expected `kEy_9!`" in html.unescape(markdown)
    assert len(markdown) < 10000


@pytest.mark.parametrize(
    "prompt",
    [
        "",
        "  ",
        {"length_chars": 10, "prefix": "too long", "filler": "x", "needle": "KEY", "suffix": "end"},
        {"length_chars": 100000, "prefix": "", "filler": "", "needle": "KEY", "suffix": ""},
    ],
)
def test_invalid_prompt_fails_before_servers_start(serving: dict, tmp_path: Path, prompt) -> None:
    serving["config"]["prompt"] = prompt
    with pytest.raises(ValueError, match="prompt"):
        evaluate(tmp_path)
    assert not serving["commands"]


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
    pipeline = pipeline_report(report)
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
    entry["matches"] = [
        {
            "decode": decode | {"parsers": {"reasoning": False}},
            "prefill": prefill | {"parsers": {"reasoning": True}},
        }
    ]
    entry["env"] = {"prefill": {"ROLE": "prefill"}, "decode": {"ROLE": "decode"}}
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
    assert report.scenario_records[0].scenario == (
        "model-mi355x-prefill-parsers-reasoning-on-mi355x-decode-parsers-reasoning-off"
    )
    assert len(serving["commands"]) == len(serving["killed"]) == 6
    for prefill_argv, decode_argv, router_argv in (serving["commands"][:3], serving["commands"][3:]):
        assert "ROLE=prefill" in prefill_argv
        assert "ROLE=decode" in decode_argv
        assert prefill_argv[prefill_argv.index("--mem-fraction-static") + 1] == "0.7"
        assert decode_argv[decode_argv.index("--mem-fraction-static") + 1] == "0.8"
        assert prefill_argv[prefill_argv.index("--reasoning-parser") + 1] == "cookbook-reasoning"
        assert "--reasoning-parser" not in decode_argv
        assert "sglang_router.launch_router" in router_argv
        assert router_argv[router_argv.index("--worker-startup-timeout-secs") + 1] == "2400"


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
    entry["matches"][0].update(features)
    entry["args"] = args
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
    serving["config"]["scenarios"][0]["matches"][0]["parsers"] = {"missing": enabled}
    report = evaluate(tmp_path)
    assert report.status == "aborted"
    assert "matches.parsers.missing: no such parser" in report.findings[0].detail
    assert serving["commands"] == []


@pytest.mark.parametrize("features", [{"attention": {"tp": 4}}, {"parsers": {"reasoning": "false"}}])
def test_invalid_features_fail_configuration(serving: dict, tmp_path: Path, features) -> None:
    serving["config"]["scenarios"][0]["matches"][0].update(features)
    with pytest.raises(ValueError):
        evaluate(tmp_path)
    assert serving["commands"] == []


def test_parser_value_tracks_the_cookbook(serving: dict, tmp_path: Path) -> None:
    entry = serving["config"]["scenarios"][0]
    entry["matches"][0]["parsers"] = {"reasoning": True}
    entry["args"] = {}
    path = tmp_path / "sglang/docs/src/snippets/configs/vendor/model.jsx"
    path.write_text(path.read_text().replace("cookbook-reasoning", "new-parser"))
    report = evaluate(tmp_path)
    assert report.status == "passed"
    for argv in serving["commands"]:
        assert argv[argv.index("--reasoning-parser") + 1] == "new-parser"


def test_same_cell_with_different_parsers_runs_separate_scenarios(serving: dict, tmp_path: Path) -> None:
    entry = serving["config"]["scenarios"][0]
    entry["matches"] = [
        {"hw": "mi355x", "parsers": {"reasoning": True}},
        {"hw": "mi355x", "parsers": {"reasoning": False}},
        {"hw": "mi355x"},
    ]
    entry["args"] = {}

    report = evaluate(tmp_path)

    assert report.status == "passed"
    assert [record.scenario for record in report.scenario_records] == [
        "model-mi355x-parsers-reasoning-on",
        "model-mi355x-parsers-reasoning-off",
        "model-mi355x",
    ]
    assert len(serving["commands"]) == len(serving["killed"]) == 6
    for argv in serving["commands"][:2]:
        assert argv[argv.index("--reasoning-parser") + 1] == "cookbook-reasoning"
    for argv in serving["commands"][2:4]:
        assert "--reasoning-parser" not in argv
        assert not any(token.startswith("--reasoning-parser=") for token in argv)
    for argv in serving["commands"][4:]:
        assert "--reasoning-parser=inherited-reasoning" in argv


def test_duplicate_parser_selection_ignores_key_order(serving: dict, tmp_path: Path) -> None:
    serving["config"]["scenarios"][0]["matches"] = [
        {"hw": "mi355x", "parsers": {"reasoning": True, "toolCall": False}},
        {"parsers": {"toolCall": False, "reasoning": True}, "hw": "mi355x"},
    ]
    with pytest.raises(ValueError, match="is repeated"):
        evaluate(tmp_path)
    assert serving["commands"] == []


def test_entry_level_features_rejected(serving: dict, tmp_path: Path) -> None:
    serving["config"]["scenarios"][0]["features"] = {"parsers": {"reasoning": True}}
    with pytest.raises(ValueError, match="features"):
        evaluate(tmp_path)
    assert serving["commands"] == []


def test_duplicate_parser_ids_fail_before_starting_servers(serving: dict, tmp_path: Path) -> None:
    path = tmp_path / "sglang/docs/src/snippets/configs/vendor/model.jsx"
    path.write_text(path.read_text().replace('id: "toolCall"', 'id: "reasoning"'))
    report = evaluate(tmp_path)
    assert report.status == "aborted"
    assert "duplicate playground parser IDs" in report.findings[0].detail
    assert serving["commands"] == []


@pytest.mark.parametrize("override", [False, True])
def test_hf_defaults_allow_role_overrides_but_device_controls_do_not(
    serving: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, override: bool
) -> None:
    inherited = {"HF_HOME": "/parent", "HF_HUB_CACHE": "/parent/hub", "HF_HUB_OFFLINE": "1"}
    selected = {"HF_HOME": "/role", "HF_HUB_CACHE": "/role/hub", "HF_HUB_OFFLINE": "0"}
    for key, value in inherited.items():
        monkeypatch.setenv(key, value)
    entry = serving["config"]["scenarios"][0]
    entry["env"] = (selected if override else {}) | {
        "HIP_VISIBLE_DEVICES": "7",
        "CUDA_VISIBLE_DEVICES": "7",
        "AKLR_DISABLE": "1",
    }
    assert evaluate(tmp_path).status == "passed"
    for argv in serving["commands"]:
        assert all(f"{key}={value}" in argv for key, value in (selected if override else inherited).items())
        assert "HIP_VISIBLE_DEVICES=0" in argv and "CUDA_VISIBLE_DEVICES=0" in argv
    assert "AKLR_DISABLE=0" in serving["commands"][1]


@pytest.mark.parametrize("action", ["available", "owns-port"])
def test_occupied_or_unowned_listener_is_never_graded(serving: dict, tmp_path: Path, action: str) -> None:
    serving["control_failures"][action] = "port is occupied by another deployment"
    report = evaluate(tmp_path)
    assert report.status == "aborted"
    assert "port is occupied" in report.scenario_records[0].off.failure
    assert not serving["evaluations"]
    assert len(serving["commands"]) == len(serving["killed"]) == (0 if action == "available" else 1)


def test_unverified_remote_cleanup_stops_the_run(serving: dict, tmp_path: Path) -> None:
    serving["config"]["hosts"] = ["remote-node"]
    serving["config"]["scenarios"][0]["matches"].append(
        {
            "hw": "mi355x",
            "parsers": {"reasoning": True},
        }
    )
    serving["control_failures"]["stop"] = "SSH unreachable"
    report = evaluate(tmp_path)
    assert report.status == "aborted"
    assert "remote cleanup was not verified" in report.findings[0].detail
    assert len(serving["commands"]) == len(serving["killed"]) == 1
    assert len(report.scenario_records) == 1
    assert report.scenario_records[0].on is None
    cleanup = [argv for argv, action in serving["controls"] if action == "stop"]
    assert len(cleanup) == 1 and cleanup[0][0] == "ssh"


@pytest.mark.parametrize("remote", [False, True])
@pytest.mark.parametrize("foreign_listener", [False, True])
def test_stage_checks_ownership_and_cleans_up(
    serving: dict, tmp_path: Path, remote: bool, foreign_listener: bool
) -> None:
    if remote:
        serving["config"]["hosts"] = ["remote-node"]
    if foreign_listener:
        serving["control_failures"]["owns-port"] = "port is owned by another deployment"

    report = evaluate(tmp_path)

    assert report.status == ("aborted" if foreign_listener else "passed")
    sides = 1 if foreign_listener else 2
    assert len(serving["commands"]) == len(serving["killed"]) == sides
    assert len(serving["evaluations"]) == (0 if foreign_listener else 2)
    actions = [action for _, action in serving["controls"]]
    assert actions.count("available") == actions.count("owns-port") == sides
    assert actions.count("stop") == (sides if remote else 0)
    if remote:
        assert all(argv[0] == "ssh" for argv, _ in serving["controls"])
