# SPDX-License-Identifier: MIT
"""AgentX image environment boundaries, without GPU dependencies."""

from __future__ import annotations

import copy
import importlib
import io
import json
import os
import sys
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

with mock.patch.object(
    sys, "path", [str(Path(__file__).resolve().parents[2] / "benchmarks/agentx"), *sys.path]
):
    runtime = importlib.import_module("agentx_ci.runtime")
    analysis = importlib.import_module("agentx_ci.performance_analysis")

IMAGE_ID = "sha256:" + "a" * 64
IMAGE_REFERENCE = "example.invalid/server@sha256:" + "b" * 64
SDK = "/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel"


class EnvironmentCase:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.inspection_path = self.root / "execution.json"
        self.defaults = {
            "PATH": "/image/bin",
            "SGLANG_USE_AITER": "1",
            "ROCM_HOME": SDK,
            "ROCM_PATH": SDK,
            "CPATH": SDK + "/include",
            "LIBRARY_PATH": SDK + "/lib",
            "PIP_CONSTRAINT": "/image/constraints.txt",
            "VIRTUAL_ENV": "/opt/venv",
        }
        self.inspection = {
            "Id": IMAGE_ID,
            "RepoDigests": [IMAGE_REFERENCE],
            "Config": {"Env": [f"{key}={value}" for key, value in self.defaults.items()]},
        }
        self.point = {
            "arm": "tp1",
            "conc": 1,
            "exp_name": "tp1-c1",
            "served_model_name": "fixture-model",
            "gpus": 1,
            "launch": {
                "command": ["python3", "-m", "sglang.launch_server"],
                "env": {"SGLANG_USE_AITER": "0", "AKLR_SGLANG_PLUGIN": "1", "AKLR_DISABLE": "0"},
            },
            "client_env": {
                "TP": "1",
                "PP_SIZE": "1",
                "PCP_SIZE": "1",
                "DCP_SIZE": "1",
                "EP_SIZE": "1",
                "DP_ATTENTION": "false",
                "TOTAL_CPU_DRAM_GB": "16",
                "DURATION": "60",
            },
            "runtime": {"port": 8888, "ready_timeout_s": 30, "point_timeout_s": 60, "poll_s": 0.1},
            "hashes": {"serve_spec_sha256": "c" * 64},
            "trace": {"repo": "example/traces", "revision": "d" * 40, "loader": "fixture"},
        }
        self.serving = {
            "schema": 3,
            "model": {"id": "fixture"},
            "target": {"id": "fixture-gpu"},
            "image": {"run": IMAGE_REFERENCE},
            "weights": {"repo": "example/model", "revision": "e" * 40},
            "hardware": {"gpu_count": 1, "gpu_name_pattern": "Fixture", "gpu_arch": "gfx950"},
            "provenance": {"inferencex": {"repo": "example/source", "sha": "f" * 40}},
            "grid_sha256": "0" * 64,
            "arms": {"tp1": {"arm_serve_sha256": "1" * 64}},
            "points": [self.point],
        }
        self.host = {
            "model_path": str(self.root / "model"),
            "tokenizer": "example/model",
            "inferencex_e2e": self.root,
            "agentx_runtime_dir": self.root / "agentx-runtime",
            "datasets_cache": self.root / "datasets",
            "trace_hub": self.root / "trace-hub",
            "client_env_overrides": {},
            "server_image_env": self.defaults,
        }
        self.aggregate = {
            "request_metrics": {
                "throughput": {"per_gpu": {"total_tput_tps": 100}},
                "latency": {"e2e_norm_intvty": {"p90": 10}, "full_response_itl": {"p90": 0.1}},
            },
            "num_requests_successful": 2,
        }
        self.environ = {
            "AGENTX_EXECUTION_FILE": str(self.inspection_path),
            "PATH": "/runtime/bin",
            "HF_HUB_CACHE": "/mounted/hub",
            "HF_TOKEN": "live-fixture-token",
            "HTTPS_PROXY": "https://fixture:live-proxy-password@proxy.invalid",
            "SGLANG_USE_AITER": "ambient",
            "ROCM_HOME": "/wrong/sdk",
            "ARBITRARY_TUNING": "unplanned",
        }

    def write_inspection(self, inspection=None):
        self.inspection_path.write_text(json.dumps(self.inspection if inspection is None else inspection))

    def run_main(self, inspection=None, *, measurements=None):
        self.write_inspection(inspection)
        serving_path = self.root / "serving.json"
        serving_path.write_text(json.dumps(self.serving))
        out = self.root / "results"
        hardware = {
            "devices": [{"name": "Fixture", "arch": "gfx950", "total_memory": 1024}],
            "peer_access": [[True]],
        }
        with ExitStack() as stack:
            stack.enter_context(mock.patch.dict(os.environ, self.environ, clear=True))
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(redirect_stderr(io.StringIO()))
            patches = {
                "aklr_sha": "2" * 40,
                "source_dirty": False,
                "observed_hardware": hardware,
                "host_settings": copy.deepcopy(self.host),
                "prepare_traces": None,
                "client_packages": {"agentx": "1.0.13"},
                "server_packages": {"sglang": "0.5.22"},
                "run_live": self.aggregate,
            }
            calls = {
                name: stack.enter_context(mock.patch.object(runtime, name, return_value=value))
                for name, value in patches.items()
            }
            if measurements is not None:
                calls["run_live"].side_effect = measurements
            code = runtime.main(["--serving", str(serving_path), "--out", str(out), "--run-id", "fixture"])
        return code, json.loads((out / "results.json").read_text()), calls

    def prepare_model_hub(self, ref_content):
        hub = self.root / "hf-hub"
        model_dir = hub / "models--example--model"
        snapshot = model_dir / "snapshots" / self.serving["weights"]["revision"]
        snapshot.mkdir(parents=True)
        main = model_dir / "refs/main"
        main.parent.mkdir()
        main.write_bytes(ref_content)
        args = SimpleNamespace(work=self.root / "work", model_path=None, duration=None)
        return hub, model_dir, snapshot, main, args


@pytest.fixture
def case(tmp_path: Path) -> EnvironmentCase:
    return EnvironmentCase(tmp_path)


@pytest.mark.parametrize("failed_mode", [0, 1], ids=["disabled", "enabled"])
def test_partial_sweep_succeeds_and_retains_point_failures_in_both_modes(case, failed_mode):
    measurements = [case.aggregate, case.aggregate]
    measurements[failed_mode] = RuntimeError("server failed")
    code, results, calls = case.run_main(measurements=measurements)
    assert code == 0
    assert calls["run_live"].call_count == 2
    points = results["control"]["points"] + results["points"]
    assert sum(point["status"] == "ok" for point in points) == 1
    failure = next(point for point in points if point["status"] == "failed")
    assert failure["error"] == "RuntimeError: server failed"
    for target in (results, results["control"]):
        assert target["run"]["completed"]
        assert "error" not in target["run"]
    assert analysis.compare_aklr(results)["status"] != "ok"


def test_sweep_fails_when_neither_mode_has_a_successful_point(case):
    code, results, calls = case.run_main(
        measurements=[RuntimeError("disabled failed"), RuntimeError("enabled failed")]
    )
    assert code == 1
    assert calls["run_live"].call_count == 2
    assert results["control"]["points"][0]["error"] == "RuntimeError: disabled failed"
    assert results["points"][0]["error"] == "RuntimeError: enabled failed"
    assert results["run"]["completed"]
    assert results["control"]["run"]["completed"]


@pytest.mark.parametrize("failed_index", [1, 2], ids=["disabled", "enabled"])
def test_process_cleanup_failure_stops_the_sweep_and_keeps_earlier_measurements(case, failed_index):
    second = copy.deepcopy(case.point)
    second.update(conc=2, exp_name="tp1-c2")
    case.serving["points"].append(second)

    def client(*_args, **kwargs):
        env = kwargs["env"]
        path = Path(env["AGENTIC_OUTPUT_DIR"]) / f"{env['RESULT_FILENAME']}.json"
        path.write_text(json.dumps(case.aggregate))
        return 0

    with (
        mock.patch.object(runtime, "port_free", return_value=True),
        mock.patch.object(runtime, "wait_ready"),
        mock.patch.object(runtime.subprocess, "Popen") as server,
        mock.patch.object(runtime, "run_client", side_effect=client),
        mock.patch.object(
            runtime.os,
            "killpg",
            side_effect=[ProcessLookupError()] * failed_index
            + [PermissionError("cannot terminate process group")],
        ) as kill,
    ):
        code, results, calls = case.run_main(measurements=runtime.run_live)
    points = results["control"]["points"] + results["points"]
    assert code == 1
    assert len(points) == failed_index + 1
    assert sum(point["status"] == "ok" for point in points) == failed_index
    assert points[-1]["status"] == "failed"
    assert points[-1]["cleanup_failed"]
    assert "PermissionError: cannot terminate process group" in points[-1]["error"]
    assert not results["run"]["completed"]
    assert results["run"]["cleanup_failed"]
    assert "cannot terminate process group" in results["run"]["error"]
    assert results["control"]["run"]["completed"] == (failed_index == 2)
    assert calls["run_live"].call_count == failed_index + 1
    assert server.call_count == failed_index + 1
    assert kill.call_count == failed_index + 1
    assert analysis.compare_aklr(results)["status"] != "ok"


def test_client_cleanup_failure_is_distinct_from_a_measurement_failure():
    with (
        mock.patch.object(runtime.subprocess, "Popen") as process,
        mock.patch.object(runtime.os, "killpg", side_effect=PermissionError("cannot stop client")),
    ):
        process.return_value.wait.return_value = 0
        with pytest.raises(runtime.CleanupError, match="PermissionError: cannot stop client"):
            runtime.run_client(["fixture-client"], timeout=1)


def test_verified_defaults_are_read_once_and_not_taken_from_ambient_values(case):
    case.write_inspection()
    with (
        mock.patch.dict(os.environ, case.environ, clear=True),
        mock.patch.object(runtime, "read_json", wraps=runtime.read_json) as reader,
    ):
        identity, defaults = runtime.execution_identity(case.serving)
        reader.assert_called_once_with(str(case.inspection_path))
        case.inspection_path.unlink()
        env = runtime.execution_env(defaults)
    assert identity["image_id"] == IMAGE_ID
    assert defaults == case.defaults
    assert env["SGLANG_USE_AITER"] == "1"
    assert env["ROCM_HOME"] == SDK
    assert env["PATH"] == "/runtime/bin"
    assert env["HF_HUB_CACHE"] == "/mounted/hub"
    assert "ARBITRARY_TUNING" not in env
    assert identity["server_environment"]["defaults_sha256"] == runtime.sha256_json(defaults)


def test_server_overrides_and_client_environment_stay_separate(case):
    out = case.root / "point"
    out.mkdir()
    (out / "tp1-c1.json").write_text(json.dumps(case.aggregate))
    with (
        mock.patch.dict(os.environ, case.environ, clear=True),
        mock.patch.object(runtime, "port_free", return_value=True),
        mock.patch.object(runtime.subprocess, "Popen") as server,
        mock.patch.object(runtime, "wait_ready"),
        mock.patch.object(runtime, "stop"),
        mock.patch.object(runtime, "run_client", return_value=0) as client,
    ):
        assert runtime.run_live(case.point, out, case.host) == case.aggregate
    server_env = server.call_args.kwargs["env"]
    client_env = client.call_args.kwargs["env"]
    assert server_env["SGLANG_USE_AITER"] == "0"
    assert server_env["ROCM_HOME"] == SDK
    assert server_env["PATH"] == "/runtime/bin"
    assert server_env["AKLR_SGLANG_PLUGIN"] == "1"
    assert server_env["AKLR_DISABLE"] == "0"
    assert server_env["HF_TOKEN"] == "live-fixture-token"
    assert client_env["DURATION"] == "60"
    assert client_env["HF_TOKEN"] == "live-fixture-token"
    for key in ("SGLANG_USE_AITER", "ROCM_HOME", "PIP_CONSTRAINT", "VIRTUAL_ENV", "AKLR_SGLANG_PLUGIN"):
        assert key not in client_env
    assert "ARBITRARY_TUNING" not in server_env
    assert "ARBITRARY_TUNING" not in client_env


def test_trace_preparation_and_package_inspection_keep_original_environment(case):
    with (
        mock.patch.dict(os.environ, case.environ, clear=True),
        mock.patch.object(runtime, "run_client", return_value=0) as prepare,
        mock.patch.object(runtime.subprocess, "check_output", return_value="{}") as packages,
    ):
        runtime.prepare_traces(case.host, [case.point], case.root / "prepare.log")
        runtime.client_packages(case.host)
    for env in (prepare.call_args.kwargs["env"], packages.call_args.kwargs["env"]):
        assert env["PATH"] == "/runtime/bin"
        for key in ("SGLANG_USE_AITER", "ROCM_HOME", "PIP_CONSTRAINT", "VIRTUAL_ENV"):
            assert key not in env


def test_exact_model_ref_keeps_pinned_offline_tokenizer(case):
    revision = case.serving["weights"]["revision"].encode()
    hub, model_dir, snapshot, main, args = case.prepare_model_hub(revision)
    with (
        mock.patch.dict(os.environ, {"HF_HUB_CACHE": str(hub)}, clear=True),
        mock.patch.object(runtime, "fetch_inferencex", return_value=case.root / "inferencex"),
    ):
        host = runtime.host_settings(args, case.serving)
    assert main.stat().st_size == 40
    assert host["model_path"] == str(snapshot)
    assert host["tokenizer"] == "example/model"
    assert (host["trace_hub"] / model_dir.name).resolve() == model_dir
    assert main.read_bytes() == revision


def test_model_ref_trailing_newline_is_rejected_before_client_setup(case):
    hub, _, _, main, args = case.prepare_model_hub(case.serving["weights"]["revision"].encode() + b"\n")
    with (
        mock.patch.dict(os.environ, {"HF_HUB_CACHE": str(hub)}, clear=True),
        mock.patch.object(runtime, "fetch_inferencex") as fetch,
    ):
        with pytest.raises(RuntimeError, match="must contain exactly the pinned revision"):
            runtime.host_settings(args, case.serving)
    assert main.stat().st_size == 41
    fetch.assert_not_called()


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"Env": "SGLANG_USE_AITER=1"},
        {"Env": [1]},
        {"Env": ["MISSING_EQUALS"]},
        {"Env": ["=missing-name"]},
        {"Env": ["BAD-NAME=value"]},
        {"Env": ["TOKEN=invalid\0secret"]},
        {"Env": ["A=1", "A=2"]},
    ],
    ids=str,
)
def test_malformed_environment_is_rejected_before_gpu_inspection(case, config):
    inspection = {**case.inspection, "Config": config}
    code, results, calls = case.run_main(inspection)
    assert code == 1
    assert "Config.Env" in results["run"]["error"]
    calls["observed_hardware"].assert_not_called()
    calls["run_live"].assert_not_called()
    assert "secret" not in json.dumps(results)


def test_mismatched_image_is_rejected_before_environment_and_gpu_use(case):
    inspection = {**case.inspection, "RepoDigests": [], "Config": {"Env": ["INVALID"]}}
    code, results, calls = case.run_main(inspection)
    assert code == 1
    assert "planned digest" in results["run"]["error"]
    calls["observed_hardware"].assert_not_called()
    calls["run_live"].assert_not_called()


@pytest.mark.parametrize(
    ("entries", "expected"),
    [(None, {}), ([], {}), (["EMPTY=", "VALUE=a=b"], {"EMPTY": "", "VALUE": "a=b"})],
    ids=str,
)
def test_empty_environment_and_values_with_equals(case, entries, expected):
    case.write_inspection({**case.inspection, "Config": {"Env": entries}})
    with mock.patch.dict(os.environ, case.environ, clear=True):
        _, defaults = runtime.execution_identity(case.serving)
    assert defaults == expected


def test_evidence_redacts_image_credentials_and_excludes_live_credentials(case):
    image_credentials = {
        "HF_TOKEN": "image-fixture-token",
        "HTTPS_PROXY": "https://fixture:image-proxy-password@proxy.invalid",
        "SERVICE_URL": "https://fixture:service-password@service.invalid",
        "DOWNLOAD_URL": "https://service.invalid/file?signature=signed-fixture-secret",
        "AZURE_STORAGE_CONNECTION_STRING": (
            "DefaultEndpointsProtocol=https;AccountName=fixture;AccountKey=image-fixture-account-value"
        ),
        "APP_CONFIG": "opaque-image-fixture-value",
    }
    case.inspection["Config"]["Env"].extend(f"{key}={value}" for key, value in image_credentials.items())
    case.write_inspection()
    with mock.patch.dict(os.environ, case.environ, clear=True):
        identity, defaults = runtime.execution_identity(case.serving)
    for key, value in image_credentials.items():
        assert defaults[key] == value
        assert identity["server_environment"]["defaults"][key] == "<redacted>"
    code, results, calls = case.run_main()
    assert code == 0
    serialized = json.dumps(results)
    for value in [*image_credentials.values(), case.environ["HF_TOKEN"], case.environ["HTTPS_PROXY"]]:
        assert value not in serialized
    for call in calls["run_live"].call_args_list:
        for name, value in image_credentials.items():
            assert call.args[2]["server_image_env"][name] == value


@pytest.mark.parametrize("name", ["SGLANG_USE_AITER", "APP_CONFIG"])
def test_redacted_image_values_still_determine_the_complete_environment_digest(case, name):
    defaults = {**case.defaults, "APP_CONFIG": "opaque-image-fixture-value"}
    original = defaults.copy()
    evidence = runtime.image_environment_evidence(defaults)
    assert defaults == original
    assert evidence["defaults"] == {name: "<redacted>" for name in defaults}
    assert evidence["defaults_sha256"] == runtime.sha256_json(defaults)
    assert runtime.image_environment_evidence(dict(reversed(list(defaults.items())))) == evidence
    changed = runtime.image_environment_evidence({**defaults, name: "changed-fixture-value"})
    assert changed["defaults"] == evidence["defaults"]
    assert changed["defaults_sha256"] != evidence["defaults_sha256"]


def test_paired_measurements_share_defaults_and_separate_legacy_environment(case):
    code, results, calls = case.run_main()
    assert code == 0
    assert analysis.compare_aklr(results)["status"] == "ok"
    for call in calls["run_live"].call_args_list:
        assert call.args[2]["server_image_env"] == case.defaults
    for target in (results, results["control"]):
        measurement = target["points"][0]["measurement"]
        assert measurement["environment_policy"] == "paired-aklr-image-env-v2"
        assert measurement["server_image_env_sha256"] == runtime.sha256_json(case.defaults)

    legacy = copy.deepcopy(results)
    for target in (legacy, legacy["control"]):
        point = target["points"][0]
        point["measurement"]["environment_policy"] = "paired-aklr-v1"
        del point["measurement"]["server_image_env_sha256"]
        target["serving"]["arms"]["tp1"]["arm_measurement_sha256"] = runtime.sha256_json(
            {point["conc"]: point["measurement"]}
        )
    assert analysis.compare(results, legacy)["arms"]["tp1"]["status"] == "new-segment"
