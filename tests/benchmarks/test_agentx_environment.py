# SPDX-License-Identifier: MIT
"""AgentX image environment boundaries, without GPU dependencies."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
from agentx_ci import performance_analysis as analysis
from agentx_ci import runtime, serving_config

IMAGE_ID = "sha256:" + "a" * 64
IMAGE_REFERENCE = "example.invalid/server@sha256:" + "b" * 64
SDK = "/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel"


class EnvironmentCase:
    def __init__(self, root: Path, gpus_idle: mock.Mock) -> None:
        self.root = root
        self.gpus_idle = gpus_idle
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
                "FRAMEWORK": "sglang",
                "TOTAL_CPU_DRAM_GB": "16",
                "DURATION": "60",
                "CONC": "1",
                "AIPERF_EXPERIMENTAL_FAST": "0",
                "WEKA_LOADER_OVERRIDE": "fixture",
            },
            "runtime": {"port": 8888, "ready_timeout_s": 30, "point_timeout_s": 60, "poll_s": 0.1},
            "hashes": {"serve_spec_sha256": "c" * 64},
            "trace": {"repo": "example/traces", "revision": "d" * 40, "loader": "fixture"},
        }
        self.serving = {
            "schema": 4,
            "model": {"id": "fixture"},
            "target": {"id": "fixture-gpu"},
            "framework": {"id": "sglang"},
            "image": {"run": IMAGE_REFERENCE},
            "weights": {"repo": "example/model", "revision": "e" * 40},
            "hardware": {"gpu_count": 1, "gpu_name_pattern": "Fixture", "gpu_arch": "gfx950"},
            "source": {"kind": "inferencex", "repo": "example/source", "sha": "f" * 40},
            "client": {"repo": "example/source", "sha": "f" * 40},
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
            "num_gpus": 1,
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

    def use_environ(self, environ=None):
        """The process environment as exactly ``environ``, for the ``with`` block only."""
        return mock.patch.dict(os.environ, self.environ if environ is None else environ, clear=True)

    def write_inspection(self, inspection=None):
        self.inspection_path.write_text(json.dumps(self.inspection if inspection is None else inspection))

    def run_main(self, inspection=None, *, measurements=None):
        self.write_inspection(inspection)
        serving_path = self.root / "serving.json"
        serving_path.write_text(
            json.dumps({**self.serving, "selection": serving_config.full_selection(self.serving)})
        )
        out = self.root / "results"
        hardware = {
            "devices": [{"name": "Fixture", "arch": "gfx950", "total_memory": 1024}],
            "peer_access": [[True]],
        }
        with self.use_environ(), pytest.MonkeyPatch.context() as patch:
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
            calls = {name: mock.Mock(return_value=value) for name, value in patches.items()}
            for name, double in calls.items():
                patch.setattr(runtime, name, double)
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
def case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    gpus_idle = mock.Mock(return_value=0.0)
    monkeypatch.setattr(runtime, "wait_gpus_idle", gpus_idle)
    return EnvironmentCase(tmp_path, gpus_idle)


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


def test_vllm_sweep_switches_the_vllm_plugin_in_each_mode(case):
    case.serving["framework"] = {"id": "vllm"}
    case.point["client_env"]["FRAMEWORK"] = "vllm"
    case.point["launch"] = {"command": ["vllm", "serve"], "env": {"VLLM_ROCM_USE_AITER": "1"}}
    code, results, calls = case.run_main()
    assert code == 0
    calls["server_packages"].assert_called_once_with("vllm")
    for target, switches in ((results["control"], ("0", "1")), (results, ("1", "0"))):
        env = target["points"][0]["measurement"]["launch"]["env"]
        assert (env["AKLR_VLLM_PLUGIN"], env["AKLR_DISABLE"]) == switches
        assert "AKLR_SGLANG_PLUGIN" not in env
    assert analysis.aklr_pairing_error(results) is None
    env = results["points"][0]["measurement"]["launch"]["env"]
    env["AKLR_SGLANG_PLUGIN"] = env.pop("AKLR_VLLM_PLUGIN")
    assert analysis.aklr_pairing_error(results) == "AKLR mode switches do not match their labels."


def test_points_outside_the_plan_framework_fail_before_traces_and_servers(case):
    case.point["client_env"]["FRAMEWORK"] = "vllm"
    code, results, calls = case.run_main()
    assert code == 1
    assert "do not use the plan's framework 'sglang'" in results["run"]["error"]
    calls["prepare_traces"].assert_not_called()
    calls["run_live"].assert_not_called()


def test_unsupported_framework_fails_before_traces_and_servers(case):
    case.point["client_env"]["FRAMEWORK"] = "trtllm"
    code, results, calls = case.run_main()
    assert code == 1
    assert "unsupported serving framework 'trtllm'" in results["run"]["error"]
    calls["prepare_traces"].assert_not_called()
    calls["run_live"].assert_not_called()


@pytest.mark.parametrize("failed_index", [1, 2], ids=["disabled", "enabled"])
def test_process_cleanup_failure_stops_the_sweep_and_keeps_earlier_measurements(case, failed_index):
    second = copy.deepcopy(case.point)
    second.update(conc=2, exp_name="tp1-c2")
    second["client_env"]["CONC"] = "2"
    case.serving["points"].append(second)

    def client(*_args, **kwargs):
        env = kwargs["env"]
        path = Path(env["AGENTIC_OUTPUT_DIR"]) / f"{env['RESULT_FILENAME']}.json"
        path.write_text(json.dumps(case.aggregate))
        return 0

    server = mock.Mock()
    kill = mock.Mock(
        side_effect=[ProcessLookupError()] * failed_index
        + [PermissionError("cannot terminate process group")]
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runtime, "port_free", mock.Mock(return_value=True))
        patch.setattr(runtime, "wait_ready", mock.Mock())
        patch.setattr(runtime.subprocess, "Popen", server)
        patch.setattr(runtime, "run_client", client)
        patch.setattr(runtime.os, "killpg", kill)
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
    process = mock.Mock()
    process.return_value.wait.return_value = 0
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runtime.subprocess, "Popen", process)
        patch.setattr(runtime.os, "killpg", mock.Mock(side_effect=PermissionError("cannot stop client")))
        with pytest.raises(runtime.CleanupError, match="PermissionError: cannot stop client"):
            runtime.run_client(["fixture-client"], timeout=1)


def test_verified_defaults_are_read_once_and_not_taken_from_ambient_values(case):
    case.write_inspection()
    reader = mock.Mock(wraps=runtime.read_json)
    with case.use_environ(), pytest.MonkeyPatch.context() as patch:
        patch.setattr(runtime, "read_json", reader)
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
    server, client = mock.Mock(), mock.Mock(return_value=0)
    with case.use_environ(), pytest.MonkeyPatch.context() as patch:
        patch.setattr(runtime, "port_free", mock.Mock(return_value=True))
        patch.setattr(runtime.subprocess, "Popen", server)
        patch.setattr(runtime, "wait_ready", mock.Mock())
        patch.setattr(runtime, "stop", mock.Mock())
        patch.setattr(runtime, "run_client", client)
        assert runtime.run_live(case.point, out, case.host) == case.aggregate
    case.gpus_idle.assert_called_once_with()
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
    prepare, packages = mock.Mock(return_value=0), mock.Mock(return_value="{}")
    with case.use_environ(), pytest.MonkeyPatch.context() as patch:
        patch.setattr(runtime, "run_client", prepare)
        patch.setattr(runtime.subprocess, "check_output", packages)
        patch.setattr(runtime, "verify_trace_loaders", mock.Mock())
        runtime.prepare_traces(case.host, [case.point], case.root / "prepare.log")
        runtime.client_packages(case.host)
    for env in (prepare.call_args.kwargs["env"], packages.call_args.kwargs["env"]):
        assert env["PATH"] == "/runtime/bin"
        for key in ("SGLANG_USE_AITER", "ROCM_HOME", "PIP_CONSTRAINT", "VIRTUAL_ENV"):
            assert key not in env


@pytest.mark.parametrize(
    ("loader", "message"),
    [("capped", "replays 'example/traces-256k'"), ("unknown", "rejects the trace loaders")],
    ids=["capped", "unknown"],
)
def test_trace_loaders_must_replay_the_recorded_trace_dataset(case, loader, message):
    client = case.root / "client"
    package = client / "infx/bench/agentic"
    package.mkdir(parents=True)
    for directory in (client / "infx", client / "infx/bench", package):
        (directory / "__init__.py").write_text("")
    (package / "traces.py").write_text(
        "LOADERS = {'fixture': 'example/traces', 'capped': 'example/traces-256k'}\n"
        "def resolve(prefix, override):\n"
        "    if override not in LOADERS:\n"
        "        raise ValueError(f'unknown WEKA_LOADER_OVERRIDE={override!r}')\n"
        "    return override, LOADERS[override]\n"
    )
    host = {**case.host, "inferencex_e2e": str(client)}
    runtime.verify_trace_loaders(host, [case.point])
    point = copy.deepcopy(case.point)
    point["client_env"]["WEKA_LOADER_OVERRIDE"] = loader
    with pytest.raises(RuntimeError, match=message):
        runtime.verify_trace_loaders(host, [point])


def test_exact_model_ref_keeps_pinned_offline_tokenizer(case):
    revision = case.serving["weights"]["revision"].encode()
    hub, model_dir, snapshot, main, args = case.prepare_model_hub(revision)
    with case.use_environ({"HF_HUB_CACHE": str(hub)}), pytest.MonkeyPatch.context() as patch:
        patch.setattr(runtime, "fetch_inferencex", mock.Mock(return_value=case.root / "inferencex"))
        host = runtime.host_settings(args, case.serving)
    assert main.stat().st_size == 40
    assert host["model_path"] == str(snapshot)
    assert host["tokenizer"] == "example/model"
    assert (host["trace_hub"] / model_dir.name).resolve() == model_dir
    assert main.read_bytes() == revision


def test_model_ref_trailing_newline_is_rejected_before_client_setup(case):
    hub, _, _, main, args = case.prepare_model_hub(case.serving["weights"]["revision"].encode() + b"\n")
    fetch = mock.Mock()
    with (
        case.use_environ({"HF_HUB_CACHE": str(hub)}),
        pytest.MonkeyPatch.context() as patch,
        pytest.raises(RuntimeError, match="must contain exactly the pinned revision"),
    ):
        patch.setattr(runtime, "fetch_inferencex", fetch)
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
    ids=[
        "no-env",
        "env-not-a-list",
        "entry-not-a-string",
        "missing-equals",
        "missing-name",
        "bad-name",
        "nul-in-value",
        "duplicate-name",
    ],
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
    ids=["null", "empty", "values"],
)
def test_empty_environment_and_values_with_equals(case, entries, expected):
    case.write_inspection({**case.inspection, "Config": {"Env": entries}})
    with case.use_environ():
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
    with case.use_environ():
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
    assert analysis.compare(results, legacy)["arms"]["tp1"]["status"] == "curve"


def amd_smi(*used_mb, wrap=False):
    gpus = [
        {
            "gpu": index,
            "mem_usage": {"total_vram": {"value": 294896, "unit": "MB"}, "used_vram": {"value": used}},
        }
        for index, used in enumerate(used_mb)
    ]
    stdout = "WARNING: fixture banner\n" + json.dumps({"gpu_data": gpus} if wrap else gpus)
    return SimpleNamespace(returncode=0, stdout=stdout, stderr="")


@pytest.mark.parametrize("wrap", [False, True], ids=["list", "wrapped"])
def test_vram_use_reads_whole_percent_from_either_amd_smi_layout(monkeypatch, wrap):
    monkeypatch.setattr(
        runtime.subprocess, "run", mock.Mock(return_value=amd_smi(283, 2949, 294896, wrap=wrap))
    )
    assert runtime.vram_used_percent() == {0: 0, 1: 1, 2: 100}


@pytest.mark.parametrize(
    ("message", "outcome"),
    [
        ("could not run", FileNotFoundError("amd-smi")),
        ("exited with code 2", SimpleNamespace(returncode=2, stdout="", stderr="no GPUs")),
        ("unreadable", SimpleNamespace(returncode=0, stdout='[{"gpu": 0}]', stderr="")),
        ("listed no GPUs", SimpleNamespace(returncode=0, stdout="[]", stderr="")),
    ],
    ids=["not-runnable", "nonzero-exit", "unreadable", "no-gpus"],
)
def test_unreadable_amd_smi_aborts_the_sweep(monkeypatch, message, outcome):
    effect = {"side_effect": outcome} if isinstance(outcome, Exception) else {"return_value": outcome}
    monkeypatch.setattr(runtime.subprocess, "run", mock.Mock(**effect))
    with pytest.raises(runtime.CleanupError, match=message):
        runtime.vram_used_percent()


def test_wait_returns_once_every_gpu_reads_zero():
    read = mock.Mock(side_effect=[{0: 0, 1: 37}, {0: 0, 1: 2}, {0: 0, 1: 0}])
    sleep = mock.Mock()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runtime, "vram_used_percent", read)
        patch.setattr(runtime.time, "sleep", sleep)
        runtime.wait_gpus_idle(timeout_s=300, poll_s=5)
    assert read.call_count == 3
    assert sleep.call_count == 2


def test_wait_times_out_naming_the_busy_gpus():
    clock = iter([0, 0, 200, 301])
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runtime, "vram_used_percent", mock.Mock(return_value={0: 0, 5: 12}))
        patch.setattr(runtime.time, "monotonic", lambda: next(clock))
        patch.setattr(runtime.time, "sleep", mock.Mock())
        with pytest.raises(runtime.CleanupError, match=r"after 300 s: GPU5 12%"):
            runtime.wait_gpus_idle(timeout_s=300, poll_s=5)
