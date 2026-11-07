# SPDX-License-Identifier: MIT
"""AgentX image environment boundaries; runnable with unittest without GPU dependencies."""

from __future__ import annotations

import copy
import importlib
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

with mock.patch.object(
    sys, "path", [str(Path(__file__).resolve().parents[2] / "benchmarks/agentx"), *sys.path]
):
    runtime = importlib.import_module("agentx_ci.runtime")
    analysis = importlib.import_module("agentx_ci.performance_analysis")

IMAGE_ID = "sha256:" + "a" * 64
IMAGE_REFERENCE = "example.invalid/server@sha256:" + "b" * 64
SDK = "/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel"


class AgentXEnvironmentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
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
                "latency": {"e2e_norm_intvty": {"p90": 10}},
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

    def run_main(self, inspection=None):
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
            code = runtime.main(["--serving", str(serving_path), "--out", str(out), "--run-id", "fixture"])
        return code, json.loads((out / "results.json").read_text()), calls

    def test_verified_defaults_are_read_once_and_not_taken_from_ambient_values(self):
        self.write_inspection()
        with (
            mock.patch.dict(os.environ, self.environ, clear=True),
            mock.patch.object(runtime, "read_json", wraps=runtime.read_json) as reader,
        ):
            identity, defaults = runtime.execution_identity(self.serving)
            reader.assert_called_once_with(str(self.inspection_path))
            self.inspection_path.unlink()
            env = runtime.execution_env(defaults)
        self.assertEqual(identity["image_id"], IMAGE_ID)
        self.assertEqual(defaults, self.defaults)
        self.assertEqual(env["SGLANG_USE_AITER"], "1")
        self.assertEqual(env["ROCM_HOME"], SDK)
        self.assertEqual(env["PATH"], "/runtime/bin")
        self.assertEqual(env["HF_HUB_CACHE"], "/mounted/hub")
        self.assertNotIn("ARBITRARY_TUNING", env)
        self.assertEqual(identity["server_environment"]["defaults_sha256"], runtime.sha256_json(defaults))

    def test_server_overrides_and_client_environment_stay_separate(self):
        out = self.root / "point"
        out.mkdir()
        (out / "tp1-c1.json").write_text(json.dumps(self.aggregate))
        with (
            mock.patch.dict(os.environ, self.environ, clear=True),
            mock.patch.object(runtime, "port_free", return_value=True),
            mock.patch.object(runtime.subprocess, "Popen") as server,
            mock.patch.object(runtime, "wait_ready"),
            mock.patch.object(runtime, "stop"),
            mock.patch.object(runtime, "run_client", return_value=0) as client,
        ):
            self.assertEqual(runtime.run_live(self.point, out, self.host), self.aggregate)
        server_env = server.call_args.kwargs["env"]
        client_env = client.call_args.kwargs["env"]
        self.assertEqual(server_env["SGLANG_USE_AITER"], "0")
        self.assertEqual(server_env["ROCM_HOME"], SDK)
        self.assertEqual(server_env["PATH"], "/runtime/bin")
        self.assertEqual(server_env["AKLR_SGLANG_PLUGIN"], "1")
        self.assertEqual(server_env["AKLR_DISABLE"], "0")
        self.assertEqual(server_env["HF_TOKEN"], "live-fixture-token")
        self.assertEqual(client_env["DURATION"], "60")
        self.assertEqual(client_env["HF_TOKEN"], "live-fixture-token")
        for key in ("SGLANG_USE_AITER", "ROCM_HOME", "PIP_CONSTRAINT", "VIRTUAL_ENV", "AKLR_SGLANG_PLUGIN"):
            self.assertNotIn(key, client_env)
        self.assertNotIn("ARBITRARY_TUNING", server_env)
        self.assertNotIn("ARBITRARY_TUNING", client_env)

    def test_trace_preparation_and_package_inspection_keep_original_environment(self):
        with (
            mock.patch.dict(os.environ, self.environ, clear=True),
            mock.patch.object(runtime, "run_client", return_value=0) as prepare,
            mock.patch.object(runtime.subprocess, "check_output", return_value="{}") as packages,
        ):
            runtime.prepare_traces(self.host, [self.point], self.root / "prepare.log")
            runtime.client_packages(self.host)
        for env in (prepare.call_args.kwargs["env"], packages.call_args.kwargs["env"]):
            self.assertEqual(env["PATH"], "/runtime/bin")
            for key in ("SGLANG_USE_AITER", "ROCM_HOME", "PIP_CONSTRAINT", "VIRTUAL_ENV"):
                self.assertNotIn(key, env)

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

    def test_exact_model_ref_keeps_pinned_offline_tokenizer(self):
        revision = self.serving["weights"]["revision"].encode()
        hub, model_dir, snapshot, main, args = self.prepare_model_hub(revision)
        with (
            mock.patch.dict(os.environ, {"HF_HUB_CACHE": str(hub)}, clear=True),
            mock.patch.object(runtime, "fetch_inferencex", return_value=self.root / "inferencex"),
        ):
            host = runtime.host_settings(args, self.serving)
        self.assertEqual(main.stat().st_size, 40)
        self.assertEqual(host["model_path"], str(snapshot))
        self.assertEqual(host["tokenizer"], "example/model")
        self.assertEqual((host["trace_hub"] / model_dir.name).resolve(), model_dir)
        self.assertEqual(main.read_bytes(), revision)

    def test_model_ref_trailing_newline_is_rejected_before_client_setup(self):
        hub, _, _, main, args = self.prepare_model_hub(self.serving["weights"]["revision"].encode() + b"\n")
        with (
            mock.patch.dict(os.environ, {"HF_HUB_CACHE": str(hub)}, clear=True),
            mock.patch.object(runtime, "fetch_inferencex") as fetch,
        ):
            with self.assertRaisesRegex(RuntimeError, "must contain exactly the pinned revision"):
                runtime.host_settings(args, self.serving)
        self.assertEqual(main.stat().st_size, 41)
        fetch.assert_not_called()

    def test_malformed_environment_is_rejected_before_gpu_inspection(self):
        invalid = [
            {},
            {"Env": "SGLANG_USE_AITER=1"},
            {"Env": [1]},
            {"Env": ["MISSING_EQUALS"]},
            {"Env": ["=missing-name"]},
            {"Env": ["BAD-NAME=value"]},
            {"Env": ["TOKEN=invalid\0secret"]},
            {"Env": ["A=1", "A=2"]},
        ]
        for config in invalid:
            with self.subTest(config=config):
                inspection = {**self.inspection, "Config": config}
                code, results, calls = self.run_main(inspection)
                self.assertEqual(code, 1)
                self.assertIn("Config.Env", results["run"]["error"])
                calls["observed_hardware"].assert_not_called()
                calls["run_live"].assert_not_called()
                self.assertNotIn("secret", json.dumps(results))

    def test_mismatched_image_is_rejected_before_environment_and_gpu_use(self):
        inspection = {**self.inspection, "RepoDigests": [], "Config": {"Env": ["INVALID"]}}
        code, results, calls = self.run_main(inspection)
        self.assertEqual(code, 1)
        self.assertIn("planned digest", results["run"]["error"])
        calls["observed_hardware"].assert_not_called()
        calls["run_live"].assert_not_called()

    def test_empty_environment_and_values_with_equals(self):
        for entries, expected in (
            (None, {}),
            ([], {}),
            (["EMPTY=", "VALUE=a=b"], {"EMPTY": "", "VALUE": "a=b"}),
        ):
            with self.subTest(entries=entries):
                self.write_inspection({**self.inspection, "Config": {"Env": entries}})
                with mock.patch.dict(os.environ, self.environ, clear=True):
                    _, defaults = runtime.execution_identity(self.serving)
                self.assertEqual(defaults, expected)

    def test_evidence_redacts_image_credentials_and_excludes_live_credentials(self):
        image_credentials = {
            "HF_TOKEN": "image-fixture-token",
            "HTTPS_PROXY": "https://fixture:image-proxy-password@proxy.invalid",
            "SERVICE_URL": "https://fixture:service-password@service.invalid",
            "DOWNLOAD_URL": "https://service.invalid/file?signature=signed-fixture-secret",
        }
        self.inspection["Config"]["Env"].extend(f"{key}={value}" for key, value in image_credentials.items())
        self.write_inspection()
        with mock.patch.dict(os.environ, self.environ, clear=True):
            identity, defaults = runtime.execution_identity(self.serving)
        for key, value in image_credentials.items():
            self.assertEqual(defaults[key], value)
            self.assertEqual(identity["server_environment"]["defaults"][key], "<redacted>")
        code, results, _ = self.run_main()
        self.assertEqual(code, 0)
        serialized = json.dumps(results)
        for value in [*image_credentials.values(), self.environ["HF_TOKEN"], self.environ["HTTPS_PROXY"]]:
            self.assertNotIn(value, serialized)

    def test_paired_measurements_share_defaults_and_separate_legacy_environment(self):
        code, results, calls = self.run_main()
        self.assertEqual(code, 0)
        self.assertEqual(analysis.compare_aklr(results)["status"], "ok")
        for call in calls["run_live"].call_args_list:
            self.assertEqual(call.args[2]["server_image_env"], self.defaults)
        for target in (results, results["control"]):
            measurement = target["points"][0]["measurement"]
            self.assertEqual(measurement["environment_policy"], "paired-aklr-image-env-v2")
            self.assertEqual(measurement["server_image_env_sha256"], runtime.sha256_json(self.defaults))

        legacy = copy.deepcopy(results)
        for target in (legacy, legacy["control"]):
            point = target["points"][0]
            point["measurement"]["environment_policy"] = "paired-aklr-v1"
            del point["measurement"]["server_image_env_sha256"]
            target["serving"]["arms"]["tp1"]["arm_measurement_sha256"] = runtime.sha256_json(
                {point["conc"]: point["measurement"]}
            )
        self.assertEqual(analysis.compare(results, legacy)["arms"]["tp1"]["status"], "new-segment")


if __name__ == "__main__":
    unittest.main()
