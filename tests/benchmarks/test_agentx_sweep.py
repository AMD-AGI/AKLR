# SPDX-License-Identifier: MIT
"""Sweep outcomes and CI summaries without launching a GPU container."""

from __future__ import annotations

import copy
import importlib
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

with mock.patch.object(
    sys, "path", [str(Path(__file__).resolve().parents[2] / "benchmarks/agentx"), *sys.path]
):
    sweep = importlib.import_module("agentx_ci.sweep")
    docker_launch = importlib.import_module("agentx_ci.docker_launch")
    serving_config = importlib.import_module("agentx_ci.serving_config")


class AgentXSweepTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.summary_path = self.root / "summary.md"
        self.enterContext(mock.patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": str(self.summary_path)}))
        self.enterContext(redirect_stdout(io.StringIO()))
        target = {
            "run": {"completed": True, "points_expected": 2},
            "points": [
                {
                    "arm": "tp8",
                    "conc": conc,
                    "status": "ok",
                    "tput_per_gpu": 100,
                    "p90_full_response_intvty": 20,
                    "raw_data": {
                        "request_metrics": {
                            "throughput": {"per_gpu": {"total_tput_tps": 100}},
                            "latency": {"full_response_itl": {"p90": 0.05}},
                        }
                    },
                }
                for conc in (1, 2)
            ],
        }
        self.results = {**copy.deepcopy(target), "control": copy.deepcopy(target)}

    def launch(self, code=0, *, save=True):
        before = json.dumps(self.results)

        def run(args, _options):
            if save:
                (args.out / "results.json").write_text(before)
            return code

        with mock.patch.object(docker_launch, "run", side_effect=run):
            result = sweep.main(["--serving", str(self.root / "serving.json"), "--out", str(self.root)])
        if save:
            self.assertEqual((self.root / "results.json").read_text(), before)
        return result, self.summary_path.read_text()

    def test_complete_sweep_reports_success(self):
        code, summary = self.launch()
        self.assertEqual(code, 0)
        self.assertIn("AgentX sweep: Success", summary)
        self.assertIn("4/4 measurement points successful", summary)

    def test_partial_sweep_counts_both_modes_and_preserves_errors(self):
        self.results["points"][0].update(status="failed", error="server failed before ready")
        code, summary = self.launch()
        self.assertEqual(code, 0)
        self.assertIn("AgentX sweep: Partial success", summary)
        self.assertIn("3/4 measurement points successful", summary)
        self.assertIn("| enabled | 1 | 1 | 0 | 2 |", summary)
        self.assertIn("| disabled | 2 | 0 | 0 | 2 |", summary)

    def test_failed_or_invalid_points_do_not_count_as_successful(self):
        for target in (self.results, self.results["control"]):
            target["points"][0]["status"] = "failed"
            target["points"][1]["tput_per_gpu"] = 0
        code, summary = self.launch()
        self.assertEqual(code, 1)
        self.assertIn("AgentX sweep: Failed", summary)
        self.assertIn("0/4 measurement points successful", summary)

    def test_missing_measurements_are_distinct_from_failed_points(self):
        self.results["run"]["completed"] = False
        self.results["points"] = []
        code, summary = self.launch()
        self.assertEqual(code, 0)
        self.assertIn("AgentX sweep: Partial success", summary)
        self.assertIn("2/4 measurement points successful", summary)
        self.assertIn("| enabled | 0 | 0 | 2 | 2 |", summary)

    def test_process_failure_and_cancellation_are_not_hidden_by_saved_points(self):
        self.results["points"].pop()
        self.results["run"]["completed"] = False
        for process_code in (1, 130, 137, 143):
            with self.subTest(process_code=process_code):
                code, summary = self.launch(process_code)
                self.assertEqual(code, process_code)
                self.assertIn("AgentX sweep: Failed", summary)
                self.assertNotIn("AgentX sweep: Partial success", summary)
                self.assertIn("3/4 measurement points successful", summary)
                self.assertIn("| enabled | 1 | 0 | 1 | 2 |", summary)
                self.assertIn("| disabled | 2 | 0 | 0 | 2 |", summary)
                self.assertIn(f"Sweep process exited with status {process_code}", summary)

    def test_process_failure_is_failed_even_when_all_points_succeeded(self):
        code, summary = self.launch(1)
        self.assertEqual(code, 1)
        self.assertIn("AgentX sweep: Failed", summary)
        self.assertNotIn("AgentX sweep: Partial success", summary)
        self.assertIn("4/4 measurement points successful", summary)
        self.assertIn("| enabled | 2 | 0 | 0 | 2 |", summary)
        self.assertIn("| disabled | 2 | 0 | 0 | 2 |", summary)

    def test_missing_results_cannot_turn_a_process_exit_into_success(self):
        code, summary = self.launch(save=False)
        self.assertEqual(code, 1)
        self.assertIn("No usable measurement results were saved", summary)

    def test_duplicate_measurements_are_not_reported_as_successful(self):
        self.results["points"][1] = copy.deepcopy(self.results["points"][0])
        code, summary = self.launch()
        self.assertEqual(code, 1)
        self.assertIn("No usable measurement results were saved", summary)


class AgentXDockerOutcomeTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(redirect_stdout(io.StringIO()))
        self.enterContext(redirect_stderr(io.StringIO()))
        self.serving = {
            "schema": 4,
            "model": {"id": "fixture"},
            "target": {"id": "mi355x"},
            "framework": {"id": "sglang"},
            "image": {"run": "example.invalid/server@sha256:" + "a" * 64},
            "hardware": {"gpu_count": 8},
            "weights": {"repo": "example/model", "revision": "b" * 40},
            "source": {"kind": "inferencex", "repo": "example/source", "sha": "c" * 40},
            "client": {"repo": "example/source", "sha": "c" * 40},
            "grid_sha256": "d" * 64,
            "arms": {"tp8": {}},
            "points": [
                {
                    "arm": "tp8",
                    "conc": 1,
                    "client_env": {"FRAMEWORK": "sglang"},
                    "hashes": {"serve_spec_sha256": "e" * 64},
                }
            ],
        }
        serving_path = self.root / "serving.json"
        serving_path.write_text(
            json.dumps({**self.serving, "selection": serving_config.full_selection(self.serving)})
        )
        self.args = SimpleNamespace(
            out=self.root / "out",
            cache=None,
            work=None,
            serving=serving_path,
            run_id="fixture",
            hub=self.root,
            image=None,
        )

    def launch(self, *, completed):
        self.commands = []

        def command(argv, **_kwargs):
            self.commands.append(argv)
            if argv[:3] == ["docker", "start", "--attach"]:
                current = json.loads((self.args.out / "results.json").read_text())
                for target in (current, current["control"]):
                    target["run"]["completed"] = completed
                    target["points"] = [
                        {"arm": "tp8", "conc": 1, "status": "failed", "error": "server startup failed"}
                    ]
                if not completed:
                    current["control"]["points"][0].update(
                        status="ok",
                        error=None,
                        tput_per_gpu=100,
                        p90_full_response_intvty=20,
                        raw_data={
                            "request_metrics": {
                                "throughput": {"per_gpu": {"total_tput_tps": 100}},
                                "latency": {"full_response_itl": {"p90": 0.05}},
                            }
                        },
                    )
                (self.args.out / "results.json").write_text(json.dumps(current))
                return 1, "container log tail"
            if "status" in argv:
                return 0, ""
            return 0, str(self.root / "git-metadata")

        with (
            mock.patch.object(docker_launch.Launcher, "command", side_effect=command),
            mock.patch.object(docker_launch.Launcher, "image", return_value=("sha256:fixture", {})),
            mock.patch.object(docker_launch.Launcher, "cleanup", return_value=None),
        ):
            code = docker_launch.run(self.args, [])
        return code, json.loads((self.args.out / "results.json").read_text())

    def test_all_failed_measurements_keep_completion_and_individual_errors(self):
        code, results = self.launch(completed=True)
        self.assertEqual(code, 1)
        for target in (results, results["control"]):
            self.assertTrue(target["run"]["completed"])
            self.assertNotIn("error", target["run"])
            self.assertEqual(target["points"][0]["error"], "server startup failed")
        self.assertIn("All measurement points failed", (self.args.out / "launcher.log").read_text())

    def test_container_runs_bash_instead_of_the_image_entrypoint(self):
        self.launch(completed=True)
        create = next(argv for argv in self.commands if argv[:2] == ["docker", "create"])
        self.assertEqual(create[create.index("--entrypoint") + 1], "bash")
        image = create.index("sha256:fixture")
        self.assertEqual(create[image + 1 : image + 3], ["-c", docker_launch.CONTAINER_COMMAND])
        self.assertIn("install_aklr.sh /agentx/cache /agentx/serving.json", docker_launch.CONTAINER_COMMAND)

    def test_container_failure_remains_failed_even_after_a_successful_point(self):
        code, results = self.launch(completed=False)
        self.assertEqual(code, 1)
        self.assertFalse(results["run"]["completed"])
        self.assertIn("Container exited with status 1", results["run"]["error"])
        self.assertEqual(results["control"]["points"][0]["status"], "ok")


if __name__ == "__main__":
    unittest.main()
