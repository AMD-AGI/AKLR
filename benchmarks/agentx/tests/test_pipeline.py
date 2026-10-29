# SPDX-License-Identifier: MIT
"""Mock sweep → report, end to end in a temporary directory, plus the PR lookup."""

from __future__ import annotations

import contextlib
import io
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agentx_ci import prs, report, sweep
from agentx_ci.common import load_config, read_json, write_json

from tests.test_recipe_resolve import build


def quiet(main, argv):
    with contextlib.redirect_stdout(io.StringIO()):
        return main(argv)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.serving = write_json(self.root / "serving.json", build())

    def sweep(self, name, *extra):
        out = self.root / name
        argv = [
            "--serving",
            str(self.serving),
            "--out",
            str(out),
            "--mock",
            "--run-id",
            name,
            *extra,
        ]
        self.assertEqual(quiet(sweep.main, argv), 0)
        return out / "results.json"

    def test_mock_sweep_writes_every_point(self):
        results = read_json(self.sweep("w1"))
        self.assertEqual(results["run"]["mode"], "mock")
        self.assertEqual(len(results["points"]), 9)
        self.assertTrue(
            all(
                p["status"] == "ok" and p["tput_per_gpu"] > 0 for p in results["points"]
            )
        )

    def test_arm_filter_and_scale(self):
        base = read_json(self.sweep("w1", "--arms", "tp8"))
        scaled = read_json(self.sweep("w2", "--arms", "tp8", "--mock-scale", "tp8=2"))
        self.assertEqual({p["arm"] for p in base["points"]}, {"tp8"})
        ratio = scaled["points"][0]["tput_per_gpu"] / base["points"][0]["tput_per_gpu"]
        self.assertAlmostEqual(ratio, 2.0, places=2)

    def test_report_compares_with_the_previous_run(self):
        site1, site2 = self.root / "site1", self.root / "site2"
        quiet(
            report.main,
            ["--results", str(self.sweep("w1")), "--out", str(site1), "--no-prs"],
        )
        w2 = self.sweep("w2", "--mock-seed", "1", "--mock-scale", "tp4=1.5")
        argv = [
            "--results",
            str(w2),
            "--baseline",
            str(site1 / "results.json"),
            "--out",
            str(site2),
            "--no-prs",
        ]
        quiet(report.main, argv)
        html = (site2 / "index.html").read_text()
        self.assertIn("MOCK DATA", html)
        self.assertEqual(html.count("<svg"), 3)
        comparison = read_json(site2 / "report.json")["comparison"]
        self.assertEqual(comparison["arms"]["tp4"]["status"], "comparable")
        self.assertGreater(comparison["arms"]["tp4"]["tput_geomean_ratio"], 1.3)
        self.assertEqual(read_json(site2 / "results.json")["run"]["id"], "w2")

    def test_report_without_baseline(self):
        site = self.root / "site"
        quiet(
            report.main,
            [
                "--results",
                str(self.sweep("w1")),
                "--out",
                str(site),
                "--baseline",
                str(self.root / "missing.json"),
            ],
        )
        self.assertIn("no previous run", (site / "index.html").read_text())

    def test_metrics_of_reads_the_inferencex_aggregate(self):
        aggregate = sweep.mock_aggregate({"arm": "tp8", "conc": 2}, seed=0)
        metrics = sweep.metrics_of(aggregate)
        self.assertEqual(
            set(metrics),
            {"tput_per_gpu", "p90_e2e_norm_intvty", "num_requests_successful"},
        )

    def test_total_cpu_dram_follows_the_gpu_share(self):
        tp4 = {"kv_offloading": "dram", "gpus": 4}
        tp8 = {"kv_offloading": "none", "gpus": 8}
        self.assertEqual(sweep.total_cpu_dram_gb(tp4, 0.85, 3000), "1275")
        self.assertEqual(sweep.total_cpu_dram_gb(tp8, 0.85, 3000), "0")

    def test_live_sweep_needs_the_model_snapshot(self):
        argv = [
            "--serving",
            str(self.serving),
            "--out",
            str(self.root / "live"),
            "--model-path",
            str(self.root / "nope"),
        ]
        with contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(quiet(sweep.main, argv), 1)
        self.assertIn("model snapshot", err.getvalue())


FAKE_SERVER = """
import http.server, sys
port = int(sys.argv[sys.argv.index("--port") + 1])
class Ok(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b"{}")
    def log_message(self, *args):
        pass
http.server.HTTPServer(("127.0.0.1", port), Ok).serve_forever()
"""
FAKE_CLIENT = """
import json, os, pathlib
env = os.environ
assert env["TOTAL_CPU_DRAM_GB"] == "0" and env["SERVED_MODEL_NAME"] and env["AKLR_FAKE"] == "client"
out = pathlib.Path(env["AGENTIC_OUTPUT_DIR"]) / (env["RESULT_FILENAME"] + ".json")
metrics = {"throughput": {"per_gpu": {"total_tput_tps": 10.0 * int(env["CONC"])}},
           "latency": {"e2e_norm_intvty": {"p90": 50.0}}}
out.write_text(json.dumps({"num_requests_successful": 7, "request_metrics": metrics}))
"""


class LivePathTests(unittest.TestCase):
    """The live path with a fake server and a fake ``infx.bench``: process lifecycle and env."""

    def test_one_point(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench = root / "ix" / "infx" / "bench"
            bench.mkdir(parents=True)
            (root / "ix" / "infx" / "__init__.py").write_text("")
            (bench / "__init__.py").write_text("")
            (bench / "__main__.py").write_text(FAKE_CLIENT)
            (root / "server.py").write_text(FAKE_SERVER)
            point = next(
                p for p in build()["points"] if p["arm"] == "tp8" and p["conc"] == 2
            )
            point["launch"]["command"] = [
                sys.executable,
                str(root / "server.py"),
                "--port",
                "${AGENTX_PORT}",
            ]
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]
            host = {
                "port": port,
                "ready_timeout_s": 30,
                "point_timeout_s": {"tp8": 60},
                "dram_utilization": 0.85,
                "memory_gb": 1000,
                "model_path": str(root),
                "ix_e2e": root / "ix",
                "agentx_runtime_dir": root / "venv",
                "client_env_overrides": {"AKLR_FAKE": "client"},
            }
            with patch.object(sweep, "POLL_S", 0.2):
                record = sweep.run_point(point, root / "out", host, None)
            self.assertEqual(record["status"], "ok", record["error"])
            self.assertEqual(record["tput_per_gpu"], 20.0)
            self.assertTrue(sweep.port_free(port))


class PrTests(unittest.TestCase):
    def test_merged_prs_dedupes_and_skips_unmerged(self):
        pulls = {
            "c1": [
                {
                    "number": 2,
                    "title": "two",
                    "html_url": "u2",
                    "user": {"login": "x"},
                    "merged_at": "2026-10-02",
                }
            ],
            "c2": [
                {
                    "number": 1,
                    "title": "one",
                    "html_url": "u1",
                    "user": {"login": "y"},
                    "merged_at": "2026-10-01",
                },
                {
                    "number": 3,
                    "title": "open",
                    "html_url": "u3",
                    "user": None,
                    "merged_at": None,
                },
            ],
            "c3": [
                {
                    "number": 2,
                    "title": "two",
                    "html_url": "u2",
                    "user": {"login": "x"},
                    "merged_at": "2026-10-02",
                }
            ],
        }

        def get(path, token):
            return pulls[path.split("/")[-2]]

        with (
            patch.object(prs, "first_parent", return_value=["c1", "c2", "c3"]),
            patch.object(prs, "get", get),
        ):
            out = prs.merged_prs("o/r", "base", "head", None, Path("."))
        self.assertIsNone(out["error"])
        self.assertEqual([pr["number"] for pr in out["prs"]], [2, 1])

    def test_lookup_errors_are_reported(self):
        with (
            patch.object(prs, "first_parent", return_value=["c1"]),
            patch.object(prs, "get", side_effect=OSError("down")),
        ):
            out = prs.merged_prs("o/r", "base", "head", None, Path("."))
        self.assertIn("down", out["error"])

    def test_same_commit_lists_nothing(self):
        self.assertEqual(prs.merged_prs("o/r", "a", "a", None, Path("."))["prs"], [])

    def test_report_repo_is_configured(self):
        self.assertEqual(load_config()["report"]["repo"], "AMD-AGI/SIKL")


if __name__ == "__main__":
    unittest.main()
