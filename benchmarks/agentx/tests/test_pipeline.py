# SPDX-License-Identifier: MIT
"""Mock sweep → report, end to end in a temporary directory, plus the PR lookup."""

from __future__ import annotations

import contextlib
import io
import os
import socket
import subprocess
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

    def test_point_filter(self):
        results = read_json(self.sweep("w1", "--points", "tp8:1,tp4:4"))
        self.assertEqual(
            sorted((p["arm"], p["conc"]) for p in results["points"]),
            [("tp4", 4), ("tp8", 1)],
        )
        argv = ["--serving", str(self.serving), "--out", str(self.root / "x")]
        with contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(quiet(sweep.main, [*argv, "--mock", "--points", "tp8:3"]), 1)
        self.assertIn("tp8:3", err.getvalue())

    def test_report_compares_with_the_previous_run(self):
        site1, site2 = self.root / "site1", self.root / "site2"
        quiet(
            report.main,
            ["--results", str(self.sweep("w1")), "--out", str(site1)],
        )
        w2 = self.sweep("w2", "--mock-seed", "1", "--mock-scale", "tp4=1.5")
        argv = [
            "--results",
            str(w2),
            "--baseline",
            str(site1 / "results.json"),
            "--out",
            str(site2),
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


class HistoryRepo:
    """A temporary main: base → squash PR → direct commit → merge-commit PR."""

    def __init__(self, root: Path):
        self.root = root
        self.git("init", "-q", "--initial-branch=main")
        self.base = self.commit("initial", "Ann")
        self.commit("feat: squash-merged change (#12)", "Alice")
        self.commit("fix a typo", "Bob")
        self.git("checkout", "-q", "-b", "topic")
        self.commit("work on topic", "Carol")
        self.git("checkout", "-q", "main")
        merge = "Merge pull request #15 from org/topic\n\nfeat: merged through a merge commit"
        self.git("merge", "-q", "--no-ff", "topic", "-m", merge, author="Dave")
        self.head = self.git("rev-parse", "HEAD")

    def git(self, *args: str, author: str = "Ann") -> str:
        env = {
            "GIT_AUTHOR_NAME": author,
            "GIT_AUTHOR_EMAIL": f"{author.lower()}@example.com",
            "GIT_COMMITTER_NAME": author,
            "GIT_COMMITTER_EMAIL": f"{author.lower()}@example.com",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "PATH": os.environ["PATH"],
        }
        done = subprocess.run(
            ["git", *args],
            cwd=self.root,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        return done.stdout.strip()

    def commit(self, subject: str, author: str) -> str:
        self.git("commit", "-q", "--allow-empty", "-m", subject, author=author)
        return self.git("rev-parse", "HEAD")


class HistoryTests(unittest.TestCase):
    URL = "https://github.com/o/r"

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.tmp = Path(temporary.name)
        (self.tmp / "main").mkdir()
        self.repo = HistoryRepo(self.tmp / "main")

    def test_first_parent_entries(self):
        out = prs.landed(self.repo.base, self.repo.head, self.repo.root, self.URL)
        self.assertIsNone(out["error"])
        got = [
            (e["kind"], e["number"], e["title"], e["author"]) for e in out["entries"]
        ]
        self.assertEqual(
            got,
            [
                ("pr", 15, "feat: merged through a merge commit", "Dave"),
                ("commit", None, "fix a typo", "Bob"),
                ("pr", 12, "feat: squash-merged change", "Alice"),
            ],
        )
        self.assertEqual(out["entries"][0]["url"], f"{self.URL}/pull/15")
        self.assertNotIn("url", out["entries"][1])
        self.assertTrue(
            out["entries"][1]["commit_url"].startswith(f"{self.URL}/commit/")
        )
        self.assertRegex(out["entries"][2]["date"], r"^\d{4}-\d{2}-\d{2}$")

    def test_same_commit_lists_nothing(self):
        out = prs.landed(self.repo.head, self.repo.head, self.repo.root, self.URL)
        self.assertEqual((out["entries"], out["error"]), ([], None))

    def test_unknown_baseline_is_a_clear_error(self):
        out = prs.landed("0" * 40, self.repo.head, self.repo.root, self.URL)
        self.assertIn("fetch-depth: 0", out["error"])
        self.assertEqual(out["entries"], [])

    def test_baseline_off_main_is_an_error(self):
        topic = self.repo.git("rev-parse", "topic")
        side = self.repo.git("rev-parse", "HEAD~1")
        out = prs.landed(topic, side, self.repo.root, self.URL)
        self.assertIn("not an ancestor", out["error"])

    def test_a_missing_baseline_is_fetched(self):
        self.repo.git("config", "uploadpack.allowAnySHA1InWant", "true")
        clone = self.tmp / "clone"
        source = f"file://{self.repo.root}"
        subprocess.run(
            ["git", "clone", "-q", "--depth=1", source, str(clone)], check=True
        )
        self.assertFalse(prs.has_commit(clone, self.repo.base))
        out = prs.landed(self.repo.base, self.repo.head, clone, self.URL)
        self.assertIsNone(out["error"])
        self.assertEqual(out["fetched"], ["unshallow"])
        self.assertEqual(len(out["entries"]), 3)

    def test_report_lists_the_history(self):
        serving = write_json(self.tmp / "serving.json", build())
        runs = {}
        for name, sha in (("w1", self.repo.base), ("w2", self.repo.head)):
            argv = ["--serving", str(serving), "--out", str(self.tmp / name), "--mock"]
            with patch.dict(os.environ, {"GITHUB_SHA": sha}):
                quiet(sweep.main, [*argv, "--run-id", name])
            runs[name] = self.tmp / name / "results.json"
        site = self.tmp / "site"
        argv = ["--results", str(runs["w2"]), "--baseline", str(runs["w1"])]
        quiet(
            report.main, [*argv, "--out", str(site), "--git-dir", str(self.repo.root)]
        )
        html = (site / "index.html").read_text()
        self.assertIn("/pull/15", html)
        self.assertIn("direct commit", html)
        url = load_config()["report"]["repo_url"]
        self.assertIn(f'href="{url}/compare/{self.repo.base}...{self.repo.head}"', html)
        self.assertIn(f'href="{url}/commit/{self.repo.base}"', html)
        history = read_json(site / "report.json")["history"]
        self.assertEqual([e["number"] for e in history["entries"]], [15, None, 12])

    def test_first_run_has_no_history(self):
        site = self.tmp / "site"
        serving = write_json(self.tmp / "serving.json", build())
        quiet(
            sweep.main,
            ["--serving", str(serving), "--out", str(self.tmp / "w1"), "--mock"],
        )
        quiet(
            report.main,
            ["--results", str(self.tmp / "w1/results.json"), "--out", str(site)],
        )
        html = (site / "index.html").read_text()
        self.assertIn("first run", html)
        self.assertNotIn("/compare/", html)
        self.assertIsNone(read_json(site / "report.json")["history"])

    def test_repo_url_is_configured(self):
        self.assertEqual(
            load_config()["report"]["repo_url"], "https://github.com/AMD-AGI/AKLR"
        )


if __name__ == "__main__":
    unittest.main()
