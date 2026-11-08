# SPDX-License-Identifier: MIT
"""AgentX runner selection; upstream resolution is mocked without network or GPU access."""

from __future__ import annotations

import argparse
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
from unittest import mock
from urllib.parse import parse_qs, urlsplit

import yaml

with mock.patch.object(
    sys, "path", [str(Path(__file__).resolve().parents[2] / "benchmarks/agentx"), *sys.path]
):
    plan = importlib.import_module("agentx_ci.plan")

RUNNER_LABEL = "agentx-mi355x"
IMAGE = "example.invalid/server@sha256:" + "a" * 64


def target(runner_label):
    return {
        "hardware": {"gpu_name_pattern": "MI355X", "gpu_arch": "gfx950", "gpu_count": 8},
        "runner_label": runner_label,
        "inferencex": {"config_key": "fixture", "expect": {"precision": "mxfp4"}},
    }


def model(targets):
    return {"weights": {"repo": "example/model", "revision": "b" * 40}, "targets": targets}


class AgentXPlanTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.config = self.root / "agentx.yaml"
        self.config.write_text(
            yaml.safe_dump(
                {
                    "inferencex": {"repo": "example/inferencex", "ref": "main"},
                    "models": {
                        "first": model(
                            {
                                "mi355x": target(f" {RUNNER_LABEL} "),
                                "other": target("agentx-other"),
                            }
                        ),
                        "second": model({"different-target-id": target(RUNNER_LABEL)}),
                    },
                    "report": {"repo_url": "https://example.invalid/repo"},
                },
                sort_keys=False,
            )
        )
        self.next_run = 0
        self.enterContext(redirect_stdout(io.StringIO()))
        self.enterContext(redirect_stderr(io.StringIO()))
        self.enterContext(mock.patch.object(plan.rr, "resolve_image", return_value=(IMAGE, [])))
        self.enterContext(mock.patch.object(plan.rr, "resolve_ref", return_value="c" * 40))
        self.resolver = self.enterContext(mock.patch.object(plan.rr, "resolve", side_effect=self.serving))
        self.catalog = {
            "models": [
                {"name": "First Model", "dbKeys": ["first"]},
                {"name": "Second Model", "dbKeys": ["second"]},
            ]
        }
        self.rows = [
            row
            for model_id in ("first", "second")
            for row in self.upstream_rows(
                self.serving(argparse.Namespace(model=model_id, target="fixture", inferencex_ref="c" * 40))
            )
        ]
        self.fetcher = self.enterContext(
            mock.patch.object(plan.rr, "fetch_json", side_effect=self.fetch_json)
        )

    def args(self, **overrides):
        self.next_run += 1
        values = {
            "config": self.config,
            "models": "all",
            "targets": "all",
            "point_policy": "all",
            "inferencex_ref": None,
            "run_image": None,
            "arms": "all",
            "points": None,
            "duration": None,
            "run_id": "fixture",
            "work": self.root / f"work-{self.next_run}",
            "out": self.root / f"plan-{self.next_run}",
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    @staticmethod
    def serving(args):
        topology = {
            "tp": 8,
            "ep": 1,
            "dp-attn": False,
            "spec-decoding": "mtp",
            "pp": 1,
            "pcp-size": 1,
            "dcp-size": 1,
            "kv-offloading": "none",
            "kv-offload-backend": None,
        }
        return {
            "model": {"id": args.model},
            "target": {"id": args.target},
            "provenance": {"inferencex": {"sha": args.inferencex_ref}},
            "image": {"inferencex": IMAGE},
            "warnings": [],
            "arms": {"tp8": {"topology": topology}},
            "points": [
                {
                    "arm": "tp8",
                    "conc": conc,
                    "exp_name": f"{args.model}-tp8-c{conc}",
                    "client_env": {
                        "DURATION": "3600",
                        "MODEL_PREFIX": args.model,
                        "FRAMEWORK": "sglang",
                        "PRECISION": "fp4",
                    },
                    "hashes": {"inferencex_recipe_fingerprint": plan.sha256_json(args.model)},
                }
                for conc in (1, 2, 3)
            ],
        }

    @staticmethod
    def upstream_rows(spec):
        measurements = [(30, 100), (20, 150), (10, 80)]
        rows = []
        for point, (interactivity, throughput) in zip(spec["points"], measurements):
            rows.append(
                {
                    **plan.rr.scope_of(spec, "mi355x"),
                    "id": point["exp_name"],
                    "conc": point["conc"],
                    "recipe_fingerprint": point["hashes"]["inferencex_recipe_fingerprint"],
                    "status": "ok",
                    "curve_date": "2026-11-08",
                    "curve_workflow_run_id": "1234",
                    "run_url": "https://github.com/SemiAnalysisAI/InferenceX/actions/runs/1234/attempts/1",
                    "prefill_tp": 8,
                    "decode_tp": 8,
                    "prefill_ep": 1,
                    "decode_ep": 1,
                    "prefill_dp_attention": False,
                    "decode_dp_attention": False,
                    "spec_method": "mtp",
                    "disagg": False,
                    "is_multinode": False,
                    "image": IMAGE,
                    "metrics": {
                        plan.rr.X: interactivity,
                        plan.rr.Y: throughput,
                        "pp": 1,
                        "pcp_size": 1,
                        "dcp_size": 1,
                        "kv_offloading": "none",
                    },
                }
            )
        return rows

    def fetch_json(self, url):
        if url == plan.MODEL_CATALOG:
            return copy.deepcopy(self.catalog), "d" * 64
        if url.startswith(plan.rr.API + "?"):
            display = parse_qs(urlsplit(url).query)["model"][0]
            prefixes = next(model["dbKeys"] for model in self.catalog["models"] if model["name"] == display)
            return copy.deepcopy([row for row in self.rows if row["model"] in prefixes]), "e" * 64
        if url == "https://api.github.com/repos/SemiAnalysisAI/InferenceX/actions/runs/1234/attempts/1":
            return {
                "id": 1234,
                "run_attempt": 1,
                "head_sha": "c" * 40,
                "status": "completed",
                "conclusion": "success",
                "html_url": "https://github.com/SemiAnalysisAI/InferenceX/actions/runs/1234/attempts/1",
            }, "f" * 64
        self.fail(f"Unexpected upstream request: {url}")

    def auto_args(self, **overrides):
        options = {
            "runner_label": RUNNER_LABEL,
            "point_policy": plan.POLICY,
            "inferencex_model": "auto",
            "inferencex_hardware": "mi355x",
            "inferencex_results": None,
            "inferencex_run_id": None,
            "duration": 600,
        }
        options.update(overrides)
        return self.args(**options)

    @staticmethod
    def pairs(manifest):
        return [(item["model"]["id"], item["target"]["id"]) for item in manifest["targets"]]

    def test_runner_filter_selects_across_models_and_persists_only_matches(self):
        args = self.args(runner_label=RUNNER_LABEL)
        manifest = plan.build(args)
        expected = [("first", "mi355x"), ("second", "different-target-id")]
        self.assertEqual(self.pairs(manifest), expected)
        self.assertEqual({item["status"] for item in manifest["targets"]}, {"resolved"})
        self.assertEqual({item["target"]["runner_label"] for item in manifest["targets"]}, {RUNNER_LABEL})
        self.assertEqual(
            [(call.args[0].model, call.args[0].target) for call in self.resolver.call_args_list], expected
        )
        self.assertEqual(json.loads((args.out / "manifest.json").read_text()), manifest)
        self.assertEqual(
            {str(path.relative_to(args.out)) for path in args.out.rglob("serving.json")},
            {f"models/{model_id}/targets/{target_id}/serving.json" for model_id, target_id in expected},
        )

    def test_default_selection_works_without_runner_label_attribute(self):
        args = self.args()
        self.assertFalse(hasattr(args, "runner_label"))
        manifest = plan.build(args)
        self.assertEqual(
            self.pairs(manifest),
            [("first", "mi355x"), ("first", "other"), ("second", "different-target-id")],
        )
        self.assertEqual({item["status"] for item in manifest["targets"]}, {"resolved"})

    def test_missing_runner_fails_before_resolution_or_manifest_creation(self):
        args = self.args(runner_label="agentx-missing")
        with self.assertRaisesRegex(
            plan.ConfigError, "no selected targets use runner label 'agentx-missing'"
        ):
            plan.build(args)
        self.resolver.assert_not_called()
        self.assertFalse((args.out / "manifest.json").exists())

    def test_cli_filter_limits_manifest_and_github_job_matrix(self):
        output = self.root / "github-output"
        out = self.root / "cli-plan"
        with mock.patch.dict(os.environ, {"GITHUB_OUTPUT": str(output)}):
            code = plan.main(
                [
                    "--config",
                    str(self.config),
                    "--runner-label",
                    RUNNER_LABEL,
                    "--work",
                    str(self.root / "cli-work"),
                    "--out",
                    str(out),
                    "--run-id",
                    "fixture",
                ]
            )
        self.assertEqual(code, 0)
        values = dict(line.split("=", 1) for line in output.read_text().splitlines())
        self.assertEqual(values["count"], "2")
        jobs = json.loads(values["matrix"])["include"]
        self.assertEqual(
            [(job["model"], job["target"], job["runner"]) for job in jobs],
            [("first", "mi355x", RUNNER_LABEL), ("second", "different-target-id", RUNNER_LABEL)],
        )
        manifest = json.loads((out / "manifest.json").read_text())
        self.assertEqual(self.pairs(manifest), [(job["model"], job["target"]) for job in jobs])

    def test_smoke_single_target_checks_use_the_filtered_selection(self):
        for smoke in ({"arms": "tp8"}, {"points": "tp8:1"}, {"duration": 60}):
            with self.subTest(smoke=smoke):
                manifest = plan.build(self.args(models="first", runner_label=RUNNER_LABEL, **smoke))
                self.assertEqual(self.pairs(manifest), [("first", "mi355x")])
                self.assertEqual(manifest["targets"][0]["status"], "resolved")
                with self.assertRaisesRegex(plan.ConfigError, "smoke filters require exactly one"):
                    plan.build(self.args(models="first", **smoke))

    def test_smoke_filters_still_reject_multiple_matching_targets(self):
        for smoke in ({"arms": "tp8"}, {"points": "tp8:1"}, {"duration": 60}):
            with self.subTest(smoke=smoke):
                with self.assertRaisesRegex(plan.ConfigError, "smoke filters require exactly one"):
                    plan.build(self.args(runner_label=RUNNER_LABEL, **smoke))
        self.resolver.assert_not_called()

    def test_pareto_single_target_check_uses_the_filtered_selection(self):
        options = {
            "runner_label": RUNNER_LABEL,
            "point_policy": plan.POLICY,
            "inferencex_hardware": "mi355x",
            "inferencex_model": "fixture",
            "inferencex_results": None,
            "inferencex_run_id": None,
        }
        # Pareto snapshot acquisition and selection are independent of the runner filter.
        with (
            mock.patch.object(
                plan, "prepare", return_value={"policy": plan.POLICY, "manifest_sha256": "fixture-selection"}
            ),
            mock.patch.object(plan, "selected_points", side_effect=lambda spec: spec["points"]),
        ):
            manifest = plan.build(self.args(models="first", **options))
        self.assertEqual(self.pairs(manifest), [("first", "mi355x")])
        self.assertEqual(manifest["targets"][0]["status"], "resolved")
        with self.assertRaisesRegex(plan.ConfigError, "Pareto selection requires exactly one"):
            plan.build(self.args(**options))

    def test_auto_pareto_resolves_independent_sources_and_multiple_points_per_arm(self):
        args = self.auto_args()
        manifest = plan.build(args)
        self.assertEqual(self.pairs(manifest), [("first", "mi355x"), ("second", "different-target-id")])
        self.assertEqual({item["status"] for item in manifest["targets"]}, {"resolved"})
        self.assertEqual(args.inferencex_model, "auto")
        requests = [call.args[0] for call in self.fetcher.call_args_list]
        self.assertEqual(requests.count(plan.MODEL_CATALOG), 1)
        self.assertEqual(
            [
                parse_qs(urlsplit(url).query)["model"][0]
                for url in requests
                if url.startswith(plan.rr.API + "?")
            ],
            ["First Model", "Second Model"],
        )
        for model_id, target_id in self.pairs(manifest):
            root = args.out / f"models/{model_id}/targets/{target_id}"
            spec = json.loads((root / "serving.json").read_text())
            selection = json.loads((root / "selection.json").read_text())
            snapshot = json.loads((root / "upstream-snapshot.json").read_text())
            self.assertEqual(selection, spec["selection"])
            self.assertEqual(selection["scope"]["model"], model_id)
            self.assertEqual(selection["scope"]["hardware"], "mi355x")
            self.assertEqual(selection["duration"], 600)
            self.assertEqual(selection["points"], ["tp8:1", "tp8:2"])
            self.assertEqual(selection["errors"], [])
            self.assertEqual({row["model"] for row in snapshot["payload"]["rows"]}, {model_id})
            self.assertEqual(
                [plan.rr.point_id(point) for point in plan.selected_points(spec)], ["tp8:1", "tp8:2"]
            )

    def test_auto_pareto_keeps_prefix_scopes_separate_in_one_display_bucket(self):
        self.catalog = {"models": [{"name": "Shared Display", "dbKeys": ["first", "second"]}]}
        args = self.auto_args()
        manifest = plan.build(args)
        self.assertEqual({item["status"] for item in manifest["targets"]}, {"resolved"})
        for model_id, target_id in self.pairs(manifest):
            root = args.out / f"models/{model_id}/targets/{target_id}"
            selection = json.loads((root / "selection.json").read_text())
            self.assertEqual(selection["scope"]["model"], model_id)
            self.assertEqual(selection["upstream_count"], 3)
            self.assertEqual(
                {point["upstream_id"] for point in selection["selected"]},
                {f"{model_id}-tp8-c1", f"{model_id}-tp8-c2"},
            )

    def test_auto_pareto_mapping_failure_preserves_other_targets_and_failure_evidence(self):
        for row in self.rows:
            if row["model"] == "second" and row["conc"] == 1:
                row["recipe_fingerprint"] = "0" * 64
        args = self.auto_args()
        manifest = plan.build(args)
        self.assertEqual([item["status"] for item in manifest["targets"]], ["resolved", "resolve-failed"])
        self.assertIn("no exact local match", manifest["targets"][1]["error"])
        self.assertEqual(json.loads((args.out / "manifest.json").read_text()), manifest)
        failed = args.out / "models/second/targets/different-target-id"
        selection = json.loads((failed / "selection.json").read_text())
        self.assertTrue(selection["errors"])
        self.assertTrue((failed / "upstream-snapshot.json").is_file())
        self.assertTrue((args.out / "models/first/targets/mi355x/serving.json").is_file())

    def test_auto_catalog_unknown_or_ambiguous_model_only_fails_that_target(self):
        for second_entries in (
            [],
            [{"name": name, "dbKeys": ["second"]} for name in ("Second Model", "Other Display")],
        ):
            with self.subTest(entries=second_entries):
                self.catalog = {"models": [{"name": "First Model", "dbKeys": ["first"]}, *second_entries]}
                manifest = plan.build(self.auto_args())
                self.assertEqual(
                    [item["status"] for item in manifest["targets"]], ["resolved", "resolve-failed"]
                )
                self.assertIn("exactly one catalog match", manifest["targets"][1]["error"])

    def test_auto_invalid_catalog_fails_closed(self):
        for catalog in ({"models": {}}, {"models": [{"name": "Broken", "dbKeys": "first"}]}):
            with self.subTest(catalog=catalog):
                self.catalog = catalog
                manifest = plan.build(self.auto_args())
                self.assertEqual({item["status"] for item in manifest["targets"]}, {"resolve-failed"})
                self.assertTrue(all("catalog" in item["error"] for item in manifest["targets"]))

    def test_auto_rejects_explicit_run_or_result_source(self):
        for override in ({"inferencex_run_id": "1234"}, {"inferencex_results": "snapshot.json"}):
            with self.subTest(override=override):
                with self.assertRaises(ValueError):
                    plan.build(self.auto_args(**override))
        self.resolver.assert_not_called()
        self.fetcher.assert_not_called()

    def test_recipe_duration_requirement_preserves_upstream_duration_without_an_override(self):
        args = self.args(require_recipe_duration=3600)
        manifest = plan.build(args)
        self.assertEqual(len(manifest["targets"]), 3)
        self.assertEqual({item["status"] for item in manifest["targets"]}, {"resolved"})
        for path in args.out.rglob("serving.json"):
            spec = json.loads(path.read_text())
            self.assertIsNone(spec["selection"]["duration"])
            self.assertEqual({point["client_env"]["DURATION"] for point in spec["points"]}, {"3600"})
        self.fetcher.assert_not_called()

    def test_recipe_duration_mismatch_is_reported_per_target(self):
        def serving(args):
            spec = self.serving(args)
            if args.model == "second":
                spec["points"][0]["client_env"]["DURATION"] = "1200"
            return spec

        self.resolver.side_effect = serving
        args = self.args(runner_label=RUNNER_LABEL, require_recipe_duration=3600)
        manifest = plan.build(args)
        self.assertEqual([item["status"] for item in manifest["targets"]], ["resolved", "resolve-failed"])
        self.assertIn("upstream DURATION must be 3600", manifest["targets"][1]["error"])
        self.assertIn("tp8:1='1200'", manifest["targets"][1]["error"])
        failed = args.out / "models/second/targets/different-target-id/serving.json"
        self.assertIsNone(json.loads(failed.read_text())["selection"]["duration"])

    def test_recipe_duration_requirement_only_checks_selected_points(self):
        def serving(args):
            spec = self.serving(args)
            spec["points"][1]["client_env"]["DURATION"] = "1200"
            return spec

        self.resolver.side_effect = serving
        args = self.args(
            models="first", runner_label=RUNNER_LABEL, points="tp8:1", require_recipe_duration=3600
        )
        manifest = plan.build(args)
        self.assertEqual(manifest["targets"][0]["status"], "resolved")
        self.assertEqual(manifest["targets"][0]["point_ids"], ["tp8:1"])

    def test_recipe_duration_requirement_rejects_invalid_or_overridden_values(self):
        for options in (
            {"require_recipe_duration": 0},
            {"require_recipe_duration": -1},
            {"require_recipe_duration": 3600, "duration": 3600},
        ):
            with self.subTest(options=options):
                with self.assertRaisesRegex(plan.ConfigError, "--require-recipe-duration"):
                    plan.build(self.args(**options))
        self.resolver.assert_not_called()


if __name__ == "__main__":
    unittest.main()
