# SPDX-License-Identifier: MIT
"""AgentX runner selection; upstream resolution is mocked without network or GPU access."""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import sys
from pathlib import Path
from unittest import mock
from urllib.parse import parse_qs, urlsplit

import pytest
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


class PlanCase:
    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.root = root
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
        monkeypatch.setattr(plan.rr, "resolve_image", mock.Mock(return_value=(IMAGE, [])))
        monkeypatch.setattr(plan.rr, "resolve_ref", mock.Mock(return_value="c" * 40))
        self.resolver = mock.Mock(side_effect=self.serving)
        monkeypatch.setattr(plan.rr, "resolve", self.resolver)
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
        self.fetcher = mock.Mock(side_effect=self.fetch_json)
        monkeypatch.setattr(plan.rr, "fetch_json", self.fetcher)

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
                        "p90_e2e_norm_intvty": interactivity,
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


@pytest.fixture
def case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PlanCase:
    return PlanCase(tmp_path, monkeypatch)


def test_runner_filter_selects_across_models_and_persists_only_matches(case):
    args = case.args(runner_label=RUNNER_LABEL)
    manifest = plan.build(args)
    expected = [("first", "mi355x"), ("second", "different-target-id")]
    assert case.pairs(manifest) == expected
    assert {item["status"] for item in manifest["targets"]} == {"resolved"}
    assert {item["target"]["runner_label"] for item in manifest["targets"]} == {RUNNER_LABEL}
    assert [(call.args[0].model, call.args[0].target) for call in case.resolver.call_args_list] == expected
    assert json.loads((args.out / "manifest.json").read_text()) == manifest
    assert {str(path.relative_to(args.out)) for path in args.out.rglob("serving.json")} == {
        f"models/{model_id}/targets/{target_id}/serving.json" for model_id, target_id in expected
    }


def test_default_selection_works_without_runner_label_attribute(case):
    args = case.args()
    assert not hasattr(args, "runner_label")
    manifest = plan.build(args)
    assert case.pairs(manifest) == [
        ("first", "mi355x"),
        ("first", "other"),
        ("second", "different-target-id"),
    ]
    assert {item["status"] for item in manifest["targets"]} == {"resolved"}


def test_missing_runner_fails_before_resolution_or_manifest_creation(case):
    args = case.args(runner_label="agentx-missing")
    with pytest.raises(plan.ConfigError, match="no selected targets use runner label 'agentx-missing'"):
        plan.build(args)
    case.resolver.assert_not_called()
    assert not (args.out / "manifest.json").exists()


def test_cli_filter_limits_manifest_and_github_job_matrix(case):
    output = case.root / "github-output"
    out = case.root / "cli-plan"
    with mock.patch.dict(os.environ, {"GITHUB_OUTPUT": str(output)}):
        code = plan.main(
            [
                "--config",
                str(case.config),
                "--runner-label",
                RUNNER_LABEL,
                "--work",
                str(case.root / "cli-work"),
                "--out",
                str(out),
                "--run-id",
                "fixture",
            ]
        )
    assert code == 0
    values = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert values["count"] == "2"
    jobs = json.loads(values["matrix"])["include"]
    assert [(job["model"], job["target"], job["runner"]) for job in jobs] == [
        ("first", "mi355x", RUNNER_LABEL),
        ("second", "different-target-id", RUNNER_LABEL),
    ]
    manifest = json.loads((out / "manifest.json").read_text())
    assert case.pairs(manifest) == [(job["model"], job["target"]) for job in jobs]


@pytest.mark.parametrize("smoke", [{"arms": "tp8"}, {"points": "tp8:1"}, {"duration": 60}], ids=str)
def test_smoke_single_target_checks_use_the_filtered_selection(case, smoke):
    manifest = plan.build(case.args(models="first", runner_label=RUNNER_LABEL, **smoke))
    assert case.pairs(manifest) == [("first", "mi355x")]
    assert manifest["targets"][0]["status"] == "resolved"
    with pytest.raises(plan.ConfigError, match="smoke filters require exactly one"):
        plan.build(case.args(models="first", **smoke))


@pytest.mark.parametrize("smoke", [{"arms": "tp8"}, {"points": "tp8:1"}, {"duration": 60}], ids=str)
def test_smoke_filters_still_reject_multiple_matching_targets(case, smoke):
    with pytest.raises(plan.ConfigError, match="smoke filters require exactly one"):
        plan.build(case.args(runner_label=RUNNER_LABEL, **smoke))
    case.resolver.assert_not_called()


def test_pareto_single_target_check_uses_the_filtered_selection(case):
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
        manifest = plan.build(case.args(models="first", **options))
    assert case.pairs(manifest) == [("first", "mi355x")]
    assert manifest["targets"][0]["status"] == "resolved"
    with pytest.raises(plan.ConfigError, match="Pareto selection requires exactly one"):
        plan.build(case.args(**options))


def test_auto_pareto_resolves_independent_sources_and_multiple_points_per_arm(case):
    args = case.auto_args()
    manifest = plan.build(args)
    assert case.pairs(manifest) == [("first", "mi355x"), ("second", "different-target-id")]
    assert {item["status"] for item in manifest["targets"]} == {"resolved"}
    assert args.inferencex_model == "auto"
    requests = [call.args[0] for call in case.fetcher.call_args_list]
    assert requests.count(plan.MODEL_CATALOG) == 1
    assert [
        parse_qs(urlsplit(url).query)["model"][0] for url in requests if url.startswith(plan.rr.API + "?")
    ] == ["First Model", "Second Model"]
    for model_id, target_id in case.pairs(manifest):
        root = args.out / f"models/{model_id}/targets/{target_id}"
        spec = json.loads((root / "serving.json").read_text())
        selection = json.loads((root / "selection.json").read_text())
        snapshot = json.loads((root / "upstream-snapshot.json").read_text())
        assert selection == spec["selection"]
        assert selection["scope"]["model"] == model_id
        assert selection["scope"]["hardware"] == "mi355x"
        assert selection["duration"] == 600
        assert selection["points"] == ["tp8:1", "tp8:2"]
        assert selection["errors"] == []
        assert {row["model"] for row in snapshot["payload"]["rows"]} == {model_id}
        assert [plan.rr.point_id(point) for point in plan.selected_points(spec)] == ["tp8:1", "tp8:2"]


def test_auto_pareto_keeps_prefix_scopes_separate_in_one_display_bucket(case):
    case.catalog = {"models": [{"name": "Shared Display", "dbKeys": ["first", "second"]}]}
    args = case.auto_args()
    manifest = plan.build(args)
    assert {item["status"] for item in manifest["targets"]} == {"resolved"}
    for model_id, target_id in case.pairs(manifest):
        root = args.out / f"models/{model_id}/targets/{target_id}"
        selection = json.loads((root / "selection.json").read_text())
        assert selection["scope"]["model"] == model_id
        assert selection["upstream_count"] == 3
        assert {point["upstream_id"] for point in selection["selected"]} == {
            f"{model_id}-tp8-c1",
            f"{model_id}-tp8-c2",
        }


def test_auto_pareto_mapping_failure_preserves_other_targets_and_failure_evidence(case):
    for row in case.rows:
        if row["model"] == "second" and row["conc"] == 1:
            row["recipe_fingerprint"] = "0" * 64
    args = case.auto_args()
    manifest = plan.build(args)
    assert [item["status"] for item in manifest["targets"]] == ["resolved", "resolve-failed"]
    assert "no exact local match" in manifest["targets"][1]["error"]
    assert json.loads((args.out / "manifest.json").read_text()) == manifest
    failed = args.out / "models/second/targets/different-target-id"
    selection = json.loads((failed / "selection.json").read_text())
    assert selection["errors"]
    assert (failed / "upstream-snapshot.json").is_file()
    assert (args.out / "models/first/targets/mi355x/serving.json").is_file()


@pytest.mark.parametrize(
    "second_entries",
    [[], [{"name": name, "dbKeys": ["second"]} for name in ("Second Model", "Other Display")]],
    ids=str,
)
def test_auto_catalog_unknown_or_ambiguous_model_only_fails_that_target(case, second_entries):
    case.catalog = {"models": [{"name": "First Model", "dbKeys": ["first"]}, *second_entries]}
    manifest = plan.build(case.auto_args())
    assert [item["status"] for item in manifest["targets"]] == ["resolved", "resolve-failed"]
    assert "exactly one catalog match" in manifest["targets"][1]["error"]


@pytest.mark.parametrize(
    "catalog", [{"models": {}}, {"models": [{"name": "Broken", "dbKeys": "first"}]}], ids=str
)
def test_auto_invalid_catalog_fails_closed(case, catalog):
    case.catalog = catalog
    manifest = plan.build(case.auto_args())
    assert {item["status"] for item in manifest["targets"]} == {"resolve-failed"}
    assert all("catalog" in item["error"] for item in manifest["targets"])


@pytest.mark.parametrize(
    "override", [{"inferencex_run_id": "1234"}, {"inferencex_results": "snapshot.json"}], ids=str
)
def test_auto_rejects_explicit_run_or_result_source(case, override):
    with pytest.raises(ValueError, match="cannot be combined"):
        plan.build(case.auto_args(**override))
    case.resolver.assert_not_called()
    case.fetcher.assert_not_called()


def test_recipe_duration_requirement_preserves_upstream_duration_without_an_override(case):
    args = case.args(require_recipe_duration=3600)
    manifest = plan.build(args)
    assert len(manifest["targets"]) == 3
    assert {item["status"] for item in manifest["targets"]} == {"resolved"}
    for path in args.out.rglob("serving.json"):
        spec = json.loads(path.read_text())
        assert spec["selection"]["duration"] is None
        assert {point["client_env"]["DURATION"] for point in spec["points"]} == {"3600"}
    case.fetcher.assert_not_called()


def test_recipe_duration_mismatch_is_reported_per_target(case):
    def serving(args):
        spec = case.serving(args)
        if args.model == "second":
            spec["points"][0]["client_env"]["DURATION"] = "1200"
        return spec

    case.resolver.side_effect = serving
    args = case.args(runner_label=RUNNER_LABEL, require_recipe_duration=3600)
    manifest = plan.build(args)
    assert [item["status"] for item in manifest["targets"]] == ["resolved", "resolve-failed"]
    assert "upstream DURATION must be 3600" in manifest["targets"][1]["error"]
    assert "tp8:1='1200'" in manifest["targets"][1]["error"]
    failed = args.out / "models/second/targets/different-target-id/serving.json"
    assert json.loads(failed.read_text())["selection"]["duration"] is None


def test_recipe_duration_requirement_only_checks_selected_points(case):
    def serving(args):
        spec = case.serving(args)
        spec["points"][1]["client_env"]["DURATION"] = "1200"
        return spec

    case.resolver.side_effect = serving
    args = case.args(models="first", runner_label=RUNNER_LABEL, points="tp8:1", require_recipe_duration=3600)
    manifest = plan.build(args)
    assert manifest["targets"][0]["status"] == "resolved"
    assert manifest["targets"][0]["point_ids"] == ["tp8:1"]


@pytest.mark.parametrize(
    "options",
    [
        {"require_recipe_duration": 0},
        {"require_recipe_duration": -1},
        {"require_recipe_duration": 3600, "duration": 3600},
    ],
    ids=str,
)
def test_recipe_duration_requirement_rejects_invalid_or_overridden_values(case, options):
    with pytest.raises(plan.ConfigError, match="--require-recipe-duration"):
        plan.build(case.args(**options))
    case.resolver.assert_not_called()
