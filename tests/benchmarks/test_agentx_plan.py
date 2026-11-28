# SPDX-License-Identifier: MIT
"""AgentX recipe serving.json contract and recipe-based plans; runs without network or GPU access."""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from unittest import mock

import pytest
import yaml

AGENTX = Path(__file__).resolve().parents[2] / "benchmarks/perf/agentx"
from agentx import plan
from agentx import serving_config as sc
from perf_utils import image as ci_image
from perf_utils.gh import load_reviewers
from perf_utils.pr_scope import requested_models

RUNNER_LABEL = "conductor-mi355x"
IMAGE = "example.invalid/server@sha256:" + "a" * 64
RUNS = (
    ("first", "mi355x", "sglang"),
    ("first", "mi355x", "vllm"),
    ("first", "other", "sglang"),
    ("second", "different-target-id", "vllm"),
)
SOURCE = {"kind": "cookbook", "url": "https://recipes.example.invalid/model.json"}


def framework(**extra):
    return {"recipe_source": dict(SOURCE), **extra}


def target(runner_label, **frameworks):
    return {
        "hardware": {"gpu_name_pattern": "MI355X", "gpu_arch": "gfx950", "gpu_count": 8},
        "runner_label": runner_label,
        "frameworks": frameworks,
    }


def model(targets, **extra):
    return {
        "family": "Example",
        "precision": "mxfp4",
        "weights": {"repo": "example/model", "revision": "b" * 40},
        "targets": targets,
        **extra,
    }


class PlanCase:
    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch, run_image: mock.Mock) -> None:
        self.root = root
        self.monkeypatch = monkeypatch
        self.run_image = run_image
        self.config = self.root / "agentx.yaml"
        self.definition = {
            "models": {
                "first": model(
                    {
                        "mi355x": target(
                            f" {RUNNER_LABEL} ",
                            sglang=framework(advisory_env={"AKLR_SGLANG_PLUGIN": "1"}),
                            vllm=framework(),
                        ),
                        "other": target("agentx-other", sglang=framework()),
                    }
                ),
                "second": model({"different-target-id": target(RUNNER_LABEL, vllm=framework())}),
            },
        }
        self.write_config()
        self.durations = {}
        self.next_run = 0

    def write_config(self):
        self.config.write_text(yaml.safe_dump(self.definition, sort_keys=False))

    def definition_of(self, model_id="first", target_id="mi355x", framework_id="sglang"):
        return plan.model_config(plan.load_config(self.config), model_id, target_id, framework_id)

    def recipe(self, model_id="first", target_id="mi355x", framework_id="sglang"):
        return {
            "schema": 4,
            "source": {"kind": "cookbook", "url": "https://recipes.example.invalid/model.json"},
            "points": [
                {
                    "arm": "tp8",
                    "conc": conc,
                    "served_model_name": "example/model",
                    "gpus": 8,
                    "launch": {
                        "command": ["vllm", "serve", "${AGENTX_MODEL_PATH}"],
                        "env": {"HF_HUB_OFFLINE": "1"},
                    },
                    "client_env": {
                        "FRAMEWORK": framework_id,
                        "DURATION": self.durations.get((model_id, conc), "3600"),
                        "CONC": str(conc),
                        "TP": "8",
                        "EP_SIZE": "1",
                        "PP_SIZE": "1",
                        "PCP_SIZE": "1",
                        "DCP_SIZE": "1",
                        "DP_ATTENTION": "false",
                        "TOTAL_CPU_DRAM_GB": "0",
                        "AIPERF_EXPERIMENTAL_FAST": "0",
                    },
                    "runtime": {"port": None, "ready_timeout_s": 600, "point_timeout_s": 4000, "poll_s": 1},
                }
                for conc in (1, 2)
            ],
        }

    def recipe_file(self, *run):
        return sc.recipe_path(self.config, self.definition_of(*run))

    def write_recipes(self, runs=RUNS):
        for run in runs:
            plan.write_json(self.recipe_file(*run), self.recipe(*run))

    def args(self, **overrides):
        self.next_run += 1
        values = {
            "config": self.config,
            "models": "all",
            "targets": "all",
            "frameworks": "all",
            "points": None,
            "duration": None,
            "run_id": "fixture",
            "out": self.root / f"plan-{self.next_run}",
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    @staticmethod
    def runs(manifest):
        return [
            (item["model"]["id"], item["target"]["id"], item["framework"]["id"])
            for item in manifest["targets"]
        ]

    def serving_of(self, args, model_id="first", target_id="mi355x", framework_id="sglang"):
        return json.loads(
            (args.out / plan.target_path(model_id, target_id, framework_id) / "serving.json").read_text()
        )

    def run_with_comment(self, body: str, *extra: str, comments: list[dict] | None = None):
        output, summary = self.root / "github-output", self.root / "github-summary"
        output.unlink(missing_ok=True)
        summary.unlink(missing_ok=True)
        pr_comments = self.root / "pr-comments.json"
        if comments is None:
            comments = [{"user": "author", "body": "Ready."}, {"user": "RuibinCheung", "body": body}]
        pr_comments.write_text(json.dumps(comments))
        allowlist = self.root / "allowlist.yaml"
        allowlist.write_text("reviewers: [RuibinCheung, xiaobochen-amd, jiejingzhangamd]\n")
        self.next_run += 1
        out = self.root / f"pr-plan-{self.next_run}"
        env = {"GITHUB_OUTPUT": str(output), "GITHUB_STEP_SUMMARY": str(summary)}
        for name, value in env.items():
            self.monkeypatch.setenv(name, value)
        code = plan.main(
            [
                "--config",
                str(self.config),
                "--runner-label",
                RUNNER_LABEL,
                "--pr-comments",
                str(pr_comments),
                "--allowlist",
                str(allowlist),
                *extra,
                "--out",
                str(out),
            ]
        )
        values = (
            dict(line.split("=", 1) for line in output.read_text().splitlines()) if output.exists() else {}
        )
        return code, values, out

    def run_manual(self, actor: str, ref: str):
        allowlist = self.root / "allowlist.yaml"
        allowlist.write_text("reviewers: [RuibinCheung, xiaobochen-amd]\n")
        summary = self.root / "github-summary"
        summary.unlink(missing_ok=True)
        self.next_run += 1
        out = self.root / f"manual-plan-{self.next_run}"
        argv = ["--config", str(self.config), "--runner-label", RUNNER_LABEL, "--allowlist", str(allowlist)]
        argv += ["--manual-actor", actor, "--manual-ref", ref, "--out", str(out)]
        self.monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
        self.monkeypatch.setenv("GITHUB_ACTIONS", "true")
        code = plan.main(argv)
        return code, out, summary.read_text() if summary.exists() else ""

    def serving_of_built(self):
        args = self.args(models="first", targets="mi355x", frameworks="sglang")
        plan.build(args)
        return self.serving_of(args)

    def plan(self, **overrides):
        self.write_recipes()
        values = {
            "config": self.config,
            "models": "all",
            "targets": "all",
            "frameworks": "all",
            "points": None,
            "duration": None,
            "run_id": "fixture",
            "out": self.root / "plan",
        }
        values.update(overrides)
        return plan.build(argparse.Namespace(**values))


@pytest.fixture
def case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    run_image = mock.Mock(return_value=IMAGE)
    monkeypatch.setattr(ci_image, "run_image", run_image)
    return PlanCase(tmp_path, monkeypatch, run_image)


@pytest.fixture
def planned(case: PlanCase) -> PlanCase:
    """The case with every run's recipe on disk, as plan reads them."""
    case.write_recipes()
    return case


def test_build_run_adds_plan_owned_fields(case):
    recipe = case.recipe()
    spec = sc.build_run(recipe, case.definition_of())
    assert "hashes" not in recipe["points"][0]
    assert spec["points"][0]["launch"]["env"] == {
        "AKLR_SGLANG_PLUGIN": "1",
        "HF_HUB_OFFLINE": "1",
        "HIP_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
    }
    definition = case.definition_of()
    for key in sc.AGENTX_TOP:
        assert spec[key] == definition[key]
    fixed = sc.template()
    assert spec["client"] == fixed["client"]
    assert spec["points"][1]["trace"] == fixed["trace"]
    assert spec["points"][1]["client_env"]["WEKA_LOADER_OVERRIDE"] == fixed["trace"]["loader"]
    assert spec["selection"]["points"] == ["tp8:1", "tp8:2"]
    assert spec["arms"]["tp8"]["concs"] == [1, 2]
    assert spec["arms"]["tp8"]["arm_serve_sha256"]
    assert spec["grid_sha256"]
    assert spec["selection"]["policy"] == sc.POLICY
    assert spec["selection"]["duration"] is None
    assert [sc.point_id(p) for p in sc.selected_points(spec)] == ["tp8:1", "tp8:2"]


def test_serve_identity_ignores_advisory_env_and_tracks_recipe_changes(case):
    base = sc.build_run(case.recipe(), case.definition_of())
    case.definition["models"]["first"]["advisory_env"] = {"EXTRA": "x"}
    case.write_config()
    advisory = sc.build_run(case.recipe(), case.definition_of())
    assert base["selection"]["selection_sha256"] == advisory["selection"]["selection_sha256"]
    changed = case.recipe()
    changed["points"][0]["launch"]["command"].append("--enforce-eager")
    changed = sc.build_run(changed, case.definition_of())
    assert base["selection"]["serve_sha256"] != changed["selection"]["serve_sha256"]
    assert base["points"][0]["hashes"] != changed["points"][0]["hashes"]
    # Point-by-point settings decide which points still compare; the selected points are unchanged.
    assert base["selection"]["selection_sha256"] == changed["selection"]["selection_sha256"]


INVALID_RECIPES = {
    "plan-owned": lambda r: r.update(grid_sha256="0" * 64),
    "plan-owned fields: ['hashes']": lambda r: r["points"][0].update(hashes={}),
    "plan-owned fields: ['selection']": lambda r: r.update(selection={"method": "measured-frontier"}),
    "plan-owned fields: ['arms']": lambda r: r.update(arms={"tp8": {}}),
    "plan-owned fields: ['trace']": lambda r: r["points"][0].update(trace={}),
    "unknown fields: ['recipe_fingerprint']": lambda r: r["points"][0].update(recipe_fingerprint="f"),
    "fixed in serving.schema.json: ['client']": lambda r: r.update(client=sc.template()["client"]),
    "fixed in serving.schema.json: ['trace']": lambda r: r.update(trace=sc.template()["trace"]),
    "plan-owned keys": lambda r: r["points"][0]["client_env"].update(WEKA_LOADER_OVERRIDE="x"),
    "must not set devices": lambda r: r["points"][0]["launch"]["env"].update(HIP_VISIBLE_DEVICES="0"),
    "agentx.yaml-owned fields: ['model']": lambda r: r.update(model={"id": "first"}),
    "agentx.yaml-owned fields: ['hardware']": lambda r: r.update(hardware={}),
    "plan reads from stack.json: ['image']": lambda r: r.update(image={"run": IMAGE}),
    "source.kind": lambda r: r["source"].update(kind="other"),
    "source.kind must be 'cookbook'": lambda r: r["source"].update(kind="inferencex"),
    "source.url must be": lambda r: r["source"].update(url="https://elsewhere.invalid/model.json"),
    "holds only kind and url; remove ['ref']": lambda r: r["source"].update(ref="v1"),
    "unique": lambda r: r["points"].append(copy.deepcopy(r["points"][0])),
    "arm name": lambda r: r["points"][0].update(arm=""),
    "fit the target": lambda r: r["points"][0].update(gpus=16),
    "advisory env": lambda r: r["points"][0]["launch"]["env"].update(AKLR_SGLANG_PLUGIN="0"),
    "FRAMEWORK must be 'sglang'": lambda r: r["points"][1]["client_env"].update(FRAMEWORK="vllm"),
    "client_env.TP must be": lambda r: r["points"][0]["client_env"].pop("TP"),
    "client_env.DCP_SIZE must be": lambda r: r["points"][0]["client_env"].update(DCP_SIZE="0"),
    "must equal gpus": lambda r: r["points"][0]["client_env"].update(TP="4"),
    "DP_ATTENTION must be": lambda r: r["points"][0]["client_env"].update(DP_ATTENTION="True"),
    "plan-owned fields: ['exp_name']": lambda r: r["points"][0].update(exp_name="first-tp8-c1"),
    "AIPERF_EXPERIMENTAL_FAST must be '0'": lambda r: r["points"][0]["client_env"].update(
        AIPERF_EXPERIMENTAL_FAST="1"
    ),
    "CONC must equal its conc": lambda r: r["points"][0]["client_env"].update(CONC="999"),
    "TOTAL_CPU_DRAM_GB must be digits": lambda r: r["points"][0]["client_env"].pop("TOTAL_CPU_DRAM_GB"),
    "runtime.ready_timeout_s needs": lambda r: r["points"][0].update(runtime={}),
    "runtime.port is required": lambda r: r["points"][0]["runtime"].pop("port"),
    "runtime.port must be null or a TCP port": lambda r: r["points"][0]["runtime"].update(port=70000),
    "runtime.poll_s needs": lambda r: r["points"][0]["runtime"].update(poll_s=float("nan")),
}


@pytest.mark.parametrize(("message", "mutate"), INVALID_RECIPES.items(), ids=INVALID_RECIPES)
def test_invalid_recipes_are_rejected(case, message, mutate):
    recipe = case.recipe()
    mutate(recipe)
    with pytest.raises(sc.RecipeError, match=message.replace("[", r"\[").replace("]", r"\]")):
        sc.build_run(recipe, case.definition_of())


@pytest.mark.parametrize(
    ("message", "mutate"),
    [
        ("40-character sha", lambda d: d["client"]["const"].update(sha="main")),
        ("owner/name repo", lambda d: d["client"]["const"].update(repo="https://evil.invalid/x")),
        ("pinned revision", lambda d: d["trace"]["const"].update(revision="main")),
        ("needs loader", lambda d: d.pop("trace")),
    ],
    ids=["client-sha", "client-repo", "trace-revision", "no-trace"],
)
def test_client_and_trace_come_from_the_schema(case, message, mutate):
    broken = json.loads(sc.RECIPE_SCHEMA.read_text())
    mutate(broken["$defs"])
    path = case.root / "serving.schema.json"
    path.write_text(json.dumps(broken))
    with pytest.raises(sc.RecipeError, match=message):
        sc.template(path)
    case.monkeypatch.setattr(sc, "RECIPE_SCHEMA", path)
    with pytest.raises(sc.RecipeError, match=message):
        sc.build_run(case.recipe(), case.definition_of())


def test_plan_names_each_point_from_model_arm_and_conc(case):
    spec = sc.build_run(case.recipe(), case.definition_of())
    assert [p["exp_name"] for p in spec["points"]] == ["first-tp8-c1", "first-tp8-c2"]


@pytest.mark.parametrize(
    "name", ["/agentx/cache", "..", ".", "a/../b", "a\\b", "", ".hidden", "tp8:1"], ids=repr
)
def test_arms_cannot_leave_the_point_directory(case, name):
    recipe = case.recipe()
    recipe["points"][0]["arm"] = name
    with pytest.raises(sc.RecipeError, match="arm name"):
        sc.build_run(recipe, case.definition_of())


@pytest.mark.parametrize(
    ("message", "extra"),
    [
        ("source.config_key must be", {"config_key": "other-key"}),
        (
            r"holds only kind and config_key; remove \['files', 'repo', 'sha'\]",
            {
                "repo": "example/upstream",
                "sha": "e" * 40,
                "files": {"configs/master.yaml": "f" * 40, "recipes/model.yaml": "f" * 40},
            },
        ),
    ],
    ids=["other-config-key", "resolved-fields"],
)
def test_inferencex_recipes_must_match_the_configured_entry(case, message, extra):
    case.definition["models"]["first"]["targets"]["mi355x"]["frameworks"]["sglang"]["recipe_source"] = {
        "kind": "inferencex",
        "repo": "example/upstream",
        "ref": "main",
        "config_key": "model-key",
        "expect": {"config_file": "configs/master.yaml", "recipe": "recipes/model.yaml"},
    }
    case.write_config()
    source = {"kind": "inferencex", "config_key": "model-key"}
    recipe = case.recipe()
    recipe["source"] = source
    sc.build_run(recipe, case.definition_of())
    recipe = case.recipe()
    recipe["source"] = {**source, **extra}
    with pytest.raises(sc.RecipeError, match=message):
        sc.build_run(recipe, case.definition_of())


def test_json_schema_matches_the_contract(case):
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((AGENTX / "recipes/serving.schema.json").read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)
    validator.validate(case.recipe())
    for path in (AGENTX / "recipes").glob("*/*/*/serving.*.json"):
        validator.validate(json.loads(path.read_text()))
    run = sc.build_run(case.recipe(), case.definition_of())
    assert not validator.is_valid(run)


@pytest.mark.usefixtures("case")  # build_run reads the run image: keep stack.json out of this test
def test_repository_recipes_match_agentx_yaml():
    config = plan.load_config(AGENTX / "agentx.yaml")
    paths = sorted((AGENTX / "recipes").glob("*/*/*/serving.*.json"))
    assert paths
    assert len(paths) == len(plan.select_runs(config))
    # A loop, not parametrize: an empty recipes/ must fail here, and an empty parameter set is skipped.
    for path in paths:
        model_id = path.name.removeprefix("serving.").removesuffix(".json")
        target_id, framework_id = path.parent.parent.name, path.parent.name
        definition = plan.model_config(config, model_id, target_id, framework_id)
        assert sc.recipe_path(AGENTX / "agentx.yaml", definition) == path, path
        spec = sc.build_run(plan.read_json(path), definition)
        assert sc.selected_points(spec), path


@pytest.mark.parametrize("family", [None, "", "../GLM", "GLM/5"], ids=repr)
def test_recipes_are_kept_by_model_family(case, family):
    assert case.recipe_file("first", "other") == case.root / "recipes/Example/other/sglang/serving.first.json"
    case.definition["models"]["first"]["family"] = family
    case.write_config()
    with pytest.raises(plan.ConfigError, match="family must name its recipes/ directory"):
        case.definition_of()


def test_frameworks_are_a_separate_layer(case):
    assert (
        case.recipe_file("first", "mi355x", "vllm")
        == case.root / "recipes/Example/mi355x/vllm/serving.first.json"
    )
    sglang = case.definition_of()
    vllm = case.definition_of(framework_id="vllm")
    assert (sglang["framework"], vllm["framework"]) == ({"id": "sglang"}, {"id": "vllm"})
    assert sglang["advisory_env"] == {"AKLR_SGLANG_PLUGIN": "1"}
    assert vllm["advisory_env"] == {}
    spec = sc.build_run(case.recipe(framework_id="vllm"), vllm)
    assert spec["framework"] == {"id": "vllm"}
    assert "AKLR_SGLANG_PLUGIN" not in spec["points"][0]["launch"]["env"]
    with pytest.raises(plan.ConfigError, match="select exactly one framework for first/mi355x"):
        plan.model_config(plan.load_config(case.config), "first", "mi355x")
    assert case.definition_of("first", "other", None)["framework"] == {"id": "sglang"}
    config = plan.load_config(case.config)
    assert plan.select_runs(config) == list(RUNS)
    assert plan.select_runs(config, frameworks="vllm") == [run for run in RUNS if run[2] == "vllm"]
    with pytest.raises(plan.ConfigError, match="unknown or disabled frameworks in the selection: sglang"):
        plan.select_runs(config, models="second", frameworks="sglang")


def test_a_framework_entry_may_run_on_the_stacks_aiter(case):
    """aiter: stack names stack.json's AITER in the run's framework identity, so that the sweep installs it and
    turns AKLR's aiter provider on with the plugin; the default, the image's own, is left out of the identity."""
    frameworks = case.definition["models"]["first"]["targets"]["mi355x"]["frameworks"]
    frameworks["vllm"]["aiter"] = "stack"
    case.write_config()
    vllm = case.definition_of(framework_id="vllm")
    assert vllm["framework"] == {"id": "vllm", "aiter": "stack"}
    assert sc.build_run(case.recipe(framework_id="vllm"), vllm)["framework"] == {
        "id": "vllm",
        "aiter": "stack",
    }
    assert case.definition_of()["framework"] == {"id": "sglang"}
    frameworks["vllm"]["aiter"] = "image"
    case.write_config()
    assert case.definition_of(framework_id="vllm")["framework"] == {"id": "vllm"}
    frameworks["vllm"]["aiter"] = "main"
    case.write_config()
    with pytest.raises(plan.ConfigError, match="frameworks.vllm.aiter must be one of image, stack"):
        case.definition_of(framework_id="vllm")


def test_framework_ids_and_flags_are_validated(case):
    frameworks = case.definition["models"]["first"]["targets"]["mi355x"]["frameworks"]
    frameworks["vllm"]["enabled"] = False
    case.write_config()
    assert case.definition_of("first", "mi355x", None)["framework"] == {"id": "sglang"}
    with pytest.raises(plan.ConfigError, match="unknown or disabled framework first/mi355x/vllm"):
        case.definition_of(framework_id="vllm")
    with pytest.raises(plan.ConfigError, match="invalid framework id 'trtllm'"):
        plan.select_runs(plan.load_config(case.config), frameworks="trtllm")
    frameworks["trtllm"] = framework()
    case.write_config()
    with pytest.raises(plan.ConfigError, match="invalid framework id 'trtllm'"):
        case.definition_of()
    del case.definition["models"]["first"]["targets"]["mi355x"]["frameworks"]
    case.write_config()
    with pytest.raises(plan.ConfigError, match="frameworks must map sglang or vllm"):
        case.definition_of()


def test_runner_filter_selects_across_models_and_persists_only_matches(planned):
    args = planned.args(runner_label=RUNNER_LABEL)
    manifest = plan.build(args)
    expected = [run for run in RUNS if run[1] != "other"]
    assert planned.runs(manifest) == expected
    assert {item["status"] for item in manifest["targets"]} == {"resolved"}
    assert json.loads((args.out / "manifest.json").read_text()) == manifest
    assert {str(path.relative_to(args.out)) for path in args.out.rglob("serving.json")} == {
        f"models/{m}/targets/{t}/frameworks/{f}/serving.json" for m, t, f in expected
    }
    assert manifest["schema"] == 4
    for item, run in zip(manifest["targets"], expected):
        spec = planned.serving_of(args, *run)
        assert spec["framework"] == item["framework"]
        assert item["point_policy"] == sc.POLICY
        assert item["point_ids"] == ["tp8:1", "tp8:2"]
        assert item["image"] == IMAGE
        assert spec["image"] == {"run": IMAGE}
        assert item["serving_sha256"] == plan.sha256_json(spec)
        assert item["selection_manifest_sha256"] == spec["selection"]["manifest_sha256"]
        assert spec["plan_id"]
    # One stack.json read per framework, shared by its runs.
    assert sorted(c.args[0] for c in planned.run_image.call_args_list) == ["sglang", "vllm"]


def test_execution_image_comes_from_the_ci_image_of_the_framework(planned):
    vllm = "example.invalid/vllm@sha256:" + "b" * 64
    planned.run_image.side_effect = lambda framework: vllm if framework == "vllm" else IMAGE
    args = planned.args()
    manifest = plan.build(args)
    for item in manifest["targets"]:
        expected = vllm if item["framework"]["id"] == "vllm" else IMAGE
        assert item["image"] == expected
        run = (item["model"]["id"], item["target"]["id"], item["framework"]["id"])
        assert planned.serving_of(args, *run)["image"] == {"run": expected}
    with pytest.raises(sc.RecipeError, match="must be pinned by digest"):
        sc.build_run(planned.recipe(), planned.definition_of(), "example.invalid/server:latest")
    planned.run_image.side_effect = ValueError("stack.json: vllm_image must be an image tag pinned by digest")
    failed = plan.build(planned.args())
    assert {item["status"] for item in failed["targets"]} == {"resolve-failed"}
    assert "vllm_image" in failed["targets"][0]["error"]


def test_default_selection_works_without_runner_label_attribute(planned):
    args = planned.args()
    assert not hasattr(args, "runner_label")
    manifest = plan.build(args)
    assert planned.runs(manifest) == list(RUNS)
    assert {item["status"] for item in manifest["targets"]} == {"resolved"}


def test_missing_runner_fails_before_manifest_creation(planned):
    args = planned.args(runner_label="agentx-missing")
    with pytest.raises(plan.ConfigError, match="no selected runs use runner label 'agentx-missing'"):
        plan.build(args)
    assert not (args.out / "manifest.json").exists()


def test_missing_or_stale_recipe_only_fails_that_target(planned):
    planned.recipe_file("second", "different-target-id", "vllm").unlink()
    planned.definition["models"]["first"]["targets"]["other"]["hardware"]["gpu_count"] = 4
    planned.write_config()
    manifest = plan.build(planned.args())
    assert [item["status"] for item in manifest["targets"]] == [
        "resolved",
        "resolved",
        "resolve-failed",
        "resolve-failed",
    ]
    assert "fit the target hardware" in manifest["targets"][2]["error"]
    assert "must be generated and reviewed" in manifest["targets"][3]["error"]


def test_cli_filter_limits_manifest_and_github_job_matrix(planned):
    output = planned.root / "github-output"
    out = planned.root / "cli-plan"
    planned.monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    code = plan.main(
        [
            "--config",
            str(planned.config),
            "--runner-label",
            RUNNER_LABEL,
            "--out",
            str(out),
            "--run-id",
            "fixture",
        ]
    )
    assert code == 0
    values = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert values["count"] == "3"
    jobs = json.loads(values["matrix"])["include"]
    assert [(job["model"], job["target"], job["framework"], job["runner"]) for job in jobs] == [
        (*run, RUNNER_LABEL) for run in RUNS if run[1] != "other"
    ]
    manifest = json.loads((out / "manifest.json").read_text())
    assert planned.runs(manifest) == [(job["model"], job["target"], job["framework"]) for job in jobs]


def test_comment_selects_only_its_requested_models(planned):
    code, values, out = planned.run_with_comment("/agentx-mi355x `second`\r\nDecode changed.\n")
    assert code == 0
    assert values["skipped"] == "false"
    jobs = json.loads(values["matrix"])["include"]
    expected = [("second", "different-target-id", "vllm")]
    assert [(job["model"], job["target"], job["framework"]) for job in jobs] == expected
    assert planned.runs(json.loads((out / "manifest.json").read_text())) == expected
    assert "selects second;" in (planned.root / "github-summary").read_text()


@pytest.mark.parametrize(
    "body", ["/agentx-mi355x all\n", "/agentx-mi355x `ALL`\n"], ids=["all", "quoted-uppercase"]
)
def test_comment_all_runs_every_enabled_model(planned, body):
    code, values, _ = planned.run_with_comment(body)
    assert code == 0
    assert values["count"] == "3"


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("/agentx-mi355x first, missing\n", "Not an enabled model id, skipped: `missing`."),
        ("/agentx-mi355x missing\n", "selects no model"),
        ("/agentx-mi355x\n", "selects no model"),
    ],
    ids=["one-unknown-id", "only-unknown-ids", "no-ids"],
)
def test_comment_scope_problems_warn_on_github(planned, capsys, body, message):
    planned.monkeypatch.setenv("GITHUB_ACTIONS", "true")
    code, _, _ = planned.run_with_comment(body)
    assert code == 0
    warnings = [line for line in capsys.readouterr().out.splitlines() if line.startswith("::warning")]
    assert len(warnings) == 1
    assert message in warnings[0]
    planned.run_with_comment("/agentx-mi355x first\n")
    assert "::warning" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "body", ["/agentx-mi355x\nfirst\n", "/agentx-mi355x , \n"], ids=["ids-on-the-next-line", "only-commas"]
)
def test_comment_without_models_skips_without_a_plan(planned, body):
    code, values, out = planned.run_with_comment(body)
    assert code == 0
    assert values == {"matrix": '{"include": []}', "count": "0", "skipped": "true"}
    assert not out.exists()
    assert (
        "@RuibinCheung's `/agentx-mi355x` comment selects no model; skipping AgentX."
        in (planned.root / "github-summary").read_text()
    )


@pytest.mark.parametrize(
    ("body", "count", "models", "note"),
    [
        ("/agentx-mi355x first, missing\n", "2", ["first", "first"], "skipped: `missing`."),
        ("/agentx-mi355x all, first\n", "2", ["first", "first"], "skipped: `all`."),
        ("/agentx-mi355x FIRST\n", "2", ["first", "first"], "selects first;"),
        ("/AgentX-MI355X first\n/agentx-mi355x second\n", "2", ["first", "first"], "selects first;"),
        ("/agentx-mi355x none, second\n", "1", ["second"], "skipped: `none`."),
    ],
    ids=["unknown-id", "all-beside-an-id", "uppercase-id", "first-line-only", "none-beside-an-id"],
)
def test_comment_never_fails_the_plan(planned, body, count, models, note):
    code, values, _ = planned.run_with_comment(body)
    assert code == 0
    assert values["count"] == count
    assert [job["model"] for job in json.loads(values["matrix"])["include"]] == models
    assert note in (planned.root / "github-summary").read_text()


@pytest.mark.parametrize(
    ("body", "note"),
    [
        ("/agentx-mi355x none\n", "skipping AgentX"),
        ("/agentx-mi355x missing\n", "skipping AgentX"),
        ("/agentx-mi355x -, other\n", "skipped: `-`, `other`."),
    ],
    ids=["none", "unknown-id", "dash-and-unknown-id"],
)
def test_comment_matching_nothing_skips(planned, body, note):
    code, values, out = planned.run_with_comment(body)
    assert code == 0
    assert values["skipped"] == "true"
    assert not out.exists()
    assert "skipping AgentX" in (planned.root / "github-summary").read_text()
    assert note in (planned.root / "github-summary").read_text()


def test_comment_with_explicit_models_is_a_cli_error(planned):
    code, values, _ = planned.run_with_comment("/agentx-mi355x first\n", "--models", "second")
    assert (code, values) == (1, {})


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("/agentx-mi355x\tsecond, first,, second\n", ("second,first", [])),
        ("/agentx-mi355x `first` second\n", ("first,second", [])),
        ("/agentx-mi355x NONE\n", ("", ["NONE"])),
        ("", ("", [])),
        ("- /agentx-mi355x first\n", ("", [])),
        ("AgentX: first\n", ("", [])),
        ("    /agentx-mi355x first\n", ("", [])),
        (" /agentx-mi355x first\n", ("", [])),
    ],
    ids=[
        "tab-and-repeats",
        "quoted-id",
        "unknown-id",
        "empty",
        "list-item",
        "old-line",
        "code-indent",
        "one-space",
    ],
)
def test_comment_parsing(body, expected):
    assert requested_models(body, ["first", "second"], plan.COMMAND) == expected


def test_comment_matches_model_ids_ignoring_case():
    enabled = ["glm52-fp4", "minimaxm31flash-fp4"]
    body = "/agentx-mi355x glm52, GLM52-FP4, minimax, glm52-fp4\n"
    assert requested_models(body, enabled, plan.COMMAND) == ("glm52-fp4", ["glm52", "minimax"])
    assert requested_models("/agentx-mi355x glm52-fp4 not all of them\n", enabled, plan.COMMAND) == (
        "glm52-fp4",
        ["not", "all", "of", "them"],
    )


@pytest.mark.parametrize(
    "comments",
    [
        [],
        [{"user": "author", "body": "/agentx-mi355x all"}],
        [{"user": "RuibinCheung", "body": "Looks good.\n/agentx-mi355x first"}],
        [{"user": "RuibinCheung", "body": "/agentx-mi355xfirst"}],
        [{"user": "RuibinCheung", "body": "    /agentx-mi355x all"}],
        [{"user": "RuibinCheung", "body": "\t/agentx-mi355x all"}],
        [{"user": "RuibinCheung", "body": "```\n/agentx-mi355x all\n```"}],
        [{"user": "RuibinCheung", "body": "LGTM"}, {"user": "someone", "body": "/agentx-mi355x first"}],
    ],
    ids=[
        "no-comments",
        "author-only",
        "not-on-the-first-line",
        "no-space-after-command",
        "space-indented",
        "tab-indented",
        "code-fenced",
        "outsider-after-reviewer",
    ],
)
def test_plan_fails_without_an_allowlisted_reviewers_request(planned, capsys, comments):
    planned.monkeypatch.setenv("GITHUB_ACTIONS", "true")
    code, values, out = planned.run_with_comment("", comments=comments)
    assert (code, values) == (1, {})
    assert not out.exists()
    message = "No AgentX reviewer has authorized this PR. Ask one of @RuibinCheung, @xiaobochen-amd, @jiejingzhangamd"
    captured = capsys.readouterr()
    assert message in (planned.root / "github-summary").read_text()
    assert message in captured.err
    errors = [line for line in captured.out.splitlines() if line.startswith("::error")]
    assert len(errors) == 1
    assert "re-run this workflow" in errors[0]


def test_latest_allowlisted_request_selects_the_models(planned):
    comments = [
        {"user": "xiaobochen-amd", "body": "/agentx-mi355x all"},
        {"user": "JIEJINGZHANGAMD", "body": "/agentx-mi355x second", "url": "https://example.invalid/c/2"},
        {"user": "author", "body": "/agentx-mi355x first"},
        {"user": "RuibinCheung", "body": "Thanks.\n/agentx-mi355x first"},
    ]
    code, values, _ = planned.run_with_comment("", comments=comments)
    assert code == 0
    assert [job["model"] for job in json.loads(values["matrix"])["include"]] == ["second"]
    assert (
        "@JIEJINGZHANGAMD's `/agentx-mi355x` comment (https://example.invalid/c/2) selects second;"
        in (planned.root / "github-summary").read_text()
    )


def test_manual_run_by_an_allowlisted_reviewer_on_main_plans(planned):
    code, out, _ = planned.run_manual("ruibincheung", "refs/heads/main")
    assert code == 0
    assert (out / "manifest.json").exists()


@pytest.mark.parametrize(
    ("actor", "ref", "message"),
    [
        (
            "someone",
            "refs/heads/main",
            "@someone may not start AgentX manually. Ask one of @RuibinCheung, @xiaobochen-amd",
        ),
        ("RuibinCheung", "refs/heads/feature", "Manual AgentX runs use main only, not refs/heads/feature"),
    ],
    ids=["outsider", "other-ref"],
)
def test_manual_run_is_refused_off_the_allowlist_or_off_main(planned, capsys, actor, ref, message):
    code, out, summary = planned.run_manual(actor, ref)
    assert code == 1
    assert not out.exists()
    assert message in summary
    assert any(
        line.startswith("::error") and message in line for line in capsys.readouterr().out.splitlines()
    )


def test_manual_actor_needs_its_ref(case):
    with pytest.raises(SystemExit, match="2"):
        plan.main(
            ["--config", str(case.config), "--manual-actor", "RuibinCheung", "--out", str(case.root / "x")]
        )


@pytest.mark.parametrize(
    "text",
    ["{}", "reviewers: []\n", "reviewers: [ok, 'not a login']\n", "reviewers: [Ok, ok]\n"],
    ids=["no-reviewers-key", "empty", "invalid-login", "duplicate-ignoring-case"],
)
def test_reviewers_file_must_list_distinct_github_logins(tmp_path, text):
    path = tmp_path / "allowlist.yaml"
    path.write_text(text)
    with pytest.raises(plan.ConfigError, match=re.escape(str(path))):
        load_reviewers(path)


def agentx_workflow():
    return yaml.safe_load((AGENTX.parents[2] / ".github/workflows/agentx-ci.yaml").read_text())


def test_allowlist_and_workflow_read_the_pr_comments():
    assert "RuibinCheung" in load_reviewers()
    steps = agentx_workflow()["jobs"]["plan"]["steps"]
    read = next(
        step for step in steps if step.get("name") == "Read the PR comments that may authorize AgentX"
    )
    assert "issues/$PR_NUMBER/comments" in read["run"]
    assert "--paginate" in read["run"]
    resolve = next(step for step in steps if step.get("id") == "resolve")
    assert '--pr-comments "$RUNNER_TEMP/$PR_COMMENTS"' in resolve["run"]
    assert '--manual-actor "$MANUAL_ACTOR" --manual-ref "$GITHUB_REF"' in resolve["run"]
    assert (
        resolve["env"]["MANUAL_ACTOR"]
        == "${{ github.event_name == 'workflow_dispatch' && github.triggering_actor || '' }}"
    )
    main_allowlist = next(step for step in steps if step.get("name") == "Read main's AgentX allowlist")
    assert main_allowlist["if"] == "github.event_name != 'schedule'"
    assert "contents/benchmarks/perf/allowlist.yaml?ref=main" in main_allowlist["run"]
    assert '> "$RUNNER_TEMP/allowlist.yaml"' in main_allowlist["run"]
    assert '--allowlist "$RUNNER_TEMP/$ALLOWLIST"' in resolve["run"]


def test_workflow_queue_allowlist_matches_the_allowlist_file():
    group = agentx_workflow()["concurrency"]["group"]
    lists = re.findall(r"fromJSON\('(\[[^']*\])'\),\s*github\.triggering_actor\)", group)
    assert len(lists) == 1
    assert json.loads(lists[0]) == load_reviewers()
    assert "github.ref != 'refs/heads/main'" in group
    assert "format('agentx-refused-{0}', github.run_id)" in group


def test_sweep_and_report_need_this_attempts_plan():
    jobs = agentx_workflow()["jobs"]
    assert jobs["plan"]["outputs"]["attempt"] == "${{ github.run_attempt }}"
    for job in ("sweep", "report"):
        assert "needs.plan.outputs.attempt == github.run_attempt" in jobs[job]["if"]


def test_smoke_points_narrow_the_selection(planned):
    full = planned.serving_of_built()
    args = planned.args(
        models="first", frameworks="sglang", runner_label=RUNNER_LABEL, points="tp8:2", duration=60
    )
    manifest = plan.build(args)
    item = manifest["targets"][0]
    assert (item["status"], item["point_ids"]) == ("resolved", ["tp8:2"])
    spec = planned.serving_of(args)
    assert spec["selection"]["points"] == ["tp8:2"]
    assert spec["selection"]["points_requested"] == "tp8:2"
    assert spec["selection"]["duration"] == 60
    assert spec["selection"]["selection_sha256"] != full["selection"]["selection_sha256"]
    assert [sc.point_id(p) for p in sc.selected_points(spec)] == ["tp8:2"]


@pytest.mark.parametrize("options", [{}, {"runner_label": RUNNER_LABEL}], ids=["any-runner", "one-runner"])
def test_smoke_points_require_one_run_and_recipe_ids(planned, options):
    with pytest.raises(plan.ConfigError, match="smoke filters require exactly one"):
        plan.build(planned.args(models="first", points="tp8:1", **options))
    manifest = plan.build(
        planned.args(models="first", frameworks="vllm", runner_label=RUNNER_LABEL, points="tp8:3")
    )
    assert manifest["targets"][0]["status"] == "resolve-failed"
    assert "not in the recipe selection" in manifest["targets"][0]["error"]


def test_duration_override_applies_to_every_target(planned):
    args = planned.args(duration=600)
    manifest = plan.build(args)
    assert {item["status"] for item in manifest["targets"]} == {"resolved"}
    for path in args.out.rglob("serving.json"):
        selection = json.loads(path.read_text())["selection"]
        assert selection["duration"] == 600
        assert selection["points_requested"] is None


def test_requirement_preserves_recipe_duration_without_an_override(case):
    manifest = case.plan(require_recipe_duration=3600)
    assert {item["status"] for item in manifest["targets"]} == {"resolved"}
    for path in (case.root / "plan").rglob("serving.json"):
        assert json.loads(path.read_text())["selection"]["duration"] is None


def test_mismatch_is_reported_per_target(case):
    case.durations[("second", 1)] = "1200"
    manifest = case.plan(runner_label=RUNNER_LABEL, require_recipe_duration=3600)
    assert [item["status"] for item in manifest["targets"]] == ["resolved", "resolved", "resolve-failed"]
    assert "recipe DURATION must be 3600" in manifest["targets"][2]["error"]
    assert "tp8:1='1200'" in manifest["targets"][2]["error"]


def test_requirement_only_checks_selected_points(case):
    case.durations[("first", 1)] = "1200"
    manifest = case.plan(
        models="first",
        frameworks="sglang",
        runner_label=RUNNER_LABEL,
        points="tp8:2",
        require_recipe_duration=3600,
    )
    assert manifest["targets"][0]["status"] == "resolved"
    assert manifest["targets"][0]["point_ids"] == ["tp8:2"]


@pytest.mark.parametrize(
    "options",
    [
        {"require_recipe_duration": 0},
        {"require_recipe_duration": -1},
        {"require_recipe_duration": 3600, "duration": 3600},
    ],
    ids=["zero", "negative", "with-duration-override"],
)
def test_requirement_rejects_invalid_or_overridden_values(case, options):
    with pytest.raises(plan.ConfigError, match="--require-recipe-duration"):
        case.plan(**options)
