# SPDX-License-Identifier: MIT
"""AgentX recipe serving.json contract and recipe-based plans; runs without network or GPU access."""

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

import yaml

AGENTX = Path(__file__).resolve().parents[2] / "benchmarks/agentx"
with mock.patch.object(sys, "path", [str(AGENTX), *sys.path]):
    plan = importlib.import_module("agentx_ci.plan")
    sc = importlib.import_module("agentx_ci.serving_config")

try:
    import jsonschema
except ImportError:
    jsonschema = None

RUNNER_LABEL = "agentx-mi355x"
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


class AgentXRecipeTestCase(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
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
            "report": {"repo_url": "https://example.invalid/repo"},
        }
        self.write_config()
        self.durations = {}
        self.enterContext(redirect_stdout(io.StringIO()))
        self.enterContext(redirect_stderr(io.StringIO()))

    def write_config(self):
        self.config.write_text(yaml.safe_dump(self.definition, sort_keys=False))

    def definition_of(self, model_id="first", target_id="mi355x", framework_id="sglang"):
        return plan.model_config(plan.load_config(self.config), model_id, target_id, framework_id)

    def recipe(self, model_id="first", target_id="mi355x", framework_id="sglang"):
        return {
            "schema": 4,
            "image": {"run": IMAGE, "recipe": "example.invalid/upstream:tag"},
            "source": {"kind": "cookbook", "url": "https://recipes.example.invalid/model.json", "ref": "v1"},
            "client": {"repo": "example/inferencex", "sha": "c" * 40},
            "trace": {"loader": "fixture", "repo": "example/traces", "revision": "d" * 40},
            "points": [
                {
                    "arm": "tp8",
                    "conc": conc,
                    "exp_name": f"{model_id}-tp8-c{conc}",
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


class AgentXRecipeContractTests(AgentXRecipeTestCase):
    def test_build_run_adds_plan_owned_fields(self):
        recipe = self.recipe()
        spec = sc.build_run(recipe, self.definition_of())
        self.assertNotIn("hashes", recipe["points"][0])
        self.assertEqual(
            spec["points"][0]["launch"]["env"],
            {"AKLR_SGLANG_PLUGIN": "1", "HF_HUB_OFFLINE": "1", "HIP_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"},
        )
        definition = self.definition_of()
        for key in sc.AGENTX_TOP:
            self.assertEqual(spec[key], definition[key])
        self.assertEqual(spec["points"][1]["trace"], recipe["trace"])
        self.assertEqual(spec["points"][1]["client_env"]["WEKA_LOADER_OVERRIDE"], "fixture")
        self.assertEqual(spec["selection"]["points"], ["tp8:1", "tp8:2"])
        self.assertEqual(spec["arms"]["tp8"]["concs"], [1, 2])
        self.assertTrue(spec["arms"]["tp8"]["arm_serve_sha256"])
        self.assertTrue(spec["grid_sha256"])
        self.assertEqual(spec["selection"]["policy"], sc.POLICY)
        self.assertIsNone(spec["selection"]["duration"])
        self.assertEqual([sc.point_id(p) for p in sc.selected_points(spec)], ["tp8:1", "tp8:2"])

    def test_serve_identity_ignores_advisory_env_and_tracks_recipe_changes(self):
        base = sc.build_run(self.recipe(), self.definition_of())
        self.definition["models"]["first"]["advisory_env"] = {"EXTRA": "x"}
        self.write_config()
        advisory = sc.build_run(self.recipe(), self.definition_of())
        self.assertEqual(base["selection"]["selection_sha256"], advisory["selection"]["selection_sha256"])
        changed = self.recipe()
        changed["points"][0]["launch"]["command"].append("--enforce-eager")
        changed = sc.build_run(changed, self.definition_of())
        self.assertNotEqual(base["selection"]["serve_sha256"], changed["selection"]["serve_sha256"])
        self.assertNotEqual(base["points"][0]["hashes"], changed["points"][0]["hashes"])
        # Point-by-point settings decide which points still compare; the selected points are unchanged.
        self.assertEqual(base["selection"]["selection_sha256"], changed["selection"]["selection_sha256"])

    def test_invalid_recipes_are_rejected(self):
        cases = {
            "plan-owned": lambda r: r.update(grid_sha256="0" * 64),
            "plan-owned fields: ['hashes']": lambda r: r["points"][0].update(hashes={}),
            "plan-owned fields: ['selection']": lambda r: r.update(selection={"method": "measured-frontier"}),
            "plan-owned fields: ['arms']": lambda r: r.update(arms={"tp8": {}}),
            "plan-owned fields: ['trace']": lambda r: r["points"][0].update(trace=r["trace"]),
            "plan-owned keys": lambda r: r["points"][0]["client_env"].update(WEKA_LOADER_OVERRIDE="x"),
            "must not set devices": lambda r: r["points"][0]["launch"]["env"].update(HIP_VISIBLE_DEVICES="0"),
            "agentx.yaml-owned fields: ['model']": lambda r: r.update(model={"id": "first"}),
            "agentx.yaml-owned fields: ['hardware']": lambda r: r.update(hardware={}),
            "pinned by digest": lambda r: r["image"].update(run="example.invalid/server:latest"),
            "source.kind": lambda r: r["source"].update(kind="other"),
            "source.kind must be 'cookbook'": lambda r: r["source"].update(kind="inferencex"),
            "source.url must be": lambda r: r["source"].update(url="https://elsewhere.invalid/model.json"),
            "40-character sha": lambda r: r["client"].update(sha="main"),
            "unique": lambda r: r["points"].append(copy.deepcopy(r["points"][0])),
            "arm name": lambda r: r["points"][0].update(arm=""),
            "fit the target": lambda r: r["points"][0].update(gpus=16),
            "pinned revision": lambda r: r["trace"].update(revision="main"),
            "advisory env": lambda r: r["points"][0]["launch"]["env"].update(AKLR_SGLANG_PLUGIN="0"),
            "FRAMEWORK must be 'sglang'": lambda r: r["points"][1]["client_env"].update(FRAMEWORK="vllm"),
            "client_env.TP must be": lambda r: r["points"][0]["client_env"].pop("TP"),
            "client_env.DCP_SIZE must be": lambda r: r["points"][0]["client_env"].update(DCP_SIZE="0"),
            "must equal gpus": lambda r: r["points"][0]["client_env"].update(TP="4"),
            "DP_ATTENTION must be": lambda r: r["points"][0]["client_env"].update(DP_ATTENTION="True"),
            "exp_name must be a single path component": lambda r: r["points"][0].update(
                exp_name="/agentx/cache"
            ),
            "unique exp_name": lambda r: r["points"][1].update(exp_name=r["points"][0]["exp_name"]),
            "owner/name repo": lambda r: r["client"].update(repo="https://evil.invalid/x"),
            "AIPERF_EXPERIMENTAL_FAST must be '0'": lambda r: r["points"][0]["client_env"].update(
                AIPERF_EXPERIMENTAL_FAST="1"
            ),
            "CONC must equal its conc": lambda r: r["points"][0]["client_env"].update(CONC="999"),
            "TOTAL_CPU_DRAM_GB must be digits": lambda r: r["points"][0]["client_env"].pop(
                "TOTAL_CPU_DRAM_GB"
            ),
            "runtime.ready_timeout_s needs": lambda r: r["points"][0].update(runtime={}),
            "runtime.port is required": lambda r: r["points"][0]["runtime"].pop("port"),
            "runtime.port must be null or a TCP port": lambda r: r["points"][0]["runtime"].update(port=70000),
            "runtime.poll_s needs": lambda r: r["points"][0]["runtime"].update(poll_s=float("nan")),
        }
        for message, mutate in cases.items():
            with self.subTest(message=message):
                recipe = self.recipe()
                mutate(recipe)
                with self.assertRaisesRegex(sc.RecipeError, message.replace("[", r"\[").replace("]", r"\]")):
                    sc.build_run(recipe, self.definition_of())

    def test_exp_names_cannot_leave_the_point_directory(self):
        for name in ("/agentx/cache", "..", ".", "a/../b", "a\\b", "", ".hidden"):
            with self.subTest(name=name):
                recipe = self.recipe()
                recipe["points"][0]["exp_name"] = name
                with self.assertRaises(sc.RecipeError):
                    sc.build_run(recipe, self.definition_of())

    def test_inferencex_recipes_must_match_the_configured_entry(self):
        expect = {"config_file": "configs/master.yaml", "recipe": "recipes/model.yaml"}
        configured = {
            "kind": "inferencex",
            "repo": "example/upstream",
            "ref": "main",
            "config_key": "model-key",
        }
        self.definition["models"]["first"]["targets"]["mi355x"]["frameworks"]["sglang"]["recipe_source"] = {
            **configured,
            "expect": expect,
        }
        self.write_config()
        source = {**configured, "sha": "e" * 40, "files": {path: "f" * 40 for path in expect.values()}}
        recipe = self.recipe()
        recipe["source"] = source
        sc.build_run(recipe, self.definition_of())
        cases = {
            "source.repo must be": lambda s: s.update(repo="example/fork"),
            "source.config_key must be": lambda s: s.update(config_key="other-key"),
            "missing the agentx.yaml recipe_source files": lambda s: s["files"].pop("recipes/model.yaml"),
        }
        for message, mutate in cases.items():
            with self.subTest(message=message):
                recipe = self.recipe()
                recipe["source"] = copy.deepcopy(source)
                mutate(recipe["source"])
                with self.assertRaisesRegex(sc.RecipeError, message):
                    sc.build_run(recipe, self.definition_of())

    @unittest.skipIf(jsonschema is None, "jsonschema is not installed")
    def test_json_schema_matches_the_contract(self):
        schema = json.loads((AGENTX / "recipes/serving.schema.json").read_text())
        jsonschema.Draft202012Validator.check_schema(schema)
        validator = jsonschema.Draft202012Validator(schema)
        validator.validate(self.recipe())
        for path in (AGENTX / "recipes").glob("*/*/*/serving.*.json"):
            validator.validate(json.loads(path.read_text()))
        run = sc.build_run(self.recipe(), self.definition_of())
        self.assertFalse(validator.is_valid(run))

    def test_repository_recipes_match_agentx_yaml(self):
        config = plan.load_config(AGENTX / "agentx.yaml")
        paths = sorted((AGENTX / "recipes").glob("*/*/*/serving.*.json"))
        self.assertTrue(paths)
        self.assertEqual(len(paths), len(plan.select_runs(config)))
        for path in paths:
            model_id = path.name.removeprefix("serving.").removesuffix(".json")
            target_id, framework_id = path.parent.parent.name, path.parent.name
            with self.subTest(recipe=f"{model_id}/{target_id}/{framework_id}"):
                definition = plan.model_config(config, model_id, target_id, framework_id)
                self.assertEqual(sc.recipe_path(AGENTX / "agentx.yaml", definition), path)
                spec = sc.build_run(plan.read_json(path), definition)
                self.assertTrue(sc.selected_points(spec))

    def test_recipes_are_kept_by_model_family(self):
        self.assertEqual(
            self.recipe_file("first", "other"), self.root / "recipes/Example/other/sglang/serving.first.json"
        )
        for family in (None, "", "../GLM", "GLM/5"):
            with self.subTest(family=family):
                self.definition["models"]["first"]["family"] = family
                self.write_config()
                with self.assertRaisesRegex(plan.ConfigError, "family must name its recipes/ directory"):
                    self.definition_of()

    def test_frameworks_are_a_separate_layer(self):
        self.assertEqual(
            self.recipe_file("first", "mi355x", "vllm"),
            self.root / "recipes/Example/mi355x/vllm/serving.first.json",
        )
        sglang = self.definition_of()
        vllm = self.definition_of(framework_id="vllm")
        self.assertEqual((sglang["framework"], vllm["framework"]), ({"id": "sglang"}, {"id": "vllm"}))
        self.assertEqual(sglang["advisory_env"], {"AKLR_SGLANG_PLUGIN": "1"})
        self.assertEqual(vllm["advisory_env"], {})
        spec = sc.build_run(self.recipe(framework_id="vllm"), vllm)
        self.assertEqual(spec["framework"], {"id": "vllm"})
        self.assertNotIn("AKLR_SGLANG_PLUGIN", spec["points"][0]["launch"]["env"])
        with self.assertRaisesRegex(plan.ConfigError, "select exactly one framework for first/mi355x"):
            plan.model_config(plan.load_config(self.config), "first", "mi355x")
        self.assertEqual(self.definition_of("first", "other", None)["framework"], {"id": "sglang"})
        config = plan.load_config(self.config)
        self.assertEqual(plan.select_runs(config), list(RUNS))
        self.assertEqual(
            plan.select_runs(config, frameworks="vllm"), [run for run in RUNS if run[2] == "vllm"]
        )
        with self.assertRaisesRegex(
            plan.ConfigError, "unknown or disabled frameworks in the selection: sglang"
        ):
            plan.select_runs(config, models="second", frameworks="sglang")

    def test_framework_ids_and_flags_are_validated(self):
        frameworks = self.definition["models"]["first"]["targets"]["mi355x"]["frameworks"]
        frameworks["vllm"]["enabled"] = False
        self.write_config()
        self.assertEqual(self.definition_of("first", "mi355x", None)["framework"], {"id": "sglang"})
        with self.assertRaisesRegex(plan.ConfigError, "unknown or disabled framework first/mi355x/vllm"):
            self.definition_of(framework_id="vllm")
        with self.assertRaisesRegex(plan.ConfigError, "invalid framework id 'trtllm'"):
            plan.select_runs(plan.load_config(self.config), frameworks="trtllm")
        frameworks["trtllm"] = framework()
        self.write_config()
        with self.assertRaisesRegex(plan.ConfigError, "invalid framework id 'trtllm'"):
            self.definition_of()
        del self.definition["models"]["first"]["targets"]["mi355x"]["frameworks"]
        self.write_config()
        with self.assertRaisesRegex(plan.ConfigError, "frameworks must map sglang or vllm"):
            self.definition_of()


class AgentXPlanTests(AgentXRecipeTestCase):
    def setUp(self):
        super().setUp()
        self.write_recipes()
        self.next_run = 0

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

    def test_runner_filter_selects_across_models_and_persists_only_matches(self):
        args = self.args(runner_label=RUNNER_LABEL)
        manifest = plan.build(args)
        expected = [run for run in RUNS if run[1] != "other"]
        self.assertEqual(self.runs(manifest), expected)
        self.assertEqual({item["status"] for item in manifest["targets"]}, {"resolved"})
        self.assertEqual(json.loads((args.out / "manifest.json").read_text()), manifest)
        self.assertEqual(
            {str(path.relative_to(args.out)) for path in args.out.rglob("serving.json")},
            {f"models/{m}/targets/{t}/frameworks/{f}/serving.json" for m, t, f in expected},
        )
        self.assertEqual(manifest["schema"], 4)
        for item, run in zip(manifest["targets"], expected):
            spec = self.serving_of(args, *run)
            self.assertEqual(spec["framework"], item["framework"])
            self.assertEqual(item["point_policy"], sc.POLICY)
            self.assertEqual(item["point_ids"], ["tp8:1", "tp8:2"])
            self.assertEqual(item["image"], IMAGE)
            self.assertEqual(item["serving_sha256"], plan.sha256_json(spec))
            self.assertEqual(item["selection_manifest_sha256"], spec["selection"]["manifest_sha256"])
            self.assertTrue(spec["plan_id"])

    def test_default_selection_works_without_runner_label_attribute(self):
        args = self.args()
        self.assertFalse(hasattr(args, "runner_label"))
        manifest = plan.build(args)
        self.assertEqual(self.runs(manifest), list(RUNS))
        self.assertEqual({item["status"] for item in manifest["targets"]}, {"resolved"})

    def test_missing_runner_fails_before_manifest_creation(self):
        args = self.args(runner_label="agentx-missing")
        with self.assertRaisesRegex(plan.ConfigError, "no selected runs use runner label 'agentx-missing'"):
            plan.build(args)
        self.assertFalse((args.out / "manifest.json").exists())

    def test_missing_or_stale_recipe_only_fails_that_target(self):
        self.recipe_file("second", "different-target-id", "vllm").unlink()
        self.definition["models"]["first"]["targets"]["other"]["hardware"]["gpu_count"] = 4
        self.write_config()
        manifest = plan.build(self.args())
        self.assertEqual(
            [item["status"] for item in manifest["targets"]],
            ["resolved", "resolved", "resolve-failed", "resolve-failed"],
        )
        self.assertIn("fit the target hardware", manifest["targets"][2]["error"])
        self.assertIn("must be generated and reviewed", manifest["targets"][3]["error"])

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
                    "--out",
                    str(out),
                    "--run-id",
                    "fixture",
                ]
            )
        self.assertEqual(code, 0)
        values = dict(line.split("=", 1) for line in output.read_text().splitlines())
        self.assertEqual(values["count"], "3")
        jobs = json.loads(values["matrix"])["include"]
        self.assertEqual(
            [(job["model"], job["target"], job["framework"], job["runner"]) for job in jobs],
            [(*run, RUNNER_LABEL) for run in RUNS if run[1] != "other"],
        )
        manifest = json.loads((out / "manifest.json").read_text())
        self.assertEqual(
            self.runs(manifest), [(job["model"], job["target"], job["framework"]) for job in jobs]
        )

    def run_with_pr_body(self, body: str, *extra: str):
        output, summary = self.root / "github-output", self.root / "github-summary"
        output.unlink(missing_ok=True)
        summary.unlink(missing_ok=True)
        pr_body = self.root / "pr-body.md"
        pr_body.write_text(body)
        self.next_run += 1
        out = self.root / f"pr-plan-{self.next_run}"
        env = {"GITHUB_OUTPUT": str(output), "GITHUB_STEP_SUMMARY": str(summary)}
        with mock.patch.dict(os.environ, env):
            code = plan.main(
                [
                    "--config",
                    str(self.config),
                    "--runner-label",
                    RUNNER_LABEL,
                    "--pr-body",
                    str(pr_body),
                    *extra,
                    "--out",
                    str(out),
                ]
            )
        values = (
            dict(line.split("=", 1) for line in output.read_text().splitlines()) if output.exists() else {}
        )
        return code, values, out

    def test_pr_description_selects_only_its_declared_models(self):
        code, values, out = self.run_with_pr_body("## Risk\nAffects: decode\n**AgentX**: `second`\n")
        self.assertEqual(code, 0)
        self.assertEqual(values["skipped"], "false")
        jobs = json.loads(values["matrix"])["include"]
        expected = [("second", "different-target-id", "vllm")]
        self.assertEqual([(job["model"], job["target"], job["framework"]) for job in jobs], expected)
        self.assertEqual(self.runs(json.loads((out / "manifest.json").read_text())), expected)
        self.assertIn("selects second;", (self.root / "github-summary").read_text())

    def test_pr_description_all_runs_every_enabled_model(self):
        for body in ("AgentX: all\n", "AgentX: `ALL`\n"):
            with self.subTest(body=body):
                code, values, _ = self.run_with_pr_body(body)
                self.assertEqual(code, 0)
                self.assertEqual(values["count"], "3")

    def test_pr_description_scope_problems_warn_on_github(self):
        cases = (
            ("AgentX: first, missing\n", "Not an enabled model id, skipped: `missing`."),
            ("AgentX: missing\n", "selects no model"),
            ("No line\n", "selects no model"),
        )
        for body, message in cases:
            with self.subTest(body=body), mock.patch.dict(os.environ, {"GITHUB_ACTIONS": "true"}):
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    code, _, _ = self.run_with_pr_body(body)
                self.assertEqual(code, 0)
                warnings = [line for line in stdout.getvalue().splitlines() if line.startswith("::warning")]
                self.assertEqual(len(warnings), 1)
                self.assertIn(message, warnings[0])
        stdout = io.StringIO()
        with mock.patch.dict(os.environ, {"GITHUB_ACTIONS": "true"}), redirect_stdout(stdout):
            self.run_with_pr_body("AgentX: first\n")
        self.assertNotIn("::warning", stdout.getvalue())

    def test_pr_description_without_models_skips_without_a_plan(self):
        template = "AgentX: <!-- model ids from agentx.yaml,\nor all -->\n"
        for body in ("## Why\nNo AgentX line.\n", template, "AgentX:\n"):
            with self.subTest(body=body):
                code, values, out = self.run_with_pr_body(body)
                self.assertEqual(code, 0)
                self.assertEqual(values, {"matrix": '{"include": []}', "count": "0", "skipped": "true"})
                self.assertFalse(out.exists())
                self.assertIn(
                    "selects no model on an `AgentX:` line; skipping AgentX.",
                    (self.root / "github-summary").read_text(),
                )

    def test_pr_description_never_fails_the_plan(self):
        cases = (
            ("AgentX: first, missing\n", "2", ["first", "first"], "skipped: `missing`."),
            ("AgentX: all, first\n", "2", ["first", "first"], "skipped: `all`."),
            ("AgentX: FIRST\n", "2", ["first", "first"], "selects first;"),
            ("AgentX: first\n* AgentX: second\n", "3", ["first", "first", "second"], "selects first,second"),
            ("AgentX: none, second\n", "1", ["second"], "skipped: `none`."),
        )
        for body, count, models, note in cases:
            with self.subTest(body=body):
                code, values, _ = self.run_with_pr_body(body)
                self.assertEqual(code, 0)
                self.assertEqual(values["count"], count)
                self.assertEqual([job["model"] for job in json.loads(values["matrix"])["include"]], models)
                self.assertIn(note, (self.root / "github-summary").read_text())

    def test_pr_description_matching_nothing_skips(self):
        for body in ("AgentX: none\n", "AgentX: missing\n", "AgentX: -, other\n"):
            with self.subTest(body=body):
                code, values, out = self.run_with_pr_body(body)
                self.assertEqual(code, 0)
                self.assertEqual(values["skipped"], "true")
                self.assertFalse(out.exists())
                self.assertIn("skipping AgentX", (self.root / "github-summary").read_text())
        self.assertIn("skipped: `-`, `other`.", (self.root / "github-summary").read_text())

    def test_pr_body_with_explicit_models_is_a_cli_error(self):
        code, values, _ = self.run_with_pr_body("AgentX: first\n", "--models", "second")
        self.assertEqual((code, values), (1, {}))

    def test_pr_description_parsing(self):
        enabled = ["first", "second"]
        self.assertEqual(
            plan.declared_models("- AgentX: second, first,, second\n", enabled), ("second,first", [])
        )
        self.assertEqual(plan.declared_models("AgentX: `first` second\n", enabled), ("first,second", []))
        self.assertEqual(plan.declared_models("**AgentX:** NONE\n", enabled), ("", ["NONE"]))
        self.assertEqual(plan.declared_models("", enabled), ("", []))
        self.assertEqual(plan.declared_models("Uses AgentX: later.\n## AgentX: first\n", enabled), ("", []))

    def test_pr_description_matches_model_ids_ignoring_case(self):
        enabled = ["glm52-fp4", "minimaxm31flash-fp4"]
        body = "AgentX: glm52, GLM52-FP4, minimax, glm52-fp4\n"
        self.assertEqual(plan.declared_models(body, enabled), ("glm52-fp4", ["glm52", "minimax"]))
        self.assertEqual(
            plan.declared_models("AgentX: glm52-fp4 not all of them\n", enabled),
            ("glm52-fp4", ["not", "all", "of", "them"]),
        )

    def test_smoke_points_narrow_the_selection(self):
        full = self.serving_of_built()
        args = self.args(
            models="first", frameworks="sglang", runner_label=RUNNER_LABEL, points="tp8:2", duration=60
        )
        manifest = plan.build(args)
        item = manifest["targets"][0]
        self.assertEqual((item["status"], item["point_ids"]), ("resolved", ["tp8:2"]))
        spec = self.serving_of(args)
        self.assertEqual(spec["selection"]["points"], ["tp8:2"])
        self.assertEqual(spec["selection"]["points_requested"], "tp8:2")
        self.assertEqual(spec["selection"]["duration"], 60)
        self.assertNotEqual(spec["selection"]["selection_sha256"], full["selection"]["selection_sha256"])
        self.assertEqual([sc.point_id(p) for p in sc.selected_points(spec)], ["tp8:2"])

    def serving_of_built(self):
        args = self.args(models="first", targets="mi355x", frameworks="sglang")
        plan.build(args)
        return self.serving_of(args)

    def test_smoke_points_require_one_run_and_recipe_ids(self):
        for options in ({}, {"runner_label": RUNNER_LABEL}):
            with self.subTest(options=options):
                with self.assertRaisesRegex(plan.ConfigError, "smoke filters require exactly one"):
                    plan.build(self.args(models="first", points="tp8:1", **options))
        manifest = plan.build(
            self.args(models="first", frameworks="vllm", runner_label=RUNNER_LABEL, points="tp8:3")
        )
        self.assertEqual(manifest["targets"][0]["status"], "resolve-failed")
        self.assertIn("not in the recipe selection", manifest["targets"][0]["error"])

    def test_duration_override_applies_to_every_target(self):
        args = self.args(duration=600)
        manifest = plan.build(args)
        self.assertEqual({item["status"] for item in manifest["targets"]}, {"resolved"})
        for path in args.out.rglob("serving.json"):
            selection = json.loads(path.read_text())["selection"]
            self.assertEqual(selection["duration"], 600)
            self.assertIsNone(selection["points_requested"])


class AgentXRecipeDurationTests(AgentXRecipeTestCase):
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

    def test_requirement_preserves_recipe_duration_without_an_override(self):
        manifest = self.plan(require_recipe_duration=3600)
        self.assertEqual({item["status"] for item in manifest["targets"]}, {"resolved"})
        for path in (self.root / "plan").rglob("serving.json"):
            self.assertIsNone(json.loads(path.read_text())["selection"]["duration"])

    def test_mismatch_is_reported_per_target(self):
        self.durations[("second", 1)] = "1200"
        manifest = self.plan(runner_label=RUNNER_LABEL, require_recipe_duration=3600)
        self.assertEqual(
            [item["status"] for item in manifest["targets"]], ["resolved", "resolved", "resolve-failed"]
        )
        self.assertIn("recipe DURATION must be 3600", manifest["targets"][2]["error"])
        self.assertIn("tp8:1='1200'", manifest["targets"][2]["error"])

    def test_requirement_only_checks_selected_points(self):
        self.durations[("first", 1)] = "1200"
        manifest = self.plan(
            models="first",
            frameworks="sglang",
            runner_label=RUNNER_LABEL,
            points="tp8:2",
            require_recipe_duration=3600,
        )
        self.assertEqual(manifest["targets"][0]["status"], "resolved")
        self.assertEqual(manifest["targets"][0]["point_ids"], ["tp8:2"])

    def test_requirement_rejects_invalid_or_overridden_values(self):
        for options in (
            {"require_recipe_duration": 0},
            {"require_recipe_duration": -1},
            {"require_recipe_duration": 3600, "duration": 3600},
        ):
            with self.subTest(options=options):
                with self.assertRaisesRegex(plan.ConfigError, "--require-recipe-duration"):
                    self.plan(**options)


if __name__ == "__main__":
    unittest.main()
