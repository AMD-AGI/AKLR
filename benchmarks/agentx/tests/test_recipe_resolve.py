# SPDX-License-Identifier: MIT
"""Resolution against InferenceX 70a7601's own output (tests/data), and the render rules."""

from __future__ import annotations

import copy
import unittest
from pathlib import Path

from agentx_ci import recipe_resolve as rr
from agentx_ci.common import load_config, read_json

DATA = Path(__file__).with_name("data")
IXOUT = read_json(DATA / "ix_points_70a7601.json")
RUN_IMAGE = "lmsysorg/sglang:v0.5.22-rocm10-mi35x@sha256:" + "1" * 64
# Pinned in the design prototype for the same InferenceX commit.
ARM_SERVE_SHA256 = {
    "tp4": "0520fc480a17304d6e2766f086e7f7eee1b5d318867b302e53b095a0c095d510",
    "tp8": "f91fd907a37cd338de2db971e759fca8a6a80c78f57c39353c372b029b8e72b1",
}


def build(ixout=IXOUT, config=None, client_defaults=None):
    return rr.build_spec(
        ixout,
        config or load_config(),
        run_image=RUN_IMAGE,
        client_defaults=client_defaults or {},
        provenance={
            "inferencex": {
                "repo": "SemiAnalysisAI/InferenceX",
                "sha": "70a7601777e6795c52a52b4dcde0022e0b93f773",
                "config_key": IXOUT["config_key"],
            }
        },
    )


class GoldenTests(unittest.TestCase):
    def test_grid_and_arm_hashes(self):
        spec = build()
        self.assertEqual(len(spec["points"]), 9)
        self.assertEqual(spec["arms"]["tp8"]["concs"], [1, 2, 4])
        self.assertEqual(spec["arms"]["tp4"]["concs"], [4, 8, 10, 12, 14, 16])
        self.assertEqual(
            {a: v["arm_serve_sha256"] for a, v in spec["arms"].items()},
            ARM_SERVE_SHA256,
        )

    def test_point_layers(self):
        point = next(
            p for p in build()["points"] if p["arm"] == "tp4" and p["conc"] == 8
        )
        command, env = point["launch"]["command"], point["launch"]["env"]
        self.assertEqual(command[:3], ["python3", "-m", "sglang.launch_server"])
        self.assertIn("--enable-metrics", command)
        self.assertEqual(command[command.index("--max-running-requests") + 1], "16")
        self.assertEqual(command.count("--served-model-name"), 1)
        self.assertEqual(env["AKLR_SGLANG_PLUGIN"], "1")
        self.assertEqual(env["HIP_VISIBLE_DEVICES"], "0,1,2,3")
        self.assertEqual(env["SGLANG_SIMULATE_ACC_LEN"], str(point["golden_al"]))
        client = point["client_env"]
        self.assertEqual(
            (client["TP"], client["EP_SIZE"], client["CONC"]), ("4", "4", "8")
        )
        self.assertEqual(
            (client["KV_OFFLOADING"], client["KV_OFFLOAD_BACKEND"]), ("dram", "hicache")
        )
        self.assertEqual(client["ENABLE_AGENTX_POWER"], "false")
        self.assertNotIn("TOTAL_CPU_DRAM_GB", client)

    def test_image_difference_is_a_warning(self):
        warnings = build()["warnings"]
        self.assertTrue(any("differs from InferenceX" in w for w in warnings))

    def test_workflow_defaults_yield_to_the_recipe(self):
        defaults = {
            "AIPERF_HTTP_TCP_USER_TIMEOUT": "1",
            "AIPERF_USE_DYNAMO_CONV_AWARE_ROUTING": "1",
        }
        client = build(client_defaults=defaults)["points"][0]["client_env"]
        self.assertEqual(client["AIPERF_HTTP_TCP_USER_TIMEOUT"], "900000")
        self.assertEqual(client["AIPERF_USE_DYNAMO_CONV_AWARE_ROUTING"], "1")

    def test_advisory_env_may_not_override_the_recipe(self):
        config = load_config()
        config["advisory_env"] = {"SGLANG_OPT_USE_TOPK_V2": "false"}
        with self.assertRaisesRegex(rr.ResolveError, "advisory env"):
            build(config=config)

    def test_a_different_model_is_rejected(self):
        ixout = copy.deepcopy(IXOUT)
        ixout["entry"]["model"] = "other/model"
        with self.assertRaisesRegex(rr.ResolveError, "entry model"):
            build(ixout)

    def test_a_grid_change_is_a_warning_and_changes_the_hash(self):
        ixout = copy.deepcopy(IXOUT)
        ixout["points"] = [p for p in ixout["points"] if p["row"]["conc"] != 16]
        spec = build(ixout)
        self.assertTrue(any("8 points" in w for w in spec["warnings"]))
        self.assertNotEqual(spec["grid_sha256"], build()["grid_sha256"])
        self.assertNotEqual(
            spec["arms"]["tp4"]["arm_serve_sha256"], ARM_SERVE_SHA256["tp4"]
        )
        self.assertEqual(
            spec["arms"]["tp8"]["arm_serve_sha256"], ARM_SERVE_SHA256["tp8"]
        )


class RuleTests(unittest.TestCase):
    def test_config_to_cli_args(self):
        args = rr.config_to_cli_args(
            {"b_flag": True, "off": False, "a-list": [1, 2], "none": None, "tp": 4}
        )
        self.assertEqual(args, ["--a-list", "1", "2", "--b-flag", "--tp", "4"])

    def test_apply_golden(self):
        recipe = {"roles": {"agg": {"env": {"KEEP": "1", "DROP": "x"}}}}
        out = rr.apply_golden(
            recipe, ["--set", 'roles.agg.env.AL="3.6"', "--unset", "roles.agg.env.DROP"]
        )
        self.assertEqual(out["roles"]["agg"]["env"], {"KEEP": "1", "AL": "3.6"})
        self.assertIn("DROP", recipe["roles"]["agg"]["env"])

    def test_runtime_settings_keeps_aiperf_exports(self):
        text = "export OPENAI_API_KEY='EMPTY'\nexport AIPERF_A='1'\n# export AIPERF_B='2'\n"
        self.assertEqual(rr.runtime_settings(text), {"AIPERF_A": "1"})

    def test_locked_versions(self):
        lock = "".join(
            f'[[package]]\nname = "{name}"\nversion = "1.{i}"\n\n'
            for i, name in enumerate(rr.DEP_PACKAGES)
        )
        self.assertEqual(rr.locked_versions(lock)["pyyaml"], "1.2")
        with self.assertRaisesRegex(rr.ResolveError, "uv.lock lacks"):
            rr.locked_versions(lock.replace('"pyyaml"', '"other"'))

    def test_ci_image_comes_from_ci_yaml(self):
        from agentx_ci.common import REPO_ROOT

        digest = "sha256:" + "2" * 64
        image, warnings = rr.ci_image(
            REPO_ROOT / ".github/workflows/ci.yaml", lookup=lambda _: digest
        )
        self.assertIn("@sha256:", image)
        if image.endswith(digest):
            self.assertEqual(len(warnings), 1)

    def test_ci_image_without_digest(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            workflow = Path(tmp) / "ci.yaml"
            workflow.write_text("env:\n  AKLR_CI_IMAGE: img:tag\n")
            digest = "sha256:" + "3" * 64
            self.assertEqual(
                rr.ci_image(workflow, lookup=lambda _: digest)[0], f"img:tag@{digest}"
            )
            image, warnings = rr.ci_image(workflow, lookup=lambda _: None)
            self.assertEqual(image, "img:tag")
            self.assertIn("none could be looked up", warnings[0])
            workflow.write_text(f"env:\n  AKLR_CI_IMAGE: img:tag@{digest}\n")
            self.assertEqual(rr.ci_image(workflow), (f"img:tag@{digest}", []))


if __name__ == "__main__":
    unittest.main()
