# SPDX-License-Identifier: MIT
"""The pre-commit recipe check applies plan's rules without Docker or a registry."""

from __future__ import annotations

import contextlib
import importlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import jsonschema
except ImportError:
    jsonschema = None

AGENTX = Path(__file__).resolve().parents[2] / "benchmarks/agentx"
FP4 = "recipes/GLM/mi355x/sglang/serving.glm52-fp4.json"


@unittest.skipIf(jsonschema is None, "jsonschema is not installed")
class CheckRecipesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with mock.patch.object(sys, "path", [str(AGENTX), *sys.path]):
            cls.check = importlib.import_module("agentx_ci.check_recipes")

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)
        shutil.copy(AGENTX / "agentx.yaml", self.root)
        shutil.copytree(AGENTX / "recipes", self.root / "recipes")

    def run_check(self, root: Path) -> tuple[int, str]:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
            code = self.check.main(["--config", str(root / "agentx.yaml")])
        return code, stderr.getvalue()

    def edit(self, relative: str, mutate) -> None:
        path = self.root / relative
        recipe = json.loads(path.read_text())
        mutate(recipe)
        path.write_text(json.dumps(recipe, indent=2) + "\n")

    def test_the_repository_recipes_pass(self):
        self.assertEqual(self.run_check(AGENTX), (0, ""))

    def test_violations_name_the_file_and_the_rule(self):
        cases = {
            "plan: point tp4-ep4-dcp4-none-spec-mtp:24 contains plan-owned fields: ['exp_name']": lambda: (
                self.edit(FP4, lambda r: r["points"][0].update(exp_name="x"))
            ),
            "schema: points/0/conc: 0 is less than the minimum of 1": lambda: self.edit(
                FP4, lambda r: r["points"][0].update(conc=0)
            ),
            "plan: recipe source.config_key must be": lambda: self.edit(
                FP4, lambda r: r["source"].update(config_key="other")
            ),
            "invalid JSON": lambda: (self.root / FP4).write_text("{"),
            "recipes/ holds only": lambda: (self.root / "recipes/GLM/notes.md").write_text("x"),
            "no enabled agentx.yaml run unknown/mi355x/sglang": lambda: shutil.copy(
                self.root / FP4, self.root / "recipes/GLM/mi355x/sglang/serving.unknown.json"
            ),
            "agentx.yaml files this run's recipe at recipes/GLM/": lambda: (
                (self.root / "recipes/Other/mi355x/sglang").mkdir(parents=True)
                or (self.root / FP4).rename(self.root / "recipes/Other/mi355x/sglang/serving.glm52-fp4.json")
            ),
        }
        for message, break_it in cases.items():
            with self.subTest(message=message):
                self.setUp()
                break_it()
                code, stderr = self.run_check(self.root)
                self.assertEqual(code, 1)
                self.assertIn(message, stderr)


if __name__ == "__main__":
    unittest.main()
