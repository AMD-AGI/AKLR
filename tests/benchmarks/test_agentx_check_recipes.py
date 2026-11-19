# SPDX-License-Identifier: MIT
"""The pre-commit recipe check applies plan's rules without Docker or a registry."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

# check_recipes imports jsonschema itself, so without it the whole module is skipped.
pytest.importorskip("jsonschema")

AGENTX = Path(__file__).resolve().parents[2] / "benchmarks/agentx"
from agentx_ci import check_recipes as check

FP4 = "recipes/GLM/mi355x/sglang/serving.glm52-fp4.json"


class CheckCase:
    """A copy of the repository's agentx.yaml and recipes/ that a test may break."""

    def __init__(self, root: Path, capsys: pytest.CaptureFixture[str]) -> None:
        self.root = root
        self.capsys = capsys
        shutil.copy(AGENTX / "agentx.yaml", self.root)
        shutil.copytree(AGENTX / "recipes", self.root / "recipes")

    def run_check(self, root: Path) -> tuple[int, str]:
        code = check.main(["--config", str(root / "agentx.yaml")])
        return code, self.capsys.readouterr().err

    def edit(self, relative: str, mutate) -> None:
        path = self.root / relative
        recipe = json.loads(path.read_text())
        mutate(recipe)
        path.write_text(json.dumps(recipe, indent=2) + "\n")


@pytest.fixture
def case(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> CheckCase:
    return CheckCase(tmp_path, capsys)


VIOLATIONS = {
    "plan: point tp4-ep4-dcp4-none-spec-mtp:24 contains plan-owned fields: ['exp_name']": lambda case: (
        case.edit(FP4, lambda r: r["points"][0].update(exp_name="x"))
    ),
    "schema: points/0/conc: 0 is less than the minimum of 1": lambda case: case.edit(
        FP4, lambda r: r["points"][0].update(conc=0)
    ),
    "plan: recipe source.config_key must be": lambda case: case.edit(
        FP4, lambda r: r["source"].update(config_key="other")
    ),
    "invalid JSON": lambda case: (case.root / FP4).write_text("{"),
    "recipes/ holds only": lambda case: (case.root / "recipes/GLM/notes.md").write_text("x"),
    "no enabled agentx.yaml run unknown/mi355x/sglang": lambda case: shutil.copy(
        case.root / FP4, case.root / "recipes/GLM/mi355x/sglang/serving.unknown.json"
    ),
    "agentx.yaml files this run's recipe at recipes/GLM/": lambda case: (
        (case.root / "recipes/Other/mi355x/sglang").mkdir(parents=True)
        or (case.root / FP4).rename(case.root / "recipes/Other/mi355x/sglang/serving.glm52-fp4.json")
    ),
}


def test_the_repository_recipes_pass(case):
    assert case.run_check(AGENTX) == (0, "")


@pytest.mark.parametrize(("message", "break_it"), VIOLATIONS.items(), ids=VIOLATIONS)
def test_violations_name_the_file_and_the_rule(case, message, break_it):
    break_it(case)
    code, stderr = case.run_check(case.root)
    assert code == 1
    assert message in stderr
