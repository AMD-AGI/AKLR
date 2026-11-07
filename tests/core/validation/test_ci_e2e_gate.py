# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Exercise the publication workflow's gate with synthetic validation artifacts."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml


@pytest.mark.parametrize(
    "statuses", [["passed"], ["failed"], ["aborted"], ["skipped"], [], ["passed", "failed"]]
)
def test_publication_requires_one_passed_e2e_report(tmp_path: Path, statuses: list[str]) -> None:
    if shutil.which("jq") is None:
        pytest.skip("the CI workflow uses jq")
    workflow = yaml.safe_load(Path(".github/workflows/ci.yaml").read_text())
    steps = workflow["jobs"]["publish-measurement"]["steps"]
    gate = next(s for s in steps if s.get("name") == "Require a completed e2e accuracy gate")
    rebuild = next(s for s in steps if s.get("name") == "Rebuild the bot branch")
    assert steps.index(gate) < steps.index(rebuild)
    report_dir = tmp_path / "validation-report"
    report_dir.mkdir()
    (report_dir / "report.json").write_text(
        json.dumps(
            {
                "stages": [{"stage_name": "bench", "status": "passed"}]
                + [{"stage_name": "e2e", "status": status} for status in statuses],
            }
        )
    )
    # Run only the gate; a sentinel stands in for rebuilding/pushing the bot branch.
    done = subprocess.run(
        ["bash", "-eu", "-c", gate["run"] + '\ntouch "$RUNNER_TEMP/publish-allowed"\n'],
        env=os.environ | {"RUNNER_TEMP": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert (done.returncode == 0) == (statuses == ["passed"])
    assert (tmp_path / "publish-allowed").exists() == (statuses == ["passed"])
