# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from pathlib import Path
from unittest.mock import Mock

import pytest

from aklr.core.validation import command
from aklr.core.validation.stages._e2e._config import E2eConfig


@pytest.mark.parametrize(
    ("flags", "expected"),
    [([], ["audit", "bench", "e2e"]), (["--stages", "audit,bench"], ["audit", "bench"])],
)
def test_stage_selection(monkeypatch: pytest.MonkeyPatch, flags: list[str], expected: list[str]) -> None:
    """Validation defaults to audit, bench and e2e; explicit stages remain supported."""
    config = E2eConfig(cookbook_repo="r", cookbook_source="main", scenarios=())
    read = Mock(return_value=config)
    monkeypatch.setattr(E2eConfig, "read", read)
    scope = Mock(root=Path("."))
    monkeypatch.setattr(command.ValidationScope, "of_working_tree", Mock(return_value=scope))
    report = Mock(exit_code=0)
    run = Mock(return_value=report)
    monkeypatch.setattr(command.Pipeline, "run", run)
    pipeline = command.Pipeline
    built = Mock(side_effect=pipeline)
    monkeypatch.setattr(command, "Pipeline", built)

    assert command.main([*flags, "--devices", "0,2-3"]) == 0
    assert [stage.name for stage in built.call_args.args[0]] == expected
    assert read.call_count == int("e2e" in expected)
    run.assert_called_once_with(scope)


def test_audit_needs_neither_devices_nor_e2e_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(command.PlatformRecord, "current", Mock(side_effect=AssertionError("GPU probe")))
    monkeypatch.setattr(E2eConfig, "read", Mock(side_effect=AssertionError("e2e config read")))
    monkeypatch.setattr(command.ValidationScope, "of_working_tree", Mock(return_value=Mock(root=Path("."))))
    monkeypatch.setattr(command.Pipeline, "run", Mock(return_value=Mock(exit_code=0)))
    assert command.main(["--stages", "audit"]) == 0
