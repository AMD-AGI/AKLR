# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The tool under test, importable as forge_task, and an AKLR stand-in to point it at."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# harness/tests/forge_task/conftest.py -> harness/scripts
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from fake_aklr import install_aklr, schema_payload, write_schema


@pytest.fixture(autouse=True)
def _aklr_stand_in_imports_fresh():
    """Driver code run in this process imports aklr from the stand-in; a real aklr another suite
    imported first, or another test's stand-in, would be found in sys.modules instead.
    """

    def imported() -> dict:
        return {
            name: module for name, module in sys.modules.items() if name == "aklr" or name.startswith("aklr.")
        }

    saved = imported()
    for name in saved:
        del sys.modules[name]
    yield
    for name in imported():
        del sys.modules[name]
    sys.modules.update(saved)


@pytest.fixture
def aklr_root(tmp_path: Path) -> Path:
    root = install_aklr(tmp_path / "aklr_checkout")
    write_schema(root, schema_payload())
    return root
