# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest

from aklr.core.validation.stages._e2e._cookbook import Cookbook

# cut from sgl-project/sglang docs/src/snippets/configs/moonshotai/kimi-k3.jsx (95560684),
# with a `//` and a backtick inside strings, a single-quoted string and a trailing comma added
_KIMI_K3 = """\
// Single `export const config` literal — no spreads/calls/IIFE.
export const config = {
  modelName: "Kimi-K3",
  dockerImages: {
    mi355x: "lmsysorg/sglang-rocm:v0.5.19-rocm720-mi35x-20260916",
    "b300|nvfp4": "lmsysorg/sglang:dev-dev-kimi-k3-nvfp4",
  },
  cellFor(s) {
    return (config.cells || []).find((c) => c.match.hw === s.hw);
  },
  flagOf(cell, name) {
    return ((cell || {}).flags || []).find((f) => f.split(/[\\s=]/)[0] === name);
  },
  // Verification marks: every cell carries "Final Verification In Progress" —
  // Flip a cell to `verified: true` once that round lands.
  cells: [
    {
      // Same ROCm/AITER profile as MI350X.
      match: { hw: "mi355x", pdMode: "unified", strategy: "balanced" },
      nnodes: 1,
      verified: false,
      verificationStatus: "in-progress",
      warn: "see http://example.com/`notes`",
      env: [
        "SGLANG_USE_AITER=1",
        'SGLANG_AITER_K3_OPT=1',
      ],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--reasoning-parser kimi_k3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "a3", pdMode: "prefill", strategy: "default" },
      nnodes: 2,
      env: [],
      flags: ["--model-path {{MODEL_NAME}}"],
    },
  ],
};
"""
_MI355X = {"hw": "mi355x", "pdMode": "unified", "strategy": "balanced"}


def _parse(text: str = _KIMI_K3) -> Cookbook:
    return Cookbook.parse(text, source="main", sha="95560684", config="moonshotai/kimi-k3")


def test_parses_the_cells() -> None:
    cookbook = _parse()
    cell = cookbook.cell(_MI355X)
    assert cell.env == ("SGLANG_USE_AITER=1", "SGLANG_AITER_K3_OPT=1")
    assert cell.flags[:3] == ("--model-path {{MODEL_NAME}}", "--tp-size 8", "--reasoning-parser kimi_k3")
    assert cell.nnodes == 1
    assert cell.notes == "verified: false; warn: see http://example.com/`notes`"
    assert cookbook.cell({"hw": "a3", "pdMode": "prefill", "strategy": "default"}).nnodes == 2
    assert cookbook.docker_image(_MI355X) == "lmsysorg/sglang-rocm:v0.5.19-rocm720-mi35x-20260916"
    assert cookbook.docker_image({"hw": "b300", "quant": "nvfp4"}) == "lmsysorg/sglang:dev-dev-kimi-k3-nvfp4"


@pytest.mark.parametrize(
    "match",
    [
        pytest.param({"hw": "h100", "pdMode": "unified", "strategy": "balanced"}, id="none"),
        pytest.param({"hw": "mi355x"}, id="partial"),
    ],
)
def test_a_match_selects_one_cell(match: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="0 cells match"):
        _parse().cell(match)


def test_several_matching_cells_fail() -> None:
    text = _KIMI_K3.replace(
        'hw: "a3", pdMode: "prefill", strategy: "default"',
        'hw: "mi355x", pdMode: "unified", strategy: "balanced"',
    )
    with pytest.raises(ValueError, match="2 cells match"):
        _parse(text).cell(_MI355X)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        pytest.param("env: [],", "env: [...base],", id="spread"),
        pytest.param("[\n    {\n      // Same", "[\n    baseCell,\n    {\n      // Same", id="variable"),
        pytest.param("env: [],", "env: (s) => [],", id="function-where-a-literal-is-needed"),
        pytest.param("env: [],", 'env: ["A=\\u{1F600}"],', id="escape-python-has-not"),
    ],
)
def test_a_format_change_fails_the_parse(old: str, new: str) -> None:
    assert old in _KIMI_K3
    with pytest.raises(ValueError, match="cookbook moonshotai/kimi-k3"):
        _parse(_KIMI_K3.replace(old, new))


def test_a_function_value_reads_as_null() -> None:
    """A key the launch does not read (glm-5.3-flash's verificationStatus) may hold a function."""
    text = _KIMI_K3.replace(
        'verificationStatus: "in-progress",',
        'verificationStatus: (s) => s.bcg !== "off" ? `a, ${s.x}` : { label: "b", hint: [1, 2] },',
    )
    assert _parse(text).cell(_MI355X).flags == _parse().cell(_MI355X).flags


def test_a_ref_named_like_a_directory_stays_a_ref(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "docs").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("subprocess.run", lambda *a, **k: (_ for _ in ()).throw(OSError("network")))
    with pytest.raises(ValueError, match="network"):
        Cookbook.fetch("docs", "moonshotai/kimi-k3")


def test_local_checkout_source(tmp_path: Path) -> None:
    path = tmp_path / "docs/src/snippets/configs/moonshotai/kimi-k3.jsx"
    path.parent.mkdir(parents=True)
    path.write_text(_KIMI_K3, encoding="utf-8")
    cookbook = Cookbook.fetch(str(tmp_path), "moonshotai/kimi-k3")
    assert cookbook.sha == "local"
    assert cookbook.cell(_MI355X).nnodes == 1
    with pytest.raises(ValueError, match="cookbook zai-org/glm-5.2"):
        Cookbook.fetch(str(tmp_path), "zai-org/glm-5.2")
