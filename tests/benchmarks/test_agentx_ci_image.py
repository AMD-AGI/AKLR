# SPDX-License-Identifier: MIT
"""Execution images come from AKLR's CI images in stack.json; runs without network or GPU access."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

AGENTX = Path(__file__).resolve().parents[2] / "benchmarks/agentx"
from agentx_ci import ci_image as ci

DIGEST = "sha256:" + "a" * 64
IMAGES = {"sglang": f"example.invalid/sglang:v1@{DIGEST}", "vllm": f"example.invalid/vllm:v1@{DIGEST}"}


def test_the_repository_names_a_ci_image_per_framework():
    images = ci.ci_images()
    assert set(images) == {"sglang", "vllm"}


@pytest.mark.parametrize(
    "unpinned", ["example.invalid/vllm:v1", DIGEST, None], ids=["tag-only", "digest-only", "missing"]
)
def test_ci_images_must_be_tags_pinned_by_digest(tmp_path, unpinned):
    stack = tmp_path / "stack.json"
    sglang, vllm = IMAGES["sglang"], IMAGES["vllm"]
    stack.write_text(json.dumps({"sglang_image": sglang, "vllm_image": vllm}))
    assert ci.ci_images(stack) == IMAGES
    stack.write_text(json.dumps({"sglang_image": sglang, "vllm_image": unpinned}))
    with pytest.raises(ValueError, match="vllm_image must be an image tag pinned by digest"):
        ci.ci_images(stack)


def test_run_image_is_the_framework_ci_image(monkeypatch):
    monkeypatch.setattr(ci, "ci_images", lambda: IMAGES)
    assert ci.run_image("vllm") == IMAGES["vllm"]
