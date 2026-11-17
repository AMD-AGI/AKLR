# SPDX-License-Identifier: MIT
"""Execution images come from AKLR's CI images in stack.json; runs without network or GPU access."""

from __future__ import annotations

import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

AGENTX = Path(__file__).resolve().parents[2] / "benchmarks/agentx"
with mock.patch.object(sys, "path", [str(AGENTX), *sys.path]):
    ci = importlib.import_module("agentx_ci.ci_image")

DIGEST = "sha256:" + "a" * 64
IMAGES = {"sglang": f"example.invalid/sglang:v1@{DIGEST}", "vllm": f"example.invalid/vllm:v1@{DIGEST}"}


class CiImageTest(unittest.TestCase):
    def test_the_repository_names_a_ci_image_per_framework(self):
        images = ci.ci_images()
        self.assertEqual(set(images), {"sglang", "vllm"})

    def test_ci_images_must_be_tags_pinned_by_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            stack = Path(tmp) / "stack.json"
            sglang, vllm = IMAGES["sglang"], IMAGES["vllm"]
            stack.write_text(json.dumps({"sglang_image": sglang, "vllm_image": vllm}))
            self.assertEqual(ci.ci_images(stack), IMAGES)
            for unpinned in ("example.invalid/vllm:v1", DIGEST, None):
                stack.write_text(json.dumps({"sglang_image": sglang, "vllm_image": unpinned}))
                with self.assertRaisesRegex(ValueError, "vllm_image must be an image tag pinned by digest"):
                    ci.ci_images(stack)

    def test_run_image_is_the_framework_ci_image(self):
        with mock.patch.object(ci, "ci_images", return_value=IMAGES):
            self.assertEqual(ci.run_image("vllm"), IMAGES["vllm"])


if __name__ == "__main__":
    unittest.main()
