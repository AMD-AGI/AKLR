# SPDX-License-Identifier: MIT
"""Execution images come from AKLR's CI images in ci.yaml; runs without network or GPU access."""

from __future__ import annotations

import importlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

AGENTX = Path(__file__).resolve().parents[2] / "benchmarks/agentx"
with mock.patch.object(sys, "path", [str(AGENTX), *sys.path]):
    ci = importlib.import_module("agentx_ci.ci_image")

IMAGES = {"sglang": "example.invalid/sglang:v1", "vllm": "example.invalid/vllm:v1"}
OLD = "sha256:" + "a" * 64
NEW = "sha256:" + "b" * 64


def inspected(stdout: str = "", returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr=stderr)


class CiImageTest(unittest.TestCase):
    def test_the_repository_names_a_ci_image_per_framework(self):
        images = ci.ci_images()
        self.assertEqual(set(images), {"sglang", "vllm"})

    def test_ci_images_must_be_tags(self):
        with tempfile.TemporaryDirectory() as tmp:
            workflow = Path(tmp) / "ci.yaml"
            sglang, vllm = IMAGES["sglang"], IMAGES["vllm"]
            workflow.write_text(f"env:\n  AKLR_CI_IMAGE: &ci_image {sglang}\n  AKLR_VLLM_CI_IMAGE: {vllm}\n")
            self.assertEqual(ci.ci_images(workflow), IMAGES)
            workflow.write_text(f"env:\n  AKLR_CI_IMAGE: {sglang}\n  AKLR_VLLM_CI_IMAGE: {vllm}@{OLD}\n")
            with self.assertRaisesRegex(ValueError, "AKLR_VLLM_CI_IMAGE must be an image tag"):
                ci.ci_images(workflow)
            workflow.write_text(f"env:\n  AKLR_CI_IMAGE: {sglang}\n")
            with self.assertRaisesRegex(ValueError, "AKLR_VLLM_CI_IMAGE must be an image tag"):
                ci.ci_images(workflow)

    def test_digest_comes_from_the_top_level_manifest(self):
        tag = IMAGES["sglang"]
        output = f"Name: {tag}\nMediaType: index\nDigest:    {NEW}\n\nManifests:\n  Name: {tag}@{OLD}\n"
        with mock.patch.object(ci.subprocess, "run", return_value=inspected(output)) as run:
            self.assertEqual(ci.resolve_digest(tag), NEW)
        self.assertEqual(run.call_args.args[0], ["docker", "buildx", "imagetools", "inspect", tag])
        with mock.patch.object(ci.subprocess, "run", return_value=inspected()):
            with self.assertRaisesRegex(ValueError, "cannot read the digest"):
                ci.resolve_digest(tag)
        with mock.patch.object(ci.subprocess, "run", return_value=inspected(returncode=1, stderr="denied\n")):
            with self.assertRaisesRegex(ValueError, "failed: denied"):
                ci.resolve_digest(tag)

    def test_run_image_pins_the_framework_ci_image(self):
        with (
            mock.patch.object(ci, "ci_images", return_value=IMAGES),
            mock.patch.object(ci, "resolve_digest", return_value=NEW) as resolve,
        ):
            self.assertEqual(ci.run_image("vllm"), f"{IMAGES['vllm']}@{NEW}")
        resolve.assert_called_once_with(IMAGES["vllm"])


if __name__ == "__main__":
    unittest.main()
