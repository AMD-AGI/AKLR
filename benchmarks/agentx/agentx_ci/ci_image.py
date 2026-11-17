# SPDX-License-Identifier: MIT
"""Each framework's execution image: AKLR's CI image for it in .github/workflows/ci.yaml, pinned at plan time."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
CI_WORKFLOW = ROOT / ".github/workflows/ci.yaml"
VARIABLES = {"sglang": "AKLR_CI_IMAGE", "vllm": "AKLR_VLLM_CI_IMAGE"}
DIGEST = re.compile(r"^Digest:\s+(sha256:[0-9a-f]{64})\s*$", re.M)


def ci_images(workflow: Path | None = None) -> dict[str, str]:
    workflow = Path(workflow or CI_WORKFLOW)
    env = yaml.safe_load(workflow.read_text(encoding="utf-8"))["env"]
    images = {}
    for framework, variable in VARIABLES.items():
        image = env.get(variable)
        if not isinstance(image, str) or not image or "@" in image:
            raise ValueError(f"{workflow}: env.{variable} must be an image tag without a digest")
        images[framework] = image
    return images


def resolve_digest(image: str) -> str:
    done = subprocess.run(
        ["docker", "buildx", "imagetools", "inspect", image],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if done.returncode:
        raise ValueError(f"docker buildx imagetools inspect {image} failed: {done.stderr.strip()[-500:]}")
    match = DIGEST.search(done.stdout)
    if not match:
        raise ValueError(f"cannot read the digest of {image} from docker buildx imagetools inspect")
    return match.group(1)


def run_image(framework: str) -> str:
    """The framework's CI image tag pinned to the digest it names now."""
    image = ci_images()[framework]
    return f"{image}@{resolve_digest(image)}"
