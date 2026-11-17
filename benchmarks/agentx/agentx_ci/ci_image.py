# SPDX-License-Identifier: MIT
"""Each framework's execution image: AKLR's CI image for it in stack.json, pinned at plan time."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STACK = ROOT / "stack.json"
KEYS = {"sglang": "sglang_image", "vllm": "vllm_image"}
DIGEST = re.compile(r"^Digest:\s+(sha256:[0-9a-f]{64})\s*$", re.M)


def ci_images(stack: Path | None = None) -> dict[str, str]:
    stack = Path(stack or STACK)
    pins = json.loads(stack.read_text(encoding="utf-8"))
    images = {}
    for framework, key in KEYS.items():
        image = pins.get(key)
        if not isinstance(image, str) or not image or "@" in image:
            raise ValueError(f"{stack}: {key} must be an image tag without a digest")
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
