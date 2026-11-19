# SPDX-License-Identifier: MIT
"""Each framework's execution image: AKLR's CI image for it in stack.json, pinned there by digest."""

from __future__ import annotations

import json
from pathlib import Path

from agentx_ci.preflight import pinned_image

ROOT = Path(__file__).resolve().parents[3]
STACK = ROOT / "stack.json"
KEYS = {"sglang": "sglang_image", "vllm": "vllm_image"}


def ci_images(stack: Path | None = None) -> dict[str, str]:
    stack = Path(stack or STACK)
    pins = json.loads(stack.read_text(encoding="utf-8"))
    images = {}
    for framework, key in KEYS.items():
        image = pins.get(key)
        if not pinned_image(image) or "@" not in image:
            raise ValueError(f"{stack}: {key} must be an image tag pinned by digest, tag@sha256:...")
        images[framework] = image
    return images


def run_image(framework: str) -> str:
    """The framework's CI image, as stack.json pins it."""
    return ci_images()[framework]
