# SPDX-License-Identifier: MIT
"""Verify the execution image and the GPUs it runs on; each framework's CI image is pinned in stack.json."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STACK = ROOT / "stack.json"
KEYS = {"sglang": "sglang_image", "vllm": "vllm_image"}


def read_json(path: str | Path):
    return json.loads(Path(path).read_text())


def sha256_json(value) -> str:
    """Hash canonical JSON."""
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode()).hexdigest()


def pinned_image(image: str) -> bool:
    return isinstance(image, str) and bool(re.fullmatch(r"(?:[^\s]+@)?sha256:[0-9a-f]{64}", image))


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


def verify_image(image: str, inspection: dict) -> dict:
    image_id = inspection.get("Id", "")
    if not pinned_image(image) or not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        raise ValueError("execution requires an immutable image and a Docker image ID")
    if image != image_id and not any(
        ref.rsplit("@", 1)[-1] == image.rsplit("@", 1)[-1] for ref in (inspection.get("RepoDigests") or [])
    ):
        raise ValueError("Docker image does not match the planned digest")
    return {"image_reference": image, "image_id": image_id}


def validate_hardware(hardware: dict, expected: dict) -> None:
    devices = hardware["devices"]
    if not devices or len(devices) != expected["gpu_count"]:
        raise ValueError("visible GPU count differs from the target's hardware requirement")
    for device in devices:
        if (
            not re.search(expected["gpu_name_pattern"], device["name"])
            or device["arch"] != expected["gpu_arch"]
        ):
            raise ValueError(f"GPU {device['name']} / {device['arch']} does not match the target's hardware")
        if type(device["total_memory"]) is not int or device["total_memory"] <= 0:
            raise ValueError("invalid GPU memory observation")
    peers = hardware["peer_access"]
    if len(peers) != len(devices) or any(
        len(row) != len(devices) or any(type(value) is not bool for value in row) for row in peers
    ):
        raise ValueError("invalid GPU peer-access topology")


# Tuning variables come from the hashed plan or the verified image defaults.
ENVIRONMENT_POLICY = "paired-aklr-image-env-v2"


def image_environment(inspection: dict) -> dict[str, str]:
    """Parse Config.Env after the inspection has passed image identity verification."""
    config = inspection.get("Config")
    if not isinstance(config, dict) or "Env" not in config:
        raise ValueError("Docker image inspection has no Config.Env")
    entries = config["Env"]
    if entries is None:
        return {}
    if not isinstance(entries, list):
        raise ValueError("Docker image Config.Env must be a list or null")
    defaults = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, str) or "=" not in entry:
            raise ValueError(f"Docker image Config.Env entry {index} must be NAME=VALUE")
        name, value = entry.split("=", 1)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) or "\0" in value:
            raise ValueError(f"Docker image Config.Env entry {index} has an invalid name or value")
        if name in defaults:
            raise ValueError(f"Docker image Config.Env entry {index} repeats a variable name")
        defaults[name] = value
    return defaults


def image_environment_evidence(defaults: dict[str, str]) -> dict:
    """Publish image-default names and their complete digest without exposing values."""
    return {
        "policy": ENVIRONMENT_POLICY,
        "scope": "server",
        "source": "docker-image-inspect.Config.Env",
        "defaults_sha256": sha256_json(defaults),
        "defaults": {name: "<redacted>" for name in sorted(defaults)},
    }


def execution_identity(serving: dict) -> tuple[dict, dict[str, str]]:
    """Return public identity and server defaults from one verified inspection."""
    path = os.environ.get("AKLR_EXECUTION_FILE")
    if not path:
        raise RuntimeError("execution image is unverified; start the run with the Docker sweep launcher")
    inspection = read_json(path)
    if not isinstance(inspection, dict):
        raise ValueError("Docker image inspection must be an object")
    identity = verify_image(serving["image"]["run"], inspection)
    defaults = image_environment(inspection)
    identity["server_environment"] = image_environment_evidence(defaults)
    return identity, defaults
