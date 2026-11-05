# SPDX-License-Identifier: MIT
"""Validate execution metadata for performance comparisons."""

from __future__ import annotations

import re

from agentx_ci.model_config import validate_model_id, validate_target_id


def pinned_image(image: str) -> bool:
    return isinstance(image, str) and bool(re.fullmatch(r"(?:[^\s]+@)?sha256:[0-9a-f]{64}", image))


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


def comparison_identity(results: dict) -> dict | None:
    """Return an identity only for a verified execution."""
    try:
        model_id = validate_model_id(results["model"]["id"])
        target_id = validate_target_id(results["target"]["id"])
        run = results["run"]
        execution = run["execution"]
        image = run["image"]["run"]
        if not pinned_image(image) or execution["image_reference"] != image:
            return None
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", execution["image_id"]):
            return None
        validate_hardware(run["hardware"], run["expected_hardware"])
        packages = run["client_packages"]
        server = run["server_packages"]
        for versions in (packages, server):
            if (
                not isinstance(versions, dict)
                or not versions
                or any(
                    not isinstance(name, str) or not isinstance(version, str)
                    for name, version in versions.items()
                )
            ):
                return None
        source = run["inferencex"]
        if not re.fullmatch(r"[0-9a-f]{40}", source["sha"]):
            return None
        traces = {}
        for point in results["points"]:
            trace = point["measurement"]["trace"]
            if (
                not trace["repo"]
                or not trace["loader"]
                or not re.fullmatch(r"[0-9a-f]{40}", trace["revision"])
            ):
                return None
            key = trace["loader"]
            if key in traces and traces[key] != trace:
                return None
            traces[key] = trace
        if not traces:
            return None
        return {
            "model_id": model_id,
            "target_id": target_id,
            "image_id": execution["image_id"],
            "inferencex": {"repo": source["repo"], "sha": source["sha"]},
            "hardware": run["hardware"],
            "client_packages": packages,
            "server_packages": server,
            "traces": traces,
        }
    except (KeyError, TypeError, ValueError, AttributeError, re.error):
        return None
