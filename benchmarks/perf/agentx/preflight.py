# SPDX-License-Identifier: MIT
"""Validate execution metadata for performance comparisons."""

from __future__ import annotations

import re

from perf_utils.image import pinned_image, validate_hardware

from agentx.model_config import validate_framework_id, validate_model_id, validate_target_id


def comparison_identity(results: dict, *, allow_empty_points: bool = False) -> dict | None:
    """Return an identity only for a verified execution."""
    try:
        model_id = validate_model_id(results["model"]["id"])
        target_id = validate_target_id(results["target"]["id"])
        framework_id = validate_framework_id(results["framework"]["id"])
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
        client = run["client"]
        if not re.fullmatch(r"[0-9a-f]{40}", client["sha"]):
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
        if not traces and not allow_empty_points:
            return None
        return {
            "model_id": model_id,
            "target_id": target_id,
            "framework_id": framework_id,
            "image_id": execution["image_id"],
            "client": {"repo": client["repo"], "sha": client["sha"]},
            "hardware": run["hardware"],
            "client_packages": packages,
            "server_packages": server,
            "traces": traces,
        }
    except (KeyError, TypeError, ValueError, AttributeError, re.error):
        return None
