# SPDX-License-Identifier: MIT
"""Write each planned group's execution.json, its image as the registry reports it, for a sweep on SaFE.

A SaFE pod runs the pinned image itself and has no Docker to inspect it with, as the Docker launcher does.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from perf_utils.image import registry_inspection

from e2e_perf.model_config import group_path
from e2e_perf.utils import read_json, write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m e2e_perf inspect-images", description=__doc__)
    parser.add_argument("--plan", type=Path, required=True, help="the plan directory, with manifest.json")
    args = parser.parse_args(argv)
    inspections: dict[str, dict] = {}
    try:
        for item in read_json(args.plan / "manifest.json")["groups"]:
            if item["status"] != "planned":
                continue
            directory = args.plan / group_path(
                item["model"]["id"], item["target"]["id"], item["framework"]["id"]
            )
            image = read_json(directory / "plan.json")["image"]["run"]
            if image not in inspections:
                inspections[image] = registry_inspection(image)
                print(f"{image}: image ID {inspections[image]['Id']}")
            write_json(directory / "execution.json", inspections[image])
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0
