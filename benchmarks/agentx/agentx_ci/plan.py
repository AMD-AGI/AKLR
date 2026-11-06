# SPDX-License-Identifier: MIT
"""Resolve model targets for serial sweep jobs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

from agentx_ci import recipe_resolve as rr
from agentx_ci.model_config import ConfigError, model_config, select_runs
from agentx_ci.recipe_resolve import POLICY, add_arguments, prepare, selected_points, validate_options
from agentx_ci.utils import DEFAULT_CONFIG, load_config, sha256_json, target_path, utc_now, write_json


def cached_resolution(cache: dict, key, resolve, *args):
    if key not in cache:
        try:
            cache[key] = resolve(*args)
        except Exception as error:
            cache[key] = error
    value = cache[key]
    if isinstance(value, Exception):
        raise value
    return value


def build(args: argparse.Namespace) -> dict:
    validate_options(args)
    config = load_config(args.config)
    selected = select_runs(config, args.models, args.targets)
    if getattr(args, "point_policy", "all") == POLICY and len(selected) != 1:
        raise ConfigError("Pareto selection requires exactly one model target per plan/source snapshot")
    definitions = [model_config(config, model_id, target_id) for model_id, target_id in selected]
    if len(selected) > 1 and (args.arms != "all" or args.points or args.duration is not None):
        raise ConfigError("--arms, --points and --duration smoke filters require exactly one selected target")
    if args.duration is not None and args.duration <= 0:
        raise ConfigError("--duration must be positive")
    args.out = args.out.resolve()
    args.work = args.work.resolve()
    if (args.out / "manifest.json").exists():
        raise ConfigError(f"plan already exists in {args.out}; choose a fresh output directory")
    manifest = {
        "schema": 3,
        "run_id": str(args.run_id),
        "created_at": utc_now(),
        "targets": [
            {
                "model": definition["model"],
                "target": definition["target"],
                "image": None,
                "status": "pending",
                "error": None,
            }
            for definition in definitions
        ],
    }
    write_json(args.out / "manifest.json", manifest)
    refs = {}
    images = {}
    for item, definition in zip(manifest["targets"], definitions):
        model_id, target_id = item["model"]["id"], item["target"]["id"]
        relative = target_path(model_id, target_id)
        source = definition["inferencex"]
        ref = args.inferencex_ref or source["ref"]
        image_reference = args.run_image or definition["run_image"]
        print(f"[{utc_now()}] resolving {model_id}/{target_id}", flush=True)
        try:
            image, warnings = cached_resolution(images, image_reference, rr.resolve_image, image_reference)
            item["image"] = image
            sha = cached_resolution(refs, (source["repo"], ref), rr.resolve_ref, source["repo"], ref)
            dst = args.out / relative / "serving.json"
            spec = rr.resolve(
                argparse.Namespace(
                    config=args.config,
                    model=model_id,
                    target=target_id,
                    inferencex_ref=sha,
                    inferencex_src=None,
                    run_image=image,
                    work=args.work / relative,
                    out=args.work / relative / "resolved-serving.json",
                )
            )
            spec["provenance"]["inferencex"]["requested_ref"] = ref
            spec["warnings"] = warnings + spec["warnings"]
            spec["plan_id"] = uuid.uuid4().hex
            spec["selection"] = prepare(spec, args, dst.parent)
            if spec["selection"]["policy"] == "all":
                spec["selection"].update(arms=args.arms, points=args.points, duration=args.duration)
            write_json(dst, spec)
            item.update(
                status="resolved",
                points=len(selected_points(spec)),
                point_policy=spec["selection"]["policy"],
                serving_sha256=sha256_json(spec),
                point_ids=[rr.point_id(p) for p in selected_points(spec)],
            )
            if spec["selection"]["policy"] == POLICY:
                item["selection_manifest_sha256"] = spec["selection"]["manifest_sha256"]
        except Exception as error:
            item.update(status="resolve-failed", error=f"{type(error).__name__}: {error}"[:1000])
            print(f"ERROR [{model_id}/{target_id}]: {item['error']}", file=sys.stderr)
        write_json(args.out / "manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agentx_bench plan", description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--models", default="all", help="enabled model ids, comma-separated, or all")
    parser.add_argument("--targets", default="all", help="enabled target ids in selected models, or all")
    parser.add_argument("--inferencex-ref")
    parser.add_argument("--run-image")
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID", "local"))
    parser.add_argument("--arms", default="all")
    parser.add_argument("--points")
    parser.add_argument("--duration", type=int)
    add_arguments(parser)
    args = parser.parse_args(argv)
    try:
        manifest = build(args)
    except (ConfigError, rr.ResolveError, ValueError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    ready = [
        {
            "model": item["model"]["id"],
            "target": item["target"]["id"],
            "runner": item["target"]["runner_label"],
            "job_index": index,
        }
        for index, item in enumerate(manifest["targets"])
        if item["status"] == "resolved"
    ]
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as output:
            output.write(f"matrix={json.dumps({'include': ready})}\ncount={len(ready)}\n")
    print(json.dumps(manifest, indent=1))
    # Defer target failures until CI saves the report.
    return 0
