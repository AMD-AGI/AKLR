# SPDX-License-Identifier: MIT
"""Resolve selected models in order and publish a manifest for serial sweep jobs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

from agentx_ci import recipe_resolve as rr
from agentx_ci.model_config import ConfigError, model_config, select_models
from agentx_ci.recipe_resolve import POLICY, add_arguments, prepare, selected_points, validate_options
from agentx_ci.utils import DEFAULT_CONFIG, REPO_ROOT, load_config, sha256_json, utc_now, write_json


def build(args: argparse.Namespace) -> dict:
    validate_options(args)
    config = load_config(args.config)
    selected = select_models(config, args.models)
    if getattr(args, "point_policy", "all") == POLICY and len(selected) != 1:
        raise ConfigError("Pareto selection requires exactly one model per plan/source snapshot")
    definitions = {key: model_config(config, key) for key in selected}
    if len(selected) > 1 and (args.arms != "all" or args.points or args.duration is not None):
        raise ConfigError("--arms, --points and --duration smoke filters require exactly one selected model")
    if args.duration is not None and args.duration <= 0:
        raise ConfigError("--duration must be positive")
    args.out = args.out.resolve()
    args.work = args.work.resolve()
    if (args.out / "manifest.json").exists():
        raise ConfigError(f"plan already exists in {args.out}; choose a fresh output directory")
    image, warnings = (
        (args.run_image, []) if args.run_image else rr.ci_image(REPO_ROOT / ".github/workflows/ci.yaml")
    )
    manifest = {
        "schema": 2,
        "run_id": str(args.run_id),
        "created_at": utc_now(),
        "image": image,
        "warnings": warnings,
        "models": [{**definitions[key]["model"], "status": "pending", "error": None} for key in selected],
    }
    write_json(args.out / "manifest.json", manifest)
    refs = {}
    for item in manifest["models"]:
        key = item["id"]
        source = definitions[key]["inferencex"]
        ref = args.inferencex_ref or source["ref"]
        identity = (source["repo"], ref)
        print(f"[{utc_now()}] resolving {key}", flush=True)
        try:
            if identity not in refs:
                # Resolve shared refs once, including failures.
                try:
                    refs[identity] = rr.resolve_ref(*identity)
                except Exception as error:
                    refs[identity] = error
            if isinstance(refs[identity], Exception):
                raise refs[identity]
            dst = args.out / "models" / key / "serving.json"
            spec = rr.resolve(
                argparse.Namespace(
                    config=args.config,
                    model=key,
                    inferencex_ref=refs[identity],
                    inferencex_src=None,
                    run_image=image,
                    work=args.work / key,
                    out=args.work / key / "resolved-serving.json",
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
            print(f"ERROR [{key}]: {item['error']}", file=sys.stderr)
        write_json(args.out / "manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agentx_bench plan", description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--models", default="all", help="enabled model ids, comma-separated, or all")
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
    ready = [item["id"] for item in manifest["models"] if item["status"] == "resolved"]
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as output:
            output.write(
                f"matrix={json.dumps({'model': ready})}\ncount={len(ready)}\nimage={manifest['image']}\n"
            )
    print(json.dumps(manifest, indent=1))
    # CI checks model failures after saving the report.
    return 0
