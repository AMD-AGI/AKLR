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

MODEL_CATALOG = "https://inferencex.semianalysis.com/api/v1/views/options"


def catalog_model(catalog: dict, prefix: str) -> str:
    models = catalog.get("models") if isinstance(catalog, dict) else None
    if not isinstance(models, list):
        raise ConfigError("InferenceX model catalog must contain a models array")
    matches = []
    for model in models:
        if (
            not isinstance(model, dict)
            or not isinstance(model.get("name"), str)
            or not model["name"].strip()
            or not isinstance(model.get("dbKeys"), list)
            or not model["dbKeys"]
            or any(not isinstance(key, str) or not key for key in model["dbKeys"])
        ):
            raise ConfigError("Invalid InferenceX model catalog entry")
        if prefix in model["dbKeys"]:
            matches.append(model["name"])
    if len(matches) != 1:
        raise ConfigError(
            f"InferenceX model prefix {prefix!r} needs exactly one catalog match; got {matches}"
        )
    return matches[0]


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
    policy = getattr(args, "point_policy", "all")
    auto_model = policy == POLICY and getattr(args, "inferencex_model", None) == "auto"
    if auto_model and args.inferencex_run_id:
        raise ConfigError("--inferencex-model auto cannot be combined with --inferencex-run-id")
    required_duration = getattr(args, "require_recipe_duration", None)
    if required_duration is not None:
        if required_duration <= 0:
            raise ConfigError("--require-recipe-duration must be positive")
        if args.duration is not None:
            raise ConfigError("--require-recipe-duration cannot be combined with --duration")
    config = load_config(args.config)
    selected = select_runs(config, args.models, args.targets)
    definitions = [model_config(config, model_id, target_id) for model_id, target_id in selected]
    runner_label = getattr(args, "runner_label", None)
    if runner_label is not None:
        definitions = [item for item in definitions if item["target"]["runner_label"] == runner_label]
        if not definitions:
            raise ConfigError(f"no selected targets use runner label {runner_label!r}")
    if policy == POLICY and not auto_model and len(definitions) != 1:
        raise ConfigError("Pareto selection requires exactly one model target unless --inferencex-model auto")
    if (
        policy == "all"
        and len(definitions) > 1
        and (args.arms != "all" or args.points or args.duration is not None)
    ):
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
    catalogs = {}
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
            target_args = args
            if auto_model:
                scope = rr.scope_of(spec, args.inferencex_hardware)
                catalog, _ = cached_resolution(catalogs, MODEL_CATALOG, rr.fetch_json, MODEL_CATALOG)
                target_args = argparse.Namespace(**vars(args))
                target_args.inferencex_model = catalog_model(catalog, scope["model"])
            spec["selection"] = prepare(spec, target_args, dst.parent)
            if spec["selection"]["policy"] == "all":
                spec["selection"].update(arms=args.arms, points=args.points, duration=args.duration)
            write_json(dst, spec)
            points = selected_points(spec)
            if required_duration is not None:
                mismatches = [
                    f"{rr.point_id(point)}={point.get('client_env', {}).get('DURATION')!r}"
                    for point in points
                    if str(point.get("client_env", {}).get("DURATION")) != str(required_duration)
                ]
                if mismatches:
                    raise ConfigError(
                        f"upstream DURATION must be {required_duration} seconds for every selected point: "
                        + ", ".join(mismatches)
                    )
            item.update(
                status="resolved",
                points=len(points),
                point_policy=spec["selection"]["policy"],
                serving_sha256=sha256_json(spec),
                point_ids=[rr.point_id(p) for p in points],
            )
            if spec["selection"]["policy"] == POLICY:
                item["selection_manifest_sha256"] = spec["selection"]["manifest_sha256"]
        except Exception as error:
            item.update(status="resolve-failed", error=f"{type(error).__name__}: {error}"[:1000])
            print(f"ERROR [{model_id}/{target_id}]: {item['error']}", file=sys.stderr)
        write_json(args.out / "manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m agentx_bench plan",
        description=__doc__,
        epilog="For per-target Pareto sources, plan supports --inferencex-model auto with --inferencex-hardware.",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--models", default="all", help="enabled model ids, comma-separated, or all")
    parser.add_argument("--targets", default="all", help="enabled target ids in selected models, or all")
    parser.add_argument("--runner-label", help="Only resolve selected targets assigned to this runner label")
    parser.add_argument("--inferencex-ref")
    parser.add_argument("--run-image")
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID", "local"))
    parser.add_argument("--arms", default="all")
    parser.add_argument("--points")
    parser.add_argument("--duration", type=int)
    parser.add_argument(
        "--require-recipe-duration",
        type=int,
        help="Require selected points' upstream DURATION without overriding it",
    )
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
