# SPDX-License-Identifier: MIT
"""Build per-run serving plans from reviewed recipes/<family>/<target>/<framework>/serving.<model>.json files."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

from agentx_ci import serving_config as sc
from agentx_ci.model_config import ConfigError, model_config, select_models, select_runs
from agentx_ci.pr_scope import declared_models
from agentx_ci.utils import (
    DEFAULT_CONFIG,
    SCHEMA,
    load_config,
    read_json,
    sha256_json,
    target_path,
    utc_now,
    write_json,
)


def load_recipe(config: str | Path, definition: dict) -> dict:
    path = sc.recipe_path(config, definition)
    if not path.is_file():
        raise ConfigError(f"missing {path}; the recipe must be generated and reviewed first")
    return sc.build_run(read_json(path), definition)


def warn(message: str) -> None:
    """Annotate the GitHub run so a skipped or narrowed AgentX scope is visible on a green check."""
    if os.environ.get("GITHUB_ACTIONS") == "true":
        escaped = message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
        print(f"::warning title=AgentX scope::{escaped}", flush=True)


def apply_pr_scope(args: argparse.Namespace) -> tuple[str | None, bool]:
    """Narrow --models to the PR description's `AgentX:` line.

    Return a note for the job summary and whether to skip AgentX, as when the line selects no model.
    """
    if args.pr_body is None:
        return None, False
    if args.models != "all":
        raise ConfigError("--pr-body cannot be combined with --models")
    declared, unmatched = declared_models(args.pr_body.read_text(), select_models(load_config(args.config)))
    missed = (
        f" Not an enabled model id, skipped: {', '.join(f'`{name}`' for name in unmatched)}."
        if unmatched
        else ""
    )
    if not declared:
        note = f"The PR description selects no model on an `AgentX:` line; skipping AgentX.{missed}"
        warn(note)
        return note, True
    if unmatched:
        warn(missed.strip())
    args.models = declared
    if declared == "all":
        return "The PR description declares `AgentX: all`; running every enabled model.", False
    return (
        f"The PR description's `AgentX:` line selects {declared}; skipping the other models.{missed}",
        False,
    )


def write_outputs(ready: list[dict], skipped: bool = False) -> None:
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as output:
            output.write(f"matrix={json.dumps({'include': ready})}\ncount={len(ready)}\n")
            output.write(f"skipped={str(skipped).lower()}\n")


def build(args: argparse.Namespace) -> dict:
    required_duration = getattr(args, "require_recipe_duration", None)
    duration = getattr(args, "duration", None)
    points = getattr(args, "points", None)
    if required_duration is not None:
        if required_duration <= 0:
            raise ConfigError("--require-recipe-duration must be positive")
        if duration is not None:
            raise ConfigError("--require-recipe-duration cannot be combined with --duration")
    if duration is not None and duration <= 0:
        raise ConfigError("--duration must be positive")
    config = load_config(args.config)
    selected = select_runs(config, args.models, args.targets, args.frameworks)
    definitions = [model_config(config, *run) for run in selected]
    runner_label = getattr(args, "runner_label", None)
    if runner_label is not None:
        definitions = [item for item in definitions if item["target"]["runner_label"] == runner_label]
        if not definitions:
            raise ConfigError(f"no selected runs use runner label {runner_label!r}")
    if points and len(definitions) != 1:
        raise ConfigError("--points smoke filters require exactly one selected model/target/framework run")
    args.out = args.out.resolve()
    if (args.out / "manifest.json").exists():
        raise ConfigError(f"plan already exists in {args.out}; choose a fresh output directory")
    manifest = {
        "schema": SCHEMA,
        "run_id": str(args.run_id),
        "created_at": utc_now(),
        "targets": [
            {
                "model": definition["model"],
                "target": definition["target"],
                "framework": definition["framework"],
                "image": None,
                "status": "pending",
                "error": None,
            }
            for definition in definitions
        ],
    }
    write_json(args.out / "manifest.json", manifest)
    for item, definition in zip(manifest["targets"], definitions):
        run = (item["model"]["id"], item["target"]["id"], item["framework"]["id"])
        label = "/".join(run)
        print(f"[{utc_now()}] loading recipe for {label}", flush=True)
        try:
            spec = load_recipe(args.config, definition)
            spec["plan_id"] = uuid.uuid4().hex
            spec["selection"] = sc.narrow_selection(spec["selection"], points, duration)
            item["image"] = spec["image"]["run"]
            write_json(args.out / target_path(*run) / "serving.json", spec)
            selected_points = sc.selected_points(spec)
            if required_duration is not None:
                mismatches = [
                    f"{sc.point_id(point)}={point.get('client_env', {}).get('DURATION')!r}"
                    for point in selected_points
                    if str(point.get("client_env", {}).get("DURATION")) != str(required_duration)
                ]
                if mismatches:
                    raise ConfigError(
                        f"recipe DURATION must be {required_duration} seconds for every selected point: "
                        + ", ".join(mismatches)
                    )
            item.update(
                status="resolved",
                points=len(selected_points),
                point_policy=spec["selection"]["policy"],
                serving_sha256=sha256_json(spec),
                point_ids=[sc.point_id(p) for p in selected_points],
                selection_manifest_sha256=spec["selection"]["manifest_sha256"],
            )
        except Exception as error:
            item.update(status="resolve-failed", error=f"{type(error).__name__}: {error}"[:1000])
            print(f"ERROR [{label}]: {item['error']}", file=sys.stderr)
        write_json(args.out / "manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agentx_bench plan", description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--models", default="all", help="enabled model ids, comma-separated, or all")
    parser.add_argument("--targets", default="all", help="enabled target ids in selected models, or all")
    parser.add_argument(
        "--frameworks", default="all", help="enabled frameworks (sglang, vllm) in selected targets, or all"
    )
    parser.add_argument("--runner-label", help="Only plan selected runs whose target uses this runner label")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID", "local"))
    parser.add_argument("--points", help="smoke run: comma-separated recipe ARM:CONC ids (one run only)")
    parser.add_argument("--duration", type=int, help="smoke run: override DURATION in seconds")
    parser.add_argument(
        "--require-recipe-duration",
        type=int,
        help="Require selected points' recipe DURATION without overriding it",
    )
    parser.add_argument(
        "--pr-body",
        type=Path,
        help="PR description file; its `AgentX:` line (model ids or all) selects the models, or skips AgentX",
    )
    args = parser.parse_args(argv)
    try:
        note, skip = apply_pr_scope(args)
        if note:
            print(note)
            if os.environ.get("GITHUB_STEP_SUMMARY"):
                with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as summary:
                    summary.write(note + "\n")
        if skip:
            write_outputs([], skipped=True)
            return 0
        manifest = build(args)
    except (ConfigError, ValueError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    ready = [
        {
            "model": item["model"]["id"],
            "target": item["target"]["id"],
            "framework": item["framework"]["id"],
            "runner": item["target"]["runner_label"],
            "job_index": index,
        }
        for index, item in enumerate(manifest["targets"])
        if item["status"] == "resolved"
    ]
    write_outputs(ready)
    print(json.dumps(manifest, indent=1))
    # Defer run failures until CI saves the report.
    return 0
