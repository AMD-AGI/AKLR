# SPDX-License-Identifier: MIT
"""Plan the groups to measure: skip a group whose run key and complete result are already published."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from perf_utils import image as ci_image
from perf_utils.gh import ALLOWLIST_FILE, ConfigError, check_manual_trigger, summarize, write_outputs
from perf_utils.pr_scope import apply_pr_scope

from e2e_perf.model_config import (
    DEFAULT_CONFIG,
    group_key,
    group_path,
    groups,
    load_config,
    load_recipe,
)
from e2e_perf.utils import SCHEMA, read_json, sha256_json, utc_now, write_json

NAME = "Daily end-to-end performance"
COMMAND = "/e2e-perf-mi355x"


def run_key(aklr_sha: str, image: str, recipe: dict, workloads: list, client: dict, weights: dict) -> str:
    """Everything a group's measurement depends on; a change in any part measures the group again."""
    return sha256_json(
        {
            "aklr_sha": aklr_sha,
            "image": image,
            "recipe_sha256": sha256_json(recipe),
            "workloads": workloads,
            "client": client,
            "weights": weights,
        }
    )


def published(latest: Path | None) -> dict:
    """The groups site-data's e2e/latest.json holds, by model/target/framework key."""
    if latest is None or not latest.is_file() or not latest.read_text().strip():
        return {}
    data = read_json(latest)
    return data.get("groups", {}) if isinstance(data, dict) and data.get("schema") == SCHEMA else {}


def unchanged(entry: dict | None, key: str) -> bool:
    return isinstance(entry, dict) and entry.get("run_key") == key and entry.get("complete") is True


def build(args: argparse.Namespace) -> dict:
    config = load_config(args.config)
    selected = groups(config, args.models, args.frameworks)
    if not selected:
        raise ConfigError("no group selected")
    out = args.out.resolve()
    if (out / "manifest.json").exists():
        raise ConfigError(f"plan already exists in {out}; choose a fresh output directory")
    previous = published(args.latest)
    manifest = {"schema": SCHEMA, "run_id": str(args.run_id), "created_at": utc_now(), "groups": []}
    for group in selected:
        ids = (group["model"]["id"], group["target"]["id"], group["framework"]["id"])
        key = group_key(*ids)
        item = {
            "model": group["model"],
            "target": group["target"],
            "framework": group["framework"],
            "status": "planned",
            "run_key": None,
            "error": None,
        }
        try:
            recipe = load_recipe(group)
            image = ci_image.run_image(group["framework"]["id"])
            item["run_key"] = run_key(
                args.commit, image, recipe, group["workloads"], config["client"], group["weights"]
            )
            item["gpus"] = recipe["gpus"]
            if not args.force and unchanged(previous.get(key), item["run_key"]):
                item["status"] = "skipped"
            else:
                write_json(
                    out / group_path(*ids) / "plan.json",
                    {
                        "schema": SCHEMA,
                        **group,
                        "image": {"run": image},
                        "recipe": recipe,
                        "recipe_sha256": sha256_json(recipe),
                        "client": config["client"],
                        "aklr_sha": args.commit,
                        "run_key": item["run_key"],
                    },
                )
        except Exception as error:
            item.update(status="resolve-failed", error=f"{type(error).__name__}: {error}"[:1000])
            print(f"ERROR [{key}]: {item['error']}", file=sys.stderr)
        manifest["groups"].append(item)
    write_json(out / "manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m e2e_perf plan", description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--models", default="all", help="model ids, comma-separated, or all")
    parser.add_argument("--frameworks", default="all", help="sglang, vllm, comma-separated, or all")
    parser.add_argument("--commit", required=True, help="the AKLR commit to measure")
    parser.add_argument("--latest", type=Path, help="site-data e2e/latest.json; missing means none published")
    parser.add_argument("--force", action="store_true", help="measure every selected group, skipping none")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID", "local"))
    parser.add_argument(
        "--pr-comments",
        type=Path,
        help=f"JSON list of PR comments ({{user, body, url}}, oldest first); the latest allowlisted reviewer's "
        f"`{COMMAND} <model ids or all>` selects the models, and without one the plan fails",
    )
    parser.add_argument("--allowlist", type=Path, default=ALLOWLIST_FILE)
    parser.add_argument("--manual-actor", help="GitHub login that started a manual run")
    parser.add_argument("--manual-ref", help="Git ref of a manual run (with --manual-actor)")
    args = parser.parse_args(argv)
    if (args.manual_actor is None) != (args.manual_ref is None):
        parser.error("--manual-actor and --manual-ref go together")
    try:
        check_manual_trigger(args, NAME)
        if args.pr_comments is not None:
            note, skip = apply_pr_scope(args, COMMAND, "e2e perf", list(load_config(args.config)["models"]))
            summarize(note)
            if skip:
                write_outputs([], skipped=True)
                return 0
        manifest = build(args)
    except (ConfigError, ValueError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    measured = [item for item in manifest["groups"] if item["status"] != "skipped"]
    ready = [
        {
            "model": item["model"]["id"],
            "target": item["target"]["id"],
            "framework": item["framework"]["id"],
            "runner": item["target"]["runner_label"],
            "job_index": index,
        }
        for index, item in enumerate(manifest["groups"])
        if item["status"] == "planned"
    ]
    write_outputs(ready, skipped=not measured)
    lines = [f"## {NAME}: plan", "", "| Group | Status |", "| --- | --- |"]
    for item in manifest["groups"]:
        key = group_key(item["model"]["id"], item["target"]["id"], item["framework"]["id"])
        status = "no change, skipped" if item["status"] == "skipped" else item["status"]
        lines.append(f"| {key} | {status} |")
    if not measured:
        lines += ["", "No change in any group; skipped."]
    summarize("\n".join(lines) + "\n")
    return 0
