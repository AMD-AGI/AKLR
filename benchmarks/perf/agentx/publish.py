# SPDX-License-Identifier: MIT
"""Merge one run's report site into the agentx/ section of the site-data branch, by target."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from agentx.model_config import ConfigError, select_runs
from agentx.report_all import indexed_targets, render_index, source_commits
from agentx.utils import DEFAULT_CONFIG, SCHEMA, load_config, read_json, target_path, utc_now, write_json

SECTION_SCHEMA = 1
SECTION_TITLE = {"en": "AgentX end-to-end performance", "zh": "AgentX 端到端性能"}


def target_dirs(section: Path) -> dict[tuple[str, str, str], Path]:
    """Map the model/target/framework directories present under a section."""
    return {
        (path.parts[-5], path.parts[-3], path.parts[-1]): path
        for path in section.glob("models/*/targets/*/frameworks/*")
        if path.is_dir()
    }


def remove_tree(path: Path, root: Path) -> None:
    """Remove a target directory and the parents it leaves empty, up to the section root."""
    shutil.rmtree(path, ignore_errors=True)
    for parent in path.parents:
        if parent == root or not parent.is_relative_to(root):
            break
        if parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()


def section_summary(manifest: dict, source: dict) -> dict:
    counts: dict[str, int] = {}
    for item in manifest["targets"]:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    return {
        "schema": SECTION_SCHEMA,
        "id": "agentx",
        "title": SECTION_TITLE,
        "path": "index.html",
        "updated_at": manifest["generated_at"],
        "last_run": source,
        "targets": len(manifest["targets"]),
        "status_counts": counts,
        "commits": source_commits(manifest),
    }


def merge(site: Path, section: Path, source: dict, *, config: Path | str = DEFAULT_CONFIG) -> dict:
    """Replace the run's targets in section, keep the others, and drop targets no longer enabled."""
    run = indexed_targets(read_json(site / "manifest.json"))
    enabled = select_runs(load_config(config))
    prior_path = section / "manifest.json"
    prior_manifest = read_json(prior_path) if prior_path.is_file() else None
    prior = (
        indexed_targets(prior_manifest) if prior_manifest and prior_manifest.get("schema") == SCHEMA else {}
    )
    if missing := sorted("/".join(key) for key in run if key not in enabled):
        raise ConfigError(f"the run measured targets that are not enabled: {', '.join(missing)}")

    entries = dict(prior)
    for key, item in run.items():
        relative = target_path(*key)
        remove_tree(section / relative, section)
        if (site / relative).is_dir():
            shutil.copytree(site / relative, section / relative)
        entries[key] = {**item, "source": source}
    entries = {key: entries[key] for key in enabled if key in entries}
    for key, path in target_dirs(section).items():
        if key not in entries:
            remove_tree(path, section)

    manifest = {
        "schema": SCHEMA,
        "run_id": source["run_id"],
        "generated_at": utc_now(),
        "targets": list(entries.values()),
    }
    write_json(prior_path, manifest)
    (section / "index.html").write_text(render_index(manifest, home="../index.html"))
    write_json(section / "section.json", section_summary(manifest, source))
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m agentx publish", description=__doc__)
    parser.add_argument("--site", type=Path, required=True, help="this run's report site")
    parser.add_argument("--section", type=Path, required=True, help="agentx/ in a site-data checkout")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-url", required=True)
    parser.add_argument("--commit", required=True, help="the main commit this run measured")
    parser.add_argument("--ref", required=True)
    parser.add_argument("--event", required=True)
    args = parser.parse_args(argv)
    source = {
        "run_id": args.run_id,
        "run_url": args.run_url,
        "commit": args.commit,
        "ref": args.ref,
        "event": args.event,
    }
    try:
        manifest = merge(args.site, args.section, source, config=args.config)
    except (ConfigError, ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    for item in manifest["targets"]:
        print(
            f"{item['model']['id']}/{item['target']['id']}/{item['framework']['id']}: "
            f"{item['status']} from run {item.get('source', {}).get('run_id', '?')}"
        )
    return 0
