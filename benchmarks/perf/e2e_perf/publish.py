# SPDX-License-Identifier: MIT
"""Merge one run's report site into site-data's e2e/ section; groups the run skipped keep their results."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from perf_utils.gh import ConfigError, repo_url

from e2e_perf.model_config import (
    DEFAULT_CONFIG,
    REPO_ROOT,
    group_key,
    group_path,
    groups,
    load_config,
)
from e2e_perf.report import TITLE, render_index
from e2e_perf.utils import SCHEMA, read_json, utc_now, write_json


def git_descends(commit: str, ancestor: str) -> bool:
    """Whether commit strictly descends from ancestor in this checkout; unknown commits do not."""
    if commit == ancestor:
        return False
    done = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "merge-base", "--is-ancestor", ancestor, commit],
        capture_output=True,
        check=False,
    )
    return done.returncode == 0


def remove_tree(path: Path, root: Path) -> None:
    """Remove a group directory and the parents it leaves empty, up to the section root."""
    shutil.rmtree(path, ignore_errors=True)
    for parent in path.parents:
        if parent == root or not parent.is_relative_to(root):
            break
        if parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()


def merge(
    site: Path,
    section: Path,
    source: dict,
    *,
    config: Path | str = DEFAULT_CONFIG,
    descends: Callable[[str, str], bool] = git_descends,
) -> dict:
    """Replace the run's groups, keep the others, drop groups no longer configured, and rebuild the index.

    A group published from a newer main commit is kept: re-running an older commit must not replace it.
    Groups are dropped only when no published group is newer than this run, so an older config cannot
    remove groups added since.
    """
    if not re.fullmatch(r"[A-Za-z0-9_-]+", source["run_id"]):
        raise ConfigError(f"invalid run id {source['run_id']!r}")
    loaded = load_config(config)
    enabled = {
        group_key(g["model"]["id"], g["target"]["id"], g["framework"]["id"]): g for g in groups(loaded)
    }
    run = {row["key"]: row for row in read_json(site / "manifest.json")["groups"]}
    if missing := sorted(key for key in run if key not in enabled):
        raise ConfigError(f"the run measured groups that are not configured: {', '.join(missing)}")
    latest_path = section / "latest.json"
    prior = read_json(latest_path) if latest_path.is_file() else {}
    entries = dict(prior.get("groups", {})) if prior.get("schema") == SCHEMA else {}
    newer = {
        key
        for key, entry in entries.items()
        if (commit := (entry.get("source") or {}).get("commit")) and descends(commit, source["commit"])
    }
    for key, row in run.items():
        if key in newer:
            print(f"{key}: kept the result of newer commit {entries[key]['source']['commit'][:12]}")
            continue
        relative = group_path(*key.split("/"))
        remove_tree(section / relative, section)
        if (site / relative).is_dir():
            shutil.copytree(site / relative, section / relative)
            if (site / relative / "results.json").is_file():
                archive = section / "runs" / source["run_id"] / relative / "results.json"
                archive.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(site / relative / "results.json", archive)
        entries[key] = {**row, "run_url": row.get("run_url") or source["run_url"], "source": source}
    if newer:
        entries = {key: entries[key] for key in [*enabled, *entries] if key in entries}
    else:
        entries = {key: entries[key] for key in enabled if key in entries}
    for path in section.glob("models/*/targets/*/frameworks/*"):
        if path.is_dir() and group_key(path.parts[-5], path.parts[-3], path.parts[-1]) not in entries:
            remove_tree(path, section)
    latest = {"schema": SCHEMA, "updated_at": utc_now(), "groups": entries}
    write_json(latest_path, latest)
    rows = list(entries.values())
    (section / "index.html").write_text(render_index(rows, repo_url(), loaded["workloads"]))
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    write_json(
        section / "section.json",
        {
            "schema": 1,
            "id": "e2e",
            "title": dict(TITLE),
            "path": "index.html",
            "updated_at": latest["updated_at"],
            "last_run": source,
            "targets": len(rows),
            "status_counts": counts,
        },
    )
    return latest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m e2e_perf publish", description=__doc__)
    parser.add_argument("--site", type=Path, required=True, help="this run's report site")
    parser.add_argument("--section", type=Path, required=True, help="e2e/ in a site-data checkout")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
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
        latest = merge(args.site, args.section, source, config=args.config)
    except (ConfigError, ValueError, OSError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    for key, row in latest["groups"].items():
        print(f"{key}: {row['status']} from run {row['source']['run_id']}")
    return 0
