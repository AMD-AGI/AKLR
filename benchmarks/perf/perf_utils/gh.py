# SPDX-License-Identifier: MIT
"""GitHub annotations, job outputs and the allowlist that authorizes a manual run."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import NoReturn

import yaml

ALLOWLIST_FILE = Path(__file__).resolve().parents[1] / "allowlist.yaml"
LOGIN = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9]|-(?=[A-Za-z0-9])){0,38}")


class ConfigError(ValueError):
    pass


def load_reviewers(path: str | Path = ALLOWLIST_FILE) -> list[str]:
    """Read the GitHub logins whose `/agentx-mi355x` comment authorizes AgentX on a PR."""
    data = yaml.safe_load(Path(path).read_text()) or {}
    reviewers = data.get("reviewers") if isinstance(data, dict) else None
    if not isinstance(reviewers, list) or not reviewers:
        raise ConfigError(f"{path}: `reviewers` must be a non-empty list of GitHub logins")
    for login in reviewers:
        if not isinstance(login, str) or not LOGIN.fullmatch(login):
            raise ConfigError(f"{path}: {login!r} is not a GitHub login")
    if len({login.lower() for login in reviewers}) != len(reviewers):
        raise ConfigError(f"{path}: duplicate reviewer")
    return reviewers


def annotate(message: str, level: str = "warning", title: str = "AgentX scope") -> None:
    """Annotate the GitHub run so a skipped, narrowed or refused scope is visible on its check."""
    if os.environ.get("GITHUB_ACTIONS") == "true":
        escaped = message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
        print(f"::{level} title={title}::{escaped}", flush=True)


def summarize(note: str) -> None:
    print(note)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as summary:
            summary.write(note + "\n")


def repo_url() -> str:
    """The repository that report commit links point at, from the Actions run, else AMD-AGI/AKLR on github.com."""
    server = os.environ.get("GITHUB_SERVER_URL") or "https://github.com"
    repository = os.environ.get("GITHUB_REPOSITORY") or "AMD-AGI/AKLR"
    return f"{server.rstrip('/')}/{repository}"


def refuse(message: str, title: str = "AgentX scope") -> NoReturn:
    annotate(message, "error", title)
    summarize(message)
    raise ConfigError(message)


def check_manual_trigger(args: argparse.Namespace, name: str = "AgentX") -> None:
    """Let only an allowlisted reviewer start a manual run of the named benchmark, and only from main."""
    if args.manual_actor is None:
        return
    allowlist = load_reviewers(args.allowlist)
    if args.manual_actor.lower() not in {login.lower() for login in allowlist}:
        reviewers = ", ".join(f"@{login}" for login in allowlist)
        refuse(
            f"@{args.manual_actor} may not start {name} manually. Ask one of {reviewers} to run it, "
            "or to add you to benchmarks/perf/allowlist.yaml.",
            f"{name} scope",
        )
    if args.manual_ref != "refs/heads/main":
        refuse(
            f"Manual {name} runs use main only, not {args.manual_ref}; choose main under `Use workflow from`.",
            f"{name} scope",
        )


def write_outputs(ready: list[dict], skipped: bool = False) -> None:
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as output:
            output.write(f"matrix={json.dumps({'include': ready})}\ncount={len(ready)}\n")
            output.write(f"skipped={str(skipped).lower()}\n")
