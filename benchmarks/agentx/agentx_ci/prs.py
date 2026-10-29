# SPDX-License-Identifier: MIT
"""What landed on main between the baseline run's AKLR commit and this run's, from git history.

``git log --first-parent base..head``: one entry per commit on main. A squash merge names its
PR as a trailing ``(#123)``; a merge commit says ``Merge pull request #123 from …`` and keeps
the PR title as the first body line. Anything else is a direct commit.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

SQUASH = re.compile(r"^(?P<title>.*\S)\s*\(#(?P<number>\d+)\)$")
MERGE = re.compile(r"^Merge pull request #(?P<number>\d+) from \S+$")
FIELDS = ("sha", "author", "date", "subject", "body")
# Fields split by 0x1f, records by 0x1e: neither occurs in commit messages.
LOG_FORMAT = "%H%x1f%an%x1f%cI%x1f%s%x1f%b%x1e"


class HistoryError(Exception):
    pass


def git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False
    )


def has_commit(cwd: Path, sha: str) -> bool:
    return git(cwd, "cat-file", "-e", f"{sha}^{{commit}}").returncode == 0


def ensure_history(
    cwd: Path, shas: tuple[str, ...], remote: str = "origin"
) -> list[str]:
    """Make ``shas`` and the history between them local; returns the fetches it needed.

    A shallow checkout is unshallowed first: a lone fetched commit would not connect to it.
    """
    done = []
    if git(cwd, "rev-parse", "--is-shallow-repository").stdout.strip() == "true":
        git(cwd, "fetch", "--no-tags", "--quiet", "--unshallow", remote)
        done.append("unshallow")
    for sha in shas:
        if has_commit(cwd, sha):
            continue
        fetched = git(cwd, "fetch", "--no-tags", "--quiet", remote, sha)
        if fetched.returncode or not has_commit(cwd, sha):
            reason = fetched.stderr.strip()[-200:] or "not found"
            raise HistoryError(
                f"commit {sha} is not in {cwd} and fetching it from {remote} failed ({reason});"
                " check out with fetch-depth: 0"
            )
        done.append(sha)
    return done


def entry(commit: dict, repo_url: str) -> dict:
    subject, body = commit["subject"], commit["body"].strip()
    out = {
        "sha": commit["sha"],
        "author": commit["author"],
        "date": commit["date"][:10],
        "commit_url": f"{repo_url}/commit/{commit['sha']}",
    }
    if m := MERGE.match(subject):
        title = body.splitlines()[0] if body else subject
        return {**out, "kind": "pr", "number": int(m["number"]), "title": title}
    if m := SQUASH.match(subject):
        return {**out, "kind": "pr", "number": int(m["number"]), "title": m["title"]}
    return {**out, "kind": "commit", "number": None, "title": subject}


def landed(base: str, head: str, cwd: Path, repo_url: str) -> dict:
    """``{"entries": [...], "error": None}`` for (base, head]; a git failure is the error, never raised."""
    out = {"base": base, "head": head, "entries": [], "error": None, "fetched": []}
    try:
        out["fetched"] = ensure_history(cwd, (base, head))
        if git(cwd, "merge-base", "--is-ancestor", base, head).returncode:
            raise HistoryError(
                f"baseline {base[:10]} is not an ancestor of {head[:10]}"
            )
        log = git(
            cwd, "log", "--first-parent", f"--format={LOG_FORMAT}", f"{base}..{head}"
        )
        if log.returncode:
            raise HistoryError(log.stderr.strip()[-300:])
    except HistoryError as error:
        out["error"] = str(error)
        return out
    url = repo_url.rstrip("/")
    for record in filter(None, (r.strip("\n") for r in log.stdout.split("\x1e"))):
        commit = dict(zip(FIELDS, record.split("\x1f"), strict=True))
        item = entry(commit, url)
        if item["number"] is not None:
            item["url"] = f"{url}/pull/{item['number']}"
        out["entries"].append(item)
    return out
