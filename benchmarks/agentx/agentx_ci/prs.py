# SPDX-License-Identifier: MIT
"""Pull requests merged between the baseline's AKLR commit and this run's, via the GitHub API."""

from __future__ import annotations

import json
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

API = "https://api.github.com"
MAX_COMMITS = 200


def get(path: str, token: str | None):
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(f"{API}{path}", headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())


def first_parent(base: str, head: str, cwd: Path) -> list[str] | None:
    """Commits that landed on the branch in (base, head]; None when the checkout lacks one of them."""
    done = subprocess.run(
        ["git", "rev-list", "--first-parent", f"{base}..{head}"],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    return done.stdout.split() if done.returncode == 0 else None


def compared(repo: str, base: str, head: str, token: str | None) -> list[str]:
    body = get(f"/repos/{repo}/compare/{base}...{head}?per_page=250", token)
    return [c["sha"] for c in body["commits"]][::-1]


def merged_prs(repo: str, base: str, head: str, token: str | None, cwd: Path) -> dict:
    """``{"prs": [...], "error": None}``; a failed lookup returns the error, never raises."""
    out = {"base": base, "head": head, "prs": [], "error": None, "truncated": False}
    if base == head:
        return out
    try:
        shas = first_parent(base, head, cwd)
        if shas is None:
            shas = compared(repo, base, head, token)
        out["truncated"] = len(shas) > MAX_COMMITS
        seen = set()
        for sha in shas[:MAX_COMMITS]:
            for pr in get(f"/repos/{repo}/commits/{sha}/pulls", token):
                if pr.get("merged_at") and pr["number"] not in seen:
                    seen.add(pr["number"])
                    out["prs"].append(
                        {
                            "number": pr["number"],
                            "title": pr["title"],
                            "url": pr["html_url"],
                            "author": (pr.get("user") or {}).get("login"),
                            "merged_at": pr["merged_at"],
                        }
                    )
    except (urllib.error.URLError, OSError, KeyError, ValueError) as error:
        out["error"] = f"{type(error).__name__}: {error}"[:300]
    out["prs"].sort(key=lambda pr: pr["merged_at"], reverse=True)
    return out
