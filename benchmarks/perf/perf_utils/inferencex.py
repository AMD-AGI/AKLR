# SPDX-License-Identifier: MIT
"""Fetch a clean sparse InferenceX checkout at a pinned commit."""

from __future__ import annotations

import shlex
import shutil
import subprocess
import time
from pathlib import Path


class FetchError(Exception):
    pass


def git(argv: list[str], cwd: Path) -> str:
    done = subprocess.run(["git", *argv], cwd=cwd, capture_output=True, text=True, timeout=300, check=False)
    if done.returncode:
        raise FetchError(f"git {shlex.join(argv)} failed: {done.stderr.strip()[-400:]}")
    return done.stdout


def fetch(
    repo: str,
    sha: str,
    dst: Path,
    sparse: tuple[str, ...],
    marker: str,
    retries: tuple[int, ...] = (15, 30, 60),
) -> Path:
    """Return dst, a clean checkout of sparse at sha; a cached one is reused when it holds marker."""
    if (dst / ".git").is_dir():
        try:
            clean = not git(["status", "--porcelain", "--untracked-files=all"], dst).strip()
            if git(["rev-parse", "HEAD^{commit}"], dst).strip() == sha and clean and (dst / marker).exists():
                return dst
        except (FetchError, subprocess.TimeoutExpired):
            pass
    error = None
    for delay in (0, *retries):
        time.sleep(delay)
        shutil.rmtree(dst, ignore_errors=True)
        dst.mkdir(parents=True)
        try:
            git(["init", "-q"], dst)
            git(["remote", "add", "origin", f"https://github.com/{repo}.git"], dst)
            git(["fetch", "-q", "--depth=1", "--filter=blob:none", "origin", sha], dst)
            git(["sparse-checkout", "set", "--no-cone", *sparse], dst)
            git(["checkout", "-q", "--detach", "FETCH_HEAD"], dst)
            return dst
        except (FetchError, subprocess.TimeoutExpired) as failure:
            error = failure
    raise FetchError(f"fetching {repo}@{sha} failed after {len(retries) + 1} attempts: {error}")
