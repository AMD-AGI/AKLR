# SPDX-License-Identifier: MIT
"""Fetch the InferenceX checkout that provides the AgentX client and trace loaders."""

from __future__ import annotations

import shlex
import shutil
import subprocess
import time
from pathlib import Path

INFERENCEX_SPARSE = (
    "/inferencex-e2e/infx/",
    "/inferencex-e2e/configs/",
    "/inferencex-e2e/benchmarks/*/srt-slurm-recipes/",
    "/inferencex-e2e/benchmarks/runtime_settings.sh",
    "/inferencex-e2e/uv.lock",
    "/inferencex-e2e/pyproject.toml",
)


class FetchError(Exception):
    pass


def sh(cmd: list[str], cwd: str | Path | None = None, timeout: int = 300) -> str:
    done = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False)
    if done.returncode:
        raise FetchError(f"{shlex.join(cmd)} failed: {done.stderr.strip()[-400:]}")
    return done.stdout


def github_url(repo: str) -> str:
    return f"https://github.com/{repo}.git"


def fetch_inferencex(repo: str, sha: str, dst: Path, retries: tuple[int, ...] = (15, 30, 60)) -> Path:
    """Fetch a clean sparse InferenceX checkout at the requested commit."""
    if (dst / ".git").is_dir():
        try:
            if sh(["git", "rev-parse", "HEAD^{commit}"], cwd=dst).strip() == sha:
                if sh(["git", "status", "--porcelain", "--untracked-files=all"], cwd=dst).strip():
                    raise FetchError("cached InferenceX checkout is dirty")
                if (dst / "inferencex-e2e/uv.lock").is_file():
                    return dst
        except (FetchError, subprocess.TimeoutExpired):
            pass
    error = None
    for delay in (0, *retries):
        time.sleep(delay)
        shutil.rmtree(dst, ignore_errors=True)
        dst.mkdir(parents=True)
        try:
            sh(["git", "init", "-q"], cwd=dst)
            sh(["git", "remote", "add", "origin", github_url(repo)], cwd=dst)
            sh(["git", "fetch", "-q", "--depth=1", "--filter=blob:none", "origin", sha], cwd=dst)
            sh(["git", "sparse-checkout", "set", "--no-cone", *INFERENCEX_SPARSE], cwd=dst)
            sh(["git", "checkout", "-q", "--detach", "FETCH_HEAD"], cwd=dst)
            return dst
        except (FetchError, subprocess.TimeoutExpired) as e:
            error = e
    raise FetchError(f"fetching {repo}@{sha} failed after {len(retries) + 1} attempts: {error}")
