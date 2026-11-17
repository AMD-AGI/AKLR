# SPDX-License-Identifier: MIT
"""Find the AgentX models an allowlisted reviewer requests with a `/agentx-mi355x` pull request comment."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from agentx_ci.model_config import ConfigError

COMMAND = "/agentx-mi355x"
ALLOWLIST_FILE = Path(__file__).resolve().parents[1] / "allowlist.yaml"
LOGIN = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9]|-(?=[A-Za-z0-9])){0,38}")


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


def is_request(body: str) -> bool:
    """Whether the comment's first line starts with the command, ignoring case."""
    words = (body or "").splitlines()[0].split(maxsplit=1) if body else []
    return bool(words) and words[0].lower() == COMMAND


def authorized_request(comments: list[dict], reviewers: list[str]) -> dict | None:
    """Return the latest request among comments (oldest first) by one of reviewers, or None."""
    allowed = {login.lower() for login in reviewers}
    for comment in reversed(comments):
        if str(comment.get("user") or "").lower() in allowed and is_request(comment.get("body") or ""):
            return comment
    return None


def requested_models(comment: str, enabled: list[str]) -> tuple[str, list[str]]:
    """Map the names after the comment's leading `/agentx-mi355x` onto enabled model ids, ignoring case.

    Only the first line counts, and it must start with the command. `all` selects every model only
    when it is the sole name. Return the selection (`all`, comma-separated model ids, or empty when
    nothing is selected) and the names that are not enabled model ids, without failing on what the
    comment says.
    """
    if not is_request(comment):
        return "", []
    words = comment.splitlines()[0].split(maxsplit=1)
    rest = words[1] if len(words) > 1 else ""
    names = [name for name in re.split(r"[\s,]+", rest.replace("`", "")) if name]
    if names and {name.lower() for name in names} == {"all"}:
        return "all", []
    ids = {model.lower(): model for model in enabled}
    unmatched = list(dict.fromkeys(name for name in names if name.lower() not in ids))
    return ",".join(dict.fromkeys(ids[name.lower()] for name in names if name.lower() in ids)), unmatched
