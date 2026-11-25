# SPDX-License-Identifier: MIT
"""Find the AgentX models an allowlisted reviewer requests with a `/agentx-mi355x` pull request comment."""

from __future__ import annotations

import re

COMMAND = "/agentx-mi355x"


def is_request(body: str) -> bool:
    """Whether the comment's first line starts with the command, ignoring case.

    Leading whitespace refuses it: GitHub renders an indented line as code, such as a quoted example.
    """
    first = body.splitlines()[0] if body else ""
    words = first.split(maxsplit=1)
    return bool(words) and not first[0].isspace() and words[0].lower() == COMMAND


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
