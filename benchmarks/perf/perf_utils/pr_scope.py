# SPDX-License-Identifier: MIT
"""Find the models an allowlisted reviewer requests with a pull request comment such as `/agentx-mi355x`."""

from __future__ import annotations

import argparse
import json
import re

from perf_utils.gh import ConfigError, annotate, load_reviewers, refuse


def is_request(body: str, command: str) -> bool:
    """Whether the comment's first line starts with the command, ignoring case.

    Leading whitespace refuses it: GitHub renders an indented line as code, such as a quoted example.
    """
    first = body.splitlines()[0] if body else ""
    words = first.split(maxsplit=1)
    return bool(words) and not first[0].isspace() and words[0].lower() == command


def authorized_request(comments: list[dict], reviewers: list[str], command: str) -> dict | None:
    """Return the latest request among comments (oldest first) by one of reviewers, or None."""
    allowed = {login.lower() for login in reviewers}
    for comment in reversed(comments):
        if str(comment.get("user") or "").lower() in allowed and is_request(
            comment.get("body") or "", command
        ):
            return comment
    return None


def requested_models(comment: str, enabled: list[str], command: str) -> tuple[str, list[str]]:
    """Map the names after the comment's leading command onto enabled model ids, ignoring case.

    Only the first line counts, and it must start with the command. `all` selects every model only
    when it is the sole name. Return the selection (`all`, comma-separated model ids, or empty when
    nothing is selected) and the names that are not enabled model ids, without failing on what the
    comment says.
    """
    if not is_request(comment, command):
        return "", []
    words = comment.splitlines()[0].split(maxsplit=1)
    rest = words[1] if len(words) > 1 else ""
    names = [name for name in re.split(r"[\s,]+", rest.replace("`", "")) if name]
    if names and {name.lower() for name in names} == {"all"}:
        return "all", []
    ids = {model.lower(): model for model in enabled}
    unmatched = list(dict.fromkeys(name for name in names if name.lower() not in ids))
    return ",".join(dict.fromkeys(ids[name.lower()] for name in names if name.lower() in ids)), unmatched


def apply_pr_scope(
    args: argparse.Namespace, command: str, name: str, enabled: list[str]
) -> tuple[str | None, bool]:
    """Narrow args.models to the latest command request of an allowlisted reviewer on the PR.

    Return a note for the job summary and whether to skip the benchmark, as when the request selects
    no model. Raise ConfigError when no allowlisted reviewer has requested the benchmark on the PR.
    """
    if args.models != "all":
        raise ConfigError("--pr-comments cannot be combined with --models")
    title = f"{name} scope"
    allowlist = load_reviewers(args.allowlist)
    request = authorized_request(json.loads(args.pr_comments.read_text()), allowlist, command)
    if request is None:
        reviewers = ", ".join(f"@{login}" for login in allowlist)
        refuse(
            f"No {name} reviewer has authorized this PR. Ask one of {reviewers} to review it and comment "
            f"`{command} <model ids or all>` on it, then re-run this workflow.",
            title,
        )
    declared, unmatched = requested_models(request["body"], enabled, command)
    source = f"@{request['user']}'s `{command}` comment" + (
        f" ({request['url']})" if request.get("url") else ""
    )
    missed = (
        f" Not an enabled model id, skipped: {', '.join(f'`{item}`' for item in unmatched)}."
        if unmatched
        else ""
    )
    if not declared:
        note = f"{source} selects no model; skipping {name}.{missed}"
        annotate(note, title=title)
        return note, True
    if unmatched:
        annotate(missed.strip(), title=title)
    args.models = declared
    if declared == "all":
        return f"{source} requests `all`; running every enabled model.", False
    return f"{source} selects {declared}; skipping the other models.{missed}", False
