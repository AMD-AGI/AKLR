# SPDX-License-Identifier: MIT
"""Read the AgentX models a pull request declares on its description's `AgentX:` line."""

from __future__ import annotations

import re

COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
LINE = re.compile(r"^[ \t]*(?:[-*][ \t]+)?\**AgentX\**[ \t]*:\**[ \t]*(.*)$", re.IGNORECASE | re.MULTILINE)


def declared_models(body: str, enabled: list[str]) -> tuple[str, list[str]]:
    """Map the `AgentX:` lines onto enabled model ids, ignoring case, without failing on what they say.

    `all` selects every model only when it is the lines' sole name. Return the selection (`all`,
    comma-separated model ids, or empty when nothing is selected) and the names that are not
    enabled model ids.
    """
    lines = LINE.findall(COMMENT.sub("", body or ""))
    names = [name for line in lines for name in re.split(r"[\s,]+", line.replace("`", "")) if name]
    if names and {name.lower() for name in names} == {"all"}:
        return "all", []
    ids = {model.lower(): model for model in enabled}
    unmatched = list(dict.fromkeys(name for name in names if name.lower() not in ids))
    return ",".join(dict.fromkeys(ids[name.lower()] for name in names if name.lower() in ids)), unmatched
