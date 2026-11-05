# SPDX-License-Identifier: MIT
"""Dispatch the AgentX benchmark CLI commands."""

from __future__ import annotations

import importlib
import sys

COMMANDS = {
    "plan": ("agentx_ci.plan", "main"),
    "resolve": ("agentx_ci.recipe_resolve", "main"),
    "sweep": ("agentx_ci.sweep", "main"),
    "pipeline": ("agentx_ci.pipeline", "main"),
    "report": ("agentx_ci.report", "main"),
}


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in COMMANDS:
        print(
            f"usage: python3 -m agentx_bench {{{','.join(COMMANDS)}}} [args]",
            file=sys.stderr,
        )
        return 2
    module, entrypoint = COMMANDS[argv[0]]
    return getattr(importlib.import_module(module), entrypoint)(argv[1:])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
