# SPDX-License-Identifier: MIT
"""Dispatch AgentX CLI commands."""

from __future__ import annotations

import importlib
import sys

COMMANDS = {
    "plan": ("agentx.plan", "main"),
    "sweep": ("agentx.sweep", "main"),
    "pipeline": ("agentx.pipeline", "main"),
    "report": ("agentx.report", "main"),
    "publish": ("agentx.publish", "main"),
    "check-recipes": ("agentx.check_recipes", "main"),
}


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in COMMANDS:
        print(
            f"usage: python3 -m agentx {{{','.join(COMMANDS)}}} [args]",
            file=sys.stderr,
        )
        return 2
    module, entrypoint = COMMANDS[argv[0]]
    return getattr(importlib.import_module(module), entrypoint)(argv[1:])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
