# SPDX-License-Identifier: MIT
"""``python3 -m agentx_ci {resolve,sweep,report}``, run from benchmarks/agentx."""

from __future__ import annotations

import importlib
import sys

COMMANDS = {
    "resolve": "agentx_ci.recipe_resolve",
    "sweep": "agentx_ci.sweep",
    "report": "agentx_ci.report",
}


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in COMMANDS:
        print(
            f"usage: python3 -m agentx_ci {{{','.join(COMMANDS)}}} [args]",
            file=sys.stderr,
        )
        return 2
    return importlib.import_module(COMMANDS[argv[0]]).main(argv[1:])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
