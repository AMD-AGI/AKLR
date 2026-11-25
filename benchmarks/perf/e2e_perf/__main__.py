# SPDX-License-Identifier: MIT
"""Dispatch daily end-to-end performance CLI commands."""

from __future__ import annotations

import importlib
import sys

COMMANDS = {
    "plan": ("e2e_perf.plan", "main"),
    "inspect-images": ("e2e_perf.registry", "main"),
    "sweep": ("e2e_perf.sweep", "main"),
    "report": ("e2e_perf.report", "main"),
    "publish": ("e2e_perf.publish", "main"),
}


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in COMMANDS:
        print(f"usage: python3 -m e2e_perf {{{','.join(COMMANDS)}}} [args]", file=sys.stderr)
        return 2
    module, entrypoint = COMMANDS[argv[0]]
    return getattr(importlib.import_module(module), entrypoint)(argv[1:])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
