# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""``python -m aklr export|validate``: forwards to the command modules."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

_COMMANDS = ("export", "validate")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m aklr")
    parser.add_argument("command", choices=_COMMANDS)
    parser.add_argument("command_argv", nargs=argparse.REMAINDER, metavar="args")
    args = parser.parse_args(argv)  # anything else: usage, exit 2
    if args.command == "export":
        from aklr.core.contract import export
        from aklr.core.platform.env import RuntimeEnv

        if args.command_argv[:1] == ["check"]:  # never reads AKLR_*
            return export.main(args.command_argv, export_dir=None)
        try:
            export_dir = RuntimeEnv().export_dir
        except ValueError as error:
            print(error, file=sys.stderr)
            return 1
        return export.main(args.command_argv, export_dir=export_dir)
    from aklr.core.validation import command

    return command.main(args.command_argv)


if __name__ == "__main__":
    sys.exit(main())
