###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

_COMMANDS = ("validate",)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m aklr")
    parser.add_argument("command", choices=_COMMANDS)
    parser.add_argument("command_argv", nargs=argparse.REMAINDER, metavar="args")
    args = parser.parse_args(argv)
    from aklr.core.validation import command

    return command.main(args.command_argv)


if __name__ == "__main__":
    sys.exit(main())
