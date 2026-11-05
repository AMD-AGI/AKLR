# SPDX-License-Identifier: MIT
"""Command-line entry point for the AgentX benchmark tools."""

from __future__ import annotations

import sys

from agentx_ci.__main__ import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
