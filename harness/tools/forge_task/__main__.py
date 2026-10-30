# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""`python harness/tools/forge_task`: run the tool from a checkout, nothing installed."""

import sys
from pathlib import Path

# Run as a directory, sys.path[0] is this package's own folder, which would make its
# modules importable under bare names; its parent makes them importable as forge_task.*.
sys.path[0] = str(Path(__file__).resolve().parent.parent)

from forge_task.cli import main

sys.exit(main())
