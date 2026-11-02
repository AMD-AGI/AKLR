###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

import contextlib
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

KILL_WAIT_SECONDS: Final = 5.0


@dataclass(frozen=True)
class ProcessGroup:
    """A process group, identified by its leader's pid."""

    group_id: int

    def signal(self, signal_number: int) -> None:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(self.group_id, signal_number)

    def kill(self) -> bool:
        self.signal(signal.SIGKILL)
        deadline = time.monotonic() + KILL_WAIT_SECONDS
        while (is_alive := self.is_alive) and time.monotonic() < deadline:
            time.sleep(0.01)
        return not is_alive

    @property
    def is_alive(self) -> bool:
        for stat_path in Path("/proc").glob("[0-9]*/stat"):
            try:
                state, _, process_group = stat_path.read_text().rpartition(")")[2].split()[:3]
            except (OSError, ValueError):
                continue
            # a zombie (an orphan no init reaps, as in a container) holds nothing
            if int(process_group) == self.group_id and state != "Z":
                return True
        return False
