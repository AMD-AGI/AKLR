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
    """A process group, identified by its leader's pid, with the groups its members'
    descendants lead: a nested runner's own (setsid), which its parent's death does not end."""

    group_id: int

    def signal(self, signal_number: int) -> None:
        for group_id in self._tree_group_ids():
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(group_id, signal_number)

    def kill(self) -> bool:
        # every group found before any member dies: a dead parent's children lose their lineage
        group_ids = self._tree_group_ids()
        for group_id in group_ids:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(group_id, signal.SIGKILL)
        deadline = time.monotonic() + KILL_WAIT_SECONDS
        while (
            is_alive := any(g in group_ids for _, g in _live_processes().values())
        ) and time.monotonic() < deadline:
            time.sleep(0.01)
        return not is_alive

    def _tree_group_ids(self) -> set[int]:
        processes = _live_processes()
        group_ids = {self.group_id}
        pids = {pid for pid, (_, group_id) in processes.items() if group_id == self.group_id}
        while pids:
            children = {pid for pid, (parent, _) in processes.items() if parent in pids}
            group_ids |= {processes[pid][1] for pid in children}
            pids = children
        return group_ids


def _live_processes() -> dict[int, tuple[int, int]]:
    """Each live (not zombie) process's (parent pid, process group), by pid."""
    processes = {}
    for stat_path in Path("/proc").glob("[0-9]*/stat"):
        try:
            state, parent, process_group = stat_path.read_text().rpartition(")")[2].split()[:3]
        except (OSError, ValueError):
            continue
        if state != "Z":
            processes[int(stat_path.parent.name)] = (int(parent), int(process_group))
    return processes
