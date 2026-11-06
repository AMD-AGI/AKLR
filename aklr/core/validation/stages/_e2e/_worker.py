# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host-local process supervision and socket ownership checks (also executed over SSH).

This standalone script uses only the standard library. The checkout and state directory
must be shared at the same absolute paths on remote hosts.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


def group_members(group: int) -> list[Path]:
    members = []
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            state, _, pgid = stat.read_text().rpartition(")")[2].split()[:3]
        except (OSError, ValueError):
            continue
        if int(pgid) == group and state != "Z":
            members.append(stat.parent)
    return members


def stop_group(group: int) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(group, signal.SIGKILL)
    deadline = time.monotonic() + 5
    while group_members(group):
        if time.monotonic() >= deadline:
            raise RuntimeError(f"process group {group} outlived SIGKILL")
        time.sleep(0.01)


def owns_port(group: int, port: int) -> bool:
    """Require every listener on the port to be owned by this group (Linux /proc)."""
    listeners = set()
    for name in ("tcp", "tcp6"):
        table = Path("/proc/net") / name
        if not table.exists():
            continue
        for line in table.read_text().splitlines()[1:]:
            fields = line.split()
            if fields[3] == "0A" and int(fields[1].rsplit(":", 1)[1], 16) == port:
                listeners.add(f"socket:[{fields[9]}]")
    owned = set()
    for process in group_members(group):
        for fd in (process / "fd").glob("*"):
            with contextlib.suppress(FileNotFoundError, ProcessLookupError):
                owned.add(os.readlink(fd))
    return bool(listeners) and listeners <= owned


def available(host: str, port: int) -> None:
    for family, kind, proto, _, address in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM):
        with socket.socket(family, kind, proto) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(address)


@contextlib.contextmanager
def locked(state: Path):
    state.parent.mkdir(parents=True, exist_ok=True)
    with state.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def stop_remote(state: Path) -> None:
    with locked(state):
        # Serialize with start: even a delayed SSH command must not launch after stop.
        state.with_suffix(".stopped").touch()
        if state.exists():
            stop_group(int(state.read_text()))
            state.unlink()


def start_remote(state: Path, argv: list[str]) -> int:
    child = None
    interrupted_by = 0

    def interrupted(signum, frame):
        nonlocal interrupted_by
        interrupted_by = signum
        # Do not raise between Popen creating the child and assigning it to `child`.
        if child is not None:
            stop_group(child.pid)

    for sig in (signal.SIGHUP, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    try:
        with locked(state):
            if interrupted_by or state.with_suffix(".stopped").exists():
                raise RuntimeError("deployment was already stopped")
            child = subprocess.Popen(argv, start_new_session=True, stdin=subprocess.DEVNULL)
            state.write_text(str(child.pid))
        if interrupted_by:
            stop_group(child.pid)
        return child.wait()
    finally:
        # Runs on SSH disconnect as well as a normal worker exit. Explicit remote stop
        # remains necessary: killing a local SSH process does not verify remote cleanup.
        if child is not None:
            with locked(state):
                stop_group(child.pid)
                child.wait()
                state.unlink(missing_ok=True)


def main() -> int:
    action, *args = sys.argv[1:]
    if action == "available":
        available(args[0], int(args[1]))
    elif action == "owns-port":
        group = int(args[0]) if args[0].isdigit() else int(Path(args[0]).read_text())
        return 0 if owns_port(group, int(args[1])) else 1
    elif action == "stop":
        stop_remote(Path(args[0]))
    elif action == "start":
        return start_remote(Path(args[0]), args[1:])
    else:
        raise ValueError(f"unknown operation: {action}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
