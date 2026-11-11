"""Shepherd's round ledger: one live round on this host at a time, and atomic snapshots of its state.

This is a storage helper, not a budget gate, a workflow engine, or a security boundary.
"""

import argparse
import fcntl
import hashlib
import json
import os
import secrets
import signal
import stat
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path


class StateError(Exception):
    pass


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise StateError(f"cannot read valid JSON from {path}: {exc}") from exc


def sync_directory(root):
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, value):
    fd, temporary = tempfile.mkstemp(prefix=".snapshot-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def guarded(root):
    # Never create a missing ledger as a side effect of a normal operation.
    if not root.is_dir():
        raise StateError("state directory missing; investigate or initialize during setup")
    with (root / ".guard").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        yield


def snapshot(root):
    value = read_json(root / "state.json")
    if (
        not isinstance(value, dict)
        or type(value.get("revision")) is not int
        or value["revision"] < 0
        or not isinstance(value.get("state"), dict)
    ):
        raise StateError("invalid state envelope; restore and reconcile, do not reinitialize")
    return value


@dataclass(frozen=True)
class Round:
    """One run of the round wrapper, identified on this host by its process, boot and PID namespace.

    The wrapper's agent and every process it starts inherit `AKLR_ROUND_ID`, which is how a later
    wrapper finds them even when they run in their own sessions.
    """

    id: str
    pid: int
    start: str
    boot_id: str
    pid_namespace: str
    config_sha256: str
    # The agent's exit code once the round ended; None while it runs, or when it never started or was killed.
    exit_code: int | None = None

    @staticmethod
    def process_start(pid):
        # Field 22 of /proc/<pid>/stat; the command name before it may contain spaces.
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]

    @classmethod
    def begin(cls, config_path):
        return cls(
            id=secrets.token_hex(12),
            pid=os.getpid(),
            start=cls.process_start("self"),
            boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            pid_namespace=os.readlink("/proc/self/ns/pid"),
            config_sha256=hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        )

    @classmethod
    def recorded(cls, path):
        if not path.exists():
            return None
        value = read_json(path)
        try:
            return cls(**value)
        except TypeError as exc:
            raise StateError(f"invalid round record {path}; investigate, do not delete it") from exc

    def processes(self):
        """PIDs this user can see whose initial environment carries this round's id."""
        marker = f"AKLR_ROUND_ID={self.id}".encode()
        found = []
        for environ in Path("/proc").glob("[0-9]*/environ"):
            with suppress(OSError):
                if marker in environ.read_bytes().split(b"\0"):
                    found.append(int(environ.parent.name))
        return found

    def is_alive(self, here):
        """False only when none of this round's processes can still run; undecidable counts as alive."""
        if self.boot_id != here.boot_id:
            return False
        if self.pid_namespace != here.pid_namespace:
            return True
        with suppress(OSError):
            if Round.process_start(self.pid) == self.start:
                return True
        return bool(self.processes())

    def run(self, root, agent, recovered_from):
        env = {**os.environ, "AKLR_ROUND_ID": self.id, "AKLR_STATE_DIR": str(root)}
        if recovered_from:
            env["AKLR_RECOVERED_FROM"] = recovered_from
        code = None
        try:
            child = subprocess.Popen(agent, env=env)
            stopping = []
            for signum in (signal.SIGTERM, signal.SIGINT):
                signal.signal(signum, lambda *_: stopping.append(time.monotonic()) or child.terminate())
            while code is None:
                try:
                    code = child.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    # A scheduler's timeout must end the round even when the agent ignores SIGTERM.
                    if stopping and time.monotonic() > stopping[0] + 10:
                        child.kill()
        finally:
            # Tool commands run in their own sessions, so only the round id finds what the agent left behind.
            while leftovers := self.processes():
                for pid in leftovers:
                    with suppress(ProcessLookupError):
                        os.kill(pid, signal.SIGKILL)
                time.sleep(0.1)
            # The exit code tells the next round whether this one's intents need reconciling.
            with guarded(root):
                atomic_json(root / "last_round.json", asdict(replace(self, exit_code=code)))
                (root / "owner.json").unlink()
                sync_directory(root)
        return code


def check_private_path(root):
    """Refuse a path another user could pre-create or swap, e.g. `/var/tmp/$USER` made by someone else."""
    for chain in (root, root.resolve()):
        for path in (chain, *chain.parents):
            if not os.path.lexists(path):
                continue
            info = os.lstat(path)
            if info.st_uid not in (os.getuid(), 0):
                raise StateError(f"{path} is owned by another user")
            if stat.S_ISLNK(info.st_mode):
                continue
            # A sticky directory such as /var/tmp keeps others from replacing what we own inside it.
            if info.st_mode & stat.S_ISVTX:
                break
            if info.st_mode & (stat.S_IWOTH | stat.S_IWGRP):
                raise StateError(f"{path} is writable by other users")


def execute(args):
    # Config keeps `$USER`-style values so it never records a username or home path.
    root = Path(os.path.expandvars(args.root)).expanduser()
    if not root.is_absolute():
        raise StateError("--root must be an absolute persistent directory")
    check_private_path(root)
    if args.command == "init":
        # A fresh host has no `/var/tmp/$USER/shepherd` yet. Every directory made here is private,
        # whatever the umask; exclusive creation of the root also rejects a lost ledger.
        for directory in [*reversed([p for p in root.parents if not p.exists()]), root]:
            try:
                directory.mkdir(mode=0o700)
            except FileExistsError as exc:
                raise StateError("directory already exists; never initialize over existing state") from exc
            directory.chmod(0o700)
        value = {
            "revision": 0,
            "state": {
                "campaigns": [],
                "last_main_sha": None,
            },
        }
        atomic_json(root / "state.json", value)
        sync_directory(root.parent)
        return value

    owner_path = root / "owner.json"
    if args.command == "round":
        if not args.agent:
            raise StateError("round needs the agent command after --")
        here = Round.begin(args.config)
        now = datetime.now(timezone.utc)
        with guarded(root):
            value = snapshot(root)
            previous = Round.recorded(owner_path)
            if previous and previous.is_alive(here):
                with (root / "reports.md").open("a") as reports:
                    reports.write(
                        f"\n{now.isoformat()}: round {previous.id} is still running; trigger skipped.\n"
                    )
                return {"busy": previous.id}
            # A round that died or failed is recovered at once: its saved intents need reconciling,
            # whatever wake_at says.
            last = Round.recorded(root / "last_round.json")
            if previous is None and last is not None and last.exit_code != 0:
                previous = last
            if previous is None and not args.now:
                wake_at = value["state"].get("wake_at")
                try:
                    wake = datetime.fromisoformat(wake_at) if isinstance(wake_at, str) else None
                except ValueError:
                    wake = None
                if (
                    last is not None
                    and last.config_sha256 == here.config_sha256
                    and wake is not None
                    and wake.tzinfo is not None
                    # A wrong wake_at can delay a round by at most two hours.
                    and now < wake <= now + timedelta(hours=2)
                ):
                    return {"waiting_until": wake_at}
            atomic_json(owner_path, asdict(here))
        code = here.run(root, args.agent, previous.id if previous else None)
        return {"round": here.id, "recovered_from": previous.id if previous else None, "exit_code": code}

    with guarded(root):
        value = snapshot(root)
        owner = Round.recorded(owner_path)
        if args.command == "show":
            return {**value, "owner": asdict(owner) if owner else None}
        if owner is None or owner.id != os.environ.get("AKLR_ROUND_ID"):
            raise StateError("only the running round may save: AKLR_ROUND_ID does not match owner.json")
        if args.revision != value["revision"]:
            raise StateError(f"stale revision: expected {value['revision']}")
        state = read_json(Path(args.input))
        if not isinstance(state, dict):
            raise StateError("--input must contain a state object, not a list or scalar")
        # The committed budget is derived from the campaigns, so a save may never drop one or
        # change the bounds it was admitted with.
        campaigns = {}
        for campaign in state.get("campaigns", []):
            if (
                not isinstance(campaign, dict)
                or not isinstance(campaign.get("id"), str)
                or campaign["id"] in campaigns
            ):
                raise StateError("every campaign needs a unique string id")
            campaigns[campaign["id"]] = campaign
        for kept in value["state"].get("campaigns", []):
            if kept["id"] not in campaigns:
                raise StateError(f"campaign {kept['id']} is missing; campaigns are never removed")
            if "bounds" in kept and campaigns[kept["id"]].get("bounds") != kept["bounds"]:
                raise StateError(f"campaign {kept['id']} changes its admitted bounds")
        value = {"revision": value["revision"] + 1, "state": state}
        atomic_json(root / "state.json", value)
        return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    sub.add_parser("show")
    round_ = sub.add_parser("round", help="run the agent command after -- as the ledger's one round")
    round_.add_argument("--config", required=True)
    round_.add_argument("--now", action="store_true", help="run even if the ledger's wake_at is ahead")
    round_.add_argument("agent", nargs=argparse.REMAINDER)
    save = sub.add_parser("save")
    save.add_argument("--revision", required=True, type=int)
    save.add_argument("--input", required=True)
    args = parser.parse_args()
    if args.command == "round" and args.agent[:1] == ["--"]:
        args.agent = args.agent[1:]
    try:
        result = execute(args)
    except (StateError, OSError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1
    print(json.dumps(result, allow_nan=False))
    return result.get("exit_code", 0)


if __name__ == "__main__":
    sys.exit(main())
