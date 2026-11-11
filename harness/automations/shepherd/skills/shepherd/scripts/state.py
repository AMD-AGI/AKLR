"""Local, durable round ownership and atomic snapshots for cooperating skill runs.

This is a storage helper, not a budget gate, a workflow engine, or a security boundary.
"""

import argparse
import fcntl
import json
import os
import secrets
import stat
import sys
import tempfile
from contextlib import contextmanager
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


def check_owner(root, token):
    value = read_json(root / "owner.json")
    if not isinstance(value, dict) or not secrets.compare_digest(str(value.get("token", "")), token):
        raise StateError("round token does not match the current owner")


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
            if info.st_mode & stat.S_IWOTH:
                raise StateError(f"{path} is world-writable")


def execute(args):
    # Config keeps `$USER`-style values so it never records a username or home path.
    root = Path(os.path.expandvars(args.root)).expanduser()
    if not root.is_absolute():
        raise StateError("--root must be an absolute persistent directory")
    check_private_path(root)
    if args.command == "init":
        # Exclusive directory creation also rejects a lost ledger in an existing directory.
        try:
            root.mkdir(mode=0o700, parents=False)
        except FileExistsError as exc:
            raise StateError("directory already exists; never initialize over existing state") from exc
        value = {
            "revision": 0,
            "state": {
                "campaigns": [],
                "committed": {"model_usd": 0, "node_hours": 0},
                "last_main_sha": None,
            },
        }
        atomic_json(root / "state.json", value)
        sync_directory(root.parent)
        return value

    with guarded(root):
        value = snapshot(root)
        owner_path = root / "owner.json"
        if args.command == "show":
            # Owner identity is observable, but don't print its token to every reader.
            owner = read_json(owner_path) if owner_path.exists() else None
            return {**value, "owner": owner.get("owner") if isinstance(owner, dict) else None}
        if args.command == "acquire":
            if owner_path.exists():
                raise StateError("busy: a round owns this ledger; no automatic takeover")
            token = secrets.token_hex(24)
            atomic_json(owner_path, {"token": token, "owner": args.owner})
            return {"token": token, **value}
        check_owner(root, args.token)
        if args.command == "release":
            owner_path.unlink()
            sync_directory(root)
            return {"released": True}
        if args.revision != value["revision"]:
            raise StateError(f"stale revision: expected {value['revision']}")
        state = read_json(Path(args.input))
        if not isinstance(state, dict):
            raise StateError("--input must contain a state object, not a list or scalar")
        value = {"revision": value["revision"] + 1, "state": state}
        atomic_json(root / "state.json", value)
        return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    sub.add_parser("show")
    acquire = sub.add_parser("acquire")
    acquire.add_argument("--owner", required=True)
    save = sub.add_parser("save")
    save.add_argument("--token", required=True)
    save.add_argument("--revision", required=True, type=int)
    save.add_argument("--input", required=True)
    release = sub.add_parser("release")
    release.add_argument("--token", required=True)
    args = parser.parse_args()
    try:
        result = execute(args)
    except (StateError, OSError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
