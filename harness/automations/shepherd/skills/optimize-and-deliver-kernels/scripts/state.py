"""Shepherd's round ledger, owned by one leased round at a time and saved as atomic, checked snapshots.

Every save is held to the configured caps (cumulative budget, GPUs in use, per-campaign model USD) and
to the campaign phase order. It is not a security boundary: the provider's spend limit and Slurm time
limits are the hard limits.
"""

import argparse
import fcntl
import hashlib
import json
import os
import pwd
import secrets
import stat
import sys
import tempfile
from contextlib import contextmanager
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


PHASES = (
    "admitted",
    "preparing",
    "optimizing",
    "finalizing",
    "validating",
    "cleaning",
    "ready_to_deliver",
    "pr_open",
)
# Phases whose campaign may still have containers on its GPU.
HOLDS_GPU = PHASES[: PHASES.index("ready_to_deliver")]
# A campaign ends with nothing on its GPU: before any container, after confirmed cleanup, or after delivery.
ENDS_FROM = ("admitted", "cleaning", "ready_to_deliver", "pr_open")


def load_config(path):
    try:
        import yaml
    except ImportError as exc:
        raise StateError("PyYAML is needed to read the configuration") from exc
    try:
        value = yaml.safe_load(Path(path).read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise StateError(f"cannot read configuration {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise StateError(f"configuration {path} is not a mapping")
    return value


@dataclass(frozen=True)
class Round:
    """One round's lease on the ledger.

    The lease is a fencing token, not a liveness check: once it expires, the round's saves are refused,
    so a round that overran cannot record (and so, by the save-intent-first rule, cannot start) any new
    side effect, and the next trigger takes over and reconciles its saved intents.
    """

    id: str
    started: str
    lease_until: str
    config_path: str
    config_sha256: str
    # The round's exit code once it ended; None while it runs, or when it never called `end`.
    exit_code: int | None = None

    @classmethod
    def begin(cls, config_path, now):
        config = load_config(config_path)
        hours = (config.get("execution") or {}).get("round_hours", 2)
        if not isinstance(hours, (int, float)) or not 0 < hours <= 24:
            raise StateError("execution.round_hours must be a number of hours in (0, 24]")
        return cls(
            id=secrets.token_hex(12),
            started=now.isoformat(),
            lease_until=(now + timedelta(hours=hours)).isoformat(),
            config_path=str(Path(config_path).resolve()),
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

    def holds_lease(self, now):
        return now < datetime.fromisoformat(self.lease_until)


class Ledger:
    """Rules every saved state must keep, checked against the previous state and the configuration."""

    def __init__(self, previous, config):
        self.previous = previous
        self.config = config

    @staticmethod
    def entries(state, kind):
        entries = {}
        for entry in state.get(kind, []):
            if not isinstance(entry, dict) or not isinstance(entry.get("id"), str) or entry["id"] in entries:
                raise StateError(f"every entry of {kind} needs a unique string id")
            entries[entry["id"]] = entry
        return entries

    @staticmethod
    def committed(entries, settled, bound):
        """Settled use where settled, else the admitted bound."""
        return sum(entry.get(settled, entry["bounds"][bound]) for entry in entries.values())

    def check(self, state):
        campaigns, pools = self.entries(state, "campaigns"), self.entries(state, "pools")
        kept = {
            "campaigns": self.entries(self.previous, "campaigns"),
            "pools": self.entries(self.previous, "pools"),
        }
        # The committed budget is derived from the campaigns (model spend) and the pools (shared node
        # allocations, node hours), so a save may never drop one or change the bounds it was admitted with.
        for kind, now, bound in (("campaigns", campaigns, "model_usd"), ("pools", pools, "node_hours")):
            for entry in now.values():
                value = (entry.get("bounds") or {}).get(bound)
                if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
                    raise StateError(
                        f"{kind} entry {entry['id']} needs bounds.{bound}, a non-negative number"
                    )
            for id_, old in kept[kind].items():
                if id_ not in now:
                    raise StateError(f"{kind} entry {id_} is missing; entries are never removed")
                if now[id_]["bounds"] != old["bounds"]:
                    raise StateError(f"{kind} entry {id_} changes its admitted bounds")
        for campaign in campaigns.values():
            self.check_campaign(campaign, kept["campaigns"].get(campaign["id"]))
        self.check_gpus(campaigns)
        self.check_budget(campaigns, pools, kept)

    def check_campaign(self, campaign, old):
        id_, phase = campaign["id"], campaign.get("phase")
        if phase not in (*PHASES, "ended"):
            raise StateError(f"campaign {id_} has unknown phase {phase!r}")
        settled = campaign.get("settled_model_usd")
        if settled is not None and settled not in (0, campaign["bounds"]["model_usd"]):
            raise StateError(f"campaign {id_} settles at 0 or its admitted bound, never in between")
        if campaign["bounds"]["model_usd"] > self.config["campaign"]["max_model_usd"]:
            raise StateError(f"campaign {id_} bounds.model_usd exceeds campaign.max_model_usd")
        if old is None:
            if phase != "admitted":
                raise StateError(f"campaign {id_} must first be saved as admitted")
            return
        before = old.get("phase")
        if before == "ended" and phase != "ended":
            raise StateError(f"campaign {id_} has ended and cannot move again")
        if phase == "ended" and before != "ended" and before not in ENDS_FROM:
            raise StateError(f"campaign {id_} cannot end from {before}: clean it up first")
        if phase in PHASES and before in PHASES and PHASES.index(phase) < PHASES.index(before):
            raise StateError(f"campaign {id_} cannot move back from {before} to {phase}")

    def check_gpus(self, campaigns):
        holding = [c for c in campaigns.values() if c["phase"] in HOLDS_GPU]
        if len(holding) > self.config["policy"]["max_parallel_campaigns"]:
            raise StateError("more campaigns hold a GPU than policy.max_parallel_campaigns")
        slots = [(c.get("pool"), c.get("gpu")) for c in holding]
        if None in (s for slot in slots for s in slot):
            raise StateError("every campaign that holds a GPU records its pool and gpu")
        if len(set(slots)) != len(slots):
            raise StateError("two campaigns hold the same GPU")

    def check_budget(self, campaigns, pools, kept):
        # Only a save that raises the commitment is held to the caps, so lowering a cap blocks new
        # admissions without blocking the rounds that settle what is already committed.
        budget = self.config["budget"]
        for kind, now, settled, bound, cap in (
            ("campaigns", campaigns, "settled_model_usd", "model_usd", "total_model_usd"),
            ("pools", pools, "settled_node_hours", "node_hours", "total_node_hours"),
        ):
            total = self.committed(now, settled, bound)
            if total > self.committed(kept[kind], settled, bound) and total > budget[cap]:
                raise StateError(f"committed {bound} {total} would exceed budget.{cap} {budget[cap]}")


class StateDir:
    """The ledger's directory: its snapshot, the round that owns it, and the commands that change them."""

    def __init__(self, root):
        # Config keeps `$USER`-style values so it never records a username or home path; a scheduler's
        # shell may start without USER.
        os.environ.setdefault("USER", pwd.getpwuid(os.getuid()).pw_name)
        self.root = Path(os.path.expandvars(root)).expanduser()
        if not self.root.is_absolute():
            raise StateError("--root must be an absolute persistent directory")
        self.check_private()
        self.owner_path = self.root / "owner.json"
        self.last_path = self.root / "last_round.json"

    def check_private(self):
        """Refuse a path another user could pre-create or swap, e.g. `/var/tmp/$USER` made by someone else."""
        for chain in (self.root, self.root.resolve()):
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

    @contextmanager
    def guarded(self):
        """The snapshot and its owner under the ledger's lock."""
        # Never create a missing ledger as a side effect of a normal operation.
        if not self.root.is_dir():
            raise StateError("state directory missing; investigate or initialize during setup")
        with (self.root / ".guard").open("a") as guard:
            fcntl.flock(guard, fcntl.LOCK_EX)
            value = read_json(self.root / "state.json")
            if (
                not isinstance(value, dict)
                or type(value.get("revision")) is not int
                or value["revision"] < 0
                or not isinstance(value.get("state"), dict)
            ):
                raise StateError("invalid state envelope; restore and reconcile, do not reinitialize")
            yield value, Round.recorded(self.owner_path)

    def init(self):
        # A fresh host has no `/var/tmp/$USER/shepherd` yet. Every directory made here is private,
        # whatever the umask; exclusive creation of the root also rejects a lost ledger.
        for directory in [*reversed([p for p in self.root.parents if not p.exists()]), self.root]:
            try:
                directory.mkdir(mode=0o700)
            except FileExistsError as exc:
                raise StateError("directory already exists; never initialize over existing state") from exc
            directory.chmod(0o700)
        value = {"revision": 0, "state": {"campaigns": [], "last_main_sha": None}}
        atomic_json(self.root / "state.json", value)
        sync_directory(self.root.parent)
        return value

    def show(self):
        with self.guarded() as (value, owner):
            return {**value, "owner": asdict(owner) if owner else None}

    def begin(self, config_path, now, force):
        with self.guarded() as (value, owner):
            if owner and owner.holds_lease(now):
                with (self.root / "reports.md").open("a") as reports:
                    reports.write(
                        f"\n{now.isoformat()}: round {owner.id} is still running; trigger skipped.\n"
                    )
                return {"busy": owner.id}
            here = Round.begin(config_path, now)
            # A round that overran its lease, failed or never ended is recovered at once: its saved intents
            # need reconciling, whatever wake_at says.
            last = Round.recorded(self.last_path)
            if owner is None and last is not None and last.exit_code != 0:
                owner = last
            if owner is None and not force and last is not None and last.config_sha256 == here.config_sha256:
                wake_at = value["state"].get("wake_at")
                try:
                    wake = datetime.fromisoformat(wake_at) if isinstance(wake_at, str) else None
                except ValueError:
                    wake = None
                # A wrong wake_at can delay a round by at most two hours.
                if wake is not None and wake.tzinfo is not None and now < wake <= now + timedelta(hours=2):
                    return {"waiting_until": wake_at}
            if owner is not None and owner.exit_code is None:
                # Its lease expired without `end`: keep its record for the report, as `end` would have.
                atomic_json(self.last_path, asdict(owner))
            atomic_json(self.owner_path, asdict(here))
            return {
                "round": here.id,
                "lease_until": here.lease_until,
                "recovered_from": owner.id if owner else None,
            }

    def save(self, round_id, revision, input_path, now):
        with self.guarded() as (value, owner):
            self.check_owner(owner, round_id)
            if not owner.holds_lease(now):
                raise StateError("this round's lease expired; record nothing more and end the round")
            if revision != value["revision"]:
                raise StateError(f"stale revision: expected {value['revision']}")
            state = read_json(Path(input_path))
            if not isinstance(state, dict):
                raise StateError("--input must contain a state object, not a list or scalar")
            Ledger(value["state"], load_config(owner.config_path)).check(state)
            value = {"revision": value["revision"] + 1, "state": state}
            atomic_json(self.root / "state.json", value)
            return value

    def end(self, round_id, exit_code):
        with self.guarded() as (_, owner):
            self.check_owner(owner, round_id)
            atomic_json(self.last_path, asdict(replace(owner, exit_code=exit_code)))
            self.owner_path.unlink()
            sync_directory(self.root)
            return {"ended": owner.id, "exit_code": exit_code}

    @staticmethod
    def check_owner(owner, round_id):
        if owner is None or owner.id != round_id:
            raise StateError("only the round named in owner.json may save or end; a later round took over")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    sub.add_parser("show")
    begin_ = sub.add_parser("begin", help="take the ledger for one round, or report busy / waiting_until")
    begin_.add_argument("--config", required=True)
    begin_.add_argument("--now", action="store_true", help="run even if the ledger's wake_at is ahead")
    save = sub.add_parser("save")
    save.add_argument("--round", required=True)
    save.add_argument("--revision", required=True, type=int)
    save.add_argument("--input", required=True)
    end = sub.add_parser("end", help="release the ledger; exit code 0 only when the round finished cleanly")
    end.add_argument("--round", required=True)
    end.add_argument("--exit-code", required=True, type=int)
    args = parser.parse_args()
    now = datetime.now(timezone.utc)
    try:
        directory = StateDir(args.root)
        if args.command == "init":
            result = directory.init()
        elif args.command == "show":
            result = directory.show()
        elif args.command == "begin":
            result = directory.begin(args.config, now, args.now)
        elif args.command == "save":
            result = directory.save(args.round, args.revision, args.input, now)
        else:
            result = directory.end(args.round, args.exit_code)
    except (StateError, OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 1
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
