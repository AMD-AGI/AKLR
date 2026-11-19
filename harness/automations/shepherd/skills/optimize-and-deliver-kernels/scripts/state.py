"""Shepherd's round ledger, owned by one leased round at a time and changed by atomic, checked patches.

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
import re
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
# Version of the ledger's layout. `begin` migrates an older ledger; a newer one is refused.
SCHEMA = 2
# What stays in the ledger once an entry is finished (a campaign ended, a pool settled): what deduplication,
# cooldown and the committed budget read. The whole entry moves to `runs/<id>/`.
KEPT = {
    "campaigns": (
        "id",
        "phase",
        "outcome",
        "bounds",
        "settled_model_usd",
        "observed_model_usd",
        "schema_path",
        "input_sha256",
        "regime",
        "contract",
        "backend",
        "branch",
        "delivered_input",
        "cooldown_until",
    ),
    "pools": ("id", "bounds", "settled_node_hours"),
}


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
    # What `observation.py` printed before the round began: the only value it may save as `idle_observation`.
    observation: str | None = None

    @classmethod
    def begin(cls, config_path, now, observation):
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
            observation=observation or None,
        )

    @classmethod
    def recorded(cls, path):
        if not path.exists():
            return None
        value = read_json(path)
        try:
            return cls(**value)
        except TypeError as exc:
            # Schema 1 identified a round by its process. Its record carries over with a lease long expired:
            # the helper that could tell whether it still runs is gone, so the next round reconciles it.
            if isinstance(value, dict) and {"id", "pid", "config_sha256"} <= value.keys():
                epoch = datetime.fromtimestamp(0, timezone.utc).isoformat()
                return cls(value["id"], epoch, epoch, "", value["config_sha256"], value.get("exit_code"))
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
            # The id names the entry's directory under `runs/` and its `remote-gpu bg --name` tasks.
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", entry["id"]):
                raise StateError(f"{kind} id {entry['id']!r} must be letters, digits, '_' or '-'")
            entries[entry["id"]] = entry
        return entries

    def cap(self, section, key):
        value = (self.config.get(section) or {}).get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise StateError(f"configuration {section}.{key} is not set: nothing can be admitted")
        return value

    @staticmethod
    def committed(entries, settled, bound):
        """Settled use where settled, else the admitted bound."""
        return sum(entry.get(settled, entry["bounds"][bound]) for entry in entries.values())

    @classmethod
    def merged(cls, value, patch):
        """JSON merge patch (RFC 7386): a mapping merges key by key, null removes the key, the rest replaces."""
        if not isinstance(patch, dict):
            return patch
        value = dict(value) if isinstance(value, dict) else {}
        for key, item in patch.items():
            if item is None:
                value.pop(key, None)
            else:
                value[key] = cls.merged(value.get(key), item)
        return value

    def patched(self, patch):
        """The previous state with the patch applied; `campaigns` and `pools` are patched by entry id."""
        if not isinstance(patch, dict):
            raise StateError("--patch must contain an object")
        kinds = {}
        for kind in KEPT:
            entries, changes = self.entries(self.previous, kind), patch.get(kind, {})
            if not isinstance(changes, dict):
                raise StateError(f"{kind} in a patch maps each entry's id to its changes")
            for id_, change in changes.items():
                if not isinstance(change, dict):
                    raise StateError(f"{kind} entry {id_} is not an object; entries are never removed")
                entries[id_] = {**self.merged(entries.get(id_), change), "id": id_}
            kinds[kind] = list(entries.values())
        # When to wake, and an idle round's skip, hold only until the next change: every save drops them
        # unless it sets them, so a round that sets no `wake_at` leaves none behind.
        dropped = (*KEPT, "wake_at", "idle_observation")
        rest = {key: item for key, item in self.previous.items() if key not in dropped}
        return {**self.merged(rest, {key: item for key, item in patch.items() if key not in KEPT}), **kinds}

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
            for entry in now.values():
                old = kept[kind].get(entry["id"]) or {}
                self.check_operations(kind, entry, old)
                # Only the helper archives an entry; a patch that changed the path would overwrite the archive.
                if entry.get("archive") != old.get("archive"):
                    raise StateError(f"{kind} entry {entry['id']}: archive is set by the helper alone")
        for campaign in campaigns.values():
            self.check_campaign(campaign, kept["campaigns"].get(campaign["id"]))
        for pool in pools.values():
            self.check_pool(pool, kept["pools"].get(pool["id"]), campaigns)
        self.check_gpus(campaigns, pools, kept["campaigns"])
        self.check_budget(campaigns, pools, kept)
        if "idle_observation" in state and not (
            isinstance(state["idle_observation"], str)
            and all(c["phase"] == "ended" for c in campaigns.values())
            and all("settled_node_hours" in p for p in pools.values())
        ):
            raise StateError(
                "idle_observation is a string, set only once every campaign ended and every pool settled"
            )

    @staticmethod
    def check_operations(kind, entry, old):
        """An operation's intent is saved before its result, in an earlier save, and is never removed."""
        operations, before = entry.get("operations", {}), old.get("operations", {})
        if not isinstance(operations, dict) or not all(isinstance(op, dict) for op in operations.values()):
            raise StateError(f"{kind} entry {entry['id']}: operations maps each operation's id to its record")
        for id_ in before:
            if id_ not in operations:
                raise StateError(
                    f"{kind} entry {entry['id']} drops operation {id_}; operations are never removed"
                )
        for id_, operation in operations.items():
            if id_ not in before and "result" in operation:
                raise StateError(
                    f"{kind} entry {entry['id']} saves operation {id_} with a result but no earlier intent: "
                    "save the intent, act, then save the result"
                )

    def check_campaign(self, campaign, old):
        id_, phase = campaign["id"], campaign.get("phase")
        if phase not in (*PHASES, "ended"):
            raise StateError(f"campaign {id_} has unknown phase {phase!r}")
        settled, bound = campaign.get("settled_model_usd"), campaign["bounds"]["model_usd"]
        holds = phase in HOLDS_GPU
        # Until it is settled a campaign counts at its bound, so it is settled in the one save that takes it
        # off its GPU, and never while it could still spend.
        if holds == (settled is not None):
            raise StateError(
                f"campaign {id_} is settled exactly when it leaves its GPU: in the save that moves it past "
                "cleaning or ends it, and not before"
            )
        if settled is not None and (isinstance(settled, bool) or settled not in (0, bound)):
            raise StateError(f"campaign {id_} settles at 0 or its admitted bound, never in between")
        if old is None:
            if phase != "admitted":
                raise StateError(f"campaign {id_} must first be saved as admitted")
            if self.config.get("mode") == "drain":
                raise StateError(f"campaign {id_}: mode drain admits no campaign")
            # Held to the cap at admission only, so lowering the cap never blocks the saves of earlier campaigns.
            if bound > self.cap("campaign", "max_model_usd"):
                raise StateError(f"campaign {id_} bounds.model_usd exceeds campaign.max_model_usd")
            return
        before = old.get("phase")
        if "settled_model_usd" in old and settled != old["settled_model_usd"]:
            raise StateError(f"campaign {id_} is settled once; its settled_model_usd cannot change")
        if before == "ended" and phase != "ended":
            raise StateError(f"campaign {id_} has ended and cannot move again")
        # A campaign leaves its GPU after confirmed cleanup, or by ending before any container started.
        if (
            before in HOLDS_GPU
            and not holds
            and before != "cleaning"
            and (before, phase) != ("admitted", "ended")
        ):
            raise StateError(f"campaign {id_} cannot leave its GPU from {before}: clean it up first")
        if phase in PHASES and before in PHASES and PHASES.index(phase) < PHASES.index(before):
            raise StateError(f"campaign {id_} cannot move back from {before} to {phase}")

    @staticmethod
    def check_pool(pool, old, campaigns):
        id_, settled = pool["id"], pool.get("settled_node_hours")
        # Before anything else: removing the amount would let a later save settle the pool again, lower.
        if old is not None and "settled_node_hours" in old and settled != old["settled_node_hours"]:
            raise StateError(f"pool {id_} is settled once; its settled_node_hours cannot change")
        if "settled_node_hours" not in pool:
            return
        if not isinstance(settled, (int, float)) or isinstance(settled, bool) or settled < 0:
            raise StateError(f"pool {id_} settled_node_hours must be a non-negative number")
        # A settled pool leaves the ledger for its archive, so it is settled after its release, never as created.
        if old is None:
            raise StateError(f"pool {id_} is settled in a later save than the one that creates it")
        if any(c["phase"] in HOLDS_GPU and c.get("pool") == id_ for c in campaigns.values()):
            raise StateError(f"pool {id_} cannot settle while a campaign holds one of its GPUs")

    def check_gpus(self, campaigns, pools, kept):
        holding = [c for c in campaigns.values() if c["phase"] in HOLDS_GPU]
        # Only a save that adds a holder is held to the cap, so lowering it never blocks cleanup.
        added = [c for c in holding if (kept.get(c["id"]) or {}).get("phase") not in HOLDS_GPU]
        if added and len(holding) > self.cap("policy", "max_parallel_campaigns"):
            raise StateError("more campaigns hold a GPU than policy.max_parallel_campaigns")
        slots = [(c.get("pool"), c.get("gpu")) for c in holding]
        if None in (s for slot in slots for s in slot):
            raise StateError("every campaign that holds a GPU records its pool and gpu")
        # An index is the same card as a number or as a string.
        if len({(pool, str(gpu)) for pool, gpu in slots}) != len(slots):
            raise StateError("two campaigns hold the same GPU")
        # A settled pool has left the ledger for its archive, `end` included.
        for campaign in added:
            try:
                end = datetime.fromisoformat(pools[campaign["pool"]]["end"])
                fits = datetime.fromisoformat(campaign["bounds"]["deadline"]) < end
            except (KeyError, TypeError, ValueError):
                fits = False
            if not fits:
                raise StateError(
                    f"campaign {campaign['id']} needs bounds.deadline before the end of a started, "
                    "unsettled pool"
                )

    def check_budget(self, campaigns, pools, kept):
        # Only a save that adds an entry is held to the caps: bounds never change and an entry is settled
        # once, so nothing else raises the commitment but settling a pool at what it really used, and
        # neither that nor a lowered cap may block the rounds that settle what is already committed.
        for kind, now, settled, bound, cap in (
            ("campaigns", campaigns, "settled_model_usd", "model_usd", "total_model_usd"),
            ("pools", pools, "settled_node_hours", "node_hours", "total_node_hours"),
        ):
            total = self.committed(now, settled, bound)
            if now.keys() - kept[kind].keys() and total > self.cap("budget", cap):
                raise StateError(f"committed {bound} {total} would exceed budget.{cap}")


class StateDir:
    """The ledger's directory: its snapshot, the round that owns it, and the commands that change them."""

    def __init__(self, root):
        # Config keeps `$USER`-style values so it never records a username or home path; a scheduler's
        # shell may start without USER.
        if "USER" not in os.environ:  # getpwuid fails for a uid without a passwd entry: ask only when needed
            os.environ["USER"] = pwd.getpwuid(os.getuid()).pw_name
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
            if value.get("schema", 1) > SCHEMA:
                raise StateError("the ledger was written by a newer state helper; update this checkout")
            yield value, Round.recorded(self.owner_path)

    def init(self):
        # A fresh host has no `/var/tmp/$USER/shepherd` yet. Every directory made here is private,
        # whatever the umask.
        for directory in reversed([p for p in (self.root, *self.root.parents) if not p.exists()]):
            directory.mkdir(mode=0o700)
            directory.chmod(0o700)
        # A first setup often finds the directory made ahead; `.guard` is left by any earlier command and
        # an empty `runs/` holds nothing. Anything else is a ledger or its remains.
        held = sorted(
            path.name
            for path in self.root.iterdir()
            if path.name != ".guard" and not (path.name == "runs" and not any(path.iterdir()))
        )
        if held:
            raise StateError(f"directory holds {', '.join(held)}; never initialize over existing state")
        value = {"schema": SCHEMA, "revision": 0, "state": {"campaigns": [], "last_main_sha": None}}
        atomic_json(self.root / "state.json", value)
        sync_directory(self.root.parent)
        return value

    def show(self):
        with self.guarded() as (value, owner):
            return {**value, "owner": asdict(owner) if owner else None}

    def archived(self, state):
        """The state with each finished entry written whole under `runs/<id>/` and cut down to `KEPT`."""
        finished = {
            "campaigns": lambda c: c["phase"] == "ended",
            "pools": lambda p: "settled_node_hours" in p,
        }
        state = dict(state)
        for kind, keys in KEPT.items():
            entries = []
            for entry in state.get(kind, []):
                if finished[kind](entry) and "archive" not in entry:
                    archive = Path("runs", entry["id"], f"{kind[:-1]}.json")
                    (self.root / archive).parent.mkdir(parents=True, exist_ok=True)
                    atomic_json(self.root / archive, entry)
                    sync_directory(self.root / "runs")
                    entry = {**{key: entry[key] for key in keys if key in entry}, "archive": str(archive)}
                entries.append(entry)
            state[kind] = entries
        return state

    def migrated(self, value, config):
        """The envelope at the current `SCHEMA`, saved, with the one it replaces kept under `schema-<old>/`."""
        old = value.get("schema", 1)
        if old == SCHEMA:
            return value
        state = dict(value["state"])
        for kind in KEPT:
            entries = []
            for entry in state.get(kind, []):
                entry = dict(entry)
                if isinstance(entry.get("operations"), list):
                    operations = {}
                    for index, op in enumerate(entry["operations"]):
                        id_ = str(op.get("id", index)) if isinstance(op, dict) else str(index)
                        # Schema 1 did not keep ids unique; a repeated one keeps its record under id#index.
                        operations[f"{id_}#{index}" if id_ in operations else id_] = op
                    entry["operations"] = operations
                settled = entry.get("settled")
                if kind == "campaigns" and isinstance(settled, dict) and "model_usd" in settled:
                    entry.setdefault("settled_model_usd", settled["model_usd"])
                    # Schema 1 named a finished campaign's outcome as its phase. Only a settled one is
                    # finished: any other unknown phase stays, and the check below refuses the ledger.
                    if entry.get("phase") not in (*PHASES, "ended"):
                        if not isinstance(entry.get("outcome"), str):
                            if "outcome" in entry:
                                entry["outcome_detail"] = entry["outcome"]
                            entry["outcome"] = entry.get("phase")
                        entry["phase"] = "ended"
                entries.append(entry)
            state[kind] = entries
        # Nothing is written unless the migrated ledger keeps every rule.
        Ledger(state, config).check(state)
        backup = self.root / f"schema-{old}"
        backup.mkdir(exist_ok=True)
        atomic_json(backup / "state.json", value)
        sync_directory(self.root)
        value = {"schema": SCHEMA, "revision": value["revision"] + 1, "state": self.archived(state)}
        atomic_json(self.root / "state.json", value)
        return value

    def begin(self, config_path, now, force, observation):
        with self.guarded() as (value, owner):
            last = Round.recorded(self.last_path)
            # `end` records the round, then releases the ledger; a crash between the two leaves a round
            # that did end, with its exit code, as the owner.
            if owner is not None and last is not None and last.id == owner.id:
                owner = None
            if owner and owner.holds_lease(now):
                return {"busy": owner.id}
            # After the busy check: a running round's next save needs the revision it read.
            value = self.migrated(value, load_config(config_path))
            here = Round.begin(config_path, now, observation)
            # A round that overran its lease, failed or never ended is recovered at once: its saved intents
            # need reconciling, whatever wake_at says.
            if owner is None and last is not None and last.exit_code != 0:
                owner = last
            if owner is None and not force and last is not None and last.config_sha256 == here.config_sha256:
                wake_at = value["state"].get("wake_at")
                try:
                    wake = datetime.fromisoformat(wake_at) if isinstance(wake_at, str) else None
                except ValueError:
                    wake = None
                # A wrong wake_at can delay a round by at most two hours; a day when the last round was idle
                # and what it waits on (the observation it saved) has not changed.
                idle = observation and observation == value["state"].get("idle_observation")
                limit = self.wake_limit(now, idle)
                if wake is not None and wake.tzinfo is not None and now < wake <= limit:
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

    def save(self, round_id, revision, patch_path, now):
        with self.guarded() as (value, owner):
            self.check_owner(owner, round_id)
            if not owner.holds_lease(now):
                raise StateError("this round's lease expired; record nothing more and end the round")
            if revision != value["revision"]:
                raise StateError(f"stale revision: expected {value['revision']}")
            if value.get("schema", 1) != SCHEMA:
                raise StateError("the ledger has an older layout; only the next begin migrates it")
            config = load_config(owner.config_path)
            if config.get("mode") not in ("run", "drain"):
                raise StateError("only a round in mode run or drain saves; observe reports its proposals")
            ledger = Ledger(value["state"], config)
            state = ledger.patched(read_json(Path(patch_path)))
            ledger.check(state)
            # The round looked at GitHub after it began, so only what it began with is known to be covered.
            if state.get("idle_observation") not in (None, owner.observation):
                raise StateError("idle_observation must be the observation this round began with")
            # `begin` ignores a wake_at beyond its limit, so a longer wait is kept as the limit: the round
            # that wakes then sets it again.
            limit = self.wake_limit(now, "idle_observation" in state)
            try:
                if datetime.fromisoformat(state["wake_at"]) > limit:
                    state["wake_at"] = limit.isoformat()
            except (KeyError, TypeError, ValueError):
                pass  # None set, or one `begin` will not read.
            value = {"schema": SCHEMA, "revision": value["revision"] + 1, "state": self.archived(state)}
            atomic_json(self.root / "state.json", value)
            return {"revision": value["revision"]}

    def end(self, round_id, exit_code):
        with self.guarded() as (_, owner):
            self.check_owner(owner, round_id)
            atomic_json(self.last_path, asdict(replace(owner, exit_code=exit_code)))
            self.owner_path.unlink()
            sync_directory(self.root)
            return {"ended": owner.id, "exit_code": exit_code}

    @staticmethod
    def wake_limit(now, idle):
        """The latest wake_at `begin` waits for."""
        return now + timedelta(hours=24 if idle else 2)

    @staticmethod
    def check_owner(owner, round_id):
        if owner is None or owner.id != round_id:
            raise StateError("only the round named in owner.json may save or end; a later round took over")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", help="the state directory; begin takes execution.state_dir without it")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    sub.add_parser("show")
    begin_ = sub.add_parser("begin", help="take the ledger for one round, or report busy / waiting_until")
    begin_.add_argument("--config", required=True)
    begin_.add_argument("--now", action="store_true", help="run even if the ledger's wake_at is ahead")
    begin_.add_argument("--observation", help="fingerprint printed by observation.py, or empty")
    save = sub.add_parser("save")
    save.add_argument("--round", required=True)
    save.add_argument("--revision", required=True, type=int)
    save.add_argument("--patch", required=True, help="file holding a JSON merge patch of the state")
    end = sub.add_parser("end", help="release the ledger; exit code 0 only when the round finished cleanly")
    end.add_argument("--round", required=True)
    end.add_argument("--exit-code", required=True, type=int)
    args = parser.parse_args()
    now = datetime.now(timezone.utc)
    try:
        root = args.root
        if root is None:
            if args.command != "begin":
                raise StateError("--root is required")
            root = load_config(args.config)["execution"]["state_dir"]
        directory = StateDir(root)
        if args.command == "init":
            result = directory.init()
        elif args.command == "show":
            result = directory.show()
        elif args.command == "begin":
            result = directory.begin(args.config, now, args.now, args.observation)
        elif args.command == "save":
            result = directory.save(args.round, args.revision, args.patch, now)
        else:
            result = directory.end(args.round, args.exit_code)
    except (StateError, OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 1
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
