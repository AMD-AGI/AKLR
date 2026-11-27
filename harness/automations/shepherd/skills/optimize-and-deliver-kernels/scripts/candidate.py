"""A candidate schema of a shepherd round, at `<op_dir>/impls/<language>/<name>/<file>.json`.

Run at the repository root.

`candidate.py list --config .shepherd/configuration.yaml --commit <main SHA> --ledger <file>` prints the
candidates at that commit, by strategy order with the configuration's filters, as {path, op, backend,
input_sha256, strategy, contract_key} sorted by path, then the skipped ones with their reason. The ledger
(a file holding `state.py show`'s output) skips what a campaign still works on, delivered, cools down or
found blocked; when that campaign had a regime, only that regime is closed, listed in the candidate's
`skip_regimes`. Competing PRs, the AKLR-baseline check and the GEMM regime split stay with the round.
Exit 2: error.

`candidate.py freshness --schema <path> --base <validation base> --main <main>` prints the changes between
the evidence base and main that can alter the candidate's verdict. Exit 0: fresh; 1: stale (relevant
changes, or the base is no longer an ancestor of main); 2: error, treat as unknown.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import PurePosixPath

OPS = PurePosixPath("aklr/plugins/aiter/provider")


def git(*args, stdin=None, check=True):
    return subprocess.run(["git", *args], input=stdin, capture_output=True, check=check)


@dataclass(frozen=True)
class Candidate:
    path: PurePosixPath

    @staticmethod
    def workload_key(workload):
        """What a workload is, without what was measured on it (accuracy, speedup)."""
        return json.dumps({k: workload.get(k) for k in ("axes", "scalars")}, sort_keys=True)

    @classmethod
    def of(cls, schema_path):
        path = PurePosixPath(schema_path)
        if len(path.parts) < 5 or path.parts[-4] != "impls" or path.suffix != ".json":
            raise ValueError(f"{schema_path} is not <op_dir>/impls/<language>/<name>/<file>.json")
        return cls(path)

    @property
    def op_dir(self):
        return self.path.parents[3]

    @property
    def name(self):
        return self.path.parts[-2]

    @property
    def op(self):
        return self.op_dir.name if self.op_dir.parent == OPS else None

    @classmethod
    def listed(cls, commit, config, state):
        """The policy's candidates at the commit and the skipped ones, each with its strategy."""
        policy = config["policy"]
        allowed, backends = policy["allowed_ops"], policy["backends"]
        # What lifts a blocked contract when it changes: the op's code (see `contract_key` below), this
        # checkout's forge_task and the runtime.
        forge_task = git("rev-parse", "HEAD:harness/scripts/forge_task", check=False).stdout.decode().strip()
        runtime = (config.get("execution") or {}).get("runtime_revision")
        now, op_trees = datetime.now(timezone.utc), {}
        listed = {}  # path -> strategy; manual wins wherever it stands, as it names the schema.
        for strategy in policy["strategies"]:
            if strategy["type"] == "manual":
                listed.update(dict.fromkeys(strategy.get("schemas") or [], "manual"))
            elif strategy["type"] == "kernel-less":
                tree = git("ls-tree", "-r", "--name-only", commit, "--", str(OPS)).stdout.decode()
                for path in tree.splitlines():
                    try:
                        cls.of(path)
                    except ValueError:
                        continue
                    listed.setdefault(path, "kernel-less")
            else:
                raise ValueError(f"unknown strategy {strategy['type']!r}")
        paths = sorted(listed)
        # Contents of each path at the commit, None when it is missing there.
        batch = git("cat-file", "--batch", stdin="".join(f"{commit}:{p}\n" for p in paths).encode()).stdout
        blobs, at = {}, 0
        for path in paths:
            header_end = batch.index(b"\n", at)
            header = batch[at:header_end].split()
            if header[-1] == b"missing":
                blobs[path], at = None, header_end + 1
                continue
            size = int(header[2])
            blobs[path] = batch[header_end + 1 : header_end + 1 + size]
            at = header_end + 1 + size + 1
        # The workloads a kernel already serves, by schema directory (one name and language).
        served = {}
        for path, blob in blobs.items():
            try:
                schema = json.loads(blob)
                if schema["kernel"]["sources"]:
                    served.setdefault(PurePosixPath(path).parent, set()).update(
                        cls.workload_key(w) for w in schema["workloads"]
                    )
            except (ValueError, TypeError, KeyError, AttributeError):
                continue  # No kernel, or unreadable: it serves nothing.
        candidates, skipped = [], []
        for path, strategy in listed.items():
            entry, blob = {"path": path, "strategy": strategy}, blobs[path]
            if blob is None:
                skipped.append({**entry, "reason": "missing at this commit"})
                continue
            try:
                candidate = cls.of(path)
            except ValueError as exc:
                skipped.append({**entry, "reason": str(exc)})
                continue
            try:
                kernel = json.loads(blob).get("kernel") or {}
                backend = kernel.get("language") or candidate.path.parts[-3]
            except (ValueError, AttributeError):
                # One unreadable file on main must not hide every other candidate.
                skipped.append({**entry, "reason": "not a schema: a JSON object whose kernel is an object"})
                continue
            entry.update(op=candidate.op, backend=backend, input_sha256=hashlib.sha256(blob).hexdigest())
            if strategy == "kernel-less":
                workloads = {cls.workload_key(w) for w in json.loads(blob).get("workloads") or []}
                # Has a kernel, or every workload is served by a kernel of a sibling schema: not
                # kernel-less, nothing to report. One unserved workload keeps the whole schema.
                if kernel.get("sources") or (
                    workloads and workloads <= served.get(candidate.path.parent, set())
                ):
                    continue
            if "all" not in allowed and candidate.op not in allowed:
                skipped.append({**entry, "reason": "op not in policy.allowed_ops"})
                continue
            if backend not in backends:
                skipped.append({**entry, "reason": "language not in policy.backends"})
                continue
            op_dir = candidate.op_dir
            if op_dir not in op_trees:
                # The op directory without `impls/`, whose measurement updates are expected.
                tree = git("ls-tree", commit, f"{op_dir}/").stdout.splitlines(keepends=True)
                kept = b"".join(
                    line.replace(f"{op_dir}/".encode(), b"")
                    for line in tree
                    if not line.endswith(b"/impls\n")
                )
                op_trees[op_dir] = git("mktree", stdin=kept).stdout.decode().strip()
            key = {
                "op_dir_tree_without_impls": op_trees[op_dir],
                "forge_task_tree": forge_task,
                "runtime_revision": runtime,
            }
            entry["contract_key"] = key
            reason, regimes = None, []
            for block in state.get("blocked_contracts", []):
                lifts = block.get("lifts_when_changed") or {}
                if block.get("op") == candidate.op and all(lifts.get(name) == key[name] for name in key):
                    reason = f"blocked contract {block.get('contract')}"
            for campaign in state["campaigns"]:
                try:
                    earlier = cls.of(campaign.get("schema_path", ""))
                except ValueError:
                    continue
                if earlier.path != candidate.path:
                    continue
                if campaign.get("phase") != "ended":
                    # Any regime: its merge would make another regime's evidence stale.
                    reason = f"campaign {campaign['id']} is undelivered"
                    continue
                cooldown = campaign.get("cooldown_until")
                try:
                    # One that cannot be read keeps cooling down.
                    cooling = cooldown is not None and not datetime.fromisoformat(cooldown) <= now
                except (TypeError, ValueError):
                    cooling = True
                delivered = campaign.get("outcome") in ("delivered", "merged")
                if delivered and campaign.get("input_sha256") == entry["input_sha256"]:
                    done = f"already delivered by campaign {campaign['id']}"
                elif cooling:
                    done = f"cooldown until {cooldown} (campaign {campaign['id']})"
                else:
                    continue
                if campaign.get("regime"):
                    regimes.append({"regime": campaign["regime"], "reason": done})
                else:
                    reason = done
            if reason:
                skipped.append({**entry, "reason": reason})
            else:
                candidates.append({**entry, **({"skip_regimes": regimes} if regimes else {})})
        candidates.sort(key=lambda entry: entry["path"])
        return {"commit": commit, "candidates": candidates, "skipped": skipped}

    def is_relevant(self, path, gitlink):
        """Relevant unless the path is known not to reach what validation reads for this candidate."""
        if gitlink:
            return True
        path = PurePosixPath(path)
        # Validation reads no Markdown.
        if path.suffix == ".md":
            return False
        # Only the `aklr` package is installed; root files such as pyproject.toml set its build and
        # dependencies, and the CI workflow pins the image and the baseline every Point is measured against.
        if path.parts[0] != "aklr":
            return (len(path.parts) == 1 and not path.name.startswith(".")) or path == PurePosixPath(
                ".github/workflows/ci.yaml"
            )
        if path.parts[1] != "plugins":
            return True
        if not path.is_relative_to(self.op_dir):
            # Another op's code may be imported by this op's reference or baseline; its kernels are not.
            return "impls" not in path.parts
        # Bench reads only the incumbents that share the candidate's name; perf-bot updates land elsewhere.
        rest = path.relative_to(self.op_dir).parts
        return not (len(rest) > 3 and rest[0] == "impls" and rest[2] != self.name)

    def relevant_changes(self, base, main):
        diff = git("diff", "--raw", "--no-renames", "-z", base, main).stdout.decode()
        # -z output alternates ":<old mode> <new mode> <old> <new> <status>" and the path.
        fields = diff.split("\0")[:-1]
        return sorted(
            path
            for meta, path in zip(fields[::2], fields[1::2])
            if self.is_relevant(path, "160000" in meta.lstrip(":").split()[:2])
        )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    listing = sub.add_parser("list", help="list the round's candidates at a commit")
    listing.add_argument("--config", required=True)
    listing.add_argument("--commit", required=True)
    listing.add_argument("--ledger", required=True, help="file holding `state.py show`'s output")
    freshness = sub.add_parser("freshness", help="exit 0 fresh, 1 stale, 2 unknown")
    freshness.add_argument("--schema", required=True, help="repository-relative schema JSON path")
    freshness.add_argument("--base", required=True, help="commit the evidence was validated against")
    freshness.add_argument("--main", required=True, help="current main commit")
    args = parser.parse_args()
    if args.command == "list":
        try:
            import yaml

            with open(args.config) as stream:
                config = yaml.safe_load(stream)
            with open(args.ledger) as stream:
                state = json.load(stream)["state"]
            listed = Candidate.listed(args.commit, config, state)
        except (ImportError, OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
            print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
            return 2
        json.dump(listed, sys.stdout, indent=2)
        print()
        return 0
    try:
        candidate = Candidate.of(args.schema)
        ancestor = git("merge-base", "--is-ancestor", args.base, args.main, check=False)
        if ancestor.returncode not in (0, 1):
            raise subprocess.CalledProcessError(ancestor.returncode, ancestor.args)
        if ancestor.returncode == 1:
            print(
                json.dumps({"base": args.base, "main": args.main, "stale": "base is not an ancestor of main"})
            )
            return 1
        relevant = candidate.relevant_changes(args.base, args.main)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps({"base": args.base, "main": args.main, "relevant": relevant}))
    return 1 if relevant else 0


if __name__ == "__main__":
    sys.exit(main())
