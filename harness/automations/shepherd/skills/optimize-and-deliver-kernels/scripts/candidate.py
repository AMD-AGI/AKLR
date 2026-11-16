"""A candidate schema of a shepherd round, at `<op_dir>/impls/<language>/<name>/<file>.json`.

Run at the repository root.

`candidate.py list --config .shepherd/configuration.yaml --commit <main SHA>` prints the candidates at that
commit, by strategy order with the configuration's filters, as {path, op, backend, input_sha256, strategy}
sorted by path, then the skipped ones with their reason. Ledger and GitHub skips (delivered, competing PR,
cooldown, blocked contract), the AKLR-baseline check and the GEMM regime split stay with the round.

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
from pathlib import PurePosixPath

OPS = PurePosixPath("aklr/ops")


def git(*args, stdin=None, check=True):
    return subprocess.run(["git", *args], input=stdin, capture_output=True, check=check)


@dataclass(frozen=True)
class Candidate:
    path: PurePosixPath

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
    def listed(cls, commit, policy):
        """The policy's candidates at the commit and the skipped ones, each with its strategy."""
        allowed, backends = policy["allowed_ops"], policy["backends"]
        listed = []
        for strategy in policy["strategies"]:
            if strategy["type"] == "manual":
                listed += [(path, "manual") for path in strategy.get("schemas") or []]
            elif strategy["type"] == "kernel-less":
                tree = git("ls-tree", "-r", "--name-only", commit, "--", str(OPS)).stdout.decode()
                for path in tree.splitlines():
                    try:
                        cls.of(path)
                    except ValueError:
                        continue
                    listed.append((path, "kernel-less"))
            else:
                raise SystemExit(f"unknown strategy {strategy['type']!r}")
        paths = sorted({path for path, _ in listed})
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
        candidates, skipped, seen = [], [], set()
        for path, strategy in listed:
            if path in seen:
                continue
            seen.add(path)
            entry, blob = {"path": path, "strategy": strategy}, blobs[path]
            if blob is None:
                skipped.append({**entry, "reason": "missing at this commit"})
                continue
            try:
                candidate = cls.of(path)
            except ValueError as exc:
                skipped.append({**entry, "reason": str(exc)})
                continue
            kernel = json.loads(blob).get("kernel") or {}
            backend = kernel.get("language") or candidate.path.parts[-3]
            entry.update(op=candidate.op, backend=backend, input_sha256=hashlib.sha256(blob).hexdigest())
            if strategy == "kernel-less" and kernel.get("sources"):
                continue  # Already has a kernel: not a kernel-less candidate, nothing to report.
            if "all" not in allowed and candidate.op not in allowed:
                skipped.append({**entry, "reason": "op not in policy.allowed_ops"})
            elif backend not in backends:
                skipped.append({**entry, "reason": "language not in policy.backends"})
            else:
                candidates.append(entry)
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
        if path.parts[1] not in ("ops", "plugins"):
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    listing = sub.add_parser("list", help="list the round's candidates at a commit")
    listing.add_argument("--config", required=True)
    listing.add_argument("--commit", required=True)
    freshness = sub.add_parser("freshness", help="exit 0 fresh, 1 stale, 2 unknown")
    freshness.add_argument("--schema", required=True, help="repository-relative schema JSON path")
    freshness.add_argument("--base", required=True, help="commit the evidence was validated against")
    freshness.add_argument("--main", required=True, help="current main commit")
    args = parser.parse_args()
    if args.command == "list":
        import yaml

        with open(args.config) as stream:
            policy = yaml.safe_load(stream)["policy"]
        json.dump(Candidate.listed(args.commit, policy), sys.stdout, indent=2)
        print()
        return 0
    try:
        candidate = Candidate.of(args.schema)
        ancestor = git("merge-base", "--is-ancestor", args.base, args.main, check=False)
        if ancestor.returncode not in (0, 1):
            raise subprocess.CalledProcessError(ancestor.returncode, ancestor.args)
        if ancestor.returncode == 1:
            print(json.dumps({"base": args.base, "main": args.main, "stale": "base is not an ancestor of main"}))
            return 1
        relevant = candidate.relevant_changes(args.base, args.main)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps({"base": args.base, "main": args.main, "relevant": relevant}))
    return 1 if relevant else 0


if __name__ == "__main__":
    sys.exit(main())
