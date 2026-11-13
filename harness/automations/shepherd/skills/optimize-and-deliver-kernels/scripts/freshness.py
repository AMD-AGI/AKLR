"""Freshness of a candidate's validation evidence against a later main.

Prints the changes between the evidence base and main that can alter the candidate's verdict. Exit 0: fresh;
1: stale (relevant changes, or the base is no longer an ancestor of main); 2: error, treat as unknown.
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import PurePosixPath


@dataclass(frozen=True)
class Candidate:
    """A schema JSON at `<op_dir>/impls/<language>/<name>/<file>.json`."""

    op_dir: PurePosixPath
    name: str

    @classmethod
    def of(cls, schema_path):
        path = PurePosixPath(schema_path)
        if len(path.parts) < 5 or path.parts[-4] != "impls" or path.suffix != ".json":
            raise ValueError(f"{schema_path} is not <op_dir>/impls/<language>/<name>/<file>.json")
        return cls(op_dir=path.parents[3], name=path.parts[-2])

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
        diff = subprocess.run(
            ["git", "diff", "--raw", "--no-renames", "-z", base, main],
            capture_output=True,
            check=True,
        ).stdout.decode()
        # -z output alternates ":<old mode> <new mode> <old> <new> <status>" and the path.
        fields = diff.split("\0")[:-1]
        return sorted(
            path
            for meta, path in zip(fields[::2], fields[1::2])
            if self.is_relevant(path, "160000" in meta.lstrip(":").split()[:2])
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", required=True, help="repository-relative schema JSON path")
    parser.add_argument("--base", required=True, help="commit the evidence was validated against")
    parser.add_argument("--main", required=True, help="current main commit")
    args = parser.parse_args()
    try:
        candidate = Candidate.of(args.schema)
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", args.base, args.main], capture_output=True, check=False
        )
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
