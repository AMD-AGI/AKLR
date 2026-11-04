#!/usr/bin/env python3
"""Which FlyDSL kernels of a revision no case page covers.

Lists every schema under `aklr/ops/*/impls/flydsl/` and `aklr/plugins/` at REV whose kernel carries source, and
matches it with the `sources:` of the pages in `cases/`:

- covered: a case names the schema file;
- renamed: a case names a file of the same schema directory that is gone (the kernel changed since, so its
  case needs a look);
- missing: no case names the directory.

Run it monthly and after a batch of kernels merges; the missing ones are the next cases to write.

    python maintenance/kb_reconcile.py                   # origin/main of this repository
    python maintenance/kb_reconcile.py --rev HEAD --json reconcile.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from kb_lint import SKILL_ROOT, read_page

REPO_ROOT = SKILL_ROOT.parents[2]
ROOTS = ("aklr/ops", "aklr/plugins")


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, check=True
    ).stdout


def tracked(rev: str) -> list[str]:
    return _git("ls-tree", "-r", "--name-only", rev, *ROOTS).split()


def flydsl_kernels(rev: str) -> list[str]:
    """Schema paths at REV whose FlyDSL kernel has source."""
    paths = [p for p in tracked(rev) if "/impls/flydsl/" in p and p.endswith(".json")]
    if not paths:
        return []
    blobs = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "cat-file", "--batch"],
        input="".join(f"{rev}:{p}\n" for p in paths).encode(),
        capture_output=True,
        check=True,
    ).stdout
    out, pos = [], 0
    for path in paths:
        header_end = blobs.index(b"\n", pos)
        size = int(blobs[pos:header_end].split()[2])
        body = blobs[header_end + 1 : header_end + 1 + size]
        pos = header_end + 1 + size + 1
        kernel = json.loads(body).get("kernel") or {}
        if any(src.get("content") for src in kernel.get("sources") or []):
            out.append(path)
    return out


def case_sources() -> dict[str, list[str]]:
    """Schema path -> the case pages that name it."""
    named: dict[str, list[str]] = {}
    for page in sorted((SKILL_ROOT / "cases").glob("*.md")):
        if page.name == "INDEX.md":
            continue
        for src in (read_page(page).front or {}).get("sources") or []:
            named.setdefault(str(src).split("#")[0], []).append(page.name)
    return named


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--rev", default="origin/main", help="revision of this repository")
    parser.add_argument("--json", type=Path, help="also write the result here")
    args = parser.parse_args(argv)
    kernels = flydsl_kernels(args.rev)
    named = case_sources()
    named_dirs = {str(Path(p).parent): pages for p, pages in named.items()}
    result: dict[str, list] = {"covered": [], "renamed": [], "missing": []}
    for path in kernels:
        if path in named:
            result["covered"].append(path)
        elif str(Path(path).parent) in named_dirs:
            result["renamed"].append([path, named_dirs[str(Path(path).parent)]])
        else:
            result["missing"].append(path)
    present = set(tracked(args.rev))
    gone = sorted(p for p in named if p.startswith(ROOTS) and p not in present)
    print(
        f"{len(kernels)} FlyDSL kernels at {args.rev}: {len(result['covered'])} covered, "
        f"{len(result['renamed'])} renamed, {len(result['missing'])} missing"
    )
    for path, pages in result["renamed"]:
        print(f"renamed  {path}  (case: {', '.join(pages)})")
    for path in result["missing"]:
        print(f"missing  {path}")
    for path in gone:
        print(f"gone     {path}  (named by {', '.join(named[path])})")
    if args.json:
        args.json.write_text(json.dumps({**result, "gone": gone}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
