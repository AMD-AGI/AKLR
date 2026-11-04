#!/usr/bin/env python3
"""Re-point the `- S<n> <path>:<line>` citations of the skill at a page whose lines moved.

Each citation of PAGE is read against PAGE at REV (the text it was written against) and moved to the line of the
working copy with the same text, evidence marks and spacing ignored. A citation whose line is gone or appears more
than once is listed for a person to fix, and left as it is.

    python maintenance/kb_reanchor.py gemm/flydsl_gemm_small_m.md            # against HEAD
    python maintenance/kb_reanchor.py gemm/flydsl_gemm_small_m.md --dry-run
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from kb_lint import SKILL_ROOT, knowledge_pages

CITATION_RE = re.compile(r"^(-\s+S\d+\s+)(\S+?):(\d+)\b", re.MULTILINE)
MARK_RE = re.compile(r"\*\(field[^)]*\)\*|\*field:\*")


def _norm(line: str) -> str:
    return " ".join(MARK_RE.sub(" ", line).split())


def _old_lines(page: Path, rev: str) -> list[str]:
    top = subprocess.run(
        ["git", "-C", str(page.parent), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    rel = page.resolve().relative_to(top).as_posix()
    follow = ["log", "--follow", "--diff-filter=R", "--name-status", "--format="]
    renamed = subprocess.run(
        ["git", "-C", top, *follow, f"{rev}..HEAD", "--", rel],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    if renamed:  # the oldest rename since `rev` gives the page's path there
        rel = renamed[-1].split("\t")[1]
    shown = subprocess.run(
        ["git", "-C", top, "show", f"{rev}:{rel}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return shown.stdout.splitlines()


def new_line(old: list[str], new: list[str], line: int) -> int | None:
    """The line of `new` holding what line `line` of `old` held, or None when it is gone or not unique."""
    if not 1 <= line <= len(old) or not _norm(old[line - 1]):
        return None
    wanted = _norm(old[line - 1])
    hits = [i + 1 for i, text in enumerate(new) if _norm(text) == wanted]
    return hits[0] if len(hits) == 1 else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "page",
        type=Path,
        help="the page whose lines moved, relative to the skill or absolute",
    )
    parser.add_argument("--rev", default="HEAD", help="the revision the citations were written against")
    parser.add_argument("--dry-run", action="store_true", help="report, change nothing")
    args = parser.parse_args(argv)
    page = (args.page if args.page.is_absolute() else SKILL_ROOT / args.page).resolve()
    target = page.relative_to(SKILL_ROOT).as_posix()
    old, new = _old_lines(page, args.rev), page.read_text(encoding="utf-8").splitlines()
    unresolved = moved = 0
    for path in knowledge_pages():
        text = path.read_text(encoding="utf-8")
        parts, last = [], 0
        for match in CITATION_RE.finditer(text):
            if match.group(2) != target:
                continue
            line = int(match.group(3))
            found = new_line(old, new, line)
            if found is None:
                unresolved += 1
                where = path.relative_to(SKILL_ROOT)
                print(f"{where}: {match.group(1).strip()} {target}:{line} — fix by hand")
                continue
            moved += found != line
            parts += [text[last : match.start()], f"{match.group(1)}{target}:{found}"]
            last = match.end()
        changed = "".join(parts) + text[last:]
        if changed != text and not args.dry_run:
            path.write_text(changed, encoding="utf-8")
    print(f"{moved} citations moved, {unresolved} to fix by hand")
    return 1 if unresolved else 0


if __name__ == "__main__":
    sys.exit(main())
