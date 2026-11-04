#!/usr/bin/env python3
"""List the numbers an edit changed in a page's text, outside its Evidence section and code.

Adding evidence ids to a card should leave its numbers alone: this prints every number the working copy
has more or fewer of than the page at REV, so each change can be matched to a correction that was meant.

    python maintenance/kb_numbers_guard.py gemm/flydsl_gemm_small_m.md            # against HEAD
    python maintenance/kb_numbers_guard.py gemm/flydsl_gemm_small_m.md --rev main
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

from kb_lint import _prose, read_page

NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?:e-?\d+)?")
# The marks themselves, not what a `*(field: …)*` mark says: rewriting it as *(field, E<n>)* moves those numbers into
# the sentence, where they count the same.
MARK_RE = re.compile(r"\*\(field(?:,\s*E\d+(?:,\s*E\d+)*)?\)\*|\*field:\*|\*\(field:|E\d+\b")


def numbers(body: str) -> Counter:
    prose = _prose(body)
    cut = re.search(r"^## Evidence\s*$", prose, re.MULTILINE)
    text = MARK_RE.sub(" ", prose[: cut.start()] if cut else prose)
    return Counter(NUMBER_RE.findall(text))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("page", type=Path)
    parser.add_argument("--rev", default="HEAD")
    args = parser.parse_args(argv)
    path = args.page.resolve()
    top = subprocess.run(
        ["git", "-C", str(path.parent), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    rel = path.relative_to(top).as_posix()
    follow = ["log", "--follow", "--diff-filter=R", "--name-status", "--format="]
    renamed = subprocess.run(
        ["git", "-C", top, *follow, f"{args.rev}..HEAD", "--", rel],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    if renamed:  # the oldest rename since --rev gives the page's path there
        rel = renamed[-1].split("\t")[1]
    old_text = subprocess.run(
        ["git", "-C", top, "show", f"{args.rev}:{rel}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    old = numbers(old_text.split("\n---\n", 1)[-1] if old_text.startswith("---\n") else old_text)
    new = numbers(read_page(path).body)
    removed, added = old - new, new - old
    for label, counter in (("removed", removed), ("added", added)):
        if counter:
            print(f"{label}: " + ", ".join(f"{n}×{c}" if c > 1 else n for n, c in sorted(counter.items())))
    if not (removed or added):
        print("numbers unchanged")
    return 1 if removed or added else 0


if __name__ == "__main__":
    sys.exit(main())
