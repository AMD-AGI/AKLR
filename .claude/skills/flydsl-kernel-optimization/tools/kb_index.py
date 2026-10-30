#!/usr/bin/env python3
"""Regenerate the INDEX.md of the folders whose pages carry the new front matter.

The table between the `generated` markers is rebuilt from each page's front matter; text outside the markers is kept.

    python tools/kb_index.py            # rewrite the indexes
    python tools/kb_index.py --check    # exit 1 if an index is out of date
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from kb_lint import SKILL_ROOT, read_page

START, END = "<!-- generated:start (tools/kb_index.py) -->", "<!-- generated:end -->"

# folder -> (heading, one line on what the folder holds, the front-matter columns of its table)
FOLDERS = {
    "patterns": (
        "Patterns",
        "Symptom pages: from what a trace or counter shows to the techniques worth trying.",
        ("signals", "ops", "basis"),
    ),
    "cases": (
        "Cases",
        "Delivered kernels: what made them fast, under which conditions, and how to reproduce them.",
        ("ops", "dtypes", "regimes", "contract", "basis"),
    ),
    "language": (
        "Language",
        "What FlyDSL v0.3.2 does, read from its source: limits, compile errors, version differences.",
        ("gens", "flydsl", "basis"),
    ),
    "pitfalls": (
        "Pitfalls",
        "Silent wrong results, FlyDSL version traps and measurement artifacts.",
        ("scope", "gens", "flydsl", "basis"),
    ),
}


def _cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)
    return str(value).replace("|", "\\|")


def render(folder: str) -> str:
    _, _, columns = FOLDERS[folder]
    rows = []
    for path in sorted((SKILL_ROOT / folder).glob("*.md")):
        if path.name == "INDEX.md" or path.name.endswith(".review.md"):
            continue
        front = read_page(path).front or {}
        title = _cell(front.get("title")) or path.stem
        rows.append(f"| [{title}]({path.name}) | " + " | ".join(_cell(front.get(c)) for c in columns) + " |")
    head = "| Page | " + " | ".join("FlyDSL" if c == "flydsl" else c.capitalize() for c in columns) + " |"
    rule = "|---" * (len(columns) + 1) + "|"
    body = "\n".join(rows) if rows else "| *(none yet)* |" + " |" * len(columns)
    return f"{START}\n{head}\n{rule}\n{body}\n{END}"


def updated(folder: str) -> tuple[Path, str]:
    index = SKILL_ROOT / folder / "INDEX.md"
    table = render(folder)
    if not index.exists():
        heading, blurb, _ = FOLDERS[folder]
        return index, f"---\nkind: index\n---\n\n# {heading}\n\n{blurb}\n\n{table}\n"
    text = index.read_text(encoding="utf-8")
    start, end = text.find(START), text.find(END)
    if start == -1 or end == -1:
        return index, text.rstrip("\n") + f"\n\n{table}\n"
    return index, text[:start] + table + text[end + len(END) :]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="report stale indexes instead of rewriting them",
    )
    args = parser.parse_args(argv)
    stale = []
    for folder in FOLDERS:
        index, text = updated(folder)
        current = index.read_text(encoding="utf-8") if index.exists() else None
        if text == current:
            continue
        stale.append(index.relative_to(SKILL_ROOT).as_posix())
        if not args.check:
            index.write_text(text, encoding="utf-8")
    verb = "out of date" if args.check else "rewritten"
    print(f"{verb}: {', '.join(stale)}" if stale else "indexes up to date")
    return 1 if args.check and stale else 0


if __name__ == "__main__":
    sys.exit(main())
