#!/usr/bin/env python3
"""Regenerate what the book derives from its pages: the INDEX.md tables and the case pages among `sources`.

- The INDEX.md of the folders whose pages carry the new front matter: the table between the `generated` markers is
  rebuilt from each page's front matter; text outside the markers is kept.
- The case pages in the `sources` of every page that is not a case page: the case pages its text links and the
  case pages whose text links it, after the public sources, where a reader of the page sees which kernels it
  rests on. The public entries are kept as they are.

    python maintenance/kb_index.py            # rewrite the indexes and the case pages of `sources`
    python maintenance/kb_index.py --check    # exit 1 if one is out of date
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import kb_lint
from kb_lint import read_page

START, END = "<!-- generated:start (maintenance/kb_index.py) -->", "<!-- generated:end -->"

# folder -> (heading, one line on what the folder holds, the columns of its table: front-matter fields, and
# `toolchain`, what the page's kernels are written with)
FOLDERS = {
    "patterns": (
        "Patterns",
        "Symptom pages: from what a trace or counter shows to the techniques worth trying.",
        ("signals", "ops", "basis"),
    ),
    "cases": (
        "Cases",
        "Worked cases, one kernel or kernel family each: the op, its shapes and dtypes, what made it fast or what went "
        "wrong, its code, and what was measured.",
        ("ops", "dtypes", "regimes", "toolchain", "accuracy", "basis"),
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


def _toolchain(front: dict) -> str:
    """What a page's kernels are written with, from the version it pins: `flydsl: 0.3.2` or `triton: 3.8.0`."""
    return ", ".join(
        f"{name} {front[key]}" for key, name in (("flydsl", "FlyDSL"), ("triton", "Triton")) if front.get(key)
    )


def render(folder: str) -> str:
    _, _, columns = FOLDERS[folder]
    rows = []
    for path in sorted((kb_lint.SKILL_ROOT / folder).glob("*.md")):
        if path.name == "INDEX.md" or path.name.endswith(".review.md"):
            continue
        front = read_page(path).front or {}
        title = _cell(front.get("title")) or path.stem
        cells = (_toolchain(front) if c == "toolchain" else _cell(front.get(c)) for c in columns)
        rows.append(f"| [{title}]({path.name}) | " + " | ".join(cells) + " |")
    head = "| Page | " + " | ".join("FlyDSL" if c == "flydsl" else c.capitalize() for c in columns) + " |"
    rule = "|---" * (len(columns) + 1) + "|"
    body = "\n".join(rows) if rows else "| *(none yet)* |" + " |" * len(columns)
    return f"{START}\n{head}\n{rule}\n{body}\n{END}"


def updated(folder: str) -> tuple[Path, str]:
    index = kb_lint.SKILL_ROOT / folder / "INDEX.md"
    table = render(folder)
    if not index.exists():
        heading, blurb, _ = FOLDERS[folder]
        return index, f"---\nkind: index\n---\n\n# {heading}\n\n{blurb}\n\n{table}\n"
    text = index.read_text(encoding="utf-8")
    start, end = text.find(START), text.find(END)
    if start == -1 or end == -1:
        return index, text.rstrip("\n") + f"\n\n{table}\n"
    return index, text[:start] + table + text[end + len(END) :]


SOURCES_FIELD_RE = re.compile(r"^sources:[ \t]*\n((?:[ \t]+-.*\n)*)", re.MULTILINE)
SOURCES_FLOW_RE = re.compile(r"^sources:[ \t]*\[(.*)\][ \t]*\n", re.MULTILINE)
CASE_ITEM_RE = re.compile(rf"^[ \t]+-[ \t]+[\"']?{kb_lint.CASES_FOLDER}/[\w.-]+\.md[\"']?[ \t]*$")


def with_case_sources(text: str, entries: list[str]) -> str:
    """A page's text with the case pages of its `sources` set to `entries`: one a line, after the public sources,
    which stay as written. A page left with no source loses the field; one that had none gets it last."""
    end = text.find("\n---", 4) + 1
    front, rest = text[4:end], text[end:]
    flow = SOURCES_FLOW_RE.search(front)
    if flow:  # a list on the key's line: one entry a line, like the others
        items = "".join(f"  - {item.strip()}\n" for item in flow.group(1).split(",") if item.strip())
        front = front[: flow.start()] + "sources:\n" + items + front[flow.end() :]
    cases = "".join(f"  - {entry}\n" for entry in entries)
    found = SOURCES_FIELD_RE.search(front)
    if found:
        public = "".join(f"{line}\n" for line in found.group(1).splitlines() if not CASE_ITEM_RE.match(line))
        field = f"sources:\n{public}{cases}" if public or cases else ""
        front = front[: found.start()] + field + front[found.end() :]
    elif cases:
        front += f"sources:\n{cases}"
    return "---\n" + front + rest


def sources_updated(path: Path) -> str | None:
    """The text of a page with the case pages of its `sources` as they should be; None for a page whose `sources`
    list none: a case page, and a page whose kind the lint does not check."""
    page = read_page(path)
    if page.front is None or page.kind == "case" or page.kind not in kb_lint.NEW_KINDS:
        return None
    return with_case_sources(page.text, sorted(set(kb_lint.linked_cases(page)) | kb_lint.citing_cases(page)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="report stale indexes and `sources` instead of rewriting them",
    )
    args = parser.parse_args(argv)
    root = kb_lint.SKILL_ROOT
    stale = []
    for path in kb_lint.knowledge_pages():
        text = sources_updated(path)
        if text is None or text == path.read_text(encoding="utf-8"):
            continue
        stale.append(path.relative_to(root).as_posix())
        if not args.check:
            path.write_text(text, encoding="utf-8")
    for folder in FOLDERS:
        index, text = updated(folder)
        current = index.read_text(encoding="utf-8") if index.exists() else None
        if text == current:
            continue
        stale.append(index.relative_to(root).as_posix())
        if not args.check:
            index.write_text(text, encoding="utf-8")
    verb = "out of date" if args.check else "rewritten"
    print(f"{verb}: {', '.join(stale)}" if stale else "indexes and `sources` up to date")
    return 1 if args.check and stale else 0


if __name__ == "__main__":
    sys.exit(main())
