#!/usr/bin/env python3
"""What forge-loop campaigns looked up in this skill, and what it did not answer.

Reads each campaign's forge_experiments/: the skill files each round opened (`skill_reads` of the
`iteration_result` events) and the `KB:` lines of each round's lesson (`KB: matched <page> §<n>`,
`KB: no-match <question>`, `KB: none`). A lesson the loop wrote itself, because the round's agent
wrote none, is counted as not recorded rather than as a round that looked nothing up.

    python maintenance/kb_gaps.py <campaign workspace or forge_experiments dir> ...
    python maintenance/kb_gaps.py ws1 ws2 --json gaps.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

KB_LINE_RE = re.compile(r"^\s*[-*]?\s*`?KB:\s*(matched|no-match|none)\b`?\s*(.*?)`?\s*$", re.IGNORECASE)
# The pages a `KB: matched` line names; it may name several and say more about each in words.
PAGE_RE = re.compile(r"[\w./+-]+\.md")
FALLBACK_MARK = "(no agent summary)"
SKILL_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Gaps:
    campaigns: int = 0
    rounds: int = 0
    rounds_opening_the_skill: int = 0
    lessons_by_agent: int = 0
    lessons_not_recorded: int = 0
    lessons_with_kb_lines: int = 0
    lessons_kb_none: int = 0
    opened: Counter = field(default_factory=Counter)
    matched: Counter = field(default_factory=Counter)
    no_match: list = field(default_factory=list)


def experiments_dir(path: Path) -> Path | None:
    for candidate in (path / "forge_experiments", path):
        if (candidate / "lessons").is_dir() or (candidate / "events.jsonl").is_file():
            return candidate
    return None


def _page(text: str) -> str:
    """A page as its path inside this skill, or as `<skill>/<path>` for a page of another skill (KernelForge records
    reads as `<skill>/<path>`), without a § reference after it."""
    page = text.split("§", 1)[0].strip().strip("`").strip()
    first, _, rest = page.partition("/")
    return rest if rest and first in (SKILL_ROOT.name, "flydsl-kernel-optimization") else page


def read_campaign(root: Path, gaps: Gaps) -> None:
    gaps.campaigns += 1
    events = root / "events.jsonl"
    if events.is_file():
        for line in events.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("type") != "iteration_result":
                continue
            gaps.rounds += 1
            reads = event.get("skill_reads") or []
            gaps.rounds_opening_the_skill += bool(reads)
            gaps.opened.update(_page(read) for read in reads)
    for lesson in sorted((root / "lessons").glob("iter_*.md")):
        text = lesson.read_text(encoding="utf-8", errors="replace")
        if FALLBACK_MARK in text:
            gaps.lessons_not_recorded += 1
            continue
        gaps.lessons_by_agent += 1
        found = [m for m in map(KB_LINE_RE.match, text.splitlines()) if m]
        gaps.lessons_with_kb_lines += bool(found)
        for match in found:
            verdict, rest = match.group(1).lower(), match.group(2)
            if verdict == "matched":
                pages = {_page(page) for page in PAGE_RE.findall(rest)} or {_page(rest)}
                gaps.matched.update(pages)
            elif verdict == "no-match":
                gaps.no_match.append(f"{root.parent.name}/{lesson.stem}: {rest}")
            else:
                gaps.lessons_kb_none += 1


def report(gaps: Gaps) -> str:
    lines = [
        f"{gaps.campaigns} campaigns, {gaps.rounds} rounds; {gaps.rounds_opening_the_skill} opened a skill file",
        (
            f"lessons: {gaps.lessons_by_agent} by the round's agent ({gaps.lessons_with_kb_lines} with KB: lines, "
            f"{gaps.lessons_kb_none} KB: none), {gaps.lessons_not_recorded} not recorded"
        ),
        "",
        "page  opened  matched",
    ]
    for page in sorted(set(gaps.opened) | set(gaps.matched), key=lambda p: (-gaps.opened[p], p)):
        lines.append(f"{page}  {gaps.opened[page]}  {gaps.matched[page]}")
    lines += [
        "",
        f"no-match questions ({len(gaps.no_match)}):",
        *(f"- {q}" for q in gaps.no_match),
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "campaigns",
        nargs="+",
        type=Path,
        help="campaign workspaces or their forge_experiments/",
    )
    parser.add_argument("--json", type=Path, help="also write the counts and questions here")
    args = parser.parse_args(argv)
    gaps = Gaps()
    for path in args.campaigns:
        root = experiments_dir(path)
        if root is None:
            print(
                f"skipped {path}: no forge_experiments/lessons or events.jsonl",
                file=sys.stderr,
            )
            continue
        read_campaign(root, gaps)
    print(report(gaps))
    if args.json:
        payload = {
            **gaps.__dict__,
            "opened": dict(gaps.opened),
            "matched": dict(gaps.matched),
        }
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
