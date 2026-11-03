#!/usr/bin/env python3
"""Move a reviewed page from staging/ into the skill.

Refuses when a page has lint errors at its new place, when the destination exists, or when a technique's `lever` is
already taken. Pages that link to each other go in one call: they move together, are checked where they will live,
and go back to staging/ together if any fails. On success it deletes each page's review notes
(staging/<name>.review.md) and rebuilds the indexes, and prints the `git add` command for the moved pages.

    python tools/kb_approve.py staging/xcd_remap.md --to techniques
    python tools/kb_approve.py staging/a.md staging/b.md --to cases
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import kb_index
from kb_lint import NEW_KINDS, SKILL_ROOT, knowledge_pages, lint_page, read_page

KIND_FOLDER = {
    "pattern": "patterns",
    "case": "cases",
    "pitfall": "pitfalls",
    "language-note": "language",
}


def _lever_owner(lever: str, exclude: Path) -> Path | None:
    for path in knowledge_pages():
        if path == exclude or path.parent.name == "staging":
            continue
        front = read_page(path).front or {}
        if front.get("kind") == "technique" and front.get("lever") == lever:
            return path
    return None


def _report(issues: list) -> bool:
    """Print the issues; True when none is an error."""
    for issue in issues:
        print(issue)
    return not any(i.level == "error" for i in issues)


def _check(arg: Path, folder: Path, to: str, batch: list[Path]) -> Path | None:
    """The destination of one staged page, or None after printing why it cannot move."""
    src = arg.resolve()
    if src.parent != SKILL_ROOT / "staging" or not src.is_file():
        print(f"not a page in staging/: {arg}")
        return None
    dst = folder / src.name
    if dst.exists():
        print(f"{dst.relative_to(SKILL_ROOT)} exists; merge into it instead")
        return None
    page = read_page(src)
    if page.kind not in NEW_KINDS:
        print(f"refused: kind `{page.kind}` is not one of {', '.join(NEW_KINDS)}; start from templates/")
        return None
    home = KIND_FOLDER.get(page.kind)
    if (home or to) != to or (not home and to in KIND_FOLDER.values()):
        where = f"{home}/" if home else "the op's package"
        print(f"refused: a {page.kind} page goes to {where}, not {to}/")
        return None
    lever = (page.front or {}).get("lever")
    owner = _lever_owner(lever, src) if lever else None
    if owner is None and lever:
        owner = next((b for b in batch if (read_page(b).front or {}).get("lever") == lever), None)
    if owner:
        print(f"refused: lever `{lever}` is already {owner.relative_to(SKILL_ROOT)}; merge into it")
        return None
    return dst


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("pages", nargs="+", type=Path, help="pages under staging/, all for one folder")
    parser.add_argument("--to", required=True, help="destination folder in the skill, for example cases")
    args = parser.parse_args(argv)

    folder = SKILL_ROOT / args.to
    if not folder.is_dir():
        print(f"no folder {args.to}/ in the skill")
        return 1
    moves: list[tuple[Path, Path]] = []
    for arg in args.pages:
        dst = _check(arg, folder, args.to, [s for s, _ in moves])
        if dst is None:
            return 1
        moves.append((arg.resolve(), dst))

    # Lint each page where it will live: links are relative to the page, and pages of one call may link each other.
    for src, dst in moves:
        src.rename(dst)
    if not _report([i for _, dst in moves for i in lint_page(read_page(dst))]):
        for src, dst in moves:
            dst.rename(src)
        print("refused: fix the errors above first; the pages are back in staging/")
        return 1
    for src, _ in moves:
        review = src.with_name(src.stem + ".review.md")
        if review.exists():
            review.unlink()
    kb_index.main([])
    repo = SKILL_ROOT.parents[2]
    staged = [dst.relative_to(repo).as_posix() for _, dst in moves]
    if args.to in kb_index.FOLDERS:
        staged.append((folder / "INDEX.md").relative_to(repo).as_posix())
    else:
        names = ", ".join(dst.name for _, dst in moves)
        print(f"{args.to}/INDEX.md is written by hand: add a row that routes to {names}")
    command = "git add " + " ".join(staged)
    for src, _ in moves:
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", src.relative_to(repo).as_posix()],
            cwd=repo,
            capture_output=True,
            check=False,
        )
        if tracked.returncode == 0:
            command += f" && git rm -q --cached {src.relative_to(repo).as_posix()}"
    for _, dst in moves:
        print(f"approved: {dst.relative_to(SKILL_ROOT)}")
    print(f"next: {command}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
