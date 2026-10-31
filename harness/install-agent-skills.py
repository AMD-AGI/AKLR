#!/usr/bin/env python3
"""Install the repository's AKLR skills for a supported coding agent."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parent
REPO_ROOT = HARNESS_DIR.parent
SKILLS_DIR = HARNESS_DIR / "skills"


def default_destination(target: str) -> Path:
    if target == "codex":
        codex_home = os.environ.get("CODEX_HOME")
        return (Path(codex_home) if codex_home else Path.home() / ".codex") / "skills"
    if target == "claude":
        return REPO_ROOT / ".claude" / "skills"
    if target == "cursor":
        return REPO_ROOT / ".cursor" / "skills"
    raise ValueError(f"unsupported target: {target}")


def available_skills() -> list[Path]:
    return sorted(path for path in SKILLS_DIR.iterdir() if path.is_dir() and (path / "SKILL.md").is_file())


def remove_existing(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def install(
    target: str,
    destination: Path,
    *,
    copy: bool,
    force: bool,
    dry_run: bool,
) -> None:
    mode = "copy" if copy else "symlink"
    print(f"Installing AKLR skills for {target} into {destination} ({mode})")

    if not dry_run:
        destination.mkdir(parents=True, exist_ok=True)

    for source in available_skills():
        output = destination / source.name
        if output.is_symlink() and not copy and output.resolve() == source.resolve():
            print(f"  unchanged: {output}")
            continue
        if output.exists() or output.is_symlink():
            if not force:
                raise FileExistsError(f"{output} already exists; rerun with --force to replace it")
            print(f"  replace:   {output}")
            if not dry_run:
                remove_existing(output)

        print(f"  install:   {source.name}")
        if dry_run:
            continue
        if copy:
            shutil.copytree(source, output)
        else:
            output.symlink_to(source, target_is_directory=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        choices=("codex", "claude", "cursor"),
        default="codex",
        help="agent environment to install for (default: codex)",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        help="override the target's default skills directory",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="copy skill directories instead of creating symlinks",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace existing skill directories",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the planned installation without changing files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    destination = args.destination or default_destination(args.target)
    install(
        args.target,
        destination.expanduser().resolve(),
        copy=args.copy,
        force=args.force,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
