# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import functools
import os
import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final, Literal

from aklr.core.contract import locate, spec
from aklr.core.contract.locate import committed_schema_paths

__all__ = ["PathKind", "ValidationScope"]

PathKind = Literal["schema_data", "document", "other"]

_SCHEMA_PATH_PATTERN: Final = re.compile(locate.REPOSITORY_SCHEMA_PATH_PATTERN)


@dataclass(frozen=True)
class ValidationScope:
    """What one pipeline run validates: the working tree's schemas against ``base_sha``."""

    root: Path
    head_sha: str
    base_sha: str | None
    schema_paths: tuple[Path, ...]
    bench_paths: tuple[Path, ...]

    @classmethod
    def of_working_tree(cls, base: str | None, schema_paths: Sequence[Path] | None) -> ValidationScope:
        # the working directory's scope only asks git where its repository is
        """The scope of the working directory's repository since the merge-base with ``base``."""
        working_directory = cls(Path.cwd(), "", None, (), ())
        stdout = working_directory._git("rev-parse", "--show-toplevel")
        root = Path(os.fsdecode(stdout.rstrip(b"\n"))).resolve()
        scope = cls(root, "", None, (), ())
        head_sha = scope._git("rev-parse", "--verify", "HEAD^{commit}")
        base_sha = scope._git("merge-base", base, "HEAD") if base else None
        scope = replace(
            scope,
            head_sha=head_sha.decode().strip(),
            base_sha=None if base_sha is None else base_sha.decode().strip(),
        )
        kinds = {p: scope.path_kind(p) for p in scope.changed_paths()}
        is_schema_only = scope.base_sha is not None and "other" not in kinds.values()
        if schema_paths is not None:
            for path in schema_paths:
                if not path.is_file() or not path.resolve().is_relative_to(root):
                    raise ValueError(f"{path} is not a file in the repository {root}")
            paths = tuple(p.resolve() for p in schema_paths)
        elif is_schema_only:
            changed = (root / p for p, kind in kinds.items() if kind == "schema_data")
            paths = tuple(sorted(p for p in changed if p.is_file()))
        else:
            op_dirs = [d for d in (root / "aklr/ops").iterdir() if d.is_dir()]
            op_dirs += [p.parent for p in (root / "aklr/plugins").rglob("contract.py")]
            paths = tuple(p for op_dir in sorted(op_dirs) for p in committed_schema_paths(op_dir))
        bench_paths = paths
        if is_schema_only:
            at_base = set(scope.paths_at_schema_roots())
            bench_paths = tuple(p for p in paths if scope.relative_path(p) not in at_base)
        return replace(scope, schema_paths=paths, bench_paths=bench_paths)

    def relative_path(self, path: Path) -> str:
        return Path(path).resolve().relative_to(self.root.resolve()).as_posix()

    def changed_paths(self) -> tuple[str, ...]:
        """The sorted change set: what differs from ``base_sha`` in the working tree."""
        if self.base_sha is None:
            return ()
        diff = self._git(
            "diff", "--name-only", "--no-renames", "--ignore-submodules=dirty", "-z",
            self.base_sha, "--",
        )  # fmt: skip
        untracked = self._git("ls-files", "--others", "--exclude-standard", "-z")
        # os.fsdecode: a file name that is not UTF-8 is still a path, of kind "other"
        paths = (diff + untracked).split(b"\0")
        return tuple(sorted({os.fsdecode(p) for p in paths if p}))

    def paths_at_schema_roots(self) -> tuple[str, ...]:
        """``paths_at_base`` of every ``locate.SCHEMA_ROOTS`` directory."""
        return tuple(p for d in locate.SCHEMA_ROOTS for p in self.paths_at_base(d))

    def paths_at_base(self, directory: str) -> tuple[str, ...]:
        if self.base_sha is None:
            return ()
        stdout = self._git("ls-tree", "-r", "--name-only", "-z", self.base_sha, "--", directory)
        return tuple(sorted(os.fsdecode(p) for p in stdout.split(b"\0") if p))

    @functools.cached_property  # one git call per run, not per workload
    def is_first_release(self) -> bool:
        """The base holds no schema data: its schemas ship on correctness alone.

        FIXME(main-kernels): main's CI gated on correctness only and its kernels are
        imported unedited."""
        return self.base_sha is not None and "schema_data" not in map(
            ValidationScope.path_kind, self.paths_at_schema_roots()
        )

    def text_at_base(self, path: str) -> str:
        """``path``'s text at the base; git's failure is a ``ValueError``."""
        if self.base_sha is None:
            raise ValueError(f"no base to read {path} at")
        blob = self._git("cat-file", "blob", f"{self.base_sha}:{path}")
        return blob.decode("utf-8", errors="replace")

    def _git(self, *args: str) -> bytes:
        directory = self.root.resolve()
        argv = ["git", "-c", f"safe.directory={directory}", "-C", str(directory), *args]
        try:
            done = subprocess.run(argv, capture_output=True, check=False)
        except OSError as e:
            raise ValueError(f"git {' '.join(args)} in {directory} failed: {e}") from e
        if done.returncode:
            stderr = done.stderr.decode(errors="replace").strip()
            raise ValueError(f"git {' '.join(args)} in {directory} failed: {stderr}")
        return done.stdout

    @staticmethod
    def path_kind(path: str) -> PathKind:
        if path.startswith("docs/") or path.endswith(".md"):
            return "document"
        match = _SCHEMA_PATH_PATTERN.fullmatch(path)
        is_schema_json = match is not None and match["language"] in spec.LANGUAGES
        if is_schema_json:
            return "schema_data"
        return "other"
