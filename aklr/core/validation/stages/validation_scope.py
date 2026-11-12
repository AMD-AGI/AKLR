###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

import functools
import os
import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final, Literal, get_args

from aklr.core.schema._collection import OPS_DIR, DefaultLayout
from aklr.core.schema._kernel import Language

__all__ = ["SCHEMA_PATH_PATTERN", "PathKind", "ValidationScope"]

PathKind = Literal["schema_data", "core", "other"]

SCHEMA_PATH_PATTERN: Final = re.compile(
    rf"^{re.escape(OPS_DIR)}/[^/]+/impls/(?P<language>[^/]+)/(?P<name>[^/]+)/(?P<uuid>[0-9a-f]{{16}})\.json$"
)


@dataclass(frozen=True)
class ValidationScope:
    root: Path
    head_sha: str
    base_sha: str | None
    schema_paths: tuple[Path, ...]
    bench_paths: tuple[Path, ...]

    @classmethod
    def of_working_tree(cls, base: str | None, schema_paths: Sequence[Path] | None) -> ValidationScope:
        stdout = cls._git(Path.cwd(), "rev-parse", "--show-toplevel")
        root = Path(os.fsdecode(stdout.rstrip(b"\n"))).resolve()
        head_sha = cls._git(root, "rev-parse", "--verify", "HEAD^{commit}")
        base_sha = cls._git(root, "merge-base", base, "HEAD") if base else None
        scope = cls(
            root,
            head_sha.decode().strip(),
            None if base_sha is None else base_sha.decode().strip(),
            (),
            (),
        )
        kinds = {p: scope.path_kind(p) for p in scope.changed_paths()}
        is_schema_only = scope.base_sha is not None and "core" not in kinds.values()
        if schema_paths is not None:
            for path in schema_paths:
                if not path.is_file() or not path.resolve().is_relative_to(root):
                    raise ValueError(f"{path} is not a file in the repository {root}")
            paths = tuple(p.resolve() for p in schema_paths)
        elif is_schema_only:
            changed = (root / p for p, kind in kinds.items() if kind == "schema_data")
            paths = tuple(sorted(p for p in changed if p.is_file()))
        else:
            paths = tuple(scope.layout.paths())
        bench_paths = paths
        # the bench skips measurement-only changes: schemas at paths the base already has
        if is_schema_only:
            at_base = set(scope.paths_at_base(OPS_DIR))
            bench_paths = tuple(p for p in paths if scope.relative_path(p) not in at_base)
        return replace(scope, schema_paths=paths, bench_paths=bench_paths)

    @property
    def layout(self) -> DefaultLayout:
        return DefaultLayout(self.root / OPS_DIR)

    def relative_path(self, path: Path) -> str:
        return Path(path).resolve().relative_to(self.root.resolve()).as_posix()

    def changed_paths(self) -> tuple[str, ...]:
        if self.base_sha is None:
            return ()
        diff = self._git(
            self.root,
            "diff", "--name-only", "--no-renames", "--ignore-submodules=dirty", "-z",
            self.base_sha, "--",
        )  # fmt: skip
        untracked = self._git(self.root, "ls-files", "--others", "--exclude-standard", "-z")
        # os.fsdecode: a file name that is not UTF-8 is still a path, of kind "other"
        paths = (diff + untracked).split(b"\0")
        return tuple(sorted({os.fsdecode(p) for p in paths if p}))

    def paths_at_base(self, directory: str) -> tuple[str, ...]:
        if self.base_sha is None:
            return ()
        stdout = self._git(
            self.root,
            "ls-tree",
            "-r",
            "--name-only",
            "-z",
            self.base_sha,
            "--",
            directory,
        )
        return tuple(sorted(os.fsdecode(p) for p in stdout.split(b"\0") if p))

    @functools.cached_property  # one git call per run, not per workload
    def is_first_release(self) -> bool:
        """The base holds no schema data: the change's schemas ship on correctness alone."""
        return self.base_sha is not None and "schema_data" not in map(
            ValidationScope.path_kind, self.paths_at_base(OPS_DIR)
        )

    def text_at_base(self, path: str) -> str:
        if self.base_sha is None:
            raise ValueError(f"no base to read {path} at")
        blob = self._git(self.root, "cat-file", "blob", f"{self.base_sha}:{path}")
        return blob.decode("utf-8", errors="replace")

    @staticmethod
    def _git(directory: Path, *args: str) -> bytes:
        directory = directory.resolve()
        # safe.directory: git refuses a bind-mounted repo owned by the host user without it
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
        """aklr/core's code is the only change that re-benches every schema."""
        match = SCHEMA_PATH_PATTERN.fullmatch(path)
        is_schema_json = match is not None and match["language"] in get_args(Language)
        if is_schema_json:
            return "schema_data"
        if path.startswith("aklr/core/") and not path.endswith(".md"):
            return "core"
        return "other"
