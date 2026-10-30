# SPDX-License-Identifier: MIT
"""Select AITER regressions from changed AKLR definitions/solutions, without imports.

Only operator-local changes may narrow the suite. Unknown impact or unavailable
history falls back to the original full file sets; an invalid test map is an
error. Benchmark correctness and the full AKLR unit suite are not filtered.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
MAP_PATH = ".github/schema-test-map.json"
API_NAME = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+")
TEST_PATH = re.compile(r"op_tests/(?:[A-Za-z0-9_/-]+/)?test_[A-Za-z0-9_]+\.py")


class SelectionError(ValueError):
    """An input cannot safely produce a selective test plan."""


class Git:
    def __init__(self, root: Path):
        self.root = root
        self._files: dict[str, frozenset[str]] = {}
        self._definitions: dict[tuple[str, str], dict[str, dict]] = {}

    def run(self, *args: str) -> bytes:
        try:
            return subprocess.run(
                ["git", *args],
                cwd=self.root,
                check=True,
                capture_output=True,
                timeout=30,
            ).stdout
        except (OSError, subprocess.SubprocessError) as error:
            raise SelectionError(f"Git inspection failed: {error}") from error

    def commit(self, ref: str) -> str:
        return self.run("rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}").decode().strip()

    def files(self, ref: str) -> frozenset[str]:
        if ref not in self._files:
            self._files[ref] = frozenset(
                self.run("ls-tree", "-r", "--name-only", "-z", ref).decode().split("\0")[:-1]
            )
        return self._files[ref]

    def read_json(self, ref: str, path: str) -> dict | None:
        if path not in self.files(ref):
            return None
        try:
            data = json.loads(self.run("show", f"{ref}:{path}"))
        except (ValueError, UnicodeError) as error:
            raise SelectionError(f"Invalid JSON at {ref}:{path}") from error
        if not isinstance(data, dict):
            raise SelectionError(f"Expected a JSON object at {ref}:{path}")
        return data

    def definitions(self, ref: str, operator: str) -> dict[str, dict]:
        key = (ref, operator)
        if key in self._definitions:
            return self._definitions[key]
        definitions = {}
        prefix = f"aklr/ops/{operator}/definitions/"
        for path in sorted(self.files(ref)):
            if path.startswith(prefix) and path.endswith(".json"):
                definition = self.read_json(ref, path)
                name = definition.get("name")
                if not isinstance(name, str) or not name or name in definitions:
                    raise SelectionError(f"Missing/duplicate definition name: {path}")
                definitions[name] = definition
        self._definitions[key] = definitions
        return definitions


def validate_test(root: Path, path: str, suite: str) -> None:
    """Lists are passed as argv or AITER_TEST, never as shell fragments."""
    if not isinstance(path, str) or not TEST_PATH.fullmatch(path):
        raise SelectionError(f"Invalid test path: {path!r}")
    parts = PurePosixPath(path).parts
    if (suite == "aiter" and len(parts) != 2) or (suite == "triton" and parts[1] != "triton_tests"):
        raise SelectionError(f"Test is not in the {suite} suite: {path}")
    test = root / path
    if not test.is_file() or test.resolve() != root.resolve() / path:
        raise SelectionError(f"Missing or symlinked test: {path}")


def load_map(root: Path, path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise SelectionError(f"Cannot read test map: {path}") from error
    if not isinstance(data, dict) or data.get("version") != 1:
        raise SelectionError("Test map must have version 1")
    rows = data.get("apis")
    if not isinstance(rows, dict) or not rows:
        raise SelectionError("Test map must contain API rows")
    seen = set(rows)
    for api, row in rows.items():
        if not API_NAME.fullmatch(api) or not isinstance(row, dict):
            raise SelectionError(f"Invalid API row: {api!r}")
        operator = row.get("operator")
        if not isinstance(operator, str) or not re.fullmatch(r"[A-Za-z_]\w*", operator):
            raise SelectionError(f"Invalid operator for {api}")
        if not api.startswith(f"aklr.ops.{operator}."):
            raise SelectionError(f"API/operator mismatch: {api}")
        aliases = row.get("backend_apis")
        if not isinstance(aliases, list) or not aliases:
            raise SelectionError(f"Missing backend APIs for {api}")
        for alias in aliases:
            if not isinstance(alias, str) or not API_NAME.fullmatch(alias) or alias in seen:
                raise SelectionError(f"Invalid/duplicate backend API: {alias!r}")
            seen.add(alias)
        for suite in ("aiter", "triton"):
            tests = row.get(suite)
            if not isinstance(tests, list):
                raise SelectionError(f"Missing {suite} test list for {api}")
            for test in tests:
                validate_test(root, test, suite)
        if not row["aiter"] and not row["triton"]:
            raise SelectionError(f"API has no regression tests: {api}")
    return rows


@dataclass
class Plan:
    mode: str = "selected"
    base: str | None = None
    head: str | None = None
    changed_paths: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    definitions: list[str] = field(default_factory=list)
    apis: list[str] = field(default_factory=list)
    aiter: list[str] = field(default_factory=list)
    triton: list[str] = field(default_factory=list)

    def full(self, root: Path, reason: str) -> Plan:
        self.mode = "full"
        self.reasons.append(reason)
        for suite, pattern in (
            ("aiter", "op_tests/test_*.py"),
            ("triton", "op_tests/triton_tests/**/test_*.py"),
        ):
            tests = sorted(p.relative_to(root).as_posix() for p in root.glob(pattern))
            if not tests:
                raise SelectionError(f"Full {suite} suite is missing")
            for test in tests:
                validate_test(root, test, suite)
            setattr(self, suite, tests)
        return self


def select_tests(
    root: Path,
    rows: dict,
    *,
    base: str | None = None,
    head: str = "HEAD",
    full: bool = False,
) -> Plan:
    plan = Plan()
    if full:
        return plan.full(root, "Full suite requested (manual/main CI)")
    if not base:
        return plan.full(root, "No PR base supplied")
    git = Git(root)
    aliases = {alias: api for api, row in rows.items() for alias in row["backend_apis"]}
    aliases.update({api: api for api in rows})
    affected_apis: set[str] = set()
    affected_definitions: set[str] = set()

    def add_definition(definition: dict, operator: str) -> None:
        name, tags = definition.get("name"), definition.get("tags")
        if not isinstance(name, str) or not name or not isinstance(tags, list):
            raise SelectionError("Definition needs a name and API tags")
        if any(not isinstance(tag, str) for tag in tags):
            raise SelectionError(f"Invalid tags in definition {name}")
        apis = [tag.removeprefix("api:") for tag in tags if tag.startswith("api:")]
        if not apis:
            raise SelectionError(f"Definition has no API tag: {name}")
        for api in apis:
            canonical = aliases.get(api)
            if canonical is None or rows[canonical]["operator"] != operator:
                raise SelectionError(f"Unmapped API/operator: {api} in {operator}")
            affected_apis.add(canonical)
        affected_definitions.add(name)

    try:
        plan.head = git.commit(head)
        plan.base = git.run("merge-base", git.commit(base), plan.head).decode().strip()
        # Treat moves as delete + add: both old and new ownership must be tested.
        plan.changed_paths = sorted(
            git.run("diff", "--name-only", "--no-renames", "-z", plan.base, plan.head, "--")
            .decode()
            .split("\0")[:-1]
        )
        broad_paths = []
        for path in plan.changed_paths:
            parts = PurePosixPath(path).parts
            if len(parts) >= 4 and parts[:2] == ("aklr", "ops"):
                operator = parts[2]
                if parts[3] in {"definitions", "impls"} and path.endswith(".json"):
                    for ref in (plan.base, plan.head):
                        data = git.read_json(ref, path)
                        if data is None:
                            continue  # Added or deleted on this side of the diff.
                        if parts[3] == "impls":
                            name = data.get("definition")
                            if not isinstance(name, str):
                                raise SelectionError(f"Solution needs definition: {path}")
                            definitions = git.definitions(ref, operator)
                            if name not in definitions:
                                raise SelectionError(f"Unknown definition {name}: {path}")
                            data = definitions[name]
                        add_definition(data, operator)
                else:
                    # Route/schema/API changes and non-JSON implementation sources
                    # affect every API in this operator, not only new definitions.
                    apis = {api for api, row in rows.items() if row["operator"] == operator}
                    if not apis:
                        raise SelectionError(f"Unmapped operator: {operator}")
                    affected_apis.update(apis)
                    for ref in (plan.base, plan.head):
                        for definition in git.definitions(ref, operator).values():
                            add_definition(definition, operator)
            elif path.endswith(".md") or path.startswith(("docs/", "aklr/tests/", ".github/scripts/tests/")):
                # The full AKLR suite and CPU CI-script tests remain separate gates.
                continue
            else:
                # Includes shared AKLR core/benchmark code, AITER, build/dependency
                # files, submodules, test helpers and the selector/map/workflow.
                broad_paths.append(path)
        plan.apis = sorted(affected_apis)
        plan.definitions = sorted(affected_definitions)
        if broad_paths:
            return plan.full(root, "Shared or unclassified changes: " + ", ".join(broad_paths))
    except (SelectionError, UnicodeError) as error:
        return plan.full(root, str(error))

    for suite in ("aiter", "triton"):
        setattr(
            plan,
            suite,
            sorted({test for api in plan.apis for test in rows[api][suite]}),
        )
    plan.reasons.append(
        "Mapped changed schema/solution APIs"
        if plan.apis
        else "No operator changes; AITER suites intentionally skipped"
    )
    return plan


def write_plan(plan: Plan, output_dir: Path) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, **asdict(plan)}
    (output_dir / "selection.json").write_text(json.dumps(payload, indent=2) + "\n")
    (output_dir / "affected-definitions.json").write_text(json.dumps(plan.definitions, indent=2) + "\n")
    for suite in ("aiter", "triton"):
        tests = getattr(plan, suite)
        # Empty means skip, not an implicit request to discover all tests.
        (output_dir / f"{suite}-selected.list").write_text("".join(test + "\n" for test in tests))
    # Escape untrusted Git paths/reasons inside a fenced JSON block.
    summary = (
        f"### AITER test selection: {plan.mode}\n\n"
        f"Operator files: **{len(plan.aiter)}**; Triton files: **{len(plan.triton)}**.\n\n"
        "Empty suites are intentionally skipped. This is file selection, not proof "
        "that a new kernel was dispatched; benchmark correctness remains required.\n\n"
        "```json\n" + json.dumps(payload, indent=2, ensure_ascii=True).replace("`", "\\u0060") + "\n```\n"
    )
    (output_dir / "summary.md").write_text(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", help="PR base revision; compare its merge-base with --head")
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--full", action="store_true", help="Force the full original suites")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--github-output", type=Path)
    parser.add_argument("--github-summary", type=Path)
    args = parser.parse_args()
    try:
        rows = load_map(ROOT, ROOT / MAP_PATH)
        plan = select_tests(ROOT, rows, base=args.base, head=args.head, full=args.full)
        summary = write_plan(plan, args.output_dir)
    except SelectionError as error:
        parser.exit(1, f"Test selection failed: {error}\n")
    print(summary)
    if args.github_output:
        with args.github_output.open("a") as output:
            output.write(
                f"mode={plan.mode}\naiter_count={len(plan.aiter)}\ntriton_count={len(plan.triton)}\n"
            )
    if args.github_summary:
        with args.github_summary.open("a") as output:
            output.write(summary + "\n")


if __name__ == "__main__":
    main()
