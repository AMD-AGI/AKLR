###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

import ast
import functools
import json
from collections.abc import Collection, Iterable, Iterator, Sequence
from itertools import starmap
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from pydantic import TypeAdapter, ValidationError

from aklr.core.schema import Schema, Tensor
from aklr.core.schema._collection import OPS_DIR, SchemaCollection
from aklr.core.validation.stages._audit._hip import check_hip_kernel
from aklr.core.validation.stages._audit._rules import SOURCE_RULES
from aklr.core.validation.stages._report._finding import Finding, FindingCode
from aklr.core.validation.stages._report._markdown import axes_text
from aklr.core.validation.stages.stage import PipelineStage, StageReport
from aklr.core.validation.stages.validation_scope import (
    SCHEMA_PATH_PATTERN,
    ValidationScope,
)

if TYPE_CHECKING:
    from aklr.core.schema._kernel import Kernel
    from aklr.core.validation.pipeline import PipelineReport

__all__ = ["AuditStage", "check_kernel"]


class AuditStage(PipelineStage):
    name = "audit"
    report_type = StageReport

    def __init__(self, *, allow_measured_edits: bool = False) -> None:
        # skips PERF_EDITED: for the bot's measurement PR only
        self._allow_measured_edits = allow_measured_edits

    def run(self, scope: ValidationScope, earlier: PipelineReport) -> StageReport:
        self.progress.write(f"checking {len(scope.schema_paths)} schemas")
        findings = []
        for path in scope.schema_paths:
            findings += self._schema_findings(path, scope)
        findings += self._collection_findings(scope)
        if scope.base_sha:
            self.progress.write(f"checking changes against base {scope.base_sha[:12]}")
            findings += self._change_findings(scope)
        return StageReport.from_findings(self.name, findings)

    def _collection_findings(self, scope: ValidationScope) -> list[Finding]:
        try:
            SchemaCollection.from_layout(scope.layout)
        except ValueError as e:
            return [Finding.unlocated(FindingCode.SCHEMA_INVALID, str(e))]
        return []

    def _schema_findings(self, path: Path, scope: ValidationScope) -> list[Finding]:
        root = scope.root.resolve()
        path = path.resolve()
        schema_path = path.relative_to(root).as_posix()

        def invalid(lines: Sequence[str]) -> list[Finding]:
            return [
                Finding(
                    code=FindingCode.SCHEMA_INVALID,
                    schema_path=schema_path,
                    axes=None,
                    detail=line,
                )
                for line in lines
            ]

        try:
            schema = TypeAdapter(Schema).validate_json(path.read_bytes())
        except ValidationError as e:
            return invalid([f"{'.'.join(map(str, d['loc']))}: {d['msg']}" for d in e.errors()])
        errors = []
        match = SCHEMA_PATH_PATTERN.fullmatch(schema_path)
        if match is None:
            errors.append(f"not at {OPS_DIR}/<op>/impls/<language>/<name>/<uuid>.json")
        elif match["name"] != schema.name:
            errors.append(f"the directory is not the name {schema.name}")
        elif schema.kernel is not None and match["language"] != schema.kernel.build.language:
            errors.append(f"not under impls/{schema.kernel.build.language}")
        if schema.kernel is not None and schema.init is None:
            errors.append("a schema with a kernel has no init to draw its inputs")
        seen = set()
        for w in schema.workloads:
            at = f"workload {{{axes_text(w.axes)}}}"
            if not set(w.axes) <= set(schema.var_axes):
                errors.append(f"{at}: axes are not of {schema.var_axes}")
            if tuple(w.axes.items()) in seen:
                errors.append(f"{at}: repeats")
            seen.add(tuple(w.axes.items()))
        if errors:
            return invalid(errors)
        if schema.kernel is None:
            return []
        if schema.kernel.build.language == "hip":
            family, rejections = "SH", check_hip_kernel(schema.kernel)
        else:
            call_keys = [*schema.inputs]
            if schema.kernel.build.destination_passing_style:
                call_keys += schema.outputs
            tensors = {**schema.inputs, **schema.outputs}
            family, rejections = (
                "SA",
                check_kernel(
                    schema.kernel,
                    call_keys=call_keys,
                    tensor_names=tuple(n for n, t in tensors.items() if isinstance(t, Tensor)),
                    output_names=tuple(schema.outputs),
                ),
            )
        return [
            Finding(
                code=FindingCode.SOURCE_REJECTED,
                schema_path=schema_path,
                axes=None,
                detail=f"{family}{rule} {path}:{line} {what}",
            )
            for path, line, rule, what in sorted(rejections)
        ]

    def _change_findings(self, scope: ValidationScope) -> list[Finding]:
        base_parts = functools.cache(lambda p: _schema_parts_of(scope.text_at_base(p)))
        root = scope.root.resolve()
        findings = []
        for path in scope.changed_paths():
            head_path = root / path
            if ValidationScope.path_kind(path) != "schema_data" or not head_path.exists():
                continue
            at_base = scope.paths_at_base(PurePosixPath(path).parent.as_posix())
            head = _schema_parts_of(head_path.read_text(encoding="utf-8", errors="replace"))
            if self._allow_measured_edits or head is None:
                continue
            # text_at_base raises on a new path; a new or unparsable base has no measurements
            base = base_parts(path) if path in at_base else None
            base_measured = base or {}
            edited = [k for k, evaluations in head.items() if evaluations != base_measured.get(k, [])]
            if edited:
                findings.append(
                    Finding(
                        code=FindingCode.PERF_EDITED,
                        schema_path=path,
                        axes=None,
                        detail=f"measured fields at {'; '.join(edited)} differ from"
                        " the base: only the bot writes evaluations",
                    )
                )
        return findings


def _schema_parts_of(text: str | None) -> dict[str, list] | None:
    if text is None:
        return None
    try:
        parsed = json.loads(text)
        return {axes_text(w["axes"]): w.get("evaluations", []) for w in parsed["workloads"]}
    except (ValueError, TypeError, KeyError):
        return None


_FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)
_SCOPE_NODES = (*_FUNCTION_NODES, ast.Lambda, ast.ClassDef)
_NAMED_NODES = (*_FUNCTION_NODES, ast.ClassDef, ast.ExceptHandler)
_DEVICE_DECORATORS = (
    *("flydsl.compiler.kernel", "flydsl.compiler.jit"),
    *("triton.jit", "triton.experimental.gluon.jit"),
)


def check_kernel(
    kernel: Kernel,
    *,
    call_keys: Collection[str],
    tensor_names: Collection[str],
    output_names: Collection[str],
) -> set[tuple[str, int, int, str]]:
    """Every SA0-SA4 rejection of a Python kernel, as (path, line, rule, what)."""
    try:
        kernel_sources = KernelSources(kernel, call_keys, tensor_names, output_names)
        parsed_sources = kernel_sources.parsed_sources
        reached = kernel_sources.reach()
        node_args = [
            (parsed_sources[path], *node_facts)
            for path, fn, input_names in reached
            for node_facts in parsed_sources[path].reached_nodes(fn, input_names)
        ]
        rejections = {
            (r.path, r.line, rule.number, r.what)
            for rule in SOURCE_RULES
            for rule_rejections in (
                rule.kernel_rejections(kernel_sources),
                *map(rule.source_rejections, parsed_sources.values()),
                *starmap(rule.node_rejections, node_args),
            )
            for r in rule_rejections
        }
    except RecursionError:
        rejections = {(kernel.get_entry_path().as_posix(), 0, 0, "nests too deeply to audit")}
    return rejections


def _bound_name(alias: ast.alias) -> str:
    return alias.asname or alias.name.partition(".")[0]


def _own_nodes(fn: ast.AST) -> Iterator[ast.AST]:
    for child in ast.iter_child_nodes(fn):
        yield child
        if not isinstance(child, _SCOPE_NODES):
            yield from _own_nodes(child)


def _bindings(nodes: Iterable[ast.AST]) -> frozenset[str]:
    names = set()
    for node in nodes:
        if isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            names.add(node.id)
        elif isinstance(node, (*_NAMED_NODES, ast.MatchAs, ast.MatchStar)):
            names.add(node.name)
        elif isinstance(node, ast.MatchMapping):
            names.add(node.rest)
        elif isinstance(node, ast.alias):
            names.add(_bound_name(node))
    names.discard(None)
    return frozenset(names)


def _own_and_class_nodes(scope: ast.AST) -> Iterator[ast.AST]:
    for node in _own_nodes(scope):
        yield node
        if isinstance(node, ast.ClassDef):
            yield from _own_and_class_nodes(node)


class KernelSources:
    def __init__(
        self,
        kernel: Kernel,
        call_keys: Collection[str],
        tensor_names: Collection[str],
        output_names: Collection[str],
    ) -> None:
        self.kernel = kernel
        self.call_keys = tuple(call_keys)
        self.tensor_names = frozenset(tensor_names)
        self.input_names = self.tensor_names - frozenset(output_names)
        self.trees: dict[str, ast.Module] = {}
        self.parse_errors: dict[str, tuple[int, str]] = {}
        for source in kernel.sources:
            try:
                self.trees[source.path] = ast.parse(source.content, filename=source.path)
            except (SyntaxError, ValueError, MemoryError) as e:
                self.parse_errors[source.path] = (
                    getattr(e, "lineno", 0) or 0,
                    str(getattr(e, "msg", e)),
                )
        self.top_level_names = {path: _bindings(_own_nodes(tree)) for path, tree in self.trees.items()}
        self.defs = {
            path: {s.name: s for s in tree.body if isinstance(s, _FUNCTION_NODES)}
            for path, tree in self.trees.items()
        }
        self.parsed_sources = {path: ParsedSource(self, path, tree) for path, tree in self.trees.items()}
        self.entry_path = kernel.get_entry_path().as_posix()
        self.entry = self.defs.get(self.entry_path, {}).get(kernel.get_entry_symbol())
        self.compile_functions: list[tuple[str, ast.FunctionDef]] = [
            (path, d)
            for path, source in self.parsed_sources.items()
            for d in self.defs[path].values()
            if d is not self.entry and source.is_compile_function(d)
        ]

    def resolve(self, node: ast.ImportFrom, name: str) -> tuple[str, str | None] | None:
        if node.level != 1:
            return None
        if node.module is None:
            path = f"{name}.py"
            return (path, None) if path in self.top_level_names else None
        path = f"{node.module}.py"
        return (path, name) if name in self.top_level_names.get(path, ()) else None

    def definition(
        self, path: str, name: str, seen: frozenset[tuple[str, str]] = frozenset()
    ) -> tuple[str, ast.FunctionDef] | None:
        if path not in self.parsed_sources or (path, name) in seen:
            return None
        if name in self.defs[path]:
            return path, self.defs[path][name]
        definition = self.parsed_sources[path].relative_imports.get(name)
        if definition is None or definition[1] is None:
            return None
        return self.definition(*definition, seen | {(path, name)})

    def function(self, path: str, node: ast.expr) -> tuple[str, ast.FunctionDef] | None:
        if isinstance(node, ast.Name):
            return self.definition(path, node.id)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            module = self.parsed_sources[path].relative_imports.get(node.value.id)
            if module is not None and module[1] is None:
                return self.definition(module[0], node.attr)
        return None

    def reach_roots(self) -> list[tuple[str, ast.FunctionDef | ast.Lambda]]:
        roots = [] if self.entry is None else [(self.entry_path, self.entry)]
        for path, compile_function in self.compile_functions:
            source = self.parsed_sources[path]
            roots += [
                (path, fn)
                for fn in _own_and_class_nodes(compile_function)
                if isinstance(fn, (*_FUNCTION_NODES, ast.Lambda))
            ]
            roots += self.named_functions(path, source.walk(compile_function))
        for path, tree in self.trees.items():
            module_nodes = list(_own_and_class_nodes(tree))
            roots += [
                (path, fn)
                for c in module_nodes
                if isinstance(c, ast.ClassDef)
                for fn in _own_and_class_nodes(c)
                if isinstance(fn, (*_FUNCTION_NODES, ast.Lambda))
            ]
            roots += self.named_functions(path, module_nodes)
        return roots

    def named_functions(self, path: str, nodes: Iterable[ast.AST]) -> list[tuple[str, ast.FunctionDef]]:
        callee_ids = self.parsed_sources[path].callee_ids
        return [
            definition
            for node in nodes
            if isinstance(node, (ast.Name, ast.Attribute))
            and isinstance(node.ctx, ast.Load)
            and id(node) not in callee_ids
            and (definition := self.function(path, node)) is not None
        ]

    def reach(self) -> list[tuple[str, ast.FunctionDef | ast.Lambda, frozenset[str]]]:
        if self.entry_path not in self.parsed_sources:
            return []
        todo = self.reach_roots()
        reached, enclosed = [], set()
        while todo:
            path, fn = todo.pop()
            source = self.parsed_sources[path]
            # device functions are compiled and compile functions run once per shape: not launch code
            if (
                (path, fn) in reached
                or source.is_device(fn)
                or (path, fn) in self.compile_functions
                or id(fn) in enclosed
            ):
                continue
            reached.append((path, fn))
            for node in source.walk(fn):
                if isinstance(node, _FUNCTION_NODES) and node is not fn:
                    enclosed.add(id(node))
                elif (
                    isinstance(node, (ast.Name, ast.Attribute))
                    and isinstance(node.ctx, ast.Load)
                    and (definition := self.function(path, node)) is not None
                ):
                    todo.append(definition)
        return [
            (
                path,
                fn,
                frozenset(a.arg for a in (*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs))
                & self.input_names,
            )
            for path, fn in reached
        ]


class ParsedSource:
    def __init__(self, kernel_sources: KernelSources, path: str, tree: ast.Module) -> None:
        self.path = path
        self.tree = tree
        self.imports: dict[str, str] = {}
        self.relative_imports: dict[str, tuple[str, str | None]] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    self.imports[_bound_name(a)] = a.name if a.asname else _bound_name(a)
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                for a in node.names:
                    self.imports[a.asname or a.name] = f"{node.module}.{a.name}"
            elif isinstance(node, ast.ImportFrom):
                for a in node.names:
                    definition = kernel_sources.resolve(node, a.name)
                    if definition is not None:
                        self.relative_imports[a.asname or a.name] = definition
        self.callee_ids: set[int] = {
            id(n.func)
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, (ast.Name, ast.Attribute))
        }

    def qualname(self, node: ast.AST) -> str | None:
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if not isinstance(node, ast.Name):
            return None
        return ".".join([self.imports.get(node.id, node.id), *reversed(parts)])

    def decorator_qualname(self, fn: ast.AST) -> str | None:
        # a second decorator (triton.autotune above triton.jit) makes the def host code
        if not isinstance(fn, _FUNCTION_NODES) or len(fn.decorator_list) != 1:
            return None
        (d,) = fn.decorator_list
        return self.qualname(d.func if isinstance(d, ast.Call) else d)

    def is_device(self, fn: ast.AST) -> bool:
        return self.decorator_qualname(fn) in _DEVICE_DECORATORS

    def is_compile_function(self, fn: ast.AST) -> bool:
        return self.decorator_qualname(fn) in ("functools.cache", "functools.lru_cache")

    def walk(self, fn: ast.AST) -> Iterator[ast.AST]:
        todo = [fn]
        while todo:
            node = todo.pop()
            yield node
            body = node.body if self.is_device(node) else ()
            todo += [c for c in ast.iter_child_nodes(node) if c not in body]

    def reached_nodes(
        self,
        scope: ast.FunctionDef | ast.Lambda | ast.ClassDef,
        input_names: frozenset[str],
    ) -> Iterator[tuple[ast.AST, frozenset[str]]]:
        own_names = _bindings(_own_nodes(scope))
        is_class = isinstance(scope, ast.ClassDef)
        # a class body's bindings shadow inputs in the body only, not in its methods
        visible_input_names = input_names - own_names if is_class else input_names
        for node in _own_nodes(scope):
            if isinstance(node, ast.ClassDef):
                yield from self.reached_nodes(node, input_names)
            elif isinstance(node, _SCOPE_NODES):
                if not self.is_device(node):
                    yield from self.reached_nodes(node, input_names - _bindings(_own_nodes(node)))
            else:
                yield node, visible_input_names
