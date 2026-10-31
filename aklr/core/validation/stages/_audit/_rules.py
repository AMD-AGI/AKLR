# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import ast
from collections.abc import Iterable
from typing import TYPE_CHECKING, ClassVar, Final, NamedTuple

if TYPE_CHECKING:
    from aklr.core.validation.stages._audit._stage import KernelSources, ParsedSource


class Rejection(NamedTuple):
    """One line a source rule rejects."""

    path: str
    line: int
    what: str

    @classmethod
    def from_node(cls, path: str, node: ast.AST, what: str) -> Rejection:
        return cls(path, getattr(node, "lineno", 0), what)


class SourceRule:
    """One SA rule over a kernel's sources."""

    number: ClassVar[int]

    def kernel_rejections(self, kernel_sources: KernelSources) -> Iterable[Rejection]:
        return ()

    def source_rejections(self, source: ParsedSource) -> Iterable[Rejection]:
        return ()

    def node_rejections(
        self,
        source: ParsedSource,
        node: ast.AST,
        input_names: frozenset[str],
    ) -> Iterable[Rejection]:
        return ()


def _source_text(node: ast.AST) -> str:
    text = ast.unparse(node).splitlines()[0]
    return f"`{text[:80]}`"


def root_name(node: ast.AST) -> str | None:
    while isinstance(node, (ast.Attribute, ast.Subscript, ast.Call)):
        node = node.func if isinstance(node, ast.Call) else node.value
    return node.id if isinstance(node, ast.Name) else None


class _ParseRule(SourceRule):
    number = 0

    def kernel_rejections(self, kernel_sources: KernelSources) -> Iterable[Rejection]:
        for path, (line, message) in kernel_sources.parse_errors.items():
            yield Rejection(path, line, f"does not parse: {message}")


class _GlobalStatementRule(SourceRule):
    number = 1

    def source_rejections(self, source: ParsedSource) -> Iterable[Rejection]:
        for node in ast.walk(source.tree):
            if isinstance(node, (ast.Global, ast.Nonlocal)):
                yield Rejection.from_node(source.path, node, _source_text(node))


# container mutators; ``copy_`` and every other non-dunder ``*_`` method too.
_MUTATORS = (
    *("append", "extend", "insert", "update", "setdefault", "pop", "clear", "add"),
    *("remove", "sort", "reverse", "popitem", "discard", "difference_update"),
    *("intersection_update", "symmetric_difference_update", "appendleft"),
    *("extendleft", "popleft"),
)


def _is_inplace_name(name: str) -> bool:
    return name.endswith("_") and not name.startswith("__")


class _InputWriteRule(SourceRule):
    number = 2

    def node_rejections(
        self,
        source: ParsedSource,
        node: ast.AST,
        input_names: frozenset[str],
    ) -> Iterable[Rejection]:
        # `a += x` runs the tensor's in-place `__iadd__`
        if (
            isinstance(node, ast.AugAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id in input_names
        ):
            what = f"writes input {node.target.id} {_source_text(node)}"
            yield Rejection.from_node(source.path, node, what)
        if isinstance(node, ast.Call):
            keywords = {k.arg: k.value for k in node.keywords}
            written = [keywords["out"]] if "out" in keywords else []
            inplace = keywords.get("inplace")
            qualname = source.qualname(node.func) or ""
            # `torch.relu_(x)` and `torch.ops.aten.fill_.Scalar(x)` write their
            # first argument, not the module
            if not (inplace is None or isinstance(inplace, ast.Constant) and inplace.value is False) or (
                root_name(node.func) in source.imports and any(map(_is_inplace_name, qualname.split(".")))
            ):
                written += node.args[:1]
            for w in written:
                for e in w.elts if isinstance(w, (ast.Tuple, ast.List)) else (w,):
                    root = root_name(e)
                    if root in input_names:
                        what = f"writes input {root} {_source_text(node)}"
                        yield Rejection.from_node(source.path, node, what)
        if isinstance(node, (ast.Subscript, ast.Attribute)) and not isinstance(node.ctx, ast.Load):
            receiver = node.value
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            attr, receiver = node.func.attr, node.func.value
            if not (attr in _MUTATORS or _is_inplace_name(attr)):
                return
        else:
            return
        root = root_name(receiver)
        if root in input_names:
            what = f"writes input {root} {_source_text(node)}"
            yield Rejection.from_node(source.path, node, what)


# in R: synchronization and timing; allocating workspace is fine.
_SYNCHRONIZATION_AND_TIMING = (
    *("synchronize", "item", "cpu", "tolist", "numpy"),
    *("Event", "elapsed_time", "perf_counter"),
)
# draw from the global RNG or copy host data: not workspace
_RNG_AND_HOST_FACTORIES = ("rand", "randn", "randint", "tensor", "as_tensor")
_INPUT_OUTPUT_CALLS = ("open", "print", "input", "breakpoint", "exit", "quit")
_FILE_CALLS = ("torch.save", "torch.load")


# process-wide torch state: precision, RNG, default dtype or device, current stream
_GLOBAL_STATE_CALLS = (
    *("torch.set_", "torch.cuda.set_", "torch.use_deterministic_algorithms"),
    *("torch.manual_seed", "torch.seed", "torch.cuda.manual_seed", "torch.cuda.seed"),
    "torch.cuda.stream",
)


class _SideEffectRule(SourceRule):
    number = 3

    def source_rejections(self, source: ParsedSource) -> Iterable[Rejection]:
        for node in ast.walk(source.tree):
            if isinstance(node, ast.Call):
                qualname = source.qualname(node.func) or ""
                if qualname.startswith(_GLOBAL_STATE_CALLS):
                    what = f"process-wide torch state {qualname}"
                    yield Rejection.from_node(source.path, node, what)
            if not isinstance(node, (ast.Name, ast.Attribute)):
                continue
            qualname = source.qualname(node)
            if qualname in _FILE_CALLS or getattr(node, "id", None) in _INPUT_OUTPUT_CALLS:
                yield Rejection.from_node(source.path, node, f"I/O {qualname}")

    def node_rejections(
        self,
        source: ParsedSource,
        node: ast.AST,
        input_names: frozenset[str],
    ) -> Iterable[Rejection]:
        if isinstance(node, ast.Attribute):
            attr, qualname = node.attr, source.qualname(node) or ""
            if attr in _SYNCHRONIZATION_AND_TIMING:
                yield Rejection.from_node(source.path, node, f"{attr} in launch")
            elif qualname.startswith("torch.") and attr in _RNG_AND_HOST_FACTORIES:
                yield Rejection.from_node(source.path, node, f"{qualname} in launch")
        elif isinstance(node, ast.Name) and node.id in source.imports:
            qualname = source.qualname(node)
            if (
                node.id in _SYNCHRONIZATION_AND_TIMING
                or qualname.startswith("torch.")
                and qualname.rpartition(".")[2] in _RNG_AND_HOST_FACTORIES
            ):
                yield Rejection.from_node(source.path, node, f"{qualname} in launch")


class _EntryRule(SourceRule):
    number = 4

    def kernel_rejections(self, kernel_sources: KernelSources) -> Iterable[Rejection]:
        entry_path = kernel_sources.entry_path
        if entry_path not in kernel_sources.parsed_sources:
            return
        tree = kernel_sources.parsed_sources[entry_path].tree
        symbol = kernel_sources.kernel.get_entry_symbol()
        if not ast.get_docstring(tree):
            yield Rejection.from_node(entry_path, tree, "no module docstring")
        entry = kernel_sources.entry
        if entry is None:
            yield Rejection.from_node(entry_path, tree, f"no module-level def {symbol}")
            return
        args, keys = entry.args, kernel_sources.call_keys
        names = [a.arg for a in args.args]
        accepted = names == list(keys) and not (
            args.posonlyargs or args.kwonlyargs or args.vararg or args.kwarg
        )
        what = f"must take exactly the parameters {', '.join(keys)}"
        if not accepted:
            yield Rejection.from_node(entry_path, entry, f"{symbol} {what}")


SOURCE_RULES: Final = (
    _ParseRule(),
    _GlobalStatementRule(),
    _InputWriteRule(),
    _SideEffectRule(),
    _EntryRule(),
)
