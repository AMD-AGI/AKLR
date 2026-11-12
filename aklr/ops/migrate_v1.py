###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

import argparse
import ast
import importlib
import io
import json
import pkgutil
import re
import sys
import tempfile
import tokenize
from dataclasses import replace
from functools import cached_property
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from pydantic import TypeAdapter

import aklr.ops
from aklr.core.schema import Const, Scalar, Schema, Tensor, Var
from aklr.core.schema._collection import DefaultLayout, SchemaCollection
from aklr.core.schema._kernel import Build, Kernel, SourceFile
from aklr.core.schema._workload import Workload

OPS_DIR = Path(aklr.ops.__file__).parent
SCHEMA_ADAPTER = TypeAdapter(Schema)
CONSTRAINT_FUNCTIONS = {"ceil_div": lambda a, b: -(-a // b), "min": min, "max": max}
LEGACY_TOP_KEYS = {
    "schema_version",
    "name",
    "description",
    "op_type",
    "axes",
    "inputs",
    "attrs",
    "outputs",
    "workloads",
    "compare",
    "kernel",
}
LEGACY_KERNEL_KEYS = {
    "language",
    "targets",
    "requires",
    "author",
    "entry_point",
    "sources",
}
LEGACY_WORKLOAD_KEYS = {"axes", "scalars", "accuracy", "speedup"}
LEGACY_ARG_KEYS = {
    "shape",
    "dtype",
    "description",
    "distribution",
    "optional",
    "strides",
}
LEGACY_DTYPES = {"float4_e2m1": "float4_e2m1fn_x2", "float8_e8m0": "float8_e8m0fnu"}


class MigrationRefused(Exception):
    pass


class LegacyKernelEntry:
    def __init__(
        self,
        source: str,
        symbol: str,
        params: list[str],
        bindings: dict[str, Any],
    ) -> None:
        self.source = source
        self.symbol = symbol
        self.params = params
        self.bindings = bindings

    def function(self, tree: ast.Module) -> ast.FunctionDef:
        found = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == self.symbol]
        if len(found) != 1:
            raise MigrationRefused(f"entry '{self.symbol}' is defined {len(found)} times")
        return found[0]

    def binding_expression(self, name: str) -> str:
        if name == "out_dtype":
            if "out" not in self.params:
                raise MigrationRefused("out_dtype is bound to out.dtype, but out is no parameter")
            return "out.dtype"
        value = self.bindings[name]
        if value is not None and not isinstance(value, (bool, int, float, str)):
            raise MigrationRefused(f"attr {name}={value!r} is no literal")
        text = repr(value)
        if ast.literal_eval(text) != value or type(ast.literal_eval(text)) is not type(value):
            raise MigrationRefused(f"attr {name}={value!r} does not round-trip as a literal")
        return text

    def rewrite(self, old_keys: list[str]) -> str:
        tree = ast.parse(self.source)
        fn = self.function(tree)
        args = fn.args
        if args.posonlyargs or args.args or args.vararg or args.kwarg:
            raise MigrationRefused(f"'{self.symbol}' is not keyword-only")
        if any(d is not None for d in args.kw_defaults):
            raise MigrationRefused(f"'{self.symbol}' has keyword defaults")
        if any(a.annotation is not None for a in args.kwonlyargs):
            raise MigrationRefused(f"'{self.symbol}' has annotated parameters")
        if [a.arg for a in args.kwonlyargs] != old_keys:
            raise MigrationRefused(f"'{self.symbol}' parameters changed while rewriting")

        lines = self.source.splitlines(keepends=True)
        offsets = [0]
        for line in lines:
            offsets.append(offsets[-1] + len(line))
        tokens = list(tokenize.generate_tokens(io.StringIO(self.source).readline))
        start = next(
            i
            for i, t in enumerate(tokens)
            if t.start[0] == fn.lineno
            and t.string == "def"
            and tokens[i + 1].string == self.symbol
            and tokens[i + 2].string == "("
        )
        depth = 0
        for close in range(start + 2, len(tokens)):
            if tokens[close].type == tokenize.OP and tokens[close].string in "([{":
                depth += 1
            elif tokens[close].type == tokenize.OP and tokens[close].string in ")]}":
                depth -= 1
                if depth == 0:
                    break
        open_tok, close_tok = tokens[start + 2], tokens[close]
        head_end = offsets[open_tok.start[0] - 1] + open_tok.start[1]
        tail_start = offsets[close_tok.end[0] - 1] + close_tok.end[1]
        line_start = offsets[fn.lineno - 1]
        line_end = self.source.find("\n", tail_start)
        one_line = "(" + ", ".join(self.params) + ")"
        def_indent = " " * fn.col_offset
        # 88: black's default line length
        if len(self.source[line_start:head_end] + one_line + self.source[tail_start:line_end]) <= 88:
            signature = one_line
        else:
            signature = "(\n" + "".join(f"{def_indent}    {p},\n" for p in self.params) + f"{def_indent})"

        first = fn.body[0]
        has_docstring = (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        )
        anchor = fn.body[1] if has_docstring and len(fn.body) > 1 else first
        if anchor.lineno <= close_tok.end[0] or (has_docstring and anchor is first):
            raise MigrationRefused(f"'{self.symbol}' body does not start on its own line")
        removed = [k for k in old_keys if k not in self.params]
        body_indent = " " * anchor.col_offset
        anchor_line = min([anchor.lineno] + [d.lineno for d in getattr(anchor, "decorator_list", [])])
        insert_at = offsets[anchor_line - 1]
        inserted = "".join(f"{body_indent}{k} = {self.binding_expression(k)}\n" for k in removed)
        new_source = (
            self.source[:head_end]
            + signature
            + self.source[tail_start:insert_at]
            + inserted
            + self.source[insert_at:]
        )
        self.verify(tree, fn, new_source, removed, has_docstring)
        return new_source

    def verify(
        self,
        old_tree: ast.Module,
        old_fn: ast.FunctionDef,
        new_source: str,
        removed: list[str],
        has_docstring: bool,
    ) -> None:
        new_tree = ast.parse(new_source)
        new_fn = self.function(new_tree)
        a = new_fn.args
        if (
            a.posonlyargs
            or a.kwonlyargs
            or a.vararg
            or a.kwarg
            or a.defaults
            or [x.arg for x in a.args] != self.params
        ):
            raise MigrationRefused("rewritten signature is not the positional parameters")
        skip = 1 if has_docstring else 0
        new_body = new_fn.body[:skip] + new_fn.body[skip + len(removed) :]
        assigns = new_fn.body[skip : skip + len(removed)]
        if [ast.dump(s) for s in new_body] != [ast.dump(s) for s in old_fn.body]:
            raise MigrationRefused("rewritten body differs beyond the bindings")
        for name, stmt in zip(removed, assigns):
            if not (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and stmt.targets[0].id == name
                and ast.unparse(stmt.value) == self.binding_expression(name)
            ):
                raise MigrationRefused(f"binding of {name} was not written as expected")
        if [ast.dump(d) for d in new_fn.decorator_list] != [
            ast.dump(d) for d in old_fn.decorator_list
        ] or ast.dump(new_fn.returns or ast.Pass()) != ast.dump(old_fn.returns or ast.Pass()):
            raise MigrationRefused("rewritten decorators or return annotation differ")
        old_rest = [ast.dump(n) for n in old_tree.body if n is not old_fn]
        new_rest = [ast.dump(n) for n in new_tree.body if n is not new_fn]
        if old_rest != new_rest:
            raise MigrationRefused("rewriting changed code outside the entry function")
        compile(new_source, "<rewritten>", "exec")


class LegacySchemaFile:
    def __init__(self, path: Path, schemas: list[Schema]) -> None:
        self.path = path
        self.data = json.loads(path.read_text())
        for side in ("inputs", "outputs"):
            for spec in self.data.get(side, {}).values():
                if spec.get("dtype") in LEGACY_DTYPES:
                    spec["dtype"] = LEGACY_DTYPES[spec["dtype"]]
        self.schemas = schemas
        self.dropped: list[str] = []

    @property
    def is_legacy(self) -> bool:
        return "schema_version" in self.data

    @cached_property
    def old_consts(self) -> dict[str, int]:
        return {k: a["value"] for k, a in self.data["axes"].items() if a["type"] == "const"}

    @cached_property
    def used_axes(self) -> set[str]:
        return {
            d
            for side in ("inputs", "outputs")
            for n, spec in getattr(self.python_schema, side).items()
            if n in self.data[side] and isinstance(spec, Tensor)
            for d in spec.dim_names
        }

    @cached_property
    def python_schema(self) -> Schema:
        hits = [
            s
            for s in self.schemas
            if s.op_type == self.data["op_type"]
            and s.schema_name({k: v for k, v in self.old_consts.items() if isinstance(s.axes.get(k), Const)})
            == self.data["name"]
        ]
        if len(hits) != 1:
            raise MigrationRefused(f"{len(hits)} python schemas are named {self.data['name']}")
        return hits[0]

    def check_known_keys(self) -> None:
        d = self.data
        if d["schema_version"] != 1:
            raise MigrationRefused(f"schema_version {d['schema_version']} is not 1")
        unknown = set(d) - LEGACY_TOP_KEYS
        unknown |= {f"kernel.{k}" for k in set(d["kernel"]) - LEGACY_KERNEL_KEYS}
        unknown |= {f"workloads.{k}" for w in d["workloads"] for k in set(w) - LEGACY_WORKLOAD_KEYS}
        unknown |= {
            f"{side}.{n}.{k}"
            for side in ("inputs", "outputs")
            for n, a in d[side].items()
            for k in set(a) - LEGACY_ARG_KEYS
        }
        if unknown:
            raise MigrationRefused(f"unknown keys {sorted(unknown)}")

    def axes(self) -> dict[str, Var | Const]:
        s, old = self.python_schema, self.data["axes"]
        extra = set(old) - set(s.axes)
        if extra:
            raise MigrationRefused(f"axes {sorted(extra)} are not in {s.op_type}'s python schema")
        env: dict[str, Any] = dict(CONSTRAINT_FUNCTIONS) | self.old_consts
        derived: dict[str, int] = {}
        for c in s.constraints:
            node = ast.parse(c, mode="eval").body
            if (
                isinstance(node, ast.Compare)
                and len(node.ops) == 1
                and isinstance(node.ops[0], ast.Eq)
                and isinstance(node.left, ast.Name)
                and node.left.id not in env
                and isinstance(s.axes.get(node.left.id), Const)
                and {n.id for n in ast.walk(node.comparators[0]) if isinstance(n, ast.Name)} <= set(env)
            ):
                derived[node.left.id] = env[node.left.id] = eval(
                    compile(ast.Expression(node.comparators[0]), c, "eval"),
                    {"__builtins__": {}},
                    env,
                )
        axes: dict[str, Var | Const] = {}
        missing = []
        for name, axis in s.axes.items():
            o = old.get(name)
            if isinstance(axis, Var):
                if (o is None or o["type"] != "var") and name in self.used_axes:
                    missing.append(f"var {name}")
                axes[name] = axis
                continue
            if o is not None and o["type"] != "const":
                raise MigrationRefused(f"axis {name} is var in the file, const in python")
            value = o["value"] if o is not None else axis.value
            if axis.value is not None and value != axis.value:
                raise MigrationRefused(f"const {name}={value} but python fixes {axis.value}")
            if value is None:
                value = derived.get(name)
            if value is None:
                if name in self.used_axes:
                    missing.append(f"const {name}")
                axes[name] = axis
                continue
            axes[name] = replace(axis, value=value)
        if missing:
            raise MigrationRefused(f"no value for axes {missing}")
        return axes

    def args(self, side: str, axes: dict[str, Var | Const]) -> dict[str, Tensor | Scalar]:
        declared: dict[str, Tensor | Scalar] = getattr(self.python_schema, side)
        old = self.data[side]
        args: dict[str, Tensor | Scalar] = {}
        for name, spec in declared.items():
            o = old.get(name)
            if o is None:
                if not spec.optional:
                    raise MigrationRefused(f"required {side[:-1]} {name} is absent from the file")
                args[name] = spec
                continue
            shape = spec.dim_names if isinstance(spec, Tensor) else None
            if (shape is None) != (o["shape"] is None) or (
                shape is not None
                and (
                    len(shape) != len(o["shape"])
                    or not all(
                        od == d
                        or (isinstance(axes[d], Const) and axes[d].value == od)
                        # v1 sized an fp4x2 last dim in fp4 values, the schema in packed pairs
                        or (
                            o["dtype"] == "float4_e2m1fn_x2"
                            and i == len(shape) - 1
                            and isinstance(axes[d], Const)
                            and 2 * axes[d].value == self.old_consts.get(od)
                        )
                        for i, (d, od) in enumerate(zip(shape, o["shape"]))
                    )
                )
            ):
                raise MigrationRefused(f"{name} shape {o['shape']} != python {shape}")
            if spec.dtype is not None and spec.dtype != o["dtype"]:
                raise MigrationRefused(f"{name} dtype {o['dtype']} != python {spec.dtype}")
            if isinstance(spec, Tensor) and spec.dtype is None:
                spec = replace(spec, dtype=o["dtype"])
            for key in ("distribution", "strides"):
                if key in o:
                    self.dropped.append(f"{side}.{name}.{key}")
            args[name] = spec
        return args

    def removed_attrs(self) -> dict[str, Any]:
        s = self.python_schema
        attrs = self.data["attrs"]
        old_inputs = set(self.data["inputs"])
        extra_inputs = old_inputs - set(s.inputs)
        if extra_inputs - set(attrs):
            raise MigrationRefused(
                f"inputs {sorted(extra_inputs - set(attrs))} left the python schema and are no attrs"
            )
        if set(self.data["outputs"]) != set(s.outputs):
            raise MigrationRefused("outputs differ from the python schema")
        consts = {k for k, a in s.axes.items() if isinstance(a, Const)}
        pins: dict[str, str] = {}
        for c in s.constraints:
            pins.update(self.pins(ast.parse(c, mode="eval").body, consts))
        unpinned = sorted(
            k
            for k, v in attrs.items()
            if not (pins.get(k) == "eq" or (pins.get(k) == "truth" and isinstance(v, bool)))
        )
        refused = [k for k in unpinned if k not in s.inputs]
        if refused:
            raise MigrationRefused(f"attrs {refused} are pinned to one value by no python constraint")
        self.dropped += [f"attrs.{k}={attrs[k]!r}" for k in unpinned]
        return {k: v for k, v in attrs.items() if k not in s.inputs}

    def pins(self, node: ast.expr, consts: set[str]) -> dict[str, str]:
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
            return {k: v for value in node.values for k, v in self.pins(value, consts).items()}
        if isinstance(node, ast.Name):
            return {node.id: "truth"}
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return {node.operand.id: "truth"} if isinstance(node.operand, ast.Name) else {}
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], (ast.Eq, ast.Is)):
            sides = [node.left, node.comparators[0]]
            fixed = [
                isinstance(x, ast.Constant) or (isinstance(x, ast.Name) and x.id in consts) for x in sides
            ]
            return {
                x.id: "eq"
                for x, other_fixed in zip(sides, reversed(fixed))
                if isinstance(x, ast.Name) and other_fixed
            }
        return {}

    def check_constraints(self, axes: dict[str, Var | Const]) -> None:
        s = self.python_schema
        env: dict[str, Any] = dict(CONSTRAINT_FUNCTIONS)
        env.update(self.data["attrs"])
        env.update({k: a.value for k, a in axes.items() if isinstance(a, Const)})
        for side in ("inputs", "outputs"):
            for name, spec in getattr(s, side).items():
                o = self.data[side].get(name)
                if o is None:
                    env[name] = None
                elif isinstance(spec, Tensor):
                    env[name] = SimpleNamespace(dtype=o["dtype"])
        for c in s.constraints:
            tree = ast.parse(c, mode="eval")
            names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
            if names - set(env):
                continue
            if not eval(compile(tree, c, "eval"), {"__builtins__": {}}, env):
                raise MigrationRefused(f"constraint {c!r} fails on this file")

    def workloads(self) -> list[Workload]:
        var_axes = set(self.python_schema.var_axes) & self.used_axes
        out = []
        for w in self.data["workloads"]:
            if set(w["axes"]) != var_axes:
                raise MigrationRefused(f"workload axes {sorted(w['axes'])} != var axes {sorted(var_axes)}")
            if w.get("scalars"):
                self.dropped.append("workloads.scalars " + json.dumps(w["scalars"], sort_keys=True))
            if w.get("accuracy") is not None or w.get("speedup"):
                self.dropped.append(
                    "workloads.accuracy/speedup without hardware "
                    + json.dumps([w.get("accuracy"), w.get("speedup")])
                )
            out.append(Workload(axes=w["axes"]))
        return out

    def kernel(self, removed: dict[str, Any]) -> Kernel | None:
        k = self.data["kernel"]
        if not k["sources"]:
            return None
        if k["requires"]:
            raise MigrationRefused(f"kernel.requires {k['requires']} has no place in the new model")
        s = self.python_schema
        params = list(s.inputs) + list(s.outputs)
        path, symbol = k["entry_point"].split("::")
        sources = {f["path"]: f["content"] for f in k["sources"]}
        if len(sources) != len(k["sources"]) or path not in sources:
            raise MigrationRefused("kernel sources have duplicate paths or miss the entry file")
        if k["language"] == "hip":
            defs = re.findall(rf"\b{re.escape(symbol)}\s*\(([^()]*)\)\s*\{{", sources[path])
            if len(defs) != 1:
                raise MigrationRefused(f"hip entry '{symbol}' is defined {len(defs)} times")
            names = [re.findall(r"\w+", p)[-1] for p in defs[0].split(",")]
            if names != params:
                raise MigrationRefused(f"hip entry parameters {names} != {params}")
        else:
            entry = LegacyKernelEntry(sources[path], symbol, params, removed)
            fn = entry.function(ast.parse(sources[path]))
            old_keys = [a.arg for a in fn.args.kwonlyargs]
            unknown = set(old_keys) - set(params) - set(removed) - {"out_dtype"}
            if unknown:
                raise MigrationRefused(
                    f"entry takes {sorted(unknown)}, which are no inputs, outputs or attrs"
                )
            new_only = set(params) - set(old_keys)
            if new_only:
                raise MigrationRefused(f"entry lacks the new parameters {sorted(new_only)}")
            sources[path] = entry.rewrite(old_keys)
        targets = []
        for t in k["targets"]:
            if not t.get("arch"):
                raise MigrationRefused(f"target {t} has no arch")
            target = "/".join(filter(None, (t["arch"], t.get("hardware"))))
            if target not in targets:
                targets.append(target)
        if not k["author"]:
            self.dropped.append("kernel.author '' -> 'unknown'")
        return Kernel(
            name=f"{self.data['name']}_{k['language']}_{self.path.stem}",
            author=k["author"] or "unknown",
            build=Build(
                language=k["language"],
                target_hardware=targets,
                entry_point=k["entry_point"],
            ),
            sources=[SourceFile(path=p, content=c) for p, c in sources.items()],
        )

    def convert(self) -> str:
        self.check_known_keys()
        for key in ("schema_version", "compare"):
            if self.data.get(key):
                self.dropped.append(key)
        axes = self.axes()
        inputs, outputs = self.args("inputs", axes), self.args("outputs", axes)
        removed = self.removed_attrs()
        self.check_constraints(axes)
        schema = replace(
            self.python_schema,
            name=self.data["name"],
            axes=axes,
            inputs=inputs,
            outputs=outputs,
            workloads=self.workloads(),
            kernel=self.kernel(removed),
        )
        text = json.dumps(SCHEMA_ADAPTER.dump_python(schema, mode="json"), indent=2) + "\n"
        if SCHEMA_ADAPTER.dump_python(SCHEMA_ADAPTER.validate_json(text), mode="json") != json.loads(text):
            raise MigrationRefused("the written json does not load back to the same schema")
        return text


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m aklr.ops.migrate_v1")
    parser.add_argument("root", nargs="?", type=Path, default=OPS_DIR)
    parser.add_argument("--write", action="store_true")
    ns = parser.parse_args()

    schemas: dict[int, Schema] = {}
    for info in pkgutil.iter_modules(aklr.ops.__path__):
        if info.ispkg:
            module = importlib.import_module(f"aklr.ops.{info.name}.schema")
            schemas.update({id(v): v for v in vars(module).values() if isinstance(v, Schema)})

    converted: dict[Path, str] = {}
    refused = 0
    for path in sorted(ns.root.glob("*/impls/*/*/*.json")):
        f = LegacySchemaFile(path, list(schemas.values()))
        if not f.is_legacy:
            continue
        rel = path.relative_to(ns.root)
        try:
            converted[rel] = f.convert()
        except MigrationRefused as e:
            refused += 1
            print(f"REFUSED {rel}: {e}")
            continue
        print(f"OK      {rel}" + "".join(f"\n          dropped {d}" for d in f.dropped))

    # the converted files must load together as a collection before --write touches the tree
    with tempfile.TemporaryDirectory() as tmp:
        for rel, text in converted.items():
            (Path(tmp) / rel).parent.mkdir(parents=True, exist_ok=True)
            (Path(tmp) / rel).write_text(text)
        SchemaCollection.from_layout(DefaultLayout(Path(tmp)))

    if ns.write:
        for rel, text in converted.items():
            (ns.root / rel).write_text(text)
    print(
        f"{len(converted)} converted{' and written' if ns.write else ' (dry run, --write to apply)'},"
        f" {refused} refused"
    )
    return 1 if refused else 0


if __name__ == "__main__":
    sys.exit(main())
