# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Final, NamedTuple

from pydantic import PositiveInt, TypeAdapter, ValidationError, field_validator, model_validator

from aklr.core.contract import locate, op, semantics, spec
from aklr.core.contract.registry import ContractRegistry, UnknownContractError
from aklr.core.contract.spec import Language, WorkloadSpec, axes_text
from aklr.core.contract.spec._sources import write_bytes_atomically
from aklr.core.platform.device import DeviceTarget

__all__ = [
    "CallRecord",
    "SchemaCalls",
    "export_schemas",
    "main",
    "read_calls",
    "repository_errors",
    "write_calls",
]

# The aklr package directory: export never writes into it.
_AKLR_DIR: Final = Path(__file__).resolve().parents[2]
_CALLS_PATTERN: Final = "calls.pid*.jsonl"
# Once per process: a restarted container reuses pids, and every save rewrites one file.
_RUN_ID: Final = uuid.uuid4().hex[:8]


def _registered_op_packages() -> list[str]:
    return sorted({c.op_package for c in ContractRegistry.default().contracts()})


class CallRecord(spec.FrozenModel):
    """One counted kind of call: the facts the runtime saw, never a verdict."""

    contract_id: spec.ContractId
    point: spec.PointSpec | None  # None: out of contract
    scalars: dict[str, int | float | bool]  # only the scalars the call gave
    fallback_reason: str | None  # None: a kernel served it; else a FallbackReason value
    target: DeviceTarget  # the device the calls ran on
    n_calls: PositiveInt

    @field_validator("target")
    @classmethod
    def _check_complete_target(cls, target: DeviceTarget) -> DeviceTarget:
        if not target.is_complete:
            raise ValueError("a call record's target names both arch and hardware")
        return target

    @model_validator(mode="after")
    def _check_point_contract(self) -> CallRecord:
        if self.point is not None and self.point.contract_id != self.contract_id:
            raise ValueError(f"point.contract_id {self.point.contract_id!r} is not {self.contract_id!r}")
        return self

    def line(self) -> str:
        return semantics.sorted_key_json(self.model_dump(mode="json"))


def write_calls(export_dir: Path, calls: Iterable[CallRecord]) -> Path:
    """Replace this process's call file by a whole snapshot of ``calls``, one sorted
    line each (no calls: an empty file); the path."""
    path = export_dir / f"calls.pid{os.getpid()}.{_RUN_ID}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(f"{line}\n" for line in sorted(map(CallRecord.line, calls)))
    write_bytes_atomically(path, text.encode())
    return path


def read_calls(directory: Path) -> list[CallRecord]:
    """Every call file in ``directory`` merged (counts of otherwise equal records
    added), sorted by line text. Raises one ``ValueError`` listing every bad line."""
    merged: dict[str, CallRecord] = {}  # by the line of the record with n_calls 1
    errors = []
    for path in sorted(directory.glob(_CALLS_PATTERN)):
        text = path.read_text(encoding="utf-8")
        lines = text.removesuffix("\n").split("\n") if text else []
        for number, line in enumerate(lines, 1):
            try:
                record = CallRecord.model_validate_json(line)
            except ValidationError as e:
                errors += op.OpContract.error_lines(f"{path}:{number}", None, e)
                continue
            key = record.model_copy(update={"n_calls": 1}).line()
            if (seen := merged.get(key)) is not None:
                record = seen.model_copy(update={"n_calls": seen.n_calls + record.n_calls})
            merged[key] = record
    if errors:
        raise ValueError("\n".join(errors))
    return sorted(merged.values(), key=CallRecord.line)


class SchemaCalls(NamedTuple):
    """The calls at the Points of one schema file: added, or committed and serving."""

    # <op>/impls/<language>/<schema_name>/<uuid>.json, under the export dir or aklr/ops/
    relative_path: PurePosixPath
    is_added: bool
    # (axes_text, n_calls): most calls first, then by axes_text
    calls: tuple[tuple[str, int], ...]
    # served: call targets kernel.targets does not match, sorted; added: ()
    mismatched_targets: tuple[DeviceTarget, ...]
    # served: the file's models do not list the export's model; added: False
    lacks_model: bool


class _ServingSchema(NamedTuple):
    relative_path: PurePosixPath  # <op>/impls/<language>/<schema_name>/<uuid>.json
    targets: tuple[DeviceTarget, ...]  # its kernel.targets
    models: tuple[str, ...]


class _ServedCalls(NamedTuple):
    n_calls_by_axes: Counter[str]  # by axes_text
    mismatched_targets: set[DeviceTarget]  # the schema's targets do not match


class _PendingSchema(NamedTuple):
    contract: op.OpContract
    point: spec.PointSpec  # the first call's: the Points of a name share one schema
    workloads: dict[str, WorkloadSpec]  # by sorted_key_json: one per (axes, scalars)
    n_calls_by_workload: Counter[str]  # by the same key
    targets: set[DeviceTarget]  # of the calls


# (op_package, schema name, frozenset(axes.items())): a Point, whatever the axis order
_PointKey = tuple[str, str, frozenset[tuple[str, int]]]


def _sorted_calls(n_calls_by_axes: Mapping[str, int]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted(n_calls_by_axes.items(), key=lambda item: (-item[1], item[0])))


def export_schemas(
    calls_dir: Path | None,
    records: Iterable[CallRecord],
    export_dir: Path,
    *,
    language: Language,
    model: str,
) -> list[SchemaCalls]:
    """Write into ``export_dir`` a ``language`` schema listing ``model`` (whose serving
    made the calls) for each schema name whose called Points no committed schema serves;
    every schema file the calls reach, sorted by path."""
    if export_dir.resolve().is_relative_to(_AKLR_DIR):
        raise ValueError(f"{export_dir}: inside the aklr source tree; export never writes the repository")
    errors: list[str] = []
    # every committed schema file, of any language, by the Points it serves
    serving: dict[_PointKey, _ServingSchema] = {}
    for pkg in _registered_op_packages():
        op_dir = locate.op_dir(pkg)
        for path in locate.committed_schema_paths(op_dir):
            try:
                s = ContractRegistry.default().parse_schema(path.read_bytes(), str(path))
            except OSError as e:
                errors.append(f"{path}: {e.strerror or e}")
                continue
            except ValueError as e:
                errors.append(str(e))
                continue
            serving_schema = _ServingSchema(
                PurePosixPath(path.relative_to(op_dir.parent)),
                s.kernel.targets,
                s.models,
            )
            for w in s.workloads:
                serving.setdefault((pkg, s.name, frozenset(w.axes.items())), serving_schema)
    pending: dict[tuple[str, str], _PendingSchema] = {}
    served: dict[_ServingSchema, _ServedCalls] = {}
    for record in records:
        if (point := record.point) is None:
            continue
        try:
            contract = ContractRegistry.default().get(record.contract_id)
        except UnknownContractError:
            errors.append(f"{calls_dir}: {record.contract_id}: unknown contract")
            continue
        try:
            contract.check_point(point)
            # no kernel serves a Point outside the domain, such as an idle rank's call of no rows
            if not contract.domain.contains(point):
                continue
            workload = contract.workload_for(point, record.scalars)
        except ValueError as e:
            errors.append(f"{calls_dir}: {point.key()}: {e}")
            continue
        pkg, name = contract.op_package, contract.name_at(point)
        serving_schema = serving.get((pkg, name, frozenset(workload.axes.items())))
        if serving_schema is not None:
            calls = served.setdefault(serving_schema, _ServedCalls(Counter(), set()))
            calls.n_calls_by_axes[axes_text(workload.axes)] += record.n_calls
            if not DeviceTarget.matches_target(record.target, serving_schema.targets):
                calls.mismatched_targets.add(record.target)
            continue
        p = pending.setdefault((pkg, name), _PendingSchema(contract, point, {}, Counter(), set()))
        key = semantics.sorted_key_json(workload.model_dump(mode="json"))
        p.workloads.setdefault(key, workload)
        p.n_calls_by_workload[key] += record.n_calls
        p.targets.add(record.target)
    files: dict[Path, bytes] = {}
    results = []
    for (pkg, name), p in sorted(pending.items()):
        # per Point, the most called scalars; ties to the least key: order-independent
        keys_by_axes: dict[str, list[str]] = {}
        for key, w in p.workloads.items():
            keys_by_axes.setdefault(axes_text(w.axes), []).append(key)
        workloads = [
            p.workloads[min(keys, key=lambda k: (-p.n_calls_by_workload[k], k))]
            for keys in keys_by_axes.values()
        ]
        kernel = spec.KernelSpec(
            language=language,
            targets=DeviceTarget.normalize_targets(list(p.targets)),
            entry_point=f"kernel{spec.SOURCE_SUFFIXES[language][0]}::run",
            sources=(),
        )
        narrowed = p.contract.narrow(p.point, workloads=workloads, kernel=kernel, models=(model,))
        relative_path = locate.op_dir(pkg).name / narrowed.relative_path()
        path = export_dir / relative_path
        errors += p.contract.schema_errors(path, narrowed)
        files[path] = narrowed.schema_text().encode()
        n_calls_by_axes: Counter[str] = Counter()
        for key, w in p.workloads.items():
            n_calls_by_axes[axes_text(w.axes)] += p.n_calls_by_workload[key]
        results.append(SchemaCalls(relative_path, True, _sorted_calls(n_calls_by_axes), (), False))
    for serving_schema, calls in served.items():
        mismatched = (
            DeviceTarget.normalize_targets(list(calls.mismatched_targets)) if calls.mismatched_targets else ()
        )
        lacks_model = model not in serving_schema.models
        results.append(
            SchemaCalls(
                serving_schema.relative_path,
                False,
                _sorted_calls(calls.n_calls_by_axes),
                mismatched,
                lacks_model,
            )
        )
    if errors:  # without repeats: each call of an unknown contract_id reports it
        raise ValueError("\n".join(dict.fromkeys(errors)))
    # replace the earlier export's files that are still empty: filled ones hold a kernel's sources
    stale_paths = [
        stale
        for pkg in _registered_op_packages()
        for stale in locate.committed_schema_paths(export_dir / locate.op_dir(pkg).name)
        if json.loads(stale.read_bytes()).get("kernel", {}).get("sources") == []
    ]
    for path, data in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        write_bytes_atomically(path, data)
    for stale in stale_paths:
        if stale not in files:  # a file written just now has no sources either
            stale.unlink()
    return sorted(results, key=lambda r: r.relative_path)


def repository_errors() -> list[str]:
    """``export check``: every problem of the repository's schemas and ``impls/``
    layout, per op. No GPU or environment; never raises."""
    errors = []
    for pkg in _registered_op_packages():
        schemas, schema_errors = ContractRegistry.default().read_schemas(pkg)
        errors += schema_errors
        errors += [f"{path}: {e}" for path, s in schemas if (e := s.kernel.requires_error) is not None]
        impls_dir = locate.op_dir(pkg) / "impls"
        for entry in sorted(impls_dir.iterdir()) if impls_dir.is_dir() else ():
            if not (entry.is_dir() and entry.name in spec.LANGUAGES):
                errors.append(f"{entry}: orphan: impls/ holds only {list(spec.LANGUAGES)} directories")
                continue
            for name_dir in sorted(entry.iterdir()):
                if not name_dir.is_dir():
                    errors.append(f"{name_dir}: orphan: impls/<language>/ holds only <name> directories")
                    continue
                errors += [
                    f"{f}: not a schema json"
                    for f in sorted(name_dir.iterdir())
                    if not (f.is_file() and f.suffix == ".json")
                ]
    return errors


def main(argv: Sequence[str], *, export_dir: Path | None) -> int:
    """``python -m aklr export [check] [--from D] [--to D] [--language L] [--model M]``;
    ``export_dir`` defaults both directories and is ``None`` only for check."""
    parser = argparse.ArgumentParser(prog="python -m aklr export")
    parser.add_argument("command", nargs="?", choices=["check"])
    parser.add_argument("--from", dest="calls_dir", type=Path, metavar="DIR")
    parser.add_argument("--to", dest="export_dir", type=Path, metavar="DIR")
    parser.add_argument("--language", choices=spec.LANGUAGES, help="of the new schema files")
    parser.add_argument("--model", help="the model whose serving made the calls, such as kimi-k3")
    args = parser.parse_args(argv)
    if args.command == "check":
        if args.calls_dir or args.export_dir or args.language or args.model:
            parser.error("export check takes no --from, --to, --language or --model")
        errors = repository_errors()
        for line in errors:
            print(line, file=sys.stderr)
        return int(bool(errors))
    if args.language is None and len(spec.LANGUAGES) > 1:
        parser.error(f"--language is required: one of {list(spec.LANGUAGES)}")
    if args.model is None:
        parser.error("--model is required: the model whose serving made the calls, such as kimi-k3")
    try:
        TypeAdapter(op.ModelName).validate_python(args.model)
    except ValidationError:
        parser.error(f"--model {args.model!r}: lower-case letters, digits, '.' and '-', such as kimi-k3")
    language = args.language or spec.LANGUAGES[0]
    assert export_dir is not None
    calls_dir, export_dir = args.calls_dir or export_dir, args.export_dir or export_dir
    try:
        records = read_calls(calls_dir)
        if not records:
            print(f"no calls.*.jsonl in {calls_dir}", file=sys.stderr)
            return 1
        results = export_schemas(calls_dir, records, export_dir, language=language, model=args.model)
    except (OSError, ValueError) as e:
        print(e, file=sys.stderr)
        return 1
    for schema_calls in results:
        relative_path, is_added, calls, mismatched_targets, lacks_model = schema_calls
        status = "added" if is_added else "served"
        line = f"{status:<8} {relative_path}   {len(calls)} workloads  "
        line += " ".join(f"{axes}:{n_calls}" for axes, n_calls in calls)
        if is_added:
            line += "   (sources empty: fill kernel.sources)"
        elif mismatched_targets:
            targets = ", ".join(str(t) for t in mismatched_targets)
            line += f"   (target mismatch: called on {targets}, which falls back)"
        if lacks_model:
            line += f"   (models lack {args.model}: add it to the file)"
        print(line)
    if n_calls := sum(r.n_calls for r in records if r.point is None):
        print(f"{n_calls} calls had no point (fallback only)")
    registry = ContractRegistry.default()
    if n_calls := sum(
        r.n_calls
        for r in records
        if r.point is not None and not registry.get(r.contract_id).domain.contains(r.point)
    ):
        print(f"{n_calls} calls were outside their contract's domain (fallback only)")
    return 0
