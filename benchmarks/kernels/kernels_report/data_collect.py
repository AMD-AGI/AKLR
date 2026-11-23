# SPDX-License-Identifier: MIT
"""What dispatch serves per Point of each contract, read from the committed schemas."""

from __future__ import annotations

import inspect
import math
import re
import subprocess
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import torch

from aklr.core.contract.baseline import OpBaseline
from aklr.core.contract.fastkey import FastKeyCodec
from aklr.core.contract.op import OpContract
from aklr.core.contract.registry import ContractRegistry
from aklr.core.contract.spec import PointSpec
from aklr.core.platform.device import DeviceTarget
from aklr.core.runtime.dispatch import DispatchTable

# The CI's GPU: the published measurements are MI355X's.
TARGET = DeviceTarget(arch="gfx950", hardware="MI355X")
SERVED = "served"
# Contract status keys, as section.json's status_counts holds them.
STATUSES = ("ok", "partial", "not-served", "not-measured")
BOT_SUBJECT = re.compile(r"perf: measurements on (?P<hardware>\S+) \(run (?P<run>\d+)\)")
MEASURED_LINE = re.compile(r"(?P<sha>[0-9a-f]{40}) (?P<path>\S+\.json)")


class SchemaErrors(Exception):
    """Committed schemas that dispatch would refuse to load, one error per line."""


@dataclass(frozen=True)
class Measurement:
    """The bot commit on main that last wrote a schema file's measurements."""

    run_id: str
    measured_at: str
    # the main commit the run measured (the file's Measured-SHA)
    commit: str
    hardware: str


@dataclass(frozen=True)
class Context:
    """The publishing run."""

    commit: str
    generated_at: str
    run_id: str
    run_url: str
    repo_url: str
    ref: str
    event: str


@dataclass(frozen=True)
class PointReport:
    # the Point's dims that vary across the contract's points, as text
    dims: dict[str, str]
    # SERVED, else dispatch's FallbackReason value
    reason: str
    # the serving kernel; None when not served
    kernel: str | None
    language: str | None
    path: str | None
    # speedup over the baseline; None when not served or the op has no baseline
    speedup: float | None
    microseconds: float | None
    # the serving kernel's kernel.requires: dispatch skips it where the installed stack misses them
    requires: dict[str, str] | None
    # committed kernels with a workload at this Point
    candidates: int


@dataclass(frozen=True)
class ContractReport:
    contract_id: str
    op_type: str
    op_package: str
    # the baseline's summary; None: the op has none, so every point not failed is served
    baseline: str | None
    min_speedup: float
    kernels: int
    # the dims every point shares
    fixed: dict[str, str]
    points: tuple[PointReport, ...]
    measured: Measurement | None

    @property
    def served(self) -> int:
        return sum(p.reason == SERVED for p in self.points)

    @property
    def speedups(self) -> list[float]:
        return [p.speedup for p in self.points if p.reason == SERVED and p.speedup is not None]

    @property
    def geomean(self) -> float | None:
        speedups = self.speedups
        return math.exp(sum(map(math.log, speedups)) / len(speedups)) if speedups else None

    @property
    def status(self) -> str:
        if self.served == len(self.points):
            return "ok"
        if self.served:
            return "partial"
        if all(p.reason == "not_measured" for p in self.points):
            return "not-measured"
        return "not-served"

    @property
    def group(self) -> str:
        """The op package without ``aklr.`` and ``plugins.``: ops.gemm, sglang.models.qwen3_5."""
        return re.sub(r"^aklr\.(plugins\.)?", "", self.op_package)


def text(value: object) -> str:
    return str(value).lower() if isinstance(value, bool) else str(value)


def point_dims(contract: OpContract, point: PointSpec) -> dict[str, str]:
    """The matched Point of ``point`` as text: any_size axes ``*``, absent tensors left out."""
    dims = {name: "*" if contract.axes[name].any_size else text(v) for name, v in point.axes.items()}
    dims |= {f"{name}.dtype": dtype for name, dtype in point.dtypes.items() if dtype is not None}
    dims |= {name: text(value) for name, value in point.attrs.items()}
    if len(contract.layout_variants.names()) > 1:
        dims["layout"] = point.layout_variant_name
    return dims


def summarize(
    contract: OpContract,
    schemas: Iterable[tuple[str, OpContract]],
    *,
    min_speedup: float,
    baseline: str | None,
    measured: Measurement | None,
) -> ContractReport:
    """``contract``'s points as dispatch serves them from ``schemas`` (repository path,
    schema) at ``min_speedup``: its own DispatchTable, each runnable the schema."""
    schemas = list(schemas)
    table = DispatchTable.from_schemas(
        contract,
        torch.device("cpu"),
        [(schema, (path, schema), schema.workloads) for path, schema in schemas],
        min_speedup=min_speedup,
    )
    codec = FastKeyCodec.from_contract(contract)
    candidates: dict[tuple, int] = {}
    labels: dict[tuple, dict[str, str]] = {}
    for _, schema in schemas:
        points = {}
        for w in schema.workloads:
            point = schema.point(w.axes)
            points.setdefault(codec.match_key(codec.to_fast_key(point)), point)
        for key, point in points.items():
            candidates[key] = candidates.get(key, 0) + 1
            labels.setdefault(key, point_dims(contract, point))

    rows = []
    for key in candidates:
        if key in table.runnables:
            path, schema = table.runnables[key]
            same = [
                w for w in schema.workloads if codec.match_key(codec.to_fast_key(schema.point(w.axes))) == key
            ]
            # the rank dispatch serves the key at: the lowest among its any_size workloads
            workload = min(same, key=schema.rank_key)
            # without a baseline, validation's speedup is untimed or over an incumbent: it ranks only
            speedup = workload.speedup if baseline is not None else None
            requires = dict(schema.kernel.requires)
            row = (
                SERVED,
                schema.name,
                schema.kernel.language,
                path,
                speedup,
                workload.microseconds,
                requires,
            )
        else:
            row = (table.fallback_reasons[key].value, None, None, None, None, None, None)
        rows.append((labels[key], *row, candidates[key]))

    names = list(dict.fromkeys(name for dims, *_ in rows for name in dims))
    shared = {n for n in names if len({dims.get(n) for dims, *_ in rows}) == 1}
    if shared == set(names):  # one point: keep its var axes as its label
        shared -= set(contract.domain.axes) & {n for n, axis in contract.axes.items() if axis.is_var}
    fixed = {n: rows[0][0][n] for n in names if n in shared}
    points = tuple(
        PointReport({n: v for n, v in dims.items() if n not in shared}, *rest) for dims, *rest in rows
    )
    return ContractReport(
        contract_id=contract.contract_id,
        op_type=contract.op_type,
        op_package=contract.op_package,
        baseline=baseline,
        min_speedup=min_speedup,
        kernels=len(schemas),
        fixed=fixed,
        points=points,
        measured=measured,
    )


def baseline_summary(baseline: OpBaseline) -> str:
    """The first sentence of ``baseline``'s ``baseline`` docstring, else its class name."""
    doc = type(baseline).baseline.__doc__
    if not doc:
        return type(baseline).__name__
    paragraph = " ".join(inspect.cleandoc(doc).split("\n\n")[0].split())
    paragraph = re.sub(r"``([^`]*)``|`([^`]*)`", lambda m: m[1] or m[2], paragraph)
    return re.split(r"(?<=[.;])\s", paragraph, maxsplit=1)[0].rstrip(".;:,")


def parse_measurements(log: str) -> dict[str, Measurement]:
    """Each schema path's newest bot measurement in ``git log -z`` output, newest first,
    of ``%H%x1f%cI%x1f%s%x1f%b``."""
    found: dict[str, Measurement] = {}
    for record in log.split("\x00"):
        fields = record.strip("\n").split("\x1f")
        if len(fields) != 4 or not (subject := BOT_SUBJECT.match(fields[2])):
            continue
        for line in fields[3].splitlines():
            if match := MEASURED_LINE.fullmatch(line.strip()):
                found.setdefault(
                    match["path"],
                    Measurement(subject["run"], fields[1], match["sha"], subject["hardware"]),
                )
    return found


def read_measurements(root: Path, rev: str = "HEAD") -> dict[str, Measurement]:
    """``parse_measurements`` of the bot's commits in ``rev``'s history."""
    log = subprocess.run(
        ["git", "log", "-z", "--format=%H%x1f%cI%x1f%s%x1f%b", "--grep=^perf: measurements on ", rev],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return parse_measurements(log)


def collect(
    root: Path,
    *,
    min_speedup: float,
    measurements: Mapping[str, Measurement],
    target: DeviceTarget = TARGET,
) -> list[ContractReport]:
    """Every contract with a committed kernel for ``target``, by contract_id; ``SchemaErrors``
    listing the schema errors dispatch would refuse to load."""
    registry = ContractRegistry.default()
    by_contract: dict[str, list[tuple[str, OpContract]]] = {}
    errors: list[str] = []
    for op_package in sorted({c.op_package for c in registry.contracts()}):
        schemas, found = registry.read_schemas(op_package)
        errors += found
        for path, schema in schemas:
            # a definition (no kernel sources) declares a scenario; it serves nothing
            if schema.kernel.sources and DeviceTarget.matches_target(target, schema.kernel.targets):
                relative = path.resolve().relative_to(root.resolve()).as_posix()
                by_contract.setdefault(schema.contract_id, []).append((relative, schema))
    if errors:
        raise SchemaErrors("\n".join(errors))
    reports = []
    for contract_id, schemas in sorted(by_contract.items()):
        contract = registry.get(contract_id)
        baseline = baseline_summary(contract.baseline)
        newest = max(
            (measurements[path] for path, _ in schemas if path in measurements),
            key=lambda m: m.measured_at,
            default=None,
        )
        reports.append(
            summarize(
                contract,
                schemas,
                # nothing to beat without a baseline: dispatch serves any point not failed
                min_speedup=min_speedup if baseline is not None else 0,
                baseline=baseline,
                measured=newest,
            )
        )
    return reports
