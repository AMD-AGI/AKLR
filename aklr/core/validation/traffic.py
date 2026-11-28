# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Real serving calls as bench inputs (the repository's ``traffic/``; its README says why).

A recording is one serving run: its server's settings, then each forward step's sequences as
``(q, kv)`` - query rows, and keys the last row attends. For a schema ``traffic/schemas.json`` lists,
bench measures each workload on ``samples`` recorded calls at the workload's dispatch key: the calls
ordered by work and split into that many parts of equal count, each part's median. ``OPS`` turns a
recording's steps into an op's calls, orders them by work, and makes a call's inputs that carry its
requests; bench draws the other inputs as usual."""

from __future__ import annotations

import json
import lzma
from collections.abc import Iterable, Iterator
from functools import cache
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    import torch

Step = tuple[str, tuple[tuple[int, int], ...]]
SUFFIX = ".traffic.jsonl.xz"


def write(path: Path, header: dict[str, Any], steps) -> None:
    """A recording: ``header`` as the first json line, then a line a step, ``[phase, [[q, kv], ...]]``."""
    lines = [header, *([phase, [list(seq) for seq in seqs]] for phase, seqs in steps)]
    text = "".join(json.dumps(line, separators=(",", ":")) + "\n" for line in lines)
    path.write_bytes(lzma.compress(text.encode(), preset=9 | lzma.PRESET_EXTREME))


@cache
def read(root: Path, name: str) -> tuple[dict[str, Any], tuple[Step, ...]]:
    """Recording ``name``'s header and steps."""
    text = lzma.decompress((root / "traffic" / f"{name}{SUFFIX}").read_bytes())
    header, *steps = map(json.loads, text.splitlines())
    return header, tuple((phase, tuple(map(tuple, seqs))) for phase, seqs in steps)


class Sample(NamedTuple):
    """One input a workload is measured on: its label among the workload's (``#i`` when several), its
    axes, and the recorded call it carries (None: drawn as the contract declares)."""

    label: str
    axes: dict[str, int]
    call: dict[str, Any] | None


class Listing(NamedTuple):
    """A schema's entry in ``schemas.json``: its recordings, and the calls a workload is measured on."""

    recordings: tuple[str, ...]
    samples: int


def listings(text: str | None) -> dict[str, Listing]:
    """``schemas.json``'s text as each schema's listing; none for no file. A schema's ``samples`` is
    the file's unless its entry sets its own."""
    if text is None:
        return {}
    listed = json.loads(text)
    return {
        name: Listing(tuple(entry["recordings"]), entry.get("samples", listed["samples"]))
        for name, entry in listed["schemas"].items()
    }


@cache
def schemas(root: Path) -> dict[str, Listing]:
    """Per schema name, its listing in the repository at ``root``."""
    path = root / "traffic" / "schemas.json"
    return listings(path.read_text() if path.exists() else None)


def changed_schemas(root: Path, changed: Iterable[str], base_text: str | None) -> set[str]:
    """The schema names whose recorded inputs differ from the base's, whose ``schemas.json`` text is
    ``base_text``: their listing changed (or appeared, or went), or a recording either lists changed
    (``changed``: paths relative to ``root``)."""
    base, head = listings(base_text), schemas(root)
    recordings = {
        Path(p).name.removesuffix(SUFFIX) for p in changed if p.startswith("traffic/") and p.endswith(SUFFIX)
    }
    return {
        name
        for name in base.keys() | head.keys()
        if base.get(name) != head.get(name)
        or recordings
        & {r for listing in (base.get(name), head.get(name)) if listing for r in listing.recordings}
    }


def op_name(contract) -> str:
    """The op a contract is of: its contract name, as ``OPS`` and ``calls`` take it."""
    return contract.contract_id.split("@")[0]


def dispatch_key(contract, axes: dict[str, int]) -> dict[str, int]:
    """The axes dispatch matches: those that are not ``any_size``."""
    return {a: v for a, v in axes.items() if not contract.axes[a].any_size}


@cache
def calls(root: Path, name: str, op: str) -> tuple[dict[str, Any], ...]:
    """The calls of ``op`` (a contract name) that recording ``name`` makes, each naming it."""
    return tuple({**call, "recording": name} for call in OPS[op].calls(*read(root, name)))


def recorded_calls(root: Path, schema, contract, axes: dict[str, int]) -> tuple[dict[str, Any], ...]:
    """The calls a workload at ``axes`` is measured on: the recorded calls at its dispatch key, ordered
    by work and split into the schema's ``samples`` parts of equal count, each part's median; none if
    ``schemas.json`` does not list the schema or no recorded call has that key."""
    listing = schemas(root).get(schema.name)
    if listing is None:
        return ()
    op, key = op_name(contract), dispatch_key(contract, axes)
    at_key = sorted(
        (c for name in listing.recordings for c in calls(root, name, op) if c["axes"].items() >= key.items()),
        key=OPS[op].work,
    )
    if not at_key:
        return ()
    n, k = len(at_key), min(listing.samples, len(at_key))
    bounds = [i * n // k for i in range(k + 1)]
    return tuple(at_key[(lo + hi) // 2] for lo, hi in pairwise(bounds))


def samples(root: Path, schema, contract, axes: dict[str, int]) -> tuple[Sample, ...]:
    """The inputs the workload at ``axes`` is measured on: one per recorded call (``recorded_calls``),
    each call's ``any_size`` axes over the workload's, inside the contract's domain or a ValueError;
    without one, the workload as drawn."""
    calls = recorded_calls(root, schema, contract, axes)
    if not calls:
        return (Sample("", dict(axes), None),)
    out = []
    for i, call in enumerate(calls):
        merged = axes | call["axes"]
        if not contract.domain.contains(schema.point_with_contract_id(merged, contract.contract_id)):
            raise ValueError(f"{schema.name}: a recorded call at {merged} is outside {contract.contract_id}")
        out.append(Sample("" if len(calls) == 1 else f"#{i}", merged, call))
    return tuple(out)


def tensors(contract, call: dict[str, Any], generator: torch.Generator) -> dict[str, torch.Tensor]:
    """The inputs that carry ``call``'s requests, on the CPU."""
    return OPS[op_name(contract)].tensors(call, generator)


def describe(call: dict[str, Any], axes: dict[str, int]) -> str:
    """``call`` in a line: its requests, and the axes it set that differ from ``axes`` (the workload's)."""
    changed = " ".join(f"{a}={v}" for a, v in call["axes"].items() if axes.get(a) != v)
    kv = " ".join(map(str, call["kv"]))
    return f"{len(call['kv'])} requests x {call['rows_per_request']} rows, kv {kv}; {changed}"


# A call: the ``axes`` it sets (its dispatch key, and ``any_size`` ones from its requests and server),
# ``rows_per_request``, each request's ``kv``, and the ``recording`` it is from.


def row_kv(call: dict[str, Any]) -> list[int]:
    """Each row's keys: row t of a request's T sees kv - T + 1 + t."""
    t = call["rows_per_request"]
    return [k - t + 1 + r for k in call["kv"] for r in range(t)]


class DsaTopkTransform:
    """sglang's DSA indexer top-k in a target verify: each request is ``draft_tokens`` rows, and its
    scores and page table are as wide as the server's page table."""

    def calls(self, header, steps) -> Iterator[dict[str, Any]]:
        t, width = header["draft_tokens"], header["table_pages"] * header["page_size"]
        for phase, seqs in steps:
            if phase == "decode" and all(q == t for q, _ in seqs):
                if max(k for _, k in seqs) > width:
                    raise ValueError(
                        f"a request of {max(k for _, k in seqs)} keys, the page table {width} wide"
                    )
                n = len(seqs)
                axes = {
                    "n_tokens": n * t,
                    "n_seqs": n,
                    "n_starts": n + 1,
                    "n_candidates": width,
                    "n_slots": width,
                }
                yield {"axes": axes, "rows_per_request": t, "kv": [k for _, k in seqs]}

    def work(self, call) -> int:
        """Keys attended, summed over rows: the scores the top-k reads."""
        return sum(row_kv(call))

    def tensors(self, call, generator: torch.Generator) -> dict[str, torch.Tensor]:
        import torch

        n, t, width = len(call["kv"]), call["rows_per_request"], call["axes"]["n_slots"]
        return {
            "lengths": torch.tensor(row_kv(call), dtype=torch.int32),
            "cu_seqlens_q": torch.arange(0, n * t + 1, t, dtype=torch.int32),
            "page_table_size_1": torch.stack(
                [torch.randperm(width, generator=generator, dtype=torch.int32) for _ in range(n)]
            ),
        }


OPS = {"dsa_topk_transform": DsaTopkTransform()}
