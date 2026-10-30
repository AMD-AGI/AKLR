# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Check the benchmark artifact before AITER consumes it through AKLR."""

import argparse
from pathlib import Path

from aklr.core.schema import SchemaCollection


def read_measured_collection(root: Path) -> SchemaCollection:
    for directory in ("definitions", "solutions", "workloads", "traces"):
        if not (root / directory).is_dir():
            raise ValueError(f"Missing {directory}/ in benchmark collection: {root}")
    collection = SchemaCollection.from_path(root)
    candidate_records = 0
    total = 0
    for records in collection.records.values():
        for record in records:
            total += 1
            solution = collection.get_solution(record.solution) if record.solution else None
            if (
                record.definition not in collection.definitions
                or solution is None
                or solution.definition != record.definition
            ):
                raise ValueError("Trace references an unknown definition or solution")
            if record.evaluation is None:
                raise ValueError(f"Trace has no evaluation: {record.solution}")
            # Failed workloads remain evidence, including failures before GPU discovery.
            # The benchmark owns the per-solution correctness/speedup verdict.
            if not record.is_successful():
                continue
            if not record.evaluation.environment.target.is_complete:
                raise ValueError(f"Trace has no complete GPU target: {record.solution}")
            if solution.author not in {"baseline", "reference"}:
                candidate_records += 1
    if not candidate_records:
        raise ValueError("No successful candidate traces; run the benchmark first")
    print(f"Schema collection: {root}")
    print(f"Validated {total} traces, including {candidate_records} successful candidate traces")
    return collection


def check_dispatch(collection: SchemaCollection) -> None:
    from aklr.core.platform import env

    if env.get_disable_proxy():
        raise ValueError("AKLR_DISABLE_PROXY must be 0 for AITER validation")
    if collection.root is None or env.get_schema_path().resolve() != collection.root.resolve():
        raise ValueError("AKLR_SCHEMA_PATH must point to the measured collection root")

    import aiter
    from aklr.core.dispatch.kernel_dispatch import get_dispatch_runtime
    from aklr.core.dispatch.winner_table import pick_winners, schema_collection_digest

    from aklr.core.platform.device import DeviceTarget
    from aklr.ops import load_routes

    load_routes()
    if not hasattr(aiter.gemm_a8w8, "aklr_proxy"):
        raise ValueError("AKLR did not proxy AITER")
    target = DeviceTarget.current()
    measured_here = any(
        record.evaluation is not None
        and record.is_successful()
        and DeviceTarget.same_target(target, record.evaluation.environment.target)
        and record.solution is not None
        and (solution := collection.get_solution(record.solution)) is not None
        and solution.author not in {"baseline", "reference"}
        for records in collection.records.values()
        for record in records
    )
    if not measured_here:
        raise ValueError(f"No successful candidate traces for the AITER runner's GPU: {target}")
    expected, _ = pick_winners(collection)
    runtime = get_dispatch_runtime()
    if (
        runtime.table.digest != schema_collection_digest(collection.root)
        or runtime.table.index != expected.index
        or runtime.table.kernels != expected.kernels
    ):
        raise ValueError("AKLR did not install the benchmark collection's winner table")
    for name in expected.kernels:
        if runtime.registry.get(name) is None:
            raise ValueError(f"Measured winner was not registered: {name}")
    calls = sum(
        DeviceTarget.same_target(target, key.target) for bucket in expected.index.values() for key in bucket
    )
    print(f"AKLR proxy enabled on {target}; {calls} measured winner rows for this GPU")
    if not calls:
        print("No rows met AKLR's existing winner rules; AITER fallback will be tested")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("schema_path", type=Path)
    parser.add_argument("--check-dispatch", action="store_true")
    args = parser.parse_args()
    collection = read_measured_collection(args.schema_path.expanduser().resolve())
    if args.check_dispatch:
        check_dispatch(collection)


if __name__ == "__main__":
    main()
