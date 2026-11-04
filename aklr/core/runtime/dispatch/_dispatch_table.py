###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 by FlashInfer-Bench Contributors
#
# Adapted from FlashInfer-Bench (https://github.com/flashinfer-ai/flashinfer-bench)
# Modified by the AKLR team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of AKLR (see LICENSE).
###############################################################################

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import TypeAdapter

from aklr.core.platform.device import DeviceTarget
from aklr.core.platform.env import RuntimeEnv
from aklr.core.runtime.builder.registry import BuilderRegistry
from aklr.core.runtime.dispatch._dispatch_key import DispatchKey, DispatchKeyFactory
from aklr.core.schema._collection import SchemaCollection

if TYPE_CHECKING:
    from aklr.core.runtime.dispatch._dispatcher import DispatchConfigRegistry


def _dispatch_table_dir() -> Path:
    return RuntimeEnv().build_cache_path / "dispatch_table"


@dataclass
class DispatchTable:
    """Kernels ranked per dispatch key from the schemas' measured workloads; cached on disk by digest."""

    UNMEASURED: ClassVar[str] = ""
    """The slot of a direct call, or of a baseline no workload was measured against."""

    digest: str
    index: dict[str, dict[str, dict[DispatchKey, list[str]]]] = field(default_factory=dict)
    """baseline -> schema name -> dispatch key -> kernel names, best first."""
    best_kernel: dict[str, dict[str, str]] = field(default_factory=dict)
    """baseline -> schema name -> the kernel ranked first on the most keys."""

    @classmethod
    def _load_from_disk(cls, digest: str) -> dict[str, Any] | None:
        index_path = _dispatch_table_dir() / f"{digest}.json"

        if index_path.exists():
            with open(index_path, "r") as f:
                return json.load(f)

        return None

    @classmethod
    def _save_to_disk(cls, digest: str, data: dict[str, Any]) -> None:
        index_path = _dispatch_table_dir() / f"{digest}.json"
        index_path.parent.mkdir(parents=True, exist_ok=True)

        # a plain write: a process loading this table meanwhile can read it half-written
        with open(index_path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load_or_build(
        cls,
        schema_collection: SchemaCollection,
        config_registry: DispatchConfigRegistry,
        target: DeviceTarget,
        min_speedup: float,
    ) -> DispatchTable:
        """Load the table cached for these inputs, or build and cache it; then build its AOT kernels."""
        digest = cls._digest(schema_collection, config_registry, target, min_speedup)
        keys = TypeAdapter(DispatchKey)

        raw = cls._load_from_disk(digest)
        if raw is not None:
            index: dict[str, dict[str, dict[DispatchKey, list[str]]]] = {}
            for baseline, by_schema in raw.get("index", {}).items():
                for schema_name, items in by_schema.items():
                    bucket: dict[DispatchKey, list[str]] = {}
                    for key_enc, kernel_names in items.items():
                        key = keys.validate_json(key_enc)
                        bucket[key] = kernel_names
                    index.setdefault(baseline, {})[schema_name] = bucket

            best_kernel: dict[str, dict[str, str]] = raw.get("best_kernel", {})

            table = cls(digest=digest, index=index, best_kernel=best_kernel)
            cls._prewarm_aot(schema_collection, config_registry, table)

            return table

        table = cls._build(schema_collection, config_registry, target, min_speedup, digest)
        to_dump: dict[str, Any] = {
            "digest": table.digest,
            "index": {},
            "best_kernel": table.best_kernel,
        }
        for baseline, by_schema in table.index.items():
            for schema_name, bucket in by_schema.items():
                to_dump["index"].setdefault(baseline, {})[schema_name] = {
                    keys.dump_json(key).decode(): kernel_names for key, kernel_names in bucket.items()
                }

        cls._save_to_disk(digest, to_dump)
        cls._prewarm_aot(schema_collection, config_registry, table)

        return table

    @classmethod
    def _build(
        cls,
        schema_collection: SchemaCollection,
        config_registry: DispatchConfigRegistry,
        target: DeviceTarget,
        min_speedup: float,
        digest: str,
    ) -> DispatchTable:
        index: dict[str, dict[str, dict[DispatchKey, list[str]]]] = {}
        best_kernel: dict[str, dict[str, str]] = {}

        baselines = {cls.UNMEASURED} | {
            baseline
            for schemas in schema_collection.schemas.values()
            for s in schemas
            for w in s.workloads
            if (e := w.evaluation(target)) is not None
            for baseline in e.speedup
        }
        for baseline in sorted(baselines):
            by_schema = index.setdefault(baseline, {})
            for schema_name, schemas in schema_collection.schemas.items():
                if config_registry.get(schema_name) is None:
                    continue
                builder = DispatchKeyFactory.specialize(schemas[0])
                per_key: dict[DispatchKey, dict[str, dict[str, float]]] = {}
                for s, w in schema_collection.filter_schemas(schema_name, target, min_speedup, baseline):
                    measured = per_key.setdefault(builder.build_from_workload(w), {}).setdefault(
                        s.kernel.name, {}
                    )
                    evaluation = w.evaluation(target)
                    for b, speedup in evaluation.speedup.items() if evaluation else ():
                        measured[b] = max(measured.get(b, 0.0), speedup.value)
                if not per_key:
                    continue
                bucket = by_schema.setdefault(schema_name, {})
                for key, kernels in per_key.items():
                    rank_by = baseline
                    # UNMEASURED ranks by the baseline most of the key's kernels were measured against
                    if baseline == cls.UNMEASURED:
                        common = Counter(b for measured in kernels.values() for b in measured)
                        rank_by = min(
                            common,
                            key=lambda b: (-common[b], b),
                            default=cls.UNMEASURED,
                        )
                    bucket[key] = sorted(kernels, key=lambda k: (-kernels[k].get(rank_by, 0.0), k))
                best_kernel.setdefault(baseline, {})[schema_name] = Counter(
                    ranked[0] for ranked in bucket.values()
                ).most_common(1)[0][0]

        return cls(digest=digest, index=index, best_kernel=best_kernel)

    @classmethod
    def _prewarm_aot(
        cls,
        schema_collection: SchemaCollection,
        config_registry: DispatchConfigRegistry,
        table: DispatchTable,
    ) -> None:
        reg = BuilderRegistry.get_instance()

        buckets = [item for by_schema in table.index.values() for item in by_schema.items()]
        for schema_name, bucket in buckets:
            if not bucket:
                continue

            config = config_registry.get(schema_name)
            if config is None:
                continue
            if not (config.aot_ratio and config.aot_ratio > 0.0):
                continue

            win_counts = Counter(ranked[0] for ranked in bucket.values())
            ranked = sorted(win_counts.items(), key=lambda kv: kv[1], reverse=True)
            cutoff = max(1, int(len(ranked) * config.aot_ratio))

            for kernel_name, _ in ranked[:cutoff]:
                kernel_schema = schema_collection.get_schema(kernel_name)
                if kernel_schema:
                    reg.build(kernel_schema)

        best = [item for by_schema in table.best_kernel.values() for item in by_schema.items()]
        for schema_name, kernel_name in best:
            config = config_registry.get(schema_name)
            if config is None:
                continue
            if config.on_miss_policy == "use_best_kernel":
                kernel_schema = schema_collection.get_schema(kernel_name)
                if kernel_schema:
                    reg.build(kernel_schema)

    @classmethod
    def _digest(  # structure: allow S4 the table's cache identity, ~80 lines of its own
        cls,
        schema_collection: SchemaCollection,
        config_registry: DispatchConfigRegistry,
        target: DeviceTarget,
        min_speedup: float,
    ) -> str:
        kept_schemas: list[dict[str, Any]] = []
        kept_workloads: list[dict[str, Any]] = []
        for schemas in schema_collection.to_dict()["schemas"].values():
            for schema in schemas:
                kernel = schema["kernel"]
                if kernel is None:
                    continue
                for drop in ("description", "tags", "reference", "constraints"):
                    schema.pop(drop, None)
                kernel["build"]["dependencies"] = sorted(kernel["build"]["dependencies"])
                kernel["sources"] = [
                    {
                        "path": sf["path"],
                        "sha1": hashlib.sha1(sf["content"].encode("utf-8")).hexdigest(),
                    }
                    for sf in kernel["sources"]
                ]
                kept_schemas.append(schema)
                for workload in schema.pop("workloads"):
                    kept_workloads.append(
                        {
                            "schema": schema["name"],
                            "kernel": schema["kernel"]["name"],
                            "axes": sorted(workload["axes"].items()),
                            "evaluation": next(
                                (e for e in workload["evaluations"] if e["hardware"] == str(target)),
                                None,
                            ),
                        }
                    )

        payload = {
            "index": "ranked",  # the cache file's index format: a new value retires files of the old one
            "cfg": config_registry.model_dump(mode="json"),
            "target": str(target),
            "min_speedup": min_speedup,
            "schemas": sorted(kept_schemas, key=lambda x: x["kernel"]["name"]),
            "workloads": sorted(
                kept_workloads,
                key=lambda x: (x["schema"], x["kernel"], x["axes"]),
            ),
        }
        str_repr = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(str_repr).hexdigest()

    def match_kernel(self, schema_name: str, key: DispatchKey, baseline: str | None) -> str | None:
        """The best kernel for ``key`` against ``baseline`` (None: a direct call), or None to fall back."""
        return next(iter(self.candidates(schema_name, key, baseline)), None)

    def candidates(self, schema_name: str, key: DispatchKey, baseline: str | None) -> list[str]:
        return self.index.get(self.slot(baseline), {}).get(schema_name, {}).get(key, [])

    def match_best_kernel(self, schema_name: str, baseline: str | None) -> str | None:
        return self.best_kernel.get(self.slot(baseline), {}).get(schema_name)

    def slot(self, baseline: str | None) -> str:
        return baseline if baseline in self.index else self.UNMEASURED
