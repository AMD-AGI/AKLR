"""Dispatch table for mapping workload keys to optimal kernels."""

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
    """Get the directory for storing dispatch table cache files.

    Returns
    -------
    Path
        The dispatch table cache directory path.
    """
    return RuntimeEnv().build_cache_path / "dispatch_table"


@dataclass
class DispatchTable:
    """Dispatch table for mapping workload keys to optimal kernels.

    This class manages a lookup table that maps workload characteristics (DispatchKey)
    to the kernels of each schema name ranked best first, judged against the baseline
    a call would otherwise run (the seam's backend). It supports caching to disk and
    ahead-of-time compilation of frequently used kernels.
    """

    UNMEASURED: ClassVar[str] = ""
    """The slot of a direct call, or of a baseline no workload was measured against."""

    digest: str
    """Hash digest identifying this table's configuration and data."""
    index: dict[str, dict[str, dict[DispatchKey, list[str]]]] = field(default_factory=dict)
    """Mapping from baseline to schema name to (key -> kernel names, best first) lookup."""
    best_kernel: dict[str, dict[str, str]] = field(default_factory=dict)
    """Mapping from baseline to schema name to the kernel winning the most keys."""

    @classmethod
    def _load_from_disk(cls, digest: str) -> dict[str, Any] | None:
        """Load dispatch table data from cache file.

        Parameters
        ----------
        digest : str
            The digest hash identifying the cached data.

        Returns
        -------
        Optional[Dict[str, Any]]
            The cached data if exists, None otherwise.
        """
        index_path = _dispatch_table_dir() / f"{digest}.json"

        if index_path.exists():
            with open(index_path, "r") as f:
                return json.load(f)

        return None

    @classmethod
    def _save_to_disk(cls, digest: str, data: dict[str, Any]) -> None:
        """Save dispatch table data to cache file.

        Parameters
        ----------
        digest : str
            The digest hash identifying the data to cache.
        data : Dict[str, Any]
            The data to save to cache.
        """
        index_path = _dispatch_table_dir() / f"{digest}.json"
        index_path.parent.mkdir(parents=True, exist_ok=True)

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
        """Load an existing dispatch table from cache or build a new one.

        This method first attempts to load a cached dispatch table based on the
        digest of the schema collection, configuration, device and threshold. If no
        cached version exists, it builds a new table from scratch and caches it.

        Parameters
        ----------
        schema_collection : SchemaCollection
            The schema collection containing benchmark data and kernels.
        config_registry : DispatchConfigRegistry
            Per-schema configuration registry for building the dispatch table.
        target : DeviceTarget
            The device the table serves.
        min_speedup : float
            ``AKLR_MIN_SPEEDUP``, see :meth:`Workload.qualifies`.

        Returns
        -------
        DispatchTable
            The loaded or newly built dispatch table.
        """
        digest = cls._digest(schema_collection, config_registry, target, min_speedup)
        keys = TypeAdapter(DispatchKey)

        # Try to load from cache
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

        # Build fresh
        table = cls._build(schema_collection, config_registry, target, min_speedup, digest)
        # Persist minimal index
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
        """Build a new dispatch table from schema collection data.

        Per baseline any workload was measured against, plus :attr:`UNMEASURED`, and per
        schema name, every workload dispatch may serve on ``target`` in place of that
        baseline is keyed, and each key ranks its kernels by speedup over that baseline
        (0 without one), then by name; :attr:`UNMEASURED` ranks by the baseline most of
        the key's kernels were measured against. The kernel ranked first on the most
        keys is the name's best kernel.

        Parameters
        ----------
        schema_collection : SchemaCollection
            The schema collection containing benchmark data.
        config_registry : DispatchConfigRegistry
            Per-schema configuration registry.
        target : DeviceTarget
            The device the table serves.
        min_speedup : float
            ``AKLR_MIN_SPEEDUP``, see :meth:`Workload.qualifies`.
        digest : str
            :meth:`_digest` of the other arguments, the table's identity.

        Returns
        -------
        DispatchTable
            The newly built dispatch table.
        """
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
        """Perform ahead-of-time compilation of frequently used kernels.

        This method pre-compiles kernels that are used frequently according to
        the AOT ratio configuration. This reduces runtime compilation overhead.

        Parameters
        ----------
        schema_collection : SchemaCollection
            The schema collection containing schemas and kernels.
        config_registry : DispatchConfigRegistry
            Per-schema configuration registry containing AOT ratio and miss policy settings.
        table : DispatchTable
            The dispatch table containing kernel mappings.
        """
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

        # Build best_kernel for schemas with on_miss_policy == "use_best_kernel"
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
        """Compute a hash digest for the schema collection and configuration.

        This method creates a deterministic hash that uniquely identifies the
        combination of schema collection data, dispatch configuration, device and
        threshold. The digest is used for caching dispatch tables.

        Parameters
        ----------
        schema_collection : SchemaCollection
            The schema collection containing benchmark data.
        config_registry : DispatchConfigRegistry
            Per-schema configuration registry.
        target : DeviceTarget
            The device the table serves.
        min_speedup : float
            ``AKLR_MIN_SPEEDUP``.

        Returns
        -------
        str
            SHA256 hash digest as a hexadecimal string.
        """
        kept_schemas: list[dict[str, Any]] = []
        kept_workloads: list[dict[str, Any]] = []
        for schemas in schema_collection.to_dict()["schemas"].values():
            for schema in schemas:
                kernel = schema["kernel"]
                # kernel-less schemas never dispatch
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
            "index": "ranked",
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
        """Find the best kernel for a given schema name and workload key.

        Parameters
        ----------
        schema_name : str
            Name of the schema.
        key : DispatchKey
            Workload characteristics key to match against.
        baseline : Optional[str]
            The backend the call falls back to, e.g. ``aiter`` or ``sglang``; None for a
            direct ``aklr.ops`` call, which has no baseline to beat.

        Returns
        -------
        Optional[str]
            The name of the best kernel, or None to fall back to ``baseline``.
        """
        return next(iter(self.candidates(schema_name, key, baseline)), None)

    def candidates(self, schema_name: str, key: DispatchKey, baseline: str | None) -> list[str]:
        return self.index.get(self.slot(baseline), {}).get(schema_name, {}).get(key, [])

    def match_best_kernel(self, schema_name: str, baseline: str | None) -> str | None:
        """The kernel winning the most keys of ``schema_name`` against ``baseline``."""
        return self.best_kernel.get(self.slot(baseline), {}).get(schema_name)

    def slot(self, baseline: str | None) -> str:
        return baseline if baseline in self.index else self.UNMEASURED
