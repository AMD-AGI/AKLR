# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import json
from pathlib import Path, PurePosixPath

import pytest

from aklr.core.contract import export
from aklr.core.contract.locate import committed_schema_paths, op_dir
from aklr.core.contract.registry import ContractRegistry
from aklr.core.contract.spec._sources import write_bytes_atomically
from tests.op_case import KERNEL_TARGET, OpCase


class TestExport:
    """Call-file reading and schema export."""

    def test_read_calls_merges_call_files(self, tmp_path: Path) -> None:
        point = OpCase.of("gemm@1.0").point
        record = export.CallRecord(
            contract_id=point.contract_id,
            point=point,
            scalars={},
            fallback_reason=None,
            target=KERNEL_TARGET,
            n_calls=2,
        )
        (tmp_path / "calls.pid1.jsonl").write_text(record.model_dump_json() + "\n")
        (tmp_path / "calls.pid2.jsonl").write_text(
            record.model_copy(update={"n_calls": 3}).model_dump_json() + "\n"
        )
        assert export.read_calls(tmp_path) == [record.model_copy(update={"n_calls": 5})]

    @pytest.mark.parametrize("sources, kept", [([], False), ([{"path": "k.py"}], True)])
    def test_export_schemas_writes_unserved_points_only(
        self, tmp_path: Path, sources: list, kept: bool
    ) -> None:
        registry = ContractRegistry.default()
        case = OpCase.of("gemm@1.0")
        gemm_dir = op_dir(case.contract.op_package)
        schemas = {
            path: registry.parse_schema(path.read_bytes(), str(path))
            for path in committed_schema_paths(gemm_dir)
        }
        committed_path, committed = next(
            (path, schema) for path, schema in schemas.items() if schema.contract_id == case.contract_id
        )
        records = [
            export.CallRecord(
                contract_id=case.contract_id,
                point=point,
                scalars={},
                fallback_reason=None,
                target=KERNEL_TARGET,
                n_calls=1,
            )
            for point in (committed.point(committed.workloads[0].axes), case.point)
        ]
        export_dir = tmp_path / "export"
        seeded = export_dir / gemm_dir.name / "impls" / "flydsl" / "old" / "old.json"
        seeded.parent.mkdir(parents=True)
        seeded.write_text(json.dumps({"kernel": {"sources": sources}}))
        results = export.export_schemas(None, records, export_dir, language="flydsl", models=["test-model"])
        assert sorted(r.is_added for r in results) == [False, True]
        served, added = sorted(results, key=lambda r: r.is_added)
        assert served.relative_path == PurePosixPath(committed_path.relative_to(gemm_dir.parent))
        assert (served.missing_models, added.missing_models) == (("test-model",), ())
        path = export_dir / added.relative_path
        assert sorted(export_dir.rglob("*.json")) == sorted([path] + ([seeded] if kept else []))
        schema = registry.parse_schema(path.read_bytes(), str(path))
        assert schema.point(schema.workloads[0].axes) == case.point
        assert schema.models == ("test-model",)
        # parsing then writing restores a schema byte for byte
        text = committed_path.read_bytes()
        assert registry.parse_schema(text, "copy").schema_text().encode() == text

    def test_a_point_outside_the_domain_is_skipped(self, tmp_path: Path) -> None:
        """An idle rank's call of no rows: no kernel can serve it, so it adds no schema."""
        case = OpCase.of("gemm@1.0")
        empty = case.point.model_copy(update={"axes": {**case.point.axes, "m": 0}})
        assert not case.contract.domain.contains(empty)
        record = export.CallRecord(
            contract_id=case.contract_id,
            point=empty,
            scalars={},
            fallback_reason="no_workload",
            target=KERNEL_TARGET,
            n_calls=3,
        )
        assert (
            export.export_schemas(
                None, [record], tmp_path / "export", language="flydsl", models=["test-model"]
            )
            == []
        )

    def test_a_point_of_other_keys_is_an_error(self, tmp_path: Path) -> None:
        """A call recorded under another form of the contract is not taken as outside the domain."""
        case = OpCase.of("gemm@1.0")
        stale = case.point.model_copy(update={"axes": {**case.point.axes, "removed": 1}})
        record = export.CallRecord(
            contract_id=case.contract_id,
            point=stale,
            scalars={},
            fallback_reason="no_workload",
            target=KERNEL_TARGET,
            n_calls=1,
        )
        with pytest.raises(ValueError, match="axes"):
            export.export_schemas(
                None, [record], tmp_path / "export", language="flydsl", models=["test-model"]
            )

    @pytest.mark.parametrize(
        ("language", "entry_point"),
        [("flydsl", "kernel.py::run"), ("hip", "kernel.hip::run")],
    )
    def test_export_entry_point_follows_the_language(
        self, tmp_path: Path, language: str, entry_point: str
    ) -> None:
        case = OpCase.of("gemm@1.0")
        record = export.CallRecord(
            contract_id=case.contract_id,
            point=case.point,
            scalars={},
            fallback_reason=None,
            target=KERNEL_TARGET,
            n_calls=1,
        )
        (added,) = export.export_schemas(None, [record], tmp_path, language=language, models=["test-model"])
        path = tmp_path / added.relative_path
        kernel = ContractRegistry.default().parse_schema(path.read_bytes(), str(path)).kernel
        assert (kernel.language, kernel.entry_point) == (language, entry_point)
        assert added.relative_path.parts[2] == language


class TestWriteBytesAtomically:
    """A failed write keeps the old file and leaves no temporary file behind."""

    def test_failed_replace_keeps_old_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        target = tmp_path / "schema.json"
        target.write_bytes(b"old")

        def fail(self: Path, _target: Path) -> None:
            raise OSError("disk gone")

        monkeypatch.setattr(Path, "replace", fail)
        with pytest.raises(OSError, match="disk gone"):
            write_bytes_atomically(target, b"new")
        assert target.read_bytes() == b"old"
        assert sorted(p.name for p in tmp_path.iterdir()) == ["schema.json"]
