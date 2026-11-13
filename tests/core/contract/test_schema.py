# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import hashlib
import json
import re
from pathlib import Path

import pytest
import torch

from aklr.core.contract import expr, locate, semantics
from aklr.core.contract.op import OpContract
from aklr.core.contract.registry import ContractRegistry
from aklr.core.contract.semantics import sorted_key_json
from aklr.core.contract.spec import attr
from tests.op_case import HIP_GEMM_KERNEL_SOURCE, OpCase


class TestSchemaFile:
    """``OpContract.narrow`` writes a schema."""

    def test_schema_text_round_trips(self, op_case: OpCase) -> None:
        """``parse_schema`` of ``op_case.schema().schema_text()`` equals the schema; the
        kernel takes every call key, and its hash is the entry point and sources."""
        schema = op_case.schema()
        parsed = ContractRegistry.default().parse_schema(schema.schema_text().encode(), "schema.json")
        assert parsed == schema
        assert parsed.point(parsed.workloads[0].axes) == op_case.point
        assert set(schema.kernel_call_keys) <= {p.name for p in schema.api_params}
        kernel = schema.kernel
        sources = [[s.path, s.content] for s in kernel.sources]
        assert (
            kernel.content_hash
            == hashlib.sha256(
                sorted_key_json({"entry_point": kernel.entry_point, "sources": sources}).encode()
            ).hexdigest()
        )

    def test_optional_attr_at_its_default_is_left_out(self) -> None:
        """A fused_moe schema written before ``has_num_local_tokens`` and
        ``has_swiglu_limit`` existed parses at their defaults and writes back byte for
        byte; its kernel takes neither them nor the inputs they gate. An interleave
        schema, ``has_swiglu_limit`` true, takes ``swiglu_limit``."""
        registry = ContractRegistry.default()
        impls = Path("aklr/ops/fused_moe/impls/flydsl")
        separated = impls / (
            "fused_moe_shuffled_16x16_model_dim6144_n_experts257_topk9_inter_dim256/f13fecf15dbe0a9d.json"
        )
        text = separated.read_bytes()
        assert b"has_num_local_tokens" not in text
        schema = registry.parse_schema(text, str(separated))
        point = schema.point(schema.workloads[0].axes)
        assert not point.attrs["has_num_local_tokens"]
        assert not point.attrs["has_swiglu_limit"]
        assert schema.schema_text().encode() == text
        keys = set(schema.kernel_call_keys)
        assert {"num_local_tokens", "swiglu_limit"}.isdisjoint(keys)
        assert "expert_mask" in keys
        data = json.loads(text)
        data["attrs"]["has_num_local_tokens"] = False
        with pytest.raises(ValueError):
            registry.parse_schema(json.dumps(data, indent=2).encode(), str(separated))
        interleave = min(
            impls.glob(
                "fused_moe_interleave_has_swiglu_limit_gate_up_interleaved_model_dim4096"
                "_n_experts256_topk6_inter_dim256/*.json"
            )
        )
        keys = registry.parse_schema(interleave.read_bytes(), str(interleave)).kernel_call_keys
        assert "swiglu_limit" in keys and "num_local_tokens" not in keys

    def test_parse_schema_lists_every_problem(self) -> None:
        data = json.loads(OpCase.of("gemm@1.0").schema().schema_text())
        data["kernel"]["language"] = "no_such_language"
        data["kernel"]["entry_point"] = "no_separator"
        with pytest.raises(ValueError) as error:
            ContractRegistry.default().parse_schema(json.dumps(data).encode(), "bad.json")
        lines = str(error.value).splitlines()
        for field in ("language", "entry_point"):
            assert any(line.startswith(f"bad.json: kernel.{field}: ") for line in lines)

    def test_kernel_requires(self) -> None:
        """``kernel.requires`` follows ``targets`` and is absent when empty; a key is a
        language, a value a version specifier. The uuid hashes it."""
        schema = OpCase.of("gemm@1.0").schema()
        data = json.loads(schema.schema_text())
        kernel = data["kernel"]
        assert "requires" not in kernel
        parse = ContractRegistry.default().parse_schema

        def with_requires(requires: object) -> bytes:
            data["kernel"] = {
                **{k: kernel[k] for k in ("language", "targets")},
                "requires": requires,
                **{k: kernel[k] for k in ("entry_point", "sources")},
            }
            return expr.json_file_text(data).encode()

        parsed = parse(with_requires({"flydsl": ">=0.3.2"}), "a.json")
        assert parsed.kernel.requires == {"flydsl": ">=0.3.2"}
        assert parsed.relative_path() != schema.relative_path()
        for bad in ({"flydsl": ""}, {"flydsl": "0.3.2"}, {"cuda": ">=12"}):
            with pytest.raises(ValueError, match=r"^a\.json: kernel\.requires"):
                parse(with_requires(bad), "a.json")

    @pytest.mark.parametrize(
        ("language", "requires", "pinned"),
        [
            ("flydsl", {"flydsl": ">=0.3.2"}, True),
            ("triton", {"triton": ">=3.8", "flydsl": ">=0.3.2"}, True),
            ("hip", {"hip": ">=7.15"}, True),
            ("flydsl", {}, False),
            ("flydsl", {"triton": ">=3.8"}, False),
            ("hip", {}, False),
        ],
        ids=["flydsl", "triton_and_more", "hip", "flydsl_empty", "flydsl_other_language", "hip_empty"],
    )
    def test_kernel_requires_error(self, language: str, requires: dict[str, str], pinned: bool) -> None:
        """A kernel names its own language in ``requires``; a definition (no sources)
        needs nothing. ``export check`` reports the error."""
        case = OpCase.of("gemm@1.0")
        schema = (
            case.schema(HIP_GEMM_KERNEL_SOURCE, language="hip")
            if language == "hip"
            else case.schema(language=language)
        )
        kernel = schema.kernel.model_copy(update={"requires": requires})
        error = kernel.requires_error
        assert (error is None) is pinned
        if not pinned:
            assert f"names no {language} version" in error
        assert kernel.model_copy(update={"sources": ()}).requires_error is None

    def test_kernel_author(self) -> None:
        """``kernel.author`` follows ``targets``, is absent when empty and is one line
        without surrounding spaces. The uuid hashes it; the content hash and
        ``same_kernel`` do not."""
        schema = OpCase.of("gemm@1.0").schema()
        data = json.loads(schema.schema_text())
        kernel = data["kernel"]
        assert "author" not in kernel
        parse = ContractRegistry.default().parse_schema

        def with_author(author: object) -> bytes:
            data["kernel"] = {
                **{k: kernel[k] for k in ("language", "targets")},
                "author": author,
                **{k: kernel[k] for k in ("entry_point", "sources")},
            }
            return expr.json_file_text(data).encode()

        parsed = parse(with_author("TEST_author"), "a.json")
        assert parsed.kernel.author == "TEST_author"
        assert parsed.schema_text().encode() == with_author("TEST_author")
        assert parsed.relative_path() != schema.relative_path()
        assert parsed.kernel.content_hash == schema.kernel.content_hash
        assert parsed.kernel != schema.kernel
        assert parsed.kernel.same_kernel(schema.kernel)
        for bad in (" TEST_author", "TEST_author\n", "a\tb", 1):
            with pytest.raises(ValueError, match=r"^a\.json: kernel\.author"):
                parse(with_author(bad), "a.json")

    @pytest.mark.parametrize(
        "content, line",
        [
            ("x = 1\ny = 2\n", None),
            ("x = 1\r\ny = 2\n", "1: forbidden character '\\r'"),
            ("x = 1\n\ty = 2\n", "2: forbidden character '\\t'"),
            ("\ufeffx = 1\n", "1: byte order mark"),
            ("x = 1\ny = 2", "2: no final newline"),
            ("x = 1\n\n", "2: blank line at the end"),
            ("# -*- coding: latin-1 -*-\nx = 1\n", "1: encoding declaration"),
        ],
        ids=["multi_line", "cr", "tab", "bom", "no_newline", "blank_end", "latin1"],
    )
    def test_source_content_is_one_checked_string(self, content: str, line: str | None) -> None:
        """A source's ``content`` is the file's text; a bad one names its line."""
        data = json.loads(OpCase.of("gemm@1.0").schema().schema_text())
        data["kernel"]["sources"][0]["content"] = content
        text = expr.json_file_text(data).encode()
        parse = ContractRegistry.default().parse_schema
        if line is None:
            (source,) = parse(text, "a.json").kernel.sources
            assert source.content == content
            return
        with pytest.raises(ValueError, match="^" + re.escape(f"a.json:kernel.py:{line}")):
            parse(text, "a.json")

    @pytest.mark.parametrize(
        ("language", "other_sources", "error"),
        [
            ("hip", {"glm5/moe.h": "#pragma once\n"}, None),
            ("hip", {"util.py": "x = 1\n"}, "paths do not end in .hip or .h"),
            ("triton", {"glm5/util.py": "x = 1\n"}, "paths are not single segments"),
            ("triton", {"util.hip": "int x;\n"}, "paths do not end in .py"),
        ],
        ids=["hip_nests", "hip_takes_no_py", "python_is_flat", "python_takes_no_hip"],
    )
    def test_kernel_paths_follow_the_language(
        self, language: str, other_sources: dict[str, str], error: str | None
    ) -> None:
        """hip sources are .hip and .h files, nested; a Python kernel's are flat .py files."""
        case = OpCase.of("gemm@1.0")
        if error is None:
            schema = case.schema("// K.\n", language=language, other_sources=other_sources)
            assert schema.kernel.entry_point == "kernel.hip::run"
            return
        with pytest.raises(ValueError, match=error):
            case.schema("x = 1\n", language=language, other_sources=other_sources)

    def test_hip_entry_file_is_a_translation_unit(self) -> None:
        data = json.loads(OpCase.of("gemm@1.0").schema("// K.\n", language="hip").schema_text())
        data["kernel"]["entry_point"] = "kernel.h::run"
        data["kernel"]["sources"][0]["path"] = "kernel.h"
        with pytest.raises(ValueError, match=r"'kernel\.h' is not a \.hip"):
            ContractRegistry.default().parse_schema(json.dumps(data).encode(), "a.json")

    def test_read_schemas_leaves_skipped_contracts_unchecked(self, tmp_path: Path) -> None:
        """A stale schema is an error, unless its contract is skipped (AKLR_DISABLE)."""
        registry, op_package = ContractRegistry.default(), "aklr.ops.gemm"
        op_dir = locate.op_dir(op_package)
        path = locate.committed_schema_paths(op_dir)[0]
        stale = tmp_path.joinpath(*op_package.split("."), path.relative_to(op_dir))
        stale.parent.mkdir(parents=True)
        stale.write_bytes(path.read_bytes().replace(b"\n  ", b"\n   ", 1))
        contract_id = json.loads(path.read_bytes())["contract_id"]
        for skipped, n_errors in [(frozenset(), 1), (frozenset({contract_id}), 0)]:
            _, errors = registry.read_schemas(op_package, skipped_contract_ids=skipped, measured_dir=tmp_path)
            assert sum(e.startswith(str(stale)) for e in errors) == n_errors


class TestMutableInput:
    """``TensorSpec.mutable``: the kernel may write the input in place, and it is
    compared as a result after the call."""

    def test_contract_rules(self) -> None:
        """A floating mutable input needs error_bounds, an input that is not mutable has
        none, and a mutable input is required and per_call."""
        contract = ContractRegistry.default().get("gdn_decode@1.0")
        assert contract.mutable_inputs == ("conv_state", "delta_state")
        assert contract.mutable_tensors == ("normalized", "conv_state", "delta_state")
        bounds = contract.inputs["delta_state"].error_bounds
        with pytest.raises(ValueError, match="delta_state: a floating mutable tensor needs"):
            _with_tensor(contract, "delta_state", error_bounds=())
        with pytest.raises(ValueError, match="projected_qkvz: only a mutable input has error_bounds"):
            _with_tensor(contract, "projected_qkvz", error_bounds=bounds)
        with pytest.raises(ValueError, match="delta_state: a mutable input is required"):
            _with_tensor(contract, "delta_state", lifetime="weight")
        presence = {"has_delta_state": attr("Whether delta_state is passed.", True, False)}
        optional = contract.model_copy(update={"attrs": presence})
        with pytest.raises(ValueError, match="delta_state: a mutable input is required"):
            _with_tensor(optional, "delta_state", present_when="has_delta_state")

    def test_compare_checks_it(self) -> None:
        """``compare_outputs`` checks a state as the call leaves it: one element off its
        reference fails it."""
        case = OpCase.of("gdn_decode@1.0")
        args = case.inputs()
        shape = case.contract.storage_shape_of("normalized", case.point)
        reference = {
            "normalized": torch.randn(shape, generator=torch.Generator().manual_seed(0), dtype=torch.float64),
            "conv_state": semantics.to_float64(args["conv_state"], "bfloat16"),
            "delta_state": semantics.to_float64(args["delta_state"], "float32"),
        }
        actual = {r: t.to(semantics.torch_dtypes(case.point.dtypes[r])[0]) for r, t in reference.items()}

        def compare() -> dict[str, semantics.CompareResult]:
            return semantics.compare_outputs(case.request.schema_json, actual, reference)

        assert all(r.is_passed for r in compare().values())
        flat = actual["delta_state"].view(-1)
        flat[0] = 0.0 if flat[0].abs() > 1 else 100.0
        results = compare()
        assert results["normalized"].is_passed and results["delta_state"].is_bound_exceeded


def _with_tensor(contract: OpContract, name: str, **update: object) -> OpContract:
    """``contract`` validated anew with ``update`` applied to its tensor ``name``."""
    fields = {f: getattr(contract, f) for f in type(contract).model_fields}
    group = "inputs" if name in contract.inputs else "outputs"
    tensors = getattr(contract, group)
    tensors = {**tensors, name: tensors[name].model_copy(update=update)}
    return OpContract.model_validate({**fields, group: tensors})
