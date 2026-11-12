# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import re
import runpy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import aklr.ops
import aklr.seam  # noqa: F401 -- declares the seam adapters
from aklr.core.runtime.dispatch import aklr_api
from aklr.core.runtime.seam import SeamRegistry
from aklr.core.schema import Const, Schema, load_json_file, save_json_file
from tests.op_case import OpCase

# the ops' own: tests add templates of their toy ops to the default ones
_TEMPLATES = tuple(t for t in Schema.templates() if t.standard.reference.__module__.startswith("aklr.ops."))
_COMMITTED = sorted(Path(aklr.ops.__file__).parent.glob("*/impls/*/*/*.json"))


class TestTemplates:
    def test_every_op_declares_its_templates(self) -> None:
        assert {t.op_type for t in _TEMPLATES} >= {
            "biased_grouped_topk",
            "fused_moe",
            "gemm_bf16_nt",
            "mega_moe",
            "mhc_pre",
            "topk_transform_paged",
        }

    def test_every_template_with_an_init_has_a_case(self) -> None:
        assert {t.schema_name({}) for t in _TEMPLATES if t.standard.init is not None} == {
            case.id for case in OpCase.all()
        }

    def test_two_templates_of_one_name_are_refused(self) -> None:
        template = OpCase.from_id("gemm_bf16_nt").template
        Schema.declare_template(template)
        with pytest.raises(ValueError, match="gemm_bf16_nt"):
            Schema.declare_template(replace(template, description="another"))

    def test_an_api_declared_after_the_templates_are_read_is_among_them(self) -> None:
        Schema.templates()
        template = replace(OpCase.from_id("gemm_bf16_nt").template, name_prefix="gemm_bf16_nt_late")

        @aklr_api(template)
        def gemm_bf16_nt_late(a, b, bias=None, out=None): ...

        assert template in Schema.templates()

    @pytest.mark.parametrize("template", _TEMPLATES, ids=lambda t: t.schema_name({}))
    def test_names_fit_a_file_name(self, template: Schema) -> None:
        name = template.schema_name({n: 2**31 for n, a in template.axes.items() if isinstance(a, Const)})
        assert len(name.encode()) <= 255
        assert re.fullmatch(r"[a-z0-9]+(_[a-z0-9]+)*", name)

    @pytest.mark.parametrize("path", _COMMITTED, ids=lambda p: p.parent.name)
    def test_committed_schema_is_read_with_its_templates_standard(self, path: Path) -> None:
        schema = load_json_file(Schema, path)
        template = schema.template()
        assert (template.op_type, template.schema_name(schema.const_axes)) == (schema.op_type, schema.name)
        assert schema.standard is template.standard

    def test_a_json_no_template_names_is_read_without_a_standard(self, tmp_path: Path) -> None:
        save_json_file(
            replace(OpCase.from_id("gemm_bf16_nt").schema(), name="gemm_bf16_nt_unnamed"), tmp_path / "s.json"
        )
        schema = load_json_file(Schema, tmp_path / "s.json")
        assert (schema.template(), schema.standard) == (None, None)

    def test_a_schema_built_in_code_keeps_the_standard_it_is_given(self) -> None:
        assert replace(OpCase.from_id("gemm_bf16_nt").schema(), standard=None).standard is None


class TestStandards:
    @pytest.mark.parametrize("template", _TEMPLATES, ids=lambda t: t.schema_name({}))
    def test_every_standard_is_saved_as_standalone_modules(self, template: Schema, tmp_path: Path) -> None:
        template.standard.save(tmp_path)
        for path in tmp_path.glob("*.py"):
            assert callable(runpy.run_path(str(path))["run"]), path.name

    @pytest.mark.parametrize(
        "template",
        [t for t in _TEMPLATES if t.standard.baseline is not None],
        ids=lambda t: t.schema_name({}),
    )
    def test_a_baseline_is_timed_under_one_seam_backend(self, template: Schema) -> None:
        # the bench keys its speedups by this backend, the name dispatch looks them up by
        assert len({a.backend for a in SeamRegistry.default().adapters(template.op_type)}) == 1


@pytest.mark.parametrize(("rows", "act_quant"), [(255, None), (256, "mxfp4")])
def test_aiters_swiglu_call_binds_the_swiglu_template_with_aiters_activation_quant(
    rows: int, act_quant: str | None
) -> None:
    """aiter keeps swiglu's activations bf16 below 256 rows (GPTOSS_SWIGLU_MXFP4_BF16_BOUND), MXFP4 from there."""
    import torch

    from aklr.ops.fused_moe.schema import fused_moe_op_schema

    (adapter,) = (
        a for a in SeamRegistry.default().adapters("fused_moe") if a.entry == "aiter.fused_moe.fused_moe"
    )
    w1 = torch.empty(8, 512, 256, dtype=torch.float4_e2m1fn_x2, device="meta")
    w1.is_shuffled = True
    bound = adapter.bind_fn(
        torch.empty(rows, 512, dtype=torch.bfloat16, device="meta"),
        w1,
        torch.empty(8, 512, 128, dtype=torch.float4_e2m1fn_x2, device="meta"),
        torch.empty(rows, 2, device="meta"),
        torch.empty(rows, 2, dtype=torch.int32, device="meta"),
        activation=SimpleNamespace(name="Swiglu"),
        quant_type=SimpleNamespace(name="per_1x32"),
        swiglu_limit=7.0,
    )
    assert bound is not None and bound["act_quant"] == act_quant
    assert fused_moe_op_schema(**bound).schema_name({}) == "fused_moe_swiglu_shuffled_16x16"
