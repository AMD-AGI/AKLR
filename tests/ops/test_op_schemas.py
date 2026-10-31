# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import re
from pathlib import Path

import pytest

import aklr.ops
from aklr.core.schema import Const, Schema, load_json_file
from aklr.ops.biased_grouped_topk import biased_grouped_topk
from aklr.ops.fused_moe import fused_moe
from tests.op_case import OpCase

_TEMPLATES = tuple(
    template
    for api in (
        *(getattr(aklr.ops, name) for name in aklr.ops.__all__),
        fused_moe,
        biased_grouped_topk,
    )
    for template in api.schemas
)
_COMMITTED = sorted(Path(aklr.ops.__file__).parent.glob("*/impls/*/*/*.json"))


class TestTemplates:
    def test_every_template_with_an_init_has_a_case(self) -> None:
        assert {t.schema_name({}) for t in _TEMPLATES if t.init is not None} == {
            case.id for case in OpCase.all()
        }

    def test_names_are_unique(self) -> None:
        names = [t.schema_name({}) for t in _TEMPLATES]
        assert len(names) == len(set(names)), names

    @pytest.mark.parametrize("template", _TEMPLATES, ids=lambda t: t.schema_name({}))
    def test_names_fit_a_file_name(self, template: Schema) -> None:
        name = template.schema_name({n: 2**31 for n, a in template.axes.items() if isinstance(a, Const)})
        assert len(name.encode()) <= 255
        assert re.fullmatch(r"[a-z0-9]+(_[a-z0-9]+)*", name)

    @pytest.mark.parametrize("path", _COMMITTED, ids=lambda p: p.parent.name)
    def test_committed_schema_is_named_by_a_template(self, path: Path) -> None:
        schema = load_json_file(Schema, path)
        assert schema.name in {
            t.schema_name(schema.const_axes) for t in _TEMPLATES if t.op_type == schema.op_type
        }
