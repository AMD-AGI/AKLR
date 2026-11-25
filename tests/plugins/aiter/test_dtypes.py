# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Where a tensor's dtype is written: read off the call into the definition, not declared by the template."""

import pytest

from aklr.core.platform.env import ENV_SCHEMA_PATH
from aklr.core.runtime.dispatch import OpVariants
from aklr.core.schema import SchemaCollection, Tensor


def test_a_template_declares_a_dtype_only_for_an_input_a_call_may_leave_out() -> None:
    declared = []
    for api in OpVariants.ops().values():
        if not api.__module__.startswith("aklr.plugins.aiter."):
            continue  # an api another test declared
        tensors = [
            (v, name, t) for v in api.variants for name, t in v.inputs.items() if isinstance(t, Tensor)
        ]
        # one Tensor is shared by an op's variants: its dtype stands for the variants whose call leaves it out
        left_out = {name for _, name, tensor in tensors if tensor.optional}
        declared += [
            f"{variant.op_name}.{name}"
            for variant, name, tensor in tensors
            if name not in left_out and tensor.dtype is not None
        ]
    assert declared == []


def test_a_definition_records_the_dtype_of_every_required_tensor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(
        ENV_SCHEMA_PATH
    )  # with no path the collection is the checkout's dataset, aklr/plugins/aiter
    ops = OpVariants.ops()
    unrecorded = []
    for name, schemas in SchemaCollection.from_path().schemas.items():
        for schema in schemas:
            api = ops.get(schema.op_type)
            variant = None if api is None else api.variant_of(schema)
            if variant is None:
                continue
            tensors = variant.inputs | variant.outputs
            unrecorded += [
                f"{name}.{key}"
                for key, tensor in (schema.inputs | schema.outputs).items()
                if isinstance(tensor, Tensor) and not tensors[key].optional and tensor.dtype is None
            ]
    assert unrecorded == []
