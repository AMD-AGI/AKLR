# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import ast

from aklr.core.contract.registry import ContractRegistry


def test_fused_moe_kernels_can_load_independently_of_the_timing_baseline():
    registry = ContractRegistry.default()
    contract = registry.get("fused_moe@1.0")
    schemas, errors = registry.read_schemas(contract.op_package)
    assert not errors
    dependencies = {
        (path.name, node.module)
        for path, schema in schemas
        for source in schema.kernel.sources
        if source.path.endswith(".py")
        for node in ast.walk(ast.parse(source.content))
        if isinstance(node, ast.ImportFrom) and (node.module or "").endswith(".baseline")
    }
    assert not dependencies


def test_public_aiter_operators_keep_the_host_callable_and_output_convention(monkeypatch):
    import sys
    from types import ModuleType

    from aklr.plugins.aiter import api

    norm = ModuleType("aiter.ops.rmsnorm")

    def rms_norm(input, weight, epsilon, use_model_sensitive_rmsnorm=0):
        return input, epsilon, use_model_sensitive_rmsnorm

    def rmsnorm2d_fwd_with_add(
        out,
        input,
        residual_in,
        residual_out,
        weight,
        epsilon,
        gemma_norm=False,
        use_model_sensitive_rmsnorm=0,
    ):
        residual_out[:] = [input + residual_in]
        out[:] = [weight]

    norm.rms_norm = rms_norm
    norm.rmsnorm2d_fwd_with_add = rmsnorm2d_fwd_with_add
    monkeypatch.setitem(sys.modules, norm.__name__, norm)
    assert api.rms_norm is norm.rms_norm
    assert api.rms_norm(input=3, weight=4, epsilon=1e-5) == (3, 1e-5, 0)
    out, residual_out = [], []
    assert api.rmsnorm2d_fwd_with_add(out, 3, 5, residual_out, 4, 1e-5) is None
    assert (out, residual_out) == ([4], [8])
    assert not hasattr(api, "fp8_paged_mqa_logits_verify")
    assert not hasattr(api, "gemm")


def test_old_aiter_measurement_tree_is_read_after_family_migration(tmp_path):
    import shutil

    registry = ContractRegistry.default()
    pkg = registry.get("add_rms_norm@1.0").op_package
    passed, errors = registry.read_schemas(pkg)
    assert not errors
    committed = next(path for path, record in passed if path.stem == "1f0b6c68db064ca4")
    relative = committed.relative_to(committed.parents[3])
    old = tmp_path / "aklr/ops/rms_norm" / relative
    old.parent.mkdir(parents=True)
    shutil.copyfile(committed, old)
    passed, errors = registry.read_schemas(pkg, measured_dir=tmp_path)
    assert not errors
    assert old in [path for path, _ in passed]
    new = tmp_path / "aklr/plugins/aiter/rms_norm" / relative
    new.parent.mkdir(parents=True)
    shutil.copyfile(old, new)
    passed, errors = registry.read_schemas(pkg, measured_dir=tmp_path)
    assert not errors
    assert new in [path for path, _ in passed]


def test_historical_measurements_report_a_corrupt_committed_schema(tmp_path, monkeypatch):
    from aklr.core.contract import locate

    registry = ContractRegistry.default()
    family = registry.get("add_rms_norm@1.0").op_package
    original = next(locate.op_dir(family).rglob("1f0b6c68db064ca4.json"))
    relative = original.relative_to(locate.op_dir(family))
    root = tmp_path / "checkout" / "aklr/plugins/aiter/rms_norm"
    committed = root / relative
    committed.parent.mkdir(parents=True)
    committed.write_text("{}\n")
    mirror = tmp_path / "reports"
    overlay = mirror / "aklr/ops/rms_norm" / relative
    overlay.parent.mkdir(parents=True)
    overlay.write_bytes(original.read_bytes())
    monkeypatch.setattr(locate, "op_dir", lambda package: root)
    results, errors = registry.read_schemas(family, measured_dir=mirror)
    assert not results and len(errors) == 1
    assert str(committed) in errors[0]


def test_new_aiter_families_publish_their_kernel_contracts_from_plugin():
    import importlib.util

    registry = ContractRegistry.default()
    contracts = [c for c in registry.contracts() if c.op_package.startswith("aklr.plugins.aiter.")]
    assert len(contracts) == 28
    for contract in contracts:
        if contract.op_type != "situv2_a4w4_moe":
            assert contract.api_name == contract.definition_name
    assert importlib.util.find_spec("aklr.ops") is None


def test_published_fused_moe_module_exposes_the_native_call_after_family_discovery(monkeypatch):
    import sys
    from types import ModuleType

    upstream = ModuleType("aiter.fused_moe")

    def fused_moe(hidden_states, w1, w2, topk_weight, topk_ids, **kwargs):
        return kwargs["output"]

    upstream.fused_moe = fused_moe
    monkeypatch.setitem(sys.modules, upstream.__name__, upstream)
    ContractRegistry.default()
    from aklr.plugins.aiter.api import fused_moe as public
    from aklr.plugins.aiter.fused_moe import fused_moe as published

    assert published is public is fused_moe
