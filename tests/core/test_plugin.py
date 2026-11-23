# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import inspect
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from aklr.core.contract.registry import UnknownContractError
from aklr.core.plugin import Plugin
from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins.aiter.rms_norm.plugin import RMS_NORM, rms_norm
from aklr.plugins.aiter.situv2_a4w4_moe.contract import SITUV2_A4W4_MOE


def test_new_family_needs_only_a_plugin_module_to_publish_and_call_its_op(tmp_path):
    # Import an isolated installation whose new family has no compatibility modules or assets.
    import aklr

    root = tmp_path / "aklr"
    shutil.copytree(Path(aklr.__file__).parent, root, ignore=shutil.ignore_patterns("impls", "__pycache__"))
    package = root / "plugins/aiter/sample"
    package.mkdir()
    (package / "__init__.py").touch()
    (package / "plugin.py").write_text(
        textwrap.dedent("""
        from aklr.plugins import Plugin
        from aklr.plugins.aiter.rms_norm.plugin import RMS_NORM

        family = Plugin("aiter")
        family.define(RMS_NORM.model_copy(update={"op_type": "sample", "contract_id": "sample@1.0"}))
        family.build()

        @family.op
        def sample(x, weight, epsilon): ...
    """)
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent("""
        from aklr.core.contract.locate import import_entry, op_dir
        from aklr.core.contract.registry import ContractRegistry
        from aklr.core.runtime.dispatch import FallbackReason

        registry = ContractRegistry.default()
        contract = registry.get("sample@1.0")
        assert contract.api_name == "aklr.plugins.aiter.sample.plugin.sample"
        assert op_dir(contract.op_package).name == "sample"
        op = import_entry(contract.api_name)
        assert registry.for_api(op) is contract
        assert op(None, None, 1e-5) is FallbackReason.NO_SCHEMA
    """),
        ],
        cwd=tmp_path,
        check=True,
    )


def test_plugin_authors_import_the_core_plugin_class():
    from aklr.plugins import Plugin as AuthorPlugin

    assert AuthorPlugin is Plugin


def test_op_binds_the_signature_and_dispatch_queries_to_a_declared_contract(default_dispatchers):
    family = Plugin("aiter")
    family.define(RMS_NORM)
    signature = inspect.unwrap(rms_norm)
    with pytest.raises(RuntimeError, match="build"):
        family.op(signature)
    op = family.build().op(signature)
    assert inspect.signature(op) == inspect.signature(rms_norm)
    for call in (op, op.serving, op.fallback_reason):
        assert call(None, None, epsilon=1e-5) is FallbackReason.NO_SCHEMA


def test_op_can_only_bind_a_contract_declared_on_its_own_instance():
    with pytest.raises(UnknownContractError, match="rms_norm"):
        Plugin("aiter").build().op(inspect.unwrap(rms_norm))


def test_family_publishes_the_declared_contract_after_build():
    family = Plugin("aiter")
    with pytest.raises(RuntimeError, match="build"):
        _ = family.contracts
    family.build(lambda plugin: plugin.define(SITUV2_A4W4_MOE))
    assert family.name == "aiter"
    assert len(family.contracts) == 1
    assert family.contracts[0] is SITUV2_A4W4_MOE


def test_failed_build_rolls_back_and_frozen_build_is_idempotent():
    family = Plugin("aiter")

    def declare(plugin):
        plugin.define(SITUV2_A4W4_MOE)

    def recurse(plugin):
        plugin.build()

    with pytest.raises(RuntimeError, match="recursive"):
        family.build(declare, recurse)
    assert family.build(declare).build(declare).build() is family
    assert family.contracts == (SITUV2_A4W4_MOE,)
    with pytest.raises(RuntimeError, match="frozen"):
        family.build(recurse)
    with pytest.raises(RuntimeError, match="frozen"):
        family.define(SITUV2_A4W4_MOE)


def test_contract_owner_and_duplicate_are_checked_before_freezing():
    contract = SITUV2_A4W4_MOE
    with pytest.raises(ValueError, match="aiter"):
        Plugin("sglang").define(contract)
    plugin = Plugin("aiter")
    plugin.define(contract)
    with pytest.raises(ValueError, match=contract.contract_id):
        plugin.build(lambda p: p.define(contract))
    assert plugin.build().contracts == (contract,)


def test_provide_keeps_preexisting_versions_when_a_fragment_conflicts():
    provider = Plugin("aiter")

    def old(args, kwargs):
        return None

    def new(args, kwargs):
        return None

    provider.provide("rmsnorm", version=1, select=old)

    def conflicting(plugin):
        plugin.provide("rmsnorm", version=2, select=new)
        plugin.provide("rmsnorm", version=1, select=new)

    with pytest.raises(ValueError, match="rmsnorm.*old.*new"):
        provider.build(conflicting)
    provider.provide("rmsnorm", version=2, select=new)
    provider.build()
    with pytest.raises(RuntimeError, match="frozen"):
        provider.provide("third", version=1, select=new)


def test_wrap_requires_one_callback_and_claims_each_supported_release():
    plugin = Plugin("sglang")
    target = "sglang.Model.forward"
    for callbacks in ({}, {"replace": str, "around": str}, {"replace": str, "around": None}, {"around": 1}):
        with pytest.raises(TypeError, match="callback|callable"):
            plugin.wrap(target, versions=("0.5.22",), **callbacks)
    plugin.wrap(target, versions=("0.5.23",), around=str)
    with pytest.raises(ValueError, match="0.5.23.*str.*bytes"):
        plugin.wrap(target, versions=("0.5.22", "0.5.23"), replace=bytes)
    plugin.wrap(target, versions=("0.5.22",), replace=bytes, eager=True)
    plugin.build()


def test_intercept_claims_all_callees_or_none_per_release():
    vllm = Plugin("vllm")
    vllm.intercept("model.forward", "lib.first", versions=("0.34.0",), replace=str, group="model")
    with pytest.raises(ValueError, match="lib.first.*str.*bytes"):
        vllm.intercept("model.other", "lib.second", "lib.first", versions=("0.34.0",), replace=bytes)
    vllm.intercept("model.other", "lib.second", versions=("0.34.0",), replace=bytes)
    with pytest.raises(TypeError, match="callee"):
        vllm.intercept("model.forward", versions=("0.34.0",), replace=str)
    with pytest.raises(ValueError, match="sglang"):
        vllm.wrap("model.forward", versions=("0.34.0",), replace=str)


def test_duplicate_declarations_name_both_registration_sites_for_the_same_callback():
    plugin = Plugin("aiter")

    def select(args, kwargs):
        return None

    plugin.provide("rmsnorm", version=1, select=select)
    with pytest.raises(ValueError, match=r"test_plugin\.py:\d+.*test_plugin\.py:\d+") as caught:
        plugin.provide("rmsnorm", version=1, select=select)
    import re

    assert len(set(re.findall(r"test_plugin\.py:(\d+)", str(caught.value)))) == 2


@pytest.mark.parametrize("name", ["", "aiter.rms_norm", "bad-name", None])
def test_namespace_must_be_a_package_name(name):
    with pytest.raises(ValueError, match="name"):
        Plugin(name)


def test_define_requires_an_op_contract():
    with pytest.raises(TypeError, match="OpContract"):
        Plugin("aiter").define(object())


def test_concurrent_build_runs_the_fragment_once_and_freezes_late_writes():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    plugin = Plugin("aiter")
    entered, release = Event(), Event()
    calls = []

    def declare(p):
        calls.append(1)
        entered.set()
        assert release.wait(5)
        p.define(SITUV2_A4W4_MOE)

    with ThreadPoolExecutor(max_workers=3) as workers:
        first = workers.submit(plugin.build, declare)
        assert entered.wait(5)
        second = workers.submit(plugin.build, declare)
        write = workers.submit(plugin.define, SITUV2_A4W4_MOE)
        release.set()
        assert first.result() is second.result() is plugin
        with pytest.raises(RuntimeError, match="frozen"):
            write.result()
    assert calls == [1] and plugin.contracts == (SITUV2_A4W4_MOE,)
