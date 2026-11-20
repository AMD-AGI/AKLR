# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import sys
import types

import pytest

from aklr.core.contract.fastkey import FastKeyCodec
from aklr.core.contract.spec import AxisSpec
from tests.op_case import OpCase


class _Runnable:
    """A kernel whose module declares RANGE_SAFE as given (``_dispatcher._is_range_safe``)."""

    def __init__(self, monkeypatch, range_safe):
        name = f"tests_range_safe_{range_safe}"
        module = types.ModuleType(name)
        module.RANGE_SAFE = range_safe
        monkeypatch.setitem(sys.modules, name, module)
        self.metadata = types.SimpleNamespace(misc={"module_name": name})


def _key(codec, case, n_tokens):
    point = case.point.model_copy(update={"axes": {**case.point.axes, "n_tokens": n_tokens}})
    return codec.match_key(codec.to_fast_key(point))


def test_a_bucketed_axis_matches_its_power_of_two_range() -> None:
    """fused_moe's n_tokens: 3000 matches 3072 (both 12 bits), not 4096 or 2047."""
    case = OpCase.of("fused_moe@1.0")
    codec = FastKeyCodec.from_contract(case.contract)
    assert _key(codec, case, 3000) == _key(codec, case, 3072)
    assert _key(codec, case, 3000) != _key(codec, case, 4096)
    assert _key(codec, case, 3000) != _key(codec, case, 2047)


def test_an_axis_is_any_size_or_bucketed_not_both() -> None:
    with pytest.raises(ValueError, match="not both"):
        AxisSpec(kind="int", description="x", lo=1, hi=8, is_var=True, any_size=True, bucketed=True)
    with pytest.raises(ValueError, match="var axis"):
        AxisSpec(kind="int", description="x", lo=1, hi=8, bucketed=True)


@pytest.mark.parametrize(("slow_n_tokens", "served"), [(None, True), (3500, False), (5000, True)])
def test_a_schema_serves_a_range_only_if_each_of_its_workloads_there_qualifies(
    monkeypatch, slow_n_tokens, served
) -> None:
    """A schema measured at 3072 tokens (2.0x) serves 3000 (same range); a second workload of
    its in that range at 0.5x leaves the range unserved, one in another range does not."""
    import torch

    from aklr.core.runtime.dispatch._dispatcher import DispatchTable

    case = OpCase.of("fused_moe@1.0")
    contract = case.contract
    codec = FastKeyCodec.from_contract(contract)

    def workload(n_tokens, speedup):
        point = case.point.model_copy(update={"axes": {**case.point.axes, "n_tokens": n_tokens}})
        w = contract.workload_for(point, {})
        return w.model_copy(update={"accuracy": "passed", "speedup": speedup})

    workloads = [workload(3072, 2.0)] + ([workload(slow_n_tokens, 0.5)] if slow_n_tokens else [])
    schema = contract.narrow(case.point, workloads=[w.model_dump(mode="json", exclude_none=True) for w in workloads],
                             kernel={"language": "triton", "targets": [{"arch": "gfx950", "hardware": "MI355X"}],
                                     "entry_point": "kernel.py::run", "sources": []})  # fmt: skip
    runnable = _Runnable(monkeypatch, True)
    table = DispatchTable.from_schemas(
        contract, torch.device("cpu"), [(schema, runnable, schema.workloads)], min_speedup=1.0
    )
    assert (table.runnables.get(_key(codec, case, 3000)) is runnable) is served


@pytest.mark.parametrize("range_safe", [True, False], ids=["range_safe", "not_range_safe"])
def test_a_measured_value_is_served_by_its_own_workload_first(monkeypatch, range_safe) -> None:
    """Measured at 3072 (2.0x) and 3500 (0.5x): its range is unserved, yet 3072 itself is served, 3500 is not, and
    3000 (measured by none) falls back, with the range's reason for a RANGE_SAFE kernel; a kernel that does not
    declare it serves its measured values alone."""
    import torch

    from aklr.core.runtime.dispatch import FallbackReason
    from aklr.core.runtime.dispatch._dispatcher import DispatchTable

    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: False
    )  # no device needed for a lookup
    case = OpCase.of("fused_moe@1.0")
    contract = case.contract

    def point(n_tokens):
        return case.point.model_copy(update={"axes": {**case.point.axes, "n_tokens": n_tokens}})

    def workload(n_tokens, speedup):
        w = contract.workload_for(point(n_tokens), {})
        return w.model_copy(update={"accuracy": "passed", "speedup": speedup})

    workloads = [workload(3072, 2.0), workload(3500, 0.5)]
    schema = contract.narrow(case.point, workloads=[w.model_dump(mode="json", exclude_none=True) for w in workloads],
                             kernel={"language": "triton", "targets": [{"arch": "gfx950", "hardware": "MI355X"}],
                                     "entry_point": "kernel.py::run", "sources": []})  # fmt: skip
    runnable = _Runnable(monkeypatch, range_safe)
    table = DispatchTable.from_schemas(
        contract, torch.device("cpu"), [(schema, runnable, schema.workloads)], min_speedup=1.0
    )
    codec = FastKeyCodec.from_contract(contract)
    table = DispatchTable(  # the lookup of a call whose key is the Point's
        lambda call: codec.to_fast_key(call), table.alloc_outputs, table.runnables, table.warmed_keys,
        table.fallback_reasons, table.match_key, table.exact_key, table.spans, table.bucketed_indices,
    )  # fmt: skip
    assert table.lookup(point(3072))[2] is runnable
    assert table.lookup(point(3500))[2] is FallbackReason.NOT_QUALIFIED
    assert table.lookup(point(3000))[2] is (
        FallbackReason.NOT_QUALIFIED if range_safe else FallbackReason.NO_WORKLOAD
    )


def test_a_kernel_not_range_safe_serves_its_measured_values_alone(monkeypatch) -> None:
    """Measured at 3072 (2.0x) without RANGE_SAFE: 3072 is served, 3000 in its range is not."""
    import torch

    from aklr.core.runtime.dispatch._dispatcher import DispatchTable

    case = OpCase.of("fused_moe@1.0")
    contract = case.contract
    codec = FastKeyCodec.from_contract(contract)
    point = case.point.model_copy(update={"axes": {**case.point.axes, "n_tokens": 3072}})
    w = contract.workload_for(point, {}).model_copy(update={"accuracy": "passed", "speedup": 2.0})
    schema = contract.narrow(case.point, workloads=[w.model_dump(mode="json", exclude_none=True)],
                             kernel={"language": "triton", "targets": [{"arch": "gfx950", "hardware": "MI355X"}],
                                     "entry_point": "kernel.py::run", "sources": []})  # fmt: skip
    runnable = _Runnable(monkeypatch, False)
    table = DispatchTable.from_schemas(
        contract, torch.device("cpu"), [(schema, runnable, schema.workloads)], min_speedup=1.0
    )
    assert table.runnables.get(codec.exact_key(codec.to_fast_key(point))) is runnable
    assert _key(codec, case, 3000) not in table.runnables


def test_a_range_is_served_only_between_its_lowest_and_highest_measured_values(monkeypatch) -> None:
    """Measured at 2100 and 3072 (both 2.0x, the range [2048, 4096)): 3000 is served, 2050 and 3500 (outside what
    was measured, where a kernel's last tile class may end) fall back."""
    import torch

    from aklr.core.runtime.dispatch import FallbackReason
    from aklr.core.runtime.dispatch._dispatcher import DispatchTable

    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: False
    )  # no device needed for a lookup
    case = OpCase.of("fused_moe@1.0")
    contract = case.contract
    codec = FastKeyCodec.from_contract(contract)

    def point(n_tokens):
        return case.point.model_copy(update={"axes": {**case.point.axes, "n_tokens": n_tokens}})

    workloads = [
        contract.workload_for(point(n), {}).model_copy(update={"accuracy": "passed", "speedup": 2.0})
        for n in (2100, 3072)
    ]
    schema = contract.narrow(case.point, workloads=[w.model_dump(mode="json", exclude_none=True) for w in workloads],
                             kernel={"language": "triton", "targets": [{"arch": "gfx950", "hardware": "MI355X"}],
                                     "entry_point": "kernel.py::run", "sources": []})  # fmt: skip
    runnable = _Runnable(monkeypatch, True)
    table = DispatchTable.from_schemas(
        contract, torch.device("cpu"), [(schema, runnable, schema.workloads)], min_speedup=1.0
    )
    table = DispatchTable(  # the lookup of a call whose key is the Point's
        lambda call: codec.to_fast_key(call), table.alloc_outputs, table.runnables, table.warmed_keys,
        table.fallback_reasons, table.match_key, table.exact_key, table.spans, table.bucketed_indices,
    )  # fmt: skip
    assert table.lookup(point(3000))[2] is runnable
    assert table.lookup(point(2100))[2] is runnable
    assert table.lookup(point(3500))[2] is FallbackReason.NO_WORKLOAD
    assert table.lookup(point(2050))[2] is FallbackReason.NO_WORKLOAD
