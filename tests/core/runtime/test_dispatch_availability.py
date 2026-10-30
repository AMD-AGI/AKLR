# Created by OpenAI
"""Workload preflight avoids input preparation without executing a kernel."""

from types import SimpleNamespace

from aklr.core.contract.fastkey import FastKeyCodec
from aklr.core.runtime.dispatch import KernelDispatcher
from tests.op_case import OpCase


def test_closed_and_missing_points_are_not_available():
    case = OpCase.of("gemm@1.0")
    dispatcher = KernelDispatcher(case.contract)
    assert not dispatcher.has_workload(case.point)
    codec = FastKeyCodec.from_contract(case.contract)
    dispatcher.table = SimpleNamespace(match_key=codec.match_key, runnables={})
    assert not dispatcher.has_workload(case.point)


def test_only_a_matching_runnable_is_available_without_execution():
    case = OpCase.of("gemm@1.0")
    dispatcher = KernelDispatcher(case.contract)
    codec = FastKeyCodec.from_contract(case.contract)
    key = codec.match_key(codec.to_fast_key(case.point))
    dispatcher.table = SimpleNamespace(match_key=codec.match_key, runnables={key: object()})
    assert dispatcher.has_workload(case.point)
    axis = next(iter(case.contract.var_axes))
    changed = case.point.model_copy(update={"axes": {**case.point.axes, axis: case.point.axes[axis] + 1}})
    assert not dispatcher.has_workload(changed)


def test_any_size_pool_capacity_matches_but_invalid_capacity_does_not():
    case = OpCase.of("gdn_prefill_group_fp8_quant@1.0")
    dispatcher = KernelDispatcher(case.contract)
    codec = FastKeyCodec.from_contract(case.contract)
    key = codec.match_key(codec.to_fast_key(case.point))
    dispatcher.table = SimpleNamespace(match_key=codec.match_key, runnables={key: object()})
    expanded = case.point.model_copy(update={"axes": {**case.point.axes, "n_slots": 8}})
    assert dispatcher.has_workload(expanded)
    invalid = case.point.model_copy(update={"axes": {**case.point.axes, "n_slots": 1}})
    assert not dispatcher.has_workload(invalid)
