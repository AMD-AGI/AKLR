# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import json
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import pytest
import torch

import aklr.ops
from aklr.core.platform.device import DeviceTarget
from aklr.core.platform.env import ENV_MIN_SPEEDUP, RuntimeEnv
from aklr.core.runtime.dispatch import (
    DispatchConfig,
    DispatchConfigRegistry,
    DispatchTable,
)
from aklr.core.schema import Evaluation, Schema, SchemaCollection, Speedup
from tests.op_case import GEMM_KERNEL as GEMM
from tests.op_case import OpCase, dispatcher

_GEMM_RETURNING = "def run(a, b, bias):\n    return a @ b.mT\n"
_UNBUILT = DispatchConfig(aot_ratio=0.0)


def _evaluation(accuracy: str, speedup: float) -> Evaluation:
    return Evaluation(
        hardware=str(DeviceTarget.current()),
        accuracy=accuracy,
        speedup={"aiter": Speedup(commit="v0", value=speedup)},
    )


@pytest.mark.gpu
@pytest.mark.usefixtures("builder_registry")
class TestKernelDispatcher:
    @pytest.mark.parametrize(
        ("source", "dps"),
        [(GEMM, True), (_GEMM_RETURNING, False)],
        ids=["dps", "returning"],
    )
    def test_serves_a_qualified_workload(self, source: str, dps: bool) -> None:
        case = OpCase.of("gemm_bf16_nt")
        call = case.call()
        schema = case.served(source, dps=dps)
        out = dispatcher(schema).dispatch(schema.name, (), {"a": call["a"], "b": call["b"]}, lambda **_: None)
        assert torch.equal(out, call["a"] @ call["b"].mT)

    def test_fills_the_callers_output(self) -> None:
        case = OpCase.of("gemm_bf16_nt")
        call = case.call()
        schema = case.served(GEMM)
        out = torch.empty(4, 32, dtype=torch.bfloat16)
        result = dispatcher(schema).dispatch(
            schema.name, (), {"a": call["a"], "b": call["b"], "out": out}, None
        )
        assert result is out and torch.equal(out, call["a"] @ call["b"].mT)

    @pytest.mark.parametrize(
        ("evaluations", "m", "baseline"),
        [
            pytest.param(None, 8, None, id="other_workload"),
            pytest.param([], 4, None, id="unmeasured"),
            pytest.param([_evaluation("failed", 2.0)], 4, None, id="failed"),
            pytest.param([_evaluation("passed", 0.5)], 4, "aiter", id="slower"),
        ],
    )
    def test_falls_back_on_a_miss(self, evaluations: list | None, m: int, baseline: str | None) -> None:
        case = OpCase.of("gemm_bf16_nt")
        schema = case.served(GEMM, evaluations=evaluations)
        a, b = torch.ones(m, 64, dtype=torch.bfloat16), case.call()["b"]
        assert (
            dispatcher(schema).dispatch(schema.name, (), {"a": a, "b": b}, lambda **_: "fallback", baseline)
            == "fallback"
        )

    def test_unmeasured_workload_is_served_at_min_speedup_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(ENV_MIN_SPEEDUP, "0")
        case = OpCase.of("gemm_bf16_nt")
        call = case.call()
        schema = case.served(GEMM, evaluations=[])
        out = dispatcher(schema).dispatch(schema.name, (), {"a": call["a"], "b": call["b"]}, lambda **_: None)
        assert torch.equal(out, call["a"] @ call["b"].mT)

    def test_miss_without_fallback_raises(self) -> None:
        case = OpCase.of("gemm_bf16_nt")
        with pytest.raises(RuntimeError, match="no fallback"):
            dispatcher().dispatch(case.schema().name, (), {"a": case.call()["a"]}, None)


@pytest.mark.gpu
@pytest.mark.usefixtures("builder_registry")
class TestDispatchTable:
    @staticmethod
    def _load(schema: Schema, **inputs: Any) -> DispatchTable:
        return DispatchTable.load_or_build(
            **{
                "schema_collection": SchemaCollection(schemas={schema.name: [schema]}),
                "config_registry": DispatchConfigRegistry(default=_UNBUILT),
                "target": DeviceTarget.current(),
                "min_speedup": 1.03,
            }
            | inputs
        )

    def test_second_load_reads_the_cache_file_of_its_digest(self) -> None:
        schema = OpCase.of("gemm_bf16_nt").served(GEMM)
        first = self._load(schema)
        (path,) = RuntimeEnv().build_cache_path.rglob(f"{first.digest}.json")
        written = path.stat().st_mtime_ns
        second = self._load(schema)
        assert json.loads(path.read_text())["digest"] == first.digest
        assert (second.digest, second.index) == (first.digest, first.index)
        assert path.stat().st_mtime_ns == written

    @pytest.mark.parametrize(
        ("change", "is_input"),
        [
            pytest.param(
                lambda s: (
                    s,
                    {
                        "config_registry": DispatchConfigRegistry(
                            default=_UNBUILT,
                            per_schema={s.name: DispatchConfig(aot_ratio=0.5)},
                        )
                    },
                ),
                True,
                id="per_schema_config",
            ),
            pytest.param(
                lambda s: (s, {"config_registry": DispatchConfigRegistry()}),
                True,
                id="schema_not_dispatched",
            ),
            pytest.param(
                lambda s: (
                    s,
                    {"target": DeviceTarget(DeviceTarget.current().arch, "other")},
                ),
                True,
                id="target",
            ),
            pytest.param(lambda s: (s, {"min_speedup": 0.0}), True, id="min_speedup"),
            pytest.param(
                lambda s: (
                    OpCase.of("gemm_bf16_nt").served(GEMM, evaluations=[_evaluation("failed", 2.0)]),
                    {},
                ),
                True,
                id="evaluation_on_the_target",
            ),
            pytest.param(
                lambda s: (
                    replace(
                        s,
                        workloads=[
                            replace(
                                w,
                                evaluations=[
                                    *w.evaluations,
                                    Evaluation(
                                        hardware=f"{DeviceTarget.current().arch}/other",
                                        accuracy="failed",
                                    ),
                                ],
                            )
                            for w in s.workloads
                        ],
                    ),
                    {},
                ),
                False,
                id="evaluation_on_other_hardware",
            ),
            pytest.param(
                lambda s: (replace(s, description="changed"), {}),
                False,
                id="description",
            ),
        ],
    )
    def test_digest_follows_every_input_of_the_table(
        self, change: Callable[[Schema], tuple[Schema, dict]], is_input: bool
    ) -> None:
        schema = OpCase.of("gemm_bf16_nt").served(GEMM)
        changed_schema, inputs = change(schema)
        changed = self._load(changed_schema, **inputs)
        assert (changed.digest != self._load(schema).digest) == is_input


@pytest.mark.gpu
@pytest.mark.usefixtures("builder_registry")
class TestOpApi:
    def test_active_dispatcher_serves_the_call(self) -> None:
        case = OpCase.of("gemm_bf16_nt")
        call = case.call()
        with dispatcher(case.served(GEMM)):
            out = aklr.ops.gemm_bf16_nt(call["a"], call["b"])
        assert torch.equal(out, call["a"] @ call["b"].mT)

    def test_call_after_the_dispatcher_stops_raises(self) -> None:
        case = OpCase.of("gemm_bf16_nt")
        call = case.call()
        with dispatcher(case.served(GEMM)):
            pass
        with pytest.raises(RuntimeError, match="no KernelDispatcher is active"):
            aklr.ops.gemm_bf16_nt(call["a"], call["b"])

    @pytest.mark.parametrize("shuffled", [False, True])
    def test_routes_by_the_schema_the_call_chooses(self, shuffled: bool) -> None:
        case = OpCase.of("gemm_bf16_nt_shuffled")
        call = case.call()
        with dispatcher(case.served(GEMM)):
            if shuffled:
                out = aklr.ops.gemm_bf16_nt(call["a"], call["b"], use_shuffled_weight=True)
                assert torch.equal(out, call["a"] @ call["b"].mT)
            else:
                with pytest.raises(RuntimeError, match="no fallback"):
                    aklr.ops.gemm_bf16_nt(call["a"], call["b"])

    def test_two_dispatchers_cannot_be_active(self) -> None:
        with dispatcher(), pytest.raises(RuntimeError, match="already active"):
            dispatcher().start()
