# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import gc
import os
import subprocess
import sys
import types
import weakref
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import aklr.ops
from aklr.core.platform import toolchain
from aklr.core.platform.device import DeviceTarget
from aklr.core.platform.env import ENV_DISABLE, ENV_ENABLE_TRACING, ENV_MIN_SPEEDUP, RuntimeEnv
from aklr.core.runtime.dispatch import (
    DispatchConfig,
    DispatchConfigRegistry,
    Dispatcher,
    OpVariants,
    aklr_api,
)
from aklr.core.schema import Evaluation, SchemaCollection, Speedup, save_json_file
from tests.op_sample import GEMM_KERNEL as GEMM
from tests.op_sample import GEMM_SAMPLE, OpSample, dispatcher

_GEMM_RETURNING = "def run(a, b, bias):\n    return a @ b.mT\n"


def _evaluation(accuracy: str, speedup: float) -> Evaluation:
    return Evaluation(
        hardware=str(DeviceTarget.current()),
        accuracy=accuracy,
        speedup={"aiter": Speedup(commit="v0", value=speedup)},
    )


@pytest.mark.gpu
@pytest.mark.usefixtures("builder_registry")
class TestDispatcher:
    @pytest.mark.parametrize(
        ("source", "dps"),
        [(GEMM, True), (_GEMM_RETURNING, False)],
        ids=["dps", "returning"],
    )
    def test_serves_a_qualified_workload(self, source: str, dps: bool) -> None:
        case = GEMM_SAMPLE
        call = case.call()
        schema = case.served(source, dps=dps)
        out = dispatcher(schema).call(schema, {"a": call["a"], "b": call["b"]}, fallback=lambda: None)
        assert torch.equal(out, call["a"] @ call["b"].mT)

    def test_fills_the_callers_output(self) -> None:
        case = GEMM_SAMPLE
        call = case.call()
        schema = case.served(GEMM)
        out = torch.empty(4, 32, dtype=torch.bfloat16)
        result = dispatcher(schema).call(schema, {"a": call["a"], "b": call["b"], "out": out})
        assert result is out
        assert torch.equal(out, call["a"] @ call["b"].mT)

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
        case = GEMM_SAMPLE
        schema = case.served(GEMM, evaluations=evaluations)
        a, b = torch.ones(m, 64, dtype=torch.bfloat16), case.call()["b"]
        assert (
            dispatcher(schema).call(schema, {"a": a, "b": b}, fallback=lambda: "fallback", baseline=baseline)
            == "fallback"
        )

    @pytest.mark.parametrize(
        "change",
        [
            pytest.param(lambda a: a.to(torch.float16), id="another_dtype"),
            pytest.param(lambda a: a[0], id="another_rank"),
        ],
    )
    def test_falls_back_on_a_call_the_kernel_was_not_recorded_for(self, change) -> None:
        case = GEMM_SAMPLE
        call = case.call()
        schema = case.served(GEMM)
        out = dispatcher(schema).call(
            schema, {"a": change(call["a"]), "b": call["b"]}, fallback=lambda: "fallback"
        )
        assert out == "fallback"

    def test_a_call_that_fell_back_is_freed_when_it_returns(self) -> None:
        case = GEMM_SAMPLE
        schema = case.served(GEMM)
        serving = dispatcher(schema)
        a = case.call()["a"][0].clone()  # of another rank: the schema cannot describe it
        freed = weakref.ref(a)
        gc.disable()  # a cycle would keep the call's tensors until a collection
        try:
            serving.call(schema, {"a": a, "b": case.call()["b"]}, fallback=lambda: "fallback")
            del a
            assert freed() is None
        finally:
            gc.enable()

    def test_a_direct_call_the_schema_cannot_describe_raises_with_the_reason(self) -> None:
        case = GEMM_SAMPLE
        schema = case.served(GEMM)
        with pytest.raises(RuntimeError, match="Dispatch miss") as raised:
            dispatcher(schema).call(schema, {"a": case.call()["a"][0], "b": case.call()["b"]})
        assert "defined dimension is 2" in str(raised.value.__cause__)

    def test_a_tuner_runs_no_kernel_on_a_call_of_another_dtype(self) -> None:
        case = GEMM_SAMPLE
        call = case.call()
        schema = case.served(GEMM)
        other = replace(schema, kernel=replace(schema.kernel, name="another"))
        asked = []
        tuner = types.SimpleNamespace(
            choose=lambda op_name, key, candidates, run: asked.append(candidates) or candidates[0]
        )
        tuned = Dispatcher(SchemaCollection(schemas={schema.name: [schema, other]}), tuner=tuner)
        half = {"a": call["a"].to(torch.float16), "b": call["b"]}
        assert tuned.call(schema, half, fallback=lambda: "fallback") == "fallback"
        assert asked == []
        tuned.call(schema, {"a": call["a"], "b": call["b"]}, fallback=lambda: "fallback")
        assert len(asked) == 1  # the tuner does choose for this key: a call of the recorded dtypes

    def test_falls_back_on_a_disabled_op_type(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(ENV_DISABLE, "another_op,gemm_bf16_nt")
        case = GEMM_SAMPLE
        call = case.call()
        schema = case.served(GEMM)
        out = dispatcher(schema).call(schema, {"a": call["a"], "b": call["b"]}, fallback=lambda: "fallback")
        assert out == "fallback"

    @pytest.mark.parametrize(("installed", "is_served"), [("3.8.0", True), ("3.7.0", False), (None, False)])
    def test_serves_only_a_kernel_whose_toolchain_is_installed(
        self, monkeypatch: pytest.MonkeyPatch, installed: str | None, is_served: bool
    ) -> None:
        monkeypatch.setattr(toolchain, "toolchain_version", lambda name: installed)
        case = GEMM_SAMPLE
        call = case.call()
        schema = case.served(GEMM)
        build = replace(schema.kernel.build, requires={"triton": ">=3.8"})
        schema = replace(schema, kernel=replace(schema.kernel, build=build))
        out = dispatcher(schema).call(schema, {"a": call["a"], "b": call["b"]}, fallback=lambda: "fallback")
        assert isinstance(out, torch.Tensor) is is_served

    def test_unmeasured_workload_is_served_at_min_speedup_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(ENV_MIN_SPEEDUP, "0")
        case = GEMM_SAMPLE
        call = case.call()
        schema = case.served(GEMM, evaluations=[])
        out = dispatcher(schema).call(schema, {"a": call["a"], "b": call["b"]}, fallback=lambda: None)
        assert torch.equal(out, call["a"] @ call["b"].mT)

    @pytest.mark.parametrize(
        ("configs", "served"),
        [
            pytest.param({"gemm_bf16_nt.default": DispatchConfig()}, True, id="its_variant"),
            pytest.param({"gemm_bf16_nt.shuffled": DispatchConfig()}, False, id="another_variant"),
        ],
    )
    def test_serves_only_the_variants_with_a_config(self, configs: dict, served: bool) -> None:
        case = GEMM_SAMPLE
        call = case.call()
        schema = case.served(GEMM)
        collection = SchemaCollection(schemas={schema.name: [schema]})
        out = Dispatcher(collection, DispatchConfigRegistry(per_schema=configs)).call(
            schema, {"a": call["a"], "b": call["b"]}, fallback=lambda: None
        )
        assert (out is not None) is served

    def test_a_miss_runs_the_best_kernel_of_its_const_axes_when_configured(self) -> None:
        case = GEMM_SAMPLE
        schema = case.served(GEMM)
        collection = SchemaCollection(schemas={schema.name: [schema]})
        a, b = torch.ones(8, 64, dtype=torch.bfloat16), case.call()["b"]
        out = Dispatcher(collection, DispatchConfig(on_miss_policy="use_best_kernel")).call(
            schema, {"a": a, "b": b}, fallback=lambda: None
        )
        assert torch.equal(out, a @ b.mT)

    def test_a_miss_runs_no_best_kernel_of_another_dtype(self) -> None:
        case = GEMM_SAMPLE
        schema = case.served(GEMM)
        collection = SchemaCollection(schemas={schema.name: [schema]})
        a, b = torch.ones(8, 64, dtype=torch.float16), case.call()["b"]
        out = Dispatcher(collection, DispatchConfig(on_miss_policy="use_best_kernel")).call(
            schema, {"a": a, "b": b}, fallback=lambda: "fallback"
        )
        assert out == "fallback"

    def test_serves_each_dtype_its_own_kernel_at_the_same_axes(self) -> None:
        case = GEMM_SAMPLE
        call = case.call()
        bf16 = case.served("def run(a, b, bias):\n    return 'bf16'\n", dps=False)
        half = replace(
            case.served("def run(a, b, bias):\n    return 'half'\n", dps=False),
            inputs={**bf16.inputs, "a": replace(bf16.inputs["a"], dtype="float16")},
        )
        half = replace(half, kernel=replace(half.kernel, name="half"))
        served = Dispatcher(SchemaCollection(schemas={bf16.name: [bf16, half]}))
        assert served.call(bf16, {"a": call["a"], "b": call["b"]}, fallback=lambda: None) == "bf16"
        assert (
            served.call(bf16, {"a": call["a"].to(torch.float16), "b": call["b"]}, fallback=lambda: None)
            == "half"
        )

    def test_falls_back_on_an_optional_tensor_the_kernel_was_recorded_without(self) -> None:
        case = GEMM_SAMPLE
        call = case.call()
        schema = case.served(GEMM)
        bias = torch.zeros(32, dtype=torch.bfloat16)
        out = dispatcher(schema).call(
            schema, {"a": call["a"], "b": call["b"], "bias": bias}, fallback=lambda: "fallback"
        )
        assert out == "fallback"

    def test_miss_without_fallback_raises(self) -> None:
        case = GEMM_SAMPLE
        with pytest.raises(RuntimeError, match="no fallback"):
            dispatcher().call(case.schema(), {"a": case.call()["a"]})


@pytest.mark.gpu
@pytest.mark.usefixtures("builder_registry")
class TestOpApi:
    def test_active_dispatcher_serves_the_call(self) -> None:
        case = GEMM_SAMPLE
        call = case.call()
        with dispatcher(case.served(GEMM)):
            out = aklr.ops.gemm_bf16_nt(call["a"], call["b"])
        assert torch.equal(out, call["a"] @ call["b"].mT)

    def test_call_after_the_dispatcher_stops_raises(self) -> None:
        case = GEMM_SAMPLE
        call = case.call()
        with dispatcher(case.served(GEMM)):
            pass
        with pytest.raises(RuntimeError, match="no Dispatcher is active"):
            aklr.ops.gemm_bf16_nt(call["a"], call["b"])

    @pytest.mark.parametrize("shuffled", [False, True])
    def test_routes_by_the_variant_the_call_matches(self, shuffled: bool) -> None:
        case = OpSample(aklr.ops.gemm_bf16_nt, "gemm_bf16_nt_shuffled", {"M": 4, "N": 64, "K": 64})
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

    def test_call_no_variant_matches_runs_the_fallback(self) -> None:
        case = OpSample(
            aklr.ops.gemm_fp8_nt_blockscaled, "gemm_fp8_nt_blockscaled", {"M": 4, "N": 256, "K": 256}
        )
        with dispatcher():
            assert (
                aklr.ops.gemm_fp8_nt_blockscaled.dispatch(
                    case.call() | {"scale_major_mode": "other"}, fallback=lambda: "fallback"
                )
                == "fallback"
            )

    @pytest.mark.parametrize(
        ("change", "match"),
        [
            pytest.param({"op_type": "another_op"}, "op types", id="another_op"),
            pytest.param({}, "named alike", id="same_variant"),
            pytest.param({"variant": {"late": "not_an_argument"}}, "not arguments", id="unknown_argument"),
        ],
    )
    def test_api_refuses_variants(self, change: dict, match: str) -> None:
        template = GEMM_SAMPLE.template
        with pytest.raises(ValueError, match=match):

            @aklr_api((template, replace(template, **change)))
            def gemm_bf16_nt_refused(a, b, bias=None, out=None, use_shuffled_weight=False): ...


@pytest.mark.parametrize(
    "runtime", ["aklr.core.runtime.dispatch.Dispatcher", "aklr.core.runtime.tracing.TracingRuntime"]
)
def test_a_runtime_the_env_cannot_start_raises_its_error_on_every_call(runtime: str) -> None:
    schemas = RuntimeEnv().schema_path / "schemas"
    schemas.mkdir()
    (schemas / "unparsable.json").write_text("{")
    module, _, name = runtime.rpartition(".")
    script = (
        f"from {module} import {name}\n"
        "errors = []\n"
        "for _ in range(2):\n"
        "    try:\n"
        f"        print({name}.get_instance())\n"
        "    except Exception as e:\n"
        "        errors.append(e)\n"
        "        print('raised', type(e).__name__, 'from', type(e.__cause__).__name__)\n"
        # one error raised again would keep every caller's frames, and their tensors, in its traceback
        "print('distinct', errors[0] is not errors[1])\n"
    )
    # a process of its own: the env starts a runtime once per process
    env = {**os.environ, ENV_DISABLE: "0", ENV_ENABLE_TRACING: "1"}
    done = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True, check=False
    )
    assert done.stdout.splitlines() == [
        "raised RuntimeError from ValidationError",
        "raised RuntimeError from ValidationError",
        "distinct True",
    ], done.stderr[-2000:]


def test_a_thread_calling_while_the_env_starts_the_dispatcher_waits_for_it() -> None:
    script = (
        "import threading\nimport time\n"
        "from aklr.core.runtime.dispatch import Dispatcher\n"
        "from aklr.core.schema import SchemaCollection\n"
        "load = SchemaCollection.from_path\n"
        "def slow(*args, **kwargs):\n"
        "    time.sleep(0.5)\n"
        "    return load(*args, **kwargs)\n"
        "SchemaCollection.from_path = slow\n"
        "got = []\n"
        "threads = [threading.Thread(target=lambda: got.append(Dispatcher.get_instance())) for _ in range(2)]\n"
        "for thread in threads:\n"
        "    thread.start()\n"
        "for thread in threads:\n"
        "    thread.join()\n"
        "print([type(d).__name__ for d in got], got[0] is got[1])\n"
    )
    env = {**os.environ, ENV_DISABLE: "0"}
    done = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True, check=False
    )
    assert done.stdout.splitlines() == ["['Dispatcher', 'Dispatcher'] True"], done.stderr[-2000:]


class TestOpVariants:
    def test_a_second_api_of_one_op_type_is_refused(self) -> None:
        assert "gemm_bf16_nt" in OpVariants.ops()
        with pytest.raises(ValueError, match="gemm_bf16_nt"):

            @aklr_api(GEMM_SAMPLE.template)
            def gemm_bf16_nt_again(a, b, bias=None, out=None, use_shuffled_weight=False): ...

    def test_an_api_declared_after_the_ops_are_read_is_among_them(self) -> None:
        OpVariants.ops()
        template = replace(GEMM_SAMPLE.template, op_type="gemm_bf16_nt_late")

        @aklr_api(template)
        def gemm_bf16_nt_late(a, b, bias=None, out=None, use_shuffled_weight=False): ...

        assert OpVariants.ops()["gemm_bf16_nt_late"] is gemm_bf16_nt_late

    def test_an_api_takes_its_variants_as_one_tuple(self) -> None:
        template = replace(GEMM_SAMPLE.template, op_type="gemm_bf16_nt_tupled")
        variants = (template, replace(template, variant={"shuffled": "use_shuffled_weight"}))

        @aklr_api(variants)
        def gemm_bf16_nt_tupled(a, b, bias=None, out=None, use_shuffled_weight=False): ...

        assert gemm_bf16_nt_tupled.variants == variants

    def test_a_json_no_variant_names_is_loaded_without_a_standard(self, tmp_path: Path) -> None:
        save_json_file(replace(GEMM_SAMPLE.schema(), name="gemm_bf16_nt_unnamed"), tmp_path / "s.json")
        schema = OpVariants.load((tmp_path / "s.json").read_bytes())
        assert (OpVariants.ops()[schema.op_type].variant_of(schema), schema.standard) == (None, None)
