# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import logging
import sys
import types
from pathlib import Path

import pytest
import torch

import aklr.ops.gemm
from aklr.core.contract.export import read_calls
from aklr.core.contract.op import OpContract
from aklr.core.contract.registry import ContractRegistry
from aklr.core.contract.spec import WorkloadSpec
from aklr.core.platform import toolchain
from aklr.core.platform.device import DeviceTarget
from aklr.core.platform.env import (
    ENV_DISABLE,
    ENV_EXPORT,
    ENV_EXPORT_DIR,
    ENV_MIN_SPEEDUP,
)
from aklr.core.runtime import LoadError
from aklr.core.runtime.builder import BuilderRegistry
from aklr.core.runtime.dispatch import (
    CallRecorder,
    DispatcherRegistry,
    DispatchTable,
    FallbackReason,
    KernelDispatcher,
    register_vllm_plugin,
)
from aklr.core.runtime.hook_registry import HookRegistry
from tests.op_case import GEMM_KERNEL_SOURCE, KERNEL_TARGET, OpCase


class TestKernelDispatcher:
    """One contract's dispatch: serving, fallback reasons and call recording."""

    def _dispatcher(
        self,
        builder_registry: BuilderRegistry,
        schemas: tuple[OpContract, ...],
        *,
        min_speedup: float = 0.0,
    ) -> KernelDispatcher:
        contract_id = schemas[0].contract_id if schemas else "gemm@1.0"
        contract = ContractRegistry.default().get(contract_id)
        dispatcher = KernelDispatcher(contract)
        if schemas:
            table = DispatchTable.from_schemas(
                contract,
                torch.device("cpu"),
                [(s, builder_registry.build(s), s.workloads) for s in schemas],
                min_speedup=min_speedup,
            )
            table.warmed_keys.update(table.runnables)
            dispatcher.set_table(table)
        return dispatcher

    @pytest.mark.parametrize("served", ["allocated", "caller", "strided_a_scale"])
    def test_api_fn_runs_served_kernel(self, builder_registry: BuilderRegistry, served: str) -> None:
        """The kernel fills the caller's ``out`` as is, else one dispatch allocates;
        sglang's column-major a_scale view reaches it as the stored tensor."""
        if served == "strided_a_scale":
            case = next(
                c
                for c in OpCase.all()
                if c.contract_id == "gemm_a8w8_blockwise_scaled@1.0"
                and c.point.layout_variant_name == "b_shuffled"
            )
            # view(-1) raises on a strided view, as the committed kernels' launch does
            source = (
                "def run(*, a_scale, out, **_):\n"
                "    out.view(-1)[: a_scale.numel()].copy_(a_scale.view(-1))\n"
            )
            dispatcher = self._dispatcher(builder_registry, (case.schema(source),))
            call = case.call(case.inputs())
            stored = call["a_scale"]
            call["a_scale"] = stored.view(*stored.shape[::-1]).mT
            result = dispatcher.dispatch(call)
            assert isinstance(result, torch.Tensor)
            expected = stored.view(-1).to(result.dtype)
            torch.testing.assert_close(result.view(-1)[: stored.numel()], expected)
            return
        caller_out = served == "caller"
        case = OpCase.of("gemm@1.0")
        dispatcher = self._dispatcher(builder_registry, (case.schema(GEMM_KERNEL_SOURCE),))
        api_fn = dispatcher.make_api_fn(aklr.ops.gemm.gemm.__wrapped__)
        call = case.call(case.inputs())
        if caller_out:
            call["out"] = torch.empty(
                case.contract.storage_shape_of("out", case.point),
                dtype=call["out_dtype"],
            )
        result = api_fn(**call)
        assert isinstance(result, torch.Tensor)
        if caller_out:
            assert result is call["out"]
        torch.testing.assert_close(result, call["a"] @ call["b"].mT, rtol=0, atol=0)

    @pytest.mark.parametrize(
        ("miss", "reason"),
        [
            ("no_table", FallbackReason.NO_SCHEMA),
            ("other_point", FallbackReason.NO_WORKLOAD),
            ("unmeasured", FallbackReason.NOT_MEASURED),
            ("slow", FallbackReason.NOT_QUALIFIED),
            ("undescribed_call", FallbackReason.OUT_OF_CONTRACT),
            ("misaligned", FallbackReason.OUT_OF_CONTRACT),
            ("non_contiguous", FallbackReason.OUT_OF_CONTRACT),
            ("aliased_out", FallbackReason.OUT_OF_CONTRACT),
            ("not_warmed", FallbackReason.NOT_WARMED),
        ],
    )
    def test_missed_call_returns_fallback_reason(
        self,
        builder_registry: BuilderRegistry,
        monkeypatch: pytest.MonkeyPatch,
        miss: str,
        reason: FallbackReason,
    ) -> None:
        case = OpCase.of("gemm@1.0")
        schema = case.schema(GEMM_KERNEL_SOURCE)
        workload = schema.workloads[0]
        with_workload = lambda **update: schema.model_copy(
            update={"workloads": (workload.model_copy(update=update),)}
        )
        schemas, min_speedup = {
            "no_table": ((), 0.0),
            "other_point": ((with_workload(axes={**workload.axes, "m": 8}),), 0.0),
            "unmeasured": ((schema,), 1.0),
            "slow": ((with_workload(accuracy="passed", speedup=0.5),), 1.0),
            "undescribed_call": ((schema,), 0.0),
            "misaligned": ((schema,), 0.0),
            "non_contiguous": ((schema,), 0.0),
            "aliased_out": ((schema,), 0.0),
            "not_warmed": ((schema,), 0.0),
        }[miss]
        dispatcher = self._dispatcher(builder_registry, schemas, min_speedup=min_speedup)
        call = case.call(case.inputs())
        call["out"] = torch.full(
            case.contract.storage_shape_of("out", case.point),
            float("nan"),
            dtype=call["out_dtype"],
        )
        if miss == "undescribed_call":
            call["a"] = call["a"].float()
        elif miss == "misaligned":
            a = call["a"]
            call["a"] = torch.empty(a.numel() + 1, dtype=a.dtype, device=a.device)[1:].view_as(a).copy_(a)
        elif miss == "non_contiguous":
            call["a"] = call["a"].mT.contiguous().mT
        elif miss == "aliased_out":
            call["out"] = call["a"].view(-1)[: call["out"].numel()].view_as(call["out"]).fill_(float("nan"))
        elif miss == "not_warmed":
            dispatcher.table.warmed_keys.clear()
            monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
        # asked first, the same reason
        assert dispatcher.fallback_reason(call) is reason
        assert dispatcher.dispatch(call) is reason
        assert call["out"].isnan().all()
        if miss == "not_warmed":  # capture never marks the key warm
            assert not dispatcher.table.warmed_keys

    def _state_dispatcher(
        self,
        builder_registry: BuilderRegistry,
        workloads: tuple[WorkloadSpec, ...] = (),
        min_speedup: float = 0.0,
    ) -> tuple[OpCase, KernelDispatcher]:
        """gdn_decode served at its case's workload, or at ``workloads``, by a kernel that
        zeroes the output and the two state pools, its mutable inputs."""
        case = OpCase.of("gdn_decode@1.0")
        schema = case.schema(
            "def run(*, normalized, conv_state, delta_state, **_):\n"
            "    normalized.zero_()\n"
            "    conv_state.zero_()\n"
            "    delta_state.zero_()\n"
        )
        if workloads:
            schema = schema.model_copy(update={"workloads": workloads})
        dispatcher = self._dispatcher(
            builder_registry,
            (schema,),
            min_speedup=min_speedup,
        )
        return case, dispatcher

    def test_any_size_axis_matches_every_value(self, builder_registry: BuilderRegistry) -> None:
        """A schema serving one value of an any_size axis serves every value of it: state
        pools of twice the workload's slots are served, and written in place."""
        case, dispatcher = self._state_dispatcher(builder_registry)
        call = case.call(case.inputs())
        for name in ("conv_state", "delta_state"):
            call[name] = torch.cat([call[name], call[name]])
        output = dispatcher.dispatch(call)
        assert not output.any() and not call["conv_state"].any() and not call["delta_state"].any()

    @pytest.mark.parametrize("slots", [0, 2], ids=["below_its_lo", "constrained"])
    def test_any_size_axis_keeps_its_domain(self, builder_registry: BuilderRegistry, slots: int) -> None:
        """An any_size axis matches only values its bounds and the constraints reading it
        admit: state pools of no slots (below n_slots' lowest) or of 2 (4 tokens <=
        n_slots + 1 fails) fall back."""
        case, dispatcher = self._state_dispatcher(builder_registry)
        call = case.call(case.inputs())
        for name in ("conv_state", "delta_state"):
            call[name] = call[name][:slots]
        assert dispatcher.dispatch(call) is FallbackReason.OUT_OF_CONTRACT

    def test_any_size_axis_needs_each_workload_qualified(self, builder_registry: BuilderRegistry) -> None:
        """Workloads differing in an any_size axis alone share a key, served only if each
        of them qualifies: one too slow leaves the case's call unserved."""
        case = OpCase.of("gdn_decode@1.0")
        workload = case.contract.workload_for(case.point, {})
        doubled = {**workload.axes, "n_slots": 2 * workload.axes["n_slots"]}
        _, dispatcher = self._state_dispatcher(
            builder_registry,
            (
                workload.model_copy(update={"accuracy": "passed", "speedup": 2.0}),
                workload.model_copy(update={"axes": doubled, "accuracy": "passed", "speedup": 0.5}),
            ),
            min_speedup=1.0,
        )
        call = case.call(case.inputs())
        assert dispatcher.dispatch(call) is FallbackReason.NOT_QUALIFIED

    def test_mutable_input_sharing_bytes_declines(self, builder_registry: BuilderRegistry) -> None:
        """A mutable input sharing a byte with another input falls back, unwritten."""
        case, dispatcher = self._state_dispatcher(builder_registry)
        call = case.call(case.inputs())
        conv_bytes = call["conv_state"].view(torch.uint8).view(-1)
        n = call["cache_indices"].numel()
        conv_bytes[: 4 * n].view(torch.int32).copy_(call["cache_indices"])
        call["cache_indices"] = conv_bytes[: 4 * n].view(torch.int32)
        before = conv_bytes.clone()
        assert dispatcher.dispatch(call) is FallbackReason.OUT_OF_CONTRACT
        assert torch.equal(conv_bytes, before)

    def test_failing_kernel_raises(
        self, builder_registry: BuilderRegistry, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A kernel's error is not caught; the log names the kernel, its toolchain here
        and the switch that turns its op off."""
        case = OpCase.of("gemm@1.0")
        schema = case.schema("def run(**_):\n    raise RuntimeError('broken')\n")
        dispatcher = self._dispatcher(builder_registry, (schema,))
        with pytest.raises(RuntimeError, match="broken"):
            dispatcher.dispatch(case.call(case.inputs()))
        assert f"the kernel serving {schema.name} at m=" in caplog.text
        assert f"{ENV_DISABLE}={case.contract_id} turns this op off" in caplog.text

    def test_best_ranked_kernel_serves(self, builder_registry: BuilderRegistry) -> None:
        case = OpCase.of("gemm@1.0")
        workload = case.schema().workloads[0]
        measured = lambda source, speedup: case.schema(source).model_copy(
            update={"workloads": (workload.model_copy(update={"accuracy": "passed", "speedup": speedup}),)}
        )
        faster = measured(GEMM_KERNEL_SOURCE, 2.0)
        slower = measured("def run(*, out, **_):\n    out.zero_()\n", 1.5)
        for schemas in [(slower, faster), (faster, slower)]:
            dispatcher = self._dispatcher(builder_registry, schemas)
            call = case.call(case.inputs())
            result = dispatcher.dispatch(call)
            torch.testing.assert_close(result, call["a"] @ call["b"].mT, rtol=0, atol=0)

    def test_fallback_reason_asks_without_serving(
        self, builder_registry: BuilderRegistry, tmp_path: Path
    ) -> None:
        """A call a kernel serves: ``None``, nothing run, allocated or counted; a missed one:
        dispatch's reason, counted as dispatch counts it. The api function's
        ``fallback_reason`` takes the api's arguments."""
        case = OpCase.of("gemm@1.0")
        dispatcher = self._dispatcher(builder_registry, (case.schema(GEMM_KERNEL_SOURCE),))
        recorder = CallRecorder(tmp_path, DeviceTarget.from_names(**KERNEL_TARGET))
        dispatcher.set_table(dispatcher.table, recorder)
        api_fn = dispatcher.make_api_fn(aklr.ops.gemm.gemm.__wrapped__)
        call = case.call(case.inputs())
        call["out"] = torch.full(
            case.contract.storage_shape_of("out", case.point),
            float("nan"),
            dtype=call["out_dtype"],
        )
        assert api_fn.fallback_reason(**call) is None
        assert call["out"].isnan().all() and not recorder.calls
        out_of_contract = {**call, "a": call["a"].float()}
        assert api_fn.fallback_reason(**out_of_contract) is FallbackReason.OUT_OF_CONTRACT
        assert {k.reason: n for k, n in recorder.calls.items()} == {FallbackReason.OUT_OF_CONTRACT: 1}

    def test_recorder_counts_calls(self, builder_registry: BuilderRegistry, tmp_path: Path) -> None:
        case = OpCase.of("gemm@1.0")
        dispatcher = self._dispatcher(builder_registry, (case.schema(GEMM_KERNEL_SOURCE),))
        recorder = CallRecorder(tmp_path, DeviceTarget.from_names(**KERNEL_TARGET))
        dispatcher.set_table(dispatcher.table, recorder)
        served = case.call(case.inputs())
        out_of_contract = {**served, "a": served["a"].float()}
        assert isinstance(dispatcher.dispatch(served), torch.Tensor)
        # the first key is saved at once
        assert [r.n_calls for r in read_calls(tmp_path)] == [1]
        assert isinstance(dispatcher.dispatch(served), torch.Tensor)
        assert dispatcher.dispatch(out_of_contract) is FallbackReason.OUT_OF_CONTRACT
        assert {k.reason: n for k, n in recorder.calls.items()} == {
            None: 2,
            FallbackReason.OUT_OF_CONTRACT: 1,
        }
        recorder.save()
        records = {r.fallback_reason: r for r in read_calls(tmp_path)}
        assert set(records) == {None, FallbackReason.OUT_OF_CONTRACT.value}
        assert (records[None].point, records[None].n_calls) == (case.point, 2)
        out_of_contract_record = records[FallbackReason.OUT_OF_CONTRACT.value]
        assert (out_of_contract_record.point, out_of_contract_record.n_calls) == (
            None,
            1,
        )
        assert {r.target for r in records.values()} == {recorder.target}


class TestDispatcherRegistry:
    """``DispatcherRegistry.init`` over the committed schemas."""

    def test_disabled_init_closes_every_dispatcher(
        self,
        default_dispatchers: DispatcherRegistry,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv(ENV_DISABLE, "1")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        default_dispatchers.init()
        dispatchers = default_dispatchers.dispatchers()
        assert set(dispatchers) == {c.contract_id for c in ContractRegistry.default().contracts()}
        assert {d.dispatch({}) for d in dispatchers.values()} == {FallbackReason.DISABLED}
        case = OpCase.of("gemm@1.0")
        assert aklr.ops.gemm.gemm(**case.call(case.inputs())) is FallbackReason.DISABLED
        # an init failing partway leaves every dispatcher closed
        (tmp_path / "file").touch()
        monkeypatch.setenv(ENV_EXPORT, "1")
        monkeypatch.setenv(ENV_EXPORT_DIR, str(tmp_path / "file"))
        with pytest.raises(LoadError):
            default_dispatchers.init()
        assert {d.dispatch({}) for d in default_dispatchers.dispatchers().values()} == {
            FallbackReason.DISABLED
        }

    def test_unimportable_baseline_is_a_load_error(
        self, default_dispatchers: DispatcherRegistry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An op's ``baseline.py`` that fails to import fails the init as a
        ``LoadError``, the error a host catches."""

        def baseline(contract: OpContract) -> None:
            raise ModuleNotFoundError("No module named 'sglang'", name="sglang")

        monkeypatch.setattr(OpContract, "baseline", baseline)
        with pytest.raises(LoadError, match="its baseline"):
            default_dispatchers.init()

    @staticmethod
    def _schema_alone(monkeypatch: pytest.MonkeyPatch, schema: OpContract) -> None:
        """``init`` reads ``schema`` alone and serves it on the CPU, as on its target."""
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
        op_package = ContractRegistry.default().get(schema.contract_id).op_package
        monkeypatch.setattr(
            ContractRegistry,
            "read_schemas",
            lambda self, package, **_: (
                [(Path("a.json"), schema)] if package == op_package else [],
                [],
            ),
        )
        monkeypatch.setattr(OpContract, "baseline", lambda contract: None)
        target = DeviceTarget.from_names(**KERNEL_TARGET)
        monkeypatch.setattr(
            DispatcherRegistry,
            "_probe_device",
            staticmethod(lambda: (torch.device("cpu"), target)),
        )

    @pytest.mark.parametrize(("installed", "is_served"), [("0.3.2", True), ("0.3.1", False)])
    def test_init_skips_schema_whose_requires_fails(
        self,
        default_dispatchers: DispatcherRegistry,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        installed: str,
        is_served: bool,
    ) -> None:
        """A schema whose ``kernel.requires`` the toolchains here miss is not served, and
        one INFO line says why."""
        case = OpCase.of("gemm@1.0")
        schema = case.schema(GEMM_KERNEL_SOURCE)
        kernel = schema.kernel.model_copy(update={"requires": {"flydsl": ">=0.3.2"}})
        self._schema_alone(monkeypatch, schema.model_copy(update={"kernel": kernel}))
        monkeypatch.setattr(toolchain, "toolchain_version", lambda name: installed)
        with caplog.at_level(logging.INFO):
            default_dispatchers.init()
        result = aklr.ops.gemm.gemm(**case.call(case.inputs()))
        assert isinstance(result, torch.Tensor) is is_served
        skipped = "skipped 1 schemas: flydsl 0.3.1 is not >=0.3.2"
        assert (skipped in caplog.text) is not is_served

    def test_unimportable_kernel_fails_the_init(
        self, default_dispatchers: DispatcherRegistry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The error names the switch that turns the kernel's op off."""
        case = OpCase.of("gemm@1.0")
        self._schema_alone(monkeypatch, case.schema("import no_such_module\n"))
        hint = f"{ENV_DISABLE}={case.contract_id} turns this op off"
        with pytest.raises(LoadError, match=hint):
            default_dispatchers.init()

    @pytest.mark.gfx950
    @pytest.mark.parametrize("has_baseline", [True, False])
    def test_init_serves_committed_schema(
        self,
        default_dispatchers: DispatcherRegistry,
        monkeypatch: pytest.MonkeyPatch,
        has_baseline: bool,
    ) -> None:
        """An unmeasured workload is served at min speedup 0; with no defined baseline
        that is the default too (nothing to beat)."""
        if has_baseline:
            monkeypatch.setenv(ENV_MIN_SPEEDUP, "0")
        else:
            monkeypatch.delenv(ENV_MIN_SPEEDUP, raising=False)
            monkeypatch.setattr(OpContract, "baseline", lambda contract: None)
        default_dispatchers.init()
        contract = OpCase.of("gemm@1.0").contract
        schemas, errors = ContractRegistry.default().read_schemas(contract.op_package)
        assert errors == []
        schema, workload = next(
            (s, w)
            for _, s in schemas
            if s.contract_id == contract.contract_id and s.kernel.sources
            for w in s.workloads
            if w.accuracy != "failed"
        )
        case = OpCase(schema.point(workload.axes))
        args = case.inputs()
        result = aklr.ops.gemm.gemm(**case.call(args, "cuda"))
        assert not isinstance(result, FallbackReason)
        comparisons = case.compare(case.contract.outputs_by_name(result), args)
        assert {o: c.is_passed for o, c in comparisons.items()} == dict.fromkeys(case.contract.outputs, True)


class TestVllmPlugin:
    """``register_vllm_plugin``: AKLR starts in a vLLM worker once its device is bound."""

    def test_init_runs_after_init_device(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The wrapped ``Worker.init_device`` runs the original, then ``init_once``, and
        returns the original's result."""
        events: list[str] = []

        class Worker:
            def init_device(self) -> str:
                events.append("init_device")
                return "bound"

        module = types.ModuleType("vllm.v1.worker.gpu_worker")
        module.Worker = Worker
        for name in ("vllm", "vllm.v1", "vllm.v1.worker"):
            package = types.ModuleType(name)
            package.__path__ = []
            monkeypatch.setitem(sys.modules, name, package)
        monkeypatch.setitem(sys.modules, module.__name__, module)
        registry = HookRegistry()
        monkeypatch.setattr(HookRegistry, "default", classmethod(lambda cls: registry))
        monkeypatch.setattr(DispatcherRegistry, "init_once", lambda self: events.append("init_once"))
        try:
            register_vllm_plugin()
            assert Worker().init_device() == "bound"
            assert events == ["init_device", "init_once"]
        finally:
            registry.restore()
