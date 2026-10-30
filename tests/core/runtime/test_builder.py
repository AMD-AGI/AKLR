# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import sys

import pytest
import torch

from aklr.core.runtime import peer_memory
from aklr.core.runtime.builder import (
    KERNEL_PACKAGE_PREFIX,
    BuilderRegistry,
    BuildError,
    Runnable,
)
from aklr.core.runtime.builder import runnable as runnable_module
from tests.op_case import GEMM_KERNEL_SOURCE, HIP_GEMM_KERNEL_SOURCE, OpCase

_IMPORT_ERROR_KERNEL_SOURCE = "raise ImportError('no kernel')\n"
_UNBOUND = r"kernel\.py::run does not take the call keys"
# out = a @ b.T times the number of calls so far, counted in its state
_COUNTING_KERNEL_SOURCE = (
    "def run(*, a, b, out, state, **_):\n    state[0] += 1\n    out.copy_(a @ b.mT * state[0])\n"
)
# the hip gemm, counting its calls the same way
_HIP_COUNTING_KERNEL_SOURCE = (
    HIP_GEMM_KERNEL_SOURCE.replace(
        "                     int k) {\n    float acc = 0.0f;",
        "                     int k, const int *calls) {\n    float acc = 0.0f;",
    )
    .replace(
        "__float2bfloat16(acc);\n}\n",
        "__float2bfloat16(acc * *calls);\n}\n\n__global__ void count(int *calls) { ++*calls; }\n",
    )
    .replace(
        "at::ScalarType out_dtype, at::Tensor &out) {\n",
        "at::ScalarType out_dtype, at::Tensor &out, at::Tensor &state) {\n"
        "    int *calls = static_cast<int *>(state.data_ptr());\n"
        "    count<<<1, 1, 0, at::hip::getCurrentHIPStream()>>>(calls);\n",
    )
    .replace(
        "static_cast<int>(a.size(1)));",
        "static_cast<int>(a.size(1)), calls);",
    )
    .replace("Tensor(a!) out) -> ()", "Tensor(a!) out, Tensor(b!) state) -> ()")
)
# out = a @ b.T times the number of ranks plus this rank's index, from its peer memory
_PEER_KERNEL_SOURCE = (
    "def run(*, a, b, out, peers, rank, **_):\n    out.copy_(a @ b.mT * (peers.numel() + rank))\n"
)
# the hip gemm, scaled the same way
_HIP_PEER_KERNEL_SOURCE = (
    HIP_GEMM_KERNEL_SOURCE.replace(
        "                     int k) {\n    float acc = 0.0f;",
        "                     int k, float factor) {\n    float acc = 0.0f;",
    )
    .replace("__float2bfloat16(acc);\n}\n", "__float2bfloat16(acc * factor);\n}\n")
    .replace(
        "at::ScalarType out_dtype, at::Tensor &out) {\n",
        "at::ScalarType out_dtype, at::Tensor &out, const at::Tensor &peers,\n         int64_t rank) {\n",
    )
    .replace(
        "static_cast<int>(a.size(1)));",
        "static_cast<int>(a.size(1)),\n        static_cast<float>(peers.numel() + rank));",
    )
    .replace("Tensor(a!) out) -> ()", "Tensor(a!) out, Tensor peers, int rank) -> ()")
)


class TestBuilderRegistry:
    """``build`` imports a schema's kernel as a ``Runnable``, once per content hash."""

    def test_imports_entry_point(self, builder_registry: BuilderRegistry) -> None:
        """The gemm case's schema builds to a ``Runnable`` writing ``a @ b.T`` into ``out``."""
        case = OpCase.of("gemm@1.0")
        runnable = builder_registry.build(case.schema(GEMM_KERNEL_SOURCE))
        assert isinstance(runnable, Runnable)
        call = case.call(case.inputs())
        call["out"] = torch.empty(case.contract.storage_shape_of("out", case.point), dtype=call["out_dtype"])
        runnable(**call)
        torch.testing.assert_close(call["out"], call["a"] @ call["b"].mT, rtol=0, atol=0)
        package = runnable.metadata.misc["module_name"].partition(".")[0]
        assert package in sys.modules
        builder_registry.cleanup()
        assert package not in sys.modules

    @pytest.mark.parametrize(
        ("source", "match"),
        [
            (_IMPORT_ERROR_KERNEL_SOURCE, r"cannot import kernel\.py::run"),
            ("def run(a, b):\n    return a @ b.mT\n", _UNBOUND),
            ("def run(*, a, b, out, scale, **_): ...\n", _UNBOUND),
        ],
        ids=["import_error", "without_out", "unknown_required"],
    )
    def test_build_error_leaves_no_module(
        self, builder_registry: BuilderRegistry, source: str, match: str
    ) -> None:
        """A kernel that cannot be imported or bound gives a ``BuildError`` naming its entry point, and leaves no package behind."""
        schema = OpCase.of("gemm@1.0").schema(source)
        with pytest.raises(BuildError, match=match):
            builder_registry.build(schema)
        package = f"{KERNEL_PACKAGE_PREFIX}flydsl_{schema.kernel.content_hash[:16]}"
        assert not [m for m in sys.modules if f"{m}.".startswith(f"{package}.")]

    def test_caches_by_content_hash(self, builder_registry: BuilderRegistry) -> None:
        """Two schemas with the same kernel build one ``Runnable``."""
        gemm_schema = OpCase.of("gemm@1.0").schema()
        bmm_schema = OpCase.of("bmm@1.0").schema()
        assert gemm_schema.name != bmm_schema.name
        assert builder_registry.build(gemm_schema) is builder_registry.build(bmm_schema)

    def test_keeps_state_between_calls(self, builder_registry: BuilderRegistry) -> None:
        """A kernel with ``state_bytes`` gets ``state``, zeroed before its first call and
        kept for the next; the caller's call is left without it."""
        case = OpCase.of("gemm@1.0")
        runnable = builder_registry.build(case.schema(_COUNTING_KERNEL_SOURCE, state_bytes=16))
        call = case.call(case.inputs())
        product = call["a"] @ call["b"].mT
        for calls in (1, 2):
            call["out"] = torch.empty(
                case.contract.storage_shape_of("out", case.point),
                dtype=call["out_dtype"],
            )
            runnable(**call)
            torch.testing.assert_close(call["out"], product * calls, rtol=0, atol=0)
        assert "state" not in call

    def test_state_is_a_call_key(self, builder_registry: BuilderRegistry) -> None:
        """With ``state_bytes`` the entry takes ``state`` too."""
        source = "def run(*, a, b, trans_b, bias, out_dtype, out): ...\n"
        case = OpCase.of("gemm@1.0")
        builder_registry.build(case.schema(source))
        with pytest.raises(BuildError, match=_UNBOUND):
            builder_registry.build(case.schema(source, state_bytes=16))

    def test_peers_are_call_keys(self, builder_registry: BuilderRegistry) -> None:
        """With ``peer_bytes`` the entry takes ``peers`` and ``rank`` too."""
        source = "def run(*, a, b, trans_b, bias, out_dtype, out, state): ...\n"
        case = OpCase.of("gemm@1.0")
        builder_registry.build(case.schema(source, state_bytes=16))
        with pytest.raises(BuildError, match=_UNBOUND):
            builder_registry.build(case.schema(source, state_bytes=16, peer_bytes=64))

    def test_peer_memory_of_one_rank(self, builder_registry: BuilderRegistry) -> None:
        """Without a group the kernel's peer memory is its own zeroed buffer: one address,
        rank 0; the caller's call is left without them."""
        case = OpCase.of("gemm@1.0")
        runnable = builder_registry.build(case.schema(_PEER_KERNEL_SOURCE, peer_bytes=64))
        call = case.call(case.inputs())
        call["out"] = torch.empty(case.contract.storage_shape_of("out", case.point), dtype=call["out_dtype"])
        runnable(**call)
        torch.testing.assert_close(call["out"], call["a"] @ call["b"].mT, rtol=0, atol=0)
        assert "peers" not in call and "rank" not in call

    def test_peer_memory_needs_a_group_of_several_ranks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without a group, a process of several ranks opens no peer memory."""
        monkeypatch.setattr(peer_memory, "_group", None)
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
        with pytest.raises(RuntimeError, match="needs set_group"):
            peer_memory.open_peer_memory(64, torch.device("cpu"))

    def test_peer_memory_opens_once_per_device(
        self, builder_registry: BuilderRegistry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The kernel gets its peer memory's addresses and rank, opened once per device
        for every later call."""
        opened = []

        def open_peer_memory(nbytes: int, device: torch.device) -> peer_memory.PeerMemory:
            opened.append((nbytes, device))
            return peer_memory.PeerMemory(torch.tensor([7, 8, 9]), 2)

        monkeypatch.setattr(runnable_module, "open_peer_memory", open_peer_memory)
        case = OpCase.of("gemm@1.0")
        runnable = builder_registry.build(case.schema(_PEER_KERNEL_SOURCE, peer_bytes=64))
        call = case.call(case.inputs())
        call["out"] = torch.empty(case.contract.storage_shape_of("out", case.point), dtype=call["out_dtype"])
        for _ in range(2):
            runnable(**call)
            torch.testing.assert_close(call["out"], call["a"] @ call["b"].mT * 5, rtol=0, atol=0)
        assert opened == [(64, torch.device("cpu"))]


@pytest.mark.gpu
class TestHipBuilder:
    """``build`` of a hip schema loads its library and returns its entry op."""

    def test_builds_entry_op(self, builder_registry: BuilderRegistry) -> None:
        """The hip gemm writes ``a @ b.T`` into ``out``."""
        case = OpCase.of("gemm@1.0")
        runnable = builder_registry.build(case.schema(HIP_GEMM_KERNEL_SOURCE, language="hip"))
        assert runnable.metadata.build_type == "hip"
        call = case.call(case.inputs(), device="cuda")
        call["out"] = torch.empty(
            case.contract.storage_shape_of("out", case.point),
            dtype=call["out_dtype"],
            device="cuda",
        )
        runnable(**call)
        torch.testing.assert_close(
            call["out"].float(),
            call["a"].float() @ call["b"].float().mT,
            rtol=1e-2,
            atol=1e-2,
        )

    @pytest.mark.parametrize(
        ("old", "new", "match"),
        [
            ("float acc = 0.0f;", "float acc = ;", "Error building extension"),
            ('m.def("run(', 'm.def("other(', "registers no op run"),
            ("Tensor(a!) out)", "Tensor(a!) output)", "not the call keys"),
            ("Tensor(a!) out)", "Tensor out)", r"must write exactly \['out'\]"),
        ],
        ids=["compile_error", "no_op", "other_names", "output_not_written"],
    )
    def test_build_error(self, builder_registry: BuilderRegistry, old: str, new: str, match: str) -> None:
        """A kernel that does not compile, or whose entry op is not the call's, gives a ``BuildError``."""
        source = HIP_GEMM_KERNEL_SOURCE.replace(old, new)
        assert source != HIP_GEMM_KERNEL_SOURCE
        with pytest.raises(BuildError, match=match):
            builder_registry.build(OpCase.of("gemm@1.0").schema(source, language="hip"))

    def test_keeps_state_between_calls(self, builder_registry: BuilderRegistry) -> None:
        """The op writes ``state`` as it writes the outputs; the state it gets is zeroed
        once on the device and kept."""
        case = OpCase.of("gemm@1.0")
        runnable = builder_registry.build(
            case.schema(_HIP_COUNTING_KERNEL_SOURCE, language="hip", state_bytes=16)
        )
        call = case.call(case.inputs(), device="cuda")
        product = call["a"].float() @ call["b"].float().mT
        for calls in (1, 2):
            call["out"] = torch.empty(
                case.contract.storage_shape_of("out", case.point),
                dtype=call["out_dtype"],
                device="cuda",
            )
            runnable(**call)
            torch.testing.assert_close(call["out"].float(), product * calls, rtol=1e-2, atol=1e-2)

    def test_state_is_written(self, builder_registry: BuilderRegistry) -> None:
        """With ``state_bytes`` the op takes ``state`` as a ``Tensor(a!)``."""
        source = _HIP_COUNTING_KERNEL_SOURCE.replace("Tensor(b!) state", "Tensor state")
        schema = OpCase.of("gemm@1.0").schema(source, language="hip", state_bytes=16)
        with pytest.raises(BuildError, match=r"must write exactly \['out', 'state'\]"):
            builder_registry.build(schema)

    def test_takes_its_peer_memory(self, builder_registry: BuilderRegistry) -> None:
        """With ``peer_bytes`` the op takes ``peers`` (on the host) and ``rank``: one rank
        without a group."""
        case = OpCase.of("gemm@1.0")
        runnable = builder_registry.build(case.schema(_HIP_PEER_KERNEL_SOURCE, language="hip", peer_bytes=64))
        call = case.call(case.inputs(), device="cuda")
        call["out"] = torch.empty(
            case.contract.storage_shape_of("out", case.point),
            dtype=call["out_dtype"],
            device="cuda",
        )
        runnable(**call)
        torch.testing.assert_close(
            call["out"].float(),
            call["a"].float() @ call["b"].float().mT,
            rtol=1e-2,
            atol=1e-2,
        )

    def test_library_that_does_not_load_is_the_kernels_fault(self, builder_registry: BuilderRegistry) -> None:
        """An undefined symbol links but fails the load: a ``BuildError`` not caused by an ``OSError``, so bench does not blame the platform."""
        source = HIP_GEMM_KERNEL_SOURCE.replace(
            "namespace {\n", "void aklr_undefined();\n\nnamespace {\n", 1
        ).replace("    gemm<<<", "    aklr_undefined();\n    gemm<<<")
        with pytest.raises(BuildError, match="undefined symbol") as error:
            builder_registry.build(OpCase.of("gemm@1.0").schema(source, language="hip"))
        assert not isinstance(error.value.__cause__, OSError)
