# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import sys
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from aklr.core.contract.op import OpContract
from aklr.core.validation import ValidationScope, VerifyReport, VerifyValidator
from aklr.core.validation.report import PlatformRecord
from aklr.core.validation.validators.verify import Extraction
from tests.op_case import (
    HIP_GEMM_KERNEL_SOURCE,
    HSACO_GEMM_KERNEL_SOURCE,
    OpCase,
    hsaco_gemm_sources,
)

_PLATFORM = PlatformRecord("gfx950", "MI355X", 1, "7", "2", "3", "0.5")
_RULES = (
    "lds_race",
    "barrier_epoch",
    "pending_read",
    "lds_bounds",
    "lds_uninitialized_read",
    "global_bounds",
)
# `in` is 64 f32 at 0x10000..0x10100 and `out` 128 f32 at 0x20000..0x20200
_LAUNCH = {
    "kernel": "exchange",
    "grid": [1, 1, 1],
    "block": [128, 1, 1],
    "dynamic_shared_bytes": 0,
    "arrays": [
        {"name": "in", "kind": "in", "dtype": "f32", "shape": [64]},
        {"name": "out", "kind": "out", "dtype": "f32", "shape": [128]},
    ],
    "params": [{"ptr": "in"}, {"ptr": "out"}],
}

# What `volta solve` 0.1.0 (ptx_verifier 886c9fd) prints: the check table of a run, then its conclusion.
_CHECKS = (
    "barrier epochs (named barriers) (Algorithm 1)",
    "barrier epochs (mbarrier phases) (Algorithm 1)",
    "mbarrier ordering (Algorithm 2)",
    "memory-conflict freedom (Algorithm 3)",
    "proxy ordering (Algorithm 4)",
    "tensor-operation ordering (Algorithms 5 and 6)",
    "resource lifetime (Algorithm 7)",
    "allocation order (PTX ISA tcgen05.alloc)",
    "allocation permit (PTX ISA tcgen05.relinquish_alloc_permit)",
    "mbarrier release order (phase chain)",
)
_WRITE_READ = """\
      MemoryConflict { kind: WriteRead, location: ByteLocation { space: Shared, addr: 256 }, earlier: EventId(67), later: EventId(136) }
        e67 async op 1   line 348: ds_write_b32 v1, v2
        e136 async op 2   line 352: ds_read_b32 v0, v0
      ... (125 more)
"""
_DUPLICATE_LOAD = """\
      MemoryConflict { kind: WriteWrite, location: ByteLocation { space: Shared, addr: 0 }, earlier: EventId(1), later: EventId(65) }
        e1 async op 0   line 90: buffer_load_dword v4, s[4:7], 0 offen lds
        e65 async op 1   line 90: buffer_load_dword v4, s[4:7], 0 offen lds
"""
_BARRIER = """\
      EpochViolation { barrier: SyncObjectId(0), generation: GenerationId(0), next: GenerationId(1), synchronization: EventId(70), operation: EventId(140), predecessor: Some(EventId(139)) }
        e70 generation 0  line 18: s_barrier
        e139 t1           line 19: ds_read_b32 v0, v0
        e140 t1           line 20: ds_write_b32 v1, v2
"""


def _log(failed: dict[str, str] | None = None, abort: str = "", deadlock: bool = False) -> str:
    """A run's output: the checks named in ``failed`` FAIL with those witnesses, the run stopped at ``abort``."""
    failed = failed or {}
    lines = ["kernel.s: AMD gfx950 kernel", "  time: 0.002s"]
    lines.append(f"  no outputs: {abort}" if abort else "  output 'out': 128 element(s) written")
    lines.append("kernel: graph verification")
    for check in _CHECKS:
        witness = next((w for name, w in failed.items() if check.startswith(name)), None)
        lines.append(f"  {check:<55} {'PASS' if witness is None else 'FAIL'}")
        lines += (witness or "").splitlines()
    if deadlock:
        lines.append("  trace: deadlock, 2 blocked thread(s)")
    reasons = ["the trace deadlocked"] * deadlock + [f"{name} failed" for name in failed]
    if reasons:
        conclusion = "REJECTED"
    elif abort:
        conclusion = "UNKNOWN"
        reasons = [f"the trace generator aborted, so the graph is only a prefix: {abort.partition(': ')[2]}"]
    else:
        conclusion = "VERIFIED"
    lines.append(f"GRAPH VERIFICATION: {conclusion} (no equivalence check requested)")
    return "\n".join(lines + [f"  {reason}" for reason in reasons]) + "\n"


_FAKE_VOLTA = """\
#!{python}
import pathlib, re, sys
if sys.argv[1] == "--help":
    print("The Volta PTX analysis engine.")
elif sys.argv[1] == "--version":
    print("volta 0.1.0")
else:
    log = pathlib.Path(sys.argv[2]).with_suffix(".log").read_text()
    print(log, end="")
    conclusion = re.search(r"^GRAPH VERIFICATION: (\\w+)", log, re.MULTILINE)
    sys.exit({{"VERIFIED": 0, "REJECTED": 1, "UNKNOWN": 2}}[conclusion[1]] if conclusion else 1)
"""


@pytest.fixture
def volta(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The volta binary, mocked on PATH: it prints the log written beside the ISA it is given."""
    directory = tmp_path / "bin"
    directory.mkdir()
    binary = directory / "volta"
    binary.write_text(_FAKE_VOLTA.format(python=sys.executable), encoding="utf-8")
    binary.chmod(0o755)
    monkeypatch.setenv("PATH", str(directory))
    return binary


def _schema(source: str | None = "def run(**_):\n    pass\n", n: int = 1) -> OpContract:
    """A gemm schema of a FlyDSL kernel serving the workloads m=1..n; None: a definition, without sources."""
    case = OpCase.of("gemm@1.0")
    workloads = [
        case.contract.workload_for(case.point.model_copy(update={"axes": case.point.axes | {"m": m}}), {})
        for m in range(1, n + 1)
    ]
    kernel = {
        "language": "flydsl",
        "targets": [{"arch": "gfx950", "hardware": "MI355X"}],
        "requires": {"flydsl": ">=0"},
        "entry_point": "kernel.py::run",
        "sources": [] if source is None else [{"path": "kernel.py", "content": source}],
    }
    return case.contract.narrow(case.point, workloads=workloads, kernel=kernel)


class _Extractor:
    """The FlyDSL ISA extraction, mocked: each workload yields one kernel whose ISA is beside the log
    volta prints for it, or yields none, for the reason given."""

    def __init__(self, tmp_path: Path, log: str = "", failure: str = "") -> None:
        self.directory, self.log, self.failure = tmp_path, log, failure
        self.extracted: list[tuple[str, int, int, str | None]] = []

    def __call__(self, root, schema_path, schema, directory, first, count, device) -> list[Extraction]:
        self.extracted.append((schema_path, first, count, device))
        extractions = []
        for workload in schema.workloads[first : first + count]:
            if self.failure:
                extractions.append(
                    Extraction(schema_path=schema_path, axes=workload.axes, failure=self.failure)
                )
                continue
            isa = self.directory / f"isa{len(self.extracted)}_{workload.axes['m']}.s"
            isa.write_text("\t.amdhsa_kernel exchange\n", encoding="utf-8")
            isa.with_suffix(".log").write_text(self.log, encoding="utf-8")
            extractions.append(
                Extraction(
                    schema_path=schema_path,
                    axes=workload.axes,
                    kernel="exchange",
                    isa_path=str(isa),
                    launch=_LAUNCH,
                )
            )
        return extractions


def _verify(tmp_path: Path, extractor: _Extractor, *schemas: OpContract, devices=()) -> VerifyReport:
    paths = []
    for number, schema in enumerate(schemas or (_schema(),)):
        path = tmp_path / f"aklr/plugins/aiter/provider/gemm/impls/flydsl/{schema.name}/{number:0>16}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(schema.schema_text(), encoding="utf-8")
        paths.append(path)
    scope = ValidationScope(tmp_path, "1" * 40, None, tuple(paths), tuple(paths))
    return VerifyValidator(devices, extractor=extractor).run(scope, _PLATFORM)


def _statuses(report: VerifyReport) -> dict[str, str]:
    """The only extracted kernel's status under each rule."""
    (record,) = report.path_records
    return {verdict.rule: verdict.status for verdict in record.verdicts}


def _details(report: VerifyReport) -> dict[str, str]:
    (record,) = report.path_records
    return {verdict.rule: verdict.detail for verdict in record.verdicts}


def test_a_run_volta_verifies_proves_every_rule(tmp_path, volta):
    report = _verify(tmp_path, _Extractor(tmp_path, _log()))
    assert report.status == "passed"
    assert report.findings == ()
    assert _statuses(report) == dict.fromkeys(_RULES, "proved")
    assert report.tools == {"volta": "volta 0.1.0 at an unrecorded commit"}


def test_two_lds_accesses_without_an_order_reject_the_race_rule_alone(tmp_path, volta):
    report = _verify(tmp_path, _Extractor(tmp_path, _log({"memory-conflict freedom": _WRITE_READ})))
    assert report.status == "failed"
    assert _statuses(report) == dict.fromkeys(_RULES, "proved") | {"lds_race": "rejected"}
    (finding,) = report.findings
    assert finding.code == "isa_rejected"
    assert str(finding.workload) == "m=1"
    assert finding.detail == (
        "VR1 lds_race exchange: WriteRead conflict at Shared 0x100"
        " between `line 348: ds_write_b32 v1, v2` and `line 352: ds_read_b32 v0, v0`"
    )


def test_a_load_to_lds_two_waves_repeat_is_unchecked_and_not_a_race(tmp_path, volta):
    report = _verify(tmp_path, _Extractor(tmp_path, _log({"memory-conflict freedom": _DUPLICATE_LOAD})))
    assert report.status == "passed"
    assert _statuses(report)["lds_race"] == "unchecked"
    assert _details(report)["lds_race"].startswith("duplicate-load (ptx_verifier#48): WriteWrite conflict")


def test_an_lds_race_the_run_stopped_at_rejects_even_beside_a_duplicate_load(tmp_path, volta):
    abort = (
        "wave 0 aborted at instruction 7: LDS race at 0x0: wave 0 lanes 0 and 32 both write it"
        " at instruction 7, in no specified order"
    )
    report = _verify(
        tmp_path, _Extractor(tmp_path, _log({"memory-conflict freedom": _DUPLICATE_LOAD}, abort))
    )
    assert report.status == "failed"
    assert _statuses(report)["lds_race"] == "rejected"
    assert _details(report)["lds_race"] == abort.partition("aborted at ")[2]


def test_a_failed_check_volta_lists_no_witness_for_is_rejected(tmp_path, volta):
    failed = {"memory-conflict freedom": "", "barrier epochs (named barriers)": ""}
    report = _verify(tmp_path, _Extractor(tmp_path, _log(failed)))
    assert report.status == "failed"
    statuses = _statuses(report)
    assert statuses["lds_race"] == statuses["barrier_epoch"] == "rejected"
    assert _details(report)["lds_race"] == "volta failed memory-conflict freedom without a witness"


def test_duplicate_loads_volta_lists_only_some_of_are_not_exempt(tmp_path, volta):
    truncated = _DUPLICATE_LOAD + "      ... (2 more)\n"
    report = _verify(tmp_path, _Extractor(tmp_path, _log({"memory-conflict freedom": truncated})))
    assert report.status == "failed"
    assert _statuses(report)["lds_race"] == "rejected"
    assert _details(report)["lds_race"].endswith("; volta omitted 2 more conflicts")


def test_a_barrier_generation_not_separated_from_the_next_rejects_the_barrier_rule(tmp_path, volta):
    failed = {"barrier epochs (named barriers)": _BARRIER}
    report = _verify(tmp_path, _Extractor(tmp_path, _log(failed)))
    assert _statuses(report) == dict.fromkeys(_RULES, "proved") | {"barrier_epoch": "rejected"}
    (finding,) = report.findings
    assert finding.detail == (
        "VR2 barrier_epoch exchange: generation 0 of barrier 0 is not separated from generation 1:"
        " `line 18: s_barrier`, `line 19: ds_read_b32 v0, v0`, `line 20: ds_write_b32 v1, v2`"
    )


def test_a_deadlocked_trace_rejects_the_barrier_rule(tmp_path, volta):
    report = _verify(tmp_path, _Extractor(tmp_path, _log(deadlock=True)))
    assert _statuses(report)["barrier_epoch"] == "rejected"
    assert _details(report)["barrier_epoch"] == "the trace deadlocked, 2 blocked thread(s)"


@pytest.mark.parametrize(
    ("abort", "rule", "number"),
    [
        pytest.param(
            "wave 0 aborted at instruction 7: LDS race at 0x0: wave 0 lanes 0 and 32 both write it at"
            " instruction 7, in no specified order",
            "lds_race",
            1,
            id="two_lanes_write_one_lds_word",
        ),
        pytest.param(
            "wave 0 aborted at instruction 7: vector register v2 in lane 0 has a pending load",
            "pending_read",
            3,
            id="vector_register_read_before_its_wait",
        ),
        pytest.param(
            "wave 0 aborted at instruction 5: scalar register s0 has a pending load",
            "pending_read",
            3,
            id="scalar_register_read_before_its_wait",
        ),
        pytest.param(
            "wave 1 aborted at instruction 8: LDS access at 0x100 is misaligned or outside the 256-byte allocation",
            "lds_bounds",
            4,
            id="lds_access_past_the_allocation",
        ),
        pytest.param(
            "wave 0 aborted at instruction 9: LDS access at 0xfc is misaligned or outside the 256-byte allocation",
            "lds_bounds",
            4,
            id="lds_access_running_past_the_allocation",
        ),
        pytest.param(
            "wave 1 aborted at instruction 6: 4-byte read at 0x10100 is not an element of an input array",
            "global_bounds",
            6,
            id="global_read_past_its_array",
        ),
        pytest.param(
            "wave 1 aborted at instruction 6: 4-byte write at 0x20200 is not an element of an output array",
            "global_bounds",
            6,
            id="global_write_past_its_array",
        ),
    ],
)
def test_a_run_that_stops_at_a_defect_rejects_its_rule_and_leaves_the_others_unchecked(
    tmp_path, volta, abort, rule, number
):
    report = _verify(tmp_path, _Extractor(tmp_path, _log(abort=abort)))
    assert report.status == "failed"
    assert _statuses(report) == dict.fromkeys(_RULES, "unchecked") | {rule: "rejected"}
    (finding,) = report.findings
    stopped = abort.removeprefix("wave 0 aborted at ").removeprefix("wave 1 aborted at ")
    assert finding.detail == f"VR{number} {rule} exchange: {stopped}"
    assert all(d.startswith("aborted: ") for r, d in _details(report).items() if r != rule)


@pytest.mark.parametrize(
    "abort",
    [
        pytest.param(
            "wave 0 aborted at instruction 9: LDS access at 0x2 is misaligned or outside the 256-byte allocation",
            id="misaligned_lds_access_inside_the_allocation",
        ),
        pytest.param(
            "wave 0 aborted at instruction 6: 4-byte read at 0x20010 is not an element of an input array",
            id="read_of_an_output_array",
        ),
        pytest.param(
            "wave 0 aborted at instruction 6: 4-byte write at 0x10000 is not an element of an output array",
            id="write_of_an_input_array",
        ),
        pytest.param(
            "wave 0 aborted at instruction 7: 4-byte write at 0x0 is not an element of an output array",
            id="write_through_an_address_no_array_is_near",
        ),
        pytest.param(
            "wave 0 aborted at instruction 7: 4-byte read at 0x30000 is not an element of an input array",
            id="read_beyond_the_page_after_the_last_array",
        ),
        pytest.param(
            "wave 0 aborted at instruction 6: buffer access at byte offset 0xfe is partially out of range",
            id="buffer_access_partly_in_range",
        ),
        pytest.param(
            "wave 0 aborted at instruction 3: unsupported instruction", id="unsupported_instruction"
        ),
    ],
)
def test_a_run_that_stops_at_what_volta_does_not_model_is_unchecked(tmp_path, volta, abort):
    report = _verify(tmp_path, _Extractor(tmp_path, _log(abort=abort)))
    assert report.status == "passed"
    assert report.findings == ()
    assert _statuses(report) == dict.fromkeys(_RULES, "unchecked")
    stopped = abort.removeprefix("wave 0 aborted at ")
    assert set(_details(report).values()) == {f"aborted: {stopped}"}


def test_an_lds_word_read_before_any_write_is_unchecked_under_its_own_reason(tmp_path, volta):
    abort = "wave 0 aborted at instruction 10: LDS read at 0x100 of a word no wave has written"
    report = _verify(tmp_path, _Extractor(tmp_path, _log(abort=abort)))
    assert report.status == "passed"
    assert _statuses(report) == dict.fromkeys(_RULES, "unchecked")
    details = _details(report)
    assert details.pop("lds_uninitialized_read") == (
        "uninitialized-read: instruction 10: LDS read at 0x100 of a word no wave has written"
    )
    assert set(details.values()) == {
        "aborted: instruction 10: LDS read at 0x100 of a word no wave has written"
    }


def test_a_race_found_before_the_run_stopped_is_still_rejected(tmp_path, volta):
    abort = "wave 0 aborted at instruction 3: unsupported instruction"
    report = _verify(tmp_path, _Extractor(tmp_path, _log({"memory-conflict freedom": _WRITE_READ}, abort)))
    assert _statuses(report) == dict.fromkeys(_RULES, "unchecked") | {"lds_race": "rejected"}
    assert _details(report)["lds_race"].endswith(
        "; the run aborted at instruction 3: unsupported instruction"
    )


def test_a_launch_volta_cannot_run_is_unchecked(tmp_path, volta):
    log = "kernel.s: AMD gfx950 kernel\nerror: unsupported launch: AMD array `in` has 1-byte elements\n"
    report = _verify(tmp_path, _Extractor(tmp_path, log))
    assert report.status == "passed"
    assert _statuses(report) == dict.fromkeys(_RULES, "unchecked")
    assert set(_details(report).values()) == {
        "unsupported: unsupported launch: AMD array `in` has 1-byte elements"
    }


def test_without_volta_every_rule_is_unchecked(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    report = _verify(tmp_path, _Extractor(tmp_path, _log()))
    assert report.status == "passed"
    assert _statuses(report) == dict.fromkeys(_RULES, "unchecked")
    assert set(_details(report).values()) == {"no-volta: the volta binary is not on PATH"}


def test_a_workload_that_does_not_compile_is_unchecked_for_that_reason(tmp_path, volta):
    report = _verify(tmp_path, _Extractor(tmp_path, failure="requires: flydsl>=0.5"))
    assert report.status == "passed"
    assert _statuses(report) == dict.fromkeys(_RULES, "unchecked")
    assert set(_details(report).values()) == {"requires: flydsl>=0.5"}


def test_every_workload_of_every_kernel_is_compiled_and_a_definition_is_not(tmp_path, volta):
    extractor = _Extractor(tmp_path, _log())
    report = _verify(tmp_path, extractor, _schema(n=6), _schema(source=None), devices=(4, 5))
    assert len(report.path_records) == 6
    assert sorted(str(record.workload) for record in report.path_records) == [f"m={m}" for m in range(1, 7)]
    assert len({schema_path for schema_path, *_ in extractor.extracted}) == 1
    assert sorted((first, count) for _, first, count, _ in extractor.extracted) == [(0, 4), (4, 4)]
    assert {device for *_, device in extractor.extracted} == {"4", "5"}


def test_the_report_counts_each_rule_and_lists_what_was_rejected(tmp_path, volta):
    abort = "wave 0 aborted at instruction 7: vector register v2 in lane 0 has a pending load"
    report = _verify(tmp_path, _Extractor(tmp_path, _log(abort=abort)))
    lines = [line for block in report.markdown_blocks() for line in block.lines()]
    assert "| rule | proved | rejected | unchecked |" in lines
    assert "| VR3 pending&#95;read | 0 | 1 | 0 |" in lines
    assert "| VR1 lds&#95;race | 0 | 0 | 1 |" in lines
    assert "| VR1 lds&#95;race | aborted | 1 |" in lines
    assert any(line.startswith("| flydsl/") and "| VR3 pending&#95;read |" in line for line in lines)
    assert report.console_lines()[1] == "  proved 0, rejected 1, unchecked 5"


def test_a_written_report_reads_back(tmp_path, volta):
    report = _verify(tmp_path, _Extractor(tmp_path, _log({"memory-conflict freedom": _WRITE_READ})))
    written = TypeAdapter(VerifyReport).dump_python(report, mode="json")
    assert written["name"] == "verify"
    assert TypeAdapter(VerifyReport).validate_python(written) == report


_TRITON_GEMM_KERNEL_SOURCE = """\
import triton
import triton.language as tl


@triton.jit
def gemm(a, b, out, k: tl.constexpr, n: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.arange(0, n)
    depth = tl.arange(0, k)
    x = tl.load(a + row * k + depth).to(tl.float32)
    w = tl.load(b + columns[:, None] * k + depth[None, :]).to(tl.float32)
    tl.store(out + row * n + columns, tl.sum(w * x[None, :], axis=1).to(tl.bfloat16))


def run(*, a, b, out, **_):
    gemm[(a.shape[0],)](a, b, out, k=a.shape[1], n=b.shape[0], num_warps=1)
"""
# the Triton gemm also flagging each row it wrote in a uint32 and a bool tensor run() allocates
_TRITON_FLAGGED_GEMM_KERNEL_SOURCE = """\
import torch
import triton
import triton.language as tl


@triton.jit
def gemm(a, b, out, flags, done, k: tl.constexpr, n: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.arange(0, n)
    depth = tl.arange(0, k)
    x = tl.load(a + row * k + depth).to(tl.float32)
    w = tl.load(b + columns[:, None] * k + depth[None, :]).to(tl.float32)
    tl.store(out + row * n + columns, tl.sum(w * x[None, :], axis=1).to(tl.bfloat16))
    tl.store(flags + row, 1)
    tl.store(done + row, True)


def run(*, a, b, out, **_):
    flags = torch.zeros(a.shape[0], dtype=torch.uint32, device=a.device)
    done = torch.zeros(a.shape[0], dtype=torch.bool, device=a.device)
    gemm[(a.shape[0],)](a, b, out, flags, done, k=a.shape[1], n=b.shape[0], num_warps=1)
"""
# the gemm case's arrays: a is m=4 x k=128, b n=16 x k=128, out m x n, all bf16
_GEMM_ARRAYS = [
    {"name": "a", "kind": "in", "dtype": "bf16", "shape": [4, 128]},
    {"name": "b", "kind": "in", "dtype": "bf16", "shape": [16, 128]},
    {"name": "out", "kind": "out", "dtype": "bf16", "shape": [4, 16]},
]
_GEMM_POINTERS = [{"ptr": "a"}, {"ptr": "b"}, {"ptr": "out"}]
# a templated kernel staging a @ b.T in a float scratch tensor run() allocates, and a second kernel
# converting it to out
_HIP_STAGED_GEMM_KERNEL_SOURCE = """\
#include <ATen/hip/HIPContext.h>
#include <ATen/ops/empty.h>
#include <hip/hip_bf16.h>
#include <optional>
#include <torch/library.h>

namespace staged {

template <int K>
__global__ void accumulate(const __hip_bfloat16 *a, const __hip_bfloat16 *b, float *scratch) {
    float acc = 0.0f;
    for (int i = 0; i < K; ++i) {
        acc += __bfloat162float(a[blockIdx.x * K + i]) * __bfloat162float(b[threadIdx.x * K + i]);
    }
    scratch[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

} // namespace staged

namespace {

__global__ void convert(const float *scratch, __hip_bfloat16 *out) {
    out[blockIdx.x * blockDim.x + threadIdx.x] = __float2bfloat16(scratch[blockIdx.x * blockDim.x + threadIdx.x]);
}

void run(const at::Tensor &a, const at::Tensor &b, bool trans_b,
         const std::optional<at::Tensor> &bias, at::ScalarType out_dtype, at::Tensor &out) {
    auto scratch = at::empty({a.size(0), b.size(0)}, a.options().dtype(at::kFloat));
    auto stream = at::hip::getCurrentHIPStream();
    staged::accumulate<128><<<a.size(0), b.size(0), 0, stream>>>(
        static_cast<const __hip_bfloat16 *>(a.data_ptr()), static_cast<const __hip_bfloat16 *>(b.data_ptr()),
        static_cast<float *>(scratch.data_ptr()));
    convert<<<a.size(0), b.size(0), 0, stream>>>(
        static_cast<const float *>(scratch.data_ptr()), static_cast<__hip_bfloat16 *>(out.data_ptr()));
}

} // namespace

#define TORCH_LIBRARY_EXPAND(ns, m) TORCH_LIBRARY(ns, m)
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
    m.def("run(Tensor a, Tensor b, bool trans_b, Tensor? bias, ScalarType out_dtype, "
          "Tensor(a!) out) -> ()",
          &run);
}
"""


def _extractions(tmp_path: Path, language: str, source: str, other_sources=None) -> list[Extraction]:
    """The gemm case's one workload extracted from the ``language`` kernel ``source``: its launches."""
    schema = OpCase.of("gemm@1.0").schema(source, language=language, other_sources=other_sources)
    schema_path = f"aklr/plugins/aiter/provider/gemm/impls/{language}/{schema.name}/{0:0>16}.json"
    (tmp_path / schema_path).parent.mkdir(parents=True)
    (tmp_path / schema_path).write_text(schema.schema_text(), encoding="utf-8")
    extractions = Extraction.of_schema(tmp_path, schema_path, schema, tmp_path / "extracted", 0, 1, None)
    for extraction in extractions:
        assert extraction.failure == ""
        assert extraction.axes == {"m": 4}
        assert f".amdhsa_kernel {extraction.kernel}\n" in Path(extraction.isa_path).read_text(
            encoding="utf-8"
        )
    return extractions


def _extracted(tmp_path: Path, language: str, source: str, other_sources=None) -> Extraction:
    """The gemm case's one workload extracted from the ``language`` kernel ``source``: its one launch."""
    (extraction,) = _extractions(tmp_path, language, source, other_sources)
    return extraction


def test_a_kernel_whose_isa_cannot_be_extracted_names_the_extraction_and_its_log(tmp_path):
    schema = OpCase.of("gemm@1.0").schema("raise RuntimeError('broken kernel')\n")
    schema_path = f"aklr/plugins/aiter/provider/gemm/impls/flydsl/{schema.name}/{0:0>16}.json"
    (tmp_path / schema_path).parent.mkdir(parents=True)
    (tmp_path / schema_path).write_text(schema.schema_text(), encoding="utf-8")
    (extraction,) = Extraction.of_schema(tmp_path, schema_path, schema, tmp_path / "extracted", 0, 1, None)
    assert extraction.kernel is None
    assert extraction.failure.startswith("extract: ")
    assert (tmp_path / "extracted" / "extractor.log").is_file()


@pytest.mark.gpu
def test_a_triton_kernel_compiles_to_its_isa_and_launch(tmp_path):
    extraction = _extracted(tmp_path, "triton", _TRITON_GEMM_KERNEL_SOURCE)
    assert extraction.kernel == "gemm"
    launch = extraction.launch
    assert (launch["grid"], launch["block"]) == ([4, 1, 1], [64, 1, 1])
    assert launch["arrays"] == _GEMM_ARRAYS
    assert launch["params"][:3] == _GEMM_POINTERS
    assert set(map(str, launch["params"][3:])) == {"{'int': 0}"}


@pytest.mark.gpu
def test_a_triton_kernel_writing_unsigned_and_bool_tensors_lays_them_out(tmp_path):
    launch = _extracted(tmp_path, "triton", _TRITON_FLAGGED_GEMM_KERNEL_SOURCE).launch
    assert launch["arrays"][3:] == [
        {"name": "workspace24", "kind": "out", "dtype": "i32", "shape": [4]},
        {"name": "workspace32", "kind": "out", "dtype": "u8", "shape": [4]},
    ]
    assert launch["params"][3:5] == [{"ptr": "workspace24"}, {"ptr": "workspace32"}]


@pytest.mark.gpu
def test_a_hip_kernel_compiles_to_its_isa_and_launch(tmp_path):
    extraction = _extracted(tmp_path, "hip", HIP_GEMM_KERNEL_SOURCE)
    assert "gemm" in extraction.kernel
    launch = extraction.launch
    assert (launch["grid"], launch["block"], launch["dynamic_shared_bytes"]) == ([4, 1, 1], [16, 1, 1], 0)
    assert launch["arrays"] == _GEMM_ARRAYS
    assert launch["params"][:4] == [*_GEMM_POINTERS, {"int": 128}]


@pytest.mark.gpu
def test_a_templated_hip_kernel_reads_a_scratch_tensor_the_call_frees(tmp_path):
    accumulate, convert = _extractions(tmp_path, "hip", _HIP_STAGED_GEMM_KERNEL_SOURCE)
    assert "accumulate" in accumulate.kernel
    assert "convert" in convert.kernel
    scratch = {"name": "workspace16", "kind": "out", "dtype": "i32", "shape": [64]}
    assert accumulate.launch["arrays"] == [*_GEMM_ARRAYS[:2], scratch]
    assert accumulate.launch["params"][:3] == [{"ptr": "a"}, {"ptr": "b"}, {"ptr": "workspace16"}]
    assert convert.launch["arrays"] == [{**scratch, "name": "workspace0", "kind": "in"}, _GEMM_ARRAYS[2]]
    assert convert.launch["params"][:2] == [{"ptr": "workspace0"}, {"ptr": "out"}]


@pytest.mark.gpu
def test_an_hsaco_kernel_compiles_to_its_isa_and_launch(tmp_path):
    extraction = _extracted(tmp_path, "hsaco", HSACO_GEMM_KERNEL_SOURCE, hsaco_gemm_sources())
    assert extraction.kernel == "gemm"
    launch = extraction.launch
    assert (launch["grid"], launch["block"], launch["dynamic_shared_bytes"]) == ([4, 1, 1], [16, 1, 1], 0)
    assert launch["arrays"] == _GEMM_ARRAYS
    assert launch["params"][:5] == [*_GEMM_POINTERS, {"int": 16}, {"int": 128}]
