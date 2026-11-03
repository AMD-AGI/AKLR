# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""KernelForge driver for one AKLR schema. Correctness is what ``python -m aklr validate`` returns for the kernel's
sources (``kernel.py`` and the modules it imports relatively) published as the schema's kernel; timing is AKLR's own
graph timer on AKLR's timing inputs, the candidate alone. ``forge_task.workspace`` fills the placeholders below.
"""

from __future__ import annotations

import argparse
import ast
import faulthandler
import hashlib
import importlib
import json
import os
import signal
import statistics
import subprocess
import sys
import types
from pathlib import Path

AKLR_ROOT = Path("@AKLR_ROOT@")
SCHEMA_JSON = Path("@SCHEMA_JSON@")
OP_DIR = "@OP_DIR@"
LANGUAGE = "@LANGUAGE@"
# One validate run per kernel serves correctness and the baseline anchor: it also times the baseline, which takes
# longer than KernelForge allows a single benchmark, so the bench mode times the candidate alone.
VALIDATE_DIR = Path("@VALIDATE_DIR@")
WORKSPACE = Path(__file__).resolve().parent
KERNEL = WORKSPACE / "kernel.py"
# The package the candidate's sources are imported into, as AKLR imports a kernel's: a package made in memory.
CANDIDATE_PACKAGE = "forge_candidate"

# Findings that mean a workload's output cannot be trusted; NOT_FASTER and REGRESSED only grade speed.
INCORRECT_CODES = frozenset(
    {
        "schema_invalid",
        "kernel_missing",
        "source_rejected",
        "build_failed",
        "candidate_crash",
        "candidate_timeout",
        "capture_unsafe",
        "input_mutation",
        "numeric_mismatch",
        "output_caching",
        "nondeterministic",
        "post_timing_mismatch",
    }
)


def _aklr():
    sys.path.insert(0, str(AKLR_ROOT))
    from aklr.core.contract.locate import schema_relative_path, schema_uuid
    from aklr.core.contract.op import OpContract
    from aklr.core.contract.registry import ContractRegistry

    return ContractRegistry.default(), OpContract, schema_relative_path, schema_uuid


def _relative_imports(source: str) -> list[str]:
    """The file names ``source`` imports from its own package: ``x.py`` for ``from .x import y`` and for each ``x``
    of ``from . import x``. A source that does not parse imports nothing; validate reports the error.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 1:
            modules = [node.module] if node.module else [alias.name for alias in node.names]
            names += [f"{module}.py" for module in modules]
    return names


def kernel_sources() -> dict[str, str]:
    """The kernel's files by name, ascending as AKLR orders them: ``kernel.py`` and every module it imports
    relatively, transitively. A name no file carries (``from . import f`` of a function) is none, nor is a file
    ``kernel.py`` does not reach, such as an agent's probe, nor the driver."""
    sources = {}
    pending = [KERNEL.name]
    while pending:
        name = pending.pop()
        path = WORKSPACE / name
        if name in sources or path == Path(__file__).resolve() or not path.is_file():
            continue
        sources[name] = path.read_text(encoding="utf-8")
        pending += _relative_imports(sources[name])
    return dict(sorted(sources.items()))


def _candidate_text() -> tuple[str, str]:
    """The schema text AKLR expects for this kernel, and the file name AKLR derives from it."""
    registry, op_contract, _, schema_uuid = _aklr()
    raw = json.loads(SCHEMA_JSON.read_text(encoding="utf-8"))
    raw["kernel"]["language"] = LANGUAGE
    raw["kernel"]["sources"] = [
        {"path": name, "content": content} for name, content in kernel_sources().items()
    ]
    contract = registry.for_schema_json(raw)
    point = contract.point_from_json(raw, raw["workloads"][0]["axes"])
    text = contract.narrow(point, workloads=raw["workloads"], kernel=raw["kernel"]).schema_text()
    return text, schema_uuid(raw["name"], op_contract.authored_parts(json.loads(text)))


def _publish() -> Path:
    """Write the candidate where AKLR keeps kernels; return its path relative to the checkout."""
    _, _, schema_relative_path, _ = _aklr()
    text, uuid = _candidate_text()
    raw = json.loads(text)
    path = Path("aklr/ops", OP_DIR, schema_relative_path(raw["name"], LANGUAGE, uuid))
    (AKLR_ROOT / path).parent.mkdir(parents=True, exist_ok=True)
    (AKLR_ROOT / path).write_text(text, encoding="utf-8")
    return path


def _platform_failed(report: dict) -> bool:
    return any(
        f["code"] == "platform_failure" for stage in report["stages"] for f in stage.get("findings", [])
    )


def _report() -> dict:
    """AKLR's validation report for the current kernel, running validate once per kernel and once more when the
    platform failed the run: a tracer fault aborts every workload, and it is the platform's, not the kernel's.
    """
    sources = sorted(kernel_sources().items())
    digest = hashlib.sha256(json.dumps(sources).encode()).hexdigest()[:16]
    report = _validate(VALIDATE_DIR / digest)
    if _platform_failed(report):
        report = _validate(VALIDATE_DIR / f"{digest}-rerun")
    return report


def _validate(out: Path) -> dict:
    report = out / "report.json"
    if report.is_file():
        return json.loads(report.read_text(encoding="utf-8"))
    path = _publish()
    # `python -m` run from the checkout's root imports that checkout's aklr.
    command = [
        sys.executable,
        "-m",
        "aklr",
        "validate",
        "--stages",
        "audit,bench",
        "--devices",
        "0",
        "--out",
        str(out),
    ]
    try:
        completed = subprocess.run(
            [*command, str(path)],
            cwd=AKLR_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
    finally:
        # Another kernel's file left in the checkout would be served beside the next one.
        (AKLR_ROOT / path).unlink(missing_ok=True)
    if not report.is_file():
        sys.stderr.write(completed.stdout[-3000:] + completed.stderr[-3000:])
        raise SystemExit(f"aklr validate wrote no report (exit {completed.returncode})")
    return json.loads(report.read_text(encoding="utf-8"))


def _stage(report: dict, name: str) -> dict:
    return next((stage for stage in report["stages"] if stage["stage_name"] == name), {})


def _case_id(axes: dict) -> str:
    return ",".join(f"{name}={value}" for name, value in axes.items())


def _outcomes(report: dict) -> dict[str, dict]:
    """Per workload: whether AKLR found its output wrong, and the baseline's graph time."""
    workloads = json.loads(SCHEMA_JSON.read_text(encoding="utf-8"))["workloads"]
    outcomes = {_case_id(w["axes"]): {"incorrect": [], "baseline_ms": None} for w in workloads}
    for stage in ("audit", "bench"):
        # A workload's own finding is filed under the stage's demotions: AKLR serves the schema's other workloads.
        found = _stage(report, stage)
        for finding in (*found.get("findings", []), *found.get("demotions", [])):
            if finding["code"] not in INCORRECT_CODES:
                continue
            # An audit finding names no workload: it rejects the kernel everywhere.
            cases = [_case_id(finding["axes"])] if finding.get("axes") else list(outcomes)
            for case_id in cases:
                outcomes[case_id]["incorrect"].append(finding["code"])
    for record in _stage(report, "bench").get("workload_records", []):
        outcome = outcomes[_case_id(record["axes"])]
        outcome["violation"] = record.get("worst_violation_fraction")
        for timing in record["timings"]:
            if timing["mode"] == "graph" and timing["versus"] == "baseline":
                outcome["baseline_ms"] = statistics.median(timing["versus_milliseconds"])
    for outcome in outcomes.values():
        # An aborted run compares nothing; a workload AKLR did not compare is not shown correct.
        if outcome.get("violation") is None and not outcome["incorrect"]:
            outcome["incorrect"].append("not_compared")
    return outcomes


def _run_correctness() -> int:
    """Report AKLR's findings per workload; KernelForge gates on ``allclose``, which holds only when AKLR passes
    every workload. AKLR grades a fraction of elements outside its error bound, not an SNR.
    """
    outcomes = _outcomes(_report())
    for case_id, outcome in outcomes.items():
        if outcome["incorrect"]:
            print(f"aklr_findings: {case_id} {' '.join(sorted(set(outcome['incorrect'])))}")
    passed = [not outcome["incorrect"] for outcome in outcomes.values()]
    print(f"allclose: {bool(passed) and all(passed)}")
    return 0


def _write_baseline(path: Path) -> int:
    """KernelForge's scoring anchor: per workload, the baseline's median graph time from AKLR's validate run, and
    the speedup AKLR serves a workload at, which KernelForge's dashboard marks each workload against.
    """
    case_times = {case_id: outcome["baseline_ms"] for case_id, outcome in _outcomes(_report()).items()}
    missing = [case_id for case_id, ms in case_times.items() if not ms]
    if missing:
        print(
            f"AKLR timed no baseline for {', '.join(missing)}: prepare with --anchor start to score against "
            "the starting kernel",
            file=sys.stderr,
        )
        return 2
    payload = {
        "wall_ms": sum(case_times.values()),
        "case_times": case_times,
        "label": "AKLR baseline",
    }
    _aklr()
    from aklr.core.platform.env import RuntimeEnv

    serving = RuntimeEnv().min_speedup
    # At or below zero AKLR serves every workload it did not fail: there is no bar to mark.
    if serving > 0:
        payload["serving_speedup"] = serving
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


def _aklr_timer():
    """AKLR's graph timer as validate's bench reads it: the budget of one reading and the clock."""
    _aklr()
    from aklr.core.validation.stages._measure import GRAPH_BUDGET_MILLISECONDS, bind_clock

    return GRAPH_BUDGET_MILLISECONDS, bind_clock


def _measure_candidate(warmup: int, rounds: int) -> dict[str, float]:
    """Per workload, the candidate's median time over ``rounds`` readings of AKLR's graph timer on the call
    validate's bench times."""
    budget, bind_clock = _aklr_timer()
    from aklr.core.utils.testing import kernel_tracer

    # rocprofiler-sdk takes the timer's kernel tracer only before the process's first GPU call
    kernel_tracer().configure()
    clock = bind_clock()
    candidate = _load_candidate()
    times = {}
    profiler_warm = False
    for case_id, call in _calls("cuda").items():

        def call_once(call=call) -> None:
            candidate.run(**call)

        # At least once, as a model's warm-up does: a per-tensor cache fills here, not in a timed graph.
        for _ in range(max(1, warmup)):
            call_once()
        clock.synchronize()
        if not profiler_warm:
            # Validate's profiler warm-up: one unrecorded reading before the first recorded one in the process.
            clock.bench_cudagraph(call_once, rep=budget)
            profiler_warm = True
        readings = [clock.bench_cudagraph(call_once, rep=budget) for _ in range(rounds)]
        times[case_id] = statistics.median(readings)
    return times


def _run_bench(warmup: int, iterations: int) -> int:
    """Emit one ``case_ms`` line per workload plus the suite aggregate."""
    times = _measure_candidate(warmup, max(1, iterations))
    for case_id, milliseconds in times.items():
        print(f"case_ms: {case_id} {milliseconds:.6f}")
    # A schema is served workload by workload, but the loop reads one aggregate: the whole set's time.
    print(f"median_ms: {sum(times.values()):.6f}")
    return 0


def _calls(device: str) -> dict[str, dict]:
    """Per workload, the call validate's bench times, on AKLR's timing inputs, built by AKLR's own trial code:
    inputs from another draw describe a call nobody scores. The call holds the schema's call keys only, as AKLR's
    builder hands the kernel: a trial call also carries API parameters an attr hides from the kernel. The outputs
    dispatch would allocate are allocated here, since the candidate is called directly.
    """
    import torch

    registry, *_ = _aklr()
    from aklr.core.contract import semantics
    from aklr.core.validation.stages._bench._evaluation._trials import TrialRequest

    text, _ = _candidate_text()
    schema = registry.parse_schema(text.encode(), str(SCHEMA_JSON))
    contract = registry.get(schema.contract_id)
    calls = {}
    for workload in schema.workloads:
        request = TrialRequest.of_workload(
            str(SCHEMA_JSON),
            schema,
            contract,
            workload,
            thresholds={},
            incumbent=None,
            baseline=None,
        )
        assembled = request.assemble_call(request.compose_inputs(None), device)
        call = {key: assembled[key] for key in schema.kernel_call_keys if key in assembled}
        for name in contract.allocated_outputs:
            call[name] = torch.empty(
                contract.storage_shape_of(name, request.point),
                dtype=semantics.torch_dtypes(request.point.dtypes[name])[0],
                device=device,
            )
        calls[_case_id(workload.axes)] = call
    return calls


def _load_candidate():
    """``kernel.py`` imported as a module of a package over the workspace, as AKLR imports it, so that its relative
    imports find the files beside it."""
    for name in [n for n in sys.modules if f"{n}.".startswith(f"{CANDIDATE_PACKAGE}.")]:
        del sys.modules[name]
    package = types.ModuleType(CANDIDATE_PACKAGE)
    package.__path__ = [str(WORKSPACE)]
    sys.modules[CANDIDATE_PACKAGE] = package
    return importlib.import_module(f"{CANDIDATE_PACKAGE}.{KERNEL.stem}")


def _run_profile() -> int:
    """Dispatch only the candidate on the largest workload, so a profiler sees its kernels."""
    import torch

    calls = _calls("cuda")
    largest = max(
        calls,
        key=lambda case_id: sum(int(part.split("=")[1]) for part in case_id.split(",")),
    )
    candidate = _load_candidate()
    for _ in range(6):
        candidate.run(**calls[largest])
    torch.cuda.synchronize()
    return 0


# A stream limited to fewer CUs than a launch has workgroups: workgroups that wait for each other cannot all run.
STRESS_CUS = (16, 64)
STRESS_TRIES = 3
# Under bounded determinism two runs may differ in the last bits (atomics land in another order): a difference counts
# against the schema's violation fraction only beyond this many output ulps of the element plus the output's mean.
STRESS_BOUNDED_ULPS = 8
# A call under stress not finished after this long is taken as hung: its workgroups wait for each other.
STRESS_SECONDS = 60.0


class StressHang(Exception):
    """A call under stress did not finish within ``STRESS_SECONDS``."""


def _settle(streams) -> None:
    """Wait for the work queued on ``streams``, at most ``STRESS_SECONDS``: a kernel whose workgroups wait for each
    other may never finish, and a synchronize would wait with it."""
    import time

    import torch

    events = []
    for stream in streams:
        event = torch.cuda.Event()
        event.record(stream)
        events.append(event)
    deadline = time.monotonic() + STRESS_SECONDS
    while not all(event.query() for event in events):
        if time.monotonic() > deadline:
            raise StressHang
        time.sleep(0.001)


def _output_names() -> list[str]:
    registry, *_ = _aklr()
    schema = registry.parse_schema(_candidate_text()[0].encode(), str(SCHEMA_JSON))
    return list(registry.get(schema.contract_id).outputs)


def _cu_limited_stream(cus: int):
    """A stream whose kernels run on the first ``cus`` CUs only, made by the HIP runtime torch loaded: a second copy
    of the runtime makes streams torch cannot use."""
    import ctypes

    import torch

    maps = Path("/proc/self/maps").read_text().splitlines()
    hip = ctypes.CDLL(next(line.split()[-1] for line in maps if "libamdhip64.so" in line))
    total = torch.cuda.get_device_properties(0).multi_processor_count
    mask = (ctypes.c_uint32 * ((total + 31) // 32))()
    for cu in range(min(cus, total)):
        mask[cu // 32] |= 1 << (cu % 32)
    handle = ctypes.c_void_p()
    if hip.hipExtStreamCreateWithCUMask(ctypes.byref(handle), len(mask), mask):
        raise RuntimeError(f"no stream limited to {cus} CUs")
    return torch.cuda.ExternalStream(handle.value)


def _stress_conditions() -> dict:
    """Ways to call the candidate under contention, each returning the calls whose outputs it wrote: on a stream
    limited to a few CUs, and beside a second call of its own on another stream, into outputs of its own."""
    import torch

    def on(stream):
        def launch(candidate, call, outputs):
            for name in outputs:
                call[name].zero_()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                candidate.run(**call)
            _settle([stream, torch.cuda.current_stream()])
            return [call]

        return launch

    side = torch.cuda.Stream()

    def beside_another(candidate, call, outputs):
        twin = {**call, **{name: torch.zeros_like(call[name]) for name in outputs}}
        for name in outputs:
            call[name].zero_()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            candidate.run(**twin)
        candidate.run(**call)
        _settle([side, torch.cuda.current_stream()])
        return [call, twin]

    conditions = {f"on {cus} CUs": on(_cu_limited_stream(cus)) for cus in STRESS_CUS}
    conditions["beside a second call"] = beside_another
    return conditions


def _compare_spec() -> tuple[str, float]:
    compare = json.loads(_candidate_text()[0]).get("compare") or {}
    return compare.get("determinism", "bitwise"), float(compare.get("max_violation_fraction", 0.0))


def _agrees(stressed, alone, determinism: str, max_fraction: float) -> bool:
    """Whether an output under contention agrees with the output alone as the schema's determinism asks: bit for bit
    under ``bitwise``; under ``bounded``, no value turned non-finite and all but ``max_fraction`` of the elements
    within a few ulps."""
    import torch

    if determinism == "bitwise" or not alone.is_floating_point():
        return torch.equal(stressed.contiguous().view(torch.uint8), alone.contiguous().view(torch.uint8))
    unit = torch.finfo(alone.dtype).eps / 2
    stressed, alone = stressed.double(), alone.double()
    finite = torch.isfinite(alone)
    if bool((finite & ~torch.isfinite(stressed)).any()):
        return False
    bound = STRESS_BOUNDED_ULPS * unit * (alone.abs() + alone[finite].abs().mean())
    off = finite & ((stressed - alone).abs() > bound)
    return float(off.double().mean()) <= max_fraction


def _run_stress(device: str = "cuda") -> int:
    """Every workload under each stress condition, against the candidate's own result alone, as the schema's
    determinism asks. AKLR's validate runs a kernel with the GPU to itself, so it passes a kernel that is right only
    while every workgroup of a launch is resident at once, and one whose concurrent calls share scratch memory. A call
    that does not finish is reported as hung and ends the check: the device is still running it."""
    candidate = _load_candidate()
    outputs = _output_names()
    determinism, max_fraction = _compare_spec()
    conditions = _stress_conditions()
    failed = []
    for case_id, call in _calls(device).items():
        candidate.run(**call)
        alone = {name: call[name].clone() for name in outputs if name in call}
        for label, launch in conditions.items():
            for _ in range(STRESS_TRIES):
                try:
                    results = launch(candidate, call, alone)
                except StressHang:
                    print(f"stress: hung, no result within {STRESS_SECONDS:g} s: {case_id} {label}")
                    print("stress: FAIL")
                    return 1
                agree = (
                    _agrees(r[name], alone[name], determinism, max_fraction)
                    for r in results
                    for name in alone
                )
                if not all(agree):
                    failed.append(f"{case_id} {label}")
                    break
    for failure in failed:
        print(f"stress: differs from the call alone: {failure}")
    print(f"stress: {'FAIL' if failed else 'PASS'}")
    return 1 if failed else 0


def main() -> int:
    # KernelForge sends SIGUSR1 to a driver that outruns its timeout before killing it: every thread's stack is the
    # only record of where it hung.
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    parser = argparse.ArgumentParser(description=f"KernelForge driver for {SCHEMA_JSON.stem}")
    parser.add_argument("--bench-mode", action="store_true")
    parser.add_argument("--profile-run", action="store_true")
    parser.add_argument(
        "--baseline-json",
        type=Path,
        help="write AKLR's baseline times as KernelForge's --baseline-json",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="write the kernel into the AKLR checkout as the schema's kernel, and keep it there",
    )
    parser.add_argument(
        "--stress",
        action="store_true",
        help="run every workload on streams limited to a few CUs and beside a second call; same answer or fail",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--iters",
        type=int,
        default=30,
        help="readings of AKLR's graph timer per workload",
    )
    args, _unknown = parser.parse_known_args()
    if args.publish:
        print(_publish())
        return 0
    if args.profile_run:
        return _run_profile()
    if args.stress:
        code = _run_stress()
        sys.stdout.flush()
        # A hung kernel cannot be waited out at interpreter exit; leaving at once drops the device context with it.
        os._exit(code)
    if args.baseline_json:
        return _write_baseline(args.baseline_json)
    if args.bench_mode:
        return _run_bench(args.warmup, args.iters)
    return _run_correctness()


if __name__ == "__main__":
    sys.exit(main())
