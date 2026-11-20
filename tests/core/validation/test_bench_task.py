# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The bench on a GPU, through `python -m aklr validate`: its accuracy check, timing and tasks."""

import contextlib
import json
import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

from aklr.core.platform.device import DeviceTarget
from aklr.core.validation.stages.bench import InputSampler
from tests.op_case import OpCase

pytestmark = pytest.mark.gpu

_AKLR = Path(__file__).resolve().parents[3]
_CASE = OpCase.of("gemm@1.0")
_BASELINE = "aklr.ops.gemm.baseline.gemm_baseline"
# a kernel takes exactly its schema's call keys, keyword-only
_DEF = f'"""A test kernel."""\n\n\ndef run(*, {", ".join(_CASE.schema().kernel_call_keys)}):\n'
_GEMM = _DEF + "    out.copy_(a @ b.mT)\n"
_WRONG = _DEF + "    out.zero_()\n"
_RAISING = _DEF + "    raise RuntimeError('broken kernel')\n"
_OTHER = _GEMM + "# another kernel\n"
_NOISY = (
    # Keep a duplicate of stderr, as a subprocess from this task would.
    _DEF + "    import os\n    import sys\n\n"
    "    sys.previous_task_stderr = os.dup(2)\n"
    "    sys.stdout.write('x' * (6 << 20))\n    sys.stdout.flush()\n    out.zero_()\n"
)
_CRASHING = (
    _DEF + "    import os\n    import sys\n\n"
    "    os.write(sys.previous_task_stderr, b'late output from the first task')\n"
    "    sys.stdout.write('the crash is here')\n    sys.stdout.flush()\n    os._exit(7)\n"
)
_WRONG_ON_RANK_ONE = (
    _GEMM + "    import torch.distributed as dist\n\n    if dist.get_rank() == 1:\n        out.zero_()\n"
)
# gemm's bench in a validation these tests start: on AKLR_TEST_RANKS ranks, each calling the op on its own
# inputs, and with AKLR_TEST_CLOCK_MS a clock of its own that reads that
_BENCH = """\
import os

from aklr.core.validation import OpBench


class GemmBench(OpBench, name="gemm"):
    @classmethod
    def ranks(cls, schema):
        return int(os.environ.get("AKLR_TEST_RANKS", "1"))

    def clock(self, fn):
        if "AKLR_TEST_CLOCK_MS" not in os.environ:
            return super().clock(fn)
        fn()
        return float(os.environ["AKLR_TEST_CLOCK_MS"])
"""


class _Repo:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._git("init", "-q")

    def _git(self, *args: str) -> str:
        argv = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(self.root), *args]
        return subprocess.run(argv, check=True, capture_output=True, text=True).stdout.strip()

    def write(self, source: str, m: int = 4, speedup: float | None = None) -> str:
        """A gemm schema of one workload at m, its kernel for this machine, stored passed at speedup."""
        target = DeviceTarget.current()
        point = _CASE.point.model_copy(update={"axes": _CASE.point.axes | {"m": m}})
        workload = _CASE.contract.workload_for(point, {})
        if speedup is not None:
            workload = workload.model_copy(update={"accuracy": "passed", "speedup": speedup})
        schema = _CASE.contract.narrow(
            _CASE.point,
            workloads=[workload],
            kernel={
                "language": "triton",
                "targets": [{"arch": target.arch, "hardware": target.hardware}],
                "requires": {"triton": ">=0"},
                "entry_point": "kernel.py::run",
                "sources": [{"path": "kernel.py", "content": source}],
            },
        )
        path = f"aklr/ops/gemm/{schema.relative_path()}"
        (self.root / path).parent.mkdir(parents=True, exist_ok=True)
        (self.root / path).write_text(schema.schema_text(), encoding="utf-8")
        return path

    def commit(self) -> None:
        self._git("add", "-A")
        self._git("commit", "-q", "--allow-empty", "-m", "c")

    def start(self, *args: str, **environment: str) -> subprocess.Popen:
        """A validation of the committed tree, running: its report goes to out/. environment: what gemm's
        bench reads."""
        self.commit()
        out = self.root / "out"
        # the run registers gemm's bench, as it does the ones in the bench's _ops/
        site = self.root.parent / "site"
        site.mkdir(exist_ok=True)
        (site / "sitecustomize.py").write_text(_BENCH)
        env = {**os.environ, **environment, "PYTHONPATH": os.pathsep.join((str(site), str(_AKLR)))}
        argv = [sys.executable, "-m", "aklr", "validate", "--stages", "audit,bench", "--out", str(out), *args]
        return subprocess.Popen(
            argv, cwd=self.root, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

    def validate(self, *args: str, **environment: str) -> dict:
        stdout, stderr = self.start(*args, **environment).communicate()
        out = self.root / "out"
        assert (out / "report.json").is_file(), stdout[-3000:] + stderr[-3000:]
        stages = json.loads((out / "report.json").read_text())["stages"]
        assert stages[0]["status"] == "passed", stages[0]
        return stages[1]


@pytest.fixture
def repo(tmp_path: Path) -> _Repo:
    return _Repo(tmp_path)


@pytest.mark.parametrize("isolate", [(), ("--isolate", "gemm")], ids=["thread", "process"])
def test_an_accurate_kernel_is_timed_against_the_baseline(repo: _Repo, isolate: tuple[str, ...]) -> None:
    path = repo.write(_GEMM)
    bench = repo.validate("--devices", "0", *isolate, path)
    (record,) = bench["records"]
    # torch's matmul against aiter's: either side may be faster, and a slower one is still timed
    assert record["versus"] == "baseline"
    assert record["verdict"] in ("passed", "regressed")
    comparison = record["comparison"]
    assert record["versus_name"] == _BASELINE
    assert len(comparison["candidate_ms"]) == len(comparison["opponent_ms"]) == 3  # one per accuracy trial
    assert bench["evaluations"][path]["m=4"]["speedup"] == comparison["speedup"]


@pytest.mark.parametrize("switch", ["ROCPROFILER_REGISTER_FORCE_LOAD", "ROCPROFILER_REGISTER_LIBRARY"])
def test_a_runner_times_under_an_inherited_rocprofiler_registration(repo: _Repo, switch: str) -> None:
    """What a parent whose rocprofiler-sdk configured hands its children: they still register their tracer."""
    value = "1"
    if switch.endswith("LIBRARY"):
        with open("/proc/self/maps") as maps:
            value = next(line.split()[-1] for line in maps if "librocprofiler-sdk.so" in line)
    bench = repo.validate("--devices", "0", repo.write(_GEMM), **{switch: value})
    (record,) = bench["records"]
    assert record["verdict"] in ("passed", "regressed")


@pytest.mark.parametrize(("source", "code"), [(_WRONG, "outputs_mismatch"), (_RAISING, "candidate_raised")])
def test_an_inaccurate_kernel_is_demoted_untimed(repo: _Repo, source: str, code: str) -> None:
    path = repo.write(source)
    bench = repo.validate("--devices", "0", path)
    (record,) = bench["records"]
    assert (record["verdict"], record["is_demoted"], record["comparison"]) == (code, True, None)
    assert record["workload"] == {"axes": {"m": 4}}
    assert [f["workload"] for f in bench["demotions"]] == [{"axes": {"m": 4}}]
    assert [f["code"] for f in bench["findings"]] == ["no_qualifying_workload"]
    assert bench["evaluations"][path]["m=4"] == {"axes": {"m": 4}, "accuracy": "failed"}


def test_a_candidate_is_timed_against_a_different_incumbent(repo: _Repo) -> None:
    repo.write(_OTHER, speedup=2.0)
    repo.commit()
    path = repo.write(_GEMM)
    bench = repo.validate("--devices", "0", "--base", "HEAD~1", path)  # validate commits the candidate
    (record,) = bench["records"]
    assert record["versus"] == "incumbent"
    ratio = record["comparison"]["speedup"]
    assert bench["evaluations"][path]["m=4"]["speedup"] == pytest.approx(ratio * 2.0, abs=1e-3)


def test_an_op_types_own_clock_times_its_workloads(repo: _Repo) -> None:
    bench = repo.validate("--devices", "0", repo.write(_GEMM), AKLR_TEST_CLOCK_MS="2.5")
    (record,) = bench["records"]
    comparison = record["comparison"]
    assert comparison["candidate_ms"] == comparison["opponent_ms"] == [2.5] * 3


def test_a_crash_is_reported_with_its_own_tasks_log(repo: _Repo) -> None:
    repo.write(_NOISY, m=8)  # the costlier task: it runs first, in the process the crashing one reuses
    path = repo.write(_CRASHING)
    bench = repo.validate("--devices", "0")
    (detail,) = [f["detail"] for f in bench["demotions"] if f["schema_path"] == path]
    assert "the crash is here" in detail
    assert "x" * 80 not in detail
    assert "late output from the first task" not in detail


def test_a_killed_validation_leaves_no_runner_behind(repo: _Repo) -> None:
    pid_path = repo.root / "runner.pid"
    # the kernel's process, a runner, names itself and stays busy
    kernel = (
        _DEF + "    import os\n    import pathlib\n    import time\n\n"
        f"    pathlib.Path({str(pid_path)!r}).write_text(str(os.getpid()))\n    time.sleep(600)\n"
    )
    with repo.start("--devices", "0", repo.write(kernel)) as validation:
        try:
            deadline = time.monotonic() + 120
            while not (pid_path.exists() and pid_path.read_text()):
                assert validation.poll() is None, validation.stderr.read()[-3000:]
                assert time.monotonic() < deadline, "runner did not enter the kernel"
                time.sleep(0.1)
            runner = int(pid_path.read_text())
            assert runner != validation.pid
            pidfd = os.pidfd_open(runner)
            try:
                validation.kill()
                validation.wait(timeout=10)
                assert select.select([pidfd], [], [], 10)[0], "runner survived its supervisor"
            finally:
                if not select.select([pidfd], [], [], 0)[0]:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(runner, signal.SIGKILL)
                os.close(pidfd)
        finally:
            validation.kill()


def test_threads_time_on_several_devices_at_once(repo: _Repo) -> None:
    paths = [repo.write(_GEMM + f"# {m}\n", m=m) for m in (4, 8)]
    bench = repo.validate("--devices", "0,1", *paths)
    assert all(r["comparison"] is not None for r in bench["records"])


@pytest.mark.gpu(2)
class TestGroup:
    def test_an_accurate_kernel_is_checked_and_timed_on_two_ranks(self, repo: _Repo) -> None:
        bench = repo.validate("--devices", "0,1", repo.write(_GEMM), AKLR_TEST_RANKS="2")
        (record,) = bench["records"]
        assert record["verdict"] in ("passed", "regressed")
        assert len(record["comparison"]["candidate_ms"]) == 3

    def test_a_kernel_wrong_on_rank_one_fails_the_whole_group_without_hanging(self, repo: _Repo) -> None:
        bench = repo.validate("--devices", "0,1", repo.write(_WRONG_ON_RANK_ONE), AKLR_TEST_RANKS="2")
        (record,) = bench["records"]
        assert record["verdict"] == "outputs_mismatch"
        (finding,) = bench["demotions"]
        assert finding["detail"].startswith("rank 1: ")

    def test_a_group_kernel_is_timed_on_every_rank_against_its_incumbent(self, repo: _Repo) -> None:
        repo.write(_OTHER, speedup=2.0)
        repo.commit()
        path = repo.write(_GEMM)
        bench = repo.validate("--devices", "0,1", "--base", "HEAD~1", path, AKLR_TEST_RANKS="2")
        (record,) = bench["records"]
        assert record["versus"] == "incumbent"
        assert len(record["comparison"]["candidate_ms"]) == 3

    def test_a_seed_draws_the_same_inputs_on_every_device(self) -> None:
        schema = _CASE.schema()
        sampler = InputSampler(schema, schema.workloads[0].axes)
        first, second = (sampler.draw(7, 0, f"cuda:{d}") for d in (0, 1))
        assert all(
            torch.equal(first[n].cpu(), second[n].cpu()) for n in first if isinstance(first[n], torch.Tensor)
        )
