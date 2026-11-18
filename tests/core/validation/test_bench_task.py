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
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from aklr.core.platform.device import DeviceTarget
from aklr.core.schema import (
    Build,
    Evaluation,
    Kernel,
    Schema,
    SourceFile,
    Speedup,
    Workload,
    save_json_file,
)
from tests.core.validation.toy_group import ToyClockBench, toy_group_schema
from tests.op_sample import GEMM_KERNEL, GEMM_SAMPLE

pytestmark = pytest.mark.gpu

_AKLR = Path(__file__).resolve().parents[3]
_DOC = '"""A test kernel."""\n\n\n'
_WRONG = "def run(a, b, bias, out):\n    out.zero_()\n"
_RAISING = "def run(a, b, bias, out):\n    raise RuntimeError('broken kernel')\n"
_OTHER = GEMM_KERNEL + "# another kernel\n"
_NOISY = (
    # Keep a duplicate of stderr, as a subprocess from this task would.
    "def run(a, b, bias, out):\n    import os\n    import sys\n\n"
    "    sys.previous_task_stderr = os.dup(2)\n"
    "    sys.stdout.write('x' * (6 << 20))\n    sys.stdout.flush()\n    out.zero_()\n"
)
_CRASHING = (
    "def run(a, b, bias, out):\n    import os\n    import sys\n\n"
    "    os.write(sys.previous_task_stderr, b'late output from the first task')\n"
    "    sys.stdout.write('the crash is here')\n    sys.stdout.flush()\n    os._exit(7)\n"
)


_GROUP_SUM = (
    "def run(x, out):\n    import torch.distributed as dist\n\n    out.copy_(x)\n    dist.all_reduce(out)\n"
)
_GROUP_WRONG_ON_RANK_ONE = _GROUP_SUM + "    if dist.get_rank() == 1:\n        out.zero_()\n"


class _Repo:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._git("init", "-q")

    def _git(self, *args: str) -> str:
        argv = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(self.root), *args]
        return subprocess.run(argv, check=True, capture_output=True, text=True).stdout.strip()

    def write(self, source: str, m: int = 4, speedup: float | None = None) -> str:
        evaluations = []
        if speedup is not None:
            measured = {"aiter": Speedup("e" * 40, speedup)}
            evaluations = [
                Evaluation(hardware=str(DeviceTarget.current()), accuracy="passed", speedup=measured)
            ]
        schema: Schema = GEMM_SAMPLE.served(_DOC + source, evaluations=evaluations)
        schema.workloads[0].axes["M"] = m
        schema = replace(schema, kernel=schema.kernel.with_unique_name())  # kernel names are unique
        path = f"aklr/plugins/aiter/gemm/impls/triton/{schema.name}/{schema.uuid}.json"
        save_json_file(schema, self.root / path)
        return path

    def commit(self) -> None:
        self._git("add", "-A")
        self._git("commit", "-q", "--allow-empty", "-m", "c")

    def write_group(
        self, source: str, speedup: float | None = None, world: int = 2, op_type: str = "toy_group"
    ) -> str:
        """A toy group op on world ranks: each rank's out is the sum of every rank's x."""
        evaluations = []
        if speedup is not None:
            measured = {"aiter": Speedup("e" * 40, speedup)}
            evaluations = [
                Evaluation(hardware=str(DeviceTarget.current()), accuracy="passed", speedup=measured)
            ]
        kernel = Kernel(
            name="toy_group",
            author="tests",
            build=Build(
                language="triton",
                target_hardware=[DeviceTarget.current().arch],
                entry_point="kernel.py::run",
                requires={"triton": ">=0"},
            ),
            sources=[SourceFile(path="kernel.py", content=_DOC + source)],
        )
        schema = toy_group_schema(
            kernel.with_unique_name(), [Workload(axes={"n": 4096}, evaluations=evaluations)], world, op_type
        )
        path = f"aklr/plugins/aiter/{op_type}/impls/triton/{schema.name}/{schema.uuid}.json"
        save_json_file(schema, self.root / path)
        return path

    def start(self, *args: str) -> subprocess.Popen:
        """A validation of the committed tree, running: its report goes to out/."""
        self.commit()
        out = self.root / "out"
        # the run registers the toy ops' benches, as it does the ones in the bench's _ops/
        site = self.root.parent / "site"
        site.mkdir(exist_ok=True)
        (site / "sitecustomize.py").write_text("import tests.core.validation.toy_group\n")
        env = {**os.environ, "PYTHONPATH": os.pathsep.join((str(site), str(_AKLR)))}
        argv = [sys.executable, "-m", "aklr", "validate", "--stages", "audit,bench", "--out", str(out), *args]
        return subprocess.Popen(
            argv, cwd=self.root, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

    def validate(self, *args: str) -> dict:
        stdout, stderr = self.start(*args).communicate()
        out = self.root / "out"
        assert (out / "report.json").is_file(), stdout[-3000:] + stderr[-3000:]
        stages = json.loads((out / "report.json").read_text())["stages"]
        assert stages[0]["status"] == "passed", stages[0]
        return stages[1]


@pytest.fixture
def repo(tmp_path: Path) -> _Repo:
    return _Repo(tmp_path)


@pytest.mark.parametrize("isolate", [(), ("--isolate", "gemm_bf16_nt")], ids=["thread", "process"])
def test_an_accurate_kernel_is_timed_against_the_baseline(repo: _Repo, isolate: tuple[str, ...]) -> None:
    path = repo.write(GEMM_KERNEL)
    bench = repo.validate("--devices", "0", *isolate, path)
    (record,) = bench["records"]
    # torch's matmul against aiter's: either side may be faster, and a slower one is still timed
    assert record["versus"] == "baseline"
    assert record["verdict"] in ("passed", "regressed")
    comparison = record["comparison"]
    assert record["versus_name"] == "aiter"
    assert len(comparison["candidate_ms"]) == len(comparison["opponent_ms"]) == 3  # one per accuracy trial
    speedup = bench["evaluations"][path]["M=4"]["speedup"]["aiter"]["value"]
    assert speedup == comparison["speedup"]


@pytest.mark.parametrize(("source", "code"), [(_WRONG, "outputs_mismatch"), (_RAISING, "candidate_raised")])
def test_an_inaccurate_kernel_is_demoted_untimed(repo: _Repo, source: str, code: str) -> None:
    path = repo.write(source)
    bench = repo.validate("--devices", "0", path)
    (record,) = bench["records"]
    assert (record["verdict"], record["is_demoted"], record["comparison"]) == (code, True, None)
    assert record["workload"] == {"axes": {"M": 4}}
    assert [f["workload"] for f in bench["demotions"]] == [{"axes": {"M": 4}}]
    assert [f["code"] for f in bench["findings"]] == ["no_admitted_workload"]
    assert bench["evaluations"][path]["M=4"] == {
        "hardware": str(DeviceTarget.current()),
        "accuracy": "failed",
        "speedup": {},
        "microseconds": None,
    }


def test_a_candidate_is_timed_against_a_different_incumbent(repo: _Repo) -> None:
    repo.write(_OTHER, speedup=2.0)
    repo.commit()
    path = repo.write(GEMM_KERNEL)
    bench = repo.validate("--devices", "0", "--base", "HEAD~1", path)  # validate commits the candidate
    (record,) = bench["records"]
    assert record["versus"] == "incumbent"
    ratio = record["comparison"]["speedup"]
    assert bench["evaluations"][path]["M=4"]["speedup"]["aiter"]["value"] == pytest.approx(
        ratio * 2.0, abs=1e-3
    )


def test_an_op_types_own_clock_times_its_workloads(repo: _Repo) -> None:
    repo.write_group(_GROUP_SUM + "# another kernel\n", speedup=2.0, world=1, op_type="toy_clock")
    repo.commit()
    path = repo.write_group(_GROUP_SUM, world=1, op_type="toy_clock")
    bench = repo.validate("--devices", "0", "--base", "HEAD~1", path)
    (record,) = bench["records"]
    comparison = record["comparison"]
    assert comparison["candidate_ms"] == comparison["opponent_ms"] == [ToyClockBench.CLOCK_MS] * 3


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
        "def run(a, b, bias, out):\n    import os\n    import pathlib\n    import time\n\n"
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
    paths = [repo.write(GEMM_KERNEL + f"# {m}\n", m=m) for m in (4, 8)]
    bench = repo.validate("--devices", "0,1", *paths)
    assert all(r["comparison"] is not None for r in bench["records"])


@pytest.mark.gpu(2)
class TestGroup:
    def test_an_accurate_group_kernel_passes_on_two_ranks(self, repo: _Repo) -> None:
        path = repo.write_group(_GROUP_SUM)
        bench = repo.validate("--devices", "0,1", path)
        (record,) = bench["records"]
        assert (record["verdict"], record["comparison"]) == ("passed", None)  # no Baseline: accuracy alone

    def test_a_kernel_wrong_on_rank_one_fails_the_whole_group_without_hanging(self, repo: _Repo) -> None:
        path = repo.write_group(_GROUP_WRONG_ON_RANK_ONE)
        bench = repo.validate("--devices", "0,1", path)
        (record,) = bench["records"]
        assert record["verdict"] == "outputs_mismatch"
        (finding,) = bench["demotions"]
        assert finding["detail"].startswith("rank 1: ")

    def test_a_group_kernel_is_timed_on_every_rank_against_its_incumbent(self, repo: _Repo) -> None:
        repo.write_group(_GROUP_SUM + "# another kernel\n", speedup=2.0)
        repo.commit()
        path = repo.write_group(_GROUP_SUM)
        bench = repo.validate("--devices", "0,1", "--base", "HEAD~1", path)
        (record,) = bench["records"]
        assert record["versus"] == "incumbent"
        assert len(record["comparison"]["candidate_ms"]) == 3

    def test_a_seed_draws_the_same_inputs_on_every_device(self) -> None:
        case = GEMM_SAMPLE
        first, second = (case.call(device=f"cuda:{d}", trial=7) for d in (0, 1))
        assert all(
            torch.equal(first[n].cpu(), second[n].cpu()) for n in first if isinstance(first[n], torch.Tensor)
        )
