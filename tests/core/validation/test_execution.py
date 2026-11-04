"""The bench's execution layer: process cleanup and runner failures."""

import subprocess
import sys
import time
from pathlib import Path

import pytest

from aklr.core.validation.stages._bench._evaluation import CandidateResult
from aklr.core.validation.stages._bench._execution import (
    ExecutionError,
    Executor,
    ProcessGroup,
    caused_by_the_task,
)
from aklr.core.validation.stages._report._finding import FindingCode


def test_killing_a_group_kills_its_descendants_own_groups(tmp_path) -> None:
    """A runner whose task started a nested runner (its own session, as the bench trial's
    candidate) takes the nested runner's children down with it."""
    pid_file = tmp_path / "grandchild.pid"
    grandchild = "import time; time.sleep(60)"
    nested = (
        "import os, subprocess, sys, time; "
        f"p = subprocess.Popen([sys.executable, '-c', {grandchild!r}]); "
        f"open({str(pid_file)!r}, 'w').write(str(p.pid)); time.sleep(60)"
    )
    outer = f"import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', {nested!r}], start_new_session=True); time.sleep(60)"
    runner = subprocess.Popen([sys.executable, "-c", outer], start_new_session=True)
    deadline = time.monotonic() + 10
    while not pid_file.exists() or not pid_file.read_text():
        assert time.monotonic() < deadline
        time.sleep(0.01)
    grandchild_pid = int(pid_file.read_text())

    assert ProcessGroup(runner.pid).kill()
    runner.wait(timeout=5)
    stat = Path(f"/proc/{grandchild_pid}/stat")
    deadline = time.monotonic() + 5
    # gone, or a zombie its new parent has yet to reap
    while stat.exists() and stat.read_text().rpartition(")")[2].split()[0] != "Z":
        assert time.monotonic() < deadline, "the nested runner's child outlived the kill"
        time.sleep(0.01)


def test_a_command_the_runner_cannot_receive_is_the_platforms_failure() -> None:
    """Receiving the task's command (importing the tensors lent with it) precedes the task:
    its failure is the platform's, not the candidate's crash."""
    with Executor.for_devices(["cpu"]) as executor:
        future = executor.submit(_never_runs, timeout_seconds=60, value=_Unloadable())
        with pytest.raises(ExecutionError) as failure:
            future.result()
    assert not caused_by_the_task(failure.value)
    (finding,) = CandidateResult.of_failure(failure.value).findings
    assert finding.code == FindingCode.PLATFORM_FAILURE


def _refuse() -> None:
    raise RuntimeError("the lent tensor cannot be imported")


class _Unloadable:
    def __reduce__(self):
        return _refuse, ()


def _never_runs(context, value) -> None:
    raise AssertionError("the task ran")
