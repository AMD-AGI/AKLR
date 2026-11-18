"""runner teardown and GPU checks, with the GPU queries and the point's stages replaced: no GPU, no process started.

A driver started with a bare PATH (no /opt/rocm/bin) once killed the runner thread in its GPU-free check after a
complete capture: no status.json, the point's GPUs never returned to the pool, and the queue hung.
"""

import json
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from op_census import runner  # noqa: E402

POINT = {"id": "m.p_tp2_conc4", "recipe": {"engine": "sglang", "roles": {"agg": {"gpus": 2}}}}


def host(tmp_path):
    return runner.Host(
        gpus=[0, 1, 2, 3],
        out=tmp_path,
        images={},
        paths={},
        infx=tmp_path,
        agentx_runtime=tmp_path,
        hf_home=tmp_path,
    )


def test_amd_smi_found_off_path(monkeypatch, tmp_path):
    exe = tmp_path / "amd-smi"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    monkeypatch.setattr(runner.shutil, "which", lambda name: None)
    monkeypatch.setattr(runner, "AMD_SMI_FALLBACKS", ("/nonexistent/amd-smi", str(exe)))
    assert runner.amd_smi() == str(exe)


def test_amd_smi_missing_is_a_clear_error(monkeypatch):
    monkeypatch.setattr(runner.shutil, "which", lambda name: None)
    monkeypatch.setattr(runner, "AMD_SMI_FALLBACKS", ("/nonexistent/amd-smi",))
    with pytest.raises(RuntimeError, match="amd-smi not found"):
        runner.amd_smi()


def stub_stages(monkeypatch, fail_in=None):
    """Every stage of a point a no-op; ``fail_in`` names one that raises."""
    for name in (
        "prefetch_weights",
        "start_server",
        "wait_ready",
        "smoke",
        "start_client",
        "wait_profiling_phase",
        "window",
    ):

        def stage(self, *a, _n=name):
            if _n == fail_in:
                raise RuntimeError(f"{_n} failed")

        monkeypatch.setattr(runner.PointRun, name, stage)


def test_failed_teardown_still_writes_status(monkeypatch, tmp_path):
    stub_stages(monkeypatch)
    free = iter([True])  # free at the start, then amd-smi is gone

    def wait(gpus, limit_mib=2048, timeout=600):
        if not next(free, False):
            raise FileNotFoundError("amd-smi")

    monkeypatch.setattr(runner, "wait_gpus_free", wait)
    cap = runner.Capture(windows=2, gap_s=0)
    st = runner.PointRun(host(tmp_path), cap, POINT, [0, 1], 60).run()
    assert st["ok"] is False and st["error"].startswith("teardown: FileNotFoundError")
    saved = json.loads((tmp_path / "m.p_tp2_conc4" / "status.json").read_text())
    assert saved == st


def test_point_does_not_start_on_busy_gpus(monkeypatch, tmp_path):
    stub_stages(monkeypatch)
    started = []
    monkeypatch.setattr(runner.PointRun, "start_server", lambda self: started.append(1))
    calls = []

    def wait(gpus, limit_mib=2048, timeout=600):
        calls.append(timeout)
        if len(calls) == 1:
            raise RuntimeError(f"GPUs {gpus} still hold memory")

    monkeypatch.setattr(runner, "wait_gpus_free", wait)
    st = runner.PointRun(host(tmp_path), runner.Capture(windows=1), POINT, [0, 1], 60).run()
    assert st["ok"] is False and "still hold memory" in st["error"] and not started


def test_queue_goes_on_after_a_point_dies(monkeypatch, tmp_path):
    """A point whose run() raises gives its GPUs back; the points after it still run."""
    ran = []

    def run(self, capture_after_client=True):
        ran.append(self.p["id"])
        if self.p["id"] == "a":
            raise RuntimeError("boom")
        return {"point": self.p["id"], "gpus": self.gpus, "ok": True}

    monkeypatch.setattr(runner.PointRun, "run", run)
    monkeypatch.setattr(runner.time, "sleep", lambda s: None)
    monkeypatch.setattr(threading, "excepthook", lambda args: None)  # the dying thread's traceback
    h = host(tmp_path)
    h.gpus = [0, 1]
    pts = [dict(POINT, id="a"), dict(POINT, id="b"), dict(POINT, id="c")]
    res = runner.run_points(h, runner.Capture(), pts, 60, max_parallel=1)
    assert ran == ["a", "b", "c"]
    assert {r["point"]: r["ok"] for r in res} == {"a": False, "b": True, "c": True}
