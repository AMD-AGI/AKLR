"""Run AgentX points on a GPU host and capture torch-profiler windows while the client replays its traces.

Per point, in ``<out>/<exp-name>/``: the server (the cookbook's command in the cookbook's image, run with Apptainer),
a smoke request, the InferenceX AgentX client at the point's concurrency, and ``windows`` profiler windows of
``window_s`` seconds each in the client's profiling phase (after its warmup), ``gap_s`` apart. Then the client and
the server stop and the GPUs are checked free before the next point takes them.

Points that need fewer GPUs than the host has run side by side on disjoint, aligned GPU groups.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from . import serving

# The AgentX client (AIPerf) logs "Credit phase start: <phase>" when it switches from its warmup (cache priming) to
# its profiling phase.
PROFILING_STARTED = re.compile(r"(?i)credit phase start:[^\n]*profiling")


@dataclass
class Host:
    """Where and how points run on this host."""

    gpus: list[int]
    out: Path  # host directory holding every run; bound into the container at ``census``
    images: dict[str, str]  # image reference (as in the cookbook) -> local Apptainer sandbox
    paths: dict[str, str]  # repository id -> local checkpoint directory
    infx: Path  # InferenceX checkout (inferencex-e2e's parent)
    agentx_runtime: Path  # AIPERF_RUNTIME_DIR with a bootstrapped venv
    hf_home: Path  # holds the AgentX trace corpora
    census: str = "/census"
    binds: list[str] = field(default_factory=lambda: ["/shared_nfs:/shared_nfs:ro"])
    base_port: int = 31000
    # Changes this host needs to start a point at all: [{"points": <regex on ids>, "env": {...}, "reason": "..."}].
    # Each one applied is written to the point's deviations.json and shown in its report.
    deviations: list = field(default_factory=list)


@dataclass
class Capture:
    windows: int = 3
    window_s: float = 10.0
    gap_s: float = 45.0
    settle_s: float = 60.0  # after the profiling phase starts
    fallback_s: float = 7200.0  # start the windows this long after the client starts if no phase line appears
    ready_timeout_s: float = 7200.0
    deadline: float = 0.0  # unix time; a point still running then stops (its server too), 0 = none


def _post(url: str, body: dict | None = None, timeout: float = 600.0) -> tuple[int, str]:
    data = json.dumps(body).encode() if body is not None else b""
    req = urllib.request.Request(url, data=data, method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode(errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")


def _get(url: str, timeout: float = 10.0) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read().decode(errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, ""
    except (urllib.error.URLError, OSError):
        return 0, ""


class PointRun:
    def __init__(self, host: Host, capture: Capture, point: dict, gpus: list[int], duration: int):
        self.h, self.c, self.p, self.gpus = host, capture, point, gpus
        self.recipe = point["recipe"]
        self.engine = (
            self.recipe["engine"]["type"]
            if isinstance(self.recipe["engine"], dict)
            else self.recipe["engine"]
        )
        self.name = re.sub(r"[^A-Za-z0-9._-]", "_", point["id"])
        self.dir = host.out / self.name
        self.cdir = f"{host.census}/{self.name}"
        self.port = host.base_port + 10 * gpus[0]
        self.url = f"http://127.0.0.1:{self.port}"
        self.duration = duration
        self.server = None
        self.client = None
        self.events = None

    def check_deadline(self) -> None:
        if self.c.deadline and time.time() > self.c.deadline:
            raise RuntimeError("deadline reached")

    def log(self, what: str, **kw) -> None:
        rec = {"t": time.time(), "event": what, **kw}
        self.events.write(json.dumps(rec) + "\n")
        self.events.flush()

    # ---- server -------------------------------------------------------------------------------------------------
    def server_argv(self) -> list[str]:
        argv = serving.server_command(self.recipe, self.h.paths, "127.0.0.1", self.port)
        if self.engine == "vllm":
            # vLLM serves /start_profile only with a profiler configured; this adds no work until a window opens.
            argv += [
                "--profiler-config",
                json.dumps(
                    {
                        "profiler": "torch",
                        "torch_profiler_dir": f"{self.cdir}/traces/_vllm",
                        "torch_profiler_with_stack": False,
                        "torch_profiler_record_shapes": False,
                        "torch_profiler_use_gzip": True,
                        "ignore_frontend": True,
                    }
                ),
            ]
        return argv

    def write_scripts(self) -> list[str]:
        env = serving.server_env(self.recipe)
        applied = [d for d in self.h.deviations if re.search(d["points"], self.p["id"])]
        for d in applied:
            env.update({k: str(v) for k, v in d.get("env", {}).items()})
        if applied:
            (self.dir / "deviations.json").write_text(json.dumps(applied, indent=1))
            self.log("deviations", applied=applied)
        # TMPDIR stays short: vLLM puts its ZMQ IPC sockets there, and an AF_UNIX path is at most 107 bytes.
        env.update(
            ROCR_VISIBLE_DEVICES=",".join(map(str, self.gpus)),
            HOME="/root",
            TMPDIR=f"/tmp/oc{self.port}",
            HF_HOME=f"{self.cdir}/hf",
        )
        lines = ["#!/bin/bash", f"# {self.p['config_key']} {self.p['row']['exp-name']} (op_census)"]
        lines += [f"export {k}={shlex.quote(str(v))}" for k, v in env.items()]
        lines += ['mkdir -p "$TMPDIR"', f"cd {self.cdir}", "exec " + shlex.join(self.server_argv())]
        (self.dir / "server.sh").write_text("\n".join(lines) + "\n")
        image = self.h.images[self.p["row"]["image"]]
        home = self.h.out.parent / "home" / Path(image).name
        home.mkdir(parents=True, exist_ok=True)
        # A --writable sandbox gets no overlay, so every bind destination must already exist in it.
        for dest in [self.h.census, *(b.split(":")[1] for b in self.h.binds)]:
            Path(image, dest.lstrip("/")).mkdir(parents=True, exist_ok=True)
        # No --rocm: it binds the host's ROCm user-space libraries (libdrm, comgr, libnuma, ...) over the image's,
        # which is not the deployed stack (and breaks images with an older glibc). /dev/kfd and /dev/dri are visible
        # without it.
        argv = [
            "apptainer",
            "-s",
            "exec",
            "--cleanenv",
            "--writable",
            "--home",
            f"{home}:/root",
            "--bind",
            f"{self.h.out}:{self.h.census}",
        ]
        for b in self.h.binds:
            argv += ["--bind", b]
        return argv + [image, "bash", f"{self.cdir}/server.sh"]

    def start_server(self) -> None:
        argv = self.write_scripts()
        (self.dir / "apptainer.sh").write_text(shlex.join(argv) + "\n")
        log = open(self.dir / "server.log", "ab")
        env = {k: v for k, v in os.environ.items() if k not in ("LD_LIBRARY_PATH", "PYTHONPATH")}
        self.server = subprocess.Popen(
            argv,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            env=env,
        )
        self.log("server_started", pid=self.server.pid, gpus=self.gpus, port=self.port)

    # vLLM logs this when the weights are on the GPUs, before it allocates its KV cache and any offload pool.
    WEIGHTS_LOADED = re.compile(r"Model loading took|Loading weights took|Load weight end")

    def prefetch_weights(self) -> None:
        """Read every checkpoint file once, 32 at a time, into the host page cache.

        A loader reading shards one by one from NFS took 26 min for Qwen3.8's 117 GB; 32 parallel sequential readers
        take seconds to minutes, and a file already cached costs nothing. Only the load gets faster."""
        from concurrent.futures import ThreadPoolExecutor

        files = []
        for repo in serving.repositories(self.recipe):
            for root, dirs, names in os.walk(self.h.paths.get(repo, "/nonexistent"), followlinks=True):
                dirs[:] = [
                    d for d in dirs if not d.startswith(".")
                ]  # e.g. .cache/huggingface: not ours to read
                files += [os.path.join(root, f) for f in names if not f.startswith(".")]

        def read(path):
            try:
                with open(path, "rb", buffering=0) as f:
                    while f.read(1 << 24):
                        pass
            except OSError:  # an unreadable side file is the loader's business, not the prefetch's
                pass

        t0 = time.time()
        with ThreadPoolExecutor(32) as pool:
            list(pool.map(read, files))
        self.log("weights_prefetched", files=len(files), s=round(time.time() - t0, 1))

    def drop_weight_cache(self) -> None:
        """Evict the checkpoint's files from the host page cache (posix_fadvise DONTNEED).

        A host KV-offload pool allocated with transparent huge pages stalls in memory compaction when a terabyte of
        page cache fragments host memory; it does not change any kernel the point runs."""
        n = 0
        for repo in serving.repositories(self.recipe):
            for root, dirs, files in os.walk(self.h.paths.get(repo, "/nonexistent"), followlinks=True):
                dirs[:] = [d for d in dirs if not d.startswith(".")]
                for f in files:
                    try:
                        fd = os.open(os.path.join(root, f), os.O_RDONLY)
                        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                        os.close(fd)
                        n += 1
                    except OSError:
                        pass
        self.log("weight_page_cache_dropped", files=n)

    def wait_ready(self) -> None:
        hc = self.recipe.get("health_check") or {}
        limit = max(self.c.ready_timeout_s, hc.get("max_attempts", 0) * hc.get("interval_seconds", 0))
        t0 = time.time()
        served = self.recipe["roles"]["agg"]["args"].get("served-model-name") or serving.checkpoint_id(
            self.recipe
        )
        offload = self.engine == "vllm" and self.p["row"].get("kv-offloading", "none") != "none"
        dropped = False
        while time.time() - t0 < limit:
            self.check_deadline()
            if self.server.poll() is not None:
                raise RuntimeError(f"server exited with {self.server.returncode} before it was ready")
            if (
                offload
                and not dropped
                and self.WEIGHTS_LOADED.search((self.dir / "server.log").read_text(errors="replace"))
            ):
                self.drop_weight_cache()
                dropped = True
            code, body = _get(f"{self.url}/v1/models")
            if code == 200 and served in body:
                self.log("server_ready", load_s=time.time() - t0)
                return
            time.sleep(10)
        raise RuntimeError(f"server not ready after {limit:.0f} s")

    def smoke(self) -> None:
        served = self.recipe["roles"]["agg"]["args"].get("served-model-name") or serving.checkpoint_id(
            self.recipe
        )
        code, body = _post(
            f"{self.url}/v1/chat/completions",
            {
                "model": served,
                "messages": [{"role": "user", "content": "Count from 1 to 5."}],
                "max_tokens": 32,
            },
        )
        self.log("smoke", status=code, body=body[:400])
        if code != 200:
            raise RuntimeError(f"smoke request failed: {code} {body[:300]}")

    # ---- client -------------------------------------------------------------------------------------------------
    def start_client(self) -> None:
        tokenizer = self.h.paths[serving.checkpoint_id(self.recipe)]
        env = serving.client_env(self.p, self.url, str(self.dir / "client"), self.duration, tokenizer)
        env.update(
            PATH=os.environ.get("PATH", "/usr/bin:/bin"),
            HOME=str(self.dir / "client_home"),
            # Not offline: AIPerf loads the trace corpus through `datasets`, which in offline mode does not
            # find what `hf download` put in the hub cache. InferenceX runs the client online too.
            HF_HOME=str(self.h.hf_home),
            AIPERF_RUNTIME_DIR=str(self.h.agentx_runtime),
            PYTHONPATH=str(self.h.infx / "inferencex-e2e"),
            PORT=str(self.port),
        )
        (self.dir / "client_env.json").write_text(json.dumps(env, indent=1))
        python = str(self.h.agentx_runtime / "venv/bin/python")
        log = open(self.dir / "client.log", "ab")
        self.client = subprocess.Popen(
            [python, "-m", "infx.bench", "agentic"],
            cwd=self.h.infx / "inferencex-e2e",
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        self.log("client_started", pid=self.client.pid, duration=self.duration)

    def wait_profiling_phase(self) -> None:
        t0 = time.time()
        while time.time() - t0 < self.c.fallback_s:
            self.check_deadline()
            if self.client.poll() is not None:
                raise RuntimeError(f"client exited with {self.client.returncode} before profiling started")
            # AIPerf may log to its artifact directory rather than to the console.
            for path in [self.dir / "client.log", *(self.dir / "client").rglob("*.log")]:
                text = path.read_text(errors="replace") if path.is_file() else ""
                m = PROFILING_STARTED.search(text)
                if m:
                    self.log(
                        "profiling_phase", file=str(path), line=text[max(0, m.start() - 80) : m.end() + 80]
                    )
                    time.sleep(self.c.settle_s)
                    return
            time.sleep(5)
        self.log("profiling_phase_fallback", after_s=self.c.fallback_s)

    # ---- windows ------------------------------------------------------------------------------------------------
    def window(self, i: int) -> None:
        wdir = f"{self.cdir}/traces/w{i}"
        (self.dir / "traces" / f"w{i}").mkdir(parents=True, exist_ok=True)
        if self.engine == "sglang":
            body = {
                "output_dir": wdir,
                "activities": ["CPU", "GPU"],
                "with_stack": False,
                "record_shapes": False,
                "profile_prefix": f"w{i}",
                "detailed_annotations": True,
            }
            code, text = _post(f"{self.url}/start_profile", body)
        else:
            code, text = _post(f"{self.url}/start_profile")
        self.log("window_start", window=i, status=code, body=text[:200], load=self.load())
        if code != 200:
            raise RuntimeError(f"start_profile failed: {code} {text[:300]}")
        time.sleep(self.c.window_s)
        t_stop = time.time()
        load = self.load()
        code, text = _post(f"{self.url}/stop_profile", timeout=1800)
        self.log(
            "window_stop", window=i, status=code, body=text[:200], stop_s=time.time() - t_stop, load=load
        )
        if self.engine == "vllm":
            self._collect_vllm(i)
        self._wait_files(self.dir / "traces" / f"w{i}")

    LOAD_METRIC = re.compile(
        r"^(sglang:num_running_reqs|sglang:num_queue_reqs|vllm:num_requests_running|"
        r"vllm:num_requests_waiting)\{[^}]*\}\s+([0-9.eE+-]+)",
        re.M,
    )

    def load(self) -> dict:
        """Requests the server is running and queueing now (its Prometheus gauges; summed over engines)."""
        code, text = _get(f"{self.url}/metrics")
        out = {}
        for name, value in self.LOAD_METRIC.findall(text if code == 200 else ""):
            out[name] = out.get(name, 0.0) + float(value)
        return out

    def _collect_vllm(self, i: int) -> None:
        src = self.dir / "traces" / "_vllm"
        dst = self.dir / "traces" / f"w{i}"
        deadline = time.time() + 900
        while time.time() < deadline:
            files = sorted(src.glob("*.json*")) if src.exists() else []
            if len(files) >= len(self.gpus) and self._stable(files):
                for f in files:
                    f.rename(dst / f.name)
                return
            time.sleep(5)
        raise RuntimeError(f"vLLM traces of window {i} did not appear in {src}")

    @staticmethod
    def _stable(files: list[Path]) -> bool:
        a = [f.stat().st_size for f in files]
        time.sleep(3)
        return a == [f.stat().st_size for f in files]

    def _wait_files(self, d: Path) -> None:
        deadline = time.time() + 900
        while time.time() < deadline:
            files = sorted(d.glob("*.json*"))
            if len(files) >= len(self.gpus) and self._stable(files):
                self.log(
                    "window_files",
                    dir=str(d),
                    files=[f.name for f in files],
                    mib=[round(f.stat().st_size / 2**20, 1) for f in files],
                )
                return
            time.sleep(5)
        raise RuntimeError(f"traces did not appear in {d}")

    # ---- teardown -----------------------------------------------------------------------------------------------
    def stop(self) -> None:
        for proc, what in ((self.client, "client"), (self.server, "server")):
            if proc is None or proc.poll() is not None:
                continue
            try:
                os.killpg(proc.pid, signal.SIGINT)
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=60)
            except ProcessLookupError:
                pass
            self.log(f"{what}_stopped", rc=proc.returncode)
        wait_gpus_free(self.gpus)
        self.log("gpus_free", gpus=self.gpus)

    def run(self, capture_after_client: bool = True) -> dict:
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "tmp").mkdir(exist_ok=True)
        (self.dir / "point.json").write_text(json.dumps(self.p, indent=1))
        self.events = open(self.dir / "events.jsonl", "a")
        status = {"point": self.name, "gpus": self.gpus}
        try:
            # A point never shares its GPUs with what a previous one left behind.
            wait_gpus_free(self.gpus, timeout=300)
            self.prefetch_weights()
            self.start_server()
            self.wait_ready()
            self.smoke()
            self.start_client()
            self.wait_profiling_phase()
            for i in range(self.c.windows):
                if i:
                    time.sleep(self.c.gap_s)
                self.check_deadline()
                self.window(i)
            status["ok"] = True
        except Exception as e:  # noqa: BLE001  -- recorded, the next point still runs
            status.update(ok=False, error=f"{type(e).__name__}: {e}")
            self.log("error", error=status["error"])
        finally:
            try:
                self.stop()
            except Exception as e:  # noqa: BLE001  -- a failed teardown must still leave a status behind
                status.update(ok=False, error=status.get("error") or f"teardown: {type(e).__name__}: {e}")
                self.log("error", error=f"teardown: {type(e).__name__}: {e}")
            (self.dir / "status.json").write_text(json.dumps(status, indent=1))
            self.events.close()
        return status


AMD_SMI_FALLBACKS = ("/opt/rocm/bin/amd-smi", "/opt/venv/bin/amd-smi")


def amd_smi() -> str:
    """amd-smi from PATH, else where ROCm installs it: a driver started with a bare PATH must still see the GPUs."""
    for c in (shutil.which("amd-smi"), *AMD_SMI_FALLBACKS):
        if c and os.access(c, os.X_OK):
            return c
    raise RuntimeError("amd-smi not found on PATH or in " + ", ".join(AMD_SMI_FALLBACKS))


def gpu_used_mib(gpus: list[int]) -> dict[int, float]:
    out = subprocess.run(
        [amd_smi(), "metric", "--mem-usage", "--json"], capture_output=True, text=True, check=False
    )
    used = {}
    try:
        data = json.loads(out.stdout)
        for rec in data.get("gpu_data", []) if isinstance(data, dict) else data:
            g = rec.get("gpu")
            if g in gpus:
                mu = rec.get("mem_usage", {}).get("used_vram", {})
                used[g] = float(mu.get("value", 0)) if isinstance(mu, dict) else float(mu)
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return used


def wait_gpus_free(gpus: list[int], limit_mib: float = 2048, timeout: float = 600) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        used = gpu_used_mib(gpus)
        if used and all(v < limit_mib for v in used.values()):
            return
        time.sleep(10)
    raise RuntimeError(f"GPUs {gpus} still hold memory after {timeout:.0f} s: {gpu_used_mib(gpus)}")


def groups(n: int, gpus: list[int]) -> list[list[int]]:
    """Aligned groups of ``n`` GPUs: 4 -> [0..3], [4..7]."""
    return [gpus[i : i + n] for i in range(0, len(gpus) - n + 1, n)]


def run_points(
    host: Host, capture: Capture, points: list[dict], duration: int, max_parallel: int = 8
) -> list[dict]:
    """Run ``points`` in order, each on the first free aligned group of the GPUs it needs, at most ``max_parallel``
    at once (servers and clients share the host's CPUs, which a launch-bound decode needs)."""
    free = set(host.gpus)
    running = [0]
    lock = threading.Condition()
    results = []
    threads = []

    def work(p, g):
        # Whatever happens to this point, its GPUs go back to the pool (the next point checks they are free first).
        res = {"point": p.get("id"), "gpus": g, "ok": False, "error": "runner thread died"}
        try:
            res = PointRun(host, capture, p, g, duration).run()
        finally:
            with lock:
                results.append(res)
                free.update(g)
                running[0] -= 1
                lock.notify_all()

    for p in points:
        need = p["recipe"]["roles"]["agg"]["gpus"]
        with lock:
            while True:
                cand = [g for g in groups(need, host.gpus) if set(g) <= free]
                if cand and running[0] < max_parallel:
                    g = cand[0]
                    free.difference_update(g)
                    running[0] += 1
                    break
                lock.wait()
        t = threading.Thread(target=work, args=(p, g), daemon=False)
        t.start()
        threads.append(t)
        time.sleep(5)
    for t in threads:
        t.join()
    return results
