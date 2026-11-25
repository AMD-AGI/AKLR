# SPDX-License-Identifier: MIT
"""Start, poll and stop a serving process and its client, and wait for idle GPUs."""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request

INFRA_ENV = (
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "TMPDIR",
    "LD_LIBRARY_PATH",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "HF_TOKEN",
    "HF_HOME",
    "HF_HUB_CACHE",
    "HF_HUB_OFFLINE",
    "HF_DATASETS_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "XDG_CACHE_HOME",
    "PIP_CACHE_DIR",
    "UV_CACHE_DIR",
    "AITER_JIT_DIR",
    "GPU_ARCHS",
    "FLYDSL_RUNTIME_CACHE_DIR",
    "FLYDSL_AUTOTUNE_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "AKLR_KERNEL_CACHE_DIR",
)


def execution_env(image_defaults: dict[str, str] | None = None) -> dict[str, str]:
    """Use verified image defaults only when explicitly requested by the server."""
    return {
        **(image_defaults or {}),
        **{name: os.environ[name] for name in INFRA_ENV if name in os.environ},
    }


def http_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return response.status < 400
    except (urllib.error.URLError, OSError):
        return False


def wait_ready(base_url: str, server: subprocess.Popen, timeout_s: float, interval_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"server exited with code {server.returncode} before ready")
        if http_ok(f"{base_url}/health") and http_ok(f"{base_url}/v1/models"):
            return
        time.sleep(min(interval_s, max(0, deadline - time.monotonic())))
    raise RuntimeError(f"server not ready after {timeout_s:.0f} s")


class CleanupError(RuntimeError):
    """A child process could not be stopped or its GPUs freed; the sweep must abort."""


GPU_IDLE_TIMEOUT_S = 300

GPU_IDLE_POLL_S = 5


def vram_used_percent() -> dict[int, int]:
    """Whole-percent VRAM use of every GPU amd-smi lists, as rocm-smi's VRAM% reports it."""
    try:
        done = subprocess.run(
            [shutil.which("amd-smi") or "/opt/rocm/bin/amd-smi", "metric", "--mem-usage", "--json"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise CleanupError(f"amd-smi could not run: {type(error).__name__}: {error}") from error
    if done.returncode:
        raise CleanupError(f"amd-smi exited with code {done.returncode}: {done.stderr.strip()[:200]}")
    text = done.stdout
    starts = [index for index in (text.find("["), text.find("{")) if index >= 0]
    try:
        data = json.loads(text[min(starts) :]) if starts else None
        gpus = data.get("gpu_data") if isinstance(data, dict) else data
        usage = {}
        for gpu in gpus:
            memory = gpu["mem_usage"]
            used, total = float(memory["used_vram"]["value"]), float(memory["total_vram"]["value"])
            usage[int(gpu["gpu"])] = int(used * 100 // total)
    except (ValueError, KeyError, TypeError, ZeroDivisionError) as error:
        raise CleanupError(f"amd-smi memory usage is unreadable: {type(error).__name__}: {error}") from error
    if not usage:
        raise CleanupError("amd-smi listed no GPUs")
    return usage


def wait_gpus_idle(timeout_s: float = GPU_IDLE_TIMEOUT_S, poll_s: float = GPU_IDLE_POLL_S) -> float:
    """Wait until every GPU's VRAM use reads 0%; servers such as HiCache free memory late."""
    began = time.monotonic()
    while True:
        busy = {gpu: percent for gpu, percent in sorted(vram_used_percent().items()) if percent}
        waited = time.monotonic() - began
        if not busy:
            return waited
        if waited >= timeout_s:
            listed = ", ".join(f"GPU{gpu} {percent}%" for gpu, percent in busy.items())
            raise CleanupError(f"GPU VRAM still in use after {timeout_s:.0f} s: {listed}")
        time.sleep(min(poll_s, max(0, timeout_s - waited)))


def stop(server: subprocess.Popen, grace_s: float = 120) -> None:
    """Stop the process group even if its leader has exited."""
    deadline = time.monotonic() + grace_s
    try:
        try:
            os.killpg(server.pid, signal.SIGTERM)
            while time.monotonic() < deadline:
                server.poll()
                os.killpg(server.pid, 0)
                time.sleep(min(0.1, max(0, deadline - time.monotonic())))
            os.killpg(server.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        server.wait()
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        raise CleanupError(
            f"Could not stop process group {server.pid}: {type(error).__name__}: {error}"
        ) from error


def run_client(command: list[str], *, timeout: float, **kwargs) -> int:
    """Stop the client's process group on exit or timeout."""
    process = subprocess.Popen(command, start_new_session=True, **kwargs)
    try:
        return process.wait(timeout=timeout)
    finally:
        stop(process)


def available_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def port_free(port: int) -> bool:
    with socket.socket() as probe:
        return probe.connect_ex(("127.0.0.1", port)) != 0
