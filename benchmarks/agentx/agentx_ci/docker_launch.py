# SPDX-License-Identifier: MIT
"""Verify the image and manage the sweep container."""

from __future__ import annotations

import codecs
import json
import os
import selectors
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from agentx_ci.preflight import pinned_image, verify_image
from agentx_ci.recipe_resolve import selected_points
from agentx_ci.utils import REPO_ROOT, begin_attempt, pending_results, read_json, utc_now, write_json

CONTAINER_COMMAND = r"""
set -euo pipefail
trap 'chown -R "$HOST_IDS" /agentx/out /agentx/cache /agentx/work' EXIT
mkdir /aklr
tar -C /src --exclude='./.git' -cf - . | tar -C /aklr -xf -
printf 'gitdir: %s\n' "$SOURCE_GIT_DIR" > /aklr/.git
cd /aklr
scripts/agentx/install_aklr.sh /agentx/cache 2>&1 | tee /agentx/out/install.log
set -a; . /agentx/cache/aklr.env; set +a
cd benchmarks/agentx
python3 -m agentx_bench sweep --in-container --serving /agentx/serving.json \
  --work /agentx/work --out /agentx/out "$@"
"""


class Cancelled(Exception):
    pass


class Launcher:
    def __init__(self, log):
        self.log = log
        self.signum = None

    def on_signal(self, signum, _frame):
        self.signum = self.signum or signum

    def check_cancelled(self):
        if self.signum is not None:
            raise Cancelled(f"Sweep cancelled by {signal.Signals(self.signum).name}")

    def note(self, text):
        self.log.write((text + "\n").encode())
        self.log.flush()

    def command(self, argv, *, stream=False, check=True, timeout=60, interruptible=True, log_output=True):
        self.note("$ " + shlex.join(map(str, argv)))
        if interruptible:
            self.check_cancelled()
        process = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True
        )
        output = bytearray()
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        deadline = time.monotonic() + timeout if timeout is not None else None
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                while selector.get_map() or process.poll() is None:
                    if interruptible:
                        self.check_cancelled()
                    if deadline is not None and time.monotonic() >= deadline:
                        raise RuntimeError(f"{argv[0]} {argv[1]} timed out after {timeout}s")
                    for key, _ in selector.select(0.1):
                        data = os.read(key.fd, 65536)
                        if not data:
                            selector.unregister(key.fileobj)
                            continue
                        output.extend(data)
                        if stream:
                            del output[:-4096]
                            sys.stdout.write(decoder.decode(data))
                            sys.stdout.flush()
                        if log_output:
                            self.log.write(data)
                            self.log.flush()
            if stream:
                sys.stdout.write(decoder.decode(b"", final=True))
            text = output.decode("utf-8", errors="replace")
            if check and process.returncode:
                raise RuntimeError(
                    f"{argv[0]} {argv[1]} failed (exit {process.returncode}): {text[-4096:].strip()}"
                )
            return process.returncode, text
        finally:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
            process.stdout.close()

    def image(self, reference):
        if not pinned_image(reference):
            raise ValueError("execution image is not pinned; regenerate the plan")
        code, text = self.command(["docker", "image", "inspect", reference], check=False, log_output=False)
        if code:
            if "No such image" not in text or "@sha256:" not in reference:
                raise RuntimeError(f"docker image inspect failed: {text.strip()}")
            self.command(["docker", "pull", reference], stream=True, timeout=None)
            _, text = self.command(["docker", "image", "inspect", reference], log_output=False)
        inspections = json.loads(text)
        if not isinstance(inspections, list) or len(inspections) != 1 or not isinstance(inspections[0], dict):
            raise ValueError("docker image inspect did not return one image")
        inspection = inspections[0]
        return verify_image(reference, inspection)["image_id"], inspection

    def cleanup(self, name, started):
        if started:
            try:
                self.command(
                    ["docker", "stop", "--time", "10", name],
                    check=False,
                    timeout=20,
                    interruptible=False,
                )
            except (OSError, RuntimeError) as error:
                self.note(str(error))
        try:
            code, text = self.command(
                ["docker", "rm", "--force", name], check=False, timeout=20, interruptible=False
            )
            if code and "No such container" not in text:
                return f"Container cleanup failed: {text.strip()}"
        except (OSError, RuntimeError) as error:
            return f"Container cleanup failed: {error}"
        return None


def save_failure(out, pending, error, cleanup_failed):
    try:
        current = read_json(out / "results.json")
        if (
            current["model"] != pending["model"]
            or current.get("target") != pending.get("target")
            or current["run"].get("id") != pending["run"].get("id")
            or current["run"].get("serving_sha256") != pending["run"].get("serving_sha256")
        ):
            current = pending
    except (OSError, ValueError, KeyError, TypeError):
        current = pending
    run = current["run"]
    run.update(completed=False, finished_at=utc_now())
    if not run.get("error"):
        run["error"] = error
    if cleanup_failed:
        run["container_cleanup_failed"] = True
    write_json(out / "results.json", current)


def all_measurements_failed(out: Path, pending: dict) -> bool:
    """Recognize a completed all-failed sweep instead of reporting a container crash."""
    try:
        current = read_json(out / "results.json")
        if any(current[key] != pending[key] for key in ("model", "target")) or any(
            current["run"].get(key) != pending["run"].get(key) for key in ("id", "serving_sha256")
        ):
            return False
        for target in (current, current["control"]):
            run, points = target["run"], target["points"]
            identities = [(point["arm"], point["conc"]) for point in points]
            if (
                run.get("completed") is not True
                or run.get("error")
                or type(run.get("points_expected")) is not int
                or not points
                or len(points) != run["points_expected"]
                or len(identities) != len(set(identities))
                or any(point.get("status") != "failed" for point in points)
            ):
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def run(args, runtime_options: list[str]) -> int:
    repo, out = REPO_ROOT.resolve(), args.out.resolve()
    cache, work = (args.cache or out / "cache").resolve(), (args.work or out / "work").resolve()
    if any(path.is_relative_to(repo) for path in (out, cache, work)):
        print("ERROR: run, work and cache directories must be outside the source checkout", file=sys.stderr)
        return 1
    try:
        attempt_id = begin_attempt(out)
        pending = read_json(out / "results.json")
        log = (out / "launcher.log").open("wb")
    except OSError as error:
        print(f"ERROR: unable to create sweep output: {error}", file=sys.stderr)
        return 1
    launcher = Launcher(log)
    handlers = {sig: signal.signal(sig, launcher.on_signal) for sig in (signal.SIGINT, signal.SIGTERM)}
    name = f"agentx-sweep-{attempt_id}"
    launch_dir = None
    created = started = False
    error = cleanup_error = None
    code = 1
    try:
        try:
            serving_path = args.serving.resolve()
            serving = read_json(serving_path)
            run_id = args.run_id or os.environ.get("GITHUB_RUN_ID", "local")
            pending = pending_results(
                serving,
                selected_points(serving),
                {
                    "id": str(run_id),
                    "attempt_id": attempt_id,
                    "aklr_sha": "unknown",
                    "source_dirty": True,
                    "host": socket.gethostname(),
                },
            )
            write_json(out / "results.json", pending)
            if args.hub is None or not args.hub.is_dir():
                raise ValueError("--hub must name an existing HF hub cache directory")
            hub = args.hub.resolve()
            reference = serving["image"]["run"]
            if args.image is not None and args.image != reference:
                raise ValueError("--image conflicts with the plan; regenerate it with --run-image")
            for path in (cache, work):
                path.mkdir(parents=True, exist_ok=True)
            _, sha = launcher.command(["git", "-C", str(repo), "rev-parse", "HEAD"])
            _, status = launcher.command(
                ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"]
            )
            for target in (pending, pending["control"]):
                target["run"].update(aklr_sha=sha.strip(), source_dirty=bool(status.strip()))
            write_json(out / "results.json", pending)
            _, git_dir = launcher.command(["git", "-C", str(repo), "rev-parse", "--absolute-git-dir"])
            _, git_common = launcher.command(
                ["git", "-C", str(repo), "rev-parse", "--path-format=absolute", "--git-common-dir"]
            )
            image_id, inspection = launcher.image(reference)
            launch_dir = Path(tempfile.mkdtemp(prefix="agentx-launch-"))
            execution_path = write_json(launch_dir / "execution.json", inspection)
            command = [
                "docker",
                "create",
                "--name",
                name,
                "--label",
                f"agentx.out={out}",
                "--pull",
                "never",
                "--init",
                "--network",
                "host",
                "--device=/dev/kfd",
                "--device=/dev/dri",
                "--ipc=host",
                "--security-opt=seccomp=unconfined",
                "--ulimit",
                "memlock=-1:-1",
            ]
            for mount in (
                f"{repo}:/src:ro",
                f"{hub}:{hub}:ro",
                f"{serving_path}:/agentx/serving.json:ro",
                f"{out}:/agentx/out",
                f"{cache}:/agentx/cache",
                f"{work}:/agentx/work",
                f"{git_common.strip()}:{git_common.strip()}:ro",
                f"{execution_path}:/agentx/execution.json:ro",
            ):
                command.extend(["-v", mount])
            for key, value in {
                "HF_HUB_CACHE": str(hub),
                "HF_HUB_OFFLINE": "1",
                "GITHUB_SHA": sha.strip(),
                "AKLR_SOURCE_DIRTY": str(bool(status.strip())).lower(),
                "HOST_IDS": f"{os.getuid()}:{os.getgid()}",
                "SOURCE_GIT_DIR": git_dir.strip(),
                "AGENTX_EXECUTION_FILE": "/agentx/execution.json",
            }.items():
                command.extend(["-e", f"{key}={value}"])
            command.extend([image_id, "bash", "-c", CONTAINER_COMMAND, "sweep"])
            command.extend(runtime_options)
            if args.run_id is None:
                command.extend(["--run-id", str(run_id)])
            launcher.check_cancelled()
            created = True
            # Finish creation before cancellation so cleanup finds the container.
            launcher.command(command, timeout=30, interruptible=False)
            launcher.check_cancelled()
            started = True
            code, text = launcher.command(
                ["docker", "start", "--attach", name], stream=True, timeout=None, check=False
            )
            if code == 1 and all_measurements_failed(out, pending):
                launcher.note("All measurement points failed; see results.json for point errors.")
            elif code:
                error = f"Container exited with status {code}: {text.strip()}"
        except (Cancelled, OSError, RuntimeError, ValueError, KeyError, TypeError) as failure:
            error = str(failure)
        finally:
            if created:
                cleanup_error = launcher.cleanup(name, started and (code != 0 or launcher.signum is not None))
            if launcher.signum is not None:
                code = 128 + launcher.signum
                error = error or f"Sweep cancelled by {signal.Signals(launcher.signum).name}"
            if error or cleanup_error:
                code = code or 1
                for message in (error, cleanup_error):
                    if message:
                        launcher.note(message)
                        print(f"ERROR: {message}", file=sys.stderr)
                save_failure(out, pending, error or cleanup_error, bool(cleanup_error))
    finally:
        try:
            if launch_dir is not None and not cleanup_error:
                shutil.rmtree(launch_dir)
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            log.close()
    return code
