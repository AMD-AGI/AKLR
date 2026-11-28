# SPDX-License-Identifier: MIT
"""Verify the framework's image, run one group's sweep in a container, and clean the container up."""

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

from perf_utils.image import pinned_image, verify_image

from e2e_perf.model_config import REPO_ROOT
from e2e_perf.runtime import pending_results
from e2e_perf.utils import begin_attempt, read_json, utc_now, write_json

CONTAINER_COMMAND = r"""
set -euo pipefail
touch -d '1 second ago' /tmp/e2e-started
trap 'find /e2e/out /e2e/cache /e2e/work -mindepth 1 ! -user "${HOST_IDS%%:*}" -cnewer /tmp/e2e-started -exec chown -h "$HOST_IDS" {} +' EXIT
mkdir /aklr
tar -C /src --exclude='./.git' -cf - . | tar -C /aklr -xf -
printf 'gitdir: %s\n' "$SOURCE_GIT_DIR" > /aklr/.git
cd /aklr
# the recipe's setup (pinned package installs into the image), before AKLR's, in both modes' one container
python3 -c 'import json, sys; sys.stdout.writelines(f"{c}\n" for c in json.load(open("/e2e/plan.json"))["recipe"].get("setup", []))' > /tmp/e2e-setup.sh
if [ -s /tmp/e2e-setup.sh ]; then
  (cd /tmp && PIP_CACHE_DIR=/e2e/cache/pip bash -euxo pipefail /tmp/e2e-setup.sh) 2>&1 | tee /e2e/out/setup.log
fi
scripts/install_aklr.sh /e2e/cache "$E2E_FRAMEWORK" 2>&1 | tee /e2e/out/install.log
set -a; . /e2e/cache/aklr.env; set +a
cd benchmarks/perf
python3 -m e2e_perf sweep --in-container --plan /e2e/plan.json --work /e2e/work --out /e2e/out "$@"
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
                    ["docker", "stop", "--time", "10", name], check=False, timeout=20, interruptible=False
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


def current_results(out: Path, pending: dict) -> dict | None:
    """The container's results, if they belong to this run of this group."""
    try:
        current = read_json(out / "results.json")
        same = all(current["run"].get(key) == pending["run"][key] for key in ("id", "run_key"))
        return current if same and current.get("model") == pending["model"] else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def all_points_failed(out: Path, pending: dict) -> bool:
    """A completed sweep whose points all failed exits 1; its results are the record, not a crash."""
    current = current_results(out, pending)
    if not current or current["run"].get("completed") is not True:
        return False
    points = [point for mode in current["modes"].values() for point in mode.get("points", [])]
    return not any(point.get("status") == "ok" for point in points)


def save_failure(out: Path, pending: dict, error: str, cleanup_failed: bool) -> None:
    """Keep the container's results if they belong to this run, and record why the sweep failed."""
    current = current_results(out, pending) or pending
    run = current["run"]
    run.update(completed=False, finished_at=utc_now())
    run["error"] = run.get("error") or error
    if cleanup_failed:
        run["container_cleanup_failed"] = True
    write_json(out / "results.json", current)


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
    name = f"e2e-perf-{attempt_id}"
    launch_dir = None
    created = started = False
    error = cleanup_error = None
    code = 1
    try:
        try:
            plan_path = args.plan.resolve()
            plan = read_json(plan_path)
            run_id = args.run_id or os.environ.get("GITHUB_RUN_ID", "local")
            pending = pending_results(
                plan,
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
            for path in (cache, work):
                path.mkdir(parents=True, exist_ok=True)
            _, sha = launcher.command(["git", "-C", str(repo), "rev-parse", "HEAD"])
            _, status = launcher.command(
                ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"]
            )
            pending["run"].update(aklr_sha=sha.strip(), source_dirty=bool(status.strip()))
            write_json(out / "results.json", pending)
            _, git_dir = launcher.command(["git", "-C", str(repo), "rev-parse", "--absolute-git-dir"])
            _, git_common = launcher.command(
                ["git", "-C", str(repo), "rev-parse", "--path-format=absolute", "--git-common-dir"]
            )
            image_id, inspection = launcher.image(plan["image"]["run"])
            launch_dir = Path(tempfile.mkdtemp(prefix="e2e-launch-"))
            execution_path = write_json(launch_dir / "execution.json", inspection)
            command = [
                "docker",
                "create",
                "--name",
                name,
                "--label",
                f"e2e.out={out}",
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
                # Serving images such as vllm-openai set ENTRYPOINT ["vllm", "serve"].
                "--entrypoint",
                "bash",
            ]
            for mount in (
                f"{repo}:/src:ro",
                f"{hub}:{hub}:ro",
                f"{plan_path}:/e2e/plan.json:ro",
                f"{out}:/e2e/out",
                f"{cache}:/e2e/cache",
                f"{work}:/e2e/work",
                f"{git_common.strip()}:{git_common.strip()}:ro",
                f"{execution_path}:/e2e/execution.json:ro",
            ):
                command.extend(["-v", mount])
            for key, value in {
                "HF_HUB_CACHE": str(hub),
                "HF_HUB_OFFLINE": "1",
                "GITHUB_SHA": sha.strip(),
                "AKLR_SOURCE_DIRTY": str(bool(status.strip())).lower(),
                "HOST_IDS": f"{os.getuid()}:{os.getgid()}",
                "SOURCE_GIT_DIR": git_dir.strip(),
                "AKLR_EXECUTION_FILE": "/e2e/execution.json",
                "E2E_FRAMEWORK": plan["framework"]["id"],
            }.items():
                command.extend(["-e", f"{key}={value}"])
            command.extend([image_id, "-c", CONTAINER_COMMAND, "sweep", *runtime_options])
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
            if code == 1 and all_points_failed(out, pending):
                launcher.note("All measurement points failed; see results.json for point errors.")
            elif code:
                error = f"Container exited with status {code}: {text.strip()[-2000:]}"
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
