# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import base64
import hashlib
import json
import os
import re
import shutil
import ssl
import subprocess
import sys
import tempfile
import time
from http.client import HTTPException
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import HTTPSHandler, Request, build_opener


class Deployment:
    def __init__(self):
        env = os.environ
        self.name = f"aklr-{env['GITHUB_RUN_ID']}-{env['GITHUB_RUN_ATTEMPT']}-{env['GITHUB_JOB']}"
        self.url = env["AKLR_SAFE_API_URL"].rstrip("/") + "/api/v1/workloads"
        if not self.url.startswith("https://"):
            raise ValueError("SaFE requires HTTPS")
        ca_urls = env.get("AKLR_SAFE_CA_URLS", "").split()
        fetch = ["curl", "-fsSL", "--retry", "3", "--proto", "=https", "--proto-redir", "=https", *ca_urls]
        certs = subprocess.check_output(fetch, text=True) if ca_urls else None
        self.http = build_opener(HTTPSHandler(context=ssl.create_default_context(cadata=certs)))

    def api(self, method, suffix="", payload=None, allow=()):
        request = Request(self.url + suffix, method=method, data=payload and json.dumps(payload).encode())
        request.add_header("Content-Type", "application/json")
        # Unredirected: a redirect never receives the key.
        request.add_unredirected_header("Authorization", "Bearer " + os.environ["USER_APIKEY"])
        for retry in range(6):
            try:
                with self.http.open(request, timeout=60) as response:
                    return json.loads(response.read() or "{}")
            except HTTPError as error:
                reason = error.read().decode(errors="replace")[:500]
                # A 404 without SaFE's errorCode is a gateway's: the workload may still be there.
                gateway = error.code == 404 and "errorCode" not in reason
                transient = gateway or error.code in {408, 429} or error.code >= 500
                # 409 on a retry: the first try had arrived.
                if not transient and (error.code in allow or (error.code == 409 and retry)):
                    return None
                if retry == 5 or not transient:
                    error.msg = f"{error.msg} from {method} {suffix or self.name}: {reason}"
                    raise
            except (OSError, HTTPException):  # HTTPException: the connection dropped during the answer
                if retry == 5:
                    raise
            time.sleep(2**retry)

    def create(self, script):
        env = os.environ
        source = base64.b64encode(Path(__file__).read_bytes()).decode()
        command = f'f=$(mktemp) && echo {source} | base64 -d > "$f" && exec python3 "$f" pod'
        passed = {name: env[name] for name in env.get("PASS_ENV", "").split() if name in env}
        payload = dict(
            workloadId=self.name,
            displayName=self.name,
            workspaceId=env["AKLR_SAFE_WORKSPACE"],
            groupVersionKind={"kind": "Deployment", "version": "v1"},
            resources=[json.loads(env["AKLR_SAFE_RESOURCES"])],
            images=[env["IMAGE"]],
            useWorkspaceStorage=False,
            entryPoints=[base64.b64encode(command.encode()).decode()],
            priority={"medium": 1, "high": 2}[env["AKLR_GPU_PRIORITY"]],
            # SaFE counts from the dispatch, the wait for a node included, and the pod limits the
            # tests itself: this only stops a Deployment no job deleted, at the job's own limit.
            timeout=480 * 60,
            # The unit tests leave the GPUs idle for longer than the workspace lets a workload be.
            lowUtilizationExempt=True,
            env=dict(
                passed,
                TIMEOUT_MINUTES=env["TIMEOUT_MINUTES"],
                AKLR_REPO=f"{env['GITHUB_SERVER_URL']}/{env['GITHUB_REPOSITORY']}",
                AKLR_COMMIT=env["GITHUB_SHA"],
                AKLR_SCRIPT=base64.b64encode(script.encode()).decode(),
            ),
        )
        self.api("POST", payload=payload)
        # SaFE answers 404 for the first moments after it accepts a workload.
        for _ in range(30):
            if self.api("GET", "/" + self.name, allow=(404,)):
                break
            time.sleep(2)
        print(f"🚀 Workload {self.name} in {env['IMAGE']} at {env['AKLR_GPU_PRIORITY']} priority")

    def delete(self):
        # Twice: SaFE does not see a workload for the first moments after it accepts one, and
        # answers a DELETE the same with or without it.
        for wait in (5, 0):
            self.api("DELETE", "/" + self.name)
            time.sleep(wait)
        print(f"🧹 Deleted the workload {self.name}")

    def collect(self, reports):
        """Print the workload's progress and test output, unpack the reports, return the exit code."""
        reports.mkdir(parents=True, exist_ok=True)
        shown = pod = boot = ""
        printed = deadline = 0
        while True:
            state = self.api("GET", "/" + self.name, allow=(404,))
            if not state or state["phase"] in {"Stopped", "Failed", "Succeeded"}:
                raise RuntimeError(f"Workload {self.name} ended before the tests: {state and state['phase']}")
            status = f"{state['phase']} {state.get('message') or ''}".strip()
            pods = state.get("pods") or []
            lines = []
            if pods and "Running" in (state["phase"], pods[0].get("phase")):
                if pod and pods[0]["podId"] != pod:
                    raise RuntimeError(f"The GPU pod {pod} was replaced during the tests")
                if not pod:  # The pod limits the tests; this limits a pod that never reports their end.
                    deadline = time.time() + (int(os.environ["TIMEOUT_MINUTES"]) + 15) * 60
                pod = pods[0]["podId"]
                logs = f"/{self.name}/pods/{quote(pod, safe='')}/logs?tailLines="
                try:
                    lines = self.api("GET", logs + "4000").get("logs") or []
                    done = [line for line in lines if line.startswith("AKLR_DONE ")]
                    if done:  # the pod's last line: read again, far enough back for every chunk
                        lines = self.api("GET", logs + str(int(done[0].split()[2]) + 4000))["logs"]
                except OSError as error:  # HTTP 500 until the container has started
                    if getattr(error, "code", 0) in {401, 403}:
                        raise
                    status += f" (pod log: {error})"
            elif pods and pods[0].get("hostIP"):
                # On its node, the queue behind it: this limits an image that never arrives.
                deadline = deadline or time.time() + 60 * 60
            if status != shown:
                print(f"⏳ Workload {self.name}: {status}")
                shown = status
            chunks = []
            for line in lines:
                if match := re.fullmatch(r"test (\d+) (\d+) \| ?(.*)", line):
                    if boot and match[1] != boot:
                        raise RuntimeError("The GPU pod's container restarted during the tests")
                    boot = match[1]
                    number = int(match[2])
                    if number > printed:
                        if number > printed + 1:
                            print(f"::warning::{number - printed - 1} lines of the pod's log were lost")
                        print(match[3])
                        printed = number
                elif line.startswith("AKLR_DATA "):
                    chunks.append(base64.b64decode(line[10:], validate=True))
                elif line.startswith("AKLR_DONE "):
                    _, code, _, checksum = line.split()
                    archive = b"".join(chunks)
                    if hashlib.sha256(archive).hexdigest() != checksum:
                        raise ValueError("The pod's log did not return the reports whole")
                    subprocess.run(["tar", "-xzC", str(reports)], input=archive, check=True)
                    # Later steps print and upload the reports: a link would hand them the runner's files.
                    for link in [path for path in reports.rglob("*") if path.is_symlink()]:
                        link.unlink()
                    return int(code)
                elif line.startswith("AKLR_ERROR "):
                    raise RuntimeError(f"The GPU pod failed: {line[11:]}")
            if deadline and time.time() > deadline:
                raise RuntimeError(
                    f"Workload {self.name} ran out of time, {status}; its log ends: {lines[-3:]}"
                )
            time.sleep(8)

    @staticmethod
    def run_in_pod():
        """Run the test script on the tested commit and write the pod's side of the log."""
        try:
            # /dev/shm is the pod's, not the container's: a restarted container finds the mark.
            started = Path("/dev/shm/aklr-tests-started")
            if started.exists():
                raise RuntimeError("the container restarted during the tests")
            started.touch()
            work = Path(tempfile.mkdtemp())
            (work / "src").mkdir()
            (work / "reports").mkdir()
            (work / "test.sh").write_bytes(base64.b64decode(os.environ["AKLR_SCRIPT"]))
            tests = subprocess.Popen(
                'git init -q . && git fetch -q --tags "$AKLR_REPO" "$AKLR_COMMIT" && '
                'git checkout -q --detach "$AKLR_COMMIT" && '
                # timeout leads a process group: what the tests left in it would hold the output open.
                # dash's kill takes the signal only as -s here.
                '{ timeout -k 60s "${TIMEOUT_MINUTES}m" bash ../test.sh & wait $!; code=$?; '
                "kill -s KILL -- -$! 2> /dev/null; exit $code; }",
                shell=True,
                cwd=work / "src",
                env=dict(os.environ, REPORTS=str(work / "reports")),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
            )
            # A restarted container runs the tests again from line 1: its start time tells them apart.
            boot = int(time.time())
            for number, line in enumerate(tests.stdout, 1):
                print(f"test {boot} {number} | {line.rstrip()}")
            code = tests.wait() & 255
            archive = Path(shutil.make_archive(str(work / "reports"), "gztar", work / "reports")).read_bytes()
            # shortcut: the reports ride the pod's log after the test output, and the node rotates that
            # log (10 MiB by default): fine while the output plus 1.35 times the reports stay below
            # that, use object storage beyond.
            chunks = [archive[start : start + 3072] for start in range(0, len(archive), 3072)]
            for chunk in chunks:
                print("AKLR_DATA", base64.b64encode(chunk).decode())
            print("AKLR_DONE", code, len(chunks), hashlib.sha256(archive).hexdigest())
        except Exception as error:
            print(f"AKLR_ERROR {error!r}")
        # A Deployment restarts a command that exits, which would run the tests again.
        while True:
            time.sleep(3600)


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)  # both logs are followed live
    if sys.argv[1] == "pod":
        Deployment.run_in_pod()
    elif sys.argv[1] == "delete":
        Deployment().delete()
    else:
        deployment = Deployment()
        deployment.create(Path(sys.argv[1]).read_text())
        sys.exit(deployment.collect(Path(os.environ["RUNNER_TEMP"], "gpu-reports")))
