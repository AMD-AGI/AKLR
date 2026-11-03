#!/usr/bin/env python3
"""GEAK's task runner for an AKLR schema: compile, correctness and performance, each done by forge_task's driver.py
beside this task. Correctness is what `python -m aklr validate` returns for kernel.py published as the schema's
kernel; performance is AKLR's own graph timer on AKLR's timing inputs, one line per workload. Reports go to build/.
"""

import argparse
import json
import os
import re
import subprocess
import sys

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRIVER = os.path.join(TASK_DIR, "driver.py")
BUILD = os.path.join(TASK_DIR, "build")


def _driver(*args):
    return subprocess.run(
        [sys.executable, DRIVER, *args], cwd=TASK_DIR, capture_output=True, text=True, check=False
    )


def _report(name, report):
    with open(os.path.join(BUILD, f"{name}_report.json"), "w") as f:
        json.dump(report, f, indent=2)


def compile_kernel():
    """Import kernel.py as AKLR imports it: a package over the task, so its relative imports resolve."""
    sys.path.insert(0, TASK_DIR)
    try:
        import driver

        driver._load_candidate()
        return True, None
    except Exception as exc:  # any import failure is the candidate's
        return False, f"{type(exc).__name__}: {exc}"


def main():
    parser = argparse.ArgumentParser(description="GEAK task runner over AKLR's forge_task driver")
    parser.add_argument("mode", choices=["compile", "correctness", "performance"])
    mode = parser.parse_args().mode
    os.makedirs(BUILD, exist_ok=True)
    if mode == "compile":
        ok, err = compile_kernel()
        _report("compile", {"status": "ok" if ok else "fail", "error": err})
        print(f"Compilation: {'PASS' if ok else 'FAIL'}")
        if err:
            print(f"Error: {err}")
        sys.exit(0 if ok else 1)
    if mode == "correctness":
        done = _driver()
        findings = re.findall(r"^aklr_findings: (.*)$", done.stdout, re.M)
        ok = done.returncode == 0 and re.search(r"^allclose: True$", done.stdout, re.M) is not None
        _report("correctness", {"status": "ok" if ok else "fail", "findings": findings})
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        for finding in findings:
            print(f"AKLR finding: {finding}")
        if not ok and not findings:
            print((done.stdout + done.stderr)[-3000:])
        sys.exit(0 if ok else 1)
    done = _driver("--bench-mode", "--warmup", "3", "--iters", "10")
    cases = [
        {"test_case_id": case, "execution_time_ms": float(ms)}
        for case, ms in re.findall(r"^case_ms: (\S+) ([0-9.eE+-]+)$", done.stdout, re.M)
    ]
    _report("performance", {"test_cases": cases})
    for case in cases:
        print(f"Perf: {case['execution_time_ms']:.6f} ms ({case['test_case_id']})")
    if done.returncode != 0 or not cases:
        print((done.stdout + done.stderr)[-3000:])
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
