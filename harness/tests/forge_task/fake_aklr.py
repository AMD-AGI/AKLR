# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A two-workload AKLR schema, and a AKLR stand-in with the contract calls, the serving bar and the
`validate` report the driver reads."""

from __future__ import annotations

import json
from pathlib import Path

SCHEMA_NAME = "gemm_n32_k64"
CASES = ("m=8", "m=16")

# The contract surface the driver and the api-params probe call, with the schema text as plain JSON.
FAKE_CONTRACT = """import hashlib, json
from pathlib import PurePosixPath


class _Param:
    def __init__(self, name):
        self.name = name


class _Narrowed:
    def __init__(self, raw):
        self.raw = raw

    def schema_text(self):
        return json.dumps(self.raw, indent=2) + "\\n"


def gemm_scaled_baseline(**params):
    return params["out"]


class _Contract:
    def __init__(self, raw):
        self.raw = raw
        inputs = [name for name in raw["inputs"] if name not in ("a", "b")]
        self.api_params = [_Param(name) for name in ("a", "b", *inputs, "trans_b", "bias", "out_dtype", "out")]

    def baseline(self):
        return gemm_scaled_baseline if "a_scale" in self.raw["inputs"] else None

    def point_from_json(self, raw, axes):
        return axes

    def narrow(self, point, *, workloads, kernel):
        return _Narrowed(dict(self.raw, workloads=workloads, kernel=kernel))


class ContractRegistry:
    @staticmethod
    def default():
        return ContractRegistry()

    def for_schema_json(self, raw):
        return _Contract(raw)


class OpContract:
    @staticmethod
    def authored_parts(schema):
        return {"kernel": schema["kernel"], "workloads": schema["workloads"]}


def schema_relative_path(name, language, uuid):
    return PurePosixPath("impls", language, name, f"{uuid}.json")


def schema_uuid(name, parts):
    return hashlib.sha256(json.dumps([name, parts], sort_keys=True).encode()).hexdigest()[:16]
"""

# `python -m aklr validate`: passes a kernel whose source contains {passing!r} on every
# workload, finds a numeric mismatch on every workload otherwise, and counts its calls. As
# AKLR does, a workload's mismatch is filed under the bench stage's demotions with the
# fraction it compared, and a schema with no workload left fails on a finding of its own.
# Its first {platform_failures} calls abort the way a tracer fault does: a platform failure
# on every workload, none of them compared. With {timed_baseline} false it compares outputs
# but times no baseline, as for a contract that defines none.
FAKE_VALIDATE = """import json, pathlib, sys
argv = sys.argv[1:]
out = pathlib.Path(argv[argv.index("--out") + 1])
candidate = json.loads(pathlib.Path(argv[-1]).read_text())
calls = pathlib.Path("validate_calls")
calls.write_text(str(int(calls.read_text()) + 1 if calls.exists() else 1))
aborted = int(calls.read_text()) <= {platform_failures}
passing = any({passing!r} in source["content"] for source in candidate["kernel"]["sources"])
findings, demotions, records = [], [], []
for index, workload in enumerate(candidate["workloads"]):
    if aborted:
        findings.append({{"code": "platform_failure", "axes": workload["axes"], "detail": "timing: rocprofiler-sdk"}})
        records.append({{"axes": workload["axes"], "worst_violation_fraction": None, "timings": []}})
        continue
    if not passing:
        demotions.append({{"code": "numeric_mismatch", "axes": workload["axes"], "detail": "off"}})
    timing = {{"mode": "graph", "versus": "baseline", "ratio_lower_bound": 0.9,
               "candidate_milliseconds": [0.01 * (index + 1)] * 3, "versus_milliseconds": [0.009 * (index + 1)] * 3}}
    timings = [timing] if {timed_baseline} else []
    violation = 0.0 if passing else 0.5
    records.append({{"axes": workload["axes"], "worst_violation_fraction": violation, "timings": timings}})
if demotions and not aborted:
    findings.append({{"code": "no_qualifying_workload", "axes": None, "detail": "no workload passes"}})
out.mkdir(parents=True, exist_ok=True)
report = {{"stages": [{{"stage_name": "audit", "findings": []}},
                      {{"stage_name": "bench", "findings": findings, "demotions": demotions,
                        "workload_records": records}}]}}
(out / "report.json").write_text(json.dumps(report))
"""


def install_aklr(
    root: Path,
    *,
    passing: str = "torch.matmul",
    platform_failures: int = 0,
    timed_baseline: bool = True,
    min_speedup: float = 1.03,
) -> Path:
    """Plant the AKLR stand-in at ``root``, a checkout `--aklr-root` can name."""
    contract = root / "aklr" / "core" / "contract"
    platform = root / "aklr" / "core" / "platform"
    for package in (root / "aklr", root / "aklr" / "core", contract, platform):
        package.mkdir(parents=True, exist_ok=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
    for module in ("registry", "op", "locate"):
        (contract / f"{module}.py").write_text(FAKE_CONTRACT, encoding="utf-8")
    (platform / "env.py").write_text(
        f"class RuntimeEnv:\n    min_speedup = {min_speedup!r}\n", encoding="utf-8"
    )
    (root / "aklr" / "__main__.py").write_text(
        FAKE_VALIDATE.format(
            passing=passing,
            platform_failures=platform_failures,
            timed_baseline=timed_baseline,
        ),
        encoding="utf-8",
    )
    return root


def schema_payload(
    name: str = SCHEMA_NAME,
    *,
    op_type: str = "gemm",
    rank: int = 2,
    sources: list | None = None,
):
    batch = ["batch"] if rank == 3 else []
    return {
        "schema_version": 6,
        "name": name,
        "contract_id": f"{op_type}@1.0",
        "description": "16-bit gemm.",
        "op_type": op_type,
        "axes": {
            "m": {"type": "var", "description": "Tokens."},
            "n": {"type": "const", "value": 32},
            "k": {"type": "const", "value": 64},
        },
        "attrs": {"trans_b": True, "has_bias": False},
        "inputs": {
            "a": {"shape": [*batch, "m", 64], "dtype": "bfloat16"},
            "b": {"shape": [*batch, 32, 64], "dtype": "bfloat16"},
        },
        "outputs": {"out": {"shape": [*batch, "m", 32], "dtype": "bfloat16"}},
        "workloads": [{"axes": {"m": int(case.split("=")[1])}} for case in CASES],
        "compare": {
            "max_violation_fraction": 0.0,
            "determinism": "bitwise",
            "element_order": "fixed",
        },
        "kernel": {
            "language": "flydsl",
            "targets": [{"arch": "gfx950", "hardware": "MI355X"}],
            "entry_point": "kernel.py::run",
            "sources": sources or [],
        },
    }


def write_schema(aklr_root: Path, payload: dict, uuid: str = "0123456789abcdef") -> Path:
    path = (
        aklr_root
        / "aklr"
        / "ops"
        / payload["op_type"]
        / "impls"
        / "flydsl"
        / payload["name"]
        / f"{uuid}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
