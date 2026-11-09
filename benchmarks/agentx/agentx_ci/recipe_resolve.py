# SPDX-License-Identifier: MIT
"""Resolve InferenceX recipes into serving.json."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import sysconfig
import time
import urllib.parse
import urllib.request
from pathlib import Path
from urllib.parse import quote

import yaml

from agentx_ci.model_config import ConfigError, model_config
from agentx_ci.performance_analysis import X, Y, frontier
from agentx_ci.preflight import pinned_image
from agentx_ci.utils import (
    DEFAULT_CONFIG,
    REPO_ROOT,
    load_config,
    read_json,
    sha256_json,
    utc_now,
    write_json,
)

INFERENCEX_SPARSE = (
    "/inferencex-e2e/infx/",
    "/inferencex-e2e/configs/",
    "/inferencex-e2e/benchmarks/*/srt-slurm-recipes/",
    "/inferencex-e2e/benchmarks/runtime_settings.sh",
    "/inferencex-e2e/uv.lock",
    "/inferencex-e2e/pyproject.toml",
)
DEP_PACKAGES = (
    "pydantic",
    "pydantic-core",
    "pyyaml",
    "annotated-types",
    "typing-extensions",
    "typing-inspection",
)
RUNTIME_SETTINGS = "inferencex-e2e/benchmarks/runtime_settings.sh"
WORKER_OWNED_ARGS = {
    "model-path",
    "served-model-name",
    "nccl-port",
    "grpc-http-sidecar-port",
    "smg-http-sidecar-port",
    "port",
    "host",
}
BOUND_CLIENT_ENV = {
    "CONC",
    "MODEL",
    "KV_OFFLOADING",
    "REQUIRE_POWER",
    "ENABLE_AGENTX_POWER",
}
POINT_CLIENT_ENV = (
    "MODEL",
    "MODEL_PREFIX",
    "PRECISION",
    "FRAMEWORK",
    "CONC",
    "TP",
    "EP_SIZE",
    "PP_SIZE",
    "PCP_SIZE",
    "DCP_SIZE",
    "DP_ATTENTION",
    "KV_OFFLOADING",
    "TOTAL_CPU_DRAM_GB",
)
FIXED_CLIENT_ENV = {
    "IS_AGENTIC": "1",
    "IS_MULTINODE": "false",
    "EVAL_ONLY": "false",
    "ENABLE_AGENTX_POWER": "false",
    "REQUIRE_POWER": "0",
    "AIPERF_EXPERIMENTAL_FAST": "0",
}


class ResolveError(Exception):
    pass


def sh(cmd: list[str], cwd: str | Path | None = None, timeout: int = 300) -> str:
    done = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False)
    if done.returncode:
        raise ResolveError(f"{shlex.join(cmd)} failed: {done.stderr.strip()[-400:]}")
    return done.stdout


def github_url(repo: str) -> str:
    return f"https://github.com/{repo}.git"


def resolve_ref(repo: str, ref: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", ref):
        return ref
    out = sh(
        [
            "git",
            "ls-remote",
            github_url(repo),
            f"refs/heads/{ref}",
            f"refs/tags/{ref}",
            f"refs/tags/{ref}^{{}}",
        ],
        timeout=60,
    )
    refs = {name: sha for sha, name in (line.split() for line in out.splitlines() if line.strip())}
    tag = f"refs/tags/{ref}"
    if tag + "^{}" in refs:
        refs[tag] = refs.pop(tag + "^{}")
    shas = set(refs.values())
    if len(shas) != 1:
        raise ResolveError(f"{repo}@{ref} resolved to {len(shas)} commits")
    return shas.pop()


def fetch_inferencex(repo: str, sha: str, dst: Path, retries: tuple[int, ...] = (15, 30, 60)) -> Path:
    """Fetch a clean sparse InferenceX checkout at the requested commit."""
    if (dst / ".git").is_dir():
        try:
            if sh(["git", "rev-parse", "HEAD^{commit}"], cwd=dst).strip() == sha:
                if sh(["git", "status", "--porcelain", "--untracked-files=all"], cwd=dst).strip():
                    raise ResolveError("cached InferenceX checkout is dirty")
                if (dst / "inferencex-e2e/uv.lock").is_file():
                    return dst
        except (ResolveError, subprocess.TimeoutExpired):
            pass
    error = None
    for delay in (0, *retries):
        time.sleep(delay)
        shutil.rmtree(dst, ignore_errors=True)
        dst.mkdir(parents=True)
        try:
            sh(["git", "init", "-q"], cwd=dst)
            sh(["git", "remote", "add", "origin", github_url(repo)], cwd=dst)
            sh(
                [
                    "git",
                    "fetch",
                    "-q",
                    "--depth=1",
                    "--filter=blob:none",
                    "origin",
                    sha,
                ],
                cwd=dst,
            )
            sh(["git", "sparse-checkout", "set", "--no-cone", *INFERENCEX_SPARSE], cwd=dst)
            sh(["git", "checkout", "-q", "--detach", "FETCH_HEAD"], cwd=dst)
            return dst
        except (ResolveError, subprocess.TimeoutExpired) as e:
            error = e
    raise ResolveError(f"fetching {repo}@{sha} failed after {len(retries) + 1} attempts: {error}")


def locked_versions(uv_lock: str) -> dict[str, str]:
    locked = dict(
        re.findall(
            r'^\[\[package\]\]\nname = "([^"]+)"\nversion = "([^"]+)"',
            uv_lock,
            re.MULTILINE,
        )
    )
    missing = [name for name in DEP_PACKAGES if name not in locked]
    if missing:
        raise ResolveError(f"uv.lock lacks {missing}")
    return {name: locked[name] for name in DEP_PACKAGES}


def ensure_deps(versions: dict[str, str], cache: Path) -> Path:
    abi = f"{sys.implementation.cache_tag}-{sysconfig.get_platform()}"
    target = cache / f"inferencex-deps-{sha256_json([versions, abi])[:16]}"
    if not (target / ".complete").exists():
        shutil.rmtree(target, ignore_errors=True)
        pins = [f"{name}=={version}" for name, version in versions.items()]
        pip = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-deps",
            "--only-binary=:all:",
        ]
        sh([*pip, "--target", str(target), *pins], timeout=600)
        (target / ".complete").write_text(json.dumps(versions))
    return target


def run_inferencex_points(
    inferencex_e2e: Path, deps: Path, config_file: str, key: str, timeout: int = 120
) -> dict:
    """Resolve points in an isolated interpreter with pinned dependencies."""
    inferencex_e2e, deps = inferencex_e2e.resolve(), deps.resolve()
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": "/nonexistent",
        "LANG": "C.UTF-8",
    }
    # -I ignores PYTHONPATH, so inject the pinned paths.
    script = Path(__file__).with_name("inferencex_points.py")
    path, argv = [str(deps), str(inferencex_e2e)], ["inferencex_points.py", config_file, key]
    boot = (
        f"import runpy, sys; sys.path[:0] = {path!r}; sys.argv = {argv!r}; "
        f"runpy.run_path({str(script)!r}, run_name='__main__')"
    )
    done = subprocess.run(
        [sys.executable, "-I", "-c", boot],
        cwd=inferencex_e2e,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        check=False,
    )
    if done.returncode:
        raise ResolveError(f"InferenceX resolution failed: {done.stderr.strip()[-2000:]}")
    return json.loads(done.stdout)


def config_to_cli_args(config: dict) -> list[str]:
    """Match srtctl.backends.sglang._config_to_cli_args."""
    args: list[str] = []
    for key, value in sorted(config.items()):
        flag = key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                args.append(f"--{flag}")
        elif isinstance(value, list):
            args.append(f"--{flag}")
            args.extend(str(v) for v in value)
        elif value is not None:
            args.extend([f"--{flag}", str(value)])
    return args


def apply_golden(recipe: dict, overrides: list[str]) -> dict:
    """Apply srtctl's build_overrides pairs."""
    out = json.loads(json.dumps(recipe))
    for flag, item in zip(overrides[::2], overrides[1::2], strict=True):
        path, _, raw = item.partition("=")
        *parents, leaf = path.split(".")
        node = out
        for part in parents:
            node = node.setdefault(part, {})
        if flag == "--set":
            node[leaf] = json.loads(raw)
        elif flag == "--unset":
            node.pop(leaf, None)
        else:
            raise ResolveError(f"unexpected override {flag}")
    return out


def canonical(value) -> str:
    return "(flag)" if value is True else str(value)


def runtime_settings(text: str) -> dict[str, str]:
    """Read InferenceX's exported AIPERF defaults."""
    pairs = re.findall(r"^export (AIPERF_\w+)='([^']*)'$", text, re.MULTILINE)
    return dict(pairs)


def arm_topology(row: dict) -> dict:
    """Group points by topology, offloading and decoding strategy."""
    return {
        "tp": int(row["tp"]),
        "ep": int(row["ep"]),
        "pp": int(row.get("pp", 1)),
        "pcp-size": int(row.get("pcp-size", 1)),
        "dcp-size": int(row.get("dcp-size", 1)),
        "dp-attn": row.get("dp-attn", False),
        "kv-offloading": row["kv-offloading"],
        "kv-offload-backend": row.get("kv-offload-backend") if row["kv-offloading"] != "none" else None,
        "spec-decoding": row.get("spec-decoding") or "none",
    }


def arm_of(row: dict) -> str:
    topology = arm_topology(row)
    parts = [f"tp{topology['tp']}", f"ep{topology['ep']}"]
    for key, label in (("pp", "pp"), ("pcp-size", "pcp"), ("dcp-size", "dcp")):
        if topology[key] != 1:
            parts.append(f"{label}{topology[key]}")
    if topology["dp-attn"]:
        parts.append("dpa")
    parts.append(str(topology["kv-offloading"]))
    backend = topology["kv-offload-backend"]
    if backend:
        parts.append(str(backend["name"]))
        if set(backend) - {"name"}:
            parts.append(sha256_json(backend)[:12])
    if topology["spec-decoding"] != "none":
        parts.extend(["spec", str(topology["spec-decoding"])])
    return "-".join(quote(part, safe="") for part in parts)


def model_precision(precision: str, model: str) -> str:
    """Distinguish MXFP4 and NVFP4 within upstream's fp4 label."""
    precision = precision.lower()
    if precision == "fp4":
        for kind in ("mxfp4", "nvfp4"):
            if re.search(rf"(?:^|[^a-z0-9]){kind}(?:$|[^a-z0-9])", model.lower()):
                return kind
    return precision


def time_limit_seconds(value: str) -> int:
    """Parse a Slurm time limit in seconds."""
    days, sep, rest = str(value).partition("-")
    parts = (rest if sep else days).split(":")
    if len(parts) > 3 or not all(part.isdigit() for part in parts) or (sep and not days.isdigit()):
        raise ResolveError(f"unsupported InferenceX time limit {value!r}")
    numbers = list(map(int, parts))
    if len(numbers) == 1:
        seconds = numbers[0] * (3600 if sep else 60)
    elif len(numbers) == 2:
        seconds = numbers[0] * (3600 if sep else 60) + numbers[1] * (60 if sep else 1)
    else:
        seconds = numbers[0] * 3600 + numbers[1] * 60 + numbers[2]
    seconds += int(days) * 86400 if sep else 0
    if seconds <= 0:
        raise ResolveError(f"InferenceX time limit must be positive, got {value!r}")
    return seconds


def runtime_settings_of(point: dict) -> dict:
    runtime = dict(point["runtime"])
    health = runtime["health_check"]
    for field in ("max_attempts", "interval_seconds"):
        value = health[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ResolveError(f"invalid InferenceX health_check.{field}: {value!r}")
    runtime.update(
        ready_timeout_s=health["max_attempts"] * health["interval_seconds"],
        poll_s=health["interval_seconds"],
        point_timeout_s=time_limit_seconds(runtime["time_limit"]),
    )
    return runtime


def resolve_point(point: dict, arm: str, config: dict, client_defaults: dict[str, str]) -> dict:
    row, env = point["row"], point["environment"]
    if row.get("framework") != "sglang":
        raise ResolveError("only the SGLang framework is supported")
    recipe = apply_golden(point["bound_recipe"], point["golden_overrides"])
    role = recipe["roles"]["agg"]
    args = dict(role.get("args") or {})
    engine_env = {**(recipe.get("environment") or {}), **(role.get("env") or {})}
    advisory = config.get("advisory_env") or {}
    clash = sorted(set(advisory) & set(engine_env))
    if clash:
        raise ResolveError(f"advisory env {clash} would override what the recipe sets")
    cli = config_to_cli_args({k: v for k, v in args.items() if k not in WORKER_OWNED_ARGS})
    if "enable-metrics" not in args:
        cli.append("--enable-metrics")
    served = args.get("served-model-name") or row["model"]
    gpus = int(row["tp"]) * int(row["pp"]) * int(row["pcp-size"])
    runtime_args = [
        "--model-path",
        "${AGENTX_MODEL_PATH}",
        "--served-model-name",
        served,
        "--host",
        "127.0.0.1",
        "--port",
        "${AGENTX_PORT}",
        "--revision",
        config["weights"]["revision"],
    ]
    runtime_env = {
        "HF_HUB_OFFLINE": "1",
        "HIP_VISIBLE_DEVICES": ",".join(map(str, range(gpus))),
    }
    command = ["python3", "-m", "sglang.launch_server", *runtime_args, *cli]
    bench_env = (recipe.get("benchmark") or {}).get("env") or {}
    backend = (row.get("kv-offload-backend") or {}).get("name")
    backend_metadata = env.get("KV_OFFLOAD_BACKEND_METADATA")
    client_env = {
        **client_defaults,
        **({"KV_OFFLOAD_BACKEND": backend} if backend else {}),
        **({"KV_OFFLOAD_BACKEND_METADATA": backend_metadata} if backend_metadata else {}),
        "DURATION": str(row["duration"]),
        **{k: str(v) for k, v in bench_env.items() if k not in BOUND_CLIENT_ENV},
        **{name: env[name] for name in POINT_CLIENT_ENV},
        **FIXED_CLIENT_ENV,
    }
    client_gpus = int(client_env["TP"]) * int(client_env["PP_SIZE"]) * int(client_env["PCP_SIZE"])
    if client_gpus != gpus:
        raise ResolveError(f"client GPU count {client_gpus} differs from launch GPU count {gpus}")
    serve = {
        "args": {k: canonical(v) for k, v in sorted(args.items())},
        "env": dict(sorted(engine_env.items())),
    }
    return {
        "exp_name": row["exp-name"],
        "arm": arm,
        "conc": int(row["conc"]),
        "variant": point["variant"],
        "served_model_name": served,
        "gpus": gpus,
        "kv_offloading": row["kv-offloading"],
        "golden_al": point["acceptance_length"],
        "serve": serve,
        "launch": {
            "command": command,
            "env": dict(sorted({**engine_env, **advisory, **runtime_env}.items())),
        },
        "client_env": dict(sorted(client_env.items())),
        "runtime": runtime_settings_of(point),
        "hashes": {
            "serve_spec_sha256": sha256_json(serve),
            "inferencex_recipe_fingerprint": row["recipe-fingerprint"],
        },
    }


def check_entry(inferencex_output: dict, expect: dict) -> None:
    """Validate the upstream model and recipe identity."""
    entry, points = inferencex_output["entry"], inferencex_output["points"]
    if not points:
        raise ResolveError("InferenceX generated no points")
    if entry.get("framework") != "sglang":
        raise ResolveError("only the SGLang framework is supported")
    expected_recipe = Path(expect["recipe"]).relative_to("inferencex-e2e").as_posix()
    for point in points:
        if point["row"].get("srt-recipe") != expected_recipe:
            raise ResolveError(f"generated recipe differs from expected {expected_recipe}")
    for key in ("model", "framework", "model-prefix"):
        if str(entry.get(key)) != str(expect[key]):
            raise ResolveError(f"entry {key}={entry.get(key)!r}, expected {expect[key]!r}")
    if not re.search(expect["runner_pattern"], str(entry.get("runner"))):
        raise ResolveError(f"runner {entry.get('runner')!r} does not match {expect['runner_pattern']!r}")
    actual_precision = model_precision(str(entry.get("precision")), str(entry["model"]))
    if actual_precision != str(expect["precision"]).lower():
        raise ResolveError(f"entry precision={actual_precision!r}, expected {expect['precision']!r}")


def build_spec(
    inferencex_output: dict,
    config: dict,
    *,
    run_image: str,
    client_defaults: dict[str, str],
    provenance: dict,
) -> dict:
    """Build serving.json from resolved InferenceX points."""
    source = config["inferencex"]
    check_entry(inferencex_output, source["expect"])
    warnings = []
    points = []
    topologies = {}
    seen = set()
    for point in inferencex_output["points"]:
        row = point["row"]
        arm = arm_of(row)
        topology = arm_topology(row)
        if arm in topologies and topologies[arm] != topology:
            raise ResolveError(f"ambiguous generated arm id {arm}")
        topologies[arm] = topology
        key = (arm, int(row["conc"]))
        if key in seen:
            raise ResolveError(f"duplicate upstream point {arm}:c{row['conc']}; cannot group unambiguously")
        seen.add(key)
        points.append(resolve_point(point, arm, config, client_defaults))
    points.sort(key=lambda q: (q["arm"], q["conc"]))
    arms = {}
    for name in dict.fromkeys(q["arm"] for q in points):
        mine = [q for q in points if q["arm"] == name]
        arms[name] = {
            "topology": topologies[name],
            "concs": [q["conc"] for q in mine],
            "arm_serve_sha256": sha256_json({q["conc"]: q["hashes"]["serve_spec_sha256"] for q in mine}),
        }
    inferencex_image = inferencex_output["entry"]["image"]
    if inferencex_image != run_image:
        warnings.append(f"run image {run_image} differs from InferenceX's {inferencex_image}")
    return {
        "schema": 3,
        "model": config["model"],
        "target": config["target"],
        "provenance": provenance,
        "image": {
            "run": run_image,
            "inferencex": inferencex_image,
            "differs": inferencex_image != run_image,
        },
        "weights": config["weights"],
        "hardware": config["hardware"],
        "arms": arms,
        "grid_sha256": sha256_json([[q["arm"], q["conc"]] for q in points]),
        "points": points,
        "warnings": warnings,
    }


def registry_digest(image: str) -> str | None:
    """Resolve a registry digest, falling back to the local image."""
    try:
        listing = sh(["docker", "buildx", "imagetools", "inspect", image], timeout=120)
        for line in listing.splitlines():
            key, _, value = line.partition(":")
            if key.strip() == "Digest" and value.strip().startswith("sha256:"):
                return value.strip()
    except (ResolveError, OSError, ValueError, subprocess.TimeoutExpired):
        pass
    try:
        repo_digests = json.loads(
            sh(["docker", "image", "inspect", "--format", "{{json .RepoDigests}}", image])
        )
    except (ResolveError, OSError, ValueError, subprocess.TimeoutExpired):
        return None
    for reference in repo_digests or []:
        if "@sha256:" in reference:
            return reference.split("@", 1)[1]
    return None


def pin_image(image: str, lookup=registry_digest) -> str:
    if pinned_image(image):
        return image
    digest = lookup(image)
    if digest is None or not pinned_image(digest):
        raise ResolveError(
            f"cannot pin execution image {image}; supply an immutable digest or restore registry access"
        )
    return f"{image}@{digest}"


def ci_image(workflow: Path, lookup=registry_digest) -> tuple[str, list[str]]:
    """Pin AKLR_CI_IMAGE and report mutable tags."""
    image = yaml.safe_load(workflow.read_text())["env"]["AKLR_CI_IMAGE"]
    if pinned_image(image):
        return image, []
    return pin_image(image, lookup), [
        f"AKLR_CI_IMAGE {image} pins no digest; pinned the tag's digest at resolve time"
    ]


def resolve_image(reference: str | None) -> tuple[str, list[str]]:
    return (pin_image(reference), []) if reference else ci_image(REPO_ROOT / ".github/workflows/ci.yaml")


def git_blob(tree: Path, path: str) -> str:
    return sh(["git", "hash-object", "--", path], cwd=tree).strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m agentx_bench resolve", description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="run definition")
    parser.add_argument("--model", help="model id; may be omitted when only one model is enabled")
    parser.add_argument("--target", help="target id; may be omitted when the model has one enabled target")
    parser.add_argument("--inferencex-ref", help="InferenceX branch, tag or sha (default: the config's ref)")
    parser.add_argument(
        "--inferencex-src", type=Path, help="use this InferenceX checkout instead of fetching"
    )
    parser.add_argument("--run-image", help="override target.run_image or the default AKLR CI image")
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="serving.json to write")
    args = parser.parse_args(argv)
    try:
        spec = resolve(args)
    except (ResolveError, ConfigError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    summary = {
        "model": spec["model"]["id"],
        "target": spec["target"]["id"],
        "inferencex_sha": spec["provenance"]["inferencex"]["sha"],
        "points": len(spec["points"]),
        "arms": {name: arm["concs"] for name, arm in spec["arms"].items()},
        "warnings": spec["warnings"],
    }
    print(json.dumps(summary, indent=1))
    return 0


def resolve(args: argparse.Namespace) -> dict:
    args.work = args.work.resolve()
    config = model_config(load_config(args.config), args.model, args.target)
    source = config["inferencex"]
    image_reference = args.run_image or config["run_image"]
    run_image, image_warnings = resolve_image(image_reference)
    if args.inferencex_src:
        inferencex = args.inferencex_src.resolve()
        if sh(["git", "status", "--porcelain", "--untracked-files=all"], cwd=inferencex).strip():
            raise ResolveError("--inferencex-src must be a clean checkout; commit changes before resolving")
        sha = sh(["git", "rev-parse", "HEAD"], cwd=inferencex).strip()
    else:
        sha = resolve_ref(source["repo"], args.inferencex_ref or source["ref"])
        inferencex = fetch_inferencex(source["repo"], sha, args.work / "inferencex")
    expect = source["expect"]
    for key in ("config_file", "recipe"):
        if not (inferencex / expect[key]).is_file():
            raise ResolveError(f"{expect[key]} not found at InferenceX {sha}")
    e2e = inferencex / "inferencex-e2e"
    versions = locked_versions((e2e / "uv.lock").read_text())
    deps = ensure_deps(versions, args.work / "cache")
    config_file = str(Path(expect["config_file"]).relative_to("inferencex-e2e"))
    inferencex_output = run_inferencex_points(e2e, deps, config_file, source["config_key"])
    settings = inferencex / RUNTIME_SETTINGS
    client_defaults = runtime_settings(settings.read_text()) if settings.is_file() else {}
    commit_time, subject = sh(["git", "log", "-1", "--format=%cI%x00%s"], cwd=inferencex).strip().split("\0")
    sources = [
        expect["config_file"],
        expect["recipe"],
        RUNTIME_SETTINGS,
        "inferencex-e2e/configs/runners.yaml",
        "inferencex-e2e/infx/launch/drivers/srt/config.py",
        "inferencex-e2e/infx/launch/drivers/srt/recipe.py",
        "inferencex-e2e/infx/launch/drivers/srt/lanes.py",
        "inferencex-e2e/infx/matrix/generate.py",
    ]
    provenance = {
        "inferencex": {
            "repo": source["repo"],
            "requested_ref": args.inferencex_ref or source["ref"],
            "sha": sha,
            "commit_time": commit_time,
            "commit_subject": subject,
            "config_key": source["config_key"],
            "sources": {
                path: git_blob(inferencex, path) for path in sources if (inferencex / path).is_file()
            },
            "deps": versions,
        },
        "resolved_at": utc_now(),
    }
    spec = build_spec(
        inferencex_output,
        config,
        run_image=run_image,
        client_defaults=client_defaults,
        provenance=provenance,
    )
    spec["warnings"] = image_warnings + spec["warnings"]
    try:
        pin_traces(e2e, deps, spec["points"])
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        raise ResolveError(f"cannot resolve pinned trace datasets: {error}") from error
    write_json(args.out, spec)
    return spec


def dataset_revision(repo: str) -> str:
    request = urllib.request.Request(
        f"https://huggingface.co/api/datasets/{quote(repo, safe='/')}/revision/main"
    )
    if os.environ.get("HF_TOKEN"):
        request.add_header("Authorization", f"Bearer {os.environ['HF_TOKEN']}")
    with urllib.request.urlopen(request, timeout=60) as response:
        revision = json.load(response)["sha"]
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ResolveError(f"invalid trace dataset revision for {repo}")
    return revision


def pin_traces(inferencex_e2e: Path, deps: Path, points: list[dict]) -> None:
    sources = [
        [p["client_env"]["MODEL_PREFIX"], p["client_env"].get("WEKA_LOADER_OVERRIDE") or None] for p in points
    ]
    paths = [str(deps.resolve()), str(inferencex_e2e.resolve())]
    script = (
        f"import sys,json; sys.path[:0]={paths!r}; "
        "from infx.bench.agentic.traces import resolve; "
        "json.dump([resolve(*source) for source in json.load(sys.stdin)],sys.stdout)"
    )
    done = subprocess.run(
        [sys.executable, "-I", "-c", script],
        input=json.dumps(sources),
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    resolved = json.loads(done.stdout)
    revisions = {repo: dataset_revision(repo) for repo in sorted({repo for _, repo in resolved})}
    for point, (loader, repo) in zip(points, resolved, strict=True):
        point["trace"] = {"loader": loader, "repo": repo, "revision": revisions[repo]}
        point["client_env"]["WEKA_LOADER_OVERRIDE"] = loader


POLICY = "inferencex-pareto"
API = "https://inferencex.semianalysis.com/api/v1/benchmarks"
ALGORITHM = "p90-e2e-normalized-max-throughput-per-gpu-max-v1"


class SelectionError(ValueError):
    pass


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--point-policy", choices=("all", POLICY), default="all")
    parser.add_argument("--inferencex-model", help="InferenceX display model, e.g. GLM-5.2 (online lookup)")
    parser.add_argument("--inferencex-hardware", help="Upstream hardware series, e.g. mi355x")
    parser.add_argument("--inferencex-results", help="Benchmark API URL or saved upstream-snapshot.json")
    parser.add_argument("--inferencex-run-id", help="Pin a GitHub Actions run with exactRun=true")


def validate_options(args: argparse.Namespace) -> None:
    policy = getattr(args, "point_policy", "all")
    options = [
        getattr(args, name, None)
        for name in ("inferencex_model", "inferencex_hardware", "inferencex_results", "inferencex_run_id")
    ]
    if policy == "all":
        if any(options):
            raise SelectionError("InferenceX source options require --point-policy inferencex-pareto")
        return
    if getattr(args, "arms", "all") != "all" or getattr(args, "points", None):
        raise SelectionError("Pareto policy cannot be combined with --arms or --points")
    if not args.inferencex_hardware:
        raise SelectionError("Pareto policy requires --inferencex-hardware")
    if not args.inferencex_results and not args.inferencex_model:
        raise SelectionError("Provide --inferencex-model or --inferencex-results")
    if args.inferencex_results and (args.inferencex_model or args.inferencex_run_id):
        raise SelectionError("An explicit result source cannot be combined with model/run query options")
    if args.inferencex_run_id and not re.fullmatch(r"[1-9][0-9]*", args.inferencex_run_id):
        raise SelectionError("--inferencex-run-id must be a positive GitHub Actions run id")


def fetch_json(url: str) -> tuple[object, str]:
    request = urllib.request.Request(url, headers={"User-Agent": "AKLR-AgentX", "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        raw = response.read()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def scope_of(spec: dict, hardware: str) -> dict:
    identities = {
        (p["client_env"]["MODEL_PREFIX"], p["client_env"]["FRAMEWORK"], p["client_env"]["PRECISION"])
        for p in spec["points"]
    }
    if len(identities) != 1:
        raise SelectionError("The serving plan must identify one model/framework/precision")
    model, framework, precision = identities.pop()
    return dict(
        model=model,
        framework=framework,
        precision=precision,
        hardware=hardware,
        benchmark_type="agentic_traces",
    )


def scoped_rows(rows: list[dict], scope: dict) -> list[dict]:
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise SelectionError("Expected an InferenceX benchmark row array")
    return [row for row in rows if all(row.get(k) == v for k, v in scope.items())]


def acquire_snapshot(args: argparse.Namespace, scope: dict) -> dict:
    source = args.inferencex_results
    if source and not source.startswith(("https://", "http://")):
        snapshot = read_json(source)
        if (
            not isinstance(snapshot, dict)
            or snapshot.get("schema") != 1
            or snapshot.get("payload_sha256") != sha256_json(snapshot.get("payload"))
        ):
            raise SelectionError("Invalid or modified upstream snapshot")
        return snapshot
    if not source:
        query = {"model": args.inferencex_model}
        if args.inferencex_run_id:
            query.update(runId=args.inferencex_run_id, exactRun="true")
        source = API + "?" + urllib.parse.urlencode(query)
    rows, response_hash = fetch_json(source)
    scoped = scoped_rows(rows, scope)
    if not scoped:
        raise SelectionError(f"No upstream measurements for {scope}")
    workflows = {}
    for run_url in sorted({row.get("run_url", "") for row in scoped}):
        match = re.fullmatch(
            r"https://github.com/SemiAnalysisAI/InferenceX/actions/runs/([0-9]+)/attempts/([0-9]+)", run_url
        )
        if not match:
            raise SelectionError(f"Missing or unsupported InferenceX run provenance: {run_url!r}")
        run_id, attempt = match.groups()
        workflow, _ = fetch_json(
            f"https://api.github.com/repos/SemiAnalysisAI/InferenceX/actions/runs/{run_id}/attempts/{attempt}"
        )
        if (
            str(workflow.get("id")) != run_id
            or str(workflow.get("run_attempt")) != attempt
            or workflow.get("status") != "completed"
        ):
            raise SelectionError(f"Unfinished or mismatched upstream workflow: {run_url}")
        workflows[run_url] = {
            key: workflow.get(key)
            for key in ("id", "run_attempt", "head_sha", "status", "conclusion", "html_url")
        }
    payload = {"rows": rows, "workflows": workflows}
    return {
        "schema": 1,
        "source_url": source,
        "fetched_at": utc_now(),
        "response_sha256": response_hash,
        "payload_sha256": sha256_json(payload),
        "payload": payload,
    }


def point_id(point: dict) -> str:
    return f"{point['arm']}:{point['conc']}"


def check_mapping(row: dict, point: dict, spec: dict) -> None:
    topology = spec["arms"][point["arm"]]["topology"]
    metrics = row["metrics"]
    expected = {
        "prefill_tp": topology["tp"],
        "decode_tp": topology["tp"],
        "prefill_ep": topology["ep"],
        "decode_ep": topology["ep"],
        "prefill_dp_attention": topology["dp-attn"],
        "decode_dp_attention": topology["dp-attn"],
        "spec_method": topology["spec-decoding"],
        "disagg": False,
        "is_multinode": False,
        "image": spec["image"]["inferencex"],
    }
    expected_metrics = {
        "pp": topology["pp"],
        "pcp_size": topology["pcp-size"],
        "dcp_size": topology["dcp-size"],
        "kv_offloading": topology["kv-offloading"],
    }
    backend = topology["kv-offload-backend"]
    if backend:
        expected_metrics["kv_offload_backend"] = backend["name"]
    mismatches = [k for k, v in expected.items() if row.get(k) != v]
    mismatches += [k for k, v in expected_metrics.items() if metrics.get(k) != v]
    if mismatches:
        raise SelectionError(
            f"Upstream point {row['id']} differs from {point_id(point)}: {', '.join(mismatches)}"
        )


def build_selection(spec: dict, snapshot: dict, scope: dict) -> dict:
    payload = snapshot["payload"]
    if not isinstance(payload, dict) or not isinstance(payload.get("workflows"), dict):
        raise SelectionError("Snapshot must contain benchmark rows and workflow provenance")
    if snapshot["payload_sha256"] != sha256_json(payload):
        raise SelectionError("Upstream snapshot checksum mismatch")
    rows = scoped_rows(payload["rows"], scope)
    if not rows:
        raise SelectionError(f"No upstream measurements for {scope}")
    curves = {(row.get("curve_date"), row.get("curve_workflow_run_id")) for row in rows}
    if len(curves) != 1 or any(value is None for value in next(iter(curves))):
        raise SelectionError("Upstream rows must belong to one published curve date/run; use an exact run")
    seen = set()
    seen_recipes = set()
    for row in rows:
        identity = row.get("id")
        if not identity or identity in seen:
            raise SelectionError(f"Missing or duplicate upstream point id: {identity}")
        seen.add(identity)
        if type(row.get("conc")) is not int or row["conc"] <= 0:
            raise SelectionError(f"Invalid upstream concurrency for point {identity}")
        fingerprint = row.get("recipe_fingerprint")
        if not isinstance(fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
            raise SelectionError(f"Missing or invalid upstream recipe fingerprint for point {identity}")
        recipe_key = (fingerprint, row["conc"])
        if recipe_key in seen_recipes:
            raise SelectionError(f"Multiple upstream observations for recipe/concurrency {recipe_key}")
        seen_recipes.add(recipe_key)
        if row.get("error") or row.get("status", "ok") != "ok":
            raise SelectionError(f"Upstream point {identity} is not a successful measurement")
        metrics = row.get("metrics", {})
        if not isinstance(metrics, dict):
            raise SelectionError(f"Missing upstream metrics for point {identity}")
        for key in (X, Y):
            value = metrics.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise SelectionError(
                    f"Upstream point {identity} lacks positive finite {key}; cannot select a frontier"
                )
        workflow = payload["workflows"].get(row.get("run_url"), {})
        if not re.fullmatch(r"[0-9a-f]{40}", workflow.get("head_sha") or ""):
            raise SelectionError(f"Missing upstream source SHA for point {identity}")
        expected_url = (
            "https://github.com/SemiAnalysisAI/InferenceX/actions/runs/"
            f"{workflow.get('id')}/attempts/{workflow.get('run_attempt')}"
        )
        if row.get("run_url") != expected_url or workflow.get("status") != "completed":
            raise SelectionError(f"Unfinished or mismatched upstream workflow for point {identity}")
    coordinates = {(p[X], p[Y]) for p in frontier([row["metrics"] for row in rows])}
    selected_rows = [row for row in rows if (row["metrics"][X], row["metrics"][Y]) in coordinates]
    local = {}
    for point in spec["points"]:
        key = (point["hashes"]["inferencex_recipe_fingerprint"], point["conc"])
        if key in local:
            raise SelectionError("Ambiguous local recipe fingerprint/concurrency")
        local[key] = point
    mapped, errors = [], []
    for row in selected_rows:
        key = (row.get("recipe_fingerprint"), row["conc"])
        point = local.get(key)
        if point is None:
            errors.append(
                f"Upstream point {row['id']} c{row['conc']} recipe {row.get('recipe_fingerprint')} has no exact local match"
            )
            continue
        try:
            check_mapping(row, point, spec)
        except SelectionError as error:
            errors.append(str(error))
            continue
        mapped.append(
            {
                "point_id": point_id(point),
                "exp_name": point["exp_name"],
                "upstream_id": row["id"],
                X: row["metrics"][X],
                Y: row["metrics"][Y],
                "recipe_fingerprint": key[0],
            }
        )
    ids = {p["point_id"] for p in mapped}
    observations = {(r.get("recipe_fingerprint"), r["conc"]) for r in rows}
    unselected = [
        {
            "point_id": point_id(p),
            "exp_name": p["exp_name"],
            "status": "not-selected",
            "reason": "off-upstream-frontier"
            if (p["hashes"]["inferencex_recipe_fingerprint"], p["conc"]) in observations
            else "no-matching-upstream-measurement",
        }
        for p in spec["points"]
        if point_id(p) not in ids
    ]
    result = {
        "schema": 1,
        "policy": POLICY,
        "algorithm": ALGORITHM,
        "created_at": utc_now(),
        "scope": scope,
        "curve_date": rows[0]["curve_date"],
        "curve_workflow_run_id": rows[0]["curve_workflow_run_id"],
        "source": {
            key: snapshot[key] for key in ("source_url", "fetched_at", "response_sha256", "payload_sha256")
        },
        "workflows": payload["workflows"],
        "local_inferencex_sha": spec["provenance"]["inferencex"]["sha"],
        "candidate_count": len(spec["points"]),
        "upstream_count": len(rows),
        "selected": mapped,
        "unselected": unselected,
        "errors": errors,
        "points": [point_id(p) for p in spec["points"] if point_id(p) in ids],
        "spec_sha256": sha256_json({k: v for k, v in spec.items() if k != "selection"}),
    }
    # Snapshot dates do not change selection identity.
    result["selection_sha256"] = sha256_json(
        {
            "scope": scope,
            "algorithm": ALGORITHM,
            "points": sorted((p["point_id"], p["recipe_fingerprint"]) for p in mapped),
        }
    )
    result["manifest_sha256"] = sha256_json(result)
    return result


def prepare(spec: dict, args: argparse.Namespace, out: Path) -> dict:
    validate_options(args)
    if getattr(args, "point_policy", "all") == "all":
        return {"policy": "all", "duration": getattr(args, "duration", None)}
    scope = scope_of(spec, args.inferencex_hardware)
    snapshot = acquire_snapshot(args, scope)
    write_json(out / "upstream-snapshot.json", snapshot)
    selection = build_selection(spec, snapshot, scope)
    selection["duration"] = getattr(args, "duration", None)
    selection["manifest_sha256"] = sha256_json({k: v for k, v in selection.items() if k != "manifest_sha256"})
    write_json(out / "selection.json", selection)
    if selection["errors"]:
        raise SelectionError("; ".join(selection["errors"]))
    if not selection["points"]:
        raise SelectionError("Upstream frontier selected no local points")
    return selection


def validate_manifest(selection: dict) -> None:
    if selection.get("policy") != POLICY or selection.get("algorithm") != ALGORITHM:
        raise SelectionError("Unsupported Pareto selection manifest")
    if selection.get("manifest_sha256") != sha256_json(
        {k: v for k, v in selection.items() if k != "manifest_sha256"}
    ):
        raise SelectionError("Selection manifest was modified; regenerate the plan")
    if selection.get("errors") or not selection.get("points"):
        raise SelectionError("Selection manifest is empty or has mapping failures")


def selected_points(spec: dict) -> list[dict]:
    selection = spec.get("selection", {})
    if selection.get("policy", "all") == "all":
        points = spec["points"]
        arms = selection.get("arms", "all")
        if arms != "all":
            wanted = set(arms.split(","))
            if wanted - set(spec["arms"]):
                raise SelectionError(f"unknown arms: {sorted(wanted - set(spec['arms']))}")
            points = [p for p in points if p["arm"] in wanted]
        if selection.get("points"):
            wanted = set(selection["points"].split(","))
            points = [p for p in points if point_id(p) in wanted]
            missing = wanted - {point_id(p) for p in points}
            if missing:
                raise SelectionError(f"no such points: {sorted(missing)}")
        if not points:
            raise SelectionError("plan selects no points")
        return points
    validate_manifest(selection)
    if selection["spec_sha256"] != sha256_json({k: v for k, v in spec.items() if k != "selection"}):
        raise SelectionError("Serving configuration changed after selection; regenerate the plan")
    wanted = set(selection["points"])
    points = [p for p in spec["points"] if point_id(p) in wanted]
    if len(points) != len(wanted):
        raise SelectionError("Selected points no longer exist in the serving plan")
    return points
