# SPDX-License-Identifier: MIT
"""Resolve the InferenceX recipe into one launch spec per point (``serving.json``).

InferenceX@ref (partial + sparse clone) → its pinned deps from uv.lock → InferenceX's own
matrix generation and point binding (``ix_points.py``) → render each point's engine args
with srtctl's rule → add the advisory and runtime layers → hashes and provenance.

Needs only the standard library, PyYAML, git and pip.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import sysconfig
import time
from pathlib import Path

import yaml

from agentx_ci.common import (
    DEFAULT_CONFIG,
    REPO_ROOT,
    load_config,
    sha256_json,
    utc_now,
    write_json,
)

IX_SPARSE = (
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
# srtctl's worker command writes its own values for these.
WORKER_OWNED_ARGS = {
    "model-path",
    "served-model-name",
    "nccl-port",
    "grpc-http-sidecar-port",
    "smg-http-sidecar-port",
}
# Bound per point by the launcher (or by sweep.py on our host), never by the recipe.
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
    "KV_OFFLOADING",
)
# One single-node AgentX point with power telemetry off, as the client validates it.
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
    done = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False
    )
    if done.returncode:
        raise ResolveError(f"{shlex.join(cmd)} failed: {done.stderr.strip()[-400:]}")
    return done.stdout


def github_url(repo: str) -> str:
    return f"https://github.com/{repo}.git"


def resolve_ref(repo: str, ref: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", ref):
        return ref
    out = sh(
        ["git", "ls-remote", github_url(repo), f"refs/heads/{ref}", f"refs/tags/{ref}"],
        timeout=60,
    )
    shas = {line.split()[0] for line in out.splitlines() if line.strip()}
    if len(shas) != 1:
        raise ResolveError(f"{repo}@{ref} resolved to {len(shas)} commits")
    return shas.pop()


def fetch_inferencex(
    repo: str, sha: str, dst: Path, retries: tuple[int, ...] = (15, 30, 60)
) -> Path:
    """Check out ``sha`` of ``repo`` into ``dst`` with only the files resolution and the client read."""
    if dst.is_dir() and sh(["git", "rev-parse", "HEAD"], cwd=dst).strip() == sha:
        return dst
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
            sh(["git", "sparse-checkout", "set", "--no-cone", *IX_SPARSE], cwd=dst)
            sh(["git", "checkout", "-q", "--detach", "FETCH_HEAD"], cwd=dst)
            return dst
        except (ResolveError, subprocess.TimeoutExpired) as e:
            error = e
    raise ResolveError(
        f"fetching {repo}@{sha} failed after {len(retries) + 1} attempts: {error}"
    )


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
    # Binary wheels (pydantic-core) are per interpreter, so the interpreter is part of the key.
    abi = f"{sys.implementation.cache_tag}-{sysconfig.get_platform()}"
    target = cache / f"ixdeps-{sha256_json([versions, abi])[:16]}"
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


def run_ix_points(
    ix_e2e: Path, deps: Path, config_file: str, key: str, timeout: int = 120
) -> dict:
    """Run InferenceX's resolution in a child with only its own code and deps on the path."""
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HOME": "/nonexistent",
        "LANG": "C.UTF-8",
    }
    # -I ignores PYTHONPATH, so the bootstrap puts InferenceX and its deps first on sys.path.
    script = Path(__file__).with_name("ix_points.py")
    path, argv = [str(deps), str(ix_e2e)], ["ix_points.py", config_file, key]
    boot = (
        f"import runpy, sys; sys.path[:0] = {path!r}; sys.argv = {argv!r}; "
        f"runpy.run_path({str(script)!r}, run_name='__main__')"
    )
    done = subprocess.run(
        [sys.executable, "-I", "-c", boot],
        cwd=ix_e2e,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        check=False,
    )
    if done.returncode:
        raise ResolveError(
            f"InferenceX resolution failed: {done.stderr.strip()[-2000:]}"
        )
    return json.loads(done.stdout)


def config_to_cli_args(config: dict) -> list[str]:
    """srtctl.backends.sglang._config_to_cli_args: sorted keys, ``_`` → ``-``, bare flags for true."""
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
    """Apply ``build_overrides``' ``--set``/``--unset`` pairs (dotted paths, JSON values) as srtctl does."""
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
    """The ``AIPERF_*`` defaults InferenceX's workflow exports for every client."""
    pairs = re.findall(r"^export (AIPERF_\w+)='([^']*)'$", text, re.MULTILINE)
    return dict(pairs)


def arm_of(row: dict, arms: dict) -> str:
    for name, selector in arms.items():
        if all(str(row.get(k)) == str(v) for k, v in selector.items()):
            return name
    return f"tp{row['tp']}-ep{row['ep']}-{row['kv-offloading']}"


def resolve_point(
    point: dict, arm: str, config: dict, client_defaults: dict[str, str]
) -> dict:
    row, env = point["row"], point["environment"]
    recipe = apply_golden(point["bound_recipe"], point["golden_overrides"])
    role = recipe["roles"]["agg"]
    args = dict(role.get("args") or {})
    engine_env = {**(recipe.get("environment") or {}), **(role.get("env") or {})}
    advisory = config.get("advisory_env") or {}
    clash = sorted(set(advisory) & set(engine_env))
    if clash:
        raise ResolveError(f"advisory env {clash} would override what the recipe sets")
    cli = config_to_cli_args(
        {k: v for k, v in args.items() if k not in WORKER_OWNED_ARGS}
    )
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
    client_env = {
        **client_defaults,
        **{name: env[name] for name in POINT_CLIENT_ENV},
        **({"KV_OFFLOAD_BACKEND": backend} if backend else {}),
        "DURATION": str(row["duration"]),
        **{k: str(v) for k, v in bench_env.items() if k not in BOUND_CLIENT_ENV},
        **FIXED_CLIENT_ENV,
    }
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
        "hashes": {
            "serve_spec_sha256": sha256_json(serve),
            "inferencex_recipe_fingerprint": row["recipe-fingerprint"],
        },
    }


def check_entry(ixout: dict, expect: dict) -> list[str]:
    """Raise when the entry stops being the model we measure; return warnings for grid changes."""
    entry, points = ixout["entry"], ixout["points"]
    if not points:
        raise ResolveError("InferenceX generated no points")
    for key in ("model", "framework", "precision", "model-prefix"):
        if str(entry.get(key)) != str(expect[key]):
            raise ResolveError(
                f"entry {key}={entry.get(key)!r}, expected {expect[key]!r}"
            )
    if not re.search(expect["runner_pattern"], str(entry.get("runner"))):
        raise ResolveError(
            f"runner {entry.get('runner')!r} does not match {expect['runner_pattern']!r}"
        )
    warnings = []
    if len(points) != expect["points"]:
        warnings.append(
            f"InferenceX now defines {len(points)} points, not {expect['points']}"
        )
    return warnings


def build_spec(
    ixout: dict,
    config: dict,
    *,
    run_image: str,
    client_defaults: dict[str, str],
    provenance: dict,
) -> dict:
    """Pure: InferenceX's points → serving.json."""
    source = config["inferencex"]
    warnings = check_entry(ixout, source["expect"])
    points = []
    for point in ixout["points"]:
        arm = arm_of(point["row"], source["arms"])
        if arm not in source["arms"]:
            warnings.append(
                f"{point['row']['exp-name']}: topology matches no arm; new arm {arm}"
            )
        points.append(resolve_point(point, arm, config, client_defaults))
    points.sort(key=lambda q: (q["arm"], q["conc"]))
    arms = {}
    for name in dict.fromkeys(q["arm"] for q in points):
        mine = [q for q in points if q["arm"] == name]
        arms[name] = {
            "selector": source["arms"].get(name),
            "concs": [q["conc"] for q in mine],
            "arm_serve_sha256": sha256_json(
                {q["conc"]: q["hashes"]["serve_spec_sha256"] for q in mine}
            ),
        }
    ix_image = ixout["entry"]["image"]
    if ix_image != run_image:
        warnings.append(f"run image {run_image} differs from InferenceX's {ix_image}")
    return {
        "schema": 1,
        "provenance": provenance,
        "image": {
            "run": run_image,
            "inferencex": ix_image,
            "differs": ix_image != run_image,
        },
        "weights": config["weights"],
        "arms": arms,
        "grid_sha256": sha256_json([[q["arm"], q["conc"]] for q in points]),
        "points": points,
        "warnings": warnings,
    }


def registry_digest(image: str) -> str | None:
    """The tag's current manifest digest from the registry, else from the local image store."""
    try:
        listing = sh(["docker", "buildx", "imagetools", "inspect", image], timeout=120)
        for line in listing.splitlines():
            key, _, value = line.partition(":")
            if key.strip() == "Digest" and value.strip().startswith("sha256:"):
                return value.strip()
        repo_digests = json.loads(
            sh(["docker", "image", "inspect", "--format", "{{json .RepoDigests}}", image])
        )
    except (ResolveError, OSError, ValueError, subprocess.TimeoutExpired):
        return None
    for reference in repo_digests or []:
        if "@sha256:" in reference:
            return reference.split("@", 1)[1]
    return None


def ci_image(workflow: Path, lookup=registry_digest) -> tuple[str, list[str]]:
    """AKLR_CI_IMAGE pinned to a digest, and a warning when ci.yaml pins none itself."""
    image = yaml.safe_load(workflow.read_text())["env"]["AKLR_CI_IMAGE"]
    if "@sha256:" in image:
        return image, []
    digest = lookup(image)
    if digest is None:
        return image, [f"AKLR_CI_IMAGE {image} pins no digest and none could be looked up"]
    return f"{image}@{digest}", [
        f"AKLR_CI_IMAGE {image} pins no digest; pinned the tag's digest at resolve time"
    ]


def git_blob(tree: Path, path: str) -> str:
    return sh(["git", "hash-object", "--", path], cwd=tree).strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m agentx_ci resolve", description=__doc__
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="run definition")
    parser.add_argument(
        "--ix-ref", help="InferenceX branch, tag or sha (default: the config's ref)"
    )
    parser.add_argument(
        "--ix-src", type=Path, help="use this InferenceX checkout instead of fetching"
    )
    parser.add_argument(
        "--run-image", help="default: AKLR_CI_IMAGE of .github/workflows/ci.yaml"
    )
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="serving.json to write")
    args = parser.parse_args(argv)
    try:
        spec = resolve(args)
    except ResolveError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    summary = {
        "inferencex_sha": spec["provenance"]["inferencex"]["sha"],
        "points": len(spec["points"]),
        "arms": {name: arm["concs"] for name, arm in spec["arms"].items()},
        "warnings": spec["warnings"],
    }
    print(json.dumps(summary, indent=1))
    return 0


def resolve(args: argparse.Namespace) -> dict:
    config = load_config(args.config)
    source = config["inferencex"]
    run_image, image_warnings = (
        (args.run_image, [])
        if args.run_image
        else ci_image(REPO_ROOT / ".github/workflows/ci.yaml")
    )
    if args.ix_src:
        ix = args.ix_src
        sha = sh(["git", "rev-parse", "HEAD"], cwd=ix).strip()
    else:
        sha = resolve_ref(source["repo"], args.ix_ref or source["ref"])
        ix = fetch_inferencex(source["repo"], sha, args.work / "inferencex")
    expect = source["expect"]
    for key in ("config_file", "recipe"):
        if not (ix / expect[key]).is_file():
            raise ResolveError(f"{expect[key]} not found at InferenceX {sha}")
    e2e = ix / "inferencex-e2e"
    versions = locked_versions((e2e / "uv.lock").read_text())
    deps = ensure_deps(versions, args.work / "cache")
    config_file = str(Path(expect["config_file"]).relative_to("inferencex-e2e"))
    ixout = run_ix_points(e2e, deps, config_file, source["config_key"])
    settings = ix / RUNTIME_SETTINGS
    client_defaults = (
        runtime_settings(settings.read_text()) if settings.is_file() else {}
    )
    commit_time, subject = (
        sh(["git", "log", "-1", "--format=%cI%x00%s"], cwd=ix).strip().split("\0")
    )
    sources = [expect["config_file"], expect["recipe"], RUNTIME_SETTINGS]
    provenance = {
        "inferencex": {
            "repo": source["repo"],
            "requested_ref": args.ix_ref or source["ref"],
            "sha": sha,
            "commit_time": commit_time,
            "commit_subject": subject,
            "config_key": source["config_key"],
            "sources": {
                path: git_blob(ix, path) for path in sources if (ix / path).is_file()
            },
            "deps": versions,
        },
        "resolved_at": utc_now(),
    }
    spec = build_spec(
        ixout,
        config,
        run_image=run_image,
        client_defaults=client_defaults,
        provenance=provenance,
    )
    spec["warnings"] = image_warnings + spec["warnings"]
    write_json(args.out, spec)
    return spec
