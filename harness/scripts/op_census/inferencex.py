"""The InferenceX cookbook: fetch it, list a target's AgentX points and bind each one to the exact recipe it ran.

Binding uses InferenceX's own code, run in a child process inside the checkout (``_bind_child``): the matrix
generator (``infx.matrix.generate test-config``) gives each point's matrix row, ``bound_variant`` composes the
recipe file with its shared blocks and selects the variant for that concurrency, and ``build_overrides`` adds the
synthetic acceptance an AgentX speculative run forces (``srtctl apply`` does that at launch). Nothing here
re-implements the recipe format, so a point binds the way InferenceX would submit it.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = "SemiAnalysisAI/InferenceX"
SPARSE = (
    "/inferencex-e2e/infx/",
    "/inferencex-e2e/configs/",
    "/inferencex-e2e/benchmarks/",
    "/inferencex-e2e/pyproject.toml",
    "/inferencex-e2e/uv.lock",
)
FRAMEWORKS = ("sglang", "vllm")


class CookbookError(Exception):
    pass


def _git(args: list[str], cwd: Path, timeout: int = 300) -> str:
    done = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False
    )
    if done.returncode:
        raise CookbookError(f"git {' '.join(args)}: {done.stderr.strip()[-400:]}")
    return done.stdout


def fetch(ref: str, dst: Path) -> str:
    """A clean sparse checkout of InferenceX at ``ref`` (branch, tag or commit) in ``dst``; returns the commit."""
    if (dst / ".git").is_dir():
        try:
            head = _git(["rev-parse", "HEAD"], dst).strip()
            if head.startswith(ref) and not _git(["status", "--porcelain"], dst).strip():
                return head
        except CookbookError:
            pass
    shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True)
    _git(["init", "-q"], dst)
    _git(["remote", "add", "origin", f"https://github.com/{REPO}.git"], dst)
    _git(["fetch", "-q", "--depth=1", "--filter=blob:none", "origin", ref], dst, timeout=600)
    _git(["sparse-checkout", "set", "--no-cone", *SPARSE], dst)
    _git(["checkout", "-q", "--detach", "FETCH_HEAD"], dst, timeout=600)
    return _git(["rev-parse", "HEAD"], dst).strip()


def config_keys(checkout: Path, target: str = "mi355x", master: str = "amd-master.yaml") -> list[str]:
    """Every single-node AgentX config of ``target`` served by SGLang or vLLM, in file order."""
    import yaml

    configs = yaml.safe_load((checkout / "inferencex-e2e/configs" / master).read_text())
    keys = []
    for key, value in configs.items():
        agentic = any(s == "agentic-coding" for s in (value.get("scenarios") or {}))
        if (
            f"-{target}-" in key
            and agentic
            and value.get("framework") in FRAMEWORKS
            and not value.get("multinode")
        ):
            keys.append(key)
    return keys


_CHILD = r"""
import json, subprocess, sys
from pathlib import Path
from infx.srt_slurm.generate import bound_variant, point_environment
from infx.srt_slurm.variants import expand_variants
from infx.srt_slurm.synthetic_acceptance import build_overrides, acceptance_length
master, out, keys = sys.argv[1], sys.argv[2], sys.argv[3:]
root = Path(".").resolve()
points = []
for key in keys:
    rows = json.loads(subprocess.run(
        [sys.executable, "-m", "infx.matrix.generate", "test-config", "--config-files", f"configs/{master}",
         "--config-keys", key], capture_output=True, text=True, check=True).stdout)
    for row in rows:
        env = point_environment(row)
        variant, recipe = bound_variant(row, env, root, expand=expand_variants)
        framework = env["FRAMEWORK"]
        points.append({
            "config_key": key, "row": row, "environment": env, "variant": variant, "recipe": recipe,
            "overrides": build_overrides(recipe, framework, env),
            "acceptance_length": acceptance_length(recipe, framework, env),
        })
json.dump(points, open(out, "w"), indent=1, default=str)
"""


def bind(checkout: Path, keys: list[str], master: str = "amd-master.yaml") -> list[dict]:
    """Each key's matrix rows, bound by InferenceX; see the module docstring."""
    project = checkout / "inferencex-e2e"
    out = project.parent / ".op_census_points.json"
    env = {**os.environ, "PYTHONPATH": str(project), "PYTHONDONTWRITEBYTECODE": "1"}
    done = subprocess.run(
        [sys.executable, "-c", _CHILD, master, str(out), *keys],
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    if done.returncode:
        raise CookbookError(f"binding failed: {done.stderr.strip()[-2000:]}")
    points = json.loads(out.read_text())
    out.unlink()
    for p in points:
        p["recipe"] = apply_overrides(p["recipe"], p.pop("overrides"))
    return points


def apply_overrides(recipe: dict, overrides: list[str]) -> dict:
    """``srtctl apply``'s ``--set path=<json>`` and ``--unset path`` on a copy of ``recipe``.

    Paths are dotted; a value is JSON (``build_overrides`` writes every value with ``json.dumps``).
    """
    recipe = json.loads(json.dumps(recipe))
    it = iter(overrides)
    for flag in it:
        operand = next(it)
        if flag == "--set":
            path, _, raw = operand.partition("=")
            keys = path.split(".")
            node = recipe
            for k in keys[:-1]:
                node = node.setdefault(k, {})
            node[keys[-1]] = json.loads(raw)
        elif flag == "--unset":
            keys = operand.split(".")
            node = recipe
            for k in keys[:-1]:
                node = node.get(k) or {}
            if isinstance(node, dict):
                node.pop(keys[-1], None)
        else:
            raise CookbookError(f"unknown srtctl override {flag!r}")
    return recipe
