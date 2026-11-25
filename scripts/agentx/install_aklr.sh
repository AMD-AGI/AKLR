#!/usr/bin/env bash
# Install this checkout into the serving image as it ships, as a user does, with its kernel caches under <cache dir>.
set -euo pipefail

cache=${1:?usage: install_aklr.sh <cache dir> <serving.json>}
AGENTX_SERVING=${2:?usage: install_aklr.sh <cache dir> <serving.json>}
export AGENTX_SERVING
root=$(pwd -P)
git config --global --add safe.directory '*'
mkdir -p "$cache"
cache=$(cd "$cache" && pwd -P)
export FLYDSL_RUNTIME_CACHE_DIR="$cache/flydsl-runtime" FLYDSL_AUTOTUNE_CACHE_DIR="$cache/flydsl-autotune" \
  TRITON_CACHE_DIR="$cache/triton" AKLR_BUILD_CACHE_DIR="$cache/aklr-kernels" PIP_CACHE_DIR="$cache/pip"
python3 -m pip install -e .
python3 - <<'PY'
import json
import os
from importlib.metadata import entry_points
from pathlib import Path

import aklr

assert Path(aklr.__file__).resolve().is_relative_to(Path.cwd().resolve()), aklr.__file__
serving = json.loads(Path(os.environ["AGENTX_SERVING"]).read_text())
framework = serving["framework"]["id"]
assert all(p["client_env"]["FRAMEWORK"] == framework for p in serving["points"]), f"points must use {framework}"
group = {"sglang": "sglang.srt.plugins", "vllm": "vllm.general_plugins"}[framework]
assert any(ep.name == "aklr" for ep in entry_points(group=group)), f"{framework} will not load AKLR"
PY

python3 - <<'PY'
import json
import os
import subprocess
import sys
from pathlib import Path

if json.loads(Path(os.environ["AGENTX_SERVING"]).read_text())["framework"]["id"] != "sglang":
    sys.exit(0)
try:
    from sglang.srt.mem_cache.unified_cache import tree_core_registry as registry
except ImportError:
    print("Rust TreeCore warmup skipped: this SGLang has no TreeCore registry")
    sys.exit(0)

try:
    manifest = registry._RUST_TREE_CORE_MANIFEST
    if manifest.is_file():
        for command in (("cargo", "--version", "--verbose"), ("rustc", "-vV")):
            subprocess.run(command, cwd=manifest.parent.parent, check=True)
    import sglang.srt.mem_cache.rust_tree_core.extension  # noqa: F401
except Exception as exc:
    print(f"WARNING: Rust TreeCore warmup failed; servers will resolve it themselves: {exc!r}", file=sys.stderr)
else:
    print("Rust TreeCore warmed up")
PY

out=${GITHUB_ENV:-$cache/aklr.env}
[[ -n ${GITHUB_ENV:-} ]] || : > "$out"
for name in FLYDSL_RUNTIME_CACHE_DIR FLYDSL_AUTOTUNE_CACHE_DIR TRITON_CACHE_DIR AKLR_BUILD_CACHE_DIR PIP_CACHE_DIR; do
  printf '%s=%s\n' "$name" "${!name}"
done >> "$out"
echo "AKLR installed from $root; environment in $out"
