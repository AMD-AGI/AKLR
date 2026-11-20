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
  TRITON_CACHE_DIR="$cache/triton" AKLR_KERNEL_CACHE_DIR="$cache/aklr-kernels" PIP_CACHE_DIR="$cache/pip"
framework=$(python3 -c 'import json, os; print(json.load(open(os.environ["AGENTX_SERVING"]))["framework"]["id"])')
# vLLM: both modes serve on stack.json's AITER (with its aiter.plugin and the FlyDSL it pins), as aiter-test does, so
# the aiter provider's routes are reachable; AKLR_DISABLE=1 sends them all back to AITER in the AKLR-off mode.
# SGLang keeps its image's AITER: on stack.json's, v0.5.23's Kimi-K3 MoE call (SiTUv2, A8W4) computes wrong and
# AITER's Triton MLA kernels need a newer Triton than the image's.
if [[ $framework == vllm ]]; then
# not under the image's site-packages; SGLang's image AITER keeps its own prebuilt modules
export AITER_JIT_DIR="$cache/aiter-jit"
AKLR_STACK_AITER=$(python3 -c 'import json; print(json.load(open("stack.json"))["aiter"])')
image_triton=$(python3 -c 'import torch, triton; print(triton.__version__)')
AITER_USE_SYSTEM_TRITON=1 python3 -m pip install "amd-aiter @ git+https://github.com/ROCm/aiter.git@$AKLR_STACK_AITER"
triton=$(python3 -c 'import torch, triton; print(triton.__version__)')
[[ $triton == "$image_triton" ]] || { echo "The image's Triton $image_triton became $triton" >&2; exit 1; }
AKLR_STACK_AITER=$AKLR_STACK_AITER python3 - <<'PY'
import json
import os
from importlib.metadata import distribution, version
from pathlib import Path

import aiter

dist = distribution("amd-aiter")
installed = Path(dist.locate_file("aiter/__init__.py")).resolve()
assert Path(aiter.__file__).resolve() == installed, aiter.__file__
commit = json.loads(dist.read_text("direct_url.json"))["vcs_info"]["commit_id"]
assert commit == os.environ["AKLR_STACK_AITER"], commit
print("amd-aiter", version("amd-aiter"), "at", installed, "; flydsl", version("flydsl"))
PY
fi
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
for name in FLYDSL_RUNTIME_CACHE_DIR FLYDSL_AUTOTUNE_CACHE_DIR TRITON_CACHE_DIR AKLR_KERNEL_CACHE_DIR PIP_CACHE_DIR \
  AITER_JIT_DIR; do
  [[ -z ${!name:-} ]] || printf '%s=%s\n' "$name" "${!name}"
done >> "$out"
echo "AKLR installed from $root; environment in $out"
