#!/usr/bin/env bash
# Install this checkout into the serving image as it ships, as a user does, with its kernel caches under <cache dir>,
# and check that the framework will load AKLR's plugin. Shared by AgentX and the daily e2e perf.
set -euo pipefail

cache=${1:?usage: install_aklr.sh <cache dir> <sglang|vllm>}
framework=${2:?usage: install_aklr.sh <cache dir> <sglang|vllm>}
case $framework in
  sglang) group=sglang.srt.plugins ;;
  vllm) group=vllm.general_plugins ;;
  *) echo "unsupported framework: $framework" >&2; exit 2 ;;
esac
root=$(pwd -P)
git config --global --add safe.directory '*'
mkdir -p "$cache"
cache=$(cd "$cache" && pwd -P)
export FLYDSL_RUNTIME_CACHE_DIR="$cache/flydsl-runtime" FLYDSL_AUTOTUNE_CACHE_DIR="$cache/flydsl-autotune" \
  TRITON_CACHE_DIR="$cache/triton" AKLR_KERNEL_CACHE_DIR="$cache/aklr-kernels" PIP_CACHE_DIR="$cache/pip"
python3 -m pip install -e .
python3 - "$group" <<'PY'
import sys
from importlib.metadata import entry_points
from pathlib import Path

import aklr

assert Path(aklr.__file__).resolve().is_relative_to(Path.cwd().resolve()), aklr.__file__
assert any(ep.name == "aklr" for ep in entry_points(group=sys.argv[1])), f"{sys.argv[1]} has no AKLR plugin"
PY

if [[ $framework == sglang ]]; then
  python3 - <<'PY'
import subprocess
import sys

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
fi

out=$cache/aklr.env
: > "$out"
for name in FLYDSL_RUNTIME_CACHE_DIR FLYDSL_AUTOTUNE_CACHE_DIR TRITON_CACHE_DIR AKLR_KERNEL_CACHE_DIR PIP_CACHE_DIR; do
  printf '%s=%s\n' "$name" "${!name}"
done >> "$out"
echo "AKLR installed from $root for $framework; environment in $out"
