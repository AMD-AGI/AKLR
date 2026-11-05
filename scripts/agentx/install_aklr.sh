#!/usr/bin/env bash
set -euo pipefail

cache=${1:?usage: install_aklr.sh <cache dir>}
root=$(pwd -P)
git config --global --add safe.directory '*'
mkdir -p "$cache"
cache=$(cd "$cache" && pwd -P)
AKLR_BASELINE_AITER=$(sed -n 's/^  AKLR_BASELINE_AITER: *//p' .github/workflows/ci.yaml)
[[ $AKLR_BASELINE_AITER =~ ^[0-9a-f]{40}$ ]] || { echo "no AKLR_BASELINE_AITER sha in ci.yaml: '$AKLR_BASELINE_AITER'" >&2; exit 1; }
export AKLR_BASELINE_AITER

GPU_ARCHS=$(python3 - <<'PY'
import torch
assert torch.version.hip, "the AgentX sweep requires ROCm PyTorch"
assert torch.cuda.is_available(), "no GPU is visible to PyTorch"
archs = {torch.cuda.get_device_properties(i).gcnArchName.split(":", 1)[0] for i in range(torch.cuda.device_count())}
print(";".join(sorted(archs)))
PY
)
export GPU_ARCHS
export AITER_JIT_DIR="$cache/aiter-jit" FLYDSL_RUNTIME_CACHE_DIR="$cache/flydsl-runtime" \
  FLYDSL_AUTOTUNE_CACHE_DIR="$cache/flydsl-autotune" TRITON_CACHE_DIR="$cache/triton" \
  AKLR_KERNEL_CACHE_DIR="$cache/aklr-kernels" PIP_CACHE_DIR="$cache/pip"
AITER_USE_SYSTEM_TRITON=1 python3 -m pip install "amd-aiter @ git+https://github.com/ROCm/aiter.git@$AKLR_BASELINE_AITER"
python3 -m pip install -e .
python3 - <<'PY'
import os
from importlib.metadata import distribution, entry_points, version
from pathlib import Path

import aiter
import aklr
import triton

if triton.__version__.split('+')[0] != '3.8.0':
    raise RuntimeError(f'triton==3.8.0 is required, found {triton.__version__}')
root = Path.cwd().resolve()
assert Path(aklr.__file__).resolve().is_relative_to(root), aklr.__file__
installed = Path(distribution("amd-aiter").locate_file("aiter/__init__.py")).resolve()
assert aiter.__file__ and Path(aiter.__file__).resolve() == installed, aiter.__file__
assert f"+g{os.environ['AKLR_BASELINE_AITER'][:7]}" in version("amd-aiter"), version("amd-aiter")
assert any(ep.name == "aklr" for ep in entry_points(group="sglang.srt.plugins")), "sglang will not load AKLR"
PY

out=${GITHUB_ENV:-$cache/aklr.env}
[[ -n ${GITHUB_ENV:-} ]] || : > "$out"
for name in GPU_ARCHS AITER_JIT_DIR FLYDSL_RUNTIME_CACHE_DIR FLYDSL_AUTOTUNE_CACHE_DIR TRITON_CACHE_DIR \
  AKLR_KERNEL_CACHE_DIR PIP_CACHE_DIR; do
  printf '%s=%s\n' "$name" "${!name}"
done >> "$out"
echo "AKLR installed from $root; environment in $out"
