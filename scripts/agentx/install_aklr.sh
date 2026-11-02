#!/usr/bin/env bash
# Install this checkout into the AKLR CI image the way ci.yaml's setup_aklr does, for the AgentX sweep.
#
#   scripts/agentx/install_aklr.sh <cache dir>
#
# Run as root inside the container, from the repository root. Writes the kernel cache variables the
# server needs to $GITHUB_ENV when set, else to <cache dir>/aklr.env (source it with set -a).
set -euo pipefail

cache=${1:?usage: install_aklr.sh <cache dir>}
root=$(pwd -P)
git config --global --add safe.directory '*'
mkdir -p "$cache"
cache=$(cd "$cache" && pwd -P)

GPU_ARCHS=$(python3 - <<'PY'
import sys
import torch
assert torch.version.hip, "the AgentX sweep requires ROCm PyTorch"
assert torch.cuda.is_available(), "no GPU is visible to PyTorch"
archs = {torch.cuda.get_device_properties(i).gcnArchName.split(":", 1)[0] for i in range(torch.cuda.device_count())}
print(f"Python {sys.version.split()[0]}; PyTorch {torch.__version__}; ROCm {torch.version.hip}; GPUs {torch.cuda.device_count()}", file=sys.stderr)
print(";".join(sorted(archs)))
PY
)
export GPU_ARCHS
unset AITER_ASM_DIR
# Not aiter's default 3rdparty/aiter/aiter/jit: a killed build's lock there would block later runs.
export AITER_JIT_DIR="$cache/aiter-jit" FLYDSL_RUNTIME_CACHE_DIR="$cache/flydsl-runtime" \
  FLYDSL_AUTOTUNE_CACHE_DIR="$cache/flydsl-autotune" TRITON_CACHE_DIR="$cache/triton" \
  AKLR_KERNEL_CACHE_DIR="$cache/aklr-kernels" PIP_CACHE_DIR="$cache/pip"
echo "GPU_ARCHS=$GPU_ARCHS"

# With its dependencies: aiter's kernels need its pinned FlyDSL, not the image's.
python3 -m pip install -e .
python3 -c "import triton, sys; v = triton.__version__.split('+')[0]; print(f'triton {triton.__version__}'); sys.exit(0 if v == '3.8.0' else f'triton==3.8.0 is required, found {triton.__version__}')"
python3 - <<'PY'
from importlib.metadata import version
from pathlib import Path

import aiter
import aklr
from aklr.ops import load_routes

root = Path.cwd().resolve()
assert Path(aklr.__file__).resolve().is_relative_to(root), aklr.__file__
load_routes()
assert hasattr(aiter.gemm_a8w8, "aklr_proxy"), "AKLR did not proxy aiter"
assert Path(aiter.__file__).resolve().is_relative_to(root / "3rdparty" / "aiter"), aiter.__file__
print("amd-aiter", version("amd-aiter"), "with AKLR from", root)
PY

out=${GITHUB_ENV:-$cache/aklr.env}
[[ -n ${GITHUB_ENV:-} ]] || : > "$out"
for name in GPU_ARCHS AITER_JIT_DIR FLYDSL_RUNTIME_CACHE_DIR FLYDSL_AUTOTUNE_CACHE_DIR TRITON_CACHE_DIR \
  AKLR_KERNEL_CACHE_DIR PIP_CACHE_DIR; do
  printf '%s=%s\n' "$name" "${!name}"
done >> "$out"
echo "AKLR installed from $root; environment in $out"
