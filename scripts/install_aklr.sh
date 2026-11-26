#!/usr/bin/env bash
# Install this checkout into the serving image as it ships, as a user does, with its kernel caches under <cache dir>,
# and check that the framework will load AKLR's plugin. Shared by AgentX and the daily e2e perf. The serving AITER is
# the image's own unless the third argument is "stack" (a daily e2e recipe's "aiter").
set -euo pipefail

cache=${1:?usage: install_aklr.sh <cache dir> <sglang|vllm> [image|stack]}
framework=${2:?usage: install_aklr.sh <cache dir> <sglang|vllm> [image|stack]}
aiter=${3:-image}
case $framework in
  sglang) group=sglang.srt.plugins ;;
  vllm) group=vllm.general_plugins ;;
  *) echo "unsupported framework: $framework" >&2; exit 2 ;;
esac
case $aiter in
  image | stack) ;;
  *) echo "unsupported AITER: $aiter" >&2; exit 2 ;;
esac
root=$(pwd -P)
git config --global --add safe.directory '*'
mkdir -p "$cache"
cache=$(cd "$cache" && pwd -P)
export FLYDSL_RUNTIME_CACHE_DIR="$cache/flydsl-runtime" FLYDSL_AUTOTUNE_CACHE_DIR="$cache/flydsl-autotune" \
  TRITON_CACHE_DIR="$cache/triton" AKLR_KERNEL_CACHE_DIR="$cache/aklr-kernels" PIP_CACHE_DIR="$cache/pip"
# stack: stack.json's AITER over the image's, as aiter-test installs it (its aiter.plugin, the FlyDSL it pins, the
# image's Triton kept), with its built modules under the cache; both modes of a run then serve on it and differ in
# AKLR alone.
if [[ $aiter == stack ]]; then
  export AITER_JIT_DIR="$cache/aiter-jit"
  stack_aiter=$(python3 -c 'import json; print(json.load(open("stack.json"))["aiter"])')
  image_triton=$(python3 -c 'import torch, triton; print(triton.__version__)')
  AITER_USE_SYSTEM_TRITON=1 python3 -m pip install "amd-aiter @ git+https://github.com/ROCm/aiter.git@$stack_aiter"
  triton=$(python3 -c 'import torch, triton; print(triton.__version__)')
  [[ $triton == "$image_triton" ]] || { echo "The image's Triton $image_triton became $triton" >&2; exit 1; }
  STACK_AITER=$stack_aiter python3 - <<'PY'
import json
import os
from importlib.metadata import distribution, version
from pathlib import Path

import aiter

dist = distribution("amd-aiter")
installed = Path(dist.locate_file("aiter/__init__.py")).resolve()
assert Path(aiter.__file__).resolve() == installed, aiter.__file__
commit = json.loads(dist.read_text("direct_url.json"))["vcs_info"]["commit_id"]
assert commit == os.environ["STACK_AITER"], commit
print("amd-aiter", version("amd-aiter"), "at", installed, "; flydsl", version("flydsl"))
PY
  # SGLang v0.5.23 tells AITER gate_mode "separated" for SiTU while it holds the MoE's w13 gate/up-interleaved under
  # AITER_SITUV2_A8W4. Its image's AITER reads that layout whatever gate_mode says; stack.json's reads the layout
  # gate_mode names (ROCm/aiter#6464), so the call computes wrong there. SGLang passes INTERLEAVE since
  # sgl-project/sglang#51000: that fix on the image's runner, for both modes; a SGLang that has it is left alone.
  if [[ $framework == sglang ]]; then
    python3 - <<'PY'
import re
from importlib.util import find_spec
from pathlib import Path

path = Path(find_spec("sglang").origin).parent / "srt/layers/moe/moe_runner/aiter.py"
text = path.read_text()
old = '            extra["gate_mode"] = GateMode.SEPARATED.value\n            if self.config.gemm1_alpha is not None:\n'
new = (
    "            from sglang.srt.layers.quantization.mxfp4 import _aiter_situ_uses_gu_interleaved_weights\n\n"
    "            interleaved = is_gfx95 and _aiter_situ_uses_gu_interleaved_weights()\n"
    '            extra["gate_mode"] = (GateMode.INTERLEAVE if interleaved else GateMode.SEPARATED).value\n'
    "            if self.config.gemm1_alpha is not None:\n"
)
if new in text:
    print("SGLang's SiTU call already passes the weights' gate/up layout:", path)
elif text.count("\n" + old) == 1:
    # v0.5.23's assignment at its own indent: a later SGLang holds the same words one level deeper
    path.write_text(text.replace("\n" + old, "\n" + new))
    print("SGLang's SiTU call now passes the weights' gate/up layout:", path)
else:
    # a SGLang with the fix of its own: it names the layout where it sets gate_mode, or keeps the one its caller
    # passed (sgl-project/sglang f8fd250)
    own = re.search(r'^ *extra\["gate_mode"\] = .*INTERLEAVE', text, re.M) or '"gate_mode" in extra' in text
    assert own, f"{path}: SiTU's gate_mode line is not v0.5.23's"
    print("SGLang's SiTU call passes the weights' gate/up layout by itself:", path)
PY
  fi
fi
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
for name in FLYDSL_RUNTIME_CACHE_DIR FLYDSL_AUTOTUNE_CACHE_DIR TRITON_CACHE_DIR AKLR_KERNEL_CACHE_DIR PIP_CACHE_DIR \
  AITER_JIT_DIR; do
  [[ -z ${!name:-} ]] || printf '%s=%s\n' "$name" "${!name}"
done >> "$out"
echo "AKLR installed from $root for $framework; environment in $out"
