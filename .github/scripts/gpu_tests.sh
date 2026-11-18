#!/usr/bin/env bash
# One GPU test job's unit tests and validation, run inside the job's stack image on a Primus-SaFE
# GPU workload that safe_job.sh submitted. The checkout is the current directory, on local disk.
#
# Environment: RUN_DIR (shared directory the reports go to), JOB, TESTS, VALIDATE, BASE,
# MEASURED_BY_BOT, and AKLR_BASELINE_AITER (aiter-test) or PLUGINS (sglang-test, vllm-test).
set -euo pipefail
: "${RUN_DIR:?}" "${JOB:?}" "${TESTS:?}"

# The runner user removes the run directory; this pod may write it as another user.
trap 'chmod -R a+rwX "$RUN_DIR" 2>/dev/null || true' EXIT

export PYTHONDONTWRITEBYTECODE=1
git config --global --add safe.directory '*'
state=$(dirname "$PWD")
cache=$state/cache
mkdir -p "$cache" "$RUN_DIR/test-reports"
# AITER JIT-builds for these archs.
GPU_ARCHS=$(python3 -c 'import torch; assert torch.version.hip and torch.cuda.is_available(), "no ROCm GPU"; print(";".join(sorted({torch.cuda.get_device_properties(i).gcnArchName.split(":")[0] for i in range(torch.cuda.device_count())})))')
echo "🔧 GPU_ARCHS=$GPU_ARCHS"
export GPU_ARCHS FLYDSL_RUNTIME_CACHE_DIR=$cache/flydsl-runtime FLYDSL_AUTOTUNE_CACHE_DIR=$cache/flydsl-autotune \
  TRITON_CACHE_DIR=$cache/triton XDG_CACHE_HOME=$cache/xdg AKLR_KERNEL_CACHE_DIR=$cache/aklr-kernels
# Bench task directories need a local disk (peak len(devices) × 16 GiB): flock, atomic renames, real free space.
export TMPDIR=$state

if [[ -n ${AKLR_BASELINE_AITER:-} ]]; then
  echo '::group::Install the baseline AITER and AKLR'
  # Not aiter's default, its site-packages tree.
  export AITER_JIT_DIR=$cache/aiter-jit
  # Keep the image's Triton: otherwise aiter's setup.py swaps in the one it pins.
  image_triton=$(python3 -c 'import torch, triton; print(triton.__version__)')
  AITER_USE_SYSTEM_TRITON=1 python3 -m pip install "amd-aiter @ git+https://github.com/ROCm/aiter.git@$AKLR_BASELINE_AITER"
  python3 -m pip install -e .
  triton=$(python3 -c 'import torch, triton; print(triton.__version__)')
  [[ $triton == "$image_triton" ]] || { echo "::error::The image's Triton $image_triton became $triton"; exit 1; }
  # The aiter Python imports is the one just installed, not the image's copy on sys.path.
  python3 - <<'PY'
import json
import os
from importlib.metadata import distribution, version
from pathlib import Path
import aiter
dist = distribution("amd-aiter")
installed = Path(dist.locate_file("aiter/__init__.py")).resolve()
assert Path(aiter.__file__).resolve() == installed, aiter.__file__
# pip's record of the commit it built: a release tag's version carries none
commit = json.loads(dist.read_text("direct_url.json"))["vcs_info"]["commit_id"]
assert commit == os.environ["AKLR_BASELINE_AITER"], commit
print("amd-aiter", version("amd-aiter"), "at", installed)
PY
else
  echo '::group::Install AKLR'
  python3 -m pip install -e .
fi
echo '::endgroup::'

echo '::group::Enable coverage and test tools'
# Install coverage over any copy the image ships, so pip never has to uninstall it.
python3 -m pip install --ignore-installed 'coverage[toml]==7.16.0'
python3 -m pip install 'pytest-timeout==2.4.0' 'expecttest==0.3.0'
site_packages=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
echo 'import coverage; coverage.process_startup()' > "$site_packages/coverage_subprocess.pth"
export COVERAGE_FILE=$state/.coverage COVERAGE_PROCESS_START=$PWD/pyproject.toml
echo '::endgroup::'

echo "::group::Run unit tests ($TESTS)"
tests=0
# shellcheck disable=SC2086 # TESTS is a list of paths
timeout -k 60s 25m python3 -m coverage run -m pytest -v $TESTS \
  --timeout=600 --timeout-method=thread \
  --junitxml="$RUN_DIR/test-reports/aklr.xml" \
  2>&1 | tee "$RUN_DIR/test-reports/aklr.log" || tests=$?
echo '::endgroup::'
# Coverage's tracer would skew the measurements.
unset COVERAGE_PROCESS_START
python3 -m coverage combine --append "$state"
cp "$COVERAGE_FILE" "$RUN_DIR/coverage_data_$JOB"
[[ $tests == 0 ]] || { echo "::error::Unit tests failed (exit $tests)"; exit "$tests"; }

[[ ${VALIDATE:-} == true ]] || exit 0
# The exit code is the conclusion; report.md's first line says which failure.
flags=(${BASE:+--base "$BASE"})
[[ ${MEASURED_BY_BOT:-} != true ]] || flags+=(--allow-measured-edits)
if [[ -n ${AKLR_BASELINE_AITER:-} ]]; then
  echo '::group::Validate aklr/ops'
  # FIXME: mega_moe's baseline JIT-builds mori's device bitcode, which misses TheRock's
  # <libdrm/drm.h> (mori/jit/core.py::_collect_include_dirs, ROCm/mori 29e0c086); measure
  # mega_moe again once the image ships a mori that finds it.
  python3 -m aklr validate "${flags[@]}" --under aklr/ops --exclude aklr/ops/mega_moe \
    --out "$RUN_DIR/validation-report"
else
  echo "::group::Validate plugins ($PLUGINS), check aklr/ops"
  plugins=0 ops=0
  # shellcheck disable=SC2086 # PLUGINS is a list of flags
  python3 -m aklr validate "${flags[@]}" $PLUGINS --out "$RUN_DIR/validation-report" || plugins=$?
  python3 -m aklr validate "${flags[@]}" --under aklr/ops --no-timing --out "$RUN_DIR/validation-check" || ops=$?
  exit $((plugins ? plugins : ops))
fi
