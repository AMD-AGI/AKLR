#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Benchmark the solutions in `aklr/ops/<op>/impls/` with the stock api CLI.
# Run from a ROCm/PyTorch environment with access to the target GPUs:
#
#   bash .github/scripts/run_benchmark.sh                       # every solution in this checkout
#   bash .github/scripts/run_benchmark.sh --definition gemm_a16w16_nt_n32_k6144
#   bash .github/scripts/run_benchmark.sh --gpus 0,1 --num-trials 3 --iterations 50
#   bash .github/scripts/run_benchmark.sh --atol 1e-2 --rtol 1e-2
#   bash .github/scripts/run_benchmark.sh --dry-run             # export the collection, stop
#   bash .github/scripts/run_benchmark.sh --dry-run --output-dir /tmp/benchmark-collection
#   bash .github/scripts/run_benchmark.sh --output-dir /tmp/benchmark-collection --run-dir /tmp/benchmark-run
#   bash .github/scripts/run_benchmark.sh -- --op-type-eval gemm:atol=1e-2,rtol=1e-3
#
# Tolerances default to the API settings (atol=rtol=1e-2). Explicit --atol/--rtol
# override per-op-type and per-definition settings; no k-based relaxation.
#
# Adapted from the user-provided aklr-bench/bench_new_solutions.sh.
# Reuses the original export and benchmark commands; CI additionally requires
# persisted traces so the following AITER tests can load the measured collection.
#
# `python -m aklr.benchmarks.api` is used exactly as it ships.
# `aklr.benchmarks.api.tools.export_benchmark_schemas` assembles its input directly
# from the declarations under `aklr/ops`: it copies
# definitions and candidate solutions through the schema models, expands the
# configured workloads, and renders each timed baseline from its schema/route.
# --output-dir is the SchemaCollection root passed unchanged to the benchmark API; it
# defaults to <run-dir>/set and must not exist. Logs and the report go under
# --run-dir. Source definitions and solutions are never modified.
#
#   definitions/  <- aklr/ops/<op>/definitions/*.json, each carrying its own
#                    `reference`: what the answer is, part of the question
#   solutions/    <- aklr/ops/<op>/impls/**/*.json, what is being measured, plus
#                    generated `author=baseline` solutions, what it costs
#   workloads/    <- WorkloadConfig in aklr/benchmarks/api/config.py, expanded to jsonl
#
# The baseline stays a solution because it is timed and its records join on its
# name; the reference is never timed, so it needs no name and lives in the
# definition, the way flashinfer-bench has it.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"

OPS_ROOT="${REPO_ROOT}/aklr/ops"
RUN_DIRECTORY=""
BENCHMARK_DIRECTORY=""
NUM_TRIALS=3
WARMUP_RUNS=15
ITERATIONS=500
TIMEOUT_SECONDS=900 # a fused_moe solution JIT-builds before it runs
LOG_LEVEL="info"
MAX_CONCURRENCY=""
GPUS=""
DRY_RUN="no"
SELECTION=()           # --definition / --solution, passed through as given
TOLERANCE_ARGUMENTS=() # Only explicit overrides; leave API defaults unset.
EXTRA_API_ARGUMENTS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
    --definition | --solution)
        SELECTION+=("$1" "$2")
        shift 2
        ;;
    --atol | --rtol)
        TOLERANCE_ARGUMENTS+=("$1" "${2:?$1 requires a value}")
        shift 2
        ;;
    --ops-root)
        OPS_ROOT="$2"
        shift 2
        ;;
    --run-dir)
        RUN_DIRECTORY="$2"
        shift 2
        ;;
    --output-dir)
        BENCHMARK_DIRECTORY="${2:?$1 requires a value}"
        shift 2
        ;;
    --num-trials)
        NUM_TRIALS="$2"
        shift 2
        ;;
    --warmup-runs)
        WARMUP_RUNS="$2"
        shift 2
        ;;
    --iterations)
        ITERATIONS="$2"
        shift 2
        ;;
    --timeout-seconds)
        TIMEOUT_SECONDS="$2"
        shift 2
        ;;
    --log-level)
        LOG_LEVEL="$2"
        shift 2
        ;;
    --max-concurrency)
        MAX_CONCURRENCY="$2"
        shift 2
        ;;
    --gpus)
        GPUS="$2"
        shift 2
        ;;
    --dry-run)
        DRY_RUN="yes"
        shift
        ;;
    --)
        shift
        EXTRA_API_ARGUMENTS=("$@")
        break
        ;;
    -h | --help)
        sed -n '5,19p' "${BASH_SOURCE[0]}"
        exit 0
        ;;
    *)
        echo "unknown argument: $1" >&2
        exit 2
        ;;
    esac
done

[[ -d "${OPS_ROOT}" ]] || {
    echo "not a directory: ${OPS_ROOT}" >&2
    exit 2
}

RUN_DIRECTORY="${RUN_DIRECTORY:-${REPO_ROOT}/.bench-runs/$(date +%Y%m%d-%H%M%S)}"
[[ -e "${RUN_DIRECTORY}" || -L "${RUN_DIRECTORY}" ]] && {
    echo "run directory exists: ${RUN_DIRECTORY}" >&2
    exit 2
}
BENCHMARK_DIRECTORY="${BENCHMARK_DIRECTORY:-${RUN_DIRECTORY}/set}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[[ -n "${GPUS}" ]] && export HIP_VISIBLE_DEVICES="${GPUS}"

# Measure with the proxy off. A solution here reaches aiter through the same
# @aklr_proxy entry point dispatch owns, so once a measured winner table is installed
# the baseline arm would be served by the very kernel it is the comparand of,
# and every speedup would collapse to 1.00x.
export AKLR_DISABLE_PROXY=1

echo "=== export collection ==="
echo "repo        : ${REPO_ROOT}"
echo "run         : ${RUN_DIRECTORY}"
echo "collection  : ${BENCHMARK_DIRECTORY}"
python3 -m aklr.benchmarks.api.tools.export_benchmark_schemas \
    "${BENCHMARK_DIRECTORY}" --ops-root "${OPS_ROOT}"

if [[ "${DRY_RUN}" == "yes" ]]; then
    echo "dry run: benchmark API collection exported to ${BENCHMARK_DIRECTORY}"
    exit 0
fi

# An explicit --output-dir may be outside the run directory.
mkdir -p "${RUN_DIRECTORY}"

# A definition whose only solution is its baseline is skipped by the benchmark,
# so the collection can hold every definition and still measure only what is new.
# One task per GPU, each in its own child: aiter binds to the first device it
# sees, so nothing here may drive two devices from one process.
python3 -c 'import torch; print("visible GPUs:", torch.cuda.device_count())'

API_ARGUMENTS=(
    "${BENCHMARK_DIRECTORY}"
    --num-trials "${NUM_TRIALS}"
    --warmup-runs "${WARMUP_RUNS}"
    --iterations "${ITERATIONS}"
    --timeout-seconds "${TIMEOUT_SECONDS}"
    --log-level "${LOG_LEVEL}"
    --dump-records
    --json-report "${RUN_DIRECTORY}/records.json"
)
[[ -n "${MAX_CONCURRENCY}" ]] && API_ARGUMENTS+=(--max-concurrency "${MAX_CONCURRENCY}")
API_ARGUMENTS+=("${SELECTION[@]+"${SELECTION[@]}"}")
API_ARGUMENTS+=("${TOLERANCE_ARGUMENTS[@]+"${TOLERANCE_ARGUMENTS[@]}"}")
API_ARGUMENTS+=("${EXTRA_API_ARGUMENTS[@]+"${EXTRA_API_ARGUMENTS[@]}"}")

echo
echo "=== python3 -m aklr.benchmarks.api ${API_ARGUMENTS[*]} ==="

set +e
python3 -m aklr.benchmarks.api "${API_ARGUMENTS[@]}" 2>&1 | tee "${RUN_DIRECTORY}/run.log"
status=${PIPESTATUS[0]}
set -e

if [[ "$status" -eq 0 ]]; then
    # The API logs report write errors separately. A successful CI benchmark
    # must also provide both its report and the collection consumed by AITER.
    test -s "${RUN_DIRECTORY}/records.json"
    python3 "${HERE}/verify_schema.py" "${BENCHMARK_DIRECTORY}"
fi

echo "=== done, exit ${status} ==="
echo "table  : ${RUN_DIRECTORY}/run.log"
echo "records: ${RUN_DIRECTORY}/records.json"
exit "${status}"
