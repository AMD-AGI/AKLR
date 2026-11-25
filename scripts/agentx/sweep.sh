#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd -P)
export PYTHONPATH="$repo/benchmarks/perf${PYTHONPATH:+:$PYTHONPATH}"
exec python3 -m agentx sweep "$@"
