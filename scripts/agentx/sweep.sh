#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd -P)
export PYTHONPATH="$repo/benchmarks/agentx${PYTHONPATH:+:$PYTHONPATH}"
exec python3 -m agentx_bench sweep "$@"
