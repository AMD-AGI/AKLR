#!/bin/bash
# Run the example ladders.
#
#   ./run.sh                      run every step on the GPU, check it against torch, print the max error
#                                 and a median time; exit 1 if any step fails its tolerance
#   ./run.sh --compile-only       compile every step for gfx950 without a GPU (CPU tensors, nothing
#                                 launched) and dump IR + ISA; then print each step's ISA against the
#                                 previous step's with ../tools/isa_diff.py
#   ./run.sh [--compile-only] gemm/step3 moe     only the steps whose path contains one of the words
#
# PYTHON   interpreter with FlyDSL 0.3.2 and torch (default: python3 on PATH)
# KBEX_DUMP, KBEX_CACHE   compile-only dump and cache roots (default /tmp/kbex/dump, /tmp/kbex/cache)
#
# Tolerance (common.py): |out - ref| <= 1e-2 + 1e-2 * |ref| elementwise, against an fp32 torch reference.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
PY=$(command -v "${PYTHON:-python3}") || { echo "no python: set PYTHON" >&2; exit 2; }

COMPILE_ONLY=0
FILTERS=()
for arg in "$@"; do
  case "$arg" in
    --compile-only) COMPILE_ONLY=1 ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) FILTERS+=("$arg") ;;
  esac
done

STEPS=()
for f in "$HERE"/gemm/step*.py "$HERE"/moe/step*.py; do
  rel=${f#"$HERE"/}
  if [ ${#FILTERS[@]} -eq 0 ]; then STEPS+=("$rel"); continue; fi
  for w in "${FILTERS[@]}"; do case "$rel" in *"$w"*) STEPS+=("$rel"); break ;; esac; done
done

DUMP=${KBEX_DUMP:-/tmp/kbex/dump}
CACHE=${KBEX_CACHE:-/tmp/kbex/cache}
fail=0
for rel in "${STEPS[@]}"; do
  name=$(echo "${rel%.py}" | tr / _)
  echo "== $rel"
  if [ "$COMPILE_ONLY" = 1 ]; then
    rm -rf "${DUMP:?}/$name" "${CACHE:?}/$name"
    # No ROCm tools on PATH (so nothing probes for a GPU), no visible devices; FLYDSL_GPU_ARCH as well as
    # ARCH because the arch-dispatched helpers (s_waitcnt, buffer-descriptor flags) read the host, not ARCH.
    env PATH=/usr/local/bin:/usr/bin:/bin HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= \
      COMPILE_ONLY=1 ARCH=gfx950 FLYDSL_GPU_ARCH=gfx950 \
      FLYDSL_RUNTIME_CACHE_DIR="$CACHE/$name" FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR="$DUMP/$name" \
      "$PY" -I -B "$HERE/$rel" 2>&1 | grep -v '^\[flydsl.compile\]'
    status=${PIPESTATUS[0]}
    isa=$(find "$DUMP/$name" -mindepth 2 -maxdepth 2 -name '*_final_isa.s' 2>/dev/null | sort | head -1)
    if [ "$status" != 0 ] || [ -z "$isa" ]; then echo "   FAILED to compile"; fail=1; continue; fi
    echo "   ISA: $isa"
  else
    "$PY" -I -B "$HERE/$rel" || fail=1
  fi
done

ISA_DIFF="$HERE/../tools/isa_diff.py"
if [ "$COMPILE_ONLY" = 1 ] && [ -f "$ISA_DIFF" ]; then
  for ladder in gemm moe; do
    prev=""
    for rel in "${STEPS[@]}"; do
      case "$rel" in "$ladder"/*) ;; *) continue ;; esac
      name=$(echo "${rel%.py}" | tr / _)
      cur=$(find "$DUMP/$name" -mindepth 2 -maxdepth 2 -name '*_final_isa.s' 2>/dev/null | sort | head -1)
      if [ -n "$prev" ] && [ -n "$cur" ]; then
        echo; echo "######## $(basename "$(dirname "$(dirname "$prev")")") -> $name"
        "$PY" -I -B "$ISA_DIFF" "$prev" "$cur"
      fi
      prev=$cur
    done
  done
fi
exit $fail
