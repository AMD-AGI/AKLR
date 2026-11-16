---
title: FlyDSL's own tests and benchmark scripts — where they live, how they run, what they measure
kind: method
lever: flydsl_repo_tests
ops: [any]
gens: [gfx950]
flydsl: 0.3.2
sources:
  - ROCm/FlyDSL@dd8373349:tests/README.md
  - ROCm/FlyDSL@dd8373349:tests/pytest.ini
  - ROCm/FlyDSL@dd8373349:tests/conftest.py
  - ROCm/FlyDSL@dd8373349:tests/kernels/conftest.py
  - ROCm/FlyDSL@dd8373349:tests/language/conftest.py
  - ROCm/FlyDSL@dd8373349:tests/language/lang_utils.py
  - ROCm/FlyDSL@dd8373349:tests/arch_compat.py
  - ROCm/FlyDSL@dd8373349:tests/test_common.py
  - ROCm/FlyDSL@dd8373349:tests/utils.py
  - ROCm/FlyDSL@dd8373349:tests/kernels/benchmark_common.py
  - ROCm/FlyDSL@dd8373349:tests/kernels/test_vec_add.py
  - ROCm/FlyDSL@dd8373349:tests/kernels/test_preshuffle_gemm.py:703
  - ROCm/FlyDSL@dd8373349:tests/kernels/test_fp4_gemm_4wave.py:143
  - ROCm/FlyDSL@dd8373349:tests/python/examples/aot_example.py
  - ROCm/FlyDSL@dd8373349:scripts/run_tests.sh
  - ROCm/FlyDSL@dd8373349:scripts/run_benchmark.sh
  - ROCm/FlyDSL@dd8373349:scripts/compare_benchmark.py
  - ROCm/FlyDSL@dd8373349:scripts/dumpir.sh
  - ROCm/FlyDSL@dd8373349:python/flydsl/utils/env.py
  - ROCm/FlyDSL@dd8373349:.github/workflows/flydsl.yaml
---

# FlyDSL's own tests and benchmarks

> Paths are relative to a FlyDSL checkout at tag `v0.3.2` (`dd8373349`). Later tags move files and add
> suites: check out the tag that matches the installed `flydsl` before running anything here.

## Route here when
- You are working inside a FlyDSL checkout: changing FlyDSL itself, or reproducing an upstream test or
  benchmark number.
- You want to copy an upstream harness, reference or helper (`tests/test_common.py`, `tests/utils.py`, a
  kernel test).

**A AKLR kernel is judged by `python -m aklr validate` (`../SKILL.md` § In AKLR) and timed per
`../gemm/measurement.md`, not by FlyDSL's own suites or `run_benchmark.sh`.** AKLR kernel sources
cannot import `tests.*` either; copy what you need.

## 1. Environment

- The tests import `tests.*` and `kernels.*` from the checkout: run from the repository root with the root on
  `PYTHONPATH`. With the pinned wheel (`flydsl==0.3.2`), `export PYTHONPATH=$PWD` is enough; FlyDSL's CI
  benchmarks a release tag the same way (a worktree at the tag plus `pip install flydsl==<version>`). With a
  source build in `build-fly/` (or `$FLY_BUILD_DIR`):
  ```bash
  export PYTHONPATH="$PWD/build-fly/python_packages:$PWD:$PYTHONPATH"
  export LD_LIBRARY_PATH="$PWD/build-fly/python_packages/flydsl/_mlir/_mlir_libs:$LD_LIBRARY_PATH"
  ```
  `scripts/run_tests.sh` and `scripts/run_benchmark.sh` export both themselves.
- pytest reads `tests/pytest.ini` (markers, `python_files = test_*.py`, `-v --tb=short`) only when the paths
  you pass are under `tests/`, or with `-c tests/pytest.ini`. A bare `pytest` at the repository root takes
  `pyproject.toml` as its config instead.

## 2. Where each category lives

| Location | Contents | GPU | Run by |
|---|---|---|---|
| `tests/mlir/{Conversion,LayoutAlgebra,Transforms}/*.mlir` | FileCheck tests of `fly-opt` passes and lowering | no; needs `build-fly/bin/fly-opt` (source build) and LLVM `FileCheck` | `run_tests.sh` only (no pytest, no lit config) |
| `tests/language/` | language conformance (numeric/vector types, storage, allocator), traced frontend-only | no | pytest |
| `tests/unit/` | compiler, JIT cache key, control-flow dispatch, layout algebra, runtime | per file (tier marker, §4) | pytest |
| `tests/system/` | end-to-end control-flow kernels | mostly (8 of 9 files `l2_device`) | pytest |
| `tests/extension/` | `coop` (warp/block reduce and scan), `random` | per file | pytest |
| `tests/kernels/test_*.py` | kernel correctness + perf harnesses; most also run as scripts (`__main__`) | yes | pytest; `run_benchmark.sh` runs them as scripts |
| `tests/test_common.py`, `tests/utils.py`, `tests/kernels/{benchmark_common.py,test_ref.py,utils/}` | timers, comparisons, quantization and shuffle helpers, torch MoE references | — | imported |
| `tests/python/examples/aot_example.py` | precompiles preshuffle GEMM configs into `FLYDSL_RUNTIME_CACHE_DIR` with `COMPILE_ONLY=1` (`--preset`, `--in_dtype`); `--run_kernel` launches and checks them | `--run_kernel` and `test_aot_compile` | script. Its tests run only when the file is named explicitly: the name does not match `test_*.py`, so the directory `run_tests.sh` passes collects nothing from it |
| `examples/*.py`, `examples/extension/*/*.py` | tutorials | yes | `run_tests.sh` step 2 |

GEMM harnesses in `tests/kernels/` (MX formats, async copy and the last three files need gfx950; the tests
skip elsewhere):
`test_preshuffle_gemm.py` (fp8/int8 with per-token scales, fp16/bf16, MXFP4 B with fp4 A, MXFP6×MXFP4,
MXFP8×MXFP8; checks that no C store lands past row M), `test_gemm_a16w16_gfx950.py` (bf16/fp16, layouts
nn/nt/tn/tt, split-K), `test_fp8_gemm_rowscale.py` (8-wave fp8 row-scale), `test_fp4_gemm_4wave.py` (MXFP4).

## 3. Running

### `scripts/run_tests.sh` — the whole suite
CI runs `RUN_TESTS_FULL=1 bash scripts/run_tests.sh`. In order:
1. `python3 -m pytest tests/kernels/ tests/language/ tests/unit/ tests/system/ tests/extension/
   tests/python/examples/`, with `-m "not large_shape"` unless `RUN_TESTS_FULL=1`.
2. Every `examples/*.py` and `examples/extension/*/*.py` as a script; fails on a non-zero exit or on
   `Result correct: False` / `All passed: False` in the output.
3. Every `tests/mlir/**/*.mlir`: each `// RUN:` line with `%fly-opt`, `%FileCheck`/`FileCheck` and `%s`
   substituted. `FileCheck` comes from the build's `MLIR_DIR` (`CMakeCache.txt`) or `PATH`; without one the
   stage is skipped, but a missing `fly-opt` fails it.

It first exports `FLYDSL_AUTOTUNE=0` and `FLYDSL_RUN_QUANT=1`, and, when `HIP_VISIBLE_DEVICES` is unset on a
multi-GPU machine, selects the GPU with the most free memory. The pytest step runs to completion; the other
two stop at the first failure; any failing step ends the script.

### pytest directly
```bash
python3 -m pytest tests/kernels/test_preshuffle_gemm.py -m "not large_shape" -k fp8
python3 -m pytest tests/language/ tests/unit/ \
  -m "l0_backend_agnostic or l1a_compile_no_target_dialect or l1b_target_dialect"
python3 -m pytest tests/python/examples/aot_example.py
```
`--flydsl-compile-backend` and `--flydsl-compile-arch` (`tests/conftest.py`) set `FLYDSL_COMPILE_BACKEND` and
`ARCH` for the session. `pytest_sessionfinish` turns exit code 5 (nothing collected) into 0, so a `-k` or `-m`
that matches nothing still passes: check the collected count.

### One MLIR test
Build `fly-opt` (`bash scripts/build.sh`, or `cmake --build build-fly --target fly-opt` in an existing build),
then replay the file's own `// RUN:` lines; a file can carry several, one per `--check-prefix`:
```bash
build-fly/bin/fly-opt tests/mlir/Transforms/canonicalize.mlir --fly-canonicalize \
  | FileCheck tests/mlir/Transforms/canonicalize.mlir
```

### One kernel test as a script
```bash
python3 tests/kernels/test_preshuffle_gemm.py --in_dtype fp8 -M 16 -N 5120 -K 8192 \
  --tile_m 16 --tile_n 128 --tile_k 256 --no_aiter_bench
```
Flags that take effect: `--in_dtype` (`fp8`/`int8`/`fp16`/`bf16`; `fp4` only with `--wfp4`), `--out_dtype`,
`-M/-N/-K`, `--tile_m/--tile_n/--tile_k`, `--num_iters` (default 20), `--num_warmup` (3), `--use_async_copy`,
`--waves_per_eu`, `--xcd_swizzle`, `--test_graph`/`-tg`, `--no_aiter_bench` (without it, fp8/int8 also time
aiter's `gemm_a8w8_bpreshuffle` when aiter imports), `--wfp4` (MXFP4 path; ignores `--in_dtype`,
`--test_graph` and `--xcd_swizzle`). **Parsed and silently ignored:** `--lds_stage`, `--dsrd_preload`,
`--dvmem_preload`, `--use_cshuffle_epilog`, `--flyc`. Other files have their own flags
(`test_gemm_a16w16_gfx950.py`: `--dtype -m -n -k --block_m … --layout`; `test_fp8_gemm_rowscale.py`:
`--num_warmups`); read the file's `__main__` block.

## 4. Markers and tiers

`tests/pytest.ini` registers:

| Marker | Meaning |
|---|---|
| `l0_backend_agnostic` | L0: no compile backend or device runtime assumed, no vendor target dialect |
| `l1a_compile_no_target_dialect` | L1a: compiles with Fly and upstream dialects only |
| `l1b_target_dialect` | L1b: target lowering (Fly→ROCDL), no GPU execution; pair with `rocm_lower` |
| `l2_device` | L2: needs a GPU, driver and runtime |
| `rocm_lower` | with L1b or L2 when the test assumes the ROCDL path |
| `large_shape` | slow or large shapes; `run_tests.sh` deselects them unless `RUN_TESTS_FULL=1` |
| `multi_gpu` | needs at least 4 GPUs; skipped when they are absent |
| `benchmark` | performance tests; `run_benchmark.sh` runs some of them with `-m benchmark` |

The tiers (`tests/README.md`, RFC ROCm/FlyDSL#275) are only partly applied: 28 of 35 files in `tests/kernels/`
and 38 of 58 in `tests/unit/` carry a tier marker. `-m l2_device` misses all of `test_pa.py` and the GPU cases of
`test_flash_attn_fwd.py`, and `-m "not l2_device"` is not a CPU-only selection; select by directory when it
matters. `.mlir` files carry no markers.

## 5. Environment variables

| Variable | Effect |
|---|---|
| `FLYDSL_COMPILE_BACKEND`, `FLYDSL_RUNTIME_KIND` | compile backend and device runtime (both default `rocm`) |
| `ARCH` | target arch for compilation |
| `COMPILE_ONLY` | compile without launching |
| `FLYDSL_RUNTIME_CACHE_DIR` | disk cache (default `~/.flydsl/cache`) |
| `FLYDSL_RUNTIME_ENABLE_CACHE` | `0` disables the disk cache; the in-process cache stays (what the key covers: `debug-flydsl-kernel.md` § What the cache key sees) |
| `FLYDSL_DUMP_IR`, `FLYDSL_DUMP_DIR` | per-stage IR dump (default dir `~/.flydsl/debug`), §8 |
| `FLYDSL_AUTOTUNE`, `FLYDSL_AUTOTUNE_CACHE_DIR` | `1` forces the `flydsl.autotune` search; its results dir (default `~/.flydsl/autotune`) |
| `FLYDSL_GPU_ARCH`, `HSA_OVERRIDE_GFX_VERSION` | arch detection hints (`flydsl/runtime/device.py`) |
| `FLYDSL_RUN_QUANT` | `1` enables `tests/kernels/test_quant.py`, which skips otherwise |
| `ROCDSL_SOFTMAX_SHAPES`, `ROCDSL_LAYERNORM_SHAPES`, `ROCDSL_RMSNORM_SHAPES` | override those tests' shapes, `"M,N,dtype;M,N,dtype"` |
| `FLYDSL_PERFTEST_USE_EVENTS`, `FLYDSL_LOG_MORE` | `run_perftest` timing mode and logging, §7 |
| `RUN_TESTS_FULL` | `1`: `run_tests.sh` includes `large_shape` |
| `FLY_BUILD_DIR` | build directory for the scripts (default `build-fly`) |
| `BENCH_LOG_DIR`, `BENCH_OUTPUT_CSV` | `run_benchmark.sh` logs (default `/tmp/flydsl_bench`) and CSV output |

## 6. Adding a test

- Place it by what it depends on: a pass → a `.mlir` file under `tests/mlir/`; language semantics →
  `tests/language/`; compiler or runtime pieces → `tests/unit/`; a kernel → `tests/kernels/`. Name it
  `test_*.py`, or a directory run does not collect it.
- Mark the tier at module level, as the in-tree kernel tests do:
  ```python
  pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

  if not torch.cuda.is_available():
      pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)
  ```
  Language tests use `l1a_compile_no_target_dialect`; tests that run the Fly→ROCDL pipeline without a GPU use
  `l1b_target_dialect` + `rocm_lower` (`tests/unit/test_layout_algebra.py`).
- Skip arch-specific cases inside the test on `flydsl.runtime.device.get_rocm_arch()`, as
  `test_preshuffle_gemm.py` does for its gfx950-only paths; add a CDNA-only file to `CDNA_ONLY_TESTS`. Mark slow
  shapes with `pytest.param(..., marks=pytest.mark.large_shape)`.
- A kernel test follows `tests/kernels/test_vec_add.py`: a `@flyc.kernel`, a `@flyc.jit` launcher taking
  `stream: fx.Stream = fx.Stream(None)` and calling `.launch(grid=..., block=..., stream=stream)`, a torch
  reference, `checkAllclose`/`verify_output` and `run_perftest` from `tests/test_common.py`, and a `__main__`
  CLI. Writing the kernel itself is FlyDSL's `flydsl-kernel-authoring` skill (`guides.md` § flydsl-kernel-authoring).
- A test without a GPU traces through `@flyc.jit` frontend-only. Under `tests/language/` an autouse fixture sets
  `COMPILE_ONLY=1`, a fixed `ARCH` that is not gfx950, `FLYDSL_RUNTIME_ENABLE_CACHE=0` and replaces `MlirCompiler.compile` with a
  no-op; `lang_utils.source_ir(body)` returns the pre-lowering MLIR and `launch_ir(kernel)` traces a launch:
  ```python
  from lang_utils import source_ir

  def test_store_is_emitted():
      def body():
          ...  # DSL code under test
      assert "fly.ptr.store" in source_ir(body)
  ```
  `tests/conftest.py` also defines `ctx`, `module` and `insert_point` fixtures (a fresh MLIR context with the
  Fly dialects registered); no test at v0.3.2 uses them.
- A FileCheck test is `// RUN: %fly-opt %s <passes> | FileCheck %s` plus `// CHECK:` lines. Every RUN line runs.
  Expected-error tests wrap the tool: `// RUN: { %fly-opt --split-input-file %s 2>&1 || true; } | FileCheck %s`.
- `run_benchmark.sh` does not discover tests. A new row needs a shape list and a loop in the script, and a
  script that prints one of the lines its parser reads (§7).

## 7. How FlyDSL measures

**`run_perftest(fn, *args, num_iters=20, num_warmup=3, testGraph=False, num_rotate_args=0)`**
(`tests/test_common.py`, copied from aiter) returns `(last output, mean µs per iteration)`:
- After `num_warmup` calls it times `num_iters` calls **under `torch.profiler`** and reports the summed device
  time of every GPU kernel the callable launches, per iteration. The first profiled iteration is dropped, and
  above 30 iterations so are IQR outliers. `FLYDSL_PERFTEST_USE_EVENTS=1` switches to one event pair around the
  loop (for runs under `rocprofv3`); `FLYDSL_LOG_MORE=1` also logs a per-call event mean and the profiler table.
- With `num_rotate_args=0` it deep-copies every tensor argument, outputs included, and cycles through the
  copies: as many as fit in min(8192 × L2 size, 90% of free memory), capped at `num_iters`. For GEMM-sized
  inputs that is one copy per iteration, so the default reads cold-cache time and allocates up to `num_iters`
  copies of every argument.
- `testGraph=True` captures the `num_iters` calls in one graph, replays it under the profiler and returns that
  time instead.
- Upstream's own comment calls it right "for a smoke check but reads a few percent low and drifts run to run"
  (`test_fp4_gemm_4wave.py`). That file's `--vs-aiter` mode is its A/B harness: 5 rotated input sets, 500
  warmup calls, 3 pairs of 500 iterations inside one event pair, best of each side.

**Correctness helpers.** `checkAllclose(a, b, rtol=1e-2, atol=1e-2, tol_err_ratio=0.05)` returns the fraction of
elements outside `torch.isclose` (0 when all pass); `tol_err_ratio` only picks the log message.
`verify_output(c_out, c_ref, atol=1e-2, rtol=1e-2, logits_diff_threshold=2e-3)` passes when under 5% of the
elements mismatch, **or** when 1 − 2Σxy / Σ(x² + y²) ≤ 2e-3, so a few wrong tiles pass it; AKLR's gate is in
`../gemm/correctness.md`. `tests/utils.py`: `pertoken_quant(x, quant_dtype=...)` returns
`(q, per_row_fp32_scale)` and scrubs NaN/Inf; `shuffle_weight(w, layout=(16, 16))` is the B preshuffle the
preshuffle GEMMs expect.

**Other timers** (`tests/kernels/benchmark_common.py`). `bench_gpu_us_torch(fn, *, warmup=20, iters=200)`: one
event pair around back-to-back calls, mean, no rotation (warm cache). `bench_kernel_us(run_fn, warmup=10,
iters=50, flush_l2=True, prep_fn=None)`: per-call events after zeroing a buffer of max(2 × torch's
`L2_cache_size`, 8 MB), IQR filter, median; with no flush and no `prep_fn` it falls back to one event pair.

**`scripts/run_benchmark.sh`.**
- `bash scripts/run_benchmark.sh` runs every op; `… gemm moe`, `--only softmax,layernorm`, `--list`,
  `--output_csv <path>` select or export. Ops: `softmax softmax_bwd layernorm rmsnorm flash_attn mla gemm moe`.
  `gemm` covers preshuffle (sync and async-copy lists), A16W16, 8-wave fp8 row-scale and 4-wave fp4; GEMM and
  MoE rows run only on CDNA. On other CDNA GPUs the gfx950-only fp8 row-scale, fp4 and MoE fp4/a8w4 rows print
  `skip`, and A16W16 prints no row (its shape list is set only on gfx950).
- Each shape runs a test file as a script, not under pytest (the rmsnorm mixed-weight rows run their `benchmark`
  tests under pytest), logging to `$BENCH_LOG_DIR/<op>_<shape>.log`. GEMM runs pass 10 warmup and 100 timed
  iterations, except A16W16 (`--num_warmup 3 --num_iters 50`); all time with `run_perftest`.
- The table (`op shape dtype TB/s TFLOPS`) is parsed from the logs: GEMM rows take the **last**
  `Throughput: … TFLOPS, BW: … TB/s` line, softmax and norm rows the **first** `Bandwidth: … GB/s` line. A log
  containing `Skipped:` counts as skipped; any failure makes the exit code 1.
- `test_fp4_gemm_4wave.py`'s `__main__` ignores every flag but `--vs-aiter` and always runs 8192³ then 16384³,
  so both `gemm_fp4` rows report the 16384³ result.
- Shape lists sit at the top of the script. Only `HGEMM_SHAPES`, `FLASH_ATTN_FUNC_SHAPES`, `MLA_DECODE_SHAPES`,
  `GEMM_USE_ASYNC_COPY` and `GEMM_WAVES_PER_EU` are read from the environment; edit the script for the rest.
- CI runs the script on the PR, on its base commit and on the latest release tag, and compares them with
  `scripts/compare_benchmark.py`, which prints per-row ratios (TFLOPS when present, else TB/s) and always exits
  0: a slowdown alone does not fail FlyDSL's CI.

## 8. IR dumps

`bash scripts/dumpir.sh python3 <script.py> [args]` runs the command from the repository root with
`FLYDSL_DUMP_IR=1` and `FLYDSL_DUMP_DIR` defaulting to `/tmp/flydsl_dump_ir` (`~/.flydsl/debug` without the
script). Each kernel gets a subdirectory of numbered stages, `NN_<stage>.mlir` through `NN_final_isa.s`. How to
read them: `../gemm/mainloop.md` §10. Do not time a run with the dump on.

## Related
`../gemm/measurement.md` · `../gemm/correctness.md` · `debug-flydsl-kernel.md`
· `../profile.md` § bisect-perf-regression (an upstream test as the probe of a FlyDSL bisect)
