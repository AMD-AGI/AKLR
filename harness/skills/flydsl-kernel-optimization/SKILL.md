---
name: flydsl-kernel-optimization
description: >
  Making FlyDSL kernels fast on gfx950 (MI350/MI355X). Includes a dense-GEMM package
  (fp8, MXFP4, MXFP8, bf16/fp16, int8, small-M decode shapes included): dispatch rounds
  and split-K, main-loop pipelines, LDS/swizzle/DMA data paths, XCD tile order and
  scheduling hints, epilogues, autotuning, attribution, correctness gates and A/B
  measurement. Also a grouped-GEMM package (training forward, dgrad and variable-K wgrad in bf16, fp8,
  MXFP8 and MXFP4; MoE expert GEMMs), an MXFP4 fused-MoE decode note, LDS and prefetch optimization, ATT trace capture and analysis,
  perf-regression bisection, debugging wrong or non-compiling kernels, and the CDNA4 (gfx950)
  ISA reference guide with guides to looking things up in it. Use when optimizing a FlyDSL
  kernel, especially a GEMM or grouped GEMM, or finding out why one is slow or wrong.
allowed-tools: Read Edit Bash Grep Glob Agent
gens: [gfx950]
arch-names: [gfx950, CDNA4, MI350, MI355]
---

# FlyDSL kernel optimization

Techniques for making a FlyDSL kernel fast on gfx950, by phase. Paths below are relative to this folder,
except those starting with `aklr/`, relative to the repository root, and `aiter/` or `csrc/`, relative to aiter's.
The language itself (layouts, `@flyc.kernel` / `@flyc.jit`, copy atoms, MFMA, loops) is the
`flydsl-kernel-authoring` skill (`../flydsl-kernel-authoring/SKILL.md`); modernizing an existing kernel's
code is the `flydsl-kernel-code-cleanup` skill.

Every API statement here is for FlyDSL v0.3.2, the version this repository pins. AKLR's CI installs with `--no-deps`
and measures on its image's FlyDSL, 0.3.4.1 (`pitfalls/measured_a_different_build.md`). File and line citations are to
ROCm/FlyDSL `v0.3.2` (`dd8373349`), AMD-AGI/Primus-Turbo main `7615a5a2` (`ca1fdd67` in `grouped_gemm/`) and
ROCm/aiter `dbd8bf5` (`AKLR_BASELINE_AITER`); check out those revisions to follow them. The files under `optimize/`,
`profile/` and `bottleneck/` that come from FlyDSL's own skills keep their `Usage: /name` lines; those commands are
not registered here, so open the file instead.

## Stop: what a AKLR kernel cannot do

`aklr/ops/README.md` § The kernel section and ABI is the authority; where this skill disagrees, the README
wins. Check a plan against this table before writing code.

| Rule | What it rules out here |
|---|---|
| Outputs pass the definition's `compare` block; under `determinism: bitwise`, the default, two runs are byte-identical (`runs_agree`, `aklr/core/contract/semantics/_compare.py`) | Reducing in arrival order (atomic split-K, stream-K fix-ups): reduce in a fixed order (`gemm/flydsl_gemm_correctness.md` §8, `gemm/flydsl_gemm_grid_pricing.md` §2) |
| A Point is served by the file with the best measured `speedup` over the baseline, and only from `AKLR_MIN_SPEEDUP` (1.03) | Judging on a mean: compare each Point with the faster of the kernel AKLR serves there and the baseline |
| Launch code (`run` and what it calls) does not synchronize, call `.item()` / `.cpu()`, print, read a clock or the environment, or change torch state (SA3) | Timing or racing configurations in `run` |
| Compilation and the choice of configuration happen in a module-level `@functools.cache` function on a Point's first call; later calls may be inside a CUDA graph capture | The first-call race of `gemm/flydsl_gemm_autotune.md` §1 as written: choose from the shape, with a table raced offline |
| Inputs other than the mutable ones are read-only (SA2); nothing derived from an argument is kept across calls, and there is no `global` or `nonlocal` (SA1) | Caching a repacked weight: repack on every call, or serve the contract's shuffled layout variant where it has one |
| Scratch is allocated in the call (`torch.empty`: contents undefined) or once per shape in the compile function and rewritten before it is read; no call depends on what an earlier call left | Counters that live across calls: zero workspaces in each call before any read, or tag flags per launch (`gemm/flydsl_gemm_small_m.md` §4) |
| No workgroup waits for another: nothing polls a flag or data that another workgroup of the launch writes, since a launch's workgroups need not all be resident (a stream limited to part of the CUs, another kernel holding CUs) | A split-K fold or split-KV merge whose owner polls its partners: merge in a separate launch, or have each publish and count in on an atomic counter and the last to arrive merge, in a fixed order |
| `run(*, <kernel_call_keys>)` writes the preallocated outputs and returns nothing; the entry file's docstring describes the kernel, with no history or speedup (SA4) | Allocating outputs; importing sibling files other than relatively (the package name is a hash) |

## How to look things up

- Before opening a page, write down the diagnosis so far (the regime and the trace or counter that shows it),
  the plan, and the one question whose answer would change the plan.
- Pick one page and one section with the tables below, and read that, not the folder.
- Then note what changed in the plan, and leave one line in the round's notes: `KB: matched <page> §<n>` or
  `KB: no-match <question>`. The no-match lines are what this skill is missing.
- Confirm a change by what its page says to check (final ISA, kernel descriptor, a counter), not by the total
  time alone.
- Before starting on an op, look in [cases/INDEX.md](cases/INDEX.md) for a delivered kernel of the same op and
  regime, and in [pitfalls/INDEX.md](pitfalls/INDEX.md) for its traps.

## By task

| Task | Read in this order |
|---|---|
| Write or optimize a dense GEMM (fp8 / MXFP4 / MXFP8 / bf16) | `optimize/gemm-optimization.md` (FlyDSL's walkthrough) if you have not built a FlyDSL GEMM before → `gemm/INDEX.md` → the card it routes to |
| A GEMM with M ≤ 256 (decode, speculative verify) | `gemm/flydsl_gemm_small_m.md` |
| Write or optimize a grouped GEMM / MoE expert GEMM (forward, dgrad, variable-K wgrad) | `grouped_gemm/INDEX.md` → the card it routes to (main loop and datapath in `gemm/`) |
| Write or optimize an attention kernel (prefill, backward, decode with split KV) | `attention/INDEX.md` → the card it routes to |
| A fused MoE at decode sizes (MXFP4, routing sort, stage 1 / stage 2) | `moe/INDEX.md` → the technique, cases and expert-GEMM card it routes to |
| See each GEMM or MoE technique added to a working kernel, one per step, with its gfx950 ISA | `examples/README.md` (compile-checked; `examples/run.sh` checks them against torch on a GPU) |
| Dump a kernel's IR and final ISA; read registers, LDS, spills, waits and hot-loop counts | `gemm/flydsl_gemm_mainloop.md` §10 (`FLYDSL_DUMP_IR=1`); `tools/isa_diff.py` compares two dumps |
| Author and optimize any other kernel, structure first | `optimize/flydsl_levers/flydsl_authoring_method.md` |
| Read aiter's shipped FlyDSL kernels (dense GEMM, MoE, attention and the rest) and their knobs | `optimize/flydsl_levers/flydsl_kernel_library.md` → `optimize/flydsl_levers/flydsl_knob_space.md` |
| LDS bank conflicts, double buffering, swizzle | `optimize/lds-optimization.md` |
| Overlap loads with compute, prefetch, software pipelining | `optimize/prefetch-data-load.md` |
| Wrong output, NaN, won't compile, a fix that didn't take | `bottleneck/debug-flydsl-kernel.md` (clear the cache first); an error message: `language/compile_errors.md` |
| Code from another FlyDSL version, or FlyDSL doing what you did not expect | `language/version_notes.md`, `language/limits.md` |
| Capture a kernel trace, find the hotspot | `profile/capture-kernel-trace.md` → `profile/kernel-trace-analysis/SKILL.md` |
| A performance regression appeared; find the commit | `optimize/bisect-perf-regression.md` |
| What an instruction, encoding, counter, wait state or hardware limit really is | `isa/INDEX.md` → `isa/lookup.md`, `isa/matrix_core.md`, `isa/memory_and_sync.md` |
| Run or extend FlyDSL's own tests and benchmarks, inside a FlyDSL checkout | `profile/flydsl_repo_tests.md` |

## By symptom

| What the timings, trace or counters show | Read (then the symptom page, for what to rule out first) |
|---|---|
| Not yet known whether it is compute-, memory- or launch-bound | `gemm/flydsl_gemm_attribution.md` §1–3 |
| Fewer workgroups than CUs, or a last dispatch round mostly empty | `gemm/flydsl_gemm_grid_pricing.md` §1–2, §10; `patterns/low_cu_utilization.md` |
| M ≤ 256 on a row bound by launch, reduction and latency, not by reading B | `gemm/flydsl_gemm_small_m.md` §2–3; `patterns/launch_floor_small_m.md` |
| LDS bank conflicts | `gemm/flydsl_gemm_datapath.md` §3, `optimize/lds-optimization.md`; `patterns/lds_bank_conflicts.md` |
| MFMAs wait on global loads; bytes / time near the measured stream bandwidth | `gemm/flydsl_gemm_datapath.md` §5, §7, `optimize/prefetch-data-load.md`; `patterns/memory_bound.md` |
| Spills, VGPR / AGPR pressure, one wave per SIMD | `gemm/flydsl_gemm_mainloop.md` §4, §7, `gemm/flydsl_gemm_grid_pricing.md` §4; `patterns/register_pressure.md` |
| The store costs more than expected | `gemm/flydsl_gemm_epilogue.md` §2–3 |
| Memory-bound with a linear tile order: tiles that share a panel land on different XCDs | `gemm/flydsl_gemm_schedule_and_locality.md` §2 |
| Stalls on `s_waitcnt` | `gemm/flydsl_gemm_schedule_and_locality.md` §4, `isa/memory_and_sync.md` §1 |
| A fixed cost per tile: time not proportional to K | `gemm/flydsl_gemm_attribution.md` §7, `gemm/flydsl_gemm_grid_pricing.md` §6 |
| Clocks drop under load; slower than the instruction count says | `gemm/flydsl_gemm_attribution.md` §4 |
| A gain that does not reproduce, or an A/B inside the noise | `gemm/flydsl_gemm_measurement.md` §1, §4 |
| Outputs that differ between runs | `gemm/flydsl_gemm_correctness.md` §3, §8 |
| A change wins on some shapes and loses on others | `gemm/flydsl_gemm_autotune.md` §1–2; `patterns/shape_divergence.md` |

## Rules that hold everywhere

- **Structure before parameters.** Fusion, pipelining and layout changes beat knob tuning; tune last.
- **Clear the compile cache** (`rm -rf "${FLYDSL_RUNTIME_CACHE_DIR:-$HOME/.flydsl/cache}"`) before believing
  any result; `bottleneck/debug-flydsl-kernel.md` § What the cache key sees tells you when a result was stale.
- **Reuse the mechanism, re-derive the number.** Constants in the cited kernels (tile and preload tables,
  wait counts, adoption margins, cache thresholds) were tuned on specific shapes.
- Numbers marked *(field)* are measured values: use them for direction and order of magnitude, not as
  targets. *(field, E1)* points to the line under the page's `## Evidence` that gives the conditions.

## Also in AKLR

- **The aiter these cards describe is the baseline**, ROCm/aiter at `AKLR_BASELINE_AITER`
  (`.github/workflows/ci.yaml`): AKLR does not ship, patch or fall back to it, so its `aiter/` paths are that
  revision's, read as reference code. Price aiter through the op's `<contract>_baseline`
  (`aklr/ops/<op>/baseline.py`), the call bench times. At a Point no kernel serves, the api returns a
  `FallbackReason` and its caller runs its own path. AKLR's plugins route a model to plugin ops only
  (`aklr/plugins/README.md`), so an `aklr/ops` kernel runs in a model only where the engine calls its api.
- **Workflow** (`aklr/ops/README.md` § Workflow): export the Points, fill `kernel.sources`, then
  `python -m aklr export check` and `python -m aklr validate --base origin/main`. The local result does not
  count; CI measures. A `platform_failure` "Kineto recorded no kernels for a single call" is a tracer fault: on a
  busy node, wait a few minutes before you rerun (`pitfalls/profiler_misses_kernels.md`).
- **AKLR does not time below about 4 µs.** Its bench runner forces `GPU_FORCE_QUEUE_PROFILING`, and a profiled
  dispatch costs about that however little it does, so a short kernel and its baseline read about the same. A
  Point whose baseline sits near that floor shows no gain in CI: check `baseline.json` before working on it.

## Folder structure

- Packages, each starting at its `INDEX.md`: `gemm/` dense GEMM · `grouped_gemm/` grouped GEMM · `attention/`
  attention · `moe/` fused MoE at decode sizes
- `patterns/` symptom pages · `cases/` delivered kernels · `pitfalls/` traps · `language/` FlyDSL v0.3.2 notes
- `optimize/` FlyDSL's own guides, and `flydsl_levers/` on aiter's FlyDSL kernels and their knobs ·
  `profile/` trace capture and analysis · `bottleneck/` debugging · `isa/` the CDNA4 ISA guide
- `examples/` GEMM and MoE ladders, one technique per step
- `templates/`, `staging/`, `tools/` and `CONTRIBUTING.md`: adding a page

## Adding knowledge

A new page starts from `templates/`, goes to `staging/`, passes `tools/kb_lint.py` and review, and is moved in
by `tools/kb_approve.py`: see `CONTRIBUTING.md`. Write the current best practice, not the history of how it was
found.
