---
name: aklr-kb
description: >
  What makes a kernel fast on gfx950 (MI350/MI355X) inside AKLR, and how to write it in FlyDSL, Triton/Gluon or
  HIP (the flydsl/, triton/ and hip/ folders: limits, errors, debugging, each language's own guides). Includes
  a dense-GEMM package (fp8, MXFP4, MXFP8, bf16/fp16, int8, small-M shapes included): dispatch rounds
  and split-K, main-loop pipelines, LDS/swizzle/DMA data paths, XCD tile order and scheduling hints,
  epilogues, autotuning, attribution, correctness gates and A/B measurement. Also a MoE package (fused MoE at
  any token count, and the expert grouped GEMM: tile-to-group mapping, order under token skew, sorted-token
  rows, transposed operands and K varying per group, in bf16, fp8, MXFP8 and MXFP4), an attention package,
  symptom pages, pitfalls and delivered kernels, trace capture and analysis,
  perf-regression bisection, the CDNA4 (gfx950) ISA reference guide with guides to looking things up in it,
  and gfx942 (MI300X) GEMM notes. Use when optimizing a kernel in any of these languages, or finding out why one
  is slow or wrong.
allowed-tools: Read Edit Bash Grep Glob Agent
gens: [gfx950, gfx942]
---

# AKLR kernel knowledge base

What makes a kernel fast on AMD GPUs inside AKLR, by op and by symptom. The op cards (`gemm/`, `moe/`,
`attention/`) and the cases come from FlyDSL kernels: their mechanisms carry to other languages, their numbers were
measured on FlyDSL code unless a sentence says otherwise (re-derive them: § Rules that hold everywhere). How to write
it in a language is in that language's folder: `flydsl/SKILL.md` (with the FlyDSL side of each op card in
`flydsl/<op>.md`), `triton/SKILL.md`, `hip/SKILL.md`.

Paths below are relative to this folder, except those starting with `aklr/`, relative to the repository root, and
`aiter/` or `csrc/`, relative to aiter's. File and line citations are to ROCm/FlyDSL `v0.3.2` (`dd8373349`),
AMD-AGI/Primus-Turbo main `7615a5a2` (`ca1fdd67` for the expert GEMM in `moe/`) and ROCm/aiter `dbd8bf5`
(`AKLR_BASELINE_AITER`); check out those revisions to follow them. FlyDSL's own skills are linked at `v0.3.2`, not
copied: `flydsl/guides.md` and `profile.md` give each link with what this repository corrects in it; a
`§` marked as FlyDSL's is that skill's own numbering.

## Stop: what a AKLR kernel cannot do

`aklr/ops/README.md` § The kernel section and ABI is the authority; where this skill disagrees, the README
wins. Check a plan against this table before writing code.

| Rule | What it rules out here |
|---|---|
| Outputs pass the definition's `compare` block; under `determinism: bitwise`, the default, two runs are byte-identical (`runs_agree`, `aklr/core/contract/semantics/_compare.py`) | Reducing in arrival order (atomic split-K, stream-K fix-ups): reduce in a fixed order (`gemm/correctness.md` §8, `gemm/grid_pricing.md` §2) |
| A Point is served by the file with the best measured `speedup` over the baseline, and only from `AKLR_MIN_SPEEDUP` (1.05) | Judging on a mean: compare each Point with the faster of the kernel AKLR serves there and the baseline |
| Launch code (`run` and what it calls) does not synchronize, call `.item()` / `.cpu()`, print, read a clock or the environment, or change torch state (SA3) | Timing or racing configurations in `run` |
| Compilation and the choice of configuration happen in a module-level `@functools.cache` function on a Point's first call; later calls may be inside a CUDA graph capture | The first-call race of `gemm/autotune.md` §1 as written: choose from the shape, with a table raced offline |
| Inputs other than the mutable ones are read-only (SA2); nothing derived from an argument is kept across calls, and there is no `global` or `nonlocal` (SA1) | Caching a repacked weight: repack on every call, or serve the contract's shuffled layout variant where it has one |
| Scratch is allocated in the call (`torch.empty`: contents undefined) or once per shape in the compile function and rewritten before it is read; no call depends on what an earlier call left | Counters that live across calls: zero workspaces in each call before any read (`gemm/small_m.md` §4). A slot stamped with this launch's 64-bit tag, queue bits included, needs no zeroing as long as no leftover word equals the tag: such a split-K fold is one launch on a `torch.empty` workspace (`gemm/wait_free_fold.md` § Folding without a counter) |
| Not in the README or its audit, but keep to it: no workgroup waits for another without a way out: a launch's workgroups need not all be resident (a stream limited to part of the CUs, another kernel holding CUs), so a wait on a flag another workgroup writes is bounded, and on timeout hands its share to the last to arrive (`attention/split_kv.md` §2 Bounded wait); the driver's `--stress` passes at every Point the file lists | A split-K fold or split-KV merge whose owner polls its partners with no bound, or gives up and writes: merge in a separate launch, have each publish and count in on an atomic counter and the last to arrive merge, or wait bounded with hand-off, in a fixed order |
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
| Write or optimize a dense GEMM (fp8 / MXFP4 / MXFP8 / bf16) | `gemm/INDEX.md` → the card it routes to |
| A GEMM with M ≤ 256 (as in decode or speculative verify) | `gemm/small_m.md` |
| A GEMM on gfx942 (MI300X) | `gemm/gfx942.md` |
| A MoE layer at any token count (MXFP4 experts, routing sort, stage 1 / stage 2), or its expert GEMMs as a grouped GEMM: per-size configuration, tile and occupancy, XCD placement, low precision and scale preshuffle, sorted-token rows, K varying per group, correctness, measurement | `moe/INDEX.md` → the card § it routes the problem to; main loop and data path in `gemm/` |
| Write or optimize an attention kernel: many query rows per head, the dQ dK dV gradients, or few query rows per head with split KV | `attention/INDEX.md` → the card it routes to |
| Dump a kernel's IR and final ISA; read registers, LDS, spills, waits and hot-loop counts | `gemm/mainloop.md` §10 (FlyDSL's `FLYDSL_DUMP_IR=1`: `flydsl/gemm.md` § mainloop §10); `analysis/isa_diff.py` compares two dumps |
| Capture a kernel trace and find the hotspot, or find the commit behind a performance regression | `profile.md` |
| EP dispatch/combine against TP for a MoE layer at a few tokens per GPU, or a collective's time in a multi-GPU trace | `communication/INDEX.md` |
| What an instruction, encoding, counter, wait state or hardware limit really is | `hardware/gfx950/INDEX.md` → `hardware/gfx950/lookup.md`, `hardware/gfx950/matrix_core.md`, `hardware/gfx950/memory_and_sync.md`; gfx942 (MI300X): `hardware/gfx942/INDEX.md` |
| The language: its API, compile errors, debugging, its own guides | the language's folder: `flydsl/SKILL.md`, `triton/SKILL.md`, `hip/SKILL.md` |
| A Triton or Gluon kernel: authoring, debugging, tuning, numerics, Gluon layouts, memory, MFMA and inline asm, AKLR's Triton kernels by family, its GEMM, MoE, attention and GatedDeltaNet kernels | `triton/SKILL.md` → the page its task table names |

## By symptom

| What the timings, trace or counters show | Read (then the symptom page, for what to rule out first) |
|---|---|
| Not yet known whether it is compute-, memory- or launch-bound | `gemm/attribution.md` §1–3 |
| Fewer workgroups than CUs, or a last dispatch round mostly empty | `gemm/grid_pricing.md` §1–2, §10; `patterns/low_cu_utilization.md` |
| M ≤ 256 on a row bound by launch, reduction and latency, not by reading B | `gemm/small_m.md` §2–3; `patterns/launch_floor_small_m.md` |
| LDS bank conflicts | `gemm/datapath.md` §3; `patterns/lds_bank_conflicts.md` |
| MFMAs wait on global loads; bytes / time near the measured stream bandwidth | `gemm/datapath.md` §5, §7; `patterns/memory_bound.md` |
| Spills, VGPR / AGPR pressure, one wave per SIMD | `gemm/mainloop.md` §4, §7, `gemm/grid_pricing.md` §4; `patterns/register_pressure.md` |
| The store costs more than expected | `gemm/epilogue.md` §2–3 |
| Memory-bound with a linear tile order: tiles that share a panel land on different XCDs | `gemm/schedule_and_locality.md` §2 |
| Stalls on `s_waitcnt` | `gemm/schedule_and_locality.md` §4, `hardware/gfx950/memory_and_sync.md` §1 |
| A fixed cost per tile: time not proportional to K | `gemm/attribution.md` §7, `gemm/grid_pricing.md` §6 |
| Clocks drop under load; slower than the instruction count says | `gemm/attribution.md` §4 |
| A gain that does not reproduce, or an A/B inside the noise | `gemm/measurement.md` §1, §4 |
| Outputs that differ between runs | `gemm/correctness.md` §3, §8 |
| A change wins on some shapes and loses on others | `gemm/autotune.md` §1–2; `patterns/shape_divergence.md` |

## Rules that hold everywhere

- **Structure before parameters.** Try fusion, pipelining and layout changes before knob tuning; tune last.
- **Reuse the mechanism, re-derive the number.** Constants in the cited kernels (tile and preload tables,
  wait counts, adoption margins, cache thresholds) were tuned on specific shapes.
- Numbers marked *(field)* are measured values: use them for direction and order of magnitude, not as
  targets. *(field, E1)* points to the line under the page's `## Evidence` that gives the conditions.

## Also in AKLR

- **The aiter these cards describe is the baseline**, ROCm/aiter at `AKLR_BASELINE_AITER`
  (`.github/workflows/ci.yaml`): AKLR does not ship, patch or fall back to it, so its `aiter/` paths are that
  revision's, read as reference code. Price aiter through the op's `<contract>_baseline`
  (`aklr/ops/<op>/baseline.py`), the call bench times. At a Point no kernel serves, the api returns a
  `FallbackReason` and its caller runs its own path. AKLR's plugins route a model to plugin ops and to some
  `aklr/ops` ops (vLLM's GEMM and MoE layers, sglang's DSA paged MQA logits; `aklr/plugins/README.md`), so an
  `aklr/ops` kernel runs in a model only where such a route or the engine calls its api.
- **Workflow** (`aklr/ops/README.md` § Workflow): export the Points, fill `kernel.sources`, then
  `python -m aklr export check` and `python -m aklr validate --base origin/main`. The local result does not
  count; CI measures. A `platform_failure` whose message names rocprofiler-sdk or the graph replays, or says "the
  kernel tracer failed" or "calibration could not restore", is a tracer fault, already measured again in fresh
  processes: rerun (`aklr/core/validation/stages/_measure.py`, `pitfalls/profiler_misses_kernels.md`).
- **A short kernel sits on AKLR's timing floor.** Up to measurement profile 18 (Kineto, with
  `GPU_FORCE_QUEUE_PROFILING` forced) AKLR read about 4 µs per kernel however little it did, so a short kernel and
  its baseline read about the same. Main's profile 19 (its own rocprofiler-sdk tracer, queue profiling not forced)
  still has that floor, about 4.4 µs: a bmm candidate 1.50× faster back to back read 1.13× on it. Time an empty
  kernel, the baseline and your candidate on current main before deciding a Point has no headroom
  (`pitfalls/aklr_dispatch_floor.md` § Root cause).

## Folders and adding a page

`CONTRIBUTING.md`: what each folder holds, the page kinds, and how a page gets in.
