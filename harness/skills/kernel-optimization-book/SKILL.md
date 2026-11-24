---
name: kernel-optimization-book
description: >
  A book on optimizing operators for AMD GPUs: what makes a kernel fast on gfx950 (MI350/MI355X), why, when it
  does not apply, and how to write it in FlyDSL, Triton/Gluon or
  HIP (the flydsl/, triton/ and hip/ folders: limits, errors, debugging, each language's own guides). Includes
  a dense-GEMM package (fp8, MXFP4, MXFP8, bf16/fp16, int8, small-M shapes included): dispatch rounds
  and split-K, main-loop pipelines, LDS/swizzle/DMA data paths, XCD tile order and scheduling hints,
  epilogues, autotuning, attribution, correctness gates and A/B measurement. Also a MoE package (fused MoE at
  any token count, and the expert grouped GEMM: tile-to-group mapping, order under token skew, sorted-token
  rows, transposed operands and K varying per group, in bf16, fp8, MXFP8 and MXFP4), an attention package,
  symptom pages, pitfalls and worked cases, trace capture and analysis,
  perf-regression bisection, the CDNA4 (gfx950) ISA reference guide with guides to looking things up in it,
  and gfx942 (MI300X) GEMM notes. Use when optimizing a kernel in any of these languages, or finding out why one
  is slow or wrong.
allowed-tools: Read Edit Bash Grep Glob Agent
gens: [gfx950, gfx942]
---

# Kernel optimization book

What makes a kernel fast on AMD GPUs, by op and by symptom. The op cards (`gemm/`, `moe/`,
`attention/`) and the cases come from FlyDSL kernels: their mechanisms carry to other languages, their numbers were
measured on FlyDSL code unless a sentence says otherwise (re-derive them: § Rules that hold everywhere). How to write
it in a language is in that language's folder: `flydsl/SKILL.md` (with the FlyDSL side of each op card in
`flydsl/<op>.md`), `triton/SKILL.md`, `hip/SKILL.md`.

Paths below are relative to this folder, except those starting with `aiter/` or `csrc/`, relative to aiter's.
File and line citations are to ROCm/FlyDSL `v0.3.2` (`dd8373349`), AMD-AGI/Primus-Turbo main `7615a5a2`
(`ca1fdd67` for the expert GEMM in `moe/`) and ROCm/aiter `dbd8bf5`; check out those revisions to follow them.
FlyDSL's own skills are linked at `v0.3.2`, not copied: `flydsl/guides.md` and `profile.md` give each link with
what this book corrects in it; a `§` marked as FlyDSL's is that skill's own numbering.

## How to look things up

- Before opening a page, write down the diagnosis so far (the regime and the trace or counter that shows it),
  the plan, and the one question whose answer would change the plan.
- Pick one page and one section with the tables below, and read that, not the folder.
- Then note what changed in the plan, and leave one line in the round's notes: `KB: matched <page> §<n>` or
  `KB: no-match <question>`. The no-match lines are what this skill is missing.
- Confirm a change by what its page says to check (final ISA, kernel descriptor, a counter), not by the total
  time alone.
- Before starting on an op, look in [cases/INDEX.md](cases/INDEX.md) for a case of the same op and
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
| A Triton or Gluon kernel: authoring, debugging, tuning, numerics, Gluon layouts, memory, MFMA and inline asm, Triton kernels by family, GEMM, MoE, attention and GatedDeltaNet kernels | `triton/SKILL.md` → the page its task table names |

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
| A kernel and its baseline of a few µs read about the same, near 4 µs | `pitfalls/profiled_timer_floor.md`; `gemm/small_m.md` §1 |
| Outputs that differ between runs | `gemm/correctness.md` §3, §8 |
| A change wins on some shapes and loses on others | `gemm/autotune.md` §1–2; `patterns/shape_divergence.md` |

## Rules that hold everywhere

- **Structure before parameters.** Try fusion, pipelining and layout changes before knob tuning; tune last.
- **Reuse the mechanism, re-derive the number.** Constants in the cited kernels (tile and preload tables,
  wait counts, adoption margins, cache thresholds) were tuned on specific shapes.
- Numbers marked *(field)* are measured values: use them for direction and order of magnitude, not as
  targets. *(field, E1)* points to the line under the page's `## Evidence` that gives the conditions.
- **Judge each size against the baseline**, never the mean over sizes.
- **A wait on another workgroup has a way out.** A launch's workgroups need not all be resident (a stream limited
  to part of the CUs, another kernel holding CUs): bound the wait, and on timeout hand the share to the last to
  arrive (`attention/split_kv.md` §2).
- **Scratch holds nothing until this call writes it.** `torch.empty` is undefined bytes: zero or stamp a workspace
  in the call before any read (`gemm/small_m.md` §4), and keep no counter across calls.
- **A caller may capture the call into a CUDA graph.** Then launch code does not synchronize, read back to the
  host or time anything: choose the configuration from the shape, with a table raced offline
  (`gemm/autotune.md` §1).

## Folders and adding a page

`CONTRIBUTING.md`: what each folder holds, the page kinds, and how a page gets in.
