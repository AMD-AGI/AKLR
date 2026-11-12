---
title: Dense GEMM on gfx950 — package map
kind: index
gens: [gfx950, gfx942]
pinned_source: ROCm/FlyDSL@dd8373349 (v0.3.2; kernels/gemm, .claude/skills/gemm-optimization); AMD-AGI/Primus-Turbo@7615a5a2 (primus_turbo/flydsl/{gemm,utils})
---

# Dense GEMM — package map

What makes a dense GEMM fast on gfx950 (MI350/MI355X), whatever the language: fp8 (per-tensor and per-row
scales), MXFP4, MXFP8, bf16/fp16 and int8, layouts NT/NN/TN. The cards state the mechanisms and their numbers; the
FlyDSL side (calls, FlyDSL's compiler, and the FlyDSL kernels of FlyDSL, aiter and Primus-Turbo) is in
`../flydsl/gemm.md`. Grouped and MoE expert GEMMs build on these cards and have their own package,
`../moe/INDEX.md`.

Unless a sentence names another language or a microbenchmark, the measured numbers (`*(field, E<n>)*`) come
from FlyDSL kernels; the evidence lines name the toolchain where it was recorded. The mechanisms are stated in ISA
and LLVM terms and should carry over to Triton or HIP, but the numbers have not been checked there.

**Three sources, used together.**
- **FlyDSL's own GEMM guide and kernels.** `../flydsl/guides.md` § gemm-optimization walks through how a FlyDSL GEMM is
  built, on `preshuffle_gemm`: tiling, the 2-stage LDS ping-pong, the XOR swizzle, `hot_loop_scheduler`, direct
  and CShuffle epilogues, B preshuffle, register budget. `../flydsl/guides.md` § lds-optimization and
  `../flydsl/guides.md` § prefetch-data-load cover the general LDS and prefetch transformations.
- **Primus-Turbo's FlyDSL GEMMs** (`primus_turbo/flydsl/{gemm,utils}` at main `7615a5a2`), built for large-M
  shapes. Their comments record why each choice was made, and the cards cite them by file and line.
- **Measured FlyDSL GEMM work**, marked `*(field, E<n>)*`: what moved wall time and what did not.

What each code base's kernels cover, and how they differ: `../flydsl/gemm.md` § INDEX Kernel map.

The cards in this package start from what FlyDSL's guide already says and add the rest. Read the guide first if
you have not built a FlyDSL GEMM before.

**Small M (M ≤ 256, as in decode)** is in `small_m.md`: the kernels there are the baseline aiter's bf16
FlyDSL GEMMs (`splitk_hgemm`, `small_m_hgemm`). Primus-Turbo does not target small M.

Numbers marked `*(field, E<n>)*` are measured values; `E<n>` names the line under the card's `## Evidence` that gives the conditions. Use them for direction and order of magnitude, not as targets.

## Start here

| You are… | Read |
|---|---|
| building your first FlyDSL GEMM, or learning how the pieces fit | `../flydsl/guides.md` § gemm-optimization, then `workflow.md` |
| starting work on a GEMM, or deciding what to do next | `workflow.md` |
| about to choose a tile, split-K or boundary body, or wanting the ceiling of an idea | `grid_pricing.md` |
| working on a GEMM with M ≤ 256 (as in decode or speculative verify), or aiter's FlyDSL bf16 tuned rows | `small_m.md` |
| designing or editing the K loop (2-stage ping-pong, 8-wave pipeline, 4-wave designs, waits, registers) | `mainloop.md` |
| working on LDS, swizzle, global→LDS DMA, B preshuffle, descriptors, transpose reads, MX scales | `datapath.md` (with `../flydsl/guides.md` § lds-optimization) |
| ordering tiles across XCDs, writing scheduling hints, mapping waves | `schedule_and_locality.md` |
| working on the C store, CShuffle, beta=1, or a fused epilogue | `epilogue.md` |
| putting several configurations behind one entry point; launch overhead; CUDA graphs | `autotune.md` |
| finding out where the time goes before changing anything | `attribution.md` (with `../profile.md` § kernel-trace-analysis) |
| writing or trusting a correctness gate | `correctness.md` |
| measuring an A/B, setting a noise floor, deciding which of a series of changes paid | `measurement.md` |
| debugging a wrong result | `../flydsl/debug-flydsl-kernel.md` |

## Kernel map

In FlyDSL: `../flydsl/gemm.md` § INDEX Kernel map.

## FlyDSL version

In FlyDSL: `../flydsl/gemm.md` § INDEX FlyDSL version.

## Cards

| Card | One line |
|---|---|
| `workflow.md` | order of operations, reading the evidence, one change per round, escalation |
| `grid_pricing.md` | dispatch rounds, last-round fill, split-K variants, alternative tiles, persistent tiles, fixed per-tile cost |
| `mainloop.md` | 2-stage ping-pong, 8-wave pipeline, 4-wave designs, inline-asm register plan and constraints, MFMA atom, K tail |
| `datapath.md` | LDS layout and swizzle, DMA and M0, i64 descriptors, transpose reads, row alignment, MX scales |
| `schedule_and_locality.md` | wave placement, XCD/L2 tile order, scheduling hints, wait strategy |
| `epilogue.md` | 64 B write requests, store classes, overlap with the last MFMAs, cache policy, beta=1, fusion |
| `autotune.md` | first-call race, never-regress margins, cache keys, launch path, CUDA graphs |
| `attribution.md` | regime probes, subtractive and additive arms, energy at the power cap, wave-cycle budget |
| `correctness.md` | gates that reach the changed path, races, waits, bounds, register classes |
| `measurement.md` | rulers, arm order, warm-up, noise floors, statistics, closing a series of changes |
| `small_m.md` | GEMM with M ≤ 256: cold rulers, dispatch rounds, split-K reduction cost, tile/stage/cache-policy choices, aiter tuned rows |
| `wait_free_fold.md` | a split-K fold in which no workgroup waits: the last split to arrive folds, in split order |

## gfx942 (MI300X)

`gfx942.md`: where FlyDSL's own skills cover gfx942 (CDNA3: MI300X, MI308) — the LDS budget and its
32 banks, the combined register budget, the synchronous-copy schedule, BF16/FP16 MFMA forms, peak figures — as
links, with this repository's notes from FlyDSL v0.3.2's source (the K=16 path of a bf16-activation kernel, DMA
and transpose reads). Not re-measured here.
