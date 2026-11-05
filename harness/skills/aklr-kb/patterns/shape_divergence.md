---
title: "Shape divergence: one kernel or configuration wins on some shapes and loses on others"
kind: pattern
signals: ["per-shape deltas of opposite sign under one change", "geometric mean flat while single cells move",
  "race winner flips between cleared-cache re-runs", "a tuned row serving every M that pads to it"]
ops: [gemm]
gens: [gfx950]
flydsl: 0.3.2
basis: source-semantics
---

# Shape divergence: a configuration wins on some shapes, loses on others

## Symptom

- A geometry change that helps one shape moves others the other way (S1); a geometric mean can sit still while one
  cell moves (S2).
- A per-shape race picks different winners across cleared-cache re-runs (S3).
- A tuned row serves every M that pads to it, not only the M it was tuned at (S4).

## Likely causes

1. **A regime boundary between the shapes.** Loop design, tile shape, L2 band, store policy and split-K all change
   sign between shapes (S5); at small M the lessons flip between streaming and latency-bound rows (S6).
2. **Dispatch rounds.** Narrowing pays where the last round is partial, widening where rounds are full (S7, S8).
3. **The kernel's own gates:** "divisible ⇒ fast" is not a rule; block-count parity and tail predicates decide (S9).
4. **Carried code.** A boundary variant costs every shape compiled into its module, including shapes that do not run
   it (S10; `gemm/grid_pricing.md` §8).
5. **Constants from elsewhere:** tile and preload tables, wait counts and L2 thresholds were tuned on other shapes (S11).
6. **The tuned table.** The lookup pads M (S12); an unparsable kernel name falls through with only a warning (S13);
   tuner filters can delete the winner silently (S14); a table raced on another build or machine regresses (S15).

## Rule out first

- **Per-shape noise.** Adopt a per-shape change only above that shape's measured spread (S16). Race noise above the
  margin flips the winner; time base and candidate as an in-process palindrome (S3, S17).
- **The operating point.** Benchmark the deployment's per-call M (S18), each arm at its own best configuration (S19).
- **An isolated loop.** Cache, swizzle and placement levers do not transfer from isolated back-to-back loops (S20).

## Candidate techniques

- Choose per shape — `gemm/autotune.md` §1–2 — incumbent first, challengers by a margin larger than the
  race's noise (S21); a model only finds suspect shapes, so validate any replacement off the scoring set (S22). In
  AKLR the choice comes from the shape, via a table raced offline (S23).
- A static lead for a known shape family, or a hard-coded choice per family — `gemm/autotune.md` §2 —
  when the race scores the wrong point (S24, S25).
- Race both loop designs per shape — `gemm/mainloop.md` §1 — the crossover depends on shape and epilogue (S26).
- A tile/round table for every scored shape, the old tile kept in the race — `gemm/grid_pricing.md` §1,
  §9 (S27, S28).
- aiter tuned rows — `gemm/small_m.md` §7 — validate a row across every M that reaches it and keep it
  only if it beats the pad fallback (S29); re-tune instead of copying rows (S30).
- Close with a full-table ablation that has an all-off arm — `gemm/measurement.md` §9 — keep a change
  only if its sign agrees across cells (S31, S32). In AKLR a speedup is judged per Point (S33).
- Other kernel families: in a grouped GEMM, find where the skew lives before reading a per-shape result —
  `moe/grouped_scheduling.md` §1 — and decide waves per SIMD per pass —
  `moe/grouped_tile_pricing.md` §2; in attention, key the tile heuristic on the axis the grid spreads
  over — `attention/grid_dispatch.md` §6.

## Evidence

- S1 gemm/grid_pricing.md:92 a geometry change that helps one shape moves others the other way
- S2 gemm/measurement.md:91 report per cell: a gm can sit still while one cell moved
- S3 gemm/autotune.md:86 race noise above the margin makes the winner flip between runs
- S4 gemm/small_m.md:270 a row serves every M that pads to it
- S5 gemm/autotune.md:20 loop designs, tiles, L2 bands, store policy, split-K change sign between shapes
- S6 gemm/small_m.md:74 the lessons flip sign between two regimes
- S7 gemm/grid_pricing.md:75 narrowing gained only on the shapes whose last round was partial
- S8 gemm/grid_pricing.md:68 last round full: intensity is the limit, widen
- S9 gemm/grid_pricing.md:249 "divisible ⇒ fast" is not a rule; evaluate the kernel's own gates
- S10 gemm/grid_pricing.md:264 each extra body variant adds code even when it does not run
- S11 flydsl/gemm.md:84 constants in both code bases were tuned on specific shapes
- S12 gemm/small_m.md:259 the lookup tries M as is, then padded at granularity 0 and 1
- S13 flydsl/gemm.md:375 a row whose kernel name the catalog cannot parse falls through
- S14 flydsl/gemm.md:380 tuner filters can delete the winner silently
- S15 ../flydsl/kernels/flydsl_knob_space.md:155 a tuned row regressed: CSV raced on another build or machine
- S16 gemm/small_m.md:66 adopt a per-shape change only above that shape's measured spread
- S17 gemm/autotune.md:90 time base and candidate in one process as a palindrome per rep
- S18 gemm/measurement.md:139 benchmark the deployment's per-call shape
- S19 gemm/measurement.md:141 each arm at its own best configuration
- S20 gemm/measurement.md:75 cache, residency, swizzle, placement levers do not transfer from isolated loops
- S21 gemm/autotune.md:65 incumbent first; a challenger must beat it by more than the race's noise
- S22 gemm/autotune.md:53 a static analysis is a cheap way to find suspect shapes; validate off the scoring set
- S23 aklr/ops/README.md:193 compilation happens on a Point's first call; later calls may be inside a graph capture
- S24 gemm/autotune.md:100 give a known shape family a static lead under a predicate
- S25 gemm/autotune.md:103 hard-coding beats a race that scores the wrong point
- S26 gemm/mainloop.md:91 the crossover depends on the shape and on the epilogue
- S27 gemm/grid_pricing.md:322 the tile/round table for every scored shape exists before the first build
- S28 gemm/grid_pricing.md:323 judged per shape, with s = 1 / the old tile still in the race
- S29 gemm/small_m.md:273 validate a row across every M that reaches it; add it only if it beats the pad
- S30 flydsl/gemm.md:378 re-tune instead of copying rows
- S31 gemm/measurement.md:164 ablate every component on the full table, including an all-off arm
- S32 gemm/measurement.md:169 keep a change only if it beats BASE-to-BASE drift and its sign agrees
- S33 aklr/ops/README.md:21 dispatch serves each Point with the best measured file
