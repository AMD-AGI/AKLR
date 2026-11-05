---
title: "Attention on gfx950 — package map"
kind: index
pinned_source: ROCm/FlyDSL@dd8373349 (v0.3.2; kernels/attention)
---

# Attention — package map

How to make an attention kernel fast and correct on gfx950 (MI350/MI355X): flash forward, flash
backward (dQ, dK, dV) and decode with split KV. The cards hold measured lessons — what moved wall time, what
did not, and how to tell which is which before spending a week on it. They are mechanisms, diagnostics and
decision rules, not a catalogue of kernels or configurations. Each card states the mechanism and its numbers;
how to write it in FlyDSL is in `../flydsl/attention.md`, one section per card section.

- FlyDSL facts, here and in `../flydsl/attention.md`, are stated for v0.3.2. Instruction rates and MFMA timings
  were measured on gfx950 with microbenchmarks.
- Numbers marked `*(field, E<n>)*` come from measured attention work (A/B on real kernels); their conditions are
  listed under Evidence at the end of each card. Read them as magnitudes: they show how large an effect was where
  it was measured, not what it will be on your kernel.
- Signed percentages of wall time, throughput or a benchmark score are speed changes (+ = faster) unless the
  sentence says faster or slower or the card states otherwise; a sign next to a named quantity (cycles, sclk, a
  counter, bytes, registers, an error norm) is the change of that quantity.
- *Occupancy N* means N co-resident workgroups per CU; waves per SIMD are stated as such. An 8-wave workgroup
  alone on a CU is occupancy 1 at 2 waves per SIMD.
- Most attention bodies are latency- and rendezvous-bound, not throughput-bound. Several lessons below
  exist because a throughput explanation looked right and was wrong.

## Start here

| You are… | Read |
|---|---|
| starting on an attention kernel, or deciding what to try next | `workflow.md` |
| finding out where the time goes (counters, probes, clocks) | `attribution.md` (with `../gemm/attribution.md`) |
| mapping the grid: XCD order, dispatch order, GQA head grouping | `grid_dispatch.md` |
| choosing the MFMA atom, laying out fragments, using S^T | `mfma_layout.md` |
| writing the softmax: exp, LSE, online rescale, numerics | `softmax.md` |
| working on LDS: transpose reads, layouts, bank conflicts | `lds.md` (with `../flydsl/guides.md` § lds-optimization) |
| fighting registers or occupancy, or overlapping MFMA with VALU | `registers_occupancy.md` |
| placing barriers, waits, scheduling hints, prefetch | `schedule_sync.md` (with `../gemm/schedule_and_locality.md`) |
| writing the backward pass: dQ/dK/dV accumulation, determinism, fusion | `backward.md` |
| writing decode or split-KV with a combine step | `decode_split_kv.md` |
| measuring an A/B on attention | `measurement.md` (with `../gemm/measurement.md`) |
| writing or trusting a correctness gate | `correctness.md` (with `../flydsl/debug-flydsl-kernel.md`) |
| writing it in FlyDSL: API spellings, the JIT and its cache, FlyDSL's own attention kernels | `../flydsl/attention.md` |

## FlyDSL's own attention code (v0.3.2, `kernels/attention/`)

In FlyDSL: `../flydsl/attention.md` § INDEX FlyDSL's own attention code (v0.3.2, `kernels/attention/`).

## Cards

| Card | One line |
|---|---|
| `workflow.md` | decision order: pin the bound, audit the grid, count redundant MFMA, then hide latency; what is usually dead |
| `attribution.md` | real clock, counter blind spots, which counter ranks which change, stall pools, subtraction and addition probes, power caps |
| `grid_dispatch.md` | XCD-major mapping, dispatch order as list scheduling, padding, GQA sharer merging |
| `mfma_layout.md` | MFMA timing on gfx950, atom choice for forward and backward, S^T, row sums through MFMA, fragment layouts, fp8 atoms |
| `softmax.md` | v_exp_f32 cost, how exp2 lowers, LSE formats, online rescale, why a fixed reference max is not a general lever |
| `lds.md` | gfx950 transpose reads and their lane maps, 16-element blocks, row alignment, bank-conflict counters |
| `registers_occupancy.md` | the 512-register pool, occupancy steps, spills, MFMA–VALU co-execution and when dual-wave pays |
| `schedule_sync.md` | sched_barrier and sched_group_barrier (TRANS mask for exp), setprio, LLVM scheduler options, hazards, waits, prefetch, barriers |
| `backward.md` | determinism by construction, the D/A law for dQ partials, atomics, workspace layout, fusion forms that win and lose, masked loops |
| `decode_split_kv.md` | combine as a fixed cost, cooperative rendezvous, the CTA/CU cliff, split-K rules, gather locality |
| `measurement.md` | attention-specific A/B protocol, FLOP accounting, host-bound small shapes, coverage traps |
| `correctness.md` | SNR + determinism gates, spike inputs, shared-LSE end-to-end gates, window conventions, probes |
