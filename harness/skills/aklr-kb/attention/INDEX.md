---
title: "Attention on gfx950 — package map"
kind: index
pinned_source: ROCm/FlyDSL@dd8373349 (v0.3.2; kernels/attention)
---

# Attention — package map

How to make an attention kernel fast and correct on gfx950 (MI350/MI355X), whatever the shape: many query rows per
head against the KV (flash attention, causal, windowed or rectangular), the gradients dQ, dK and dV, and few query
rows per head against a long KV (split KV and a combine, often gathering from a sparse/top-k or paged KV). The cards
hold measured lessons — what moved wall time, what did not, and how to tell which is which before spending a week
on it. They are mechanisms, diagnostics and decision rules, not a catalogue of kernels or configurations. Each card
states the mechanism and its numbers; how to write it in FlyDSL is in `../flydsl/attention.md`, under the card
section it serves.

- FlyDSL facts, here and in `../flydsl/attention.md`, are stated for v0.3.2. Instruction rates and MFMA timings
  were measured on gfx950 with HIP inline-asm microbenchmarks, not inside a kernel.
- Numbers marked `*(field, E<n>)*` were measured on FlyDSL attention kernels unless the sentence names another
  language (Triton, Gluon, HIP) or another kind of measurement (an inline-asm microbenchmark, a PyTorch model, a
  simulator); their conditions are listed under Evidence at the end of each card. The mechanisms are stated in ISA
  and LLVM terms; each number was measured in one language and not re-checked in the others. Read them as
  magnitudes: they show how large an effect was where it was measured, not what it will be on your kernel.
- Signed percentages of wall time, throughput or a benchmark score are speed changes (+ = faster) unless the
  sentence says faster or slower or the card states otherwise; a sign next to a named quantity (cycles, sclk, a
  counter, bytes, registers, an error norm) is the change of that quantity.
- *Occupancy N* means N co-resident workgroups per CU; waves per SIMD are stated as such. An 8-wave workgroup
  alone on a CU is occupancy 1 at 2 waves per SIMD.
- Most attention bodies are latency- and rendezvous-bound, not throughput-bound. Several lessons below
  exist because a throughput explanation looked right and was wrong.

## Route by shape

The same mechanisms apply wherever the shape occurs; pick the rows that describe your call.

| Shape of the call | What usually decides its time | Read |
|---|---|---|
| Many query rows per head against the KV (a long sequence, as in prefill) | MFMA atom, softmax VALU under the MFMAs, transposed LDS reads, occupancy, barriers | `workflow.md`; `mfma_layout.md` §1, §5; `softmax.md` §1–§3; `lds.md` §1–§5; `registers_occupancy.md` §8; `schedule_sync.md` §2–§4 |
| Causal, sliding-window or rectangular (Sq ≠ Skv) masks | unequal work per q-tile, where the padding falls, the axis the grid spreads over, mask conventions | `grid_dispatch.md` §4–§6; `gradients.md` §8; `correctness.md` §9; `measurement.md` §5, §8 |
| Few query rows per head (one or a few query tokens per sequence, as in decode) against a long KV | how the KV range is split, the combine as a fixed cost, CTAs per CU, launch and host time | `split_kv.md` §1–§5; `grid_dispatch.md` §2, §3; `mfma_layout.md` §5; `measurement.md` §6; `schedule_sync.md` §5, §10; `softmax.md` §6.3, §6.5; `registers_occupancy.md` §8 |
| KV rows gathered by index: sparse/top-k selection or a paged pool | unique 128 B lines per row, the index distribution, bytes rather than FLOPs | `split_kv.md` §6–§7; `measurement.md` §5; `attribution.md` §9 |
| Gradients dQ, dK, dV | dQ workspace bytes and its reduce, determinism, the fusion form, masked q-loops | `gradients.md` §1–§8; `grid_dispatch.md` §2, §10; `correctness.md` §5; `measurement.md` §4 |
| GQA: several q heads per kv head | XCD placement of a kv head's sharers, sharer merging, pipelining along the head axis | `grid_dispatch.md` §2, §9–§10 |
| fp8 operands or fp8 P | only the wide-K fp8 MFMAs raise throughput; scales along the contraction; the error budget | `mfma_layout.md` §8; `softmax.md` §6.1, §6.5 |

## Route by symptom

| What you see | Read |
|---|---|
| Body tuning has stopped paying | `workflow.md` Step 2; `grid_dispatch.md` §1 |
| A counter reads "LDS-bound", "latency-bound" or "X% waiting" | `attribution.md` §2–§4; `lds.md` §5 |
| `v_exp_f32` is the largest VALU class | `softmax.md` §1–§2; `schedule_sync.md` §1–§2 |
| Spills, or one more wave per SIMD is wanted | `registers_occupancy.md` §1–§6 |
| The combine or a second launch is a visible share of the call | `split_kv.md` §1–§3 |
| A gathering kernel does not respond to compute-side changes | `split_kv.md` §6–§7; `attribution.md` §9 |
| The dQ workspace, its reduce or an atomic alternative dominates | `gradients.md` §2–§6 |
| A causal, windowed or rectangular shape is slower than its FLOPs suggest | `grid_dispatch.md` §4–§6; `gradients.md` §8 |
| NaN or inf, one output wrong while the others are right, or a metric at exactly 1.000 | `correctness.md` §1–§8; `softmax.md` §4, §6.1 |
| A delta under ~1%, a noisy cell, or two rulers that disagree | `measurement.md` §1–§4, §6 |

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
| computing dQ, dK and dV: accumulation, determinism, fusion | `gradients.md` |
| splitting the KV range for few query rows per head, with a combine step | `split_kv.md` |
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
| `mfma_layout.md` | MFMA timing on gfx950, atom choice per GEMM (the O kernel, the dQ/dK/dV kernels, P·V with few query rows), S^T, row sums through MFMA, fragment layouts, fp8 atoms |
| `softmax.md` | v_exp_f32 cost, how exp2 lowers, LSE formats, online rescale, why a fixed reference max is not a general lever |
| `lds.md` | gfx950 transpose reads and their lane maps, 16-element blocks, row alignment, bank-conflict counters |
| `registers_occupancy.md` | the 512-register pool, occupancy steps, spills, MFMA–VALU co-execution and when dual-wave pays |
| `schedule_sync.md` | sched_barrier and sched_group_barrier (TRANS mask for exp), setprio, LLVM scheduler options, hazards, waits, prefetch, barriers |
| `gradients.md` | determinism by construction, the D/A law for dQ partials, atomics, workspace layout, fusion forms that win and lose, masked loops |
| `split_kv.md` | few query rows per head: combine as a fixed cost, cooperative rendezvous, the CTA/CU cliff, split-K rules, gather locality |
| `measurement.md` | attention-specific A/B protocol, FLOP accounting, host-bound small shapes, coverage traps |
| `correctness.md` | SNR + determinism gates, spike inputs, shared-LSE end-to-end gates, window conventions, probes |
