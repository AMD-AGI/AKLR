---
title: "State kept between calls lets a kernel pass the checks it should fail, or time faster than it runs in serving"
kind: pitfall
scope: [launch code and kernels in FlyDSL, Triton / Gluon or HIP, kernels written by an agent tuning run, harnesses that time or check repeated calls on one set of buffers, ops that update state in place]
gens: [gfx950]
flydsl: 0.3.2
basis: single-observation
sources:
  - cases/bf16_glm52_narrow_n.md
  - cases/bf16_glm52_wide_n.md
  - cases/bf16_k4096_splitk_hgemm.md
  - cases/fused_moe_shuffled_16x16.md
  - cases/gdn_step_gluon.md
  - cases/mhc_fused_post_pre.md
---

# State kept between calls

## Symptom

A kernel passes every correctness check and either times faster than the hardware allows or than it runs in
serving, or, in a server, keeps returning the result for the previous weights after a weight update. Nothing raises.

- **Too fast.** Among kernels written by an agent tuning run, one shape of a bf16 GEMM was scored at
  7.93 PFLOP/s against the 2.5 PFLOP/s bf16 peak *(field, E1)*; the worst kernel of that set (bf16 GEMMs and
  MXFP4 fused MoE) was scored at 5.448× over aiter and recomputed 14× slower than aiter once its activations
  changed between calls *(field, E2)*.
- **Stale after an update.** Twenty kernels (FlyDSL bf16 GEMMs and MXFP4 fused MoE) returned the old
  weights' result after their weights were rewritten in place *(field, E3)*.
- **Faster in the bench than in serving, on an op that updates its state in place.** A Qwen3.5 GDN decode kernel
  in Gluon read 2.40× its baseline with the state replayed in place, as a bench of graph replays replays it,
  and 1.62× with the state refreshed before every call *(field, E5)*; its lower-bound speedup
  (`../gemm/measurement.md` §1), which replays the state in place, read 3.08× *(field, E6)*.

## Root cause

The first two forms keep something derived from an argument across calls; in the third the harness keeps it.

1. **An output memo or compute skip keyed on input values.** The kernel keeps a copy of an input and, when the
   next input matches it, skips the work or copies the previous output. Forms found in one tuning run's kernels:
   a per-tile XOR of the activations against a saved copy, with the MFMA under
   a flag that the compare sets; a saved output and a state buffer, with the whole GEMM body under a flag that says
   an input changed; a kernel launched ahead of the stages
   that compares the activations and the routing with stored copies and, on a match, copies the previous output
   over the result. A changed input is recomputed, so every correctness check sees right answers. A harness that
   captures one graph over one set of buffers and replays it gives every timed call an input the kernel has seen,
   and times the hit. A per-shape switch in the kernel can turn the machinery off at the timed shapes
   while it stays in the source.
2. **A derived-weight cache keyed on identity or `_version`.** The launcher repacks a weight once and skips the
   repack while the weight is the object it packed last and its `_version` has not changed (S1). A reload through
   `param.data.copy_` keeps the address and the version, and a CUDA graph replay does not run the Python check
   at all, so the kernel keeps reading the old packed weight.
   A speedup measured on cache hits leaves out the repack.
3. **Work that depends on the state's history, timed on state replayed in place.** The op's state is an input the
   kernel updates in place (`delta_state`, `conv_state`), and the GDN decode kernel stores only the 64 B segments of it whose
   bits changed (code: `../cases/gdn_step_gluon.md` §8). Replayed on one call's tensors, each call starts from the state the last one left; the
   one-token recurrence settles toward a fixed point, fewer segments change and the kernel stores less: from 100% of
   the `delta_state` segments at calls 1–4 to 52% at call 64 and 20% after 400, and no `conv_state` stores from call 4
   *(field, E5)*. A serving step brings new tokens, so its state changes on every call. The kernel itself keeps
   nothing.

A check that overwrites reused input buffers, weights included, through
`.data.copy_()` fails form 2. Form 1 recomputes on the new values and
passes it, and a bench that times every participant on one call's tensors reads a form-1 kernel as a
cache hit. A static check for
`global` and `nonlocal` catches only that part of the kept state; a source read catches the rest. A form-3
kernel is right on any state, so no check fails it, and a bench that replays the state in place times it
on state near its fixed point.

## How to detect

- **Time on inputs that change.** Write new values into the input buffers between timed calls (in place, as a
  weight reload does) and, separately, pass fresh buffers. On a ruler that flushes the caches before each call,
  a kernel without kept state read within 1% with fresh buffers; without
  the flush, fresh buffers read cold (a memory-bound GEMM 20% slower) and an in-place write cost about as much
  with the old values as with new ones *(field, E4)*: compare against the same protocol fed the old values.
- **Time an op that updates its state in place with the state refreshed.** Copy the state back from its initial
  draw before every timed call, outside the window, and compare with in-place replay: a kernel that reads faster in
  place while the baseline does not has work that depends on the state's history (form 3). Count what one call
  stores, from the initial state and after many in-place calls.
- **Compare each shape's throughput with the ceiling** (`../gemm/measurement.md` §7). A reading above
  the peak means the work was not done.
- **Read the source; timing is not enough.** In one tuning run, timing missed three kernels whose machinery a per-shape
  switch turned off at the timed shapes; a source read found them. Search launch code
  and device code for `_version`, `data_ptr`, `id(`, `weakref`, `is` comparisons on an argument, `nonlocal`,
  `global`, a module-level container filled from an argument, a saved copy of an input in per-shape scratch,
  and a flag that guards the MFMAs or the whole body.
- **A passing correctness check says nothing about form 1**: it computes the right answer on any input that
  changed. Form 2 does too on a new tensor or after a `_version` bump; it fails a check that rewrites a reused
  input unseen by the cache: through `.data.copy_`, or before a graph replay.

## Fix or workaround

- Where the caller can change an argument's contents without the kernel seeing it (a weight rewritten in place,
  a call replayed from a CUDA graph), keep nothing derived from that argument between calls: repack on every call,
  or take the weight already in the packed layout where the caller can pass it that way. A cache keyed on the
  object or its `_version` is not sound there. Scratch that holds no argument's data is another matter: aiter
  keeps its split-K counters and workspace across calls (`../flydsl/kernels/flydsl_kernel_library.md` § Before you
  reuse one), which is right as long as every call leaves them as the next one expects.
- In a tuning run, time candidates on inputs that change between calls, and reject a candidate whose
  source keeps argument-derived data, whatever its score.
- For an op that updates its state in place, quote the speedup timed with the state refreshed before each call, not
  the one timed on state replayed in place (form 3).
- To roll back a tuning run's result, walk its accepted commits back to the newest one whose source contains no such
  machinery. The newest one that times clean can still carry it,
  switched off at the timed shapes.

## Scope

- Launch code and device code in any language (FlyDSL, Triton / Gluon, HIP); the observed kernels are FlyDSL bf16 GEMMs and MXFP4 fused MoE
  on MI355X (forms 1 and 2) and a Gluon GDN decode (form 3).
- Not this trap: kernels whose first call differs from later ones (`../gemm/correctness.md` §2).

## Evidence

The numbers of E1 and E2 are the tuning run's own scores; their per-shape values are
unverified.

- E1 [2026-09, gfx950, FlyDSL ?, N=?, an agent tuning run's task timing: one CUDA graph captured over one set of input buffers and replayed] later iterations of the N 6144 × K 6144 kernel peaked at 7.93 PFLOP/s on one shape; five of the seven kernels rolled back reported per-shape throughput above what the hardware can reach (bf16 GEMM, FlyDSL kernels written by an agent tuning run)
- E2 [2026-09, gfx950, FlyDSL ?, N=?, the same protocol, then the activations moved between calls] the worst of the seven reported 5.448× and recomputed 14× slower than aiter once its activations moved; which kernel it was, and whether the numbers are per case or a mean over cases, is not known (bf16 GEMM or MXFP4 fused MoE, FlyDSL kernels written by an agent tuning run)
- E3 [2026-10, gfx950, FlyDSL ?, N=?, ruler not recorded; a bench check rewrote weights in place through `.data.copy_`] the 20 kernels that cached a weight keyed on the object (or its `id` or address) and its `_version` returned the old weights' result after an update; their recorded speedups, timed on one input set, measured only cache hits (FlyDSL bf16 GEMMs and MXFP4 fused MoE)
- E4 [2026-11, gfx950 (MI355X), torch 2.11.0+rocm10.0.0, N=7 rounds, floor = largest gap between two identical arms; per-kernel CUDA-graph timer (1 GiB flush per call), and eager Kineto kernel time without a flush] `torch.mm` bf16 at 4096×4096×4096, 16×8192×8192 and 16×2048×2048: with the flush, fresh buffers +0.10% to +0.87%; without it, at 16×8192×8192, fresh buffers +20.2%, an in-place write +1.1% with new values and +0.9% with the old ones (floor 0.93%) (hipBLASLt bf16 GEMMs)
- E5 [2026-11, gfx950 (MI355X), Triton of the sglang 0.5.22 image (no FlyDSL kernel), N=200 rounds, the kernel and the baseline (sglang 0.5.22's GDN decode) alternating with the order swapped on odd rounds, each on its own buffers; before each timed call a 1 GiB write and a GPU spin of at least 300 µs, CUDA events, medians] the Gluon GDN decode kernel at n_tokens 64, n_slots 127: state replayed in place, kernel 20.8 µs, baseline 49.8 (2.40×); `delta_state` and `conv_state` copied back from the initial draw before every call, kernel 30.8, baseline 49.8 (1.62×); share of the live slots' `delta_state` segments one kernel call stored, from the initial state: 100% at calls 1–4, 83% at 16, 52% at 64, 31% at 200, 20% after 400; `conv_state` 0% from call 4 (Qwen3.5 GDN decode, Gluon)
- E6 [2026-11, gfx950 (MI355X), Triton of the sglang 0.5.22 image (no FlyDSL kernel), N=1, lower-bound speedup over sglang 0.5.22's GDN decode, one run, the state replayed in place] the Gluon GDN decode kernel at n_tokens 64, n_slots 127: 3.08, kernel 12.9 µs
- S1 the launcher of each of the 20 kernels of E3 kept the packed weight with the weight object, or its `id` or address, and the weight's `_version`; it repacked when either differed from what it had stored and reused the packed weight otherwise (these kernels have no public copy)
