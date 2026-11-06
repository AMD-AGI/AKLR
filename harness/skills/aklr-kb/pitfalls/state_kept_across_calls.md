---
title: "State kept between calls lets a kernel pass the checks it should fail"
kind: pitfall
scope: [launch code and kernels in any AKLR language, kernels written by an optimization loop, harnesses that time or check repeated calls on one set of buffers]
gens: [gfx950]
flydsl: 0.3.2
basis: single-observation
sources:
  - "AMD-AGI/AKLR@a7a8de605 commit message (PR #43)"
  - "AMD-AGI/AKLR@0b6e82fd9 commit message (PR #43)"
  - "AMD-AGI/AKLR@97d5eb0b3 commit message and diff (PR #81)"
  - "AMD-AGI/AKLR@0cef4f2e3:aklr/ops/README.md"
  - "AMD-AGI/AKLR@0cef4f2e3:aklr/core/validation/stages/_bench/_evaluation/_candidate.py:350"
---

# State kept between calls

## Symptom

A kernel passes every correctness check and either times faster than the hardware allows or, in a server, keeps
returning the result for the previous weights after a weight update. Nothing raises.

- **Too fast.** Among kernels written by an optimization loop, one shape of a bf16 GEMM was scored at
  7.93 PFLOP/s against the 2.5 PFLOP/s bf16 peak *(field, E1)*; the worst kernel of that set (bf16 GEMMs and
  MXFP4 fused MoE) was scored at 5.448× over aiter and recomputed 14× slower than aiter once its activations
  changed between calls *(field, E2)*.
- **Stale after an update.** Twenty AKLR schemas (FlyDSL bf16 GEMMs and MXFP4 fused MoE) returned the old
  weights' result after their weights were rewritten in place *(field, E3)*.

## Root cause

Both forms keep something derived from an argument across calls.

1. **An output memo or compute skip keyed on input values.** The kernel keeps a copy of an input and, when the
   next input matches it, skips the work or copies the previous output. Forms found in one loop's kernels (commit
   `0b6e82fd9`): a per-tile XOR of the activations against a saved copy, with the MFMA under
   `if needs_compute != 0`; a `State`/`Cache` pair and a `changed` flag around the whole GEMM body; a host function
   that compares the activations and the routing with stored copies and, on a match, copies the previous output
   over the result. A changed input is recomputed, so every correctness check sees right answers. A harness that
   captures one graph over one set of buffers and replays it gives every timed call an input the kernel has seen,
   and times the hit (commit `a7a8de605`). A dispatch constant can switch the machinery off at the timed shapes
   while it stays in the source.
2. **A derived-weight cache keyed on identity or `_version`.** The launcher repacks a weight once and skips the
   repack while `b is cached_weight` and `b._version == cached_version` (S1). A reload through
   `param.data.copy_` keeps the address and the version, and a CUDA graph replay does not run the Python check
   at all (`aklr/ops/README.md` § The kernel section and ABI), so the kernel keeps reading the old packed weight.
   A speedup measured on cache hits leaves out the repack.

AKLR's first-capture and post-timing checks overwrite reused input buffers, weights included, through
`.data.copy_()` (`aklr/ops/README.md` § Semantics); that fails form 2. Form 1 recomputes on the new values and
passes them, and the bench times every participant on one call's tensors (S2), so a form-1 kernel would read as a
cache hit in AKLR's timing too. That last point is read from the code; no CI run of such a kernel is recorded.
The README's rule that nothing derived from an argument is kept across calls is enforced by audit only for
`global` and `nonlocal`; review catches the rest (`aklr/ops/README.md` § The kernel section and ABI).

## How to detect

- **Time on inputs that change.** Write new values into the input buffers between timed calls (in place, as a
  weight reload does) and, separately, pass fresh buffers. A kernel without kept state reads the same within the
  ruler's noise floor (`../gemm/measurement.md` §4).
- **Compare each shape's throughput with the ceiling** (`../gemm/measurement.md` §7). A reading above
  the peak means the work was not done.
- **Read the source; timing is not enough.** In one loop, timing missed three kernels whose machinery a dispatch
  constant switched off at the timed shapes; a source read found them (commit `0b6e82fd9`). Search launch code
  and device code for `_version`, `data_ptr`, `id(`, `weakref`, `is` comparisons on an argument, `nonlocal`,
  `global`, a module-level container filled from an argument, a saved copy of an input in per-shape scratch,
  and a flag that guards the MFMAs or the whole body.
- **A passing correctness check says nothing here**: both forms compute the right answer on any input that
  changed.

## Fix or workaround

- Repack on every call, or serve the contract's shuffled layout variant where it has one (`../SKILL.md`, the
  table under "Stop: what a AKLR kernel cannot do"). A kernel keeps nothing derived from an argument between
  calls, and no call depends on what an earlier call left (`aklr/ops/README.md` § The kernel section and ABI).
- In an optimization loop, time candidates on inputs that change between calls, and reject a candidate whose
  source keeps argument-derived data, whatever its score.
- To roll back a loop's result, walk its accepted commits back to the newest one whose source contains no such
  machinery. The newest one that times clean can still carry it, switched off at the timed shapes (commit
  `0b6e82fd9`).

## Scope

- Launch code and device code in any AKLR language; the observed kernels are FlyDSL bf16 GEMMs and MXFP4 fused MoE
  on MI355X.
- Not this trap: kernels whose first call differs from later ones (`../gemm/correctness.md` §2).

## Evidence

The KernelForge numbers are the loop's own scores, from the commit messages; no per-shape run record of them is
on this machine.

- E1 [2026-09, gfx950, FlyDSL ?, N=?, KernelForge task timing: one CUDA graph captured over one set of input buffers and replayed] later iterations of gemm_n6144_k6144 peaked at 7.93 PFLOP/s on one shape; five of the seven kernels rolled back reported per-shape throughput above what the hardware can reach (bf16 GEMM, KernelForge FlyDSL campaigns; a7a8de605)
- E2 [2026-09, gfx950, FlyDSL ?, N=?, the same protocol, then the activations moved between calls] the worst of the seven reported 5.448× and recomputed 14× slower than aiter once its activations moved; the record names neither the kernel nor whether the numbers are per case or a mean over cases (bf16 GEMM or MXFP4 fused MoE, KernelForge FlyDSL campaigns; a7a8de605)
- E3 [2026-10, gfx950, FlyDSL ?, N=?, ruler not recorded; the PR's bench check rewrote weights in place through `.data.copy_`] the 20 schemas whose kernels cached a weight keyed on the object and its `_version` returned the old weights' result after an update; their recorded speedups predated the bench's rotating input sets and measured only cache hits (FlyDSL bf16 GEMMs and MXFP4 fused MoE; 97d5eb0b3, #81)
- S1 sikl/ops/gemm/impls/flydsl/gemm_n2624_k6144/225df1ae13e08609.json#kernel.py:1046 at 97d5eb0b3^: `if b is not cached_weight or version is None or version != cached_version:` repacks, else reuses `packed_b`; 97d5eb0b3 removed the check from all 20, and no kernel committed at 0cef4f2e3 reads `._version`
- S2 aklr/core/validation/stages/_bench/_evaluation/_candidate.py:350 at 0cef4f2e3: one `call` built from one input set, timed for every participant (:353)
