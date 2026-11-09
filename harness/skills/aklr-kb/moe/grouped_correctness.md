---
title: "Grouped GEMM — correctness gates: skewed and empty groups, large G, proven bodies, runtime splits"
kind: method
lever: grouped_correctness
ops: [grouped_gemm]
gens: [gfx950]
dtypes: [bf16, fp8, mxfp8, mxfp4]
regimes: [forward, dgrad, wgrad, variable-k, skew, split-k]
flydsl: 0.3.2
sources:
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_bf16_kernel.py:651-657
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/utils/gemm_helper.py:118-197
  - AMD-AGI/Primus-Turbo@ca1fdd67:benchmark/ops/training/config.py:487-497
  - AMD-AGI/Primus-Turbo@ca1fdd67:tests/pytorch/ops/test_grouped_gemm.py:209-224
---

# Correctness for a grouped GEMM

> What a grouped GEMM's gate must add to a dense one. Most cases marked `*(field, E<n>)*` below passed a gate that
> reported clean; use them to design the gate before trusting it. The general rules — gates that reach the changed
> path, calling twice, races found by bitwise repetition at the timing shapes, waits and barriers, inline-asm
> hazards, random E8M0 exponents — are in `../gemm/correctness.md`. Debugging a kernel already known
> to be wrong starts in `../flydsl/debug-flydsl-kernel.md`. Numbers marked `*(field, E<n>)*` are measured
> values; their conditions are listed under Evidence at the end.

## Route here when
- You changed group lookup, tile decode, descriptors, boundary or tail bodies, barriers, waits, fills, or a
  runtime split policy in a grouped kernel.
- A gate passes on balanced inputs and you have not run it on skewed ones, with empty groups, or at large G.
- You changed a dense kernel or a helper that grouped kernels import.

## 1. Determinism is not correctness

A run-to-run check passes an output that is wrong the same way every time. Every grouped gate compares against
an fp32 reference built from the dequantized operands, with SNR, on skewed distributions as well as balanced.
- *(field, E1)* A wgrad whose tiles over-ran their group read about 6 dB on an unbalanced distribution and passed on
  a balanced one.
  - Balanced groups end on tile boundaries, so an over-running tile never reaches the next group.
- Keep the dense gate's mechanics: at least two calls, NaN-filled outputs between calls, bitwise repetition at
  the shapes you time (`../gemm/correctness.md` §1–§3).

## 2. The distribution set every gate runs

- Balanced.
- Mild random skew. Turbo's benchmark generator draws per-group weights from 0.2–1.0, at most 5:1 between groups
  (`benchmark/ops/training/config.py:487-497`); that is not heavy skew.
- Heavy skew: one group with about half of the tokens. Near one-hot.
- Empty groups: one, two, and many (Turbo's test generator spreads the tokens over the non-empty groups,
  `tests/pytorch/ops/test_grouped_gemm.py:209-224`).
- Group lengths that are not multiples of any tile size, and the deployment's measured histogram.
- A G and sizes large enough to cross 2^31 elements in any offset product (§4).

## 3. Empty and zero-length groups

- **Clamp record counts at zero with a signed max before any unsigned clamp.** A per-tile record count such as
  `M_g · OUT − tile_offset` goes negative for an empty group; a later unsigned minimum against `0xFFFFFFFF` turns
  it into the maximum, the descriptor stops bounding the loads and the kernel faults. *(field, E2)* The fault
  appeared only with empty groups; balanced inputs and other backends ran clean. The fix is one scalar op per
  tile, outside the main loop.
- **Every path that builds a descriptor needs the clamp.** *(field, E3)* Fixing the shared helper did not fix a kernel
  that built its descriptor inline. Search for every descriptor construction before declaring the fix done.
- **Check the empty group's output block.** Zero for beta=0; unchanged for beta=1 (C += 0). The mechanisms,
  Turbo's included, are in `grouped_group_mapping.md` §8 and `grouped_variable_k.md` §11.
- **An uninitialized output fakes nondeterminism.** With a `torch.empty` output, empty groups and padding leave
  regions the kernel never writes, and an output fingerprint picks up whatever the allocator left there.
  *(field, E4)* 5–6 of 24 fingerprint cells changed across six runs of one binary with one seed, on different cells
  each time, including cells without split-K. Fingerprint only the region the contract writes (truncate at the
  last group offset and the real N) or prefill the output; until then, run the baseline against itself at least
  four times and accept differences only on cells that were stable.

## 4. Large G: 64-bit offsets

- A cumulative group offset times a row stride outgrows int32 long before one group does, and the failure is
  silent: the last groups come out wrong with no fault and no error. The measured case and the fix (a 64-bit
  descriptor base per group or tile, bounded by the group's own extent, as Turbo's helpers do per tile,
  `gemm_helper.py:118-197`) are in `grouped_group_mapping.md` §7.
- Gate with a G, a tokens-per-group and an output size whose products exceed 2^31, and check the last groups.

## 5. Out-of-bounds reads that only skew exposes

- **Operand layout decides whether the descriptor can clamp an over-run.** With tokens as the inner dimension
  (MX wgrad operands `[OUT, M_total]`) it cannot, and an over-running K step accumulates the next group's real
  tokens; the guard is a run-time K check (`grouped_variable_k.md` §3). Only an unbalanced gate sees it (§1).
- Values that feed a descriptor, the record count as well as the base, must be wave-uniform
  (`grouped_group_mapping.md` §6, with the base-only case). A per-lane value gives "a non-deterministic
  wrong answer rather than a fault" (Turbo's comment, `grouped_gemm_bf16_kernel.py:651-657`): gate with SNR on
  repeated runs, not with a fault check.

## 6. Health lines per dtype and reference

- **Separate a test's minimum line from the kernel's clean band.** Minimum lines (10 dB for fp4 in some test
  suites) sit far below a correct kernel; measure the baseline's band on each gate cell and treat leaving it, or
  jitter between runs, as the failure. *(field, E5)* One MXFP4 grouped NT kernel's clean band against the
  dequantized reference was 55.38–55.53 dB (55.09–55.53 dB with the wgrad cells of the same gate); *(field, E6)*
  a race that corrupted half of a boundary tile read 36–41 dB, changed every run, and still passed a 10 dB
  minimum.
- **Pick the reference that can see the error.** *(field, E7)* MXFP4 grouped NT outputs read 14.8–16.0 dB against an
  fp32 reference of the unquantized operands, all of it quantization; an error smaller than the quantization noise
  is invisible there. Use the dequantized operands.
- *(field, E8)* Healthy fp8 grouped GEMMs read about 56 dB with rowwise scales and 40–47 dB with tensorwise scales;
  on a wgrad whose clean reading was ~56 dB, output below 54 dB was corrupted; readings of 15 or 29 dB were wrong
  data or races. Judge each kernel against its own band (first bullet).
- **Kernel or upstream?** Run the same inputs through a second backend. Identical SNR to many digits means the
  difference comes from quantization or the reference path, not from the kernel.

## 7. Prove a body ran

A boundary or tail body guarded by a runtime predicate can pass every gate because it never executes.
- **Poison an address only that body uses** (e.g. its scale offset) and check that SNR collapses **only** on the
  distribution that triggers it. *(field, E9)* Balanced: 55.42 / 55.47 dB, unchanged; skewed (3.6% of blocks with a
  ragged remainder): 17.90 / 17.07 dB. Both halves prove something: the body runs, and the predicate does not
  catch the main path.
- To price the code a body adds even when it does not run, make its predicate false **at run time** (an expression
  that is always false but not a compile-time constant); a constant lets dead-code elimination delete the body,
  and the arm collapses into the baseline.

## 8. Barriers, waits and fills

- **Bisect barrier deletions one at a time.** *(field, E10)* In a distance-2 hand-pipelined fp8 per-tensor grouped NT
  loop (8 waves), three "opening" rendezvous around MFMA groups: deleting the first raced, the second was
  bit-identical, the third raced, all three raced. The read-only last K steps were the least deletable: the
  cross-wave visibility of fills issued two steps earlier lands there. The safe deletion was also 1.1–2.3% slower.
  Gate each deletion by bitwise repetition at the timing shapes.
- **A graded `vmcnt(N)` counts instructions, not bytes.** *(field, E11)* Narrowing a boundary tile's fill by dropping
  one half of it, with the wait unchanged, raced (0 of 10 repeats identical). Whether SNR sees it depends on
  whether the stale half reaches the scored output (§6). Narrow both halves (same issue count) or drain fully.
- The safe in-flight tail and multi-pool counting are in `../gemm/correctness.md` §4.

## 9. Runtime splits: cover every split the policy can pick

A kernel that chooses a split factor on the device and divides with a precomputed reciprocal table silently drops
part of the contraction when the table lacks a split the policy can reach — no fault, deterministic output.
- Assert at compile time that the table covers every reachable split.
- Build a gate shape that forces the highest split: *(field, E12)* only a large-M SNR gate (M ≈ 16384, chosen to
  force the high-split branch) caught a missing entry; small-M benchmarks did not leave the low splits, and the
  determinism gate passed the consistently truncated output.
- Gate a split hand-off on a grid larger than one dispatch round: *(field, E13)* racy dense variants reported error 0
  on grids that fit the device at once (`../gemm/small_m.md` §4).

## 10. Changes elsewhere break grouped kernels

- **Rerun the grouped gates after changing a dense kernel or a shared helper.** *(field, E14)* A dense kernel stopped
  importing a module constant that the grouped kernel imported through it, so the grouped module no longer loaded, and
  a dense-only scorer saw nothing. *(field, E15)* Separately, a test suite reported 2108 passes while a grouped module
  failed to import, because its cases ran other backends. Add an import smoke for every grouped module and a build
  smoke for every launch mode.
- **Probes built from contiguous tensors never reach strided branches.**
  - *(field, E16)* On a dense MXFP4 GEMM, a rule that took one operand's allocated row stride for both operands
    passed every contiguous probe.
- **A fallback `except` in a race must record what it swallowed**, so that "never won" and "never built" look
  different in the log (`../gemm/autotune.md` §7, `grouped_host_autotune.md` §8).

## Verify
- [ ] The gate runs balanced, heavy-skew, one-hot and empty-group distributions against a dequantized fp32
      reference, at least twice, with bitwise repetition at the timing shapes.
- [ ] Every descriptor path clamps record counts at zero and uses 64-bit bases; a > 2^31 shape is in the gate.
- [ ] Empty groups produce zeros (beta=0) or leave C unchanged (beta=1); fingerprints cover only written regions.
- [ ] Each new runtime-selected body has a poison probe that fails only on the distribution that triggers it.
- [ ] The highest runtime split is forced by a gate shape; reciprocal tables are asserted complete.
- [ ] Dense or helper changes reran the grouped gates and the import smoke.

## Related
`../gemm/correctness.md` · `../flydsl/debug-flydsl-kernel.md` · `grouped_group_mapping.md` ·
`grouped_variable_k.md` · `grouped_lowprec.md` · `grouped_measurement.md`

## Evidence

A FlyDSL version is given only where the cited run, or another round in the same container and venv, printed it
(`flydsl.__version__`, an import path with its version, or a `pip install flydsl==` in the run's own setup); otherwise
`FlyDSL ?`. `Turbo fork <rev>` names the Primus-Turbo fork commit that ran, where the record gives one.

- E1 [2026-07, gfx950, FlyDSL ?, Turbo fork rev ?, N=?, SNR against a reference, unbalanced and balanced distributions] a wgrad whose tiles over-ran their group read `b_grad_snr` ~6 dB with `balance=False` and passed on balanced groups (grouped MX wgrad; local:/workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/05-mxfp4-mxfp8-deadends.md:276)
- E2 [2026-07, gfx950, FlyDSL ?, Turbo da4074d8, N=?, fault reproducer] with 13 empty groups of 32 the FlyDSL wgrad hit a GPU memory-access fault; the Triton backend, and a balanced distribution without empty groups, ran clean (fp8 variable-K wgrad, gate_up K=2880, N=5760, G=32; local:/workspace/code/gpt_oss_docker/.claude/memory/project_wgrad_empty_group_fault.md:11)
- E3 [2026-07, gfx950, FlyDSL ?, Turbo da4074d8, N=?, fault reproducer] of the three wgrad kernels, the two that build descriptors through the shared rebase helper passed; the 4-wave kernel, which built its descriptor inline through the library call, was the one that faulted (fp8 variable-K wgrad; local:/workspace/code/gpt_oss_docker/.claude/memory/project_wgrad_empty_group_fault.md:15)
- E4 [2026-08, gfx950, FlyDSL ?, Turbo fork rev ?, N=6 runs of one binary, `torch.manual_seed(0)`, 24-cell byte fingerprint] 5–6 cells' fingerprints changed, a different set each run, including NT cells without split-K (gpt-oss MXFP4 grouped GEMM; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260817_092541/rounds/round-16/agent.txt:20)
- E5 [2026-09, gfx950, FlyDSL 0.2.4, Turbo fork rev ?, N=24 gate cells, SNR against fp32 products of the dequantized operands, unequal group lengths] the base read 55.09–55.53 dB: the NT cells (fc2_fwd, fc1_dgrad) 55.38–55.53, the wgrad cells 55.09–55.17; the gate requires SNR > 10 dB (gpt-oss MXFP4 grouped MoE MLP GEMMs, three geometries; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260925_090609/goal.md:65; rounds/round-04/agent.txt:83-84)
- E6 [2026-09, gfx950, FlyDSL 0.2.4, Turbo fork rev ?, N=? (several scored runs), the same SNR gate] with one fill of the half-N boundary body dropped and its drain unchanged, fc2_fwd read 55.42 → 41.35 / 39.74 dB and fc1_dgrad 55.47 → 37.95 / 35.91 dB, varying from run to run and above the 10 dB gate (gpt-oss MXFP4 grouped NT; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260925_090609/rounds/round-04/agent.txt:66)
- E7 [2026-07, gfx950, FlyDSL ?, Turbo fork rev ?, N=18 configurations, SNR against an fp32 reference] narrow and wide stores alike read 15.4 dB (eligible configurations) and 14.8 dB (gpt-oss); 8 waves against fp32 read 16.04 dB, all of it quantization loss and equal to 4 waves (MXFP4 grouped NT; local:/workspace/code/gpt_oss_docker/.claude/memory/project_mxfp4_grouped_fp4vs8_occ.md:34)
- E8 [2026-07, gfx950, FlyDSL ?, Turbo fork rev ?, N=?, ?] healthy FlyDSL fp8 grouped GEMMs read ~56 dB rowwise and ~40–47 dB tensorwise; a wgrad passes at 56 dB and below 54 dB its output is corrupted; collapses such as 15 / 29 / 46 dB were wrong data or races (FlyDSL grouped fp8 production kernels; local:/workspace/code/gpt_oss_docker/myskill/optimizer/methodology/02-correctness-race-cache.md:60)
- E9 [2026-09, gfx950, FlyDSL 0.2.4, Turbo fork rev ?, N=? (two cells per distribution), the campaign SNR gate; reverted, 32 repeats bit-exact] poisoning the M body's A-scale granule left balanced cells at 55.42 / 55.47 dB and took skewed cells to 17.90 / 17.07 dB; under skew 19 of 529 M blocks (3.59%) have a ragged remainder, under balanced none (gpt-oss MXFP4 grouped NT, seed 0, 4096 tokens per group; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260925_090609/rounds/round-08/agent.txt:19)
- E10 [2026-09, gfx950, FlyDSL 0.2.2, Turbo fork rev ?, N=? (both arm orders), same-process element-wise cross-check against the base, inert control arm] deleting the opening rendezvous before c00 raced, before c01 stayed bit-identical but ran 1.1–2.3% slower, before c10 raced; deleting every opening rendezvous raced (gpt-oss tensorwise fp8 grouped NT, distance-2 pipeline, 8 waves, 489 barriers per binary; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260904_151102/memory.md:292)
- E11 [2026-08, gfx950, FlyDSL ?, Turbo fork rev ?, N=10 bit-exact repeats] dropping the b1 half of a boundary tile's fill with the wait unchanged: 0 of 10 repeats identical (gpt-oss-20b dgrad NN, tensorwise fp8 grouped; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260817_065611/memory.md:154)
- E12 [2026-08, gfx950, FlyDSL ?, Turbo fork rev ?, N=?, SNR against a reference] only the large-M gate `_verify_wg_sa.py` (M≈16384, forcing the high-split branch) caught a reciprocal table that missed a reachable split; the small-M bench gate and the determinism gate passed (tensorwise fp8 grouped wgrad with a runtime split-K; local:/workspace/code/gpt_oss_docker/myskill/optimizer/methodology/08-splitk-crosslane-lowprec-wgrad.md:192)
- E13 [2026-09, gfx950, FlyDSL 0.3.2, aiter fork base 5a11152e2, N=?, ?] racy split-K variants reported err 0.00000 on the fully co-resident grids (1,32 / 1,128 / 48,32) (dense bf16 small-M split-K GEMM; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260904_151953/memory.md:229)
- E14 [2026-09, gfx950, FlyDSL ?, Turbo fork rev ?, N=?, ?] after a dense MXFP4 kernel stopped referencing `_MXFP4_PRESHUF_BLK`, which the grouped MXFP4 kernel imported through it, the grouped module no longer loaded; the dense-only scorer did not see it (fixed in Turbo 13da3f1a; local:/workspace/code/gpt_oss_docker/.claude/memory/reference_dense_kernel_breaks_grouped.md:25)
- E15 [2026-08, gfx950, FlyDSL ?, Turbo fork rev ?, N=1 test-suite run, pytest] `tests/pytorch/ops/test_grouped_gemm_fp4.py` reported 2108 passed while `grouped_gemm_mxfp4_kernel.py` failed to import; its cases ran the CK and Triton backends (mxfp4 PR branch; local:/workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/08-sync-build.md:77)
- E16 [2026-09, gfx950, FlyDSL ?, Turbo fork rev ?, N=?, ?] a dense MXFP4 kernel's row-stride rule took operand A's allocated stride for both operands; every probe built from contiguous `torch.randn` tensors passed, since A and B then share a stride (fixed in Turbo 2c9e65ee; local:/workspace/code/gpt_oss_docker/.claude/memory/reference_dense_kernel_breaks_grouped.md:64)
