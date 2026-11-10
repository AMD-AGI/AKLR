---
title: "GEMM — correctness gates that actually exercise the change, races, and silent-wrong traps"
kind: method
lever: gemm_correctness
ops: [gemm, grouped_gemm, fused_moe]
gens: [gfx950]
dtypes: [fp8, bf16, mxfp4, mxfp8]
regimes: [pipelined, split-k, inline-asm]
flydsl: 0.3.2
sources:
  - AMD-AGI/Primus-Turbo@7615a5a2:primus_turbo/flydsl/gemm/gemm_fp8_kernel.py
  - AMD-AGI/Primus-Turbo@7615a5a2:primus_turbo/flydsl/gemm/gemm_mxfp4_kernel.py
  - ROCm/FlyDSL@dd8373349:kernels/gemm/gemm_a16w16_gfx950.py:362
  - ROCm/FlyDSL@dd8373349:python/flydsl/expr/rocdl/__init__.py:142
---

# Correctness for a GEMM

> A GEMM that is fast and wrong usually passed its gate. Each case marked `*(field, E<n>)*` below went through
> a gate that reported clean. Use them to design the gate before trusting it. Numbers marked `*(field, E<n>)*`
> are measured values; their conditions are listed under Evidence at the end. Use them for direction and order
> of magnitude, not as targets. Debugging a kernel that is already known wrong starts in
> `../flydsl/debug-flydsl-kernel.md`. "A16" is FlyDSL's `kernels/gemm/gemm_a16w16_gfx950.py`.

## Route here when
- You changed the pipeline, a wait, a barrier, the LDS layout, register classes or inline asm.
- A gate passes but you have not shown it runs the code you changed.
- Outputs differ between runs, or only on some shapes, or only after the first call.

**Static checks first; they cost nothing.** LDS stays within 160 KiB per workgroup on gfx950; tile packing and
vector widths match the operand layout; the swizzle or padding is applied identically by the writer and the
reader (`../flydsl/guides.md` § gemm-optimization (FlyDSL's §3.4)); accumulator and output conversions cannot overflow; edges hold
for shapes that do not divide the tile.

## 1. A gate must execute the path you changed

- **Coverage first, repetition second.** A race shows up as a jittering error. A path the gate
  never runs shows up as a stable, clean number that looks exactly like a pass, and running the gate
  more times cannot change that. A gate whose shapes gave the kernel only one or two K blocks made the
  runtime dispatcher take the short branch, so the steady-state loop never ran.
  - *(field, E1)* A steady-state bug in a grouped MXFP4 GEMM passed repeatedly at 55.09 dB; with four times the K
    the same code read 19.40 dB on one of its two weight-gradient GEMMs (55.46 dB on the other).
- Before trusting a gate, compute for its shapes the trip counts the kernel derives (`k_iters`,
  phases per tile, groups that reach the steady loop). A 0 or 1 is an alarm. Keep a second probe
  that reuses the gate's reference and scales the shape up.
- Include the operands production feeds: padded or strided rows, unequal strides, ragged M/N/K.
  A probe that only uses contiguous tensors never reaches the strided branch.
- A previously kept change is not a trusted baseline until a gate that covers it has passed.

**Shapes every GEMM gate needs.** Each one forces a path that only runs at an edge:
1. **The minimum trip count.** Take a K where the main loop runs zero times (e.g. `K = 2·BLOCK_K` for a
   loop of `K_ITERS-2` iterations after a two-tile prologue). A loose prologue wait that relies on "the loop
   will drain it later" then has nothing after it, and the epilogue reads LDS that is still being filled.
   `s_barrier` orders waves; it does not drain DMA. Turbo's dense fp8 8-wave kernel special-cases exactly this
   (`gemm_fp8_kernel.py:257-262`).
2. **More tiles than CUs.** Only the second tile on a CU can read what the first one left in LDS; the
   first tile finds an empty slot and computes the right answer. Turbo's MXFP4 kernel leaves the k=1 prologue
   fill in flight only for a caller whose loop wait was computed with that fill outstanding; a caller
   written against a drained prologue reads k=1 before it lands. Skipping the drain is an explicit opt-in
   flag (`k1_watermark`), and the flag is part of the cache key (`gemm_mxfp4_kernel.py:1510-1525`, 765).
3. **The deepest branch of every runtime choice** (the largest split factor, the widest tile, the
   persistent path). Pick a shape large enough to force it.
4. **Ragged M, N and K, and K that is not a multiple of the K tile**, for every layout the kernel
   accepts.
5. **The same gate for kernels you did not write.** A shipped kernel's default configuration is also an
   input to verify, not a reference.

Run each gate with at least two calls, fill C with NaN between calls, compare against an fp32
reference built from the dequantized operands, and compare repeated outputs bitwise (§3).

## 2. Call twice; compare from the second call

*(field, E2)* A kernel was clean on its first call after compilation (55.54 dB) and 33 dB from the
second call on, with different elements wrong on each call. The harness scored the first call, so the race
stayed hidden. Every gate calls the kernel at least twice and checks the second and
later results. Work that depends on a first call (atomic counters, semaphores, lazily initialized
workspaces) must be warmed before a probe starts counting.

## 3. Races: bitwise repetition, at the timing shape

- Detect a race by repeating the same inputs and comparing outputs **bitwise**, not by SNR: on a grouped fp8
  GEMM, a race that flipped about 0.17% of repeated runs barely showed in SNR *(field, E3)*.
- Repeat enough: *(field, E4)* 200 runs looked clean where 500 exposed a 1.5e-5 race. Use fresh random
  inputs per pass, and for rates near 1e-4 tens of thousands of runs.
- Repeat **at the shapes you time**. *(field, E5)* Narrow shapes of a grouped MXFP8 GEMM passed a determinism
  check while the timing shapes had 134–389 bad rows; *(field, E6)* M = 32768 failed every time while M ≤ 8192
  was clean 8/8.
- Deeper prefetch fails first: a race that only appears on the largest K is pointing at a
  pipeline-drain seam.
- **Test anything one workgroup hands to another (a split-K fold) where residency and timing change**: on streams
  limited to part of the CUs (16 and 64), beside a second launch of the same kernel, with two input sets
  alternating between calls so that a stale partial always shows. *(field, E23)* Such a test caught partials stored
  without `sc0 sc1` on every launch; leaving out the `vmcnt(0)` before the counter or the fold loads' cache bits did
  not fail it, so keep what the ISA requires even when the test passes.

## 4. Waits, barriers and the scheduler

- **Read-write distance decides how much may stay in flight.** With two buffers the next phase reads
  what was just written (distance 1): drain fully. With three (distance 2) a partial drain is safe,
  and only then.
- **A partial `vmcnt(N)` must count every pool still in flight**, not the first one. *(field, E7)* Two
  three-deep pools of a grouped fp8 GEMM left 8 writes in flight while the wait accounted for 4: a 1-in-30000
  race.
- **A longer in-flight tail is less safe even when the issue-order count says it is enough.**
  *(field, E5)* On a grouped MXFP8 GEMM, a wait sized to leave a whole iteration in flight raced on the wide
  shapes; keep the tail at most one iteration's issue count minus its first group, and prove a longer one by
  repetition.
- **`spill ≠ 0` voids every hand-written partial `vmcnt`.** Scratch loads and stores share `vmcnt`
  with your DMA, so `vmcnt(1)` no longer means "one DMA left". Read `.private_segment_fixed_size`
  and the spill counts first. When spills are zero and it still races, show those numbers
  (`llvm-readelf --notes`) as the evidence: the race has another channel, and more `vmcnt(0)` will not fix
  it.
- **Waits written as instruction counts break when the wave count changes.** Recompute them per tile,
  not per instruction, after changing the CTA's waves; the failure signature is "all outputs right,
  determinism passes, every shape passes" until it is not.
- **An inline-asm `s_waitcnt vmcnt(0)` does not stop LLVM from moving an earlier load below it,** where
  the wait no longer covers it. *(field, E8)* In a grouped MXFP4 GEMM, two drains before an epilogue that
  reused the operand LDS still let the next tile's global→LDS prefetch sink past them and land on staged data,
  only on the deepest-K shape. `sched_barrier(0)` after the drains fixed it (0/10 bad reads vs 3/30) at
  about 0.7% of time. A mask that let ALU/MFMA cross (`0x40F`) made it worse on one token distribution (6/11
  bad, against 1/15 for the clean tree; 0/11 on the other).
- `s_barrier` does not drain `vmcnt`; `lgkmcnt` is 4 bits (max 15), `vmcnt` 6 bits (max 63).
- **No result may depend on another workgroup being resident.** A spin on another workgroup's flag or data
  assumes that workgroup is running; on a stream limited to part of the CUs, or beside another process, it may not
  be, and a capped spin that then writes anyway returns garbage. *(field, E22)* A split-K GEMM whose splits polled
  returned millions of NaN on a 16-CU stream. Fold split-K by arrival instead (`small_m.md` §4), or bound the wait
  and hand the share to the split that arrives last (`../attention/split_kv.md` §2, Bounded wait; not measured on
  a GEMM fold).
- **Read another workgroup's data with a load that misses the CU cache: `sc1` (aux=16), `nt` (aux=2) or `sc0 sc1`.**
  A default or `sc0` load hits the CU's stale copy (`../hardware/gfx950/memory_and_sync.md` §2.4), even when both workgroups
  are on one XCD; `buffer_inv sc0` is a no-op outside TG-split mode.
- **Reading a kernel that polls** (aiter's generic split-K kernel spins): LLVM deletes a bounded poll loop whose
  body has no `s_sleep` and reads an unbounded one's flag once, `sc` bits notwithstanding. The volatile aux bit
  (bit 31) keeps the loop but waits `vmcnt(0)` after every load (10.3–15.1 µs against 5.15–9.41 µs at M = 8–112
  *(field, E9)*); a monotonic, agent-scope atomic load keeps it as one `global_load … sc1` per pass.

In FlyDSL: `../flydsl/gemm.md` § correctness §4.

## 5. LDS placement and DMA

- **An LDS-DMA `M0` base must never go negative.**
  - *(field, E11)* A ring that pre-decremented one slot's base by a K step (so one immediate could serve both
    the global and LDS address) rotated that role onto the slot at LDS offset 0. Wave 0's `M0` underflowed and
    its writes were dropped — not wrapped — so the result was deterministic and looked unlike a race: exactly
    one tile index wrong (55.5 → 14.8 dB), or, with a static rotation, rows 0–31 of every tile wrong
    *(field, E12)*.
  - Keep any skewed role off the slot at offset 0.
- **Separate LDS buffers are laid out by the backend, not in declaration order.** An arm that stops using
  one lets the others be re-packed, so read `group_segment_fixed_size` before interpreting an LDS-placement result.
  A pool addressed as one base plus immediates must be one allocation.

In FlyDSL: `../flydsl/gemm.md` § correctness §5.

## 6. Bounds

- **MFMA ignores EXEC.** With EXEC set by inline asm to lanes 0–31, 32–63, lane 0 only, or none,
  `v_mfma_f32_16x16x32_bf16` writes full results in all 64 lanes, while a `v_mov` in the same asm block
  is masked as expected. A divergent `if tid < 32:` around an MFMA
  therefore overwrites the accumulators of the "inactive" lanes; when the operand loads were only used
  inside the branch, LLVM sank them into it and every lane came out as garbage. Bound ragged tiles through
  the buffer descriptor (operands read as zero) or select whole tiles; never mask an MFMA with a branch.

- Bound ragged edges with the descriptor's `num_records`: out-of-range loads return zero and
  stores are dropped at no instruction cost. *(field, E13)* On a MoE scale-scatter kernel the descriptor bound
  cost nothing measurable (MoE operator score 1.13583 against 1.13511 unbounded); a software predicate in its
  store loop has been timed only with debug instrumentation in place, so its own cost is not known.
- Bound indices derived from data with the buffer's real size in the descriptor, not a maximal `num_records`.
- **A store masked by swapping in a sentinel byte offset (0x7FFFFFFF) is dropped only when `num_records` really
  bounds it.** Under a maximal `num_records` (`0xFFFFFFFF`) that offset is in range, so a "masked" lane writes 2 GiB past the base
  *(field, E16)*; in a fused MoE sort kernel such a store faulted, and an out-of-range index in its place ran
  correct *(field, E19)*.
- **Guard a load by its offset, not by `if`.** Each `if`-guarded load gets its own branch and `s_waitcnt vmcnt(0)`
  *(field, E17)*; on a MoE sort kernel 8 such row loads took 14.3–18.3 µs, 9.2–10.5 with an offset past the bound *(field, E18)*.
- Prove a grid or index decode is an in-bounds permutation offline before timing it: an
  out-of-bounds write can look like a speedup.

In FlyDSL: `../flydsl/gemm.md` § correctness §6.

## 7. Register classes and inline asm

- **asm that writes an MFMA accumulator into AGPRs hides the hazard from the compiler.** gfx950 needs
  several wait states between an MFMA writing an AGPR and a `v_accvgpr_read` of it, more for MFMAs with
  more passes. LLVM's gfx950 hazard model inserts 8 for `v_mfma_f32_16x16x32_bf16` and for fp4
  `v_mfma_scale_f32_16x16x128_f8f6f4`, 12 for its fp8 form, and 20 for fp8 `32x32x64`. Read the count for
  your instruction from an `llc` probe with the intrinsic. *(field, E14)* Two of 64
  accumulators lost the last K iteration's contribution (SNR still 37–45 dB). Put the wait states in
  the asm string of the tail iteration only.
- **The reverse too:** asm that *reads* an MFMA destination loses the compiler's `s_nop`. A drop in
  `s_nop` count after moving code into asm is a red flag, not a saving. *(field, E21)* On a split-K GEMM an inline-asm
  `v_max3_f32` that read the accumulators right after the main loop got no wait states and about 1.5% of the
  outputs came out wrong, by exact powers of two; written as `math.absf` and `arith.maxnumf`, the compiler emitted
  the same `v_max3_f32` with its wait states (seen once).
- **An untied asm pin (`"=a,v"`) silently deletes the operand.** It compiles, runs, removes spills and
  runs faster, because it emits no copy and the output is uninitialized.
- **Moving an asm output from AGPR to arch VGPR needs `&` (early clobber).** Without it the register
  allocator may legally place an input (an LDS base, an offset) inside the output range. Signature:
  some shapes right, others collapse, correlated with tile count rather than K; every ISA count stays
  the same. `&` is incompatible with a tied (`"0"`) operand.
- In `$n` numbering, tied inputs count: in `"=v,0,v,v"`, `$1` is the tied input.
- The constraint rules for a whole-loop asm region (literal `src2=0` for a zero accumulator, `~{scc}`
  around an asm loop, every `=&s` output written) are in `mainloop.md` §4; how to set the
  `amdgpu-agpr-alloc` attribute in FlyDSL is in `../flydsl/gemm.md` § mainloop §4.

## 8. Reproducibility and tolerance

- **Split-K with atomics is not bitwise reproducible.** A split-K that adds its partials into the bf16 output
  with packed atomics (`buffer_atomic_pk_add_bf16`) sums in arrival order and rounds each partial to bf16 as it
  lands. Gate it on a calibrated tolerance against an independent reference,
  not on bit equality with the previous build. In AKLR, check the schema's `compare.determinism`
  (`aklr/ops/README.md` § Schema fields): under `bitwise`, atomic split-K fails outright, so reduce the splits
  in a fixed order.
- A changed reduction order is a changed result: compare against a third reference rather than
  old-vs-new maximum difference.
- Timed autotuning that picks configurations per run makes outputs differ across runs; pin the
  configuration for determinism checks.
- Compare only the region the kernel contract writes, using the harness's own input tensors.

In FlyDSL: `../flydsl/gemm.md` § correctness §8.

## 9. Low precision: validate in order

1. conversion and packing; 2. scale handling; 3. the math against a dequantized reference; only then
4. performance. Block-scaled probes need random E8M0 exponents (e.g. `torch.randint(122, 132)`), not
scales quantized from `randn` data: N(0,1) blocks almost always share one power-of-two exponent, so a
kernel applying the wrong scale set reads the same as a correct one.
   - *(field, E15)* A tail phase of a grouped MXFP4 GEMM that kept reusing the scales from two phases earlier
     read 49.6 dB with `randn`-derived scales and 11–14 dB the moment the exponents were random.

## Related
`../flydsl/debug-flydsl-kernel.md` · `mainloop.md` · `datapath.md` ·
`schedule_and_locality.md` · `measurement.md`

## Evidence

A FlyDSL version is given only where the cited run, or another round in the same container and venv, printed it
(`flydsl.__version__`, an import path with its version, or a `pip install flydsl==` in the run's own setup); otherwise
`FlyDSL ?`. The Kimi-K3 forge runs installed `flydsl==0.3.2`. The month is the run's where the record dates it; where
only a note records the number, the line says so. "Turbo" is Primus-Turbo.

- E1 [2026-09, gfx950, FlyDSL 0.2.4, N=1, SNR probe built on the gate's own reference with only `m_per` (rows per group, the K of a weight-gradient GEMM) changed; one reading per token distribution] a quarter-M boundary body that reused the N body's half-width steady-state fill: at `m_per=256`, the gate's value (1–2 K blocks per group, so the dispatcher takes the `nval<4` branch and the steady-state loop never runs), fc1_wgrad 55.09 / fc2_wgrad 55.09 dB; at `m_per=1024` 55.46 / 19.40 dB (19.35 on the skewed distribution); the gate had kept reporting 55.09 dB (gpt-oss-20b MXFP4 grouped MoE MLP, the two variable-K weight-gradient GEMMs)
- E2 [2026-09, gfx950, FlyDSL 0.2.4, N=?, SNR of each call of one compiled closure, the harness's process state rebuilt from its own functions] the `_ROWSPLIT` path (fp4 row pitch ≡ 64 mod 128): first call after compilation 55.537 dB, every later call 33.136 dB (a second probe: call 0 55.54, calls 1 and later 33–34 dB); `compile_with_scratch_out` runs the kernel once while compiling and the harness's SNR read that call; 20 of 20 fixed-seed repeats differed bitwise (gpt-oss dense MXFP4 GEMM, K=2944, `g_QKV_fwd` and `g_op_fwd`)
- E3 [2026-07, gfx950, FlyDSL ?, N=?, bitwise repeat test] bit flips at about 0.17% of repeated runs barely show in SNR; only bitwise repeats of 30000 or more runs catch them (Turbo grouped fp8 tensorwise weight-gradient GEMM, 4 waves; unverified)
- E4 [2026-05, gfx950, FlyDSL ?, N=?, bitwise repeat gate (det) over 200, 500 and 1000 runs] the path-J `ds_read_b64_tr_b8` read hint `vmcnt_hint` ≥ 4 raced at det 1.5e-5, which surfaced only at ≥ 500 runs; 200 runs gave a false pass; `vmcnt_hint=3` det 0 over 1000 runs (Turbo dense fp8 tensorwise TN GEMM, 8192×8192×28672; Primus-Turbo 424fe26b)
- E5 [2026-07, gfx950, FlyDSL ?, N=?, repeated runs compared bitwise by the agent's probe; the bench's own det gate on narrow shapes] the A1 pool moved to prefetch distance 2 with `vmcnt(8)`, a whole iteration in flight by issue order: at N=2944 389 and at N=5632 134 of 131072 rows differed, maxabs 16.0, while the bench's det gate on narrow shapes passed; `vmcnt(6)`, one iteration's issue count minus its first group, 4 shapes × 4 repeats bitwise identical (gpt-oss MXFP8 grouped GEMM, NT forward)
- E6 [2026-09, gfx950, FlyDSL 0.2.4, N=8, fixed-seed bitwise repeat probe, 8 repeats per M] N=5120, K=2944, scanning M: M=32768 (2560 tiles) 8 of 8 differed; M=8192 (640 tiles), 4096 (320) and 2048 (160) 0 of 8 (the kernel and round of E2)
- E7 [2026-07, gfx950, FlyDSL ?, N=1, bitwise repeat test over 30000 runs] the first two-pool three-buffer build: 1 race in 30000 runs; the wait counted one pool's writes (`_n_outstanding = nsb`, 4) while two three-buffer pools left 2×nsb = 8 in flight; with the sum counted, 0 of 30000 on all 7 shapes (Turbo grouped fp8 tensorwise weight-gradient GEMM, 4 waves; the build became Turbo's default)
- E8 [2026-09, gfx950, FlyDSL 0.2.4, N=10, corrupted gate readings counted across repeated bench runs (the gate data are identical every run), per arm and token distribution; price from same-session palindrome pairs B A A B B A A B, 4 reads per arm] DeepSeek-V3 geometry (K=7168; the gpt-oss and Qwen3 geometries, K 2880 and 4096, clean): clean tree 1/15 + 2/15 = 3/30 readings corrupted; `sched_barrier(0)` after the two drains, together with an SRD hoist shown bitwise-equivalent, 0/5 + 0/5; the mask `0x40F` on the clean tree 6/11 on the balanced distribution and 0/11 on the skewed one; hoist + `sched_barrier(0)` −0.72 gm (pairs +0.65 / +0.32 / +1.07 / +0.82 on scores of 107–110), the hoist alone +0.03 gm (gpt-oss-20b-shaped fused MoE FC2 dgrad, an MXFP4 grouped GEMM with a dGLU and dual-quantization epilogue)
- E9 [2026-10, gfx950, FlyDSL 0.3.2, N=?, the record lists driver runs (warmup 3, 20 iterations) and AKLR bench_cudagraph A/B runs and does not say which timed these two arms] one reducer polling all split partials with the volatile aux bit set (`sc0 sc1`): m=8–112 at 10.3–15.1 µs, suite median 0.2884 ms; the same reducer without it: m=8 5.150, 24 6.138, 40 7.483, 64 7.970, 80 8.968, 96 9.412, 112 8.646 µs, suite median 0.2234 ms; per-M times with the bit set are not recorded (gemm_n144_k7168 forge run)
- E11 [2026-09, gfx950, FlyDSL ?, N=?, correctness bisect probe; wrong outputs bitwise repeatable] a three-slot A ring whose rotation put the role with the base pre-decremented by a K block on the leaf at LDS offset 0: SNR 55.535 → 14.8 dB, exactly 512 of 2048 256×256 blocks wrong (one tile index per workgroup), `block_rel_max` 1.25; with the period-4 slot table ((0,1),(2,1),(0,2),(1,2)) 55.535 dB and `|C|sum` bitwise equal to the baseline (MXFP4 dense NT GEMM, whole-loop kernel)
- E12 [2026-10, gfx950, FlyDSL ?, N=?, correctness probe on the campaign's shapes] a barrier-free three-slot rotation, static per tile: rel 8e-4, M rows 0..31 of every tile wrong in all N columns (wave 0's first M0 window, 4096 B); slot pairs (0,1), (0,2), (1,2) passed and (1,0), (2,1), (2,0) failed (MXFP4 dense dgrad/wgrad GEMM, training shapes)
- E13 [2026-09, gfx950, FlyDSL ?, N=1, operator-level score of the MoE bench (noise 0.2%), arms paired in the same period, one reading each for HEAD and the descriptor arm] the bound `(row_ok) & (row < len)` in the buffer descriptor (`num_records_bytes`) 1.13583 against unbounded HEAD 1.13511; the same bound as a software predicate in the store loop 1.1029, reported as the reading with debug instrumentation in place; no reading of the predicate without it is recorded. Whether the 1.1029 → 1.1358 gap is the instrumentation's host overhead or the predicate's cost is unresolved (GLM-5.2 TP4/EP4 MoE operator, the quantization scale scatter)
- E14 [2026-09, gfx950, FlyDSL ?, N=1, first correctness test of six shapes, then a per-K-iteration probe (`--keep-ki`)] inline-asm MFMAs writing their accumulators to AGPRs with no wait states before `v_accvgpr_read`: all six shapes failed at SNR 37–45 dB; the errors sat in 2 accumulator quads per wave, and only the tail K iteration's contribution was lost; with 18 wait states after the tail iteration's MFMAs, five shapes were bitwise equal to the baseline and the sixth within the baseline's own split-K nondeterminism (MXFP8 dense NT GEMM, five production linears with activation quantization)
- E15 [2026-08, gfx950, FlyDSL ?, N=1, the agent's SNR probe, same session, md5-gated, interleaved with the baseline] a folded tail phase (`emit_peel_fold`) that never re-issued the trailing half K block's scales: 49.60 dB with scales quantized from `randn` data; with random E8M0 exponents (`torch.randint(122, 132)`) 10.98 / 11.04 / 11.13 / 14.00 dB on the four scored fwd/dgrad shapes; the bench's SNR shapes (K=1024 and 256) never ran the fold (gpt-oss MXFP4 grouped GEMM, fwd/dgrad)
- E16 [2026-10, gfx950, FlyDSL 0.3.2, N=1, GPU probe: one 64-lane uint8 `buffer_store` with every lane masked off, into a 64-byte view of a 3 GiB tensor] `create_buffer_resource(max_size=True)`: the byte at base + 0x7FFFFFFF was written; `max_size=False, num_records_bytes=64`: nothing written (aiter/ops/flydsl/kernels/buffer_ops.py:370, :674; range rule `../hardware/gfx950/memory_and_sync.md` §2; a fused MoE sort + quant kernel's first form hit `HSA_STATUS_ERROR_MEMORY_FAULT` with masked stores)
- E17 [2026-10, gfx950, FlyDSL 0.3.2, N=1, compile-only, `21_final_isa.s`] 8 row loads under per-row `if`: 8 `s_and_saveexec` / `s_cbranch_execz` blocks, each load followed by `s_waitcnt vmcnt(0)` before the next branch; the same loads with an out-of-range offset selected: 8 `buffer_load_dword` back to back, then `vmcnt(6)`…`vmcnt(0)`
- E18 [2026-10, gfx950, FlyDSL 0.3.2, N=?, per-kernel Kineto median, 1 GiB read flush before each call] a fused MoE sort + quant kernel prefetching 8 rows per thread at 40 / 64 / 112 tokens: loads guarded by per-row `if` 14.26 / 15.96 / 18.32 µs, unguarded 9.52 / 10.34 / 10.64, out-of-range offset for missing rows 9.16 / 10.22 / 10.48 (896 experts, top-16, model dim 3584)
- E19 [2026-10, gfx950, FlyDSL 0.3.2, N=1, one run] a scatter fused into a MoE sorting kernel, storing through a max-size descriptor with `buffer_store(mask=)`: GPU memory fault, a write to a read-only page; the same kernel with an out-of-range element index instead ran and matched (max_diff 0) (fused MoE EP8 prefill, n_tokens 9397 and 16384)
- E21 [2026-10, gfx950, FlyDSL 0.3.2, N=1, single observation, the fix session's correctness check] `_pack16e` reading MFMA results through inline-asm `v_max3_f32` / `v_max_f32_e64`, once the pack ran straight after the main loop: about 1.5% of elements wrong, off by exact powers of two; the same max written as `math_d.absf` + `arith.MaxNumFOp` compiled to `v_max3_f32` with the wait states and passed (bf16 split-K GEMM n6144 k16384; comment "plain ops, not inline asm" at aklr/ops/gemm/impls/flydsl/gemm_n6144_k16384/8f7ffe4ce850ba6e.json#kernel.py:144)
- E22 [2026-10, gfx950, FlyDSL 0.3.2, N=?, the campaign's stress test: CU-masked streams of 16/32/64/128 CUs and a second launch beside the kernel] splits that polled flags and planes up to `_MAX_POLLS`, then wrote NaN: up to 4.7 M NaN at m=256 and m=512, calls of up to 140 ms on 16 CUs; the arrival-fold rewrite passed every mask (bf16 split-K GEMM n6144 k16384)
- E23 [2026-10, gfx950, FlyDSL 0.3.2, N=340 launches, ordering test: two input sets alternating, full, 64-CU and 16-CU streams, a second launch of the kernel alongside] the arrival fold: 0 differing launches of 340; with the partial stores' `sc0 sc1` removed, every launch differed; without the `vmcnt(0)` before the counter, or without the fold loads' cache bits, no launch differed (bf16 split-K GEMM n6144 k16384)
