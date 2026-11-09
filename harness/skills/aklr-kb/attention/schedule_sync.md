---
title: "Attention — scheduling hints, fences, priorities, waits, barriers and DMA pipelining"
kind: topic
lever: attn_schedule_sync
ops: [attention]
gens: [gfx950]
dtypes: [bf16, fp8]
regimes: [prefill, backward, decode, causal]
flydsl: 0.3.2
sources:
  - ROCm/FlyDSL@dd8373349:kernels/attention/flash_attn_utils.py:74
  - ROCm/FlyDSL@dd8373349:kernels/attention/flash_attn_utils.py:1754
---

# Scheduling, synchronisation and pipelining in an attention kernel

> An attention body interleaves MFMAs with a transcendental-heavy softmax, LDS reads and DMA, all between
> barriers. The hint vocabulary (mask bits, greedy group filling, mask rules for unrolled epilogues), the
> single `vmcnt` and LDS-DMA waits are in `../gemm/schedule_and_locality.md` §3–§4 and
> `../gemm/datapath.md` §4; how to rank scheduling arms is in `attribution.md` §2
> and §5. This card adds what attention adds. Regime first: when co-resident waves already fill the
> MFMA shadow (no bare VALU between MFMAs), reordering rarely pays; where bare VALU remains it still can, and it
> paid more at 2 waves/SIMD than at 3 (§4). At one workgroup per CU, interleaving inside the workgroup is the only
> cover (`registers_occupancy.md` §8).
>
> Numbers marked `*(field, E<n>)*` are measured values; their conditions are listed under Evidence at the end.
> Use them for direction and order of magnitude, not as targets.

## Route here when
- You are writing or changing `sched_group_barrier` recipes, `sched_barrier` fences, `iglp_opt`, `s_setprio`.
- You want to move work across a barrier, delete a wait or change a wait depth.
- You are adding a prefetch, a second LDS buffer, or cache hints to K/V/Q/dO loads.
- A scheduling change came back numerically wrong on some shapes.

## 1. Scheduling primitives and masks
- **exp is TRANS, not VALU.** `0x400` selects transcendentals including `v_exp_f32`; `0x002` (VALU) does
  not; `0x008` matches both 16x16 and 32x32 MFMAs. A `{1 MFMA, 2 VALU} × 4` recipe left every `v_exp`
  after the last MFMA; only `{1 MFMA, 2 TRANS}` placed exps between MFMAs.
- FlyDSL's gfx950 forward uses `0x008 / 0x002 / 0x400` groups of `{1 MFMA, n VALU}` and
  `{1 MFMA, 3 EXP}`, halved at D=64, and wraps every rendezvous as
  `sched_barrier(0); s_barrier; sched_barrier(0)` (`_dualwave_sync_barrier`).

In FlyDSL: `../flydsl/attention.md` § schedule_sync §1.

## 2. `sched_group_barrier` recipes
- **Classes not named in a region's groups go to the end of the region.** *(field, E1)* A recipe naming only MFMA
  and DS-read groups cut its read stalls as intended but pushed the region's unnamed exp chain behind every group:
  −7.7%, VGPR, AGPR and LDS unchanged. Name every class the region contains.
- **Naming a class pins it.** *(field, E2)* A 1-read-per-MFMA group placed each LDS read beside its consumer:
  read-to-wait distance 32.4 → 19.7, 3.48% slower. To issue reads early, lead with a group larger than the
  region's read count.
- **Count every MFMA in the region, row-sum MFMAs included.** *(field, E3)* A D=64 forward QK region held 8
  `32x32x16` plus 4 `16x16x32` row-sum MFMAs; 8 anchors let the row sums take half and serialized the QK
  chain after the exps. With 9 registers freed first, 10 anchors gave +1.3% in a same-process lab (+0.8% on the
  official bench) and let LLVM pack 38 VALU ops into 32 `v_pk_fma_f32`; the same anchors without the freed
  registers −0.3%; 12 broke the packing. D=128
  sat at 16, ±2 costing 0.5%. Sweep per head_dim, on the current register budget.
- **A recipe governs only its own region.** `sched_barrier`, `s_setprio` and `s_barrier` cut regions.
  *(field, E4)* A recipe placed after an internal `sched_barrier(0)` steered a region holding 2 MFMAs; moving
  or deleting it produced byte-identical ISA. After restructuring (e.g. loop unswitching) whole recipe
  sweeps can go inert.
- **Hash the opcode stream of the final ISA before benchmarking an arm:**
  `grep -o "^\s*[a-z_0-9]*" <isa>.s | md5sum`. Equal hashes cost no GPU time.

In FlyDSL: `../flydsl/attention.md` § schedule_sync §2.

## 3. `iglp_opt`
- One strategy per region. Position inside the region and repeated calls compile byte-identically; the
  only choices are which regions get a call and which id. *(field, E5)* A software-pipelining change that
  merged two regions destroyed the exp interleave of `iglp_opt(3)`: 32 exps became one bare run,
  `lgkmcnt(0)` 53 → 119.
- `iglp_opt` and `sched_group_barrier` drive the same LLVM mutation. *(field, E6)* Where a hand recipe plus fences
  already interleaved exps, adding `iglp_opt(2)` gained nothing; *(field, E2)* replacing a working `iglp_opt` with
  hand groups started 0.9% behind.

## 4. `sched_barrier` fences: price each site
- **A fence wrapping a real `s_barrier` is a tax.** With mask 0 nothing crosses, so the matrix pipe idles
  through the barrier wait. MFMAs touch no LDS: relaxing both fences to `0x008` keeps DS ordering across the
  rendezvous. *(field, E7)* +0.6–0.84% at D=128, ~0 at D=64. *(field, E8)* On another forward, removing both fences
  +0.61–0.96% (16 exps moved ahead of the barrier); removing either one alone changed nothing or lost.
  Prove safety by counting `ds_*`/`buffer_load` per barrier-delimited region: equal counts mean no memory
  op crossed.
- **Cluster-boundary fences do work.** *(field, E9)* Removing the fence at the head of a memory cluster: −1.5%
  (spills 31 → 35); a DS-read-only mask (`0x100`) was flat. *(field, E10)* Deleting (not relaxing) the fence before
  the row-sum MFMAs inside a compute cluster cost 2.5% on another forward. Relaxing such a fence to `0x008` paid
  where bare VALU remained between MFMAs (next bullet); *(field, E11)* on the other forward partial masks read
  −0.15 to −0.50%: price each site.
- **A register fence inside a compute cluster may relax to `0x008`.** *(field, E12)* Three such fences relaxed
  together: +3.1% on the two-cell score, all of it from the D=128 cell at 2 waves/SIMD (that cell 0.930 → 0.9855),
  noise at 3 (D=64); `.vgpr_count` unchanged, spills 24 → 22 at D=64 and 0 → 1 at D=128. Count
  `naked_work` (VALU and TRANS between adjacent MFMAs) in the ISA: when it reaches 0 the lever is spent.
- An MFMA-only fence makes MFMAs close ranks on both sides, leaving the fenced block with no cover.
- **A source-level LDS read prefetch needs a fence per MFMA.** LLVM sinks each read beside its MFMA (depth 4 in
  source, 2 in flight); `sched_barrier(0)` after each MFMA kept 5, same VGPRs. *(field, E44)* +2.6% with
  `s_setprio(1)` over the run, 1.2–1.8% without (`../gemm/schedule_and_locality.md` §3).

## 5. `s_setprio`
- Only bits [1:0] count: priorities 0–3. *(field, E13)* 1 vs 2 or 3 is byte-identical apart from the immediate
  and measured flat; all waves running one stream compare only in-region against out-of-region.
- Raise priority only where this wave has an MFMA run and its co-resident siblings do not, at ≥ 2 waves per
  SIMD on an MFMA-pipe-serialised body (removing it there −2.8%) *(field, E14)*. In a dual-wave forward protect the
  VALU-dense window instead (priority on the pure-MFMA QK region −0.45 to −1.9%; releasing the P·V
  window early −3.5%) *(field, E15)*.
  - Always drop back to 0.
- **It is a scheduling anchor.** *(field, E16)* Deleting 4 `s_setprio` at occupancy 1 (inert by hardware
  arbitration): −65 instructions, two regions merged, `lgkmcnt(0)` 44 → 88, −4.7%, and a dS fence race
  (NaN). Price a removal together with the reschedule.
- Sign depends on regime. *(field, E17)* `s_setprio(1)` around a backward GEMM1 at occupancy 2: −3.3% (it starves
  the co-resident workgroup); forward sweeps with priority off: −0.8% *(field, E18)* and −2.4% *(field, E19)*.
- With producer waves feeding MFMA waves, do not raise the MFMA waves. *(field, E45)* On an MLA decode, priority 2 on
  them (softmax, or QK and P·V) was 3–7% slower on every shape; on the producers, 1% faster to 3% slower.

## 6. LLVM scheduler strategy and options
- The strategy is the LLVM option `amdgpu-sched-strategy` (values such as `max-ilp`, `max-memory-clause`,
  `iterative-ilp`). Options are LLVM `cl::opt`s; an unknown strategy value (e.g. `max-occupancy`) is accepted and
  silently ignored.
- Confirm the opcode hash changed (§2), then rerun SNR and determinism.
- On a body not yet hand-scheduled: *(field, E20)* `max-memory-clause` with post-RA scheduling +0.6–0.7%,
  `max-ilp` +0.4%; `iterative-ilp` was fastest but reordered hand fences and hazard guards into NaN.
- On a hand-scheduled body only the downside is left: *(field, E21, E42)* backward −0.35% to −0.85%,
  *(field, E22)* forward −0.1% to −1.1%.
- LLVM's SLP vectorizer packs independent scalar f32 FMAs into `v_pk_fma_f32`; the LLVM option
  `slp-threshold=100000` keeps them scalar. *(field, E46)* With the softmax exponent rewritten as one scalar fma
  per score, together with the threshold: 0 packed ops on the main path, +6.85% on an fp8 MLA decode (the
  threshold's own share unpriced).

In FlyDSL: `../flydsl/attention.md` § schedule_sync §6.

## 7. Hazards
- Inline-asm results are invisible to the hazard recognizer (`../gemm/correctness.md` §7).
- *(field, E23)* MFMAs consuming an inline-asm `v_cvt_pk_bf16_f32` (the f32 → bf16 pack of P) directly got `s_nop 0`
  instead of the required wait states: non-deterministic, half the q rows (alternate groups of 4) biased, 24.99 dB.
  Fix: `sched_barrier(0)` + `s_nop 1` before those MFMAs, or a compiler-emitted `v_cvt_pk_bf16_f32` (an f32 → bf16
  `fptrunc`).
- **A clamp can be the hazard carrier.** When an inline-asm instruction reads an MFMA result, a compiler-visible
  VALU op on that value (a `min(s, 0)` that never fires on real data) is what makes LLVM insert the wait states.
  - *(field, E24)* Every variant without it failed the SNR gate; *(field, E25)* reducing it from one per element
    to one per 4-element vector: 1.1% (dK/dV) to 2.4% (dQ) less kernel time; one anchor per kv-half read 3.5%
    less on dQ but reads only one vector of its group, the hole the next bullet describes.
- **An anchor must read the exact value it protects.** *(field, E26)* One anchor per group instead of per
  vector: 0.31–2.53% faster, 16.9–19.7 dB and non-deterministic, only at q_split 4 with BLOCK_KV 128 or
  at BLOCK_KV 64; the main shapes passed. Gate hazard edits on a shape grid with bitwise determinism
  (`correctness.md`).

In FlyDSL: `../flydsl/attention.md` § schedule_sync §7.

## 8. Waits and barriers
- **Before moving an instruction across a barrier, ask which write-after-read its `s_waitcnt` protects.**
  `s_barrier` waits on no counter (`../hardware/gfx950/memory_and_sync.md` §5). *(field, E27)* MFMAs moved past
  a single-slot LDS barrier took the waits that retired the previous head's reads with them; an
  `s_waitcnt lgkmcnt(0)` before the barrier restored it at zero cost.
- **A hand-written `lgkmcnt(0)` right before a barrier is a tax** when every read it drains already has a
  consumer before the barrier: the consumers' graded waits retire them. A barrier that carries workgroup-scope
  fences (`fence syncscope("workgroup") release` before `s_barrier` in the LLVM IR) also gets its own
  `s_waitcnt lgkmcnt(0)` from the backend; a bare `llvm.amdgcn.s.barrier` gets none on gfx950 (LLVM adds a full
  wait before `s_barrier` only on targets without barrier back-off, `SIInsertWaitcnts.cpp`; check the ISA). That
  drain gives visibility, not write-after-read safety: delete the
  hand-written one only after checking, per buffer, that the LDS each read targets is overwritten no earlier
  than two barriers later. *(field, E28)* +0.54%; the same removal in prologue and
  epilogues another +0.19–0.32%.
- **A full drain can sink to the cluster that consumes it** when the buffer it guards is overwritten two
  barriers later. *(field, E29)* +1.4%; without a stagger gate an 8-wave configuration degraded (rel_l2
  0.0027 → 0.0294) while the scored SNR did not move.
- **Count depth, not waits.** `lgkmcnt` ≤ 15, `vmcnt` ≤ 63, `expcnt` ≤ 7 on gfx950. *(field, E30)* Moving a
  prefetch ahead of partial stores: `vmcnt(0)` 16 → 1, wait depths (7..0) → (11..4), +3.9%; grouping DMA
  cut `vmcnt` waits 89 → 18 but turned graded waits into full drains: −5.3%.
- A barrier is also a scheduling boundary and, in a dual-wave body, the phase lock. Price a
  barrier family at N, N/2 and 0 and re-price after nearby edits (`attribution.md` §5); the
  rendezvous pool tracks co-resident workgroups and barrier-domain width
  (`grid_dispatch.md` §9).

In FlyDSL: `../flydsl/attention.md` § schedule_sync §8.

## 9. Moving work into exposed segments
- Cover the first uncovered wait of an exposed segment with MFMAs whose operands are already in registers.
  *(field, E31)* Handing 8 trailing MFMAs to sit right after a transposed-read burst, ahead of its first wait:
  +0.8–0.9% raw, +0.74% corrected (two sessions).
- **Dose ≤ the target body's ring depth.** *(field, E32)* Within the depth: 0.96%, then 1.20–1.35% fewer wave
  cycles; one past it 14.4% more, two past 74.3% more. Take the dose from the target body, not the source.
- **Place the hand-off right after the reads it covers.** *(field, E32)* The same 8 MFMAs after the whole next
  GEMM bought more cover and cost 23.3% more wave cycles: no other MFMA segment may sit between hand-off and
  reads.
- Ask who covers the wait left behind. *(field, E32)* The same move on a body whose MFMAs were already covering the
  approach to a fence: 1.98% more wave cycles (on another D=64 build the same move read +0.9% wall *(field, E31)*;
  the two readings are unreconciled).
- Batching reads loses to covering waits. *(field, E31)* Grouped reads −1.4% against the covered hand-off.

## 10. Pipelining and DMA
- **One in-order `vmcnt` for loads and stores:** issue a prefetch before heavy partial stores, never after.
  *(field, E30)* Same prefetch, before the stores +3.4%, a few instructions after them −2.4%.
- **An LDS DMA in flight forces `vmcnt(0)` before every `ds_read` of the same allocation** (mechanism in
  `../gemm/datapath.md` §4). *(field, E33)* With a second K/V buffer, drains 16 → 85 and 109. With
  compiler-visible reads, DMA overlaps only register-resident work; asm reads that carry their own waits avoid the
  drain (`../gemm/datapath.md` §4) at the price of hand-placed `lgkmcnt` (`lds.md` §2).
- **K/V double buffering usually loses: it hides latency; measured cost was throughput.** *(field, E34)* dq
  `BLOCK_KV` 96/64/32 double-buffered: −2.9/−3.2/−2.3%; *(field, E33)* a 256-VGPR body duplicated hoisted LDS
  addresses into 424 B of scratch: −16.5%. The winning form is a single buffer with a hand-off: issue the next DMA
  where the transposed-read registers die (`SQ_WAIT_ANY` −41% *(field, E35)*). At one workgroup per CU a second
  slot that removes a write-after-read barrier can pay (+2.4%, +3.1–3.8% with a prefetch retune on one dQ kernel)
  *(field, E36)*, but it is not a rule: *(field, E37)* on another such backward it lost 6.0% because the second
  slot moved the next head's `ds_write` burst into a transposed-read burst (`lds.md` §9);
  *(field, E38)* on a dual-wave forward, deepening the ring to drop one barrier lost 2.2% (LDS doubled, spills 4 →
  9; the deeper ring alone cost 1.65%).
- One barrier per iteration needs 2 buffers when every wave reads a buffer in the iteration after its refill
  (FlyDSL's `kernels/gemm/preshuffle_gemm.py:509-524`, `pipeline_2stage`; *(field, E43)*), 3 when some waves
  read it a step later.
- A deeper ring relaxes write-after-read only, never read-after-write.
- Register prefetch of cold per-head loads (LSE, delta): `grid_dispatch.md` §10.
- `buffer_load … lds` gets no cross-workgroup vL1D reuse; judge request pressure by `TCP_TCC_READ_REQ`,
  not by DMA instruction count (`grid_dispatch.md` §2, §9).
- **Non-temporal hints on attention operand streams were negative.** *(field, E39)* Q `nt` −0.95% (waves of one
  workgroup share Q lines); dkdv Q/dO DMA `nt` −7.6% *(field, E33)* (reused across workgroups); O store `nt` +0.11%
  (noise). Before marking a stream `nt`, ask whether lanes or waves of one workgroup share its lines.
  Workspace partials are in `gradients.md` §5.
- **Decode prefetch depth:** a prologue depth of 2 pays only at ≤ 2 CTAs per CU. *(field, E40)* It was 1.7–2.7%
  faster up to 512 CTAs (2 per CU on 256 CUs) and 2.0–2.5% slower from 576 up; *(field, E41)* a steady-state depth
  ≥ 2 was 4.6–6.6% slower. Account the extra prologue tile so the first iteration consumes it, or it becomes the
  steady-state depth (`split_kv.md`).

In FlyDSL: `../flydsl/attention.md` § schedule_sync §10.

## Verify
- Every recipe arm has a changed opcode hash, and every region's MFMAs and TRANS ops are named.
- Fence relaxations keep per-region memory-op counts; hazard edits pass a shape grid with bitwise determinism.
- Wait changes are judged by `vmcnt(0)` count and depth distribution, then by `SQ_WAVE_CYCLES` /
  `SQ_WAIT_INST_LDS`, then by wall time.

## Related
`registers_occupancy.md` · `lds.md` · `attribution.md` ·
`correctness.md` · `grid_dispatch.md` · `split_kv.md` ·
`../gemm/schedule_and_locality.md` · `../gemm/correctness.md` ·
`../flydsl/guides.md` § prefetch-data-load · `../flydsl/debug-flydsl-kernel.md`

## Evidence

`FlyDSL <version>` is given only where the cited run, or another round in the same container and venv, printed it
(`flydsl.__version__`, an import path with its version, or a `pip install flydsl==` in the run's own setup); `FlyDSL ?`
otherwise. The month is that of the round's timestamps in the campaign's `run.jsonl`; a line resting on a note kept
outside a run says which date it uses. "Campaign" paths are under
`/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/`; "myskill" is
`/workspace/code/gpt_oss_docker/myskill/optimizer/`. Wall and throughput percentages are signed as the card signs them
(negative = slower) unless the line says faster or slower; time, cycle and counter percentages follow the quantity.

- E1 [2026-09, gfx950, FlyDSL ?, N=6, same-process paired A/B `_r7_ab.py` on b4, 80 reps, Thue-Morse order, against the same session's untouched tree; its control leg read +0.057%] copying GEMM2's 1:1 `sched_mfma/sched_dsrd` hint pair onto GEMM3's k-step loop: the `{LDS_TR:12}` read family's issue excess 934 → 566 as intended, but the region's unnamed exp chain's excess 566 → 2142, VGPR/AGPR/spill/LDS bit-identical, wall −7.695%, 0 of 6 pairs faster, intervals disjoint (gpt-oss D=64 attention backward, Hq=64 Hkv=8 S=8192 causal, B=4; campaign 20260910_112555 round 14; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260910_112555/memory.md:428)
- E2 [2026-08, gfx950, FlyDSL ?, N=6, six-round position-balanced three-arm rotation] a hand `sched_group_barrier` pipeline naming one LDS read per group (`"r1,m2,t2,v4"`): LDS issue-to-retire cover distance 32.4 → 19.7–20.3, wall +3.48% slower (0 of 6 rounds faster; the same rotation's spread 0.6%); a leading group larger than the region's read count (`"r24,m32"`) kept cover at 32.8 and still read +0.90% slower (1 of 6), recorded as the cost of replacing the deployed `iglp_opt(2)` mutation (gpt-oss-20b attention backward, B=4 Sq=Skv=8192 Hq=64 Hkv=8 D=64, fused body; campaign 20260819_065627 round 8; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260819_065627/memory.md:270)
- E3 [2026-09, gfx950, FlyDSL ?, N=5, five lab sessions, each with a same-binary A-vs-A null of 0.01–0.41%; official bench three runs before and three after] D=64 QK region with 8 `32x32x16` + 4 row-sum `16x16x32` MFMAs: `K_HEAD` 3 → 2 (9 VGPRs freed, spill 31 → 22) plus anchors 8 → 10 read +1.52/+1.28/+1.21/+1.30/+1.45% in the lab and +0.82% on the official bench; `K_HEAD` alone +0.70/+0.81%, anchors alone −0.30%; the scheduler packed 26 `v_pk_fma` + 12 `v_fma` into 32 `v_pk_fma_f32`; VALU-group sweep 5/6/7/8/9/11 pairs → 0/+1.29/+1.30/+0.24/0/+0.17%, 12 anchors undid the packing; D=128 kept 16 anchors, ±2 read −0.5% both ways against a 0.18% null (FlyDSL dualwave_swp attention forward, d64 B4 Hq64 Hkv8 S8192 causal, d128 B1 Hq32 Hkv8 S8192 causal; campaign 20260924_001030 round 13; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260924_001030/memory.md:358)
- E4 [2026-08, gfx950, FlyDSL ?, N=1, final-ISA opcode-sequence md5, no timing] the P·V `sched_group_barrier` prescription sat after `softmax_half`'s internal `sched_barrier(0)` in a region holding 2 MFMAs; moving it into the region with the 8 P·V MFMAs and 16 exps, deleting it, or both gave an opcode md5 identical to the deployed build (gpt-oss D=64 attention forward, dual-wave, B=4 Hq=64 Hkv=8 S=8192; campaign 20260831_084030 round 12; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260831_084030/memory.md:265)
- E5 [2026-09, gfx950, FlyDSL ?, N=1, ISA dump at the round's ISA gate, not benched] `G2_LAG` (GEMM2 delayed one q-half into the next half's GEMM1 + softmax region): VGPR 467 → 484, in-loop `lgkmcnt(0)` 53 → 119, `s_nop` 63 → 81, and the 32 exps formed one bare run with no MFMA interleaved, i.e. `iglp_opt(3)`'s exp interleave was lost (gpt-oss D=64 attention backward, Hq=64 Hkv=8 S=8192, batches 24 and 32; campaign 20260905_095125 round 21; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260905_095125/memory.md:353)
- E6 [2026-08, gfx950, FlyDSL ?, N=3, scored bench, 3 of 3 for the iglp arm; the combined arm's reading count is not stated] `iglp_opt(2)` with the two mutually exclusive fences deleted: spill 27, −6.5%; the same on top of the packs-only carry: spill 0, ISA fingerprint identical to packs-only alone, −0.57% = packs-only ±0 (gpt-oss D=64 attention forward, dual-wave, B=4; campaign 20260831_084030 round 13; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260831_084030/memory.md:277)
- E7 [2026-09, gfx950, FlyDSL ?, N=3, three paired repeats per mask; official bench three runs] the two `sched_barrier(0)` around each real `s_barrier` relaxed to `0x008` at all 25 call sites: d128 +0.6–0.84%, d64 ≈ 0; official bench 1.00661 → 1.00802/1.009992/1.010122, O/LSE bit-identical; `0x00A` −0.49% at d64; main-loop-only relaxation d128 +0.42% against +0.60% for all sites (FlyDSL dualwave_swp attention forward, d128 B1 Hq32 Hkv8 S8192 D128 causal, d64 B4 Hq64 Hkv8 S8192 D64 causal; campaign 20260923_144851 round 3; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260923_144851/rounds/round-03/agent.txt:8)
- E8 [2026-08, gfx950, FlyDSL ?, N=10, in-process screen n=10; scored ruler 2 palindromic pairs; sustained min-of-30000 at a pinned 1400 W] both fences around `s_barrier` dropped: 1.84608 against 1.85744 ms (+0.61%), scored +0.55%, sustained +0.96% wall (−1.85% real cycles); leading fence alone ISA-inert, trailing fence alone −0.13% (gpt-oss D=64 attention forward, dual-wave, B=4 Hq=64 Hkv=8 S=8192; campaign 20260831_084030 round 6; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260831_084030/memory.md:198)
- E9 [2026-09, gfx950, FlyDSL ?, N=?, lab A/B with a same-binary null control] deleting the `_sched_barrier(0)` at the head of the memory cluster: −1.5%, spill 31 → 35 and two scratch ops in the fast loop; a half-open mask `0x100` (DS reads only): flat (FlyDSL dualwave_swp attention forward, d64 B4 Hq64 Hkv8 S8192 causal; campaign 20260924_001030 round 13; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260924_001030/memory.md:365)
- E10 [2026-08, gfx950, FlyDSL ?, N=2, two readings against the batch's controls, palindromic] `SBSM=0`, dropping only the `_sched_barrier(0)` inside `softmax_half` (before its row-sum MFMAs): spill 4, 246 instructions, 1.8961/1.8718 ms, −2.5% (gpt-oss D=64 attention forward, dual-wave, B=4; campaign 20260831_084030 round 11; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260831_084030/memory.md:253)
- E11 [2026-09, gfx950, FlyDSL ?, N=5, 5–6 palindromic passes per arm] partial `sched_barrier` masks, the fences they were applied to not named in the record: `0x008` −0.15%, `0x002` −0.20%, an added fence after the row-sum MFMAs −0.12% (round 17); `0x8`/`0xA` on the 6 folded cluster syncs −0.21%, `0x40A` −0.50%, the same masks on the 4 cluster-head fences −0.29% (round 18) (gpt-oss D=64 attention forward, dual-wave, B=4; campaign 20260831_084030 rounds 17–18; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260831_084030/memory.md:331)
- E12 [2026-09, gfx950, FlyDSL ?, N=2, two official-bench readings, 0.04% apart] three softmax → MFMA register fences inside the compute clusters relaxed to `0x008`: score 0.946331 and 0.945965, +3.1% over the round-2 tree; per cell d128 0.988372/0.98555, d64 0.906078/0.907971; d128 fast-arm `naked_work` 58 → 54 → 0; d64 `.vgpr_count` 168 (spill 24 → 22), d128 256 (spill 0 → 1); the round's own reading: the occupancy-2 d128 cell 0.930 → 0.9855 took all of the gain, the occupancy-3 d64 cell moved only within noise (FlyDSL dualwave_swp attention forward, d128 B1 Hq32 Hkv8 S8192 at 2 waves/SIMD, d64 B4 Hq64 Hkv8 S8192 at 3; campaign 20260924_001030 round 3; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260924_001030/memory.md:150; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260924_001030/memory.md:154)
- E13 [2026-07, gfx950, FlyDSL 0.2.2, N=3, same-session interleaved A/B ×3] `s_setprio` level on the P·V cluster: 2 → 1127.3 TF/s, level with 1 (round 6); level 2 1131.4, level 3 1133.9, level 1 against 3 interleaved: 1131.5/1131.8/1131.5 against 1131.9/1132.5/1131.9, +0.05% (round 10) (hd64 attention forward, Sq=Skv=16384, B=1, Hq=128 Hkv=16, full causal; campaign 20260728_082734 rounds 6 and 10; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260728_082734/memory.md:207)
- E14 [2026-08, gfx950, FlyDSL ?, N=9, paired win-count] `G2NP`, removing the `s_setprio` pair around GEMM2: 9 of 9 pairs slower, −2.8% (fused single-pass attention backward, KV-outer with dQ folded in, B=3 S=8192 Hq64/Hkv8 D64 full causal; campaign 20260804_003925 round 6; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260804_003925/memory.md:78)
- E15 [2026-08, gfx950, FlyDSL ?, N=?, screen against a same-session control, count not stated] `s_setprio` 1/2/3 over the QK run: −0.70% / −0.45% / −1.9%; releasing the P·V window early: 1.9070 against 1.8425 ms, −3.5%; releasing it late: 0.0%; priority 2 over the softmax clusters: −0.9% (gpt-oss D=64 attention forward, dual-wave, B=4; campaign 20260831_084030 round 8; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260831_084030/memory.md:222)
- E16 [2026-09, gfx950, FlyDSL ?, N=6, b4 palindromic A/B as the round's other arms, 80 reps; intervals disjoint, 6 of 6 per myskill] deleting GEMM2's 4 `s_setprio` (65 instructions): trip +6.7%, in-loop `lgkmcnt(0)` 44 → 88, wall +4.678% slower, SNR 52.3 / NaN / NaN (gpt-oss D=64 attention backward, occupancy 1, 4-wave workgroups, B=4; campaign 20260910_112555 round 11; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260910_112555/memory.md:342; myskill 00-decision-index.md:330 gives 6/6)
- E17 [2026-07, gfx950, FlyDSL ?, N=?, scored bench `_bench_campaign_bwd_gptoss.py`] `s_setprio(1)` bracket around the dK/dV kernel's GEMM1: 799.7 TF/s, −3.3%, the record's reason "starves the co-resident WG" (hd64 attention backward, gpt-oss configuration Hq=64 Hkv=8 D=64, Sq=Skv=8192, B=2, SBHD, full causal; campaign 20260728_123346 round 8; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260728_123346/memory.md:147)
- E18 [2026-07, gfx950, FlyDSL 0.2.2, N=?, scored bench] `s_setprio` turned off: 1118.7 TF/s, −0.8% (hd64 attention forward, Sq=Skv=16384, B=1, Hq=128 Hkv=16, full causal; campaign 20260728_082734 round 6; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260728_082734/memory.md:165)
- E19 [2026-07, gfx950, FlyDSL ?, N=?, same-process interleaved A/B `_ab_fwd_cfg.py`] forward configuration sweep: `setprio=False` −2.4%; no run record is kept, the month is that of the note (modified 2026-07-30) (gpt-oss SBHD attention forward, Hq64/Hkv8 D64 B2 full causal, S=8192 and 16384; local:/workspace/code/gpt_oss_docker/.claude/memory/project_fwd16384_campaign.md:199)
- E20 [2026-07, gfx950, FlyDSL 0.2.2, N=2, palindromic balanced A/B, two readings for the kept arm and one for the others, every arm through the ISA and SNR gates] `amdgpu-sched-strategy` arms against base 1168.5/1169.3/1171.3 TF/s: `max-memory-clause` + post-RA scheduling 1178.2/1178.6 (recorded as +0.8% beside the readings, +0.6–0.7% in the round's summary), `max-ilp` alone 1175.9, `iterative-ilp` 1178.7, fastest, but SNR NaN and not deterministic (hd64 attention forward, Sq=Skv=16384, B=1, Hq=128 Hkv=16, full causal; campaign 20260728_082734 round 15; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260728_082734/memory.md:261)
- E21 [2026-07, gfx950, FlyDSL ?, N=3, scored bench, three runs per arm] dK/dV with `max-memory-clause` 837.5 against 842.5 TF/s, −0.59% (3 of 3 worse); dQ −0.13%; `max-ilp` 835.3, −0.85% (hd64 attention backward, gpt-oss configuration, Sq=Skv=8192, B=2; campaign 20260728_123346 round 14; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260728_123346/rounds/round-14/agent.txt:1)
- E22 [2026-08, gfx950, FlyDSL ?, N=2, two readings for `max-ilp`, the others' count not stated; controls 1.8377/1.8387/1.8489 ms, spread 0.61%] strategy re-sweep on the hand-scheduled tree (deployed `max-memory-clause`): `max-ilp` −0.10%, `max-occupancy` −1.1%, `minregexcess` −0.8% (both spill 4 → 5); `max-occupancy` is not a strategy value LLVM recognises (E42), and the record does not show which strategy that arm's build ran (gpt-oss D=64 attention forward, dual-wave, B=4; campaign 20260831_084030 round 11; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260831_084030/memory.md:253)
- E23 [2026-07, gfx950, FlyDSL 0.2.2, N=1, SNR and run-to-run determinism gate per arm] row-sum MFMAs consuming the inline-asm `v_cvt_pk` result directly: SNR 25.01 dB, then 24.99 dB with an extra `lgkmcnt(0)`, not deterministic; bad rows exactly `(q//4)%2==0`; `sched_barrier(0)` + `s_nop 7` before those MFMAs: SNR 51.61, deterministic, and `s_nop 1` was kept (hd64 attention forward, Sq=Skv=16384, B=1; campaign 20260728_082734 round 12; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260728_082734/memory.md:225)
- E24 [2026-07, gfx950, FlyDSL ?, N=?, same-process interleaved `_probe_ratio_goal.py`] every variant that removed the clamp (prime, d2, s_nop, latekeep, margin) failed the SNR gate, −79 to 26.9 dB; no run record is kept, the month is that of the note (modified 2026-07-26) (hd64 attention backward, dK/dV fold path, Sq=16384; local:/workspace/code/gpt_oss_docker/.claude/memory/project_dkdv_clamp_hazard_truth.md:25)
- E25 [2026-07, gfx950, FlyDSL ?, N=2, kernel-scoped interleaved A/B, median of 9 per arm, two rounds, arms non-overlapping] dQ: one clamp per accumulator v4 5.2116 → 5.0884 ms (time −2.36%), one per kv-half 5.0273 ms (time −3.54%, `v_min` 72 → 3, kept) (round 15); dK/dV: one per v4 7.3061 → 7.2256 ms (time −1.10%, `v_min` 256 → 64) (round 16); round 16 records that dQ's per-kv-half anchor has the same structural hole as a per-group anchor and is "empirically validated, not proven" (hd64 attention backward, Sq=Skv=16384, B=1, Hq=128 Hkv=16; campaign 20260726_152547 rounds 15–16; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260726_152547/memory.md:247; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260726_152547/memory.md:261)
- E26 [2026-07, gfx950, FlyDSL ?, N=2, kernel-scoped interleaved A/B, median of 9, two rounds; SNR/determinism grid over Sq and BLOCK_KV] dK/dV anchors grouped per whole GEMM1a block: −2.53% kernel time but dK/dV 16.89/17.85 dB, not deterministic at Sq 4096 and 8192 (q_split 4, BLOCK_KV 128); per mt-pair −1.79% kernel time (round 16); a min3 anchor per group of three v4: −0.31% kernel time, 19.7 dB and not deterministic at BLOCK_KV 64 (round 17); the scored shape passed (hd64 attention backward, Sq=Skv=16384, B=1; campaign 20260726_152547 rounds 16–17; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260726_152547/memory.md:259)
- E27 [2026-09, gfx950, FlyDSL ?, N=?, ?] handing GEMM2's last MFMAs across the head-step's WAR `s_barrier` needed an `s_waitcnt lgkmcnt` before that barrier (the round's diff); myskill records the wait as measured at zero cost, without a reading (llama-8b D=128 a16 paired attention backward; campaign 20260919_155238 round 7; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260919_155238/rounds/round-07/diff.txt:134; myskill pitfalls/13-meta-hd64-flash-bwd.md:2721)
- E28 [2026-08, gfx950, FlyDSL ?, N=6, scored ruler, palindromic pairs in two opposite-order batches] the hand-written `lgkmcnt(0)` before the hot loop's barriers removed: +0.54%, 5 of 6 pairs positive (round 5); the same in the prologue and 7 epilogue clusters: `d64_gptoss_b4` +0.32% (4 of 4 pairs), composite +0.19% (round 7) (gpt-oss D=64 attention forward, dual-wave, B=4; campaign 20260831_084030 rounds 5 and 7; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260831_084030/memory.md:186)
- E29 [2026-07, gfx950, FlyDSL 0.2.2, N=3, same-session interleaved A/B, 3 candidate readings against 4 base readings, intervals disjoint] memory-cluster drain sunk to the consuming compute cluster, gated off under stagger: 1132.0/1132.8/1131.2 against 1116.3/1115.8/1117.0/1116.9 TF/s, recorded as +1.39% (myskill: +1.42%, 1116.5 → 1132.4); ungated, the 8-wave stagger configuration failed 1 of 22 checks, full-causal rel_l2 0.0294 > 0.020, SWA 0.0027 → 0.0112, while the scored SNR stayed 51.26; gated, rel_l2 back to 0.0027/0.0028 (hd64 attention forward, Sq=Skv=16384, B=1; campaign 20260728_082734 round 9; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260728_082734/memory.md:195)
- E30 [2026-08, gfx950, FlyDSL ?, N=3, scored bench, three runs; the two probe arms single readings] next head's Q/dO prefetch moved ahead of GEMM3's 4 dQ partial stores: ratio 0.87539 → 0.90939/0.90877/0.91268, D128 804.8 → 836.4/832.7/833.9 TF/s, whole-kernel `vmcnt(0)` 16 → 1; probe 831.2 against 804.0 (+3.4%), the same prefetch a few instructions after the stores 785.0 (−2.4%); `dma_grp=2`: `s_waitcnt vmcnt` 89 → 18, 0.8291 (−5.3%); wait depths (7..0) → (11..4) are from myskill (gpt-oss D=128 fused attention backward, B=2 Hq64/Hkv8 Sq=Skv=8192 full causal, BLOCK_KV 128, 4 waves, occupancy 1; campaign 20260810_114422 round 9; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260810_114422/memory.md:159; myskill pitfalls/13-meta-hd64-flash-bwd.md:326)
- E31 [2026-09, gfx950, FlyDSL 0.2.4, N=2, same-process frozen-reference ABBA `_bench_attn_bwd2.py`, guard-corrected, two sessions] `g2_hs=1` (GEMM2's last 8 MFMAs handed to the exposed transposed-read burst): score 1.0137/1.0151 raw, 1.0171/1.0134 corrected, against 1.0059 raw / 1.0078 corrected; the same plus burst-issued reads (`g2_rd_at=1`): 0.9996 raw / 1.0031 corrected, 1.4% below the hand-off alone, one session (gpt-oss D=64 attention backward, a16 dQ path, occupancy 1, 4 waves, Hq=64 Hkv=8 S=8192 causal, B=4/2/1; campaign 20260911_114527 round 5; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260911_114527/memory.md:304)
- E32 [2026-09, gfx950, FlyDSL ?, N=?, rocprofv3 PMC `SQ_WAVE_CYCLES` per arm] `g2_hs` dose on the D=128 body (`g2d=2`): 0/1/2/3/4 tiles → 0 / −0.96% / −1.20 to −1.35% / +14.4% / +74.3% wave cycles; the same 8 MFMAs landed after the whole GEMM3: +23.3% wave cycles though `SQ_WAIT_INST_LDS` was lower; on the D=64 body (`g3_defer=False`): +1.98% wave cycles, LDS wait up 4.0% (llama-8b D=128 a16 paired attention backward, S=4096 and 8192, and the D=64 body of the same file; campaign 20260919_155238 round 7; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260919_155238/goal.md:535)
- E33 [2026-07, gfx950, FlyDSL ?, N=1, scored bench, one reading per arm; ISA dump] dK/dV Q/dO DMA marked `nt`: 731.5 against 791.3 TF/s, −7.6%; dK/dV Q/dO LDS double buffer with next-head DMA prefetch: 663.0, −16.5%, from 424 B of scratch for duplicated LDS address sets; with the DMA in flight, `vmcnt(0)` drains 16 → 85, and 16 → 109 with the hand-over moved past the last `ds_read` (hd64 attention backward, Sq=Skv=16384, B=1, Hq=128 Hkv=16; campaign 20260726_152547 round 7; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260726_152547/memory.md:181)
- E34 [2026-07, gfx950, FlyDSL ?, N=?, ?] dQ K/V double buffering at BLOCK_KV 96/64/32: −2.9%/−3.2%/−2.3%, SNR 52.29; at BLOCK_KV 32 its LDS was 2/3 of the deployed single buffer and it still lost (hd64 attention backward, dQ kernel, Sq=Skv=16384, B=1; the campaign memory's list of closed levers, written at campaign start; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260726_152547/memory.md:59)
- E35 [2026-07, gfx950, FlyDSL ?, N=?, PMC counters] register-neutral hand-over, next DMA issued where half the `k_tr` registers die: 793.5 against 787.6 TF/s, kept; `SQ_WAIT_ANY` −41%, coexec +13% (hd64 attention backward, dQ kernel, Sq=Skv=16384, B=1; campaign 20260726_152547 round 6; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260726_152547/memory.md:173)
- E36 [2026-07, gfx950, FlyDSL ?, N=?, same-host interleaved, min-based] `db` (cross-pair double buffer that removes the post-P·V WAR barrier, 2 → 1 barrier per iteration, occupancy 1) alone +2.4%; with the QK prefetch depth retuned 6 → 4 (`pf4`): pro cr4 488 → 506.3 TF/s (+3.8%), flash cr4 340 → 350.7 (+3.1%); no run record is kept, the month is the note's (dated 2026-07-14 in its heading) (DeepSeek-V4 sparse-MLA attention backward, dQ kernel `pvk32`; local:/workspace/code/gpt_oss_docker/.claude/memory/project_dsv4_bwd_dq_occ_truth.md:84)
- E37 [2026-08, gfx950, FlyDSL ?, N=?, scored bench] `QDO_PP` (second Q/dO LDS slot, head-step WAR barrier dropped): `s_barrier` 35 → 19, VGPR 455 → 481, 0.8201, −6.0%; the cause was later re-measured on the D=64 body with registers held flat and attributed to the next head's `ds_write` burst landing inside GEMM3's transposed-read burst (myskill) (gpt-oss D=128 fused attention backward, B=2 Sq=Skv=8192; campaign 20260810_114422 round 9; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260810_114422/memory.md:159; myskill pitfalls/13-meta-hd64-flash-bwd.md:2361)
- E38 [2026-08, gfx950, FlyDSL ?, N=?, scored bench per arm; the decomposition from in-process readings] 4-deep K/V ring with one barrier at the iteration boundary: fwd64 0.98704 (−2.2%); the 4-deep ring with both barriers kept 1.9030 against 1.8721 ms (deepening alone −1.65%), so the barrier gave back 1.30%; the build also paid LDS ×2 and spill 4 → 9 (gpt-oss D=64 attention forward, dual-wave, B=4; campaign 20260831_084030 round 5; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260831_084030/memory.md:186)
- E39 [2026-08, gfx950, FlyDSL ?, N=?, ruler not stated; `nt` verified in the ISA] Q loaded non-temporal (`BufferCopy128b(2)`): −0.95%; O stored non-temporal: +0.11%; both: −0.9% (gpt-oss D=64 attention forward, dual-wave, B=4; campaign 20260831_084030 round 7; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260831_084030/memory.md:210)
- E40 [2026-09, gfx950, FlyDSL 0.3.2, N=?, same-block ABBA] prologue gather prefetch depth 2 (`_XPF_PRIME=2`, steady state 1): producer time −1.68 to −2.70% (faster) at ≤ 2 CTAs per CU, +2.0 to 2.5% (slower) at ≥ 576 CTAs (GLM-5.2 sparse-MLA decode producer `flydsl_sparse_mla_partial`, TP4/EP4 decode shapes; campaign 20260912_134949 round 7; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260912_134949/rounds/round-07/agent.txt:63; version printed in the same campaign and venv, rounds/round-06/deep_raw.jsonl:5889)
- E41 [2026-09, gfx950, FlyDSL 0.3.2, N=?, ?] steady-state gather prefetch depth `_XPF_DEPTH` 1 → 2: seq48/60/84 +4.56/+6.01/+6.55% slower, output bit-identical, depth 3 worse (GLM-5.2 sparse-MLA decode producer; campaign 20260912_134949 round 1; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260912_134949/memory.md:129; version printed in the same campaign and venv, rounds/round-06/deep_raw.jsonl:5889)
- E42 [2026-08, gfx950, FlyDSL ?, N=5 for `max-ilp` (a 5v5 ABBA), ? for `max-memory-clause`, ISA checked per arm] LLVM scheduler strategies re-priced on another hand-scheduled backward body: `max-memory-clause` −0.51%, `max-ilp` +0.15% on 2 points then −0.35% on the 5v5 ABBA; `amdgpu-sched-strategy=max-occupancy` compiled byte-identical to the base: not a strategy name LLVM knows, silently ignored (gpt-oss D=64 attention backward, per-tensor fp8 E4M3, B=4 Hq=64 Hkv=8 S=8192 full causal; campaign 20260830_060642 round 7; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260830_060642/memory.md:233; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260830_060642/memory.md:235; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260830_060642/goal.md:18)
- E43 [2026-10, gfx950, FlyDSL 0.3.2, N=?, AKLR correctness pass (bounded determinism, max_violation_fraction 0) on every kept build] two bf16 LDS tiles refilled from registers (fp8 gathered, converted, `ds_write`) and read the next step, one `s_barrier` per step: max_diff 0 on all 11 workloads; once the P·V waves ran a step behind the QK waves the kernel needed three rotating tiles, still one barrier per step (dense fp8 MLA decode, 96 heads, query length 1, page size 1, batch 8–112; local:/shared_nfs/kyle/aklr_mla/runs/L3/workspace/forge_experiments/lessons/iter_005.md:24; the three-tile ring: commit d08bd402e6cc in local:/shared_nfs/kyle/aklr_mla/runs/L3/workspace, kernel.py:83, :427, :552)
- E44 [2026-10, gfx950, FlyDSL 0.3.2, N=? for the ablation's paired timing on AKLR's graph timer (noise floor 0.12%); N=5–7 paired in-process Kineto rounds of 10 calls for the arms; ISA re-read for this card by compiling both commits on FlyDSL 0.3.2] `sched_barrier(0)` after each of 18 logits `v_mfma_f32_16x16x32_bf16` with a 4-deep `ds_read_b128` prefetch written in source: waits `lgkmcnt(1)` → `lgkmcnt(4)`, `.vgpr_count` 164 both, no spill; with `s_setprio(1)`/`(0)` around the run +2.56% (bounds +2.48 … +2.64%, 11 of 11 batch sizes); fence alone 0.988 / 0.986 / 0.982 at batch 112 / 32 / 8 against 0.973 / 0.957 / 0.974 with priority; depth 5 fenced 168 VGPRs and slower (one 12-wave workgroup per CU, 3 waves per SIMD; dense MLA decode, 96 heads, fp8 KV; MI355X; local:/shared_nfs/kyle/aklr_mla/runs/L4/workspace/forge_experiments/ablation.md:19; local:/shared_nfs/kyle/aklr_mla/runs/L4/workspace/forge_experiments/lessons/iter_007.md:8; :38; :33; commits 30b410e and 055f736 in local:/shared_nfs/kyle/aklr_mla/runs/L4/workspace)
- E45 [2026-10, gfx950, FlyDSL 0.3.2, N=1, AKLR graph timer via `driver.py --bench-mode`, warmup 3, 20 iterations, one run per arm; the later round's ratios from its own graph-replay probe, control ±0.7% (±2% at b8)] `s_setprio(2)` on the 6 MFMA waves during the softmax: b8 0.1427 / b112 1.5895 ms against 0.1331 / 1.5374 (7.2% / 3.4% slower), during the QK and P·V phases 0.1427 / 1.6061 (7.2% / 4.5% slower), slower on all 11 batch sizes; on the 2 producer waves, priority 2: b8 1.2% faster, b112 0.3% slower, priority 3: 1.1% faster on both; in a later round, time ratios (candidate / base) of priority around the softmax 1.066 / 1.048 / 1.052 (b8 / b48 / b112) and of producer priority 1.014 / 0.989 / 1.027 / 1.017 (b8 / b16 / b48 / b112) (dense MLA decode over paged fp8 KV, one 8-wave workgroup per CU: 2 producer waves gather, convert fp8 → bf16 and store K to LDS for 6 MFMA waves, 256 VGPRs; local:/shared_nfs/kyle/aklr_mla/runs/L1/workspace/forge_experiments/lessons/iter_006.md:29; local:/shared_nfs/kyle/aklr_mla/runs/L1/workspace/forge_experiments/lessons/iter_012.md:25; local:/shared_nfs/kyle/aklr_mla/runs/L1/workspace/forge_experiments/lessons/iter_012.md:38)
- E46 [2026-10, gfx950, FlyDSL 0.3.2, N=2 for the candidate and 1 for the incumbent on the driver bench (11-case total, cache cleared per run); the gain from the campaign's 3-pair ablation] softmax exponent rewritten as `exp2(fma(s, scale2, -m))` per score plus `slp-threshold=100000`: main-path packed ops 4 `v_pk_fma_f32` (the rewrite under default SLP; 12 packed mul/add before it) → 0, 256 VGPRs and 29 spills unchanged; total 7.768 → 7.288 / 7.300 ms; ablation +6.85% when kept, +6.88% measured by the ablation (one-sided 95% bounds +6.82…+6.94%), 11/11 cases faster; the rewrite under default SLP was not timed, so SLP's own share is unpriced (96-head MLA decode, fp8 KV, batch 8–112; local:/shared_nfs/kyle/aklr_mla/runs/L2/workspace/forge_experiments/lessons/iter_006.md:11; local:/shared_nfs/kyle/aklr_mla/runs/L2/workspace/forge_experiments/ablation.md:18)
