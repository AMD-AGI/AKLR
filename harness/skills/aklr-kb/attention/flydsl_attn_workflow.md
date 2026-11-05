---
title: "FlyDSL attention — working order: pin the bound, audit the grid, price the MFMA, then hide latency"
kind: method
lever: flydsl_attn_workflow
ops: [attention]
gens: [gfx950]
dtypes: [bf16]
regimes: [prefill, backward, causal, sliding-window, gqa]
flydsl: 0.3.2
---

# Working on a FlyDSL attention kernel

> The order in which to look at a flash-attention forward or backward kernel (fwd, dq, dkdv, fused dq+dkdv,
> and their helper kernels). It does not say which configuration wins. Numbers marked `*(field, E<n>)*` are
> measured values; their conditions are listed under Evidence at the end. Use them for direction and order of
> magnitude, not as targets. The round discipline — one change per
> round, hypothesis before build, escalation after rollbacks — is in `../gemm/workflow.md` and
> applies unchanged.

## Route here when
You are starting on an attention kernel, or a run of rounds has stopped paying and you need to know which
layer you have not audited.

| The open question | Card |
|---|---|
| Where the time goes; which counter or probe answers which question | `flydsl_attn_attribution.md` |
| Grid shape, XCD placement, dispatch order, padding, GQA sharer merging | `flydsl_attn_grid_dispatch.md` |
| MFMA atom per GEMM, operand and accumulator layouts, P/dS as the next B operand | `flydsl_attn_mfma_layout.md` |
| exp2, LSE format, online-softmax rescale, row sums | `flydsl_attn_softmax.md` |
| Transposed LDS reads, row stride, bank-conflict readings | `flydsl_attn_lds.md` (with `../flydsl/guides.md` § lds-optimization) |
| VGPR budget, occupancy steps, spills, AGPR | `flydsl_attn_registers_occupancy.md` |
| `sched_group_barrier`, fences, `s_setprio`, waits, barriers | `flydsl_attn_schedule_sync.md` |
| dQ/dK/dV accumulation, determinism, workspaces, fusion, masked q-loops | `flydsl_attn_backward.md` |
| Decode, split-KV, the combine | `flydsl_attn_decode_split_kv.md` |
| Rulers, noise floors, FLOP accounting, host-bound shapes | `flydsl_attn_measurement.md` |
| Gates, spike inputs, mask conventions | `flydsl_attn_correctness.md` |
| The kernel is wrong, not slow | `../flydsl/debug-flydsl-kernel.md` |

## Step 0 — measure the kernel that ships
- **Audit the dispatch per cell** before trusting any number: record which backend and which code path each
  scored shape really runs (`flydsl_attn_measurement.md` §8). A window wide enough to cover the whole sequence
  may be normalised to full causal; a framework default may not be FlyDSL at all.
- **Have a gate that reaches the changed path** (`flydsl_attn_correctness.md`) before the first timing.
- **Dump the facts per path**, not per code base: the MFMA atom each GEMM issues (fwd and bwd in one code base
  can use different atoms), `.vgpr_count`, `.agpr_count`, spill, waves per SIMD, LDS bytes per workgroup, and
  a per-trip opcode histogram. Every later decision reads from this table.

## Step 1 — pin the bound, per kernel and per head_dim
- **Do not assume throughput-bound.** Backward bodies ran the MFMA pipe at 44–50% duty at the deployed clock (52–57%
  when computed at the clock read under profiling) *(field, E1, E2)*; the rest was waits, rendezvous and serial VALU
  chains between GEMMs. But the bound is per cell: re-derive the
  per-pipe census for each head_dim with `v_exp_f32` at 8 cycles (the MFMA-to-exp ratio grows with D: exps fill two
  thirds of the shadow at D=64 and one third at D=128, `flydsl_attn_mfma_layout.md` §1); a reorder that paid on one D
  was noise on the other *(field, E3)*.
- **Do not accept "issue-bound" from a paper count.** Paper issue budgets fail easily: one was off by 4×
  *(field, E4)* because it compared per-SIMD issue slots with per-wave pipe cycles, when all waves of a SIMD share one
  MFMA pipe. Confirm with the PMC split and a deletion probe.
- Order of evidence: true clock → per-kernel additive compute probe → LDS port ratio → stall-pool split →
  subtractive probes. All of them are in `flydsl_attn_attribution.md`.

| What the evidence shows | What it usually means | Next |
|---|---|---|
| MFMA duty well under 80% at the true clock, large share where neither pipe is busy | latency, rendezvous, serial exp/pack chains | price the stall pool (Step 4) |
| MFMA busy + VALU busy near the timeline, low co-execution, occupancy 1 | the two pipes run nearly serially | overlap inside the workgroup (`flydsl_attn_schedule_sync.md`) |
| One pipe saturated in a per-class census (exp at 8 cycles, `flydsl_attn_softmax.md` §1) | throughput on that pipe | delete its work; reordering pays nothing |
| `SQ_VALU_MFMA_BUSY_CYCLES` under ~4× `SQ_LDS_IDX_ACTIVE` | LDS read bound (in fwd, usually the PV transposed reads) | `flydsl_attn_lds.md` |
| sclk far under peak with power at the cap | energy-bound | price in wall and clock (`flydsl_attn_attribution.md` §8) |

## Step 2 — audit the grid and dispatch mapping before the kernel body
Four index/order changes outside the kernel, the two XCD-major decodes among them bit-identical, were worth +7% on one
backward together with an LDS hand-over and an LSE fold kept in the same rounds *(field, E5)*; many rounds of tile,
buffering and traversal changes inside the body had been worth ~1% *(field, E6)*. Ask, in order:
1. **Which data do co-resident workgroups share, and does it stay in one XCD's L2?** Before writing a remap,
   compute what `flat_id % 8` already equals for the current grid: many grids make the XCD remap a
   constructive no-op, which three lines of index algebra settle without a benchmark.
2. **Is the dispatch order the list-schedule order?** Take the per-workgroup trip-count profile first; its
   shape (monotone, tent, flat) picks the order.
3. **Where does the padding land** — on the most expensive tile or the cheapest?

- Judge this layer with `SQ_WAIT_ANY` and co-execution, not with L2 hit rate: a hit rate rising from 86.5% to
  94.8% (misses −62%) moved the wall only within noise, and +0.54% once kept together with an LDS hand-over
  *(field, E7)* (`flydsl_attn_grid_dispatch.md` §2).
- Sweep the batch axis before building a persistent or work-stealing scheduler for the tail; on one forward the
  grid quantisation and tail were ≤0.2% *(field, E8)*.
- Mechanics and rules for each question: `flydsl_attn_grid_dispatch.md` §1–§8.

## Step 3 — compute the redundant-MFMA ratio before "reduce MFMA"
Removing MFMAs pays only while the body still issues MFMAs the math does not need. Do the arithmetic first:
```
issued MFMA FLOPs   = Σ over kernels of (MFMA instructions per trip × trips × FLOP per instruction)
causal-exact FLOPs  = the scored convention (5 GEMMs for bwd, attended fraction only)
redundancy          = issued / causal-exact − 1
```
- A causally trimmed D=64 backward issued only 3.1% more than causal-exact, the half-triangles of the
  diagonal band at tile granularity. That caps every "fewer MFMA" idea on that body at 3.1%.
- Above ~15%, this is the first lever; under ~5%, go to Step 4.
- With redundancy to remove, audit droppable GEMMs and global correction terms first: dropping one GEMM from a
  dq kernel gained 9.8% *(field, E9)*. Fusion and drop rules are in `flydsl_attn_backward.md` §7.
- Redundancy that buys a constraint is not free. A deterministic two-kernel backward recomputes S and dP in
  both kernels (7 GEMM-equivalents against the 5 the FLOP convention counts, ~29% of MACs); removing
  it needs float atomics or a per-trip dQ partial store (`flydsl_attn_backward.md` §1–§4).

## Step 4 — price the stall pool, then hide latency
- Every route to a speed-up goes through the stall pool; only some routes also need more co-execution. Price
  the pool by source (barrier, wait counters, LDS return, prologue/epilogue) before touching the schedule
  (`flydsl_attn_attribution.md` §3 and §6). Eight rounds of scheduling work on one body returned 0.4–4.0% in
  total while its stall pool had not been priced *(field, E10)*.
- **When co-resident waves already fill the MFMA shadow** (no bare VALU between MFMAs in the ISA, high coexec / MFMA
  busy), reordering inside one wave is worth about zero; only deleting instructions pays (forced exp
  interleave in a backward with two co-resident workgroups per CU: 0.00%) *(field, E11)*. Where bare VALU remains between MFMAs,
  fence relaxation still paid at 2 waves/SIMD and was noise at 3 (`flydsl_attn_schedule_sync.md` §4): count
  `naked_work` first. With one workgroup per CU,
  interleaving inside the workgroup is the only cover.
- Then, by what the census names: `flydsl_attn_schedule_sync.md` (anchors, fences, priorities, barriers),
  `flydsl_attn_mfma_layout.md` (atom and chain structure), `flydsl_attn_lds.md` (transposed reads),
  `flydsl_attn_registers_occupancy.md` (what a freed register buys), `flydsl_attn_softmax.md` (exp and rescale
  cost).

## Step 5 — numerics, occupancy, determinism
Check these in order once the structure is settled: exp mode and LSE format (`flydsl_attn_softmax.md`),
whether an occupancy step is reachable without spills (`flydsl_attn_registers_occupancy.md`), and whether the
target requires bit-reproducible outputs (`flydsl_attn_backward.md` §1). A determinism requirement closes
every "relax determinism to drop MFMAs" option; decide it before Step 3, not after.

## Usually dead
Each entry held on the bodies where it was measured; re-open one only with a probe that shows its precondition
changed.
- **LDS swizzle, padding or bank-conflict work** when MFMA busy cycles are ≥4× `SQ_LDS_IDX_ACTIVE`
  (4× on a dq body, 8× on a dkdv body *(field, E12)*): the port is not the limit (`flydsl_attn_lds.md` §5).
- **Raising occupancy by force** (`waves_per_eu`, AGPR allocation) when the code does not already nearly fit: the
  compiler spills (`flydsl_attn_registers_occupancy.md` §2). *(field, E13)* One more wave
  per SIMD made a backward 32% slower and two more 5× slower; *(field, E14)* forcing AGPR splits took a forward's
  spills from 31 to 52–77.
- **Dual-wave, 8-wave, warp specialisation or stagger in a backward that already runs two workgroups per CU.** Those
  co-resident workgroups already overlap with no barrier; the rebuilt version adds a barrier tax. Its best stagger,
  which raced (NaN, not deterministic), reached 1029 TF/s against a 1116 TF/s baseline *(field, E15)*.
- **Pushing exp into MFMA shadows with two co-resident workgroups per CU**: 0.00% and noise-level results; raising
  MFMA priority to force it, −3.3% *(field, E11, E16)*.
- **Pure reordering on a power-capped kernel whose shadow is already full**: identical instructions in a new order pay
  0 ± 2% (co-execution 44% of SIMD cycles, 52% of MFMA busy) *(field, E17)*; at low co-execution reorders still gain
  (`flydsl_attn_attribution.md` §8).
- **K/V LDS double buffering to hide DMA with two co-resident workgroups per CU**: the occupancy already hides most
  DMA latency and the prefetch's LDS writes compete with the compute reads; *(field, E18)* a deletion probe showed 8.7%
  headroom and the built prefetch lost 4.3–4.7% (`flydsl_attn_attribution.md` §5). At one workgroup per CU a second
  slot that removes a write-after-read barrier can pay, but not as a rule (`flydsl_attn_schedule_sync.md` §10).
- **A fixed reference max for the softmax shift** as a general lever: exact only inside a data-dependent window,
  silently wrong outside it (`flydsl_attn_softmax.md`).
- **Lazy (threshold) O-rescale ported into a P-store pipeline**: the branch broke its overlap; −31% on one path,
  neutral on another *(field, E19)*; on dense decodes with no such pipeline it paid (`flydsl_attn_softmax.md`
  §6.3).
- **Q-outer fused dq+dkdv**: reducing dK/dV over q takes float atomics (not deterministic) or a dK/dV partial
  workspace that saves no bytes (`flydsl_attn_backward.md` §2). A KV-outer fusion with a narrow KV band
  (`BLOCK_KV = 64`) was 1.95–2.55× slower *(field, E20)*; the working fused structure is a KV-outer single pass with a
  wide band (`flydsl_attn_backward.md` §7).
- **fp32 atomic dQ**: each dQ element takes ~Skv/(2·BLOCK_KV) contributions; 105.5 ms against 6.34 ms
  *(field, E21)* (`flydsl_attn_backward.md` §3).
- **Bigger tiles past the register wall**: a dkdv `BLOCK_Q` of 128 lost 67% *(field, E22)*.
- **Tuning hand-written `vmcnt` drain levels**: removing every one of them priced at +0.3% on one forward
  *(field, E23)*; the barrier family, not the wait family, held the time.

## Rules that save rounds
- **Before declaring a lever dead in one regime, align the other levers.** A regime that still runs a different
  softmax, mask or tile path than the one where the lever was measured can flip its sign.
- **A gate condition whose reason nobody can state is usually scope residue.** A `window_left < 0` gate kept
  sliding windows on a path nobody had tuned for them; window-gated configuration (compile-time guarded:
  q_split, tile sizes, register prefetch, dead-tile skip) took one window shape from 216 to 298 effective TF/s
  *(field, E24)*.
- **"The levers are exhausted" usually means one layer is.** List the layers (grid and dispatch, body, waits and
  barriers, helper kernels, host path) and name the last probe that priced each.
- **Re-derive per head_dim.** Anchor counts, tile sizes and which pipe binds change with D; D=64 and D=128
  sharing one file does not share gains (`flydsl_attn_measurement.md` §8).
- **Static proxies do not rank candidates.** −351 instructions was worth 0.0% and a zero-instruction scheduling
  change +0.85% on one body *(field, E25, E26)*; a 31% longer LDS read-to-wait distance lost 0.65% on an MFMA-bound cell *(field, E27)*.
  Rank with the counter that matches the change (`flydsl_attn_attribution.md` §2).

## Related
`INDEX.md` · `../gemm/workflow.md` (round discipline) · `../gemm/attribution.md` ·
`../flydsl/kernels/flydsl_authoring_method.md` · `../hardware/gfx950/matrix_core.md`

## Evidence

Every workload below is an attention kernel. A FlyDSL version is given only where the cited run, or another round in the
same container and venv, printed it (`flydsl.__version__`, an import path with its version, or a `pip install flydsl==`
in the run's own setup); otherwise `FlyDSL ?`. Months are those of the round timestamps (`run.jsonl`), or of the dated
entry where only myskill or memory records the number. A sign on a wall-time change is the record's own.

- E1 [2026-08, gfx950, FlyDSL ?, N=?, issued MFMA FLOPs ÷ wall over the clock band measured under load (2078–2377 MHz), checked against PMC] issued MFMA 5.669e12 / 5.2924 ms = 1071 TF/s ⇒ MFMA-pipe busy ≈ 44–49%, against the PMC's 44.0% (gpt-oss D=64 attention backward, b4; campaign 20260829_150647, round-14 replan; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260829_150647/goal.md:339)
- E2 [2026-09, gfx950, FlyDSL 0.2.4, N=?, rocprofv3 PMC, clock from `GRBM_GUI_ACTIVE` (~1823 MHz under profiling)] `MFMA_BUSY` per trip 20,626 cycles of a ≈36,350-cycle trip ⇒ duty ≈52–57% at that clock; the same campaign's replan, at the sustained unprofiled 2070 MHz, put the duty at ≈50% (b4 1073–1086 / b2 1095 / b1 1102–1110 TF against a 2171 TF on-clock peak) and called 52–57% slightly high (local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260911_114527/goal.md:168) (gpt-oss D=64 attention backward, b4; campaign 20260911_114527 round 7; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260911_114527/memory.md:364)
- E3 [2026-09, gfx950, FlyDSL ?, N=2, official bench, two runs] three `sched_barrier(0)` fences relaxed to 0x008: speedup 0.946331 / 0.945965 (+3.1%), the D=128 cell 0.930 → 0.9855 (2 waves/SIMD), while the D=64 cell (3 waves/SIMD) moved only inside the noise (0.9099 → 0.9061) (FlyDSL dualwave attention forward, D=128 and D=64 cells; campaign 20260924_001030 round 3; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260924_001030/memory.md:150; local:/workspace/code/gpt_oss_docker/myskill/optimizer/methodology/15-attention-optimize.md:143)
- E4 [2026-07, gfx950, FlyDSL ?, N=?, PMC instruction count] an issue budget of "285×4 = 1140 slots against 1152 MFMA cycles = 99% issue-bound" (81% in the plan) compared per-SIMD issue slots with per-wave pipe cycles; with 4 waves sharing one pipe the issue occupancy is 960 / 4608 = 20.8%; no run record was found; month of the myskill commit (hd64 attention forward; local:/workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/13-meta-hd64-flash-bwd.md:1246)
- E5 [2026-07, gfx950, FlyDSL ?, N=3, campaign bench, three reads per round] four kept rounds outside the kernel body: dq XCD-major decode, kept together with a register-neutral LDS hand-over, +0.54%; dkdv XCD-major decode +2.81%; descending-q_tile dispatch, kept together with a dq LSE fold into the MFMA C-init that changed dQ's bits (SNR 52.16 → 50.26 dB), +2.50%; padding moved to the cheapest tile +0.96% (compounded ≈ +7%, computed from the four) (Meta hd64 attention backward, Sq=Skv=16384, B=1, hw-exp path; campaign 20260726_152547 rounds 6, 7, 10, 11; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260726_152547/memory.md:174, :186, :210, :222, :169, :205, :207)
- E6 [2026-07, gfx950, FlyDSL ?, N=?, ?] weeks of tile size, double buffering and traversal changes inside the kernel had been worth about 1%; a summary in myskill, no run record, dated 2026-07-27 there (Meta hd64 attention backward; local:/workspace/code/gpt_oss_docker/myskill/optimizer/methodology/15-attention-optimize.md:41)
- E7 [2026-07, gfx950, FlyDSL ?, N=3, campaign bench, three reads] dq XCD-major decode: L2 hit 86.53 → 94.84%, misses −62%, wall within noise in the first build; kept at +0.54% (same campaign, round 6; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260726_152547/memory.md:169)
- E8 [2026-08, gfx950, FlyDSL ?, N=3, palindromic arms, one process per arm] stall census: grid and tail ≤0.2% (gpt-oss D=64 attention forward; campaign 20260831_084030 round 13; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260831_084030/memory.md:281)
- E9 [2026-07, gfx950, FlyDSL ?, N=3, campaign bench, three reads] dropping the dq kernel's B-GEMM and its rho/R correction: 703.9 → 773.0 (+9.81%) (Meta hd64 deterministic attention backward, dq kernel; campaign 20260715_142909 round 7; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260715_142909/memory.md:84)
- E10 [2026-08, gfx950, FlyDSL ?, N=?, campaign ruler] eight rounds of scheduling, iglp, sched_group, region-mutation and chain-splitting arms measured +0.4% to +4.0%, while the stall pool had never been priced (gpt-oss D=64 attention backward; campaign 20260829_150647, round-14 replan; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260829_150647/goal.md:366)
- E11 [2026-07, gfx950, FlyDSL ?, N=?, campaign bench, ISA-verified interleave] exp interleave forced with `sched_barrier(0)` (MVVVV…, 56 runs of exactly 4): wall moved 0.00% (gpt-oss hd64 attention backward, dkdv, Hq=64, Hkv=8, Sq=Skv=8192, B=2, SBHD, two workgroups per CU; campaign 20260728_123346 round 7; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260728_123346/memory.md:137)
- E12 [2026-07, gfx950, FlyDSL ?, N=?, rocprofv3 PMC] LDS / MFMA busy ratio dq 0.25, dkdv 0.125: 4× and 8× LDS-port headroom (Meta hd64 deterministic attention backward; campaign 20260715_142909 round 11; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260715_142909/memory.md:127)
- E13 [2026-07, gfx950, FlyDSL ?, N=?, ?] forcing `waves_per_eu` 3 cost 32% (VGPR squeezed to 84 + 180 B scratch), 4 cost 5× (368 B scratch); no run record was found; month of the myskill commit (Meta hd64 attention backward, dq kernel; local:/workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/13-meta-hd64-flash-bwd.md:44)
- E14 [2026-09, gfx950, FlyDSL ?, N=?, ISA dump (spill counts)] `amdgpu-agpr-alloc` forced to 32,32 / 64,64: spill 31 → 52 / 77 (FlyDSL dualwave attention forward, D=64 cell; campaign 20260924_001030 round 13; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260924_001030/memory.md:365)
- E15 [2026-07, gfx950, FlyDSL ?, N=?, campaign bench] 8-wave NT=2 994.8; with the stagger 1029.8, but NaN / not deterministic (a WAR race on the shared Q/dO tile); baseline ~1104–1116 TF in the campaign goal, 1116.7 TF read at the start of that session (local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260720_143502/memory.md:104) (Meta hd64 attention backward, dkdv, hw-exp; campaign 20260720_143502 round 6; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260720_143502/memory.md:105)
- E16 [2026-07, gfx950, FlyDSL ?, N=?, campaign bench, ISA-verified] `sched_group_barrier` MFMA:TRANS at 1:2 / 2:4 / 4:8 (applied, `nop_cyc` 222 → 85): +0.26 / −0.04 / −0.07%; dkdv GEMM1 `s_setprio(1)`: 799.7, −3.3% (same campaign as E11, round 8; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260728_123346/rounds/round-08/agent.txt:13)
- E17 [2026-09, gfx950, FlyDSL ?, N=3, instruction-identical pair, three batches reproduced to 0.01%] pure reorders paid 0 ± 2% on a forward whose coexec covered 43.64% of SIMD cycles (52% of MFMA busy, computed from the two counters) (gpt-oss D=64 attention forward; campaign 20260901_123110 round 2; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260901_123110/rounds/round-02/agent.txt:24; table in local:/workspace/code/gpt_oss_docker/myskill/optimizer/methodology/03-profiling-utilization.md:768)
- E18 [2026-07, gfx950, FlyDSL ?, N=?, ?] skipping the per-tile K/V DMA +8.7%; the built double-buffered prefetch −4.3~4.7%; no run record was found; the entry is dated 2026-07-21 (Meta hd64 attention backward, dq kernel, fast-exp path; local:/workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/13-meta-hd64-flash-bwd.md:158)
- E19 [2026-07, gfx950, FlyDSL ?, N=?, ?] lazy O-rescale ported on its own: flash cr4 654 → ~450 TF (−31%), pro cr4 neutral; recorded in a memory note dated 2026-07-12 (DeepSeek-V4 sparse-MLA attention forward, K32 path; local:/workspace/code/gpt_oss_docker/.claude/memory/project_dsv4_fwd_lazy_rescale.md:13)
- E20 [2026-07, gfx950, FlyDSL ?, N=?, campaign bench] fused dq+dkdv with a transient dQ: 1.95–2.55× slower; myskill records it as the KV-outer fusion at BLOCK_KV=64 (local:/workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/13-meta-hd64-flash-bwd.md:139) (Meta hd64 deterministic attention backward; campaign 20260715_142909, measured earlier in the session and listed in round 11; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260715_142909/memory.md:130)
- E21 [2026-08, gfx950, FlyDSL 0.2.4, N=?, campaign bench] the real fp32-atomic dQ (512 `buffer_atomic_add_f32`, 486 VGPR, 0 scratch): 105.5 ms against 6.34 ms, versus 1.229 ms for the whole dQ split-K round trip (D=128 attention backward, cell d128_70b_b2 (B=2); campaign 20260822_152617 round 4; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260822_152617/rounds/round-04/agent.txt:5)
- E22 [2026-07, gfx950, FlyDSL ?, N=?, campaign bench] dkdv BLOCK_Q=128 → 254 (−67%, register-walled) (same campaign as E12, round 11; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260715_142909/memory.md:127)
- E23 [2026-09, gfx950, FlyDSL ?, N=?, illegal subtractive probes] removing the hand-written vmcnt drain +0.3%, all `s_waitcnt` +0.4%, the barrier family +3.2% (FlyDSL dualwave attention forward; campaign 20260923_144851 round 3; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260923_144851/memory.md:136)
- E24 [2026-08, gfx950, FlyDSL ?, N=3, campaign bench, three reads] SWA(128) effective TF 216.5 → 294.6 (round 3) → 298.1 (round 4); round 1 found the window trim already present on the split path SWA takes, and the gains came from window-gated configuration (q_split, BLOCK_M, BLOCK_KV, register prefetch, dead-tile skip, prologue batching) (gpt-oss attention backward, SWA(128), B=3, S=8192, Hq64/Hkv8, D64; campaign 20260808_153438; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260808_153438/memory.md:59)
- E25 [2026-08, gfx950, FlyDSL ?, N=?, campaign bench, single-variable arms] −351 instructions / −234 shuttle / −8 dwords / spill 1→0 = 0.0% (d64 attention backward with fp8 operands, dkdv; campaign 20260828_090905 round 9; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260828_090905/rounds/round-09/agent.txt:12)
- E26 [2026-08, gfx950, FlyDSL ?, N=?, campaign bench] `iglp_opt` id 3, never tried before, is worth +0.85% (same kernel; campaign 20260828_090905 round 13; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260828_090905/memory.md:310)
- E27 [2026-09, gfx950, FlyDSL ?, N=?, official bench] `V_HEAD` 1→2 on the D=128 cell: LDS read cover distance 17.8 → 23.4 (+31%), instructions −4, `s_waitcnt` −3, VGPR and spill unchanged; the D=128 cell 0.985395 → 0.97886 (−0.65%) (FlyDSL dualwave attention forward; campaign 20260924_001030 round 4; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260924_001030/rounds/round-04/agent.txt:38)
