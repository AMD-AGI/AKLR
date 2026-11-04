---
title: "FlyDSL GEMM — wave placement, XCD order, scheduling fences and epilogue cost that are easy to get backwards"
kind: topic
lever: flydsl_gemm_schedule_and_locality
ops: [gemm]
gens: [gfx950]
dtypes: [bf16, fp8, mxfp4]
regimes: [compute-bound, register-saturated]
flydsl: 0.3.2
sources:
  - ROCm/FlyDSL@dd8373349:python/flydsl/expr/rocdl/__init__.py:99
  - ROCm/FlyDSL@dd8373349:python/flydsl/expr/rocdl/__init__.py:142
  - ROCm/FlyDSL@dd8373349:python/flydsl/expr/rocdl/__init__.py:877
  - ROCm/FlyDSL@dd8373349:python/flydsl/expr/rocdl/universal.py:266
  - ROCm/FlyDSL@dd8373349:kernels/gemm/fp8_gemm_4wave.py:135
---

# Wave placement, XCD order, scheduling and epilogue in a FlyDSL GEMM

> Each section here is a place where the obvious move measured in the wrong direction. Read the rule,
> then check whether your kernel is in the regime it was measured in. "GH" is Primus-Turbo's
> `primus_turbo/flydsl/utils/gemm_helper.py` at main `7615a5a2`; "PS" and "A16" are FlyDSL's
> `kernels/gemm/preshuffle_gemm.py` and `kernels/gemm/gemm_a16w16_gfx950.py`.
>
> Numbers marked `*(field, E<n>)*` are measured values; their conditions are listed under Evidence at the end.
> Use them for direction and order of magnitude, not as targets.

## Route here when
- A change that should be neutral (re-labelling waves, a new swizzle, a relaxed fence) moved wall time.
- You are choosing a tile order across XCDs, or porting one kernel's order to another.
- You are writing `rocdl.sched_*` hints or `sched_barrier` fences.
- You want to fuse work into the epilogue of a register-heavy GEMM.
- Accumulators no longer fit, or buffer loads sit inside a `v_readfirstlane` loop.

**Somewhere else if:** the grid, tile size, split-K or a boundary variant is the question →
`flydsl_gemm_grid_pricing.md`; the general order of operations → `flydsl_gemm_workflow.md`.

## 1. Co-resident waves must not read the same heavy fragment

A workgroup's waves stripe across the CU's 4 SIMDs; with 8 waves, wave `w` and wave `w + 4` share a SIMD.
If those two read the same LDS fragment of the heavier operand at the same moment, their LDS requests are
in phase and neither hides the other's latency.

*(field, E1)* One 8-wave grouped fp8 GEMM kernel at 2 waves per SIMD with a 2(M)×4(N) wave grid. Writing
the same `(wave_m, wave_n)` bijection three ways — identical tile split, LDS bytes, instruction counts:

| What the co-resident pair shares | Result |
|---|---|
| `wave_n` only (same B fragment, the lighter side) | baseline |
| neither | same as baseline |
| `wave_m` only (same A fragment, 8 of the wave's 12 `ds_read`s) | **~25% slower** |

*(field, E1)* Transposing the wave grid to 4(M)×2(N), the co-resident pair still sharing `wave_n`, cost ~16%
on its own. Cache, LDS, bank-conflict and MFMA
counters were bit-identical between arms; only `SQ_WAIT_ANY / SQ_WAVE_CYCLES` moved (31.7% → 50.0%).

So: any rewrite of the wave grid — new MFMA atom, different `n_tiles` split, changed boundary granularity
— must state the `wave_id → (wave_m, wave_n)` mapping explicitly and recheck which waves co-reside. It is
index algebra that can cost a quarter of the kernel, and ordinary counters will not show it.

## 2. Tile order across XCDs

gfx950 has 8 XCDs of 32 CUs each, and L2 is per XCD: an operand that another XCD pulled in is not an L2
hit. The hardware hands workgroup ids to the XCDs round-robin (`block_id % 8`), so a linear tile order
spreads tiles that share a panel over all 8 L2s; the basic swizzle remaps `block_id → (xcd, local_id)` so a
contiguous run of tiles lands on one XCD. The GEMM-specific rules:

**Decide by what co-resident workgroups reuse.**
- Each group of tiles streams **a slab of its own** (shared inside the group, not across the grid) that is larger
  than one XCD's L2: keeping the group's workgroups on the same XCD at the same time (cluster across all 8 XCDs)
  is K-dependent: *(field, E2)* on a grouped fp8 NN dgrad, 8 XCDs with GROUP_M = 4 against 4 XCDs with
  GROUP_M = 8 gained 0.79% at K = 2944 and lost 1.91% at K = 5760, both slabs larger than the L2 slice. Key the
  choice on K too, and sweep.
- All tiles read a **shared slice**: clustering by XCD only fragments the
  sharing; the hardware round-robin (`block_id % 8`) with a narrow group-major band wins.
- **Split-K: give each XCD a contiguous run of one split's tiles**, not all splits of a few tiles: splits of one tile
  read disjoint K-slices, so an XCD's L2 then holds one K-slice of A and B. *(field, E3, E4)* 46.5 → 37.8 µs on a
  split-4 grid at M = 893; the tile-major run was 11% slower there and 2% slower at M = 500. *(field, E34)* With the
  fold removed, XCDs that each ran half of K (two K groups by XCD parity) made the main loop 3.5% faster than all of
  K at M = 256 (A 8 MiB). These orders were all timed with folds in which splits waited for each other, and on
  another such kernel, whose fold exchanged partials inside one XCD's L2 when a tile's splits shared it, one split per
  XCD lost: race the order with the fold you ship (`flydsl_gemm_small_m.md` §4).

*(field, E5)* On grouped fp8 GEMMs, porting one kernel's winning band/XCD setting to another kernel made it 0.8% to
2.3% slower (the identity XCD order with GROUP_M = 4, recommended for grouped kernels in general, lost 5.7%). Predict
the direction from the reuse class, then sweep every group size up to 8 and past it (9 beat every value ≤ 8 that was
tried — 3, 4, 6, 7 at one shape, 2, 3, 4, 6, 7 at the other — by 1–4% on two shapes, and 4 won at the third
*(field, E6)*), not 3–5 points, on every shape class you are scored on: *(field, E7)* adjacent `GROUP_M` values
differed by up to ~14% (9% paired).
Re-sweep the group size after every tile change: *(field, E8)* at one shape, groups 4 and 8 lost up to 2.4% at one
tile; at another, group 4 won 2.3% and group 2 3–4.4%, while groups 6 and 8 won about 0.4%.

**FlyDSL's kernels expose the order as an argument:** `xcd_swizzle` in PS (`xcd_remap_bx_by`, in
`kernels/common/mma/mfma_preshuffle_pipeline.py`), `use_xcd_remap` in `fp8_gemm_4wave`, and `group_m`
(`BlockSwizzle`) in A16. A16's swizzle falls back to the plain order when `group_m` is 0 and for grids under
256 workgroups or not a multiple of 8 (`kernels/gemm/gemm_a16w16_gfx950_utils.py:105, 123`).

**How Turbo orders tiles** (Primus-Turbo main `7615a5a2`, `primus_turbo/flydsl/utils/gemm_helper.py` = GH):
- `xcd_remap_pid` (GH:1639) gives each XCD a contiguous slice of logical tiles, for any total. It is a
  bijection, so it cannot change results.
- `block_mn` (GH:1967) is a `GROUP_M` (1D) or `group_n` (2D) super-tile decode, also a bijection.
- `xcd_band_remap_pid` (GH:2694) gives each XCD every `num_xcd`-th run of `band` tiles. That keeps reuse
  inside a run while every XCD samples the whole tile range, which balances tiles of unequal cost.
- **Gate the band on L2 bytes.** Once one band's operand slice outgrows an XCD's L2 slice (4 MiB on
  gfx950), clustering buys no reuse and only costs balance. Turbo's dense fp8 table then switches to the
  plain round-robin (`_NT_BAND_L2_BYTES`, `gemm/gemm_fp8_kernel.py:3186-3192`). The MXFP4 kernel picks
  the widest power-of-two N band whose B slice fits one XCD's L2, and no band once a single N tile fills
  it (`_mxfp4_l2_band`, `gemm/gemm_mxfp4_kernel.py:3131-3139`).
- **Make each XCD's resident set a rectangle.** The dense fp8 TN whole loop sizes `(group_m, group_n)`
  so the tiles one XCD holds pull the fewest operand bytes through its L2. It snaps the group height to a
  divisor of 8, "which is what keeps its resident set a rectangle instead of a staircase" (`_tn4_band_group` and
  `_tn_wave4_band`, `gemm_fp8_kernel.py:3280-3297`).
- **A persistent loop strides by the grid, not by 1.** Then one dispatch round's tiles stay inside one
  band and the workgroups co-resident on an XCD share a B tile from L2. A consecutive walk per workgroup
  widened the live band and cost 2.7–4.0% *(field, E9)* (`gemm_mxfp4_kernel.py:2962-2966`).
- **Keep the tile decode cheap.** It sits in front of the first load at occupancy 1. Use unsigned
  division and modulo, make N a compile-time constant so divisions become multiplies, and when dividing
  by a runtime value, use a fixed-point reciprocal that is proven exact over the reachable range.

**2D bands:** compare `GM · GN` with the per-XCD concurrent window (resident workgroups per XCD: 32 at one
workgroup per CU). `GM · GN` near the window makes the footprint depend on where the window starts inside
the band; a 1D band is steadier there. Use 2D only when `GM · GN` is much larger than the window.

**Mechanics:**
- One locality remap at a time — stacking two usually defeats both.
- Persistent kernels remap the work id, not `block_idx`.
- Prove the remap is a bijection over every grid you actually launch (a chunked launch can be narrower
  than the deployment shape). `x*(G//8) + min(x, G%8) + block_id//8` with `x = block_id % 8` is a
  permutation for any G, so it needs no divisibility gate.
- FlyDSL's own `_xcd_swizzle` (`kernels/gemm/fp8_gemm_4wave.py`) is a contiguous-chunk remap plus
  `GROUP_M = 4`, used only when the grid has ≥ 4·256 workgroups and a multiple of 8; otherwise linear.

**Judge by wall time; explain with stall counters, not L2 hit rate.** *(field, E10)* An XCD order that was 3% faster
on a grouped fp8 GEMM under balanced routing (2.5% under heavy skew, 0.03% on the canonical distribution) left
`TCC_HIT`, `TCC_REQ` and `TCP_TCC_READ_REQ` unchanged; the gain showed in `MemUnitStalled` (12.4% → 10.7%), i.e.
TCP→TCC request queueing. Elsewhere hit rate rose sharply with no wall change. Read `SQ_WAIT_ANY` and `MemUnitStalled`
to explain a result. *(field, E11)* Differences of 1–2% on this axis flipped sign between back-to-back sweeps;
10%-scale effects (14–16%, and 23–25% the other way) agreed with the shipped heuristic. *(field, E12, E13)* That
reading was on a grouped NN dgrad; a later band sweep on the same pass improved `MemUnitStalled` in both the winning
and the losing cell, and only on a grouped NT forward did it agree with wall time (the faster band, +1.7–2.6%, had
6.4% less `MemUnitStalled` at the same L2 hit rate). Decide by wall time.

## 3. Scheduling hints and fences

FlyDSL's helpers are thin wrappers over `sched_group_barrier`:

| Helper | Mask | Classes |
|---|---|---|
| `rocdl.sched_mfma(n)` | `0x008` | MFMA/WMMA |
| `rocdl.sched_vmem(n)` | `0x020` | **VMEM reads only** |
| `rocdl.sched_dsrd(n)` | `0x100` | DS reads |
| `rocdl.sched_dswr(n)` | `0x200` | DS writes |

`rocdl.sched_barrier(mask)` takes the same bit vocabulary — `0x001` non-memory/non-side-effect (this
**includes VALU**), `0x002` VALU, `0x004` SALU, `0x008` MFMA, `0x010` all VMEM, `0x020`/`0x040` VMEM
read/write, `0x080` all DS, `0x100`/`0x200` DS read/write, `0x400` transcendental, `0x800` LDS DMA. A set
bit means that class **may cross** the fence; `0` pins everything.

**FlyDSL's `hot_loop_scheduler` is the reference use.** `preshuffle_gemm` builds one hint table per tile.
On gfx950 it spreads the LDS reads and VMEM loads evenly over all MFMAs,
for both sync and async copy (`https://github.com/ROCm/FlyDSL/blob/v0.3.2/.claude/skills/gemm-optimization/SKILL.md` §5.5);
the sync path also places its LDS writes at the tail.
`preload=(dsrd, dvmem)` sets how many go ahead of the first MFMA
(PS:439-494).

Fences inside a fully unrolled epilogue usually exist to stop the scheduler from hoisting work and
lengthening live ranges. What must stay pinned is the arithmetic, not only the memory operations:
- *(field, E14)* Relaxing such a fence from `0` to `0x001` in a grouped MXFP4 GEMM's epilogue: VGPR 476 → 512,
  spills 0 → 30 — VALU intermediates hoisted across the fence.
- *(field, E15, E16, E17)* `0x004 | 0x008 | 0x080` (SALU, MFMA, DS may cross; VALU and VMEM pinned) in the same
  epilogue: no spills, unchanged VGPR, `s_nop` 79 → 53. That makes it safe, not faster: beside a barrier removal it
  added 0.16 to 0.6 points of the campaign score; with a vector-sum rewrite in the same region it added 55–56
  instructions per shape and was net negative; on a dense MXFP4 GEMM's fused dGELU epilogue any relaxation was 2.3%
  slower. Race the mask on the current code whenever the region changes.

**LLVM also moves work across waits you place by hand; pin it with `rocdl.sched_barrier(0)`.** Loads issued up
front for latency sink behind the MFMAs to save VGPRs (a fence after them was +0.6–2.6% on rows with ≥ 12 loads
*(field, E18)*). MFMAs issued before an inline-asm `s_waitcnt` + `s_barrier` sank past it, 8 of 28 (a fence in
front cut the time 0.8–3.8% at M ≥ 893 and 1.4–1.7% at M = 64–96, and added 1.1–2.4% at M = 56, 60 and 112
*(field, E19)*; race it).

Gate any mask change on the ISA's `vgpr_spill_count` before benchmarking. Hint counts must come from the
loop in front of you; constants copied from another kernel are worse than none.

**Race LLVM's scheduler per tile:** `flyc.compile[{"llvm_options": {"amdgpu-sched-strategy": "iterative-ilp"}}]`
(or `max-ilp`, `max-memory-clause`, `iterative-minreg`, `iterative-maxocc`); a misspelled name silently compiles the default.
*(field, E20)* On a 256-VGPR bf16 tile `iterative-ilp` cut spills 9 → 0 and ~2% of the time; `iterative-minreg` won <1% on another tile and doubled this one.
*(field, E36)* On a fused-MoE fp8 × MXFP4 expert GEMM whose waits are placed by hand none paid: stage 2 ran 1–7% slower
under `max-memory-clause`, `max-ilp` and `iterative-minreg`; on stage 1 `max-ilp` and `iterative-ilp` took the
registers from 240 to 280 and ran 15–19% slower in noisy runs.

**A count mismatch is never reported.** `sched_group_barrier` groups are filled greedily in program
order. For 4 independent MFMAs and 4 global loads:
- with no hints the order is `L L L L M M M M`;
- `(mfma 1, vmem 1) × 4` gives `M L M L M L M L`;
- `(mfma 1, vmem 3) × 4` front-loads the loads: `M L L L M L M M`;
- `(mfma 1, vmem 2) × 2` gives `M L L M L L M M`: the groups pin the head and the leftover MFMAs follow;
- the compiler prints no warning in any case.
Read the scheduled ISA after changing a hint table.

**Kernels with hand-placed waits use none of this.** Turbo's dense kernels contain no
`sched_group_barrier`. They use `rocdl.sched_barrier(0)` around hand-written waits, `s_setprio` at two
waves per SIMD, and `s_nop` issue pads inside whole-loop asm (`flydsl_gemm_mainloop.md` §2–§3).

**Where each K step's fragment reads are fenced from its MFMAs by `sched_barrier(0)`** (aiter's `splitk_hgemm`
loop), read step kk+1's fragments before step kk's MFMAs. *(field, E21, E22)* +0.5–4% on most bf16 tiles, up to ~1% worse
on two, 2–4× slower on tiles already at 256 VGPRs (it spilled); prefetching across the stage barrier lost. Race it per tile.
In the same loop, race which K steps the next stage's DMAs go in. *(field, E23)* All of them in step 0 lost up to 5% to half in
each of steps 0–1; thirds over steps 0–2 measured −2.4 to +1.7% by M.

**Prefetching across the stage barrier can pay once the barrier moves before the last K step's final MFMA groups**, with
the freed stage refilled under them. *(field, E24)* +~1% on an 8-wave bf16 2-stage loop; moving the barrier alone lost up to 4.4%.

## 4. gfx950 has one memory counter for loads and stores

`s_waitcnt vmcnt(N)` and `s_waitcnt lgkmcnt(N)` assemble for gfx950; `s_waitcnt_vscnt`,
`s_wait_loadcnt`, `s_wait_storecnt` and `s_wait_dscnt` do not (the assembler reports "instruction not
supported on this GPU" — they belong to later ISAs). Stores in flight count in `vmcnt`, so waiting for a
load also waits for every earlier store. Consequences:
- Dripping C stores into the main loop serializes them against the next operand wait.
- Non-memory work (MFMA, VALU, SALU) occupies neither counter; in a GEMM loop the MFMAs are the bulk of
  it, so they are the medium to hide a wait under.

**Write a wait as "what may still be in flight", not as a formula of the block shape.** At a phase
barrier, leave outstanding only the fills that nothing reads before a later phase, and only the
`ds_read`s whose consumers come after the barrier. Then race the pair:
- On one whole loop, the autotune race offers `(vmcnt, lgkmcnt)` of `(10, 9)` and `(16, 15)`, the
  deeper pair only for K ≥ 8192 and adopted only at a 2% margin (`_wl_opts`, `_WL_MARGIN`,
  `gemm_mxfp4_kernel.py:3208, 3274`). *(field, E32)* At K = 4096 on that loop, `(13, 12)` and `(16, 15)` read
  within 0.02% of `(10, 9)` on two shapes.
- `lgkmcnt` 15 is the encoding limit of the 4-bit field, not a tuning boundary. The depth at which a wait
  stops being bitwise repeatable is per kernel: *(field, E25, E26)* another whole loop, on a grouped fp8 GEMM, was
  fastest at 12 on a score over balanced, moderate and heavy routing (15 within 0.1%) and at 15 under
  capacity-collapsed routing, and stayed bitwise repeatable at 15 over 4500 launches. Prove any deeper setting by
  repetition (`flydsl_gemm_correctness.md` §3).
- *(field, E27)* The lgkm-side partial drain carried over between kernels and precisions (+0.3% on a grouped fp8
  GEMM, +0.6–0.7% with two further drain edits). A one-`s_nop` pad after the barrier helped MXFP4 and was worth
  0.00% on the grouped fp8 GEMM. Port the lgkm side first.

## 5. Hiding an LDS write when registers are nearly full

The usual advice is to put independent work (the next global load, address math, an independent MFMA
group) between a `ds_write` and the `lgkmcnt` wait that guards its read. When VGPR headroom is small, any
insertion lengthens a live range and loses:

*(field, E28)* Occupancy 1, 440/472 VGPR + 256 AGPR, fully unrolled epilogue, same gap three ways:

| Move | Result |
|---|---|
| issue the `ds_write` itself earlier, as soon as its data is ready | **+0.6%** |
| insert the next stage's `buffer_load`s into the gap | −2.0% |
| move the `ds_write` into the previous MFMA stream | −2.6% |

Rule of thumb from that kernel: below ~15% VGPR headroom, move the write, do not insert work. A real
latency fix barely changes the instruction count (5315 → 5307 here *(field, E28)*); a large count change means you
changed something else.

## 6. Epilogue fusion competes with the tile that made the GEMM fast

The tile that wins the GEMM often leaves one wave per SIMD. Then no other wave hides the epilogue's VALU
work, and every `v_*` instruction the fusion adds shows up in the tile time.

*(field, E29, E30)* At one wave per SIMD, an exact GELU added to the epilogue cost +70%; rewritten as independent
vector chains it cost about 1.1× the same math in a separate high-occupancy kernel. The price is set by the
extra `v_*` instructions per element, the read width of any extra operand, and whether the fusion adds a
new input stream.

Before planning an epilogue fusion, read the kernel's occupancy. At one wave per SIMD, write element-wise
math as independent vector chains, count the added `v_*` per element, and compare against a separate
kernel. The count alone does not set the price: *(field, E33)* in the same epilogue an FP6 row pack, also written
as parallel chains, cost 1.87× the GELU's time per added instruction on one shape and 2.17× on another.

## 7. Accumulators, descriptors and large operands

- **Accumulators beyond the AGPR file.** A wide tile can need more than 256 accumulator dwords per lane.
  Place them yourself (part AGPR, part VGPR). *(field, E31)* Leaving a 128×512 tile's accumulators to the
  compiler through `fx.gemm` produced 613 `v_accvgpr_*` moves and 18 scratch loads in the hot loop;
  explicit placement of a 256×384 tile ran at 500/512 registers with no spill. Check: count
  `v_accvgpr_read/write` and `scratch_` in the hot loop.
- **Descriptor fields must be wave-uniform, and operands past 4 GiB need an i64 rebase.** See
  `flydsl_gemm_datapath.md` §8. Signature of a non-uniform descriptor: `v_readfirstlane_b32` plus an
  `s_and_saveexec_b64` loop around the hot loop's `buffer_load`s.
- **Ragged edges.** `fx.rocdl.make_buffer_tensor(t)` defaults to `max_size=True`, which sets
  `num_records = 0xFFFFFFFF` — no out-of-range masking. To have out-of-range loads return zero and
  out-of-range stores dropped, pass `max_size=False` or `num_records_bytes`, and derive the bound from
  traced values (`../flydsl/debug-flydsl-kernel.md` § What the cache key sees).

## Verify
- The `wave_id → (wave_m, wave_n)` mapping and its co-resident pairs are written down for the kernel.
- A remap is checked as a bijection over every launched grid width, with the exit on the raw id.
- A fence change shows `vgpr_spill_count` unchanged before any timing.
- An XCD/order change is judged on wall time and stall counters across all shape classes.

## Related
`flydsl_gemm_grid_pricing.md` · `flydsl_gemm_workflow.md` · `../flydsl/debug-flydsl-kernel.md`

## Evidence

A FlyDSL version is given only where the cited run, or another round in the same container and venv, printed it
(`flydsl.__version__`, an import path with its version, or a `pip install flydsl==` in the run's own setup); otherwise
`FlyDSL ?`. The Kimi-K3 forge runs (`gemm_n…_k…`) installed `flydsl==0.3.2`. The month is the run's: the round
timestamps in each campaign's `run.jsonl`, the K3 lesson files (2026-10-24…26). "Turbo" is Primus-Turbo. Where the
number on the card came to it through myskill, the line names the myskill card too.

- E1 [2026-09, gfx950, FlyDSL 0.2.2, N=?, in-process paired A/B, 16 launches per window in both orders, against an inert control arm that read ±0.06–0.31%; output cross-check 0] the same `(wave_m, wave_n)` bijection written three ways on the shipped kernel (16-wide MFMA atom, 2(M)×4(N) waves): the co-resident pair sharing `wave_m` −26.5% (`shipm1`), sharing neither +0.19% against the inert arm (within its floor); the 4(M)×2(N) grid sharing `wave_n` −15.9% (`w4`); for `shipm1` the TCP/TCC request, L2 hit/miss and EA counters agreed within 0.3%, `SQ_INSTS_MFMA` and MFMA busy cycles were identical, and `SQ_WAIT_ANY/SQ_WAVE_CYCLES` went 31.7% → 50.0%; 8 of each wave's 12 `ds_read`s read A; myskill gives −25.1 to −26.5% over the balanced and unbalanced cells (gpt-oss-20b grouped per-tensor fp8 GEMM forward (NT), Turbo `grouped_gemm_fp8_tensorwise_flydsl_kernel`, Down cells at B=24; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260904_151102/rounds/round-13/agent.txt:24; local:/workspace/code/gpt_oss_docker/myskill/optimizer/methodology/04-occupancy-and-tile.md:113)
- E2 [2026-09, gfx950, FlyDSL 0.2.2, N=12, in-process paired probe: 3 process pairs × both orders × 3 draws per cell] 8 XCDs with GROUP_M = 4 against the 4-XCD, GROUP_M = 8 lead on the dgrad: Down (K=2944, B[g] 8.67 MB) +0.79%, GateUP (K=5760, B[g] 16.96 MB) −1.91%, 11 of 12 reads negative; the campaign bench read Down's gain as +1.4 to 2.1%; the KernelForge grouped card carries the same two numbers (gpt-oss-20b grouped per-tensor fp8 GEMM dgrad (NN), B=24; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260904_151102/memory.md:256; Hyperloom 8df49821c:src/kernelforge/knowledge/local_wiki/resources/languages/flydsl/skills/grouped_gemm/flydsl_grouped_scheduling.md:79)
- E3 [2026-10, gfx950, FlyDSL 0.3.2, N=4, driver bench (`driver.py --bench-mode`), against three incumbent readings] split-major (tile, split) order, each XCD a contiguous run of one split's tiles, for the distributed fold at m=893, split 4: 37.85, 37.85, 37.87, 37.84 µs against 46.55, 46.60, 46.58; own CUDA-graph timer 52.37 → 44.17; output byte-identical (gemm_n2112_k7168 forge run, iter 7; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n2112_k7168/forge_experiments/lessons/iter_007.md:12)
- E4 [2026-10, gfx950, FlyDSL 0.3.2, N=?, AKLR `bench_cudagraph` (L2 flushed by a buffer read) or the driver bench, variant against incumbent] tile-major order inside each XCD for the same fold: m=893 41.89 against 37.56 µs (+11.5%), m=500 34.62 against 33.89 (+2.2%) (gemm_n2112_k7168 forge run, iter 9; 192×192 tiles, 2×4 waves, 3 stages, split 4; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n2112_k7168/forge_experiments/lessons/iter_009.md:25)
- E5 [2026-08, gfx950, FlyDSL ?, N=?, isolated kernel A/B at the deployment M of 4096 rows per expert, bitwise-identical outputs] band settings moved between the three kernels of one campaign: the wgrad winner (identity XCD order, GROUP_M = 2) on the NT forward −2.26%; on the wgrad the winner beat the previous order by 0.78–0.98%, and the NT/NN winner (8 XCDs, GROUP_M = 4) "falls back to REF"; on the NN dgrad, identity order with GROUP_M = 4, the order a KB card then recommended for grouped kernels, −5.72% (gpt-oss-20b down-projection padN, grouped tensorwise fp8, G=32, M=131072, K=N=2944: forward NT, dgrad NN, wgrad TN; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260817_065611/memory.md:178; :166 for the dgrad arm; myskill methodology/05-lds-swizzle-prefetch-sched.md:154)
- E6 [2026-10, gfx950, FlyDSL 0.3.2, N=?, driver graph-timer medians (warmup 3, 20 iterations), FlyDSL cache cleared before each sweep] GROUP_M sweeps in ms: m=2732 (160×256, 3 stages) 9: 1.129–1.133 against 7: 1.173–1.175, 3: 1.194, 4: 1.189, 6: 1.191, 18: 1.203; m=9397 head 9: 3.440–3.442 against 2: 3.538, 3: 3.483, 4: 3.487, 6: 3.486, 7: 3.521, 12: 3.545, 18: 3.519; at m=16384, 4 won (5.886–5.896 against 8: 5.997, 16: 6.112) (gemm_n7168_k35840 forge run, iter 3; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n7168_k35840/forge_experiments/lessons/iter_003.md:56)
- E7 [2026-10, gfx950, FlyDSL 0.3.2, N=5–7, own CUDA-graph A/B script: 512 MB zeroed to flush, candidate order alternated, median of 5–7 reps; paired deltas from a graph-timer harness, 4–5 interleaved rounds per arm] GROUP_M at m=16384: 3 157.2 µs, 4 179.5, 7 165.4 against the base row's 163.8; paired against GROUP_M 4 (172.68 µs), 3 was 14.00 and 6 13.78 µs faster (gemm_n7168_k768 forge run, iters 7 and 12; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n7168_k768/forge_experiments/lessons/iter_007.md:56; iter_012.md:35)
- E8 [2026-10, gfx950, FlyDSL 0.3.2, N=?, scratch script: 10 launches per CUDA graph, 512 MB flush before each replay, median of 20 replays; 4 interleaved repeats in iter 8, one process per variant in iter 9 with the repeat count not stated] at m=16384 with the 416×192 tile, group 8 1659 and group 4 1638 µs against 1620 for the canonical order; 416×192 with group 4 about 1623 against about 1620 in iter 9; with 320×256, against about 1576 in plain row order: group 8 about 1570, 6 about 1570, 4 about 1539, 3 1516–1528, 2 1506–1527 (gemm_n8448_k7168 forge run, iters 8 and 9; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n8448_k7168/forge_experiments/lessons/iter_008.md:26; iter_009.md:16)
- E9 [2026-09, gfx950, FlyDSL ?, N=?, campaign score (`_bench_campaign_flux.py`), repeat count not stated] persistent tile loop walking consecutive post-remap tiles per workgroup instead of striding by the grid: score 0.9858 against 1.0059, the three cells with more than one tile per workgroup −2.74/−3.60/−3.98% against a +0.11% control mean; Turbo keeps the result as a source comment, added in 4b583333 (#514) (Flux MXFP4 dense GEMM, persistent path; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260927_035256/memory.md:127; Primus-Turbo 7615a5a2:primus_turbo/flydsl/gemm/gemm_mxfp4_kernel.py:2966)
- E10 [2026-08, gfx950, FlyDSL ?, N=5, isolated kernel, 5-round palindrome, bitwise-identical outputs; PMC pass count not stated] 8 XCDs with GROUP_M = 4 on the NN dgrad at the deployment M: balanced +2.95%, heavy skew +2.49%, canonical distribution +0.03%; PMC: `MemUnitStalled` 12.4% → 10.7% (−13.7%) while `TCC_REQ`, `TCP_TCC_READ_REQ` and `TCC_HIT` (76%) did not move (gpt-oss-20b down-projection padN dgrad NN, grouped tensorwise fp8, G=32, M=131072, K=N=2944, B[g] 8.67 MB; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260817_065611/memory.md:166; rounds/round-06/agent.txt:25)
- E11 [2026-09, gfx950, FlyDSL 0.2.4, N=2, two interleaved in-process sweeps of `num_xcds` over 18 cells] differences of 1–2% changed sign between the sweeps: QKV_dgrad xcd=1 1.76% faster, then 2.3% slower; FC2_fwd xcd=2 0.96% faster, then 2.9% slower; the record calls the 10%-scale effects confirmed: FC1_dgrad and FC1_wgrad xcd=16 14–16% faster than xcd=1, one 32768×4096×3072 cell with `(4,0,16)` 23–25% slower (Turbo MXFP4 dense GEMM, 6 gpt-oss and 12 Llama training shapes; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260925_144843/rounds/round-07/agent.txt:29)
- E12 [2026-09, gfx950, FlyDSL 0.2.2, N=?, rocprofv3 PMC of the two band arms, per tile] the arms of E2 (8 XCDs, GROUP_M = 4 against 4 XCDs, GROUP_M = 8): `MemUnitStalled` −1.58% on Down (wall +0.79%) and −2.98% on GateUP (wall −1.91%); `TCC_EA0_RDREQ` per tile +9.23% and −5.02%; `SQ_INSTS_MFMA`, `SQ_LDS_IDX_ACTIVE` and TCP→TCC requests per tile unchanged (gpt-oss-20b grouped per-tensor fp8 GEMM dgrad (NN), B=24; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260904_151102/rounds/round-04/agent.txt:80)
- E13 [2026-09, gfx950, FlyDSL 0.2.2, N=4, in-process paired A/B in four processes; one PMC comparison] on the forward, 8 XCDs with GROUP_M = 4 against the 4-XCD, GROUP_M = 8 lead: balanced +2.23/+2.27/+2.61/+1.88%, unbalanced +1.74%; `MemUnitStalled` −6.4% with identical L2 hit rates (gpt-oss-20b grouped per-tensor fp8 GEMM forward (NT), Down at B=24; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260904_151102/rounds/round-01/agent.txt:9)
- E14 [2026-09, gfx950, FlyDSL 0.2.4, N=1, ISA of one build, no timing] the epilogue's two `sched_barrier(0)` fences relaxed to `0x1`: fc2 `vgpr_count` 476 → 512, `vgpr_spill_count` 0 → 30, `private_segment_fixed_size` 0 → 124, 187 extra `v_accvgpr_write_b32` (gpt-oss-20b MoE FC2 dgrad, grouped MXFP4 GEMM with a fused dGLU and dual MXFP4 quantization epilogue, `grouped_gemm_fp4_dglu_impl`; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260923_120547/memory.md:257)
- E15 [2026-09, gfx950, FlyDSL 0.2.4, N=4, campaign bench (geomean over 8 cells), four reads, against a baseline that drifted +0.69 points the same day] mask `0x4|0x8|0x80` on the same two fences: spill 0/0, `vgpr_count` 476 unchanged, LDS unchanged, `s_nop` 79 → 53, `v_pk_mul_f32` 1152 → 1113; bench 108.47 / 108.19 / 108.70 / 106.49, "+0.16 to 0.6" over the round's first change (a barrier removal) alone; with both changes the six fc2 cells were all faster, +1.22 to +2.76 (+1.06–1.09% after detrending against the fc1 control) (same kernel; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260923_120547/memory.md:319)
- E16 [2026-09, gfx950, FlyDSL 0.2.4, N=5, campaign bench, five reads per arm] four rounds later, on a tree that also had a vector-sum rewrite, the same mask added 55/56/56 instructions to the three fc2 shapes; the arm with it overlapped the baseline, the arm without it did not, and the fences went back to `sched_barrier(0)` (same kernel; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260923_120547/memory.md:409)
- E17 [2026-10, gfx950, FlyDSL ?, N=?, same-session probe, four arms, fc2 minimum] on a dense MXFP4 GEMM's fused dGELU and dual MXFP4 quantization epilogue (ISA gate passed: vgpr 440, spill 0), mask 0 238.67 µs against `0x8c` 244.15, `0x80` 244.71, `0x48c` 244.13: any relaxation +2.3% (Turbo `gemm_mxfp4_kernel_main.py` dgrad with fused gelu_backward; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20261012_105732/memory.md:157)
- E18 [2026-10, gfx950, FlyDSL 0.3.2, N=4–8, AKLR `bench_cudagraph` (rep=20) on its 8 timing input sets, 4–8 alternating rounds, median; outputs bitwise equal] `sched_barrier(0)` after the up-front slice loads, as speed ratios: on rows with ≥ 16 loads against HEAD, m=80 1.017, m=96 1.006, m=112 1.026; extended to rows with ≥ 12 loads, against the previous build, m=56 1.006, m=60 1.010, m=64 1.010; before the fence the m=112 ISA had about 11 of 24 loads in flight (gemm_n144_k7168 forge run, iter 10; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n144_k7168/forge_experiments/lessons/iter_010.md:11)
- E19 [2026-10, gfx950, FlyDSL 0.3.2, N=2, AKLR `bench_cudagraph` in one process per race, rows interleaved, median; each row run twice, the three slower rows over 12 rounds; MFMA counts from the distill probe's ISA] `sched_barrier(0)` in front of the mid-tile inline-asm sync, before → after in µs: m=9397 161.93 → 155.71, m=16384 273.14 → 270.99, m=2732 66.85 → 65.61, m=893 31.24 → 30.84, m=64 9.038 → 8.885, m=80 9.773 → 9.630, m=96 10.363 → 10.213; slower at m=112 11.359 → 11.631, m=56 8.867 → 8.972, m=60 8.956 → 9.053; at m=9397, 28 of 56 hot-loop MFMAs ahead of the barrier with the fence, 20 without (gemm_n1536_k7168 forge run, iter 12; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n1536_k7168/forge_experiments/lessons/iter_012.md:45; local:/workspace/code/gpt_oss_docker/work/sikl/kb_evidence/sources/k3_distill/gemm_n1536_k7168.distill.md:24)
- E20 [2026-10, gfx950, FlyDSL 0.3.2, N=2, temporary CUDA-graph bench (median of 10 replays of 50 calls, ±0.8 µs), repeated at m=9397 and m=16384; spill counts from the distill probe's ISA] `amdgpu-sched-strategy`: m=9397 (256 VGPRs) `iterative-ilp` 54.581 / 54.504 µs against canonical 55.542 / 55.517, spills 9 → 0; `iterative-minreg` 110.897 µs there (47 spills), and 91.934 / 91.690 at m=16384 against 92.478 / 92.161 (gemm_n2304_k1536 forge run, iter 8; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n2304_k1536/forge_experiments/lessons/iter_008.md:35; local:/workspace/code/gpt_oss_docker/work/sikl/k3_forge/final3_local/distill/gemm_n2304_k1536/distill.md:9)
- E21 [2026-10, gfx950, FlyDSL 0.3.2, N=4–5, scratch paired harness on AKLR's graph timer, 4–5 interleaved rounds per arm, median delta in µs; byte-identical outputs] FRAG_PF=1 (step kk+1's LDS fragments read before step kk's MFMAs) against the previous build: m48 −0.039, m64 −0.027, m80 −0.060, m96 −0.045, m112 −0.009, m893 −0.682, m2732 +0.270 µs, and m112 +0.027 in a later run; FRAG_PF=2 (across the stage barrier) against FRAG_PF=1: m893 +1.310, m112 +0.144 µs; the run records no base times for these rows, and m893 had read 13–14 µs in iter 7 (gemm_n7168_k768 forge run, iter 12; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n7168_k768/forge_experiments/lessons/iter_012.md:16)
- E22 [2026-10, gfx950, FlyDSL 0.3.2, N=1–3, driver bench (warmup 3, 20 iterations), 1–3 runs per figure; spill counts from the distill probe] FPF=1 at m=9397 233.3 µs and at m=16384 177.9 against 55.78 and 94.39 (4.2× and 1.9×), spills 9 → 100 and 0 → 64, on 256-VGPR tiles (gemm_n2304_k1536 forge run, iter 4; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n2304_k1536/forge_experiments/lessons/iter_004.md:21; local:/workspace/code/gpt_oss_docker/work/sikl/k3_forge/final3_local/distill/gemm_n2304_k1536/distill.md:12)
- E23 [2026-10, gfx950, FlyDSL 0.3.2, N=4, AKLR `bench_cudagraph` with L2 flushed, alternating-order rounds; ratio = base ÷ variant, above 1 = faster; untouched buckets read 0.9837–1.0011] the next stage's DMA slots, against half in each of K-steps 0 and 1, over m=8…112: all in K-step 0 0.9486–0.9998; thirds over K-steps 0–2 0.9760–1.0172 (gemm_n18432_k1536 forge run, iter 4; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n18432_k1536/forge_experiments/lessons/iter_004.md:63)
- E24 [2026-10, gfx950, FlyDSL 0.3.2, N=2, driver bench (warmup 3, 20 iterations, FlyDSL cache cleared, one build per process), two alternating runs per arm; the run's ablation: 3 reads] at m=35107 / 104093 / 342073: the per-K-block barrier moved after group 2 of the last k-step, fragments prefetched, DMA slots unchanged: 0.103212 / 0.293706 / 0.953269 ms against the incumbent's 0.098838 / 0.283756 / 0.951373 (+4.4%, +3.5%, +0.2%); the same with the freed stage refilled after the barrier: 0.098449 / 0.278492 / 0.942474 and 0.098475 / 0.278769 / 0.944043 against 0.098906 / 0.284187 / 0.952022 and 0.099041 / 0.284887 / 0.954338; the kept round's ablation +1.30% (+1.26 … +1.34%) (gemm_n3072_k512 forge run, iter 7; 2×4 waves, 2 LDS stages; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n3072_k512/forge_experiments/lessons/iter_007.md:11; local:/workspace/code/gpt_oss_docker/work/sikl/k3_forge/final3_docs/gemm_n3072_k512/ablation.md:18)
- E25 [2026-07, gfx950, FlyDSL ?, N=1, campaign ruler `_bench_tw_vs_mx.py` (geomean of t_mx / t_tw over 14 configurations: forward, dgrad and wgrad of two projections under balanced, moderate or heavy routing), one reading per arm in the session] per-phase lgkm drain of the whole loop: `lgkmcnt(0)` 0.97432, (8) 0.97691, (12) 0.97748, (15) 0.97667 ("no better than 12"), (16) does not assemble; 15 "still bit-exact over 1500 reps" in the round's stress test, as myskill also notes (gpt-oss tensorwise fp8 grouped GEMM, weight-gradient pass (TN, variable K), `_wholeloop_asm_3buf`; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260731_073404/rounds/round-06/agent.txt:8; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260731_073404/memory.md:160; local:/workspace/code/gpt_oss_docker/myskill/optimizer/methodology/06-register-wholeloop-emit.md:305)
- E26 [2026-08, gfx950, FlyDSL ?, N=8, campaign bench, reads per arm as stated] the same kernel family under capacity-collapsed routing: `_WL_ELGK` 15 903.1 µs (8 reads) against 909.3 at 12 (11 reads), kept, and 14 910.2; the kept build bitwise repeatable over 4500 launches (1500 reps × 3 distributions) (gpt-oss-20b down-projection fp8 grouped wgrad, TN, tensorwise; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260826_133925/memory.md:166)
- E27 [2026-07, gfx950, FlyDSL ?, N=1, the campaign ruler of E25, one reading per arm in the session; the harness's gain for the round over three re-reads] Turbo's MXFP4 4-wave emit knobs moved to the tensorwise fp8 wgrad whole loop: the lgkm-side partial drain alone 0.97432 → 0.97748 (+0.32%), with the tail-phase and prologue drains → 0.98027 (+0.61%), and the harness logged +0.72% for the round; myskill's port table gives "+0.8% gm @12"; one `s_nop` after the barrier 0.98016 against 0.98027, "0.00%", where the MXFP4 kernel lists it at +21T (same kernel as E25; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260731_073404/rounds/round-06/agent.txt:5; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260731_073404/memory.md:158; local:/workspace/code/gpt_oss_docker/myskill/optimizer/methodology/06-register-wholeloop-emit.md:292)
- E28 [2026-10, gfx950, FlyDSL ?, N=5, campaign score of the whole chain (GEMM + quantization), five reads against a same-session baseline, null control ±0.12%] `stage_col`'s four `ds_write_b128` moved after `pack_pair`, ahead of the row write-back: score 3291.29 → 3310.01 (+0.57%; reads 3306.1 / 3306.9 / 3310.0 / 3311.7 / 3313.0), fc2_dgrad 230.5 → 228.6 µs, ISA 5315 → 5307 instructions; the next stage's `buffer_load`s in the gap −2.0% (fc2 +10.7 µs), the `ds_write` in the previous MFMA stream −2.6% (fc2 +13.4 µs); the kernel at occupancy 1, vgpr 440 of 472, agpr 256 (Turbo MXFP4 dense GEMM dgrad with a fused gelu_backward and dual MXFP4 quantization epilogue, `gemm_mxfp4_kernel_main.py`; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20261012_105732/rounds/round-13/agent.txt:47)
- E29 [2026-10, gfx950, FlyDSL ?, N=?, decomposition probe `_probe_decomp.py` under the scoring ruler's timing discipline] GELU in the epilogue: +70.2% (+105.2 µs on fc1_double over 149.72 µs for the plain GEMM; +201.8 µs on fc1_single); a standalone GELU 88.84 µs (Turbo A6W4 dense forward GEMM with bias and GELU epilogue, `gemm_a6w4_kernel_tuned.py`; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20261012_073126/goal.md:20)
- E30 [2026-10, gfx950, FlyDSL ?, N=?, kernel-trace steady-state minimum, three-arm subtraction; PMC wave-cycle budget] fc1_double: plain GEMM 148.0 µs, + bias 150.0, + GELU 221.5; GELU 73.5 µs in the epilogue (the `Vec[4]` form) against 66.7 µs for the same GELU in the high-occupancy quantization kernel, 1.10×; 96.9% of the added wave-cycles in `SQ_ACTIVE_INST_ANY` (Turbo A6W4 dense forward GEMM; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20261012_143244/rounds/round-01/agent.txt:7; goal.md:410)
- E31 [2026-10, gfx950, FlyDSL ?, N=1, ISA of one build per arm] tile (128, 512): its first build missed the AGPR accumulator path (the MFMA-form gate was tied to BM 256) and had 613 `v_accvgpr_*` and 18 scratch spills, which myskill records as `scratch_load`s in the hot loop; gated on the accumulator footprint it had no spills and still ran 13.9% slower; tile (256, 384), 256 accumulator dwords per lane placed in AGPRs and 128 in arch VGPRs by inline asm: 500/512 registers, no spill (Turbo A6W4 dense forward GEMM, six training shapes; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20261010_100503/memory.md:178; myskill methodology/04-occupancy-and-tile.md:200)
- E32 [2026-09, gfx950, FlyDSL ?, N=4, in-process palindrome A/B after a 90 s warm-up, minimum of 4 reads, swizzle triple held fixed (ruler as given in the myskill note the round wrote)] `(wlv, elgk)` = (10, 9) / (13, 12) / (16, 15) at K = 4096, where the autotune offered (16, 15) only for K ≥ 8192: FC1_fwd +0.00 / +0.02 / −0.01%, QKV_fwd +0.00 / −0.01 / −0.00% (Turbo MXFP4 dense GEMM `gemm_mxfp4_flydsl_kernel`, Llama training shapes, round 10; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260910_041317/memory.md:250; local:/workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/05-mxfp4-mxfp8-deadends.md:847)
- E33 [2026-10, gfx950, FlyDSL ?, N=1, campaign chain bench (GEMM + quantizer), one run per arm, null control −0.92%, quantizer row phase priced by a three-arm subtraction; added instructions counted in the ISA against a same-structure build; outputs byte-identical to the standalone quantizer] FP6 (E2M3) row pack moved into the GEMM epilogue, amax as a depth-5 tree: +5.237 `v_*` per element, `s_waitcnt` +4, vgpr 508, no spill; fc1_double chain 251.9 → 266.4 µs with the quantizer's row phase −32.1 µs, so +46.6 µs in the epilogue (+106.8 µs on fc1_single), against 24.9 µs at the GELU-calibrated price of 4.75 µs per added instruction per element: 1.87× (2.17× on fc1_single) (Turbo A6W4 dense forward GEMM with bias and GELU epilogue, `gemm_a6w4_kernel_tuned.py`, round 3; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20261012_143244/rounds/round-03/agent.txt:11; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20261012_143244/goal.md:359; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20261012_143244/memory.md:204)
- E34 [2026-10, gfx950, FlyDSL 0.3.2, N=8–10 readings per arm, AKLR graph timer, arms interleaved in one process against the canonical kernel, median; outputs bitwise equal] with the fold removed (wrong output on purpose), two K groups (XCD x runs the splits of group x % 2) 50.26 µs against 52.08 for one group at m=256 (256×192 tiles, split 8, A 256 × 16384 bf16 = 8 MiB); with the fold on, every split summing a share of the tile after the others arrived and its flags in the XCD-local L2 only when a tile's splits shared an XCD (device scope otherwise): 8 groups (one split per XCD) 72.18 against 57.64 µs at m=256, 66.64 / 66.71 against 57.60 / 57.72 in the previous iteration (bf16 GEMM n6144 k16384; local:/shared_nfs/kyle/moe/k3six/runs/gemm_l/workspace/forge_experiments/lessons/iter_017.md:36; :31; :5; local:/shared_nfs/kyle/moe/k3six/runs/gemm_l/workspace/forge_experiments/lessons/iter_016.md:45)
- E36 [2026-10, gfx950, FlyDSL 0.3.2, N=2–4 alternating rounds per arm, each arm in its own process on one GPU, torch.profiler kernel times with a 512 MB flush; the stage-1 strategy runs shared the node with jobs on other GPUs] stage 2 (64×256×128, atomic) default → strategy: `max-memory-clause` 421.6 / 415.6 / 414.8 → 449.7 / 444.9 / 440.1 µs at n_tokens 16384 and 281.3 / 282.5 / 283.9 → 292.4 / 286.5 / 308.6 at 9397; `max-ilp` 429.6 / 420.5 / 426.0 → 433.6 / 429.5 / 424.9 and 281.4 / 274.0 → 286.6 / 276.1 / 279.9 / 281.0; `iterative-minreg` 275.6 / 278.3 / 267.8 → 279.6 / 282.3 / 274.7 at 9397; stage 1 (64×256×256, 108 VGPR + 132 AGPR by default): `max-ilp` 128 + 152 registers, 532–549 / 824 µs, `iterative-ilp` 128 + 152, 535–557 / 831–832, against 464 / 693 for the default in the same rounds (fused MoE SiTUv2, fp8 activations × MXFP4 weights, 112 of 896 experts local, top-16, model dim 3584, inter 3072; local:/shared_nfs/kyle/moe/k3six/runs/ep8cmp/workspace/forge_experiments/lessons/iter_023.md:25; local:/shared_nfs/kyle/moe/k3six/runs/ep8cmp/workspace/forge_experiments/lessons/iter_021.md:31)
