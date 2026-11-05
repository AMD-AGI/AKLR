---
title: "GEMM — price the grid, the tile and each structural idea before building it"
kind: topic
lever: gemm_grid_pricing
ops: [gemm]
gens: [gfx950]
dtypes: [bf16, fp8, mxfp4, mxfp6]
regimes: [small-M, ragged-last-round, large-K, padded-K]
flydsl: 0.3.2
sources:
  - ROCm/FlyDSL@dd8373349:kernels/gemm/gemm_a16w16_gfx950.py:71
  - AMD-AGI/Primus-Turbo@7615a5a2:primus_turbo/flydsl/gemm/gemm_mxfp4_kernel.py
  - AMD-AGI/Primus-Turbo@7615a5a2:primus_turbo/flydsl/gemm/gemm_mxfp8_kernel.py
  - AMD-AGI/Primus-Turbo@7615a5a2:primus_turbo/flydsl/gemm/gemm_fp8_kernel.py
---

# Pricing a GEMM's grid and tile

> Most GEMM wins on gfx950 — and most wasted attempts — are decided by arithmetic you can do before
> writing code: how many workgroups there are, how full the last dispatch round is, what a split
> costs, and what a structural change could return at most. Do that arithmetic first.
>
> Numbers marked `*(field, E<n>)*` are measured values; their conditions are listed under Evidence at the end.
> Use them for direction and order of magnitude, never as a target for your shape.

## Route here when
- You are choosing `tile_m` / `tile_n`, or deciding whether to widen or narrow a tile.
- A small-M or ragged shape sits far from roofline and the loop itself looks fine.
- You are considering split-K, stream-K, a tail/boundary body, or a K-padding change.
- You are about to build a structural change and want its ceiling first.

**Somewhere else if:** the loop structure is the question → `mainloop.md`; wave placement,
XCD order, scheduling hints or the epilogue → `schedule_and_locality.md`; the arguments of a
shipped kernel → the entry points in `INDEX.md`'s kernel map, and aiter's FlyDSL bf16 rows →
`small_m.md`.

FlyDSL's guide starts the tile choice from `preshuffle_gemm`'s constraints and a table by M
(`../flydsl/guides.md` § gemm-optimization (FlyDSL's §1.3–§1.5)), and its checklist asks for enough blocks to fill the GPU (§13).
This card is the arithmetic behind "enough": dispatch rounds, last-round fill, and what split-K or a different
tile can return.

## 1. Count workgroups and dispatch rounds — before any profile

```
tiles      = ceil(M/tile_m) * ceil(N/tile_n) * split_k
slots      = NCU * wg_per_cu                      # NCU = 256 on gfx950; query it
rounds     = ceil(tiles / slots)
last_round = tiles - (rounds - 1) * slots
eff        = tiles / (rounds * slots)             # fraction of CU slots busy over the launch
```

`wg_per_cu` comes from the occupancy arithmetic (gfx950: 512 registers per SIMD in granules of 8,
160 KiB LDS and 32 wave slots per CU; `W` = waves per workgroup; `regs_per_wave` is the arch VGPRs rounded up
to 4 plus the AGPRs, `mainloop.md` §7; LDS is allocated in 1280-byte blocks):

```
waves_per_simd = min(8, floor(512 / round_up(regs_per_wave, 8)))
wg_per_cu      = min(floor(4 * waves_per_simd / W), floor(163840 / round_up(lds_bytes_per_wg, 1280)), floor(32 / W))
```

A register-heavy GEMM is usually 1, which makes `slots = 256`.

How to read it:

| Reading | What it means | Direction |
|---|---|---|
| `tiles < NCU` | every workgroup walks all of K serially; latency is set by K, not M | split K, or shrink the tile |
| `last_round == slots` (`eff ≈ 1`) | no idle slot to recover; intensity is the limit | **widen** the tile |
| `last_round < slots` | CUs idle through the last round | **narrow** the tile (or split) so the last round fills |

Narrowing pays even when it does not remove a round — filling a partial last round is enough.
Widening full rounds pays even at the cost of a pipeline stage. *(field, E1)* On a feed-bound bf16 GEMM, halving the
rounds with taller tiles (3 → 2 stages to fit LDS) gained 7–8%: each CU streams fewer A+B bytes.
*(field, E2, E3)* On related MXFP6/MXFP4 GEMM kernels, widening `tile_n` 256→384 on forward shapes was worth
+14.6% on a six-shape score by ablation; narrowing 256→192 gained on exactly the 6 of 12 backward shapes whose last round was
partial and on none of the others (+16% where the last round filled 144 of 256 CUs, +4–6% where it filled 128).

*(field, E4)* A ragged last round can flip the XCD order: grouping won ~7% with a near-full last round and lost 2–6% at
2% more M (12-tile last round); running those rows as a second small-tile launch kept the grouped head (~4% net).
*(field, E5)* On another kernel that head + tail pair lost ~4% to one launch of shorter tiles filling the last round:
race both.

*(field, E6)* Small-M example: bf16 NT, N=6016, K=7168, tile 128×128, `split_k=1` gives
`ceil(6016/128) = 47` workgroups for every M ≤ 128. The kernel ran ~112 µs flat from M=8 to M=112,
while a split-K library kernel ran ~32 µs at M=8. The grid alone explains it — read the launched grid
from a trace before reading any counter.

The usual rule of thumb, "≥1024 workgroups and a multiple of 8", is not sufficient: 1152 tiles at one
workgroup per CU is 4.5 rounds with the last round half idle.

A narrower tile also costs reuse: LDS/global reads per tile scale with `tile_m + tile_n`, MFMA work with
`tile_m · tile_n`. A geometry change that helps one shape moves others the other way, so evaluate it
across the whole shape set you are scored on.

## 2. Split-K: gate on wave quantization, then price the reduce

**Gate.** Propose a split `s` only if `eff(tiles · s) > eff(tiles) + 0.05`; on ties take the smaller
`s`, because partial-sum bytes and reduce traffic scale with `s`. Keep `s = 1` in the candidate set and
choose by measurement with a margin, so a split can never regress a shape. "Few tiles" is only one case
of this gate — a grid that is full but ragged (1.5 or 3.5 rounds) qualifies too.

**Price.** The split recovers at most the idle part of the launch, `t · (1 − eff)`. The reduce costs
roughly `(s · M · N · partial_bytes + M · N · out_bytes) / BW`. If the reduce costs more than the idle
time it recovers, no split wins. `partial_bytes` can be 3: fp32 rounded to a 16-bit significand, 4 values in
3 dwords (~7% faster at M=893 with split 7, slower where partials were a small share of the bytes
*(field, E7)*), or 2: four 14-bit integers sharing one power-of-two scale, peak error 0.65–0.68 of the contract's
bound against torch's 0.65–0.67, 0.5–2% faster than the 3-byte form at M 256–512 in single readings *(field, E33)*;
gate it on accuracy. M=4096, N=2880, K=32768 on 256×256 tiles (192 tiles, one round,
`eff` = 0.75): with N and K fixed, that grid took 9.8% more time per tile than a 252-tile grid (`eff` = 0.98)
*(field, E32)*, so filling the round would save an estimated ~15 µs, while an `s=4` bf16 reduce costs an
estimated ~42 µs ⇒ `s=1` is correct; the autotuner that picked it was right.

**How Primus-Turbo implements it** (main `7615a5a2`, files under `primus_turbo/flydsl/gemm/`):
- The gate above is Turbo's MXFP4 gate. `_mxfp4_wave_eff = tiles / (ncu · ceil(tiles/ncu))`; no split
  when `eff > 0.95` or `K < 2048`; a split must raise `eff` by more than 0.05; the race gets `s = 1`
  plus the two best splits (`_ksplit_candidates`, `gemm_mxfp4_kernel.py:3489-3511`).
- **Reduce with pairwise `torch.add(out=)` launched after the GEMM**, not inside it (not at small M:
  `small_m.md` §4). It "runs at the pure write floor where
  `sum(dim=0)` reaches half of it, and the reduce is a sixth of a split launch -- the fold's form decides
  whether a split pays at all" (`_fold_mxfp4_partials`, `gemm_mxfp4_kernel.py:3540-3546`).
- **A bandwidth reduce beats an atomic-fused one** here, "since same-address HBM atomics serialise"
  (`gemm_mxfp4_kernel.py:4536-4538`). Atomic split-K is also not bitwise reproducible.
- **MXFP8 splits only the tail round.** `k_split = NCU / tail` when the tail divides the CU count, with
  at least 16 K iterations per slice; head and tail must both be whole rounds (`_mx_ksplit`,
  `gemm_mxfp8_kernel.py:654-672`). *(field, E8)* A uniform split netted out at zero.
- **Dense fp8 TN splits one window.** Only the last partial round's `rem` tiles are sliced s ways
  (s ∈ {2, 3, 4}), choosing the s with the shortest window time (rounds × K/s; the smaller s on a tie) (`_dense_tn_split`,
  `gemm_fp8_kernel.py:1857-1872`). At s=2 the two slices hand off through a flag accessed at system
  scope (`sc0 sc1`) inside the GEMM, so no reduce kernel is launched (`_tn4_handoff_wait`,
  `gemm_fp8_kernel.py:1325`). That hand-off is a wait between workgroups; in your own kernel, fold by arrival
  instead (below).

Refinements, in the order they usually pay:
- **Tail split.** Split only the tiles of the ragged last round. Makespan matches a uniform split, but
  partial bytes cover only the tail. If the whole-wave part and the tail are two launches on one stream
  they run back to back, so **each must be whole rounds by itself** — otherwise the idle round has only
  moved (e.g. 3.5 rounds → 3 + 1 = 4). It lets a wider tile run whole rounds: *(field, E9, E10)* one round of
  256×256 plus a one-round split-K tail beat a near-full two-round launch of 160-row tiles by ~8%.
- **Stream-K** when you want the idle slots without a full reduce: only boundary tiles need a fixup.
- The split must cut K into whole K-blocks the kernel accepts; shipped kernels assert their own
  constraints (for aiter's bf16 HGEMM split-K, `../flydsl/gemm.md` § small_m §4).
- If the scoring harness rejects GPU work from kernels you did not write, the reduce has to be your own
  kernel in the same language, not a framework op.
- **Or reduce in the same launch, without any workgroup waiting for another**: each split publishes its partial and
  counts in on its tile's counter, and the split that arrives last folds every partial in split order, which is
  bitwise reproducible (`small_m.md` §4). Forms that wait — reducer workgroups polling their tile's
  flags, or every split summing a share of the tile once all have arrived — rely on the whole grid being resident
  and can fail on a stream limited to part of the CUs. The arrival fold runs on one CU, so race it against a separate
  reduce kernel per shape.

**The reduce kernel itself.** It has many more workgroups than CUs, so occupancy is not its limit; the
number of waves per workgroup only decides whether the sum crosses waves (barriers and LDS round-trips).
Try one wave per workgroup first. *(field, E14)* An attention decode's combine kernel, written in Triton, went
from 4 waves to 1: `s_barrier` 5→0, time −3.7…−6.7%. One wave stops paying where each output merges many
partials: race wider workgroups there (`../attention/flydsl_attn_decode_split_kv.md` §1). A changed reduction tree
is not bit-identical to the old one — gate it against an independent reference, not old-vs-new max diff.

## 3. Asymmetric operand widths: the best tile is not square

Per K element, a workgroup moves `tile_m · A_row_bytes + tile_n · B_col_bytes` to produce
`tile_m · tile_n` MACs:

```
traffic_per_MAC = A_row_bytes / tile_n + B_col_bytes / tile_m
```

Under an accumulator cap `tile_m · tile_n ≤ C`, this is minimized at

```
tile_m : tile_n = B_col_bytes : A_row_bytes
```

With equal operand widths (bf16×bf16, fp8×fp8) that is square, which is where the square-tile habit comes
from. With MXFP6 A (96 B per 128-K row) and MXFP4 B (64 B per 128-K column) the optimum is
`tile_m : tile_n = 2 : 3`, e.g. 256×384. *(field, E15, E16)* 256×384 beat 256×256 on each of the four shapes it
was tried on: by 3.7%, 6.4% and 6.9% where both tiles ran whole rounds, and by 11.2% where it also turned 4.5
rounds into 3; 128×512 lost 13.9%.

Then check two things before building it:
- **Divisibility.** `N % tile_n == 0` on your shapes. Non-power-of-two widths (192, 384) often divide model
  dimensions of the form `3·2^k` better than 512 does.
- **Accumulator placement.** A wide tile can exceed the 256-dword AGPR file per lane; see
  `schedule_and_locality.md` §7.

*(field, E17)* On two shapes run at `tile_m = 128`, widening only `tile_n` (128×256 → 128×384, which the traffic
formula ranks better) made them 4–9% slower; *(field, E18)* the ratio's own tile, 256×384, cut the same two
shapes' time by 16–19% (one other setting changed with it). Test the tile the ratio gives, not the old `tile_m`
with a new `tile_n`. The ratio tells you where to look, not what will win.

## 4. Occupancy moves in workgroup-sized steps

The register tier gives waves per SIMD (`waves_per_simd` in §1). A workgroup's waves
must all be resident on one CU, so the reachable occupancy is quantized by workgroup size:

```
waves_per_cu = floor(4 * waves_per_simd / W) * W    # W = waves per workgroup
```

With an 8-wave workgroup only 2/4/6/8 waves per SIMD are reachable: cutting from 128 to 96 registers
(per-SIMD tier 4 → 5) buys nothing. Align a register-reduction target to a reachable tier before chasing
it.

The compiler's `; Occupancy:` comment in the ISA does **not** apply this quantization. Example:
`llc -mcpu=gfx950` reports a kernel using 96 VGPRs with `amdgpu-flat-work-group-size=512,512` as
occupancy 5; the CU can hold only two such workgroups, i.e. 4 waves per SIMD.

## 5. K granularity: the MFMA decides what a K-tail can save

The MFMA's K is a hard granule. A tail body that skips K padding saves instructions only if

```
ceil(K_real / inst_k) < ceil(K_pad / inst_k)
```

`K_real = 2880`, `K_pad = 2944`, `inst_k = 128` gives 23 and 23 — the tail body saves no MFMA, only the
last iteration's feed bytes. Confirm the granule from counters: `SQ_INSTS_MFMA` should equal
`M·N·K / (m·n·k of the instruction)`.

A bigger MFMA shape is not a throughput lever. *(field, E19)* On gfx950 with fp8 A and B, measured on a grouped
GEMM, `v_mfma_scale_f32_32x32x64_f8f6f4` occupies the matrix pipe exactly twice as long as
`v_mfma_scale_f32_16x16x128_f8f6f4` (`SQ_VALU_MFMA_BUSY_CYCLES / SQ_INSTS_MFMA` = 64 vs 32), so pipe time per
FLOP is unchanged; it needs 4× the accumulator registers per instruction and leaves fewer independent
accumulation chains to hide dependencies. Swapping only the main-loop MFMA of that grouped GEMM was 0.8–3.0%
slower on 12 of 12 shapes. Use the larger shape to fix K granularity, not to go faster.

Whether an instruction exists on the target is a five-second check:
`echo '<instruction>' | llvm-mc -arch=amdgcn -mcpu=gfx950 -show-encoding`.

## 6. Fit the per-tile fixed cost before polishing the loop

At fixed M and N, sweep K and fit

```
t_tile = a · n_kblocks + F
```

`F` — prologue, epilogue, tail peel, address and descriptor setup, drain — is paid once per tile, and its
share `F / (a · n_kblocks + F)` grows as K shrinks. Across a shape table it usually explains the TF/s
ranking better than any other variable. *(field, E20, E21)* `F` = 2.21 K-blocks of 256: an estimated 16% of tile
time at K=2944 (11.5 blocks), 1.7% at K=32768, and the table's TF/s order was essentially the inverse of that
share. By the fit, removing one K-block of `F` is worth ~4–5% over that table — a projection, not a measurement,
but already larger than the 1–2% that swizzle and candidate-table changes returned *(field, E22)*.

Kernel-argument preload (`small_m.md` §6) trims `F` at any M: *(field, E23)* −0.2 to −0.6% at M ≈ 900–16k, not only at decode M.

Two traps when reading such a table:
- **Compare absolute µs, not TF/s, across padded shapes.** TF/s counts useful FLOPs, so padding waste
  shows up as "efficiency" instead of as work done.
- **"Divisible ⇒ fast" is not a rule.** *(field, E24)* K=2816 (11 K-blocks) ran slower than K=2944 padded to 12
  blocks, because an odd block count fell off an unroll-2 pipeline's store-folding path. Before adding a
  shape, evaluate the kernel's own gates (parity of block counts, tail predicates) for it.

## 7. Bound the gain before building: the illegal arm

For any proposal that shortens a wait, removes a transfer or skips work, first build an arm that removes
it entirely — the results are wrong, the time is the ceiling. Prove the arm really compiled in with an
output checksum (an absurd sum shows it did); a "change" that never reached the binary reads as noise.
If the ceiling is below what you need, skip the idea.
- *(field, E25)* Removing the whole first prologue wait bounded the gain at 0.72%, so a half-wait reorganization
  was worth at most an estimated ~0.35% and was not built.

## 8. Boundary variants are not free

Each extra body variant (a quarter-N or quarter-M tail, a half-K body) adds code even when it never runs.
*(field, E26)* On a grouped GEMM, one variant added 739 instructions (+39.5% of the module) and cost 0.32 points
averaged over the four cells measured, all compiled into that module, with VGPR, SGPR and LDS identical across arms — code size alone.

```
net = hit_rate × gain_per_tile × shapes_that_trigger − carry_cost × shapes_in_module
```

Measure the hit rate on the scored input distribution first: a runtime ragged tail can be 0% on balanced
inputs. When a variant does pay, prefer re-assigning waves over computing less — a ragged tile wastes
whole waves pinned to an empty region, and re-tiling the waves (e.g. 2×2 → 4×1) serves the N tail, while
the same body rotated 90° serves the M tail. Write the gate as geometry (`m_tail <= rows_per_wave`), not
as shape constants.

## 9. Pricing an alternative tile shape

A second tile shape trades arithmetic intensity for CU fill. Price both sides before adding it to the race.
- **Operand bytes per output cell are `(bm + bn) / (bm · bn)`.** Turbo's dense fp8 whole loop lets its
  narrower 256×192 tile join the race only when its gain in resident workgroups exceeds its loss in
  intensity: `coverage(rect) / coverage(square) > intensity(rect) / intensity(square)` (`_nt4_geoms`,
  `gemm_fp8_kernel.py:2102-2119`).
- **A narrow tile wins only where its extra tiles still fit in the same number of dispatch rounds**, so
  each round is shorter. Turbo's MXFP4 picks BLOCK_N=192 over 256 only when the 192 grid needs no more
  rounds and its priced time is lower: a k-block at the pessimistic feed ratio `(BM + BN) / (2·BM)`, plus
  the C store the narrow tile can no longer fold (`_mxfp4_pick_block_n`, `gemm_mxfp4_kernel.py:3426-3445`).
  The store penalty was solved from one measured shape; re-derive it for your kernel.
- **Dead quadrants still issue their MFMAs.** The descriptor bound drops only their loads and stores. An
  exact tile (e.g. 192 rows when M divides by 192) therefore shortens every workgroup's own path. Turbo
  adds exact tiles only when they fit in one CU pass and cut the per-workgroup MFMAs per K block
  (`gemm_fp8_kernel.py:2066-2070, 2134-2144`).
- **Work freed on only some workgroups returns less than it looks.** On a power-capped part, idle CUs
  come back as clock for the others, not as wall time (`_nt4_wg_path` docstring, `gemm_fp8_kernel.py:2134`).
- **The race decides, not the plan.** An analytic ranking only orders the builds.

## 10. Persistent tiles per workgroup

- With `tiles_per_wg = ceil(tiles / ncu)`, a tile count between two whole CU passes rounds the resident
  workgroup count down, and the rest of the device never gets a wave. "This, not the tile count, is the
  signal" (`_nt4_resident`, `gemm_fp8_kernel.py:2089-2095`).
- Leftover work ids may repeat the last tile. That is harmless for an overwriting store and wrong for
  beta=1, which needs one tile per workgroup (`epilogue.md` §7).
- Turbo's MXFP4 kernel walks up to 4 tiles per workgroup, and only when `tiles % (ncu · tpw) == 0` and
  K/256 ≤ 56; at longer K the per-tile launch is already amortized (`_MXFP4_TPW_*`,
  `gemm_mxfp4_kernel.py:3467-3485`). Stride the persistent loop by the grid
  (`schedule_and_locality.md` §2).
- **Passing that gate is not a win: race it.** *(field, E27)* A small-M bf16 GEMM at K=256 with 2560 tiles lost
  5.5–12.5% at 2 tiles per workgroup under every store placement (and 1–5% as a two-tile walk of another variant
  of the kernel), though a stores-off probe had priced an estimated ~7% gain.
  *(field, E28)* At K=512 a 1-per-CU grid walking 6–60 tiles each, whose last two K-blocks fill the next tile's stages
  and with no workgroup barrier between tiles (C staged per wave), beat one tile per workgroup by 3–5%.
- **Race a start delay on half of each XCD's workgroups** (`rocdl.s_sleep` before the first fill when
  `pid // 8` is odd; a few thousand clocks). *(field, E29, E30)* +1–3% at 6–20 tiles per workgroup, null at ~60.
- **Keeping B resident in LDS does not pay when B already sits in L2.** *(field, E31)* At K=128 with a sub-MB B, a
  persistent kernel holding each workgroup's B slice gained ~1% at M=16384 and lost 10–22% at M=9397.

## Verify
- The tile/round table for every scored shape exists before the first build.
- A split or tile change is judged per shape, with `s = 1` / the old tile still in the race.
- A structural change has an illegal-arm ceiling, and the arm is proven live by a checksum.
- The win does not come from one shape while its neighbours regress.

## Related
`workflow.md` (order of operations) · `schedule_and_locality.md` (wave
placement, XCD order, scheduling) · `autotune.md` (racing the candidates) ·
`small_m.md` (split-K and tiles at decode sizes)

## Evidence

A FlyDSL version is given only where the cited run, or another round in the same container and venv, printed it
(`flydsl.__version__`, an import path with its version, or a `pip install flydsl==` in the run's own setup); otherwise
`FlyDSL ?`. The Kimi-K3 forge runs installed `flydsl==0.3.2`. The month is the run's; where the record is a note written
later, the line says the date it gives. "Turbo" is Primus-Turbo; a "K3 forge run" is a Kimi-K3 forge-loop campaign on
AKLR's bf16 GEMM schema of that N and K.

- E1 [2026-10, gfx950, FlyDSL 0.3.2, N=3–5, own A/B script: CUDA events, L2 flushed before each call, kernels alternated, median over 3–5 repeats of the 30-call median] taller tiles with 2 stages (BK 64) against the port's 3-stage schedule: m=9397 as one round of 416×192 tiles 214.80 against 233.88 µs (−8.2%; 2 rounds → 1), m=16384 as two rounds (one 384-row and one 320-row tile per workgroup) 374.56 against 403.63 µs (−7.2%; 4 rounds → 2; round counts from the run's distill); with the compute removed m=9397 still took 186.4 against 213.4 µs (feed-bound) (gemm_n2112_k7168 forge run, iters 10–12; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n2112_k7168/forge_experiments/lessons/iter_011.md:31)
- E2 [2026-10, gfx950, FlyDSL ?, N=?, closing ablation after the campaign, ruler not recorded] pinning `tile_n` back from 384 to 256 in the delivered A6W4 forward kernel cost 14.58% (`tile_n` 384 worth +14.58% of the campaign's +16.94%); no run record of the ablation was found (Turbo A6W4 (MXFP6 A, MXFP4 B) dense forward GEMM, six training shapes, campaign 20261010_100503, ablation dated 2026-10-11; myskill ddc8f2d; local:/workspace/code/gpt_oss_docker/myskill/optimizer/methodology/04-occupancy-and-tile.md:610)
- E3 [2026-10, gfx950, FlyDSL ?, N=?, closing ablation after the campaign, ruler not recorded] narrowing the tile 256 → 192 in the delivered MXFP4 backward kernel: the 6 of 12 shapes whose last round was partial gained and the other 6 moved −0.7% to +0.5%; out_proj_wgrad (last round 144 of 256 CUs → 256) +15.97%, the three with 128 of 256 +6.42%, +6.13% and +3.96%, the two with 64 of 256 +8.29% and +6.91%; no run record of the ablation was found (Turbo MXFP4 dense backward GEMM, 12 training shapes, campaign 20261010_100506, ablation dated 2026-10-11; myskill ddc8f2d; local:/workspace/code/gpt_oss_docker/myskill/optimizer/methodology/04-occupancy-and-tile.md:592)
- E4 [2026-10, gfx950, FlyDSL 0.3.2, N=?, interleaved in-process bench_cudagraph arms, medians; the two-launch result by driver bench] XCD grouping on 256×256×64 tiles (28 tile columns): m=9216 (1008 tiles, last round 240, computed) 168.1 µs ungrouped against 156.0 µs with group 3 (−7.2%); m=9397 (1036 tiles, last round 12) 173.4–173.6 µs ungrouped against 177.4–183.9 µs for every group size tried (+2.2% to +6.1%); 9216 rows grouped plus the remaining 181 rows as a second launch of 96×64 tiles: driver bench 172.95 → 166.38 µs (−3.8%) (gemm_n7168_k1536 forge run, iter 10; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n7168_k1536/forge_experiments/lessons/iter_010.md:33)
- E5 [2026-10, gfx950, FlyDSL 0.3.2, N=1, driver --bench-mode --bench-case, warmup 3, 10 iterations] m=9397: one launch of 224×256 tiles (1008 tiles, computed) at group height 4 0.6198 ms; 256×256 for the first 8192 rows, then 128×256 or 160×256 tiles in a second launch, 0.6421 or 0.6438 ms (+3.6%, +3.9%) (gemm_n6144_k7168 forge run, iter 4; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n6144_k7168/forge_experiments/lessons/iter_004.md:11)
- E6 [2026-10, gfx950, FlyDSL ?, N=5, cold L2 (512 MB zeroed before each call), torch profiler kernel device time, median of 5] bf16 NT, N=6016, K=7168: the forge-delivered FlyDSL kernel `23b8990` (128×128 tile, split-K 1, 47 workgroups at M ≤ 128) 112 µs at m=8 and 114 µs at m=112; aiter's baseline (hipBLASLt with split-K 16) 32 µs at m=8; a trace confirmed the grid. The note is dated 2026-10-16; its run log is not kept on this machine (Kimi-K3 gemm_a16w16_nt_n6016_k7168; local:/workspace/code/gpt_oss_docker/.claude/memory/reference_bf16_gemm_impl_bench_k3_1016.md:21)
- E7 [2026-10, gfx950, FlyDSL 0.3.2, N=5, scratch harness on AKLR bench_cudagraph, cold L2, median of 5 readings, interleaved against canonical] partials rounded to a 16-bit significand and packed 4 per 3 dwords: m=893 at split 7, 29.2 → 27.0 µs (−7.5%); the single-launch small-M rows, whose partials are few bytes, slower (m=16 7.49 → 7.63, m=24 7.9 → 8.01, m=32 8.01 → 8.15 µs); driver allclose with max_diff 0 (gemm_n1280_k7168 forge run, iter 8; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n1280_k7168/forge_experiments/lessons/iter_008.md:34)
- E8 [2026-09, gfx950, FlyDSL ?, N=?, ?] Turbo's commit message on the MXFP8 dense GEMM's tail-round split-K: splitting only the tail tiles refills the last round at a partial-sum cost proportional to the tail, "which is why a uniform split netted out at zero"; shapes and timer not stated (Turbo MXFP8 dense GEMM; Primus-Turbo d522ee0a (#505))
- E9 [2026-10, gfx950, FlyDSL 0.3.2, N=?, driver graph-timer medians, warmup 3, 20 iterations; the incumbent's time is from the previous iteration, not re-measured] m=2732: a 256×256 head of one round (252 tiles, the first 2304 rows) plus the last 428 rows as a one-round split-K tail (2 × 32 tiles × split 4 = 256 workgroups, computed) 1.0436 and 1.0454 ms (1.0465 in the full bench) against 1.129–1.133 ms for 160×256 tiles at GROUP_M 9 (504 tiles, 1.97 rounds, computed) (−7.3% to −7.9%) (gemm_n7168_k35840 forge run, iters 3–4; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n7168_k35840/forge_experiments/lessons/iter_004.md:11)
- E10 [2026-10, gfx950, FlyDSL 0.3.2, N=3, forge-loop ablation ladder, 3 pairs, driver --bench-mode] the head + tail round (06f6ca7f) against its parent: +8.27% at m=2732, +0.56% [+0.54, +0.58] over all cases (gemm_n7168_k35840 forge run; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n7168_k35840/forge_experiments/ablation.json:195)
- E14 [2026-09, gfx950, Triton (sglang `triton_sparse_mla.py`), N=7, isolated paired race: real launcher, HIP graph of 32 calls, min of 7, palindromic order, inert control arm spread ≤ 0.7%] `_sparse_mla_reduce_kernel` `num_warps` 4 → 1: −3.8 / −3.7 / −6.7 / −4.5% at seq 1/2/6/12; ISA: one wave 47 VGPRs and 0 `s_barrier`, two and four waves 27–30 VGPRs and 5 `s_barrier`; e2e +0.318% (floor 0.134%) (GLM-5.2 TP8/EP1 sparse-MLA attention decode, its split-K combine kernel; sglang campaign 20260925_134931 round 4; local:/workspace/code/gpt_oss_docker/sync/sglang_optimizer/sglang_campaigns/20260925_134931/rounds/round-04/agent.txt:15)
- E15 [2026-10, gfx950, FlyDSL ?, N=?, the round's per-shape timing (timer not stated in the round record); checksums unchanged] linear1 (16384×21504×3072): 256×256 568.9 µs, 256×384 547.8 µs (−3.7%), 128×512 647.9 µs (+13.9%) (Turbo A6W4 (MXFP6 A, MXFP4 B) dense forward GEMM; campaign 20261010_100503 round 6; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20261010_100503/rounds/round-06/agent.txt:11)
- E16 [2026-10, gfx950, FlyDSL ?, N=3, campaign score: training-time-weighted harmonic mean TF/s of six shapes, three reads per arm] 256×384 on the four tile_m=256 shapes (linear1 in E15; linear2 334.0 → 312.5 µs (−6.4%), fc1 160.9 → 149.8 µs (−6.9%), qkv 125.4 → 111.3 µs (−11.2%; 1152 workgroups in 4.5 rounds → 768 in 3, the round's arithmetic); fc2 and out_proj unchanged); on the six-shape score: harness 3952.65 → 4159.02 (+5.22%); the round's own four reads give 4163.9 (+5.3%); the round report writes +5.4%, as myskill does (Turbo A6W4 dense forward GEMM; campaign 20261010_100503 round 6; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20261010_100503/memory.md:187)
- E17 [2026-10, gfx950, FlyDSL ?, N=?, the round's per-shape timing] on the two tile_m=128 shapes, 128×384 against 128×256 (traffic per MAC 0.75 against 0.875) with the accumulators in AGPR assembly: fc2 (8192×3072×12288) +4.3%, out_proj (8192×3072×3072) +9.2% (the round's table gives 154.6 → 160.7 and 46.8 → 51.0 µs, +3.9% and +9.0%) (Turbo A6W4 dense forward GEMM; campaign 20261010_100503 round 6; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20261010_100503/rounds/round-06/agent.txt:15)
- E18 [2026-10, gfx950, FlyDSL ?, N=?, campaign score as in E16 and the round's per-shape timing; `fi` 0 → 2 changed in the same step] fc2 and out_proj moved from 128×256 to the ratio's 256×384 (256 tiles, one round): fc2 154.1 → 125.1 µs (−18.8%), out_proj 47.0 → 39.3 µs (−16.4%), score 4195.89 → 4324.73 (+3.07%), checksums unchanged (Turbo A6W4 dense forward GEMM; campaign 20261010_100503 round 7; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20261010_100503/memory.md:194)
- E19 [2026-08, gfx950, FlyDSL ?, N=?, rocprofv3 PMC; campaign bench `_bench_campaign_mx8tw.py` per cell] main-loop MFMA swapped from `v_mfma_scale_f32_16x16x128_f8f6f4` to `v_mfma_scale_f32_32x32x64_f8f6f4` (MFMA count 1104 → 552, the rest of the ISA byte-identical, VGPR 246 → 244, no spill): SQ_VALU_MFMA_BUSY_CYCLES / SQ_INSTS_MFMA 32.0 and 64.0, aggregate busy cycles 1.10939e9 on both; all 12 NT cells 0.8–3.0% slower, the 6 wgrad cells unchanged (gpt-oss MXFP8 grouped GEMM, NT kernel; campaign 20260804_041346 round 14; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260804_041346/rounds/round-14/agent.txt:5)
- E20 [2026-09, gfx950, FlyDSL 0.2.4, N=?, gemm18 scorer, per-cell TF/s] F solved from four pairs of cells that differ only in K, all at wave efficiency 1.0: 1.85 / 2.72 / 1.95 / 2.33 → F = 2.21 K-blocks of 256 per tile; the round converts it to 16.1% of tile time at K=2944, 12.1% at K=4096 and 1.7% at K=32768 (Turbo MXFP4 dense GEMM, 256×256×256 tiles, 6 gpt-oss and 12 Llama training shapes; campaign 20260925_144843 round 6; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260925_144843/rounds/round-06/agent.txt:13)
- E21 [2026-09, gfx950, FlyDSL 0.2.4, N=?, gemm18 scorer, per-cell TF/s] the 18 cells' TF/s order (4440 → 5662) was essentially the inverse of F's share; 32768×4096×2944 4448.5 against 32768×4096×4096 5059.3 TF/s (−12.1%); this round writes F's share at K=2944 as 19.2% (F / 11.5 blocks) where round 6 wrote 16.1% (F / (11.5 + F)) (same kernel; campaign 20260925_144843 round 7; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260925_144843/memory.md:221)
- E22 [2026-09, gfx950, FlyDSL 0.2.4, N=?, ?] the round's summary puts swizzle and candidate-table changes at 1–2%; no run is named for that figure (same kernel; campaign 20260925_144843 round 7; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260925_144843/memory.md:221)
- E23 [2026-10, gfx950, FlyDSL 0.3.2, N=3, bench_cudagraph in the same process as the incumbent, median of 3] kernel-argument preload count 16: m=893 74.21 → 73.92, m=2732 184.03 → 183.64, m=9397 625.45 → 621.71, m=16384 1056.47 → 1053.95 µs (−0.2% to −0.6%) (gemm_n6016_k7168 forge run, iter 5; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n6016_k7168/forge_experiments/lessons/iter_005.md:45)
- E24 [2026-09, gfx950, FlyDSL ?, N=?, timer not stated] K scan at M=32768, N=4096: K=2816 (11 K-blocks of 256) 190.9 µs, K=2944 (11.5 blocks, padded to 12) 178.2 µs, K=3072 185.4 µs; by the kernel's gate `(ki >= 4) and (ki % 2 == 0 or half_k)` the folded C store is off at ki = 11 (Turbo MXFP4 dense GEMM, 256×256×256 tiles; campaign 20260923_074233 goal; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260923_074233/goal.md:57; myskill methodology/04:292, :308)
- E25 [2026-09, gfx950, FlyDSL 0.2.4, N=1, the round's probe script, one run per arm, per-arm JIT cache directories, absum to prove the arm is live] no wait on the k=0 prefill (wrong output, absum 8.67e35 against 1.06e8): 214.00 → 212.47 µs (−0.72%) (Llama out_proj_dgrad shape; Turbo MXFP4 dense GEMM; campaign 20260925_144843 round 7; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260925_144843/memory.md:194)
- E26 [2026-09, gfx950, FlyDSL 0.2.4, N=5, campaign score: frozen-denominator geomean over 8 cells, each cell drift-corrected against 4 inert wgrad cells] a runtime M-tail body compiled into the NT module: module 1870 → 2609 instructions (+739, +39.5%), VGPR 480 / SGPR 96 / LDS 163840 identical across arms; −0.31 and −0.34 points on the two cells where it does not fire (B−A), −0.32 mean over 4 NT cells with the body compiled but its predicate false (C−A); gm −0.134 ± 0.045 (gpt-oss plain MXFP4 grouped MoE GEMMs fc2_fwd and fc1_dgrad; campaign 20260925_090609 round 8; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260925_090609/rounds/round-08/agent.txt:17)
- E27 [2026-10, gfx950, FlyDSL 0.3.2, N=4–6, AKLR graph timer, arms interleaved in one process, 4–6 reps each, medians] persistent 2-tile grid (1280 workgroups) by store placement tail / eager / after_issue: m=14 16.69 / 17.50 / 16.46 µs and m=12 16.42 / 16.54 / 15.73 µs against 15.56 and 14.91 µs with one tile per workgroup (+5.5% to +12.5%); a two-tile walk of the skinny kernel (2560 workgroups): m=8 14.97–15.16 against 14.44 µs, m=12 15.10–15.19 against 14.91, m=14 15.71–15.78 against 15.57 (+0.9% to +5.0%) (gemm_n163840_k256 forge run, iter 11; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n163840_k256/forge_experiments/lessons/iter_011.md:85)
- E28 [2026-10, gfx950, FlyDSL 0.3.2, N=1, own CUDA-graph harness, all arms in one process, 10 launches per graph, median of 10 replays] the same 256×256 body as a persistent 256-workgroup grid against one tile per workgroup: m=35107 0.1143 against 0.1200, m=104093 0.3019 against 0.3183, m=342073 0.9738 against 1.0022 ms (−2.8% to −5.2%); 1656 / 4884 / 16044 tiles, about 6.5 / 19 / 63 per workgroup (computed) (gemm_n3072_k512 forge run, iter 2; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n3072_k512/forge_experiments/lessons/iter_002.md:22)
- E29 [2026-10, gfx950, FlyDSL 0.3.2, N=3, forge-loop ablation ladder, 3 pairs, driver --bench-mode] the start-delay round (27d6d9c4; the delay is applied at m ≤ 200k only) against its parent: m=35107 +2.92%, m=104093 +1.06%, m=342073 +0.02%; +1.33% [+1.17, +1.49] over the three (gemm_n3072_k512 forge run; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n3072_k512/forge_experiments/ablation.json:133)
- E30 [2026-10, gfx950, FlyDSL 0.3.2, N=?, own CUDA-graph harness, all arms in one process, 10 launches per graph, median of 10 replays; repeated incumbent runs agreed to about ±1%] start-delay arms: m=342073 every arm 0.9690–0.9841 ms against bases of 0.9743–0.9786; m=104093 0.2789–0.2821 against 0.2837–0.2902 ms; m=35107 0.0960–0.0973 against 0.1012–0.1031 ms (gemm_n3072_k512 forge run, iter 4; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n3072_k512/forge_experiments/lessons/iter_004.md:23)
- E31 [2026-10, gfx950, FlyDSL 0.3.2, N=2, AKLR graph timer, best of 2 medians of 3, same process] persistent kernel keeping each workgroup's B slice in LDS against the tiled kernel: m=16384 15.02 µs (repeats 15.04, 15.08) against 15.21 (−1.2%); m=9397 11.44–12.70 against 10.44 µs (+9.6% to +21.6%); B is 1536 × 128 bf16 = 384 KB (computed) (gemm_n1536_k128 forge run, iter 2; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n1536_k128/forge_experiments/lessons/iter_002.md:23)
- E32 [2026-09, gfx950, FlyDSL ?, N=5, the round's probe script: CUDA events around 20 back-to-back calls, best of 5 reps after a warm-up loop, no L2 flush] N=2880, K=32768 fixed, M scanned, unsplit at every M (the kernel's own split-K choice): time per tile × 256 = 266.9 µs at 132 tiles (`eff` 0.52), 240.2 at 156, 217.1 at 192 (`eff` 0.75; M=4096, 162.86 µs), 210.7 at 228, 199.9 at 240 (`eff` 0.94), 197.8 at 252 (`eff` 0.98), 311.5 at 264 (two rounds), 289.0 at 288; 217.1 against 197.8 = +9.8%, 199.9 against 197.8 = +1.1% (Turbo MXFP4 dense GEMM, 256×256 tiles, gpt-oss out_proj wgrad shape at M=4096; campaign 20260925_083359 round 1; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260925_083359/rounds/round-01/deep_raw.jsonl:5906)
- E33 [2026-10, gfx950, FlyDSL 0.3.2, N=1 per arm, driver on AKLR's graph timer and AKLR's inputs, and an in-process probe on randn inputs with an L2 flush before each call] split-K partials as four 14-bit integers on one shared power-of-two scale (2 dwords per 4 values) against 4 values per 3 dwords, on a fold in which every split summed a share of the tile: driver m=512 92.97 → 91.44 µs, m=256 59.43 → 59.15; probe 99.98 → 98.19 and 65.07 → 63.76; peak error 0.656 → 0.659 (m=512) and 0.646 → 0.679 (m=256) of the contract's bound; on AKLR's inputs the kept kernel read 0.649 / 0.680 against torch's 0.649 / 0.666 (bf16 GEMM n6144 k16384, 256×192 and 256×96 tiles, split 4 and 8; local:/shared_nfs/kyle/moe/k3six/runs/gemm_l/workspace/forge_experiments/lessons/iter_008.md:21; local:/shared_nfs/kyle/moe/k3six/runs/gemm_l/workspace/forge_experiments/lessons/iter_021.md:14)
