---
title: "fp8 block-scaled GEMM with a row-major a_scale, shuffled B — read the scales in the caller's layout; the margin at m ≥ 2 includes the baseline's relayout"
kind: case
ops: [gemm]
dtypes: [fp8]
regimes: ["M 1-64", "M 128-512", "M 1024-4096"]
contract: bounded
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=1
measured:
  - {metric: "speedup over aiter, CI, per Point", value: "1.486–5.571; 13 of 13 Points ≥ 1.03", gpu: MI355X, workload: "gemm_a8w8_blockwise_scaled_b_shuffled_row_major_a_scale_n512_k4096, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", date: "2026-10-28"}
  - {metric: "speedup over aiter, CI, per Point", value: "1.375–3.298; 13 of 13 Points ≥ 1.03", gpu: MI355X, workload: "gemm_a8w8_blockwise_scaled_b_shuffled_row_major_a_scale_n1024_k4096, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", date: "2026-10-28"}
  - {metric: "speedup over aiter, CI, per Point", value: "1.025–2.611; 12 of 13 Points ≥ 1.03", gpu: MI355X, workload: "gemm_a8w8_blockwise_scaled_b_shuffled_row_major_a_scale_n4096_k256, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", date: "2026-10-28"}
  - {metric: "speedup over aiter, CI, per Point", value: "0.993–2.266; 12 of 13 Points ≥ 1.03", gpu: MI355X, workload: "gemm_a8w8_blockwise_scaled_b_shuffled_row_major_a_scale_n4096_k512, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", date: "2026-10-28"}
sources:
  - aklr/ops/gemm/impls/flydsl/gemm_a8w8_blockwise_scaled_b_shuffled_row_major_a_scale_n512_k4096/e5cda3e1b4f63183.json
  - aklr/ops/gemm/impls/flydsl/gemm_a8w8_blockwise_scaled_b_shuffled_row_major_a_scale_n1024_k4096/7c4c895c7743b287.json
  - aklr/ops/gemm/impls/flydsl/gemm_a8w8_blockwise_scaled_b_shuffled_row_major_a_scale_n4096_k256/0660d74d64e95f05.json
  - aklr/ops/gemm/impls/flydsl/gemm_a8w8_blockwise_scaled_b_shuffled_row_major_a_scale_n4096_k512/b52039c93b6d86ac.json
---

# fp8 block-scaled GEMM with a row-major a_scale and shuffled B

## Applies when

- Contract `gemm_a8w8_blockwise_scaled@1.0`, layout variant `b_shuffled_row_major_a_scale` (`operand_format`
  `none:preshuffled`, `a_scale.is_row_major` true): b in aiter's 16×16 `shuffle_weight` order, and the a-scales
  row-major, `S_a[i, kb] = a_scale_flat[i·SK + kb]` with SK = K / 128, so one row's scales are contiguous. a and b
  fp8 e4m3fn, one fp32 b-scale per 128×128 of b, `trans_b`, `scale_major_mode` K, no bias, bf16 out.
- (N, K) = (512, 4096), (1024, 4096), (4096, 256) and (4096, 512), each its own schema serving m = 1, 2, 4, …,
  4096. The same shapes in the column-major variant are in [a8w8_small_n_large_k.md](../cases/a8w8_small_n_large_k.md) and
  [a8w8_n4096_k_sweep.md](../cases/a8w8_n4096_k_sweep.md).
- Compare: each schema's `compare` block is `determinism: bounded`, `max_violation_fraction: 0.0`, against the
  contract's tolerance (atol = rtol = 1e-2 for bf16 out; `aklr/ops/gemm/contract.py`). The reductions in these four
  kernels run in a fixed order; run-to-run bit equality is not measured.

## Counter-signals

- **At m ≥ 2 the measured margin includes a copy in the baseline.** For this layout variant the bench times aiter
  with `x_scale.T.contiguous().T` passed in its place, because aiter's preshuffled-B GEMM reads a row-major `x_scale`
  as column-major memory: a copy kernel inside each timed baseline call. At m = 1 the (1, SK)
  a_scale is contiguous in both orders, `.contiguous()` returns it unchanged, and no copy runs. With the same kernel
  configuration at m = 1 and m = 2 in each file, the four files read 4.569, 2.285, 1.025 and 0.993 at m = 1 against
  5.571, 3.220, 1.942 and 1.922 at m = 2 *(field, E1, E2, E3, E4)*; the column-major files of the same shapes
  show no such step (m = 1 / m = 2: 4.638 / 4.504, 2.383 / 2.400, 1.039 / 1.035, 0.986 / 0.996) *(field, E5)*. With
  aiter dbd8bf5 the copy adds about 4.1 µs to a 4.4 µs aiter call at N 4096, K 256, m = 2 and 16 *(field, E7)*, so
  n4096_k256's 1.942 at m = 2 would be about 1.01 without it (estimated); at the other three shapes the copy's share
  is not measured.
- **N 4096 at m = 1 is not served:** n4096_k256 1.025 and n4096_k512 0.993 *(field, E3, E4)*. Across the five
  measurements committed in the predecessor repository, n4096_k256's m = 1 reads 1.006–1.033, on either side of
  1.03, and n4096_k512's 0.989–1.009, below it in all five *(field, E6)*.
  n4096_k512's m = 1 Point runs its register body, four 16×16×32 fp8 MFMAs per K
  block on a 16×16 tile.
- **Cross-workgroup split-K counters live between calls.** n512_k4096 (m ≤ 256) and n1024_k4096 (m = 256) keep
  their arrival counters in memory the builder zeroes once per shape; the last arriver resets them. Both sum the
  splits in split order, not arrival order. No AKLR call may depend on what an earlier call left (`../SKILL.md`,
  Stop table), and two streams calling the same Point at once would share the counters.

## Mechanism

What the four kernels do with the row-major scales: a row's K-block scales are adjacent, so a single 16-byte load
fetches four of them. n512_k4096 and n1024_k4096 (m ≥ 1024) multiply them by the b-scales once per workgroup into an
LDS table the K loop reads; n4096_k512's register body loads one 16-byte vector per lane and row block that covers all
four K blocks of K = 512.

Shared with the column-major kernels: B straight from the 16×16 preshuffled layout into MFMA operand registers on
most paths (`../gemm/datapath.md` §10); one MFMA step per 128-wide K block folded with S_a·S_b
(`../gemm/mainloop.md` §5); A through XOR-swizzled LDS (`../gemm/datapath.md` §2) on the paths that stage it, by DMA
on some paths (`../gemm/datapath.md` §5); a configuration per m from a table (`../SKILL.md`, Stop
table).

Per file (functions are in the `kernel.py` each schema embeds):

**n512_k4096** (`_pick_config`, `_pick_ksplit`, `_pick_wsplit`, `_pick_nw`).
- Tiles 32×64 (m ≤ 256 and m = 1024), 32×32 with 2 waves (512), 64×64 (2048), 64×128 (4096); 4 waves along N
  otherwise; standard MFMA orientation, 2-byte stores.
- A from global into registers 3–6 K tiles ahead, then into LDS in MFMA-read order with `Swizzle<3,4,4>`, 2–4 K tiles
  per LDS stage and one barrier per stage; B into registers 2–4 K tiles ahead; the whole K loop unrolled.
- K split across workgroups: 8 ways at m ≤ 128, 4 at m = 256 (64–256 workgroups). Each thread stores its fp32 partial
  to its own workspace slot with sc0|sc1, bumps its own counter (`raw_ptr_buffer_atomic_fadd` of 1.0), and the thread
  that arrives last for its slot sums the partials in split order, writes bf16 and resets the counter
  (`../gemm/grid_pricing.md` §2, `../gemm/small_m.md` §4).
- K split inside the workgroup: two groups of waves at m = 512, 2048, 4096, summed through LDS in group order.

**n1024_k4096** (`_select_decode` for m ≤ 128, `_select_tiles` / `_select_ngroups` / `_select_mgroups` above).
- m ≤ 128: one workgroup per (16- or 32-row tile, 16 columns), K split over 4 waves (m 1, 2) or 8 (m 4–128) in
  contiguous ranges; A and B loaded up front, A hopping through per-wave LDS at m ≥ 4; wave 0 sums the partials from
  LDS in wave order; no workspace, counter or atomic; XCD remap at m = 128; 64 to 256 workgroups.
- m ≥ 256: MFMA issued with B as the first operand, so a lane holds one row's 4 adjacent columns: one a-scale per
  16-row block and 8-byte bf16 stores (`../gemm/epilogue.md` §3). Two 256-thread groups per workgroup
  split K (m 256–2048) and exchange halves of their rows through LDS; at m = 4096 two groups stacked along N share one
  A ring (a 64×256 tile). A through a 4-stage swizzled LDS ring: async DMA at m 256, 512, 2048, register-staged at
  1024 and 4096. B into registers two K tiles ahead. Each XCD gets a compact block of tiles. 256 workgroups.
- m = 256 also splits K 2 ways across workgroups: the last arriving split adds the two fp32 planes in plane order and
  restores the counter.

**n4096_k256** (`_cfg_for_m`) — the whole K = 256 per tile, no split.
- Weights are the MFMA A operand, so an accumulator holds 4 adjacent columns of one row; a lane packs two such
  tiles into 8 adjacent columns and writes them as one 16-byte store with sc1. Activations and their scales load
  straight into registers.
- m ≤ 256: weights straight into registers; wave tiles of 32 or 64 weight columns by 16 or 32 rows, 64 to 512
  workgroups.
- m ≥ 512: each workgroup copies its 128-column weight slice into LDS once (4 KiB per 16 columns, 64-byte pad
  between them) and four waves read their fragments from there while the workgroup walks 1, 2, 4 or 8 M tiles
  (m 512, 1024, 2048, 4096), keeping the grid at 256 workgroups. Above m = 1024 each wave transposes its 16×128 output
  through a private LDS scratch (68-dword pitch) so a quarter-wave stores one 256-byte row segment.

**n4096_k512** (`_pick_config`, `_small_cfg`, `_LDS_MIN_M`).
- m ≤ 128, register body: MFMA issued as (B, A) with four `v_mfma_f32_16x16x32_fp8_fp8` per 128-wide K block into a
  fresh accumulator, then `acc + p·(S_a·S_b)`; A and B straight from global with a 1–3 block prefetch; each lane owns
  4·tn adjacent output columns; 256 workgroups.
- m 256, 512: 32×128 / 64×128 workgroup tile, a 3-stage LDS ring filled by buffer→LDS DMA for both A and B, each K
  block staged as 1 KiB row-contiguous chunks with an XOR-swizzled 16-byte order; one
  `v_mfma_scale_f32_16x16x128_f8f6f4` per K block; stores with sc1.
- m ≥ 1024: 128×128 tile, 2×2 waves of 4×4 MFMA tiles, the same DMA scheme on a 2-stage ring; two M tiles per
  workgroup at m = 4096; row groups 2 and 3 of the last K block go out through the free LDS stage as full 128-byte
  rows. 256 or 512 workgroups.

## Result

Each file is the only schema at its Points, so the faster of AKLR's served kernel and aiter is aiter, and the
`speedup` CI wrote is the yardstick: a one-sided lower bound (α = 0.05) of the aiter / kernel time ratio over 32
rounds of CUDA-graph replays with L2 flushed before each call (`aklr/core/validation/stages/_measure.py`); here the
aiter time includes the a_scale copy at m ≥ 2 (Counter-signals). A Point is served from 1.03.

The predecessor repository's CI ran an editable install whose `setup.py` replaced any other FlyDSL with its pin, so
these runs used FlyDSL 0.3.2, not the CI image's 0.3.4.1 (not checked in the runs' logs;
`../pitfalls/measured_a_different_build.md`).

| File | m 1 | m 2–64 | m 128–512 | m 1024–4096 | Served |
|---|---|---|---|---|---|
| n512_k4096 | 4.569 *(field, E1)* | 4.677–5.571 *(field, E1)* | 2.967–4.190 *(field, E1)* | 1.486–2.532 *(field, E1)* | 13 of 13 |
| n1024_k4096 | 2.285 *(field, E2)* | 2.747–3.298 *(field, E2)* | 1.678–2.560 *(field, E2)* | 1.375–1.653 *(field, E2)* | 13 of 13 |
| n4096_k256 | 1.025 *(field, E3)* | 1.912–1.969 *(field, E3)* | 1.989–2.180 *(field, E3)* | 1.818–2.611 *(field, E3)* | 12 of 13 |
| n4096_k512 | 0.993 *(field, E4)* | 1.866–1.924 *(field, E4)* | 1.900–2.266 *(field, E4)* | 1.661–2.125 *(field, E4)* | 12 of 13 |

50 of the 52 Points are served; all 52 pass accuracy (`accuracy: passed` in each workload).

## Reproduce

On an MI355X, in a checkout of main (its files carry later CI numbers than the table), one schema at a time:

```
python -m aklr validate aklr/ops/gemm/impls/flydsl/gemm_a8w8_blockwise_scaled_b_shuffled_row_major_a_scale_n512_k4096/e5cda3e1b4f63183.json
```

and the same for the other three paths under `sources`. Each runs the audit and bench stages and writes
`aklr_validation/report.md` with the per-Point speedup (`aklr/ops/README.md` § Commands). Add `--base origin/main`
only on a branch that edits the file: with the base set, bench measures only schemas that are not at the base, so on
an unchanged checkout it skips the bench (`aklr/core/validation/stages/validation_scope.py`, `of_working_tree`). The
local result does not count; CI's measurement is the one dispatch reads.

## Rollback

Delete the schema file (in a PR): its Points go unserved, since no other file serves them. `AKLR_DISABLE`
takes a whole contract id and would also drop every other `gemm_a8w8_blockwise_scaled@1.0` kernel. The two m = 1
Points below 1.05 are not served without any rollback. If the bench stops relayouting a_scale in the timed baseline,
re-measure all four files before trusting their m ≥ 2 margins.

## Evidence

`<schema>.json#kernel.py:<line>` is a line of the `kernel.py` source a schema embeds in `kernel.sources`.

- E1 [2026-10, gfx950, FlyDSL 0.3.2, N=1, the predecessor repository's CI] speedup over aiter 1.486–5.571 per Point, 13 of 13 ≥ 1.03; m 1 4.569, m 2 5.571, m 2–64 4.677–5.571, m 128–512 2.967–4.190, m 1024–4096 1.486–2.532 (gemm_a8w8_blockwise_scaled_b_shuffled_row_major_a_scale_n512_k4096, m 1–4096)
- E2 [2026-10, gfx950, FlyDSL 0.3.2, N=1, the predecessor repository's CI] speedup over aiter 1.375–3.298 per Point, 13 of 13 ≥ 1.03; m 1 2.285, m 2 3.220, m 2–64 2.747–3.298, m 128–512 1.678–2.560, m 1024–4096 1.375–1.653 (gemm_a8w8_blockwise_scaled_b_shuffled_row_major_a_scale_n1024_k4096, m 1–4096)
- E3 [2026-10, gfx950, FlyDSL 0.3.2, N=1, the predecessor repository's CI] speedup over aiter 1.025–2.611 per Point, 12 of 13 ≥ 1.03; m 1 1.025, m 2 1.942, m 2–64 1.912–1.969, m 128–512 1.989–2.180, m 1024–4096 1.818–2.611 (gemm_a8w8_blockwise_scaled_b_shuffled_row_major_a_scale_n4096_k256, m 1–4096)
- E4 [2026-10, gfx950, FlyDSL 0.3.2, N=1, the predecessor repository's CI] speedup over aiter 0.993–2.266 per Point, 12 of 13 ≥ 1.03; m 1 0.993, m 2 1.922, m 2–64 1.866–1.924, m 128–512 1.900–2.266, m 1024–4096 1.661–2.125 (gemm_a8w8_blockwise_scaled_b_shuffled_row_major_a_scale_n4096_k512, m 1–4096)
- E5 [2026-10, gfx950, FlyDSL 0.3.2, N=1, the predecessor repository's CI] column-major variant, m 1 / m 2: n512_k4096 4.638 / 4.504, n1024_k4096 2.383 / 2.400, n4096_k256 1.039 / 1.035, n4096_k512 0.986 / 0.996 (gemm_a8w8_blockwise_scaled_b_shuffled_n512_k4096, _n1024_k4096, _n4096_k256, _n4096_k512)
- E6 [2026-10, gfx950, FlyDSL 0.3.2, N=5, the predecessor repository's CI; FlyDSL of the runs: the pin each run's `setup.py` installs, not read from the runs' logs] m = 1 across the five committed measurements, in commit order: n4096_k256 1.025, 1.006, 1.013, 1.033, 1.025; n4096_k512 1.004, 0.989, 1.009, 0.996, 0.993 (the row-major n4096_k256 and n4096_k512 schemas)
- E7 [2026-11, gfx950 (MI355X), aiter dbd8bf5, N=5, AKLR `bench_cudagraph`, median per-call kernel time] aiter's timed baseline copying the row-major a_scale inside the call / given scales copied beforehand, N 4096 × K 256 (default CK): m 2 8.456 / 4.392 µs, m 16 8.528 / 4.406 µs (aiter `gemm_a8w8_blockscale_bpreshuffle` through `aklr/ops/gemm/baseline.py`)
- S1 aklr/ops/gemm/baseline.py:58 for an a_scale marked `is_row_major` with a shuffled b (the `b_shuffled_row_major_a_scale` variant) the timed baseline gets `a_scale.T.contiguous().T` (:59); the docstring at :46 gives the reason
- S2 aklr/ops/gemm/contract.py:298 the `b_shuffled_row_major_a_scale` layout variant; compare `bounded` at :281
- S3 aklr/ops/gemm/impls/flydsl/gemm_a8w8_blockwise_scaled_b_shuffled_row_major_a_scale_n512_k4096/e5cda3e1b4f63183.json#kernel.py:344 cross-workgroup fixup: per-thread counter (`raw_ptr_buffer_atomic_fadd`, :376), partials summed in split order (:384), counters zeroed once (:595); split factors in `KS_MAP` at :546
- S4 aklr/ops/gemm/impls/flydsl/gemm_a8w8_blockwise_scaled_b_shuffled_row_major_a_scale_n1024_k4096/7c4c895c7743b287.json#kernel.py:1212 `_select_decode`; plane sum in plane order at :915; counters zeroed once at :1286
- S5 aklr/ops/gemm/impls/flydsl/gemm_a8w8_blockwise_scaled_b_shuffled_row_major_a_scale_n4096_k512/b52039c93b6d86ac.json#kernel.py:162 the register body's `mfma_f32_16x16x32_fp8_fp8`; configuration per m at :820
- S6 aklr/ops/gemm/impls/flydsl/gemm_a8w8_blockwise_scaled_b_shuffled_row_major_a_scale_n4096_k256/0660d74d64e95f05.json#kernel.py:73 `_cfg_for_m`
