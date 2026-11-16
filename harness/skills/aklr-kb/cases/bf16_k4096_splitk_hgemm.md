---
title: "bf16 GEMM, N 256–2048 × K 4096, and later files built from the same kernels for other N, K and m — K split over waves or workgroups, folded in a fixed order"
kind: case
ops: [gemm]
dtypes: [bf16]
regimes: ["M 1-256", "M 512-4096"]
contract: bitwise
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=1
measured:
  - {metric: "speedup over aiter, AKLR validate profile 19, per Point", value: "1.360–1.508; 5 of 5 Points at or above AKLR_MIN_SPEEDUP", gpu: MI355X, workload: "gemm_n64_k4096 file 3af3d3de3d0f2285, measured as its predecessor, which differs only in passing the stream of the input's device; m 256, 512, 1024, 2048, 4096", date: "2026-11-09"}
  - {metric: "speedup over aiter, AKLR validate profile 19, per Point", value: "1.207–1.308; 4 of 4 Points at or above AKLR_MIN_SPEEDUP", gpu: MI355X, workload: "gemm_n128_k6144 file 0ebeb298609e55ab, measured as its predecessor, which differs only in passing the stream of the input's device; m 256, 512, 1024, 2048", date: "2026-11-09"}
  - {metric: "speedup over aiter, AKLR validate profile 19, per Point", value: "1.379–1.627; 3 of 3 Points at or above AKLR_MIN_SPEEDUP", gpu: MI355X, workload: "gemm_n256_k6144 file d47047da480ab92f, measured as its predecessor, which lacks the barrier before the fold stores at m 1024; m 256, 512, 1024", date: "2026-11-09"}
  - {metric: "speedup over aiter, CI, per Point", value: "0.875–1.401; 9 of 13 Points ≥ 1.03 (m 1–256)", gpu: MI355X, workload: "gemm_n2048_k4096, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", run_ref: "6d7cfec (#102, CI run 42549655045, measurement profile 19)", date: "2026-11-09"}
  - {metric: "speedup over aiter, CI, per Point", value: "0.579–1.214; 5 of 13 Points ≥ 1.03 (m 1, 2, 64, 128, 256)", gpu: MI355X, workload: "gemm_n1024_k4096, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", run_ref: "6d7cfec (#102, CI run 42549655045, measurement profile 19)", date: "2026-11-09"}
  - {metric: "speedup over aiter, CI, per Point", value: "1.000–1.233; 11 of 13 Points ≥ 1.03 (m 4–4096)", gpu: MI355X, workload: "gemm_n256_k4096, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", run_ref: "6d7cfec (#102, CI run 42549655045, measurement profile 19)", date: "2026-11-09"}
  - {metric: "speedup over aiter, CI, per Point", value: "0.792–1.248; 4 of 13 Points ≥ 1.03 (m 128, 256, 512, 2048)", gpu: MI355X, workload: "gemm_n512_k4096, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", run_ref: "d908d01 (#71, CI run 42000899019); #102 left these values unchanged", date: "2026-11-05"}
  - {metric: "speedup over aiter, CI, per Point", value: "0.858–1.400; 9 of 13 Points ≥ 1.03 (m 1–256)", gpu: MI355X, workload: "gemm_n2048_k4096, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", date: "2026-10-28"}
  - {metric: "speedup over aiter, CI, per Point", value: "0.584–1.290; 5 of 13 Points ≥ 1.03 (m 1, 2, 64, 128, 256)", gpu: MI355X, workload: "gemm_n1024_k4096, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", date: "2026-10-28"}
  - {metric: "speedup over aiter, CI, per Point", value: "0.992–1.207; 10 of 13 Points ≥ 1.03 (m 4–64, 256–4096)", gpu: MI355X, workload: "gemm_n256_k4096, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", date: "2026-10-28"}
  - {metric: "speedup over aiter, CI, per Point", value: "0.793–1.237; 5 of 13 Points ≥ 1.03 (m 128, 256, 512, 2048, 4096)", gpu: MI355X, workload: "gemm_n512_k4096, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", date: "2026-10-28"}
  - {metric: "speedup over aiter, AKLR validate profile 19, per Point", value: "1.153–1.269; 7 of 7 Points at or above AKLR_MIN_SPEEDUP", gpu: MI355X, workload: "gemm_n512_k4096 file e42f6c833c101f69, m 1, 2, 4, 8, 16, 32, 64", date: "2026-11-09"}
  - {metric: "speedup over aiter, AKLR validate profile 19, per Point", value: "1.433–1.886; 8 of 8 Points at or above AKLR_MIN_SPEEDUP", gpu: MI355X, workload: "gemm_n2048_k2048 file 697eeb2233f25b25, m 1, 2, 4, 8, 16, 32, 64, 128", date: "2026-11-09"}
sources:
  - aklr/ops/gemm/impls/flydsl/gemm_n64_k4096/3af3d3de3d0f2285.json
  - aklr/ops/gemm/impls/flydsl/gemm_n128_k6144/0ebeb298609e55ab.json
  - aklr/ops/gemm/impls/flydsl/gemm_n256_k6144/d47047da480ab92f.json
  - aklr/ops/gemm/impls/flydsl/gemm_n2048_k4096/095a7d687f48faac.json
  - aklr/ops/gemm/impls/flydsl/gemm_n1024_k4096/4f306c3bd2f6b0c1.json
  - aklr/ops/gemm/impls/flydsl/gemm_n256_k4096/9200d6d7a56e5076.json
  - aklr/ops/gemm/impls/flydsl/gemm_n512_k4096/09c4acd65577d824.json
  - aklr/ops/gemm/impls/flydsl/gemm_n512_k4096/e42f6c833c101f69.json
  - aklr/ops/gemm/impls/flydsl/gemm_n2048_k2048/697eeb2233f25b25.json
---

# bf16 GEMM at N 256–2048 × K 4096, and files built from the same kernels: split K, fold in a fixed order

Four FlyDSL kernels for the `gemm@1.0` contract on DeepSeek-V4 shapes, one schema file per N, are described
in § Applies when through § Result; § Reproduce and § Rollback hold for every file on this page. Each other section
adds files built from the same kernels for other shapes or m ranges, with its own Points, numbers and
counter-signals. Each schema embeds its kernel as `kernel.sources[0].content` (`kernel.py`); the function names below
are in that file. "m" is the Point's M.

## Applies when

- Contract `gemm@1.0`, layout variant `row_major` (`b.is_shuffled` false): `a` (m, k) and `b` (n, k) bf16,
  `trans_b`, no bias, bf16 `out`. `compare` is bitwise: `determinism: bitwise`, `max_violation_fraction` 0.0
  (`CompareSpec` defaults, `GEMM` in `aklr/ops/gemm/contract.py`). Every reduction these kernels run at a served
  Point adds in a fixed order.
- The four DeepSeek-V4 files: K = 4096 with N = 256, 512, 1024 or 2048, a weight of 2, 4, 8 or 16 MiB; each lists 13
  Points, m = 1, 2, 4, …, 4096.
- Their served Points on main as of #102 (6d7cfec, CI run 42549655045, measurement profile 19): n2048 m 1–256; n1024
  m 1, 2, 64–256; n256 m 4–4096; n512 m 128–512 and 2048. #102 left n512's values as #71 measured them: m 4096 at
  1.004, below the threshold, where the run in § Result had it above.
- No other file serves those Points, so the yardstick there is aiter.

## Counter-signals

- **Repacking the weight on every call does not pay at N = 1024.** At m = 4, 512, 1024 and 2048, n1024's
  launch runs `_build_weight_pack` (the whole 8 MiB `b` into MFMA lane order, a second launch) before
  `_build_packed_splitk_launcher`: 0.584, 0.819, 0.884, 0.973 *(field, E2)*. The module docstring still
  describes a cached relayout invalidated by tensor identity or version; the code
  (`build_gemm_a16w16_nt_n1024_k4096_module`) repacks on every call.
- **n512 at m ≤ 64 and n1024 at m 8–32, both splitting K only inside the workgroup on 32–128 workgroups, stay below
  aiter in CI; two kernels in one CI run, not a rule.** n512 at m ≤ 64 (16×16 tiles,
  4 waves splitting K, 32, 64 and 128 workgroups): 0.793–0.941 *(field, E4)*. n1024 at m 8–32 (16×16 tiles, 4–8
  waves splitting K, 64 and 128 workgroups): 0.907–0.980 *(field, E2)*. The count alone does not decide: n256 at
  m 4–64 runs 64 workgroups with split-K 8 or 4 across workgroups and was served at 1.036–1.130 in the predecessor
  repository's CI *(field, E3)*.
- **At m ≥ 512, these CI results do not isolate the contribution of a K split:** n2048
  (0.858–1.026) *(field, E1)* and n1024 (0.819–0.973) *(field, E2)* were not served there, and n1024 runs a split-K
  launcher at m 512–2048 (after a per-call repack); n256 was served at m 512–4096 *(field, E3)* and n512 at m 512,
  2048 and 4096 *(field, E4)*; later runs moved n512's m 4096 below 1.03 (the schema's recorded speedup). They are
  not a reason to rule a K split out on another configuration; compare split and unsplit versions of the same
  pipeline before attributing either a gain or a loss to the split. n512's kernel ported to narrower N at M 256–4096:
  `../gemm/mainloop.md` §11.
- **Arrival counters that live across calls.** n2048's counters are zeroed once per shape in the cached builder;
  n256's are allocated on the first call inside `launch` and kept in a closure list (with per-address pointer
  caches). The reducing block resets each counter, so the next call finds zeros. `SKILL.md`'s Stop table asks
  for scratch rewritten by every call before it is read: for a new kernel, zero counters on every call
  (`../gemm/small_m.md` §4).
- **System-scope partials.** n2048 stores and reloads its fp32 partials with `sc0 sc1` (`CPOL_SYS`);
  `../gemm/small_m.md` §4 found agent scope sufficient across XCDs on other kernels. Not raced here.
- **Code that no Point runs:** n512's cross-workgroup split-K with a separate reduce kernel (`_build_reduce`; the
  in-workgroup split sets `SPLITK = 1` at m ≤ 1024 and m 2048, 4096 have split 1), although its docstring
  describes it; n1024's cross-workgroup last-arriver path (`_build_splitk_launcher`); n256's bf16-atomic split-K
  (`IS_SPLIT_K` is false whenever `SPLIT_FP32` is set, which the builder does for every split). CI measures none
  of them.

## Mechanism

Shared by the four kernels:

- **MFMA `mfma_f32_16x16x32_bf16` into fp32 accumulators** wherever an MFMA runs (`../gemm/mainloop.md`
  §5); n1024 at m 1 and 2 runs none.
- **Operands into LDS by buffer→LDS DMA** (`buffer_load_dwordx4 … lds`, 16 B per lane) in XOR-swizzled 16 B
  chunks, read back as 16 B fragments (`../gemm/datapath.md` §2, §5); n1024 at m 1 and 2 loads `a` and `b` straight
  into registers instead.
- **K split for parallelism at small M, partials kept in fp32 and summed in a fixed order, one bf16 rounding**
  (`../gemm/grid_pricing.md` §2, `../gemm/small_m.md` §4, §5,
  `../gemm/correctness.md` §8). n2048 ports aiter's split-K HGEMM but replaces its bf16 atomic
  adds into C, which follow arrival order, with an fp32 workspace and a last-arriver fold.
- **64, 256 or 512 workgroups at every Point one CI run of the predecessor repository put at or above 1.03** (§ Result;
  `../gemm/grid_pricing.md` §1, `../gemm/small_m.md` §3). n256 at m 128, served as of #102, runs 128 workgroups.
- **A per-m configuration table**, compiled on a Point's first call inside the `functools.lru_cache` function
  `_aklr_launch`.

At the Points that CI run of the predecessor repository put at or above 1.03 (§ Result), and n256 at m 128:

| | n2048_k4096 | n1024_k4096 | n256_k4096 | n512_k4096 |
|---|---|---|---|---|
| Path, tile, waves | m 1–8: `_compile_skw8`, one 8×8 output tile per workgroup, 8 waves. m 16, 32: `_compile_skw_av`, 16×8 / 16×16 tile, 4 waves, a private 6-slot LDS ring per wave. m 64–256: `compile_hgemm`, 32×32 / 32×64 / 64×64 tile, 4 waves, 2–4-stage LDS ring | m 1: `_build_gemv_launcher`, one output column per wave, 4 waves; 16 B loads of `a` and `b` per 512-wide K chunk, `v_dot2_f32_bf16` into one fp32 chain, a 6-step `ds_bpermute` butterfly. m 2: `_build_multi_gemv_launcher`, each `b` vector feeds both rows' chains. m 64–256: `_build_lds_splitk_launcher`, 16×16 / 16×32 / 32×32 tile, 4 waves | m 4–64: 16×32 tile, 1 wave (m ≤ 16), or 32×32, 2 waves. m 128–512: 32×32, 2 waves. m 1024: 32×32, 2 waves. m 2048, 4096: 32×64 / 64×64, 2 waves along N × 2 along K | m 128–512: 16×16 / 16×32 / 32×32 tile, 4 waves along K, a private 4-stage LDS ring per wave. m 2048: 64×64, 2×2 waves, 5 stages; m 4096: 128×64, 4×2 waves, 4 stages; both an unrolled ping-pong with fenced next-tile fragment reads |
| K split and fold | Inside the workgroup (m ≤ 32): wave w takes K slice w, partials summed through LDS in slice order. Across workgroups (m 64–256, split 4): each slice stores its fp32 tile to a workspace, then takes a ticket from a per-tile counter (agent-scope `atomicrmw add`); the slice holding ticket 3 sums slices 0–3 in index order (its own from registers at m 128 and 256; at m 64 all four reloaded from the workspace), rounds once, stores, resets the counter | m 1, 2: none (one wave holds all of K). m 64–256: each wave owns a contiguous quarter of K; the K loop waits only on that wave's own DMA, no `s_barrier`; partials parked in LDS, summed over the 4 waves in fixed order | Split-K 8 (m ≤ 32) or 4 (m 64–512) across workgroups (`SPLIT_FP32`): fp32 partials in MFMA fragment order, `sc1` stores, one counter per (tile, wave), or per tile at m 32 and 128; the last arriver loads slices 0…s−1 in index order, sums, resets the counter. m ≥ 2048: the two K waves fold in fp32 through LDS in fixed order | m 128–512: wave 0 sums the 4 partials through LDS in wave order. m 2048, 4096: none |
| Workgroups | 256 (m ≤ 32), 512 (m 64–256) | 256 at each served Point | 64 (m 4–64), 128 (m 128), 256 (m 256, 1024–4096), 512 (m 512) | 256 at each served Point |
| Tile order | m 32: the two workgroups that share a `b` tile on one XCD. Otherwise linear; the split-K path asserts no remap | linear | m ≥ 512: each XCD a contiguous eighth of the M tiles with all N tiles (`swizzle_for_cache_reuse`) | m 128–512: each XCD a box of 8 M tiles × 4 N tiles (`_XCD_REMAP`); m 2048, 4096 linear |

XCD order: `../gemm/schedule_and_locality.md` §2, `../gemm/small_m.md` §6.

## Result

`speedup` as CI writes it into each schema's `workloads`: a lower bound of the CUDA-graph time ratio aiter /
kernel (`aklr/core/validation/stages/_bench/_report.py`). The table is one CI run of the predecessor repository; bold: at or
above 1.03 (`AKLR_MIN_SPEEDUP`'s default then, `aklr/ops/README.md` § Environment variables) in that run. Per Point:

The predecessor repository's CI installed the code with `--no-deps`, but the editable install ran that repository's
`setup.py`, which at that commit replaced any other FlyDSL with its pin, so these runs used FlyDSL 0.3.2, the
repository's pin, not the CI image's 0.3.4.1 (`../pitfalls/measured_a_different_build.md`).

| File | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 | ≥ 1.03 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| n2048 *(field, E1)* | **1.394** | **1.395** | **1.400** | **1.373** | **1.278** | **1.197** | **1.205** | **1.151** | **1.076** | 1.026 | 1.002 | 0.988 | 0.858 | 9 of 13 |
| n1024 *(field, E2)* | **1.288** | **1.290** | 0.584 | 0.980 | 0.907 | 0.973 | **1.163** | **1.190** | **1.208** | 0.819 | 0.884 | 0.973 | 0.944 | 5 of 13 |
| n256 *(field, E3)* | 1.000 | 1.002 | **1.101** | **1.081** | **1.130** | **1.071** | **1.036** | 0.992 | **1.109** | **1.207** | **1.153** | **1.172** | **1.057** | 10 of 13 |
| n512 *(field, E4)* | 0.925 | 0.898 | 0.877 | 0.850 | 0.793 | 0.851 | 0.941 | **1.187** | **1.194** | **1.237** | 1.025 | **1.116** | **1.059** | 5 of 13 |

29 of the 52 Points were at or above 1.03 in that run. As of #102 (6d7cfec, CI run 42549655045, measurement profile
19) 29 are served, not the same 29: n256 also at m 128, and n512 not at m 4096 (1.004 in CI run 42000899019, #71,
whose n512 values #102 left unchanged) *(field, E64)*. The `speedup` in each schema's `workloads` is the current value.

## N 64–256 at m 256–4096: n512's K split over waves, ported

Three names take n512's m 128–512 path (K over the waves of one workgroup, a private LDS ring per wave; § Mechanism)
to narrower N at m 256–4096, one new file each: `gemm_n64_k4096/3af3d3de3d0f2285.json` (m 256–4096, 512 KiB of
weight), `gemm_n128_k6144/0ebeb298609e55ab.json` (m 256–2048, 1.5 MiB) and `gemm_n256_k6144/d47047da480ab92f.json`
(m 256–1024, 3 MiB). No other file serves these Points (the older K 6144 files read below aiter at these m in CI,
#102), so the yardstick is aiter; narrow N at small m is on [bf16_glm52_narrow_n.md](bf16_glm52_narrow_n.md). CI
reads 1.211–1.624 at all 12 Points (#125), at or above `AKLR_MIN_SPEEDUP` (default 1.05, `aklr/ops/README.md` §
Environment variables; `qualifies()` in `aklr/core/contract/spec/_workloads.py`), so dispatch serves the new files.
N 2048 × K 2048 at m 256–4096 starts from n512's `_build_gemm` too:
[bf16_large_m_whole_k_tiles.md](bf16_large_m_whole_k_tiles.md).

Each workgroup owns one output tile. Except at N 64 m 4096, it splits its share of K over KW waves: the share is all
of K, or a half or a quarter at N 256 m 512 and 1024, where K is split across workgroups as well (below). Each wave
streams its part through its own LDS ring (`buffer_load_dwordx4 … lds` into XOR-swizzled 16 B chunks), waits only on
its own `vmcnt` and `lgkmcnt` and passes no `s_barrier` in the K loop; after one barrier the waves' fp32 partials meet
in LDS and are added in a fixed order. N 64 at m 4096 splits K four ways across workgroups only: its four waves split
the 64×64 output tile 2 × 2, each running the workgroup's quarter of K from one shared six-slot ring with a barrier
per K tile, so no wave partials meet in LDS; the workgroups' partials are folded through fp32 planes in global memory
(below). Every output is rounded to bf16 once, after the last fold. Every Point launches 256 workgroups, one round
(`../gemm/grid_pricing.md` §1).

| | n64_k4096 | n128_k6144 | n256_k6144 |
|---|---|---|---|
| Tile, K waves, BK, ring slots per wave | m 256: 8×8, 8 waves, BK 128, 3 slots; m 512: 16×8, m 1024: 16×16, 8 waves, BK 128, 2 slots; m 2048: 32×16, 4 waves, BK 64, 4 slots; m 4096: 64×64, four waves over the tile (2 × 2), BK 64, six shared slots | m 256: 8×16, 8 waves, 2 slots; m 512: 16×16, 8 waves, 2 slots; m 1024: 32×16, 4 waves, 3 slots; m 2048: 32×32, 3 waves, 3 slots; BK 128 | 4 waves, 2 slots; m 256: 16×16, BK 128; m 512: 32×32, BK 128; m 1024: 64×64, BK 64 |
| K walk | m 256–2048: a contiguous K range per wave; workgroup (x, y) starts at tile (x + R·y) mod tiles, R 5 (11 at m 2048); m 4096: each split walks its quarter of K from its start | contiguous per wave, start rotated by the tile's coordinates; at m 1024 each wave one tile further; odd column tiles walk K backwards at m 512 and 2048 | wave w takes BK tiles w, w + 4, …; the start is set per XCD, then rotated by the tile's place in its XCD box |
| Fold in the workgroup | m 256–2048: wave 0 adds the waves' partials in wave order; m 4096: none | a fixed pairwise tree ((p0 + p1) + (p2 + p3)) … at m 256–1024, wave order at m 2048; wave 0 keeps its own partial in registers at m 512; fragments spread over the waves at m 1024 and 2048 | every thread adds its outputs over the four waves in order: one column at m 256; read from a padded row-major LDS tile, 4 adjacent columns at m 512 and 4 rows × 4 adjacent columns at m 1024; at m 1024 that tile overruns the wave's own ring, so a barrier first waits for every wave to finish its K loop |
| Across workgroups | m 4096 (below) | none | m 512 split 2, m 1024 split 4 (below) |
| Tile order, C store | 2-D grid in launch order; A first in the MFMA, 2-byte stores | each XCD a box of 8 × 4 or 4 × 8 tiles; B first, 8 B stores | each XCD a box of tiles, a tile's K splits on different XCDs; 2-byte stores at m 256, 8 B with `sc1` at m 512, 1024 |
| Also | kernel arguments preloaded | preloaded; LLVM scheduler strategy per m (`max-ilp` at m 512, `iterative-ilp` otherwise); `s_setprio 3` while issuing DMA at m 256–1024; K loop unrolled at m 1024, 2048 | preloaded; at m 512 the A and B descriptors start at the tile's first row |

**The folds across workgroups** keep the fp32 partial planes and the arrival words in a `torch.empty` workspace
allocated by each call, and no workgroup waits for another:
- n64_k4096 at m 4096 (`_build_fold`): 64×64 tiles, all of N, K split 4 (BK 64), four waves as 2 × 2; 64 row tiles ×
  4 splits = 256 workgroups. Before the first DMA every thread loads its tile's 64-bit counter word (counters 256 B
  apart; thread 0 alone uses it) and forms a 56-bit launch tag from the dispatch id and the queue pointer. After the
  K loop, if the word's high bits are not this tag, thread 0 compare-and-swaps the value it read to `tag << 8`. The
  partials are stored `sc0 sc1`; after `vmcnt(0)` and a barrier thread 0 adds 1 (monotonic, agent scope), and the
  split that gets back `tag << 8 | 3` loads the other three planes (`sc0 sc1`), adds the four in split order, its own
  from registers, and stores.
- n256_k6144 at m 512 (split 2): each workgroup folds its waves, then stores its partial with `sc1`. Once a wave's
  stores are acknowledged its lane 0 exchanges the launch tag into the wave's 8 B word of the tile's 128 B slot; the
  wave that reads back this launch's tag arrived second, swaps a sentinel back and adds its partner's partial to its
  own, kept in registers. At m 1024 (split 4) thread 0 counts in with a compare-and-swap chain on the slot's word,
  where a word with another tag counts as no arrival; the fourth arrival loads the four planes (`sc1`), adds them in
  split order and stores, and the word goes back to the sentinel. The tag is the low 59 bits of
  dispatch id · H1 + queue address · H2, with odd 64-bit multipliers H1 and H2.

Counter-signals:
- **These folds read scratch this call has not written.** `SKILL.md`'s Stop table, and § Counter-signals above, ask
  for counters zeroed in each call. N 64 at m 4096 and N 256 at m 512 and 1024 instead read words of an uninitialised
  `torch.empty` workspace and rely on the launch tag: the output is right only while no leftover word carries this
  launch's tag (N 64: the high 56 bits of the counter word; N 256: the 59-bit tag), and while the tag differs between
  launches that can be given the same workspace. Checked on every listed Point of the three files: the forge driver's
  conditions at 20 tries (streams limited to 16 and 64 CUs; a second concurrent call, which the driver runs on the
  same inputs), one graph replayed 24 times with three input sets rotated into its buffers, and two calls at once on
  two streams with different inputs, unrestricted and on the same 64 CUs; every try gave the output of the call alone
  *(field, E63)*. Not checked: a full model's decode graph, dispatch-id wraparound, and a garbage word equal to the
  tag.
- **Do not use the bare dispatch id as the tag.** It is a small integer and each queue counts its own: with split 2, a
  first arriver whose uninitialised word equals it (dispatch id 0 on a new queue over zeroed memory, or a word another
  queue left at the same dispatch id) folds before its partner has written. Hashing in the queue address makes that
  improbable, not impossible: within one queue, distinct dispatch ids below 2^59 give distinct tags (H1 is odd); two
  queues q1 and q2 share a tag where did2 ≡ did1 + (q1 − q2)·H2·H1⁻¹ (mod 2^59), which in general needs dispatch ids
  far apart; a tag below 2^32 has a probability of about 2^-27 (estimated, for a uniformly distributed hash). These
  runs resolve no cost of the hash: the same kernel with the bare-id tag read 1.394, 1.625 and 1.460, its intervals
  overlapping its predecessor's *(field, E62)*.
- **Not covered:** N 128 at m 4096 and N 256 at m 2048–4096 are not in these files; the other file of each K 6144 name
  lists them.

AKLR validate (measurement profile 19) on one MI355X at main 0e02a81, not CI: N 64 1.360–1.508 *(field, E60)*, N 128
1.207–1.308 *(field, E61)*, N 256 1.379–1.627 *(field, E62)* over aiter, each measured as its file's predecessor (N
256 without the m 1024 barrier), each highest at m 512; every Point passed accuracy. Reproduce with `python -m aklr
validate` on `aklr/ops/gemm/impls/flydsl/gemm_n64_k4096/3af3d3de3d0f2285.json`,
`aklr/ops/gemm/impls/flydsl/gemm_n128_k6144/0ebeb298609e55ab.json` and
`aklr/ops/gemm/impls/flydsl/gemm_n256_k6144/d47047da480ab92f.json` (§ Reproduce). To roll back, delete the file: its
Points are then served by no file (the older K 6144 files read below the threshold there), and the caller runs its
own path.

## Reproduce

On an MI355X with the repository's FlyDSL 0.3.2, from the repository root:

```
python -m aklr validate aklr/ops/gemm/impls/flydsl/gemm_n2048_k4096/095a7d687f48faac.json
python -m aklr validate aklr/ops/gemm/impls/flydsl/gemm_n1024_k4096/4f306c3bd2f6b0c1.json
python -m aklr validate aklr/ops/gemm/impls/flydsl/gemm_n256_k4096/9200d6d7a56e5076.json
python -m aklr validate aklr/ops/gemm/impls/flydsl/gemm_n512_k4096/09c4acd65577d824.json
```

The report goes to `aklr_validation/report.md`; a run outside CI checks the file, the served numbers are CI's
(`aklr/ops/README.md` § Workflow). Add `--base origin/main` only on a branch that edits the file: with the base set,
bench measures only schemas that are not at the base, so on an unchanged checkout it skips the bench
(`aklr/core/validation/stages/validation_scope.py`, `of_working_tree`). To read a kernel:
`git show <rev>:<schema> | python3 -c 'import json,sys; print(json.load(sys.stdin)["kernel"]["sources"][0]["content"])'`.

## Rollback

Delete the schema file: dispatch stops serving that file's Points. `AKLR_DISABLE=gemm@1.0` stops serving the
whole contract, every other bf16 GEMM schema included. A re-measured Point below 1.05 stops being served
without a rollback, and a `failed` accuracy is never served. Roll back for a wrong result CI does not catch, for
example after a FlyDSL or LLVM upgrade, or a split-K counter left non-zero by an aborted launch.

## N 512 at m 1–64: both operands in registers, K split over two waves

`gemm_n512_k4096/e42f6c833c101f69.json` lists m 1–64 of the same name, where `09c4acd6…` (16×16 tiles, four waves
splitting K through LDS rings, 32–128 workgroups) stays below aiter (§ Counter-signals). CI reads 1.139–1.269 at
the seven Points (#125), so dispatch serves all seven. Its `run` sends an m outside `_PTABLE` to `torch.matmul`; no
listed Point reaches that.

- **One round of small tiles.** Every Point launches 256 workgroups of two waves: m 1–4 one M-block with 2-column
  tiles, m 8 and 16 two M-blocks of 4 or 8 rows with 4-column tiles, m 32 and 64 four M-blocks of 8 or 16 rows with
  8-column tiles. The M-blocks of one B tile get workgroup ids 8 apart, so that they share an XCD
  (`../gemm/small_m.md` §3, `../gemm/schedule_and_locality.md` §2).
- **Packed MFMA operands.** Each 16-row operand of `mfma_f32_16x16x32_bf16` holds R real rows (of A, or B's columns)
  times S = 16 / R K slices, so no loaded byte is padding; of the 16×16 result only the entries (i, j) with
  i mod S = j mod S are partials of real outputs. B is the first operand.
- **No LDS ring.** Each wave loads its half of K for A and B with 16 B `buffer_load`s straight into registers. This
  departs from `../gemm/small_m.md` §5, where B straight to registers was 2.6× slower than B through LDS on o_proj
  (m 8, N 6144, K 16384). Here no full LDS ring was built; at m 64, DMA-ing A into a per-wave LDS ring instead of
  registers, without reading it back, read 5.11 against 4.90 µs *(field, E9)*. At m ≤ 32 (`_compile_packed`) every
  load is issued before the first MFMA; at m = 64 (`_compile_m64`) the loads run 10 of the wave's 16 K windows ahead
  of the MFMAs, and odd M-blocks start 8 windows later than even ones, so that the even and odd M-blocks of a B tile
  do not read the same lines at once.
- **Fold in a fixed order.** At m ≤ 32 the partials go through LDS: one pass adds each output's K slices in
  ascending order, a second adds the two waves in order, then one bf16 rounding and a 4 or 8 B store. At m = 64 the
  K slices of an output sit in one lane quad and are added in registers with DPP broadcasts; wave 1 passes its sums
  to wave 0 through LDS, and wave 0 adds and stores 2-byte values (`../gemm/correctness.md` §8).
- The three pointer arguments are preloaded into SGPRs (`amdgpu-kernarg-preload-count` 6; `../gemm/small_m.md` §6).

AKLR validate (measurement profile 19) on one MI355X at main 0e02a81, not CI: 1.153–1.269 over aiter at the seven
Points, highest at m = 32 *(field, E5)*. At m 1–8 the kernel read 4.20–4.38 µs per round and aiter 4.86–5.33
*(field, E6)*; whether profile 19 has a floor near there was not measured (`../pitfalls/aklr_dispatch_floor.md`).
Under stress every listed Point gave the output of the call alone *(field, E8)*. Reproduce with
`python -m aklr validate aklr/ops/gemm/impls/flydsl/gemm_n512_k4096/e42f6c833c101f69.json` (§ Reproduce). To roll
back, delete that file: m 1–64 are then served by no file, since `09c4acd6…` is below the threshold there, and the
caller runs its own path.

## N 2048 × K 2048 at m 1–128: n2048's in-workgroup K split at K 2048

`gemm_n2048_k2048/697eeb2233f25b25.json` lists m 1, 2, 4, …, 128 and runs adaptations of n2048's `_compile_skw8`
(m ≤ 8) and `_compile_skw_av` (m 16–128) at K = 2048, 8 MiB of weight. No other file serves these Points, so the
yardstick is aiter; CI reads 1.410–1.856 (#125), and dispatch serves all eight Points. Every Point is one launch of
256 workgroups of four waves, each wave owning a 512-wide quarter of K. B is the MFMA's first operand, so a lane holds
four adjacent output columns and stores them in one 8 B write; the partials meet in LDS and are added in wave order
before one bf16 rounding, by wave 0 at m ≤ 8 and, at m 16–128, by wave f mod 4 for fragment f. Other m go to
`torch.matmul`.

- m ≤ 8, `_compile_skw8`: 8×8 output tiles (MFMA rows 8–15 repeat rows 0–7). No ring: each wave DMAs its whole K
  quarter of A and B (16 `buffer_load_dwordx4 … lds`) into its own LDS area before the loop and waits for it line by
  line; the partials meet in a separate LDS array and wave 0 adds them.
- m 16–128, `_compile_skw_av`: each wave streams its quarter line by line (64 bf16 per row) through a private ring,
  refills a slot once its `ds_read`s of it retire, and waits only on its own `vmcnt`, with no `s_barrier` in the K
  loop. Tile and slots: 16×8 and 8 (m 16), 16×16 and 8 (m 32), 16×32 and 3 (m 64), 32×32 and 2 (m 128). The M-tiles
  of one B tile get workgroup ids 8 apart. At m 64 and 128, workgroup (m-tile, n-group) starts at K line
  2·m-tile + 3·n-group mod 8 (`../gemm/small_m.md` §6). Kernel arguments are preloaded only where a B tile has
  several M-tiles (m 32–128).

AKLR validate (profile 19), not CI: 1.433–1.886 over aiter, highest at m = 32 *(field, E7)*. Under stress every listed
Point gave the output of the call alone *(field, E8)*. Reproduce with
`python -m aklr validate aklr/ops/gemm/impls/flydsl/gemm_n2048_k2048/697eeb2233f25b25.json`; to roll back, delete
it: its Points are then served by no file.

## Evidence

- E60 [2026-11, gfx950, FlyDSL 0.3.2, N=16, AKLR validate at 0e02a81, measurement profile 19: one-sided lower bound of the aiter / kernel time ratio over 16 rounds of CUDA-graph replays with L2 flushed before each call, one MI355X, aiter 0.1.21.dev66+gdbd8bf5bd] the predecessor of 3af3d3de3d0f2285, which differs only in passing the stream of the input's device: 1.431, 1.508, 1.502, 1.414, 1.360 at m 256, 512, 1024, 2048, 4096; accuracy passed (gemm_n64_k4096)
- E61 [2026-11, gfx950, FlyDSL 0.3.2, N=16, the ruler of E60] the predecessor of 0ebeb298609e55ab, which differs only in passing the stream of the input's device: 1.258, 1.308, 1.253, 1.207 at m 256, 512, 1024, 2048; accuracy passed (gemm_n128_k6144)
- E62 [2026-11, gfx950, FlyDSL 0.3.2, N=16, the ruler of E60; audit and bench passed, 0 findings] the predecessor of d47047da480ab92f, which lacks the barrier before the fold stores at m 1024 (m 256 and 512 compile to the same code): 1.379, 1.627, 1.449 at m 256, 512, 1024, kernel medians 6.35, 7.98, 10.95 µs, worst violation fraction 0; intervals 1.379–1.398, 1.627–1.635, 1.449–1.469. The same kernel with the bare dispatch id as its tag, same ruler, two runs: 1.394 and 1.393, 1.625 and 1.625, 1.460 and 1.453; intervals 1.394–1.405, 1.625–1.634, 1.460–1.482 in the first; kernel medians 6.33, 7.98, 10.72 µs
- E63 [2026-11, gfx950, FlyDSL 0.3.2, N=20, eager and op-level CUDA-graph stress on one MI355X, main 0e02a81 with each schema, each try compared byte for byte with an eager call alone] the predecessors of 3af3d3de3d0f2285, 0ebeb298609e55ab and d47047da480ab92f (E60–E62), every listed Point: the forge driver's conditions (streams limited to 16 and 64 CUs; a second concurrent call on the same inputs) and one captured graph replayed on the same inputs, 20 tries each; one graph replayed 24 times with three AKLR input sets rotated into its buffers; two calls at once on two streams with different inputs, unrestricted and both on the same 64 CUs, 20 tries each: 0 tries differ, no hang
- E64 [2026-11, gfx950, FlyDSL 0.3.2 (the pin of baseline aiter dbd8bf5, read from the workflow), N=1, AKLR CI perf runs; measurement profile 19 for #102] Points at or above 1.03: n2048 m 1–256 (0.875–1.401), n1024 m 1, 2, 64–256 (0.579–1.214), n256 m 4–4096 (1.000–1.233; m 128 1.031) as of #102 (6d7cfec, CI run 42549655045); n512 m 128–512 and 2048 (0.792–1.248; m 4096 1.004) from d908d01 (#71, CI run 42000899019), which #102 left unchanged (each schema's `workloads` at 6d7cfec)
- S60 harness/scripts/forge_task/assets/aklr_forge_driver.py:437 `_stress_conditions`; `beside_another` at :456 gives the second call the same inputs (`twin = {**call, …}`, :457)
- S61 aklr/ops/gemm/impls/flydsl/gemm_n64_k4096/3af3d3de3d0f2285.json#kernel.py:21 `_CONFIGS` and `_FOLD` (:27); counter word loaded by every thread and 56-bit tag formed before any DMA at :304-313; stale-word compare-and-swap at :395-409; `vmcnt(0)` and barrier at :411; count-in add at :412-419; fold in split order at :428-449; `torch.empty` per call at :461
- S62 aklr/ops/gemm/impls/flydsl/gemm_n128_k6144/0ebeb298609e55ab.json#kernel.py:22 `_CONFIGS` (PRO bits at :120-122: closed-form offsets, late fragment setup, prologue offsets, spread fold, `s_setprio`, pairwise tree); `_tree_sum` at :76; scheduler strategy hint at :474-477
- S63 aklr/ops/gemm/impls/flydsl/gemm_n256_k6144/d47047da480ab92f.json#kernel.py:23 `_CONFIGS` and `_KROT` (:29); H1 and H2 at :51-52; `_arrive` at :101; partial store `sc1` at :409-415; `_exchange` at :419 (tag at :421-427, two-split exchange at :430-450, compare-and-swap chain at :452); per-wave lane 0 after `vmcnt(0)` at :468-475; fold at :491-525 (at m 1024 after a barrier at :367-372, since the padded fold tile runs past the wave's ring); `torch.empty` per call at :584
- E1 [2026-10, gfx950, FlyDSL 0.3.2, N=1, the predecessor repository's CI] speedup over aiter per Point 0.858–1.400, 9 of 13 Points ≥ 1.03 (m 1–256) (gemm_n2048_k4096, m 1–4096)
- E2 [2026-10, gfx950, FlyDSL 0.3.2, N=1, the predecessor repository's CI] speedup over aiter per Point 0.584–1.290, 5 of 13 Points ≥ 1.03 (m 1, 2, 64, 128, 256) (gemm_n1024_k4096, m 1–4096)
- E3 [2026-10, gfx950, FlyDSL 0.3.2, N=1, the predecessor repository's CI] speedup over aiter per Point 0.992–1.207, 10 of 13 Points ≥ 1.03 (m 4–64, 256–4096) (gemm_n256_k4096, m 1–4096)
- E4 [2026-10, gfx950, FlyDSL 0.3.2, N=1, the predecessor repository's CI] speedup over aiter per Point 0.793–1.237, 5 of 13 Points ≥ 1.03 (m 128, 256, 512, 2048, 4096) (gemm_n512_k4096, m 1–4096)
- E5 [2026-11, gfx950, FlyDSL 0.3.2, N=16, AKLR validate at 0e02a81, measurement profile 19: one-sided lower bound of the aiter / kernel time ratio over 16 rounds of CUDA-graph replays with L2 flushed before each call, one MI355X, aiter 0.1.21.dev66+gdbd8bf5bd] e42f6c833c101f69: 1.166, 1.153, 1.158, 1.189, 1.210, 1.269, 1.263 at m 1, 2, 4, 8, 16, 32, 64; accuracy passed at every Point (gemm_n512_k4096)
- E6 [2026-11, gfx950, FlyDSL 0.3.2, N=16, the run of E5] per-round times at m 1, 2, 4, 8: kernel 4.20–4.31, 4.22–4.38, 4.21–4.36, 4.23–4.38 µs; aiter 4.97–5.07, 4.86–5.20, 4.88–5.15, 4.98–5.33 µs (`candidate_milliseconds`, `versus_milliseconds`)
- E7 [2026-11, gfx950, FlyDSL 0.3.2, N=16, the ruler of E5] 697eeb2233f25b25: 1.485, 1.561, 1.567, 1.583, 1.596, 1.886, 1.748, 1.433 at m 1, 2, 4, 8, 16, 32, 64, 128; accuracy passed (gemm_n2048_k2048)
- E8 [2026-11, gfx950, FlyDSL 0.3.2, N=20, eager and op-level CUDA-graph stress on one MI355X, main 0e02a81 with each schema, each try compared byte for byte with an eager call alone] e42f6c833c101f69 and 697eeb2233f25b25, every listed Point: the forge driver's conditions (streams limited to 16 and 64 CUs; a second concurrent call on the same inputs) and one captured graph replayed on the same inputs, 20 tries each; one graph replayed 24 times with three AKLR input sets rotated into its buffers; two calls at once on two streams with different inputs, unrestricted and both on the same 64 CUs, 20 tries each: 0 tries differ; a full model's decode graph was not run
- E9 [2026-11, gfx950, FlyDSL 0.3.2, N=3, the campaign's in-process probe, 2–3 alternating rounds, each the median of 10 graph readings; outputs wrong by design, not shipped] gemm_n512_k4096 m 64, the `_compile_m64` of that round: base 4.901–4.905 µs; A DMA'd into a per-wave LDS ring (`buffer_load_dwordx4 … lds`, 10 deep) instead of registers, never read back, 5.109–5.113
- S1 aklr/ops/gemm/impls/flydsl/gemm_n512_k4096/e42f6c833c101f69.json#kernel.py:46 `_PTABLE` (TM, M-blocks, TN, waves, R) per m; tile numbering at :95-96; `_compile_m64` loads D = 10 windows ahead (:215, :297-305) and starts odd M-blocks NWIN / 2 windows later (:263); kernarg preload count 6 at :377
- S2 aklr/ops/gemm/impls/flydsl/gemm_n2048_k2048/697eeb2233f25b25.json#kernel.py:132 `_compile_skw_av` (XCD numbering at :175-185, line rotation at :206-215, fold at :300-319, fragment f to wave f mod W at :307); `_compile_skw8` at :335 (whole K quarter DMA'd at :377-380); `_select_config` at :435; preload only where MT > 1 at :456; `torch.matmul` for other m at :462-467
