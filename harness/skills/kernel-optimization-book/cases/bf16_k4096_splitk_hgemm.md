---
title: "bf16 GEMM, N 256–2048 × K 4096, and later kernels built from the same designs for other N, K and m — K split over waves or workgroups, folded in a fixed order"
kind: case
ops: [gemm]
dtypes: [bf16]
regimes: ["M 1-256", "M 512-4096"]
accuracy: bitwise
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=1
measured:
  - {metric: "speedup over aiter's tuned bf16 GEMM, per size, one-sided lower bound over 16 rounds of CUDA-graph replays", value: "1.360–1.508; all 5 sizes at or above 1.03", gpu: MI355X, workload: "N 64 × K 4096, measured on the build before the current one, which differs only in passing the stream of the input's device; m 256, 512, 1024, 2048, 4096", date: "2026-11-09"}
  - {metric: "speedup over aiter's tuned bf16 GEMM, per size, one-sided lower bound over 16 rounds of CUDA-graph replays", value: "1.207–1.308; all 4 sizes at or above 1.03", gpu: MI355X, workload: "N 128 × K 6144, measured on the build before the current one, which differs only in passing the stream of the input's device; m 256, 512, 1024, 2048", date: "2026-11-09"}
  - {metric: "speedup over aiter's tuned bf16 GEMM, per size, one-sided lower bound over 16 rounds of CUDA-graph replays", value: "1.379–1.627; all 3 sizes at or above 1.03", gpu: MI355X, workload: "N 256 × K 6144, measured on the build before the current one, which lacks the barrier before the fold stores at m 1024; m 256, 512, 1024", date: "2026-11-09"}
  - {metric: "speedup over aiter's tuned bf16 GEMM, per size, a repeat measurement", value: "0.875–1.401; 9 of 13 sizes ≥ 1.03 (m 1–256)", gpu: MI355X, workload: "N 2048 × K 4096, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", date: "2026-11-09"}
  - {metric: "speedup over aiter's tuned bf16 GEMM, per size, a repeat measurement", value: "0.579–1.214; 5 of 13 sizes ≥ 1.03 (m 1, 2, 64, 128, 256)", gpu: MI355X, workload: "N 1024 × K 4096, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", date: "2026-11-09"}
  - {metric: "speedup over aiter's tuned bf16 GEMM, per size, a repeat measurement", value: "1.000–1.233; 11 of 13 sizes ≥ 1.03 (m 4–4096)", gpu: MI355X, workload: "N 256 × K 4096, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", date: "2026-11-09"}
  - {metric: "speedup over aiter's tuned bf16 GEMM, per size, a repeat measurement", value: "0.792–1.248; 4 of 13 sizes ≥ 1.03 (m 128, 256, 512, 2048)", gpu: MI355X, workload: "N 512 × K 4096, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", date: "2026-11-05"}
  - {metric: "speedup over aiter's tuned bf16 GEMM, per size", value: "0.858–1.400; 9 of 13 sizes ≥ 1.03 (m 1–256)", gpu: MI355X, workload: "N 2048 × K 4096, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", date: "2026-10-28"}
  - {metric: "speedup over aiter's tuned bf16 GEMM, per size", value: "0.584–1.290; 5 of 13 sizes ≥ 1.03 (m 1, 2, 64, 128, 256)", gpu: MI355X, workload: "N 1024 × K 4096, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", date: "2026-10-28"}
  - {metric: "speedup over aiter's tuned bf16 GEMM, per size", value: "0.992–1.207; 10 of 13 sizes ≥ 1.03 (m 4–64, 256–4096)", gpu: MI355X, workload: "N 256 × K 4096, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", date: "2026-10-28"}
  - {metric: "speedup over aiter's tuned bf16 GEMM, per size", value: "0.793–1.237; 5 of 13 sizes ≥ 1.03 (m 128, 256, 512, 2048, 4096)", gpu: MI355X, workload: "N 512 × K 4096, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", date: "2026-10-28"}
  - {metric: "speedup over aiter's tuned bf16 GEMM, per size, one-sided lower bound over 16 rounds of CUDA-graph replays", value: "1.153–1.269; all 7 sizes at or above 1.03", gpu: MI355X, workload: "N 512 × K 4096, the kernel for m 1–64; m 1, 2, 4, 8, 16, 32, 64", date: "2026-11-09"}
  - {metric: "speedup over aiter's tuned bf16 GEMM, per size, one-sided lower bound over 16 rounds of CUDA-graph replays", value: "1.433–1.886; all 8 sizes at or above 1.03", gpu: MI355X, workload: "N 2048 × K 2048, m 1, 2, 4, 8, 16, 32, 64, 128", date: "2026-11-09"}
---

# bf16 GEMM at N 256–2048 × K 4096, and kernels built from the same designs: split K, fold in a fixed order

Four FlyDSL kernels for `out = a @ b.T` on DeepSeek-V4 shapes, one per N, are described in § Applies when through
§ Result. Each other section adds kernels built from the same designs for other shapes or m ranges, with its own
sizes, numbers and counter-signals. "m" is M.

## Applies when

- `a` (m, k) and `b` (n, k) bf16, row-major and not preshuffled, `out = a @ b.T`, no bias, bf16 `out`. Two runs
  must give the same bytes: every reduction these kernels run at a size where they beat aiter adds in a fixed
  order.
- The four DeepSeek-V4 kernels: K = 4096 with N = 256, 512, 1024 or 2048, a weight of 2, 4, 8 or 16 MiB; each is
  built for 13 sizes, m = 1, 2, 4, …, 4096.
- Their sizes at 1.03 or more over the baseline in the latest of three runs: N 2048 m 1–256; N 1024
  m 1, 2, 64–256; N 256 m 4–4096; N 512 m 128–512 and 2048. The latest run kept N 512's values from the run
  before it: m 4096 at 1.004, below 1.03, where the first run, the one in § Result, had it above.
- The yardstick there is aiter's tuned bf16 GEMM.

## Counter-signals

- **Repacking the weight on every call does not pay at N = 1024.** At m = 4, 512, 1024 and 2048, N 1024's
  launch first packs the whole 8 MiB `b` into MFMA lane order, a second launch, and then runs a split-K GEMM on the
  packed copy: 0.584, 0.819, 0.884, 0.973 *(field, E2)*. The module docstring still
  describes a cached relayout invalidated by tensor identity or version; the code repacks on every call.
- **N 512 at m ≤ 64 and N 1024 at m 8–32, both splitting K only inside the workgroup on 32–128 workgroups, stay
  below aiter; two kernels in one run, not a rule.** N 512 at m ≤ 64 (16×16 tiles,
  4 waves splitting K, 32, 64 and 128 workgroups): 0.793–0.941 *(field, E4)*. N 1024 at m 8–32 (16×16 tiles, 4–8
  waves splitting K, 64 and 128 workgroups): 0.907–0.980 *(field, E2)*. The count alone does not decide: N 256 at
  m 4–64 runs 64 workgroups with split-K 8 or 4 across workgroups and read 1.036–1.130 in the same run
  *(field, E3)*.
- **At m ≥ 512, these results do not isolate the contribution of a K split:** N 2048
  (0.858–1.026) *(field, E1)* and N 1024 (0.819–0.973) *(field, E2)* were below 1.03 there, and N 1024 runs a
  split-K launcher at m 512–2048 (after a per-call repack); N 256 was at 1.03 or more at m 512–4096 *(field, E3)*
  and N 512 at m 512, 2048 and 4096 *(field, E4)*; later runs moved N 512's m 4096 below 1.03. They are
  not a reason to rule a K split out on another configuration; compare split and unsplit versions of the same
  pipeline before attributing either a gain or a loss to the split. N 512's kernel ported to narrower N at
  M 256–4096: `../gemm/mainloop.md` §11.
- **Arrival counters that live across calls.** N 2048's counters are zeroed once per shape in the cached builder;
  N 256's are allocated on the first call and kept in a closure list (with per-address pointer
  caches). The reducing block resets each counter, so the next call finds zeros. A call then depends on what the
  call before it left: for a new kernel, zero counters on every call
  (`../gemm/small_m.md` §4).
- **System-scope partials.** N 2048 stores and reloads its fp32 partials with `sc0 sc1`;
  `../gemm/small_m.md` §4 found agent scope sufficient across XCDs on other kernels. Not raced here.
- **Code that no measured size runs:** N 512's cross-workgroup split-K with a separate reduce kernel (its
  cross-workgroup split factor is 1 wherever the in-workgroup split runs, at m ≤ 1024, and m 2048, 4096 have
  split 1), although its docstring
  describes it; N 1024's cross-workgroup last-arriver path; N 256's bf16-atomic split-K
  (every split takes the fp32-workspace path in its place). None of them was measured.

## Mechanism

Shared by the four kernels:

- **MFMA `mfma_f32_16x16x32_bf16` into fp32 accumulators** wherever an MFMA runs (`../gemm/mainloop.md`
  §5); N 1024 at m 1 and 2 runs none.
- **Operands into LDS by buffer→LDS DMA** (`buffer_load_dwordx4 … lds`, 16 B per lane) in XOR-swizzled 16 B
  chunks, read back as 16 B fragments (`../gemm/datapath.md` §2, §5); N 1024 at m 1 and 2 loads `a` and `b` straight
  into registers instead.
- **K split for parallelism at small M, partials kept in fp32 and summed in a fixed order, one bf16 rounding**
  (`../gemm/grid_pricing.md` §2, `../gemm/small_m.md` §4, §5,
  `../gemm/correctness.md` §8). N 2048 ports aiter's split-K HGEMM but replaces its bf16 atomic
  adds into C, which follow arrival order, with an fp32 workspace and a last-arriver fold:

  ```python
  for f in range_constexpr(C_FRAGS_LEN):  # every K slice: its fp32 tile to the workspace, sc0 sc1
      buffer_store(c_frags[f], W_.rsrc, ws_own_off + f * 256, CPOL_SYS)
  ...
  old = llvm.AtomicRMWOp(  # thread 0 takes a ticket from the tile's counter
      llvm.AtomicBinOp.add,
      cnt_ptr,
      _i32c(1),
      llvm.AtomicOrdering.monotonic,
      syncscope="agent",
      alignment=4,
  ).result
  ...
  is_last = std_arith.CmpIOp(std_arith.CmpIPredicate.eq, arrived, _i32c(SPLIT_K - 1)).result
  ...  # only the slice holding the last ticket goes on
  parts = []
  for f in range_constexpr(C_FRAGS_LEN):
      for sk in range_constexpr(SPLIT_K):
          off = ws_tile_off + (sk * (N_TILES * BLOCK_MN_SIZE) + f * 256)
          is_own = arith.cmpi(arith.CmpIPredicate.eq, ks_idx, fx.Index(sk))
          off = arith.select(is_own, fx.Index(WS_FLOATS), off)  # its own slice: an out-of-range load, no traffic
          ld = buffer_load(ws_sized_rsrc, off, C_FRAG, T.f32, CPOL_SYS)
          parts.append(arith.select(is_own, c_frags[f], ld))  # and the register copy in its place
  for f in range_constexpr(C_FRAGS_LEN):
      acc = parts[f * SPLIT_K]
      for sk in range_constexpr(1, SPLIT_K):  # slices 0, 1, 2, 3 in index order, whichever arrived last
          acc = std_arith.AddFOp(_unwrap(acc), _unwrap(parts[f * SPLIT_K + sk])).result
      ...
          val = std_arith.TruncFOp(T.bf16, v_extract(acc, kk)).result  # one rounding, after the sum
  ...
  llvm.StoreOp(_i32c(0), create_llvm_ptr(cnt_addr2, address_space=1))  # the reducing block resets the counter
  ```

  Not this: every slice adding its bf16 tile into `out` with atomic adds, as aiter's split-K HGEMM does. The sum
  then follows arrival order.
- **64, 256 or 512 workgroups at every size the run of § Result put at or above 1.03** (§ Result;
  `../gemm/grid_pricing.md` §1, `../gemm/small_m.md` §3). N 256 at m 128, at 1.03 or more in the latest run, runs
  128 workgroups.
- **A per-m configuration table**, compiled on a size's first call inside a `functools.lru_cache` function.

At the sizes the run of § Result put at or above 1.03, and N 256 at m 128:

| | N 2048 × K 4096 | N 1024 × K 4096 | N 256 × K 4096 | N 512 × K 4096 |
|---|---|---|---|---|
| Path, tile, waves | m 1–8: one 8×8 output tile per workgroup, 8 waves. m 16, 32: 16×8 / 16×16 tile, 4 waves, a private 6-slot LDS ring per wave. m 64–256: the split-K HGEMM port, 32×32 / 32×64 / 64×64 tile, 4 waves, 2–4-stage LDS ring | m 1: one output column per wave, 4 waves; 16 B loads of `a` and `b` per 512-wide K chunk, `v_dot2_f32_bf16` into one fp32 chain, a 6-step `ds_bpermute` butterfly. m 2: the same, each `b` vector feeding both rows' chains. m 64–256: 16×16 / 16×32 / 32×32 tile, 4 waves | m 4–64: 16×32 tile, 1 wave (m ≤ 16), or 32×32, 2 waves. m 128–512: 32×32, 2 waves. m 1024: 32×32, 2 waves. m 2048, 4096: 32×64 / 64×64, 2 waves along N × 2 along K | m 128–512: 16×16 / 16×32 / 32×32 tile, 4 waves along K, a private 4-stage LDS ring per wave. m 2048: 64×64, 2×2 waves, 5 stages; m 4096: 128×64, 4×2 waves, 4 stages; both an unrolled ping-pong with fenced next-tile fragment reads |
| K split and fold | Inside the workgroup (m ≤ 32): wave w takes K slice w, partials summed through LDS in slice order. Across workgroups (m 64–256, split 4): each slice stores its fp32 tile to a workspace, then takes a ticket from a per-tile counter (agent-scope `atomicrmw add`); the slice holding ticket 3 sums slices 0–3 in index order (its own from registers at m 128 and 256; at m 64 all four reloaded from the workspace), rounds once, stores, resets the counter | m 1, 2: none (one wave holds all of K). m 64–256: each wave owns a contiguous quarter of K; the K loop waits only on that wave's own DMA, no `s_barrier`; partials parked in LDS, summed over the 4 waves in fixed order | Split-K 8 (m ≤ 32) or 4 (m 64–512) across workgroups: fp32 partials in MFMA fragment order, `sc1` stores, one counter per (tile, wave), or per tile at m 32 and 128; the last arriver loads slices 0…s−1 in index order, sums, resets the counter. m ≥ 2048: the two K waves fold in fp32 through LDS in fixed order | m 128–512: wave 0 sums the 4 partials through LDS in wave order. m 2048, 4096: none |
| Workgroups | 256 (m ≤ 32), 512 (m 64–256) | 256 at each of those sizes | 64 (m 4–64), 128 (m 128), 256 (m 256, 1024–4096), 512 (m 512) | 256 at each of those sizes |
| Tile order | m 32: the two workgroups that share a `b` tile on one XCD. Otherwise linear; the split-K path asserts no remap | linear | m ≥ 512: each XCD a contiguous eighth of the M tiles with all N tiles | m 128–512: each XCD a box of 8 M tiles × 4 N tiles; m 2048, 4096 linear |

XCD order: `../gemm/schedule_and_locality.md` §2, `../gemm/small_m.md` §6.

## Result

The speedup is a lower bound of the CUDA-graph time ratio aiter / kernel, against aiter's tuned bf16 GEMM. The
table is one run; bold: at or above 1.03 in that run. Per size:

These runs used FlyDSL 0.3.2, a pin installed over a machine image that ships 0.3.4.1
(`../pitfalls/measured_a_different_build.md`).

| Kernel | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 | ≥ 1.03 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N 2048 *(field, E1)* | **1.394** | **1.395** | **1.400** | **1.373** | **1.278** | **1.197** | **1.205** | **1.151** | **1.076** | 1.026 | 1.002 | 0.988 | 0.858 | 9 of 13 |
| N 1024 *(field, E2)* | **1.288** | **1.290** | 0.584 | 0.980 | 0.907 | 0.973 | **1.163** | **1.190** | **1.208** | 0.819 | 0.884 | 0.973 | 0.944 | 5 of 13 |
| N 256 *(field, E3)* | 1.000 | 1.002 | **1.101** | **1.081** | **1.130** | **1.071** | **1.036** | 0.992 | **1.109** | **1.207** | **1.153** | **1.172** | **1.057** | 10 of 13 |
| N 512 *(field, E4)* | 0.925 | 0.898 | 0.877 | 0.850 | 0.793 | 0.851 | 0.941 | **1.187** | **1.194** | **1.237** | 1.025 | **1.116** | **1.059** | 5 of 13 |

29 of the 52 sizes were at or above 1.03 in that run. In the latest run 29 are, not the same 29: N 256 also at
m 128, and N 512 not at m 4096 (1.004, a value the latest run kept from the run before it) *(field, E64)*.

## N 64–256 at m 256–4096: N 512 × K 4096's K split over waves, ported

Three kernels take N 512 × K 4096's m 128–512 path (K over the waves of one workgroup, a private LDS ring per wave;
§ Mechanism) to narrower N at m 256–4096: N 64 × K 4096 (m 256–4096, 512 KiB of
weight), N 128 × K 6144 (m 256–2048, 1.5 MiB) and N 256 × K 6144
(m 256–1024, 3 MiB). The yardstick is aiter's tuned bf16 GEMM (the older K 6144 kernels read below aiter at these
m); narrow N at small m is on [bf16_glm52_narrow_n.md](bf16_glm52_narrow_n.md). A separate run, its ruler not stated
in the record, read 1.211–1.624, all 12 sizes at or above 1.03 *(field, E65)*.
N 2048 × K 2048 at m 256–4096 starts from the N 512 × K 4096 kernel too:
[bf16_large_m_whole_k_tiles.md](bf16_large_m_whole_k_tiles.md).

Each workgroup owns one output tile. Except at N 64 m 4096, it splits its share of K over KW waves: the share is all
of K, or a half or a quarter at N 256 m 512 and 1024, where K is split across workgroups as well (below). Each wave
streams its part through its own LDS ring (`buffer_load_dwordx4 … lds` into XOR-swizzled 16 B chunks), waits only on
its own `vmcnt` and `lgkmcnt` and passes no `s_barrier` in the K loop; after one barrier the waves' fp32 partials meet
in LDS and are added in a fixed order. N 64 at m 4096 splits K four ways across workgroups only: its four waves split
the 64×64 output tile 2 × 2, each running the workgroup's quarter of K from one shared six-slot ring with a barrier
per K tile, so no wave partials meet in LDS; the workgroups' partials are folded through fp32 planes in global memory
(below). Every output is rounded to bf16 once, after the last fold. Every size launches 256 workgroups, one round
(`../gemm/grid_pricing.md` §1).

| | N 64 × K 4096 | N 128 × K 6144 | N 256 × K 6144 |
|---|---|---|---|
| Tile, K waves, BK, ring slots per wave | m 256: 8×8, 8 waves, BK 128, 3 slots; m 512: 16×8, m 1024: 16×16, 8 waves, BK 128, 2 slots; m 2048: 32×16, 4 waves, BK 64, 4 slots; m 4096: 64×64, four waves over the tile (2 × 2), BK 64, six shared slots | m 256: 8×16, 8 waves, 2 slots; m 512: 16×16, 8 waves, 2 slots; m 1024: 32×16, 4 waves, 3 slots; m 2048: 32×32, 3 waves, 3 slots; BK 128 | 4 waves, 2 slots; m 256: 16×16, BK 128; m 512: 32×32, BK 128; m 1024: 64×64, BK 64 |
| K walk | m 256–2048: a contiguous K range per wave; workgroup (x, y) starts at tile (x + R·y) mod tiles, R 5 (11 at m 2048); m 4096: each split walks its quarter of K from its start | contiguous per wave, start rotated by the tile's coordinates; at m 1024 each wave one tile further; odd column tiles walk K backwards at m 512 and 2048 | wave w takes BK tiles w, w + 4, …; the start is set per XCD, then rotated by the tile's place in its XCD box |
| Fold in the workgroup | m 256–2048: wave 0 adds the waves' partials in wave order; m 4096: none | a fixed pairwise tree ((p0 + p1) + (p2 + p3)) … at m 256–1024, wave order at m 2048; wave 0 keeps its own partial in registers at m 512; fragments spread over the waves at m 1024 and 2048 | every thread adds its outputs over the four waves in order: one column at m 256; read from a padded row-major LDS tile, 4 adjacent columns at m 512 and 4 rows × 4 adjacent columns at m 1024; at m 1024 that tile overruns the wave's own ring, so a barrier first waits for every wave to finish its K loop |
| Across workgroups | m 4096 (below) | none | m 512 split 2, m 1024 split 4 (below) |
| Tile order, C store | 2-D grid in launch order; A first in the MFMA, 2-byte stores | each XCD a box of 8 × 4 or 4 × 8 tiles; B first, 8 B stores | each XCD a box of tiles, a tile's K splits on different XCDs; 2-byte stores at m 256, 8 B with `sc1` at m 512, 1024 |
| Also | kernel arguments preloaded | preloaded; LLVM scheduler strategy per m (`max-ilp` at m 512, `iterative-ilp` otherwise); `s_setprio 3` while issuing DMA at m 256–1024; K loop unrolled at m 1024, 2048 | preloaded; at m 512 the A and B descriptors start at the tile's first row |

**The folds across workgroups** keep the fp32 partial planes and the arrival words in a `torch.empty` workspace
allocated by each call, and no workgroup waits for another:
- N 64 × K 4096 at m 4096: 64×64 tiles, all of N, K split 4 (BK 64), four waves as 2 × 2; 64 row tiles ×
  4 splits = 256 workgroups. Before the first DMA every thread loads its tile's 64-bit counter word (counters 256 B
  apart; thread 0 alone uses it) and forms a 56-bit launch tag from the dispatch id and the queue pointer. After the
  K loop, if the word's high bits are not this tag, thread 0 compare-and-swaps the value it read to `tag << 8`. The
  partials are stored `sc0 sc1`; after `vmcnt(0)` and a barrier thread 0 adds 1 (monotonic, agent scope), and the
  split that gets back `tag << 8 | 3` loads the other three planes (`sc0 sc1`), adds the four in split order, its own
  from registers, and stores.

  ```python
  _TAG_MUL = 0x9E3779B97F4A7C15  # Fibonacci-hash multiplier
  _COUNT_BITS = 8  # low bits of a counter word count arrivals, the rest hold the launch tag
  _TAG_MASK = (1 << (64 - _COUNT_BITS)) - 1
  ...
  # The launch tag and the old counter word are fetched before any DMA.
  ctr_off = fx.Int32(bid_x) * _CTR_STRIDE + CTR_OFF
  w0 = _rocdl_d.raw_ptr_buffer_load(i64, rsrc_w, _irv(ctr_off), zero, aux=_auxattr())
  did = _llvm.call_intrinsic(i64, "llvm.amdgcn.dispatch.id", [], [], [])
  qp = _llvm.call_intrinsic(ir.Type.parse("!llvm.ptr<4>"), "llvm.amdgcn.queue.ptr", [], [], [])
  qpi = _llvm.PtrToIntOp(i64, qp).result
  tag = _std_arith.AndIOp(
      _std_arith.AddIOp(did, _std_arith.MulIOp(qpi, _i64c(_TAG_MUL)).result).result, _i64c(_TAG_MASK)
  ).result
  tag_word = _std_arith.ShLIOp(tag, _i64c(_COUNT_BITS)).result
  ...  # the K loop and the partial stores
  stale = _std_arith.CmpIOp(  # the word read at the start does not carry this launch's tag
      _std_arith.CmpIPredicate.ne, _std_arith.ShRUIOp(w0, _i64c(_COUNT_BITS)).result, tag
  ).result
  if_r = _scf.IfOp(_std_arith.AndIOp(_irv(is_t0), stale).result, [], has_else=False)
  with ir.InsertionPoint(if_r.then_block):
      _llvm.AtomicCmpXchgOp(  # thread 0 swaps the value it read for tag << 8
          gptr,
          w0,
          tag_word,
          _llvm.AtomicOrdering.monotonic,
          _llvm.AtomicOrdering.monotonic,
          syncscope="agent",
      )
      _scf.YieldOp([])
  # Every partial store of this workgroup is acknowledged before it counts in.
  _asm("s_waitcnt vmcnt(0)\n\ts_barrier", "~{memory}")
  ...
      old = _llvm.atomicrmw(
          _llvm.AtomicBinOp.add, gptr, _i64c(1), _llvm.AtomicOrdering.monotonic, syncscope="agent"
      )
      last = _std_arith.CmpIOp(  # the split that gets back tag << 8 | S - 1 folds
          _std_arith.CmpIPredicate.eq, old, _std_arith.OrIOp(tag_word, _i64c(S - 1)).result
      )
  ```

  Not this: `tag = did`, the bare dispatch id (Counter-signals below).
- N 256 × K 6144 at m 512 (split 2): each workgroup folds its waves, then stores its partial with `sc1`. Once a wave's
  stores are acknowledged its lane 0 exchanges the launch tag into the wave's 8 B word of the tile's 128 B slot; the
  wave that reads back this launch's tag arrived second, swaps a sentinel back and adds its partner's partial to its
  own, kept in registers. At m 1024 (split 4) thread 0 counts in with a compare-and-swap chain on the slot's word,
  where a word with another tag counts as no arrival; the fourth arrival loads the four planes (`sc1`), adds them in
  split order and stores, and the word goes back to the sentinel. The tag is the low 59 bits of
  dispatch id · H1 + queue address · H2, with odd 64-bit multipliers H1 and H2.

Counter-signals:
- **These folds read scratch this call has not written.** A workspace from `torch.empty` holds undefined bytes, and
  § Counter-signals above asks for counters zeroed in each call. N 64 at m 4096 and N 256 at m 512 and 1024 instead
  read words of an uninitialised
  `torch.empty` workspace and rely on the launch tag: the output is right only while no leftover word carries this
  launch's tag (N 64: the high 56 bits of the counter word; N 256: the 59-bit tag), and while the tag differs between
  launches that can be given the same workspace. Checked on every size of the three kernels: at 20 tries each,
  streams limited to 16 and 64 CUs and a second concurrent call on the same inputs; one graph replayed 24 times with
  three input sets rotated into its buffers; and two calls at once on
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
  overlapping the hashed-tag build's *(field, E62)*.
- **Not covered:** N 128 at m 4096 and N 256 at m 2048–4096 are not in these kernels; the other kernel of each
  K 6144 shape covers them ([bf16_glm52_narrow_n.md](bf16_glm52_narrow_n.md)).

Measured as a one-sided lower bound of the aiter / kernel time ratio over 16 rounds of CUDA-graph replays with L2
flushed before each call, on one MI355X: N 64 1.360–1.508 *(field, E60)*, N 128
1.207–1.308 *(field, E61)*, N 256 1.379–1.627 *(field, E62)* over aiter, each measured on the build before the
current one (N 256 without the m 1024 barrier), each highest at m 512; every size passed the accuracy check.

## N 512 at m 1–64: both operands in registers, K split over two waves

A second N 512 × K 4096 kernel covers m 1–64, where the first (16×16 tiles, four waves
splitting K through LDS rings, 32–128 workgroups) stays below aiter (§ Counter-signals). A separate run, its ruler
not stated in the record, read 1.139–1.269 at the seven sizes *(field, E66)*. An m outside the kernel's table goes
to `torch.matmul`; no measured size reaches that.

- **One round of small tiles.** Every size launches 256 workgroups of two waves: m 1–4 one M-block with 2-column
  tiles, m 8 and 16 two M-blocks of 4 or 8 rows with 4-column tiles, m 32 and 64 four M-blocks of 8 or 16 rows with
  8-column tiles. The M-blocks of one B tile get workgroup ids 8 apart, so that they share an XCD
  (`../gemm/small_m.md` §3, `../gemm/schedule_and_locality.md` §2).
- **Packed MFMA operands.** Each 16-row operand of `mfma_f32_16x16x32_bf16` holds R real rows (of A, or B's columns)
  times S = 16 / R K slices, so no loaded byte is padding; of the 16×16 result only the entries (i, j) with
  i mod S = j mod S are partials of real outputs. B is the first operand.

  ```python
  lr = l % 16  # the lane's row of the 16-row MFMA operand
  g = l // 16
  q = lr // S  # which of the R real rows
  sg = lr % S  # which of the S K slices
  voff = arith.index_cast(
      T.i32, (fx.Index(q) * k + fx.Index(wid) * KW + fx.Index(g) * (S * 8) + fx.Index(sg) * 8) * 2
  )
  ...
  acc[sa][sb] = rocdl.mfma_f32_16x16x32_bf16(
      T.vec(4, T.f32), [bv[win][sb], av[win][sa], acc[sa][sb], 0, 0, 0]
  )
  ...
  # Stage 1: pair t = o * W + w sums its S slices in ascending order.
  s_ = None
  for sig in range_constexpr(S):
      i = qi * S + sig  # entry (i, j) of the 16x16 result with i % S == j % S
      j = qj * S + sig
      fidx = (base + j + (i // 4) * 16) * 4 + i % 4
      v = llvm.LoadOp(T.f32, _lds_ptr(red_i64, fidx * 4), alignment=4).result
      s_ = v if s_ is None else std_arith.AddFOp(s_, v).result
  ```
- **No LDS ring.** Each wave loads its half of K for A and B with 16 B `buffer_load`s straight into registers. This
  departs from `../gemm/small_m.md` §5, where B straight to registers was 2.6× slower than B through LDS on o_proj
  (m 8, N 6144, K 16384). Here no full LDS ring was built; at m 64, DMA-ing A into a per-wave LDS ring instead of
  registers, without reading it back, read 5.11 against 4.90 µs *(field, E9)*. At m ≤ 32 every
  load is issued before the first MFMA; at m = 64 the loads run 10 of the wave's 16 K windows ahead
  of the MFMAs, and odd M-blocks start 8 windows later than even ones, so that the even and odd M-blocks of a B tile
  do not read the same lines at once.
- **Fold in a fixed order.** At m ≤ 32 the partials go through LDS: one pass adds each output's K slices in
  ascending order, a second adds the two waves in order, then one bf16 rounding and a 4 or 8 B store. At m = 64 the
  K slices of an output sit in one lane quad and are added in registers with DPP broadcasts; wave 1 passes its sums
  to wave 0 through LDS, and wave 0 adds and stores 2-byte values (`../gemm/correctness.md` §8).
- The three pointer arguments are preloaded into SGPRs (`amdgpu-kernarg-preload-count` 6; `../gemm/small_m.md` §6).

Measured as a one-sided lower bound of the aiter / kernel time ratio over 16 rounds of CUDA-graph replays with L2
flushed before each call, on one MI355X: 1.153–1.269 over aiter at the seven
sizes, highest at m = 32 *(field, E5)*. At m 1–8 the kernel read 4.20–4.38 µs per round and aiter 4.86–5.33
*(field, E6)*; whether this timer has a floor near there was not measured here; three timers that were checked read every
short kernel at about 4 µs or more (`../pitfalls/profiled_timer_floor.md`).
Under stress every size gave the output of the call alone *(field, E8)*.

## N 2048 × K 2048 at m 1–128: N 2048 × K 4096's in-workgroup K split at K 2048

The N 2048 × K 2048 kernel is built for m 1, 2, 4, …, 128 and runs adaptations of N 2048 × K 4096's m 1–8 path
(m ≤ 8) and of its m 16, 32 path (m 16–128) at K = 2048, 8 MiB of weight. The
yardstick is aiter's tuned bf16 GEMM; a separate run, its ruler not stated in the record, read 1.410–1.856 at the
eight sizes *(field, E67)*. Every size is one launch of
256 workgroups of four waves, each wave owning a 512-wide quarter of K. B is the MFMA's first operand, so a lane holds
four adjacent output columns and stores them in one 8 B write; the partials meet in LDS and are added in wave order
before one bf16 rounding, by wave 0 at m ≤ 8 and, at m 16–128, by wave f mod 4 for fragment f. Other m go to
`torch.matmul`.

- m ≤ 8: 8×8 output tiles (MFMA rows 8–15 repeat rows 0–7). No ring: each wave DMAs its whole K
  quarter of A and B (16 `buffer_load_dwordx4 … lds`) into its own LDS area before the loop and waits for it line by
  line; the partials meet in a separate LDS array and wave 0 adds them.
- m 16–128: each wave streams its quarter line by line (64 bf16 per row) through a private ring,
  refills a slot once its `ds_read`s of it retire, and waits only on its own `vmcnt`, with no `s_barrier` in the K
  loop. Tile and slots: 16×8 and 8 (m 16), 16×16 and 8 (m 32), 16×32 and 3 (m 64), 32×32 and 2 (m 128). The M-tiles
  of one B tile get workgroup ids 8 apart. At m 64 and 128, workgroup (m-tile, n-group) starts at K line
  2·m-tile + 3·n-group mod 8 (`../gemm/small_m.md` §6). Kernel arguments are preloaded only where a B tile has
  several M-tiles (m 32–128).

Measured with the same ruler as the two sections above, on one MI355X: 1.433–1.886 over aiter, highest at m = 32
*(field, E7)*. Under stress every size gave the output of the call alone *(field, E8)*.

## Evidence

- E60 [2026-11, gfx950, FlyDSL 0.3.2, N=16, one-sided lower bound of the aiter / kernel time ratio over 16 rounds of CUDA-graph replays with L2 flushed before each call, one MI355X, aiter 0.1.21.dev66+gdbd8bf5bd] N 64 × K 4096 on the build before the current one, which differs only in passing the stream of the input's device: 1.431, 1.508, 1.502, 1.414, 1.360 at m 256, 512, 1024, 2048, 4096; accuracy passed
- E61 [2026-11, gfx950, FlyDSL 0.3.2, N=16, the ruler of E60] N 128 × K 6144 on the build before the current one, which differs only in passing the stream of the input's device: 1.258, 1.308, 1.253, 1.207 at m 256, 512, 1024, 2048; accuracy passed
- E62 [2026-11, gfx950, FlyDSL 0.3.2, N=16, the ruler of E60] N 256 × K 6144 on the build before the current one, which lacks the barrier before the fold stores at m 1024 (m 256 and 512 compile to the same code): 1.379, 1.627, 1.449 at m 256, 512, 1024, kernel medians 6.35, 7.98, 10.95 µs, worst violation fraction 0; intervals 1.379–1.398, 1.627–1.635, 1.449–1.469. The same kernel with the bare dispatch id as its tag, same ruler, two runs: 1.394 and 1.393, 1.625 and 1.625, 1.460 and 1.453; intervals 1.394–1.405, 1.625–1.634, 1.460–1.482 in the first; kernel medians 6.33, 7.98, 10.72 µs
- E63 [2026-11, gfx950, FlyDSL 0.3.2, N=20, eager and op-level CUDA-graph stress on one MI355X, each try compared byte for byte with an eager call alone] the N 64 × K 4096, N 128 × K 6144 and N 256 × K 6144 builds of E60–E62, every size: streams limited to 16 and 64 CUs; a second concurrent call on the same inputs; one captured graph replayed on the same inputs, 20 tries each; one graph replayed 24 times with three input sets rotated into its buffers; two calls at once on two streams with different inputs, unrestricted and both on the same 64 CUs, 20 tries each: 0 tries differ, no hang
- E64 [2026-11, gfx950, FlyDSL 0.3.2 (the pin of aiter dbd8bf5, installed over the image for these runs), N=1, repeat measurements of E1–E4, the later one with the ruler of E60] sizes at or above 1.03: N 2048 m 1–256 (0.875–1.401), N 1024 m 1, 2, 64–256 (0.579–1.214), N 256 m 4–4096 (1.000–1.233; m 128 1.031) in the later run; N 512 m 128–512 and 2048 (0.792–1.248; m 4096 1.004) from the run before it, whose N 512 values the later run kept
- E1 [2026-10, gfx950, FlyDSL 0.3.2, N=1, lower bound of the CUDA-graph time ratio aiter / kernel] speedup over aiter per size 0.858–1.400, 9 of 13 sizes ≥ 1.03 (m 1–256) (N 2048 × K 4096, m 1–4096)
- E2 [2026-10, gfx950, FlyDSL 0.3.2, N=1, the ruler of E1] speedup over aiter per size 0.584–1.290, 5 of 13 sizes ≥ 1.03 (m 1, 2, 64, 128, 256) (N 1024 × K 4096, m 1–4096)
- E3 [2026-10, gfx950, FlyDSL 0.3.2, N=1, the ruler of E1] speedup over aiter per size 0.992–1.207, 10 of 13 sizes ≥ 1.03 (m 4–64, 256–4096) (N 256 × K 4096, m 1–4096)
- E4 [2026-10, gfx950, FlyDSL 0.3.2, N=1, the ruler of E1] speedup over aiter per size 0.793–1.237, 5 of 13 sizes ≥ 1.03 (m 128, 256, 512, 2048, 4096) (N 512 × K 4096, m 1–4096)
- E5 [2026-11, gfx950, FlyDSL 0.3.2, N=16, one-sided lower bound of the aiter / kernel time ratio over 16 rounds of CUDA-graph replays with L2 flushed before each call, one MI355X, aiter 0.1.21.dev66+gdbd8bf5bd] N 512 × K 4096, the kernel for m 1–64: 1.166, 1.153, 1.158, 1.189, 1.210, 1.269, 1.263 at m 1, 2, 4, 8, 16, 32, 64; accuracy passed at every size
- E6 [2026-11, gfx950, FlyDSL 0.3.2, N=16, the run of E5] per-round times at m 1, 2, 4, 8: kernel 4.20–4.31, 4.22–4.38, 4.21–4.36, 4.23–4.38 µs; aiter 4.97–5.07, 4.86–5.20, 4.88–5.15, 4.98–5.33 µs
- E7 [2026-11, gfx950, FlyDSL 0.3.2, N=16, the ruler of E5] N 2048 × K 2048: 1.485, 1.561, 1.567, 1.583, 1.596, 1.886, 1.748, 1.433 at m 1, 2, 4, 8, 16, 32, 64, 128; accuracy passed
- E8 [2026-11, gfx950, FlyDSL 0.3.2, N=20, eager and op-level CUDA-graph stress on one MI355X, each try compared byte for byte with an eager call alone] the N 512 × K 4096 kernel for m 1–64 and N 2048 × K 2048, every size: streams limited to 16 and 64 CUs; a second concurrent call on the same inputs; one captured graph replayed on the same inputs, 20 tries each; one graph replayed 24 times with three input sets rotated into its buffers; two calls at once on two streams with different inputs, unrestricted and both on the same 64 CUs, 20 tries each: 0 tries differ; a full model's decode graph was not run
- E9 [2026-11, gfx950, FlyDSL 0.3.2, N=3, an in-process probe, 2–3 alternating rounds, each the median of 10 graph readings; outputs wrong by design] N 512 × K 4096 at m 64, the kernel for m 1–64 as it stood at the time: base 4.901–4.905 µs; A DMA'd into a per-wave LDS ring (`buffer_load_dwordx4 … lds`, 10 deep) instead of registers, never read back, 5.109–5.113
- E65 [2026-11, gfx950, FlyDSL 0.3.2, N=?, a separate run; ruler not stated in the record] N 64 × K 4096: 1.333, 1.414, 1.480, 1.445, 1.386 at m 256, 512, 1024, 2048, 4096; N 128 × K 6144: 1.261, 1.310, 1.263, 1.211 at m 256, 512, 1024, 2048; N 256 × K 6144: 1.396, 1.624, 1.452 at m 256, 512, 1024; all 12 sizes at or above 1.03, accuracy passed at each
- E66 [2026-11, gfx950, FlyDSL 0.3.2, N=?, a separate run; ruler not stated in the record] N 512 × K 4096, the kernel for m 1–64: 1.141, 1.139, 1.142, 1.174, 1.184, 1.268, 1.269 at m 1, 2, 4, 8, 16, 32, 64; accuracy passed at each
- E67 [2026-11, gfx950, FlyDSL 0.3.2, N=?, a separate run; ruler not stated in the record] N 2048 × K 2048: 1.410, 1.517, 1.515, 1.535, 1.544, 1.856, 1.752, 1.437 at m 1, 2, 4, 8, 16, 32, 64, 128; accuracy passed at each
