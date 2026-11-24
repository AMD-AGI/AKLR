---
title: "fp8 block-scaled GEMM with a row-major a_scale, shuffled B — read the scales in the caller's layout; the margin at m ≥ 2 includes the baseline's relayout"
kind: case
ops: [gemm]
dtypes: [fp8]
regimes: ["M 1-64", "M 128-512", "M 1024-4096"]
accuracy: bounded
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=1
measured:
  - {metric: "speedup over aiter's tuned fp8 block-scaled GEMM, per size", value: "1.486–5.571; 13 of 13 sizes ≥ 1.03", gpu: MI355X, workload: "N 512 × K 4096, row-major a_scale, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", date: "2026-10-28"}
  - {metric: "speedup over aiter's tuned fp8 block-scaled GEMM, per size", value: "1.375–3.298; 13 of 13 sizes ≥ 1.03", gpu: MI355X, workload: "N 1024 × K 4096, row-major a_scale, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", date: "2026-10-28"}
  - {metric: "speedup over aiter's tuned fp8 block-scaled GEMM, per size", value: "1.025–2.611; 12 of 13 sizes ≥ 1.03", gpu: MI355X, workload: "N 4096 × K 256, row-major a_scale, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", date: "2026-10-28"}
  - {metric: "speedup over aiter's tuned fp8 block-scaled GEMM, per size", value: "0.993–2.266; 12 of 13 sizes ≥ 1.03", gpu: MI355X, workload: "N 4096 × K 512, row-major a_scale, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", date: "2026-10-28"}
---

# fp8 block-scaled GEMM with a row-major a_scale and shuffled B

## Applies when

- a (m, k) and b (n, k) fp8 e4m3fn, b in aiter's 16×16 `shuffle_weight` order, and the a-scales
  row-major, `S_a[i, kb] = a_scale_flat[i·SK + kb]` with SK = K / 128, so one row's scales are contiguous. One fp32
  b-scale per 128×128 of b, `out = (a·S_a) @ (b·S_b).T`, no bias, bf16 out.
- (N, K) = (512, 4096), (1024, 4096), (4096, 256) and (4096, 512), each with its own kernel, built for
  m = 1, 2, 4, …, 4096. The same shapes with column-major a-scales are in
  [a8w8_small_n_large_k.md](../cases/a8w8_small_n_large_k.md) and
  [a8w8_n4096_k_sweep.md](../cases/a8w8_n4096_k_sweep.md).
- Accuracy: the output is compared with a reference within a tolerance (atol = rtol = 1e-2 for bf16 out), not bit
  for bit. The reductions in these four kernels run in a fixed order; run-to-run bit equality is not measured.

## Counter-signals

- **At m ≥ 2 the measured margin includes a copy in the baseline.** For this scale layout the baseline is aiter
  called with `x_scale.T.contiguous().T`, because aiter's preshuffled-B GEMM reads a row-major `x_scale`
  as column-major memory: a copy kernel inside each timed baseline call. At m = 1 the (1, SK)
  a_scale is contiguous in both orders, `.contiguous()` returns it unchanged, and no copy runs. With the same kernel
  configuration at m = 1 and m = 2 in each kernel, the four kernels read 4.569, 2.285, 1.025 and 0.993 at m = 1
  against 5.571, 3.220, 1.942 and 1.922 at m = 2 *(field, E1, E2, E3, E4)*; the column-major kernels of the same
  shapes show no such step (m = 1 / m = 2: 4.638 / 4.504, 2.383 / 2.400, 1.039 / 1.035, 0.986 / 0.996)
  *(field, E5)*. With
  aiter dbd8bf5 the copy adds about 4.1 µs to a 4.4 µs aiter call at N 4096, K 256, m = 2 and 16 *(field, E7)*, so
  N 4096 × K 256's 1.942 at m = 2 would be about 1.01 without it (estimated); at the other three shapes the copy's
  share is not measured. Against a baseline that is handed the scales already in its own layout, re-measure all
  four kernels before trusting their m ≥ 2 margins.
- **N 4096 at m = 1 is below 1.03:** N 4096 × K 256 1.025 and N 4096 × K 512 0.993 *(field, E3, E4)*. Across five
  runs of the ruler, N 4096 × K 256's m = 1 reads 1.006–1.033, on either side of
  1.03, and N 4096 × K 512's 0.989–1.009, below it in all five *(field, E6)*.
  At m = 1 N 4096 × K 512 runs its register body, four 16×16×32 fp8 MFMAs per K
  block on a 16×16 tile.
- **Cross-workgroup split-K counters live between calls.** N 512 × K 4096 (m ≤ 256) and N 1024 × K 4096 (m = 256)
  keep their arrival counters in memory the builder zeroes once per shape; the last arriver resets them. Both sum
  the splits in split order, not arrival order. A call then depends on what the call before it left, and two
  streams calling the same size at once would share the counters.

## Mechanism

What the four kernels do with the row-major scales: a row's K-block scales are adjacent, so a single 16-byte load
fetches four of them. N 512 × K 4096 and N 1024 × K 4096 (m ≥ 1024) multiply them by the b-scales once per workgroup
into an LDS table the K loop reads; N 4096 × K 512's register body loads one 16-byte vector per lane and row block
that covers all four K blocks of K = 512. The LDS table of N 512 × K 4096:

```python
# Stage the block's combined scales sa[r, kt] * sb[n_blk, kt] in LDS,
# transposed to [kt, r], once before the K loop: the K loop then reads
# them with LDS loads (lgkmcnt) and never waits on the vmem counter.
sS = fx.make_view(lds.sc.ptr, fx.make_layout((SK, tile_m), (tile_m, 1)))
for i in range_constexpr(n_scale_loads):
    idx4 = tid + i * total_threads
    r = idx4 // (SK // 4)
    kt4 = (idx4 % (SK // 4)) * 4
    sa4 = Vec(rocdl.raw_ptr_buffer_load(  # row bx_m + r, K blocks kt4..kt4+3: one 16-byte load
        v4f32, sa_rsrc, fx.Int32((bx_m + r) * (SK * 4) + kt4 * 4), fx.Int32(0)
    ))
    sb4 = Vec(rocdl.raw_ptr_buffer_load(
        v4f32, sb_rsrc, fx.Int32(sb_row_byte + kt4 * 4), fx.Int32(0)
    ))
    prod = sa4 * sb4
    for j in range_constexpr(4):
        sS[kt4 + j, r] = prod[j]
```

Not this: the column-major index `a_scale_flat[kb·m + i]` on this payload. It is how aiter's preshuffled-B GEMM
reads `x_scale`, which is why the baseline needs the relayout copy (Counter-signals).

Shared with the column-major kernels: B straight from the 16×16 preshuffled layout into MFMA operand registers on
most paths (`../gemm/datapath.md` §10); one MFMA step per 128-wide K block folded with S_a·S_b
(`../gemm/mainloop.md` §5); A through XOR-swizzled LDS (`../gemm/datapath.md` §2) on the paths that stage it, by DMA
on some paths (`../gemm/datapath.md` §5); a configuration per m from a table.

Per kernel:

**N 512 × K 4096**.
- Tiles 32×64 (m ≤ 256 and m = 1024), 32×32 with 2 waves (512), 64×64 (2048), 64×128 (4096); 4 waves along N
  otherwise; standard MFMA orientation, 2-byte stores.
- A from global into registers 3–6 K tiles ahead, then into LDS in MFMA-read order with `Swizzle<3,4,4>`, 2–4 K tiles
  per LDS stage and one barrier per stage; B into registers 2–4 K tiles ahead; the whole K loop unrolled.
- K split across workgroups: 8 ways at m ≤ 128, 4 at m = 256 (64–256 workgroups). Each thread stores its fp32 partial
  to its own workspace slot with sc0|sc1, bumps its own counter (`raw_ptr_buffer_atomic_fadd` of 1.0), and the thread
  that arrives last for its slot sums the partials in split order, writes bf16 and resets the counter
  (`../gemm/grid_pricing.md` §2, `../gemm/small_m.md` §4). The counters live between calls, zeroed once per shape:
  a call depends on what the call before it left, and two streams calling one size at once share them
  (Counter-signals). A K split kept inside the workgroup needs no counter
  ([a8w8_small_n_large_k.md](a8w8_small_n_large_k.md)).

  ```python
  def publish():  # every thread: its fp32 partial to its own workspace slot, then + 1.0 on its own counter
      cv = Vec(frag_C.load())
      for q in range_constexpr(n_v4):
          part = Vec.from_elements([cv[q * 4 + e] for e in range_constexpr(4)], Float32)
          _rocdl_d.raw_ptr_buffer_store(
              part.ir_value(), ws_rsrc,
              fx.Int32((((tile_id * KS + bid_z) * n_v4 + q) * total_threads + tid) * 16).ir_value(),
              fx.Int32(0).ir_value(),
              aux=_CPOL_SYS,  # 17: sc0|sc1
          )
      rocdl.s_waitcnt(vmcnt=0)
      return rocdl.raw_ptr_buffer_atomic_fadd(
          fx.Float32(1.0), cnt_rsrc, fx.Int32(slot * 4), fx.Int32(0), aux=17
      )

  def final_sum():  # only the thread that arrives last for its slot
      ...
          for z in range_constexpr(KS):  # split 0, 1, ...: the order does not depend on who arrived last
              v4 = Vec(rocdl.raw_ptr_buffer_load(
                  v4f32, ws_rsrc,
                  fx.Int32((((tile_id * KS + z) * n_v4 + q) * total_threads + tid) * 16),
                  fx.Int32(0),
                  aux=17,
              ))
              acc4 = v4 if acc4 is None else acc4 + v4
  ```

  Not this: the last arriver starting the sum with its own partial from registers, as the column-major
  N 4096 × K 8192 kernel does at m 64 and 128 ([a8w8_n4096_k_sweep.md](../cases/a8w8_n4096_k_sweep.md)). The
  association then depends on which split arrives last, and the output bits can differ between runs.
- K split inside the workgroup: two groups of waves at m = 512, 2048, 4096, summed through LDS in group order.

**N 1024 × K 4096** — one kernel for m ≤ 128, another above.
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

**N 4096 × K 256** — the whole K = 256 per tile, no split.
- Weights are the MFMA A operand, so an accumulator holds 4 adjacent columns of one row; a lane packs two such
  tiles into 8 adjacent columns and writes them as one 16-byte store with sc1. Activations and their scales load
  straight into registers.
- m ≤ 256: weights straight into registers; wave tiles of 32 or 64 weight columns by 16 or 32 rows, 64 to 512
  workgroups.
- m ≥ 512: each workgroup copies its 128-column weight slice into LDS once (4 KiB per 16 columns, 64-byte pad
  between them) and four waves read their fragments from there while the workgroup walks 1, 2, 4 or 8 M tiles
  (m 512, 1024, 2048, 4096), keeping the grid at 256 workgroups. Above m = 1024 each wave transposes its 16×128 output
  through a private LDS scratch (68-dword pitch) so a quarter-wave stores one 256-byte row segment.

**N 4096 × K 512**.
- m ≤ 128, register body: MFMA issued as (B, A) with four `v_mfma_f32_16x16x32_fp8_fp8` per 128-wide K block into a
  fresh accumulator, then `acc + p·(S_a·S_b)`; A and B straight from global with a 1–3 block prefetch; each lane owns
  4·tn adjacent output columns; 256 workgroups.

  ```python
  # one 16-byte load per lane and row block: the four K-block scales of row m0 + mi*16 + l16
  sa = []
  for mi in range_constexpr(tm):
      sa.append(Vec(fx.rocdl.raw_ptr_buffer_load(v4f32, rSA, fx.Int32((m0 + mi * 16 + l16) * 16), zero)))
  ...
  for kb in range_constexpr(nkb):
      a_frag, b_frag = frags.pop(kb)
      for mi in range_constexpr(tm):
          s = sa[mi][kb] * sb[kb]
          for ni in range_constexpr(tn):
              p = Vec.filled([4], 0.0, fx.Float32)  # fresh accumulator per K block
              for c in range_constexpr(2):
                  for h in range_constexpr(2):
                      p = Vec(fx.rocdl.mfma_f32_16x16x32_fp8_fp8(
                          v4f32, [b_frag[ni][c][h], a_frag[mi][c][h], p]))  # (B, A)
              acc[mi][ni] = acc[mi][ni] + p * s
  ```
- m 256, 512: 32×128 / 64×128 workgroup tile, a 3-stage LDS ring filled by buffer→LDS DMA for both A and B, each K
  block staged as 1 KiB row-contiguous chunks with an XOR-swizzled 16-byte order; one
  `v_mfma_scale_f32_16x16x128_f8f6f4` per K block; stores with sc1.
- m ≥ 1024: 128×128 tile, 2×2 waves of 4×4 MFMA tiles, the same DMA scheme on a 2-stage ring; two M tiles per
  workgroup at m = 4096; row groups 2 and 3 of the last K block go out through the free LDS stage as full 128-byte
  rows. 256 or 512 workgroups.

## Result

The baseline at every size is aiter's tuned fp8 block-scaled GEMM for preshuffled weights
(`gemm_a8w8_blockscale_bpreshuffle`). The yardstick is a one-sided lower bound (α = 0.05) of the aiter / kernel time
ratio over 32 rounds of CUDA-graph replays with L2 flushed before each call; here the
aiter time includes the a_scale copy at m ≥ 2 (Counter-signals). A size counts as won from 1.03.

These runs used FlyDSL 0.3.2, a pin installed over a machine image that ships 0.3.4.1 (not checked in the runs' logs;
`../pitfalls/measured_a_different_build.md`).

| Kernel | m 1 | m 2–64 | m 128–512 | m 1024–4096 | At 1.03 or more |
|---|---|---|---|---|---|
| N 512 × K 4096 | 4.569 *(field, E1)* | 4.677–5.571 *(field, E1)* | 2.967–4.190 *(field, E1)* | 1.486–2.532 *(field, E1)* | 13 of 13 |
| N 1024 × K 4096 | 2.285 *(field, E2)* | 2.747–3.298 *(field, E2)* | 1.678–2.560 *(field, E2)* | 1.375–1.653 *(field, E2)* | 13 of 13 |
| N 4096 × K 256 | 1.025 *(field, E3)* | 1.912–1.969 *(field, E3)* | 1.989–2.180 *(field, E3)* | 1.818–2.611 *(field, E3)* | 12 of 13 |
| N 4096 × K 512 | 0.993 *(field, E4)* | 1.866–1.924 *(field, E4)* | 1.900–2.266 *(field, E4)* | 1.661–2.125 *(field, E4)* | 12 of 13 |

50 of the 52 sizes are at 1.03 or more; all 52 pass the accuracy check.

## Evidence

- E1 [2026-10, gfx950, FlyDSL 0.3.2, N=1, one-sided lower bound (α = 0.05) of the aiter / kernel time ratio over 32 rounds of CUDA-graph replays with L2 flushed before each call, the row-major a_scale copied to aiter's layout inside each timed aiter call] speedup over aiter 1.486–5.571 per size, 13 of 13 ≥ 1.03; m 1 4.569, m 2 5.571, m 2–64 4.677–5.571, m 128–512 2.967–4.190, m 1024–4096 1.486–2.532 (N 512 × K 4096, row-major a_scale, m 1–4096)
- E2 [2026-10, gfx950, FlyDSL 0.3.2, N=1, the ruler of E1] speedup over aiter 1.375–3.298 per size, 13 of 13 ≥ 1.03; m 1 2.285, m 2 3.220, m 2–64 2.747–3.298, m 128–512 1.678–2.560, m 1024–4096 1.375–1.653 (N 1024 × K 4096, row-major a_scale, m 1–4096)
- E3 [2026-10, gfx950, FlyDSL 0.3.2, N=1, the ruler of E1] speedup over aiter 1.025–2.611 per size, 12 of 13 ≥ 1.03; m 1 1.025, m 2 1.942, m 2–64 1.912–1.969, m 128–512 1.989–2.180, m 1024–4096 1.818–2.611 (N 4096 × K 256, row-major a_scale, m 1–4096)
- E4 [2026-10, gfx950, FlyDSL 0.3.2, N=1, the ruler of E1] speedup over aiter 0.993–2.266 per size, 12 of 13 ≥ 1.03; m 1 0.993, m 2 1.922, m 2–64 1.866–1.924, m 128–512 1.900–2.266, m 1024–4096 1.661–2.125 (N 4096 × K 512, row-major a_scale, m 1–4096)
- E5 [2026-10, gfx950, FlyDSL 0.3.2, N=1, the ruler of E1, no copy in the aiter call] column-major a_scale, m 1 / m 2: N 512 × K 4096 4.638 / 4.504, N 1024 × K 4096 2.383 / 2.400, N 4096 × K 256 1.039 / 1.035, N 4096 × K 512 0.986 / 0.996 (the kernels of the same four shapes for column-major a-scales)
- E6 [2026-10, gfx950, FlyDSL 0.3.2, N=5, the ruler of E1; FlyDSL of the runs: a pin installed over the image's, not read from the runs' logs] m = 1 across five runs, in order: N 4096 × K 256 1.025, 1.006, 1.013, 1.033, 1.025; N 4096 × K 512 1.004, 0.989, 1.009, 0.996, 0.993 (the row-major N 4096 × K 256 and N 4096 × K 512 kernels)
- E7 [2026-11, gfx950 (MI355X), aiter dbd8bf5, N=5, per-kernel times of CUDA-graph replays summed per call, median] aiter's baseline copying the row-major a_scale inside the timed call / given scales copied beforehand, N 4096 × K 256 (default CK): m 2 8.456 / 4.392 µs, m 16 8.528 / 4.406 µs (aiter `gemm_a8w8_blockscale_bpreshuffle`)
