---
title: "fp8 block-scaled GEMM, N 512–1536 × K 4096, shuffled B — K split over the waves of one workgroup"
kind: case
ops: [gemm]
dtypes: [fp8]
regimes: ["M 1-64", "M 128-512", "M 1024-4096"]
accuracy: bounded
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=1
measured:
  - {metric: "speedup over aiter's tuned fp8 block-scaled GEMM, per size", value: "1.278–4.638; 13 of 13 sizes ≥ 1.03", gpu: MI355X, workload: "N 512 × K 4096, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", date: "2026-10-28"}
  - {metric: "speedup over aiter's tuned fp8 block-scaled GEMM, per size", value: "0.966–2.400; 12 of 13 sizes ≥ 1.03", gpu: MI355X, workload: "N 1024 × K 4096, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", date: "2026-10-28"}
  - {metric: "speedup over aiter's tuned fp8 block-scaled GEMM, per size", value: "1.323–4.461; 13 of 13 sizes ≥ 1.03", gpu: MI355X, workload: "N 1536 × K 4096, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", date: "2026-10-28"}
---

# fp8 block-scaled GEMM at N 512–1536, K 4096: split K inside the workgroup

## Applies when

- a (m, k) and b (n, k) fp8 e4m3fn, b in aiter's 16×16 `shuffle_weight` order, one fp32 a-scale per 1×128 of a,
  read from a column-major payload (`S_a[i, kb] = a_scale_flat[kb·m + i]`), one fp32 b-scale per 128×128 of b,
  `out = (a·S_a) @ (b·S_b).T`, no bias, bf16 out.
- Few output columns and a long K: N = 512, 1024 or 1536 with K = 4096 (32 scale blocks along K). Each (N, K)
  pair has its own kernel, built for m = 1, 2, 4, …, 4096.
- At m ≤ 16 a grid of 16×16 output tiles has only 32 (N 512), 64 (N 1024) or 96 (N 1536) workgroups, one per CU,
  so 160–224 of the 256 CUs stay idle (`../gemm/grid_pricing.md` §1, `../gemm/small_m.md` §3). The K split stays
  inside each workgroup and adds waves to those CUs, not work to the others: confined to exactly that many CUs, the
  kernels take the same time *(field, E5)*.
- Accuracy: the output is compared with a reference within a tolerance (atol = rtol = 1e-2 for bf16 out), not bit
  for bit. Every reduction in these three kernels runs in a fixed order with no atomics on data; run-to-run bit
  equality is not measured.

## Counter-signals

- **N 1024 at m = 4096 is below aiter: 0.966** *(field, E2)*, and below 1.0 in each of five runs of the ruler
  *(field, E4)*. That size runs the N 1024 × K 4096 kernel's unsplit main body: a 64×128
  tile, four waves along N, A written from registers into a 2-stage swizzled LDS ping-pong, B into registers two
  tiles ahead, no K split, 2-byte stores, 512 workgroups. At m = 4096 N 512 × K 4096 keeps its four-wave K split
  (64×128 tile, A by DMA, 8-byte stores) and N 1536 × K 4096 runs a 64×192 tile whose two waves share one A tile in
  LDS, with transposed 8-byte stores; they read 1.278 and 1.449
  *(field, E1, E3)*.
- **The margin shrinks as m grows.** From 3.802–4.638 (N 512), 1.999–2.400 (N 1024) and 3.233–4.461 (N 1536) at
  m ≤ 64 to 1.278–1.959, 0.966–1.416 and 1.323–1.449 at m ≥ 1024 *(field, E1, E2, E3)*.
- **Cross-workgroup split-K is present in the N 512 and N 1024 kernels but no measured size reaches it.** N 512 ×
  K 4096 keeps a last-arriving-wave fused split-K with a separate reduce kernel; N 1024 × K 4096 a
  last-arriving-split path. Both keep their arrival counters in memory the builder zeroes once per shape
  (`torch.zeros`), relying on the counter arithmetic or the last arriver to leave it reusable. A call then depends
  on what the call before it left, so a variant that needs these paths needs that rework first.

## Mechanism

What the three kernels share:

- **One 16×16×128 fp8 MFMA per 128-wide K block**, into a zeroed fp32 partial (`v_mfma_scale_f32_16x16x128_f8f6f4`
  with unit E8M0 scales, or the same intrinsic with zero scale operands, which the N 1536 × K 4096 source notes is lowered
  to the unscaled opcode), then `acc = fma(partial, S_a·S_b, acc)`: the block scales are applied once per
  K block on the fp32 partial (`../gemm/mainloop.md` §5).
- **B straight to registers.** The 16×16 preshuffled weight is already in MFMA operand order: each lane issues two
  16-byte loads per K block into the B operand, and B does not go through LDS (`../gemm/datapath.md` §10;
  §7 on why that suits an operand each wave owns alone).
- **K split across the waves of one workgroup.** Each wave runs its own share of the 32 K blocks into its own
  accumulators; after one barrier the partials are summed through LDS in a fixed wave order and stored. No
  workspace, no counter, one launch (`../gemm/grid_pricing.md` §2, `../gemm/small_m.md` §4,
  `../gemm/correctness.md` §8). In the N 1024 × K 4096 kernel:

  ```python
  if const_expr(num_waves > 1):
      red_ptr = lds.red.ptr
      if wave != fx.Int32(0):  # waves 1, 2, ...: their fp32 partials to LDS
          wbase = (wave - fx.Int32(1)) * (n_chunks * 64 * 4)
          for c in range_constexpr(n_chunks):
              v4 = Vec.from_elements([acc[c * 4 + e] for e in range_constexpr(4)], Float32)
              fx.ptr_store(v4, fx.add_offset(red_ptr, wbase + (c * 64 + lane) * 4))
      gpu.barrier()
      if wave == fx.Int32(0):  # wave 0 adds them to its own in wave order, then stores
          parts = [acc[e] for e in range_constexpr(acc_size)]
          for w in range_constexpr(1, num_waves):
              for c in range_constexpr(n_chunks):
                  off = (w - 1) * (n_chunks * 64 * 4) + (c * 64 + lane) * 4
                  v4 = Vec(fx.ptr_load(fx.add_offset(red_ptr, off), T.vec(4, Float32.ir_type)))
                  for e in range_constexpr(4):
                      parts[c * 4 + e] = parts[c * 4 + e] + v4[e]
          finish(Vec.from_elements(parts, Float32))
  ```

  Not this: K split across workgroups, each split storing its partial plane to a workspace and bumping an arrival
  counter that lives between calls, the scheme the row-major N 512 × K 4096 kernel runs at m ≤ 256
  ([a8w8_row_major_a_scale.md](a8w8_row_major_a_scale.md), Counter-signals). A call then depends on what the call
  before it left (Counter-signals).
- **A through a wave-private LDS area for most sizes**, XOR-swizzled (N 512, N 1024) or padded to a 144-byte row
  (N 1536) (`../gemm/datapath.md` §2, §3), filled by DMA in N 512 at m ≥ 1024 and N 1024 at m 128–2048, and from
  registers otherwise (`../gemm/datapath.md` §5).
- **Configuration per m from a table**, compiled in an `lru_cache` function on a size's first call. Tile order
  across XCDs is remapped at mid and large m
  (`../gemm/schedule_and_locality.md` §2).

Per kernel:

**N 512 × K 4096** — every size takes the in-workgroup split.
- Workgroup tile 16×16 with K split over 4 waves (m 1, 2), 16 waves (m 4, 8), 8 waves (m 16–64) or 4 waves
  (m 128); 16×32 (256), 32×32 (512), 32×64 (1024), 64×64 (2048), 64×128 (4096) with 4 waves. Wave w takes a
  contiguous range of K blocks. Workgroups: 32 (m ≤ 16), 64 (32), 128 (64), 256 (m ≥ 128).
- A: m = 2 loads straight into MFMA operand registers. m = 1 and 4–512 load whole 128-byte rows (8 lanes per row)
  and re-lay them through the wave's LDS area, 16-byte chunk c of row r at chunk c ⊕ (r >> 1), read back with
  `ds_read_b128` (m 1, 4, 8 stage only 8 rows). m ≥ 1024 DMA A into the same layout (`raw_ptr_buffer_load_lds`,
  swizzle inverted on the global address) with hand-placed `s_waitcnt`, so the other slot's DMA stays in flight.

  ```python
  # LDS layout per (slot, mi): 16 rows x 128 B, chunk c of row r stored at
  # chunk c ^ (r >> 1) (conflict-free ds_read_b128).  Offsets in f32 units.
  wr = lane // 8  # the row this lane loaded, and
  wc = lane % 8   # its 16-byte chunk of that row
  st_lo = wr * 32 + fx.Int32(std_arith.XOrIOp(_uw(wc), _uw(wr // 2)).result) * 4
  st_hi = (wr + 8) * 32 + fx.Int32(std_arith.XOrIOp(_uw(wc), _uw(wr // 2 + 4)).result) * 4
  lrow = l16 % STG_ROWS  # the row this lane feeds to the MFMA
  fr = lrow // 2
  ld_lo = lrow * 32 + fx.Int32(std_arith.XOrIOp(_uw(g), _uw(fr)).result) * 4
  ld_hi = lrow * 32 + fx.Int32(std_arith.XOrIOp(_uw(g + 4), _uw(fr)).result) * 4
  ```
- Scales: one b-scale row held across lanes, picked per K block with `v_readlane`; at m ≤ 64 and m = 512 the
  S_a·S_b products of the wave's K range are formed once and picked per block with `ds_bpermute`; at m ≥ 1024 the
  wave's a-scales are staged once into LDS and the fold trails the MFMAs by 3 (m 1024, 4096) or 5 (m 2048) MFMAs.
- The MFMA is issued with B as the first operand, so the accumulator is Cᵀ: a lane holds 4 adjacent output columns,
  stored as one 8-byte bf16 store (`../gemm/epilogue.md` §3).
- Reduction: partials to LDS, barrier, wave w sums accumulators w, w + KW, … over the waves in wave order; m 1024
  and 2048 use a fixed pairwise tree; at m = 512 each wave keeps the accumulator it reduces in registers.

  ```python
  # partials -> LDS as [wave][idx][lane] v4f32 (ds_write_b128, conflict-free)
  for idx in range_constexpr(NACC):
      vec_d.StoreOp(accs[idx], lds_view, [_idx(wave * (WREG_BYTES // 4) + (idx * 64 + lane) * 4)])
  _gpu.BarrierOp()
  for q in range_constexpr((NACC + NWAVE - 1) // NWAVE):
      ridx = wave + q * NWAVE  # every wave reduces its own share of the accumulators

      def _red():
          parts = [
              vec_d.LoadOp(v4f32, lds_view, [_idx(w * (WREG_BYTES // 4) + (ridx * 64 + lane) * 4)]).result
              for w in range_constexpr(NWAVE)
          ]
          if const_expr(m in (1024, 2048)):
              # Independent pairs shorten the cross-wave addition chain.
              for level in range_constexpr(NWAVE.bit_length() - 1):
                  parts = [std_arith.AddFOp(parts[2 * w], parts[2 * w + 1]).result
                           for w in range_constexpr(NWAVE >> (level + 1))]
              s_ = parts[0]
          else:
              s_ = parts[0]
              for w in range_constexpr(1, NWAVE):  # wave 0, 1, 2, ...
                  s_ = std_arith.AddFOp(s_, parts[w]).result
  ```
- XCD order: each XCD gets an 8×4 block of tiles at m 512 and 1024, 4×8 at 2048.

**N 1024 × K 4096** — the in-workgroup split for m ≤ 2048, an unsplit body at m = 4096.
- m ≤ 2048: tile 16×16 (m ≤ 64), 32×32 (128, 256), 32×64 (512), 64×64 (1024, 2048); K split over 4 waves
  (m ≤ 4, m ≥ 128) or 8 waves (m 8–64), each a contiguous run of 8 or 4 K blocks prefetched through an 8-, 4-, 2- or
  1-deep ring; waves 1… write partials to LDS and wave 0 adds them in wave order. Workgroups: 64 (m ≤ 16),
  128 (32, 128), 256 (64, 256–1024), 512 (2048).
- A: m ≤ 4 straight into operand registers; m 8–64 row loads through registers into the wave's swizzled LDS
  (`Swizzle(3,4,4)`) and back; m ≥ 128 async DMA (`BufferLoadAsyncLDS128b` with `asyncmark` / `wait_asyncmark`)
  into a wave-private swizzled LDS ring, swizzle inverted on the global address.
- Scales: each wave loads its run of b-scales once through a uniform pointer; at m ≤ 4 also its a-scales.
- m = 4096: the unsplit 64×128 body described under Counter-signals.
- Standard MFMA orientation (`MFMA_Scale` atom through `fx.gemm`); 2-byte bf16 stores.
- XCD order from m = 256: the 8 XCDs form a 2×4 (m 256, 512) or 4×2 (m 1024, 2048) grid of tile blocks.

**N 1536 × K 4096** — a configuration per m plus flags the builder adds per m.
- Workgroup tile 16×16 (m ≤ 32), 16×32 (64), 16×64 (128), 16×96 (256), 32×96 (512), 64×96 (1024, 2048),
  64×192 (4096). K split over 4 waves (m ≤ 64, 256–1024) or 2 (128, 2048); none at 4096, where two waves sit side
  by side along N. Wave wk takes K blocks wk, wk + WK, … (interleaved). From m = 256 every size launches 1024 waves,
  one per SIMD.
- A: m = 1 loads 32 contiguous bytes of a row per lane straight into registers. m 2–2048: each K-split wave fetches
  whole 128-byte rows two K blocks ahead into its own LDS double buffer (row stride padded to 144 bytes, no
  swizzle) and reads the operand back. m = 4096: one A tile per workgroup in LDS, shared by its two waves; the next
  block's LDS stores go out before MFMA group 1 and the barrier and reads before group 2.
- The fold `acc = fma(p, S_a·S_b, acc)` trails its MFMA by 16, 8 or 6 MFMAs. The b-scale row is read once per wave
  and picked with `v_readlane` at m ≤ 128, 2048 and 4096. A 96-column wave that straddles two 128-column scale
  blocks walks its tiles rotated so the block change falls at a fixed tile.
- Stores: transposed MFMA (8-byte stores of 4 adjacent columns) at m ≤ 128 and 4096; at m 512–2048 each thread sums
  4 or 8 adjacent columns from the LDS partials and stores them packed; 2-byte stores at m = 256.
- Reduction: wave partials to LDS, summed in wave order.
- XCD order: 2-D grouping of the grid over the XCDs at m 512, 1024 (4 column groups) and 2048 (2); contiguous runs
  per XCD at 4096.

## Result

The baseline at every size is aiter's tuned fp8 block-scaled GEMM for preshuffled weights
(`gemm_a8w8_blockscale_bpreshuffle`). Each value is a
one-sided lower bound (α = 0.05) of the aiter / kernel time ratio over 32 rounds of CUDA-graph replays with L2
flushed before each call; a size counts as won from 1.03.

These runs used FlyDSL 0.3.2, a pin installed over a machine image that ships 0.3.4.1 (not checked in the runs' logs;
`../pitfalls/measured_a_different_build.md`).

| Kernel | m 1–64 | m 128–512 | m 1024–4096 | At 1.03 or more |
|---|---|---|---|---|
| N 512 × K 4096 | 3.802–4.638 *(field, E1)* | 2.430–2.922 *(field, E1)* | 1.278–1.959 *(field, E1)* | 13 of 13 |
| N 1024 × K 4096 | 1.999–2.400 *(field, E2)* | 1.289–1.553 *(field, E2)* | 1.180–1.416 (m 1024, 2048); 0.966 (m 4096) *(field, E2)* | 12 of 13 |
| N 1536 × K 4096 | 3.233–4.461 *(field, E3)* | 1.600–2.448 *(field, E3)* | 1.323–1.449 *(field, E3)* | 13 of 13 |

38 of the 39 sizes are at 1.03 or more; all 39 pass the accuracy check.

## Evidence

- E1 [2026-10, gfx950, FlyDSL 0.3.2, N=1, one-sided lower bound (α = 0.05) of the aiter / kernel time ratio over 32 rounds of CUDA-graph replays with L2 flushed before each call] speedup over aiter 1.278–4.638 per size, 13 of 13 sizes ≥ 1.03; m 1–64 3.802–4.638, m 128–512 2.430–2.922, m 1024–4096 1.278–1.959 (N 512 × K 4096, m 1–4096)
- E2 [2026-10, gfx950, FlyDSL 0.3.2, N=1, the ruler of E1] speedup over aiter 0.966–2.400 per size, 12 of 13 sizes ≥ 1.03; m 1–64 1.999–2.400, m 128–512 1.289–1.553, m 1024–2048 1.180–1.416, m 4096 0.966 (N 1024 × K 4096, m 1–4096)
- E3 [2026-10, gfx950, FlyDSL 0.3.2, N=1, the ruler of E1] speedup over aiter 1.323–4.461 per size, 13 of 13 sizes ≥ 1.03; m 1–64 3.233–4.461, m 128–512 1.600–2.448, m 1024–4096 1.323–1.449 (N 1536 × K 4096, m 1–4096)
- E4 [2026-10, gfx950, FlyDSL 0.3.2, N=5, the ruler of E1; FlyDSL of the runs: a pin installed over the image's, not read from the runs' logs] N 1024 × K 4096 at m = 4096: 0.968, 0.968, 0.958, 0.966, 0.966 in five runs (N 1024 × K 4096, m 4096)
- E5 [2026-11, gfx950 (MI355X), FlyDSL 0.3.2, torch 2.11.0+rocm10.0.0, N=6, torch profiler kernel time, median of 30 calls with L2 flushed, per round] at m 1, 8 and 16 each kernel on a stream masked to exactly as many CUs as it has workgroups (32, 64, 96) takes 0.981–1.000 of its time on all 256 CUs (N 512, N 1024 and N 1536 × K 4096)
