---
title: "fp8 block-scaled GEMM, shuffled B: N 4096 × K 256–8192, N 8192 / 32768 × K 1024, N 16384 / 65536 × K 1536, N 7168 × K 384 (m 64 and 128 only) — one structure per shape; skinny paths at large N and small m; at N 4096 with K 256, 512 and 1024, most of the 13 sizes per kernel stay below 1.03 over aiter"
kind: case
ops: [gemm]
dtypes: [fp8]
regimes: ["M 1-64", "M 128-512", "M 1024-4096", skinny, large-N, short-K]
accuracy: bounded
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=1
measured:
  - {metric: "speedup over aiter's tuned fp8 block-scaled GEMM, per size, kernel times from a rocprofiler-sdk kernel tracer", value: "1.010–1.857; 5 of 13 sizes ≥ 1.03", gpu: MI355X, workload: "N 4096 × K 256, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", date: "2026-11-09"}
  - {metric: "speedup over aiter's tuned fp8 block-scaled GEMM, per size, kernel times from Kineto", value: "0.986–1.474; 5 of 13 sizes ≥ 1.03", gpu: MI355X, workload: "N 4096 × K 512, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", date: "2026-11-05"}
  - {metric: "speedup over aiter's tuned fp8 block-scaled GEMM, per size, kernel times from a rocprofiler-sdk kernel tracer", value: "0.809–1.110; 2 of 13 sizes ≥ 1.03", gpu: MI355X, workload: "N 4096 × K 1024, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", date: "2026-11-09"}
  - {metric: "speedup over aiter's tuned fp8 block-scaled GEMM, per size, kernel times from Kineto", value: "1.044–1.302; 13 of 13 sizes ≥ 1.03", gpu: MI355X, workload: "N 4096 × K 2048, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", date: "2026-11-05"}
  - {metric: "speedup over aiter's tuned fp8 block-scaled GEMM, per size, kernel times from a rocprofiler-sdk kernel tracer", value: "1.151–2.139; 13 of 13 sizes ≥ 1.03", gpu: MI355X, workload: "N 4096 × K 4096, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", date: "2026-11-09"}
  - {metric: "speedup over aiter's tuned fp8 block-scaled GEMM, per size, kernel times from Kineto", value: "1.092–2.590; 13 of 13 sizes ≥ 1.03", gpu: MI355X, workload: "N 4096 × K 8192, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", date: "2026-11-05"}
  - {metric: "speedup over aiter's tuned fp8 block-scaled GEMM, per size, kernel times from a rocprofiler-sdk kernel tracer", value: "1.401–1.847; 13 of 13 sizes ≥ 1.03", gpu: MI355X, workload: "N 8192 × K 1024, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", date: "2026-11-09"}
  - {metric: "speedup over aiter's tuned fp8 block-scaled GEMM, per size, kernel times from a rocprofiler-sdk kernel tracer", value: "1.061–2.152; 13 of 13 sizes ≥ 1.03", gpu: MI355X, workload: "N 32768 × K 1024, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", date: "2026-11-09"}
  - {metric: "speedup over aiter's tuned fp8 block-scaled GEMM, per size, kernel times from a rocprofiler-sdk kernel tracer, in a separate run", value: "1.328–1.945; 5 of 5 sizes ≥ 1.03", gpu: MI355X, workload: "N 16384 × K 1536, m 1, 2, 4, 16, 32", date: "2026-11-09"}
  - {metric: "speedup over aiter's tuned fp8 block-scaled GEMM, per size, kernel times from a rocprofiler-sdk kernel tracer, in a separate run", value: "1.026–1.092; 5 of 6 measured sizes ≥ 1.03 (the kernel is built for those 5)", gpu: MI355X, workload: "N 65536 × K 1536, m 1, 2, 4, 8, 16, 32, measured on the build before the current one (E11)", date: "2026-11-09"}
  - {metric: "speedup over aiter's tuned fp8 block-scaled GEMM, per size, kernel times from a rocprofiler-sdk kernel tracer, in a separate run", value: "1.020–1.155; 2 of 8 measured sizes ≥ 1.03 (the kernel is built for those 2)", gpu: MI355X, workload: "N 7168 × K 384, m 1, 2, 4, 8, 16, 32, 64, 128", date: "2026-11-09"}
---

# fp8 block-scaled GEMM at N 4096 across K, at large N with K 1024 and 1536, and at N 7168 with K 384

## Applies when

- a (m, k) and b (n, k) fp8 e4m3fn, b in aiter's 16×16 `shuffle_weight` order, one fp32 a-scale per 1×128 of a,
  read from a column-major payload (`S_a[i, kb] = a_scale_flat[kb·m + i]`), one fp32 b-scale per 128×128 of b,
  `out = (a·S_a) @ (b·S_b).T`, no bias, bf16 out.
- N = 4096 with K = 256, 512, 1024, 2048, 4096 or 8192 (2 to 64 scale blocks along K), N = 8192 or 32768 with
  K = 1024, N = 16384 or 65536 with K = 1536 (12 scale blocks), and N = 7168 with K = 384 (3). Each (N, K) pair
  has its own kernel. The six N 4096 kernels and the two K 1024 kernels are built for m = 1, 2, 4, …, 4096;
  N 16384 × K 1536 for m = 1, 2, 4, 16, 32 (every m its workload has), N 65536 × K 1536 for m = 1, 2, 4, 8, 16 and
  N 7168 × K 384 for m = 64, 128. The workloads' other sizes, N 65536 at m = 32 and N 7168 at
  m = 1–32 and 256, have no kernel and run aiter.
- The yardstick is aiter's tuned fp8 block-scaled GEMM for preshuffled weights, and this page counts a size as won
  from 1.03 over it: 77 of the 104 sizes of the eight kernels measured first, and all 12 of the sizes the three
  newer kernels are built for *(field, E18)* (Result).
- Accuracy: the output is compared with a reference within a tolerance (atol = rtol = 1e-2 for bf16 out), not bit
  for bit. One kernel's split-K depends on it (Counter-signals).
- Read the section for your shape before reusing a design: the eleven kernels are eleven different kernels, not
  one kernel tuned eleven ways. N 16384 × K 1536 and N 65536 × K 1536 are built like N 32768 × K 1024's skinny path
  (m ≤ 64), and N 7168 × K 384 like N 4096 × K 256's whole-K tile.

## Counter-signals

- **N 4096 with K 256, 512 or 1024 at small m is at or near aiter.** N 4096 × K 256 reads 1.010–1.026 at m ≤ 64 and
  1.022 at m = 128 *(field, E1)*. N 4096 × K 512 reads 0.986–1.000 at m ≤ 128 *(field, E2)*; it splits K nowhere, and
  at m ≤ 64 one 16×64 tile holds all of K (64 workgroups at m ≤ 16). N 4096 × K 1024 clears 1.03 in the run of the
  Result table only at m = 2048, 4096 (1.105, 1.110); its unsplit four-wave body reads 0.809–1.005 at m = 1, 2 and
  32–1024, and its eight-wave in-workgroup K split at m = 4, 8, 16 reads 0.998–1.000 in that run after 1.073–1.080
  in the run before it *(field, E3, E9)* and 1.04–1.05 in a paired run where the Kineto-timed and the tracer-timed
  rulers (Result) agreed within 0.5% *(field, E15)*. There the kernel takes
  4.35–4.37 µs, about an empty kernel's 4.34 µs under the same timer, so the ratio only measures how far aiter sits
  above that floor.
- **N 7168 with K 384 at m ≤ 32 reads 1.020–1.028** *(field, E12)*. The kernel's time is flat in m, 4.23–4.31 µs
  from m = 1 to 128, and so is aiter's up to m = 32, 4.38–4.41 µs; the margin opens only where aiter's time grows,
  4.47 µs at m = 64 and 4.90 µs at m = 128 *(field, E12)*. At m ≤ 16 the kernel runs a 16×16 one-wave tile (448
  workgroups), at m = 32 a 16×64 four-wave tile (224 workgroups). The workload's m = 256 was not measured.
- **Sizes within a few percent of 1.03 move across it between runs.** Across the six to ten runs of the ruler on
  each kernel (the earliest on an earlier version of the ruler, the middle ones timed through Kineto, the latest
  through a rocprofiler-sdk kernel tracer): N 4096 × K 256 m ≤ 64
  0.986–1.043, at or above 1.03 only in the first run, and m = 128 1.002–1.038; N 4096 × K 512 m = 256 0.997–1.044;
  N 4096 × K 1024 m = 4, 8, 16 0.998–1.085, below 1.03 in four of ten runs, the latest among them, and m = 32
  1.004–1.032; N 4096 × K 2048 m = 2048 1.023–1.073; N 4096 × K 8192 m = 4096 1.005–1.100; N 32768 × K 1024 m = 64
  1.016–1.026 in five runs, then 1.061 in the latest *(field, E9)*. Which side of 1.03 such a size lands on depends
  on the run. Between the two runs of the three newer kernels cited here the ruler changed: kernel times read
  through Kineto, then through the rocprofiler-sdk kernel tracer. Every kernel and aiter time read
  0.2–3.2% higher in the second run, the ratios moved by −1.2% to +1.0%, and N 7168 × K 384 m = 8 (1.031 → 1.028) and
  N 65536 × K 1536 m = 32 (1.035 → 1.026) crossed 1.03 *(field, E13, E11, E12)*.
- **The skinny path's margin shrinks at the m where it widens to 128 columns per workgroup, but the width is not the
  main cause.** N 32768 × K 1024 at m = 64 (64-row slab, 8 waves, 256 workgroups) reads 1.061 in the run of the
  Result table, against 1.319–1.500 at m ≤ 32 with 16- or 32-row slabs and 64 columns (512 workgroups), and
  1.016–1.026 in the five runs before *(field, E8, E9)*; compiled with 64 columns at m = 64 it is about 1% slower
  *(field, E16)*. N 65536 × K 1536 at
  m = 32 (32-row slab, 8 waves, 512 workgroups, 4 K blocks of weights in flight instead of 8) reads 1.026 against
  1.047–1.092 at m ≤ 16; its time rises from 16.66 µs at m = 16 to 18.60 µs, aiter's from 18.22 to 19.09 µs
  *(field, E11)*; with 64 columns there it is about 2% faster, a third of that drop *(field, E16)*.
- **At N 65536 the margin is 5–9%, under a read-roof cap.** At m ≤ 16 the kernel reads its 100.7 MB of weights in
  16.50–16.66 µs, an estimated 6.0–6.1 TB/s; aiter, at 17.34–18.22 µs, reaches an estimated 5.5–5.8 TB/s
  *(field, E11)*. Under the same timer the fastest of 17 read-only kernels over the same 100.7 MB reached 6.1 TB/s
  (6.8 TB/s only over ten times the bytes), so the kernel is within about 2% of a pure read at this size, and a kernel
  at that speed would be an estimated 1.06–1.11 over aiter at these sizes *(field, E17)*. N 16384 × K 1536's margins
  do not carry over.
- **N 16384 × K 1536's margin follows aiter's time, not the kernel's.** The kernel takes 5.17–5.61 µs at all five
  sizes; aiter takes 9.70–10.73 µs at m = 1, 4, 16 (1.868–1.945) but 6.91–7.92 µs at m = 2, 32 (1.328–1.408)
  *(field, E10)*. For this shape aiter 0.1.21.dev66+gdbd8bf5bd's tuned table names its Triton kernel at m = 1, 4, 16
  and a CK kernel at m = 2, 32 (S12). The a8w8 lookup, `get_CKGEMM_config`, tries M exactly and then padded, as
  `../gemm/small_m.md` §7 describes for the bf16 table (S13). Another aiter build or table moves these numbers:
  re-measure after an aiter upgrade.
- **N 4096 × K 8192 at m = 64 and 128 reduces its split-K partials in arrival order.** Split-K 4 across workgroups:
  each split stores its fp32 partial, bumps an agent-scope counter, and the last to arrive adds its own partial
  first and then the others from split index (own + 1) mod 4 onwards. The association depends on which split
  arrives last, so the output bits can differ between runs. The accuracy bar here is a tolerance, so this passes;
  where two runs must give the same bytes these sizes need a fixed order (`../gemm/correctness.md` §8,
  `../gemm/small_m.md` §4). At m = 256 (split 2) the two-term sum does not depend on the order.

  ```python
  def finish():  # runs only in the split that arrived last
      sums = [  # its own partial first, from its registers
          Vec.from_elements([acc[4 * j + i] for i in range_constexpr(4)], Float32)
          for j in range_constexpr(n_vec)
      ]
      for d in range_constexpr(1, split_k):  # then the others, from split (own + 1) mod split_k onwards
          peer = (fx.Int32(bid_z) + d) & (split_k - 1)
          pbase = (peer * n_tiles + tile_lin) * slot_bytes
          for j in range_constexpr(n_vec):
              ld = rocdl_d.RawPtrBufferLoadOp(
                  v4ty, ws_rsrc, _iv(voff + j * 4096), _iv(pbase), aux=aux
              ).result
              sums[j] = sums[j] + Vec(ld)
  ```
- **The same split-K keeps its counters between calls**: the builder zeroes them once per shape (`torch.zeros`) and
  the last arriver subtracts the split count. A call then depends on what the call before it left, and two streams
  calling the same size at once would share the counters.

## Mechanism

What the eleven kernels share:

- **One 16×16×128 fp8 MFMA per 128-wide K block**, into a zeroed fp32 partial, folded into the fp32 accumulator
  with S_a·S_b, by one FMA in most bodies (`../gemm/mainloop.md` §5). N 4096 × K 256 and N 7168 × K 384 keep their
  two or three K-block partials apart and combine them once at the end. The one-FMA fold, in the N 65536 × K 1536
  kernel:

  ```python
  # the a-scale of row i and K block kb sits at kb * m + i of the column-major payload
  fx.copy(atom_f32x1, fx.slice(sa_div, (None, fx.Int32(kb * c_m + mt * 16 + l16))), r)
  ...
  row.append(sa_v * sb_s[kb])  # S_a·S_b per row and K block, formed before the K loop
  ...
  for k in range_constexpr(KSTEPS):  # one 128-wide K block per step
      ...
      part = mfma.call_partial([w_frag], a_frags)  # the MFMA adds into a fresh zeroed fp32 partial
      for mt in range_constexpr(MT):
          s = Vec.filled(4, sab[mt][k], fx.Float32).ir_value()
          acc[mt] = _math.fma(part[mt], s, Vec(acc[mt]).ir_value())  # acc += partial * S_a·S_b
  ```
- **B straight to registers** from the 16×16 preshuffled layout, two 16-byte loads per lane per K block
  (`../gemm/datapath.md` §10), except where noted below (N 4096 × K 4096 at m ≥ 64 and N 32768 × K 1024 at
  m ≥ 128 DMA B into LDS, where several waves share it: `../gemm/datapath.md` §7).
- **A through XOR-swizzled LDS** for most sizes (`../gemm/datapath.md` §2), written from registers
  or by DMA (`../gemm/datapath.md` §5). The skinny paths (N 32768 × K 1024 at m ≤ 64, N 16384 × K 1536,
  N 65536 × K 1536) instead store the whole activation slab once with its row stride padded by 16 bytes
  (`../gemm/datapath.md` §3).
- **S_a·S_b staged in LDS once per workgroup**, in the prologue, at some or all sizes of five kernels (N 4096 ×
  K 512, K 2048 and K 8192, N 8192 and N 32768 × K 1024), so the K loop reads scales with LDS loads only.
- **No K split at K ≤ 2048** except N 4096 × K 1024 at m 4–16. K 4096 splits inside the workgroup at m ≤ 256; K 8192
  splits inside the workgroup at m ≤ 32 and 256–512, and across workgroups at m 64–256
  (`../gemm/grid_pricing.md` §2).
- **Configuration per m from a table**, compiled in a cached function (`lru_cache`, `functools.cache`) on a size's
  first call; XCD-aware tile order in five kernels (`../gemm/schedule_and_locality.md` §2).

Per kernel:

**N 4096 × K 256** — the whole K = 256 in one tile.
- Tile 16×16 with one wave (m ≤ 16), 16×64 (32, 64), 32×64 (128, 256), 32×128 (512, 1024), 64×128 (2048, 4096)
  with four waves; 128 to 2048 workgroups.
- MFMA roles swapped (weights are the A operand): a lane holds 4 adjacent output columns.
- A straight into registers at m ≤ 4, else written from registers into XOR-swizzled LDS. With 128-column tiles,
  S_a is staged in LDS behind the A tile and S_b read by scalar loads.
- Stores: m ≤ 64 direct 8-byte stores of 4 columns; above, C transposed through XOR-swizzled LDS (its own LDS tile
  at m 512, 1024) and written as row-contiguous 16-byte stores; cache policy sc0|sc1 up to m = 1024, nt above
  (`../gemm/epilogue.md` §3, §5). From m = 2048 consecutive workgroups take adjacent column tiles.

**N 4096 × K 512** — no K split at any m.
- m ≤ 64: 16×64 tile holding all of K in one LDS stage (64, 128, 256 workgroups at m ≤ 16, 32, 64). m = 128:
  32×64, two 256-deep stages. m 256, 512: 32×128, every A K tile DMA'd (`BufferCopyLDS128b`) into its own LDS
  buffer in the prologue, B in 4 register stages. m 1024–4096: 64×128 the same way; at m 2048 and 4096 each
  workgroup walks 2 and 4 adjacent 128-column subtiles over its resident A and issues the next subtile's first MFMA
  before this subtile's epilogue. 512 workgroups from m = 512.
- S_a·S_b pre-multiplied into an LDS table in the prologue at m ≥ 4 (m 1, 2 load S_a per tile).
- Standard MFMA orientation. Stores: 2-byte on the register path (m ≤ 128); on the DMA path the bf16 subtile goes
  row-major through XOR-swizzled LDS and out as 16-byte stores with sc1.

**N 4096 × K 1024** — its own in-workgroup split kernel at m 4, 8, 16.
- m 4, 8, 16: a 16×16 tile per workgroup, 8 waves, one K block per wave, A and B straight into registers; wave 0
  sums the 8 partials from LDS in K-block order. 256 workgroups. m 4, 8 swap the MFMA operands for 8-byte stores;
  m = 16 stores 2 bytes per value.

  ```python
  # every wave: its K block's scaled fp32 partial into its own LDS slot
  p = fx.make_view(fx.add_offset(lds.partials.ptr, fx.make_int_tuple(wave * 256 + lane * 4)), fx.make_layout(4, 1))
  p.store(total)
  gpu.barrier()
  if wave == 0:  # wave 0 adds the slots in K-block order, the same order in every run
      sums = Vec.filled(4, 0.0, Float32)
      for kb in range_constexpr(split_k):
          q = fx.make_view(fx.add_offset(lds.partials.ptr, fx.make_int_tuple(kb * 256 + lane * 4)), fx.make_layout(4, 1))
          sums = sums + Vec(q.load())
  ```
- Other m: four waves along N, A written from registers into a 2-stage XOR-swizzled LDS ping-pong (async DMA,
  `BufferLoadAsyncLDS128b`, at m = 256), B into registers double-buffered, FlyDSL `preshuffle_gemm`-style
  MFMA/VMEM/DS interleave hints (`../gemm/schedule_and_locality.md` §3,
  `../flydsl/gemm.md` § mainloop §8). Tiles 16×64×256 (m ≤ 32), 16×128×256 (64, 128), 32×128×256 (256),
  64×128×256 (512), 64×256×128 (m ≥ 1024); 64 to 1024 workgroups; 2-byte stores.

**N 4096 × K 2048** — no K split.
- Tiles 16×64×512 (m ≤ 64), 32×64×512 (128, 256), 64×64×512 (512), 64×128×256 (1024), 128×128×128 (2048,
  4096); four waves along N; 64 to 1024 workgroups.
- A written from registers into a 2-stage XOR-swizzled LDS ping-pong, B into registers; S_a·S_b for all 16 K
  blocks in LDS; interleave hints at m ≤ 1024. At 32 ≤ m ≤ 1024 the N tiles run along grid x, so workgroups
  sharing a B panel land on one XCD.
- Epilogue: lane pairs swap bf16 halves with one DPP `quad_perm`, then each lane stores 4 bytes (2 adjacent
  columns) per row (`../gemm/epilogue.md` §3).

**N 4096 × K 4096** — every path issues the MFMA transposed and stores 8 bytes (4 adjacent columns).
- m ≤ 32: 16×16 (m ≤ 16) or 16×32 (32) workgroup tile, A and B straight into registers, K split over 8 waves
  (m 1, 2) or 4 (m 4–32); tile i is reduced by wave i mod KS from the others' LDS slots in a fixed order.
- m 64–256: two independent halves of K per workgroup, each with its own async-DMA LDS ring for A and B
  (`BufferLoadAsyncLDS128b`); the halves hand partials over through the then-idle B ring. Tiles 32×32, 32×64,
  64×64.
- m ≥ 512: 64×128 (512) or 128×128 tile, A and B DMA'd into a 5-, 3-, 2-, 2-stage LDS ring (m 512, 1024, 2048,
  4096), A XOR-swizzled on the global side so each lane's DMA slot stays linear; at m 2048 and 4096 each XCD gets a
  compact patch of tiles.
- 256 workgroups up to m = 1024, 512 at 2048, 1024 at 4096.

**N 4096 × K 8192** — a skinny kernel of its own at m ≤ 32.
- m ≤ 32: 16×16 or 32×16 tile, four waves each taking 16 contiguous K blocks; A by async DMA
  (`RawPtrBufferLoadAsyncLdsOp`) into a wave-private swizzled LDS ring at m ≥ 2, B into registers; the waves'
  partials summed from LDS in wave order. 256 workgroups.
- m 64, 128: 64×64 / 64×128 tiles with split-K 4 across workgroups (Counter-signals). m 256: split-K 2 with two
  4-wave groups per workgroup taking alternate K blocks of the workgroup's share, summed through LDS. m 512: the two groups
  without the cross-workgroup split. m 1024: 64×128 unsplit; m 2048, 4096: 64×256, its two 128-column halves
  sharing the A reads.
- Main body: A written from registers into 2-stage LDS, B in a register ring, S_a·S_b in LDS, and a block order
  that gives each XCD contiguous (N tile, split) panels so all K slices of a tile share one L2.

**N 8192 × K 1024** — no K split; four waves.
- Tiles 32×64 (m ≤ 64), 64×64 (128), 64×128 (256, 512), 128×128 (m ≥ 1024); 128 to 2048 workgroups.
- MFMA transposed at m ≤ 64 and m ≥ 256 (a lane's 4 adjacent columns share one combined scale); standard at
  m = 128. At m ≤ 16 only the first 16-row fragment is computed.
- A from registers into LDS at m ≤ 64; from m = 128 async DMA into a 2- or 3-slot LDS ring whose rows are permuted
  and XOR-swizzled; B into registers, two K blocks ahead below m = 1024, one from m = 1024.
- Combined S_a·S_b per (K block, row) staged in LDS in the prologue; XCD-aware block order. Stores: m ≥ 256
  through an XOR-swizzled LDS epilogue as 16-byte row stores with nt; m ≤ 64 4 adjacent bf16 per lane directly;
  m = 128 2 bytes per value.

**N 32768 × K 1024** — no K split.
- m ≤ 64: weights as the MFMA A operand loaded straight into registers; the activation slab (all of K for 16, 32 or
  64 rows) goes through LDS once with a row stride padded by 16 bytes, behind one barrier; each wave owns one
  16-column tile over all of K.
- m = 128: 8 waves, 128×128 tile, 3-stage LDS ring; the two wave groups alternate, one issuing loads and LDS reads
  while the other runs its MFMAs at raised priority (`s_setprio`). m = 256: 8 waves, 128×128 with the operands
  swapped; m ≥ 512: 4 waves, 256×128 swapped, two workgroups per CU.
- From m = 128 both operands are DMA'd into LDS (`BufferCopyLDS128b`): activations XOR-swizzled, weights in their
  preshuffled order. S_a·S_b for the workgroup tile is folded into an LDS table at kernel start. Swapped tiles pair
  adjacent tiles across lanes (`permlane16_swap`) into 16-byte stores of 8 columns
  (`../gemm/epilogue.md` §3).

**N 16384 × K 1536** — N 32768 × K 1024's skinny path, compiled per m.
- 256 workgroups of four waves, one dispatch round on 256 CUs (`../gemm/grid_pricing.md` §1); each wave owns one
  16-column tile over all 12 K blocks, the workgroup 64 columns. One M block covers all rows: 16 (m ≤ 16) or 32
  (m = 32). No K split, no reduction.
- Weights are the MFMA A operand, two 16-byte loads per lane per K block straight into registers: all 12 K blocks
  issued up front at m ≤ 4, 9 in flight at m 16 and 32. Cache policy `nt` (aux 2) on these loads at m ≤ 4 and
  m = 32, the default at m = 16 (`../gemm/small_m.md` §6).
- Activations: only the loads that cover the m real rows are issued (one 16-byte load per thread at m 1, 2, two at
  m = 4; the buffer range is m·K, so the rest reads zero), into LDS at a 1552-byte row stride. At m ≤ 4 the slab
  lands behind one barrier. At m 16 and 32 it comes in six 256-byte K slices whose loads are interleaved, in K
  order, with the weight loads of the first nine K blocks; each slice is written to LDS behind its own barrier just
  before the K loop reaches it. Lanes past m read row 0; their stores fall outside the output's buffer range and are
  dropped.
- Scales: the workgroup's 12 b-scales as invariant loads through the constant address space. At m = 1 the a-scales
  come the same way; at m 2–32 one 16-byte load per thread copies the whole raw a_scale into LDS. S_a·S_b per row
  and K block is formed in registers before the K loop.
- The compile requests a preload of the five pointer arguments into SGPRs (`amdgpu-kernarg-preload-count` 5);
  whether it took is read from the ISA (`../gemm/small_m.md` §6). The launcher compiles once per m and calls the
  compiled function directly.
- Stores: a lane holds 4 adjacent columns of one row; m ≤ 16 8-byte stores; m = 32 stages C through LDS (row stride
  72 bf16) and writes 16-byte row stores.

**N 65536 × K 1536** — the same skinny path with one configuration for every m it is built for.
- m ≤ 16: 1024 workgroups of four waves, each wave one 16-column tile over all 12 K blocks; 16-row slab;
  `waves_per_eu` 4. (m = 32 would run 32 rows and 128 columns, Counter-signals.)
- Weights as the MFMA A operand, two 16-byte loads per lane per K block, 8 of the 12 K blocks in flight, every load
  `nt` (aux 2), which the source gives for a read-once weight stream.
- The slab goes from global to registers in six 16-byte loads per thread, each pass's offset in the lane's
  `voffset`; the buffer's range check returns zeros for slab rows past m (`../gemm/datapath.md` §8), then to
  LDS at a 1552-byte row stride, behind one barrier; the LDS read base passes through a `v_mov` so the per-read
  constants fold into the `ds_read` offsets.

  ```python
  RS = K + 16  # padded LDS row stride (bytes): conflict-free ds_read_b128
  ...
  def issue_w(k):  # K block k of the weights: two 16-byte loads per lane, straight into registers
      regs = []
      for h in range_constexpr(2):
          r = fx.make_rmem_tensor(fx.make_layout(16, 1), fx.Float8E4M3FN)
          fx.copy(
              atom_w16b,  # fx.rocdl.BufferCopy128b(2): cache policy nt
              fx.slice(w_div, (None, lane16)),
              r,
              soffset=fx.Int32(w_base + (k * 2048 + h * 1024)),
          )
          regs.append(r)
      return regs

  for k in range_constexpr(W):  # W = 8 K blocks issued before the slab is stored
      w_regs[k] = issue_w(k)
  ...
  for p in range_constexpr(PASSES):  # slab: registers -> LDS at row stride RS
      cid = tid + p * NTHREADS
      st_off = (cid // CH_PER_ROW) * RS + (cid % CH_PER_ROW) * 16
      ...
  gpu.barrier()  # the one barrier; every wave then reads the slab for all of K
  ...
  for k in range_constexpr(KSTEPS):
      ...
      if const_expr(k + W < KSTEPS):  # keep W K blocks of weights in flight
          w_regs[k + W] = issue_w(k + W)
  ```
- Scales: the workgroup's 12 b-scales in three 16-byte loads moved to SGPRs with `readfirstlane`, a lane's a-scales
  one 4-byte load per K block; S_a·S_b formed in registers before the K loop.
- Stores: 8-byte stores of 4 adjacent columns of one row. The launcher compiles once per m, at the first call,
  and calls the compiled function directly.

**N 7168 × K 384** — N 4096 × K 256's whole-K tile at K = 384, three K blocks.
- Weights are the MFMA A operand, loaded straight into registers for all three K blocks before the first MFMA; tokens
  are the B operand. No K split.
- m = 64: 16×64 tile, four waves of 16 columns, 448 workgroups; A copied 8 bytes per thread into XOR-swizzled LDS
  behind one barrier. m = 128: 32×64 tile, four waves, 448 workgroups; A copied 16 bytes per thread.
- Scales: a lane loads its row's a-scale and its 128-column block's b-scale per K block with raw buffer loads; the
  three K-block partials stay in separate accumulators and are scaled and summed once after the last MFMA.
- Stores with sc0|sc1 (aux 17): m = 64 8-byte stores of 4 adjacent columns straight from the accumulators, rows past
  m dropped by the output's buffer range; m = 128 transposes C through XOR-swizzled LDS (reusing the A buffer after a
  barrier) and writes 16-byte row stores (`../gemm/epilogue.md` §3, §5).

## Result

Each kernel is measured against aiter's tuned fp8 block-scaled GEMM for preshuffled weights
(`gemm_a8w8_blockscale_bpreshuffle`). The yardstick
is a one-sided lower bound of the aiter / kernel time ratio over CUDA-graph replays with L2 flushed before each call,
read at 16 rounds and, where a threshold is still undecided, again at 32, each look at α = 0.025. The table counts
the sizes at 1.03 or more.

The eight N 4096 and K 1024 rows come from two runs of that ruler on an MI355X: N 4096 × K 256, N 4096 ×
K 1024, N 4096 × K 4096, N 8192 × K 1024 and N 32768 × K 1024 from the later run, which timed the graph replays
through a rocprofiler-sdk kernel tracer; N 4096 × K 512, N 4096 × K 2048 and N 4096 × K 8192, which the later run
did not re-measure, from the run before it, which timed them through Kineto.
Both runs used ROCm/aiter@dbd8bf5 as the baseline, and the FlyDSL that aiter pins, 0.3.2 (read from the runs' setup,
not checked in the runs' logs; `../pitfalls/measured_a_different_build.md`). The N 16384 × K 1536, N 65536 × K 1536
and N 7168 × K 384 rows come from a separate run of the tracer-timed ruler on one MI355X, with FlyDSL 0.3.2 and aiter
0.1.21.dev66+gdbd8bf5bd, after 16 rounds (32 at N 7168 × K 384 m = 8); the N 65536 × K 1536 row is of the build
before the current one (E11). Another separate run of these three kernels, its ruler not stated in the record, read
1.03 or more at every size they are built for *(field, E18)*.

| Kernel | m 1–64 | m 128–512 | m 1024–4096 | ≥ 1.03 |
|---|---|---|---|---|
| N 4096 × K 256 | 1.010–1.026 *(field, E1)* | 1.022 (128), 1.058–1.189 (256, 512) *(field, E1)* | 1.240–1.857 *(field, E1)* | 5 of 13 |
| N 4096 × K 512 | 0.991–1.000 *(field, E2)* | 0.986 (128), 1.044–1.061 (256, 512) *(field, E2)* | 1.177–1.474 *(field, E2)* | 5 of 13 |
| N 4096 × K 1024 | 0.991–1.004 (1–32), 0.956 (64) *(field, E3)* | 0.809–1.005 *(field, E3)* | 0.918 (1024), 1.105–1.110 (2048, 4096) *(field, E3)* | 2 of 13 |
| N 4096 × K 2048 | 1.131–1.215 *(field, E4)* | 1.140–1.302 *(field, E4)* | 1.044–1.112 *(field, E4)* | 13 of 13 |
| N 4096 × K 4096 | 1.447–2.139 *(field, E5)* | 1.193–1.407 *(field, E5)* | 1.151–1.405 *(field, E5)* | 13 of 13 |
| N 4096 × K 8192 | 1.624–2.590 *(field, E6)* | 1.395–1.468 *(field, E6)* | 1.092–1.266 *(field, E6)* | 13 of 13 |
| N 8192 × K 1024 | 1.416–1.626 *(field, E7)* | 1.401–1.486 *(field, E7)* | 1.785–1.847 *(field, E7)* | 13 of 13 |
| N 32768 × K 1024 | 1.319–1.500 (1–32), 1.061 (64) *(field, E8)* | 1.520–1.876 *(field, E8)* | 1.814–2.152 *(field, E8)* | 13 of 13 |
| N 16384 × K 1536 (separate run) | 1.328–1.945 (1, 2, 4, 16, 32) *(field, E10)* | — | — | 5 of 5 |
| N 65536 × K 1536 (separate run) | 1.047–1.092 (1–16), 1.026 (32) *(field, E11)* | — | — | 5 of 6 |
| N 7168 × K 384 (separate run) | 1.020–1.028 (1–32), 1.055 (64) *(field, E12)* | 1.155 (128) *(field, E12)* | — | 2 of 8 |

All 104 sizes of the first eight rows pass the accuracy check, and so do the 19 of the last three (worst violation
fraction 0). The kernels of the last three rows take 5.17–5.61 µs (N 16384 × K 1536), 16.50–16.66 µs at m ≤ 16
(N 65536 × K 1536) and 4.23–4.31 µs (N 7168 × K 384) *(field, E10, E11, E12)*.

## Evidence

- E1 [2026-11, gfx950, FlyDSL 0.3.2, N=1, one-sided lower bound of the aiter / kernel time ratio over CUDA-graph replays with L2 flushed before each call, kernel times from a rocprofiler-sdk kernel tracer, aiter dbd8bf5] speedup over aiter 1.010–1.857 per size, 5 of 13 ≥ 1.03; m 1–64 1.010–1.026, m 128 1.022, m 256 1.058, m 512 1.189, m 1024–4096 1.240–1.857 (N 4096 × K 256, m 1–4096)
- E2 [2026-11, gfx950, FlyDSL 0.3.2, N=1, the ruler of E1 with kernel times from Kineto] speedup over aiter 0.986–1.474 per size, 5 of 13 ≥ 1.03; m 1–64 0.991–1.000, m 128 0.986, m 256 1.044, m 512 1.061, m 1024–4096 1.177–1.474 (N 4096 × K 512, m 1–4096)
- E3 [2026-11, gfx950, FlyDSL 0.3.2, N=1, the ruler of E1] speedup over aiter 0.809–1.110 per size, 2 of 13 ≥ 1.03; m 1 0.991, 2 0.999, 4 0.998, 8 1.000, 16 1.000, 32 1.004, 64 0.956, 128 0.809, 256 0.862, 512 1.005, 1024 0.918, 2048 1.105, 4096 1.110 (N 4096 × K 1024, m 1–4096)
- E4 [2026-11, gfx950, FlyDSL 0.3.2, N=1, the ruler of E1 with kernel times from Kineto] speedup over aiter 1.044–1.302 per size, 13 of 13 ≥ 1.03; m 1–64 1.131–1.215, m 128–512 1.140–1.302, m 1024–4096 1.044–1.112 (N 4096 × K 2048, m 1–4096)
- E5 [2026-11, gfx950, FlyDSL 0.3.2, N=1, the ruler of E1] speedup over aiter 1.151–2.139 per size, 13 of 13 ≥ 1.03; m 1–64 1.447–2.139, m 128–512 1.193–1.407, m 1024–4096 1.151–1.405 (N 4096 × K 4096, m 1–4096)
- E6 [2026-11, gfx950, FlyDSL 0.3.2, N=1, the ruler of E1 with kernel times from Kineto] speedup over aiter 1.092–2.590 per size, 13 of 13 ≥ 1.03; m 1–64 1.624–2.590, m 128–512 1.395–1.468, m 1024–4096 1.092–1.266 (N 4096 × K 8192, m 1–4096)
- E7 [2026-11, gfx950, FlyDSL 0.3.2, N=1, the ruler of E1] speedup over aiter 1.401–1.847 per size, 13 of 13 ≥ 1.03; m 1–64 1.416–1.626, m 128–512 1.401–1.486, m 1024–4096 1.785–1.847 (N 8192 × K 1024, m 1–4096)
- E8 [2026-11, gfx950, FlyDSL 0.3.2, N=1, the ruler of E1] speedup over aiter 1.061–2.152 per size, 13 of 13 ≥ 1.03; m 1–32 1.319–1.500, m 64 1.061, m 128–512 1.520–1.876, m 1024–4096 1.814–2.152 (N 32768 × K 1024, m 1–4096)
- E9 [2026-11, gfx950, FlyDSL 0.3.2, N=6–10, every run of the E1 ruler on each kernel, in order; the earliest runs on an earlier version of the ruler, the middle ones with kernel times from Kineto, the latest (E1, E3, E5, E7, E8) from the rocprofiler-sdk tracer] near-threshold sizes: N 4096 × K 256 m 1–64 0.986–1.043 (≥ 1.03 only in the first run), m 128 1.002–1.038 (ten runs); N 4096 × K 512 m 256 0.997–1.044 (seven); N 4096 × K 1024 m 4, 8, 16 0.998–1.085 (below 1.03 in the third, sixth, eighth and tenth of its ten runs; 1.073–1.080 in the ninth), m 32 1.004–1.032 (ten); N 4096 × K 2048 m 2048 1.023–1.073 (eight); N 4096 × K 8192 m 4096 1.005–1.100 (six); N 32768 × K 1024 m 64 1.016–1.026 in its first five runs and 1.061 in the sixth
- E10 [2026-11, gfx950, FlyDSL 0.3.2, N=1, the ruler of E1 in a separate run on one MI355X, 16 rounds per size, aiter 0.1.21.dev66+gdbd8bf5bd] speedup over aiter m 1 1.871, 2 1.328, 4 1.868, 16 1.945, 32 1.408; median kernel / aiter time 5.17 / 9.70, 5.20 / 6.91, 5.23 / 9.78, 5.52 / 10.73, 5.61 / 7.92 µs; worst violation fraction 0 (N 16384 × K 1536, m 1, 2, 4, 16, 32)
- E11 [2026-11, gfx950, FlyDSL 0.3.2, N=1, the ruler of E10, 16 rounds per size] speedup over aiter m 1 1.047, 2 1.056, 4 1.061, 8 1.065, 16 1.092, 32 1.026; median kernel / aiter time 16.52 / 17.34, 16.50 / 17.46, 16.52 / 17.54, 16.57 / 17.65, 16.66 / 18.22, 18.60 / 19.09 µs; worst violation fraction 0; measured on the build before the current one, which was also built for m 32; the current build moves each slab pass's offset from `soffset` into the lane's offset, the same 295 instructions in a compile-only ISA dump at m 16, and calls the compiled launcher directly (N 65536 × K 1536, m 1–32)
- E12 [2026-11, gfx950, FlyDSL 0.3.2, N=1, the ruler of E10, 16 rounds per size, 32 at m 8] speedup over aiter m 1 1.021, 2 1.020, 4 1.021, 8 1.028, 16 1.025, 32 1.027, 64 1.055, 128 1.155; median kernel time 4.31, 4.29, 4.29, 4.28, 4.27, 4.29, 4.23, 4.23 µs, aiter 4.40, 4.39, 4.38, 4.40, 4.38, 4.41, 4.47, 4.90 µs; worst violation fraction 0; measured on a build with the same kernel code that was also built for m 1–32 (N 7168 × K 384, m 1–128)
- E13 [2026-11, gfx950, FlyDSL 0.3.2, N=1, the ruler of E10 with kernel times from Kineto, 16 rounds per size, 32 at N 7168 × K 384 m 1, 8, 16] the same three kernels: N 65536 × K 1536 m 1–16 1.045–1.094, m 32 1.035; N 7168 × K 384 m 1–32 1.020–1.031 (m 8 1.031), m 64 1.055, m 128 1.154; N 16384 × K 1536 m 1, 2, 4, 16, 32 1.894, 1.333, 1.888, 1.966, 1.395; the tracer-timed runs (E10–E12) read every median kernel and aiter time 0.2–3.2% higher
- S12 ROCm/aiter@dbd8bf5:aiter/configs/model_configs/dsv4_a8w8_blockscale_bpreshuffle_tuned_gemm.csv:671 N 16384 × K 1536 on 256 CUs: `triton` at m 1, 4, 8, 16 (:671, :673–675), CK kernel 7 at m 2, 32 (:672, :676); no other bpreshuffle table of that commit lists this (N, K)
- S13 ROCm/aiter@dbd8bf5:aiter/ops/gemm_op_a8w8.py:453 `get_CKGEMM_config` looks up (gfx, cu_num, M, N, K) with M as given, then `get_padded_m` at granularity 0 and 1 (:480); `gemm_a8w8_blockscale_bpreshuffle` (:913) uses it
- E15 [2026-11, gfx950 (MI355X), FlyDSL 0.3.2, aiter dbd8bf5, N=8, per-kernel times of CUDA-graph replays from the rocprofiler-sdk tracer, kernel and aiter paired in each round in one process (the Kineto-timed version in a second process, N=6)] the eight-wave split against aiter at m 4, 8, 16: 1.051, 1.053, 1.041; kernel 4.349–4.370 µs, an empty Triton kernel 4.337 µs; the Kineto-timed version read 0.30–0.48% higher (N 4096 × K 1024)
- E16 [2026-11, gfx950 (MI355X), FlyDSL 0.3.2, aiter dbd8bf5, N=8, the timer of E15, all arms in each round in one process] the skinny path as built (128 columns per workgroup) against the same kernel compiled with 64: N 32768 × K 1024 m 64 9.794 against 9.899 µs; N 65536 × K 1536 (the current build) m 32 18.413 against 17.972 µs, aiter 18.915 µs, m 16 16.628 µs against aiter 18.166 µs
- E17 [2026-11, gfx950 (MI355X), FlyDSL 0.3.2, aiter dbd8bf5, N=6, the timer of E15, all arms in each round in one process] N 65536 × K 1536 (the current build) m 1–16 16.544–16.791 µs, 6.08–5.99 TB/s on 100.66 MB of weights, aiter 17.348–18.298 µs; the fastest of 17 Triton and HIP read-only kernels over the same 100.66 MB 16.434 µs, 6.125 TB/s, over 1.007 GB 6.84 TB/s; checksums and GEMM outputs correct
- E18 [2026-11, gfx950, FlyDSL 0.3.2, N=?, a separate run; ruler not stated in the record] the three newer kernels at the sizes they are built for: N 16384 × K 1536 1.856, 1.316, 1.849, 1.933, 1.401 at m 1, 2, 4, 16, 32; N 65536 × K 1536 1.047, 1.058, 1.056, 1.061, 1.087 at m 1, 2, 4, 8, 16; N 7168 × K 384 1.044, 1.154 at m 64, 128; all 12 at or above 1.03, accuracy passed at each
