---
title: "bf16 GEMM, wide N (2624–6144) × K 2048–6144 — one wave per output column on the unpacked weight at M ≤ 2; a weight repack per call loses above"
kind: case
ops: [gemm]
dtypes: [bf16]
regimes: ["M 1-2", "M 4-4096"]
accuracy: bitwise
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=1
measured:
  - {metric: "speedup over aiter's tuned bf16 GEMM, per size", value: "0.418–1.148; 1 of 13 sizes ≥ 1.03 (m 1), where it is the fastest kernel; at m 2, 4, 8, 64 a Triton kernel for the same shape is the fastest", gpu: MI355X, workload: "N 2624 × K 6144, FlyDSL, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", date: "2026-10-28"}
  - {metric: "speedup over aiter's tuned bf16 GEMM, per size", value: "0.250–1.465; 2 of 13 sizes ≥ 1.03 (m 1, 2)", gpu: MI355X, workload: "N 3072 × K 6144, FlyDSL, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", date: "2026-10-25"}
  - {metric: "speedup over aiter's tuned bf16 GEMM, per size", value: "0.355–1.198; 2 of 13 sizes ≥ 1.03 (m 1, 2), the fastest kernel at 1 (m 1); at m 2–64 a Triton kernel for the same shape is the fastest", gpu: MI355X, workload: "N 6144 × K 2048, FlyDSL, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", date: "2026-10-28"}
  - {metric: "speedup over aiter's tuned bf16 GEMM, per size", value: "0.418–1.208; 2 of 13 sizes ≥ 1.03 (m 1, 2)", gpu: MI355X, workload: "N 6144 × K 3072, FlyDSL, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", date: "2026-10-28"}
---

# bf16 GEMM at wide N: a wave per output column at M ≤ 2

Four FlyDSL kernels for `out = a @ b.T` on GLM-5.2 shapes, one kernel per shape. Two of the shapes also have a
Triton kernel: N 2624 × K 6144 (the QKV projection) and N 6144 × K 2048 (the output projection). "m" is M.

## Applies when

- `a` (m, k) and `b` (n, k) bf16, row-major and not preshuffled, `out = a @ b.T`, no bias, bf16 `out`. Two runs
  must give the same bytes. The FlyDSL paths that beat aiter keep each output's whole K in one wave; the
  Triton N 2624 kernel at m = 64 splits K six ways across workgroups and a second kernel adds the six fp32 slices in
  slice order.
- N × K = 2624 × 6144, 3072 × 6144, 6144 × 2048, 6144 × 3072: a weight of 30.75, 36, 24 and 36 MiB, a streaming
  row bound by the read of `b` (`../gemm/small_m.md` §2).
- m = 1 and 2 only. The FlyDSL kernel is the fastest at N 2624 m 1, N 3072 m 1, 2, N 6144 × K 2048 m 1,
  N 6144 × K 3072 m 1, 2. The Triton kernels are the fastest at N 2624 m 2, 4, 8, 64 and N 6144 × K 2048 m 2–64;
  at the other sizes aiter is.
- Judge a kernel at a size against the fastest other implementation there: aiter's tuned bf16 GEMM, except at the
  sizes where the Triton kernels beat it.

## Counter-signals

- **Every size that repacks the weight on every call loses; the pack is the whole loss at some, not at others.**
  Above the m ≤ 2 path each launch
  first runs a pack kernel that rewrites the whole `b` into MFMA order, then the MFMA GEMM on the packed copy
  (N 2624 at m ≥ 2; N 3072 at m ≥ 4; N 6144 × K 2048 at m ≥ 4; N 6144 × K 3072 at m ≥ 4). Those sizes read
  N 2624 0.418–0.996, N 3072 0.250–0.735, N 6144 × K 2048 0.355–0.861, N 6144 × K 3072 0.418–0.711
  *(field, E1, E2, E3, E4)*.

  ```python
  # Not this: the call of the N 2624 kernel at m >= 2
  def launch(a, b, out):
      stream = torch.cuda.current_stream()
      pack(b, packed_b, n, k, wpad, mt, stream)  # a launch that rewrites the whole weight, on every call
      compiled(a, packed_b, partial if has_partial else out, out, counters, packed_a, *config, stream)
  ```
  - **The cost at these shapes:** the pack reads the weight and writes the packed copy, and the GEMM reads the copy
    again: about three passes over 24–36 MiB instead of one, plus a second launch per call (estimated). The pack
    launch alone takes 13–16 µs, 56–64% of the call at m = 16 and 10% at N 6144 × K 3072 m = 4096; with the pack
    skipped the same GEMM reads 1.27 at N 2624 m = 1024 and about 1.0 at N 6144 × K 3072 m = 16, but still 0.87 and
    0.78 at N 6144 × K 3072 m = 256 and 4096 *(field, E7)*.
  - A kernel that may not depend on what an earlier call left cannot keep the packed copy between calls, and the
    caller hands over only the row-major weight, so no pre-shuffled weight can reach it; the host-side preshuffle of
    `../gemm/datapath.md` §10 pays only when done once. At these weight sizes, read `b` in its (n, k)
    layout.
- **At m 2–64 the Triton kernels win on the two shapes that have one.** They read the unpacked weight once per
  call: N 2624 m 2, 4, 8, 64 at 1.077–1.195 *(field, E5)* and N 6144 × K 2048 m 2–64 at 1.078–1.285 *(field, E6)*.
  At N 6144 × K 2048, m = 2, the FlyDSL m ≤ 2 path itself reads 1.075 *(field, E3)* against Triton's 1.139
  *(field, E6)*, about 0.94× the Triton kernel (estimated from the two ratios). The Triton N 6144 × K 2048 kernel
  at m ≤ 16:

  ```python
  pid = tl.program_id(0)
  pid = (pid % NXCD) * (N_TILES // NXCD) + (pid // NXCD)  # each XCD takes a contiguous run of N tiles
  offs_m = tl.arange(0, BLOCK_M)
  offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)  # one N tile per workgroup, over all rows of a
  ...
  a_ptrs = A + offs_m[:, None] * K + offs_k[None, :]
  b_ptrs = B + offs_n[None, :] * K + offs_k[:, None]  # b in its (n, k) layout: no pack
  m_mask = offs_m < M
  acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
  for _ in tl.range(0, K // BLOCK_K):  # all of K in the workgroup: no split, no reduction
      if EVEN_M:
          a = tl.load(a_ptrs)
      else:
          a = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)
      b = tl.load(b_ptrs, cache_modifier=".cg")
      acc = tl.dot(a, b, acc=acc)
      a_ptrs += BLOCK_K
      b_ptrs += BLOCK_K
  ```
- **Packing on every call.** N 6144 × K 2048's docstring says packed weights are reused while their identity and
  version stay unchanged; the code packs on every call.
- **Compilation runs inside the first call at a size.** The built launches are kept in a module-level dict, so
  compilation and scratch allocation run inside the first call at each m.

## Mechanism

At every size where a FlyDSL kernel is the fastest: **one wave owns one output column and reads that column's
weight row, unpacked, once.** Each lane issues 16 B loads of the `b` row and of each `a` row straight into registers (no LDS, no
MFMA), accumulates in a few independent fp32 chains, and the wave reduces them; one launch, no split across waves
or workgroups, so the order of the sum is fixed. The weight is read once per call at full width, which is what a
streaming row needs (`../gemm/small_m.md` §2); the grid is N/4 or N/8 workgroups, 656–1536, more
than the 256 CUs (`../gemm/grid_pricing.md` §1). `../gemm/small_m.md` §5 keeps `b`
on the LDS path for MFMA tiles; at m ≤ 2 these kernels issue no MFMA.

| | N 2624 × K 6144 m 1 | N 3072 × K 6144 m 1, 2 | N 6144 × K 2048 m 1 | N 6144 × K 3072 m 1, 2 |
|---|---|---|---|---|
| Workgroup | 4 waves, 4 columns | 4 waves, 4 columns | 4 waves, 4 columns | 4 waves (m 1) or 8 (m 2), one column each |
| Workgroups | 656 | 768 | 1536 | 1536 (m 1), 768 (m 2) |
| K steps per lane | 12 × 16 B, all 12 issued before the first dot | 12 × 16 B | 4 × 16 B | 6 × 16 B; at m 2 both rows' `a` loads issued with the `b` load |
| Products | `v_dot2_f32_bf16` into 4 fp32 chains | m 1: fp32 FMA on 8-wide vectors; m 2: `v_dot2_f32_bf16` into 4 chains, the weight base in a scalar buffer descriptor | `v_dot2_f32_bf16` into 4 chains | `v_dot2_f32_bf16` into 4 chains |
| Wave reduction | DPP `row_shr` / `row_bcast` adds, the sum in lane 63 | `fx.coop.warp_reduce` | 6-step xor shuffle | balanced local sum, then `warp_reduce` |

The K loop of the N 2624 × K 6144 kernel at m 1:

```python
# M rows of a, CN columns per wave, UK steps of loads issued together
col = (fx.Int32(fx.gpu.block_idx.x) * (WAVES // KW) + wave // KW) * CN  # the wave's own output column
...
for ki, state in range(0, K // KW // 512 // UK, init=accum):
    aa = []
    bb = []
    for u in range_constexpr(UK):  # every 16 B load of the UK steps is issued before the first dot
        kv = (wave % KW) * (K // KW // 8) + (fx.Int32(ki) * UK + u) * 64 + lane
        for ni in range_constexpr(CN):
            safe_col = (col + ni < N).select(col + ni, 0)
            bb.append(_load_eight(b_buf, safe_col * (K // 8) + kv, atom))  # the b row as the caller laid it out
        for row in range_constexpr(M):
            aa.append(_load_eight(a_buf, row * (K // 8) + kv, atom))
    values = list(state)
    for u in range_constexpr(UK):
        for row in range_constexpr(M):
            for ni in range_constexpr(CN):
                for v in range_constexpr(4):
                    av = fx.Vector.from_elements([aa[u * M + row][v * 2], aa[u * M + row][v * 2 + 1]], fx.BFloat16)
                    bv = fx.Vector.from_elements([bb[u * CN + ni][v * 2], bb[u * CN + ni][v * 2 + 1]], fx.BFloat16)
                    idx = (row * CN + ni) * ACC + v % ACC  # ACC independent fp32 chains per output
                    values[idx] = fx.Float32(fx.rocdl.fdot2_f32_bf16(
                        T.f32, av.ir_value(), bv.ir_value(), fx.Float32(values[idx]).ir_value(), clamp=False,
                    ).res)
    accum = yield values
```

## Result

The speedup is a lower bound of the CUDA-graph time ratio aiter / kernel, against aiter's tuned bf16 GEMM. Bold:
the kernel with the highest speedup at that size, from 1.03. Per size:

These runs used FlyDSL 0.3.2, a pin installed over a machine image that ships 0.3.4.1 (not checked in the runs' logs;
`../pitfalls/measured_a_different_build.md`).

| Kernel | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 | Fastest at |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FlyDSL N 2624 × K 6144 *(field, E1)* | **1.148** | 0.437 | 0.432 | 0.438 | 0.418 | 0.422 | 0.555 | 0.615 | 0.795 | 0.765 | 0.996 | 0.871 | 0.822 | 1 of 13 |
| Triton N 2624 × K 6144 *(field, E5)* | – | **1.081** | **1.077** | **1.100** | – | – | **1.195** | – | – | – | – | – | – | 4 of 4 |
| FlyDSL N 3072 × K 6144 *(field, E2)* | **1.465** | **1.331** | 0.250 | 0.257 | 0.265 | 0.303 | 0.301 | 0.315 | 0.370 | 0.431 | 0.606 | 0.626 | 0.735 | 2 of 13 |
| FlyDSL N 6144 × K 2048 *(field, E3)* | **1.198** | 1.075 | 0.383 | 0.425 | 0.378 | 0.355 | 0.375 | 0.459 | 0.548 | 0.635 | 0.766 | 0.685 | 0.861 | 1 of 13 |
| Triton N 6144 × K 2048 *(field, E6)* | – | **1.139** | **1.146** | **1.285** | **1.137** | **1.078** | **1.105** | – | – | – | – | – | – | 6 of 6 |
| FlyDSL N 6144 × K 3072 *(field, E4)* | **1.208** | **1.119** | 0.420 | 0.418 | 0.435 | 0.421 | 0.424 | 0.491 | 0.532 | 0.564 | 0.711 | 0.696 | 0.701 | 2 of 13 |

"–": the Triton kernel is not built for that size. The FlyDSL kernels are the fastest at 6 of their 52 sizes; at
N 6144 × K 2048, m = 2, the FlyDSL kernel passes 1.03 but the Triton kernel is faster.

## Evidence

- E1 [2026-10, gfx950, FlyDSL 0.3.2, N=1, lower bound of the CUDA-graph time ratio aiter / kernel] speedup over aiter per size 0.418–1.148, 1 of 13 sizes ≥ 1.03 (m 1); m ≥ 2 (weight repacked per call) 0.418–0.996 (N 2624 × K 6144, FlyDSL, m 1–4096)
- E2 [2026-10, gfx950, FlyDSL 0.3.2, N=1, the ruler of E1] speedup over aiter per size 0.250–1.465, 2 of 13 sizes ≥ 1.03 (m 1, 2); m ≥ 4 (repacked) 0.250–0.735 (N 3072 × K 6144, FlyDSL, m 1–4096)
- E3 [2026-10, gfx950, FlyDSL 0.3.2, N=1, the ruler of E1] speedup over aiter per size 0.355–1.198, 2 of 13 sizes ≥ 1.03 (m 1, 2); m ≥ 4 (repacked) 0.355–0.861 (N 6144 × K 2048, FlyDSL, m 1–4096)
- E4 [2026-10, gfx950, FlyDSL 0.3.2, N=1, the ruler of E1] speedup over aiter per size 0.418–1.208, 2 of 13 sizes ≥ 1.03 (m 1, 2); m ≥ 4 (repacked) 0.418–0.711 (N 6144 × K 3072, FlyDSL, m 1–4096)
- E5 [2026-10, gfx950, Triton, N=1, the ruler of E1] speedup over aiter per size 1.077–1.195 at m 2, 4, 8, 64, the fastest kernel at each (N 2624 × K 6144, Triton)
- E6 [2026-10, gfx950, Triton, N=1, the ruler of E1] speedup over aiter per size 1.078–1.285 at m 2–64, the fastest kernel at each (N 6144 × K 2048, Triton)
- E7 [2026-11, gfx950 (MI355X), FlyDSL 0.3.2, aiter dbd8bf5, N=6, per-kernel times of CUDA-graph replays summed per call, median per arm per round] the call as built (pack, then GEMM) / the same GEMM on a copy packed beforehand / the pack launch alone / aiter, µs: N 6144 × K 3072 m 16 23.51 / 9.99 / 15.03 / 10.33, m 256 41.74 / 24.92 / 14.64 / 21.68, m 4096 160.79 / 145.90 / 15.51 / 113.87; N 2624 × K 6144 m 1024 59.79 / 46.75 / 13.22 / 59.35, m 16 23.08 / 10.44 / 13.04 / 9.66
