---
title: "bf16 GEMM, narrow N (32–256) × K 6144 — whole dot products per wave or workgroup at M ≤ 8, split-K with a fixed-order fold at M ≥ 2048; N 64 × K 4096 at m 4–128 with the whole K in one workgroup"
kind: case
ops: [gemm]
dtypes: [bf16]
regimes: ["M 2-8", "M 16-1024", "M 2048-4096", "M 4-128 (N 64 × K 4096)"]
accuracy: bitwise
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=1
measured:
  - {metric: "speedup over aiter's tuned bf16 GEMM, per size", value: "0.565–1.195; 2 of 13 sizes ≥ 1.03 (m 2, 4)", gpu: MI355X, workload: "N 128 × K 6144, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", date: "2026-10-28"}
  - {metric: "speedup over aiter's tuned bf16 GEMM, per size", value: "0.490–1.201; 5 of 13 sizes ≥ 1.03 (m 2, 4, 8, 2048, 4096)", gpu: MI355X, workload: "N 256 × K 6144, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", date: "2026-10-28"}
  - {metric: "speedup over aiter's tuned bf16 GEMM, per size", value: "0.694–1.841; 3 of 13 sizes ≥ 1.03 (m 4, 8, 2048)", gpu: MI355X, workload: "N 32 × K 6144, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", date: "2026-10-28"}
  - {metric: "speedup over aiter's tuned bf16 GEMM, per size, a repeat measurement", value: "0.557–1.094; 2 of 13 sizes ≥ 1.03 (m 2, 4)", gpu: MI355X, workload: "N 128 × K 6144, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", date: "2026-11-09"}
  - {metric: "speedup over aiter's tuned bf16 GEMM, per size, a repeat measurement", value: "0.492–1.195; 4 of 13 sizes ≥ 1.03 (m 2, 4, 8, 4096)", gpu: MI355X, workload: "N 256 × K 6144, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", date: "2026-11-09"}
  - {metric: "speedup over aiter's tuned bf16 GEMM, per size, a repeat measurement", value: "0.711–1.907; 4 of 13 sizes ≥ 1.03 (m 4, 8, 128, 2048)", gpu: MI355X, workload: "N 32 × K 6144, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", date: "2026-11-09"}
  - {metric: "speedup over aiter's tuned bf16 GEMM, per size, one-sided lower bound over 16 rounds of CUDA-graph replays", value: "1.046–1.330 at m 4–128; m 1, 2 (1.014, 1.019) left out of the kernel", gpu: MI355X, workload: "N 64 × K 4096, measured on a build with the same kernel code, apart from taking the current stream without naming the device, that was also built for m 1, 2; m 1, 2, 4, 8, 16, 32, 64, 128", date: "2026-11-09"}
---

# bf16 GEMM at narrow N, K 6144: one dot product per wave at small M

Three of this page's four FlyDSL kernels are the GLM-5.2 set, one kernel per N. "m" is M.

## Applies when

- `a` (m, k) and `b` (n, k) bf16, row-major and not preshuffled, `out = a @ b.T`, no bias, bf16 `out`. Two runs
  must give the same bytes: every reduction these kernels run at a size where they beat the baseline adds in a
  fixed order.
- K = 6144 with N = 32, 128 or 256: a weight of 384 KiB, 1.5 MiB or 3 MiB, a narrow-N row bound by launch,
  reduction and memory latency rather than by reading `b` (`../gemm/small_m.md` §2).
- Sizes at 1.03 or more over the baseline in the later of two runs: N 128 m 2, 4; N 256 m 2, 4,
  8, 4096; N 32 m 4, 8, 128, 2048. § Result shows the earlier run. Each kernel is built for 13 sizes,
  m = 1, 2, 4, …, 4096.
- The yardstick is aiter's tuned bf16 GEMM. N 128 × K 6144 and N 256 × K 6144 each have a second kernel, built
  for m 256–2048 and m 256–1024. Those, and N 64 × K 4096 at m 256–4096, are in
  [bf16_k4096_splitk_hgemm.md](bf16_k4096_splitk_hgemm.md) § N 64–256 at m 256–4096.

## Counter-signals

- **The middle of the M range was below 1.03 in all three kernels in the earlier run:** N 128 at m ≥ 8
  0.565–0.977 *(field, E1)*, N 256 at m 16–1024 0.490–0.970 *(field, E2)*, N 32 at m 16–1024 0.694–1.012
  *(field, E3)*. In the later run N 32 is above it at m 128 (1.081) *(field, E8)*. These sizes run split-K
  MFMA paths, and all but N 256 at m = 1024 first pack `b` into MFMA fragment order on every call (N 128 at every
  such m; N 256 on the path for m 16–512; N 32 for m ≥ 16), a second launch over the whole weight. The sizes at
  1.03 or more that pack are N 32 at m = 2048, at 1.062 *(field, E3)*, and, in the later run, N 32 at m = 128
  *(field, E8)*.
- **m = 1 is at aiter's speed in every kernel** (0.993–1.002 *(field, E1, E2, E3)*), and so is N 32 at m = 2: its
  4×4×4 MFMA path with 16 K partitions across 128 workgroups reads 0.997 *(field, E3)*.
- **A reduction in arrival order at N 128, m = 16.** 24 K partitions add their fp32 MFMA tiles into an fp64
  workspace with `BufferAtomicAdd(Float64)`; the last producer converts to bf16. Adding
  fp32 partials in fp64 is exact only while all partials and running sums fit one 53-bit window, which a wide
  magnitude spread breaks (estimated), so the output can depend on arrival order where two runs must give the
  same bytes (`../gemm/correctness.md` §8). The accuracy check passed at that size, which is slower than aiter
  (0.631 *(field, E1)*). Do not copy it.
- **Counters that live across calls.** N 32's split paths (m = 2 and m 16–2048, the m = 2048 that beats aiter
  included) zero their arrival counters once per shape and rely on `atomicrmw uinc_wrap` returning each to 0 at
  its last arrival; N 128 at m ≥ 512 advances a counter by one generation per call and never resets it. A call
  then depends on what earlier calls left; scratch that every call rewrites before it reads it does not.
- **Compilation runs inside the first call at a size.** The built launches are kept in a module-level dict, so
  compilation and scratch allocation run inside the first call at each m.
- **Packing on every call.** N 128's docstring says the packed weight is refreshed for a new or modified weight,
  N 32's that `b` is packed on first use and repacked when replaced or mutated; both kernels pack on every call.

## Mechanism

**m ≤ 8 (N 128 m 2, 4; N 256 m 2, 4, 8; N 32 m 4, 8): the whole K of an output element stays in one workgroup.** No
MFMA, no split across workgroups, `b` read in its row-major layout with 16 B buffer loads straight into
registers, fp32 accumulation, one launch:

| | N 128 | N 256 | N 32 |
|---|---|---|---|
| Work per workgroup | one column, 1 row (m 2) or 2 rows (m 4); 4 waves | one output element; 4 waves | one output element; 1 wave |
| Workgroups | 128 × m/rows = 256 at m 2 and 4 | m × 256: 512, 1024, 2048 | m × 32: 128, 256 |
| Inner product | each thread loads its 24 bf16 of the `b` row once and reuses them for the workgroup's rows; m 2: fp32 products summed in the thread; m 4: `v_dot2_f32_bf16` in 4 independent chains | 3 steps of 16 B of `a` and `b` per thread; `v_dot2_f32_bf16` into 4 fp32 accumulators | 12 steps of 16 B per lane; `v_dot2_f32_bf16` into 4 fp32 accumulators |
| Fold | 6-step xor shuffle per wave, then thread 0 sums the 4 waves' partials from LDS in wave order | the same | 6-step xor shuffle; lane 0 stores |

With N this narrow, an MFMA tile covers few columns and the grid is small (`../gemm/grid_pricing.md` §1,
`../gemm/small_m.md` §3); here the whole K of each output element stays in one workgroup and the fold never
leaves it (`../gemm/small_m.md` §5: splitting N across warps needs no LDS reduction, splitting K
does). The N 32 kernel, one wave per output element:

```python
output = fx.block_idx.x  # one output element per workgroup
row = output // n
col = output % n
...
acc = [fx.Float32(0.0) for _ in range(vec // 2)]  # 4 independent fp32 chains
for step in range_constexpr(k // (64 * waves * vec)):  # 12 steps of 16 B per lane
    kk = step * 64 * waves * vec + tid * vec
    av = Vector(_bload(a_rsrc, row * k + kk, vec, T.bf16))
    bv = Vector(_bload(b_rsrc, col * k + kk, vec, T.bf16))  # b row-major, straight into registers
    for i in range_constexpr(vec // 2):
        ax = av.shuffle(av, [i * 2, i * 2 + 1])
        bx = bv.shuffle(bv, [i * 2, i * 2 + 1])
        acc[i] = fx.Float32(rocdl_d.fdot2_f32_bf16(
            mT.f32(), _ir_val(ax), _ir_val(bx), _ir_val(acc[i]), clamp=False).result)
total = acc[0]
for i in range_constexpr(1, vec // 2):
    total = total + acc[i]
for bit in range_constexpr(6):  # 6-step xor shuffle: every lane ends with the wave's sum
    total = total + total.shuffle_xor(1 << bit, 64)
...
if lane == fx.Int32(0):
    _bstore(total.to(fx.BFloat16), c_rsrc, output)
```

**m ≥ 2048: split-K MFMA with a fold in a fixed order.**

- N 256, m 2048 and 4096: 64×64 tile with 4 waves (m 2048) or 128×128 with 8 waves (m 4096);
  `a` and `b`, unpacked, go by `cdna4.BufferLoadAsyncLDS128b` into a 2-stage LDS ring and from there by 16 B reads
  into `mfma_f32_16x16x32_bf16`. Split-K 4 writes fp32 partials in fragment order; a second launch
  sums partitions 0–3 in order and stores bf16. 512 and 256 workgroups.

  ```python
  # the second launch: P holds the SPLIT fp32 planes in the fragment order the GEMM launch wrote them
  idx = fx.block_idx.x * 256 + fx.thread_idx.x
  pv = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(P)),
                    fx.make_layout((SPLIT, M * N // 4, 4), (M * N, 4, 1)))
  load = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
  values = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32)
  if idx < M * N // 4:
      total = fx.Vector.filled(4, 0.0, fx.Float32)
      for s in range_constexpr(SPLIT):  # partition 0, 1, 2, 3: the same order in every run
          fx.copy_atom_call(load, pv[s, idx, None], values)
          total = total + values.load()
      ...
      for i in range_constexpr(4):
          C[row + i, col] = total[i].to(fx.BFloat16)  # rounded once, after the sum
  ```

  Not this, the N 128 path at m = 16 (Counter-signals): every partition adds its tile into one shared plane as it
  arrives, so the order of the additions is the order of arrival.

  ```python
  atomic = fx.make_copy_atom(fx.rocdl.BufferAtomicAdd(fx.Float64), fx.Float64)
  thread_copy = fx.make_tiled_copy_C(atomic, tiled_mma).get_slice(tid)
  values = fx.make_fragment_like(frag_C, fx.Float64)
  values.store(frag_C.load().to(fx.Float64))
  fx.copy(atomic, thread_copy.retile(values), thread_copy.partition_S(gC))
  ```
- N 32, m 2048: a 32×32 output tile per workgroup, K split over its 8 waves and over 4
  workgroups (256 workgroups of 512 threads); `b` packed per call (384 KiB); `mfma_f32_16x16x32_bf16`. The 8 wave
  partials are summed through LDS in wave order; owner waves store an fp32 plane and take a ticket from a wrapping
  counter (`atomicrmw uinc_wrap`, agent scope); the last of the 4 sums planes 0–3 in order and stores bf16, all in
  the GEMM launch.

Fold forms: a separate reduce launch against an in-launch last-arriver (`../gemm/grid_pricing.md` §2,
`../gemm/small_m.md` §4); both are bitwise-stable because the order is fixed
(`../gemm/correctness.md` §8).

## Result

The speedup is a lower bound of the CUDA-graph time ratio aiter / kernel, against aiter's tuned bf16 GEMM. The
table is the earlier of the two runs; bold: at or above 1.03 in that run. Per size:

These runs used FlyDSL 0.3.2, a pin installed over a machine image that ships 0.3.4.1 (not checked in the run's log;
`../pitfalls/measured_a_different_build.md`).

| Kernel | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 | ≥ 1.03 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N 128 *(field, E1)* | 1.002 | **1.195** | **1.182** | 0.600 | 0.631 | 0.565 | 0.584 | 0.662 | 0.650 | 0.747 | 0.738 | 0.942 | 0.977 | 2 of 13 |
| N 256 *(field, E2)* | 0.997 | **1.065** | **1.201** | **1.125** | 0.490 | 0.500 | 0.494 | 0.531 | 0.595 | 0.784 | 0.970 | **1.047** | **1.069** | 5 of 13 |
| N 32 *(field, E3)* | 0.993 | 0.997 | **1.257** | **1.841** | 0.694 | 0.695 | 0.752 | 1.012 | 0.979 | 0.947 | 0.974 | **1.062** | 0.811 | 3 of 13 |

10 of the 39 sizes were at or above 1.03 in that run. In the later run 10 are, not the same 10: N 256 no longer at
m 2048, N 32 also at m 128 *(field, E8)*. Against a later aiter (`a6c759b1`) a separate run of the N 256 kernel read
0.43–1.04 over its 13 sizes *(field, E9)*; which of them reached 1.03 is not in the record.

## N 64 × K 4096 at m 4–128: the whole K in one workgroup, on `v_dot2`

The N 64 × K 4096 kernel is built for m 4, 8, …, 128 (512 KiB of weight) with the design of the m ≤ 8 paths
above: no MFMA, no LDS ring, `b` in its row-major layout loaded straight into registers, the whole K of an output in
one workgroup. The yardstick is aiter's tuned bf16 GEMM. A separate run read 1.020–1.286, at or above 1.03 at
m 8–128 and not at m 4 *(field, E10)*; later separate runs read 0.972–1.048, at or above 1.03 only at m 128
*(field, E11)*: there the other five sizes are aiter's. Narrow N at larger m:
[bf16_k4096_splitk_hgemm.md](bf16_k4096_splitk_hgemm.md). m 1 and 2 are left out of the kernel: within 2% of aiter
(1.014, 1.019) *(field, E4)*.

- Each workgroup owns an R × C block of outputs. Every thread takes a strided set of 8-element K chunks of those rows
  and columns, issues all its 16 B loads before the first use, and accumulates with `v_dot2_f32_bf16` in fp32.
- Per m: 1 × 1 outputs and one wave (m 4, 256 workgroups); 2 × 2 and one wave (m 8, 16, 32: 128, 256 and 512
  workgroups); 4 × 4 and 4 × 8 with eight waves (m 64, 128: 256 workgroups). Workgroup ids are remapped so that the
  ids of one XCD cover one block of rows × columns of the tile grid and its L2 reads a fraction of A and of B
  (`../gemm/schedule_and_locality.md` §2).

  ```python
  # Workgroups are dealt round-robin to the XCDs: give each XCD an (row_blocks/xr) x (col_blocks/xc)
  # block of tiles so that its L2 reads only m/xr rows of a and n/xc rows of b.
  xcd = bid & fx.Int32((1 << _XCD_BITS) - 1)
  loc = bid >> fx.Int32(_XCD_BITS)
  rb = (xcd >> fx.Int32(lxc)) * rt + (loc >> fx.Int32(lct))
  cb = (xcd & fx.Int32(xc - 1)) * ct + (loc & fx.Int32(ct - 1))
  ```
- Fold, in a fixed order: at m 4, six `shuffle_xor` steps across the wave; with several outputs per workgroup, a
  reduce-scatter that exchanges lane halves with `permlane32_swap` and `permlane16_swap`, then DPP for the lower lane
  bits; with eight waves, the wave sums meet in LDS and are added in wave order. One 2-byte store per output, `sc0
  sc1`.

  ```python
  if const_expr(outs == 1):  # one output per workgroup: xor-shuffle steps across the wave
      total = vals[0]
      for st in range_constexpr(low_bits):
          total = total + total.shuffle_xor(1 << (low_bits - 1 - st), _WAVE)
  else:  # several outputs: reduce-scatter, lane l ends with output l >> low_bits
      total = _butterfly(vals, lane, n_half, low_bits)
  o_lane = lane >> fx.Int32(low_bits)
  owner = (lane & fx.Int32((1 << low_bits) - 1)) == fx.Int32(0)

  if const_expr(W > 1):  # several waves: their sums meet in LDS and are added in wave order
      smem = fx.SharedAllocator()
      part = smem.allocate(Storage).peek().ptr
      if owner:
          fx.ptr_store(total, part + wave * outs + o_lane)
      fx.gpu.barrier()
      if tid < fx.Int32(outs):
          t = fx.ptr_load(part + tid, result_type=fx.Float32)
          for w in range_constexpr(1, W):
              t = t + fx.ptr_load(part + w * outs + tid, result_type=fx.Float32)
          oi = tid // C
          oj = tid % C
          _bstore(t.to(fx.BFloat16), c_rsrc, (r0 + oi) * n + c0 + oj)
  ```
- The launch requests kernel-argument preload (`amdgpu-kernarg-preload-count` 16) and passes the three tensors with
  static layouts (`flyc.from_torch_tensor`). In compiler output read while the kernel was being written, that
  preloaded all three pointers (kernarg size 24, 8 user SGPRs), where dynamic-layout arguments preloaded only the
  first *(field, E7)*; the final build's metadata was not read.

Measured as a one-sided lower bound of the aiter / kernel time ratio over 16 rounds of CUDA-graph replays with L2
flushed before each call, on one MI355X: 1.046–1.330 over aiter, highest at m 8,
thin at m 4 *(field, E4)*. The kernel read 4.08–4.20 µs per round at every m from 1 to 128 while aiter read
4.19–5.95 *(field, E5)*: the gain is aiter's time growing with m, and whether this timer has a floor near 4.1 µs was
not measured here; three timers that were checked read every short kernel at about 4 µs or more
(`../pitfalls/profiled_timer_floor.md`). Under stress every size gave the output of the call
alone *(field, E6)*.

## Evidence

- E1 [2026-10, gfx950, FlyDSL 0.3.2, N=1, lower bound of the CUDA-graph time ratio aiter / kernel] speedup over aiter per size 0.565–1.195, 2 of 13 sizes ≥ 1.03 (m 2, 4) (N 128 × K 6144, m 1–4096)
- E2 [2026-10, gfx950, FlyDSL 0.3.2, N=1, the ruler of E1] speedup over aiter per size 0.490–1.201, 5 of 13 sizes ≥ 1.03 (m 2, 4, 8, 2048, 4096) (N 256 × K 6144, m 1–4096)
- E3 [2026-10, gfx950, FlyDSL 0.3.2, N=1, the ruler of E1] speedup over aiter per size 0.694–1.841, 3 of 13 sizes ≥ 1.03 (m 4, 8, 2048) (N 32 × K 6144, m 1–4096)
- E4 [2026-11, gfx950, FlyDSL 0.3.2, N=16, one-sided lower bound of the aiter / kernel time ratio over 16 rounds of CUDA-graph replays (32 at m 1, 2) with L2 flushed before each call, one MI355X, aiter 0.1.21.dev66+gdbd8bf5bd] N 64 × K 4096 on an earlier build (the same kernel except `torch.cuda.current_stream()` without `a.device`, also built for m 1, 2): 1.014, 1.019, 1.046, 1.330, 1.186, 1.207, 1.233, 1.313 at m 1, 2, 4, 8, 16, 32, 64, 128; m 1, 2 judged not faster; accuracy passed
- E5 [2026-11, gfx950, FlyDSL 0.3.2, N=16, the run of E4] per-round times, kernel 4.08–4.20 µs at every m (m 1 4.16–4.20, m 128 4.09–4.16); aiter 4.19–4.33 (m 1), 4.35–4.42 (m 4), 5.43–5.95 (m 8), 5.36–5.52 (m 128)
- E6 [2026-11, gfx950, FlyDSL 0.3.2, N=20, eager and op-level CUDA-graph stress on one MI355X, each try compared byte for byte with an eager call alone] N 64 × K 4096 on the earlier build of E4, whose one difference was `torch.cuda.current_stream()` without `a.device`, every size: streams limited to 16 and 64 CUs; a second concurrent call on the same inputs; one captured graph replayed on the same inputs, 20 tries each; one graph replayed 24 times with three input sets rotated into its buffers; two calls at once on two streams with different inputs, unrestricted and both on the same 64 CUs, 20 tries each: 0 tries differ
- E7 [2026-11, gfx950, FlyDSL 0.3.2, N=1, compiler output read while the kernel was being written, not timed] N 64 × K 4096 with `amdgpu-kernarg-preload-count` 16: dynamic-layout tensor arguments preloaded only the first pointer (`preload_length 2`); static-layout `flyc.from_torch_tensor` arguments preloaded all three (kernarg size 24, user SGPR count 8)
- E8 [2026-11, gfx950, FlyDSL 0.3.2 (the pin of aiter dbd8bf5, installed over the image for the run), N=1, a repeat of the E1–E3 measurement with the ruler of E4] sizes at or above 1.03: N 128 m 2, 4 (0.557–1.094); N 256 m 2, 4, 8, 4096 (0.492–1.195; m 2048 1.021); N 32 m 4, 8, 128, 2048 (0.711–1.907; m 128 1.081)
- E9 [2026-11, gfx950, FlyDSL ?, N=1, a separate run against aiter a6c759b1; ruler not stated in the record] the N 256 × K 6144 kernel: 0.43–1.04 over its 13 sizes (m 1–4096), no size at 1.05; per-size values not in the record
- E10 [2026-11, gfx950, FlyDSL 0.3.2, N=?, a separate run; ruler not stated in the record] N 64 × K 4096: 1.020, 1.286, 1.111, 1.143, 1.159, 1.227 at m 4, 8, 16, 32, 64, 128; 5 of 6 sizes at or above 1.03 (not m 4); accuracy passed at each
- E11 [2026-11, gfx950, FlyDSL ?, N=?, later separate runs; ruler not stated in the record] N 64 × K 4096: 0.990, 0.995, 0.994, 0.972, 0.979, 1.048 at m 4, 8, 16, 32, 64, 128; 1 of 6 sizes at or above 1.03 (m 128), none at 1.05; accuracy passed at each
