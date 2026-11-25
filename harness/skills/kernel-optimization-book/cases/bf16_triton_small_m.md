---
title: "bf16 GEMM at M ≤ 64 in plain Triton, N 6144 × K 2048 and N 2624 × K 6144 — one N tile per workgroup over all rows of `a`, the row-major weight read once with `.cg`, N tiles in runs per XCD; split-K 6 with a fixed-order reduce kernel at N 2624 × K 6144, M 64"
kind: case
ops: [gemm]
dtypes: [bf16]
regimes: ["M 2-64 (N 6144 × K 2048)", "M 2-8 and 64 (N 2624 × K 6144)"]
accuracy: bitwise
gens: [gfx950]
triton: 3.8.0
basis: measured N=1
measured:
  - {metric: "speedup over aiter's `gemm_a16w16`, per size, one reference run of the ruler", value: "1.048 (N 2624 × K 6144, M 2) to 1.292 (N 6144 × K 2048, M 8)", gpu: MI355X, workload: "N 6144 × K 2048 at M 2, 4, 8, 16, 32, 64; N 2624 × K 6144 at M 2, 4, 8, 64", date: "2026-11"}
  - {metric: "call time on the ruler's timer (Kineto), over 4 runs, beside a plain read of the weight", value: "N 6144 × K 2048: 5.94–6.38 µs at M 2–16, 7.16–7.22 µs at M 32, 8.53–8.56 µs at M 64, read 5.53–5.61 µs; N 2624 × K 6144: 7.98–8.40 µs at M 2–8, 12.49–12.65 µs at M 64, read 6.90–7.02 µs", gpu: MI355X, workload: "N 6144 × K 2048 (25.17 MB weight) at M 2–64; N 2624 × K 6144 (32.24 MB) at M 2, 4, 8, 64", date: "2026-11"}
---

# bf16 GEMM at M ≤ 64 in plain Triton (N 6144 × K 2048, N 2624 × K 6144)

Two plain `@triton.jit` kernels for `out = a @ b.T` at M ≤ 64, one per weight shape, a family of the Triton and
Gluon kernel library that [triton/kernels/kernel_library](../triton/kernels/kernel_library.md) maps; with the two
norms of [cases/rmsnorm_triton_hidden6144](rmsnorm_triton_hidden6144.md) they are the library's four `@triton.jit`
kernels. Why plain Triton fits them, and what its compiler adds (coalescing, `num_stages` pipelining, buffer
instructions): [triton/authoring](../triton/authoring.md) §2. Tiles by M, split-K under a bitwise output, M = 1 and
re-tuning for cold-cache timing: [triton/gemm/small_m_gemm](../triton/gemm/small_m_gemm.md), the page that explains
this family; what Triton 3.8.0 compiles the two kernels to: [triton/gemm](../triton/gemm.md) §9. This page has the
kernels by size, their code, and what was measured.

## Applies when

- The op is `out = a @ b.T` with `a` (M, K) and `b` (N, K) bf16, row-major and not preshuffled, no bias, bf16
  `out`, at one of two weight shapes: N 6144 × K 2048 (a 25 MB weight) or N 2624 × K 6144 (32 MB), GLM-5.2's output
  projection and QKV projection ([cases/bf16_glm52_wide_n](bf16_glm52_wide_n.md) has the FlyDSL kernels for the
  two shapes).
- The path replaced is aiter's tuned `gemm_a16w16`; a speedup is against it.
- M is one of the kernel's sizes: 2, 4, 8, 16, 32, 64 at N 6144 × K 2048 and 2, 4, 8, 64 at N 2624 × K 6144.
  M = 1, N 2624 × K 6144 at M 16 and 32, and M from 128 have no Triton kernel (§ Counter-signals).
- The family is held to `bitwise` accuracy: every run on the same inputs gives the same bytes as the first, and the
  output's error bound, an SQNR floor of 45 dB, holds besides ([triton/numerics](../triton/numerics.md) §1). That
  requirement decides how K may be summed: these kernels split it only where a second kernel then adds the partials
  in a fixed order ([triton/gemm](../triton/gemm.md) §1, §9). Neither kernel contains an atomic; each of the 9 sizes
  that run without a split gave one output over 50 runs, inside the accuracy bound, and N 2624 × K 6144 at M = 64,
  the size that splits K, one output in 50 runs *(field, E1)*.
- `bitwise` compares runs of one kernel, not two kernels: a new tile, split count or K rotation changes bits and
  stays correct ([triton/numerics](../triton/numerics.md) §8).

## Counter-signals

- **The thinnest margins** are N 2624 × K 6144 at M 2, 4 and 8 (1.048, 1.070, 1.062) and N 6144 × K 2048 at M 32
  (1.064) *(field, E2)*.
- **Other languages at these sizes.** Both GEMM shapes also have a FlyDSL kernel, and at each size the kernel to
  beat is the best measured one of any language. FlyDSL kernels are the best at M = 1 of both shapes (1.191 at
  N 6144 × K 2048, 1.134 at N 2624 × K 6144) and measure 0.360–1.045 at the Triton kernels' sizes; at
  N 2624 × K 6144, M 16 and 32, and in both shapes from M 128, the only kernels are FlyDSL ones below 1.0
  (0.396–0.998), so aiter's GEMM stays the fastest there *(field, E3)*. A new
  kernel at a shared size has to beat the best one already there.
- **M = 1 was not tuned.** Called at M = 1, both kernels run a 16-row MFMA tile carrying one row. In one A/B on the
  ruler's timer the FlyDSL kernels for the two shapes took 0.975× the Triton time at N 6144 × K 2048 (5.86 against
  6.01 µs) and 0.933× at N 2624 × K 6144 (7.27 against 7.79 µs), lower in every round
  *(field, E4)* ([triton/gemm/small_m_gemm](../triton/gemm/small_m_gemm.md) §7).
- **The host branches reach M that are not sizes.** Both helpers branch on thresholds of M (entry 6 shows one), so
  an M between the sizes takes the branch of the next threshold, M = 1 the M ≤ 16 branch with its own `BLOCK_K` or
  warp count, and N 2624 × K 6144 at M = 16 and at 17–63 the no-split and the split branch. The N 2624 × K 6144
  kernel was not tuned or measured at M = 32, and at M = 16 only copies of it were timed
  ([triton/gemm/small_m_gemm](../triton/gemm/small_m_gemm.md) §6).
- **Nothing guards M above 64 or another shape.** One M tile of at most 64 rows is all of M and the host checks no
  upper bound: at M = 96 and 128 both kernels ran but left the rows above 63 uncomputed, and the N 2624 × K 6144
  reduce kernel read scratch rows that no split had written ([triton/gemm](../triton/gemm.md) §9). Neither host
  asserts the shape, and both ran at other shapes without an error
  ([triton/kernels/kernel_library](../triton/kernels/kernel_library.md) §1); at another N the N 6144 × K 2048
  kernel's remaps need not be permutations of the grid (entry 3).
- **`.cg` and the XCD remap pay behind a flush and cost when the weight stays cached.** Without `.cg` on the weight
  load both kernels took 1.1–3.4% more time with the caches flushed before each call (no stable difference at
  N 6144 × K 2048, M = 32) but 0.58–0.95× the time when the next calls re-read the same weight, 1 run
  *(field, E5)*. In program order, without the remap, they took up to 1.8% more time
  flushed, but 5–16% less at N 6144 × K 2048, M ≤ 16 when the next calls re-read the same weight, 1 run
  *(field, E6)*. The ruler flushes before each call
  ([triton/tuning](../triton/tuning.md) §1).
- **The split's reduce is a second launch.** At N 2624 × K 6144, M = 64 the call read 12.49–12.65 µs over 4 runs
  *(field, E7)*: 4.2 µs more than its GEMM kernel alone under the ruler's timer
  (Kineto), the timer's cost per kernel rather than the reduce kernel's own time
  *(field, E8)*, and 1.87 µs more under a device-graph event timer
  *(field, E9)*. What a launch costs under each timer:
  [triton/authoring](../triton/authoring.md) §7.
- **XCD runs cross split boundaries at M = 64.** The remap over the 246-program grid hands each XCD 30 or 31
  consecutive split-major tiles; the 41-tile splits do not align with them, so five XCDs straddle two splits
  (computed from the remap; [triton/gemm](../triton/gemm.md) §9).

## Mechanism

- **bf16 GEMM at M ≤ 64**, `out = a @ b.T` with `b` (N, K), no bias: one kernel for N 6144 × K 2048 (M = 2, 4, 8, 16, 32, 64) and one for
  N 2624 × K 6144 (M = 2, 4, 8, 64). One workgroup per N tile (192 at N 6144 × K 2048, 164 at N 2624 × K 6144), each `tl.dot` accumulating in fp32,
  the weight loaded with `.cg`; each XCD streams one contiguous run of N tiles at N 2624 × K 6144 for M ≤ 8 and at N 6144 × K 2048 for M ≤ 16,
  8-tile chunks (3 runs per XCD) at N 6144 × K 2048 for M 32, 64 (the N 2624 × K 6144 remap is a permutation for any grid size); at
  some M a per-workgroup K phase. One launch, except N 2624 × K 6144 at M = 64: split-K 6 into fp32 scratch (246 workgroups), then
  a second kernel adds the six partials in a fixed order (split 6 measured fastest of seven split counts at this size, split 4 at 1.11–1.15× its time: [triton/gemm/small_m_gemm](../triton/gemm/small_m_gemm.md) §6):
  entry 4 of § Code.

| Shape | M | Tile M × N, `BLOCK_K` | Workgroups × warps | N tiles per XCD | K phase | Launches |
|---|---|---|---|---|---|---|
| N 6144 × K 2048 | 2, 4, 8, 16 | 16 × 32, 256 | 192 × 2 | one run of 24 | none | 1 |
| | 32 | 32 × 32, 64 | 192 × 4 | 3 runs of 8 | `KROT` 1 | 1 |
| | 64 | 64 × 32, 256 | 192 × 4 | 3 runs of 8 | `KROT` 1 | 1 |
| N 2624 × K 6144 | 2, 4, 8 | 16 × 16, 512 | 164 × 4 | one run of 20 or 21 | `KROT` 1 | 1 |
| | 64 | 64 × 64, 128, K split 6 ways | 246 × 8, then the reduce kernel, 164 × 4 | one run of 30 or 31 split-major tiles | `KROT` 0 | 2 |

- **One kernel per shape, its sizes chosen by host branches on M.** The N 6144 × K 2048 kernel has two kernel
  functions, one launched at M ≤ 16 and one at M = 32 and 64; the N 2624 × K 6144 kernel has one GEMM function for
  its four sizes and the reduce function. M is a runtime argument, so M = 2, 4 and 8 run one binary in each kernel
  ([triton/gemm](../triton/gemm.md) §9); the constexprs and the launch options come from a `@functools.cache`
  compile function (entry 6).
- **The weight is read once**, by the source's construction, not by a measured HBM-traffic count. One M tile covers
  all of `a` (16 rows for M ≤ 16, then 32 or 64), and each workgroup owns one N tile across all rows of `a`, walks
  all of K (a sixth of it at N 2624 × K 6144, M = 64), accumulates in fp32 registers and stores once
  ([triton/gemm/small_m_gemm](../triton/gemm/small_m_gemm.md) §1). The weight tile is loaded in the layout `tl.dot`
  takes, with `.cg` (entry 1).
- **Tile order and K phase.** The program id is remapped so that each XCD gets N tiles in contiguous runs (entries
  2 and 3). Where `KROT` is 1, a workgroup starts its K walk at step `(pid // NXCD) % n_kb`, `pid` the program id
  before the remap and `n_kb` the number of `BLOCK_K` steps. Flipped one at a time with the output checked, on the
  ruler's timer (rocprofiler-sdk), `KROT = 0` took 11.6% more time at N 6144 × K 2048, M = 64, 1.1% more at M = 32
  and 1.6–2.4% more at N 2624 × K 6144, M = 2, 4, 8, while `KROT = 1` at N 2624 × K 6144, M = 64 took 0.4% more,
  inside that size's A/A threshold, 1 run *(field, E10)*; M = 1 and 16 were not timed.
- **`num_stages = 3` on every GEMM launch** pipelines the plain `tl.dot` loop with direct-to-LDS copies; the
  N 2624 × K 6144 kernel's 16-row path compiles to none of them at `num_stages=1`
  ([triton/gemm](../triton/gemm.md) §9). On that kernel's M ≤ 16 launch `num_stages=1` took 2.2–2.3× the time of 3
  at M = 2, 4, 8 (18.33–18.61 against 7.97–8.36 µs, 3 runs) *(field, E11)*, and fewer stages
  slowed the N 6144 × K 2048 kernel at M = 2, 4, 8, 16 by 1.533–1.625×, one run *(field, E13)*
  ([triton/authoring](../triton/authoring.md) §2).
- **Warps beyond the tile replicate it.** At N 2624 × K 6144, M ≤ 8 the 16 × 16 tile runs on 4 warps, compiled to
  `warpsPerCTA = [4, 1]`, every wave issuing all 192 MFMAs: the extra warps add copy bandwidth, not MFMA
  parallelism (kept, not separated; [triton/gemm](../triton/gemm.md) §9).
- **Resources** *(field, E14)*: 7 compiled kernels, `num_warps` 2, 4 or 8,
  `waves_per_eu` 2 in 6, VGPR 26–106, LDS up to about 148 KiB of the pipeliner's buffers,
  `v_mfma_f32_16x16x32_bf16`. Each GEMM kernel but the split-K reduce holds `buffer_load_dwordx4 … lds`.

## Code

### 1. The weight tile loaded as `[BLOCK_K, BLOCK_N]`, with no `tl.trans` (N 2624 × K 6144)

From the N 2624 × K 6144 kernel's GEMM function, which runs all four of its sizes. For a row-major (N, K) weight
and `out = a @ b.T` it builds the `[BLOCK_K, BLOCK_N]` tile directly,
`b_offs = offs_k[:, None] + offs_n[None, :] * K`, so `tl.dot(a, b)` needs no `tl.trans`, and it loads the weight
with `cache_modifier=".cg"`. `pid_n` is the workgroup's N tile and `pid_k` its K split (0 where K is not split),
`k_chunk` the K range of one split, `n_kb` its number of `BLOCK_K` steps and `kphase` the step this workgroup's K
walk starts at; `EVEN_M` is set when M fills the tile, so the `a` load takes no row mask.

```python
# the N 2624 × K 6144 kernel: one N tile (pid_n) per workgroup over all rows of A; pid_k is the K split (entry 4), kphase the first step of this workgroup's K walk
offs_m = tl.arange(0, BLOCK_M)
offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
offs_k = tl.arange(0, BLOCK_K)
row_ok = offs_m[:, None] < M
a_offs = offs_m[:, None] * K + offs_k[None, :] + pid_k * k_chunk
b_offs = offs_k[:, None] + offs_n[None, :] * K + pid_k * k_chunk  # [BLOCK_K, BLOCK_N], as tl.dot takes it
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
for i in tl.range(0, n_kb):
    kb = ((i + kphase) % n_kb) * BLOCK_K
    if EVEN_M:
        a = tl.load((A + kb) + a_offs)
    else:
        a = tl.load((A + kb) + a_offs, mask=row_ok, other=0.0)
    b = tl.load((B + kb) + b_offs, cache_modifier=".cg")
    acc = tl.dot(a, b, acc=acc)
```

Not this: loading `[BLOCK_N, BLOCK_K]` and calling `tl.trans`. In this kernel that added `ds_bpermute_b32`
shuffles (12 and 16 against 0 in the M = 2, 4, 8 and M = 64 tiles) *(field, E15)* and
took 1.09–1.17× the time at M = 2, 4, 8 and 1.01× at M = 64 in two runs on the ruler's timer, one read from Kineto
and one from rocprofiler-sdk *(field, E16, E15)*.

Explained in [triton/gemm/small_m_gemm](../triton/gemm/small_m_gemm.md) §2.

### 2. An XCD remap that is a permutation for any grid size (N 2624 × K 6144)

The N 2624 × K 6144 kernel remaps the program id so that each XCD gets one contiguous run of tiles: `per` tiles on
the first `tall` XCDs and `per - 1` on the rest. `pid` is the launch's program id, `GRID` the number of programs
(164 at M ≤ 8, one per N tile; 246 at M = 64, N tiles times K splits) and `NXCD` is 8; the result is the tile the
workgroup computes. It relies on workgroup `i` running on XCD `i % 8`, as on the tested grids on the default stream
*(field, E17)*; on a new stream the XCD index rotates by a constant and workgroups
congruent mod 8 still share an XCD. Enumerated from the formula, the remap is a permutation for every grid of
1–4096 tiles, a grid that 8 does not divide among them (164 = 8 × 20 + 4).

```python
# pid: the launch's program id; the result is the N tile (times the K split at M = 64) this workgroup computes
@triton.jit
def _xcd_remap(pid, GRID: tl.constexpr, NXCD: tl.constexpr):
    per: tl.constexpr = (GRID + NXCD - 1) // NXCD
    tall: tl.constexpr = GRID % NXCD if GRID % NXCD else NXCD
    xcd = pid % NXCD
    local = pid // NXCD
    if xcd < tall:
        tile = xcd * per + local
    else:
        tile = tall * per + (xcd - tall) * (per - 1) + local
    return tile
```

Not this: `tile = xcd * (GRID // 8) + local` on a grid that 8 does not divide. On this kernel's 164 tiles at M = 8
it skipped tiles 160–163 (columns 2560–2623 stayed NaN) and computed tiles 20, 40, 60 and 80 twice, yet timed
0.999× and 0.993× the kernel with its own remap in two runs (back-to-back calls, no flush), within the run spread
*(field, E18)*: timing does not flag a remap that leaves tiles unwritten.

Explained in [triton/gemm/small_m_gemm](../triton/gemm/small_m_gemm.md) §3.

### 3. Runs of N tiles per XCD on a 192-tile grid (N 6144 × K 2048)

The N 6144 × K 2048 kernel has 192 N tiles, a multiple of `NXCD` = 8, and two kernel functions with a remap each.
The function launched at M ≤ 16 gives each XCD one run of `N_TILES // NXCD` = 24 tiles:

```python
# the kernel function launched at M <= 16: one run of N_TILES // NXCD tiles per XCD
pid = tl.program_id(0)
pid = (pid % NXCD) * (N_TILES // NXCD) + (pid // NXCD)
```

The function launched at M = 32 and 64 deals runs of `CHUNK` = 8 tiles, 3 runs per XCD over the 192 tiles. The
elided line takes the K phase from the program id before the remap, `kphase = ((pid // NXCD) * KROT) & (n_kb - 1)`:

```python
# the kernel function launched at M = 32 and 64: runs of CHUNK tiles per XCD
pid = tl.program_id(0)
# ... (the K phase, taken from the unmapped pid)
blk: tl.constexpr = NXCD * CHUNK
local = pid // NXCD
pid = (local // CHUNK) * blk + (pid % NXCD) * CHUNK + (local % CHUNK)
```

Not this: either remap on the other kernel's grid of 164 tiles. Neither is a permutation there (enumerated from the
formulas): the `CHUNK` remap misses 12 tiles and sends 12 programs past the grid. Check bijectivity for each grid
size used (entry 2).

Explained in [triton/gemm/small_m_gemm](../triton/gemm/small_m_gemm.md) §3.

### 4. Split-K into fp32 scratch and a fixed-order reduce kernel (N 2624 × K 6144, M 64)

Both kernels must be bitwise reproducible, so a K split has to be summed in a fixed order. At M = 64 the
N 2624 × K 6144 kernel splits K six ways (41 N tiles × 6 splits = 246 workgroups): each split stores its fp32
partial tile into its plane of a `[split][m][n]` scratch, and a second kernel adds the planes in split order
0 to 5, the same order in every run, then rounds to bf16 once, after the sum. With one split (the helper's
M ≤ 16 branch; of the kernel's sizes M = 2, 4, 8) the same GEMM function rounds its tile and stores it to the
output, which the host passes as `W`. The end of the GEMM function (`pid_k` is the split, `row_ok` the mask of the
rows below M):

```python
# the end of the GEMM kernel. One split (M ≤ 16): round and store to the output, which the host passes as W.
# SPLIT_K > 1 (M = 64): split pid_k stores its fp32 partial tile into plane pid_k of the scratch W
if SPLIT_K == 1:
    c_ptrs = W + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, acc.to(W.dtype.element_ty), mask=row_ok)
else:
    w_ptrs = W + pid_k * (M * N) + offs_m[:, None] * N + offs_n[None, :]
    tl.store(w_ptrs, acc, mask=row_ok)
```

The second kernel. `TOTAL` is M × N, the size of one plane, and each program adds `BLOCK` outputs:

```python
# second kernel: adds the planes in split order 0, 1, ..., SPLIT_K - 1, then rounds once
@triton.jit
def _reduce_kernel(W, C, TOTAL, SPLIT_K: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ok = offs < TOTAL
    acc = tl.load(W + offs, mask=ok, other=0.0, cache_modifier=".cg")
    for s in tl.static_range(1, SPLIT_K):
        acc += tl.load(W + s * TOTAL + offs, mask=ok, other=0.0, cache_modifier=".cg")
    tl.store(C + offs, acc.to(C.dtype.element_ty), mask=ok)
```

The host helper's arm for M = 64, cached per pair of operand shapes as entry 6's is. It gives each reduce program
1024 outputs and returns the scratch size in elements with the launch closure:

```python
# host, the per-shape helper's M = 64 arm (cached): split_k = 6, total = m * n; it returns the scratch size with the launch
def launch(a, b, out, ws):
    _gemm_kernel[(n_tiles * split_k,)](
        a,
        b,
        ws,
        # ... (m, the constexprs and the launch options)
    )
    _reduce_kernel[(triton.cdiv(total, 1024),)](
        ws, out, total, SPLIT_K=split_k, BLOCK=1024, num_warps=4
    )

return launch, split_k * total
```

The host entry allocates the scratch in the call with `torch.empty`, which holds undefined bytes: every element is
written by the GEMM launch before the reduce launch reads it:

```python
# host entry, per call: ws_elems is the helper's scratch size, 0 (no scratch) where K is not split
ws = (
    torch.empty(ws_elems, dtype=torch.float32, device=a.device)
    if ws_elems
    else None
)
launch(a, b, out, ws)
```

Not this: each split adding its partial into `out` with a float atomic add. The sum then follows the arrival order,
and two runs need not give the same bytes ([triton/numerics](../triton/numerics.md) §8): on a test GEMM with this
kernel's M = 64 shapes a float-atomic K split gave 100 different outputs in 100 runs, the fixed-order reduce one
output, bit-equal to a sequential sum in split order *(field, E1)*.

Explained in [triton/gemm/small_m_gemm](../triton/gemm/small_m_gemm.md) §6 and [triton/gemm](../triton/gemm.md) §9.

### 5. The launch options, kept and not separated (N 6144 × K 2048, M 32)

The N 6144 × K 2048 kernel's launch for 16 < M ≤ 32, of its sizes M = 32: a 32-row tile, `BLOCK_K` 64, runs of
`CHUNK` 8 tiles per XCD, `KROT` 1, and `UNROLL` 4, the `loop_unroll_factor` of the K loop's `tl.range`.
`waves_per_eu = 2` and `num_stages = 3` are on every GEMM launch of both kernels and `matrix_instr_nonkdim = 16` on
all but this kernel's M ≤ 16 launch, which passes none. Of these three options only `num_stages` was timed on its
own, at M ≤ 16 on both kernels (§ Mechanism), not at this launch's M = 32; the other two are kept, not separated:
used, their own effect not measured. Of this launch's constexprs, `KROT` was flipped alone (§ Mechanism). In this
build `schedule_hint` changes nothing, and `UNROLL` neither the MFMA count nor the absence of a K loop: Triton 3.8.0
reads the hint nowhere (S1), and this launch compiled to byte-identical AMDGCN with
and without it *(field, E19)*; compiled at k = 2048 the K loop is fully unrolled, 64
MFMAs and no loop, with `UNROLL` 4 or 1 alike *(field, E21)*. Triton 3.7.0 turned the
hint into the `iterative-ilp` scheduler: to keep that scheduler on Triton 3.8.0, pass
`llvm_fn_attrs=[["amdgpu-sched-strategy", "iterative-ilp"]]`, which gave the same attribute on 3.8.0 and was
dropped by 3.7.0 *(field, E19)*, and have the host refuse to build the kernel on a
Triton older than 3.8 ([triton/tuning](../triton/tuning.md) §9).

```python
# the N 6144 × K 2048 kernel's launch for 16 < m <= 32; grid = (n // 32,)
_gemm_xcd_kernel[grid](
    a,
    b,
    out,
    m,
    N=n,
    K=k,
    BLOCK_M=32,
    BLOCK_N=32,
    BLOCK_K=64,
    NXCD=8,
    CHUNK=8,
    EVEN_M=even_m,
    UNROLL=4,
    KROT=1,
    num_warps=4,
    num_stages=3,
    waves_per_eu=2,
    matrix_instr_nonkdim=16,
    schedule_hint="memory-bound-attention",
)
```

Explained in [triton/gemm/small_m_gemm](../triton/gemm/small_m_gemm.md) §9.

### 6. A `@functools.cache` compile function that returns the launch closure (N 6144 × K 2048)

The configuration of both kernels comes from a `@functools.cache` compile function of the two operand shapes that
returns the launch closure: a module-level def whose cache holds compiled code, constants and per-shape scratch
only (neither of these two kernels caches scratch; a cached workspace is the counter-example of entry 7). The
N 6144 × K 2048 kernel builds its closure once per (a_shape, b_shape): host branches on M choose the kernel
function, its constexprs and the launch options, and `run` looks the closure up and calls it. M itself is passed at
run time (`m`), `EVEN_M` is set at M = 16, where M fills the 16-row tile, and `block_k` 512 serves M = 1, which is
not one of the kernel's sizes. The N 2624 × K 6144 kernel's helper returns its scratch size with the closure, and
its host allocates the scratch in each call (entry 4).

```python
@functools.cache
def _launch_for(a_shape, b_shape):
    m, k = a_shape
    n = b_shape[0]
    grid = (n // 32,)
    if m <= 16:
        even_m = m == 16
        block_k = 512 if m == 1 else 256

        def launch(a, b, out):
            _gemm_m16_kernel[grid](
                a,
                b,
                out,
                m,
                N=n,
                K=k,
                BLOCK_M=16,
                BLOCK_N=32,
                BLOCK_K=block_k,
                N_TILES=n // 32,
                NXCD=8,
                EVEN_M=even_m,
                num_warps=2,
                num_stages=3,
                waves_per_eu=2,
            )

        return launch
    # ... the branches for m <= 32 and m <= 64 return their own closures


def run(*, a, b, trans_b, bias, out_dtype, out):
    launch = _launch_for(tuple(a.shape), tuple(b.shape))
    launch(a, b, out)
```

Not this: `@triton.autotune` or `@triton.heuristics` above `@triton.jit` to choose the configuration. An autotuner
times its unpruned configurations at a key's first miss in a process, unless one configuration remains or the disk
cache holds the choice, so its pick among equal-cost configurations is a timing artifact: in a probe, 16 processes
run four at a time on one GPU split 10 and 6 between two such configurations *(field, E22)*, so
processes can run one size with different binaries.

Explained in [triton/authoring](../triton/authoring.md) §5.

### 7. Not this: a split-K workspace cached per (device, stream) (another kernel: N 32 × K 6144, M 6–96)

This block is a counter-example, and it is not from the two kernels above: it is from a third Triton bf16 GEMM,
N 32 × K 6144 at M 6–96, which folds its K split inside one launch, each split counting in on its tile's counter and
the last to arrive adding the partials in split order ([gemm/wait_free_fold](../gemm/wait_free_fold.md)). That
kernel keeps its partials and counters (`_MAX_PART` fp32 elements, `_MAX_TILES` counters) in a `functools.cache`
keyed on the device and the current stream, zeroes the counters once and relies on the folding split's reset to 0.
Under capture the key is the capture stream, and torch's default capture stream is one per process, so graphs
replayed at once, or a replay beside an eager call on the capture stream, share one set of partials and counters:
eager calls on the capture stream beside replays on another stream gave 30–33 wrong outputs of 400, and two graphs
replayed on two streams 164–214 of 400 (relative error up to 2.2), while eager calls on one or two streams, and
beside a matmul process, gave none *(field, E24)*. The right form is scratch that each call
owns, as entry 4 allocates it, so that overlapping calls share no synchronization memory. For this kernel,
allocating and zeroing the workspace for each call made under capture fixed the wrong outputs, but the zeroing is
captured and runs on every replay: 0.77–1.10 per size over aiter's tuned bf16 GEMM in graph replays, against
1.33–1.95 with the cached workspace *(field, E25)*; carving each capture's slice from a
region zeroed once before capture was not tested.

```python
# Not this: one workspace per (device, stream), zeroed once, shared by every graph captured on the stream
@functools.cache
def _workspace(device, stream):
    p = torch.empty(_MAX_PART, device=device, dtype=torch.float32)
    cnt = torch.zeros(_MAX_TILES, device=device, dtype=torch.int32)
    return p, cnt

p, cnt = _workspace(a.device, torch.cuda.current_stream(a.device).cuda_stream)
```

Explained in [gemm/wait_free_fold](../gemm/wait_free_fold.md) § Cost and risk.

## Result

- **Speedups** over aiter's `gemm_a16w16` *(field, E26)*: 1.048 (N 2624 × K 6144,
  M 2) to 1.292 (N 6144 × K 2048, M 8).
- **Per size**, one measurement each *(field, E2)*; how much of the spread over M is the baseline's
  was not separated ([triton/gemm](../triton/gemm.md) §9):

  | M | 2 | 4 | 8 | 16 | 32 | 64 |
  |---|---|---|---|---|---|---|
  | N 6144 × K 2048 | 1.116 | 1.129 | 1.292 | 1.138 | 1.064 | 1.116 |
  | N 2624 × K 6144 | 1.048 | 1.070 | 1.062 | no kernel | no kernel | 1.174 |

- **Call time against a plain read of the weight**, the ruler's timer (Kineto), over 4 runs
  *(field, E7)*: at small M the call ran close to the read, within 1.2× at
  N 6144 × K 2048, M 2–16 and within about 1.22× at N 2624 × K 6144, M 2–8; at M = 64 it took 1.52–1.54× and
  1.78–1.83× the same run's read.

  | Shape (weight) | Small M | M 32 | M 64 | Plain read of the weight |
  |---|---|---|---|---|
  | N 6144 × K 2048 (25.17 MB) | 5.94–6.38 µs at M 2–16 | 7.16–7.22 µs | 8.53–8.56 µs | 5.53–5.61 µs |
  | N 2624 × K 6144 (32.24 MB) | 7.98–8.40 µs at M 2–8 | no kernel | 12.49–12.65 µs | 6.90–7.02 µs |

- **Under an earlier form of the ruler**, in the runs that tuned the configurations (Triton version not recorded,
  repeat count unknown), the lower bounds against aiter were 1.09–1.31 (N 6144 × K 2048, M 2–64), 1.085–1.106
  (N 2624 × K 6144, M 2, 4, 8) and 1.236 (N 2624 × K 6144, M 64) *(field, E27)*.
  [cases/bf16_glm52_wide_n](bf16_glm52_wide_n.md) tables another run per size, beside the FlyDSL kernels.

The ruler, its timer and the bar a size is counted against: [triton/tuning](../triton/tuning.md) §1.

## Evidence

- E1 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2, repeated calls compared bitwise] the N 6144 × K 2048 kernel at m = 2 – 64 and the N 2624 × K 6144 kernel at m = 2, 4, 8: one distinct output in 50 runs at each of the 9 sizes, inside the accuracy bound, 0 atomics in the sources; a test GEMM with 6 fp32 partial planes [6, 64, 2624] summed in split order by a second kernel: one output in 100 runs, bit-equal to the sequential sum (reverse order differs in 90,596 values), a float-atomic K split 100 distinct outputs in 100 runs; the N 2624 × K 6144 kernel itself at m = 64: one output in 50 runs; captured into a CUDA graph and timed there at m = 64 with its per-call scratch, no finding other than speed
- E2 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1 per value, the ruler (Kineto), speedups as [triton/tuning](../triton/tuning.md) §1 defines them, against aiter's `gemm_a8w8_blockscale` for the fp8 block-scaled GEMMs and `gemm_a16w16` for bf16] one measurement per size, from five measurement runs: n3072 m = 1, 2, 4, 8, 16, 32, 128: 1.429, 1.397, 1.307, 1.435, 1.406, 1.418, 1.477 and n7168 m = 1, 2, 8, 16: 1.664, 1.811, 1.840, 1.725 (one run); n3072 m = 64, 256, 8193: 1.412, 1.199, 1.486, N 2624 × K 6144 m = 2, 4, 8, 64: 1.048, 1.070, 1.062, 1.174 and N 6144 × K 2048 m = 2, 4, 8, 16, 32, 64: 1.116, 1.129, 1.292, 1.138, 1.064, 1.116 (a second); n3072 m = 6144, 8192: 1.246, 1.298 (a third); n3072 m = 1024, 2048, 4096: 1.191, 1.339, 1.356 and n7168 m = 32, 64, 128, 256: 1.759, 1.488, 1.114, 1.138 (a fourth); n7168 m = 4: 1.679 (a fifth); a later run read n7168 m = 1 at 1.728
- E3 [2026-11, gfx950 (MI355X), FlyDSL ? and HIP kernels beside Triton 3.8.0 ones, N=1, a reference run of the ruler, stored speedup against the op's baseline] one run: the FlyDSL bf16 GEMM N 6144 × K 2048 M 1 1.191, M 2, 4, 8, 16, 32, 64 1.045, 0.403, 0.445, 0.392, 0.361, 0.360, M 128–4096 0.450–0.864; the FlyDSL bf16 GEMM N 2624 × K 6144 M 1 1.134, M 2, 4, 8, 16, 32, 64 0.420, 0.415, 0.419, 0.396, 0.410, 0.553, M 128–4096 0.623–0.998; the HIP add + RMSNorm (hidden size 6144) 1.038 at 32 tokens, 1.067 at 128, 1.040–1.072 at 512–8192; the Triton add + RMSNorm 1.035 at 32, 1.062 at 128
- E4 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, the ruler's timer (Kineto), 4 alternated rounds per arm; GEMV against MFMA under the same timer, 5 alternated rounds] the N 6144 × K 2048 and N 2624 × K 6144 Triton kernels run at m = 1 and 2 against FlyDSL kernels for N 6144 × K 2048 and N 2624 × K 6144 run at m = 1, 2: FlyDSL / Triton time at m = 1 0.9754 (5.86 against 6.01 µs; pairs 0.973–0.979) and 0.9332 (7.27 against 7.79 µs; pairs 0.926–0.944), FlyDSL lower in every round; at m = 2 Triton / FlyDSL 0.9167 at the N 6144 × K 2048 shape (5.97 against 6.51 µs) and 0.40 at the N 2624 × K 6144 shape (8.03–8.05 against 19.98–20.09 µs per round); a plain-Triton GEMV (fp32 `tl.sum` over all K, `.cg` on B, six (BN, BK) tiles) against the 16-row MFMA tile at m = 1 on five shapes: best GEMV / best MFMA 0.972 at N 6144 × K 2048, 1.174 at (2624, 6144), 1.049 at (4096, 2048), 1.090 at (2048, 7168), 1.019 at (6144, 4096), max error 3.5e-03
- E5 [2026-11, gfx950 (MI355X), Triton 3.8.0, torch 2.11.0+rocm10.0.0, N=1, CUDA events, median of 40 calls × 15 rounds; cold: a 1 GiB flush before each call; warm: one call after the flush, then a spin kernel before each call; graph: 20 back-to-back calls] the two kernels without `cache_modifier=".cg"` on the B load, time over the kernels with it, N 6144 × K 2048 m = 2–64 and N 2624 × K 6144 m = 2, 4, 8, 64: cold 1.011–1.034 (N 6144 × K 2048 m = 32 1.000, not stable), warm 0.728–0.947, graph 0.585–0.875
- E6 [2026-11, gfx950 (MI355X), Triton 3.8.0, torch 2.11.0+rocm10.0.0, N=1, the three timers of E5] the two kernels with program order (`pid = raw`) in place of the XCD remap, time over the remapped kernels: cold at most 1.018 (N 6144 × K 2048 m = 2–64, N 2624 × K 6144 m = 2, 4, 8, 64); N 6144 × K 2048 m = 2, 4, 8, 16 warm 0.913–0.949, graph 0.838–0.910
- E7 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=4, the ruler's timer (Kineto), in runs of the ruler for the GEMMs; a Gluon read-and-sum of the same bytes under the same timer for the plain read] over 4 runs: N 6144 × K 2048 (25.17 MB weight) at m = 2–16 5.94–6.38 µs (3.95–4.23 TB/s), m = 32 7.16–7.22, m = 64 8.53–8.56 µs, plain read 5.53–5.61 µs (4.48–4.55 TB/s); N 2624 × K 6144 (32.24 MB) at m = 2–8 7.98–8.40 µs, m = 64 12.49–12.65 µs, plain read 6.90–7.02 µs; m = 64 over the same run's read 1.52–1.54 (N 6144 × K 2048) and 1.78–1.83 (N 2624 × K 6144). Two of the runs per m: N 6144 × K 2048 m = 2, 4, 8, 16, 32, 64 5.95 / 5.98, 5.98 / 5.98, 6.05 / 6.07, 6.36 / 6.34, 7.21 / 7.19, 8.56 / 8.54 µs, read 5.61 / 5.59; N 2624 × K 6144 m = 2, 4, 8, 64 8.03 / 8.01, 8.12 / 8.09, 8.39 / 8.35, 12.49 / 12.62 µs, read 7.02 / 6.97
- E8 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, the ruler's timer (Kineto), median of 3; Kineto trace of a graph replay for the kernel list] the N 2624 × K 6144 kernel at m = 64: whole call 12.57 µs, its GEMM kernel alone 8.374 µs, difference 4.2 µs; the replay runs the GEMM kernel and the reduce kernel at m = 64, the GEMM kernel only at m = 2
- E9 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, one GPU run, one size, bf16, preallocated device graphs timed by events, 1 GiB flush outside each event window; confirmation by 99 % family-corrected paired-log Student-t intervals against the larger of the local and global A/A floors plus the AB/BA order gap] (M, N, K) = (64, 2624, 6144), `BLOCK_M` 64, `BLOCK_K` 128, `KROT` 0, 8 warps, 3 stages, `waves_per_eu` 2: all 21 split × `BLOCK_N` configurations accurate on 3 seeds and byte-identical over 100 repeats; competitor / split 6 (`BLOCK_N` 64) time 1.744 / 1.242 / 1.085 / 1.215 / 1.178 for splits 1 / 2 / 3 / 8 / 12, 1.095 (1.068–1.123) for split 4, unresolved; split 6 whole / partial 16.140 / 14.273 µs, difference 1.868 µs (1.466–2.269); clocks and package versions not recorded (plain-Triton copies of the N 2624 × K 6144 kernel)
- E10 [2026-11, gfx950 (MI355X), Triton 3.8.0, torch 2.11.0+rocm10.0.0, N=1, the ruler's timer (rocprofiler-sdk), 30 random-order rounds] each kernel's `KROT` flipped, outputs within the accuracy bound, flipped over unflipped time: N 6144 × K 2048, 1 → 0: 1.116 at m = 64, 1.011 at m = 32; N 2624 × K 6144, 1 → 0: 1.016, 1.024, 1.019 at m = 2, 4, 8; m = 64, 0 → 1: 1.004, under that size's A/A threshold of 1.7 %
- E11 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3 (the toy MoE and phase launches N=4), Gluon kernels in CUDA graphs: a split reduction of 16 splits × 16384 fp32 outputs, 200 replays per arm, compared bitwise with a partial + merge launch pair; one stream writing a 16,777,216-float region, reading it last, then overwriting it, 20 eager calls and 20 replays; the ruler on one GPU, a kernel (A) and a copy with one change (B) alternated, 4 runs per arm, median kernel time; a Gluon stream reading a 1 GiB bf16 tensor, timed with CUDA events; small Triton kernels timed as a CUDA graph replayed back to back (wall time) and by the ruler's timer (Kineto)] fold with its counter zeroed earlier in the same graph: 200 of 200 replays equal with `acq_rel` and 200 of 200 with `relaxed`, whose scratch the reference replay had just filled with the same partials (E12); without the zeroing node wrong from the second replay on, 199 of 199; replay time 76.73 µs (zero + fold) against 7.6 µs (partial + merge); the last reader saw the values from before the overwrite in 20 of 20 eager calls and 20 of 20 replays; the N 2624 × K 6144 bf16 GEMM with `num_stages` 3 → 1 on the m ≤ 16 launch, A against B at m = 2, 4, 8: 7.97, 8.11, 8.35 against 18.34, 18.43, 18.49 µs; 8.03, 8.13, 8.36 against 18.45, 18.53, 18.61 µs; 8.00, 8.08, 8.34 against 18.33, 18.43, 18.49 µs; speedup B/A 0.433–0.453 (2.21–2.30× the time), the unchanged m = 64 launch 1.003, 1.000, 0.9948, both arms accurate in every run; the stream at 96 programs × 4 warps with one 16-byte load per lane in flight (0.375 MB) against eight (3 MB): 885.2 against 3995, 876 against 3945, 885.9 against 3985 GB/s; a toy decode MoE (32 experts, top-4, hidden 256, intermediate 128, 2 tokens) recomputing top-k in its 40 consumer programs against a selection launch, same output (relative L2 0): 9.67 / 9.69 against 11.41 / 12.76, 9.89 / 9.52 against 11.13 / 12.75, 9.67 / 9.78 against 11.47 / 12.68, 9.875 / 9.793 against 11.621 / 13.193 µs (replayed / the ruler's timer); removing one of two elementwise phase launches saved 1.15, 1.87, 1.79; 1.61, 1.88, 1.80; 1.84, 1.74, 1.76; −0.102, 1.587, 0.325 µs of replay time at 16384, 131072, 1048576 elements, outputs equal, and 4.19, 4.19, 4.17, 4.17 µs on the ruler's timer at 131072 elements (the N 2624 × K 6144 bf16 GEMM and probe kernels)
- E12 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, Gluon fold kernels in CUDA graphs, 3000 replays per arm and size, the partials set to NaN before every replay, counters zeroed by an earlier node] 8 and 32 splits × 16384 and 262144 fp32 outputs: `relaxed` 3000 of 3000 replays wrong at each size, `acq_rel` and `relaxed` with `.wt` partial stores 0 of 3000; one `buffer_wbl2` and one `buffer_inv` in each `acq_rel` fold, none in the `relaxed` ones (Gluon fold probes)
- E13 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, the ruler on one GPU: `num_stages` A/B 4 runs per arm, the Gluon bf16 GEMM skeleton of [triton/authoring](../triton/authoring.md) §10 against the plain-Triton N 6144 × K 2048 GEMM and the skeleton with one-step K prefetch 3 runs per arm in a fixed order; Gluon fp8 GEMMs against aiter's plain-Triton kernel in 5 flushed graph rounds in one process; the skeleton's direct calls against the op's reference, 20 eager calls, 20 replays and a second process per input] fewer stages: the N 6144 × K 2048 bf16 GEMM at m = 2, 4, 8, 16 1.533–1.625× slower, RMSNorm at 1024, 2048, 4096 tokens 1.047–1.120×, 1.092–1.095×, 1.051–1.052× slower; unshuffled E4M3 fp8 GEMMs with fp32 block scales (N 3072, K 1536) at m = 2, 8, 16 0.777–0.781×, 0.755–0.764×, 0.759–0.768× of aiter's time, both within bounds; the skeleton at N 6144, K 2048 and M = 2, 3, 4, 8, 12, 16, 17, 32, 33, 48, 64 with one positive and one signed input each: bitwise stable, 0–1 bf16 ulp on the positive input, 0.960–0.984 of the bound on the signed one, 8192 B of `metadata.shared` passed at every launch while the amdgcn's group segment read 0; checked and timed at m = 2, 4, 8, 16, 32, 64: no correctness, build or capture failure, all six sizes in each of 3 runs at speedup 0.373–0.434 over the baseline; 2.882–3.158× the time of the plain-Triton N 6144 × K 2048 GEMM at m = 2, 4, 8, 16, and 1.695–1.729× faster with one-step K prefetch
- E14 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, every kernel built and run once, each with its own `TRITON_CACHE_DIR`, on 4 GPUs; the cached `.amdgcn` and `.json` read] `num_warps`, `waves_per_eu`, VGPR (`.vgpr_count`, AGPRs included), AGPR, LDS, spills, MFMA opcodes and the count of every opcode (loads, stores and atomics included, buffer loads into LDS apart) of every kernel function of the library (a `@gluon.jit` or `@triton.jit` def of a kernel's sources) compiled while running each of the library's 154 kernels once: 897 compiled-cache records among 1709 compiled kernels, one per compilation, not deduplicated; N 3072 fp8 GEMM LDS 9, 9, 12, 36, 48, 49 KiB at M 1, 2, 4, 8, 16, 32; N 7168 at M 16, 32, 64 64.0, 66.1, 72.0 KiB; MM64-FP4 at 2048, 4096, 6144, 8192, 16384 tokens: 0 atomic instructions (global, buffer, flat, ds), popcounts (`s_bcnt`) 4, 2, 4, 8, 8; every GDN prefill kernel (26) holds `v_mfma_f32_16x16x32_bf16` and `v_mfma_f32_16x16x4_f32`; 45 GDN decode compiled kernels, 0 MFMA; RMSNorm loads with the nt bit, stores sc0 sc1 only, 16 B of LDS (all 154 Triton kernels)
- E15 [2026-11, gfx950 (MI355X), Triton 3.8.0, torch 2.11.0+rocm10.0.0, N=1, the ruler's timer (rocprofiler-sdk), 30 random-order rounds; ISA of the kernels the N 2624 × K 6144 kernel launches] the N 2624 × K 6144 kernel with B loaded as `[BLOCK_N, BLOCK_K]` and `tl.trans` against its direct tile: time ratio 1.155, 1.148, 1.087, 1.009 at m = 2, 4, 8, 64; `ds_bpermute_b32` 12 (m = 2, 4, 8 tile) and 16 (m = 64 tile) against 0; at m = 2, 4, 8, 48 against 192 `v_mfma_f32_16x16x32_bf16` and one loop back-branch against none
- E16 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, A/B on the ruler's timer (Kineto), 4 alternated rounds; ISA of a stand-alone `tl.dot` test kernel with the two tiers' tile sizes, not of the N 2624 × K 6144 kernel] the N 2624 × K 6144 kernel with B loaded as `[BLOCK_N, BLOCK_K]` and `tl.trans` against its direct tile: time ratio 1.172, 1.163, 1.121, 1.009 at m = 2, 4, 8, 64; the test kernel with B loaded the same two ways, 16 × 16 × 512 tile, 2 warps: 509 against 350 instructions (16 against 0 `ds_bpermute_b32`); 64 × 64 × 128 tile, 8 warps: 331 against 261 (6 against 0)
- E17 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3, XCC and CU ids read from the hardware in each workgroup, launched on the default stream] XCD = workgroup id % 8 with no mismatch on 1-D grids of 8, 100, 2048 and 4096 workgroups and on 6 multi-axis grids (3 launches each), and with offset 0 in 5 further launches of 4096, in each of 3 runs; 256 CUs (`multi_processor_count`, and 32 distinct CU ids on each of 8 XCCs) in each run; the N 2624 × K 6144 kernel's m ≤ 16 grid read from its source: 164
- E18 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2, CUDA events over 20 back-to-back calls (no flush), median of 9, 5 alternated rounds] the N 2624 × K 6144 kernel at m = 8 with the remap `xcd * (GRID // 8) + local` on 164 tiles: columns 2560 – 2623 NaN in all 8 rows, bitwise equal elsewhere; broken remap / the kernel's own remap 11.93 / 11.94 µs (0.999×; round ranges 11.90 – 12.05 and 11.88 – 12.53), the other run 12.59 / 12.69 µs (0.993×; 12.53 – 12.73 and 12.55 – 12.72)
- E19 [2026-11, gfx950 (offline), Triton 3.8.0 and 3.7.0, N=4, offline compiles of the two kernels] the m = 1 tiles (`BLOCK_M` 16) compiled with M a runtime i32 and `matrix_instr_nonkdim` 16 on both, not as the kernels' launches call them at m = 1 (E20): 64 (N 6144 × K 2048) and 48 (N 2624 × K 6144) `v_mfma_f32_16x16x32_bf16`, 0 `v_dot2`, in each of 4 runs; the N 6144 × K 2048 kernel's m = 32 kernel on 3.8.0 with and without `schedule_hint="memory-bound-attention"`: byte-identical AMDGCN; on 3.7.0 (3.7.0+gitb4e20bbe.rocm7.14.0a20260608) the hinted LLIR differs only in `"amdgpu-sched-strategy"="iterative-ilp"`, the TTGIR is identical and the AMDGCN differs; `llvm_fn_attrs=[["amdgpu-sched-strategy", "iterative-ilp"]]` gives that attribute on 3.8.0 and none on 3.7.0
- E20 [2026-11, gfx950 (MI355X), Triton 3.8.0, torch 2.11.0+rocm10.0.0, N=2, the `.amdgcn` of the JIT entry each call compiled; the ruler's timer (rocprofiler-sdk), 30 random-order rounds] the kernels' launches at m = 1 compile M as the constant 1: the N 2624 × K 6144 kernel 192 `v_mfma_f32_16x16x32_bf16` with no loop back-branch, the N 6144 × K 2048 kernel 64, 0 `v_dot2` in both; the N 2624 × K 6144 kernel with `do_not_specialize=['M']`: 48 with one back-branch, outputs bitwise equal, 1.077× the specialized time
- E21 [2026-11, gfx950 (offline), Triton 3.8.0, N=1, compiled-kernel census of an offline build] the N 6144 × K 2048 kernel's m = 32 kernel (`BLOCK_M` 32, `BLOCK_N` 32, `BLOCK_K` 64, k 2048, 4 warps) built with `UNROLL` 4 and 1: 64 `v_mfma` in each, no backward branch in either
- E22 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, CUDA-graph captures and JIT probes on one GPU, with a tripwire on `JITFunction._do_compile` during each capture; the captured checks of E23 on an RMSNorm copy whose JIT key changed per call; an autotuner probe in 16 fresh processes] a constexpr warmed at 16 and launched at 32 inside a capture compiled there with one `_do_compile` call, and the tripwire raised "compiled in the capture"; the RMSNorm copy failed at every size; two fresh processes sharing a warm `TRITON_CACHE_DIR` each called `_do_compile` once (first launches 0.5631 s and 0.001563 s); an autotuner with two equal-cost configurations, listed in alternating order, timed both at its first call and neither afterwards in each of 16 processes run four at a time, 6 picking the first-listed configuration and 10 the second; a `torch.empty` tensor allocated in a capture kept its address over 3 replays and held the third replay's values; `.item()`, `.cpu()`, `.tolist()`, `torch.cuda.synchronize()` and `torch.nonzero` failed the capture (a plain-Triton RMSNorm copy and probe kernels)
- E23 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2 (the two-phase grid N=4), launches of small Gluon probe kernels with the binaries counted in the JIT cache; planted copies on one GPU checked against the op's reference in captured graphs with poisoned outputs (a first capture and two replays with new inputs, fresh captures per input set, a repeat, a check after the timing) and timed by the ruler, and direct calls against the op's reference; two toy Triton elementwise phases of 131072 elements as two launches and as one grid split by program id, timed as a CUDA graph replayed back to back (wall time) and by the ruler's timer (Kineto)] `INSTR=[16, 16, 32]` as a constexpr failed at the launch with `unhashable type: 'list'` while `triton.compile` of the same kernel succeeded, and a tuple launched; scales 0.5, 1e-6, 3.0, 2^-20 with flags True and False: 1 binary; a 1 KiB view at offset 0 of a 4 KiB buffer got `tt.divisibility = 16` and `tt.pointer_range = 32`, at offset 8 only the range, a 2 GiB + 16 KiB tensor only the divisibility, 3 binaries, `is_within_2gb` called twice per launch of a two-tensor kernel, cache hits included; an RMSNorm copy with `epsilon` hard-coded to 1e-5 passed every check at its sizes, and called with `epsilon` 1e-3 on 64 × 6144 inputs scaled by 0.01 had a maximum relative error of 2.235, the original kernel 0; a copy of an fp8 GEMM (N 3072, m = 16) storing 8 of its 16 rows: wrong values at the first capture with 24576 = 8 × 3072 non-finite mismatches, the original none (one run); the poison filled uint8, int32 and fp4 outputs with 0xFF bytes, fp32 and bf16 with NaN; a `run` keeping its first output in a module-level dict passed a source scan and gave wrong values at the first capture, the fresh captures, the repeat and the check after the timing; a math error: wrong values at the captures and after the timing; a changed reduction order: the same in one run, only a slower time in the other; state kept across calls: wrong values after the timing only; no difference between repeated runs in any; the two phases as one grid 6.33 / 4.30, 5.66 / 4.38, 5.66 / 4.33, 9.191 / 4.443 µs (replayed / the ruler's timer) against two launches 7.42 / 8.49, 7.39 / 8.57, 7.43 / 8.50, 8.471 / 8.610 µs, replay ratio 0.854, 0.766, 0.762, 1.085, outputs equal in every run (copies of a plain-Triton RMSNorm at hidden 6144 and of an fp8 GEMM, probe kernels)
- E24 [2026-11, gfx950 (MI355X), Triton of the sglang 0.5.22 image (no FlyDSL kernel), N=400 outputs per m and arm (200 rounds of one call on each of two streams), 2000 calls per m beside the matmul process; each output against fp32 `a @ bᵀ` (wrong: non-finite or relative max error > 2e-2) and bit for bit against a serial call; counters zeroed before each arm] a split-K kernel with the cached workspace at m 6–96: eager calls on a capture stream alternating with replays, on another stream, of a graph captured on it: 30–33 wrong of 400; two graphs captured on that one stream, replayed on two streams: 164–214 wrong of 400, worst relative error 2.2; eager on one stream, eager on two streams, and beside a second process running bf16 8192 × 8192 matmuls on the same GPU: 0 wrong (bf16 GEMM n32 k6144, Triton)
- E25 [2026-11, gfx950 (MI355X), Triton of the sglang 0.5.22 image (no FlyDSL kernel), N=1 run per build, lower-bound speedup over aiter's tuned bf16 GEMM per size, one MI355X] the kernel of E24 at m 6–96 with the cached workspace 1.33–1.95 per size; a copy that allocates and zeroes the partials and counters for each call made under capture and keeps them for that graph's replays 0.77–1.10, with 0 wrong outputs in E24's graph arms (bf16 GEMM n32 k6144, Triton)
- E26 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, a reference run of the ruler, stored speedup against the op's baseline] bf16 GEMM at M ≤ 64: lowest 1.048 at N 2624 × K 6144, M 2, highest 1.292 at N 6144 × K 2048, M 8, both in one run
- E27 [2026-10, gfx950 (MI355X), Triton version not recorded, N=?, an earlier form of the ruler: lower bound of the aiter / kernel time ratio over 32 rounds, 8 rotated input sets, graph replay, 1 GiB flush per call] lower bounds against aiter's `gemm_a16w16`: the N 6144 × K 2048 kernel at m = 2–64 1.09–1.31; the N 2624 × K 6144 kernel at m = 2, 4, 8 1.085–1.106, at m = 64 1.236; recorded as ranges only, raw reports not kept
- S1 triton-lang/triton@c01b677:third_party/amd/backend/compiler.py:104 `schedule_hint`: "Experimental; right now no effect", the field (:105) the only mention of `schedule_hint` in Triton 3.8.0's Python
