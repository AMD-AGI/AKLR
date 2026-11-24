---
title: "bf16 GEMM, N 6016–8448 × K 7168 and N 7168 × K 35840 (Kimi-K3) — split K 2–8 ways at m ≤ 112 with an in-launch fixed-order fold, a 4-wave AGPR tile or a split tail at large m"
kind: case
ops: [gemm]
dtypes: [bf16]
regimes: ["M 8-112", "M 893-16384"]
accuracy: bitwise
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=1
measured:
  - {metric: "speedup over aiter's tuned bf16 GEMM, lower bound of the CUDA-graph time ratio, one run, per size", value: "0.969–1.441; 15 of 17 sizes ≥ 1.03 (not m 9397, 16384); 15 of 17 in a repeat measurement", gpu: MI355X, workload: "N 6016 × K 7168, m 8, 12, 16, 24, 32, 40, 48, 56, 60, 64, 80, 96, 112, 893, 2732, 9397, 16384", date: "2026-10-26"}
  - {metric: "speedup over aiter's tuned bf16 GEMM, lower bound of the CUDA-graph time ratio, one run, per size", value: "0.952–1.871; 15 of 17 sizes ≥ 1.03 (not m 9397, 16384); 15 of 17 in a repeat measurement", gpu: MI355X, workload: "N 6144 × K 7168, m 8–16384 (17 sizes)", date: "2026-10-26"}
  - {metric: "speedup over aiter's tuned bf16 GEMM, lower bound of the CUDA-graph time ratio, one run, per size", value: "1.085–1.552; 13 of 13 sizes ≥ 1.03; 10 of 13 in a repeat measurement", gpu: MI355X, workload: "N 6288 × K 7168, m 8, 12, 16, 24, 32, 40, 48, 56, 60, 893, 2732, 9397, 16384", date: "2026-10-26"}
  - {metric: "speedup over aiter's tuned bf16 GEMM, lower bound of the CUDA-graph time ratio, one run, per size", value: "0.949–2.000 (2.000 is the bound's cap); 15 of 17 sizes ≥ 1.03 (not m 2732, 9397); 13 of 17 in a repeat measurement", gpu: MI355X, workload: "N 8448 × K 7168, m 8–16384 (17 sizes)", date: "2026-10-26"}
  - {metric: "speedup over aiter's tuned bf16 GEMM, lower bound of the CUDA-graph time ratio, one run, per size", value: "0.875–1.231; 11 of 15 sizes ≥ 1.03 (not m 80, 893, 9397, 16384); 9 of 15 in a repeat measurement", gpu: MI355X, workload: "N 7168 × K 35840, m 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 893, 2732, 9397, 16384", date: "2026-10-26"}
---

# bf16 GEMM at N 6016–8448, K 7168 and at K 35840 (Kimi-K3): a few K splits folded inside the launch

Five FlyDSL kernels for `out = a @ b.T` on Kimi-K3 shapes, one kernel per (N, K), each built for a list of m;
n6016, n6144 and n8448 share
the structure of the n2112 kernel in [k3_bf16_splitk_narrow_n.md](../cases/k3_bf16_splitk_narrow_n.md), and the
large-m paths of the four K 7168 kernels come from one further tuning pass over the four. "m" is M. The four
K 7168 kernels are named by their N below (n6016, n6144, n6288, n8448); k35840 is N 7168 × K 35840.

## Applies when

- `a` (m, k) and `b` (n, k) bf16, row-major and not preshuffled, `out = a @ b.T`, no bias, bf16 `out`. Two runs
  must give the same bytes, and the result is held to the error bound of an fp32 dot product over K, with K split
  at most 16 ways. Every fold these kernels run adds its terms in an order fixed by index, or adds
  two terms.
- K = 7168 with N = 6016, 6144, 6288 or 8448 (a weight of 82–115.5 MiB), and N 7168 × K 35840 (490 MiB). At
  m ≤ 112 a grid of 48- to 256-wide column tiles already has 28–128 tiles per row block, so a split of 2–8 fills
  the 256 CUs (`../gemm/small_m.md` §3, §4).
- Sizes: m 8–16384 (17) for n6016, n6144, n8448; m 8–60 and 893–16384 (13) for n6288; 15 for k35840.
- At every size the yardstick is aiter's tuned bf16 GEMM, and a size counts as won from 1.03 over it.
- **62 of their 79 sizes read at or above 1.03** in a repeat measurement *(field, E16)*; the count per kernel is in
  § Result.

## Counter-signals

- **Below 1.03 in the one-run measurement:** n6016 at m 9397 (1.025) and 16384 (0.969), n6144 at 9397 (1.021) and
  16384 (0.952), on the 4-wave AGPR tile below *(field, E1, E2)*; n8448 at 2732 (0.985) and 9397 (0.949), on its
  own 4-wave AGPR tile without a persistent walk *(field, E4)*; k35840 at m 80 (1.013), 893 (0.875), 9397 (0.989) and 16384 (0.927)
  *(field, E5)*. In k35840's own ablation the final kernel is slower than the starting kernel at
  m 893 (−13.1%) and 16384 (−7.9%) *(field, E10)*.
- **What each wait does when it runs out:**

  | Kernel | Fold (sizes) | Tag a waiter matches | Poll bound | When the bound is hit | Every split must be resident |
  |---|---|---|---|---|---|
  | n6016 | split 1 polls split 0's tagged chunks (m ≤ 32); last split (40–96, 893); appended reducers (112) | this launch's `dispatch.id ^ queue.ptr` (m ≤ 32); otherwise the same every call | 65536, `s_sleep 1` | stores NaN | no: producers never wait |
  | n6144 | last split (m ≤ 112, 893) | the same every call | 65536 | stores NaN | no |
  | n8448 | appended reducer workgroups (m ≤ 112) | the same every call | 65536 | stores NaN | no |
  | n6288 | the two K halves of a column strip swap halves of the output (m 32–60) | this launch's tag, inside each 8-byte (value, tag) pair | 2²², `s_sleep 0` | adds whatever the slot holds and stores it | each strip's two halves (256 workgroups in all) |
  | k35840 | every split waits for all, each fragment's owner folds (S 8 at m ≤ 112 and the 9397 tail, 4 in the 2732 tail, 2 at 893) | this launch's tag | 2²², `s_sleep 0` | sums whatever the planes hold | yes, 224–256 |

  A grid that is not resident at once can turn the last two into silently wrong sums: n6288 on a stream limited to
  64 or 16 CUs or while another kernel held half of every XCD, k35840 while another kernel held one whole XCD past
  the poll bound; a ~8 ms matmul chain on another stream left both bit-exact *(field, E15)*. Both kernels call the
  bound a guard against a hang; n6288's comment says a split that gives up writes
  "an unfolded value", but the code adds the stale slot to its own term. A fully resident grid, as on an idle GPU,
  hides a broken hand-off (`../gemm/small_m.md` §4).
- **Flags read before this call writes them.** Where the tag is the same every call (n6016 above m 32, n6144,
  n8448), the workspace comes from `torch.empty` and the fold clears the flags at the end: a call depends on the
  previous call having cleared them, and a timed-out fold leaves late producers' flags set for the next call.
  A workspace from `torch.empty` holds undefined bytes, so scratch is to be written in the same call before it is
  read. Each kernel's own comment says the flags hold "this call's tag".
- **n8448 rounds its partials to 24 bits at m 24–112** (add 0x80 to the bit pattern and keep the top
  24 bits, round to nearest with ties away from zero) before the fp32 fold; the kernel's own description says the
  partials are summed and rounded once. Added error ≤ 2⁻¹⁶ of each partial, about a quarter of the accuracy bound's
  accumulation term at K 7168 (estimated, as in [k3_bf16_splitk_narrow_n.md](../cases/k3_bf16_splitk_narrow_n.md)).
- **Code no measured size reaches:** n8448's first-table rows 4096, 12288 and 1<<30, which the AGPR tile's table
  overrides, and with them its asymmetric main loop; n6016's swizzle branch for 4 chunks per row (K tiles are 64
  wide).

## Mechanism

Shared by the five kernels:

- **MFMA `mfma_f32_16x16x32_bf16` into fp32**; A and B by buffer→LDS DMA into XOR-swizzled multi-stage rings
  (`../gemm/datapath.md` §2, §5).
- **Splits at m ≤ 112 are few (at most 8; none in n6288 at m ≤ 24)**, partials in fp32 (n8448: 24-bit at m 24–112) in a workspace allocated by
  `torch.empty` per call, folded in the same launch and rounded once to bf16; no atomics on data
  (`../gemm/small_m.md` §4, `../gemm/correctness.md` §8).
- **XCD-aware orders:** grouped tile runs per XCD at large m (3–9 block rows, or 5–6 column blocks, per group);
  n8448's producers in split-major runs; n6288's two K halves of a strip on one XCD; k35840's K slices one per XCD
  at split 8 (four XCDs per slice at split 2) (`../gemm/schedule_and_locality.md` §2). In k35840:

  ```python
  # XCD-banded order: each XCD owns one K slice of a contiguous run of tiles, so its L2
  # holds only that slice of A; each split writes fp32 partials.
  XPS = _XCDS // SPLIT  # XCDs per K slice
  TILES_PER = M_BLOCKS * N_BLOCKS // XPS
  x = pid % _XCDS  # the XCD workgroup pid runs on
  split_idx = x // XPS
  tpid = (x % XPS) * TILES_PER + pid // _XCDS
  block_m_idx = tpid % M_BLOCKS
  block_n_idx = tpid // M_BLOCKS
  ```
- **Kernel pointer arguments preloaded into SGPRs** in n6144 and n8448 (`fx.Pointer` arguments): all four, 8 dwords.
  n6016 (count 16) and n6288's large-m kernel (12) take `fx.Tensor` arguments, with which preloading stops after the
  first pointer (2 SGPRs in the N 2304 × K 1536 kernel, [k3_bf16_short_k_large_m.md](k3_bf16_short_k_large_m.md)
  § Mechanism); their own ISA was not read (`../gemm/small_m.md` §6). k35840 sets no count.

| | m ≤ 112 | m 893 | m 2732 | m 9397 | m 16384 |
|---|---|---|---|---|---|
| n6016 | m ≤ 32: 16×48 / 32×48, 1–2 waves, split 2, 252 workgroups, split 0's fp32 values go out as 16-byte chunks of three values + the launch tag, no flag; m 40–96: 48–96 × 96, split 4, last split folds; m 112: 128×96, split 4, 252 producers + 252 appended reducers; B `nt` | 224×192, 8 waves, split 2, last split; 256 | 352×192 unsplit, grouped (3 rows); 256 | the AGPR tile: 224×256, 4 waves, accumulators in AGPRs, B-first MFMA, C staged per wave and stored `nt`; 1008 | the AGPR tile 256×256, persistent: 256 workgroups × 6 tiles |
| n6144 | 16–128 × 48 or 96, 1–6 waves (3 N-waves on several rows), split 2 (m ≤ 40) or 4, last split folds; 256 workgroups | 224×192, split 2; 256 | 352×192 unsplit, grouped (4 rows); 256 | as n6016 | as n6016 |
| n6288 | m 8–24: 256 workgroups, one per CU, each a strip of 24 or 25 columns in a 32-wide tile, K over the 4 waves (a quarter of each K tile), LDS fold in wave order. m 32–60: 128 strips of 49 or 50 columns in 64-wide tiles × 2 K halves = 256; each half publishes the partner's half of the output first, then polls for its own; the strip's lines are touched before the K loop | 128×192, 8 waves, 3 stages; 231 | 288×256, 8 waves, bands of 4 block rows per XCD; 250 | the AGPR tile 192×320, 4 waves, AGPR accumulators, 5-column-block groups, C `nt`; 980 | the same; 1720 |
| n8448 | 16–112 × 256, 2–4 waves, split 7 (K 1024), 231 producers + 132–231 appended reducers; fp32 planes at m ≤ 16, 24-bit at 24–112; padding rows not stored at m ≤ 48 | 128×256 unsplit, column-major order, fragments for two K steps read before the first MFMA; 231 | the AGPR tile 192×256, 4 waves, A-first MFMA, C through LDS, `nt`; 495 | 256×256; 1221 | 224×256; 2442 |
| k35840 | 32–128 × 256, 4–8 waves, split 8 (K 4480), each XCD one slice, 224 workgroups; B `nt` | 256×224, 8 waves, split 2; 256 | 256×256 unsplit over 2304 rows (252) + a split-4 tail launch over 428 rows (256) | 256×256 unsplit over 9216 rows (1008) + a split-8 tail launch over 181 rows (256) | 256×256 unsplit, grouped (4 rows); 1792 |

k35840's fold: every split stores the fragments it does not own (agent-scope `sc1`), waits `vmcnt(0)` and a
barrier, stores its flag, polls all S flags, and each fragment's owner sums planes 0 … S−1 in order, its own from
registers. Its 8-wave tiles use the pipeline of `../gemm/mainloop.md` §2.

The AGPR tile of the four K 7168 kernels gets its accumulators into AGPRs through an LLVM option at compile time;
n6288's:

```python
@functools.cache
def _compiled4(m):
    """The 4-wave kernel for m, compiled with the AGPR-accumulator form of MFMA and its
    pointer arguments preloaded into user SGPRs."""
    launch = _build4(m, *_CONFIG4)
    opts = {"amdgpu-kernarg-preload-count": 12, "amdgpu-mfma-vgpr-form": False}
    dev = torch.cuda.current_device()
    a = torch.empty((m, _K), dtype=torch.bfloat16, device=dev)
    b = torch.empty((_N, _K), dtype=torch.bfloat16, device=dev)
    c = torch.empty((m, _N), dtype=torch.bfloat16, device=dev)
    return flyc.compile[{"llvm_options": opts}](
        launch, a, b, c, fx.Stream(torch.cuda.current_stream(dev))
    )
```

n6288's exchange between the two K halves of a strip:

```python
# ks: this half (0 or 1); elem(p): pass p's element, its fp32 sum over the waves, and whether it is inside the output
did = llvm.call_intrinsic(T.i64, "llvm.amdgcn.dispatch.id", [], [], [])
qptr = llvm.call_intrinsic(
    ir.Type.parse("!llvm.ptr<4>"), "llvm.amdgcn.queue.ptr", [], [], []
)
tag64 = fx.Int64(
    std_arith.XOrIOp(did, llvm.PtrToIntOp(T.i64, qptr).result).result
)
tag_v = rocdl.readfirstlane(
    T.i32,
    as_ir_value(fx.Int32(tag64) ^ fx.Int32(tag64 >> 32) ^ fx.Int32(_MAGIC)),
)
# publish the partner's passes first so their trip to L2 overlaps the own sums
for q in range_constexpr(OUT_PASSES // 2):
    e2, tot2, ok2 = elem((1 - ks) + 2 * q)
    pair = vector.from_elements(
        vec2_i32, [std_arith.BitcastOp(T.i32, tot2).result, tag_v]
    )
    w_off = arith.select(ok2, t * PAIR_BYTES + e2 * 8, fx.Int32(_OOB))
    rocdl_d.RawPtrBufferStoreOp(
        pair, w_rsrc, _as_i32(w_off), _c32(0), aux=_aux(_LD_AUX)
    )
own = []
for q in range_constexpr(OUT_PASSES // 2):
    own.append(elem(ks + 2 * q))
...  # poll the partner's pairs for this half's passes until each carries tag_v, or the poll bound is hit
for i in range_constexpr(nv):
    e, tot, ok = own[i]
    other = std_arith.BitcastOp(
        T.f32, vector.extract(loop.results[i], [], [0])
    ).result
    # two fp32 terms: the sum is the same in either split order
    outv = std_arith.TruncFOp(
        T.bf16, std_arith.AddFOp(tot, other).result
    ).result
```

Not this: the last lines add the partner's value whether or not its tag matched. When the poll gives up, the stale
slot goes into the sum, and the output is wrong with every value finite (§ Counter-signals).

**What each part bought** (ablations; each kept step timed beside the version before it; the large-m
steps come from a second tuning pass over the four K 7168 kernels):

- n6016: the n2112 port with N-tail tiles +18.5–71.4% at m ≤ 112 over the starting kernel; grouped
  large-m tiles, in the step that also widened the m 9397 and 16384 tiles to 256, +7.7–11.7% at m ≥ 893; split 2
  at m 24–32 and 224×192 at 893 +1.6–7.4%; one-wave 16×48 tiles at
  m ≤ 16 +3.0–5.2%; preload +1.1–1.6% at m ≤ 96; the tagged chunks at m ≤ 32 +0.8–2.0% *(field, E6)*; the 4-wave
  AGPR tile +7.2% / +7.1% at m 9397 / 16384 *(field, E11)*.
- n6144: the port +20.4–71.5% at m ≤ 112; grouped order with 256×256 tiles and last-split folds at mid m
  +4.3–15.3% at m ≥ 893; split 2 over 224×192 at m 893 +13.0%; three N-waves +3.1% at m 80 *(field, E7)*; the 4-wave
  AGPR tile +6.0% / +6.7% at m 9397 / 16384 *(field, E11)*.
- n6288: the strip and tiled kernels +12.8–47.0% at m ≤ 60; split 2 for m 32–60 with the scratch touched early
  +1.0–5.5%; the tagged pair exchange instead of flags +2.6–7.6% at m 32–60; publishing the partner's share first
  +0.6–1.2% *(field, E8)*; the 192×320 4-wave tile +1.4% / +2.5% at m 9397 / 16384 *(field, E11)*.
- n8448: the port +8.6–50.5% at m ≤ 112; appended reducers on every small-m row +0.8–5.3%; 24-bit planes with
  padding masks −0.3% to +4.6% by size (+3.8–4.6% at m 96, 112); a deeper A pipeline +5.9% at m 2732 *(field, E9)*;
  `nt` C with regrouping +6.5% / +3.0% / +2.1% at m 2732 / 9397 / 16384 *(field, E11)*.
- k35840: the split-K tiled kernel +0.4–17.2% at m ≤ 112; the fold inside one launch with one K slice per XCD
  +2.8–8.1% at m ≤ 112 and +10.4% at 893; retuned large-m grouping with the m 9397 tail run as split-K +3.6% at
  m 2732 and 9397 and +2.3% at 16384; the m 2732 tail +8.3% *(field, E10)*.

## Result

The last column counts the sizes at or above 1.03 in a repeat measurement, a separate run whose ruler is not stated
in the record *(field, E16)*.
The repeat ran FlyDSL 0.3.2, the version of the one-run numbers below, not the 0.3.4.1 of its container image
(`../pitfalls/measured_a_different_build.md`).

The other columns are one run each on MI355X, FlyDSL 0.3.2, of a lower bound of the CUDA-graph time ratio
aiter / kernel, capped at 2.000; "≥ 1.03 in one run" counts the sizes at or above 1.03 in it. For the four
K 7168 kernels the run timed the version with the large-m rework beside the earlier kernel; the reworked version
is the final one, and its
small-m code compiles to the earlier kernel's ISA *(field, E12)*.

| Kernel | m ≤ 112 | m 893 | m 2732 | m 9397 | m 16384 | ≥ 1.03 in one run | ≥ 1.03 in the repeat |
|---|---|---|---|---|---|---|---|
| n6016 *(field, E1)* | 1.134–1.441 | 1.161 | 1.096 | 1.025 | 0.969 | 15 of 17 | 15 of 17 |
| n6144 *(field, E2)* | 1.060–1.871 | 1.123 | 1.091 | 1.021 | 0.952 | 15 of 17 | 15 of 17 |
| n6288 *(field, E3)* | 1.223–1.552 (m ≤ 60) | 1.109 | 1.085 | 1.102 | 1.146 | 13 of 13 | 10 of 13 |
| n8448 *(field, E4)* | 1.165–2.000 | 1.085 | 0.985 | 0.949 | 1.127 | 15 of 17 | 13 of 17 |
| k35840 *(field, E5)* | 1.013–1.195 | 0.875 | 1.231 | 0.989 | 0.927 | 11 of 15 | 9 of 15 |

## Evidence

"One-run ratio bound" is a lower bound of the CUDA-graph time ratio aiter / kernel per size, capped at 2.000, from
one run on MI355X.

- E1 [2026-10, gfx950, FlyDSL 0.3.2, N=1, one-run ratio bound] n6016 (the version with the large-m rework): m 8 1.134, 12 1.161, 16 1.182, 24 1.176, 32 1.212, 40 1.274, 48 1.340, 56 1.364, 60 1.378, 64 1.371, 80 1.427, 96 1.427, 112 1.441, 893 1.161, 2732 1.096, 9397 1.025, 16384 0.969
- E2 [2026-10, gfx950, FlyDSL 0.3.2, N=1, one-run ratio bound] n6144: m 8 1.060, 12 1.090, 16 1.104, 24 1.158, 32 1.190, 40 1.851, 48 1.871, 56 1.823, 60 1.804, 64 1.314, 80 1.600, 96 1.506, 112 1.466, 893 1.123, 2732 1.091, 9397 1.021, 16384 0.952
- E3 [2026-10, gfx950, FlyDSL 0.3.2, N=1, one-run ratio bound] n6288: m 8 1.504, 12 1.282, 16 1.255, 24 1.223, 32 1.278, 40 1.319, 48 1.269, 56 1.552, 60 1.515, 893 1.109, 2732 1.085, 9397 1.102, 16384 1.146
- E4 [2026-10, gfx950, FlyDSL 0.3.2, N=1, one-run ratio bound] n8448: m 8 1.165, 12 1.237, 16 1.253, 24 2.000, 32 1.342, 40 1.885, 48 1.881, 56 1.824, 60 1.844, 64 1.398, 80 1.703, 96 1.688, 112 1.700, 893 1.085, 2732 0.985, 9397 0.949, 16384 1.127
- E5 [2026-10, gfx950, FlyDSL 0.3.2, N=1, one-run ratio bound] k35840: m 8 1.069, 16 1.053, 24 1.062, 32 1.073, 40 1.078, 48 1.063, 56 1.036, 64 1.035, 80 1.013, 96 1.078, 112 1.195, 893 0.875, 2732 1.231, 9397 0.989, 16384 0.927
- E6 [2026-10, gfx950, FlyDSL 0.3.2, N=?, paired bench of each step against the version before it; remeasured steps 3 pairs] n6016: port +18.5–71.4% at m ≤ 112 (recorded); grouped tiles + 256-wide tiles at 9397 / 16384 +10.7, 7.7, 8.1, 11.7% at 893–16384 (recorded); split 2 + 224×192 +4.1 / +1.6 / +7.4% at 24 / 32 / 893; one-wave tiles +3.0, 4.1, 5.2% at m 8–16; preload +1.1–1.6% at m ≤ 96; tagged chunks +1.7, 1.9, 2.0, 0.8, 0.9% at m 8–32
- E7 [2026-10, gfx950, FlyDSL 0.3.2, N=?, paired bench of each step against the version before it; remeasured steps 3 pairs] n6144: port +20.4–71.5% at m ≤ 112 (recorded); grouped order + 256×256 + last split +15.3, 14.0, 4.3, 13.3% at 893–16384 (recorded); 224×192 split 2 +13.0% at 893; three N-waves +3.1% at 80
- E8 [2026-10, gfx950, FlyDSL 0.3.2, N=?, paired bench of each step against the version before it; remeasured steps 3 pairs] n6288: two kernels +12.8–47.0% at m ≤ 60 (recorded); split 2 + early scratch touch +1.0, 2.0, 5.5, 5.3, 4.6% at m 32–60; tagged exchange +2.6, 6.1, 5.0, 7.6, 6.7% at m 32–60; partner first +0.6–1.2%
- E9 [2026-10, gfx950, FlyDSL 0.3.2, N=?, paired bench of each step against the version before it; remeasured steps 3 pairs] n8448: port +8.6–50.5% at m ≤ 112 (recorded); appended reducers +0.8–5.3%; 24-bit planes + masks +1.1, 0.5, −0.3, 1.8, 1.1, 3.1, 1.8, 0.4, 0.4, −0.0, 1.7, 3.8, 4.6% at m 8–112; deeper A +5.9% at 2732
- E10 [2026-10, gfx950, FlyDSL 0.3.2, N=?, paired bench of each step against the version before it; remeasured steps 3 pairs] k35840: split-K tiled kernel +3.3, 1.1, 2.4, 2.4, 5.2, 2.4, 2.4, 0.4, 9.7, 2.2, 17.2% at m 8–112; in-launch fold + XCD slices +2.8–8.1% at m ≤ 112 and +10.4% at 893 (recorded); regrouping + 9397 tail +3.6, 3.6, 2.3% at 2732, 9397, 16384; 2732 tail +8.3%; all changes off +5.3–24.6% at m ≤ 112, −13.1% at 893, +7.2% at 2732, +6.3% at 9397, −7.9% at 16384
- E11 [2026-10, gfx950, FlyDSL 0.3.2, N=?, paired bench of each step against the version before it; remeasured steps 3 pairs] 4-wave AGPR tile n6016 +7.2 / +7.1%, n6144 +6.0 / +6.7% at m 9397 / 16384 (recorded); n6288 192×320 +1.4 / +2.5%; n8448 `nt` C + regrouping +6.5 / 3.0 / 2.1% at 2732 / 9397 / 16384
- E12 [2026-10, gfx950, FlyDSL 0.3.2, N=1, one-run ratio bound] earlier → final at m 2732 / 9397 / 16384: n6016 1.099 → 1.096, 0.923 → 1.025, 0.876 → 0.969; n6144 1.092 → 1.091, 0.927 → 1.021, 0.850 → 0.952; n6288 1.079 → 1.085, 1.076 → 1.101, 1.145 → 1.146; n8448 0.914 → 0.985, 0.894 → 0.949, 1.109 → 1.127; outputs bit-identical, same ISA up to m 893
- E15 [2026-11, gfx950 (MI355X), FlyDSL 0.3.2, N=1–10 tries per condition, eager calls compared byte for byte with a call on the idle GPU] wrong output, every value finite, no hang: n6288 at m 32 and 60 in every try on a stream limited to 64 or 16 CUs and while another kernel held half the CUs of every XCD for 3 s; k35840 at m 8, 112 and 893 in 4 of 6 tries while another kernel held all of one XCD for 3 s; both 0 of 10 per size beside a ~8 ms matmul chain on another stream
- E16 [2026-11, gfx950, FlyDSL 0.3.2, N=1, a separate run; ruler not stated in the record] sizes at or above 1.03, of 79: n6016 15 of 17 (0.926–1.458; not m 2732 0.926, 9397 0.970), n6144 15 of 17 (0.926–1.808; not m 2732 0.926, 9397 0.960), n6288 10 of 13 (0.937–1.331; not m 2732 0.937, 9397 0.943, 16384 0.951), n8448 13 of 17 (0.911–2.118; not m 893 0.911, 2732 0.917, 9397 0.972, 16384 0.951), k35840 9 of 15 (0.890–1.320; not m 8 1.027, 32 1.027, 64 1.016, 893 0.905, 9397 0.996, 16384 0.890): 62; accuracy passed at each
