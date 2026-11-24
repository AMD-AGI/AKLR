---
title: "mHC fused post + pre, 4 streams × hidden 4096, with norm weight — the post combine, the 24-logit projection and a 4×4 Gram in one pass over the residual; the RMSNorm comes from the Gram, so the finish is element-wise"
kind: case
ops: [mhc]
dtypes: [bf16, fp32]
regimes: ["tokens 1-64", "tokens 128-512", "tokens 1024-4096"]
accuracy: bounded
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=1
measured:
  - {metric: "speedup over aiter's mhc_fused_post_pre, one-sided lower bound of the time ratio over 32 rounds of CUDA-graph replays with a 1 GiB cache flush before each call, per size", value: "1.100–1.510; 13 of 13 sizes ≥ 1.03", gpu: MI355X, workload: "mHC fused post + pre with a norm weight, 4 streams × hidden 4096, n_tokens 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", date: "2026-10-28"}
---

# mHC fused post + pre at hidden 4096: one pass over the residual, a Gram for the norm

One FlyDSL kernel for the fused post + pre step of mHC, the op aiter's `mhc_fused_post_pre` computes, with a norm
weight. "T" is the number of tokens.

## Applies when

- A norm weight, 4 streams, hidden 4096: bf16 `x` (T, 4096) and
  `residual` (T, 4, 4096), fp32 `post_mix` (T, 4) and `comb_mix` (T, 4, 4), an fp32 projection (24, 16384) with
  `mix_scale` (3) and `mix_bias` (24), a bf16 `norm_weight` (4096), and six scalars. Outputs: `next_residual` (the
  post step, bf16), then the next pre step on it rounded to bf16: `next_post_mix` (T, 4, 1) and `next_comb_mix`
  (T, 4, 4) in fp32, and the rms-normed `layer_input` (T, 4096) bf16.
- Sizes: T = 1, 2, 4, …, 4096 (13), with the scalars eps 1e-6, `post_multiplier` 2.0 and
  `sinkhorn_iters` 20 at every size.
- Accuracy: bounded, atol = rtol = 1e-2 on all four outputs.
- Yardstick: aiter's
  `mhc_fused_post_pre` (`aiter/ops/mhc.py`): on gfx950 its fused HIP kernel below 1024 tokens, `mhc_post` then
  `mhc_pre` from 1024 on (read from the source, S2). At hidden 7168 the same op has a HIP kernel for T 1–16 and a
  FlyDSL kernel for T 1–128.

## Counter-signals

- **The margin is smallest at T ≤ 64**: 1.100–1.185 there, 1.225 at 128 and 1.325–1.510 from 256 *(field, E1)*.
  All 13 sizes read at or above 1.03 in each of five measurements, the lowest reading 1.042 (T = 2048, the
  first run); in the last three runs the lowest was 1.095 *(field, E2)*.
- **Compilation inside launch code.** The cached builder returns a launch
  closure that compiles a kernel pair for every new tuple of the six scalars and keeps it in a dict.
  The scalars are compile-time constants, and the Sinkhorn loop is unrolled `sinkhorn_iters` times at trace time. A
  scalar tuple first seen inside a CUDA-graph capture would compile there: compile every variant a size's scalars
  select on its first call, before any capture.
- **Scratch on the current device.** The partials buffer (T × NK × 34 fp32) is allocated once per T with
  `torch.device("cuda")`,
  not on an argument's device, and the cache key holds no device. The projection kernel rewrites every row of it
  before the finish kernel reads it, but every call of a T shares it, so two streams calling one size at once would
  race (`../pitfalls/state_kept_across_calls.md`).
- **NaN when the residual streams cancel, unless the Gram form is clamped.** Without a clamp, the finish kernel
  feeds `Σ_{a,b} pre[a] · pre[b] · G[a][b]` straight to the rsqrt. The form is ≥ 0 exactly,
  but where `Σ_a pre[a] · next_residual[a]` is near 0 its fp32 sum is rounding residue of either sign, and below
  `−4096 · norm_eps` the rsqrt, and with it every `layer_input` element of the token, is NaN. *(field, E3)* With
  streams that cancel exactly and norm_eps 1e-6, 1034 of 6400 tokens (T 64) and 7616 of 51200
  (T 512) came out NaN from a build without the clamp; with the form clamped at 0 before the rsqrt,
  none. The random inputs of the accuracy check did not reach it, so that check passed. The kernel clamps the form
  at 0 before the rsqrt (§ Mechanism), as the hidden-7168 kernel does.
- **An LDS reservation that guards correctness.** The multi-wave projection kernel reserves 96 KiB of LDS so one
  workgroup sits on a CU;
  its comment reports wrong `next_residual` values when two co-resided, and also with 8 waves per workgroup. The
  cause is not given and was not reproduced here. A port that drops the reservation or raises the wave count needs
  a bitwise check of `next_residual` first.

  ```python
  LDS_RESERVE = 96 * 1024
  allocator = SmemAllocator(None, arch="gfx950", global_sym_name="kmmp_w")
  # Reserve 96 KB of LDS (> half of gfx950's 160 KB/CU) so at most one k_mm_p
  # workgroup is resident per CU.  When two co-resided (KC=2, 512-WG grids),
  # the second one deterministically produced wrong nres values (stream 1,
  # lanes 48-63, even elements); with one WG per CU every config is
  # bit-exact.  All selected grids have <= 256 WGs, so this costs nothing.
  allocator.ptr = max(S * 24 * LDW * 2, LDS_RESERVE)
  ```
- **The projection runs in bf16.** The fp32 projection weight is rounded to bf16 on every call (in registers, or into
  LDS in the multi-wave kernel) and multiplied with the bf16 `next_residual`; the op's inputs hold no pre-rounded
  weight, and
  nothing derived from it is kept between calls. The outputs pass the 1e-2 bound; their distance to it was not read.
- **Each call goes through a `@flyc.jit` launcher**, which re-binds its arguments on every call
  (`../flydsl/attention.md` § measurement §6, on FlyDSL 0.3.2). The measured ratio is GPU time in graph replays, so
  that host cost is outside it.
- **Paths no measured size takes are untested.** Each of the 13 sizes takes one path through the two kernels
  (§ Mechanism). Neither timed nor checked for accuracy: the finish kernel's general body, which every measured T
  bypasses through a branch written for its range; a Sinkhorn reduction split over several waves up to T 256, where
  it runs in one; 32 hidden slices on any branch but T 512's two-wave one (T ≥ 1024 has 16 slices); and a second
  hidden chunk in the single-wave tile kernel, which runs only at T = 16, with one chunk.

## Mechanism

Two launches per call: a projection kernel, then a finish kernel.

**Projection kernel: post, projection and Gram in one pass.** A workgroup owns a tile of tokens and a slice of the
hidden dimension. For each of its elements it computes `next_residual[j] = x · post_mix[j] + Σ_i residual[i] ·
comb_mix[i][j]` with fp32 FMAs, rounds it to bf16 and stores it: that output is final here. From the rounded values
it accumulates, for its slice, the 24 mix logits (bf16 MFMAs against the bf16-rounded projection) and the 10
distinct entries of the 4 × 4 Gram matrix `G[a][b] = Σ_h F[a][h] · F[b][h]` of the rounded streams F
(`v_dot2_f32_bf16`; at T 512 and 2048
16x16x32 MFMAs of which only the diagonal is kept), and writes those 34 fp32 partials per (token, slice) to a
partials buffer.
The reduction over hidden is thus split NK ways and reduced later in a fixed order
(`../gemm/grid_pricing.md` §2, `../gemm/correctness.md` §8). The Gram accumulation:

```python
# av[j]: the 8 bf16 elements a lane holds of rounded stream j
S = 4  # streams
GI = [(a, b) for a in range(S) for b in range(a, S)]  # the 10 entries on and above the diagonal
NG = len(GI)

def _gram_dot2(av, gram):
    """Accumulate Gram products from the rounded BF16 residuals in FP32.

    Native dot2 instructions avoid the FP32 expansion and accumulate two
    products per instruction in the multi-wave projection.
    """
    pairs = [[fx.Vector.from_elements([av[j][2 * e], av[j][2 * e + 1]],
                                     dtype=fx.BFloat16) for e in range(4)]
             for j in range(S)]
    out = []
    for gi in range(NG):
        a, b = GI[gi]
        val = gram[gi]
        for e in range(4):
            val = fx.Float32(fx.rocdl.fdot2_f32_bf16(
                _ir.F32Type.get(), fx.as_ir_value(pairs[a][e]),
                fx.as_ir_value(pairs[b][e]), fx.as_ir_value(val), clamp=False).result)
        out.append(val)
    return out
```

| T | Projection kernel | Hidden slices (NK) | Projection workgroups | Finish kernel grid, threads |
|---|---|---|---|---|
| 1, 2, 4, 8 | 4-token-row kernel: 1 wave, 4-token rows, `mfma_f32_4x4x4bf16_1k`; two workgroups split the 24 logits 12 / 12 | 64 | 128 (256 at T = 8) | T × 9, 64 |
| 16 | single-wave tile kernel: 1 wave, a 16-token tile, `mfma_f32_16x16x32_bf16`; activations and all 16 weight vectors loaded before any compute | 128 | 128 | T × 9, 64 |
| 32, 64 | multi-wave kernel, 2 or 4 waves, 1 tile each | 128 | 128 | T × 9, 64 |
| 128, 256 | multi-wave kernel, 4 waves, 1 tile each | 64 | 128, 256 | T × 5, 64 |
| 512 | multi-wave kernel, 4 waves, 1 tile each | 32 | 256 | T × 2, 256 |
| 1024, 2048, 4096 | multi-wave kernel, 4 waves, 1, 2 or 4 tiles each | 16 | 256 | T × 1, 192 |

The multi-wave kernel converts its workgroup's weight slab (24 logits × 4 streams × the slice, fp32) to bf16 into
LDS once, and
every wave reads its B operand from there (`../gemm/datapath.md` §7 on operands several waves share);
each 16-token tile loads its mixes and first 32-hidden chunk up front and the next chunk while the current one
computes; the combine uses packed fp32 FMAs from T 256. Every grid of the multi-wave kernel is at most 256
workgroups, one per CU
under the LDS reservation. At T 512 the workgroups that share a weight slab land on one XCD, at T 4096 each XCD
takes adjacent token blocks (`../gemm/schedule_and_locality.md` §2); at T 4096 the partial rows leave
through LDS as 8-byte pairs.

**Finish kernel: reduce, gates, Sinkhorn, then the norm from the Gram.** Per token it sums the NK partials of each
value
in a fixed tree: per-lane adds, a reduce-scatter across lane bits 5 and 4 with `permlane32_swap` /
`permlane16_swap`, then DPP adds over bits 0–3. Then:

- the projection's rms from the Gram diagonal, `rsqrt(Σ_a G[a][a] / 16384 + eps)`;
- pre gates `sigmoid(·) + pre_eps` and post gates `sigmoid(·) · post_multiplier`;
- the combination mix: a softmax over each row of the 4 × 4 logits, then a column normalisation and
  `sinkhorn_iters` − 1 (19) row-then-column rounds, with one element per lane of a 16-lane row, row sums by DPP
  `quad_perm`, column sums by DPP `row_ror` 4 and 8, no LDS;

  ```python
  _DPP_XOR1 = 0xB1      # quad_perm [1,0,3,2]
  _DPP_XOR2 = 0x4E      # quad_perm [2,3,0,1]
  _DPP_ROR4 = 0x124     # row_ror:4
  _DPP_ROR8 = 0x128     # row_ror:8 (== xor 8 inside a 16-lane row)

  def _sinkhorn_el(el, rms, s2, mbv, eps_v, iters):
      """_sinkhorn_lane with the lane's element already selected."""
      # lane L of a 16-lane row holds element (L // 4, L % 4) of the 4 × 4 matrix
      z = el * rms * s2 + mbv
      mx = fx.max(z, _dpp(z, _DPP_XOR1))
      mx = fx.max(mx, _dpp(mx, _DPP_XOR2))
      e = _exp(z - mx)
      tot = e + _dpp(e, _DPP_XOR1)
      tot = tot + _dpp(tot, _DPP_XOR2)
      eps = _f32(eps_v)
      c = e * _rcp(tot) + eps  # softmax over the row, the four lanes of a quad

      def col(c):
          cs = c + _dpp(c, _DPP_ROR4)
          cs = cs + _dpp(cs, _DPP_ROR8)
          return c * _rcp(cs + eps)

      def row(c):
          rs = c + _dpp(c, _DPP_XOR1)
          rs = rs + _dpp(rs, _DPP_XOR2)
          return c * _rcp(rs + eps)

      c = col(c)
      for _ in range(iters - 1):
          c = col(row(c))
      return c
  ```
- `layer_input = (Σ_a pre[a] · next_residual[a]) · rsqrt(Σ_{a,b} pre[a] · pre[b] · G[a][b] / 4096 + norm_eps) ·
  norm_weight`. The RMSNorm's sum of squares comes from the Gram already reduced, so the epilogue is element-wise:
  it reads `next_residual` once more, as bf16, and runs no second reduction over the hidden dimension.

  ```python
  ssq = _f32(0.0)
  for gi in fx.range_constexpr(NG):  # the 10 Gram entries; an off-diagonal one counts twice
      a, b = GI[gi]
      v = pre[a] * pre[b] * gmat[a][b]
      if a != b:
          v = v + v
      ssq = ssq + v
  # the Gram form can round below 0 when the streams cancel; rsq of a negative is NaN (the 7168 kernel clamps too)
  ssq = fx.Float32(_arith.MaximumFOp(fx.as_ir_value(ssq), fx.as_ir_value(_f32(0.0))).result)
  nsc = _rsq(ssq * _f32(1.0 / float(D)) + _f32(norm_eps))  # D = 4096
  ```

  Not this: the form fed to the rsqrt without the clamp. Where the streams cancel, the fp32 sum can round far enough
  below 0 that every `layer_input` element of that token is NaN (§ Counter-signals).

The work is split by role. Up to T 256 one workgroup per token reduces only the 24 values that Sinkhorn and the post
gates need and writes those outputs, while 4 or 8 more reduce only the 14 the epilogue needs, each over 1024 or 512
hidden elements; at T 512 two waves share the Sinkhorn reduction through LDS beside one epilogue workgroup; from T
1024 one workgroup per token holds both, wave 0 on Sinkhorn and waves 1–2 on the epilogue. No step uses atomics.

## Result

The yardstick at every size is aiter's `mhc_fused_post_pre`.
The speedup is a one-sided lower bound (α = 0.05) of aiter / kernel time over 32 rounds of CUDA-graph replays with a
1 GiB cache flush before each call; a size counts as won from 1.03.

The numbers were measured on FlyDSL 0.3.2, a pin installed over a container image that ships
0.3.4.1 (`../pitfalls/measured_a_different_build.md`), on the build before the clamp, the same kernel without the
clamp's two lines.

| T | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 | ≥ 1.03 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| *(field, E1)* | 1.167 | 1.185 | 1.117 | 1.144 | 1.125 | 1.132 | 1.100 | 1.225 | 1.325 | 1.393 | 1.423 | 1.397 | 1.510 | 13 of 13 |

13 of the 13 sizes are at or above 1.03, and all pass the accuracy check.

## Evidence

E1 and E2 measured the build before the clamp, the same kernel without the clamp's two lines.

- E1 [2026-10, gfx950, FlyDSL 0.3.2, N=1, one-sided lower bound (α = 0.05) of aiter / kernel time over 32 rounds of CUDA-graph replays with a 1 GiB cache flush before each call] speedup over aiter per size 1.100–1.510, 13 of 13 ≥ 1.03; T 1 1.167, 2 1.185, 4 1.117, 8 1.144, 16 1.125, 32 1.132, 64 1.100, 128 1.225, 256 1.325, 512 1.393, 1024 1.423, 2048 1.397, 4096 1.510
- E2 [2026-10, gfx950, FlyDSL 0.3.2, N=5, the ruler of E1, five runs] per-size range and count at or above 1.03 in run order: 1.042–1.217, 1.134–1.439, 1.095–1.504, 1.134–1.451, 1.100–1.510, 13 of 13 each time; the kernel text is the same in all five
- E3 [2026-11, gfx950 (MI355X), FlyDSL 0.3.2, N=100 calls per T and input family, the build without the clamp and a copy with the clamp inserted before its rsqrt on the same inputs, outputs checked for non-finite values and against a reference implementation] inputs whose four residual streams cancel exactly (`x` 0, `comb_mix` the identity, equal pre gates: the true form and `layer_input` are 0), norm_eps 1e-6: tokens with a non-finite `layer_input` without the clamp 1034 of 6400 at T 64 and 7616 of 51200 at T 512, with `ssq = max(ssq, 0)` 0; the random inputs of the accuracy check at the same T: none in either build
- S2 aiter/ops/mhc.py:560 aiter runs `mhc_post` and `mhc_pre` from 1024 tokens on gfx950
