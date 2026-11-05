---
title: "mHC fused post + pre, 4 streams × hidden 4096, with norm weight — the post combine, the 24-logit projection and a 4×4 Gram in one pass over the residual; the RMSNorm comes from the Gram, so the finish is element-wise"
kind: case
ops: [mhc]
dtypes: [bf16, fp32]
regimes: ["tokens 1-64", "tokens 128-512", "tokens 1024-4096"]
contract: bounded
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=1
measured:
  - {metric: "speedup over aiter (mhc_fused_post_pre), CI, per Point", value: "1.100–1.510; 13 of 13 Points ≥ 1.03", gpu: MI355X, workload: "mhc_fused_post_pre_has_norm_weight_hidden_size4096, n_tokens 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", run_ref: "5571f4589 (CI run 40613440602)", date: "2026-10-28"}
sources:
  - aklr/ops/mhc/impls/flydsl/mhc_fused_post_pre_has_norm_weight_hidden_size4096/3af43505318bd593.json
---

# mHC fused post + pre at hidden 4096: one pass over the residual, a Gram for the norm

One FlyDSL kernel for `mhc_fused_post_pre@1.0` (PR #52). The schema embeds it as `kernel.sources[0].content`
(`kernel.py`); the function names below are in that file. "T" is the Point's `n_tokens`.

## Applies when

- Contract `mhc_fused_post_pre@1.0`, `has_norm_weight` true, 4 streams, hidden 4096: bf16 `x` (T, 4096) and
  `residual` (T, 4, 4096), fp32 `post_mix` (T, 4) and `comb_mix` (T, 4, 4), an fp32 projection (24, 16384) with
  `mix_scale` (3) and `mix_bias` (24), a bf16 `norm_weight` (4096), and six scalars. Outputs: `next_residual` (the
  post step, bf16), then the next pre step on it rounded to bf16: `next_post_mix` (T, 4, 1) and `next_comb_mix`
  (T, 4, 4) in fp32, and the rms-normed `layer_input` (T, 4096) bf16.
- Points: T = 1, 2, 4, …, 4096 (13), with the captured scalars eps 1e-6, `post_multiplier` 2.0 and
  `sinkhorn_iters` 20 in every workload.
- Compare: `determinism: bounded`, `max_violation_fraction` 0.0, atol = rtol = 1e-2 on all four outputs
  (`aklr/ops/mhc/contract.py`).
- Yardstick: `mhc_fused_post_pre_baseline` (`aklr/ops/mhc/baseline.py`), aiter's
  `mhc_fused_post_pre` (`aiter/ops/mhc.py`): on gfx950 its fused HIP kernel below 1024 tokens, `mhc_post` then
  `mhc_pre` from 1024 on (read from the source). This file is the only one of its schema name, in any language; the
  hidden-7168 schema of the same contract is a definition with no kernel.

## Counter-signals

- **The margin is smallest at T ≤ 64**: 1.100–1.185 there, 1.225 at 128 and 1.325–1.510 from 256 *(field, E1)*.
  All 13 Points were served in each of the five committed measurements, the lowest reading 1.042 (T = 2048, a run
  whose FlyDSL is not recorded); in the three runs on FlyDSL 0.3.4.1 the lowest was 1.095 *(field, E2)*.
- **Compilation inside launch code.** The builder (`_aklr_launch`, a module-level `lru_cache`) returns a `launch`
  closure that compiles a kernel pair (`_make_jit`) for every new tuple of the six scalars and keeps it in a dict.
  The scalars are compile-time constants, and the Sinkhorn loop is unrolled `sinkhorn_iters` times at trace time. A
  scalar tuple first seen inside a graph capture would compile there; the README has a Point's first call compile
  every variant its scalars select, in the compile function (`aklr/ops/README.md` § The kernel section and ABI).
- **Scratch on the current device.** `part` (T × NK × 34 fp32) is allocated once per T with `torch.device("cuda")`,
  not on an argument's device, and the cache key holds no device. The projection kernel rewrites every row of it
  before `k_fin` reads it, but every call of a T shares it, so two streams calling one Point at once would race
  (`../pitfalls/state_kept_across_calls.md`).
- **An LDS reservation that guards correctness.** `k_mm_p` reserves 96 KiB of LDS so one workgroup sits on a CU;
  its comment reports wrong `next_residual` values when two co-resided, and also with 8 waves per workgroup. The
  cause is not given and was not reproduced here. A port that drops the reservation or raises the wave count needs
  a bitwise check of `next_residual` first.
- **The projection runs in bf16.** The fp32 projection weight is rounded to bf16 on every call (in registers, or into
  LDS in `k_mm_p`) and multiplied with the bf16 `next_residual`; the contract has no pre-rounded weight to read, and
  nothing derived from it is kept between calls. The outputs pass the 1e-2 bound; their distance to it was not read.
- **Each call goes through a `@flyc.jit` launcher** (`run` or `run_p`), which re-binds its arguments on every call
  (`../flydsl/attention.md` § measurement §6, on FlyDSL 0.3.2). CI's ratio is GPU time in graph replays, so that
  host cost is outside it.
- **Stale docstring.** The module docstring names the projection kernel `k_proj`; the code has three, `k_mm4`,
  `k_mm` and `k_mm_p`.
- **Code no Point reaches** (from the configuration functions `_config`, `_PCFG` and `_make`): `k_fin`'s generic
  body (with `_kfin_loads_early`, `_kfin_full_epilogue` and `_sinkhorn_lane`), its multi-wave Sinkhorn branch
  (`_KFIN_W4_KW` is 1), its `R32` branch and `_kfin_s32_sk` (T 512 takes the two-wave branch, T ≥ 1024 has 16
  slices), and `k_mm`'s chunk loop (`k_mm` runs only at T = 16, with one chunk). Comments throughout carry timing
  deltas measured outside CI; they are not reproduced here.

## Mechanism

Two launches per call: a projection kernel, then `k_fin`.

**Projection kernel: post, projection and Gram in one pass.** A workgroup owns a tile of tokens and a slice of the
hidden dimension. For each of its elements it computes `next_residual[j] = x · post_mix[j] + Σ_i residual[i] ·
comb_mix[i][j]` with fp32 FMAs, rounds it to bf16 and stores it: that output is final here. From the rounded values
it accumulates, for its slice, the 24 mix logits (bf16 MFMAs against the bf16-rounded projection) and the 10
distinct entries of the 4 × 4 Gram matrix `G[a][b] = Σ_h F[a][h] · F[b][h]` of the rounded streams F
(`v_dot2_f32_bf16`; at T 512 and 2048
16x16x32 MFMAs of which only the diagonal is kept), and writes those 34 fp32 partials per (token, slice) to `part`.
The reduction over hidden is thus split NK ways and reduced later in a fixed order
(`../gemm/grid_pricing.md` §2, `../gemm/correctness.md` §8).

| T | Projection kernel | Hidden slices (NK) | Projection workgroups | `k_fin` grid, threads |
|---|---|---|---|---|
| 1, 2, 4, 8 | `k_mm4`: 1 wave, 4-token rows, `mfma_f32_4x4x4bf16_1k`; two workgroups split the 24 logits 12 / 12 | 64 | 128 (256 at T = 8) | T × 9, 64 |
| 16 | `k_mm`: 1 wave, a 16-token tile, `mfma_f32_16x16x32_bf16`; activations and all 16 weight vectors loaded before any compute | 128 | 128 | T × 9, 64 |
| 32, 64 | `k_mm_p`, 2 or 4 waves, 1 tile each | 128 | 128 | T × 9, 64 |
| 128, 256 | `k_mm_p`, 4 waves, 1 tile each | 64 | 128, 256 | T × 5, 64 |
| 512 | `k_mm_p`, 4 waves, 1 tile each | 32 | 256 | T × 2, 256 |
| 1024, 2048, 4096 | `k_mm_p`, 4 waves, 1, 2 or 4 tiles each | 16 | 256 | T × 1, 192 |

`k_mm_p` converts its workgroup's weight slab (24 logits × 4 streams × the slice, fp32) to bf16 into LDS once, and
every wave reads its B operand from there (`../gemm/datapath.md` §7 on operands several waves share);
each 16-token tile loads its mixes and first 32-hidden chunk up front and the next chunk while the current one
computes; the combine uses packed fp32 FMAs from T 256. Every `k_mm_p` grid is at most 256 workgroups, one per CU
under the LDS reservation. At T 512 the workgroups that share a weight slab land on one XCD, at T 4096 each XCD
takes adjacent token blocks (`../gemm/schedule_and_locality.md` §2); at T 4096 the partial rows leave
through LDS as 8-byte pairs.

**`k_fin`: reduce, gates, Sinkhorn, then the norm from the Gram.** Per token it sums the NK partials of each value
in a fixed tree: per-lane adds, a reduce-scatter across lane bits 5 and 4 with `permlane32_swap` /
`permlane16_swap`, then DPP adds over bits 0–3 (`_reduce_all`, `_wsplit_reduce`). Then:

- the projection's rms from the Gram diagonal, `rsqrt(Σ_a G[a][a] / 16384 + eps)`;
- pre gates `sigmoid(·) + pre_eps` and post gates `sigmoid(·) · post_multiplier`;
- the combination mix: a softmax over each row of the 4 × 4 logits, then a column normalisation and
  `sinkhorn_iters` − 1 (19) row-then-column rounds, with one element per lane of a 16-lane row, row sums by DPP
  `quad_perm`, column sums by DPP `row_ror` 4 and 8, no LDS;
- `layer_input = (Σ_a pre[a] · next_residual[a]) · rsqrt(Σ_{a,b} pre[a] · pre[b] · G[a][b] / 4096 + norm_eps) ·
  norm_weight`. The RMSNorm's sum of squares comes from the Gram already reduced, so the epilogue is element-wise:
  it reads `next_residual` once more, as bf16, and runs no second reduction over the hidden dimension.

The work is split by role. Up to T 256 one workgroup per token reduces only the 24 values that Sinkhorn and the post
gates need and writes those outputs, while 4 or 8 more reduce only the 14 the epilogue needs, each over 1024 or 512
hidden elements; at T 512 two waves share the Sinkhorn reduction through LDS beside one epilogue workgroup; from T
1024 one workgroup per token holds both, wave 0 on Sinkhorn and waves 1–2 on the epilogue. No step uses atomics.

## Result

The file is the only kernel of its schema name, so the yardstick at every Point is aiter's `mhc_fused_post_pre`.
`speedup` is a one-sided lower bound (α = 0.05) of aiter / kernel time over 32 rounds of CUDA-graph replays with a
1 GiB cache flush before each call (`aklr/core/validation/stages/_measure.py`); a Point is served from 1.03
(`AKLR_MIN_SPEEDUP`).

CI installs the repository with `--no-deps`, so these runs used the FlyDSL of the CI image, 0.3.4.1, not the
repository's pin 0.3.2 (`../pitfalls/measured_a_different_build.md`). CI measured the file under its name before
#117, `b4b4a7b28dc91a1a.json`, whose `kernel.py` differs from the committed one only in identifiers renamed from
`sikl` to `aklr`; the rename carried the numbers over.

| T | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 | Served |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| *(field, E1)* | 1.167 | 1.185 | 1.117 | 1.144 | 1.125 | 1.132 | 1.100 | 1.225 | 1.325 | 1.393 | 1.423 | 1.397 | 1.510 | 13 of 13 |

13 of the 13 Points are served, and all pass accuracy (`accuracy: passed` in each workload).

## Reproduce

On an MI355X, in a checkout that holds this file with these numbers (main at 0cef4f2e3):

```
python -m aklr validate aklr/ops/mhc/impls/flydsl/mhc_fused_post_pre_has_norm_weight_hidden_size4096/3af43505318bd593.json
```

It runs the audit and bench stages and writes `aklr_validation/report.md` with the per-Point speedup
(`aklr/ops/README.md` § Commands). Add `--base origin/main` only on a branch that edits the file: with the base set,
bench measures only schemas that are not at the base, so on an unchanged checkout it skips the bench
(`aklr/core/validation/stages/validation_scope.py`, `of_working_tree`). The local result does not count; CI's
measurement is the one dispatch reads.

## Rollback

Delete the schema file (in a PR): its Points go unserved, since no other file serves them.
`AKLR_DISABLE=mhc_fused_post_pre@1.0` drops this kernel and nothing else (the hidden-7168 file is a definition). A
Point CI measures below 1.03 stops being served without a rollback. Roll back for a wrong result CI does not catch,
such as a model whose scalars differ from the captured ones and are first seen inside a graph capture, a process that
calls the kernel on two devices or two streams, or a FlyDSL or LLVM upgrade that changes LDS residency.

## Evidence

`<schema>.json#kernel.py:<line>` is a line of the `kernel.py` the schema embeds; `<schema>` is
aklr/ops/mhc/impls/flydsl/mhc_fused_post_pre_has_norm_weight_hidden_size4096/3af43505318bd593.

- E1 [2026-10, gfx950, FlyDSL 0.3.4.1, N=1, AKLR CI] speedup over aiter per Point 1.100–1.510, 13 of 13 ≥ 1.03; T 1 1.167, 2 1.185, 4 1.117, 8 1.144, 16 1.125, 32 1.132, 64 1.100, 128 1.225, 256 1.325, 512 1.393, 1024 1.423, 2048 1.397, 4096 1.510 (mhc_fused_post_pre_has_norm_weight_hidden_size4096, T 1–4096; 5571f4589, CI run 40613440602)
- E2 [2026-10, gfx950, FlyDSL ?, N=5, AKLR CI; FlyDSL of the runs: not recorded, 0.3.1, then 0.3.4.1 three times] per-Point range and served count in commit order: 1.042–1.217, 1.134–1.439, 1.095–1.504, 1.134–1.451, 1.100–1.510, 13 of 13 each time; the kernel text is the same in all five (mhc_fused_post_pre_has_norm_weight_hidden_size4096, b4b4a7b28dc91a1a.json; baa858028, cc6e3b96c with CI run 39908769447, cca5b12df with CI run 40058570127, 102877ca0 with CI run 40154665266, 5571f4589 with CI run 40613440602)
- S1 aklr/ops/mhc/contract.py:227 compare `BOUNDED`; tolerance atol = rtol = 1e-2 at :39, on all four outputs at :151–169
- S2 aklr/ops/mhc/baseline.py:79 `mhc_fused_post_pre_baseline` calls aiter's `mhc_fused_post_pre`; aiter/ops/mhc.py:560 aiter runs `mhc_post` and `mhc_pre` from 1024 tokens on gfx950
- S3 aklr/ops/mhc/reference.py:103 the reference rounds `next_residual` to bf16 before the pre step
- S4 aklr/ops/mhc/impls/flydsl/mhc_fused_post_pre_has_norm_weight_hidden_size4096/3af43505318bd593.json#kernel.py:1783 `_config`; `_PCFG` at :1818; NK at :1839; `part` on `torch.device("cuda")` at :1843; per-scalar compile in `launch` at :1855
- S5 aklr/ops/mhc/impls/flydsl/mhc_fused_post_pre_has_norm_weight_hidden_size4096/3af43505318bd593.json#kernel.py:1247 `_make_kmm_p`: weight slab to bf16 LDS at :1372, 96 KiB reservation at :1270 with its comment at :1265, Gram MFMAs at T 512 and 2048 at :1262
- S6 aklr/ops/mhc/impls/flydsl/mhc_fused_post_pre_has_norm_weight_hidden_size4096/3af43505318bd593.json#kernel.py:788 `_kfin_epilogue`: sum of squares from the Gram at :796–803; rms from the Gram diagonal at :637
- S7 aklr/ops/mhc/impls/flydsl/mhc_fused_post_pre_has_norm_weight_hidden_size4096/3af43505318bd593.json#kernel.py:589 `_sinkhorn_el` with DPP row and column sums; fixed reduction tree in `_reduce_all` at :173
- S8 aklr/ops/mhc/impls/flydsl/mhc_fused_post_pre_has_norm_weight_hidden_size4096/3af43505318bd593.json#kernel.py:832 `ROLE`, `WSPLIT` at :835, `R32` at :836, `SKW` at :848, generic `k_fin` body from :1123; `_KFIN_W4_KW = 1` at :377
