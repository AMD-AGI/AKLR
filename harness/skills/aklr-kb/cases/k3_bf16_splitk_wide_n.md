---
title: "bf16 GEMM, N 6016–8448 × K 7168 and N 7168 × K 35840 (Kimi-K3) — split K 2–8 ways at m ≤ 112 with an in-launch fixed-order fold, a 4-wave AGPR tile or a split tail at large m"
kind: case
ops: [gemm]
dtypes: [bf16]
regimes: ["M 8-112", "M 893-16384"]
contract: bitwise
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=1
measured:
  - {metric: "speedup over aiter, local validate lower bound per Point; no CI measurement on main", value: "0.969–1.441; 15 of 17 Points ≥ 1.03 (not m 9397, 16384); 0 served", gpu: MI355X, workload: "gemm_n6016_k7168, m 8, 12, 16, 24, 32, 40, 48, 56, 60, 64, 80, 96, 112, 893, 2732, 9397, 16384", run_ref: "local:/workspace/code/gpt_oss_docker/work/sikl/k3_forge/final_r2ab6_local/report.md:33", date: "2026-10-26"}
  - {metric: "speedup over aiter, local validate lower bound per Point; no CI measurement on main", value: "0.952–1.871; 15 of 17 Points ≥ 1.03 (not m 9397, 16384); 0 served", gpu: MI355X, workload: "gemm_n6144_k7168, m 8–16384 (17 Points)", run_ref: "local:/workspace/code/gpt_oss_docker/work/sikl/k3_forge/final_r2ab6_local/report.md:67", date: "2026-10-26"}
  - {metric: "speedup over aiter, local validate lower bound per Point; no CI measurement on main", value: "1.085–1.552; 13 of 13 Points ≥ 1.03; 0 served", gpu: MI355X, workload: "gemm_n6288_k7168, m 8, 12, 16, 24, 32, 40, 48, 56, 60, 893, 2732, 9397, 16384", run_ref: "local:/workspace/code/gpt_oss_docker/work/sikl/k3_forge/final_r2ab6_local/report.md:97", date: "2026-10-26"}
  - {metric: "speedup over aiter, local validate lower bound per Point; no CI measurement on main", value: "0.949–2.000 (2.000 is the bound's cap); 15 of 17 Points ≥ 1.03 (not m 2732, 9397); 0 served", gpu: MI355X, workload: "gemm_n8448_k7168, m 8–16384 (17 Points)", run_ref: "local:/workspace/code/gpt_oss_docker/work/sikl/k3_forge/final_r2ab6_local/report.md:127", date: "2026-10-26"}
  - {metric: "speedup over aiter, local validate lower bound per Point; no CI measurement on main", value: "0.875–1.231; 11 of 15 Points ≥ 1.03 (not m 80, 893, 9397, 16384); 0 served", gpu: MI355X, workload: "gemm_n7168_k35840, m 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 893, 2732, 9397, 16384", run_ref: "local:/workspace/code/gpt_oss_docker/work/sikl/k3_forge/final3_local/validate/report.md:46", date: "2026-10-26"}
sources:
  - aklr/ops/gemm/impls/flydsl/gemm_n6016_k7168/fe6fa078b3de6610.json
  - aklr/ops/gemm/impls/flydsl/gemm_n6144_k7168/41ac253b2d1a6704.json
  - aklr/ops/gemm/impls/flydsl/gemm_n6288_k7168/18df46ab10cda50d.json
  - aklr/ops/gemm/impls/flydsl/gemm_n8448_k7168/5beee8319bd5c527.json
  - aklr/ops/gemm/impls/flydsl/gemm_n7168_k35840/5ac66d6dae624f93.json
---

# bf16 GEMM at N 6016–8448, K 7168 and at K 35840 (Kimi-K3): a few K splits folded inside the launch

Five FlyDSL kernels for the `gemm@1.0` contract on Kimi-K3 shapes (definitions #69, kernels #115, merge
a8fe490b1), one schema file per shape, written by forge-loop campaigns and reviewed; n6016, n6144 and n8448 share
the structure of the n2112 kernel in [k3_bf16_splitk_narrow_n.md](../cases/k3_bf16_splitk_narrow_n.md), and the
large-m paths of the four K 7168 kernels come from one further campaign over the four. Each schema embeds its kernel as `kernel.sources[0].content`
(`kernel.py`); function names below are in that file. "m" is the Point's M.

## Applies when

- Contract `gemm@1.0`, layout variant `row_major` (`b.is_shuffled` false): `a` (m, k) and `b` (n, k) bf16,
  `trans_b`, no bias, bf16 `out`. `compare` is bitwise (`determinism: bitwise`, `max_violation_fraction` 0.0,
  the `CompareSpec` defaults), with the fp32 dot-product error bound of `_dot_k()` (`max_split_k` 16; `GEMM` in
  `aklr/ops/gemm/contract.py`). Every fold these kernels run adds its terms in an order fixed by index, or adds
  two terms.
- K = 7168 with N = 6016, 6144, 6288 or 8448 (a weight of 82–115.5 MiB), and N 7168 × K 35840 (490 MiB). At
  m ≤ 112 a grid of 48- to 256-wide column tiles already has 28–128 tiles per row block, so a split of 2–8 fills
  the 256 CUs (`../gemm/small_m.md` §3, §4).
- Points: m 8–16384 (17) for n6016, n6144, n8448; m 8–60 and 893–16384 (13) for n6288; 15 for k35840.
- Each schema name has only this file (no file of another language or an older kernel under the same name on
  main), so at every Point the yardstick is aiter.
- **No Point of these files is served today.** No CI run has written `speedup` or `accuracy` into their
  `workloads` (main at 0cef4f2e3); dispatch serves a workload only from `AKLR_MIN_SPEEDUP` 1.03 and falls back with
  `not_measured` (`aklr/ops/README.md` § Environment variables; `aklr/core/contract/spec/_workloads.py:38`).

## Counter-signals

- **Below 1.03 in the local validate:** n6016 at m 9397 (1.025) and 16384 (0.969), n6144 at 9397 (1.021) and
  16384 (0.952), on the 4-wave AGPR tile below *(field, E1, E2)*; n8448 at 2732 (0.985) and 9397 (0.949), on its
  own 4-wave AGPR tile without a persistent walk *(field, E4)*; k35840 at m 80 (1.013), 893 (0.875), 9397 (0.989) and 16384 (0.927)
  *(field, E5)*. In k35840's own ablation the delivered kernel is slower than the campaign's starting kernel at
  m 893 (−13.1%) and 16384 (−7.9%) *(field, E10)*.
- **What each wait does when it runs out:**

  | File | Fold (Points) | Tag a waiter matches | Poll bound | When the bound is hit | Every split must be resident |
  |---|---|---|---|---|---|
  | n6016 | split 1 polls split 0's tagged chunks (m ≤ 32); last split (40–96, 893); appended reducers (112) | this launch's `dispatch.id ^ queue.ptr` (m ≤ 32); otherwise the same every call | 65536, `s_sleep 1` | stores NaN | no: producers never wait |
  | n6144 | last split (m ≤ 112, 893) | the same every call | 65536 | stores NaN | no |
  | n8448 | appended reducer workgroups (m ≤ 112) | the same every call | 65536 | stores NaN | no |
  | n6288 | the two K halves of a column strip swap halves of the output (m 32–60) | this launch's tag, inside each 8-byte (value, tag) pair | 2²², `s_sleep 0` | adds whatever the slot holds and stores it | each strip's two halves (256 workgroups in all) |
  | k35840 | every split waits for all, each fragment's owner folds (S 8 at m ≤ 112 and the 9397 tail, 4 in the 2732 tail, 2 at 893) | this launch's tag | 2²², `s_sleep 0` | sums whatever the planes hold | yes, 224–256 |

  A grid that is not resident at once (another kernel holding CUs, or fewer CUs) turns the last two into silently
  wrong sums. Both kernels call the bound a guard against a hang; n6288's comment says a split that gives up writes
  "an unfolded value", but the code adds the stale slot to its own term. A fully resident grid, as on an idle GPU,
  hides a broken hand-off (`../gemm/small_m.md` §4).
- **Flags read before this call writes them.** Where the tag is the same every call (n6016 above m 32, n6144,
  n8448), the workspace comes from `torch.empty` and the fold clears the flags at the end: a call depends on the
  previous call having cleared them, and a timed-out fold leaves late producers' flags set for the next call
  (`aklr/ops/README.md` § The kernel section and ABI: scratch rewritten before it is read). Each
  file's `_count_ready` docstring says the flags hold "this call's tag".
- **n8448 rounds its partials to 24 bits at m 24–112** (`_pack24`: add 0x80 to the bit pattern and keep the top
  24 bits, round to nearest with ties away from zero) before the fp32 fold; the module docstring says the partials
  are summed and rounded once. Added error ≤ 2⁻¹⁶ of each partial, about a quarter of the bound's accumulation
  term at K 7168 (estimated, as in [k3_bf16_splitk_narrow_n.md](../cases/k3_bf16_splitk_narrow_n.md)).
- **`run` departs from the ABI.** n6288's `run` allocates the output when `out` is `None` ("the bench harness");
  `aklr/ops/README.md` § The kernel section and ABI has `run` never allocate an output. AKLR passes `out`.
- **Code no Point reaches:** n8448's `_CONFIGS` rows 4096, 12288 and 1<<30, which `_CONFIGS4` overrides, and with
  them its asymmetric main loop; n6016's `_swz` branch for 4 chunks per row (K tiles are 64 wide).

## Mechanism

Shared by the five kernels:

- **MFMA `mfma_f32_16x16x32_bf16` into fp32**; A and B by buffer→LDS DMA into XOR-swizzled multi-stage rings
  (`../gemm/datapath.md` §2, §5).
- **Splits at m ≤ 112 are few (at most 8; none in n6288 at m ≤ 24)**, partials in fp32 (n8448: 24-bit at m 24–112) in a workspace allocated by
  `torch.empty` per call, folded in the same launch and rounded once to bf16; no atomics on data
  (`../gemm/small_m.md` §4, `../gemm/correctness.md` §8).
- **XCD-aware orders:** grouped tile runs per XCD at large m (3–9 block rows, or 5–6 column blocks, per group);
  n8448's producers in split-major runs; n6288's two K halves of a strip on one XCD; k35840's K slices one per XCD
  at split 8 (four XCDs per slice at split 2) (`../gemm/schedule_and_locality.md` §2).
- **Kernel pointer arguments preloaded into SGPRs** in n6016, n6144 and n8448 (8–16 dwords) and in n6288's
  large-m kernel (`../gemm/small_m.md` §6); k35840 does not.

| | m ≤ 112 | m 893 | m 2732 | m 9397 | m 16384 |
|---|---|---|---|---|---|
| n6016 | m ≤ 32: 16×48 / 32×48, 1–2 waves, split 2, 252 workgroups, split 0's fp32 values go out as 16-byte chunks of three values + the launch tag, no flag; m 40–96: 48–96 × 96, split 4, last split folds; m 112: 128×96, split 4, 252 producers + 252 appended reducers; B `nt` | 224×192, 8 waves, split 2, last split; 256 | 352×192 unsplit, grouped (3 rows); 256 | `_build4`: 224×256, 4 waves, accumulators in AGPRs, B-first MFMA, C staged per wave and stored `nt`; 1008 | `_build4` 256×256, persistent: 256 workgroups × 6 tiles |
| n6144 | 16–128 × 48 or 96, 1–6 waves (3 N-waves on several rows), split 2 (m ≤ 40) or 4, last split folds; 256 workgroups | 224×192, split 2; 256 | 352×192 unsplit, grouped (4 rows); 256 | as n6016 | as n6016 |
| n6288 | m 8–24: 256 workgroups, one per CU, each a strip of 24 or 25 columns in a 32-wide tile, K over the 4 waves (a quarter of each K tile), LDS fold in wave order. m 32–60: 128 strips of 49 or 50 columns in 64-wide tiles × 2 K halves = 256; each half publishes the partner's half of the output first, then polls for its own; the strip's lines are touched before the K loop | 128×192, 8 waves, 3 stages; 231 | 288×256, 8 waves, bands of 4 block rows per XCD; 250 | `_build4` 192×320, 4 waves, AGPR accumulators, 5-column-block groups, C `nt`; 980 | the same; 1720 |
| n8448 | 16–112 × 256, 2–4 waves, split 7 (K 1024), 231 producers + 132–231 appended reducers; fp32 planes at m ≤ 16, 24-bit at 24–112; padding rows not stored at m ≤ 48 | 128×256 unsplit, column-major order, fragments for two K steps read before the first MFMA; 231 | `_build4` 192×256, 4 waves, A-first MFMA, C through LDS, `nt`; 495 | 256×256; 1221 | 224×256; 2442 |
| k35840 | 32–128 × 256, 4–8 waves, split 8 (K 4480), each XCD one slice, 224 workgroups; B `nt` | 256×224, 8 waves, split 2; 256 | 256×256 unsplit over 2304 rows (252) + a split-4 tail launch over 428 rows (256) | 256×256 unsplit over 9216 rows (1008) + a split-8 tail launch over 181 rows (256) | 256×256 unsplit, grouped (4 rows); 1792 |

k35840's fold: every split stores the fragments it does not own (agent-scope `sc1`), waits `vmcnt(0)` and a
barrier, stores its flag, polls all S flags, and each fragment's owner sums planes 0 … S−1 in order, its own from
registers. Its 8-wave tiles use the pipeline of `../gemm/mainloop.md` §2.

**What each part bought** (campaign ablations; each kept round timed beside the version before it; the large-m
rounds come from a second campaign over the four K 7168 kernels):

- n6016: the n2112 port with N-tail tiles +18.5–71.4% at m ≤ 112 over the campaign's starting kernel; grouped
  large-m tiles +7.7–11.7% at m ≥ 893; split 2 at m 24–32 and 224×192 at 893 +1.6–7.4%; one-wave 16×48 tiles at
  m ≤ 16 +3.0–5.2%; preload +1.1–1.6% at m ≤ 96; the tagged chunks at m ≤ 32 +0.8–2.0% *(field, E6)*; the 4-wave
  AGPR tile +7.2% / +7.1% at m 9397 / 16384 *(field, E11)*.
- n6144: the port +20.4–71.5% at m ≤ 112; grouped order with 256×256 tiles and last-split folds at mid m
  +4.3–15.3% at m ≥ 893; split 2 over 224×192 at m 893 +13.0%; three N-waves +3.1% at m 80 *(field, E7)*; the 4-wave
  AGPR tile +6.0% / +6.7% at m 9397 / 16384 *(field, E11)*.
- n6288: the strip and tiled kernels +12.8–47.0% at m ≤ 60; split 2 for m 32–60 with the scratch touched early
  +1.0–5.5%; the tagged pair exchange instead of flags +2.6–7.6% at m 32–60; publishing the partner's share first
  +0.6–1.2% *(field, E8)*; the 192×320 4-wave tile +1.4% / +2.5% at m 9397 / 16384 *(field, E11)*.
- n8448: the port +8.6–50.5% at m ≤ 112; appended reducers on every small-m row +0.8–5.3%; 24-bit planes with
  padding masks −0.3% to +4.6% by Point (+3.8–4.6% at m 96, 112); a deeper A pipeline +5.9% at m 2732 *(field, E9)*;
  `nt` C with regrouping +6.5% / +3.0% / +2.1% at m 2732 / 9397 / 16384 *(field, E11)*.
- k35840: the split-K tiled kernel +0.4–17.2% at m ≤ 112; the fold inside one launch with one K slice per XCD
  +2.8–8.1% at m ≤ 112 and +10.4% at 893; the split tails +3.6% (9397) and +8.3% (2732) *(field, E10)*.

## Result

`speedup` is absent from every workload of these files: CI has not measured them, so dispatch serves none of
their 79 Points. When CI does measure them it runs the FlyDSL of its image (0.3.4.1 in
the image of the latest perf runs), not the 0.3.2 of the local numbers below
(`../pitfalls/measured_a_different_build.md`).

The only per-Point measurements are the campaigns' own `python -m sikl validate` runs on MI355X (the repository
before its rename to AKLR), FlyDSL 0.3.2: a lower bound of the CUDA-graph time ratio aiter / kernel, one run each,
capped at 2.000; "≥ 1.03" counts them against the dispatch threshold (that validate itself demoted below 1.02). For the four K 7168 files the run timed the
large-m campaign's module beside the earlier kernel; the module is what the follow-up commits delivered, and its
small-m code compiles to the earlier kernel's ISA *(field, E12)*.

| File | m ≤ 112 | m 893 | m 2732 | m 9397 | m 16384 | ≥ 1.03 locally | Served |
|---|---|---|---|---|---|---|---|
| n6016 *(field, E1)* | 1.134–1.441 | 1.161 | 1.096 | 1.025 | 0.969 | 15 of 17 | 0 |
| n6144 *(field, E2)* | 1.060–1.871 | 1.123 | 1.091 | 1.021 | 0.952 | 15 of 17 | 0 |
| n6288 *(field, E3)* | 1.223–1.552 (m ≤ 60) | 1.109 | 1.085 | 1.102 | 1.146 | 13 of 13 | 0 |
| n8448 *(field, E4)* | 1.165–2.000 | 1.085 | 0.985 | 0.949 | 1.127 | 15 of 17 | 0 |
| k35840 *(field, E5)* | 1.013–1.195 | 0.875 | 1.231 | 0.989 | 0.927 | 11 of 15 | 0 |

## Reproduce

On an MI355X with the repository's FlyDSL 0.3.2, from the repository root, one schema at a time:

```
python -m aklr validate aklr/ops/gemm/impls/flydsl/gemm_n6016_k7168/fe6fa078b3de6610.json
python -m aklr validate aklr/ops/gemm/impls/flydsl/gemm_n6144_k7168/41ac253b2d1a6704.json
python -m aklr validate aklr/ops/gemm/impls/flydsl/gemm_n6288_k7168/18df46ab10cda50d.json
python -m aklr validate aklr/ops/gemm/impls/flydsl/gemm_n8448_k7168/5beee8319bd5c527.json
python -m aklr validate aklr/ops/gemm/impls/flydsl/gemm_n7168_k35840/5ac66d6dae624f93.json
```

The report goes to `aklr_validation/report.md`; a local run checks the file, the served numbers are CI's
(`aklr/ops/README.md` § Workflow). Add `--base origin/main` only on a branch that edits the file: with the base set,
bench measures only schemas that are not at the base, so on an unchanged checkout it skips the bench
(`aklr/core/validation/stages/validation_scope.py`, `of_working_tree`). Validate's bitwise check runs on an
otherwise idle GPU, where every split is resident; test n6288 and k35840 also with a grid that is not. To read a
kernel:
`git show <rev>:<schema> | python3 -c 'import json,sys; print(json.load(sys.stdin)["kernel"]["sources"][0]["content"])'`.

## Rollback

Delete the schema file: its Points go unserved, since no other file serves them. `AKLR_DISABLE=gemm@1.0`
stops serving the whole contract, every other bf16 GEMM schema included. Until CI writes `speedup`, nothing is
served and nothing needs rolling back; afterwards a Point measured below 1.03 is not served, and a `failed`
accuracy never is. Roll back for a wrong result CI does not catch: a silently wrong sum from n6288 or k35840 when
their splits are not resident, NaN or a stale fold after a timed-out launch of the others, or a change of FlyDSL or
LLVM.

## Evidence

`<schema>.json#kernel.py:<line>` is a line of the `kernel.py` a schema embeds in `kernel.sources`; "local validate"
is `python -m sikl validate` on MI355X in the campaign containers, the ratio lower bound over aiter per Point.

- E1 [2026-10, gfx950, FlyDSL 0.3.2, N=1, local validate] n6016 (the large-m group's module, 9f69f697): m 8 1.134, 12 1.161, 16 1.182, 24 1.176, 32 1.212, 40 1.274, 48 1.340, 56 1.364, 60 1.378, 64 1.371, 80 1.427, 96 1.427, 112 1.441, 893 1.161, 2732 1.096, 9397 1.025, 16384 0.969 (gemm_n6016_k7168; local:/workspace/code/gpt_oss_docker/work/sikl/k3_forge/final_r2ab6_local/report.md:33)
- E2 [2026-10, gfx950, FlyDSL 0.3.2, N=1, local validate] n6144 (bb3c3d06): m 8 1.060, 12 1.090, 16 1.104, 24 1.158, 32 1.190, 40 1.851, 48 1.871, 56 1.823, 60 1.804, 64 1.314, 80 1.600, 96 1.506, 112 1.466, 893 1.123, 2732 1.091, 9397 1.021, 16384 0.952 (gemm_n6144_k7168; local:/workspace/code/gpt_oss_docker/work/sikl/k3_forge/final_r2ab6_local/report.md:67)
- E3 [2026-10, gfx950, FlyDSL 0.3.2, N=1, local validate] n6288 (5cd58204): m 8 1.504, 12 1.282, 16 1.255, 24 1.223, 32 1.278, 40 1.319, 48 1.269, 56 1.552, 60 1.515, 893 1.109, 2732 1.085, 9397 1.102, 16384 1.146 (gemm_n6288_k7168; local:/workspace/code/gpt_oss_docker/work/sikl/k3_forge/final_r2ab6_local/report.md:97)
- E4 [2026-10, gfx950, FlyDSL 0.3.2, N=1, local validate] n8448 (94dc70c7): m 8 1.165, 12 1.237, 16 1.253, 24 2.000, 32 1.342, 40 1.885, 48 1.881, 56 1.824, 60 1.844, 64 1.398, 80 1.703, 96 1.688, 112 1.700, 893 1.085, 2732 0.985, 9397 0.949, 16384 1.127 (gemm_n8448_k7168; local:/workspace/code/gpt_oss_docker/work/sikl/k3_forge/final_r2ab6_local/report.md:127)
- E5 [2026-10, gfx950, FlyDSL 0.3.2, N=1, local validate] k35840: m 8 1.069, 16 1.053, 24 1.062, 32 1.073, 40 1.078, 48 1.063, 56 1.036, 64 1.035, 80 1.013, 96 1.078, 112 1.195, 893 0.875, 2732 1.231, 9397 0.989, 16384 0.927 (gemm_n7168_k35840; local:/workspace/code/gpt_oss_docker/work/sikl/k3_forge/final3_local/validate/report.md:46)
- E6 [2026-10, gfx950, FlyDSL 0.3.2, N=?, forge-loop paired bench; remeasured rounds 3 pairs] n6016: port +18.5–71.4% at m ≤ 112 (recorded); grouped tiles +10.7, 7.7, 8.1, 11.7% at 893–16384 (recorded); split 2 + 224×192 +4.1 / +1.6 / +7.4% at 24 / 32 / 893; one-wave tiles +3.0, 4.1, 5.2% at m 8–16; preload +1.1–1.6% at m ≤ 96; tagged chunks +1.7, 1.9, 2.0, 0.8, 0.9% at m 8–32 (gemm_n6016_k7168; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n6016_k7168/forge_experiments/ablation.json)
- E7 [2026-10, gfx950, FlyDSL 0.3.2, N=?, forge-loop paired bench; remeasured rounds 3 pairs] n6144: port +20.4–71.5% at m ≤ 112 (recorded); grouped order + 256×256 + last split +15.3, 14.0, 4.3, 13.3% at 893–16384 (recorded); 224×192 split 2 +13.0% at 893; three N-waves +3.1% at 80 (gemm_n6144_k7168; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n6144_k7168/forge_experiments/ablation.json)
- E8 [2026-10, gfx950, FlyDSL 0.3.2, N=?, forge-loop paired bench; remeasured rounds 3 pairs] n6288: two kernels +12.8–47.0% at m ≤ 60 (recorded); split 2 + early scratch touch +1.0, 2.0, 5.5, 5.3, 4.6% at m 32–60; tagged exchange +2.6, 6.1, 5.0, 7.6, 6.7% at m 32–60; partner first +0.6–1.2% (gemm_n6288_k7168; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n6288_k7168/forge_experiments/ablation.json)
- E9 [2026-10, gfx950, FlyDSL 0.3.2, N=?, forge-loop paired bench; remeasured rounds 3 pairs] n8448: port +8.6–50.5% at m ≤ 112 (recorded); appended reducers +0.8–5.3%; 24-bit planes + masks +1.1, 0.5, −0.3, 1.8, 1.1, 3.1, 1.8, 0.4, 0.4, −0.0, 1.7, 3.8, 4.6% at m 8–112; deeper A +5.9% at 2732 (gemm_n8448_k7168; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n8448_k7168/forge_experiments/ablation.json)
- E10 [2026-10, gfx950, FlyDSL 0.3.2, N=?, forge-loop paired bench; remeasured rounds 3 pairs] k35840: split-K tiled kernel +3.3, 1.1, 2.4, 2.4, 5.2, 2.4, 2.4, 0.4, 9.7, 2.2, 17.2% at m 8–112; in-launch fold + XCD slices +2.8–8.1% at m ≤ 112 and +10.4% at 893 (recorded); 9397 tail +3.6%; 2732 tail +8.3%; all changes off +5.3–24.6% at m ≤ 112, −13.1% at 893, +7.2% at 2732, +6.3% at 9397, −7.9% at 16384 (gemm_n7168_k35840; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n7168_k35840/forge_experiments/ablation.json; local:/workspace/code/gpt_oss_docker/work/sikl/k3_forge/final3_docs/gemm_n7168_k35840/ablation.md)
- E11 [2026-10, gfx950, FlyDSL 0.3.2, N=?, forge-loop paired bench, group campaign; remeasured rounds 3 pairs] 4-wave AGPR tile n6016 +7.2 / +7.1%, n6144 +6.0 / +6.7% at m 9397 / 16384 (recorded); n6288 192×320 +1.4 / +2.5%; n8448 `nt` C + regrouping +6.5 / 3.0 / 2.1% at 2732 / 9397 / 16384 (r2_n6k_k7168; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/r2_n6k_k7168/forge_experiments/ablation.json)
- E12 [2026-10, gfx950, FlyDSL 0.3.2, N=1, local validate] earlier → delivered at m 2732 / 9397 / 16384: n6016 1.099 → 1.096, 0.923 → 1.025, 0.876 → 0.969; n6144 1.092 → 1.091, 0.927 → 1.021, 0.850 → 0.952; n6288 1.079 → 1.085, 1.076 → 1.101, 1.145 → 1.146; n8448 0.914 → 0.985, 0.894 → 0.949, 1.109 → 1.127; outputs bit-identical, same ISA up to m 893 (#115 description draft, local:/workspace/code/gpt_oss_docker/work/sikl/PR_k3_gemm_kernels.md:63)
- S1 aklr/ops/gemm/contract.py:95 `_dot_k()` (fp32, `max_split_k` 16) on `GEMM` at :128
- S2 aklr/ops/gemm/impls/flydsl/gemm_n6016_k7168/fe6fa078b3de6610.json#kernel.py:25 `_CONFIGS`; preload 16 and AGPR form at :121, :124; `_count_ready` "this call's tag" at :139; launch tag at :619, NaN at :678; last-split poll at :732; `_CONFIGS4` at :786
- S3 aklr/ops/gemm/impls/flydsl/gemm_n6144_k7168/41ac253b2d1a6704.json#kernel.py:26 `_CONFIGS`; `_count_ready` at :127; poll at :427, NaN at :436; `_CONFIGS4` at :505; preload 8 at :822
- S4 aklr/ops/gemm/impls/flydsl/gemm_n8448_k7168/5beee8319bd5c527.json#kernel.py:1 docstring "sum … in fixed order and round once"; `_RED_FLAGS` at :41; `_pack24` (+0x80, keep the top 24 bits) at :109; `_count_ready` at :199; reducer poll at :619; `_CONFIGS4` at :710
- S5 aklr/ops/gemm/impls/flydsl/gemm_n6288_k7168/18df46ab10cda50d.json#kernel.py:22 `_SKINNY`; `_SPIN_LIMIT` "writes an unfolded value" at :33; launch tag at :354; partner's pairs first at :380; poll at :424; own + partner sum at :441; `_CONFIG4` at :727; `run` allocating `out` at :986
- S6 aklr/ops/gemm/impls/flydsl/gemm_n7168_k35840/5ac66d6dae624f93.json#kernel.py:27 `_CONFIGS`; `_TAIL_SPLITS` at :80; `_SPIN_LIMIT` at :90; one K slice per XCD at :283; fold at :538; dispatch tag at :661
