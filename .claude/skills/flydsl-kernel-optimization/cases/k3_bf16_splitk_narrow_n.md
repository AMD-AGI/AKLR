---
title: "bf16 GEMM, N 144–3584 × K 7168 (Kimi-K3) — K split 4–16 ways across workgroups at m ≤ 112 and folded in a fixed order inside the launch; tiled or lightly split grids at m ≥ 893"
kind: case
ops: [gemm]
dtypes: [bf16]
regimes: ["M 7-112", "M 893-16384"]
contract: bitwise
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=1
measured:
  - {metric: "speedup over aiter, local validate lower bound per Point; no CI measurement on main", value: "1.271–2.000 (2.000 is the bound's cap); 17 of 17 Points ≥ 1.03; 0 served", gpu: MI355X, workload: "gemm_n144_k7168, m 8, 12, 16, 24, 32, 40, 48, 56, 60, 64, 80, 96, 112, 893, 2732, 9397, 16384", run_ref: "0714610831be (commit message table; branch kyle/skill-kernel-loop)", date: "2026-10-25"}
  - {metric: "speedup over aiter, local validate lower bound per Point; no CI measurement on main", value: "1.272–2.000; 11 of 11 Points ≥ 1.03; 0 served", gpu: MI355X, workload: "gemm_n768_k7168, m 7, 14, 21, 28, 35, 42, 49, 56, 70, 84, 98", run_ref: "7dc4dd631 (commit message table; branch kyle/skill-kernel-loop)", date: "2026-10-25"}
  - {metric: "speedup over aiter, local validate lower bound per Point; no CI measurement on main", value: "0.962–2.000; 14 of 15 Points ≥ 1.03 (not m 9397); 0 served", gpu: MI355X, workload: "gemm_n1280_k7168, m 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 893, 2732, 9397, 16384", run_ref: "9120f0be7 (commit message table; branch kyle/skill-kernel-loop)", date: "2026-10-25"}
  - {metric: "speedup over aiter, local validate lower bound per Point; no CI measurement on main", value: "0.956–1.594; 15 of 17 Points ≥ 1.03 (not m 9397, 16384); 0 served", gpu: MI355X, workload: "gemm_n1536_k7168, m 8–16384 (17 Points)", run_ref: "bea458605 (commit message table; branch kyle/skill-kernel-loop)", date: "2026-10-25"}
  - {metric: "speedup over aiter, local validate lower bound per Point; no CI measurement on main", value: "1.034–1.697; 17 of 17 Points ≥ 1.03; 0 served", gpu: MI355X, workload: "gemm_n2112_k7168, m 8–16384 (17 Points)", run_ref: "513005dab (commit message table; branch kyle/skill-kernel-loop)", date: "2026-10-25"}
  - {metric: "speedup over aiter, local validate lower bound per Point; no CI measurement on main", value: "1.229–2.000; 11 of 11 Points ≥ 1.03; 0 served", gpu: MI355X, workload: "gemm_n3584_k7168, m 7, 14, 21, 28, 35, 42, 49, 56, 70, 84, 98", run_ref: "4a8e9860b (commit message table; branch kyle/skill-kernel-loop)", date: "2026-10-25"}
sources:
  - aklr/ops/gemm/impls/flydsl/gemm_n144_k7168/c86923da04869cbc.json
  - aklr/ops/gemm/impls/flydsl/gemm_n768_k7168/28d19101c5b0d2f4.json
  - aklr/ops/gemm/impls/flydsl/gemm_n1280_k7168/572083dd5cfea91f.json
  - aklr/ops/gemm/impls/flydsl/gemm_n1536_k7168/0b9e2a207d909b40.json
  - aklr/ops/gemm/impls/flydsl/gemm_n2112_k7168/2b8c0ff5a4fa7981.json
  - aklr/ops/gemm/impls/flydsl/gemm_n3584_k7168/631331203e1c39ad.json
---

# bf16 GEMM at N 144–3584, K 7168 (Kimi-K3): split K across workgroups, fold it inside the launch

Six FlyDSL kernels for the `gemm@1.0` contract on Kimi-K3 shapes (definitions #69, kernels #115, merge
a8fe490b1), one schema file per shape, written by forge-loop campaigns and reviewed. Each schema embeds its
kernel as `kernel.sources[0].content` (`kernel.py`); function names below are in that file. "m" is the Point's M.
The same fold forms at N 6016–8448 and at K 35840 are in [k3_bf16_splitk_wide_n.md](../cases/k3_bf16_splitk_wide_n.md).

## Applies when

- Contract `gemm@1.0`, layout variant `row_major` (`b.is_shuffled` false): `a` (m, k) and `b` (n, k) bf16,
  `trans_b`, no bias, bf16 `out`. `compare` is bitwise (`determinism: bitwise`, `max_violation_fraction` 0.0,
  the `CompareSpec` defaults), with the fp32 dot-product error bound of `_dot_k()` (`max_split_k` 16; `GEMM` in
  `aklr/ops/gemm/contract.py`). Every fold these kernels run adds its terms in an order fixed by index.
- K = 7168 with N = 144, 768, 1280, 1536, 2112 or 3584 (a weight of 2–49 MiB). At m ≤ 112 the output has 9–224
  column tiles for 256 CUs, so K is split across workgroups to fill the device (`../gemm/flydsl_gemm_small_m.md`
  §3, §4; `../gemm/flydsl_gemm_grid_pricing.md` §1, §2).
- Points: n768 and n3584 m 7–98 (11); n144, n1536 and n2112 m 8–16384 (17); n1280 m 8–16384 (15).
- Each schema name has only this file (no file of another language or an older kernel under the same name on
  main), so at every Point the yardstick is aiter.
- **No Point of these files is served today.** No CI run has written `speedup` or `accuracy` into their
  `workloads` (main at 0cef4f2e3); dispatch serves a workload only from `AKLR_MIN_SPEEDUP` 1.03 and falls back with
  `not_measured` (`aklr/ops/README.md` § Environment variables; `aklr/core/contract/spec/_workloads.py:38`).

## Counter-signals

- **Below 1.03 in the local validate:** n1280 at m 9397 (0.962), an unsplit 192×256 grid of 245 workgroups
  *(field, E3)*; n1536 at m 9397 (0.956) and 16384 (0.976), unsplit 224×256 (252 workgroups) and 256×192 (512,
  two rounds) *(field, E4)*.
- **Every fold waits on other workgroups; what a wait does when it runs out:**

  | File | Fold (Points) | Tag a waiter matches | Poll bound | When the bound is hit | Every split must be resident |
  |---|---|---|---|---|---|
  | n144 | last split (m ≤ 112); every split waits for all, each fragment's owner folds (m ≥ 893) | one NaN bit pattern, the same every call | none | the wave spins on | yes: at m ≤ 112 the XCD remap can dispatch the folding split before its producers (grid ≤ 256 asserted); at m ≥ 893 every split waits for all |
  | n768 | reducer workgroups appended to the grid (m 14–98) | `MAGIC ^ split<<16 ^ sub-tile`, the same every call | 65536 polls, `s_sleep 1` | stores NaN | no |
  | n1280 | appended reducers (m 16–112, 893); owner + tail (2732) | the same every call | 65536 | stores NaN | no |
  | n1536 | every workgroup folds its share after all arrive (m ≤ 112, 893, 2732) | this launch's `dispatch.id ^ queue.ptr` | 2²² polls, `s_sleep 0` | sums whatever the slots hold and stores it | yes, 256 workgroups |
  | n2112 | last split (8–32), appended reducers (40–112), each of 4 splits folds a quarter (893) | the same every call | 65536 | stores NaN | at m 893 (220) |
  | n3584 | each of 4 splits folds a quarter of the tile's rows (21–98) | the same every call | 65536 | stores NaN | each tile's 4 splits (224 workgroups in all) |

  A grid that is not resident at once (another kernel holding CUs, or fewer CUs) turns these waits into a hang
  (n144), NaN, or a silently wrong sum (n1536). n1536's comment calls the bound a guard against a hang, not a path
  that runs. A fully resident grid, as on an idle GPU, hides a broken hand-off (`../gemm/flydsl_gemm_small_m.md` §4).
- **Flags read before this call writes them.** Where the tag is the same every call, the workspace comes from
  `torch.empty` and each fold clears its flags (or tags) at the end "for the next call" (n3584's comment): a call
  depends on the previous call having cleared them, and a fold that timed out leaves late producers' flags set
  for the next call to match. `aklr/ops/README.md` § The kernel section and ABI asks for scratch rewritten before
  it is read, or `state_bytes`; n1536's per-launch tag needs no clearing. n2112's `_count_ready` docstring says the
  flags hold "this call's tag".
- **Partials rounded to 24 bits before the fold.** n1280 at m 56–112 and 893 and its tail partial at 2732, and
  n3584 at m 21–56, store each fp32 partial with a 16-bit significand (four in three dwords) and sum them in fp32.
  The added error is at most 2⁻¹⁶ of each partial; against the bound's accumulation term,
  2·(6·√7168 + 2)·2⁻²⁴ ≈ 6.1·10⁻⁵ of Σ|a||b|, that is about a quarter of the budget (estimated). Deterministic, but
  less headroom for an FlyDSL or compiler change.
- **Stale docstrings.** n1280's says A and B stream into LDS by DMA, large m runs one workgroup per tile, and every
  partial crossing workgroups is 24-bit; at m 16–112 the operands load straight into registers, m 2732 runs two
  workgroups per tile, and m 16–48 keep fp32 partials.
- **Code no Point reaches:** n144's `_pick` fallbacks for an m not in its table; n1280's two-launch split
  (`_build_reduce`, `launch_split`); n1536's `_config` branches for m 897–1024 and 2817–4096 and its 256-row choice
  at m 9397.

## Mechanism

Shared by the six kernels:

- **MFMA `mfma_f32_16x16x32_bf16` into fp32**, except n768 at m 7 (`mfma_f32_4x4x4bf16_1k`).
- **At m ≤ 112, K is split S ways across workgroups** (S = 14 or 7 in n144, 16 in n768 and n1280, 8 in n1536,
  7 in n2112, 4 in n3584), most of them also across the waves of a workgroup; partials go in fp32 (or 24-bit) to a
  workspace allocated by `torch.empty` per call, and are folded in the same launch, in split order, then rounded
  once to bf16 (`../gemm/flydsl_gemm_small_m.md` §4, `../gemm/flydsl_gemm_correctness.md` §8). No atomics on data.
- **Hand-off through scoped stores and loads:** producers store partials with `sc1` or `sc0 sc1`, wait
  `vmcnt(0)` (and a barrier where the workgroup publishes together), then store a flag; waiters poll with `sc1`
  loads (n144 with agent-scope atomic 64-bit loads); no fences. n3584 stores its planes and flags with no cache
  bits: all splits of a tile run on one XCD and meet in its L2.
- **Splits placed on XCDs:** n144 (m ≤ 112), n2112 (m 893) and n1280 (m 893) give each XCD a contiguous
  split-major run;
  n768 gives each XCD a contiguous eighth of K; n3584 puts all four splits of a tile on one XCD
  (`../gemm/flydsl_gemm_schedule_and_locality.md` §2).
- **A per-m configuration table** compiled in a `functools.cache` function on a Point's first call.

| | m ≤ 112 | m 893 | m 2732 | m 9397 | m 16384 |
|---|---|---|---|---|---|
| n144 | `_build`: 9 strips of 16 columns × 16–48 rows; 4 waves split K; S 14 (7 at m 112); operands straight from global into registers; 126 or 252 workgroups; split S−1 folds | `_build_large`: one 32-row tile spans all 144 columns, S 8; async DMA into a 4-stage LDS ring; 224 workgroups; every split waits for all, owners fold | 80 rows, S 7; 245 | 160 rows, S 4; 236 | 128 rows, S 2; 256 |
| n768 | m 7: `_build4`, 3 columns per workgroup, K over 8 waves, 256 workgroups, no cross-workgroup split. m 14–98: `_build16`, (16–112) × 48 tiles, 7 waves split K inside, S 16, 256 producers + 24–168 appended reducer workgroups with one reducer wave per 16×16 sub-tile; operands straight into registers | – | – | – | – |
| n1280 | m 8: `skw`, 16×8 tiles, K over 7 waves with a private 6-slot ring each, 160 workgroups. m 16–112: `sk1l`, (16–112) × 80 tiles, 7 waves split K, S 16, appended reducers; operands straight into registers | 128×256, S 7, 245 producers + 140 reducers, 24-bit planes | 128×256: an owner computes K 0–3711, a tail workgroup the rest and publishes a 24-bit partial the owner adds; 256 workgroups | 192×256 unsplit; 245 | 256×320 unsplit; 256 |
| n1536 | (16–112) × 48, 3 waves, S 8, 256 workgroups; m ≤ 48: (value, tag) pairs per output; m 56–112: fp32 planes + one tagged flag per split; every workgroup folds its share | 112×192, 12 waves, S 4; 256 | 176×192, S 2, each XCD one K half; 256 | 224×256 unsplit; 252 | 256×192 unsplit; 512 |
| n2112 | 16–128 × 64, 2–4 waves, S 7; last split at m 8–32 (231 workgroups); appended reducers at m 40–112 (363–429); B `nt` | 192×192, 8 waves, S 4, each split folds a quarter; 220 | 128×192 unsplit; 242 | 416×192 unsplit; 253 | persistent: 256 workgroups, two rounds of 384×192 and 320×192 |
| n3584 | m 7, 14: 256 workgroups of 14 columns, K over 4 waves, no cross-workgroup split. m 21–98: 56 tiles of 64 columns × S 4 = 224 workgroups; each wave owns 16 columns, so no in-workgroup fold; split s folds rows s·⌈m/4⌉… of the tile from planes laid out on whole 128-byte lines, its own rows from LDS; m 21–56 poll tags stored inside 24-bit-packed partials, m 70–98 per-reader flags | – | – | – | – |

At m ≥ 893 (n1280, n1536, n2112) A and B go by buffer→LDS DMA into XOR-swizzled multi-stage rings
(`../gemm/flydsl_gemm_datapath.md` §2, §5), in an XCD-grouped tile order; n1280's 8-wave loop raises `s_setprio`
around its first MFMA group (`../gemm/flydsl_gemm_mainloop.md` §2).

**What each part bought** (campaign ablations; each kept round timed beside the version before it):

- n144: one kernel with the in-launch fixed-order fold at every m +5.3–94.0% at m ≤ 112 over the campaign's
  starting kernel; the LDS-staged loop with the split-K fold at m ≥ 893 +32.4–66.2%; the XCD remap of the splits
  with a leaner hand-off +2.4–25.9% at m ≤ 112; spreading the large-m fold over the splits +3.0–17.7% at m ≥ 893.
  All changes off: +26.2–148.8% per Point *(field, E7)*.
- n768: the campaign's own bench timed aiter at 10.4–14.8 µs and the kernel at 6.4–9.7 µs per Point; its ablation
  record keeps only means *(field, E8)*.
- n1280: the in-launch flag-gated reducers +5.2–33.0% at m 16–112; one-round re-tiles +10.0% / +23.0% at m 9397 /
  16384; the joint split/tile XCD remap with the 8-wave pipeline +5.1% / +7.1% at m 893 / 2732; the owner + tail
  split +5.0% at m 2732 (and +18.9% at 16384); load-issue order +2.1–9.4% at m 16–48 *(field, E9)*. The 24-bit
  planes are not priced in this record (its ablation arm failed).
- n1536: the single-launch fold with dispatch-tagged partials +33.9–84.9% at m ≤ 112; XCD-grouped order with
  re-raced tiles +3.6–19.2% at m ≥ 893; 256 workgroups with flags above m 48 +4.2–15.6% at m ≤ 112; the 12-wave
  192-column split tiles +20.7% / +8.2% at m 893 / 2732 *(field, E10)*.
- n2112: the pipelined kernel with the flag-gated fold +39.1–82.2% at m ≤ 112; the appended reducers +3.7–9.8% at
  m 40–112; the split-major XCD order at m 893 +22.8%; contiguous per-XCD tiles +4.4–18.1% at m ≥ 2732 *(field, E11)*.
- n3584: the 224-workgroup K-split kernel +15.1–76.1%; the in-producer reduce-scatter +1.4–11.9% at m 21–98;
  `nt` B with A issued first +2.9–6.4% at every Point; columns split across the 4 waves +2.7–4.5% at m 21–98;
  whole-line planes +0.3–1.4% at m 28–98. All changes off: +23.2–125.2% *(field, E12)*.

## Result

`speedup` is absent from every workload of these files: CI has not measured them, so dispatch serves none of
their 88 Points and aiter runs them all. When CI does measure them it runs the FlyDSL of its image (0.3.4.1 in
the image of the latest perf runs), not the 0.3.2 of the local numbers below
(`../pitfalls/measured_a_different_build.md`).

The only per-Point measurements are the campaigns' own `python -m sikl validate` runs on MI355X (the repository
before its rename to AKLR), FlyDSL 0.3.2: a lower bound of the CUDA-graph time ratio aiter / kernel, one run each,
capped at 2.000; "≥ 1.03" counts them against the dispatch threshold (that validate itself demoted below 1.02).

| File | m ≤ 112 | m 893 | m 2732 | m 9397 | m 16384 | ≥ 1.03 locally | Served |
|---|---|---|---|---|---|---|---|
| n144 *(field, E1)* | 1.695–2.000 | 1.420 | 1.300 | 1.535 | 1.271 | 17 of 17 | 0 |
| n768 *(field, E2)* | 1.272–2.000 | – | – | – | – | 11 of 11 | 0 |
| n1280 *(field, E3)* | 1.277–2.000 | 1.309 | 1.188 | 0.962 | 1.096 | 14 of 15 | 0 |
| n1536 *(field, E4)* | 1.150–1.521 | 1.594 | 1.173 | 0.956 | 0.976 | 15 of 17 | 0 |
| n2112 *(field, E5)* | 1.059–1.697 | 1.635 | 1.044 | 1.034 | 1.269 | 17 of 17 | 0 |
| n3584 *(field, E6)* | 1.229–2.000 | – | – | – | – | 11 of 11 | 0 |

"–": no such Point.

## Reproduce

On an MI355X with the repository's FlyDSL 0.3.2, from the repository root, one schema at a time:

```
python -m aklr validate aklr/ops/gemm/impls/flydsl/gemm_n144_k7168/c86923da04869cbc.json
python -m aklr validate aklr/ops/gemm/impls/flydsl/gemm_n768_k7168/28d19101c5b0d2f4.json
python -m aklr validate aklr/ops/gemm/impls/flydsl/gemm_n1280_k7168/572083dd5cfea91f.json
python -m aklr validate aklr/ops/gemm/impls/flydsl/gemm_n1536_k7168/0b9e2a207d909b40.json
python -m aklr validate aklr/ops/gemm/impls/flydsl/gemm_n2112_k7168/2b8c0ff5a4fa7981.json
python -m aklr validate aklr/ops/gemm/impls/flydsl/gemm_n3584_k7168/631331203e1c39ad.json
```

The report goes to `aklr_validation/report.md`; a local run checks the file, the served numbers are CI's
(`aklr/ops/README.md` § Workflow). Add `--base origin/main` only on a branch that edits the file: with the base set,
bench measures only schemas that are not at the base, so on an unchanged checkout it skips the bench
(`aklr/core/validation/stages/validation_scope.py`, `of_working_tree`). Validate's bitwise check runs the kernel on
an otherwise idle GPU, where every split is resident; test the folds also with a grid that is not
(`../gemm/flydsl_gemm_small_m.md` §4). To read a kernel:
`git show <rev>:<schema> | python3 -c 'import json,sys; print(json.load(sys.stdin)["kernel"]["sources"][0]["content"])'`.

## Rollback

Delete the schema file: its Points fall back to aiter, since no other file serves them. `AKLR_DISABLE=gemm@1.0`
sends the whole contract to aiter, every other bf16 GEMM schema included. Until CI writes `speedup`, nothing is
served and nothing needs rolling back; afterwards a Point measured below 1.03 is not served, and a `failed`
accuracy never is. Roll back for a wrong result CI does not catch: NaN or a stale fold after a timed-out or aborted
launch, a silently wrong sum from n1536 when its splits are not resident, a hang of n144, or a change of FlyDSL or
LLVM.

## Evidence

`<schema>.json#kernel.py:<line>` is a line of the `kernel.py` a schema embeds in `kernel.sources`; "local validate"
is `python -m sikl validate` on MI355X in the campaign containers, the ratio lower bound over aiter per Point.

- E1 [2026-10, gfx950, FlyDSL 0.3.2, N=1, local validate] n144: m 8 1.994, 12 2.000, 16 2.000, 24 2.000, 32 2.000, 40 1.995, 48 2.000, 56 1.965, 60 2.000, 64 1.999, 80 1.871, 96 1.795, 112 1.695, 893 1.420, 2732 1.300, 9397 1.535, 16384 1.271 (gemm_n144_k7168; commit message of 0714610831be)
- E2 [2026-10, gfx950, FlyDSL 0.3.2, N=1, local validate] n768: m 7 2.000, 14 1.712, 21 1.550, 28 1.573, 35 1.445, 42 1.440, 49 1.336, 56 1.347, 70 1.272, 84 1.619, 98 1.495 (gemm_n768_k7168; commit message of 7dc4dd631)
- E3 [2026-10, gfx950, FlyDSL 0.3.2, N=1, local validate] n1280: m 8 2.000, 16 1.589, 24 1.412, 32 1.453, 40 1.277, 48 1.302, 56 1.622, 64 1.614, 80 1.470, 96 1.352, 112 1.387, 893 1.309, 2732 1.188, 9397 0.962, 16384 1.096 (gemm_n1280_k7168; commit message of 9120f0be7)
- E4 [2026-10, gfx950, FlyDSL 0.3.2, N=1, local validate] n1536: m 8 1.150, 12 1.173, 16 1.180, 24 1.238, 32 1.287, 40 1.278, 48 1.313, 56 1.272, 60 1.269, 64 1.284, 80 1.466, 96 1.521, 112 1.326, 893 1.594, 2732 1.173, 9397 0.956, 16384 0.976 (gemm_n1536_k7168; commit message of bea458605)
- E5 [2026-10, gfx950, FlyDSL 0.3.2, N=1, local validate] n2112: m 8 1.059, 12 1.119, 16 1.109, 24 1.582, 32 1.226, 40 1.587, 48 1.346, 56 1.556, 60 1.427, 64 1.446, 80 1.406, 96 1.697, 112 1.547, 893 1.635, 2732 1.044, 9397 1.034, 16384 1.269 (gemm_n2112_k7168; commit message of 513005dab)
- E6 [2026-10, gfx950, FlyDSL 0.3.2, N=1, local validate] n3584: m 7 1.306, 14 1.229, 21 1.541, 28 1.535, 35 1.627, 42 1.619, 49 1.531, 56 1.763, 70 2.000, 84 1.486, 98 1.685 (gemm_n3584_k7168; commit message of 4a8e9860b)
- E7 [2026-10, gfx950, FlyDSL 0.3.2, N=?, forge-loop paired bench; remeasured rounds 3 pairs] n144: one-kernel fold +62.0, 88.7, 94.0, 77.8, 77.0, 57.4, 60.3, 47.2, 48.1, 49.6, 13.9, 7.3, 5.3% at m 8–112 (recorded); LDS-staged large-m loop +32.4, 48.4, 61.6, 66.2% at m 893–16384 (recorded); XCD remap + hand-off +2.4–25.9% at m ≤ 112, +1.1–11.9% at m ≥ 893 (recorded); spread fold +6.1, 12.3, 17.7, 3.0% at m 893–16384 (recorded); all off +26.2–148.8% (gemm_n144_k7168; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n144_k7168/forge_experiments/ablation.json)
- E8 [2026-10, gfx950, FlyDSL 0.3.2, N=?, campaign driver bench] n768 per Point, aiter → kernel: m 7 14.48 → 6.41 µs, 14 11.33 → 6.70, 21 10.54 → 6.83, 28 10.94 → 6.96, 35 10.47 → 7.23, 42 10.77 → 7.36, 49 10.41 → 7.71, 56 10.71 → 7.86, 70 10.76 → 8.46, 84 14.84 → 9.16, 98 14.75 → 9.70; the ablation file holds per-round means only (gemm_n768_k7168; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards/demo_n768_12h/forge_experiments/dashboard.json; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards/demo_n768_12h/forge_experiments/ablation.json)
- E9 [2026-10, gfx950, FlyDSL 0.3.2, N=?, forge-loop paired bench; remeasured rounds 3 pairs] n1280: in-launch reducers +5.2–33.0% at m 16–112 (recorded); re-tiles +10.0 / +23.0% at 9397 / 16384 (recorded); XCD remap + 8-wave pipe +5.1 / +7.1% at 893 / 2732; owner + tail +5.0% at 2732 and +18.9% at 16384; load order +9.4, 2.7, 2.8, 3.1, 2.1% at m 16–48; 24-bit packing arm failed (gemm_n1280_k7168; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n1280_k7168/forge_experiments/ablation.json)
- E10 [2026-10, gfx950, FlyDSL 0.3.2, N=?, forge-loop paired bench; remeasured rounds 3 pairs] n1536: single-launch fold +33.9–84.9% at m ≤ 112 (recorded); XCD-grouped order + tiles +17.6, 19.2, 3.6, 17.3% at m 893–16384 (recorded); 256 flagged workgroups +4.2–15.6% at m ≤ 112 (recorded); 12-wave split tiles +20.7 / +8.2% at 893 / 2732 (gemm_n1536_k7168; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n1536_k7168/forge_experiments/ablation.json)
- E11 [2026-10, gfx950, FlyDSL 0.3.2, N=?, forge-loop paired bench; remeasured rounds 3 pairs] n2112: pipelined kernel + fold +39.1–82.2% at m ≤ 112 (recorded); reducer workgroups +3.9, 3.7, 4.7, 4.7, 5.1, 6.1, 6.9, 9.8% at m 40–112 (recorded); split-major order +22.8% at 893; per-XCD tiles +7.3, 18.1, 4.4% at 2732, 9397, 16384 (recorded); larger tiles +8.3 / +10.5% at 9397 / 16384 (gemm_n2112_k7168; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n2112_k7168/forge_experiments/ablation.json)
- E12 [2026-10, gfx950, FlyDSL 0.3.2, N=?, forge-loop paired bench; remeasured rounds 3 pairs] n3584: K-split kernel +15.1–76.1% (recorded); reduce-scatter +4.6, 1.4, 2.0, 2.7, 1.5, 2.1, 7.8, 11.9, 11.3% at m 21–98; `nt` B + A first +2.9–6.4% (recorded); columns over waves +2.7–4.5% at m 21–98 (recorded); whole-line planes +0.3–1.4%; all off +23.2–125.2% (gemm_n3584_k7168; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n3584_k7168/forge_experiments/ablation.json)
- S1 aklr/ops/gemm/contract.py:95 `_dot_k()` (fp32, `max_split_k` 16) on `GEMM` at :128; `_SLACK`, `_SQRT_LENGTH_MULTIPLE`, `_CONSTANT_MULTIPLE` at aklr/core/contract/semantics/_compare.py:249
- S2 aklr/ops/gemm/impls/flydsl/gemm_n144_k7168/c86923da04869cbc.json#kernel.py:25 the NaN-pattern MAGIC; agent-scope atomic flag store and poll at :70 and :82; `_CONFIG` at :102; `_LARGE` at :679; unbounded polls at :355 and :597; co-residency assert at :189
- S3 aklr/ops/gemm/impls/flydsl/gemm_n768_k7168/28d19101c5b0d2f4.json#kernel.py:22 `_CONFIGS`; splits per XCD at :163; bounded poll at :232, NaN at :250, flag clear at :259; workspace per call at :391
- S4 aklr/ops/gemm/impls/flydsl/gemm_n1280_k7168/572083dd5cfea91f.json#kernel.py:1 docstring; `p24 = ma >= 4` at :1328; operands loaded straight from global at :1364; `_CONFIGS` at :1473; NaN on timeout at :747, :924, :1447
- S5 aklr/ops/gemm/impls/flydsl/gemm_n1536_k7168/0b9e2a207d909b40.json#kernel.py:21 poll bound "only guards against a hang"; `_config` at :48; dispatch tag at :292; (value, tag) pairs at :493; the fold after the poll loop sums without checking at :596
- S6 aklr/ops/gemm/impls/flydsl/gemm_n2112_k7168/2b8c0ff5a4fa7981.json#kernel.py:181 `_count_ready` "this call's tag" over a static `_MAGIC` (:64); bounded polls with NaN at :536, :616, :694; split-major order at :311
- S7 aklr/ops/gemm/impls/flydsl/gemm_n3584_k7168/631331203e1c39ad.json#kernel.py:29 `CONFIGS`; `TAGGED_PLANES` at :46; plane and flag stores without cache bits, polls with `sc1` (:56, :57); `_pack24` at :102; all splits of a tile on one XCD at :228; polls at :528 and :607, NaN at :537; "clear the tags or flags for the next call" at :641
