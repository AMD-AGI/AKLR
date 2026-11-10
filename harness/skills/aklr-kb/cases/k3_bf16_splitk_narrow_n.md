---
title: "bf16 GEMM, N 144–3584 × K 7168 (Kimi-K3) — K split 4–16 ways across workgroups at m ≤ 112 and folded in a fixed order inside the launch; tiled or lightly split grids at m ≥ 893; N 1024 and 2048 × K 7168 at m 1–128 with the same in-workgroup ring and a fold that does not wait; N 384 × K 7168 (DeepSeek-V4-Pro) at m 7–256 on narrower tiles, with that fold at m 256"
kind: case
ops: [gemm]
dtypes: [bf16]
regimes: ["M 1-256", "M 893-16384"]
contract: bitwise
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=1
measured:
  - {metric: "speedup over aiter, local validate lower bound per Point", value: "1.271–2.000 (2.000 is the bound's cap); 17 of 17 Points ≥ 1.03; CI serves 17 of 17 as of #102", gpu: MI355X, workload: "gemm_n144_k7168, m 8, 12, 16, 24, 32, 40, 48, 56, 60, 64, 80, 96, 112, 893, 2732, 9397, 16384", date: "2026-10-25"}
  - {metric: "speedup over aiter, local validate lower bound per Point", value: "1.272–2.000; 11 of 11 Points ≥ 1.03; CI serves 11 of 11 as of #102", gpu: MI355X, workload: "gemm_n768_k7168, m 7, 14, 21, 28, 35, 42, 49, 56, 70, 84, 98", date: "2026-10-25"}
  - {metric: "speedup over aiter, local validate lower bound per Point", value: "0.962–2.000; 14 of 15 Points ≥ 1.03 (not m 9397); CI serves 14 of 15 as of CI run 42000899019 (d908d01, #71), unchanged by #102", gpu: MI355X, workload: "gemm_n1280_k7168, m 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 893, 2732, 9397, 16384", date: "2026-10-25"}
  - {metric: "speedup over aiter, local validate lower bound per Point", value: "0.956–1.594; 15 of 17 Points ≥ 1.03 (not m 9397, 16384); CI serves 15 of 17 as of #102", gpu: MI355X, workload: "gemm_n1536_k7168, m 8–16384 (17 Points)", date: "2026-10-25"}
  - {metric: "speedup over aiter, local validate lower bound per Point", value: "1.034–1.697; 17 of 17 Points ≥ 1.03; CI serves 15 of 17 as of #102", gpu: MI355X, workload: "gemm_n2112_k7168, m 8–16384 (17 Points)", date: "2026-10-25"}
  - {metric: "speedup over aiter, local validate lower bound per Point", value: "1.229–2.000; 11 of 11 Points ≥ 1.03; CI serves 11 of 11 as of #102", gpu: MI355X, workload: "gemm_n3584_k7168, m 7, 14, 21, 28, 35, 42, 49, 56, 70, 84, 98", date: "2026-10-25"}
  - {metric: "speedup over aiter, AKLR validate profile 19, per Point; no CI measurement yet", value: "1.069–1.466; 8 of 8 Points at or above AKLR_MIN_SPEEDUP", gpu: MI355X, workload: "gemm_n1024_k7168 file 23da2fcbe6d9defd, measured as its predecessor, which differed only in taking the current stream without naming the device, m 1, 2, 4, 8, 16, 32, 64, 128", date: "2026-11-09"}
  - {metric: "speedup over aiter, AKLR validate profile 19, per Point; no CI measurement yet", value: "1.114–1.224; 8 of 8 Points at or above AKLR_MIN_SPEEDUP", gpu: MI355X, workload: "gemm_n2048_k7168 file 1f5a93c79dbd12bf, m 1, 2, 4, 8, 16, 32, 64, 128", date: "2026-11-09"}
  - {metric: "speedup over aiter, AKLR validate profile 19, per Point; no CI measurement yet", value: "1.142–1.360; 21 of the 21 Points up to m 256 at or above AKLR_MIN_SPEEDUP", gpu: MI355X, workload: "gemm_n384_k7168 file d65ffa02dee32841, m 7, 14, …, 224 (7 × b) and 256", run_ref: "this file's AKLR validate, local, not CI", date: "2026-11-10"}
sources:
  - aklr/ops/gemm/impls/flydsl/gemm_n144_k7168/c86923da04869cbc.json
  - aklr/ops/gemm/impls/flydsl/gemm_n768_k7168/28d19101c5b0d2f4.json
  - aklr/ops/gemm/impls/flydsl/gemm_n1280_k7168/572083dd5cfea91f.json
  - aklr/ops/gemm/impls/flydsl/gemm_n1536_k7168/0b9e2a207d909b40.json
  - aklr/ops/gemm/impls/flydsl/gemm_n2112_k7168/2b8c0ff5a4fa7981.json
  - aklr/ops/gemm/impls/flydsl/gemm_n3584_k7168/631331203e1c39ad.json
  - aklr/ops/gemm/impls/flydsl/gemm_n1024_k7168/23da2fcbe6d9defd.json
  - aklr/ops/gemm/impls/flydsl/gemm_n2048_k7168/1f5a93c79dbd12bf.json
  - aklr/ops/gemm/impls/flydsl/gemm_n384_k7168/d65ffa02dee32841.json
---

# bf16 GEMM at N 144–3584, K 7168 (Kimi-K3): split K across workgroups, fold it inside the launch

Six of this page's nine FlyDSL `gemm@1.0` files are the Kimi-K3 set,
one schema file per shape, written by forge-loop campaigns and reviewed. Each schema embeds its
kernel as `kernel.sources[0].content` (`kernel.py`); function names below are in that file. "m" is the Point's M.
The same fold forms at N 6016–8448 and at K 35840 are in [k3_bf16_splitk_wide_n.md](../cases/k3_bf16_splitk_wide_n.md).

## Applies when

- Contract `gemm@1.0`, layout variant `row_major` (`b.is_shuffled` false): `a` (m, k) and `b` (n, k) bf16,
  `trans_b`, no bias, bf16 `out`. `compare` is bitwise (`determinism: bitwise`, `max_violation_fraction` 0.0,
  the `CompareSpec` defaults), with the fp32 dot-product error bound of `_dot_k()` (`max_split_k` 16; `GEMM` in
  `aklr/ops/gemm/contract.py`). Every fold these kernels run adds its terms in an order fixed by index.
- K = 7168 with N = 144, 768, 1280, 1536, 2112 or 3584 (a weight of 2–49 MiB). At m ≤ 112 the output has 9–224
  column tiles for 256 CUs, so K is split across workgroups to fill the device (`../gemm/small_m.md`
  §3, §4; `../gemm/grid_pricing.md` §1, §2).
- Points: n768 and n3584 m 7–98 (11); n144, n1536 and n2112 m 8–16384 (17); n1280 m 8–16384 (15).
- Each schema name has only this file (no file of another language or an older kernel under the same name on
  main), so at every Point the yardstick is aiter.
- **CI serves 83 of their 88 Points** as of #102 (6d7cfec, CI run 42549655045, measurement profile 19), n1280's as
  of CI run 42000899019 (d908d01, #71), which #102 left unchanged; the count per file is in § Result, and the
  current value of each Point is the `speedup` in its schema's `workloads`. Dispatch serves a workload only from
  `AKLR_MIN_SPEEDUP` 1.03 (`aklr/ops/README.md` § Environment variables; `aklr/core/contract/spec/_workloads.py:38`).

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
  that runs. A fully resident grid, as on an idle GPU, hides a broken hand-off (`../gemm/small_m.md` §4).
- **Flags read before this call writes them.** Where the tag is the same every call, the workspace comes from
  `torch.empty` and each fold clears its flags (or tags) at the end "for the next call" (n3584's comment): a call
  depends on the previous call having cleared them, and a fold that timed out leaves late producers' flags set
  for the next call to match. `aklr/ops/README.md` § The kernel section and ABI asks for scratch rewritten before
  it is read; n1536's per-launch tag needs no clearing. n2112's `_count_ready` docstring says the
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
  once to bf16 (`../gemm/small_m.md` §4, `../gemm/correctness.md` §8). No atomics on data.
- **Hand-off through scoped stores and loads:** producers store partials with `sc1` or `sc0 sc1`, wait
  `vmcnt(0)` (and a barrier where the workgroup publishes together), then store a flag; waiters poll with `sc1`
  loads (n144 with agent-scope atomic 64-bit loads); no fences. n3584 stores its planes and flags with no cache
  bits: all splits of a tile run on one XCD and meet in its L2.
- **Splits placed on XCDs:** n144 (m ≤ 112), n2112 (m 893) and n1280 (m 893) give each XCD a contiguous
  split-major run;
  n768 gives each XCD a contiguous eighth of K; n3584 puts all four splits of a tile on one XCD
  (`../gemm/schedule_and_locality.md` §2).
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
(`../gemm/datapath.md` §2, §5), in an XCD-grouped tile order; n1280's 8-wave loop raises `s_setprio`
around its first MFMA group (`../gemm/mainloop.md` §2).

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

"Served" counts the Points at or above 1.03 in CI as of #102 (6d7cfec, CI run 42549655045, measurement profile 19),
n1280 as of CI run 42000899019 (d908d01, #71), which #102 left unchanged; the `speedup` in each schema's `workloads`
is the current value *(field, E19)*. CI runs the FlyDSL of its image (0.3.4.1 in the image of the latest perf runs),
not the 0.3.2 of the local numbers below (`../pitfalls/measured_a_different_build.md`).

The other columns are the campaigns' own runs of the predecessor repository's `validate` (before the rename to
AKLR) on MI355X, FlyDSL 0.3.2: a lower bound of the CUDA-graph time ratio aiter / kernel, one run each,
capped at 2.000; "≥ 1.03" counts them against the dispatch threshold (that validate itself demoted below 1.02).

| File | m ≤ 112 | m 893 | m 2732 | m 9397 | m 16384 | ≥ 1.03 locally | Served |
|---|---|---|---|---|---|---|---|
| n144 *(field, E1)* | 1.695–2.000 | 1.420 | 1.300 | 1.535 | 1.271 | 17 of 17 | 17 of 17 |
| n768 *(field, E2)* | 1.272–2.000 | – | – | – | – | 11 of 11 | 11 of 11 |
| n1280 *(field, E3)* | 1.277–2.000 | 1.309 | 1.188 | 0.962 | 1.096 | 14 of 15 | 14 of 15 |
| n1536 *(field, E4)* | 1.150–1.521 | 1.594 | 1.173 | 0.956 | 0.976 | 15 of 17 | 15 of 17 |
| n2112 *(field, E5)* | 1.059–1.697 | 1.635 | 1.044 | 1.034 | 1.269 | 17 of 17 | 15 of 17 |
| n3584 *(field, E6)* | 1.229–2.000 | – | – | – | – | 11 of 11 | 11 of 11 |

"–": no such Point.

## N 1024 and 2048 × K 7168 at m 1–128: the same in-workgroup ring, and a fold that does not wait

Two later files, one per schema name: `gemm_n1024_k7168/23da2fcbe6d9defd.json` and
`gemm_n2048_k7168/1f5a93c79dbd12bf.json` (14 and 28 MiB of weight), each listing m 1, 2, 4, …, 128. The other file
of each name (`70dd928e…`, `21d17158…`) has no kernel, so the yardstick is aiter; until CI writes their `speedup`,
dispatch does not serve them (`qualifies()` in `aklr/core/contract/spec/_workloads.py`; `AKLR_MIN_SPEEDUP`, default
1.03, `aklr/ops/README.md` § Environment variables). Every Point launches 256 workgroups, and every fold adds in a
fixed order.

**N 1024 × K 7168** (`23da2fcb…`):
- m 1, 2: `_build_gemv_launcher`, one builder for both m that combines `gemm_n1024_k4096`'s `_build_gemv_launcher`
  (m 1) and `_build_multi_gemv_launcher` (m 2) ([bf16_k4096_splitk_hgemm.md](bf16_k4096_splitk_hgemm.md)): no MFMA;
  each wave owns one output column and its lanes split K. Every 16 B load of A's rows and B's column over
  the whole K is issued before the first `v_dot2_f32_bf16`, one fp32 chain per row, then a 6-step `ds_bpermute`
  butterfly and a store from lane 0.
- m 4–128: `_build_skw`, the per-wave ring of n1280's `skw` path (m 8 in the table above): four waves of 1792 K each
  (28 lines of 64), each streaming its lines through a private XOR-swizzled ring in LDS (8, 6, 6, 6, 6 and 4 slots at
  m 4–128) and waiting only on its own `vmcnt`, no `s_barrier` in the K loop. Narrower tiles than n1280's: 16×4
  (m 4, 8; B rows past 4 read zero through an offset past `num_records`), two 8-row M-blocks × 8 columns (m 16),
  16×8 (m 32), 16×16 (m 64), 32×16 (m 128). B is the MFMA's first operand; the partials meet in each wave's dead
  ring and wave f mod 4 adds fragment f in wave order. From m 16, workgroup p starts at K line p·`ROT` mod 28, with
  `ROT` 1 (m 16–64) or 27 (m 128).
- Kernel arguments go to SGPRs through `amdgpu-kernarg-preload-count`; only A's pointer lands there, since preloading
  stops at the first aggregate argument (the code's comment).

**N 2048 × K 7168** (`1f5a93c7…`):
- m 1–8: `_compile_skw_av` of `gemm_n2048_k4096`: 16×8 tiles (B rows 8–15 repeat rows 0–7), four waves of 1792 K
  each, 10 slots per wave, fold in wave order inside the workgroup.
- m 16–128: `_compile_skw_sk`: K split four ways across workgroups (1792 each) and inside each over one wave (m 16,
  32) or two (m 64, 128); tiles 16×32, 32×32, 64×32, and 64×64 in two M-tiles (m 128). The four splits and M-tiles of
  an N tile get workgroup ids 8 apart. Each workgroup adds its waves' partials in order and stores its fp32 plane with
  `sc0 sc1` to a `torch.empty` workspace allocated by the call. Then, per (tile, wave) slot, each wave waits
  `vmcnt(0)`, stores this launch's 64-bit tag `dispatch.id ^ rotl(queue.ptr, 32)` into its slot for its split, waits
  again and reads the slots of all four splits; a wave that finds all four tags loads the four planes (`sc0 sc1`),
  adds them in split order, rounds once and stores. Unlike every fold in the table under § Counter-signals, no
  workgroup waits for another; two waves that both fold a fragment write the same bytes.

Counter-signals:
- **The slots are read before this call writes them.** `SKILL.md`'s Stop table asks for scratch written in the call
  before it is read, and for counters zeroed in each call unless stamped with the launch's 64-bit tag. N 2048 × K 7168
  at m 16–128 reads the slots of an
  uninitialised `torch.empty` workspace: its output is right only while no leftover word in a slot equals this
  launch's tag, and while the tag differs between launches that can be given the same workspace. Checked, on every
  listed Point of both files: the forge driver's conditions at 20 tries (streams limited to 16 and 64 CUs; a second
  concurrent call, which the driver runs on the same inputs), one graph replayed 24 times with three input sets
  rotated into its buffers, and two calls at once on two streams with different inputs; every try gave the output of
  the call alone *(field, E15)*. Not checked: a full model's decode graph, dispatch-id wraparound, and a garbage word
  equal to the tag.
- **The K split across workgroups starts at m 16.** At m 8 a 4-way split read 7.71 µs against 7.00 for the whole K in
  one workgroup *(field, E18)*.
- **The GEMV stops at m 2.** It beat the per-wave ring at m 1 and 2 and lost at m 4 *(field, E16)*.
- **No K-start rotation at m ≤ 8.** At N 1024, m 4, every rotation tried read 5.27–5.34 µs against 5.246 without. At
  m 8 the first rotation form read 5.86–6.01 against 5.985 without on the ring it was tried on, but the shipped
  unrotated 6-slot ring read 5.793, below the best rotation there, 5.856 *(field, E17)*.
- **Fallback.** Both `run`s send an m outside their tables to `torch.matmul`; no listed Point reaches it.
- **At N 384 a split across workgroups was faster at m 8** (§ N 384 × K 7168 below). N 2048's whole-K grid has 256
  workgroups; at N 384 its whole-K builder has 48. Why the split loses at one N and wins at the other was not
  measured.

AKLR validate (measurement profile 19) on one MI355X at main 0e02a81, not CI: N 1024 1.069–1.466 over aiter, highest
at m 1 (the GEMV), lowest at m 32 and 128 (1.071, 1.069) *(field, E13)*; N 2048 × K 7168 1.114–1.224, highest at
m 128 (split across workgroups) *(field, E14)*. Reproduce with `python -m aklr validate` on
`aklr/ops/gemm/impls/flydsl/gemm_n1024_k7168/23da2fcbe6d9defd.json` and
`aklr/ops/gemm/impls/flydsl/gemm_n2048_k7168/1f5a93c79dbd12bf.json` (§ Reproduce). To roll back, delete the file: its
Points are then served by no file, and the caller runs its own path. Roll back N 2048 × K 7168 for a wrong or NaN
output under a CUDA graph that reuses its workspace.

## N 384 × K 7168 at m 7–256 (DeepSeek-V4-Pro's router): N 1024's ring on narrower tiles, a tagged fold at m 256

One file, `gemm_n384_k7168/d65ffa02dee32841.json` (5.25 MiB of weight), serves 21 of the 27 Points of
`9e1347bc…`, the router GEMM of DeepSeek-V4-Pro at TP8 (hidden 7168, 384 routed experts; #118): m = 7 × b up to
224, and 256. No other file serves this schema name, so the yardstick is aiter. m 512–16384 are not in the file
and stay with the caller's path.

A forge-loop campaign tuned N 1024's file at m 1, 2, 4, …, 256, the Points before #118. This file keeps its two
MFMA paths and takes each new m's row from the campaign's rows, then from a race of a few candidates per m
(below); no forge-loop round ran at these m. `_SKW_CONFIG` and `_SK_CONFIG` list the rows (S10).

- m 7–154: N 1024's `_build_skw` (four waves of 1792 K, a private ring each, fold in wave order) on tiles narrow
  enough for 144–240 workgroups in one round: 16×2 at m 7 (B rows past 2 read zero); 8-row M-blocks × 4, 8 or 16
  columns at m 14–70; 16×16 at m 84–154. At m 42, 84 and 126 the order is XCD-blocked (`XG` 2): XCD `pid % 8` owns
  a block of half the M-tiles and a quarter of the N-tiles (`../gemm/schedule_and_locality.md` §2).
- m 168–256: `_build_sk` on m 256's grid: 32×64 tiles, 8 M-tiles whatever m is (rows at or past m read zero and
  are not stored), K split 4 ways across workgroups (1792 each, 448 per wave, a 2-slot ring), 48 tiles × 4
  splits. The splits of a tile have ids 8 apart, on one XCD, and each XCD holds a 2 × 3 block of tiles with all
  their splits (`XG` 4). Each workgroup adds its waves' partials through LDS; then each lane stores, per
  fragment, two 16 B slots, each two fp32 partial sums beside this launch's 64-bit tag, `dispatch.id` · H1 +
  `queue.ptr` · H2 with H1, H2 odd (`../gemm/wait_free_fold.md` § How to write it, step 5), with `sc0 sc1`; waits
  `vmcnt(0)`; loads the same slots of the other three splits; and, where all tags match, writes the split-ordered
  sum (§ Folding without a counter there). No workgroup waits; two splits that both write store the same bytes. The
  slots are a `torch.empty` per call.

What each kept round bought in the campaign, at its Points, each round timed beside the one before it
(forge-loop paired bench, 3–4 pairs) *(field, E20)*:
- the port with narrowed tiles: 1.49–2.30× per Point over the starting `torch.matmul` (m 256 1.16×);
- the XCD-blocked order at m 64–256, with 16×2 tiles at m 8: m 256 +17.2%, m 8 +7.6%, m 128 +4.0%, m 64 +2.5%;
- static-layout tensors, so that the inherited kernarg preload covers every operand pointer of the `_build_skw`
  launches (`../flydsl/gemm.md` § small_m §6, its E12): +1.3–1.4% at m 16–64, +0.6% at m 128 and 256, +0.3–0.9% at
  m ≤ 8, against a noise floor of 0.4%.
  The m 256 launch passes `fx.Pointer`s; the preload length in this file's ISA was not checked;
- the split across workgroups at m 256 with the 2 × 3 XCD block: 9.44–9.47 → 8.48–8.50 µs, +11.4% at that Point.
  Without the block the same split read 2.5–4.5% faster than the whole K; on 3 ring slots the split read 8.55 µs with
  2 × 3 tiles per XCD and 11.17 with 1 × 6 *(field, E21)*.

Rows that filled fewer workgroups lost, at these m: the first derivation ran 120 workgroups at m 35, 70, 140 and
154 and read 1.07–1.13 over aiter (1.00 at m 154); rows of 216–240 workgroups read 1.23–1.26 there. m 168–224 on
the whole-K path with 32-row tiles (144–168 workgroups) read 1.08–1.11, and on m 256's split grid 1.19–1.33
*(field, E25)*. m 42 and 84 stay on 144 workgroups (1.14, 1.15): their tile counts give no one-round grid nearer
256 on these tiles.

AKLR validate (measurement profile 19) on one MI355X, `--base` at main 49de226, not CI: 1.142–1.360 over aiter
*(field, E22)*. Checked at every listed Point, each try compared byte for byte with the call alone: the forge driver's `--stress` (streams
limited to 16 and 64 CUs, a second call on the same inputs), one graph replayed 24 times with three input sets
rotated into its buffers, and two calls at once on two streams with different inputs, unrestricted and both on the
same 64 CUs, 20 tries each: none differed *(field, E24)*.

Counter-signals:
- **The rows of m 7–224 come from the campaign's rows and a race of a few candidates**, not from a
  forge-loop round at those m (above); the campaign's measurements are at its own Points.
- **At m 168–224 the split grid computes 256 rows**, so 12–34% of its MFMA work is on rows that read zero; it still
  beat the whole-K tiles there (E25).
- **Faster forms the campaign did not keep or did not race.** At m 128 one K split per XCD (split 4, 32×32 tiles)
  read 6.43 µs against the whole K's 6.92, +7.7% at that Point, below the keep gate, a mean over nine Points
  *(field, E21)*. A K split 8 ways across workgroups with one wave each (N 2048's `_compile_skw_sk`, 16×16 tiles,
  192 workgroups), timed separately, read 4.38–4.44 µs at m 1 and 4.56 at m 8 *(field, E23)*; neither this
  campaign nor the rows of m 7 and 14 raced it. At m 8 a file built on it measured 1.247 over aiter where this
  file's predecessor measured 1.205, each validated on its own *(field, E22)*; the predecessor's ring at m 8 read 4.68 µs
  in its campaign's bench *(field, E21)*.
- **The tags sit in a `torch.empty` workspace**, as N 2048's at m 16–128 (Counter-signals above), and the m 256
  fold is right only while no leftover word equals this launch's tag. Each 16 B slot carries its own tag, so a
  slot seen half written (new tag, old sums) would be wrong; no test here forces that.

To reproduce: `python -m aklr validate aklr/ops/gemm/impls/flydsl/gemm_n384_k7168/d65ffa02dee32841.json --base
origin/main` on a branch without the file (§ Reproduce). To roll back, delete the file: its Points go back to aiter.

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

The report goes to `aklr_validation/report.md`; this run checks the file, the served numbers are CI's
(`aklr/ops/README.md` § Workflow). Add `--base origin/main` only on a branch that edits the file: with the base set,
bench measures only schemas that are not at the base, so on an unchanged checkout it skips the bench
(`aklr/core/validation/stages/validation_scope.py`, `of_working_tree`). Validate's bitwise check runs the kernel on
an otherwise idle GPU, where every split is resident; test the folds also with a grid that is not
(`../gemm/small_m.md` §4). To read a kernel:
`git show <rev>:<schema> | python3 -c 'import json,sys; print(json.load(sys.stdin)["kernel"]["sources"][0]["content"])'`.

## Rollback

Delete the schema file: its Points go unserved, since no other file serves them. `AKLR_DISABLE=gemm@1.0` stops serving
the whole contract, every other bf16 GEMM schema included. A Point CI measures below 1.03 is not served, and a
`failed` accuracy never is. Roll back for a wrong result CI does not catch: NaN or a stale fold after a timed-out or
aborted launch, a silently wrong sum from n1536 when its splits are not resident, a hang of n144, or a change of
FlyDSL or LLVM.

## Evidence

`<schema>.json#kernel.py:<line>` is a line of the `kernel.py` a schema embeds in `kernel.sources`; "local validate"
is the predecessor repository's `validate` (before the rename to AKLR) outside CI on MI355X, the ratio lower bound over aiter per Point.

- E1 [2026-10, gfx950, FlyDSL 0.3.2, N=1, local validate] n144: m 8 1.994, 12 2.000, 16 2.000, 24 2.000, 32 2.000, 40 1.995, 48 2.000, 56 1.965, 60 2.000, 64 1.999, 80 1.871, 96 1.795, 112 1.695, 893 1.420, 2732 1.300, 9397 1.535, 16384 1.271 (gemm_n144_k7168)
- E2 [2026-10, gfx950, FlyDSL 0.3.2, N=1, local validate] n768: m 7 2.000, 14 1.712, 21 1.550, 28 1.573, 35 1.445, 42 1.440, 49 1.336, 56 1.347, 70 1.272, 84 1.619, 98 1.495 (gemm_n768_k7168)
- E3 [2026-10, gfx950, FlyDSL 0.3.2, N=1, local validate] n1280: m 8 2.000, 16 1.589, 24 1.412, 32 1.453, 40 1.277, 48 1.302, 56 1.622, 64 1.614, 80 1.470, 96 1.352, 112 1.387, 893 1.309, 2732 1.188, 9397 0.962, 16384 1.096 (gemm_n1280_k7168)
- E4 [2026-10, gfx950, FlyDSL 0.3.2, N=1, local validate] n1536: m 8 1.150, 12 1.173, 16 1.180, 24 1.238, 32 1.287, 40 1.278, 48 1.313, 56 1.272, 60 1.269, 64 1.284, 80 1.466, 96 1.521, 112 1.326, 893 1.594, 2732 1.173, 9397 0.956, 16384 0.976 (gemm_n1536_k7168)
- E5 [2026-10, gfx950, FlyDSL 0.3.2, N=1, local validate] n2112: m 8 1.059, 12 1.119, 16 1.109, 24 1.582, 32 1.226, 40 1.587, 48 1.346, 56 1.556, 60 1.427, 64 1.446, 80 1.406, 96 1.697, 112 1.547, 893 1.635, 2732 1.044, 9397 1.034, 16384 1.269 (gemm_n2112_k7168)
- E6 [2026-10, gfx950, FlyDSL 0.3.2, N=1, local validate] n3584: m 7 1.306, 14 1.229, 21 1.541, 28 1.535, 35 1.627, 42 1.619, 49 1.531, 56 1.763, 70 2.000, 84 1.486, 98 1.685 (gemm_n3584_k7168)
- E7 [2026-10, gfx950, FlyDSL 0.3.2, N=?, forge-loop paired bench; remeasured rounds 3 pairs] n144: one-kernel fold +62.0, 88.7, 94.0, 77.8, 77.0, 57.4, 60.3, 47.2, 48.1, 49.6, 13.9, 7.3, 5.3% at m 8–112 (recorded); LDS-staged large-m loop +32.4, 48.4, 61.6, 66.2% at m 893–16384 (recorded); XCD remap + hand-off +2.4–25.9% at m ≤ 112, +1.1–11.9% at m ≥ 893 (recorded); spread fold +6.1, 12.3, 17.7, 3.0% at m 893–16384 (recorded); all off +26.2–148.8% (gemm_n144_k7168)
- E8 [2026-10, gfx950, FlyDSL 0.3.2, N=?, campaign driver bench] n768 per Point, aiter → kernel: m 7 14.48 → 6.41 µs, 14 11.33 → 6.70, 21 10.54 → 6.83, 28 10.94 → 6.96, 35 10.47 → 7.23, 42 10.77 → 7.36, 49 10.41 → 7.71, 56 10.71 → 7.86, 70 10.76 → 8.46, 84 14.84 → 9.16, 98 14.75 → 9.70; the ablation file holds per-round means only (gemm_n768_k7168)
- E9 [2026-10, gfx950, FlyDSL 0.3.2, N=?, forge-loop paired bench; remeasured rounds 3 pairs] n1280: in-launch reducers +5.2–33.0% at m 16–112 (recorded); re-tiles +10.0 / +23.0% at 9397 / 16384 (recorded); XCD remap + 8-wave pipe +5.1 / +7.1% at 893 / 2732; owner + tail +5.0% at 2732 and +18.9% at 16384; load order +9.4, 2.7, 2.8, 3.1, 2.1% at m 16–48; 24-bit packing arm failed (gemm_n1280_k7168)
- E10 [2026-10, gfx950, FlyDSL 0.3.2, N=?, forge-loop paired bench; remeasured rounds 3 pairs] n1536: single-launch fold +33.9–84.9% at m ≤ 112 (recorded); XCD-grouped order + tiles +17.6, 19.2, 3.6, 17.3% at m 893–16384 (recorded); 256 flagged workgroups +4.2–15.6% at m ≤ 112 (recorded); 12-wave split tiles +20.7 / +8.2% at 893 / 2732 (gemm_n1536_k7168)
- E11 [2026-10, gfx950, FlyDSL 0.3.2, N=?, forge-loop paired bench; remeasured rounds 3 pairs] n2112: pipelined kernel + fold +39.1–82.2% at m ≤ 112 (recorded); reducer workgroups +3.9, 3.7, 4.7, 4.7, 5.1, 6.1, 6.9, 9.8% at m 40–112 (recorded); split-major order +22.8% at 893; per-XCD tiles +7.3, 18.1, 4.4% at 2732, 9397, 16384 (recorded); larger tiles +8.3 / +10.5% at 9397 / 16384 (gemm_n2112_k7168)
- E12 [2026-10, gfx950, FlyDSL 0.3.2, N=?, forge-loop paired bench; remeasured rounds 3 pairs] n3584: K-split kernel +15.1–76.1% (recorded); reduce-scatter +4.6, 1.4, 2.0, 2.7, 1.5, 2.1, 7.8, 11.9, 11.3% at m 21–98; `nt` B + A first +2.9–6.4% (recorded); columns over waves +2.7–4.5% at m 21–98 (recorded); whole-line planes +0.3–1.4%; all off +23.2–125.2% (gemm_n3584_k7168)
- S1 aklr/ops/gemm/contract.py:95 `_dot_k()` (fp32, `max_split_k` 16) on `GEMM` at :128; `_SLACK`, `_SQRT_LENGTH_MULTIPLE`, `_CONSTANT_MULTIPLE` at aklr/core/contract/semantics/_compare.py:249
- S2 aklr/ops/gemm/impls/flydsl/gemm_n144_k7168/c86923da04869cbc.json#kernel.py:25 the NaN-pattern MAGIC; agent-scope atomic flag store and poll at :70 and :82; `_CONFIG` at :102; `_LARGE` at :679; unbounded polls at :355 and :597; co-residency assert at :189
- S3 aklr/ops/gemm/impls/flydsl/gemm_n768_k7168/28d19101c5b0d2f4.json#kernel.py:22 `_CONFIGS`; splits per XCD at :163; bounded poll at :232, NaN at :250, flag clear at :259; workspace per call at :391
- S4 aklr/ops/gemm/impls/flydsl/gemm_n1280_k7168/572083dd5cfea91f.json#kernel.py:1 docstring; `p24 = ma >= 4` at :1328; operands loaded straight from global at :1364; `_CONFIGS` at :1473; NaN on timeout at :747, :924, :1447
- S5 aklr/ops/gemm/impls/flydsl/gemm_n1536_k7168/0b9e2a207d909b40.json#kernel.py:21 poll bound "only guards against a hang"; `_config` at :48; dispatch tag at :292; (value, tag) pairs at :493; the fold after the poll loop sums without checking at :596
- S6 aklr/ops/gemm/impls/flydsl/gemm_n2112_k7168/2b8c0ff5a4fa7981.json#kernel.py:181 `_count_ready` "this call's tag" over a static `_MAGIC` (:64); bounded polls with NaN at :536, :616, :694; split-major order at :311
- S7 aklr/ops/gemm/impls/flydsl/gemm_n3584_k7168/631331203e1c39ad.json#kernel.py:29 `CONFIGS`; `TAGGED_PLANES` at :46; plane and flag stores without cache bits, polls with `sc1` (:56, :57); `_pack24` at :102; all splits of a tile on one XCD at :228; polls at :528 and :607, NaN at :537; "clear the tags or flags for the next call" at :641
- E13 [2026-11, gfx950, FlyDSL 0.3.2, N=16, AKLR validate at 0e02a81, measurement profile 19: one-sided lower bound of the aiter / kernel time ratio over 16 rounds of CUDA-graph replays with L2 flushed before each call, one MI355X, aiter 0.1.21.dev66+gdbd8bf5bd] 23da2fcbe6d9defd, measured as its predecessor, whose one difference was `torch.cuda.current_stream()` without `a.device`: 1.466, 1.323, 1.305, 1.203, 1.225, 1.071, 1.129, 1.069 at m 1, 2, 4, 8, 16, 32, 64, 128; accuracy passed (gemm_n1024_k7168)
- E14 [2026-11, gfx950, FlyDSL 0.3.2, N=16, the ruler of E13] 1f5a93c79dbd12bf: 1.183, 1.184, 1.199, 1.159, 1.114, 1.180, 1.154, 1.224 at m 1, 2, 4, 8, 16, 32, 64, 128; accuracy passed (gemm_n2048_k7168)
- E15 [2026-11, gfx950, FlyDSL 0.3.2, N=20, eager and op-level CUDA-graph stress on one MI355X, main 0e02a81 with each schema, each try compared byte for byte with an eager call alone] 23da2fcbe6d9defd (its predecessor, as in E13) and 1f5a93c79dbd12bf, every listed Point: the forge driver's conditions (streams limited to 16 and 64 CUs; a second concurrent call on the same inputs) and one captured graph replayed on the same inputs, 20 tries each; one graph replayed 24 times with three AKLR input sets rotated into its buffers; two calls at once on two streams with different inputs, unrestricted and both on the same 64 CUs, 20 tries each: 0 tries differ
- E16 [2026-11, gfx950, FlyDSL 0.3.2, N=?, AKLR `bench_cudagraph` with L2 flushed before every call, the campaign's sweep; readings per configuration not recorded] N 1024 × K 7168: m 1 GEMV 4.48 against the ring (4 waves, 4 columns, 8 slots) 5.02 µs; m 2 two-row GEMV 4.89 against 5.13; m 4 ring 5.13 against the GEMV 7.32
- E17 [2026-11, gfx950, FlyDSL 0.3.2, N=7, AKLR `bench_cudagraph` with L2 flushed before each call, medians of 7 readings] N 1024 × K 7168 ring: m 4 rotated starts (multipliers 1, 3, 5, 9, 13 and two other forms) 5.27–5.34 against 5.246 µs unrotated; m 8 first form 5.86–6.01 against 5.985 unrotated, best rotation (1, 8) 5.856 against the unrotated 4-wave, 4-column, 6-slot ring 5.793
- E18 [2026-11, gfx950, FlyDSL 0.3.2, N=5, AKLR `bench_cudagraph` with L2 flushed before every call, median of 5 readings, one measurement per configuration] N 2048 × K 7168 m 8: split 4 across workgroups 7.71 against the whole K in one workgroup 7.00 µs
- E19 [2026-11, gfx950, FlyDSL 0.3.2 (the pin of baseline aiter dbd8bf5, read from the workflow), N=1, AKLR CI perf runs; measurement profile 19 for #102] Points at or above 1.03 as of #102 (6d7cfec, CI run 42549655045): n144 17 of 17, n768 11 of 11, n1536 15 of 17 (not m 9397, 16384), n2112 15 of 17 (not m 9397, 16384), n3584 11 of 11; n1280 14 of 15 (not m 9397) from d908d01 (#71, CI run 42000899019), which #102 left unchanged (each schema's `workloads` at 6d7cfec)
- S8 aklr/ops/gemm/impls/flydsl/gemm_n1024_k7168/23da2fcbe6d9defd.json#kernel.py:24 `_build_gemv_launcher` (loads issued at :51-56, butterfly at :64-71); `_build_skw` at :102 (`ROT` at :200-202, fold at :271-292); `_SKW_CONFIG` at :304; preload hint and its comment at :313-315; `torch.matmul` fallback at :345
- S9 aklr/ops/gemm/impls/flydsl/gemm_n2048_k7168/1f5a93c79dbd12bf.json#kernel.py:301 `_compile_skw_sk`: XCD numbering at :346-352, plane store `sc0 sc1` at :394, `vmcnt(0)` at :395, tag at :397-405, slot store at :408, `vmcnt(0)` at :409, slot loads at :410-415, `readfirstlane` at :422, fold in split order at :368-381; `_compile_skw_av` at :229; `_CONFIGS` and `_SK_CONFIGS` at :442 and :450; `torch.empty` per call at :467
- E20 [2026-11, gfx950, FlyDSL 0.3.2, N=3–4 pairs, forge-loop paired keep gate on AKLR's graph timer, each kept round timed beside the round before it] gemm_n384_k7168, per Point at m 1, 2, 4, 8, 16, 32, 64, 128, 256: the port of N 1024's file with narrowed tiles, over the starting `torch.matmul`: 2.023, 1.973, 1.921, 1.790, 2.301, 2.220, 1.487, 1.535, 1.161; the XCD-blocked order at m 64–256 and 16×2 tiles at m 8: 1.014, 1.011, 1.001, 1.076, 1.000, 1.000, 1.025, 1.040, 1.172; the kernarg preload: 1.004, 1.009, 1.003, 1.003, 1.014, 1.014, 1.013, 1.006, 1.006; the split across workgroups at m 256: 0.981, 0.991, 1.000, 1.001, 1.001, 1.003, 1.001, 1.000, 1.114
- E21 [2026-11, gfx950, FlyDSL 0.3.2, N=2–3 readings per arm, each a median of 5 calls of AKLR `bench_cudagraph` with L2 flushed before every call, interleaved with the kept kernel] gemm_n384_k7168 m 256: the kept whole-K path (`XG` 2) 9.44–9.47 µs; split 4 across workgroups, 32×64 tiles, 2 ring slots, `XG` 4: 8.48–8.50; with 3 slots, `XG` 4 / 8 (2 × 3 / 1 × 6 tiles per XCD): 8.55 / 11.17; without the XCD block (64×32 tiles, three tries), the split at m 256 paired against the kept kernel: 1.025, 1.045, 1.044, each rejected by the keep gate; m 128: the kept whole-K path 6.92–6.95, split 4 with one K slice per XCD on 32×32 tiles 6.43; m 8: the kept ring 4.684 (driver `--bench-mode`); the keep gate's noise floor 0.0043
- E22 [2026-11, gfx950, FlyDSL 0.3.2, N=16 rounds, AKLR validate measurement profile 19, one MI355X, aiter 0.1.21.dev66+gdbd8bf5bd, each file validated on its own] d65ffa02dee32841 with `--base` main 49de226 at m 7, 14, 21, 28, 35, 42, 49, 56, 70, 84, 98, 112, 126, 140, 154, 168, 182, 196, 210, 224, 256: 1.297, 1.283, 1.285, 1.303, 1.264, 1.142, 1.227, 1.251, 1.237, 1.154, 1.248, 1.260, 1.322, 1.273, 1.236, 1.184, 1.260, 1.225, 1.258, 1.321, 1.360; its predecessor at the campaign's Points (m 1, 2, 4, …, 256, main d7c0a3e): 1.244, 1.250, 1.251, 1.205, 1.269, 1.316, 1.296, 1.327, 1.359; a file built on N 2048's `_compile_skw_sk` at every m, not delivered: 1.245, 1.238, 1.253, 1.247, 1.280, 1.223, 1.224, 1.318, 1.333; all accuracy passed
- E23 [2026-11, gfx950, FlyDSL 0.3.2, N=1–2 readings per configuration, the run's own timing (timer not recorded)] gemm_n384_k7168 with N 2048's builders ported: `_compile_skw_av` (whole K, 16×8 tiles, 48 workgroups), best of four rows, m 1 / 8: 5.55 / 6.09 µs; `_compile_skw_sk`, one wave per workgroup, split 8 on 16×16 tiles (192 workgroups), 14 ring slots: m 1 4.38 and 4.44, m 8 4.56; split 7, 16 slots: 4.41 / 4.50
- S10 aklr/ops/gemm/impls/flydsl/gemm_n384_k7168/d65ffa02dee32841.json#kernel.py:41 `_build_skw`, its XCD-blocked order at :114; `_build_sk` at :308: tile count from the row, `m <= MT * TM`, at :321, XCD block at :363, tag at :481, slot stores `sc0 sc1` at :501, `vmcnt(0)` at :502, the other splits' slots loaded at :509, output store where every tag matched at :545; `_SK_CONFIG` and `_SKW_CONFIG` at :560 and :564; kernarg preload at :585; `torch.empty` per call at :617; `torch.matmul` for any other m at :634
- E24 [2026-11, gfx950, FlyDSL 0.3.2, N=3–24 (the driver's 3 tries per `--stress` condition, 24 graph replays, 20 tries per concurrent condition); each byte-compared with the call alone on AKLR's input sets] d65ffa02dee32841 at all 21 Points: the forge driver's `--stress` passed; one captured graph replayed 24 times with three input sets (outputs differ between sets) rotated into its buffers, 0 differ; two calls at once on two streams with different inputs, unrestricted and both on the same first 64 CUs, 20 tries each, 0 differ
- E25 [2026-11, gfx950, FlyDSL 0.3.2, N=16 rounds, AKLR validate measurement profile 19, local, each file validated on its own, no `--base`] rows of the same kernel file at the 21 Points: 120 workgroups at m 35, 70, 140, 154 (8-row × 16, 16×16, 32×16, 32×16 tiles): 1.069, 1.132, 1.067, 1.002 (0.995 with `--base`); 216–240 workgroups (8-row × 8, 8-row × 16, 16×16, 16×16): 1.262, 1.240, 1.255, 1.227; m 168, 182, 196, 210, 224 with 32×16 whole-K tiles (144–168 workgroups): 1.080, 1.106, 1.095, 1.105, 1.077; on m 256's split grid: 1.191, 1.262, 1.227, 1.257, 1.330
