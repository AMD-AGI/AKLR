---
title: "bf16 GEMM, short K or large M only (Kimi-K3: N 1536 × K 128, N 163840 × K 256, N 2304 × K 1536, N 3072 × K 512) — the whole K in one tile or one workgroup's waves, persistent grids at the largest M; N 7168 and 3584 × K 512 at m 2–128, one of them a copy of aiter's FlyDSL HGEMM"
kind: case
ops: [gemm]
dtypes: [bf16]
regimes: ["M 1-14", "M 2-128", "M 893-16384", "M 35107-342073"]
contract: bitwise
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=1
measured:
  - {metric: "speedup over aiter, local validate lower bound per Point", value: "1.023–1.716; 12 of 17 Points ≥ 1.03 (not m 8, 12, 16, 32, 64); CI serves 7 of 17 as of #102", gpu: MI355X, workload: "gemm_n1536_k128, m 8, 12, 16, 24, 32, 40, 48, 56, 60, 64, 80, 96, 112, 893, 2732, 9397, 16384", run_ref: "local:/workspace/code/gpt_oss_docker/work/sikl/k3_forge/final_r2_local/report.md:67", date: "2026-10-26"}
  - {metric: "speedup over aiter, local validate lower bound per Point", value: "1.527–1.628; 11 of 11 Points ≥ 1.03; CI serves 11 of 11 as of #102", gpu: MI355X, workload: "gemm_n163840_k256, m 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14", run_ref: "e44184aec (commit message table; branch kyle/skill-kernel-loop)", date: "2026-10-25"}
  - {metric: "speedup over aiter, local validate lower bound per Point", value: "1.087–1.322; 4 of 4 Points ≥ 1.03; CI serves 4 of 4 as of CI run 41976547645 (90b9d50), unchanged by #102", gpu: MI355X, workload: "gemm_n2304_k1536, m 893, 2732, 9397, 16384", run_ref: "local:/workspace/code/gpt_oss_docker/work/sikl/k3_forge/final3_local/validate/report.md:39", date: "2026-10-26"}
  - {metric: "speedup over aiter, local validate lower bound per Point", value: "1.113–1.294; 3 of 3 Points ≥ 1.03; CI serves 3 of 3 as of CI run 42000899019 (d908d01, #71), unchanged by #102", gpu: MI355X, workload: "gemm_n3072_k512, m 35107, 104093, 342073", run_ref: "local:/workspace/code/gpt_oss_docker/work/sikl/k3_forge/final3_local/validate/report.md:43", date: "2026-10-26"}
  - {metric: "speedup over aiter, AKLR validate profile 19, per Point; no CI measurement yet", value: "1.092–1.353 at m 2–64; m 1 (1.022) and 128 (0.978) not listed", gpu: MI355X, workload: "gemm_n7168_k512 kernel of file d0d6e9e5c93fbd88, measured on a file with the same kernel and m 1, 2, 4, 8, 16, 32, 64, 128", run_ref: "local:/workspace/code/gpt_oss_docker/work/kb_evidence/v19_revalidation/bf16_small/bf16_small_d8bd1ca29ef0a990/report.json", date: "2026-11-09"}
  - {metric: "speedup over aiter, AKLR validate profile 19, per Point; no CI measurement yet", value: "1.181–1.303 at m 32–128; m 1–16 (1.020–1.036) not listed", gpu: MI355X, workload: "gemm_n3584_k512 kernel of file dacbceacd8d1ae5f, measured on a file that imported the same HGEMM from aiter 0.1.21.dev66+gdbd8bf5bd, with m 1, 2, 4, 8, 16, 32, 64, 128", run_ref: "local:/workspace/code/gpt_oss_docker/work/kb_evidence/v19_revalidation/bf16_small/bf16_small_449a025fc7aa0de6/report.json", date: "2026-11-09"}
sources:
  - aklr/ops/gemm/impls/flydsl/gemm_n1536_k128/f0579ac60c343047.json
  - aklr/ops/gemm/impls/flydsl/gemm_n163840_k256/1220a995dd7d23c0.json
  - aklr/ops/gemm/impls/flydsl/gemm_n2304_k1536/0c3c675002f2fa94.json
  - aklr/ops/gemm/impls/flydsl/gemm_n3072_k512/4c6a228ee0e2d0ed.json
  - aklr/ops/gemm/impls/flydsl/gemm_n7168_k512/d0d6e9e5c93fbd88.json
  - aklr/ops/gemm/impls/flydsl/gemm_n3584_k512/dacbceacd8d1ae5f.json
---

# bf16 GEMM at short K or large M only (Kimi-K3): no K loop to split, a persistent grid where M is large

Four of this page's six FlyDSL `gemm@1.0` files are the Kimi-K3 set (definitions #69, kernels #115, merge
a8fe490b1), one schema file per shape, written by forge-loop campaigns and reviewed. Each schema embeds its
kernel as `kernel.sources[0].content` (`kernel.py`); function names below are in that file. "m" is the Point's M.
None of the four splits K across workgroups.

## Applies when

- Contract `gemm@1.0`, layout variant `row_major` (`b.is_shuffled` false): `a` (m, k) and `b` (n, k) bf16,
  `trans_b`, no bias, bf16 `out`. `compare` is bitwise (`determinism: bitwise`, `max_violation_fraction` 0.0,
  the `CompareSpec` defaults), with the fp32 dot-product error bound of `_dot_k()` (`GEMM` in
  `aklr/ops/gemm/contract.py`). Every output is one fp32 sum in a fixed order.
- Shapes:
  - N 1536 × K 128 (a 384 KiB weight) at m 8–16384: K fits one LDS tile.
  - N 163840 × K 256 (80 MiB) at m 1–14: a read-once weight over a very wide N.
  - N 2304 × K 1536 at m 893–16384 and N 3072 × K 512 at m 35107–342073: only large-M Points, compute-bound
    tiles; the second has 8 K blocks of 64.
- Each schema name has only this file (no file of another language or an older kernel under the same name on
  main), so at every Point the yardstick is aiter.
- **CI serves 25 of their 35 Points** as of #102 (6d7cfec, CI run 42549655045, measurement profile 19), n2304's as
  of CI run 41976547645 (90b9d50) and n3072's as of CI run 42000899019 (d908d01, #71), which #102 left unchanged;
  the count per file is in § Result, and the current value of each Point is the `speedup` in its schema's
  `workloads`. Dispatch serves a workload only from `AKLR_MIN_SPEEDUP` 1.03 (`aklr/ops/README.md` § Environment
  variables; `aklr/core/contract/spec/_workloads.py:38`).

## Counter-signals

- **n1536_k128 at m ≤ 112 sits on the threshold.** Its 13 Points there read 1.023–1.038 in the latest local run,
  5 of them below 1.03 *(field, E1)*; the same code read 1.010–1.038 in the commit's run and 1.021–1.123 in a third
  run's two arms *(field, E2)*. CI's runs move them too: 10 of the 13 served in CI run 42000899019 (#71), 3 (m 12,
  16, 60) as of #102 *(field, E17)*. These Points run a 16-column,
  one-wave tile on 96 workgroups for 256 CUs (code read).
- **n1536_k128: a retune of the three largest m buckets did not pay.** Run beside the delivered kernel in one
  window, it passed the local validate's bar (a lower bound of 1.02) at 14 and 12 of 17 Points against the delivered
  kernel's 14 and 16 in two runs *(field, E11)*; the campaign's ablation file still scores that version as its best.
- **n2304's two-tile walk requires `m % 256 == 0`** (asserted when the persistent kernel is built); m 16384 is the
  only Point that takes it. n3072 sizes its persistent grid from the CU count of the current device at compile time
  and starts half of each XCD's workgroups late only up to m 200000 (`_STAGGER_MAX_M`), so m 342073 runs without
  the delay.
- **Code no Point reaches:** n1536_k128's `_CONFIGS` row 2731 (no Point between 893 and 2732).
- Nothing these kernels compute crosses a workgroup, so there are no flags, counters or workspace: no state lives
  between calls, and no reduction depends on arrival order (code read).

## Mechanism

Shared by the four kernels:

- **MFMA `mfma_f32_16x16x32_bf16` with B as the first operand**, so a lane holds four consecutive output columns
  of one row (`../gemm/epilogue.md` §3), fp32 accumulators.
- **Kernel pointer arguments preloaded into SGPRs** (`amdgpu-kernarg-preload-count` 6 or 16;
  `../gemm/small_m.md` §6).
- **A per-m configuration** compiled in a `functools.cache` function on a Point's first call.

Per file:

**n1536_k128** (`_build_tiled` at every m).
- The whole K = 128 is one LDS tile: one round of buffer→LDS DMAs for A and B (`buffer_load_dwordx4 … lds`,
  XOR-swizzled 16-byte chunks), one `s_waitcnt vmcnt(0)` + barrier, four MFMA K steps; no ring, no K loop
  (`../gemm/datapath.md` §2, §5).
- m ≤ 112: a TILE_M × 16 tile with TILE_M the m bucket (16 … 112), one wave, 96 workgroups, B `nt`.
- m 893: 32×192, 4 waves, grouped XCD order (`GROUP_M` 4), 224 workgroups. m 2732: 64×96, 4 waves, `GROUP_M` 8,
  688 workgroups. m 9397: 128×128, `GROUP_M` 16, 888 workgroups. m 16384: 128×192, `GROUP_M` 8, 1024 workgroups.
- Stores: C staged through LDS and written as 16-byte rows with `sc1`; at m 9397 and 16384 B's rows are fetched
  permuted so a lane's two adjacent 16-column fragments are contiguous output columns, and C goes out as 16-byte
  stores straight from registers (`C_DIRECT`).

**n163840_k256** (`_build_skinny`; m ≤ 16 asserted).
- One 16-row MFMA block (all m rows) by 16 or 32 columns per workgroup; the waves split K and an LDS fold sums
  their partials in wave order:
  - m 1: 4 waves, each a quarter of K, 16 columns per workgroup, 10240 workgroups;
  - m 2, 3: 8 waves, each an eighth of K, 32 columns, 5120 workgroups;
  - m 4–14: 8 waves as 2 column halves × 4 K quarters, 32 columns, 5120 workgroups.
- B goes straight from global into registers, every K step's load issued before the first MFMA; each B element
  feeds one wave, so LDS would add no reuse. A goes to registers directly at m ≤ 3 and through LDS (row stride
  padded by 8 elements) from m 4.
- Adjacent column tiles are dealt to one XCD in groups of 8 (`_grouped_tile`), so a C row's neighbouring tiles
  finish in one L2 (`../gemm/small_m.md` §6); C leaves as 16-byte row stores from m 2.

**n2304_k1536** (`_build`, per-m keyword table).

| m | Tile, waves | Stages, K tile | Workgroups | Also |
|---|---|---|---|---|
| 893 | 64×128, 8 (2×4) | 3, 128 | 252 | grouped XCD order (`GM` 7), fragment prefetch, K loop unrolled, C `sc1` |
| 2732 | 144×192, 12 (3×4) | 3, 64 | 228 | `GM` 7, refills issued inside the compute after the first fragment reads, unrolled, C `nt` |
| 9397 | 448×192, 8 (4×2) | 2, 64 | 252, one round | C staged and stored in two passes, `sc1`; LLVM scheduler `iterative-ilp` |
| 16384 | 256×288, 8 (4×2) | 2, 64 | 256 for 512 tiles | two tiles per workgroup: the second walks K in reverse, starting on the slices the first just left in L2; tile 0's C leaves through stage 1 while stage 0 loads tile 1; scheduler `iterative-minreg`; C `sc1` |

Grids sized to whole rounds of the 256 CUs (`../gemm/grid_pricing.md` §1, §10); the scheduler
strategy goes in through `amdgpu-sched-strategy` (`../gemm/schedule_and_locality.md` §3).

**n3072_k512** (`_build`, one configuration for every m).
- 256×256 tiles, 8 waves (2×4), two LDS stages of K 64, `GROUP_M` 2 grouped order, `waves_per_eu` 2.
- A persistent grid of 256 workgroups (`G`, from the device's CU count) walks head tiles pid, pid + G, … over
  the first P whole 256-row panels, then the remaining rows as 128-row tail items, handed first to the workgroups
  with one head tile fewer (`_plan`):
  - m 35107: P 128 panels, 1536 head tiles (6 per workgroup) and 228 tail items;
  - m 104093: P 405, 4860 head tiles (18 or 19 per workgroup) and 48 tail items;
  - m 342073: P 1337, 16044 tiles (62 or 63 per workgroup), no tail.
- No workgroup barrier between items: each K block's barrier sits mid-block, the block after next is filled behind
  it, and each wave stages its own C rows (32 per pass) through a private LDS area and stores them `nt`.
- At m ≤ 200000, odd workgroups of each XCD sleep before their first fill, so the two halves reach their
  epilogues out of phase (`../gemm/grid_pricing.md` §10).

**What each part bought** (campaign ablations; each kept round timed beside the version before it):

- n1536_k128: the single-pass kernel with per-m tiles and preloaded pointers +0.6–4.4% at m ≤ 112 and
  +22.3–42.4% at m ≥ 893 over the campaign's starting kernel; `sc1` C stores +3.1–5.0% at m ≤ 112 and +8.3–20.3%
  above *(field, E6)*; the register-direct C store +5.9% (16384) and +3.8% (9397) *(field, E7)*.
- n163840_k256: the K-split-over-waves port +18.8–41.3%; kernel-argument preload +1.6–3.0% at every Point; the
  XCD grouping +0.0–1.6%; the two-half workgroup for m ≥ 4 +0.1–2.6% at m 4–14. Against the starting kernel the
  delivered one is +49.7–62.7% *(field, E8)*.
- n2304_k1536: grouping, vector C staging and `nt` C +2.5–8.1%; BK 128 with 3 stages +6.2% at m 893; the unrolled
  K loop +1.2% / +3.2% at m 893 / 2732; the scheduler strategy +2.3% / +0.7% at m 9397 / 16384; the persistent
  two-tile walk +0.9% at m 16384. Against the starting kernel: +9.5–23.2% *(field, E9)*.
- n3072_k512: the 256×256 port with grouped order and `nt` C +6.4–16.7%; the persistent loop with per-wave C
  +5.3–10.0%; tail items +2.3% at m 35107; the start delay +2.9% / +1.1% at m 35107 / 104093; the mid-block barrier
  +0.6–2.2%. Against the starting kernel: +13.8–31.5% *(field, E10)*.

## Result

"Served" counts the Points at or above 1.03 in CI as of #102 (6d7cfec, CI run 42549655045, measurement profile 19),
n2304 as of CI run 41976547645 (90b9d50) and n3072 as of CI run 42000899019 (d908d01, #71), which #102 left unchanged;
the `speedup` in each schema's `workloads` is the current value *(field, E17)*. CI runs the FlyDSL of its image
(0.3.4.1 in the image of the latest perf runs), not the 0.3.2 of the local numbers below
(`../pitfalls/measured_a_different_build.md`).

The other columns are the campaigns' own `python -m sikl validate` runs on MI355X (the repository
before its rename to AKLR), FlyDSL 0.3.2: a lower bound of the CUDA-graph time ratio aiter / kernel, one run each;
"≥ 1.03" counts them against the dispatch threshold (that validate itself demoted below 1.02).

| File | Points | Lower bound per Point | ≥ 1.03 locally | Served |
|---|---|---|---|---|
| n1536_k128 *(field, E1)* | m 8–112 | 1.023–1.038 | 8 of 13 | 3 of 13 |
| | m 893, 2732, 9397, 16384 | 1.243, 1.716, 1.432, 1.435 | 4 of 4 | 4 of 4 |
| n163840_k256 *(field, E3)* | m 1–14 | 1.527–1.628 | 11 of 11 | 11 of 11 |
| n2304_k1536 *(field, E4)* | m 893, 2732, 9397, 16384 | 1.322, 1.165, 1.153, 1.087 | 4 of 4 | 4 of 4 |
| n3072_k512 *(field, E5)* | m 35107, 104093, 342073 | 1.294, 1.248, 1.113 | 3 of 3 | 3 of 3 |

## N 7168 and 3584 × K 512 at m 2–128: all of K in one workgroup, at decode sizes

Two later files at small m, one per schema name: `gemm_n7168_k512/d0d6e9e5c93fbd88.json` (m 2–64, 7 MiB of weight)
and `gemm_n3584_k512/dacbceacd8d1ae5f.json` (m 32–128, 3.5 MiB). No other file serves these Points, so the yardstick
is aiter; until CI writes their `speedup`, dispatch does not serve them (`qualifies()` in
`aklr/core/contract/spec/_workloads.py`; `AKLR_MIN_SPEEDUP`, default 1.03, `aklr/ops/README.md` § Environment
variables). Like the files above, these two do not split K across workgroups; unlike n163840's, they do not split it
across waves either: each workgroup reads its rows of A and B over all of K once, and every output is one MFMA chain
in K order. N 7168 at m 1 (1.022) and m 128 (0.978) and N 3584 at
m ≤ 16 (1.020–1.036) are not listed: within about 4% of aiter or slower *(field, E12, E13)*.

**N 7168 × K 512** (`d0d6e9e5…`, `_build_whole`), one launch:
- Each workgroup DMAs all of K for its rows of A and B into LDS before computing: 8 chunks of 64, each chunk one group
  of `buffer_load_dwordx4 … lds` (64, 80 or 96 KiB of LDS). Chunk c then waits (`s_waitcnt vmcnt`, `s_barrier`)
  only until its own DMAs have landed, and its MFMAs run while the later chunks land (`../gemm/datapath.md` §5).
- LDS rows are XOR-swizzled in 16 B slots with one key per row pair (`(row / 2) mod 8`), so that one
  `ds_read_b128` pass covers every bank (`../gemm/datapath.md` §2).
- The waves split M × N, not K: 16×64 tiles with two waves along N (m ≤ 16, 112 workgroups), 32×32 with 2 × 2 waves
  (m 32, 224), 32×64 with two waves along N and two M-blocks (m 64, 224).
- `mfma_f32_16x16x32_bf16` with B as the first operand, so a lane holds 4 adjacent output columns. At m ≤ 16 and 64,
  B's LDS rows are filled from permuted columns so that a lane's two N fragments hold 8 adjacent columns, stored as
  one 16 B write; at m 32, 8 B writes (`../gemm/epilogue.md` §3). C goes out with `sc1`; rows past m fall outside C's
  descriptor and are dropped, and A rows past m read row 0.

**N 3584 × K 512** (`dacbceac…`): aiter's FlyDSL HGEMM at split-K 1, from copies of aiter's FlyDSL files at
dbd8bf5bd that the schema carries, with a 39-line `kernel.py` that chooses the tile.
- `_launch_for(m, n, k)` (a `functools.cache` compile function) calls `compile_hgemm_kernel("bf16", n, k, …)` with
  the tile of `_config(m)`, and `launch` runs it through `_run_compiled` on the current stream. It passes a null
  pointer in the bias, semaphore and signal slots: the kernel binds the semaphore and signal only with split-K > 1
  and the bias only with `HAS_BIAS`, and `_config` sets neither (S7, S8).
- The copies are `splitk_hgemm.py`, `tensor_shim.py` (only the definitions the HGEMM and the launch use),
  `buffer_ops.py` and `vector.py`; they import one another relatively and nothing from aiter. Beyond the imports,
  two host-side lines differ from aiter, as they do in the SiTUv2 MoE schema's copies: `buffer_ops.py` does not
  read `FLYDSL_GPU_ARCH`, and `tensor_shim.py` builds its default strides without `tolist` (S9).
- Tiles: 32×32 with K tiles of 256 and two waves along N (m 32 and 64, 112 and 224 workgroups); 32×64 with K tiles of
  128, four waves along N, three stages and B through LDS as well (m 128, 224 workgroups). Below m 128 only A goes
  through LDS. Everything inside the kernel is aiter's code at dbd8bf5bd; its knobs are in
  `../flydsl/kernels/flydsl_knob_space.md`.

Counter-signals:
- **N 3584 keeps aiter's HGEMM as of dbd8bf5bd.** It was measured importing that code from aiter
  0.1.21.dev66+gdbd8bf5bd, the commit CI pins as `AKLR_BASELINE_AITER` (`.github/workflows/ci.yaml`); the schema now
  carries the copies, so a later aiter changes neither its import nor its kernel, and a later fix to aiter's HGEMM
  reaches it only through a new copy.
- **The baseline is aiter's own dispatch.** Its tuned `gemm_a16w16` (`gemm_baseline`) picks its own kernel at these
  shapes; which one it ran was not recorded, so the gain is this tile choice against whatever aiter's dispatch chose.
- **Near 4.2–4.4 µs the kernel time stops moving with m.** N 7168 read 4.22–4.30 µs per round at every listed m from
  2 to 64 while aiter went from 4.64 to 5.77, 0.41–0.58 µs above the kernel at m 2–16; N 3584 read 4.16–4.32 at
  m 1–16 against aiter's 4.32–4.46, about a tenth of a microsecond apart *(field, E14, E15)*. The gain is aiter's time
  growing with m. Whether profile 19 has a floor near there was not measured (`../pitfalls/aklr_dispatch_floor.md`).
- **No K split at K 512.** Splitting K = 512 to fill the grid cost 15–18% on another kernel (`../gemm/small_m.md`
  §4), so at m ≤ 16 N 7168 runs 112 workgroups, under half a round.

AKLR validate (measurement profile 19) on one MI355X at main 0e02a81, not CI: N 7168 × K 512 1.092–1.353 over aiter,
highest at m 64 *(field, E12)*; N 3584 × K 512 1.181–1.303, highest at m 64, measured on the file that imported the
HGEMM from aiter *(field, E13)*. Under stress every listed Point gave the output of the call alone *(field, E16)*.
Reproduce on an MI355X with aiter at `AKLR_BASELINE_AITER` installed, with `python -m aklr validate` on
`aklr/ops/gemm/impls/flydsl/gemm_n7168_k512/d0d6e9e5c93fbd88.json` and
`aklr/ops/gemm/impls/flydsl/gemm_n3584_k512/dacbceacd8d1ae5f.json` (§ Reproduce). To roll back, delete the file: its
Points are then served by no file, and the caller runs its own path.

## Reproduce

On an MI355X with the repository's FlyDSL 0.3.2, from the repository root, one schema at a time:

```
python -m aklr validate aklr/ops/gemm/impls/flydsl/gemm_n1536_k128/f0579ac60c343047.json
python -m aklr validate aklr/ops/gemm/impls/flydsl/gemm_n163840_k256/1220a995dd7d23c0.json
python -m aklr validate aklr/ops/gemm/impls/flydsl/gemm_n2304_k1536/0c3c675002f2fa94.json
python -m aklr validate aklr/ops/gemm/impls/flydsl/gemm_n3072_k512/4c6a228ee0e2d0ed.json
```

The report goes to `aklr_validation/report.md`; a local run checks the file, the served numbers are CI's
(`aklr/ops/README.md` § Workflow). Add `--base origin/main` only on a branch that edits the file: with the base set,
bench measures only schemas that are not at the base, so on an unchanged checkout it skips the bench
(`aklr/core/validation/stages/validation_scope.py`, `of_working_tree`). n3072's m 342073 compares a 2 GB output
against the reference; validate scales the trial's time limit with the output size.
To read a kernel:
`git show <rev>:<schema> | python3 -c 'import json,sys; print(json.load(sys.stdin)["kernel"]["sources"][0]["content"])'`.

## Rollback

Delete the schema file: its Points go unserved, since no other file serves them. `AKLR_DISABLE=gemm@1.0` stops serving
the whole contract, every other bf16 GEMM schema included. A Point CI measures below 1.03 is not served, and a
`failed` accuracy never is. Roll back for a wrong result CI does not catch, for example after a FlyDSL or LLVM
upgrade.

## Evidence

`<schema>.json#kernel.py:<line>` is a line of the `kernel.py` a schema embeds in `kernel.sources`; "local validate"
is `python -m sikl validate` on MI355X in the campaign containers, the ratio lower bound over aiter per Point.

- E1 [2026-10, gfx950, FlyDSL 0.3.2, N=1, local validate] n1536_k128: m 8 1.025, 12 1.023, 16 1.024, 24 1.031, 32 1.023, 40 1.032, 48 1.030, 56 1.038, 60 1.037, 64 1.024, 80 1.033, 96 1.033, 112 1.032, 893 1.243, 2732 1.716, 9397 1.432, 16384 1.435 (gemm_n1536_k128; local:/workspace/code/gpt_oss_docker/work/sikl/k3_forge/final_r2_local/report.md:67)
- E2 [2026-10, gfx950, FlyDSL 0.3.2, N=1, local validate] n1536_k128 at m ≤ 112: 1.010–1.038 in the commit's run of the first delivered version; 1.048–1.123 and 1.021–1.103 for that version (602fb113) and the large-m group's module (437aa0ab) side by side in another run; the delivery check compared the final ISA at m ≤ 893 (gemm_n1536_k128; commit message of fb552046c; local:/workspace/code/gpt_oss_docker/work/sikl/k3_forge/final_r2ab_local/report.md:118; commit message of 7b091112d)
- E3 [2026-10, gfx950, FlyDSL 0.3.2, N=1, local validate] n163840_k256: m 1 1.628, 2 1.588, 3 1.591, 4 1.573, 5 1.575, 6 1.573, 7 1.569, 8 1.566, 10 1.562, 12 1.551, 14 1.527 (gemm_n163840_k256; commit message of e44184aec)
- E4 [2026-10, gfx950, FlyDSL 0.3.2, N=1, local validate] n2304_k1536: m 893 1.322, 2732 1.165, 9397 1.153, 16384 1.087 (gemm_n2304_k1536; local:/workspace/code/gpt_oss_docker/work/sikl/k3_forge/final3_local/validate/report.md:39)
- E5 [2026-10, gfx950, FlyDSL 0.3.2, N=1, local validate] n3072_k512: m 35107 1.294, 104093 1.248, 342073 1.113 (gemm_n3072_k512; local:/workspace/code/gpt_oss_docker/work/sikl/k3_forge/final3_local/validate/report.md:43)
- E6 [2026-10, gfx950, FlyDSL 0.3.2, N=?, forge-loop paired bench, recorded rounds] n1536_k128: single-pass kernel +2.0, 0.6, 1.2, 1.8, 1.6, 2.1, 2.1, 4.4, 3.5, 2.7, 2.8, 2.9, 3.5% at m 8–112, +22.3, 42.4, 26.4, 25.4% at 893, 2732, 9397, 16384; `sc1` C +3.1–5.0% at m ≤ 112, +10.0, 20.3, 8.7, 8.3% above (gemm_n1536_k128; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n1536_k128/forge_experiments/ablation.json)
- E7 [2026-10, gfx950, FlyDSL 0.3.2, N=3, forge-loop paired bench, group campaign] n1536_k128 C from registers with permuted B rows: +5.9% at m 16384, +3.8% at 9397, +1.3% at 2732 (r2_n7168; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/r2_n7168/forge_experiments/ablation.json)
- E8 [2026-10, gfx950, FlyDSL 0.3.2, N=?, forge-loop paired bench; remeasured rounds 3 pairs] n163840_k256: port +18.8–41.3% (recorded); preload +1.6–3.0% (recorded); XCD grouping +0.0–1.6%; two-half workgroup +0.1–2.6% at m 4–14; all changes off +49.7–62.7% (gemm_n163840_k256; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n163840_k256/forge_experiments/ablation.json)
- E9 [2026-10, gfx950, FlyDSL 0.3.2, N=?, forge-loop paired bench; remeasured rounds 3 pairs] n2304_k1536: grouping + vector C staging + `nt` C +7.9, 8.1, 2.5, 2.6% (recorded); BK 128 +6.2% at 893; unroll +1.2 / +3.2% at 893 / 2732; scheduler strategy +2.3 / +0.7% at 9397 / 16384; persistent +0.9% at 16384; all changes off +23.2, 15.8, 15.6, 9.5% at m 893, 2732, 9397, 16384 (gemm_n2304_k1536; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n2304_k1536/forge_experiments/ablation.json; local:/workspace/code/gpt_oss_docker/work/sikl/k3_forge/final3_docs/gemm_n2304_k1536/ablation.md)
- E10 [2026-10, gfx950, FlyDSL 0.3.2, N=?, forge-loop paired bench; remeasured rounds 3 pairs] n3072_k512 at m 35107 / 104093 / 342073: port +9.2 / 16.7 / 6.4% (recorded); persistent loop +10.0 / 7.3 / 5.3% (recorded); tail items +2.3 / 0.1 / 0.0%; start delay +2.9 / 1.1 / 0.0%; mid-block barrier +0.6 / 2.2 / 1.1%; all changes off +28.2 / 31.5 / 13.8% (gemm_n3072_k512; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n3072_k512/forge_experiments/ablation.json; local:/workspace/code/gpt_oss_docker/work/sikl/k3_forge/final3_docs/gemm_n3072_k512/ablation.md)
- E11 [2026-10, gfx950, FlyDSL 0.3.2, N=2, local validate, one window per run] n1536_k128 delivered beside the dropped last round: 14 and 16 of 17 Points past the local bar against 14 and 12 (gemm_n1536_k128; commit message of fb552046c)
- S1 aklr/ops/gemm/contract.py:128 `GEMM` with `_UNSHUFFLED_VARIANTS` (:124) and `_dot_k()` (:95); `CompareSpec` defaults bitwise at aklr/core/contract/spec/_args.py:217
- S2 aklr/ops/gemm/impls/flydsl/gemm_n1536_k128/f0579ac60c343047.json#kernel.py:26 `_CONFIGS`; whole-K DMA and one wait at :260; `C_DIRECT` stores at :281; `sc1` stores at :20 and :58
- S3 aklr/ops/gemm/impls/flydsl/gemm_n163840_k256/1220a995dd7d23c0.json#kernel.py:20 `_CONFIGS`; `_grouped_tile` at :47; B loads first at :134; B-first MFMA at :154; wave-order fold at :182 and :197
- S4 aklr/ops/gemm/impls/flydsl/gemm_n2304_k1536/0c3c675002f2fa94.json#kernel.py:35 `_CONFIGS`; scheduler strategy at :114; two-tile assertion at :173; second tile in reverse K at :521
- S5 aklr/ops/gemm/impls/flydsl/gemm_n3072_k512/4c6a228ee0e2d0ed.json#kernel.py:49 `_plan`; grid from the CU count at :110; start delay at :476; tail items at :492; `nt` C at :454
- E12 [2026-11, gfx950, FlyDSL 0.3.2, N=16, AKLR validate at 0e02a81, measurement profile 19: one-sided lower bound of the aiter / kernel time ratio over 16 rounds of CUDA-graph replays with L2 flushed before each call, one MI355X, aiter 0.1.21.dev66+gdbd8bf5bd] d8bd1ca29ef0a990 (the kernel of d0d6e9e5c93fbd88, also listing m 1 and 128): 1.022, 1.100, 1.092, 1.108, 1.132, 1.239, 1.353, 0.978 at m 1, 2, 4, 8, 16, 32, 64, 128; m 128 demoted `not_faster`; accuracy passed (local:/workspace/code/gpt_oss_docker/work/kb_evidence/v19_revalidation/bf16_small/bf16_small_d8bd1ca29ef0a990/report.json)
- E13 [2026-11, gfx950, FlyDSL 0.3.2, N=16, the ruler of E12; 32 rounds at m 2] 449a025fc7aa0de6 (also listing m 1–16; its `kernel.py` imported the HGEMM from that aiter and passed B's pointer in the bias, semaphore and signal slots, where dacbceacd8d1ae5f carries copies of aiter's files, S9, and passes a null pointer): 1.036, 1.020, 1.026, 1.030, 1.031, 1.181, 1.303, 1.286 at m 1, 2, 4, 8, 16, 32, 64, 128; m 2 demoted `not_faster`; accuracy passed (local:/workspace/code/gpt_oss_docker/work/kb_evidence/v19_revalidation/bf16_small/bf16_small_449a025fc7aa0de6/report.json)
- E14 [2026-11, gfx950, FlyDSL 0.3.2, N=16, the run of E12] per-round times, kernel against aiter: m 1 4.22–4.28 against 4.32–4.37 µs; m 2 4.22–4.25 against 4.64–4.70, m 4 4.25–4.30 against 4.66–4.72, m 8 4.26–4.30 against 4.72–4.78, m 16 4.25–4.28 against 4.80–4.86, m 32 4.23–4.26 against 5.25–5.31, m 64 4.23–4.26 against 5.73–5.77; m 128 4.86–4.92 against 4.78–4.84 (`candidate_milliseconds`, `versus_milliseconds`; local:/workspace/code/gpt_oss_docker/work/kb_evidence/v19_revalidation/bf16_small/bf16_small_d8bd1ca29ef0a990/report.json)
- E15 [2026-11, gfx950, FlyDSL 0.3.2, N=16, the run of E13] per-round times, kernel against aiter: m 1–16 4.16–4.32 against 4.32–4.46 µs; m 32–128 4.29–4.40 against 5.08–5.67 (local:/workspace/code/gpt_oss_docker/work/kb_evidence/v19_revalidation/bf16_small/bf16_small_449a025fc7aa0de6/report.json)
- E16 [2026-11, gfx950, FlyDSL 0.3.2, N=20, eager and op-level CUDA-graph stress on one MI355X, main 0e02a81 with each schema, each try compared byte for byte with an eager call alone] d0d6e9e5c93fbd88 and the predecessor of dacbceacd8d1ae5f (the kernel of E13, listing m 32–128), every listed Point: the forge driver's conditions (streams limited to 16 and 64 CUs; a second concurrent call on the same inputs) and one captured graph replayed on the same inputs, 20 tries each; one graph replayed 24 times with three AKLR input sets rotated into its buffers; two calls at once on two streams with different inputs, unrestricted and both on the same 64 CUs, 20 tries each: 0 tries differ (local:/workspace/code/gpt_oss_docker/work/kb_evidence/v19_final_schemas/stress/README.txt; local:/workspace/code/gpt_oss_docker/work/kb_evidence/v19_final_schemas/stress/results/g1/bf16_small_d0d6e9e5c93fbd88/stress.log:25; local:/workspace/code/gpt_oss_docker/work/kb_evidence/v19_final_schemas/stress/graph/results/g1/bf16_small_d0d6e9e5c93fbd88/graph.log:7; local:/workspace/code/gpt_oss_docker/work/kb_evidence/v19_final_schemas/stress/concurrent/results/g1/bf16_small_d0d6e9e5c93fbd88/concurrent.log:13; the predecessor's task id at local:/workspace/code/gpt_oss_docker/work/kb_evidence/v19_final_schemas/stress/tasks.json:129, its logs results/g1/<id>/stress.log:14, graph/results/g4/<id>/graph.log:5 and concurrent/results/g3/<id>/concurrent.log:8 in that directory)
- E17 [2026-11, gfx950, FlyDSL 0.3.2 (the pin of baseline aiter dbd8bf5, read from the workflow), N=1, AKLR CI perf runs; measurement profile 19 for #102] Points at or above 1.03 as of #102 (6d7cfec, CI run 42549655045): n1536_k128 7 of 17 (m 12, 16, 60, 893–16384; m 8–112 0.964–1.034), n163840_k256 11 of 11; n2304_k1536 4 of 4 from 90b9d50 (CI run 41976547645) and n3072_k512 3 of 3 from d908d01 (#71, CI run 42000899019), which #102 left unchanged; n1536_k128 in d908d01 (#71): 14 of 17, m 8–112 1.018–1.037, not m 24, 48, 80 (each schema's `workloads` at 6d7cfec and 0e02a81)
- S6 aklr/ops/gemm/impls/flydsl/gemm_n7168_k512/d0d6e9e5c93fbd88.json#kernel.py:23 `_CONFIGS` (tile, waves, paired 16 B stores) per m; `_C_AUX` `sc1` at :31; every chunk's DMAs issued at :139-144; per-chunk wait at :179-181; permuted B rows at :126-130; B first in the MFMA at :191-193
- S7 aklr/ops/gemm/impls/flydsl/gemm_n3584_k512/dacbceacd8d1ae5f.json#kernel.py:13 relative imports of `compile_hgemm_kernel` and (:14) `_run_compiled`, `ptr_arg`; `_config` at :17-22; a null pointer in the bias, semaphore and signal slots at :31-33
- S8 aklr/ops/gemm/impls/flydsl/gemm_n3584_k512/dacbceacd8d1ae5f.json#splitk_hgemm.py:1093 `launch_hgemm_kernel(C, A, B, BIAS, m, semaphore, signal, stream)`; semaphore and signal bound only under `IS_SPLIT_K` at :341, the bias only under `HAS_BIAS` at :262 and :1082; without `B_TO_LDS` only A has LDS storage at :196-208; the copy of ROCm/aiter@dbd8bf5bd:aiter/ops/flydsl/kernels/splitk_hgemm.py (:1094, :342, :263, :1083, :197-209 there) with its aiter import made relative at :15
- S9 aklr/ops/gemm/impls/flydsl/gemm_n3584_k512/dacbceacd8d1ae5f.json#buffer_ops.py:39 `_get_buffer_flags` with no `FLYDSL_GPU_ARCH` read (:65); its tensor_shim.py:76 default strides without `tolist`; the same lines in aklr/ops/fused_moe/impls/flydsl/fused_moe_situv2_act_quant8_gate_up_interleaved_model_dim3584_n_experts896_topk16_inter_dim384/033eff70aa1f4ee2.json#buffer_ops.py:39 and its tensor_shim.py:175
