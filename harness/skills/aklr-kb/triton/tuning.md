---
title: "Tuning a Triton or Gluon kernel for AKLR's Points on gfx950"
kind: method
ops: [gemm, moe, attention, gdn]
gens: [gfx950]
triton: 3.8.0
---

# Tuning a Triton or Gluon kernel for AKLR's Points

How to make a correct Triton or Gluon kernel faster at the Points of its AKLR schema on gfx950 (MI355X): what AKLR's
number means and how far it moves on its own, how to A/B one change, which knobs moved the ratio on AKLR's committed
kernels and which did not, and how those kernels set their configuration per Point. Every speedup here is per Point;
reading the compiled kernel is `debugging.md`, the mechanics of loads, LDS and pipelines `gluon/memory.md`.

## Route here when

- A kernel passes `validate` for its schema and you want it faster at its Points.
- You must decide whether a change helped, or whether a gap between two stored speedups means anything.
- You are choosing a split count, a prefetch depth, a grid, `num_warps`, `waves_per_eu`, a launch option or a cache
  hint, or copying one from a committed kernel.
- Elsewhere: writing the kernel (`authoring.md`), the closest committed kernel and each family's compiled resources
  (`kernels/kernel_library.md`), what the contract allows numerically (`numerics.md`).

## 1. The ruler: what a stored speedup is and how far it moves

Each measured workload stores `speedup`, written by CI: per Point, a rounded one-sided lower bound on the geometric mean of paired-round time ratios
against the schema's baseline (S1, S2; a workload with no timed baseline stores 1.0). Every call is replayed in a CUDA graph behind a 1 GiB flush (S1) that reads, not
writes, its buffer (vector loads, one store per program to a small sums tensor; S3) *(field, E34)*. A call's time is the sum of its kernels' device durations, flush excluded, from AKLR's rocprofiler-sdk collector since #33 (Kineto before; S4). Before #33 that sum came from Kineto and read 4–10% above an independent Kineto sum of the same kernels replayed without the flush *(field, E35)*. Round counts, thresholds and the clamp:
`harness/skills/aklr-read-validation/references/ruler.md` ("the ruler page"). For tuning:

- **A bound, not a mean.** At the same round count, of two records with the same mean log ratio, the larger sample standard deviation of log ratios lowers the unrounded bound
  (S2): AKLR's `TimingRecord` stores 1.442 and 1.412 for 16 rounds at exp(mean log ratio) = 1.45 with sample standard deviations of the per-round log ratios of 0.01 and 0.05.
- **Compare speedup at the same Point.** At 90b9d50 the bf16 GEMM `f806ad2b086d757d` stores lower bounds 1.116, 1.129 and 1.292 at m = 2, 4 and 8 *(field, E1)*, while one compiled kernel serves all three: m is a runtime argument and the constexprs are the same (S5, S6). One binary does not mean equal kernel time, and the stored values do not say which side moved. For a single timing round at Point p, define the raw time ratio R_p = baseline_time_p / candidate_time_p. Across Points q and p, R_q/R_p = (baseline_time_q/baseline_time_p) × (candidate_time_p/candidate_time_q); a stored speedup adds a Point-specific confidence-bound adjustment and rounding. Compare changes at the same Point and report both times.
- **Equal axes are not equal workloads.** One GDN prefill source stored 1.444 and 1.370 in one CI run on two scenarios
  of 16384 tokens in 2 sequences, split by `max_seq_len` as 10000 + 6384 and 8192 + 8192 (S48) *(field, E37)*. Compare
  within one scenario's Point.
- **The measured trivial-launch slope under AKLR is about 4 µs** (under Kineto, before #33): 4.04–4.07 µs per added trivial kernel with `GPU_FORCE_QUEUE_PROFILING=1`, as the runner set it then, and 4.05–4.08 without it, as the runner runs since #33 (S63); torch kernels alike *(field, E35)*; FlyDSL kernels:
  [../pitfalls/aklr_dispatch_floor.md](../pitfalls/aklr_dispatch_floor.md). A call's empirical launch-count budget is its launch count times
  that, the baseline's likewise: in 3 runs add + RMSNorm (hidden 6144) read 4.07–4.24 µs and its baseline 4.19–4.72 (4
  to 256 tokens), the m = 16 fp8 GEMM (1 launch) 4.21–4.33, the 2-token Qwen3.5 GDN decode 8.55–8.87 and the 8-token
  Qwen2 router 8.33–8.51 (2 launches each, S63): 0.14–0.35 µs per launch above the measured slope in the run means, an estimated headroom at fixed launch count
  rather than a proven upper bound on an optimization. Where both sides sit on it, launch counts set the ratio (one Point tested): the GDN decode's baseline has five
  launches (S63), 21.20–21.78 µs, speedup 2.47 against 5/2 *(field, E36)*. Read both sides' times in `report.json`.

How far the number moves with no change to the kernel:

| Repeated | Spread |
|---|---|
| One schema validated 3 times in a row on one idle MI355X (`exp` of `log_ratio_mean`) | 2.4% fp8 GEMM N=3072 m=16 (`f3970decacc97225`; 2.8% in a second set of three); 0.9% Qwen2 router, 256 tokens (`c3889614985b700a`); 1.7% Qwen3.5 GDN decode, 2 tokens (`299133886d479017`); 0.9% DeepSeek-V2 MoE inter_dim 256, 2 tokens (`d3bbbce5c0bbc61e`) *(field, E2)* |
| The 4 runs of one arm, kernel unchanged | 0.07% (fp8 GEMM m=4096) to 2.6% (fp8 GEMM m=16) over 11 sessions; the Qwen2 router at 8 tokens (`4801e216dcff290f`) 1.7% and 2.2% (1.200 to 1.226); the m = 16 GEMM over four sessions in one job: means 1.451 to 1.472 *(field, E3)* |
| The same sources at the same Point in two CI runs | median 2.2%, p90 6.3%; 17.5% of pairs more than 5% apart *(field, E4)* |
| One re-measurement of each of the 154 schemas at 43e112e (its lowest Point) against its stored value | plugin schemas: 84 of 106 within ±3%, 94 within ±5%, family medians 0.991 to 1.06 (Qwen2-MoE fp8 router + experts) *(field, E5)* |

- Compare a candidate only with A runs interleaved with it (§2). A gap of a few percent between stored speedups of
  different CI runs or files cannot be read as a property of the kernels.
- **Leave a margin over the serving threshold** (`AKLR_MIN_SPEEDUP`, `aklr/ops/README.md` § Environment variables).
  Over the nine CI measurement runs on main up to 43e112e, 9 Points stored values on both sides of it, and 3 Points stored exactly 1.030 at 43e112e. The median
  gap between two runs of a Point is 0.9% for the plain-Triton GEMMs and RMSNorm, 2.0 to 2.7% for GDN, DeepSeek-V2 and
  MiniMax-M2, 3.1% for the fp8 GEMM and 3.7% for Qwen2-MoE; the fp8 GEMM's 12 Points at m ≤ 64 with two or more runs
  moved 4.0 to 11.8% *(field, E4)*, about the whole spread of its m ≤ 64 Points in §14.
- **The baseline's build is part of the ruler.** A baseline that reaches aiter, the `gemm` op's directly (S7), a
  plugin's through SGLang's path (S8), can move with the aiter build: the bf16 n6144 GEMM's baseline moved 13.3% at m=8
  from the image's aiter to CI's *(field, E46)*. CI installs the aiter commit `AKLR_BASELINE_AITER` over the image's
  (S9); install it before comparing a local ratio with CI's (`.github/README.md` § Install). A re-run of the 154 schemas at 43e112e (its aiter build not recorded) read the aiter-baselined schemas at 0.985 to 1.11 of CI (fp8 GEMM, median 1.005) and 0.997 to 1.03
  (bf16 GEMM, RMSNorm), lowest Point *(field, E5)*. With identical baseline times the baseline cancels algebraically; in the measured interleaved A/Bs, B/A of the ratios matched A/B
  of the call times within 0.2% over 8 pairs *(field, E3)*.
- **The yardstick.** Among files whose workloads qualify (including accuracy), dispatch serves at a fixed-size Point the file with the highest stored speedup that reaches
  `AKLR_MIN_SPEEDUP` (default 1.03; a 1.02 file alone was `not_qualified`), so a kernel counts per Point against the
  faster of the kernel main serves there and the baseline (`aklr/ops/README.md`: files sharing a name, § Environment
  variables; `../CONTRIBUTING.md` § Evidence); §2 step 3 times both in the same rounds.
- **A benchmark of your own is not a substitute for `validate`.** In one comparison of 7 A/B pairs (router split-K, fp8 GEMM prefetch / tile order, MoE `.cg`), hot eager, flushed eager and graph replay without the flush missed its recorded B/A by >1% on 6, 4 and 5 pairs. AKLR's `bench_cudagraph` came within 0.9% on 6, but read 0.930 versus `validate`'s 0.871–0.905 for m = 16 `PREFETCH` 5 → 3: A read 4.49 µs versus 4.23–4.33 *(field, E47)* ([debugging.md](debugging.md) §11).
  Settle 1–2% with `validate` (the ruler page, "Sweeps are not the gate"). It times each Point on one fixed input set; another fixed set moved routed kernels' B/A by 0.04–1.5% (one run). At k = 2048, m = 64 a source tier lost to the re-tuned tier both flushed (1.05–1.08× time) and hot (1.08–1.13×, 5 runs); hot and flushed sweeps picked the same fastest tested configuration at m = 16 and 64 (one run), so a hot cache did not explain that loss (`gemm/decode_gemm.md` §8) *(field, E40, E47)*.

## 2. How to A/B one change

1. **One Point, one knob.** Change the knob in the branch the Point takes (files keep tables for other sizes, §14),
   with the constants the branch derives from it (scratch shapes, grid, flags): §5 shows what a lone change can do.
2. **Write the copy beside the original, at its uuid.** Load the file with `json.loads`, pop `accuracy` and `speedup`
   from every workload (`aklr/ops/README.md` § Workflow, steps 3 and 4), replace the text in the `content` of the
   source you edit (assert that it occurs once) and write the result with `json_file_text` to
   `<same directory>/<schema_uuid(name, OpContract.authored_parts(schema))>.json`, as [authoring.md](authoring.md) §3
   does (compact JSON, sorted keys, indent 4 or no final newline audit as `schema_invalid`). `python -m aklr export
   check` names the expected path of a stale name (S10).
3. **If A is the file main serves at the Point, time B against it in the same rounds:**
   `python -m aklr validate --base origin/main --devices 0 --out <dir> <B>.json`. B's record then holds a second
   `timings` entry, `versus: "incumbent"` (A, the served file at the merge base), with as many rounds as the baseline
   entry, every participant timed in each round in a seeded random order (S59): its `exp(log_ratio_mean)` is the geometric mean of the paired-round A/B time ratios,
   this page's B/A speedup ratio. The record's `baseline` field names the served file: check that it is A. The exit code is not the A/B: a B slower than A gets a `regressed` demotion, which counts as passed, and a file qualifies (exit 0) when any of its Points is correct and beats the baseline, so B exits 0 even if every Point regressed against A, as long as one of them is still correct and beats the baseline (S64). `PREFETCH` 5 → 3 at m = 16 read B/A 0.881 and exited 0, both timings kept *(field, E6)*: read the incumbent entry.
4. **Otherwise alternate A and B on one idle GPU**, A B A B …, 4 runs each:
   `python -m aklr validate --stages audit,bench --devices 0 --out ab/<A|B>_<i> <file>.json`, one `--out` directory
   per run (a reused one, the default `aklr_validation/` included, is overwritten by the next run). Do not run two
   validates on one GPU at once: two started together printed 0.80 and 1.04 for the m = 16 GEMM (0.83 and 0.95 in a
   second try), 1.42–1.47 alone *(field, E6)*. Give each arm its own `TRITON_CACHE_DIR`. Before you read a B/A near
   1.00, confirm that the change reached the binary: if the change affects generated code, the arms' `.amdgcn` must differ; otherwise record that this is a same-binary control ([debugging.md](debugging.md) §3),
   and each arm's cached `<kernel>.json` lists the options it compiled with (`num_warps`, `waves_per_eu`,
   `llvm_fn_attrs`). A one-Point schema took 15 to 25 s per run, so 4 + 4 runs take about 3 minutes *(field, E6)*.
5. **Compare means.** In each run's `<out>/report.json` the stage with `stage_name: "bench"` holds one
   `workload_records[]` entry per Point (`axes` names it), whose `timings[]` entry with `versus: "baseline"` holds
   `log_ratio_mean` (its `exp` is the run's geometric mean of per-round ratios; the console prints the lower bound, lower under load: 0.80 for a
   mean of 0.857 *(field, E6)*) and `candidate_milliseconds` (the call time per timing round: take the median); a
   reader: [debugging.md](debugging.md) §10. Average each arm's 4 runs; report B/A of the speedup ratios and A/B of the call times (or label B/A call-time change separately).
6. **Read it against the noise.** With 4 runs per arm an identical copy's mean ratio read up to 1.3% off the original
   (fp8 GEMM m=16; ≤ 0.4% at 6 other Points) and single runs of an unchanged kernel spread up to 2.6% within one session
   (§1). A 1.2% change (`group_m` 4 → 1, fp8 GEMM m = 4096) put all 4 B call times above all 4 A, the ratios agreeing,
   while an identical copy separated in 0 of 7 A-vs-A sessions (one per Point) *(field, E3)*. Smaller effects need a
   paired ruler: the incumbent timing of step 3 gave `group_m` 4 → 1 at m = 4096 B/A 0.981 ± 0.08% (95%, 16 rounds)
   *(field, E6)*. Rulers that load both files in one process and pair their rounds: the fixed-slot prefetch comparison in [gemm.md](gemm.md) §11 and, on FlyDSL kernels, `../gemm/measurement.md` §1–2.
7. **Read B's accuracy result** (`acc ok` in the log): a faster variant can be wrong (`numerics.md`) and still exit 0 when another Point qualifies,
   with `numeric_mismatch` demotions in `report.json`; a one-Point file fails with `no_qualifying_workload` ([authoring.md](authoring.md) §11).
8. **Clean up, then gate.** Delete every copy you do not ship: `validate --base origin/main` without a path includes
   changed and untracked schemas in the audit; in schema-only changes it benches paths not present at the merge base, including new-uuid losing copies (S59). Ship a win as B's file replacing A's (CI then
   re-measures A's other Points too) or as a second file narrowed to the Points where it wins (dispatch served a file
   narrowed to P at P and the original at its other Points; `aklr/ops/README.md` § Workflow, steps 3 and 4). Then gate
   with `python -m aklr validate --base origin/main` (the ruler page).

## 3. What moved the ratio: the grid's parallelism and the prefetch depth

Four changes lost roughly 9–37% of speedup, and each removed loads in flight: fewer programs, or fewer stages per program; a
fifth, on thousands of one-wave programs, barely moved. Three knobs were also moved the other way (committed value in
bold; each B against the A runs interleaved with it; call time = the sum of the call's kernels, as the bench times it):

| Knob (Point, file) | Values: programs × waves per program | B/A | Call µs, A → B |
|---|---|---|---|
| KV splits, gated attention decode (32 rows, `d9cbde237fd34de9`) | **16: 512 × 4** → 8: 256 × 4 | 0.63 | 44.18 → 69.77 *(field, E7)* |
| | **16** → 32: 1024 × 4 | 0.95 | 44.10 → 46.56 *(field, E7)* |
| Split-K of the router projection (256 tokens, `c3889614985b700a`) | **4: 256 × 4** → 2: 128 × 4 | 0.75 | 9.91 → 13.29 *(field, E8)* |
| | **4** → 8: 512 × 4 | 0.97 | 9.97 → 10.25 *(field, E8)* |
| Router look-ahead (stages issued ahead), same Point | **2** → 1, 256 × 4 | 0.89 | 10.01 → 11.28 *(field, E8)* |
| `PREFETCH`, async K groups (128 K elements sharing one block scale; K = 1536 has 12) in flight per wave, fp8 GEMM (m=16, `f3970decacc97225`) | **5** → 3, 192 × 1 | 0.87 | 4.23 → 4.85 *(field, E10)* |
| | **5** → 7 | 1.00 | 4.29 → 4.29 *(field, E10)* |
| Gate-up K split with its scratch, grid and activation launch, DeepSeek-V2 MoE (2 tokens, `d3bbbce5c0bbc61e`) | **4: 2306 × 1** → 2: 1154 × 1 | 1.00 (0.996) | 19.20 → 19.28 *(field, E11)* |

Grids from S11–S14 and S49. In the gated-attention file the head tile follows the split count (`head_tile = 16 if SPLITS
== 16 else HEADS`, S13), so both changed arms also ran an 8-row head tile: those rows measure the two changes together
([attention.md](attention.md) §7). The router reads split, tile and look-ahead from one `@gluon.constexpr_function`,
`_shape_config(rows)`, in kernel (`M: gl.constexpr`) and launch alike (S11, S12; [authoring.md](authoring.md) §5).

What to do with it:

- **Prefetch depth is instance-specific: deeper can still help.** In the table, less depth or fewer splits lost 9–37% of speedup and double splits 2–6% (speedup losses, not call-time increases) *(field, E7, E8, E9, E10, E12)*. At fp8 GEMM m = 64 (`4d4c`, committed `waves_per_eu=6`) depth 4 → 6 cut the mean call time 4.33% (paired-log 95% interval, widened for arm order, 0.9268–0.9876; one run of four adjacent AB/BA pairs per candidate); with `waves_per_eu=0`, which spills nothing, the change was unresolved (0.9446–1.0402). Shallower depths lost at m = 32 and 64 with spill-free controls too, and no fixed ratio between what a shallower depth loses and a deeper one gains followed *(field, E51)*.
  **Split neighbours are Point-specific.** In a separate Qwen3.5 gated-attention comparison (accurate arms, 6 counterbalanced rounds per arm, 3 forward and 3 reverse, one run), halving the split raised the call time +44.07%, +56.44% and +78.22% at 16, 64 and 128 rows, doubling it +3.257%, −2.690% and −21.82%; at 16 rows a half split with the head tile pinned cost +42.77%. These are call-time changes, not the speedup losses above *(field, E52)*.
  Halving cost more under a conservative joint bound (two paired-log 99% intervals sharing the A arm, at least 98% joint coverage per group, not 99% simultaneous coverage of the table); the doubling's 99% intervals were 1.018381–1.046948, 0.970714–0.975498 and 0.780215–0.783308. The fastest tested grids were 512, 1024 and 1024 four-wave programs, so 256–512 is no universal optimum. Measure both neighbours: a split also changes tiles and merge width, and in the 16-row file the head tile. CU residency was not measured, no exhaustive search was run, and look-ahead and prefetch were not re-measured in this comparison.
- **At the table's Points, for the two split knobs moved both ways the committed value was the fastest of three; `PREFETCH` 7 was observed within 2% of 5
  in four of five runs** *(field, E7, E8, E9, E10)*. At each Point try half and double of each split and the depth ±2.
- **Count the grid against the 256 CUs, but measure neighbours.** In the table above the fastest tested grids were 256 to 512 four-wave programs (1024 to 2048 waves), not a general optimum *(field, E52)*:
  one workgroup per CU (256, §6) where two per CU were best lost 37% (with the narrower head tile), one per two CUs
  where one per CU was best lost 25% *(field, E7, E8)*. Thousands of one-wave programs barely moved: halving the MoE
  gate-up's 2306 programs cost 0.4%, the runs overlapping (paired 95% interval 0.991–1.001) *(field, E11)*.
- **Merge a split in a fixed order.** Split output dimensions (N tiles, heads, value rows) first: they need no partials.
  A split of K or of the keys writes partials to scratch and adds them in a fixed order, in a merge pass or the next
  launch, not with float atomics: at 43e112e 65 of the 154 Triton schemas, the gated-attention and DSA files among them, compare
  `bitwise` across runs (S61); an atomic sum's order changes the bits ([numerics.md](numerics.md) §8).
- **Reduce in a launch the call already has**, as the three splits above do (routing, attention finish, activation launch: S62). A separate reduce adds a launch (§1): measured under Kineto before #33, the n2624 k6144 split-K GEMM at m = 64 read 12.57 µs with its reduce launch and 8.37 without, while an extra kernel cost 1.8 µs in a bare CUDA graph ([authoring.md](authoring.md) §7) *(field, E35)*.
  At (M, N, K) = (64, 2624, 6144) a separate split + reduce graph with split 6 was faster than the pilot-selected splits 1, 2, 3, 8 and 12; split 6 against 4 stayed unresolved, neither faster nor shown equivalent (one cold-cache run, 32 confirmation pairs per comparison, BM64/BK128 and backend options fixed). Its whole-minus-partial estimate, 1.868 µs, is neither the reduce's own time nor an allocation-inclusive cost; AITER and M = 2/4/8/16 were not tested *(field, E49)*.
- **A decode-size search starting point, not a universal optimum:** at three Qwen2 router Points (64, 128, 256 tokens) and gated attention (32 rows), the fastest measured half / committed / double split had 256 or 512 four-wave programs (one or two per CU); at 64 tokens double was observed 0.7% ahead, the difference not resolved. Half lost 4–37% of speedup, double changed it by −6% to +1%.
  Try neighbours and prefetch depth: router look-ahead 2 → 4 reduced call time 3% at 256 tokens (one run), → 1 lost 11–13% speedup; fp8 GEMM m = 16 `PREFETCH` 3 lost 9.5–13% of speedup (B/A 0.871–0.905), 4 and 7 stayed within 3.3% of 5. These measurements do not establish a floor-reaching rule or §5's knobs for other Points *(field, E7, E8, E9, E10, E48)*.
- **At a Point of several dispatch rounds** (prefill, M ≥ 1024) the knobs measured here moved little: tile order
  (`group_m` 4 → 1) 1.2–2.1% at m = 4096 in four runs of 4 alternated rounds (fp8 GEMM `aef7bbb0`, §12), removing
  `waves_per_eu` 0.0–0.15% at 8192 tokens in two (MiniMax gate-up `51ae4994`) *(field, E3, E25)*. Tile by M:
  [gemm.md](gemm.md) §3, §7, [moe.md](moe.md) §2, [gluon/mfma.md](gluon/mfma.md) §4; occupancy: §8.

## 4. Why (inferred): bytes in flight set a decode kernel's bandwidth

In the stream curve below, MB of bytes in flight means MiB; the router and GEMM estimates later use decimal MB. On a Gluon stream of 1 GiB (`buffer_load`, `.cg`, 16 bytes per lane) the bandwidth followed the bytes in flight
(programs × waves × independent load instructions per lane × 1 KiB per wave-load), not the program count, for 256 to 8192
programs of 1 to 8 warps with up to 8 loads per lane (equal bytes within 10.4%, 3 runs): 256 × 4 warps × 8 loads read
1.07× of 2048 × 4 × 1 (7.1–7.2 TB/s), 4 MB in flight 5.7–6.3 TB/s; at 3 MB 192 one-wave programs × 16 loads read
2.45–2.47 TB/s, half of 384 × 4 × 2 *(field, E12)* ([gluon/memory.md](gluon/memory.md) §7). Up to 2 MB the rate grows
nearly in proportion (1.16–1.17 TB/s at 0.5 MB, 3.87–3.88 at 2 MB), fitting a loaded latency of about 0.45–0.55 µs
(estimated, bytes ÷ rate). More loads per wave cost registers: each lane below keeps `U` loads in flight (`U` × 8 bf16),
and `U` = 1, 2, 4, 8 compile to that many `buffer_load_dwordx4 … sc0 nt` per iteration and 22, 38, 70, 140 VGPRs:

```python
@gluon.jit
def stream_sum(x_ptr, out_ptr, n_per_prog, U: gl.constexpr, BLOCK: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [1, 64], [1, 4], [1, 0])
    rows = gl.arange(0, U, layout=gl.SliceLayout(1, layout))
    cols = gl.arange(0, BLOCK, layout=gl.SliceLayout(0, layout))
    offs = rows[:, None] * BLOCK + cols[None, :]
    base = gl.program_id(0) * n_per_prog
    acc = gl.zeros([U, BLOCK], gl.float32, layout)
    for i in range(0, n_per_prog, U * BLOCK):
        acc += gl.amd.cdna4.buffer_load(x_ptr, base + i + offs, cache=".cg").to(gl.float32)
    gl.store(out_ptr + gl.program_id(0), gl.sum(gl.sum(acc, axis=1), axis=0))
```

Launch it with `num_warps=4` and `BLOCK=2048`, the layout's width (8 elements × 64 lanes × 4 warps): a smaller `BLOCK`
makes lanes load the same addresses, a larger one adds loads per lane, and `n_per_prog` must be a multiple of `U *
BLOCK` (no mask). The counts hold when `n_per_prog` is a multiple of 16, which the JIT specializes; for the tested runtime i32 builds with no divisibility attribute (and aligned pointer attributes),
  it emits 8 `buffer_load_ushort` per row at U = 1, 2 and 4 (Triton 3.8.0, gfx950).

Inferred, not isolated (no counter was read): §3's four reduced-parallelism/depth variants (each in 3 more runs) came from fewer programs
or fewer stages per program; bytes in flight were counted for the router only, and doubling its split raised them yet
lost 2–4% *(field, E12)*. The m = 16 GEMM issues 12, 20 or 28 direct-to-LDS copies of 1 KiB per wave before its first
wait at `PREFETCH` 3, 5 or 7, then waits at `vmcnt(4)`, `vmcnt(12)` or `vmcnt(20)` through the loop (compiles with
Triton 3.8.0 for gfx950): about 0.8–1.6, 2.4–3.1 or 3.9–4.7 MB in flight over its 192 programs (estimated;
[gemm.md](gemm.md) §5). That 3 sits on the curve's rising part and that the launch floor (§1) is why 7 changed speedup by about 0–3.2% in those m = 16 runs
is inferred. The router stages 12 KiB per program per stage: 256 programs with 2 stages ahead hold about 6 MB, 1 stage
ahead or 128 programs about 3 MB (estimated). Budget bytes in flight this way only for grids of a multi-wave program per
CU or more: at 3 MB the curve over-predicted those by about 5–7%, smaller grids 1.3–2.1× (3 runs) *(field, E41)*.

## 5. What did not move the ratio, on the Points measured

Each row is one change on one Point, 4 runs per arm: small observed changes are measurements at that Point, not equivalence proofs, and the knob remains a candidate elsewhere. The
m = 16 GEMM, the 2-token GDN decode and the 8-token router read near their launch floor (§1), so their rows say "no
loss", not "no effect": re-test such a knob at a Point above the floor before you drop it.

| Change | Point (file) | B/A | Call µs, A → B |
|---|---|---|---|
| `waves_per_eu` 3 → unset on the one-wave `_private_k128_gemm` | fp8 GEMM m=16 (`f3970decacc97225`) | 1.00 | 4.25 → 4.26 *(field, E13)* |
| `waves_per_eu` 2 → unset on the 8-wave `_attention_split` | DSA, 24 tokens (`2ee56a6dbabc48a6`) | 1.01 | 18.89 → 18.71 *(field, E13)* |
| `waves_per_eu` 3 → unset on the gate-up | MiniMax MXFP4 E256, 8192 tokens (`51ae499426db920d`) | 1.00 | 510.9 → 511.8 *(field, E13)* |
| `amdgpu-sched-strategy` removed from both launches | Qwen2 router, 256 tokens (`c3889614985b700a`) | 1.00 (0.997 in a second run) | 9.91 → 9.94 *(field, E14)* |
| `.cs` removed from the recurrent state's `buffer_load` / `buffer_store` | Qwen3.5 GDN decode, 2 tokens (`299133886d479017`) | 1.00 | 8.75 → 8.79 *(field, E15)* |
| that state's `buffer_load` / `buffer_store` → `gl.load` / `gl.store`, `.cs` kept | same | 1.00 | 8.83 → 8.81 *(field, E15)* |
| hidden states and weight loaded with `gl.load` and converted to the MFMA operands through LDS, instead of async copies to LDS | Qwen2 router, 8 tokens (`4801e216dcff290f`) | 0.99 (1.00 in a second run); difference not resolved | 8.45 → 8.45 *(field, E16)* |
| `BANK_XOR` off → on (XOR-swizzled LDS addresses) | fp8 GEMM m=16 (`f3970decacc97225`) | 1.00 (0.995) | 4.25 → 4.26 *(field, E17)* |
| program-id → N-tile permutation removed (one program per tile) | fp8 GEMM N=7168 m=64 (`06755ba3ee5eb085`) | 1.00 | 6.26 → 6.26 *(field, E17)* |

Plus the numerics switches (§9). An LDS-bound loop read a row-major tile at 0.57–0.59 of the XOR-swizzled rate (4 runs)
*(field, E39)* (`gluon/layouts.md` §5), while the m = 16 GEMM is sensitive to prefetch depth (a memory-wait explanation is inferred, not counter-isolated); async copies streamed at 0.96–1.01× the
rate of register loads with 16–32 waves per CU (3 runs) *(field, E22)* (`gluon/memory.md` §8).

**Coupled constants.** Splits 8 → 4 alone in the DSA 24-token file failed to build (`build_failed`: `NameError:
cache_shared is not defined`; its 64-head tile path exists only for 8 splits; [attention.md](attention.md) §4), and
`gate_fold` 4 → 1 alone in the MiniMax MXFP4 1-token file ([moe.md](moe.md) §7) crashed the compiler (`candidate_crash`,
exit −11: SIGSEGV in `make_amdgcn`, gone without the launch's `amdgpu-sched-strategy`; [debugging.md](debugging.md) §6),
in 4 of 4 runs each *(field, E19)*. §3's MoE K split moved scratch, grid, `SPLITS` and the activation launch together.

## 6. Grid size and occupancy

gfx950 has 256 CUs (8 XCDs of 32) *(field, E24)*; this page counts a grid in workgroups per CU (256 is one per CU). A
dispatch round is the workgroups the CUs hold at once, set by registers and LDS (`../gemm/grid_pricing.md` §1, §4; an
under-filled grid's symptoms: `../patterns/low_cu_utilization.md`). If registers and LDS permit one or two workgroups per CU, a grid of that density can bound residency: 256 four-wave programs supply an average of one wave per SIMD across 1024 SIMDs, and 512 supply two; actual placement and concurrent residency were not measured.
Below one per CU the m = 16 GEMM's 192 one-wave programs lean on each wave's depth (§3, §4). Grids per Point:

| Kernel | Points | How the grid is formed | Programs × waves |
|---|---|---|---|
| Qwen2 router projection | 8 to 256 tokens | split-K × expert tiles × token tiles held at 256: split 8 at 8 to 64 tokens, 4 at 128 to 256 | 256 × 1 at 8 and 16 tokens, 256 × 4 at 64 to 256 (S12, S15) |
| Qwen3.5 gated attention partials | 1 to 256 rows | (row, head, split) at 1 to 2 rows; (row, split) from 4 rows; rows × splits = 512 from 16 rows | 256 and 512 at 1 and 2 rows; 128 at 4; 256 at 8; 512 × 4 from 16 rows (S13, S16) |
| DSA split, 64 heads | 8 to 32 tokens | tokens × head groups × splits; 144 to 160 KiB of LDS, so one workgroup per CU (160 KiB is the most a workgroup may take, `../hardware/gfx950/memory_and_sync.md`) and grid size / 256 is a CU-coverage upper bound, not measured wave occupancy | 256 × 4 at 8 tokens; 192 to 256 × 8 at 12 to 32 (S17, S46) *(field, E20)* |
| Qwen3.5 GDN decode recurrence | 1 to 256 rows | value-row tiles per head at 1 to 4 rows, one program per (token, head) from 8 rows | 256 at 1, 2, 4, 16 rows; 128 at 8; 512 at 32; 1024 to 4096 × 16 at 64 to 256 (S18) |
| Qwen3.5 GDN prefill, `gdn_prefill_group_fp8_quant_n_v_heads16…` (5 files), sequential phase | one to 8 sequences | value block chosen so sequences × 16 heads × 128 / BV reaches 256 at the sequence count the file is written for (8, 4 or 2) | 256 at the several-sequence Points except 160 at five sequences and 192 at three; 128 at one sequence (S19) |
| fp8 block-scaled GEMM | m = 1 to 64 | one narrow N tile per program | 768 × 1 (m ≤ 4), 192 × 2 (8), 192 / 384 / 768 × 1 (16 / 32 / 64) (S14); N=7168: 448 × 1 to 224 × 4 (`gemm.md`) |

- **Across the 15 GDN prefill files the sequential phase runs 128 to 512 programs, and the speedups stored at 43e112e are not monotonic
  in program count**: 128 1.188–2.638 (seven one-sequence Points), 160 0.997, 192 1.029, 256 1.339–2.629, 384 1.198,
  512 1.889 *(field, E33)*. A cost of under-filling is not measured here.
- The `86987a66c0cdba39` Qwen2 experts source schedules a device-counted list using 16-program route tiles and 256-program cohorts. CPU execution of its scheduling AST covered 16,384 nonzero configurations without missing or duplicate work; the worst pass-size spread was 4080. This is coverage, not a measured balance or speed advantage; zero jobs were not run on a GPU *(field, E50)*.

## 7. `num_warps`

- **It is part of the layouts.** Gluon sets the module's warp count from the launch option before it verifies the
  layouts, and rejects a layout whose `warps_per_cta` multiplies to another count (S20, S21). Changing `num_warps` at
  launch is a code change unless the layouts read `gl.num_warps()`, a constexpr of the launch option (S22), as the
  gated-attention files do: `gl.BlockedLayout([1, 16], [4, 16], [gl.num_warps(), 1], [1, 0])` (S36).
- It must be a power of two, defaults to 4 and sets the work-group size LLVM schedules for (S23).
- **Choosing it.** One wave when a program's work fits a wave (a GEMV row, a per-token selection): the work-group then
  has 64 lanes and LLVM drops `s_barrier` (0 in the one-wave m=16 GEMM, 16 in the four-wave router projection *(field, E20)*). More waves when they split an MFMA tile (8 warps halve each wave's `v_mfma` on a 32 × 256 tile, not a 32 × 32
  one; `gluon/layouts.md`); a 144 KiB-LDS loop at one workgroup per CU ran 3–4% faster at 8 warps than 4 (3 runs)
  *(field, E41)*. Warps in the 897 compiled kernels of the 154 schemas at 43e112e: 4 in 514, 1 in 276, 2 in 76, 8 in 27, 16 in 4 *(field, E20)*.
- **More than 4 warps caps registers.** A workgroup's ceil(num_warps / 4) waves share each SIMD, so a wave keeps at
  most 512 / ceil(num_warps / 4) registers whatever `waves_per_eu` says: a plain-Triton 256 × 256 × 64 bf16 tile
  compiled to 482 VGPRs at 4 warps, 256 with 100 spilled at 8, 128 with 131 spilled at 16. Keep a reduced axis inside
  each wave: a row sum over an axis spread across 4 warps compiled to an LDS write, an LDS read and an `s_barrier`, and
  to none with whole rows in each wave (all compile with Triton 3.8.0 for gfx950), as in the GDN decode's "128-element
  reduction stays wave-local and needs no LDS exchange" (S34).
- **What the committed kernels launch** (the census at 43e112e *(field, E20)*; per family:
  [kernels/kernel_library.md](kernels/kernel_library.md) §5–§14): one wave for the fp8 GEMM at N=3072 m ≤ 64 except m=8
  and at N=7168 M ≤ 8 (10 Points); 8 in 27 kernels: DSA 4, Qwen3-Next GDN decode 9, MiniMax-M2 MoE 8, Qwen3.5 GDN
  prefill 4 (`_chunk_boundaries`, `_propagate_chunks`), Qwen2-MoE experts 1, bf16 GEMM 1; 16 for the Qwen3.5 GDN decode
  recurrence from 64 rows and the Qwen2 router projection at 2232 tokens (`4c2bed7e3e8c1e94`). Router selection and
  decode GEMVs: not checked. At 43e112e no Point had two committed files that differ only in `num_warps`, so no committed pair
  isolates the warp count's effect: when you vary it, vary the layouts with it and A/B it.

## 8. `waves_per_eu`: a register cap

**What it does.** A nonzero value adds the LLVM attribute `"amdgpu-waves-per-eu"="N,N"`; 0, the default, adds nothing
in the PyPI Triton 3.8.0 (S24; the image's build writes `"0,0"`). On gfx950 a wave may hold 512 registers, arch and
accumulation together (`../hardware/gfx950/lookup.md`; 482 compiled unspilled); N waves per SIMD cap each wave at
512 / N and spill the rest: a 4-warp bf16 MFMA probe compiled at 0, 1, 2, 4, 8 used 438, 438, 256, 128, 64 VGPRs with
0, 0, 181, 353, 558 spilled *(field, E21)*. LLVM does not see the LDS, which Triton allocates at launch (S25; all 897 kernels of the census at 43e112e report a zero static LDS segment *(field, E20)*), so the cap applies whatever the LDS
allows: a 4-warp kernel whose 64 KiB of LDS allow 2 waves per SIMD was still capped from 170 to 128 VGPRs at 4, spilled
58 and ran 1.43× slower *(field, E21)*. Take a kernel's LDS from its `metadata.shared` (S50), not from the ISA.

**Below the workgroup's own count it need not tighten the register budget; at or above it, it can spill a kernel that fits.** A workgroup of
`num_warps` waves already puts ceil(num_warps / 4) on each SIMD. A plain-Triton 256 × 256 × 64 bf16 tile on 8 warps
compiled to the same code at `waves_per_eu` 1 as with none (256 VGPRs, 128 AGPRs, 100 spilled); on 4 warps a
128 × 256 × 64 tile used 218 VGPRs unspilled, and 256 with 100 spilled at 2 (both compile with Triton 3.8.0). Compare
both builds' spill counts, not only their VGPRs (the GDN prefill row below).

**On committed kernels** (VGPRs = arch + AGPR, compiles with Triton 3.8.0 for gfx950; the census at 43e112e, compiled on the GPU node, holds the same counts for the committed settings, 88, 228, 256 (64 spilled), 190, and no unset build *(field, E20)*):

| Kernel (Point) | Warps | Setting | VGPRs with it | VGPRs unset | Measured |
|---|---|---|---|---|---|
| fp8 GEMM `_private_k128_gemm` (m=16) | 1 | 3 (cap 168) | 88 | 84 | B/A 1.00 (§5) |
| DSA `_attention_split` (24 tokens) | 8 | 2 (cap 256, which 8 warps impose anyway) | 228 | 222 | B/A 1.01 (§5) |
| Qwen3.5 GDN prefill `_build_chunk_tables_paired` (4097 tokens) | 1 | 2 (cap 256) | 256, 64 spilled, 260 B scratch | 322, no spill | unset 1.04× faster: 281.2 → 270.3 µs, arms separated *(field, E13)* |
| same kernel (8192 tokens) | 1 | 2 | 190 | 183 with PyPI Triton 3.8.0, 184 with the image's build *(field, E20)* | B/A 1.00 (0.9985) *(field, E13)* |

The GDN prefill file sets 2 for both of its Points (S26): it fits at 8192 tokens and spills at 4097, where the grid
(1032 one-wave programs, 129 chunks × 8 head pairs) cannot place 2 waves on most SIMDs anyway. That Point stores 1.029
*(field, E33)*; without the cap the call ran 3.9% faster (281.2 → 270.3 µs, speedup 1.017 → 1.057, the 4 runs per arm
not overlapping: 1.014–1.019 against 1.054–1.064); at 8192 tokens removing it changed nothing (B/A 0.999) *(field, E13)*.

When to try it:
- the grid has several rounds of programs: on a one-wave streaming kernel of 264 VGPRs (168, 96 spilled, under 3) the
  cap made 8192 programs 1.50× faster (1376 → 917 µs) and 256 programs, under one wave per SIMD, 13% slower (70.3 → 79.8
  µs) *(field, E13)*. The LDS-full case (the DSA split's 160 KiB) is untested;
- the natural count sits just above a tier (512 / N rounded down to 8, as compiled: 256, 168, 128, 96, 80, 72, 64). Even
  a small cut can cost: the Qwen3-Next decode template (not measured at 43e112e) compiles to 138 and 131 VGPRs at 16 and
  64 rows; forced to 4 on the committed file, both kernels compiled to 128 VGPRs, spilling 9 and 20 (2.6%, 5.3% more
  instructions), and the call ran 1.60× slower at both (speedup ratio 0.62–0.63; its comment says ~19%, S27) *(field, E13)*. On FlyDSL attention kernels `../attention/registers_occupancy.md` §2 records a forward kernel 5.47× slower under a cap and a sparse-MLA backward dQ kernel 2× and 4× slower (repeat counts unknown there). No committed AKLR A/B shows a cap helping (§5) *(field, E13)*: measure every cap.

Read the compiled count first (`debugging.md`) at every Point the setting applies to. At 43e112e the committed kernels set 1 to 8
on the fp8 GEMM's decode Points but none at N = 7168, M = 2 (N = 3072, m = 1 to 64: 8, 4, 4, 2, 3, 8, 6), 2 on every
bf16 GEMM tile kernel (not the split-K reduce), 6 and 8 on the 16-wave GDN decode recurrence, none in any DeepSeek-V2
MoE or Qwen3-Next file ([kernels/kernel_library.md](kernels/kernel_library.md) §5–§14): each Point's search result.

## 9. Launch options and attributes

| Option | What it changes in the binary (Triton 3.8.0, gfx950) | Measured on a committed kernel |
|---|---|---|
| `llvm_fn_attrs=[["amdgpu-sched-strategy", s]]` (`max-ilp`, `iterative-ilp`, `iterative-minreg` in AKLR's files, switched per Point: S28); a separate CPU test on Triton 3.7.0 found no `llvm_fn_attrs` field in `HIPOptions` and its constructor rejected the keyword (`TypeError`). AKLR's requirement helper marked `triton >=3.8` unmet on 3.7.0, and, per its source, 3.7.0's launcher would reject the keyword with `KeyError` ([authoring.md](authoring.md) §3, S8; not run); dispatch filtering was not run on 3.7.0. Declare `"requires": {"triton": ">=3.8"}` to record that version boundary ([authoring.md](authoring.md) §3) *(field, E50)* | LLVM's scheduler strategy; the VGPR count moves by kernel: on a 256 × 128-tile bf16 MFMA loop (8 warps, 16 steps of K = 32) `max-ilp` 145, `iterative-ilp` 148, `iterative-minreg` 88 against 134 unset. The option is "Experimental" (S29) and applied after `waves_per_eu` and the denormal mode, overwriting them: `[["amdgpu-waves-per-eu", "1,1"]]` on `waves_per_eu=4` leaves only `"1,1"` and 134 VGPRs instead of 128 with 10 spilled, and `[["denormal-fp-math-f32", "preserve-sign"]]` replaces the default `ieee` (all compile with Triton 3.8.0 for gfx950) | removed from the router at 256 tokens: B/A 1.00 *(field, E14)* |
| `llvm_fn_attrs=[["amdgpu-ieee", "false"]]` (Qwen2 routing launches) | the kernel descriptor's `.amdhsa_ieee_mode` goes from 1 to 0, so MODE.IEEE (S54) is off: with it on, a raw `v_max_f32` of a signalling NaN and 1.0 returned a quiet NaN in either order, with it off 1.0 *(field, E42)*; and a max no longer canonicalizes its operands: `gl.max` over rows of 4 elements, four rows per thread, went from 12 to 4 `v_max_f32` (3 → 1 per row, one `v_max3_f32` per row either way; compiles with Triton 3.8.0 for gfx950) | removed from the router's routing launch at 256 tokens: B/A 1.01 (1.00 in a second run), 9.99 → 9.95 µs, accuracy still ok *(field, E14)* |
| `allow_flush_denorm=True` (on the split and merge launches of four of the five DSA 64-head files, 12 to 32 tokens; not the 8-token one) | `denormal-fp-math-f32` becomes `preserve-sign` in the LLVM IR (S29), but the gfx950 code does not change: the same instructions and `.amdhsa_float_denorm_mode_32` 3 with and without it, on a kernel of exp, log, division, sqrt, exp2 and a multiply-add (compiles with Triton 3.8.0 for gfx950; [numerics.md](numerics.md) §6) | removing it at 12/16/24/32 tokens gave B/A 1.0007/1.0033/0.9986/0.9970; all compared arms passed accuracy *(field, E18)* |
| `enable_fp_fusion` (default on; `False` on at least one launch in 42 of the 154 schema files at 43e112e, chosen per Point in 4 more, 3 of which turn it off at one of their Points: `96e51c1f85a71b7e` at 16 tokens, `61ed32f62fa797df` and `1d5568a78971fb62` at 512) | on: LLVM may contract a multiply and an add into one FMA, one rounding (S31); off: `v_pk_mul_f32` + `v_pk_add_f32` instead of `v_pk_fma_f32` (compiles with Triton 3.8.0 for gfx950); 23.5% of 10^6 random `a * b + c` differed fused against unfused *(field, E42)* | off → on at DeepSeek-V2 inter_dim 512, 8 tokens: same gfx950 code on all 3 launches; B/A 1.00, 56.27 → 56.34 µs, accuracy still ok *(field, E18)* |
| `num_stages` | nothing in Gluon: `gluon_to_ttgir` has no `num_stages`-driven pipeliner (S32); a Gluon loop compiled at 1 and 3 gives identical code *(field, E21)*. In plain Triton it drives the software pipeliner (S33): a 128 × 128 × 64 bf16 GEMM with 8 warps issued 0, 8 and 12 direct-to-LDS loads at 1, 2 and 3; the block ping-pong pass, added above 1, changed that GEMM only at 3 (2 `amdg.cond_barrier`, no `s_setprio`; compiles with Triton 3.8.0 for gfx950) | — |
| `schedule_hint` | nothing: "Experimental; right now no effect" (S29); identical code with and without (compiles with Triton 3.8.0 for gfx950). The n6144 decode GEMM passes it (S45); Triton 3.7.0 turned `schedule_hint="memory-bound-attention"` into the `iterative-ilp` scheduler ([gemm.md](gemm.md) §9): on 3.8.0 `llvm_fn_attrs=[["amdgpu-sched-strategy", "iterative-ilp"]]` sets the same attribute | — |
| `matrix_instr_nonkdim` | plain Triton only: the dot lowering picks the MFMA with it (S58); Gluon picks the instruction in the layout (`gluon/mfma.md`) | — |
| `waves_per_eu`, `num_warps` | §8, §7 | §5 |
| any option, even one that compiles to nothing | enters the JIT's in-memory cache key before backend normalization (S57): `kpack=2` after `kpack=1` creates another raw-options entry, but gfx950 normalizes both to 1, so this does not prove a second distinct compiled binary; flipping back reuses the first entry | — |

`enable_fp_fusion` changes rounding and `amdgpu-ieee` the NaN result of a raw `v_max_f32` (`gl.maximum` stayed
bit-identical on 10^6 finite pairs) *(field, E42)*: decide both with `validate`'s accuracy check and `numerics.md`.

## 10. Pipelining by hand: depth, placement, unrolling

Gluon has no `num_stages` pipeliner (S32): the depth is in the code, as in the router's ring of `AHEAD + 2` slots,
`AHEAD` ahead, a barrier between a slot's last read and its refill (S11; [gluon/memory.md](gluon/memory.md) §11–12).

- **Depth per Point.** Make it a constexpr chosen per Point, as the committed kernels do: the router's look-ahead is 1
  at 64 rows and 2 at 128 and 256 (S11, S55); the N=3072 fp8 GEMM keeps 5 K groups ahead at m = 16 and 32 and 4 at m =
  64 (S14); the N=7168 GEMM rings 3 slots with 2 ahead at M = 64 and 4 with 3 at M = 32, with a per-step wait table
  (S35); a stage less or more: §3. Each slot costs LDS: keep `metadata.shared` (S50) at most 160 KiB per workgroup: 192
  KiB compiles, and the first launch raises `OutOfResources`, "out of resource: shared memory, Required: 196608,
  Hardware limit: 163840" (S60). Recount workgroups per CU past one dispatch round (§6).
- **Placement per Point.** Where a load is issued relative to the math is a knob too: gated attention issues the V
  gather before QK, or before or after the softmax sum, chosen per row count (S36; not A/B-tested); the Qwen3-Next
  decode template (not measured at 43e112e) issues the state load early at 1 and 32 rows, late at 128 and 256 (S37):
  early costs 7, 0, 77 and 12 more VGPRs (compiles with Triton 3.8.0 for gfx950), and at 32 rows late ran 10.28 µs
  against 8.39 at equal VGPRs *(field, E43)*.
- **Latency-bound programs: issue order and live ranges.** Where a decode program is one chain of loads and dependent
  math, the committed files reorder it rather than cut work. No item below was timed in a kernel: compare the variants'
  VGPRs, spills and `s_waitcnt` order ([debugging.md](debugging.md) §2–3), then A/B them (§2).
  - Issue the large load first: "Expose the large state fetch before preparing scalar gates" (S51).
  - Shorten live ranges: Q held in LDS "until state writeback to shorten its register lifetime", beta computed "near
    its use to shorten the gate live range", Q loaded after the state update, SiLU deferred to the epilogue, and
    `loop_range(..., disable_licm=True)` (S30), which kept 34 of 64 loop-invariant `exp2` in the loop: 40 VGPRs
    against 74 with all hoisted (compiles with Triton 3.8.0 for gfx950).
  - Evaluate scalar transcendentals in neighbouring lanes: one `gl.exp` over a 4-lane group gives the gates' four
    exponentials, and one division SiLU(v) and sigmoid(b) (S34; [gdn.md](gdn.md) §4).
  - Keep the MFMA off the critical path: fold each K group's scales `DELAY` groups late through a tuple queue (S52),
    alternate two accumulators by K step (1.02–1.03× in 5 runs on a bare chain of dependent bf16 MFMAs *(field, E41)*),
    read operands ahead and issue the MFMA before the refill, load scales before the copies they would queue behind
    (S40; [gemm.md](gemm.md) §5), pin an accumulator with an empty tied asm ([gluon/wave_ops.md](gluon/wave_ops.md) §8);
    fully unroll a one-wave loop and ask for `max-ilp` (S53).
  - From the Qwen3-Next decode files: state and index pointers first among the arguments, which Triton marks for SGPR
    preloading (S47). Their comment that decay and beta before the conv "costs enough registers to halve occupancy on
    the 4-warp instance" did not reproduce: 200 VGPRs either way at 128 rows (compiles with Triton 3.8.0 for gfx950).
- **Unrolling.** `gl.static_range` emits the body once per iteration at trace time (S38), so the loop variable can
  index a Python tuple (of buffer pointers, checked); the code grows with the trip count: 68, 117, 233, 451
  instructions at 16, 32, 64, 128 steps. For a loop that stays a loop, `from triton.language.core import range as
  loop_range` (`gl` has no `range`) and pass `loop_unroll_factor=N` to unroll it N times in Triton's IR (S39). A
  factor of 1 does not stop LLVM unrolling a small body: a one-load loop at factor 1 compiled to 8 loads per iteration
  and a remainder loop (all compile with Triton 3.8.0 for gfx950); check the loop in the ISA (`debugging.md` §3).

## 11. Cache hints

In 3 runs on a pure 1 GiB stream, loads with `.cg` or `.cs` (the `nt` bit) ran 13–14% faster than unhinted ones, and no
hinted store beat the unhinted one by more than 0.74% (`.cs` lost 5.5–7.2%) *(field, E22)*; the bits each string sets,
the keyword of each call (`cache=` on the buffer ops, `cache_modifier=` on `gl.load` / `gl.store`; `gl.load(cache=)`
raises a `TypeError`) and the per-path rates: [gluon/memory.md](gluon/memory.md) §6. After a 1 GiB read flush (a Gluon stream, eager launches timed by events, 4 runs) `.cg` gained +3.9–7.4% at 64 MB, +9.0–10.4% at 128, +5.9–8.8% at 192, +10.8–11.5% at 512 and +9.9–11.7% at 1 GiB; the 1 GiB gain exceeded the 64 MB one by 3.2–6.0 percentage points (3.2 in the run where 64 MB gained 7.4%), and at 192 MB the gain was below 128 MB's in all four runs, so the gain does not grow steadily with size *(field, E34)*. In real kernels the hint
moved the call 0.6 to 3% at 256 tokens and about 1% at 2 tokens, 2.8–3.1 µs above its floor (§1) *(field, E23, E36)*:

| Change | Point (file) | B/A | Call µs, A → B |
|---|---|---|---|
| `.cg` removed from the weight loads | MiniMax MXFP4 E64, 256 tokens (`1d277d76892ee692`) | 0.97, all 4 B runs below all 4 A runs; its 128-token Point, unchanged, 1.00 | 104.8 → 107.9 |
| `.cg` added where the author left it off | DeepSeek-V2 MoE inter_dim 512, 256 tokens (`2422a87b3b814299`) | 0.99, all 4 B runs below all 4 A runs | 226.4 → 228.0 |
| `.cg` removed from the gate-up weights | DeepSeek-V2 MoE inter_dim 256, 2 tokens (`d3bbbce5c0bbc61e`) | 0.99, all 4 B runs below all 4 A runs | 19.09 → 19.26 |

Both 256-token files choose the hint per Point (S41, S42); there and at 2 tokens the committed choice was the faster
*(field, E23)*: A/B weight loads with and without `.cg` at each Point; stores: unhinted unless an A/B says otherwise.

## 12. Program-id remapping for XCD and L2 reuse

Workgroups are dealt to the 8 XCDs round-robin, each with its own L2 (`../gemm/schedule_and_locality.md` §2) *(field, E24)*. In a probe, 8 programs reading the same chunk finished 1.8 to 2.1× sooner when their ids were congruent mod 8
(one XCD; 3 runs) *(field, E24)* ([gluon/memory.md](gluon/memory.md) §14). In kernels the effect was small: dropping the
m=4096 fp8 GEMM's grouped tile order (`group_m` 4 → 1, leaving M-major order with the 4-way N interleave, S43) cost
1.2–2.1% in four runs (25.38 → 25.87 µs, all 4 B runs slower than all 4 A runs, in one) *(field, E3, E25)*; removing the
N=7168 m=64 GEMM's permutation (S44), where each tile has one reader, did nothing (§5). Remap when several programs read
the same data; a remap must be a permutation of the grid for every grid size (`gemm/decode_gemm.md` §3).

## 13. Budget a Point before designing: matrix-core and memory rates

Four measured budgets suggest bottleneck candidates, not strict lower bounds: MFMA time (FLOP ÷ the rate below, scaled as a low-grid estimate by 1024 ÷ min(grid waves, 1024) when
it supplies fewer than 1024 waves; actual SIMD placement and residency were not measured *(field, E26)*), stream time (unique bytes ÷ the read-flushed rate at that size: about
4.5 TB/s at 64 MB, 6.0 at 1 GiB, 6.6–6.8 with `nt` loads *(field, E22, E34)*), the latency floor that the bytes in
flight set (§4), and about 4 µs per launch behind the flush, whatever the work (under Kineto, §1) *(field, E35)*. Below 64 MB, where
the Triton GEMMs' weights lie (4.7 to 32 MB), a plain read stream under AKLR's bench read 25–34 MB at 4.48–4.83 TB/s, 14–16 MiB
at 3.16–3.72 and 4.5 MiB at 1.04–1.11 (4.24–4.64 µs for the smaller streams, 4 runs) *(field, E44)*; whole calls read the bf16 n6144 GEMM's 25 MB in
5.94–6.37 µs at m ≤ 16 ([gemm/decode_gemm.md](gemm/decode_gemm.md) §1), the N = 7168 fp8 GEMM's 15.7 MB at m = 64 in
6.24–6.36 µs ([gemm.md](gemm.md) §11), 3 runs *(field, E36)*: use the measured-size plain-read budgets as starting estimates, then measure;
64 MB was not tested with this stream, and smaller weights do not obey a general 4–4.5 TB/s rule. The matrix cores ran bf16 `mfma` at 2447–2468 TFLOP/s, e4m3 `mfma_scaled` at 4899–4934 and e2m1
(MXFP4) at 9382–9591 in the 16 × 16 shape (one-wave programs at 2 waves per SIMD on all 1024 SIMDs, chained MFMAs, 4
runs); 32 × 32 within 1% for bf16, fp8 and fp32, 2.4–4.4% lower for e2m1 *(field, E26)*: pick the shape for the tile and
the layouts ([gluon/mfma.md](gluon/mfma.md) §1); kernels using the two shapes were not compared.

The m = 16 fp8 GEMM does 0.15 GFLOP, about 0.16 µs of MFMA with 192 one-wave programs supplying at most 192 SIMDs, and streams a
4.7 MB weight, about 1.05 µs at 4.5 TB/s (all estimated), yet reads 4.21–4.33 µs, 0.2 µs above the one-launch floor; at
3 groups 4.82–4.90 µs, so that change showed, and `PREFETCH` 7 was up to about 3.3% faster in these runs (§3; why is inferred, §4). In 3 runs the largest
budget was launch time or stream time at two of three Points: the launch floor is 94–96% of the m = 16 GEMM's time, the stream floor
76–78% of the MiniMax MXFP4 layer's 103–105 µs at 256 tokens; at the m = 4096 fp8 GEMM no floor dominates (8.0 µs
stream, 7.9 MFMA of 25.4) *(field, E36, E45)*, and its tile order moved it (§12). Tune a floor-bound Point in its launch
count (§1), a stream-bound one in its bytes and hints (§11), a latency-bound one per §3, §4, §10, a compute-bound one
from its tile and MFMA shape ([gemm.md](gemm.md) §3, [gluon/mfma.md](gluon/mfma.md) §4; these two untested here).

## 14. Per-Point configuration: how the files write it, and what changes as M grows

**How it is written.** With shapes as constexprs each Point compiles its own binary on its first call; a runtime int
shares one across the Points of one specialization class (§1). The configuration is Python; of the 154 files at 43e112e, 131
branch on the token count (`if m == …` or a threshold) to choose a kernel and its constexprs (S14), 71 use dataclass or NamedTuple
configuration records, such as a MiniMax MXFP4 prefill policy switching at 1536, 3072, 5120 and 6656 tokens and a Qwen3-Next
decode config per row count (S56), 51 contain a `@gluon.constexpr_function`; some helpers are shared by kernel and launch code (§3), 11 a
`functools` cache, such as the plain-Triton GEMMs' launch builder keyed by the shapes (S45). How each keys Triton's
binaries: [authoring.md](authoring.md) §5. Constexpr against runtime token counts was measured once: the Qwen3-Next
prefill twins, first measured on main in d908d01, read 0.1–2.4% apart over 11 Points in that CI run, the constexpr
file ahead at 6 and the runtime one at 5 *(field, E38)* ([gdn.md](gdn.md) §8).

**Only the branch your Point takes is evidence.** Generated files keep tables for sizes they do not serve, and the
tables disagree: in the 8-token DSA file `_decode_profile_m2(8)` gives `waves_per_eu` 1 but serves only other token
counts, while `_launch_geometry(8)` gives the 2 that the `_attention_partials_m8` launch passes (S46) and its compiled
producer carries (`"amdgpu-waves-per-eu"="2,2"`, Triton 3.8.0). Resolve the branch before you copy a constant.

**What changes as M grows**, by family (details per family in `gemm.md`, `moe.md`, `attention.md`, `gdn.md`):

| Family | As the Point grows | CI speedup stored at 43e112e |
|---|---|---|
| fp8 block-scaled GEMM, N=3072 (15 files, one per Point) | m ≤ 64: one- or two-wave programs over narrow N tiles, 4 of the 12 K groups to all of them in flight; 128 to 256: four-wave tiles with a 4-slot LDS ring; ≥ 1024: four-wave tiles, single-buffered (64 rows at m = 1024, 128 at 8192) | 1.307–1.435 at m ≤ 64; 1.199, 1.477 at 256, 128; 1.191–1.486 from 1024 *(field, E27)* |
| fp8 block-scaled GEMM, N=7168 (9 files) | 448 one-wave programs at M ≤ 8; 224 programs of 2 to 4 waves at 16 to 64; four-wave at 128 and 256 | 1.664–1.840 at M ≤ 32; 1.488 at 64; 1.114–1.138 at 128 and 256 *(field, E27)* |
| bf16 decode GEMMs, plain Triton (2 files, 10 Points) | one N tile per workgroup over all of K; split-K plus a fixed-order reduce only at n=2624 m=64 (`gemm/decode_gemm.md`) | 1.048–1.292 *(field, E28)* |
| MoE layers: DeepSeek-V2, MiniMax-M2 FP8 and MXFP4, Qwen2-MoE experts | 3 or 4 launches at 1 to 4 tokens; 3, 5 or 6 at 8 to 16; 5 to 7 at 32 to 256; 6 to 8 at ≥ 1024, where at least 26 of the 32 Points launch a kernel that allocates LDS (`moe.md`) | 1.097–4.115 at ≤ 16 tokens (34 Points); 1.030–1.275 at 32 to 256 (23); 1.152–1.785 at ≥ 1024 (32) *(field, E29)* |
| DSA sparse MLA decode | 64 heads: head tile and split count chosen together to keep 192 to 256 programs; 8 heads: split at 32 and 128 tokens, one program per token without a merge at 256 | 64 heads 1.030–2.188; 8 heads 1.050–1.196 *(field, E30)* |
| Qwen3.5 gated attention decode | FMA dot products per (row, head, split) at 1 and 2 rows; all 8 heads in one MFMA tile from 4 rows; splits shrink as rows grow (rows × splits = 512 from 16 rows) | 2.182 at 1 row; 1.671–1.803 at 2 to 32; 1.401, 1.204, 1.066 at 64, 128, 256 *(field, E31)* |
| Qwen3.5 GDN decode | each head's state split over programs at 1 to 4 rows; one program owns a (token, head) state from 8 rows, 16 waves from 64 | 2.336–2.507 at ≤ 16 rows; 1.282–1.651 at 32 to 256 *(field, E32)* |
| Qwen3.5 GDN prefill | value block of the sequential phase chosen for the sequence count the file is written for (§6) | 0.997–2.638 *(field, E33)* |
| Qwen3-Next GDN decode and prefill | per-size files plus a nine-row template (decode); constexpr and runtime-count twins (prefill) | not measured at 43e112e |

Families added since 43e112e (DSA indexer, Qwen3-Next gated attention and MoE, DSA with 16 heads, Qwen2-MoE MXFP4) and the files added to existing ones: [kernels/kernel_library.md](kernels/kernel_library.md).

Of the MoE layers' three ranges in the table, the 32 to 256-token one is the lowest *(field, E29)*. Why is not measured;
the decomposition changes there ([moe.md](moe.md) §2). A weight-bytes/time estimate leaves a bandwidth hypothesis to test, not proof of a stream limit: at 64 tokens on
MiniMax-M2 MXFP4 (one Point) the bench's routing touches about 77% of the 256 experts (simulated), whose weights over
the layer's time give 4.9–5.0 TB/s for AKLR and 3.9 for the baseline in 3 runs (estimated, no trace) *(field, E45)*.

## Evidence

AKLR paths and line numbers are on main `0e02a81` unless a line names another commit; each E line names the commit,
CI run or PR its numbers come from. #64 renamed the 15 n3072_k1536 GEMM schemas after 43e112e: they are named as on
main, except in paths pinned to an earlier commit.

- E1 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, AKLR CI bench (graph replay, 1 GiB read flush, Kineto kernel sum), stored speedup against the schema's baseline] gemm_n6144_k2048 `f806ad2b086d757d` stored 1.116, 1.129 and 1.292 at m = 2, 4 and 8, one compiled kernel serving all three (AMD-AGI/AKLR@90b9d50 (CI run 41976547645):aklr/ops/gemm/impls/triton/gemm_n6144_k2048/f806ad2b086d757d.json)
- E2 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, `aklr validate --stages audit,bench` on one GPU (graph replay, 1 GiB read flush, Kineto kernel sum), three consecutive runs per schema] A-vs-A, max ÷ min of exp(log_ratio_mean) over the three runs: fp8 GEMM `f3970decacc97225` (N=3072, m = 16) 2.39%, and 2.77% over three `--base` runs (1.4784, 1.4386, 1.4749); Qwen2 router `c3889614985b700a` (256 tokens) 0.91%; Qwen3.5 GDN decode `299133886d479017` (2 tokens) 1.73% (2.4222, 2.4105, 2.4522); DeepSeek-V2 MoE `d3bbbce5c0bbc61e` (inter_dim 256, 2 tokens) 0.94% (the four schemas at AMD-AGI/AKLR@43e112e)
- E3 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, `aklr validate --stages audit,bench` on one GPU (graph replay, 1 GiB read flush, Kineto kernel sum), 4 alternated runs per arm] an identical copy against the committed file at 7 Points, max ÷ min of an arm's 4 runs: 0.07% (fp8 GEMM `aef7bbb04ccbe710`, m = 4096), 0.12% (fp8 GEMM, m = 64), 2.39% (`f3970decacc97225`, m = 16; 2.55% in an A-only session), 0.95% (router `c3889614985b700a`, 256 tokens), 2.03% (GDN decode, 2 tokens), 1.16% (MoE `d3bbbce5c0bbc61e`, 2 tokens), 2.14% (router `4801e216dcff290f`, 8 tokens); the router at 8 tokens, A arms 1.2029–1.2228 and 1.1998–1.2263; m = 16 session means 1.4507, 1.4642, 1.4722, 1.4536 (four sessions, one job); the copy's mean ratio off by at most 1.31% (m = 16; 0.01–0.41% elsewhere); B/A of the mean ratios and A/B of the mean call times within 0.20% over 8 pairs; `group_m` 4 → 1 at m = 4096: 25.65 → 25.96 µs (+1.2%), every B call time above every A, while identical copies separated in 0 of 7 sessions; at m = 16 and 4096: `bench_cudagraph` 4.229 and 25.74 µs, validate 4.296 and 25.65, a hot eager loop 20.76 and 27.59, a flushed eager loop 6.16 and 29.2 (schemas at AMD-AGI/AKLR@43e112e)
- E4 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2 to 8 CI runs per Point, AKLR CI bench (graph replay, 1 GiB read flush, Kineto kernel sum), stored speedup against the schema's baseline] the absolute log ratio between any two stored values of the same sources at the same Point: median 2.2%, p90 6.3%, p95 8.1%, 17.5% of pairs above 5% (1314 pairs over the 122 Triton/Gluon Points with two or more runs on main); per-family median 0.9% plain-Triton GEMM and RMSNorm, 2.0% Qwen3.5 GDN prefill, 2.3% DeepSeek-V2, 2.4% MiniMax-M2 FP8, 2.6% MiniMax-M2 MXFP4, 2.7% Qwen3.5 GDN decode, 3.1% fp8 GEMM, 3.7% Qwen2-MoE; 9 Points stored values on both sides of 1.03, and 3 stored exactly 1.030 at 43e112e (`b3913402078d9226` at 4 tokens, `0bb2f7cbe49d1be3` at 8, `a1c4a2129730d1a0` at 128); the 12 fp8 GEMM Points at m ≤ 64 with two or more runs moved 4.0–11.8% (max ÷ min over 3–7 runs) (the nine "perf: measurements" commits on main from AMD-AGI/AKLR@a29581b to 43e112e, CI runs 41197527705 to 41991630592)
- E5 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, `aklr validate --stages audit,bench` on one GPU (graph replay, 1 GiB read flush, Kineto kernel sum), each schema at its lowest Point, the aiter build not recorded] against the value stored in the file: plugin schemas 84 of 106 within ±3%, 94 within ±5%, family medians 0.9911 to 1.06 (Qwen2-MoE fp8 router + experts); fp8 GEMM 0.9848 to 1.11 (median 1.005); bf16 GEMM and RMSNorm (4 schemas) 0.9971 to 1.03 (the 154 Triton schemas of AMD-AGI/AKLR@43e112e)
- E6 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2, `aklr validate --stages audit,bench` on one GPU (graph replay, 1 GiB read flush, Kineto kernel sum)] one-Point runs took 14.9–22.3 s (16.7 warm, 17.7 cold, 22.3 with `--base`) and 15.3–17.7 s in the other run (178 earlier ablation runs: 15 to 25 s); two validates of `f3970decacc97225` started together on device 0 printed 0.80 (exit 1) and 1.04, then 0.83 and 0.95, against 1.42–1.47 alone; printed lower bound against exp(log_ratio_mean): 1.4720 → 1.47, 0.8572 → 0.80, 1.1586 → 1.04; `validate --base` of `f3970decacc97225` with `PREFETCH` 5 → 3 (one run): an incumbent entry of 16 rounds like the baseline entry, exp(log_ratio_mean) 0.8814, equal to the recomputed time ratio, bench passed, exit 0, demotion `regressed`; incumbent B/A of `aef7bbb04ccbe710` with `group_m` 4 → 1 at m = 4096: 0.9812, 95% half-width 0.080%, 16 rounds (fp8 GEMM N=3072 at m = 16 and 4096; AMD-AGI/AKLR@43e112e)
- E7 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, `aklr validate --stages audit,bench` on one GPU (graph replay, 1 GiB read flush, Kineto kernel sum), 4 alternated runs per arm] KV splits 16 → 8: mean ratio 1.790 → 1.129 (0.631), call 44.18 → 69.77 µs, pairs 0.627–0.634; 16 → 32: 1.791 → 1.693 (0.945), 44.10 → 46.56 µs, all 4 B runs below all 4 A runs; the head tile went from 16 to 8 rows in both; an earlier session read 0.633 and 0.949 (Qwen3.5 gated attention decode `d9cbde237fd34de9`, 32 rows; AMD-AGI/AKLR@43e112e)
- E8 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, `aklr validate --stages audit,bench` on one GPU (graph replay, 1 GiB read flush, Kineto kernel sum), 4 alternated runs per arm] split-K 4 → 2: 1.927 → 1.441 (0.7476), 9.906 → 13.29 µs, pairs 0.743–0.755; 4 → 8: 1.915 → 1.864 (0.9732), 9.971 → 10.25 µs, pairs 0.961–0.986, every B below every A, upper 95% bound 0.989; look-ahead 2 → 1: 1.917 → 1.699 (0.8865), 10.01 → 11.28 µs, pairs 0.880–0.892, every pair slower, grid 256 × 4 in both; speedup lost by the smaller value: router split 2 25.2%, gated-attention splits 8 36.7% (512 → 256 programs), look-ahead 1 11.4%, fp8 GEMM m = 16 `PREFETCH` 3 9.5%; by the larger: split 8 2.7%, gated-attention splits 32 5.7%, `PREFETCH` 7 none (1.013) (Qwen2 router `c3889614985b700a`, 256 tokens, 256 programs × 4 warps; AMD-AGI/AKLR@43e112e)
- E9 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3, `aklr validate --stages audit,bench` on one GPU (graph replay, 1 GiB read flush, Kineto kernel sum), 3 alternated runs per arm with GPU_FORCE_QUEUE_PROFILING=1 and 2 without] call time over `PREFETCH` 5: `PREFETCH` 3 1.115, 1.140, 1.125 (without the variable 1.129, 1.148, 1.122), `PREFETCH` 7 0.985, 1.001, 0.969 (without 1.026, 0.999, 0.968); B/A of the mean speedups: `PREFETCH` 3 0.898, 0.877, 0.888, `PREFETCH` 7 1.016, 0.999, 1.032; `PREFETCH` 5 arms 4.26–4.33 µs, `PREFETCH` 3 arms 4.82–4.90 µs (fp8 GEMM `f3970decacc97225`, m = 16; AMD-AGI/AKLR@43e112e)
- E10 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1 (5 → 4: N=2), `aklr validate --stages audit,bench` on one GPU (graph replay, 1 GiB read flush, Kineto kernel sum), 4 alternated runs per arm] `PREFETCH` 5 → 3: 1.475 → 1.285 (0.871), 4.232 → 4.851 µs, every pair slower; 5 → 7: 1.448 → 1.449 (1.001), 4.294 → 4.293 µs, pairs 0.990–1.004; 5 → 4: 1.456 → 1.459 (1.002), 4.272 → 4.263 µs, and in a second run 1.445 → 1.449 (1.003), 4.312 → 4.301 µs, the second run's 5 → 3 0.881 and 5 → 7 1.003; an earlier session read 0.873 and 1.007 (fp8 GEMM `f3970decacc97225`, m = 16; AMD-AGI/AKLR@43e112e)
- E11 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, `aklr validate --stages audit,bench` on one GPU (graph replay, 1 GiB read flush, Kineto kernel sum), 4 alternated runs per arm] gate-up K split 4 → 2 with its scratch, grid (2306 → 1154 one-wave programs) and activation launch: 2.183 → 2.175 (0.996), 19.20 → 19.28 µs; single runs A 2.167–2.195, B 2.157–2.191, B below A in each pair, paired 95% interval 0.991–1.001, the arms' runs overlapping (DeepSeek-V2 MoE `d3bbbce5c0bbc61e`, inter_dim 256, 2 tokens; AMD-AGI/AKLR@43e112e)
- E12 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3, a standalone Gluon stream (1 GiB bf16 read, `buffer_load` `.cg`, 16 bytes per lane, eager launches); the B/A re-measurements with `aklr validate`] GB/s by MB in flight over 3 runs: 0.375 MB 876–886, 0.5 MB 1164–1172, 1 MB 2159–2294, 2 MB 3725–3929, 4 MB 5697–6333, 8 MB and more 6354–7167; 256 × 4 warps × U = 8 7129–7167 against 2048 × 4 × U = 1 6674–6684 (1.067–1.070); equal bytes at different program counts (256 to 8192 programs) within 1.104, the four pairs at 4 to 8 times the programs 1.028–1.072; at 3 MB 384 × 4 × U = 2 4920–4962, 768 × 2 × U = 2 4898–4947, 96 × 4 × U = 8 3945–3995, 192 × 1 × U = 16 2454–2467; bytes ÷ rate 0.444–0.542 µs from 0.375 to 2 MB; 1, 2, 4, 8 `buffer_load_dwordx4 … sc0 nt` per iteration and 22, 38, 70, 140 VGPRs at U = 1, 2, 4, 8; the router at 256 tokens 12288 B per program per stage, 6.29 MB at 256 programs 2 ahead, 3.15 MB at 128 programs or 1 ahead; §3's changes re-measured beside it, B/A: attention splits 8 0.630–0.635, router split 2 0.748–0.753, look-ahead 1 0.881–0.890, `PREFETCH` 3 0.877–0.898, router split 8 0.961–0.978 (the stream: `stream_sum` of §4, #89; the A/Bs: AMD-AGI/AKLR@43e112e)
- E13 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, `aklr validate --stages audit,bench` on one GPU (graph replay, 1 GiB read flush, Kineto kernel sum), 4 alternated runs per arm; a standalone Gluon stream with eager launches] `waves_per_eu` removed: `f3970decacc97225` (m = 16), 3 → unset: B/A 0.9973, 4.251 → 4.258 µs; `2ee56a6dbabc48a6` (DSA, 24 tokens), 2 → unset: 1.008, 18.89 → 18.71 µs, unset faster in all 4 pairs; `51ae499426db920d` (MiniMax MXFP4 gate-up, 8192 tokens), 3 → unset: 1.441 → 1.440 (0.9992), 510.9 → 511.8 µs; none of the three below 0.997; Qwen3-Next decode template `f44007374038041f`, `waves_per_eu=4` forced on `_fused_heads` at 16 and 64 rows: 128 VGPRs, 9 and 20 spilled, call-time ratio 1.598 and 1.595, speedup ratio 0.6237 and 0.6258; Qwen3.5 GDN prefill `2ecece9650b95f71`, `waves_per_eu=2` removed from `_build_chunk_tables_paired`, which A compiled to 256 VGPRs with 64 spilled at 4097 tokens and 190 unspilled at 8192: at 4097 tokens speedup 1.017 → 1.057 (B/A 1.04), call 281.2 → 270.3 µs, A runs 1.0136–1.0186, B 1.0537–1.0638; at 8192 tokens B/A 0.9985, the runs overlapping; the stream: one-wave programs streaming 256 KiB each with a 64 × 256 fp32 ballast, 264 VGPRs unspilled unset, 168 with 96 spilled at 3: 8192 programs 1376 → 917 µs (0.666), 256 programs 70.3 → 79.8 µs (1.135), unset repeats within 0.4% (schemas at AMD-AGI/AKLR@43e112e; the stream: #89)
- E14 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2, `aklr validate --stages audit,bench` on one GPU (graph replay, 1 GiB read flush, Kineto kernel sum), 4 alternated runs per arm] `amdgpu-sched-strategy` removed from both launches: 1.930 → 1.925 (0.9975), 9.91 → 9.938 µs, the code of both kernels changed; `amdgpu-ieee` false removed from the routing launch: 1.910 → 1.921 (1.006), 9.99 → 9.946 µs, accuracy ok in every B run; a second run: sched 0.9969, 9.968 → 9.998 µs, ieee 1.004, 9.98 → 9.944 µs (Qwen2 router `c3889614985b700a`, 256 tokens; AMD-AGI/AKLR@43e112e)
- E15 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, `aklr validate --stages audit,bench` on one GPU (graph replay, 1 GiB read flush, Kineto kernel sum), 4 alternated runs per arm] `.cs` removed from the state's `buffer_load` and `buffer_store`: 2.452 → 2.442, 8.748 → 8.789 µs; the state through `gl.load` and `gl.store`, `.cs` kept: 2.429 → 2.432, 8.827 → 8.813 µs (Qwen3.5 GDN decode `299133886d479017`, 2 tokens; AMD-AGI/AKLR@43e112e)
- E16 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2, `aklr validate --stages audit,bench` on one GPU (graph replay, 1 GiB read flush, Kineto kernel sum), 4 alternated runs per arm] B loads hidden states and weight with `gl.load` (24 `global_load_dwordx4`, no async copy) and converts them through LDS (24 `ds_write_b128`, 32 `ds_read_b128`) instead of async copies to LDS: 1.186 → 1.178 (0.9927), 8.447 → 8.45 µs; a second run 1.169 → 1.172 (1.003), 8.41 → 8.403 µs; A-vs-A at this Point: single runs 2.14% apart, the copy's mean 0.26% off (Qwen2 router `4801e216dcff290f`, 8 tokens; AMD-AGI/AKLR@43e112e)
- E17 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, `aklr validate --stages audit,bench` on one GPU (graph replay, 1 GiB read flush, Kineto kernel sum), 4 alternated runs per arm] `f3970decacc97225` (m = 16), `BANK_XOR` on: 1.460 → 1.453 (0.995), 4.245 → 4.263 µs; `06755ba3ee5eb085` (N=7168, m = 64), tile permutation removed: 1.479 → 1.481 (1.001), 6.262 → 6.262 µs (fp8 GEMMs; AMD-AGI/AKLR@43e112e)
- E18 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, `aklr validate --stages audit,bench` on one GPU (graph replay, 1 GiB read flush, Kineto kernel sum), 4 alternated runs per arm] DeepSeek-V2 MoE `a2bb3d83284b20ff` (inter_dim 512, 8 tokens), `enable_fp_fusion` False → True on its 3 launches: 1.147 → 1.146 (0.9994), 56.27 → 56.34 µs, accuracy ok in all 8 runs, A's and B's compiled launches (`_front_split`, `_expert`, `_down_reduce_tiny`) identical in instructions and `.amdhsa_` fields (633, 1554 and 440 instructions, compiled offline with PyPI Triton 3.8.0); `allow_flush_denorm` removed from the split and merge launches of the DSA 64-head files at 12, 16, 24 and 32 tokens: B/A 1.0007, 1.0033, 0.9986, 0.9970, accuracy ok, the 8 compiled kernels identical with and without it (AMD-AGI/AKLR@43e112e)
- E19 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=4, `aklr validate --stages audit,bench` on one GPU (graph replay, 1 GiB read flush, Kineto kernel sum)] DSA 64-head `2ee56a6dbabc48a6` (24 tokens) with splits 8 → 4 alone: `build_failed`, CompilationError, NameError `cache_shared` is not defined; MiniMax MXFP4 `655723f593921b08` (1 token) with `gate_fold` 4 → 1 alone: `candidate_crash`, runner exit −11; 4 of 4 runs each (AMD-AGI/AKLR@43e112e)
- E20 [2026-11, gfx950 (MI355X), Triton 3.8.0 (the image's build), N=1, the compiled kernels of every AKLR Triton schema at its Points, read from their `.amdgcn`] 897 kernels: `num_warps` 4: 514, 1: 276, 2: 76, 8: 27 (DSA 4, Qwen3-Next GDN decode 9, MiniMax-M2 MoE 8, Qwen3.5 GDN prefill `_chunk_boundaries` and `_propagate_chunks` 4, Qwen2-MoE experts 1, bf16 GEMM 1), 16: 4 (Qwen3.5 GDN decode `_update_and_norm` at 64, 128 and 256 rows, Qwen2 router `4c2bed7e3e8c1e94` `_router_projection`); the fp8 GEMM one wave at its 10 Points of N=3072 m ≤ 64 except 8 and N=7168 M ≤ 8; 20 Points with two or more files, none differing only in `num_warps`; nonzero `.group_segment_fixed_size` in 0 of 897; DSA 64-head split producers 147696 to 163840 B of LDS; committed settings, with the same counts compiled offline: `f397` `_private_k128_gemm` 88 VGPRs (unset 84), `2ee5` `_attention_split` 228 (unset 222), `2ece` `_build_chunk_tables_paired` 256 with 64 spilled and 260 B scratch at 4097 tokens (unset 322, none spilled), 190 at 8192 (unset 184 with the image's Triton, 183 with PyPI Triton 3.8.0 offline); `s_barrier` 0 in `f397` `_private_k128_gemm` (1 warp), 16 in `c388` `_project_split_k` (4 warps) (the 154 schemas of AMD-AGI/AKLR@43e112e)
- E21 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2, standalone Gluon kernels compiled for gfx950 and timed on MI355X] a 4-warp bf16 MFMA kernel (128 × 256 fp32 accumulator) at `waves_per_eu` 0, 1, 2, 4, 8: 438, 438, 256, 128, 64 VGPRs, 0, 0, 181, 353, 558 spilled, 438 = 256 arch + 182 AGPRs; a 256 × 256 × 64 tile at 4 warps 482 unspilled; a 4-warp kernel with 65536 B of LDS: 170 VGPRs unset, 128 with 58 spilled at 4, 282.7–282.8 against 404.5–404.9 µs over 8192 programs (1.43×), static LDS 0 against `metadata.shared` 65536; a Gluon loop at `num_stages` 1 and 3: identical code (compiled without a GPU) (#89)
- E22 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3, standalone Gluon streams: 1 GiB bf16, 2048 programs × 4 warps, 16 bytes per lane; a 64 × 64 bf16 tile stream through registers or LDS, 1024 and 2048 four-wave programs] reads, hinted over none, buffer and global: `.cg` 1.132–1.135, `.cs` 1.130–1.137 (`.cg` 6785–6823 GB/s against 5993–6013 unhinted); writes: best hinted store over none 0.998–1.007 (buffer) and 1.002–1.006 (global), the unhinted, `.wb` and `.cg` stores (one instruction) up to 1.5% apart, `.cs` 0.928–0.945; tile stream, async copies over register-only loads: best async mode 0.987–1.008 at 1024 programs (16 waves per CU), 0.973–0.987 at 2048 (32), every async mode 0.957–1.008; a fourth run of the tile stream: best async mode 1.008 at 1024 and 0.986 at 2048 programs (async copies 5875 and 5559 GB/s, register-only 5829 and 5639) (#89)
- E23 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, `aklr validate --stages audit,bench` on one GPU (graph replay, 1 GiB read flush, Kineto kernel sum), 4 alternated runs per arm] MiniMax MXFP4 E64 `1d277d76892ee692`: `.cg` removed at 256 tokens 1.181 → 1.145 (0.969), 104.8 → 107.9 µs, every B run below every A, upper 95% bound 0.976; its 128-token Point 1.001; DeepSeek-V2 inter_dim 512 `2422a87b3b814299`, 256 tokens: `.cg` added 1.039 → 1.033 (0.994), 226.4 → 228.0 µs, every B below every A; DeepSeek-V2 inter_dim 256 `d3bbbce5c0bbc61e`, 2 tokens: `.cg` removed from the gate-up weights 2.205 → 2.180 (0.9885), 19.09 → 19.26 µs, every B below every A, pairs 1.0007–1.0158 slower (AMD-AGI/AKLR@43e112e)
- E24 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3, a standalone Gluon probe: 2048 programs × 4 warps, 8 readers per chunk, a 1 GiB read before each launch, 3 rounds; hardware ids read by spinning probes] 64 KB chunks 21.3–21.6 µs with consecutive ids against 11.7–12.1 µs with ids congruent mod 8 (medians 1.78–1.80×, single rounds from 1.76); 512 KB chunks 154.4–160.5 against 74.1–76.8 µs (medians 2.09–2.12×, single rounds up to 2.17); one reader per chunk 0.98–1.01×; every program on XCC pid mod 8 in 15 launches of 4096 programs; 256 CUs, XCC ids 0–7 with 32 CUs each (#89)
- E25 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2, `aklr validate --stages audit,bench` on one GPU (graph replay, 1 GiB read flush, Kineto kernel sum), 4 alternated runs per arm] `group_m` 4 → 1: 1.351 → 1.325 (0.981), 25.38 → 25.87 µs (+1.9%), single runs A 25.36–25.40, B 25.85–25.88 µs; an earlier session read 25.84 → 26.31 µs; two other runs: call-time ratio 1.013 and 1.021, speedup ratio 0.988 and 0.978; MiniMax MXFP4 gate-up `51ae499426db920d`, 8192 tokens, `waves_per_eu` removed: call-time ratio 0.9985 and 1.000, speedup ratio 1.004 and 0.9996 (fp8 GEMM `aef7bbb04ccbe710`, m = 4096; AMD-AGI/AKLR@43e112e)
- E26 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=4, a standalone Gluon MFMA probe: one-wave programs, 2 waves per SIMD on all 1024 SIMDs, chained MFMAs] TFLOP/s at 16 × 16 and 32 × 32: bf16 2447–2468 and 2435–2452, e4m3 `mfma_scaled` 4899–4934 and 4907–4949, e2m1 9382–9591 and 9026–9341; 32 × 32 over 16 × 16: bf16 0.991–0.999, fp8 0.996–1.000, e4m3 1.000–1.003, fp32 0.997–1.002, e2m1 0.956–0.976; bf16 rate at P one-wave programs over 2048 programs 0.0627–0.0634, 0.1250–0.1256, 0.2496–0.2509, 0.4985–0.5003, 0.9896–0.9918 at P = 64, 128, 256, 512, 1024 (#89)
- E27 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, AKLR CI bench (graph replay, 1 GiB read flush, Kineto kernel sum), stored speedup against the schema's baseline] fp8 GEMM N=3072: 1.307 (m = 4) to 1.435 (m = 8) at m ≤ 64, 1.477 at 128, 1.199 at 256, 1.191 (m = 1024) to 1.486 (m = 8193) from 1024; N=7168: 1.664 (M = 1) to 1.840 (M = 8) at M ≤ 32, 1.488 at 64, 1.114 and 1.138 at 128 and 256 (AMD-AGI/AKLR@43e112e (CI run 41991630592):aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/{d6228c711bd687b6,0e15a0289735acb0,3480e97a5fcda6f6}.json; AMD-AGI/AKLR@90b9d50 (CI run 41976547645):aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/{199c45f4249e7f55,831926aed9523c0f}.json; AMD-AGI/AKLR@a29581b (CI run 41197527705):aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/8888fe378338cc8e.json; AMD-AGI/AKLR@43e112e (CI run 41991630592):aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n7168_k2048/{362ce3bb6200382e,1e021d44010accb1}.json; AMD-AGI/AKLR@a29581b (CI run 41197527705):aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n7168_k2048/{06755ba3ee5eb085,1bbf98a368b3fcba,fee7e092eab87fc5}.json)
- E28 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, AKLR CI bench (graph replay, 1 GiB read flush, Kineto kernel sum), stored speedup against the schema's baseline] bf16 GEMMs 1.048 (n = 2624, m = 2) to 1.292 (n = 6144, m = 8); RMSNorm 1.006 (4 tokens) to 1.212 (2048 tokens), add + RMSNorm 1.024 to 1.093 (AMD-AGI/AKLR@90b9d50 (CI run 41976547645):aklr/ops/gemm/impls/triton/gemm_n2624_k6144/b88c8d5f28d86777.json, aklr/ops/gemm/impls/triton/gemm_n6144_k2048/f806ad2b086d757d.json, aklr/ops/rms_norm/impls/triton/rms_norm_hidden_size6144/cf637bff0784c454.json; AMD-AGI/AKLR@43e112e (CI run 41991630592):aklr/ops/rms_norm/impls/triton/add_rms_norm_hidden_size6144/b3913402078d9226.json)
- E29 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, AKLR CI bench (graph replay, 1 GiB read flush, Kineto kernel sum), stored speedup against the schema's baseline] MoE Points (DeepSeek-V2, MiniMax-M2 FP8 and MXFP4, Qwen2-MoE experts and router + experts): ≤ 16 tokens 1.097 (DeepSeek-V2 inter_dim 512, 16 tokens) to 4.115 (MiniMax FP8 E256, 1 token), 34 Points; 32 to 256 tokens 1.030 (DeepSeek-V2 inter_dim 256, 128) to 1.275 (MiniMax MXFP4 E64, 64), 23 Points; ≥ 1024 tokens 1.152 (DeepSeek-V2 inter_dim 512, 1024) to 1.785 (MiniMax FP8 E64, 16384), 32 Points (the 89 Points of the DeepSeek-V2 and MiniMax-M2 schemas and of the Qwen2-MoE experts and router + experts schemas, aklr/plugins/sglang/models/{deepseek_v2,minimax_m2,qwen2_moe}/impls/triton/ at AMD-AGI/AKLR@43e112e, each value written by its file's last perf: measurements commit); by family DeepSeek-V2 1.030 to 2.186, MiniMax FP8 1.102 to 4.115, MiniMax MXFP4 1.178 to 3.633, Qwen2-MoE (router, router + experts, experts) 1.063 to 1.967 (AMD-AGI/AKLR@3fb9033 (CI run 41812750710):aklr/plugins/sglang/models/deepseek_v2/impls/triton/deepseek_v2_mxfp4_moe_shuffled_16x16_model_dim6144_n_routed256_inter_dim512/3c6e26ffcce7196d.json; AMD-AGI/AKLR@93eba6d (CI run 41964469977):aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_fp8_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts256_inter_dim384/e75e553b34178760.json; AMD-AGI/AKLR@369a3c0 (CI run 41753978959):aklr/plugins/sglang/models/deepseek_v2/impls/triton/deepseek_v2_mxfp4_moe_shuffled_16x16_model_dim6144_n_routed256_inter_dim256/a1c4a2129730d1a0.json; AMD-AGI/AKLR@43e112e (CI run 41991630592):aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_mxfp4_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts64_inter_dim1536/{dfa86447cdb3cc1b,b0651fe84db2f670}.json, aklr/plugins/sglang/models/deepseek_v2/impls/triton/deepseek_v2_mxfp4_moe_shuffled_16x16_model_dim6144_n_routed256_inter_dim512/3432dd09808b8428.json, aklr/plugins/sglang/models/deepseek_v2/impls/triton/deepseek_v2_mxfp4_moe_shuffled_16x16_model_dim6144_n_routed256_inter_dim256/d3bbbce5c0bbc61e.json, aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_fp8_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts64_inter_dim1536/{82e9371086a875bd,098200aca0238880}.json, aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_mxfp4_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts256_inter_dim512/4da30fd46e6f82da.json, aklr/plugins/sglang/models/qwen2_moe/impls/triton/qwen2_moe_fp8_router_experts_shuffled_16x16_model_dim4096_n_routed512_inter_dim256/99f47a166d6d62fc.json, aklr/plugins/sglang/models/qwen2_moe/impls/triton/qwen2_moe_router_model_dim4096_n_routed512_topk10/c3889614985b700a.json)
- E30 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, AKLR CI bench (graph replay, 1 GiB read flush, Kineto kernel sum), stored speedup against the schema's baseline] DSA 64 heads 1.030 (8 tokens) to 2.188 (24 tokens); 8 heads 1.050 (256 tokens) to 1.196 (128 tokens) (AMD-AGI/AKLR@43e112e (CI run 41991630592):aklr/plugins/sglang/layers/dsa_backend/impls/triton/dsa_sparse_mla_decode_n_heads64_kv_dim576_topk2048/{0bb2f7cbe49d1be3,2ee56a6dbabc48a6}.json; AMD-AGI/AKLR@a29581b (CI run 41197527705):aklr/plugins/sglang/layers/dsa_backend/impls/triton/dsa_sparse_mla_decode_n_heads8_kv_dim576_topk2048/{fbe6ddab5dac5958,ae32cc132679692c}.json)
- E31 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, AKLR CI bench (graph replay, 1 GiB read flush, Kineto kernel sum), stored speedup against the schema's baseline] Qwen3.5 gated attention decode 2.182 (1 row), 1.699, 1.671, 1.803, 1.708, 1.789 (2, 4, 8, 16, 32 rows), 1.401, 1.204, 1.066 (64, 128, 256 rows) (AMD-AGI/AKLR@43e112e (CI run 41991630592):aklr/plugins/sglang/models/qwen3_5/impls/triton/gated_attention_decode_group_fp8_quant_qkv_dim4608_n_kv_heads1_head_dim256/{f5ba496e04f8e989,42a24a8f5e63bd0c,3d0f64ffff6790c8,10df1ae2dc9d3c69,6dd321aac1b1d85a,d9cbde237fd34de9,eeb171871a296094,14e91857b4038e02,dfb1ab1cdd5bf263}.json)
- E32 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, AKLR CI bench (graph replay, 1 GiB read flush, Kineto kernel sum), stored speedup against the schema's baseline] Qwen3.5 GDN decode 2.336 (16 rows) to 2.507 (2 rows) at ≤ 16 rows; 1.282 (256) to 1.651 (32) at 32 to 256 rows (AMD-AGI/AKLR@43e112e (CI run 41991630592):aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_decode_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072/{a4f80b6a3877b48e,20f63094b85c00a2}.json; AMD-AGI/AKLR@90b9d50 (CI run 41976547645):aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_decode_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072/299133886d479017.json; AMD-AGI/AKLR@a29581b (CI run 41197527705):aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_decode_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072/c2c3a42a3909684a.json)
- E33 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, AKLR CI bench (graph replay, 1 GiB read flush, Kineto kernel sum), stored speedup against the schema's baseline] the 15 Qwen3.5 GDN prefill files under aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_prefill_group_fp8_quant_*/, tokens / sequences, programs of the sequential phase from the grids, stored value: 1d5568a78971fb62 512/1 128 2.638 and 1024/2 512 1.889, 1fd6adb2b91cf5c2 1024/1 256 2.423, b8c54bde37ee77c7 16384/2 256 1.339 (AMD-AGI/AKLR@43e112e (CI run 41991630592)); 3b99d19659746a36 16384/2 256 1.444, 7a69337f81190952 10000/1 128 1.188, 19b20285ceb08b14 1025/1 128 2.256, 47092631ad3a6e79 16384/1 128 1.741, 12b560c88540bef8 5000/1 128 1.199 and 15000/3 384 1.198, 5ead04831eb8048c 8192/1 128 1.193, 24ede969ed66a87a 15361/8 and 16384/8 256 1.454 and 1.475, 382853eb770e30b3 2048/1 128 2.092 (AMD-AGI/AKLR@369a3c0 (CI run 41753978959)); 61ed32f62fa797df 512/2 256 2.629 (AMD-AGI/AKLR@62a5fe8 (CI run 41778680267)); 0a07f9f9342c6cf1 2049/2 and 4096/2 256 1.374 and 1.452 (AMD-AGI/AKLR@5f63e8a (CI run 41956295934)); 2ecece9650b95f71 4097/3 192 1.029 and 8192/4 256 1.424, c53b6d874d511c02 8193/5 160 0.997 and 15360/8 256 1.473 (AMD-AGI/AKLR@3fb9033 (CI run 41812750710)); each value written by its file's last commit up to 43e112e
- E34 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=4, a standalone Gluon stream reading S = 64 MB to 1 GiB after the previous launch, a 1 GiB read or a 1 GiB write, eager launches timed by events] after a 1 GiB read, `.cg` over none 1.039–1.074, 1.090–1.104, 1.059–1.088, 1.108–1.115, 1.099–1.117 at 64, 128, 192, 512, 1024 MB (192 MB below 128 MB in every run; the 1 GiB gain above the 64 MB one by 3.2–6.0 points, 3.2 in the run where 64 MB gained 7.4%); no hint 4.43–4.50 TB/s at 64 MB and 5.97–6.02 at 1 GiB, `.cg` 6.61–6.67 at 1 GiB; AKLR's flush kernel compiled to 4 vector loads of its buffer and 1 store per program, to its sums tensor (#89; the flush kernel: aklr/core/utils/testing.py at AMD-AGI/AKLR@43e112e)
- E35 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=5, AKLR `bench_cudagraph` at 43e112e, before #33 (graph replay, 1 GiB read flush, Kineto kernel sum), chains of 1 to 32 trivial one-warp Gluon kernels] per added kernel 4.04–4.07 µs with GPU_FORCE_QUEUE_PROFILING=1 and 4.05–4.08 without, in 5 runs; in the last run one kernel 4.21 µs (4.31 unforced) beside a 169.5 µs flush; the bench against an independent Kineto sum of the same kernels replayed without the flush 4.21 / 3.82 to 130.16 / 124.81 µs (4–10% above; 4.06 against 3.90 µs per kernel); torch kernels 4.17–4.19 µs; the n2624 k6144 split-K GEMM at m = 64 12.57 µs with its reduce and 8.37 without; a bare CUDA graph 1.815 and 1.844 µs per extra kernel (AMD-AGI/AKLR@43e112e)
- E36 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3, `aklr validate --stages audit,bench` on one GPU (graph replay, 1 GiB read flush, Kineto kernel sum), per-run median call times; 4 validate runs per schema in each run, 2 for add + RMSNorm and for each gemm_n6144_k2048 Point] add + RMSNorm `b3913402078d9226` at its 7 Points, 4 to 256 tokens: candidate 4.07–4.24, baseline 4.19–4.72 µs; `f3970decacc97225` m = 16 (1 launch) 4.21–4.33 µs; `299133886d479017` 2 tokens (2 launches, baseline 5) 8.55–8.87 against 21.20–21.78 µs, ratio 2.44–2.51, mean speedup 2.470–2.474 against 5/2; `4801e216dcff290f` 8 tokens (2 launches) 8.33–8.51 µs; the floor slope measured beside them 4.029–4.066 µs; per-launch excess of the run means 0.181–0.237 (GEMM), 0.273–0.345 (GDN), 0.140–0.171 (router) µs, single runs up to 0.41; `d3bbbce5c0bbc61e` 2 tokens (4 launches) 19.00–19.17 µs (runs 18.85–19.34), 2.79–3.09 µs above 4 floors; gemm_n6144_k2048 (25.17 MB of weight) 5.94–5.98, 5.95–6.00, 6.04–6.09, 6.34–6.37 µs at m = 2, 4, 8, 16, and `06755ba3ee5eb085` at m = 64 6.24–6.36 µs for 15.74 MB moved, 2.48–2.52 TB/s (AMD-AGI/AKLR@43e112e)
- E37 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, AKLR CI bench (graph replay, 1 GiB read flush, Kineto kernel sum), stored speedup against the schema's baseline] byte-identical GDN prefill sources at n_tokens 16384, n_sequences 2, n_slots 4, n_starts 3: 1.444 in the max_seq_len 10000 scenario, 1.370 in the max_seq_len 8192 one, one run (AMD-AGI/AKLR@369a3c0 (CI run 41753978959):aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_prefill_group_fp8_quant_max_seq_len10000_n_v_heads16_key_dim128_value_dim128_conv_dim3072/3b99d19659746a36.json, aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_prefill_group_fp8_quant_max_seq_len8192_n_v_heads16_key_dim128_value_dim128_conv_dim3072/b8c54bde37ee77c7.json)
- E38 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, AKLR CI bench (graph replay, 1 GiB read flush, Kineto kernel sum), stored speedup against the schema's baseline] Qwen3-Next GDN prefill, a file with constexpr token and sequence counts against its twin with runtime counts at the same Point: 0.1% (162f/a99f 4096/1; 70f4/010e 13312/13 and 15360/15) to 2.4% (4489/d8f9 3072/3) apart over 11 Points, the constexpr file ahead at 6 and the runtime one at 5; one run, recorded on main in d908d01, the child of 43e112e (AMD-AGI/AKLR@d908d01 (CI run 42000899019):aklr/plugins/sglang/models/qwen3_next/impls/triton/)
- E39 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=4, a standalone Gluon LDS probe: 2048 one-wave programs read a 64 × 64 bf16 LDS tile 2048 times as an MFMA operand] XOR-swizzled over plain row-major time for operands 0 and 1: 0.584 and 0.567, 0.588 and 0.570 (532.9 and 511.7 against 906.8 and 898.5 µs), 0.583 and 0.571, 0.578 and 0.572 (#89)
- E40 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=4, in-process timing of gemm_n6144_k2048 at m = 64, weight hot or after a 1 GiB read flush] the source's tier (BLOCK_K 128, 8 warps, KROT 0) over the committed one: 1.054–1.082 flushed, 1.080–1.128 hot; committed tier 8.53–8.78 µs flushed, 6.99–7.05 µs hot (`f806ad2b086d757d`; AMD-AGI/AKLR@43e112e)
- E41 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3, standalone Gluon probes, 3 to 5 runs each] `v_mfma` per wave on a 32 × 256 tile 8 at 4 warps, 4 at 8; on a 32 × 32 tile 4 and 4; an LDS loop with 147,456 B per workgroup (one per CU), 256 programs: 276.2–277.2 µs at 4 warps, 267.0–267.4 µs at 8 (1.033–1.037×); two accumulators alternated on a chain of dependent bf16 [16, 16, 32] MFMAs, 1024 one-wave programs: 1.020–1.029× (499.5–504.7 against 509.3–519.4 µs, 5 runs), the accumulators alternating in 17 of 17 MFMAs; the 1 GiB stream of E12 at 3 MB in flight, measured ÷ predicted from its curve: 384 × 4 × U = 2 0.944–0.952, 768 × 2 × U = 2 0.938–0.949, 96 × 4 × U = 8 0.757–0.766, 192 × 1 × U = 16 0.471–0.472; 192 × 1 × U = 32 (6 MB) 1505–1521 GB/s (#89)
- E42 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, standalone Gluon kernels run on MI355X] raw `v_max_f32` of the signalling NaN 0x7FA00000 and 1.0, either order: 0x7FE00000 with IEEE mode on, 1.0 off; `gl.max` over a [256, 4] tile, four rows per thread: 12 `v_max_f32` on, 4 off; 1,000,064 random normal a·b + c: 23.53% differ fused against unfused, fused equal to the fp64 result rounded once; `gl.maximum` bit-identical with IEEE on and off (#89)
- E43 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, `aklr validate` (graph replay, 1 GiB read flush, Kineto kernel sum), 4 rounds; compiled VGPRs] at 32 rows the committed state load first 8.39 µs, moved after the first wait 10.28 µs (B/A 0.81), 160 VGPRs each, no spill, its other Points 0.997–1.006; early minus late state-load VGPRs at 1, 32, 128, 256 rows: 7, 0, 77, 12 (Qwen3-Next GDN decode template `f44007374038041f`; AMD-AGI/AKLR@43e112e)
- E44 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=4, AKLR `bench_cudagraph` at 43e112e (graph replay, 1 GiB read flush, Kineto kernel sum)] a plain Gluon `.cg` read stream, first run: 4.5 MiB 4.24 µs (1.11 TB/s), 14 MiB 4.55 µs (3.23), 16 MiB 4.57 µs (3.67), 25.17 MB 5.59 µs (4.50), 32.2 MB 6.97 µs (4.63), 32 MiB 7.06 µs (4.76); over the 4 runs 25–34 MB at 4.48–4.83 TB/s, 14–16 MiB at 3.16–3.72, 4.5 MiB at 1.04–1.11 (4.24–4.64 µs for the smaller streams); 64 MB not tested (#89; AMD-AGI/AKLR@43e112e)
- E45 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3, `aklr validate --stages audit,bench` on one GPU (graph replay, 1 GiB read flush, Kineto kernel sum); routing simulated from the schema distributions, 5 seeds] MiniMax-M2 MXFP4 E256 top 8, 64 tokens: experts touched 77.2% on average, 74.6% least, 495.3 MB of their weights; AKLR 99.09–101.7 µs (4.87–5.00 TB/s), baseline 127.0–127.8 µs (3.88–3.90 TB/s), speedup 1.249–1.285; floors (launches × the slope 4.03–4.07 µs, FLOP ÷ 4923 TFLOP/s, bytes ÷ 4.5 TB/s, 6.0 above 64 MB) against the call: the m = 16 GEMM's launch floor 94–96% of 4.25–4.27 µs, `1d277d76892ee692` at 256 tokens stream 80.22 of 103.20–104.92 µs (76–78%), the m = 4096 GEMM stream 8.04 and MFMA 7.85 of 25.36–25.38 µs (32%) (AMD-AGI/AKLR@43e112e)
- E46 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, `aklr validate`, the image's Python alternated with a venv holding CI's pinned aiter (`AKLR_BASELINE_AITER`), 3 rounds] the baseline (aiter `gemm_a16w16`), image / pinned: 7.02 / 7.23, 6.97 / 7.20, 7.02 / 7.95 (13.3%), 7.50 / 7.16 (4.5%), 7.58 / 7.65, 9.28 / 9.64 µs at m = 2, 4, 8, 16, 32, 64; the candidate within 0.1–1.7% (gemm_n6144_k2048 `f806ad2b086d757d`; AMD-AGI/AKLR@43e112e)
- E47 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, 7 A/B pairs, 5 alternated rounds per arm and timer] against validate's recorded B/A, eager hot, eager flushed and graph replay without the flush missed by more than 1% on 6, 4 and 5 pairs; AKLR's `bench_cudagraph` (Kineto kernel sum) came within 0.9% on 6, with m = 16 `PREFETCH` 5 → 3 at 0.9299 (A/B times 4.487/4.825 µs); another fixed input set moved the routed kernels' B/A by 0.04% (DeepSeek-V2 `d3bbbce5c0bbc61e`, 0.9984 → 0.9980) and 1.5% (MiniMax `1d277d76892ee692`, 0.9784 → 0.9635); the fastest of 8 bf16 gemm_n6144_k2048 configurations hot and flushed identical at m = 16 and 64, at m = 32 different by 0.8% with validate favouring the hot pick; at m = 64 the source's tier (BLOCK_K 128, 8 warps, KROT 0) over the committed one 1.079 flushed and 1.125 hot (AMD-AGI/AKLR@43e112e)
- E48 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, `aklr validate --stages audit,bench` on one GPU (graph replay, 1 GiB read flush, Kineto kernel sum), 4 alternated runs per arm] at 64, 128 and 256 tokens the fastest measured of half, committed and double split had 512, 256 and 256 four-wave programs; at 64 tokens double 8.356 against 8.414 µs, pairs 0.979–1.001, not resolved; at 256 tokens look-ahead 4 against 2: 9.583 against 9.883 µs (Qwen2 router; AMD-AGI/AKLR@43e112e)
- E49 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, cold-cache device-graph timing, 32 random balanced confirmation pairs per comparison, family-corrected 99% intervals with A/A and order controls] Point (M, N, K) = (64, 2624, 6144), bf16: split 1/2/3/4/6/8/12 × BN 16/32/64 with BM 64, BK 128, NXCD 8, KROT 0, 8 warps, 3 stages, `waves_per_eu` 2, MFMA 16; 3 input seeds pass the full-output contract and 100 bitwise repeats per configuration and seed, timing on seed 12011 only; the pilot picked BN 32 for splits 1/2/3 and BN 64 for 4/6/8/12; time over split 6: split 1/2/3/8/12 1.744/1.242/1.085/1.215/1.178, split 6 faster; split 4 1.095 (interval 1.068–1.123), short of the decision margin the A/A pilot set (about 6.9%), not resolved; whole minus partial 1.868 µs (interval 1.466–2.269) (gemm_n2624_k6144 split + reduce; AMD-AGI/AKLR@43e112e)
- E50 [2026-11, CPU only, Triton 3.8.0 and 3.7.0, N=1, source executed without a GPU] the scheduling AST of the Qwen2-MoE experts source `86987a66c0cdba39`: 16,384 nonzero configurations, 4,202,496 pass/cohort sets, no missing or duplicate work, worst pass-size spread 4080; zero-job and balance configurations not run on a GPU; on Triton 3.7.0 `HIPOptions` has no `llvm_fn_attrs` field and rejects the keyword (`TypeError`), and AKLR's requirement helper marks `triton >=3.8` unmet; a 3.7.0 launch not run (AMD-AGI/AKLR@43e112e)
- E51 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, four adjacent AB/BA-balanced pairs per candidate, paired-log 95% intervals widened by the order effect] fp8 GEMM `4d4c` (m = 64, committed `waves_per_eu=6`), depth 4 → 6: time ratio 0.9567, interval 0.92676–0.98761; with `waves_per_eu=0` (no spill) 0.94460–1.04020, unresolved; shallower: `9793` (m = 32) ratio 1.1923 (spill-free 1.1471), `4d4c` (m = 64) 1.7790 (spill-free 1.7716); MoE `1c17` (4 tokens) shallower / deeper 1.0368 / 1.0155, the conservative intervals showing no two-to-one asymmetry (AMD-AGI/AKLR@43e112e)
- E52 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, 6 counterbalanced rounds per arm (3 forward, 3 reverse), paired-log 99% intervals] Qwen3.5 gated attention decode at 16, 64 and 128 rows: half split call time +44.07%, +56.44%, +78.22%, double +3.257%, −2.690%, −21.82%; at 16 rows half with the head tile pinned +42.77%; every arm accurate; fastest tested grids 512, 1024 and 1024 four-wave programs; the double split's intervals 1.018381–1.046948, 0.970714–0.975498, 0.780215–0.783308; a conservative union bound sharing the A arm gives at least 98% joint coverage per group; look-ahead, prefetch and CU residency not measured (AMD-AGI/AKLR@43e112e)
- S1 aklr/core/validation/stages/_measure.py:60-92 the measurement profile: CUDA-graph replay, a 1 GiB read flush before each call in the graph, gpu_duration_sum from AKLR's kernel tracer (`kineto` before #33), Student-t lower bound, one trial task per device
- S2 aklr/core/validation/stages/_bench/_report.py:84-93 the stored speedup is the lower bound against the baseline, from the unclamped rounds (_measure.py:119-124)
- S3 aklr/core/utils/testing.py:63-75 the flush reads its 1 GiB buffer, "written lines would stay dirty for the next kernel", one store per program to `sums`
- S4 aklr/core/utils/testing.py:206-224 in `bench_cudagraph` each captured call follows its `flush_l2_cache()`, outside the call's profiler scope; a call's time is the sum of the scope's kernel durations (`_collect_scope_times`, :139-160) from AKLR's rocprofiler-sdk collector (`kernel_tracer`, :41-60), averaged over 10 replays; Kineto before #33
- S5 aklr/ops/gemm/impls/triton/gemm_n6144_k2048/f806ad2b086d757d.json#kernel.py:121-142 one launch for m ≤ 16: m runtime, the same constexprs at m = 2, 4, 8
- S6 triton-lang/triton@c01b677:python/src/specialize.cc:274-283 a runtime int is keyed by its width (i32 or i64), the value 1 (a constexpr) and divisibility by 16 (`D`)
- S7 aklr/ops/gemm/baseline.py:15-21 the gemm baseline is aiter's tuned gemm_a16w16
- S8 aklr/plugins/sglang/models/qwen2_moe/baseline.py:63-78 a plugin baseline reaches aiter through SGLang's path (tgemm.mm or aiter's Triton GEMM)
- S9 .github/workflows/ci.yaml:6-7 AKLR_BASELINE_AITER, installed over the image's aiter
- S10 aklr/ops/README.md:31-33 the uuid hashes the authored parts; export check names the expected path
- S11 aklr/plugins/sglang/models/qwen2_moe/impls/triton/qwen2_moe_router_model_dim4096_n_routed512_topk10/c3889614985b700a.json#kernel.py:27-34 split, tile and look-ahead per row count; :19-24 the `_ShapeConfig` fields; :113-129 the ring: AHEAD stages issued ahead, one refilled per step, DEPTH = AHEAD + 2 slots, a barrier after each wait
- S12 aklr/plugins/sglang/models/qwen2_moe/impls/triton/qwen2_moe_router_model_dim4096_n_routed512_topk10/c3889614985b700a.json#kernel.py:294 grid rows / 32 × 512 / expert_tile × split, four warps
- S13 aklr/plugins/sglang/models/qwen3_5/impls/triton/gated_attention_decode_group_fp8_quant_qkv_dim4608_n_kv_heads1_head_dim256/d9cbde237fd34de9.json#attention.py:358-379 splits per row count, grid (rows, splits), four warps; :116 `head_tile: gl.constexpr = 16 if SPLITS == 16 else HEADS`
- S14 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/f3970decacc97225.json#kernel.py:1001-1034 per-m branches: depth 5 at m = 16 and 32, grid m / 16 × n / 16 one-wave programs, waves_per_eu 3; the other Points' files: b11f8d1f0bba8fed.json#kernel.py:914 m = 1 `_narrow_small_gemm[(n // 4,)]`, one warp, 6232bdbd302e4ec2.json#kernel.py:979 m = 2 and 08bd966bd0c72914.json#kernel.py:852 m = 4 the same, a5c62e519ce3fe45.json#kernel.py:852 m = 8 `_packed_m8_gemm[grid_n]` with grid_n n // 16 (:826) and 2 warps (:859), 97933c3a190ede28.json#kernel.py:417 m = 32 `[(m // 16 * 192,)]` with prefetch 5 on the contiguous dense path (:47) and waves_per_eu 8 (:53-55), 4d4cb3c49f122deb.json#kernel.py:443 m = 64 `[(m // 16 * (n // 16),)]` with depth 4 under dense scales and waves_per_eu 6 (:439-458)
- S15 aklr/plugins/sglang/models/qwen2_moe/impls/triton/qwen2_moe_router_model_dim4096_n_routed512_topk10/4801e216dcff290f.json#kernel.py:208-216 8 tokens: (1, 8, 32) one-wave programs; 5f2049852cd84a76.json#kernel.py:229-235, a87882774fd41b18.json#kernel.py:259-266, cd47a5e858af6ab4.json#kernel.py:263-269 the 16-, 64- and 128-token grids
- S16 aklr/plugins/sglang/models/qwen3_5/impls/triton/gated_attention_decode_group_fp8_quant_qkv_dim4608_n_kv_heads1_head_dim256/f5ba496e04f8e989.json#attention.py:153-164 1 row: (rows, heads, 32 splits); 3d0f64ffff6790c8.json#attention.py:224-265 4 rows: 32 splits, (rows, splits); dfb1ab1cdd5bf263.json#attention.py:334-343 256 rows: 2 splits, splits × rows
- S17 aklr/plugins/sglang/layers/dsa_backend/impls/triton/dsa_sparse_mla_decode_n_heads64_kv_dim576_topk2048/2ee56a6dbabc48a6.json#kernel.py:584-617 head tile and splits from the token count, grid tokens × head groups × splits, 8 warps
- S18 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_decode_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072/2c2a09e220f2706c.json#kernel.py:312-322 1 row: value tiles of 8 rows, (tokens × 16, 16) programs; 4f257a3e566b20d1.json#kernel.py:206-211 8 rows: (tokens, heads) four-wave programs; fc261961f0ad572d.json#kernel.py:300-322 128 rows: one 16-wave program per (token, head), waves_per_eu 6 or 8
- S19 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_prefill_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072/c53b6d874d511c02.json#kernel.py:755 _delta_chunks[heads, 2, batch] with BV 64 (:507); 2ecece9650b95f71.json#kernel.py:1812 _chunk_recurrent[batch, heads, 128 // bv] with bv 32 (:1738); 382853eb770e30b3.json#kernel.py:1614 _state_scan[batch, heads, head_dim // state_value_rows] with 16 rows (:1539)
- S20 triton-lang/triton@c01b677:python/triton/experimental/gluon/_runtime.py:27-32 ttg.num-warps set from the launch option before layouts are verified
- S21 triton-lang/triton@c01b677:lib/Dialect/TritonGPU/IR/Dialect.cpp:3869-3881 a layout's warp count must equal the module's
- S22 triton-lang/triton@c01b677:python/triton/experimental/gluon/language/_semantic.py:650-654 gl.num_warps() returns the launch option as a constexpr
- S23 triton-lang/triton@c01b677:third_party/amd/backend/compiler.py:116-117 num_warps a power of two (default 4 at :78); :460 amdgpu-flat-work-group-size 1, 64 × num_warps
- S24 triton-lang/triton@c01b677:third_party/amd/backend/compiler.py:462-468 waves_per_eu becomes "amdgpu-waves-per-eu"="N,N"; 0 adds nothing
- S25 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/TritonGPUToLLVM.cpp:263-274 LDS is a zero-size external array, sized at launch
- S26 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_prefill_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072/2ecece9650b95f71.json#kernel.py:1778-1793 _build_chunk_tables_paired[chunks, heads // 2], one warp, waves_per_eu=2 whenever m > 4096; chunks = cdiv(m, 32) at :1736-1737
- S27 aklr/plugins/sglang/models/qwen3_next/impls/triton/qwen3_next_gdn_decode_group_fp8_quant_n_v_heads8_key_dim128_value_dim128_conv_dim2048/f44007374038041f.json#gdn_decode_config.py:91-96 "MEASURED AND LEFT OFF: setting it to 4 at M=16/64 does pin VGPRs at 128 … ~19% SLOWER than leaving them at 129/131 VGPRs and 3 waves. The allocator hits the target by spilling and rematerializing"
- S28 aklr/plugins/sglang/models/qwen2_moe/impls/triton/qwen2_moe_router_model_dim4096_n_routed512_topk10/c3889614985b700a.json#kernel.py:294-306 scheduler strategy per row count on the projection, amdgpu-ieee false on the routing launch
- S29 triton-lang/triton@c01b677:third_party/amd/backend/compiler.py:103-105 schedule_hint: "Experimental; right now no effect"; :107-110 llvm_fn_attrs "Experimental"; :479-481 applied last, after waves_per_eu (:467-468) and the denormal mode (:473-474: allow_flush_denorm selects preserve-sign, else ieee)
- S30 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_decode_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072/299133886d479017.json#kernel.py:130 Q held in LDS until state writeback, :136 beta near its use; 4f257a3e566b20d1.json#kernel.py:152 Q loaded after the state update; 20f63094b85c00a2.json#kernel.py:159 epilogue inputs early, SiLU deferred to :202; aklr/plugins/sglang/layers/dsa_backend/impls/triton/dsa_sparse_mla_decode_n_heads64_kv_dim576_topk2048/0bb2f7cbe49d1be3.json#kernel.py:557, :646 `loop_range(..., disable_licm=True)`; triton-lang/triton@c01b677:python/triton/language/core.py:3713-3715 disable_licm "often useful to avoid creating long liveranges within a loop", lowered at python/triton/compiler/code_generator.py:1299-1300
- S31 triton-lang/triton@c01b677:python/src/llvm.cc:139-140 enable_fp_fusion sets AllowFPOpFusion to Fast (passed at third_party/amd/backend/compiler.py:505)
- S32 triton-lang/triton@c01b677:third_party/amd/backend/compiler.py:326-348 gluon_to_ttgir runs no `num_stages`-driven pipeliner; its add_warp_pipeline (:339) only lowers `gl.amd.warp_pipeline_stage` blocks
- S33 triton-lang/triton@c01b677:third_party/amd/backend/compiler.py:285-301 plain Triton: schedule_loops and pipeline with num_stages, async copies on gfx950, block ping-pong when num_stages > 1
- S34 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_decode_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072/2c2a09e220f2706c.json#kernel.py:78-110 `_value_and_gates`: "Evaluate value SiLU, decay, and BF16 beta in neighboring lanes", one gl.exp over a [BV, 4] tile, one division; :214 "The 128-element reduction stays wave-local and needs no LDS exchange"
- S35 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n7168_k2048/06755ba3ee5eb085.json#kernel.py:160-166 M = 64: 3 slots, 2 ahead, a per-step wait table; ae4fa8a440b98f09.json#kernel.py:186-187 M = 32: 4 slots, 3 ahead
- S36 aklr/plugins/sglang/models/qwen3_5/impls/triton/gated_attention_decode_group_fp8_quant_qkv_dim4608_n_kv_heads1_head_dim256/6dd321aac1b1d85a.json#attention.py:53-86 "Stage 0 issues V before QK; stages 1/2 bracket the sum reduction"; :336-337 the stages per row count; :24-30 the load and MFMA layouts read `gl.num_warps()`
- S37 aklr/plugins/sglang/models/qwen3_next/impls/triton/qwen3_next_gdn_decode_group_fp8_quant_n_v_heads8_key_dim128_value_dim128_conv_dim2048/f44007374038041f.json#gdn_decode_config.py:76-83 state_prefetch early at M=1/32, late at M=128/256
- S38 triton-lang/triton@c01b677:python/triton/compiler/code_generator.py:1208-1216 static_range: the body is generated once per iteration
- S39 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/c56c343a3be1283d.json#kernel.py:524 `_loop_range(loop_end, loop_unroll_factor=1)`, imported at :6; triton-lang/triton@c01b677:python/triton/language/core.py:3701-3703 the factor drives "the Triton IR level loop unroller", "Less than 2 … implies no unrolling"; triton-lang/triton@c01b677:third_party/amd/backend/compiler.py:337 gluon_to_ttgir runs the loop unroller
- S40 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/a5c62e519ce3fe45.json#kernel.py:610-611, 647-650 two accumulators alternating by K product; f3970decacc97225.json#kernel.py:380-488 READ_AHEAD, EARLY_MFMA and REFILL_BEFORE, set per Point at :1001-1034; 08bd966bd0c72914.json#kernel.py:574-579 scales preloaded, live at m = 4 (:850-859)
- S41 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_mxfp4_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts64_inter_dim1536/1d277d76892ee692.json#kernel.py:711 WEIGHT_CACHE .cg except at 128 tokens (:603)
- S42 aklr/plugins/sglang/models/deepseek_v2/impls/triton/deepseek_v2_mxfp4_moe_shuffled_16x16_model_dim6144_n_routed256_inter_dim512/2422a87b3b814299.json#kernel.py:434 no .cg on the up projection above 128 tokens with 128-row tiles
- S43 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/aef7bbb04ccbe710.json#kernel.py:1094 group_m 2 or 4 by m; :630-638 the grouped tile order and the N interleave (`N_INTERLEAVE` 4 at M = 4096, :634)
- S44 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n7168_k2048/06755ba3ee5eb085.json#kernel.py:167-172 the m=64 program-id → tile permutation
- S45 aklr/ops/gemm/impls/triton/gemm_n6144_k2048/f806ad2b086d757d.json#kernel.py:116-120 `@functools.cache` `_launch_for(a_shape, b_shape)`, a launch-closure builder keyed by the shapes; :168 `schedule_hint="memory-bound-attention"` at m ≤ 32
- S46 aklr/plugins/sglang/layers/dsa_backend/impls/triton/dsa_sparse_mla_decode_n_heads64_kv_dim576_topk2048/0bb2f7cbe49d1be3.json#kernel.py:273 the dead profile's waves_per_eu (1 at 8 rows); :311-319 and :881-882 the live _launch_geometry (2); :892-903 the 8-token grid m × splits × head groups, 4 warps
- S47 aklr/plugins/sglang/models/qwen3_next/impls/triton/qwen3_next_gdn_decode_group_fp8_quant_n_v_heads8_key_dim128_value_dim128_conv_dim2048/f44007374038041f.json#kernel.py:482-484 decay and beta before the conv "costs enough registers to halve occupancy on the 4-warp instance"; 371905e46b19ee36.json#kernel.py:233-234 state and index pointers first, "shortening the dependency chain through the hardware-preloaded kernel arguments"; triton-lang/triton@c01b677:third_party/amd/backend/compiler.py:483-490 every argument marked for SGPR preloading
- S48 aklr/core/contract/semantics/_distributions.py:276-292 `BoundedCuSeqlensDistribution`: the first sequence takes `max_seqlen`, the others the rest
- S49 aklr/plugins/sglang/models/deepseek_v2/impls/triton/deepseek_v2_mxfp4_moe_shuffled_16x16_model_dim6144_n_routed256_inter_dim256/d3bbbce5c0bbc61e.json#kernel.py:497 gate-up grid m + routes × 4 K splits × 32 column tiles, routes = 9 per token (:479), one warp
- S50 triton-lang/triton@c01b677:third_party/amd/backend/compiler.py:522 metadata["shared"], the LDS bytes a launch asks for
- S51 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_decode_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072/20f63094b85c00a2.json#kernel.py:167-172 the state load issued before the gate loads
- S52 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n7168_k2048/0b1065ae9c0652b7.json#kernel.py:419-429 in `_packed_small`, partials queued in `pending`, folded DELAY groups later, drained after the loop; DELAY 3 at M = 4 (:691)
- S53 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_mxfp4_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts64_inter_dim1536/f82c87618c3b5cd4.json#kernel.py:1353 `loop_range(..., loop_unroll_factor=UNROLL)`; :1758-1764 UNROLL 6, one warp, max-ilp
- S54 harness/skills/aklr-kb/hardware/gfx950/cdna4/ch03_kernel_state.md:102 MODE.IEEE: signalling NaN inputs quieted and propagated per IEEE 754-2008, min and max included
- S55 aklr/plugins/sglang/models/qwen2_moe/impls/triton/qwen2_moe_router_model_dim4096_n_routed512_topk10/a87882774fd41b18.json#kernel.py:37-38 64 tokens: AHEAD 1, DEPTH 3; cd47a5e858af6ab4.json#kernel.py:38-39 128 tokens: AHEAD 2
- S56 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_mxfp4_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts256_inter_dim512/84116a96e535f174.json#kernel.py:917-920 `_mxfp4_policy_2` thresholds 1536, 3072, 5120; :933 and :949 the 6656 threshold; aklr/plugins/sglang/models/qwen3_next/impls/triton/qwen3_next_gdn_decode_group_fp8_quant_n_v_heads8_key_dim128_value_dim128_conv_dim2048/f44007374038041f.json#gdn_decode_config.py:51 `GdnDecodeConfig`, one per row count in `CONFIGS` (:187)
- S57 triton-lang/triton@c01b677:python/triton/runtime/jit.py:592-593 the binary cache key holds the specialization and str(options)
- S58 triton-lang/triton@c01b677:third_party/amd/backend/compiler.py:273 make_ttgir's accelerate_matmul pass takes matrix_instr_nonkdim
- S59 aklr/core/validation/stages/_bench/_stage.py:172-180 with --base, the base file served at the Point is the incumbent, held to retain, the candidate to entry against the baseline; :200 the record's baseline names the served file; :338-357 the timings are kept beside the bench result; aklr/core/validation/stages/_measure.py:287-291 every participant timed in each round, in a seeded random order; aklr/core/validation/stages/validation_scope.py:50-59 without paths the changed schema files are validated, and :73-84 untracked files count as changed
- S60 triton-lang/triton@c01b677:python/triton/compiler/compiler.py:466-468 a launch whose metadata.shared exceeds the device's limit raises OutOfResources for "shared memory"
- S61 aklr/core/contract/semantics/_compare.py:142-149 `runs_agree`: bitwise means every run's output byte-identical to the first's
- S62 aklr/plugins/sglang/models/qwen2_moe/impls/triton/qwen2_moe_router_model_dim4096_n_routed512_topk10/c3889614985b700a.json#kernel.py:269-271 the split-K planes summed in `_finish_routing`, launched at :300; aklr/plugins/sglang/models/qwen3_5/impls/triton/gated_attention_decode_group_fp8_quant_qkv_dim4608_n_kv_heads1_head_dim256/d9cbde237fd34de9.json#attention.py:381-396 the finish launches that merge the KV splits; aklr/plugins/sglang/models/deepseek_v2/impls/triton/deepseek_v2_mxfp4_moe_shuffled_16x16_model_dim6144_n_routed256_inter_dim256/d3bbbce5c0bbc61e.json#kernel.py:376-378 the gate-up K splits summed in `_activate_quantize`, launched at :516
- S63 aklr/core/utils/testing.py:24-38 `graph_kernel_timing_environment` drops `GPU_FORCE_QUEUE_PROFILING` ("Forced queue timing adds overhead and corrupts graph replay's heap below HIP 7.16"), entered by every runner process (aklr/core/validation/stages/_bench/_execution/_runner.py:525-528) since #33; before it the runner set the variable to "1"; aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/f3970decacc97225.json#kernel.py:1016 the one launch at m = 16; aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_decode_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072/299133886d479017.json#kernel.py:234, :242 its two launches; aklr/plugins/sglang/models/qwen2_moe/impls/triton/qwen2_moe_router_model_dim4096_n_routed512_topk10/4801e216dcff290f.json#kernel.py:213, :218 its two launches; aklr/plugins/sglang/models/deepseek_v2/impls/triton/deepseek_v2_mxfp4_moe_shuffled_16x16_model_dim6144_n_routed256_inter_dim256/d3bbbce5c0bbc61e.json#kernel.py:494-519 its four launches; aklr/plugins/sglang/models/qwen3_5/baseline.py:154, 162, 171, 183, 228 the GDN decode baseline's five calls: split, conv update, recurrence, gated norm, aiter's group quant
- S64 aklr/core/validation/stages/_bench/_report.py:52 `REGRESSED` (slower than the incumbent) is measured as "passed", a demotion; :111-151 a schema qualifies when any workload is correct and beats the baseline (:73-82), else it gets `no_qualifying_workload`
