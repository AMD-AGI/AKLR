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
kernels and which did not, and how those kernels configure each Point. Source references are at the end;
measured examples are finite, not universal optimum or equivalence claims. Compiler inspection: `debugging.md`;
loads, LDS and pipelines: `gluon/memory.md`.

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
writes, its buffer (vector loads, one store per program to a small sums tensor; S3). A call's time is the
sum of its kernels' Kineto durations, flush excluded (S4), 4–10% above an independent Kineto sum of the same kernels
replayed without the flush. Round counts, thresholds and the clamp:
`harness/skills/aklr-read-validation/references/ruler.md` ("the ruler page"). For tuning:

- **A bound, not a mean.** At the same round count, of two records with the same mean log ratio, the larger sample standard deviation of log ratios lowers the unrounded bound
  (S2): AKLR's `TimingRecord` stores 1.442 and 1.412 for 16 rounds at exp(mean log ratio) = 1.45 with sample standard deviations of the per-round log ratios of 0.01 and 0.05.
- **Compare speedup at the same Point.** A raw ratio is `baseline_time / candidate_time`; across Points
  both terms can change even if the candidate binary is shared. Stored speedup also includes confidence-bound
  adjustment and rounding. Report both times, not just the ratio.
- **Equal axes are not equal workloads.** One GDN prefill source stored 1.444 and 1.370 in one CI run on two scenarios
  of 16384 tokens in 2 sequences, split by `max_seq_len` as 10000 + 6384 and 8192 + 8192 (S48). Compare
  within one scenario's Point.
- **The measured trivial-launch slope under AKLR is about 4 µs**: 4.04–4.07 µs per added trivial kernel (4.05–4.08 without the queue profiling), torch kernels alike, with or without
  the `GPU_FORCE_QUEUE_PROFILING` the runner sets (S63); FlyDSL kernels:
  [../pitfalls/aklr_dispatch_floor.md](../pitfalls/aklr_dispatch_floor.md). A call's empirical launch-count budget is its launch count times
  that, the baseline's likewise: add + RMSNorm (hidden 6144) read 4.07–4.24 µs and its baseline 4.19–4.72 (4
  to 256 tokens), the m = 16 fp8 GEMM (1 launch) 4.21–4.33, the 2-token Qwen3.5 GDN decode 8.55–8.87 and the 8-token
  Qwen2 router 8.33–8.51 (2 launches each, S63): 0.14–0.35 µs per launch above the measured slope in the measured means, an estimated headroom at fixed launch count
  rather than a proven upper bound on an optimization. Where both sides sit on it, launch counts set the ratio (one Point tested): the GDN decode's baseline has five
  launches (S63), 21.20–21.78 µs, speedup 2.47 against 5/2. Read both sides' times in `report.json`.

Repeated runs and different CI jobs can move stored ratios by several percent without a source change.
For example, the short n3072/M16 fp8 GEMM moved about 2–3% across local validates. That is not a fixed noise
threshold: calibrate A/A under the same process, workload, temperature and order as the candidate comparison.

- Compare a candidate- Compare a candidate only with A runs interleaved with it (§2). A gap of a few percent between stored speedups of
  different CI runs or files cannot be read as a property of the kernels.
- **Leave a margin over the serving threshold** (`AKLR_MIN_SPEEDUP`, `aklr/ops/README.md` § Environment variables).
  A Point near 1.03 can cross the serving threshold after remeasurement; leave a measured margin,
  not one inferred from a different Point or old CI run.
- **The baseline's build is part of the ruler.** A baseline that reaches aiter, the `gemm` op's directly (S7), a
  plugin's through SGLang's path (S8), can move with the aiter build: the bf16 n6144 GEMM's baseline moved 13.3% at m=8
  from the image's aiter to CI's. CI installs the aiter commit `AKLR_BASELINE_AITER` over the image's
  (S9); install it before comparing a local ratio with CI's (`.github/README.md` § Install). A ratio measured with a different baseline build is not comparable to CI's stored result.
  Use paired incumbent time to evaluate a change, while retaining the baseline gate.
- **The yardstick.** Among files whose workloads qualify (including accuracy), dispatch serves at a fixed-size Point the file with the highest stored speedup that reaches
  `AKLR_MIN_SPEEDUP` (default 1.03; a 1.02 file alone was `not_qualified`), so a kernel counts per Point against the
  faster of the kernel main serves there and the baseline (`aklr/ops/README.md`: files sharing a name, § Environment
  variables; `../CONTRIBUTING.md` § Evidence); §2 step 3 times both in the same rounds.
- **Your own sweep is not admission.** Eager/hot/unflushed graph timers can rank implementations differently
  from AKLR's flushed Kineto-duration timer. Use sweeps to narrow the search, then `validate` to gate it.
  Routed inputs can also change a comparison; retain fresh-input accuracy coverage and state the measured workload.

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
   `python -m aklr validate --base origin/main --devices 0 --out <dir> <B>.json`. B's record holds `versus: "incumbent"` alongside the baseline timing. Check that the record's
   `baseline` names A. A regression can exit 0 when another Point qualifies: inspect demotions.
4. **Otherwise alternate A and B on one idle GPU**, A B A B … each:
   `python -m aklr validate --stages audit,bench --devices 0 --out ab/<A|B>_<i> <file>.json`, one `--out` directory
   per run (a reused one, the default `aklr_validation/` included, is overwritten by the next run). Do not run two
   validates on one GPU at once. Give arms separate `TRITON_CACHE_DIR` and report directories. Confirm that the
   edit reached the binary: code-generating changes should change ISA; ignored options are same-binary controls.
5. **Compare means.** In each run's `<out>/report.json` the stage with `stage_name: "bench"` holds one
   `workload_records[]` entry per Point (`axes` names it), whose `timings[]` entry with `versus: "baseline"` holds
   `log_ratio_mean` (its `exp` is the run's geometric mean of per-round ratios; the console prints the lower bound, lower under load: 0.80 for a
   mean of 0.857 ) and `candidate_milliseconds` (the call time per timing round: take the median); a
   reader: [debugging.md](debugging.md) §10. Average each arm's; report B/A of the speedup ratios and A/B of the call times (or label B/A call-time change separately).
6. **Use paired uncertainty, not range overlap.** Calibrate A-vs-A locally, counterbalance order, and use
   uncertainty of the paired mean/log ratio to judge a small effect. Overlapping samples do not imply no effect;
   a point estimate near 1 does not prove equivalence. State the equivalence band before testing it.
7. **Read B's accuracy verdict** (`acc ok` in the log): a faster variant can be wrong (`numerics.md`) and still exit 0,
   with `numeric_mismatch` demotions in `report.json`.
8. **Clean up, then gate.** Delete every copy you do not ship: `validate --base origin/main` without a path includes
   changed and untracked schemas in the audit; in schema-only changes it benches paths not present at the merge base, including new-uuid losing copies (S59). Ship a win as B's file replacing A's (CI then
   re-measures A's other Points too) or as a second file narrowed to the Points where it wins (dispatch served a file
   narrowed to P at P and the original at its other Points; `aklr/ops/README.md` § Workflow, steps 3 and 4). Then gate
   with `python -m aklr validate --base origin/main` (the ruler page).

## 3. What moved the ratio: the grid's parallelism and the prefetch depth

Representative Point-specific comparisons show why grid and depth should be tuned together. B/A below
is a speedup ratio; call-time increases are not the same percentage.

| Knob (Point, file) | Values: programs × waves per program | B/A | Call µs, A → B |
|---|---|---|---|
| KV splits, gated attention decode (32 rows, `d9cbde237fd34de9`) | **16: 512 × 4** → 8: 256 × 4 | 0.63 | 44.18 → 69.77  |
| | **16** → 32: 1024 × 4 | 0.95 | 44.10 → 46.56  |
| Split-K of the router projection (256 tokens, `c3889614985b700a`) | **4: 256 × 4** → 2: 128 × 4 | 0.75 | 9.91 → 13.29  |
| | **4** → 8: 512 × 4 | 0.97 | 9.97 → 10.25  |
| Router look-ahead (stages issued ahead), same Point | **2** → 1, 256 × 4 | 0.89 | 10.01 → 11.28  |
| `PREFETCH`, async K groups (128 K elements sharing one block scale; K = 1536 has 12) in flight per wave, fp8 GEMM (m=16, `083b493a1861424c`) | **5** → 3, 192 × 1 | 0.87 | 4.23 → 4.85  |
| | **5** → 7 | 1.00 | 4.29 → 4.29  |
| Gate-up K split with its scratch, grid and activation launch, DeepSeek-V2 MoE (2 tokens, `d3bbbce5c0bbc61e`) | **4: 2306 × 1** → 2: 1154 × 1 | 1.00 (0.996) | 19.20 → 19.28  |

Grids from S11–S14 and S49. In the gated-attention file the head tile follows the split count (`head_tile = 16 if SPLITS
== 16 else HEADS`, S13), so both changed arms also ran an 8-row head tile: those rows measure the two changes together
([attention.md](attention.md) §7). The router reads split, tile and look-ahead from one `@gluon.constexpr_function`,
`_shape_config(rows)`, in kernel (`M: gl.constexpr`) and launch alike (S11, S12; [authoring.md](authoring.md) §5).

What to do with it:

- **Measure both neighbours.** Try half/double splits and depth ±2, with legal coupled tile, scratch and
  merge configurations. The committed choice was fastest in some three-way comparisons, not all Points.
- **Deeper can help.** At n3072/M64 fp8 GEMM, depth4→6 reduced mean call time about 4.3%; its spill-free
  control remained unresolved. A universal “extra prefetch does nothing” or two-to-one loss rule is false.
- **Count grid work, not a universal optimum.** Router examples favored 256/512 four-wave programs;
  separate gated-attention Points favored 512/1024. Splits also change head tiles and merge width.
  CU count is a starting budget, not measured residency or a guarantee of the best grid.
- **Avoid oversplitting.** Independent work can improve utilization but adds partials, merge work and live state.
  Thousands of one-wave MoE programs can already supply enough parallelism; measure the full chain.
- **Merge a split in a fixed order.** Split output dimensions (N tiles, heads, value rows) first: they need no partials.
  A split of K or of the keys writes partials to scratch and adds them in a fixed order, in a merge pass or the next
  launch, not with float atomics: 65 of the 154 Triton schemas, the gated-attention and DSA files among them, compare
  `bitwise` across runs (S61); an atomic sum's order changes the bits ([numerics.md](numerics.md) §8).
- **Reduce in a launch the call already has**, as the three splits above do (routing, attention finish, activation launch: S62). A separate reduce adds device-graph cost (§1; the older launch-floor experiment read 4.2 µs at k = 6144, m = 64, 1.8 µs in a bare CUDA graph, [authoring.md](authoring.md) §7).
  At M64/N2624/K6144, a finite full-call split comparison favored split6 over several selected neighbours,
  while split6 versus4 remained unresolved. Its preallocated graph whole-minus-partial cost was about 1.9 µs,
  not intrinsic reduce time. Other M/N/K and vendor comparisons need their own legal configurations and ruler
  ([gemm/decode_gemm.md](gemm/decode_gemm.md) §6).
- **At a Point of several dispatch rounds** (prefill, M ≥ 1024) the knobs measured here moved little: tile order
  (`group_m` 4 → 1) 1.2–2.1% at m = 4096 in four runs of 4 alternated rounds (fp8 GEMM `5ea30df1`, §12), removing
  `waves_per_eu` 0.0–0.15% at 8192 tokens in two (MiniMax gate-up `51ae4994`). Tile by M:
  [gemm.md](gemm.md) §3, §7, [moe.md](moe.md) §2, [gluon/mfma.md](gluon/mfma.md) §4; occupancy: §8.

## 4. Why (inferred): bytes in flight set a decode kernel's bandwidth

For the stream curve below, byte budgets are in MiB (the original log labels say MB); the router and GEMM estimates later use decimal MB. On a Gluon stream of 1 GiB (`buffer_load`, `.cg`, 16 bytes per lane) the bandwidth followed the bytes in flight
(programs × waves × independent load instructions per lane × 1 KiB per wave-load), not the program count, for 256 to 8192
programs of 1 to 8 warps with up to 8 loads per lane (equal bytes within 10.4%): 256 × 4 warps × 8 loads read
1.07× of 2048 × 4 × 1 (7.1–7.2 TB/s), 4 MB in flight 5.7–6.3 TB/s; at 3 MB 192 one-wave programs × 16 loads read
2.45–2.47 TB/s, half of 384 × 4 × 2  ([gluon/memory.md](gluon/memory.md) §7). Up to 2 MB the rate grows
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

**Inference, not isolated cause.** Less grid parallelism or prefetch can reduce independent memory work,
but also changes registers, scheduling and occupancy. For a router, estimate bytes as programs × bytes/stage ×
stages ahead; for DMA, count actual committed copies and waits. A pure-stream curve can overpredict small,
one-wave grids, and extra split work can lose despite more estimated bytes in flight. Confirm with the actual
K loop, legal output checks and A/B rather than imposing a fixed byte threshold.

## 5. Small observed effects are not equivalence

Changes to scheduler strategy, buffer versus pointer access, `.cs`, `BANK_XOR` or `waves_per_eu` produced
similar times at several measured short-kernel Points. That does not prove no loss, exact equality or no effect
elsewhere. Near the process-dependent launch floor, retest at an above-floor Point and use paired uncertainty.
For example, GDN buffer/pointer cold-device comparisons depended on inputs despite similar earlier means.

LDS swizzling can matter in an LDS-bound loop without moving a DMA/prefetch-bound GEMM. Likewise, similar
async-copy and register-load stream rates do not establish equal throughput at low residency. Check the
actual memory path and stage boundaries before transferring a result ([gluon/memory.md](gluon/memory.md) §8).

**Coupled constants.** Splits 8 → 4 alone in the DSA 24-token file failed to build (`build_failed`: `NameError:
cache_shared is not defined`; its 64-head tile path exists only for 8 splits; [attention.md](attention.md) §4), and
`gate_fold` 4 → 1 alone in the MiniMax MXFP4 1-token file ([moe.md](moe.md) §7) crashed the compiler (`candidate_crash`,
exit −11: SIGSEGV in `make_amdgcn`, gone without the launch's `amdgpu-sched-strategy`; [debugging.md](debugging.md) §6),
on the measured compiler build. Read every constexpr consumer before changing a knob; change the
legal configuration as a unit, then validate each affected Point.

## 6. Grid size and occupancy

gfx950 has 256 CUs (8 XCDs of 32); this page counts a grid in workgroups per CU (256 is one per CU). A
dispatch round is the workgroups the CUs hold at once, set by registers and LDS (`../gemm/grid_pricing.md` §1, §4; an
under-filled grid's symptoms: `../patterns/low_cu_utilization.md`). If registers and LDS permit one or two workgroups per CU, a grid of that density can bound residency: 256 four-wave programs supply an average of one wave per SIMD across 1024 SIMDs, and 512 supply two; actual placement and concurrent residency were not measured.
Below one per CU the m = 16 GEMM's 192 one-wave programs lean on each wave's depth (§3, §4). Grids per Point:

| Kernel | Points | How the grid is formed | Programs × waves |
|---|---|---|---|
| Qwen2 router projection | 8 to 256 tokens | split-K × expert tiles × token tiles held at 256: split 8 at 8 to 64 tokens, 4 at 128 to 256 | 256 × 1 at 8 and 16 tokens, 256 × 4 at 64 to 256 (S12, S15) |
| Qwen3.5 gated attention partials | 1 to 256 rows | (row, head, split) at 1 to 2 rows; (row, split) from 4 rows; rows × splits = 512 from 16 rows | 256 and 512 at 1 and 2 rows; 128 at 4; 256 at 8; 512 × 4 from 16 rows (S13, S16) |
| DSA split, 64 heads | 8 to 32 tokens | tokens × head groups × splits; 144 to 160 KiB of LDS, so one workgroup per CU (160 KiB is the most a workgroup may take, `../hardware/gfx950/memory_and_sync.md`) and grid size / 256 is a CU-coverage upper bound, not measured wave occupancy | 256 × 4 at 8 tokens; 192 to 256 × 8 at 12 to 32 (S17, S46)  |
| Qwen3.5 GDN decode recurrence | 1 to 256 rows | value-row tiles per head at 1 to 4 rows, one program per (token, head) from 8 rows | 256 at 1, 2, 4, 16 rows; 128 at 8; 512 at 32; 1024 to 4096 × 16 at 64 to 256 (S18) |
| Qwen3.5 GDN prefill, `gdn_prefill_group_fp8_quant_n_v_heads16…` (5 files), sequential phase | one to 8 sequences | value block chosen so sequences × 16 heads × 128 / BV reaches 256 at the sequence count the file is written for (8, 4 or 2) | 256 at the several-sequence Points except 160 at five sequences and 192 at three; 128 at one sequence (S19) |
| fp8 block-scaled GEMM | m = 1 to 64 | one narrow N tile per program | 768 × 1 (m ≤ 4), 192 × 2 (8), 192 / 384 / 768 × 1 (16 / 32 / 64) (S14); N=7168: 448 × 1 to 224 × 4 (`gemm.md`) |

- **Across the 15 GDN prefill files the sequential phase runs 128 to 512 programs, and the stored speedups are not monotonic
  in program count**: 128 1.188–2.638 (seven one-sequence Points), 160 0.997, 192 1.029, 256 1.339–2.629, 384 1.198,
  512 1.889. A cost of under-filling is not measured here.
- Qwen2 experts schedule a device-counted job list using route tiles and cohorts. Coverage of nonzero
  jobs does not establish balance or speed; test zero jobs and skewed counts before generalizing the schedule.

## 7. `num_warps`

- **It is part of the layouts.** Gluon sets the module's warp count from the launch option before it verifies the
  layouts, and rejects a layout whose `warps_per_cta` multiplies to another count (S20, S21). Changing `num_warps` at
  launch is a code change unless the layouts read `gl.num_warps`, a constexpr of the launch option (S22), as the
  gated-attention files do: `gl.BlockedLayout([1, 16], [4, 16], [gl.num_warps(), 1], [1, 0])` (S36).
- It must be a power of two, defaults to 4 and sets the work-group size LLVM schedules for (S23).
- **Choosing it.** One wave when a program's work fits a wave (a GEMV row, a per-token selection): the work-group then
  has 64 lanes and LLVM drops `s_barrier` (0 in the one-wave m=16 GEMM, 16 in the four-wave router projection ). More waves when they split an MFMA tile (8 warps halve each wave's `v_mfma` on a 32 × 256 tile, not a 32 × 32
  one; `gluon/layouts.md`); a 144 KiB-LDS loop at one workgroup per CU ran 3–4% faster at 8 warps than 4 (3 runs)
. Warps in 897 compiled AKLR kernels: 4 in 514, 1 in 276, 2 in 76, 8 in 27, 16 in 4.
- **More than 4 warps caps registers.** A workgroup's ceil(num_warps / 4) waves share each SIMD, so a wave keeps at
  most 512 / ceil(num_warps / 4) registers whatever `waves_per_eu` says: a plain-Triton 256 × 256 × 64 bf16 tile
  compiled to 482 VGPRs at 4 warps, 256 with 100 spilled at 8, 128 with 131 spilled at 16. Keep a reduced axis inside
  each wave: a row sum over an axis spread across 4 warps compiled to an LDS write, an LDS read and an `s_barrier`, and
  to none with whole rows in each wave (all compile with Triton 3.8.0 for gfx950), as in the GDN decode's "128-element
  reduction stays wave-local and needs no LDS exchange" (S34).
- **What the committed kernels launch** (compiled kernels of the census; per family:
  [kernels/kernel_library.md](kernels/kernel_library.md) §5–§14): one wave for the fp8 GEMM at N=3072 m ≤ 64 except m=8
  and at N=7168 M ≤ 8 (10 Points); 8 in 27 kernels: DSA 4, Qwen3-Next GDN decode 9, MiniMax-M2 MoE 8, Qwen3.5 GDN
  prefill 4 (`_chunk_boundaries`, `_propagate_chunks`), Qwen2-MoE experts 1, bf16 GEMM 1; 16 for the Qwen3.5 GDN decode
  recurrence from 64 rows and the Qwen2 router projection at 2232 tokens (`4c2bed7e3e8c1e94`). Router selection and
  decode GEMVs: not checked. No Point has two committed files that differ only in `num_warps`, so no committed pair
  isolates the warp count's effect: when you vary it, vary the layouts with it and A/B it.

## 8. `waves_per_eu`: a register cap

**What it does.** A nonzero value adds the LLVM attribute `"amdgpu-waves-per-eu"="N,N"`; 0, the default, adds nothing
in the PyPI Triton 3.8.0 (S24; the image's build writes `"0,0"`). On gfx950 a wave may hold 512 registers, arch and
accumulation together (`../hardware/gfx950/lookup.md`; 482 compiled unspilled); N waves per SIMD cap each wave at
512 / N and spill the rest: a 4-warp bf16 MFMA probe compiled at 0, 1, 2, 4, 8 used 438, 438, 256, 128, 64 VGPRs with
0, 0, 181, 353, 558 spilled. LLVM does not see the LDS, which Triton allocates at launch (S25; all 897
AKLR kernels of the census report a zero static LDS segment ), so the cap applies whatever the LDS
allows: a 4-warp kernel whose 64 KiB of LDS allow 2 waves per SIMD was still capped from 170 to 128 VGPRs at 4, spilled
58 and ran 1.43× slower. Take a kernel's LDS from its `metadata.shared` (S50), not from the ISA.

**Below the workgroup's own count it need not tighten the register budget; at or above it, it can spill a kernel that fits.** A workgroup of
`num_warps` waves already puts ceil(num_warps / 4) on each SIMD. A plain-Triton 256 × 256 × 64 bf16 tile on 8 warps
compiled to the same code at `waves_per_eu` 1 as with none (256 VGPRs, 128 AGPRs, 100 spilled); on 4 warps a
128 × 256 × 64 tile used 218 VGPRs unspilled, and 256 with 100 spilled at 2 (both compile with Triton 3.8.0). Compare
both builds' spill counts, not only their VGPRs (the GDN prefill row below).

**On committed kernels** (VGPRs = arch + AGPR, compiles with Triton 3.8.0 for gfx950; the census, compiled on the GPU
node, holds the same counts for the committed settings, 88, 228, 256 (64 spilled), 190, and no unset build ):

| Kernel (Point) | Warps | Setting | VGPRs with it | VGPRs unset | Measured |
|---|---|---|---|---|---|
| fp8 GEMM `_private_k128_gemm` (m=16) | 1 | 3 (cap 168) | 88 | 84 | small observed difference; not equivalence (§5) |
| DSA `_attention_split` (24 tokens) | 8 | 2 (cap 256, which 8 warps impose anyway) | 228 | 222 | small observed difference; not equivalence (§5) |
| Qwen3.5 GDN prefill `_build_chunk_tables_paired` (4097 tokens) | 1 | 2 (cap 256) | 256, 64 spilled, 260 B scratch | 322, no spill | unset 1.04× faster: 281.2 → 270.3 µs, arms separated  |
| same kernel (8192 tokens) | 1 | 2 | 190 | 183 with PyPI Triton 3.8.0, 184 with the image's build  | B/A 1.00 (0.9985)  |

The GDN prefill file sets 2 for both of its Points (S26): it fits at 8192 tokens and spills at 4097, where the grid
(1032 one-wave programs, 129 chunks × 8 head pairs) cannot place 2 waves on most SIMDs anyway. That Point stores 1.029
; without the cap the call ran 3.9% faster (281.2 → 270.3 µs, speedup 1.017 → 1.057, the per arm
not overlapping: 1.014–1.019 against 1.054–1.064); at 8192 tokens removing it gave a near-unity observed ratio (0.999), not proof of no effect.

When to try it:
- the grid has several rounds of programs: on a one-wave streaming kernel of 264 VGPRs (168, 96 spilled, under 3) the
  cap made 8192 programs 1.50× faster (1376 → 917 µs) and 256 programs, under one wave per SIMD, 13% slower (70.3 → 79.8
  µs). The LDS-full case (the DSA split's 160 KiB) is untested;
- the natural count sits just above a tier (512 / N rounded down to 8, as compiled: 256, 168, 128, 96, 80, 72, 64). Even
  a small cut can cost: the Qwen3-Next decode template (not measured at 43e112e) compiles to 138 and 131 VGPRs at 16 and
  64 rows; forced to 4 on the committed file, both kernels compiled to 128 VGPRs, spilling 9 and 20 (2.6%, 5.3% more
  instructions), and the call ran 1.60× slower at both (speedup ratio 0.62–0.63; its comment says ~19%, S27). Compare spill and runtime effects at every affected Point; a budget hint can help one grid and hurt
  another. Do not infer a universal benefit from occupancy arithmetic or comments.

Read the compiled count firstRead the compiled count first (`debugging.md`) at every Point the setting applies to. The committed kernels set 1 to 8
on the fp8 GEMM's decode Points but none at N = 7168, M = 2 (N = 3072, m = 1 to 64: 8, 4, 4, 2, 3, 8, 6), 2 on every
bf16 GEMM tile kernel (not the split-K reduce), 6 and 8 on the 16-wave GDN decode recurrence, none in any DeepSeek-V2
MoE or Qwen3-Next file ([kernels/kernel_library.md](kernels/kernel_library.md) §5–§14): each Point's search result.

## 9. Launch options and attributes

| Option | What it changes in the binary (Triton 3.8.0, gfx950) | Measured on a committed kernel |
|---|---|---|
| `llvm_fn_attrs=[["amdgpu-sched-strategy", s]]` (`max-ilp`, `iterative-ilp`, `iterative-minreg` in AKLR's files, switched per Point: S28); declare `requires: {"triton": ">=3.8"}` when using this 3.8 API | LLVM's scheduler strategy; the VGPR count moves by kernel: on a 256 × 128-tile bf16 MFMA loop (8 warps, 16 steps of K = 32) `max-ilp` 145, `iterative-ilp` 148, `iterative-minreg` 88 against 134 unset. The option is "Experimental" (S29) and applied after `waves_per_eu` and the denormal mode, overwriting them: `[["amdgpu-waves-per-eu", "1,1"]]` on `waves_per_eu=4` leaves only `"1,1"` and 134 VGPRs instead of 128 with 10 spilled, and `[["denormal-fp-math-f32", "preserve-sign"]]` replaces the default `ieee` (all compile with Triton 3.8.0 for gfx950) | removed from the router at 256 tokens: B/A 1.00  |
| `llvm_fn_attrs=[["amdgpu-ieee", "false"]]` (Qwen2 routing launches) | the kernel descriptor's `.amdhsa_ieee_mode` goes from 1 to 0, so MODE.IEEE (S54) is off: with it on, a raw `v_max_f32` of a signalling NaN and 1.0 returned a quiet NaN in either order, with it off 1.0; and a max no longer canonicalizes its operands: `gl.max` over rows of 4 elements, four rows per thread, went from 12 to 4 `v_max_f32` (3 → 1 per row, one `v_max3_f32` per row either way; compiles with Triton 3.8.0 for gfx950) | removed from the router's routing launch at 256 tokens: B/A 1.01 (1.00 in a second run), 9.99 → 9.95 µs, accuracy still ok  |
| `allow_flush_denorm=True` (on the split and merge launches of four of the five DSA 64-head files, 12 to 32 tokens; not the 8-token one) | `denormal-fp-math-f32` becomes `preserve-sign` in the LLVM IR (S29), but the gfx950 code does not change: the same instructions and `.amdhsa_float_denorm_mode_32` 3 with and without it, on a kernel of exp, log, division, sqrt, exp2 and a multiply-add (compiles with Triton 3.8.0 for gfx950; [numerics.md](numerics.md) §6) | removing it at 12/16/24/32 tokens gave B/A 1.0007/1.0033/0.9986/0.9970; all compared arms passed accuracy  |
| `enable_fp_fusion` (default on; `False` on at least one launch in 42 of the 154 kernel directories, chosen per Point in 4 more, 3 of which turn it off at one of their Points: `96e51c1f85a71b7e` at 16 tokens, `61ed32f62fa797df` and `1d5568a78971fb62` at 512) | on: LLVM may contract a multiply and an add into one FMA, one rounding (S31); off: `v_pk_mul_f32` + `v_pk_add_f32` instead of `v_pk_fma_f32` (compiles with Triton 3.8.0 for gfx950); 23.5% of 10^6 random `a * b + c` differed fused against unfused  | off → on at DeepSeek-V2 inter_dim 512, 8 tokens: same gfx950 code on all 3 launches; B/A 1.00, 56.27 → 56.34 µs, accuracy still ok  |
| `num_stages` | nothing in Gluon: `gluon_to_ttgir` has no `num_stages`-driven pipeliner (S32); a Gluon loop compiled at 1 and 3 gives identical code. In plain Triton it drives the software pipeliner (S33): a 128 × 128 × 64 bf16 GEMM with 8 warps issued 0, 8 and 12 direct-to-LDS loads at 1, 2 and 3; the block ping-pong pass, added above 1, changed that GEMM only at 3 (2 `amdg.cond_barrier`, no `s_setprio`; compiles with Triton 3.8.0 for gfx950) | — |
| `schedule_hint` | nothing: "Experimental; right now no effect" (S29); identical code with and without (compiles with Triton 3.8.0 for gfx950). The n6144 decode GEMM passes it (S45); Triton 3.7.0 turned `schedule_hint="memory-bound-attention"` into the `iterative-ilp` scheduler ([gemm.md](gemm.md) §9): on 3.8.0 `llvm_fn_attrs=[["amdgpu-sched-strategy", "iterative-ilp"]]` sets the same attribute | — |
| `matrix_instr_nonkdim` | plain Triton only: the dot lowering picks the MFMA with it (S58); Gluon picks the instruction in the layout (`gluon/mfma.md`) | — |
| `waves_per_eu`, `num_warps` | §8, §7 | §5 |
| any option, even one that compiles to nothing | enters the JIT's in-memory cache key before backend normalization (S57): `kpack=2` after `kpack=1` creates another raw-options entry, but gfx950 normalizes both to 1, so this does not prove a second distinct compiled binary; flipping back reuses the first entry | — |

`enable_fp_fusion` changes rounding and `amdgpu-ieee` the NaN result of a raw `v_max_f32` (`gl.maximum` stayed
bit-identical on 10^6 finite pairs) : decide both with `validate`'s accuracy check and `numerics.md`.

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
  against 8.39 at equal VGPRs.
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
    alternate two accumulators by K step (1.02–1.03× on a bare chain of dependent bf16 MFMAs ),
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

In on a pure 1 GiB stream, loads with `.cg` or `.cs` (the `nt` bit) ran 13–14% faster than unhinted ones, and no
hinted store beat the unhinted one by more than 0.74% (`.cs` lost 5.5–7.2%); the bits each string sets,
the keyword of each call (`cache=` on the buffer ops, `cache_modifier=` on `gl.load` / `gl.store`; `gl.load(cache=)`
raises a `TypeError`) and the per-path rates: [gluon/memory.md](gluon/memory.md) §6. After a 1 GiB Gluon read flush `.cg` gained +3.9–7.4% at 64 MB, +9.0–10.4% at 128, +5.9–8.8% at 192, +10.8–11.5% at 512 and +9.9–11.7% at 1 GiB (eager-event timer).
The gain was not monotonic with size; a stream result does not establish the best hint in a fused call.
In real kernels the hintIn real kernels the hint
moved the call 0.6 to 3% at 256 tokens and about 1% at 2 tokens, 2.8–3.1 µs above its floor (§1) :

| Change | Point (file) | B/A | Call µs, A → B |
|---|---|---|---|
| `.cg` removed from the weight loads | MiniMax MXFP4 , 256 tokens (`1d277d76892ee692`) | 0.97, all 4 B runs below all 4 A runs; its 128-token Point, unchanged, 1.00 | 104.8 → 107.9 |
| `.cg` added where the author left it off | DeepSeek-V2 MoE inter_dim 512, 256 tokens (`2422a87b3b814299`) | 0.99, all 4 B runs below all 4 A runs | 226.4 → 228.0 |
| `.cg` removed from the gate-up weights | DeepSeek-V2 MoE inter_dim 256, 2 tokens (`d3bbbce5c0bbc61e`) | 0.99, all 4 B runs below all 4 A runs | 19.09 → 19.26 |

Both 256-token files choose the hint per Point (S41, S42); there and at 2 tokens the committed choice was the faster
: A/B weight loads with and without `.cg` at each Point; stores: unhinted unless an A/B says otherwise.

## 12. Program-id remapping for XCD and L2 reuse

The measured MI355X probe grids placed linear workgroup ids round-robin on 8 XCDs, each with its own L2
([../gemm/schedule_and_locality.md](../gemm/schedule_and_locality.md) §2). Congruent ids helped a multi-reader
probe, but that observation is not a portable scheduler contract or proof of L2 causation in every kernel.
At n3072/M4096, removing grouped tile order cost about 2%; n7168/M64 was compatible with a ±1.5% band,
while M2 changed measurably. Measure the actual Point, model both A and B locality, and prove bijectivity for
**every** launched grid size ([gemm/decode_gemm.md](gemm/decode_gemm.md) §3).

## 13. Budget a Point before designing: matrix-core and memory rates

Use four estimates to choose the next experiment, not to certify a bottleneck:

1. **Matrix core:** FLOPs divided by a matched instruction/operand microbenchmark rate. Register-resident
   gfx950 probes reached roughly 2.45 PFLOP/s for bf16, 4.9 for e4m3 scaled MFMA and 9.4–9.6 for e2m1.
   These are probe rates, not guaranteed GEMM throughput. Low grid-wave supply can reduce utilization.
2. **Memory:** unique bytes divided by measured-size, matched-cache stream throughput. Under AKLR's
   timer, 25–34 MB streams reached about 4.5–4.8 TB/s; a 4.5 MiB stream was close to the short-kernel floor.
   Do not apply the 1 GiB streaming rate to tiny weights or infer physical traffic from logical bytes.
3. **Latency/parallelism:** count independent loads, committed groups and waits (§4), then sweep legal
   grid/depth neighbours. Bytes-in-flight arithmetic does not isolate scheduling or register effects.
4. **Launch/ruler:** compare the call's launch count with locally measured trivial-kernel costs (§1).
   A near-floor call may benefit from fewer launches, but full-call fusion can introduce repeated work.

A weight-bytes/time estimate is not a hardware counter. Match the regime and ruler, inspect ISA and actual
call stages, then A/B the candidate. Compute-dominated, stream-dominated and under-filled Points need different
experiments; no paper roofline gap is proof that its entire headroom is reachable.

## 14. Per-Point configuration: how the files write it, and what changes as M grows

**How it is written.** With shapes as constexprs each Point compiles its own binary on its first call; a runtime int
shares one across the Points of one specialization class (§1). The configuration is Python; of the 154 files, 131
branch on the token count (`if m == …` or a threshold) to choose a kernel and its constexprs (S14), 71 use dataclass or NamedTuple
configuration records, such as a MiniMax MXFP4 prefill policy switching at 1536, 3072, 5120 and 6656 tokens and a Qwen3-Next
decode config per row count (S56), 51 contain a `@gluon.constexpr_function`; some helpers are shared by kernel and launch code (§3), 11 a
`functools` cache, such as the plain-Triton GEMMs' launch builder keyed by the shapes (S45). How each keys Triton's
binaries: [authoring.md](authoring.md) §5. Constexpr against runtime token counts was measured once: the Qwen3-Next
prefill twins, first measured on main in d908d01, read 0.1–2.4% apart over 11 Points in that CI run, the constexpr
file ahead at 6 and the runtime one at 5  ([gdn.md](gdn.md) §8).

**Only the branch your Point takes is evidence.** Generated files keep tables for sizes they do not serve, and the
tables disagree: in the 8-token DSA file `_decode_profile_m2(8)` gives `waves_per_eu` 1 but serves only other token
counts, while `_launch_geometry(8)` gives the 2 that the `_attention_partials_m8` launch passes (S46) and its compiled
producer carries (`"amdgpu-waves-per-eu"="2,2"`, Triton 3.8.0). Resolve the branch before you copy a constant.

**What changes as M grows**, by family (details per family in `gemm.md`, `moe.md`, `attention.md`, `gdn.md`):

| Family | As the Point grows | CI speedup |
|---|---|---|
| fp8 block-scaled GEMM, N=3072 (15 files, one per Point) | m ≤ 64: one- or two-wave programs over narrow N tiles, 4 of the 12 K groups to all of them in flight; 128 to 256: four-wave tiles with a 4-slot LDS ring; ≥ 1024: four-wave tiles, single-buffered (64 rows at m = 1024, 128 at 8192) | 1.307–1.435 at m ≤ 64; 1.199, 1.477 at 256, 128; 1.191–1.486 from 1024  |
| fp8 block-scaled GEMM, N=7168 (9 files) | 448 one-wave programs at M ≤ 8; 224 programs of 2 to 4 waves at 16 to 64; four-wave at 128 and 256 | 1.664–1.840 at M ≤ 32; 1.488 at 64; 1.114–1.138 at 128 and 256  |
| bf16 decode GEMMs, plain Triton (2 files, 10 Points) | one N tile per workgroup over all of K; split-K plus a fixed-order reduce only at n=2624 m=64 (`gemm/decode_gemm.md`) | 1.048–1.292  |
| MoE layers: DeepSeek-V2, MiniMax-M2 FP8 and MXFP4, Qwen2-MoE experts | 3 or 4 launches at 1 to 4 tokens; 3, 5 or 6 at 8 to 16; 5 to 7 at 32 to 256; 6 to 8 at ≥ 1024, where at least 26 of the 32 Points launch a kernel that allocates LDS (`moe.md`) | 1.097–4.115 at ≤ 16 tokens (34 Points); 1.030–1.275 at 32 to 256 (23); 1.152–1.785 at ≥ 1024 (32)  |
| DSA sparse MLA decode | 64 heads: head tile and split count chosen together to keep 192 to 256 programs; 8 heads: split at 32 and 128 tokens, one program per token without a merge at 256 | 64 heads 1.030–2.188; 8 heads 1.050–1.196  |
| Qwen3.5 gated attention decode | FMA dot products per (row, head, split) at 1 and 2 rows; all 8 heads in one MFMA tile from 4 rows; splits shrink as rows grow (rows × splits = 512 from 16 rows) | 2.182 at 1 row; 1.671–1.803 at 2 to 32; 1.401, 1.204, 1.066 at 64, 128, 256  |
| Qwen3.5 GDN decode | each head's state split over programs at 1 to 4 rows; one program owns a (token, head) state from 8 rows, 16 waves from 64 | 2.336–2.507 at ≤ 16 rows; 1.282–1.651 at 32 to 256  |
| Qwen3.5 GDN prefill | value block of the sequential phase chosen for the sequence count the file is written for (§6) | 0.997–2.638  |
| Qwen3-Next GDN decode and prefill | per-size files plus a nine-row template (decode); constexpr and runtime-count twins (prefill) | not measured at 43e112e |

Of the MoE layers' three ranges in the table, the 32 to 256-token one is the lowest. Why is not measured;
the decomposition changes there ([moe.md](moe.md) §2). A weight-bytes/time estimate leaves a bandwidth hypothesis to test, not proof of a stream limit: at 64 tokens on
MiniMax-M2 MXFP4 (one Point) the bench's routing touches about 77% of the 256 experts (simulated), whose weights over
the layer's time give 4.9–5.0 TB/s for AKLR and 3.9 for the baseline (estimated, no trace).

**Applicability.** Per-Point configurations are starting designs, not global optima. Resource budgets do not
prove residency or a floor-reaching rule; source job coverage does not prove balance. Keep new Point, backend,
zero-job and toolchain changes in the validation scope instead of borrowing another Point's conclusion.
## Evidence

AKLR source paths refer to main `43e112e` unless another commit is stated.
Triton compiler paths refer to `triton-lang/triton@c01b677` (v3.8.0).
Measurements in the text are gfx950 examples, not universal performance guarantees.

- S1 aklr/core/validation/stages/_measure.py:66-94 the measurement profile: CUDA-graph replay, a 1 GiB read flush before each call in the graph, gpu_duration_sum, Student-t lower bound, one trial task per device
- S2 aklr/core/validation/stages/_bench/_report.py:84-93 the stored speedup is the lower bound against the baseline, from the unclamped rounds (_measure.py:119-126)
- S3 aklr/core/utils/testing.py:49-56 the flush reads its buffer: "written lines stay dirty, and the next kernel pays their write-back"
- S4 aklr/core/utils/testing.py:111-149 a replay's time is the sum of its kernels' Kineto durations, the flush kernel excluded
- S5 aklr/ops/gemm/impls/triton/gemm_n6144_k2048/f806ad2b086d757d.json#kernel.py:121-142 one launch for m ≤ 16: m runtime, the same constexprs at m = 2, 4, 8
- S6 triton-lang/triton@c01b677:python/src/specialize.cc:274-283 a runtime int is keyed by its width (i32 or i64), the value 1 (a constexpr) and divisibility by 16 (`D`)
- S7 aklr/ops/gemm/baseline.py:15-21 the gemm baseline is aiter's tuned gemm_a16w16
- S8 aklr/plugins/sglang/models/qwen2_moe/baseline.py:58-68 a plugin baseline reaches aiter through SGLang's path (tgemm.mm or aiter's Triton GEMM)
- S9 .github/workflows/ci.yaml:6-7 AKLR_BASELINE_AITER, installed over the image's aiter
- S10 aklr/ops/README.md:31-33 the uuid hashes the authored parts; export check names the expected path
- S11 aklr/plugins/sglang/models/qwen2_moe/impls/triton/qwen2_moe_router_model_dim4096_n_routed512_topk10/c3889614985b700a.json#kernel.py:27-34 split, tile and look-ahead per row count; :19-24 the `_ShapeConfig` fields; :113-129 the ring: AHEAD stages issued ahead, one refilled per step, DEPTH = AHEAD + 2 slots, a barrier after each wait
- S12 aklr/plugins/sglang/models/qwen2_moe/impls/triton/qwen2_moe_router_model_dim4096_n_routed512_topk10/c3889614985b700a.json#kernel.py:294 grid rows / 32 × 512 / expert_tile × split, four warps
- S13 aklr/plugins/sglang/models/qwen3_5/impls/triton/gated_attention_decode_group_fp8_quant_qkv_dim4608_n_kv_heads1_head_dim256/d9cbde237fd34de9.json#attention.py:358-379 splits per row count, grid (rows, splits), four warps; :116 `head_tile: gl.constexpr = 16 if SPLITS == 16 else HEADS`
- S14 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/083b493a1861424c.json#kernel.py:1000-1033 per-m branches: depth 5 at m = 16 and 32, grid m / 16 × n / 16 one-wave programs, waves_per_eu 3; the other Points' files: bd017b0794d5db31.json#kernel.py:913 m = 1 `_narrow_small_gemm[(n // 4,)]`, one warp, c194fbea23151544.json#kernel.py:978 m = 2 and d6228c711bd687b6.json#kernel.py:851 m = 4 the same, 0e15a0289735acb0.json#kernel.py:851 m = 8 `_packed_m8_gemm[grid_n]` with grid_n n // 16 (:825) and 2 warps (:858), fcf21622f6c91bbb.json#kernel.py:416 m = 32 `[(m // 16 * 192,)]` with prefetch 5 on the contiguous dense path (:46) and waves_per_eu 8 (:52-54), 54f6affa5c286a70.json#kernel.py:442 m = 64 `[(m // 16 * (n // 16),)]` with depth 4 under dense scales and waves_per_eu 6 (:438-457)
- S15 aklr/plugins/sglang/models/qwen2_moe/impls/triton/qwen2_moe_router_model_dim4096_n_routed512_topk10/4801e216dcff290f.json#kernel.py:208-216 8 tokens: (1, 8, 32) one-wave programs; 5f2049852cd84a76.json#kernel.py:229-235, a87882774fd41b18.json#kernel.py:259-266, cd47a5e858af6ab4.json#kernel.py:263-269 the 16-, 64- and 128-token grids
- S16 aklr/plugins/sglang/models/qwen3_5/impls/triton/gated_attention_decode_group_fp8_quant_qkv_dim4608_n_kv_heads1_head_dim256/f5ba496e04f8e989.json#attention.py:153-164 1 row: (rows, heads, 32 splits); 3d0f64ffff6790c8.json#attention.py:224-265 4 rows: 32 splits, (rows, splits); dfb1ab1cdd5bf263.json#attention.py:334-343 256 rows: 2 splits, splits × rows
- S17 aklr/plugins/sglang/layers/dsa_backend/impls/triton/dsa_sparse_mla_decode_n_heads64_kv_dim576_topk2048/2ee56a6dbabc48a6.json#kernel.py:584-617 head tile and splits from the token count, grid tokens × head groups × splits, 8 warps
- S18 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_decode_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072/2c2a09e220f2706c.json#kernel.py:312-322 1 row: value tiles of 8 rows, (tokens × 16, 16) programs; 4f257a3e566b20d1.json#kernel.py:206-211 8 rows: (tokens, heads) four-wave programs; fc261961f0ad572d.json#kernel.py:300-322 128 rows: one 16-wave program per (token, head), waves_per_eu 6 or 8
- S19 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_prefill_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072/c53b6d874d511c02.json#kernel.py:755 _delta_chunks[heads, 2, batch] with BV 64 (:507); 2ecece9650b95f71.json#kernel.py:1812 _chunk_recurrent[batch, heads, 128 // bv] with bv 32 (:1738); 382853eb770e30b3.json#kernel.py:1614 _state_scan[batch, heads, head_dim // state_value_rows] with 16 rows (:1539)
- S20 triton-lang/triton@c01b677:python/triton/experimental/gluon/_runtime.py:27-32 ttg.num-warps set from the launch option before layouts are verified
- S21 triton-lang/triton@c01b677:lib/Dialect/TritonGPU/IR/Dialect.cpp:3869-3881 a layout's warp count must equal the module's
- S22 triton-lang/triton@c01b677:python/triton/experimental/gluon/language/_semantic.py:650-654 gl.num_warps returns the launch option as a constexpr
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
- S36 aklr/plugins/sglang/models/qwen3_5/impls/triton/gated_attention_decode_group_fp8_quant_qkv_dim4608_n_kv_heads1_head_dim256/6dd321aac1b1d85a.json#attention.py:53-86 "Stage 0 issues V before QK; stages 1/2 bracket the sum reduction"; :336-337 the stages per row count; :24-30 the load and MFMA layouts read `gl.num_warps`
- S37 aklr/plugins/sglang/models/qwen3_next/impls/triton/qwen3_next_gdn_decode_group_fp8_quant_n_v_heads8_key_dim128_value_dim128_conv_dim2048/f44007374038041f.json#gdn_decode_config.py:76-83 state_prefetch early at M=1/32, late at M=128/256
- S38 triton-lang/triton@c01b677:python/triton/compiler/code_generator.py:1208-1216 static_range: the body is generated once per iteration
- S39 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/14c8e25ca1a39f6d.json#kernel.py:523 `_loop_range(loop_end, loop_unroll_factor=1)`, imported at :5; triton-lang/triton@c01b677:python/triton/language/core.py:3701-3703 the factor drives "the Triton IR level loop unroller", "Less than 2 … implies no unrolling"; triton-lang/triton@c01b677:third_party/amd/backend/compiler.py:337 gluon_to_ttgir runs the loop unroller
- S40 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/0e15a0289735acb0.json#kernel.py:609-610, 646-649 two accumulators alternating by K product; 083b493a1861424c.json#kernel.py:379-487 READ_AHEAD, EARLY_MFMA and REFILL_BEFORE, set per Point at :1000-1033; d6228c711bd687b6.json#kernel.py:573-578 scales preloaded, live at m = 4 (:847-858)
- S41 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_mxfp4_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts64_inter_dim1536/1d277d76892ee692.json#kernel.py:711 WEIGHT_CACHE.cg except at 128 tokens (:603)
- S42 aklr/plugins/sglang/models/deepseek_v2/impls/triton/deepseek_v2_mxfp4_moe_shuffled_16x16_model_dim6144_n_routed256_inter_dim512/2422a87b3b814299.json#kernel.py:434 no.cg on the up projection above 128 tokens with 128-row tiles
- S43 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/5ea30df18b03ab8d.json#kernel.py:1093 group_m 2 or 4 by m; :629-637 the grouped tile order and the N interleave (`N_INTERLEAVE` 4 at M = 4096, :633)
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
- S59 aklr/core/validation/stages/_bench/_stage.py:172-180 with --base, the base file served at the Point is the incumbent, held to retain, the candidate to entry against the baseline; :200 the record's baseline names the served file; :338-357 the timings are kept beside the verdict; aklr/core/validation/stages/_measure.py:292-296 every participant timed in each round, in a seeded random order; aklr/core/validation/stages/validation_scope.py:50-59 without paths the changed schema files are validated, and :73-84 untracked files count as changed
- S60 triton-lang/triton@c01b677:python/triton/compiler/compiler.py:466-468 a launch whose metadata.shared exceeds the device's limit raises OutOfResources for "shared memory"
- S61 aklr/core/contract/semantics/_compare.py:142-149 `runs_agree`: bitwise means every run's output byte-identical to the first's
- S62 aklr/plugins/sglang/models/qwen2_moe/impls/triton/qwen2_moe_router_model_dim4096_n_routed512_topk10/c3889614985b700a.json#kernel.py:269-271 the split-K planes summed in `_finish_routing`, launched at :300; aklr/plugins/sglang/models/qwen3_5/impls/triton/gated_attention_decode_group_fp8_quant_qkv_dim4608_n_kv_heads1_head_dim256/d9cbde237fd34de9.json#attention.py:381-396 the finish launches that merge the KV splits; aklr/plugins/sglang/models/deepseek_v2/impls/triton/deepseek_v2_mxfp4_moe_shuffled_16x16_model_dim6144_n_routed256_inter_dim256/d3bbbce5c0bbc61e.json#kernel.py:376-378 the gate-up K splits summed in `_activate_quantize`, launched at :516
- S63 aklr/core/validation/stages/_bench/_execution/_runner.py:515 `GPU_FORCE_QUEUE_PROFILING="1"` in every runner process; aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/083b493a1861424c.json#kernel.py:1015 the one launch at m = 16; aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_decode_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072/299133886d479017.json#kernel.py:234, :242 its two launches; aklr/plugins/sglang/models/qwen2_moe/impls/triton/qwen2_moe_router_model_dim4096_n_routed512_topk10/4801e216dcff290f.json#kernel.py:213, :218 its two launches; aklr/plugins/sglang/models/deepseek_v2/impls/triton/deepseek_v2_mxfp4_moe_shuffled_16x16_model_dim6144_n_routed256_inter_dim256/d3bbbce5c0bbc61e.json#kernel.py:494-519 its four launches; aklr/plugins/sglang/models/qwen3_5/baseline.py:154, 162, 171, 183, 228 the GDN decode baseline's five calls: split, conv update, recurrence, gated norm, aiter's group quant
- S9 .github/workflows/ci.yaml:6-7 `AKLR_BASELINE_AITER`, installed over the image baseline.
