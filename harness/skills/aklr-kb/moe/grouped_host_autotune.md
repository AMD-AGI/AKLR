---
title: "Grouped GEMM — host path, launch modes and autotune"
kind: method
lever: grouped_host_autotune
ops: [grouped_gemm]
gens: [gfx950]
dtypes: [bf16, fp8, mxfp8, mxfp4]
regimes: [forward, dgrad, wgrad, skew, cuda-graph, autotune, beta-accumulate]
flydsl: 0.3.2
sources:
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_fp8_kernel.py
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_mxfp8_kernel.py
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_mxfp4_kernel.py
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/pytorch/kernels/grouped_gemm/grouped_gemm_utils.py
---

# Host path and autotune for grouped GEMMs

> The host side of a grouped GEMM: keeping the steady state free of device→host copies, launching under CUDA
> graphs and in eager mode, what to cache and under which key, and how to race configurations when the token
> distribution is unknown at dispatch. The dense rules (first-call race, margins, palindrome timing, cache keys,
> the compiled-object launch path) are in `../gemm/autotune.md` (the compiled-object launch path:
> `../flydsl/gemm.md` § autotune §6) and are not repeated. "FP8", "MX8",
> "MX4" are Primus-Turbo's `grouped_gemm_{fp8,mxfp8,mxfp4}_kernel.py` under `primus_turbo/flydsl/grouped_gemm/`,
> at main `ca1fdd67`. FlyDSL facts are v0.3.2.
> Numbers marked `*(field, E<n>)*` are measured values; their conditions are listed under Evidence at the end. Use
> them for direction and order of magnitude, not as targets.

## Route here when
- A grouped call costs more host time than kernel time, or stalls on a device→host copy.
- A kernel behaves differently under CUDA-graph capture than in eager mode.
- A cache grows with every step, or returns another call's result.
- The autotuner picks a configuration that loses on the deployed distribution or token count.

## 1. The steady state reads no device data on the host
- **Group metadata stays on the device.** Turbo builds `group_offs [G+1]` (int64) on the GPU from the group
  lengths (`grouped_gemm_compute_offs`, `grouped_gemm_utils.py:28-33`) and the kernels read it from a device
  buffer. Grids come from shapes alone (`grouped_group_mapping.md` §3). Turbo's grouped kernels synchronize
  only inside the first-call race (FP8:2601, 4675).
- **The host may read shapes, never contents.** `M_total` is `tensor.shape[0]` and G is `len(group_offs) − 1`,
  so per-call choices may use `M_total // G`: MX4 picks its NT and wgrad blocking from it (MX4:1164-1170,
  2083-2091), and FP8 adds a coarse M-derived regime to its key (`_np_regime`, FP8:2487; §3).
- **Decisions that need the distribution are made in the kernel.** Reading `group_offs` on the device to choose a
  walk direction or a split (`grouped_variable_k.md` §5, §9) keeps the host path sync-free and graph-safe.
- **Derived offsets are one device kernel, not a chain of torch ops.** *(field, E1)* Computing an MX quantizer's padded
  group lengths and offsets with ~7 small torch kernels cost ~31 µs per call (each ~4.4 µs, launch-bound); a
  single-thread device kernel that derives them from the tight int64 offsets (the HIP reference's form) takes
  ~9 µs. Launch it from the quantizer's host launcher. Moving a quantizer's group search to host-side torch ops costs
  the same way (`grouped_lowprec.md` §5).

## 2. Two launch modes: eager and CUDA-graph capture
- **Make the first call of every key eager.** Nothing can be timed during capture. A tuner reached under capture
  must return a static configuration and not cache it, so a later eager call still races (MX8:1722-1723).
- **Launch helpers from the GEMM's host launcher.** Scale repack then GEMM (MX8:1101-1120), GEMM then split-K reduce
  (FP8:4617): one launcher call, stream order for free.
- **A run-time dispatch layer costs host time**, as a device-side split policy for a wgrad did; re-measure the
  host path after adding any dispatch logic (`../gemm/autotune.md` §6).

In FlyDSL: `../flydsl/moe.md` § grouped_host_autotune §2.

## 3. Cache compiled objects and launch closures, keyed on static shape
- **Never key on tensor identity** (`../gemm/autotune.md` §5). In grouped training every call brings new
  activations, gradients and `group_offs`, so an identity-keyed cache of quantized activations or derived offsets
  hits ~0 in training and returns stale data when an id is reused. A weight-side cache keyed on
  `(id(weight), weight._version)` holds within one step: the forward fills it, the backward reuses it, the
  optimizer bumps the version. Its gain is capped at quantization time over step time.
- **Key the race on the static shape plus, at most, a coarse regime derived from M; never on `M_total` itself.**
  The token count is a run-time argument (`m_total`, an int32). The static part is op, N, K, G, dtypes, beta and
  strides/padding. Add an M-derived regime only where M changes which body is right, as a rule rather than a
  per-M retune: FP8's `_np_regime` (FP8:2487-2494) is 1 when an NN dgrad grid at `BLOCK_M = 128` fits in one round
  of CUs, `G · ceil(pm/128) · ceil(N/256) ≤ num_CUs` with `pm = M_total // G`, and 0 otherwise; regime 1 takes a
  fixed `BLOCK_M = 128` body with no race (FP8:2553-2555; `grouped_tile_pricing.md` §3). Its key is
  `(op, N, K, G, out dtype, operand dtypes, regime, persistent mode, capped CU count, beta, n_stride)`, the CU
  count entering only when the grid is capped (`num_cu if capped else 0`, FP8:2674-2687). Its wgrad key is "never
  per m_total" (FP8:4759); MX8's NT key has no `M_pad` (MX8:933).
- **Why it matters.** `M_total` changes every step in MoE training; a compile cache keyed on it compiles per step and
  keeps growing. *(field, E8)* One quantizer cache keyed on the total token count ran the host out of memory in a
  sweep of token counts. Bound every compile cache anyway: MX4 clears its caches past 32 entries
  (`_bound_caches`, `_GMXFP4_CACHE_CAP`, MX4:1153-1163).
- **A host plan cache keyed on `M_total` grows the same way.** A per-call plan (output shape, workspace,
  launcher entry) keyed on `M_total` does not recompile, but it gains an entry for every distinct token count.
  Bound it like a compile cache, or keep `M_total` out of its key.

## 4. Never re-pad weights per call
- *(field, E3)* An MXFP4 entry that padded the weight `[G, N, K/2]` (~65 MB) on every call for a non-256 N took
  0.142 ms instead of 0.090 ms (1867 against 2957 TF/s).
- Handle ragged dims in the kernel instead: a column mask on the store for a free N (out-of-range columns go to an
  index the descriptor drops), descriptor row bounds for a free M, zero scales past K for an MX contraction tail
  (`k_real`, MX4:364-373). *(field, E4)* Replacing the per-call N pad with a masked store took one wide-N MXFP4 forward
  from 1127 to 3079 TF/s.
- Produce the layout the kernel reads in the quantizer. *(field, E5)* An MXFP4 wgrad that re-packed column-wise operands
  to per-group 512 alignment with gather ops on every call lost its largest per-call backward overhead once the
  quantizer emitted that layout directly.

## 5. Racing configurations when the distribution is unknown
**What Turbo's grouped tuners do** (each runs once per static key):

| Kernel | Candidates | Probe | Numeric gate | Adoption |
|---|---|---|---|---|
| fp8 NT/NN (`_autotune_np_dispatch`, FP8:2497) | ≤4, cand[0] a static lead | balanced, 1024 and 8192 tokens per group (FP8:2483) | rel-MSE < (2e-2)² against the base at each point | palindrome ratio < 0.985 at every point (FP8:2622) |
| fp8 wgrad (`_autotune_wgrad_dispatch`, FP8:4743) | geometry-ordered list | balanced only (FP8:4772) | gate all first, then time survivors in one interleaved window set (FP8:4920) | faster by 0.2% at every point (FP8:4664) |
| MXFP8 NT (`_select_nt_cfg`, MX8:1005) | 5 | balanced, 2048 and 8192 (MX8:936-939) | against the base, shared output zeroed first | geomean 1.5% faster (MX8:1057) |
| MXFP8 wgrad (`_select_wgrad_cfg`, MX8:1711) | 9 (MX8:1683) | the first call's own operands and distribution | against the base | 1.5% faster; base config under capture or at beta=1 |
| MXFP4 | static tables, no race | — | — | chosen from host-visible extents (MX4:1099-1170) |

Turbo's backend dispatcher likewise tunes on load-balanced group lengths (`grouped_gemm_utils.py:36-60`).
- **Why balanced.** The choice then depends on the static shape only, never on whichever distribution the first
  call happens to carry (`_balanced_group_offs` docstring, FP8:2453-2465). A race on the first call's own data
  caches a choice made for that one distribution.
- **What balanced cannot see.** A configuration that ties on a balanced load can separate under skew. Turbo keeps
  such a measured optimum as a static lead because "the online race cannot find it" (FP8:2470-2475). An XCD remap
  that wins on a balanced wgrad load and collapses under skew (`grouped_scheduling.md` §2) is exactly what a
  balanced-only race adopts. Two designs keep it out, and the owner picks: exclude such candidates by construction
  and race balanced only (Turbo's fp8 per-tensor wgrad list is all `num_xcd = 1`, FP8:4654-4656), or score each
  candidate by the worse of a balanced and a skewed probe.
- **Ties under balance flip between runs.** The config key has no distribution, so one cached choice serves both
  balanced and skewed calls; candidates within the race's noise flip across cleared-cache runs unless timed as a
  palindrome (`../gemm/autotune.md` §2).
- **Race points are tokens per group, and they are not the deployment's.** Turbo scores two spread points because
  "the swizzle is not M-invariant" (MX8:936-939). A candidate that wins only at the deployed token count becomes a
  static lead (`../gemm/autotune.md` §2; how the race misses it: `grouped_measurement.md` §8).
  *(field, E6)* Check the lead at every token count its predicate covers: a wgrad lead measured +1.47 / +0.27 /
  +0.78–0.98 / +0.95% at 1024 / 2048 / 4096 / 8192 tokens per group, while an NN lead lost at 1024. Then check that
  the real entry lands on the lead, by timing the entry against both compiled candidates in one process.
- **Separate defaults per path.** NT and wgrad must not share a default configuration (a shared one collapsed an
  MXFP4 wgrad, `grouped_scheduling.md` §1); fixing each path's choice per shape family can beat a race on
  synthetic data (`../gemm/autotune.md` §2).
- **Zero the probe output before each candidate.** Candidates share the buffer; one that skips some of its tiles
  would inherit the base's values there and pass the numeric gate (MX8:1040-1042).

## 6. beta in the race
The measured facts (read-back cost by dtype, ranking differences between beta=0 and beta=1, the extra compile
per surviving candidate) are in `grouped_variable_k.md` §10. The mechanics of timing at beta=1:
- Run beta=1 candidates on scratch: every timed launch adds another product into the buffer.
- A beta=1 probe on an uninitialized buffer reads garbage and fails a finiteness gate, and a rel-MSE comparison
  against a reference fails once results accumulate. Either zero the probe output before every launch so
  `C = 0 + acc` matches the beta=0 reference (FP8's NT/NN race, FP8:2500-2502, 2598-2599), or gate at beta=0 and
  time beta=1 builds of the survivors (FP8's wgrad race, FP8:4934-4945).

In FlyDSL: `../flydsl/moe.md` § grouped_host_autotune §6.

## 7. Price helper kernels by launching them once more
A grouped call carries helper kernels: scale repack, quantizer, padded offsets, split-K reduce, output tail. Price
each by adding 0/1/2/4 extra launches in one process and fitting the slope (`../gemm/attribution.md`
§3), not by its trace duration, which overstated such kernels 2–3× (`grouped_measurement.md` §6).
- *(field, E7)* On an MXFP4 grouped kernel the kernel trace put the scale repack at 2.1–2.6% and an output-tail kernel
  at 0.5–0.9% of each call, an overstatement of about 2× (repack) and 3× (tail); even so they outweigh the
  whole family of steady-state SALU trims (~0.15%).
- Removing a whole helper launch pays. Shrinking the work inside a launch-bound helper does not
  (`grouped_sorted_tokens.md` §7); a helper bound by its own work, like a VALU-bound scale repack, does
  pay (`grouped_lowprec.md` §2).

## 8. Do not let a race swallow build errors
- A catch-all around candidate builds turns a build bug into "no working config" (`../gemm/autotune.md`
  §7). Turbo's fp8 wgrad race collects `(config, error)` pairs and logs them with the winner (FP8:4866, 4950-4954).
- Grouped geometry is derived from G and tiles per group (split pieces, slice heads, tiles per workgroup). Assert
  at build time that it fits the smallest G the callers use; otherwise a small expert count fails every fast
  candidate and the race silently settles on the fallback kernel.
- Check every candidate's LDS bytes against the 160 KiB limit before timing it (`../gemm/autotune.md` §7).

In FlyDSL: `../flydsl/moe.md` § grouped_host_autotune §8.

## Verify
- A trace of the steady state shows no host-side copy, and host enqueue time is below GPU time.
- The first call of every key is eager, and no race is entered under capture.
- No compile cache is keyed on `M_total` or on tensor identity, and every compile and plan cache is bounded.
- The race's winners were checked on skewed loads and at the deployed tokens per group.
- Every candidate built at least once in a smoke test, with build errors printed.

In FlyDSL: `../flydsl/moe.md` § grouped_host_autotune Verify.

## Related
`grouped_group_mapping.md` · `grouped_variable_k.md` · `grouped_lowprec.md` ·
`grouped_measurement.md` · `grouped_correctness.md` · `../gemm/autotune.md` ·
`../gemm/attribution.md` · `../attention/measurement.md` ·
`../flydsl/debug-flydsl-kernel.md`

## Evidence

A FlyDSL version is given only where the cited run, or another round in the same container and venv, printed it
(`flydsl.__version__`, an import path with its version, or a `pip install flydsl==` in the run's own setup); otherwise
`FlyDSL ?`. "Turbo" is Primus-Turbo. A line whose only reference is a myskill or memory note gives the month of that
note ("recorded by"); such notes rarely state N. Secondary references in a line sit in the directory of its first
`local:` path unless they carry their own.

- E1 [2026-07, gfx950, FlyDSL ?, N=?, rocprof kernel trace and wrapper time] grouped MXFP8 quantizer: the padded lengths and offsets computed by ~7 small torch kernels (`ceil` ×2, 2 × `cumsum`, `fill`, `copy`, ~4.4 µs each, launch-bound) ≈31 µs per call; the HIP reference's single-thread `compute_padded_layout_gpu <<<1,1>>>` ~9 µs (Turbo MXFP8 grouped quantization path; recorded by 2026-07-05, myskill 99db6ee; local:/workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/02-measurement-noise.md:431-432)
- E3 [2026-07, gfx950, FlyDSL ?, N=?, direct scan with the pad outside the timed region vs the entry] fat-N MXFP4 grouped forward, M=8192: the entry's per-call `F.pad` of the weight `[G,N,K/2]` (~65 MB) for a non-256 N took 0.090 → 0.142 ms (2957 → 1867 TF/s) (gpt-oss MoE shape N5760/K2816; local:/workspace/code/gpt_oss_docker/.claude/memory/project_mxfp4_grouped_gemm.md:59)
- E4 [2026-07, gfx950, FlyDSL ?, N=?, median; fp4/fp8 forward ratio against the MXFP8 grouped kernel] masked store for a free non-256 N instead of the per-call pad: fat-N N5760/K2816 forward 1127 → 3079 TF/s (fp4/fp8 0.72–0.89× → 1.46–1.49×), SNR ~15 dB on non-256 N (Turbo MXFP4 grouped GEMM; commit:/workspace/code/gpt_oss_docker/sync/tensorwise/Primus-Turbo@5f0d7fdc, 2026-07-14; local:/workspace/code/gpt_oss_docker/.claude/memory/project_mxfp4_grouped_gemm.md:61)
- E5 [2026-07, gfx950, FlyDSL ?, N=?, full forward+backward vs Triton, median] the quantizer emitting the per-group 512-aligned column-wise layout removed the per-call searchsorted + gather + index_select repack, which the commit names the dominant per-call backward cost; forward+backward vs Triton gate_up 0.79–1.06× → 1.43–1.74×, down 0.64–0.94× → 1.13–1.63× (Turbo MXFP4 grouped wgrad, MoE shapes; commit:/workspace/code/gpt_oss_docker/sync/tensorwise/Primus-Turbo@6dc9bac9, 2026-07-14; local:/workspace/code/gpt_oss_docker/.claude/memory/project_mxfp4_grouped_gemm.md:66-67)
- E6 [2026-08, gfx950, FlyDSL ?, N=4, `PROBE_ROUNDS=4`, median per arm, balanced routing, all arms bitwise identical] static wgrad lead `(group_m=2, num_xcd=1)` against the race's own pick: +1.47 / +0.27 / +0.78–0.98 / +0.95% at 1024 / 2048 / 4096 / 8192 tokens per expert; the NN dgrad lead (x8g4) −0.94% at 1024 (gpt-oss-20b down projection, tensorwise fp8 grouped GEMM, padded operands; campaign 20260817_065611 rounds 6–7; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260817_065611/remote_data.md:180-191; :28)
- E7 [2026-09, gfx950, FlyDSL 0.2.4, N=?, rocprofv3 `--kernel-trace` per cell; in-process additive probe (0/1/2/4 extra launches, slope)] MXFP4 grouped call: the preshuffle `kern_0` 2.08–2.60% of each op and `_grouped_gemm_output_tail_kernel` 0.5–0.9% by the trace, against ~0.15% for the whole steady-state SALU family; additive probe: `kern_0` 20.32 µs traced vs ≈10.2 µs in stream (≈2×), the idle tail kernel 5.10 µs traced vs 1.579 µs (3.2×) (gpt-oss-20b plain MXFP4 grouped MoE GEMMs; campaign 20260925_090609 rounds 2 and 10, `kyle_mxfp4_204` `/opt/venv-syncv4`; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260925_090609/rounds/round-02/agent.txt:37; goal.md:254; rounds/round-10/agent.txt:18)
- E8 [2026-07, gfx950, FlyDSL ?, N=?, full end-to-end autograd sweep over B × M shapes; host RSS probe] the grouped MXFP4 dual-quant compile cache (`_GQ_MXFP4_CACHE`), keyed on total_M and unbounded, accumulated compiled kernels until the container hit its host-RAM cgroup limit (OOM 137, seen as a segfault mid MLIR build); with the cache bounded, a 132-distinct-shape probe held RSS flat and the sweep passed 1148 / 32 skipped (Turbo MXFP4 grouped GEMM; commit:/workspace/code/gpt_oss_docker/sync/tensorwise/Primus-Turbo@8d6f3854, 2026-07-15; local:/workspace/code/gpt_oss_docker/.claude/memory/project_mxfp4_grouped_gemm.md:69)
