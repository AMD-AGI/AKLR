---
name: aklr-kb/triton
description: >
  AKLR Triton/Gluon on gfx950/3.8.0: ABI, validation, timing, layouts,
  memory, MFMA, numerics and operator designs.
allowed-tools: Read Edit Bash Grep Glob Agent
---

# Triton and Gluon kernels for AKLR on gfx950

AKLR Triton/Gluon on gfx950 (MI355X), Triton 3.8.0: 154 schemas at 43e112e (150 Gluon, 4 plain `tl.*`),
280 on main 0e02a81 (the same 4 plain) *(field, E1)*. [Evidence](#evidence) gives each number's conditions and
commit, run or PR; `aklr/…` paths are on main unless pinned.

## By task

|Task|Read|
|---|---|
|Start|[library](kernels/kernel_library.md) §3; [authoring](authoring.md) §10|
|ABI, scratch, keys, launches, capture, fallbacks|[authoring](authoring.md) §2–§9|
|Validation|[authoring](authoring.md) §11; [debugging](debugging.md) §5–§6|
|IR, ISA, VGPRs, LDS, occupancy|[debugging](debugging.md) §1–§3|
|Compile errors|[debugging](debugging.md) §4; [MFMA](gluon/mfma.md) §10|
|Wrong values, faults, nondeterminism|[debugging](debugging.md) §5–§6; [numerics](numerics.md) §10|
|Profiling, host cost, other builds, kernel time, micro-benchmarks|[debugging](debugging.md) §7–§11|
|A/B, split, grid, prefetch|[tuning](tuning.md) §1–§4, §6, §10|
|Options, cache hints, XCD|[tuning](tuning.md) §7–§9, §11–§12|
|Budget a Point|[tuning](tuning.md) §13|
|Rounding, quantization|[numerics](numerics.md) §1–§4|
|SwiGLU, formats|[library](kernels/kernel_library.md) §3; [MoE](moe.md) §8; [numerics](numerics.md) §2–§3; [wave ops](gluon/wave_ops.md) §3; [MFMA](gluon/mfma.md) §7|
|Accuracy, math, atomics|[numerics](numerics.md) §5–§8|
|Layouts|[layouts](gluon/layouts.md) §6–§8|
|Bounds, async copies, waits|[memory](gluon/memory.md) §3–§5, §8–§12|
|MFMA, scales, small M|[MFMA](gluon/mfma.md) §1–§2, §5, §8|
|Inline asm, hazards|[wave ops](gluon/wave_ops.md) §2, §6–§9|
|fp8 GEMM|[GEMM](gemm.md) §3, §5|
|bf16 decode GEMM, split-K|[decode GEMM](gemm/decode_gemm.md); [GEMM](gemm.md) §9|
|MoE|[MoE](moe.md) §2, §5–§9|
|Decode attention|[attention](attention.md) §1–§2, §5, §10|
|GDN decode, prefill (schedule per Point: §8)|[GDN](gdn.md)|
|Ruler|`harness/skills/aklr-read-validation/references/ruler.md`|
|SA0–SA4, ABI|`aklr/ops/README.md` § The kernel section and ABI|
|Cross-language|[KB](../SKILL.md)|

## Rules for every Triton or Gluon kernel in AKLR

- Constexprs specialize the configuration, not always the code: compiled 8/16/32/64-row masks vanished once M
  covered the tile, but `range(M)` kept branches at M = 23–128 (42–68 VGPRs) and M = 19 took 74 VGPRs, 66 with a
  runtime M *(field, E7)*. A key first seen in graph capture compiles there and trips AKLR's tripwire: warm every key
  eagerly *(field, E16)*.
- At 43e112e no file used `triton.autotune` or `heuristics`; autotune re-times its configs at each new key
  *(field, E16)*. No plain JIT globals ([authoring](authoring.md) §4).
- All 154 `run`s at 43e112e are keyword-only, return nothing and call no `synchronize()`, `.item()`, `.cpu()` or
  `.tolist()` *(field, E8)*; implicit syncs untested. `.item()` and `torch.cuda.synchronize()` fail in a global
  capture *(field, E16)*.
- Reset counters, sentinels and padding every call: a pointer-cached output went stale in a CPU test; whether AKLR
  catches it in general is untested *(field, E8)* ([authoring](authoring.md) §6, §8).
- In one launch a Gluon last-arriver (acq_rel) and a cooperative-grid barrier passed values across workgroups
  correctly; a plain spin on a partner in an oversubscribed grid timed out. Two launches beat the tested fold but took
  1.16× the replay time of a 1–2-token count scan (one run) *(field, E9)* ([authoring](authoring.md) §7).
- Under `bitwise`, no float atomics: reduce in a fixed order ([numerics](numerics.md) §8).
- Skipping a reference rounding: in the router (256 tokens) and GDN decode (128 rows) it lowered the error yet failed
  validate twice; in DeepSeek-V2 (8 tokens) and gated attention (32 rows) it passed twice, gated attention's scale
  error rising. A pass is not non-inferiority *(field, E14)*. Compute in fp32: bf16 × a Python float stays bf16
  ([numerics](numerics.md) §1).
- Every tested layout mismatch in an elementwise op, store or load mask (Blocked, Slice, MFMA, DotOperand, Linear)
  failed to compile. Give `arange`, `zeros` and `full` a layout; a Blocked layout's warps multiply to `num_warps`
  (default 4). Auto, Coalesced and other warp layouts untested *(field, E10)* ([layouts](gluon/layouts.md) §1).
- Gluon has no automatic `num_stages` pipeline; its warp-pipeline pass handles explicit stage annotations.
  `matrix_instr_nonkdim` is unused, `kpack` becomes 1 on gfx950, and `schedule_hint` has no effect in 3.8.0
  ([authoring](authoring.md) §2, [tuning](tuning.md) §9).
- A buffer op has no tensor bound: mask lanes leaving the tensor, keep all accessed bytes below 2^31 − 2. Tested
  1/2/4 B accesses past the range read 0 and dropped writes; 8/16 B vectors were partly read or written by dword, no
  fault; typed offsets 2^32 + d wrapped to d (d = 0, 4, 64) *(field, E2)* ([memory](gluon/memory.md) §3–§4).
- `gl.barrier()` and a plain `.load` do not wait for an async copy (the tile read stale); `wait_group` does, for
  committed groups only ([memory](gluon/memory.md) §8).
- LDS is `metadata.shared`, not the amdgcn's group segment; 163,840 B ran, 262,144 B failed at the first launch
  *(field, E3)* ([debugging](debugging.md) §2, §4).
- Constexprs passed at launch must hash: tuples, not lists ([debugging](debugging.md) §4).
- A masked e8m0 scale load takes `other=127` (2^0): `0xFF` made the masked row NaN, no `other` reads 0 (2^−127)
  ([MFMA](gluon/mfma.md) §2.2).
- Read stderr after compiling inline asm; LLVM pads no hazard inside it ([wave ops](gluon/wave_ops.md) §9).
- Equal bytes in flight do not give equal rate (programs × warps × 16 B loads per lane): on a 1 GiB stream
  equal-byte groups spread up to 10.4%, and at 3 MB 96×4×8 and 192×1×16 read 0.80× and 0.50× of 384×4×2 (3 runs)
  *(field, E4)*; in another stream (one run) four 8 MiB configurations stayed within 10% of the median, but 3/4/6 MiB
  384×1×8, 256×1×16, 384×1×16 read 0.888×, 0.760×, 0.861× of four-warp peers *(field, E15)*. At equal bytes the
  256-token router's call time read 1.000, 1.033, 1.260 for programs × look-ahead 256×2, 512×1, 128×4 (one run)
  *(field, E11)*: do not size a decode kernel by bytes alone or pool implementations.
- Router split 4 → 2 (256 tokens) and gated attention KV splits 16 → 8 (32 rows, head tile changed too) lost: B/A
  0.75, 0.63. Router look-ahead 2 → 1 lost at 256 programs (1.12–1.14× call time); at 512 programs depths 1, 2, 4
  read about 1.03× of 256 × 2. fp8 GEMM m = 16 `PREFETCH` 5 → 3: B/A 0.87–0.91; a gain at 7 is unresolved, not
  absent. No split or depth fits all Points *(field, E5, E11)* ([tuning](tuning.md) §3–§4).
- Compare each Point with the faster of its served kernel and baseline. In two A/A studies (four runs per arm) eight
  runs spanned up to 5.93%, one arm 3.26%, an identical copy's mean 1.3% off *(field, E5)* ([tuning](tuning.md)
  §1–§2). Report both times: a ratio across Points mixes baseline and candidate changes.

## In AKLR

- `gemm@1.0` stub: the stored layout was served; non-contiguous, misaligned and output-aliasing calls were out of
  contract *(field, E16)*. Strides may fold ([authoring](authoring.md) §4); masks may drop ([memory](gluon/memory.md) §4).
- Choose the configuration per Point from shapes and dtypes: host branches, a `@gluon.constexpr_function` both kernel
  and launch read, or a `@functools.cache` compile function ([authoring](authoring.md) §5).
- MXFP4 and e8m0 tensors go into a launch as `.view(torch.uint8)`; give a kernel that calls 3.8-only API
  `"requires": {"triton": ">=3.8"}` ([authoring](authoring.md) §3–§4).
- AKLR times a call as the sum of its kernels' device durations after a 1 GiB read flush, gaps excluded:
  rocprofiler-sdk collector since #33, Kineto before (S1). A trivial kernel cost 4.16–4.32 µs under Kineto
  *(field, E6)* ([tuning](tuning.md) §1).
- A flush preceded each of 85 timed calls; it stores only block sums. Flushed over hot time: 1.19–1.28× at 64–192
  MB, 1.00–1.01× at 512–1024 MB *(field, E12)*. Re-time with AKLR: at n6144 m = 64 the source tier lost hot too,
  1.08–1.13× the re-tuned time *(field, E13)* ([decode GEMM](gemm/decode_gemm.md) §8).
- CI requires 3.8.0; matched builds differ from 3.7.0. Check the version and re-time on CI's image
  ([authoring](authoring.md) §11).

## Evidence

- E1 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, source census] `aklr/**/impls/triton/*/*.json`: 154 at 43e112e (150 Gluon), 280 at 0e02a81 (276 Gluon) (AKLR@43e112e, AKLR@0e02a81)
- E2 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2, Gluon buffer-op probes; byte recheck N=1] 1/2/4 B past the range read 0, writes dropped; 93,511 out-of-range bytes per run neither read nor written; offsets 2^32 + d wrapped to d (#89)
- E3 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, one Gluon launch per LDS size] 163,840 B ran, 262,144 B failed (#89)
- E4 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3, 1 GiB Gluon read stream, eager] equal bytes within 10.4% over 256–8192 programs, 1–8 warps, ≤ 8 loads per lane; at 3 MB 0.80× and 0.50× (#89; tuning.md E12)
- E5 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2 to 8 per item, `aklr validate` (Kineto), 4 alternated runs per arm] B/A 0.75 (router split, 8 runs), 0.63 (attention, 7), 0.87–0.91 (`PREFETCH` 3, 7 runs); `PREFETCH` 7: 7 runs, look-ahead 6, A/A 2 (AKLR@43e112e)
- E6 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=5, `bench_cudagraph`, Kineto, before #33] one trivial Gluon kernel 4.16–4.32 µs (AKLR@43e112e)
- E7 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, compile-only scan of M = 1–128, some rows in three compiles] `range(M)` branches at M = 23–128, 42–68 VGPRs; M = 19 74 VGPRs, 66 with a runtime M (#89)
- E8 [2026-11, CPU only, Triton 3.8.0, N=1, source scan and a CPU test] 154 of 154 `run`s; 7 cached elements stale (AKLR@43e112e)
- E9 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, Gluon probes, graph replay] 0 wrong in 1000 calls per Point with the counter zeroed per call (1999 of 1999 wrong without); two launches no slower than the fold at 9 sizes, 1.16× the count scan at 1–2 tokens, 0.89× at 16; the oversubscribed spin (4096 programs) timed out 3448–3454 of 4095 times in 2 runs, 0 in two launches (#89)
- E10 [2026-11, CPU only, Triton 3.8.0, N=2, compile-only for gfx950] 63 mismatch/control pairs, 24 warp builds per run (#89)
- E11 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, `aklr validate` (Kineto), 4 rounds per arm] router `c3889614985b700a` 1.000/1.033/1.260; DeepSeek-V2 `d3bbbce5c0bbc61e` split 4 → 2 1.004×, unresolved (AKLR@43e112e)
- E12 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2, `bench_cudagraph` (Kineto)] 85 of 85 calls flushed; 1 GiB over a 4 GiB flush 0.992–1.000; flushed/hot 1.19–1.28× (64–192 MB), 1.00–1.01× (512–1024 MB); Gluon eager probe (4 runs) 1.10–1.22×, 0.99–1.01×; HBM reads not measured (AKLR@43e112e)
- E13 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=4, in-process timing] BLOCK_K 128, 8 warps, KROT 0 over 256, 4, 1, hot: 1.08–1.13× (`f806ad2b086d757d`; AKLR@43e112e)
- E14 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, `aklr validate` twice per variant, one input set] router `c3889614985b700a`, GDN decode `fc261961f0ad572d`: 2 of 2 failed; gated attention scale relative L2 0.003233 → 0.003269 (AKLR@43e112e)
- E15 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, 1 GiB bf16 Gluon stream, 12 timing blocks] paired 95% intervals 0.8665–0.8692, 0.7437–0.7452, 0.8539–0.8546; A/A within ±1% (#89)
- E16 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, capture probes and `aklr validate`] a capture-compiling variant failed bench without `capture_unsafe`; a new autotune key re-timed 3 configs, 0 compiles (AKLR@43e112e)
- S1 aklr/core/utils/testing.py:206-224 each call captured after `flush_l2_cache()` (:70-75), timed as its kernels' summed durations
