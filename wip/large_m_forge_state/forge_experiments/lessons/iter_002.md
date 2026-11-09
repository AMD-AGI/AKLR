# Iteration 2 record

## 1. Profiling the aiter baseline (scratch script in /tmp; completed)

I profiled AKLR's aiter `fused_moe` with the torch profiler. Per-call kernel times:

| Workload | Sorting kernels | Quant / scale-sort kernels | Stage-1 kernel | Stage-2 kernel |
|---|---|---|---|---|
| n_tokens=256 | 5.4 + 3.7 µs | 7.2 + 4.6 µs | `t32x128x256` asm, 100.5 µs | `t32x128x256`, 57.2 µs |
| n_tokens=3816 | — | — | `t128x128x256`, 162.4 µs | 181.8 µs |
| n_tokens=32768 | P23 sort 77.2 µs | quant 167.4 µs, scale sort 31.6 µs | `t64x128x256`, 870 µs | persist kernel, 1315 µs |

## 2. Inlined FlyDSL two-stage MoE in `kernel.py` (completed and submitted)

- **Build:** a generator script (`/tmp/w2/gen.py`) inlined aiter's `buffer_ops`, `vector`, `kernels_common`, `layout_utils`, `mfma_epilogues`, `mfma_preshuffle_pipeline` and `mixed_moe_gemm_2stage_common` into `kernel.py`. Each module sits in its own namespace. The generator replaces the one environment read with `None` and replaces `GateMode` and `get_cu_num` with local definitions.
- **Pipeline:** torch sync-free sort into packed ids, torch MXFP4 quantization emitting packed codes plus shuffled sorted E8M0 scales, stage 1 with bf16 output, a second torch quantization, then atomic stage 2 into `out` after `out.zero_()`.
- **Smoke test:** relative error against the old torch kernel was 0.0035 at n_tokens=256, 3816 and 32768, with no NaNs.
- **Correctness:** `driver.py` reported `allclose: True` on all 25 workloads.
- **Bench** (`--warmup 3 --iters 20`):

| n_tokens | 256 | 512 | 640 | 3112 | 3816 | 10000 | 13152 | 32768 |
|---|---|---|---|---|---|---|---|---|
| ms | 0.879 | 1.005 | 1.093 | 2.679 | 3.194 | 7.823 | 10.159 | 24.405 |

  The sum over all 25 workloads was 67.39 ms; the incumbent's raw mean was 501.4 ms. Every workload was faster than the incumbent torch path and slower than the aiter baseline.

The submitted config:

HELD-FIXED: tile_m=32 for n_tokens≤1024 and 64 above; stage-1 tile_n=128, tile_k=256, waves_per_eu=3, b_nt=2, k_wave=1, xcd_swizzle=0, v2_output_layout=False, bf16 output; stage-2 tile_n=128, tile_k=256, b_nt=0, atomic, persist_m=-1 when blocks>256.

## 3. Cleanup (completed)

I removed `import os` and an unused list, cleared the FlyDSL cache and re-ran correctness: `allclose: True`. The cleaned file was not re-benchmarked.

NEGATIVES: none
DISPROOF: none
KB: none

SCOPE: measured on n_tokens=10000, n_tokens=13152, n_tokens=256, n_tokens=272, n_tokens=288, n_tokens=304, n_tokens=3112, n_tokens=320, n_tokens=32768, n_tokens=336, n_tokens=352, n_tokens=368, n_tokens=3816, n_tokens=384, n_tokens=400, n_tokens=416, n_tokens=424, n_tokens=432, n_tokens=448, n_tokens=464, n_tokens=480, n_tokens=496, n_tokens=512, n_tokens=576, n_tokens=640 | held fixed tile_m=32 for n_tokens≤1024 and 64 above, tile_n=128, tile_k=256, waves_per_eu=3, b_nt=2, k_wave=1, xcd_swizzle=0, v2_output_layout=False, persist_m=-1 when blocks>256. | no measured negative | no feasibility claim

OUTCOME: KEEP | mean case speedup 0.179406x vs best 0.179406x | wall 67.3979 ms vs best 67.3979 ms | session ended: candidate_submitted
