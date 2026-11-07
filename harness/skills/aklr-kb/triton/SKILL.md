---
name: aklr-kb/triton
description: >
  AKLR Triton/Gluon on gfx950/3.8.0: ABI, validation, timing, layouts,
  memory, MFMA, numerics and operator designs.
allowed-tools: Read Edit Bash Grep Glob Agent
---

# Triton and Gluon kernels for AKLR on gfx950

AKLR Triton/Gluon on gfx950 (MI355X), Triton 3.8.0: 154 committed kernels, 150 Gluon/4 plain `tl.*` *(E1)*.
Source/probe data; [Evidence](#evidence) gives conditions. `aklr/…` is repo-relative.

## By task

|Task|Read|
|---|---|
|Start|[library](kernels/kernel_library.md) §3;[ABI](authoring.md) §10|
|ABI/scratch/keys/capture|[ABI](authoring.md) §2–§8|
|Validation|[ABI](authoring.md) §11;[dbg](debugging.md) §5–§6|
|IR/ISA/VGPR/LDS/occupancy|[dbg](debugging.md) §1–§3|
|Compile errors|[dbg](debugging.md) §4;[MFMA](gluon/mfma.md) §10|
|Values/faults/nondeterminism|[dbg](debugging.md) §5–§6;[math](numerics.md) §10|
|Profile/timer|[dbg](debugging.md) §7, §11|
|A/B/split/grid/prefetch|[tune](tuning.md) §1–§4, §6, §10|
|Options/cache/XCD|[tune](tuning.md) §7–§9, §11–§12|
|Budget|[tune](tuning.md) §13|
|Rounding/quantization|[math](numerics.md) §1–§4|
|SwiGLU/format|[library](kernels/kernel_library.md) §3;[MoE](moe.md) §8;[math](numerics.md) §2–§3;[wave](gluon/wave_ops.md) §3;[MFMA](gluon/mfma.md) §7|
|Accuracy/math/atomics|[math](numerics.md) §5–§8|
|Layouts|[layout](gluon/layouts.md) §6–§8|
|Bounds/async/waits|[memory](gluon/memory.md) §3–§5, §8–§12|
|MFMA/scales/small M|[MFMA](gluon/mfma.md) §1–§2, §5, §8|
|Asm/hazards|[wave](gluon/wave_ops.md) §2, §6–§9|
|fp8 GEMM|[GEMM](gemm.md) §3, §5|
|bf16 M64/N2624/K6144: split6/4 unresolved;small-M/AITER untested|[decode](gemm/decode_gemm.md);[GEMM](gemm.md) §9|
|MoE|[MoE](moe.md) §2, §5–§9|
|Decode attention|[attn](attention.md) §1–§2, §5, §10|
|GDN decode / prefill: D/CKPT beat measured RSUM/MAP at five Points, not all ≤2048|[GDN](gdn.md) §8 (N=1 study, five Point jobs)|
|Ruler|`harness/skills/aklr-read-validation/references/ruler.md`|
|SA0–SA4/ABI|`aklr/ops/README.md` § The kernel section and ABI|
|Cross-language|[KB](../SKILL.md)|

## Rules for every Triton or Gluon kernel in AKLR

- Constexprs specialize by configuration, not necessarily distinct code per Point. Compiled 8/16/32/64-row
  masks vanish when M covers the tile; `range(M)` keeps branches at M23–128 (42–68 VGPRs in that range); the M1–128 scan reaches
  74 VGPRs at M19 (runtime 66)
  *(E7)*. Captured new keys compile/trip; warm eagerly *(E16)*.
- 154 files: no autotune/heuristics; autotune times configs/new key, builds missing code *(E16)*.
  No plain JIT globals ([ABI](authoring.md) §4).
- 154 `run`s: keyword-only/no return *(E8)*; no synchronize/item/cpu/tolist; implicit syncs untested.
  Item/synchronize fail global capture *(E16)*.
- Reset counters/sentinels/padding. CPU cached-pointer output went stale; universal AKLR detection untested
  *(E8)* ([ABI](authoring.md) §6, §8).
- Gluon last-arriver acq_rel/cooperative-grid barriers handed off correct values within one launch; plain
  oversubscribed partner spins timed out. Two launches beat the tested fold but lost to 1–2-token count scans
  (1.16× replay time, N=1) *(E9)* ([ABI](authoring.md) §7).
- Under `bitwise`, no float atomics: reduce in a fixed order ([math](numerics.md) §8).
- Rounding skips: S1/S4 error↓/fail×2, S6/S7 pass×2; S7 scales error↑. Non-significance≠non-inferiority (E14).
  Compute in fp32: bf16 × a Python float stays bf16 ([math](numerics.md) §1).
- Tested elementwise/store/load-mask layout mismatches fail (Blocked/Slice/MFMA/DotOperand/Linear).
  Give `arange`/`zeros`/`full` layouts; tested Blocked warp products equal `num_warps` (default 4).
  Auto/Coalesced and other warp layouts untested *(E10, compile-only)* ([layouts](gluon/layouts.md) §1).
- Gluon has no automatic `num_stages` pipeline; its warp-pipeline pass handles explicit stage annotations.
  `matrix_instr_nonkdim` is unused, `kpack` becomes 1 on gfx950, and `schedule_hint` has no effect in 3.8.0 ([ABI](authoring.md) §2, [tuning](tuning.md) §9).
- A buffer op has no tensor bound: mask lanes leaving the tensor and keep all accessed bytes below 2^31 − 2.
  Tested 1/2/4 B boundary accesses zero/drop; 8/16 B vectors can partially read/write by dword, without fault.
  Typed offsets 2^32+d wrapped to d for tested d=0/4/64 combinations *(E2)* ([memory](gluon/memory.md) §3–§4).
- `gl.barrier()` and a plain `.load` do not wait for an async copy (the tile read stale); `wait_group` does, for
  committed groups only ([memory](gluon/memory.md) §8).
- LDS is `metadata.shared`, not the amdgcn's group segment; over 160 KiB the first launch fails *(field, E3)*
  ([dbg](debugging.md) §2, §4).
- Constexprs passed at launch must hash: tuples, not lists ([dbg](debugging.md) §4).
- A masked e8m0 scale load takes `other=127` (2^0): `0xFF` made the masked row NaN, no `other` reads 0 (2^−127)
  ([MFMA](gluon/mfma.md) §2.2).
- Read stderr after compiling inline asm; LLVM pads no hazard inside it ([wave ops](gluon/wave_ops.md) §9).
- Equal in-flight bytes do not fix rate. Old 1 GiB: ≈10.4% max/min−1 in logged equal-byte groups
  (256–8192 programs, 1–8 warps, ≤8 loads/lane, N=3); at 3 MB, 96×4×8/192×1×16 read 0.80×/0.50× of 384×4×2 *(E4)*.
  New 1 GiB bf16 stream (N=1 run, 12 timing blocks): four tested 8 MiB configs within 10% of level
  median; 3/4/6 MiB, 384×1×8/256×1×16/384×1×16 reached 0.888/0.760/0.861× *(E15)*.
  Triples: programs×warps×16B loads/lane. Do not pool implementations or infer decode sizing:
  Router c388, 256 tokens: equal bytes gave call-time ratios
  1.000/1.033/1.260 for programs×look-ahead 256×2/512×1/128×4 *(E11, N=1, four rounds)*.
- Router 256-token split4→2 and attention 32-row KV16→8 lost (B/A speedup 0.75/0.63);
  attention's head tile also changed. Router look-ahead 2→1 lost at 256 programs (1.12–1.14× call time),
  but depths 1/2/4 at 512 programs all read ≈1.03× of 256×2. fp8 GEMM M16 PREFETCH5→3 gave B/A
  0.87–0.91; P7 gain unresolved, not absent. No universal split/depth choice *(E5, E11)* ([tuning](tuning.md) §3–§4).
- Compare each Point with the faster served kernel/baseline. In two A/A studies with four runs/arm:
  eight-run span ≤5.93%, single-arm span ≤3.26%; copy mean offset ≤1.3%
  *(field, E5)* ([tuning](tuning.md) §1–§2).

Compare at the same Point; report both times. Cross-Point ratios mix baseline and candidate changes, not guaranteed better decisions.

## In AKLR

- `gemm@1.0` stub: aligned/contiguous served; bad stride/alignment/alias refused *(E16)*.
  Shape strides may fold ([ABI](authoring.md) §4); masks may drop ([memory](gluon/memory.md) §4).
- Choose the configuration per Point from shapes and dtypes: host branches, a `@gluon.constexpr_function` both kernel
  and launch read, or a `@functools.cache` compile function ([ABI](authoring.md) §5).
- MXFP4 and e8m0 tensors go into a launch as `.view(torch.uint8)`; give a kernel that calls 3.8-only API
  `"requires": {"triton": ">=3.8"}` ([ABI](authoring.md) §3–§4).
- AKLR: 1 GiB read before each timed graph call, stores only block sums (85/85 calls/flushes, N=2).
  Flushed/hot time: 1.12–1.28× at 64–192 MB, 0.99–1.01× at 512–1024 MB *(E12)*. Re-time with AKLR;
  n6144 M64 source tier lost hot too: 1.08–1.13× re-tuned time *(E13)* ([decode](gemm/decode_gemm.md) §8).
  Trivial launch ≈4 µs *(E6)* ([tuning](tuning.md) §1).
- CI requires 3.8.0; matched builds differ from 3.7.0. Check the version and re-time on CI's image
  ([ABI](authoring.md) §11).

## Evidence

L = `/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs`;
T = `/tmp/kbmig/checks_study`. Cases: `triton/checks/`.
N counts runs, not samples/seeds/shards; PASS is not claim support.

- E16 `rt_graph_capture_compile.py` .01–.04; L/870_rt_graph_capture_compile_cleanup.log: 22 executed/21 PASS; 3 logs≠3 passes/check. Limited scope, rejection unproved.
- E1 [source N=1] 154/150 Gluon/4 plain: `census_kernel_inventory.py` .01, L/300_census_kernel_inventory.log.
- E2 [N=2 boundary runs; byte recheck N=1] `s_jit_key_and_buffer_range.py` .07/.07.bytes: partial vectors, 93511 out-of-range bytes each without data/writes; L/{810_s_jit_key_and_buffer_range,850_s_jit_key_and_buffer_range_r1}.log.
- E3 [GPU N=1] LDS163840 ran, 262144 launch failed: `rt_launch_only_errors.py` .03, L/150_rt_launch_only_errors_r1.log.
- E4 [N=3 stream runs] `bench_bytes_in_flight.py` .01/.04/.11/.23; L/{200_bench_bytes_in_flight,800_bench_bytes_in_flight_rep2,880_bench_bytes_in_flight_rep3}.log. No decode equivalence.
- E5 [validate, four rounds/arm] Router split N=8, attention N=7, P3/P4/P7 N=7/2/7, look-ahead N=6: `ab_router256_split_lookahead.py` .01/.02 and `ab_gemm_m16_prefetch_depth.py` .01/.04/.05; L/700_ab_router256_split_lookahead.log, L/700_ab_gemm_m16_prefetch_depth.log (repeat evidence in [tuning](tuning.md)). A/A four Points N=2: `ab_noise_a_vs_a.py` .01, L/700_ab_noise_a_vs_a.log and L/895_ab_noise_a_vs_a_rep4.log.
- E6 [N=5] trivial launch 4.16–4.32 µs: `bench_launch_cost.py` .01, L/200_bench_launch_cost.log + 001–004.
- E7 [compile-only, full scan N=1] `s_constexpr_folding_and_load_width.py` .01, T/s_constexpr_folding_and_load_width.audit_fixed_local.log: real folding/VGPR counterexample; row subset repeated in three compile runs.
- E8 [CPU N=1] `d_source_contract_and_point_census.py` .01/.02/.03/.11, T/local_logs/d_source_contract_and_point_census.log.
- E9 [GPU N=1] `s_inlaunch_order_vs_launch_end.py` .01/.02/.04, L/800_s_inlaunch_order_vs_launch_end.log: 1000 calls/Point; fold 2000 replays×9 sizes needs zeroing. Oversubscription N=2: `rt_cross_workgroup_order.py` .01, L/{100_rt_cross_workgroup_order,150_rt_cross_workgroup_order_r1}.log: 3448–3454/4095 timeouts.
- E10 [compile-only N=2] `s_gluon_compile_rejections.py` .01/.02, T/local_logs/s_gluon_compile_rejections.audit_run{1,2}.log: 63 mismatch/control pairs; 24 warp builds/run.
- E11 [GPU N=1, four rounds/arm] `s_decode_split_grid_router_moe.py` .01–.03, L/800_s_decode_split_grid_router_moe.log; d3bb split4→2 unresolved (1.004×, L/700_ab_moe_decode_gateup_ksplit.log).
- E12 [GPU N=2] `s_aklr_timer_flush_floor.py` .01/.02, L/{800_s_aklr_timer_flush_floor,850_s_aklr_timer_flush_floor_r1}.log: AKLR/4GiB 0.992–1.000, re-read/cold 0.997–1.009. Separate Gluon eager probe N=4 (200/800/880/870), `bench_flush_kind.py` .02: hot/flushed 1.102–1.224 (64–192 MB), 0.991–1.014 (512–1024); no direct HBM proof. L/870_bench_flush_kind_cleanup.log:9; [tuning](tuning.md).
- E13 [N=5] n6144/K2048 M64 `ab_bf16_decode_gemm_design.py` .19 (four runs; [decode](gemm/decode_gemm.md) §8), plus E12's 850_r1 .03: BK128/8w/K0 loses hot too versus BK256/4w/K1.
- E15 [N=1 run, 12 timing blocks] `s_stream_bytes_in_flight.py` .01, L/850_s_stream_bytes_in_flight_r1.log.
  8 MiB: 512×1×16/256×2×16/256×8×4/2048×4×1. 3/4/6 MiB vs same-level four-warp peers:
  paired-log 95% CIs [0.8665,0.8692]/[0.7437,0.7452]/[0.8539,0.8546]. Seven A/A CIs within ±1%;
  .01 FAIL: counterexample.

### 未验证

- E8 (`d_source_contract_and_point_census` .01–.03): full output writes, implicit-sync absence and universal cache detection untested.
- E7/E10: general folding/register savings, Auto/Coalesced behavior and non-Blocked warp rejection unverified (cases/checks above).
- E11/E12: unmeasured split/depth choices and every kernel's first weight read from HBM unverified (cases/checks above).
- `s_prefetch_depth_asymmetry.py` .01: L/850_s_prefetch_depth_asymmetry_r1.log has four paired rounds, M32/64 only, not M16; no universal P7 no-gain proof.
- E14 [one input/arm] `s_rounding_points_vs_bound.py` .01–.06,
  L/850_s_rounding_points_vs_bound_r1.log. Later finite slot/cost controls: [MoE](moe.md); no universal guarantee.
- E15: decode/occupancy causality untested; paired CIs ≠ level-median CIs.

- S1 harness/skills/aklr-kb/triton/SKILL.md:109 measurement conditions.
- S2 aklr/core/utils/testing.py:186 each timed call is captured behind `flush_l2_cache()` (:49-55), which reads a 1 GiB buffer
