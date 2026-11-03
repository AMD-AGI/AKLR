---
title: "Fused MoE (MXFP4 experts) on gfx950 — package map"
kind: op
ops: [moe]
dtypes: [mxfp4, bf16, fp8]
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=1
---

# Fused MoE — package map

Where to start on a fused MoE kernel in FlyDSL for AKLR: what the op computes, what bounds it at each token count,
which cards to read, which kernels to start from, and the AKLR files that govern it. The expert GEMMs are grouped
GEMMs; their card is `../grouped_gemm/flydsl_grouped_moe_inference.md`, and the training grouped-GEMM cards apply as
techniques (`../grouped_gemm/INDEX.md`). "m" is a Point's `n_tokens`; a route is one (token, expert) pair.

## What it computes

`fused_moe@1.0` (`aklr/ops/fused_moe/contract.py`, reference in `reference.py`): for each token and each of its top-k
experts e (`topk_ids`), stage 1 computes act(x · W1_gate,eᵀ, x · W1_up,eᵀ), stage 2 projects that row with W2_e, and
the output row is the sum over the token's k routes of `topk_weights` × that projection.

- `act` is SiLU(gate) · up (`activation` silu), its clamped form with `swiglu_limit` (`gate_mode` interleave), GELU, or
  SiTUv2 with `beta` and `linear_beta`. Under expert parallelism (`has_expert_mask`) the ids are global and only the
  experts set in `expert_mask` contribute.
- Weights are fp4 e2m1 with one E8M0 scale per 1 × 32: `w1` (experts, 2 · inter, model_dim), `w2` (experts,
  model_dim, inter). Layout variants: `row_major`, `shuffled_16x16` (aiter's 16×16 preshuffle, shuffled scales) and
  `gate_up_interleaved` (gate and up rows interleaved, 2-D scales). `act_quant` 4 quantizes the activations to MXFP4,
  8 to fp8 (SiTUv2 only); gate mode interleave keeps bf16 activations.
- The var axis is `n_tokens`; model dim, experts, top-k and inter dim are fixed per schema name.
- Compare: `determinism: bounded`, `max_violation_fraction` 1e-3, one whole-output SQNR ≥ 13 dB bound against the
  float64 reference rounded to the output dtype, with no bias test. Atomic accumulation passes it, and so do fp8
  partial rows and bf16 truncation ([fused_moe_shuffled_16x16.md](../cases/fused_moe_shuffled_16x16.md), Counter-signals).
- Every AKLR kernel computes it in two GEMM stages. How the routes reach their expert's rows is the main choice at
  decode sizes: a separate sort kernel, a workgroup set per route with no sort, or a sort folded into stage 1.

## Bottleneck by regime

| Regime | What bounds the time | Read |
|---|---|---|
| m 1–2 (6–18 routes) | Little work per call, so the launches and the sort weigh (estimated). At m 1 every AKLR kernel runs two launches, no sort and a workgroup set per route, and all four are served, at 1.26–3.36× *(field, E1)*; at m 2 the 33-expert kernel is served with four launches | the two cases |
| m 8–64 with 257 experts | 72–576 routes over 257 experts, so most hit experts are read for one or two routes (estimated); each graph node adds a fixed cost (`../moe/flydsl_moe_decode.md` §1). Folding the sort into stage 1 saved 3.5–6.4 µs of 39–102 µs *(field, E2)* | `../moe/flydsl_moe_decode.md` §1–§2 |
| Decode with tens of local experts | Measured on aiter's FlyDSL MXFP4 kernels: bandwidth. At 60 tokens a 75-layer step read 5.77 TB/s against a 6.27 TB/s read roof measured in the same run (92%); from 10 to 60 tokens the time grew 2.38× while the experts hit grew 2.6× *(field, E3)* | `../grouped_gemm/flydsl_grouped_moe_inference.md` §3–§5 |
| Kimi-K3 SiTUv2, fp8 × MXFP4, m 8–112: 896 experts (TP) or 112 of 896 local (EP8), top-16 | Bandwidth: the weights of the experts the routes touch, 2.19 MB each at inter 384 and 17.5 MB at inter 3072 (estimated from the layout); count the touched experts from AKLR's own inputs (`../patterns/memory_bound.md`, Rule out first). From 32–40 tokens up the delivered kernels read them at 6.0–6.45 TB/s against a 6.5 TB/s read roof, aiter at 5.0–5.97, so at most 5–8% is left; at 8–24 tokens the per-launch costs around the GEMMs weigh, and EP8 measured 1.28–1.59 there *(field, E5)* | `../patterns/memory_bound.md`; `../grouped_gemm/flydsl_grouped_moe_inference.md` §2, §7; `../moe/flydsl_moe_decode.md` |
| Kimi-K3 SiTUv2 at m 893–16384 | Every expert is touched; from ~9k tokens the GEMMs are compute-bound. At 16384 tokens the delivered kernels ran stage 1 at 1970 (TP) and 1889 (EP8) TF/s of useful FLOPs (padding rows not counted) and stage 2 at 1347 and 1764, where aiter ran 1431/1459 and 929/1267 *(field, E6)*; dense fp8 GEMMs of squat shapes reached ~2300–2550 TF/s on MI355X *(field, E7)*. TP's per-route rows and their reduce kernel took 190 of 1533 µs | `../grouped_gemm/flydsl_grouped_moe_inference.md` §2, §6; `../gemm/flydsl_gemm_mainloop.md` §7 |
| m 64–4096 with 33 local experts | Rows repeat per expert: tile-M 32–128, several row tiles of one expert per weight sweep, activations staged through LDS. Served at 1.030–1.609 over aiter at every m from 64 *(field, E4)* | [fused_moe_shuffled_16x16.md](../cases/fused_moe_shuffled_16x16.md) |

## Techniques that apply

| Task | Read |
|---|---|
| Decide the launch structure of a decode-sized call; count graph nodes | `../moe/flydsl_moe_decode.md` §1 |
| Remove the sort launch at 8–64 tokens | `../moe/flydsl_moe_decode.md` §2 |
| Judge a port whose stage 2 uses atomics against its source | `../moe/flydsl_moe_decode.md` §3 |
| Map sorted tokens to expert blocks, exit on the device, persistent stage 2 | `../grouped_gemm/flydsl_grouped_moe_inference.md` §2 |
| Grid shape and tile-N at decode sizes | `../grouped_gemm/flydsl_grouped_moe_inference.md` §3 |
| Cache policy of the weight loads | `../grouped_gemm/flydsl_grouped_moe_inference.md` §4 |
| Tile-M against the sort unit | `../grouped_gemm/flydsl_grouped_moe_inference.md` §5 |
| Stage 2: atomics or a fixed-order reduce | `../grouped_gemm/flydsl_grouped_moe_inference.md` §6, `../gemm/flydsl_gemm_correctness.md` §8 |
| Sort and quantization kernels around the GEMMs | `../grouped_gemm/flydsl_grouped_moe_inference.md` §7 |
| Measuring at decode sizes | `../grouped_gemm/flydsl_grouped_moe_inference.md` §8, `../gemm/flydsl_gemm_small_m.md` §1 |
| Split K over the waves of one workgroup | `../gemm/flydsl_gemm_small_m.md` §4 |
| Kernel-argument preload; rows bound by launch and latency | `../gemm/flydsl_gemm_small_m.md` §6, `../patterns/launch_floor_small_m.md` |
| Preshuffled operands and MX scales | `../gemm/flydsl_gemm_datapath.md` §10, §11 |
| MFMA atom: scaled fp4, or bf16 on dequantized fp4 | `../gemm/flydsl_gemm_mainloop.md` §5 |
| Tile order across XCDs | `../gemm/flydsl_gemm_schedule_and_locality.md` §2 |

## Kernels to start from

- **AKLR's own.** `../optimize/flydsl_levers/flydsl_kernel_library.md` § AKLR's own FlyDSL MoE kernels lists the five
  kernels (eight schema files) and what to read each for. Which Points each file serves, and how it gets there:
  [fused_moe_shuffled_16x16.md](../cases/fused_moe_shuffled_16x16.md) (257, 65 and 33 experts) and
  [fused_moe_interleave_swiglu.md](../cases/fused_moe_interleave_swiglu.md) (clamped SwiGLU). Where the library's items
  run on main:
  - `1fee5982…`: the tile descriptors written by the slot-`% BM` route run at m 32–256 and the loop-carried
    prefetch at m 128, 256, 2048, Points the file does not serve; at its served m 1, 2 the `ds_bpermute` gather runs.
  - `ef15e878…`: `_route_block` runs at no Point (routing is `_quantize_route_fused` at m 2–32, `_route_expert` at
    m ≥ 64, none at m 1); several row tiles per expert (`GROUP_M` > 1) only at m ≥ 512; fp8 rows at m 16, 32, 64 and
    m ≥ 128.
  - `1bf41ccb…`: `_make_small_route` runs at m 4–128, none served; `FUSE_SPLIT_QUANT` (m 1) and `FULL_STAGE_A`
    (m ≥ 256) run at served Points.
  - `4fbd9d02…`: one stage-2 binary serves all four token counts; stage 1 compiles per count.
- **aiter's FlyDSL MoE families** and their knobs: `../optimize/flydsl_levers/flydsl_kernel_library.md` § Other
  FlyDSL families in the baseline aiter (MoE). aiter is the baseline; price it through `fused_moe_baseline`, which
  calls `aiter.fused_moe.fused_moe` (`../grouped_gemm/flydsl_grouped_moe_inference.md` §8).
- **FlyDSL v0.3.2's `kernels/moe/`** and how aiter's copies differ: `../grouped_gemm/flydsl_grouped_moe_inference.md`
  §1.

## In AKLR

- The contract and its compare: `aklr/ops/fused_moe/contract.py`. The api:
  `aklr/ops/fused_moe/api.py`. The baseline call and its environment (`AITER_BF16_FP8_MOE_BOUND=16` for gate mode
  interleave, sglang's DeepSeek-V4 setting): `aklr/ops/fused_moe/baseline.py`.
- Kernel form, scratch and compile function: `aklr/ops/README.md` § The kernel section and ABI (`../SKILL.md`, Stop
  table). Workflow and CI: § Workflow. The serving threshold: § Environment variables. Kernel-less files (a
  definition: loaded and audited, not served): § Schema fields.
- `AKLR_DISABLE=fused_moe@1.0` stops serving every MoE schema.
- Kernel-less `fused_moe` files on main at 0cef4f2e3, counted from `aklr/ops/fused_moe/impls/flydsl/`: 17 of the 25
  files, 200 Points, none served (S1). By schema name, after `fused_moe_`:

| Name | Points | Added by |
|---|---|---|
| `has_expert_mask_shuffled_16x16_model_dim4096_n_experts32_topk2_inter_dim4096_global_experts257` | 53, m 8–65536 | aa5e96bf3 |
| `shuffled_16x16_model_dim4096_n_experts256_topk1_inter_dim512` | 51, m 8–65536 | aa5e96bf3 |
| `shuffled_16x16_model_dim6144_n_experts257_topk9_inter_dim512` | 13, m 1–4096 | 97d5eb0b3 (#81) |
| `situv2_act_quant8_gate_up_interleaved_model_dim3584_n_experts896_topk16_inter_dim384` | 17, m 8–16384 | a0e3cbea1 (#69) |
| `situv2_has_expert_mask_act_quant8_gate_up_interleaved_model_dim3584_n_experts112_topk16_inter_dim3072_global_experts896` | 17, m 8–16384 | a0e3cbea1 (#69) |
| `interleave_has_swiglu_limit_gate_up_interleaved_model_dim7168_n_experts384_topk6_inter_dim384` / `…_inter_dim768` | 9 and 6, m 1–256 | 31222a0d9 |
| `interleave_has_expert_mask_has_swiglu_limit_gate_up_interleaved_model_dim7168_n_experts48_…` / `…_n_experts96_…` (inter 3072, 384 global) | 6 and 4, m 1–128 | 31222a0d9 |
| the four `…_has_num_local_tokens_…` names (hidden 4096 or 7168) | 1 each, m 1024 or 2048 | 39a51ac02 (#71), 31222a0d9 |
| the kernel-less siblings of the four clamped-SwiGLU kernel files | 5 each, m 16–256 | 39a51ac02 (#71) |

## Evidence

- E1 [2026-10, gfx950, FlyDSL 0.3.4.1, N=1, AKLR CI] speedup over aiter at m 1: 2.472 (257 experts, 1fee59821837c26f), 1.258 (33 experts, ef15e87894a8ae06), 1.353 (65 experts, 1bf41ccba6750300), 3.015–3.359 (clamped SwiGLU, four files; 9dcb4d796b597d0b 3.089, a5a343437ee191a7 3.015 from 102877ca0) (5571f4589, CI run 40613440602)
- E2 [2026-10, gfx950, FlyDSL 0.3.2, N=1, bench_cudagraph: graph replay, 1 GiB flush per call, Kineto kernel time, median; one run per arm] 39.09–102.04 µs per call with a separate sort node, 34.38–95.67 µs with it folded into stage 1, at m 8–64 (257 experts, inter 256; local:/workspace/code/gpt_oss_docker/scratch/glm52_sikl_port/evidence/moe_triage_AB.md:13)
- E3 [2026-09, gfx950, FlyDSL ?, N=?, wall clock of a 75-layer MoE step over 8 weight sets against a read-roof kernel in the same process; per-layer times from a minibench] aiter's MXFP4 MoE (`mfma_moe1`/`mfma_moe2`, GLM-5.2 TP4/EP4: 64 local experts, inter 2048, hidden 6144) at 60 tokens: 5.77 TB/s against a 6.27 TB/s roof, 92.0%; 10 → 60 tokens 82.2 → 195.6 µs per layer (2.38×) with 20 → 52 experts hit (2.6×) (local:/workspace/code/gpt_oss_docker/.claude/memory/reference_glm52_moe_bandwidth_bound.md:21; :60; :81)
- E4 [2026-10, gfx950, FlyDSL 0.3.4.1, N=1, AKLR CI] speedup over aiter at m 64–4096: 1.030, 1.147, 1.109, 1.257, 1.609, 1.081, 1.070 (fused_moe_shuffled_16x16_model_dim6144_n_experts33_topk9_inter_dim2048, file ef15e87894a8ae06; 5571f4589, CI run 40613440602)
- E5 [2026-10, gfx950, FlyDSL ?, N=1, `python -m aklr validate --base origin/main` run locally on one MI355X: graph replay, 1 GiB flush per call, one-sided lower bound against aiter; touched experts counted from the same inputs] speedup over aiter at m 8–112: TP 1.373, 1.287, 1.243, 1.194, 1.168, 1.084, 1.069, 1.061, 1.058, 1.054, 1.043, 1.039, 1.038; EP8 1.592, 1.427, 1.347, 1.281, 1.220, 1.278, 1.282, 1.276, 1.262, 1.253, 1.245, 1.229, 1.240; weight bytes of the touched experts over the median candidate time 6.04–6.20 TB/s (TP, m 40–112) and 6.20–6.45 TB/s (EP8, m 32–112), aiter 5.0–5.97 TB/s; read roof 6.5 TB/s from a read-only kernel (the eight K3 kernel files of branch kyle/k3-moe-kernels; local:/shared_nfs/kyle/moe/k3rebase/val2/report.json; local:/shared_nfs/kyle/moe/kf/scan/touched.json; local:/workspace/code/gpt_oss_docker/.claude/memory/reference_mi355_measured_ceilings.md)
- E6 [2026-10, gfx950, FlyDSL 0.3.2, N=1, eager calls on the validate inputs, L2 flushed between calls, torch.profiler kernel times; useful FLOPs 2 · routes · N · K per GEMM] at 16384 tokens, the delivered kernels against aiter: TP stage 1 732.4 µs = 1970 TF/s (aiter 1008.2 µs = 1431), stage 2 535.8 µs = 1347 (776.8 µs = 929), other kernels 264.5 µs with the reduce 190.4 (480.3 with 374.8); EP8 stage 1 762.6 µs = 1889 (987.2 µs = 1459), stage 2 408.1 µs = 1764 (568.3 µs = 1267), other 57.5 µs (508.5); at 9397 tokens TP 1701 / 1200, EP8 1660 / 1511 TF/s (local:/shared_nfs/kyle/moe/kf/flops_run/r2/tp8.txt; local:/shared_nfs/kyle/moe/kf/flops_run/r2/ep8.txt; script local:/shared_nfs/kyle/moe/kf/flops_run/prof_flops.py)
- E7 [2026-09, gfx950, FlyDSL ?, N=?, FlyDSL's own GEMM bench and hipBLASLt on one idle MI355X] dense fp8 GEMM at M=131072: N5632 K2560 2288 / 2474, N6144 K3072 2396 / 2602, N2560 K2560 2167 / 2488, N3072 K3072 2369 / 2664 TF/s (FlyDSL / hipBLASLt); fp8 16384³ 3319 / 2813 (local:/workspace/code/gpt_oss_docker/.claude/memory/reference_mi355_measured_ceilings.md)
- S1 aklr/ops/README.md:76 a schema file with empty `sources` is a definition: loaded and audited, not served or benched
