---
title: "AKLR's Triton and Gluon kernels on gfx950: the family map"
kind: topic
ops: [gemm, moe, attention, gdn, norm]
dtypes: [fp8, mxfp4, bf16, fp32]
regimes: [all]
gens: [gfx950]
triton: 3.8.0
---

# AKLR's Triton and Gluon kernels: the family map

AKLR main (`43e112e`) commits 154 Triton-language kernels for gfx950 (MI355X): 150 Gluon (`@gluon.jit`) and 4 plain
`@triton.jit`, in eleven families. Per family: what it replaces, its Points, files, launches, decomposition, techniques,
compiled resources and CI speedups, to start a new kernel from the closest committed one; packaging and validating it:
[authoring](../authoring.md) §10, §11. The page’s Evidence section records source facts, measured values and their limits; the compiled-resource numbers *(field, E2)* come from the census case `rt_census_compiled_kernels`.

## Route here when

- You start a Triton or Gluon kernel for a new schema and look for the committed kernel nearest in op, dtype and token count, or need to know that AKLR has no Triton kernel for your op or shape (§3).
- You change or tune a family and need its per-Point files and launch structure; the ablations run on these files are in [tuning](../tuning.md) §3, §5, §9, §11, §12.
- You want a shipped example of a technique: async LDS rings, MXFP4 `mfma_scaled`, atomic-ticket sorts, split-KV merges, fused fp8 group quantization (§4).

## 1. Using a committed kernel as the starting point

- **Find it.** A kernel is one schema file, `aklr/<ops|plugins/sglang/...>/impls/triton/<scenario>/<uuid>.json`; its files are the `content` of each `kernel.sources` entry, its entry is `kernel.entry_point` (`kernel.py::run`, or `main.py::run` in the 15 N 3072 fp8 GEMMs). The format is in `aklr/ops/README.md` § The kernel section and ABI. This page names a file by the first four hex digits of its uuid, unique among the 154 Triton files: `find aklr -path '*/impls/triton/*' -name '8411*.json'` finds it. A citation `<schema>#kernel.py:<line>` is a line of a source the schema stores; `python -c 'import json,sys,pathlib; [pathlib.Path(s["path"]).write_text(s["content"]) for s in json.load(open(sys.argv[1]))["kernel"]["sources"]]' <schema.json>` writes the sources into the current directory.
- **One design per Point.** 116 of the 154 files list one Point, the others 2–14; only Qwen3-Next's Points are held by more than one Triton file (§13, §14); both bf16 GEMM scenarios share Points with a FlyDSL file, and `add_rms_norm` has a HIP file at its token counts with another epsilon (§6). Under the token-bound host-AST rule (shape-derived `m`, `rows`, `tokens` or `num_tokens`, repeated assignments left unknown), 73 of the 116 single-Point files (`5ea3`, `2422` among them) retain a decidable host branch or conditional-expression arm for another Point. Follow your Point's branch before you copy a constant (`harness/skills/aklr-port-generated-kernels/references/porting-notes.md`).
- **The shape is in the code.** A file serves one scenario and writes its constant axes as literals: N 3072's 12 K groups as `gl.static_range(12)` (`54f6`, S6), N 7168's `STAGES = 8` stages of 256 K and its 224-tile order `tile * 8 - quotient * 223` (`0675`, S5), 256 experts as `(256,)` vectors and top-8 plus the shared expert as `row * 9` (`d3bb`, S10). Many files also assert the shape on the host (`assert n == 3072 and k == 1536`, S6); no DeepSeek or plain-Triton file does, and the host code of `d3bb`, `b88c`, `f806` ran at other shapes without an error. For another shape read every literal and assert, not only your Point's branch, and recount the grid against the 256 CUs ([tuning](../tuning.md) §6). Buffer accesses take 32-bit offsets and reach only byte offsets from their scalar base below 2^31 − 2, the descriptor's `num_records` *(field, E17)*; of the 137 files using them, 30 check a size against 2^31 on the host (`cache_buffer` in `2ee5`, S24), 4 `static_assert` one, and 103 have neither kind of guard ([gluon/memory](../gluon/memory.md) §3). MoE traps: [moe](../moe.md) §12.
- **Triton 3.8 APIs.** 12 of the 15 N 3072 files call `llvm_fn_attrs`, `compute_efficient_padded_shared_layout` or `gl.amd.slice`, which Triton 3.7 lacks, and declare no `requires`. In the CPU check, 3.7.0 rejects `{"triton": ">=3.8"}` as unmet and `7266` fails to compile without it; in one GPU run, an unmet `>=3.9` under 3.8.0 skips the copy without building or timing it, and dispatch serves M = 6144 from the original (E19; [authoring](../authoring.md) §3).
- **Copy a Point's whole configuration.** Its constants were chosen together: changing one alone made the 24-token DSA file fail to build (`build_failed`, a `CompilationError` in `_attention_split`) and the 1-token MiniMax MXFP4 file crash (`candidate_crash`, exit −11); moving a constant with what it feeds: [tuning](../tuning.md) §5. Where one could be moved both ways, the committed router split-K (256 tokens) was the fastest of the three values tried, and so was the 32-row gated-attention file's pair of KV split count and head tile (tied, S40; with the 64- and 16-row files' values: 0.63 and 0.95); the fp8 GEMM at M = 16 lost prefetching 3 K groups instead of 5 (0.87), not 4, while the measured PREFETCH7/5 time ratios were about 0.9998 and 0.9968; gain or equivalence was not established *(field, E1)*. Re-tune with `python -m aklr validate` ([tuning](../tuning.md) §2), not a neighbour's value.
- **Helpers are copied, not shared.** A schema imports its schema-local helpers only from its own files, so families copy their helpers: `initial_state.py` is byte-identical in all 26 GDN prefill files, `_math.py` in the 10 DeepSeek files that use it.
- **Reading the numbers.** Resources come from compiling every schema once on MI355X with Triton 3.8.0 *(field, E2)*; "VGPR" is a kernel's `.vgpr_count`, architectural VGPRs plus AGPRs (a 4-warp 256 × 256 `tl.dot` for gfx950 with 256 of each reports 512). A CI speedup is the lower bound CI stored for one Point against that op's baseline (`harness/skills/aklr-read-validation/references/ruler.md`); dispatch serves a Point from `AKLR_MIN_SPEEDUP` up (`aklr/ops/README.md` § Environment variables). It describes a whole design at one Point against one op's baseline: do not rank families by it. A file's own values are its `workloads[].speedup`; read them before you start from a file, because the family ranges below hide weak Points, such as N 7168 at M = 128 (1.114), DeepSeek inter 256 at 128 tokens (1.030) and DSA 64 heads at 8 tokens (1.030) *(field, E3, E6, E10)*.

## 2. The families

Directories are under `aklr/`; a leading `…/` stands for `plugins/sglang/models/`, a `…` inside a name for the middle
of a scenario name (shapes, layout). Launches are the kernel nodes of one call at the schema's Points (served or not),
a `q.to(torch.bfloat16)` conversion included. Contract is the schema's `compare.determinism` ([numerics](../numerics.md)
§1, §8): the `bitwise` files use no atomic and each repeated bit for bit at its first Point; all 31 files
with an atomic (tickets, job counters, LDS scatters) are `bounded`. The fp8 GEMM's `bounded` is marked
`# FIXME(main-kernels)` (`aklr/ops/gemm/contract.py:269`), and none of its 24 files uses an atomic. In the MoE rows a
route is one (token, expert) pair, a ticket the position that a relaxed `atomic_add` on its expert's counter returns, a
job a run of up to BM routes of one expert, the rows of one gate-up tile.

| Family (§) | Schema directories | Replaces | Points | Files per schema | Launches | Language | Contract |
|---|---|---|---|---|---|---|---|
| fp8 block-scaled GEMM (§5) | `ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/`, `…_n7168_k2048/` | aiter `gemm_a8w8_blockscale` | 15 + 9: M 1–8193, 1–256 | 3 (`main.py`, `adapter.py`, `kernel.py`) at N 3072, 1 at N 7168 | 1 | Gluon | bounded |
| bf16 decode GEMM (§6) | `ops/gemm/impls/triton/gemm_n6144_k2048/`, `gemm_n2624_k6144/` | aiter `gemm_a16w16` | 6 + 4: M 2–64 | 1 | 1; 2 at N 2624, M 64 | Triton | bitwise |
| RMSNorm, add + RMSNorm (§6) | `ops/rms_norm/impls/triton/rms_norm_hidden_size6144/`, `add_rms_norm_hidden_size6144/` | aiter `rmsnorm2d_fwd`, `rmsnorm2d_fwd_with_add` | 14: 3–4096; 7: 4–256 | 1 | 1 | Triton | bitwise |
| DeepSeek-V2/GLM MXFP4 MoE (§7) | `…/deepseek_v2/impls/triton/deepseek_v2_mxfp4_moe_…_inter_dim{256,512}/` | SGLang `DeepseekV2MoE.forward_normal` | 24: 1–4192 | 1–5 | 3–6 | Gluon | bounded |
| MiniMax-M2 MXFP4 MoE (§8) | `…/minimax_m2/impls/triton/minimax_m2_mxfp4_moe_…/` | SGLang `MiniMaxM2MoE.forward_normal` | 30: 1–16384 | 1 | 3–8 | Gluon | bounded |
| MiniMax-M2 FP8 MoE (§9) | `…/minimax_m2/impls/triton/minimax_m2_fp8_moe_…/` | SGLang `MiniMaxM2MoE.forward_normal` | 28: 1–16384 | 1 | 3–8 | Gluon | bounded |
| Qwen2-MoE router, fp8 experts, fp8 router + experts (§10) | `…/qwen2_moe/impls/triton/qwen2_moe_{router,fp8_experts,fp8_router_experts}_…/` | SGLang `Qwen2MoeSparseMoeBlock._forward_router_experts` | 7: 8–4096; 3: 8–128; 4: 1–32 | 1 | 2; 3–5; 4–7 | Gluon | bounded |
| DSA sparse MLA decode (§11) | `plugins/sglang/layers/dsa_backend/impls/triton/dsa_sparse_mla_decode_n_heads{64,8}_kv_dim576_topk2048/` | SGLang `DeepseekSparseAttnBackend._forward_triton_decode` | 5 + 3: 8–32, 32–256 | 1–2 | 2–3 | Gluon | bitwise |
| Qwen3.5 gated attention decode (§12) | `…/qwen3_5/impls/triton/gated_attention_decode_group_fp8_quant_…/` | SGLang `Qwen3_5AttentionDecoderLayer.self_attention` | 9: 1–256 | 3 | 3 | Gluon | bitwise |
| GatedDeltaNet decode (§13) | `…/qwen3_5/impls/triton/gdn_decode[_group_fp8_quant]_…/`, `…/qwen3_next/impls/triton/qwen3_next_gdn_decode_group_fp8_quant_…/` | SGLang `Qwen3_5GatedDeltaNet.forward`, `Qwen3GatedDeltaNet.forward` at decode | 8 + 1: 1–256; 9: 1–256 | 1 (3 in `f440`) | 2; 1–2 | Gluon | bitwise |
| GatedDeltaNet chunked prefill (§14) | `…/qwen3_5/impls/triton/gdn_prefill_group_fp8_quant[_max_seq_len<L>]_…/`, `…/qwen3_next/impls/triton/qwen3_next_gdn_prefill_group_fp8_quant[_max_seq_len<L>]_…/` | the same two `forward` at prefill | 21: 512–16384 tokens in 1–8 sequences; 13: 1024–16384 in 1–16 | 2–3 | 5–8 | Gluon | bitwise |

## 3. Which family to start from

| Your op | Start from | What you inherit |
|---|---|---|
| GEMM, e4m3 A and B, fp32 scales per 1 × 128 of A and 128 × 128 of B | §5's current N 3072 / K 1536 and N 7168 / K 2048 files; nearest-shape selection is unverified | CPU host-AST replay: one launch at each of 24 current Points; `7266` rejects independently strided A and B. Source calls use unit-scale `mfma_scaled` and async LDS APIs; fp32 group-scale def-use was not checked (E20) |
| GEMM, bf16, M ≤ 64 | §6: `f806` (K 2048), `b88c` (K 6144) | the weight loaded with `.cg`, programs remapped across XCDs, no atomics; split-K with a separate fixed-order reduce kernel in `b88c` only (M = 64) |
| GEMM with MXFP4 operands (e2m1, e8m0 per 32) | the expert GEMMs of §7 or §8 at your token count | `mfma_scaled` on e2m1 with e8m0 scale layouts, MXFP4 activation quantization in the kernel |
| MiniMax router projection, bf16 hidden and fp32 weights | the specific tested Points of §8 and §9, not a general fp32-accuracy recommendation | ISA and sampled projections at 58 Points; full-logit comparisons at cb88@2048, b94f@4096 and bb66@1024 gave L2 error no worse than executed FP32 in 96/96 cases, maxabs in 93/96. A separate 28-input study covered 13 additional arithmetic branches, including M1–16 VALU and M32 FP32 MFMA: L2 noninferiority 27/28, maxabs 25/28. These finite observations do not certify every specialization or top-8 equivalence (E21; [numerics](../numerics.md) §5) |
| bf16 router requiring the contract’s once-BF16 rounding | §10’s finite router + expert observations, not whole-layer FP64 equivalence | fp64 VALU products at 1–4 tokens, fp64 MFMA at 32; mixed-precision experts still require their original contract |
| bf16 router with top-k | §10 router, 8–256 tokens (`4801`, `5f20`, `a878`, `cd47`, `c388`) | a split-K projection of exactly 256 CTAs, then per-row top-k; outputs bitwise identical over 20 repeats/check; 2232/4096 not run by this check (E25) |
| MoE layer, MXFP4 experts and a shared expert | §7's source map, not a measured transfer recommendation | current host-selected gate-up launches, grids and W1 provenance were recorded in the 89-Point / 58-schema MoE CPU census; ninth-route and shared-combine semantics remain unverified (E20) |
| MoE layer, MXFP4 experts, no shared expert, all or some experts on this rank | §8's E256 and E64 source maps | gate-up provenance, grids, integer buffers and source load pointers recorded; `expert_offset`, effective job rows and ticket/ballot classification remain unverified (E20) |
| MoE layer, fp8 experts with 128 × 128 block scales | §9 and §10's router + experts source maps | gate-up host paths recorded; job-queue counts, SWAR sort, register-resident quantization and sigmoid shared-gate semantics remain unverified (E20) |
| MoE experts, routing given | §10 fp8 experts | per-route split-K (`5c35`), an LDS `atomic_scatter_add` sort (in one CTA in `8698`; also in `d425`), persistent grid-stride stages (`8698`) |
| SwiGLU + requantization as its own launch | `d3bb`'s `_activate_quantize` (§7: K-split gate-up partials summed in order, SwiGLU, MXFP4 per 32; S10); fp8 per 1 × 128: `_activate_quantize` of `5cac` (§9) or `5c35` (§10) (S44) | no standalone op on main: these are launches inside the MoE files ([moe](../moe.md) §8) |
| Decode attention over gathered rows (top-k indices) | §11 | key splits merged by a separate atomic-free launch (7 of 8 files; `fbe6` runs one workgroup per token, no merge), LDS-staged K and V |
| Decode attention over a paged KV cache the op writes, fp8 output | §12 | a prologue task grid, split-KV partials, a finish that writes the fp8 values and their scales: three launches at all 9 Points |
| Q/K RMSNorm, RoPE or mRoPE, fp8 paged KV-cache write | §12's `prologue.py` (`_qk_norm_rope_cache` in `d9cb`, `dfb1`) | a task grid of one-wave CTAs, a whole number per token (320 at 32 rows, 1,024 at 256; [attention](../attention.md) §11) |
| A per-slot recurrent state stepped once per decode token | §13 | the state split over CTAs or owned by one, an in-place history shift |
| A chunked linear recurrence over long sequences | §14 | parallel chunk phases, one sequential launch, then a parallel epilogue |
| Row-wise normalization | §6 RMSNorm (plain Triton; `rms_norm` is not served at 3–18 tokens, nor at 64 at `43e112e` (1.024), where `d908d01`'s CI run stored 1.040 *(field, E5)*); in Gluon only fused: the Q/K RMSNorm of §12's prologue, the gated RMSNorm of §13 and §14 | one row per program, grid-stride from 1024 rows |
| Many Points from one source | a host branch per Point (`cf63`: 14 Points, `f806`: 6), or `f440`'s table (§13), unmeasured at `43e112e` but measured on main in `d908d01`: the per-size files beat it at 7 of 9 Points *(field, E13)* | one configuration row per token count; an unlisted count raises |
| An output quantized to fp8 per 1 × 128 for the next GEMM | the finish of `3b99` (§14) or `d9cb` (§12), the last launch of its call | amax, scale and e4m3 codes in the epilogue; `3b99` was bit-exact against a reference quantization of its own bf16 output on 100000 random groups ([numerics](../numerics.md)) |

**Not covered by a Triton kernel.** These contracts have only FlyDSL or HIP kernels on main, or none. Start from
[authoring](../authoring.md) §10 and borrow:

Recommendation: price an uncovered Point before implementing it. The finite BF16 N16/K128 M1/M64 examples have measured legal baselines, but do not prove that this ordering saves development effort or selects the best strategy in general.

| Contract (kernels on main) | Nearest Triton code |
|---|---|
| `gemm_a4w4_mxfp4@1.0` (none) | the e2m1 `mfma_scaled` step of `6557`'s `_project_gate_up_fold` (§8, S42), without its routing: that call looped over K on row-major operands and [rows, K/32] scales matched torch's fp32 matmul at M = N = 32, K = 512 (`6557`'s function itself not run) |
| `per_group_quant_fp8@1.0` (HIP) | `4f25`'s `_store_quantized_head` (§13, S43): its codes and amax × fl32(1/448) scales equal the op's reference bit for bit on random bf16 groups ([numerics](../numerics.md) §2); the op stores its scales group-major, [n_groups, n_tokens] |
| `per_group_quant_mxfp8@1.0` (HIP) | `d3bb`'s `_quantize_groups` (S10) as its routed experts call it (`shared=False`): the e8m0 exponent rounded up from the bits of peak / 6 (MXFP4; `shared=True` takes the even rule); the op's reference rounds amax × (1/448) up the same way (`aklr/ops/quant/reference.py`) |
| `biased_grouped_topk@1.0` (FlyDSL) | §7's selection: sigmoid plus fp32 correction bias, top-8 of 256, renormalized × 2.5 (`d3bb`'s `_select_routes`, S10) |
| `fused_moe@1.0`, `mega_moe@1.0` (FlyDSL) | the MXFP4 expert GEMMs and layers of §7–§8: `fused_moe@1.0` is `per_1x32` only (33 schemas) and shares their `shuffled_16x16` layout |
| `flash_mla_with_kvcache@1.0` (FlyDSL), `sparse_prefill_mxfp8@1.0` (HIP) | §11's gathered-row decode; no Triton softmax-attention kernel attends over a prompt (§14's GatedDeltaNet prefill runs a linear recurrence over it) |
| `gemm_a8w8@1.0` (none), `bmm@1.0`, `fp8_paged_mqa_logits@1.0`, `topk_transform_paged@1.0`, `mhc_*@1.0` | nothing close |

Within the covered contracts, fp8 block-scaled GEMMs exist at N 3072 / K 1536 and N 7168 / K 2048 only, with row-major
`b`. For another N or K start from a ring whose slots do not grow with K (the 8-slot skeleton of [gemm](../gemm.md) §5
uses a fixed 32 KiB LDS allocation with 8 slots (matched CPU-offline builds at N7168/K2048 and N4096/K7168)): §5's N 7168 files, built for K 2048, stream `STAGES = 8` stages of 256 K through
4 A + 6 B LDS slots at M = 16 (`1db6`), 4 + 4 at M = 32 (`ae4f`) and 3 + 3 at M = 64 (`0675`), 64–72 KiB *(field, E2)*;
`1db6` takes its 6 B slots only when `a`'s row stride is 2048 (S5, S46). The N 3072 files at M = 16 and 32 do not carry
over: they give each 128-K group its own LDS slot (S6), 48 and 49 KiB at K 1536 (M = 1, 2, 4, 8: 9, 9, 12, 36 KiB)
*(field, E2)*, about 224 KiB at K 7168 (estimated), above the 160 KiB one work-group may request ([gfx950
memory](../../hardware/gfx950/memory_and_sync.md) §4). Longer K and a preshuffled `b` exist only as FlyDSL kernels
(`aklr/ops/gemm/impls/flydsl/gemm_a8w8_blockwise_scaled_b_shuffled_*`, K 256–16384); long-K evidence in plain Triton is
the bf16 K 6144 GEMM ([gemm/decode_gemm](../gemm/decode_gemm.md) §5). bf16 GEMMs exist at N 6144 / K 2048 and N 2624 /
K 6144, norms at hidden 6144 only.

## 4. Techniques across the families

Counts are files whose sources use the construct (a text scan of the 154 files, comments excluded, dead branches
included); Example names one shipped instance.

| Technique | Where (files) | Example | Read |
|---|---|---|---|
| `buffer_load` / `buffer_store` (`gl.amd.cdna4`; the same functions as `gl.amd.cdna3` in 37 files): scalar base, 32-bit offsets | 137 files (`buffer_load` in 128, `buffer_store` in 93), every Gluon family | `2ee5`, gated on the cache's size (S24) | [gluon/memory](../gluon/memory.md) §3 |
| async direct-to-LDS copies (`async_copy.buffer_load_to_shared`, `commit_group`, `wait_group`) | fp8 GEMM 24, DeepSeek 6, Qwen2 router 6 and experts 1, DSA 4, gated attention 3, MiniMax FP8 2; staging the 8-token router's hidden tile this way gained nothing measurable over `gl.load`s converted through LDS (8.45 µs both; the weight's load layout changed with it) *(field, E1)* ([tuning](../tuning.md) §5) | `7266` (S1) | [gluon/memory](../gluon/memory.md) §8, §12 |
| padded LDS layouts (`PaddedSharedLayout`, `compute_efficient_padded_shared_layout`) | fp8 GEMM 14, MiniMax MXFP4 11, DSA 5, Qwen2 router 5, DeepSeek 4, gated attention 3, MiniMax FP8 2 | `7266` (S1) | [gluon/memory](../gluon/memory.md) §9, [gluon/layouts](../gluon/layouts.md) §4 |
| `mfma_scaled` on e4m3 with `None` scales | fp8 GEMM 24 (each 128-K group's product multiplied by the fp32 block scales), MiniMax FP8 9, Qwen2 router + experts 4 | `7266` (S1) | [gluon/mfma](../gluon/mfma.md) §2, [numerics](../numerics.md) §4 |
| `mfma_scaled` on e4m3 with `None` scales into a running accumulator | DSA 5 (`0bb2`, `1148`, `2ee5`, `2f70`, `f60f`) | `2ee5` (S24) | [attention](../attention.md) §10 |
| `mfma_scaled` on e2m1 with e8m0 scales | MiniMax MXFP4 23, DeepSeek 19 | `6557` (S42) | [gluon/mfma](../gluon/mfma.md) §2, §3, [numerics](../numerics.md) §3 |
| K groups folded into the MFMA rows at tiny M | all 8 fp8 GEMM files at M ≤ 8; the MiniMax MXFP4 decode file `6557` (`gate_fold`) | `bd01` (S2), `6557` (S42) | [gluon/mfma](../gluon/mfma.md) §8, [gemm](../gemm.md) §4 |
| `gl.inline_asm_elementwise` | 83: MiniMax MXFP4 22, DeepSeek 19, Qwen3.5 GDN decode 8, Qwen3-Next GDN decode 7, fp8 GEMM 6, MiniMax FP8 6, Qwen2 router 5, Qwen3.5 GDN prefill 4, DSA 3, Qwen2 router + experts 3 | `d3bb`'s `_next_expert` (S10) | [gluon/wave_ops](../gluon/wave_ops.md) §1, §2 |
| relaxed global integer atomics (tickets, job counters, bitmasks); LDS `atomic_scatter_*` | DeepSeek 12, MiniMax MXFP4 11, MiniMax FP8 6; LDS: Qwen2 experts 2, MiniMax FP8 1 | `8411` (S15), `8698` (S23) | [moe](../moe.md) §6, [numerics](../numerics.md) §8 |
| fixed-grid graph capture/replay | MiniMax `8411`, `51ae`, `147b`, `82e9`: no capture/accuracy findings; two input sets have identical eager grids (6/6/6/8 launches); different routing, worst-case sizing and device early exit not independently tested (E24) | `8411` (S15) | [authoring](../authoring.md) §8 |
| several roles in one grid, split by program id | program-id branches calling different helpers (a static scan that also counts bounds branches): DeepSeek 18 of 19, MiniMax MXFP4 19 of 23, MiniMax FP8 7 of 9, Qwen2 router + experts 4 of 4, gated attention 7 of 9 files; a finish branch in 6 Qwen3.5 GDN prefill files | `5751` (S9) | [moe](../moe.md) §3 |
| a fixed-order merge launch instead of float atomics (bitwise contracts) | bf16 GEMM at N 2624, M 64 (a second kernel adds the six partials in order); DSA and gated attention, which use no atomic and repeat bit for bit | `f60f` (S25), `b88c` (S7) | [attention](../attention.md) §2, [numerics](../numerics.md) §8 |
| program-id tile-order remaps: a non-AKLR probe with eight readers of the same chunk favored ids congruent mod 8 by 1.76–2.17× in three runs *(field, E18)*; actual-Point A/Bs below do not establish a unique L2 cause *(field, E23)* | The host census records program-id tile-order implementations in four families, and four legal original-Point remap/identity comparisons exist. These are tile-order/address and finite timing facts, not evidence of actual XCD placement or a unique L2 cause. The b88c source describes contiguous N streaming per XCD and K-phase staggering; no located author record establishes the full-family XCD-L2 motive. | `bd01`, `083b` (S3) | [gluon/memory](../gluon/memory.md) §14, [../../gemm/schedule_and_locality.md](../../gemm/schedule_and_locality.md) §2 |
| MiniMax router instruction observations, not a precision guarantee | Three MiniMax Points have complete-logit comparisons to executed GPU FP32 and same-input FP64 (L2 noninferiority 96/96, maxabs 93/96). Qwen2 M1/2/4/32 passed all 32 original SQNR cases; small-M once-BF16 logits matched and M32 FP64 partials had zero measured error, but mixed-precision experts are not FP64-equivalent. GDN passed its 168 recorded original contracts. In the additional same-input two-Point random/cancel GPU FP32 recurrence diagnostic, the predeclared 4×FP32-error + 4×eps32×oracle-norm limits passed only short/random; worst/random and both cancel inputs exceeded the limits. All four original contracts passed. The saved actual/FP32/FP64 tensors were downloaded, all twelve file hashes matched, and full-state shapes and error metrics were independently recomputed on CPU. The diagnostic remains 1/4, not a blanket FP32-accuracy guarantee. | `99f4` remains a source citation, not precision evidence (S22) | [numerics](../numerics.md) §5 |
| fused e4m3 quantization of an output or GEMM input | every file of gated attention, GDN decode and prefill (the bf16 op's `cd00` excepted), MiniMax FP8 and the Qwen2 experts: 64 files | `4f25` (S43) | [numerics](../numerics.md) §2 |
| `enable_fp_fusion=False` on launches | 42: DeepSeek 19, Qwen3.5 GDN prefill 11, Qwen3-Next GDN prefill 11, DSA 1; set per Point in 4 more | `d3bb` (S10) | [numerics](../numerics.md) §6, [tuning](../tuning.md) §9 |
| `llvm_fn_attrs`; `amdgpu-ieee` only in the Qwen2 routers | 26: Qwen2 router 5, Qwen3.5 GDN prefill 5, fp8 GEMM 4, MiniMax MXFP4 4, Qwen3-Next GDN decode 4, MiniMax FP8 3, DeepSeek 1 | `c388` (S20) | [tuning](../tuning.md) §9 |
| no MFMA at all | 22: GDN decode 18, gated attention at 1–2 rows 2, RMSNorm 2 | `2991` | [gluon/mfma](../gluon/mfma.md) §2, [gdn](../gdn.md) §1 |

**Measured remap scope, not a library-wide L2 rule.** In one run of seven actual GEMM/MoE Points, each with N=4 balanced AB/BA pairs and zero accuracy findings, A is the committed order and B changes only the tile order. Every B sample was slower than every A sample, but only five Points met the prespecified >=1% plus full-arm-separation criterion (mean B/A >=1.01, all B slower than all A, paired CI above 1); check .01 therefore FAILs validly. Mean B/A is the ratio of arm means; CI below is the 95% t interval of the paired mean ratio, not an independent-run interval *(field, E23)*.

| Actual Point | Mean B/A | Paired-ratio 95% CI | Criterion |
|---|---|---|---|
| N3072/K1536, M64 `54f6` | 1.11993 | 1.0965–1.1436 | met |
| N3072/K1536, M128 `3480` | 1.05936 | 1.0566–1.0621 | met |
| N3072/K1536, M2048 `14c8` | 1.00651 | 1.0054–1.0076 | resolved change, below 1% |
| N3072/K1536, M4096 `5ea3` | 1.01862 | 1.0181–1.0191 | met |
| N3072/K1536, M8192 `514a` | 1.00935 | 1.0085–1.0102 | interval crosses 1.01; >=1% not established |
| N7168/K2048, M128 `1bbf` | 1.04045 | 1.0391–1.0418 | met |
| DeepSeek inter256, 2048 tokens `c524` | 1.02654 | 1.0104–1.0428 | met; **whole MoE call**, not an expert GEMM |

CPU maps assuming `pid % 8` estimate locality, not actual CTA placement or L2 fills: the `14c8`/`5ea3`/`514a` edits reduce modeled weight-reader XCD counts while increasing estimated total A+B fill. Thus "all shared-weight consumers on one XCD" does not describe every winning order. There are no hardware L2 counters or mechanism-isolating measurements here; the L2 explanation remains unknown. For the actual M256 `199c` Point at N3072/K1536, replacing `pn ^ (pm * 2)` with `pn` passed accuracy and reduced mean time 6.0875 → 5.906 us: B/A 0.97012, paired CI 0.9689–0.9713, N=4 balanced pairs in one run. This is measured M256 evidence, not an extrapolation from M128 *(field, E23)*.

**One weight reader does not imply no timing effect.** Removing the permutation at N7168/K2048, M64 `0675` was equivalent within its prespecified ±1.5% band: B/A 0.99305, CI 0.9856–1.0005, N=4 balanced pairs, not exact zero. N3072/K1536 M1 `bd01` (N=12, B/A 1.00434, CI 0.9895–1.0196) and M16 `083b` (N=12, B/A 1.00803, CI 0.9953–1.0211) remain unresolved against their ±1.5% and ±1.3% bands. N7168/K2048 M2 `ff2d` (N=12) instead has B/A 1.02350 and CI 1.0101–1.0372: a resolved >1% increase despite mixed pair signs. All are balanced pairs within one run with zero accuracy findings; check .02 validly FAILs. Pair signs alone prove neither significance nor equivalence; these controls establish neither unique L2 causation nor absence of L2 involvement *(field, E23)*.

In the 897 compiled AKLR kernels *(field, E2)*: direct-to-LDS `buffer_load … lds` in 71 (fp8 GEMM 24 of 24, MiniMax FP8
17 of 188, DeepSeek 10 of 116, bf16 GEMM 6 of 7, Qwen2 router 6 of 14, DSA 4 of 15, gated attention 3 of 27, Qwen2
experts 1 of 13); transposing LDS reads, all `ds_read_b64_tr_b8`, only in DSA 8, gated attention 6, DeepSeek 5, MiniMax
MXFP4 3; `v_cvt_scalef32_pk_fp4_f32` in DeepSeek 36, MiniMax MXFP4 62; global or buffer atomics only in DeepSeek 15,
MiniMax FP8 17, MiniMax MXFP4 18; `v_permlane16_swap` / `v_permlane32_swap` in 144; `ds_bpermute_b32` in 291.

## 5. fp8 block-scaled GEMM

`a` (M, K) and `b` (N, K) in e4m3, fp32 scales per 1 × 128 of `a` and 128 × 128 of `b`, bf16 out; 15 files at N 3072,
K 1536 and 9 at N 7168, K 2048, one Point each. Op card: [../../gemm/INDEX.md](../../gemm/INDEX.md); Triton side:
[gemm](../gemm.md).

| M | N 3072, K 1536 | N 7168, K 2048 |
|---|---|---|
| 1–4 | `bd01` (1) `c194` (2) `d622` (4) `_narrow_small_gemm`: 768 programs × 1 wave, 4M outputs each | `362c` (1) `ff2d` (2) `0b10` (4): 448 × 1 wave, 16M outputs each |
| 8 | `0e15` `_packed_m8_gemm`: 192 × 2 waves | `1e02` `_packed_eight`: 448 × 1 wave |
| 16–64 | `083b` (16) `fcf2` (32) `54f6` (64): 192–768 × 1 wave, 256 outputs each | `1db6` (16) `ae4f` (32) `0675` (64) `_pipeline_output`: 224 × 2–4 waves, 32M outputs each |
| 128–256 | `3480` (128) `199c` (256) `_decode_kernel`: 192 × 4 waves, 2048 and 4096 outputs each | `1bbf` (128) `fee7` (256): 224 and 448 × 4 waves, 4096 outputs each, 32 × 32 × 64 MFMA |
| 1024–8193 | `8888` (1024) `14c8` (2048) `5ea3` (4096) `7266` (6144) `514a` (8192) `8319` (8193): 512–1560 × 4 waves, 6144–24576 outputs per bulk program | — |

- **Decomposition.** One launch, one output tile per program over all of K: no split-K, atomics or scratch. Each 128-K group produces a unit-scale partial, sometimes through several `mfma_scaled` calls, that the two fp32 scales multiply before it is added (S1); the epilogue casts to bf16 and stores.
- **Techniques.** In all 8 files at M ≤ 8 the MFMA rows hold (token, K group) pairs and the matching-group diagonal is
  kept (S2; [gemm](../gemm.md) §4, [../../gemm/small_m.md](../../gemm/small_m.md)). All 24 stage tiles with async
  `buffer_load_to_shared`, `commit_group` and `wait_group` ([gemm](../gemm.md) §5, [gluon/memory](../gluon/memory.md) §12).
  At N 3072, the program-id address maps under the `pid % 8` model give one contiguous N run per group at M 1–4 (N 7168 too at M 2, 4), stripes
  that keep all M tiles of an N tile in one group at M 16–64, and windows of 2 or 4 ids on one N tile at M 2048–8193 (S3;
  [gemm](../gemm.md) §8, [../../gemm/schedule_and_locality.md](../../gemm/schedule_and_locality.md) §2); N 7168 has no
  remap at M 1, 16, 32. M = 8193 compiles the bulk at M = 8192 and adds 16-row tail programs (S4; [gemm](../gemm.md) §7).
- **Copy from.** The per-group step of `7266` (S1; shown and explained in [gemm](../gemm.md) §2), and the N 7168 host:
  one launch of `_pipeline_output` with shape and strides as constexprs and `waves_per_eu` by M (S5).
- **Resources** *(field, E2)*: 24 kernels; `num_warps` 1 (10), 2 (2), 4 (12); `waves_per_eu` 1–8 in 13; VGPR 40–242,
  AGPRs in 8 (8–32); LDS 9–72 KiB; no spills; `v_mfma_scale_f32_16x16x128_f8f6f4` in 21, `…_32x32x64_…` in 3.
- **CI speedups** *(field, E3)*: 1.114 (N 7168, M 128) to 1.840 (N 7168, M 8) over the 24 Points.

## 6. Plain Triton: bf16 decode GEMM and RMSNorm

The four `@triton.jit` files. Why plain Triton fits them, and what its compiler adds (coalescing, `num_stages`
pipelining, buffer instructions): [authoring](../authoring.md) §2. Each GEMM kernel but the split-K reduce holds
`buffer_load_dwordx4 … lds`; the 6 RMSNorm kernels load with `buffer_load_dwordx4`, no `global_load` *(field, E2)*.

- **bf16 decode GEMM**, `trans_b`, no bias: `gemm_n6144_k2048/` (`f806`, M = 2, 4, 8, 16, 32, 64) and `gemm_n2624_k6144/`
  (`b88c`, M = 2, 4, 8, 64). One workgroup per N tile (192 at N 6144, 164 at N 2624), each `tl.dot` accumulating in fp32,
  the weight loaded with `.cg`; each XCD streams one contiguous run of N tiles in `b88c` at M ≤ 8 and `f806` at M ≤ 16,
  8-tile chunks (3 runs per XCD) in `f806` at M 32, 64 (`b88c`'s `_xcd_remap` is a permutation for any grid size); at
  some M a per-workgroup K phase. One launch, except `b88c` at M = 64: split-K 6 into fp32 scratch (246 workgroups), then
  a second kernel adds the six partials in a fixed order. The configuration comes from a `@functools.cache` compile
  function, `_launch_for(a_shape, b_shape)`, that returns the launch closure (S7). Tiles by M, split-K under the bitwise
  contract, M = 1 and re-tuning for cold-cache timing: [gemm/decode_gemm](../gemm/decode_gemm.md).
- **RMSNorm**: `rms_norm_hidden_size6144/` (`cf63`, 3–4096 tokens), `add_rms_norm_hidden_size6144/` (`b391`, 4–256
  tokens). The 6144-wide row as three 2048-wide chunks in registers, the sum of squares in fp32, `.cg` loads, `.wt`
  stores, one program per row *(field, E2)*; `rms_norm` from 1024 rows gives each program `m // 512` rows, the next
  row's loads pipelined and the weight loaded once (S8):

  ```python
  for it in tl.range(0, LOOP, num_stages=NS):
      row = tl.program_id(0) + it * tl.num_programs(0)
      rmask = cmask & (row < M)
      base = row.to(tl.int64) * H + cols
      x0 = tl.load(x_ptr + base, mask=rmask, other=0.0, cache_modifier=".cg").to(tl.float32)
  ```
- **Resources** *(field, E2)*: GEMM 7 kernels, `num_warps` 2, 4 or 8, `waves_per_eu` 2 in 6, VGPR 26–106, LDS up to
  about 148 KiB of the pipeliner's buffers, `v_mfma_f32_16x16x32_bf16`; RMSNorm 6 kernels, 4 warps, VGPR 49–79, 16 B of LDS.
- **CI speedups** *(field, E4, E5)*: GEMM 1.048 (N 2624, M 2) to 1.292 (N 6144, M 8); `rms_norm` 1.006 (4 tokens) to
  1.212 (2048), below the default `AKLR_MIN_SPEEDUP` and so not served at 3–64 tokens; `add_rms_norm` 1.024 (8 tokens,
  not served) to 1.093 (256).
- **Other languages at these Points.** The three scenarios also hold a FlyDSL or HIP file, and dispatch serves each
  Point with the best measured file of any language (`aklr/ops/README.md`, the paragraph on files that share a
  `<name>`). The HIP `add_rms_norm` (`fc82`) stored 1.038 and 1.067 at 32 and 128 tokens against `b391`'s 1.035 and
  1.062, so it is the one served there, and it alone holds 512–8192 tokens. FlyDSL files serve M = 1 of both GEMM
  scenarios (1.191 at N 6144, 1.134 at N 2624) and store 0.360–1.045 at the Triton files' Points; at N 2624, M 16 and
  32, and in both scenarios from M 128, the only files are FlyDSL ones below 1.0 (0.396–0.998), so no AKLR kernel is
  served there *(field, E16)*. A new kernel at a shared Point has to beat the file served there.

## 7. DeepSeek-V2 / GLM MXFP4 MoE

SGLang's `DeepseekV2MoE` on one rank: bf16 router logits, the sigmoid top-8 of 256 with an fp32 correction bias, weights
renormalized × 2.5, the shared expert as slot 256 at weight 1, MXFP4 experts (e2m1 with e8m0 per 32, aiter's 16 × 16
shuffled weights) and their weighted sum. `inter_dim256` and `inter_dim512` specify this rank’s expert intermediate dimension; their original TP8/TP4 capture attribution is not certified by the retained metadata; 19 files, 24 Points. The baseline
replays aiter's router GEMM, `biased_grouped_topk` and `fused_moe`. Triton side: [moe](../moe.md); op card:
[../../moe/INDEX.md](../../moe/INDEX.md). Files with their Points in parentheses; inter 512 has no Point at 32–128.

| Tokens | inter 256 | inter 512 | Launches | Decomposition |
|---|---|---|---|---|
| 1–2 | `5751` (1) `d3bb` (2) | `7e9c` (1, 2) | 3–4 | router (split-K tiles, or one full-K program per expert in `d3bb`) beside the hidden-state MXFP4 quantization (192 + 24 programs in `5751`); the gate-up of every route split along K (4 splits in `5751`, `d3bb`, 12 in `7e9c` at 1 token) and finished by a small launch, or whole (`7e9c` at 2); a last launch of down and combine over (token, column tile): 16 columns in `5751`, `d3bb`, 32 and 64 in `7e9c` at 1 and 2 |
| 4–16 | `c3e3` (4) `e9fd` (8, 16) | `4798` (4) `a2bb` (8) `3c6e` (16) | 3 (`3c6e` 5) | router + quantization; one launch runs the gate-up of every route, no separate activation launch; down + combine. `3c6e`: `_front`, `_select_masks`, two `_grouped_experts`, `_combine` |
| 32–256 | `f8cb` (32) `a1c4` (128) `1204` (256) | `2422` (256) | 5 (`2422` 6) | router, quantization and counter initialization in the first launches; a selection launch on relaxed atomics; gate-up over expert jobs (16 to 128 rows in `2422`; `a1c4`, `1204` derive their job from the counts); down; a last, atomic-free `_reduce_parts` launch writes `out` |
| 1024–4192 | `c524` (1024, 2048, 3072) `699f` (4096, 4192) | `3432` (1024) `a841` (2048) `67d9` (3072) `3b66` (4096) `b058` (4192) | 6 | full-K router + quantization; route selection; a prepare launch writes job lists (in `3432`, `a841`, `67d9`: full 128-row tiles first, then tails by falling height down to 16 rows, the mix changing with the routing); grouped gate-up; routed down; the shared expert's down, which writes `out` (S14) |

- **Techniques.** Router tiles and the MXFP4 quantization of the hidden states share one launch, split by program id (S9;
  [moe](../moe.md) §3). At 1–16 tokens every file but `3c6e` selects experts in the kernel with a rank loop; in `d3bb`
  each gate-up program recomputes its route's expert with rank + 1 argmax rounds instead of waiting for a top-k launch
  (S10; [moe](../moe.md) §4, [../../moe/decode.md](../../moe/decode.md)). A gate-up N tile holds matching gate and up
  rows; in full-K gate-up designs such as `1204`, SwiGLU and the MXFP4 requantization run in its epilogue (S11; [moe](../moe.md) §7, §8). The decode down
  multiplies a token's routes (MFMA rows) by their experts' column bands and keeps the diagonal, weighted and summed in
  slot order in registers (S12; [moe](../moe.md) §9). At 32–256 tokens, `1204`'s gate-up programs scan the per-expert
  counts with `gl.associative_scan` and make the job descriptor wave-uniform with `v_readfirstlane_b32` (S13;
  [gluon/wave_ops](../gluon/wave_ops.md) §6). MXFP4 operands go to `mfma_scaled` with e8m0 scales in
  `get_mfma_scale_layout` layouts; activations are packed with `v_cvt_scalef32_pk_fp4_f32` inline asm
  ([numerics](../numerics.md) §3).
- **Copy from.** `7e9c`'s `fused_moe` is a whole 1–2-token call in 3–4 launches (S12). The selection recomputed per
  program (S10):

  ```python
  def _expert_at_rank(Logits, Bias, row, rank, SPLITS: gl.constexpr):
      layout: gl.constexpr = gl.BlockedLayout([4], [64], [gl.num_warps()], [0])
      e, _, score = _routing_scores(Logits, Bias, row, SPLITS)
      available = gl.full((256,), True, gl.int1, layout)
      idx = _next_expert(score, available, e)
      for j in range(rank):
          available = available & (e != idx)
          score = gl.where(e == idx, -float("inf"), score)
          idx = _next_expert(score, available, e)
      return idx
  ```
- **Resources** *(field, E2)*: 116 kernels; `num_warps` 1 (57), 2 (8), 4 (51); `waves_per_eu` unset in all; VGPR
  22–280, AGPRs in 2 (16, 24); LDS 0–74.8 KiB; no spills; scaled 16 × 16 × 128 MFMA present in 54 of 116 DeepSeek compiled-cache records in the recovered four final shards (all 154 schemas, 897 AKLR records). The older 60-opcode-truncated table showed 53; it was incomplete. These are cache records, not deduplicated unique binaries (local:/tmp/kbmig/checks_study/census_recovered_final_counts_1107.json); bf16 16 × 16 × 32 in 23,
  scaled 32 × 32 × 64 in 8.
- **CI speedups** *(field, E6)*: 1.030 (inter 256, 128 tokens) to 2.186 (inter 256, 2 tokens) over the 24 Points.

## 8. MiniMax-M2 MXFP4 MoE

MiniMax-M2's MoE layer on one rank: an fp32 router (sigmoid plus correction bias, top-8 of 256, renormalized), MXFP4 experts, no shared expert. `n_experts256_inter_dim512` holds all 256 experts (12 files); `n_experts64_inter_dim1536` holds the 64 from `expert_offset`, routes to other experts adding nothing (11 files); 30 Points. CPU baseline-closure inspection found router `tgemm.mm` with its first operand cast to fp32, `moe_fused_gate` and `fused_moe` at `per_1x32`; vendor execution was not measured (E20).

| Tokens | E256 | E64 | Launches | Decomposition |
|---|---|---|---|---|
| 1–16 | `6557` (1) `fbd9` (2) `96e5` (4, 8, 16) | `b065` (1, 2) `f82c` (4) `3df5` (8, 16) | unverified | actual router at these 10 Points: no MFMA, fp32 VALU; 492 sampled projection comparisons pass (E21). Full-K/split-K reduction, expert decomposition and down + weighted combine remain unverified |
| 32–256 | `0101` (32) `d861` (64) `01ce` (128) `4da3` (256) | `dfa8` (32, 64) `1d27` (128, 256) | 5 | split-K router tiles, counter and job initialization and the hidden quantization in the first two launches; per-token top-8 with relaxed atomic tickets, a job opened every 16th ticket (every 64th in `01ce`, `4da3`); gate-up over jobs (`1d27`: each program finds its job by scanning the counts); down; combine |
| 1024–16384 | `bb66` (1024) `147b` (2048, 4096) `e6fc` (6144) `51ae` (8192) `8411` (16384) | `abc7` (1024) `cb88` (2048) `b94f` (4096) `62df` (6144) `f839` (8192) `8b22` (16384) | unverified | router at these 12 Points: fp32 MFMA in all listed E256 Points and `abc7`; bf16 MFMA in the other five E64 Points, with three exposed bf16 terms checked on finite packed samples. All 912 sampled projection comparisons pass (E21). E64 describes the expert shard: router E = 256. Sort, gate-up/quantization, down/reuse and combine semantics remain unverified |

- **Techniques.** Full-call launch counts and expert selection at the 10 small-token Points remain unverified by E21
  ([moe](../moe.md) §4); the router-only observations are listed above.
  Job queues from relaxed atomics; a publish tensor built on one wave masks its copies so each route takes one ticket
  (S17; [moe](../moe.md) §6). At E64 a counting sort from wave ballots and popcounts (S16;
  [gluon/wave_ops](../gluon/wave_ops.md) §6), atomic-free from 2048 tokens *(field, E2)*. `8411` replays a captured graph with new inputs
  correctly and keeps identical grids over 6 eager launches at two input sets; routing difference, worst-case sizing
  and device-count exit were not independently tested (S15, E24; [authoring](../authoring.md) §8). Full-logit comparisons have executed at cb88@2048, b94f@4096 and bb66@1024 over four distributions and eight seeds: L2 error versus the same-input FP64 oracle was <= executed GPU FP32 error in 96/96 cases, maxabs in 93/96. The additional 13-branch study used seed0 random/cancellation inputs at each selected Point, plus two nonzero-residual controls (28 inputs total). Relative to same-input FP64, L2 error was no worse than executed GPU IEEE FP32 in 27/28 inputs, maxabs in 25/28; strict elementwise noninferiority held in only 5/28. The M32 from32 and M1/2/4/16 VALU branches were included. This is finite FP32-scale evidence, not every tile/split specialization or guaranteed top-8 equivalence.
- **Copy from.** The relaxed ticket of `8411`'s select launch and the scatter that turns it into a slot (S15; shown in
  [moe](../moe.md) §6; checked with one count block, not `8411`'s several); the gate-up fold of `6557` (S42), which packs
  `gate_fold` K chunks into the gate-up's 16 MFMA rows, set per token count.
- **Resources** *(field, E2)*: 183 kernels; `num_warps` 1 (35), 2 (7), 4 (139), 8 (2); `waves_per_eu` 3 in 4, 4 in 1;
  VGPR 9–244, no AGPRs; LDS 0–35 KiB; no spills; scaled 16 × 16 × 128 MFMA in 69, fp32 16 × 16 × 4 in 12, bf16
  16 × 16 × 32 in 5, fp32 32 × 32 × 2 in 3, scaled 32 × 32 × 64 in 1.
- **CI speedups** *(field, E7)*: 1.178 (E256, 256 tokens) to 3.633 (E64, 1 token) over the 30 Points.

## 9. MiniMax-M2 FP8 MoE

The same layer with e4m3 experts and fp32 scales per 128 × 128 block, both GEMM inputs quantized per 1 × 128 in the kernels: `n_experts256_inter_dim384` (5 files) and `n_experts64_inter_dim1536` (4 files), 28 Points. The baseline source closure contains router `tgemm.mm` with a first-operand fp32 cast, `moe_fused_gate` and `fused_moe` at `per_128x128`; no vendor baseline was executed by that CPU check (E20).

| Tokens | E256 | E64 | Launches | Decomposition |
|---|---|---|---|---|
| 1–16 | `e75e` (1, 2, 4, 8, 16) | `1c17` (1, 2, 4, 8, 16) | unverified | actual router at these 10 Points: no MFMA, fp32 VALU; 180 sampled projection comparisons pass (E21). Quantization, per-route expert design, warp ownership and down + combine remain unverified |
| 32–256 | `5cac` (32, 64) `b43d` (128) `8a5a` (256) | `0982` (32, 64) `94af` (128, 256) | unverified | actual router uses fp32 MFMA at `5cac` 32/64, `b43d` 128 and `0982` 32/64; bf16 MFMA at `8a5a` 256 and `94af` 128/256. All 3456 sampled projection comparisons pass; individual inline bf16 terms remain UNKNOWN (E21). Job queues, quantization, persistent gate-up and down/combine remain unverified |
| 1024–16384 | `f770` (1024, 2048, 4096, 8192, 16384) | `82e9` (the same five) | 6–8 | a prepare launch first (`82e9`'s `_prepare_inputs`; `f770`'s `_prepare_router_and_quantize` from 8192 tokens) or the router first (`f770` up to 4096); router; selection and sort (`f770`: launches with a `CHUNK` constexpr; `82e9`: an atomic-free SWAR counting sort, `_rank_counts` and `_prefix_scatter`); gate-up with fused SwiGLU + fp8 quantization; down; combine |

- **Techniques.** One warp per route under `AMDMFMALayout(..., warps_per_cta=[8, 1, 1])`, the route as the batch
  dimension (S18; [moe](../moe.md) §9); a counting sort with four 8-bit counters per 32-bit word and one
  `gl.associative_scan` (S19; [moe](../moe.md) §6); inline asm in 6 files: `v_rcp_f32` in all six, `v_exp_f32` in five,
  `v_readlane_b32` in four, `s_ff1_i32_b64` in three, `v_max_f64` / `v_min_f64` only in `94af`
  ([gluon/wave_ops](../gluon/wave_ops.md) §4, §6). At cb88@2048 and b94f@4096, the actual three-BF16 RTZ/RTZ/RTZ split with flagged FP64 residual correction has finite full-logit FP32 comparisons. The other tested bb66 branch uses FP32 MFMA. Additional runtime controls covered 62df/f839 RTZ/RTZ/RTZ with flagged FP64 correction, f770 RTZ/RNE/RNE in inline/prepared paths, and 82e9 RNE/RNE/RNE without that correction. In the two finite nonzero-residual controls, 16 flags and 256 residual entries were nonzero; deleting only the correction call gave zero logits, while the original output changed by 2^-23 exactly. Ordinary random/cancellation inputs had zero residual and did not exercise that body. Three BF16 terms alone are not an accuracy theorem; arbitrary-input and high-precision top-8 equivalence remain unestablished.
- **Copy from.** The SWAR rank of every route among same-expert routes in a 512-route block (S19):

  ```python
  ids = gl.load(Ids + route, route < R, -1) - OFFSET
  matches = (route[None, :] < R) & (ids[None, :] >= 0) & (ids[None, :] < E)
  matches &= group[:, None] == ids[None, :] // 4
  shift = (ids & 3) * 8
  increment = gl.full((512,), 1, gl.uint32, gl.SliceLayout(0, layout)) << shift
  packed = gl.where(matches, increment[None, :], 0)
  prefix = gl.associative_scan(packed, 1, _add) - packed
  rank = prefix >> shift[None, :] & 255
  ```

  The 8-bit counters hold because a block's 512 routes are 512 // TOPK = 64 tokens, each choosing an expert once:
  the file asserts `gl.static_assert(512 % TOPK == 0 and 512 // TOPK < 256)` (S19; TOPK 3 fails it). Over 255 routes of
  one expert in a block carry into the next expert's counter: with 300, the neighbour's ranks came out one too high.
- **Resources** *(field, E2)*: 188 kernels; `num_warps` 1 (73), 2 (13), 4 (96), 8 (6); `waves_per_eu` 1 in 4; VGPR
  14–327, AGPRs in 2 (35, 71); LDS 0–131.9 KiB; an 8-SGPR spill in `_pack_routes`; scaled 16 × 16 × 128 MFMA in 47,
  bf16 16 × 16 × 32 in 13, scaled 32 × 32 × 64 in 10, fp8 16 × 16 × 32 in 9, fp32 16 × 16 × 4 in 7.
- **CI speedups** *(field, E8)*: 1.102 (E64, 64 tokens) to 4.115 (E256, 1 token) over the 28 Points. Across §7–§9 the
  32–256-token Points are the weakest band whatever the design, and inside one 1–16-token file the value falls as
  tokens grow: band ranges in [moe](../moe.md) §2, band medians in [tuning](../tuning.md) §14.

## 10. Qwen2-MoE: router, fp8 experts, fp8 router + experts

Qwen3.5-397B-A17B's MoE block at TP4 (by the contract's docstring): model dim 4096, 512 routed experts, top-10; the
experts schemas add inter dim 256, one shared expert weighted by `sigmoid(hidden · shared_gate_weight)` (513 experts in
all) and e4m3 experts with fp32 128 × 128 block scales. The route tries the router + experts op first, then the router
followed by the experts op (`aklr/plugins/sglang/models/qwen2_moe/api.py`).

| Contract, directory `qwen2_moe_…` | Tokens: files | Launches | Decomposition |
|---|---|---|---|
| `router_model_dim4096_n_routed512_topk10/` | 8 `4801`, 16 `5f20`, 64 `a878`, 128 `cd47`, 256 `c388`, 2232 `4c2b`, 4096 `ca82` | 2 | a split-K projection (S20), exactly 256 programs at 8–256 tokens, then per-row selection over the stored bf16 logits: top-10 with lower-id ties, softmax weights within 2.3e-8 of those 10 logits, outputs bitwise identical over 20 repeats/check at 8–256 tokens; once-rounded split sums untested (E25); a source-described prefill GEMM with multi-row selection at 2232 and 4096, not run by this check |
| `fp8_experts_shuffled_16x16_…_n_experts513_…/` | 8 `5c35`, 64 `8698`, 128 `d425` | 3; 5; 5 | routing given; 8 tokens: per-route split-K gate-up, activation quantization, down with the ordered 11-route sum; 64 and 128: a route sort in LDS (by one CTA in `8698`), then gate-up, quantization, down, ordered reduce (`8698`: persistent stages over device-counted tasks) |
| `fp8_router_experts_shuffled_16x16_…/` | 1 `7d7e`, 2 `04e9`, 4 `ddb8`, 32 `99f4` | 4; 4; 4; 7 | router logits in fp64 (VALU), rounded once to bf16; every gate-up CTA replays the top-k of its row; activation quantization; one CTA per output tile multiplies all of a token's routes in one MFMA and keeps the diagonal; at 32 tokens fp64 MFMA router partials, a counting sort, sparse gate-up, quantization, grouped down, ordered sum |

- **Techniques.** One `@gluon.constexpr_function` gives the host its grid and the kernel its tile geometry (S20;
  [tuning](../tuning.md) §14); CPU AST inspection records `llvm_fn_attrs` arguments at the actual Qwen2 launch sites,
  not GPU scheduler effects (E20; [tuning](../tuning.md) §9); the top-k replayed per CTA (S21; [moe](../moe.md) §4); fp64 products for a router that
  must round like an fp64 reference, on the VALU at 1–4 tokens and on fp64 MFMAs at 32 (S22; [numerics](../numerics.md)
  §5); a one-CTA counting sort with `atomic_scatter_add` on an LDS descriptor (S23; [moe](../moe.md) §6).
- **Copy from.** `c388`'s `_shape_config`, which returns `_ShapeConfig(split, expert_tile, k_tile, lookahead,
  routing_group)` per row count to both the launch and the kernel (S20; shown in [tuning](../tuning.md) §3).
- **Resources** *(field, E2)*: router 14 kernels, `num_warps` 1, 4 or 16, VGPR 41–192, LDS 0–130 KiB, bf16 MFMA;
  experts 13 kernels, 1–8 warps, `waves_per_eu` 2 or 4 in 3, VGPR 16–110, LDS 0–12 KiB, bf16 and fp16 MFMA; router +
  experts 19 kernels, 1–4 warps, VGPR 19–128, LDS 0–16 KiB, scaled 16 × 16 × 128, fp64 16 × 16 × 4 and fp16 MFMA. No
  AGPRs, no spills.
- **CI speedups** *(field, E9)*: router 1.074 (8 tokens) to 1.967 (256); experts 1.076 (128) to 1.208 (8); router +
  experts 1.063 (32) to 1.172 (1).

## 11. DSA sparse MLA decode

Each token's 64 or 8 heads attend 2048 rows of an fp8 latent cache (576 values: 512 latent and 64 RoPE) named by
top-k indices, q in e4m3, `kv_b_proj` absorbed, bf16 out; 8 files, one Point each. The baseline is SGLang's
`triton_sparse_mla_decode_splitk`. Triton side: [attention](../attention.md) §3, §4, §5; op card:
[../../attention/INDEX.md](../../attention/INDEX.md).

| Tokens (heads) | Files | Launches | Decomposition |
|---|---|---|---|
| 8 (64) | `0bb2` | 3 | `q.to(torch.bfloat16)`; a producer of 256 programs × 4 warps; a merge of 512 programs |
| 12–32 (64) | `1148` (12) `f60f` (16) `2ee5` (24) `2f70` (32) | 2 | 192–256 programs of 8 warps per (token, 32- or 64-head tile, key split), 256 gathered rows per LDS tile (two tiles per program at 32 tokens, which has 4 splits, S45), QK on `mfma_scaled` e4m3 with unit scales (S24); a merge launch |
| 32, 128 (8) | `43ff` (32) `ae32` (128) | 3 | `q.to`; a producer; a merge |
| 256 (8) | `fbe6` | 2 | `q.to`; one workgroup per token over all 2048 keys, no merge |

- **Techniques.** Head tile and split count follow the token count; at 64 heads the grid is 192–256 programs, one
  LDS-heavy workgroup per CU (S24; [../../attention/decode_split_kv.md](../../attention/decode_split_kv.md) §4, §5).
  Partials merged by a separate launch in a fixed order, scaled by `exp(m_s - max)` with a guarded `1 / den` (S25;
  bitwise contract, [attention](../attention.md) §2); enough tokens to fill the machine drop the merge (S26). Four files
  (`0bb2`, `43ff`, `ae32`, `fbe6`) convert `q` in `run`, one more kernel per call (S27).
- **Copy from.** The 64-head host of `2ee5` (S24; shown in [attention](../attention.md) §4): head tile and split count
  from the row count, the partial records sized from them, the buffer path gated on the cache's size.
- **Resources** *(field, E2)*: 15 kernels; `num_warps` 1 (3), 2 (2), 4 (6), 8 (4); `waves_per_eu` 1 in 2, 2 in 3;
  VGPR 23–228, AGPRs in 1 (22); LDS 0–160 KiB; no spills; bf16 32 × 32 × 16 MFMA in 4, scaled 32 × 32 × 64 in 4, bf16
  16 × 16 × 32 in 3, scaled 16 × 16 × 128 in 1, fp16 32 × 32 × 16 in 1.
- **CI speedups** *(field, E10)*: 1.030 (64 heads, 8 tokens) to 2.188 (64 heads, 24 tokens).

## 12. Qwen3.5 gated attention decode

One token per row, 8 query heads on 1 KV head of dim 256: Q and K RMS-normalized, mRoPE on 64 channels, K and V written
to the e4m3 paged cache, attention over the row's slots, `sigmoid(gate)`, then e4m3 per 128 channels with fp32 scales; 9
files (`kernel.py`, `prologue.py`, `attention.py`), one Point each, 1–256 rows. The baseline chains SGLang's fused QK
norm, Triton mRoPE and FP8 KV write, aiter's `paged_attention_ragged`, `fused_sigmoid_mul` and aiter's per-1 × 128
quantization. Triton side: [attention](../attention.md) §6, §7, §8, §9.

| Rows | Files | Decomposition (3 launches: prologue, attention, finish) |
|---|---|---|
| 1–2 | `f5ba` (1) `42a2` (2) | one CTA per (row, head, split), 32 splits, FMA dot products, no MFMA |
| 4 | `3d0f` | one program per (row, split), 4 × 32, all 8 query heads in it on MFMA |
| 8 | `10df` | MFMA partials, async copies into 34,816 bytes of LDS |
| 16–64 | `6dd3` (16) `d9cb` (32) `eeb1` (64) | a (row, split) grid, rows × splits = 512 (32, 16 and 8 splits), tiles dealt to the splits cyclically; K and V gathered into registers, V through a padded LDS layout; a 16-row head tile at 16 splits, else 8 (S40) |
| 128–256 | `14e9` (128) `dfb1` (256) | a 1-D grid of 4 and 2 splits (rows × splits = 512), one contiguous chunk per split, async copies into LDS |

- **Techniques.** The prologue is a per-token task grid of one-wave CTAs (K norm, RoPE and cache write; V cache write; Q
  heads and the gate copy; `10df`, `14e9`, `d9cb`, `f5ba` and `42a2` give the cache-write tasks the low program ids). The
  attention launch is flash-decoding ([../../attention/decode_split_kv.md](../../attention/decode_split_kv.md) §5): per
  split an unnormalized numerator with its (max, sum); the finish merges in a fixed order, rounds to bf16, applies the
  gate and quantizes, bit for bit as emulated in the 32-row file ([numerics](../numerics.md) §2). At 256 rows the 1-D
  grid is a bijection whose blocks of 128 CTAs hold both splits of 64 rows (S29). CPU enumeration of rows 1–1024
  and SPLITS {2,3,4,8,16,32} found the map bijective exactly at multiples of 64; each valid block of 64*SPLITS
  consecutive CTAs covers every split of the same 64 rows (6144 grids, S48). This is finite evidence, not a proof
  for arbitrary SPLITS. Every split writes a record, inactive
  ones neutral (S30), and a replayed graph matched the eager call (32-row file). `run` is two library calls (S28).
- **Copy from.** The split-to-row map of the 256-row file (S29):

  ```python
  # Flatten the launch into groups containing all splits of 64 rows.
  cta = gl.program_id(0)
  row = cta % 64 + cta // (64 * SPLITS) * 64
  split = cta // 64 % SPLITS
  ```

  In the same finite enumeration, SPLITS=1 is bijective at every tested row count (1024 more grids, 7168 total),
  so the multiple-of-64 condition must be qualified by the tested SPLITS>=2 values. At 96 rows and 2 splits,
  it misses 32 pairs, emits 32 out-of-range CTAs and reaches row 127; at 256 rows it is bijective. Both source
  expressions matched, with execution agreement on five sample grids (S48). This does not broaden the entry
  contract: the 128- and 256-row files assert `rows in (128, 256)` (S29), and at 16–64 rows the files launch a
  (row, split) grid instead (S40).
- **Resources** *(field, E2)*: 27 kernels; `num_warps` 1 (9), 2 (1), 4 (17); `waves_per_eu` 2 in 2, 3 in 1; VGPR
  25–192, AGPRs in 1 (63); LDS 0–70 KiB; no spills; bf16 16 × 16 × 32 MFMA in 7, fp16 in 1.
- **CI speedups** *(field, E11)*: 1.066 (256 rows) to 2.182 (1 row).

## 13. GatedDeltaNet decode

Per row: a width-4 causal convolution over the projected Q, K, V with the conv history shifted in place, the gates, the gated delta-rule update of a 128 × 128 fp32 state per (slot, value head), a gated RMSNorm and e4m3 per 128 with fp32 scales. Qwen3.5 (16 value heads): 8 files, 1–256 rows, plus `gdn_decode_…/cd00` (64 rows, bf16 output). Qwen3-Next (8 value heads, grouped projections): 9 files over 9 Points, `f440` listing all of them. These file/Point counts are the historical source map, not re-counted by E20. CPU baseline closures contain conv update, recurrence, norm and quant calls; GPU baseline execution was not tested (E20). No kernel uses MFMA *(field, E2)*. Triton side: [gdn](../gdn.md) §2–6.

| Model, rows | Files | Launches | Decomposition |
|---|---|---|---|
| Qwen3.5, 1–4 | `2c2a` (1) `2991` (2) `409c` (4) | 2 | each head's 128 value rows split over CTAs, 256 at 1, 2 and 4 rows; then a launch of 40 programs per row; the conv history shifted in place |
| Qwen3.5, 8–256 | `4f25` (8) `a4f8` (16) `20f6` (32) `cd00` (64, bf16 op) `fc26` (128) `c2c3` (256) | 2 | convolution with exclusive channel ownership, history shifted in place; one CTA per (row, head) owns the state, 4 waves at 8–32 rows, 16 from 64, the norm (and quantization) in its epilogue |
| Qwen3-Next, 1–8 | `b31b` (1) `bba1` (2) `3719` (4, 8) | 2 | producers of one or two waves per (row, value head or head pair, block of value rows), then a finisher |
| Qwen3-Next, 16–256 | `35b9` (16) `38a7` (32) `0c0b` (64) `0678` (128, 256) `c784` (128, 256) | 1 | one workgroup per (row, key head) owns the conv history and both value heads' states |

- **Techniques.** At `2c2a` / `2991` / `409c` (1 / 2 / 4 rows), moving the history shift before the reader launch failed 20/20 trials each over two case executions; shifting inside failed 14/20, 5/20 and 17/20, respectively, while the committed second-launch shift passed 20/20 each (E22; S31; [gdn](../gdn.md) §3). The race need not fail every trial. One-owner `4f25` (8 rows) and `0678` (128) passed 40/40 each; replicated `c784` (128) passed 100/100 with its barrier and failed 100/100 without it. A delayed-reader 4-wave micro-kernel was exact 40/40 with one owner or a barrier, but mismatched 40/40 with neither; an older unskewed probe had seen no race (E22; S32, S41). The template keeps one configuration row per row count and raises on an unlisted count (S33). Inline asm supplies `v_med3_f32` (all 8 Qwen3.5 files, the ±448 clamp; `gl.clamp` emits the same instruction) and, in Qwen3-Next, `v_rcp_f32`, `v_exp_f32`, `v_log_f32`, `v_div_fixup_f32`, `v_pk_fma_f32`, `v_cvt_scalef32_pk_fp8_f32`, `s_setreg_imm32_b32` and DPP wave reductions (`v_max_f32` / `v_add_f32` with `row_shr`, then `v_readlane_b32`) ([gluon/wave_ops](../gluon/wave_ops.md) §3, §4, §6, §7).
- **Copy from.** The exclusive-owner shift of `0678`: one 1-D tile over the key group's 512 conv channels with `gl.static_assert(NW * 64 <= conv_width)`, so no lane or wave holds a copy and each thread shifts the history it read, with no barrier (S41):

  ```python
  conv_layout: gl.constexpr = gl.BlockedLayout([1], [64], [NW], [0])
  # This packed tile has no replicated lanes/waves. Every history read and
  # write has one owner, even though Q/K are subsequently shared by two heads.
  c = gl.arange(0, conv_width, layout=conv_layout)
  ```

  `c784` keeps replicated lanes and orders the reads before the shift with `gl.barrier()` (S32). In one CI run on
  main in `d908d01` it was the slowest of the three 128- and 256-row designs, 1.466 and 1.151 against `0678`'s 1.782 and
  1.425 *(field, E13)*; the designs differ in more than the shift ([gdn](../gdn.md) §2).
- **Resources** *(field, E2)*: Qwen3.5 18 kernels, `num_warps` 1, 2, 4 or 16, `waves_per_eu` 1, 6 or 8 in 5, VGPR
  16–146, LDS 0–2 KiB; Qwen3-Next 27 kernels, 1–8 warps, VGPR 17–208, LDS 0–2.5 KiB. No AGPRs, no spills.
- **CI speedups**: Qwen3.5 1.282 (256 rows) to 2.507 (2 rows), the bf16 op 1.461 *(field, E12)*. Qwen3-Next stores no
  speedup on main at `43e112e`, so dispatch does not serve it at the default `AKLR_MIN_SPEEDUP`; a CI run of the same
  files on main in `d908d01` gave the best file per Point 1.457 (256 rows) to 3.350 (32 rows), and the per-size files
  above `f440` at 7 of 9 Points (equal at 4 rows, `f440` ahead at 256) *(field, E13)*.

## 14. GatedDeltaNet chunked prefill

Packed sequences of the same layer: Qwen3.5, 15 files over 21 Points (512–16384 tokens in 1–8 sequences); Qwen3-Next, 11 files over 13 Points (1024–16384 in 1–16). Files: `kernel.py`, `initial_state.py` and, in 9 Qwen3.5 files, `packing.py`; these are historical file/Point counts, not re-counted by E20. CPU baseline closures contain `causal_conv1d_fn`, `fused_gdn_gating`, `chunk_gated_delta_rule`, norm and quant calls (`per_token_group_quant_fp8` in Qwen3.5); GPU baseline execution was not checked (E20). Triton side: [gdn](../gdn.md) §7–12.

Every call zeroes the state slots of sequences without an initial state (S34; tested on `24ed`), and 9 Qwen3.5 files repack the projections into grouped order. Historical launch trace at 34 Points: prepare → chunk factors → summaries for long sequences → one sequential launch → parallel output epilogue; 5–8 launches. At 21 Qwen3.5 Points the final conv history matched the last three entries of [old history, or zeros when fresh; the sequence's inputs]. Sequences shorter than three tokens retain old/zero entries. Early history commit in `0a07` (2049 tokens / 2 sequences) and `4489` (3072 / 3 and 3072 / 2) changed 12,693–29,318 fp8 values and failed accuracy on all three inputs in each of two case executions; committed files passed. Final conv/delta states were unchanged: wrong outputs expose the early read hazard (E22; S37). Gated RMSNorm/fp8 epilogue semantics beyond the historical launch trace were not tested by this audit.

Files with their Points as tokens/sequences, then the scenario's `_max_seq_len<L>` after the semicolon (∅: a scenario without the suffix).

| Point shape | Qwen3.5 | Qwen3-Next | Sequential pass |
|---|---|---|---|
| up to about 2K tokens per sequence | `61ed` (512/2; 256) `1d55` (512/1, 1024/2; 512) `1fd6` (1024/1; 1024) `19b2` (1025/1; 1025) `3828` (2048/1; ∅) | `11f6` (1024/1, 2048/2; ∅) | 64-token chunks, chunk-start checkpoints, output in parallel per (head, chunk) |
| several sequences | `0a07` (2049/2, 4096/2) `2ece` (4097/3, 8192/4) `c53b` (8193/5, 15360/8) `24ed` (15361/8, 16384/8), all ∅ | `4489`/`d8f9` (3072/3, 4096/4, 8192/8, 12288/12, 16384/16; ∅), `70f4`/`010e` (13312/13, 15360/15; ∅) | per (sequence, head, value-row strip); the Qwen3.5 pass has 160–256 programs (`24ed`: 16 heads × 2 strips of 64 value rows × 8 sequences = 256; `c53b` at 5 sequences: 160; S35) |
| long sequences, 4096–16384 tokens in 1–4 | maps from an identity seed: `5ead` (8192/1; 8192) `7a69` (10000/1; 10000); reverse summaries: `12b5` (5000/1, 15000/3; 5000) `b8c5` (16384/2; 8192) `3b99` (16384/2; 10000) `4709` (16384/1; 16384) | `162f`/`a99f` (4096/1; 4096), `c3a7`/`f137` (5120/1, 10240/2; 5120), `98af`/`ed2a` (16384/4; 5120) | a parallel summary launch per run of chunks (320–864 CTAs), one sequential launch (64–384), a parallel replay of the chunks (320–1024) |

- **Techniques.** bf16 MFMA for the Gram products of the bf16-rounded Q and K, fp32 MFMA for the state algebra (S38's
  comment; [gdn](../gdn.md) §9, [numerics](../numerics.md) §5). `24ed` permutes K inside every 16-block of both operands
  so that an fp32 accumulator becomes the next MFMA's A operand by a trivial `convert_layout` (S39; [gdn](../gdn.md)
  §11, [gluon/layouts](../gluon/layouts.md) §7). Every Qwen3-Next file but `11f6` has a twin, the same source with `M`
  and `BATCH` (in `70f4` / `010e` also `TOTAL`) as constexprs in one and runtime ints in the other, permitting binary reuse only when the complete JIT specialization key is unchanged (S36); changing the annotations alone does not establish reuse across all Points. The original probe for the historical claim of 3 runtime binaries for 11 Points versus 11 constexpr binaries has not been located; in the CI run
  main recorded in d908d01, the twins came within 2.4 % of each other, either way *(field, E15)*.
- **Copy from.** The per-sequence pass of `24ed` (S35) and `_initialize_fresh` (S34).
- **Resources** *(field, E2)*: Qwen3.5 123 kernels, `num_warps` 1, 2, 4 or 8, `waves_per_eu` 2, 4 or 5 in 15, VGPR
  3–256, no AGPRs, LDS 0–32 KiB, spills in 2 (64 VGPRs in `_build_chunk_tables_paired`, 1 SGPR); Qwen3-Next 117
  kernels, 1–4 warps, VGPR 2–316, AGPRs in 5 (20–60), LDS 0–32 KiB, no spills. fp32 16 × 16 × 4 MFMA in 57 kernels
  of each family, bf16 16 × 16 × 32 in 24 of each; fp32 32 × 32 × 2 in 4 (Qwen3.5) and 2 (Qwen3-Next), bf16
  32 × 32 × 16 in 2 (Qwen3.5). All 26 files hold both bf16 16 × 16 × 32 and fp32 16 × 16 × 4 MFMAs.
- **CI speedups**: Qwen3.5 0.997 (8193 tokens in 5 sequences) to 2.638 (512 in 1); 0.997 and 1.029 (4097 in 3) are
  below the default `AKLR_MIN_SPEEDUP` *(field, E14)*. Qwen3-Next has none at `43e112e`; the same files' CI run on main in `d908d01` gave the best file per Point 1.233 (13312 tokens in 13 sequences) to 1.741 (5120 in 1) *(field, E15)*.


**Unverified scope retained.** E20/E21 do not establish nearest-file transfer, fp32 GEMM group-scale def-use, shared ninth-route/combine semantics, `expert_offset`, effective job rows, sort/job counts, register quantization or complete MoE/down/combine behavior. Small-M optimality, AITER performance comparisons, full-output fp32/fp64 accuracy, arbitrary inputs and top-8 equivalence are unmeasured. The N 2624 / K 6144, M = 64 split6-versus-split4 ranking is unresolved. §14's ≤2048 row is a source map, not a skip-summaries recommendation: the final five-Point study belongs to [gdn](../gdn.md), not evidence for all ≤2048 shapes (cases `d_source_contract_and_point_census`, `d_router_instruction_point_census`, `s_gdn_decode_split_and_nearest_file`, `d_gdn_short_scan_vs_summaries`).

## Evidence

- E24 [2026-11, gfx950, Triton 3.8.0, N=2 valid .20/.21 runs, AKLR validate/eager profiler] graph all 22 checks executed, 21 PASS, .01 FAIL, 0 SKIP. Three logs are not three successful runs/check: 100 .11/.20/.21 had case exceptions; .20/.21 have two valid runs (150/870). Four MiniMax files capture/replay correctly and keep identical eager grids at numeric/0 and numeric/1 (6/6/6/8 launches); 8411 alone has 6. Routing difference, worst-case sizing and device early exit unproved; all old s_/d_ verdicts/counts retained (local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/150_rt_graph_capture_compile_r1.log; local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/870_rt_graph_capture_compile_cleanup.log; local:/tmp/kbmig/checks_study/verdicts/rt_graph_capture_compile.audit2.json).
- E25 [2026-11, gfx950, Triton 3.8.0, N=3 case runs, rt_moe_topk_selection .02/.05 equality/grid probes] 3 valid case logs (100/898/870), each 17 checks executed, 15 PASS, .14/.16 FAIL, 0 SKIP; not all claims pass. Five 8–256-token routers grid=256, outputs bitwise identical over 20 repeats/check; stored bf16 top-10/lower-id ties and softmax weights within 2.3e-8. 2232/4096 and once-rounded hidden.W/split sums untested (local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/100_rt_moe_topk_selection.log; local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/898_rt_moe_topk_selection_rep2.log; local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/870_rt_moe_topk_selection_cleanup.log; local:/tmp/kbmig/checks_study/verdicts/rt_moe_topk_selection.audit2.json).

- E23 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1 complete GPU run; 4 balanced AB/BA pairs per shared Point and 199c/0675, 12 per bd01/083b/ff2d; aklr validate --stages audit,bench with separate arm caches] all 12 Points completed, 58 kernels compiled, zero A/B accuracy findings; B differs only in tile order. Seven shared Points: 5/7 meet mean B/A >=1.01 plus full-arm separation and paired CI above 1; 14c8 is below 1%, 514a crosses the 1.01 threshold. Actual 199c M256 passes its separate comparison. 0675 is equivalent within ±1.5%; bd01/083b unresolved against ±1.5%/±1.3%; ff2d has a resolved >1% increase. Per-Point ratios and paired-mean-ratio 95% t intervals are in §4; these are within-run pairs, not independent full-run repetitions. CPU pid%8 tile maps are estimates, not hardware L2 counters or CTA placements; full-library remap counts and unique L2 causes remain unverified. Case triton/checks/s_xcd_remap_shared_operand.py (.01, .01.m256, .02), local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/850_s_xcd_remap_shared_operand_r2.log:1–3,5–19; final audit local:/tmp/kbmig/checks_study/verdicts/s_xcd_remap_shared_operand.audit.json (final_checked=true, rerun_requested=false). Probe evidence stays separately scoped in E18.
- S48 local:/tmp/aklr_triton/harness/skills/aklr-kb/triton/checks/s_xcd_remap_shared_operand.py:850 CPU split-map enumeration (.03), local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/850_s_xcd_remap_shared_operand_r2.log:4 finite enumeration of rows 1–1024 and SPLITS {1,2,3,4,8,16,32}: 7168 grids; six SPLITS>=2 values bijective exactly at multiples of 64, all SPLITS=1 grids bijective; block property and both dfb1/14e9 source expressions agree, five sample-grid executions; 96 rows / 2 splits misses 32 pairs, has 32 out-of-range CTAs and max row 127. Not a proof for arbitrary SPLITS.
- E19 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1 GPU execution; separate CPU Triton 3.7.0 requirement check] unmet `>=3.9` under 3.8.0: validate rc=3, no traceback, no_matching_toolchain, zero workloads timed and zero copy builds; dispatch serves M 6144 from 7266. Met `>=3.8`: rc=0, one timed workload and one build. CPU 3.7.0 reports `>=3.8` unmet; the final audit also records 7266's missing-helper compile error under 3.7.0, without a standalone local log. Case triton/checks/s_dispatch_contract_and_requires.py (.03, .03.cpu), local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/800_s_dispatch_contract_and_requires.log:775; CPU requirement corroboration: case triton/checks/d_source_contract_and_point_census.py (.10.t37), local:/tmp/kbmig/checks_study/local_logs/d_source_contract_and_point_census.log; audit local:/tmp/kbmig/checks_study/verdicts/s_dispatch_contract_and_requires.audit.json. No end-to-end GPU run under 3.7; .03.cpu was SKIP in the GPU log.
- E20 [2026-11, CPU, installed Triton 3.8.0, N=1 source/AST run, no GPU measurements] host metadata replay: 24 GEMM Points each one launch; 7266 rejects strided A/B; 89 MoE Points in 58 schemas record gate-up W1 provenance/grids, not job semantics. Six baseline closures record calls and router first-operand fp32 casts; Qwen2 launch AST records llvm_fn_attrs, not scheduler effects. Case triton/checks/d_source_contract_and_point_census.py (.07, .10.source, .12.inventory, .13.gemm, .13.gemm_launch, .14.host, .14.strides, .14.gate_up), local:/tmp/kbmig/checks_study/local_logs/d_source_contract_and_point_census.log:114,204–206,239; 25 checks ran, eight explicit skips remain unverified.
- E21 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1 full census split into two disjoint log parts, not N=2; one numeric/1 input per Point; no timing] 58 real router projections, hidden bf16[M,3072], weights fp32[256,3072], M among 1/2/4/8/16/32/64/128/256/1024/2048/4096/6144/8192/16384, not every M in every schema. All 8520 finite sampled comparisons against math.fsum pass explicit per-sample tolerances; zero mismatch/nonfinite/poison. Subsets: MXFP4 small 10 Points / 492 comparisons, MXFP4 large 12 / 912, FP8 small 10 / 180, FP8 middle 8 / 3456, bf16 mechanism 18 / 1896; subsets overlap, not independent runs. Case triton/checks/d_router_instruction_point_census.py (point.*, source.*, split_cpu), local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/850_d_router_instruction_point_census_r1_part0of2.log and local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/850_d_router_instruction_point_census_r1_part1of2.log; NUMERIC/TRACE identify actual Point, grid, constexprs, dtype, samples and ISA. Exposed E64 MXFP4 three-term values have finite checks; FP8 inline terms remain UNKNOWN. No whole-op accuracy or ranking follows from CASE PASS. Per-Point metadata and ISA hashes: local:/tmp/kbmig/checks_study/verdicts/d_router_instruction_point_census.audit.json.
- E22 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2 full case executions; trials within each execution are not independent full runs] case triton/checks/s_barrier_war_races.py (.05, .06), local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/800_s_barrier_war_races.log:6–7 and local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/850_s_barrier_war_races_r1.log:6–7. Decode Points 2c2a@1,2991@2,409c@4: committed passes 20/20 each, shift-before fails 20/20 each, shift-inside fails 14/20,5/20,17/20. 4f25@8 and 0678@128 pass 40/40 each; c784@128 passes 100/100 with barrier, fails 100/100 without. Delayed-reader 4-wave micro probe: replicated/no-barrier mismatches 40/40; barrier or exclusive owner exact 40/40. Prefill 0a07@2049/2 and 4489@3072/3,3072/2: early commit fails all three inputs per execution, changes 12693–29318 fp8 values, committed passes; final states unchanged. Historical 34-Point launch order and 21-Point Qwen3.5 history evidence: case triton/checks/rt_launch_trace_gdn.py (.12), local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/100_rt_launch_trace_gdn.log (N=1, separate from these two executions). The .05 check FAIL demanded every in-reader trial fail, not a failure of committed kernels. Older unskewed probe: case triton/checks/rt_gdn_state_semantics.py (.09,.10), local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/100_rt_gdn_state_semantics.log.


- E1 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1 (router split-K: N=3), python -m aklr validate --stages audit,bench, one GPU, copy and original alternated, 4 rounds per arm] copies of committed files with one constant changed; ratio of the copy's speedup to the original's at one Point: gated attention 32 rows (d9cbde237fd34de9) KV splits 16 → 8 0.63, 16 → 32 0.95 (the head tile follows the split count: 16 → 8 rows in both); Qwen2 router 256 tokens (c3889614985b700a) split-K 4 → 2 0.75 (9.91 → 13.29 µs a call), 4 → 8 0.97 (9.97 → 10.25 µs, all 4 copy runs below all 4 originals), repeated at 0.748 / 0.967 and 0.749 / 0.961; Qwen2 router 8 tokens (4801e216dcff290f) hidden tile by gl.load converted through LDS instead of async copies, the weight's load layout changing with it: 8.447 → 8.45 µs, ratio 0.99; fp8 GEMM N 3072 M 16 (083b493a1861424c) K groups prefetched 5 → 3 0.87, 5 → 4 1.00, 5 → 7 1.00; DSA 24 tokens (2ee56a6dbabc48a6) KV splits 8 → 4 alone: build_failed, a CompilationError in _attention_split (NameError: cache_shared is not defined), in all 4; MiniMax MXFP4 E256 1 token (655723f593921b08) gate fold 4 → 1 alone: candidate_crash, exit −11, in all 4 (case triton/checks/ab_gattn32_kv_splits.py (.06, .07), local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/700_ab_gattn32_kv_splits.log; case triton/checks/ab_gemm_m16_prefetch_depth.py (.04, .08), local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/700_ab_gemm_m16_prefetch_depth.log; case triton/checks/ab_coupled_constants_break.py (.15, .16, .17), local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/400_ab_coupled_constants_break.log; case triton/checks/ab_router256_split_lookahead.py (.03), local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/700_ab_router256_split_lookahead.log, repeats case triton/checks/bench_bytes_in_flight.py (.13), local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/200_bench_bytes_in_flight.log, local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/800_bench_bytes_in_flight_rep2.log; case triton/checks/ab_router8_async_staging.py (.05), local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/700_ab_router8_async_staging.log)
- E2 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, python -m aklr validate --stages audit,bench once per schema on its own TRITON_CACHE_DIR over 4 GPUs; the cached .amdgcn and .json parsed] num_warps, waves_per_eu, VGPR (.vgpr_count, AGPRs included), AGPR, LDS, spills, MFMA opcodes and the count of every opcode (loads, stores and atomics included, buffer loads into LDS apart) of every AKLR kernel (a @gluon.jit / @triton.jit def of the schema's sources) compiled while validating each of the 154 schemas once: 897 of the 1709 compiled kernels; N 3072 fp8 GEMM LDS 9, 9, 12, 36, 48, 49 KiB at M 1, 2, 4, 8, 16, 32; 1db6, ae4f, 0675 64.0, 66.1, 72.0 KiB; E64 MiniMax MXFP4 cb88, b94f, 62df, f839, 8b22: 0 atomic instructions (global, buffer, flat, ds), popcounts (s_bcnt) 4, 2, 4, 8, 8; every GDN prefill file (26) holds v_mfma_f32_16x16x32_bf16 and v_mfma_f32_16x16x4_f32; 45 GDN decode kernels, 0 MFMA; RMSNorm loads with the nt bit, stores sc0 sc1 only, 16 B of LDS (case triton/checks/rt_census_compiled_kernels.py (.05, .34–.56), local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/860_rt_census_compiled_kernels_merge.log, merged from the four shard logs local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/750_rt_census_compiled_kernels_shard0.log to _shard3.log)
- E3 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, AKLR CI bench (graph replay, 1 GiB read flush), speedup against the schema's baseline] lowest 1.114 at N 7168 M 128 (AMD-AGI/AKLR@a29581b (CI run 41197527705):aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n7168_k2048/1bbf98a368b3fcba.json); highest 1.840 at N 7168 M 8 (AMD-AGI/AKLR@43e112e (CI run 41991630592):aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n7168_k2048/1e021d44010accb1.json); case triton/checks/census_ci_speedups.py (.08, .10; needs a checkout holding 43e112e, local:/tmp/kbmig/audit7_runs/census_ci_speedups.log)
- E4 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, AKLR CI bench (graph replay, 1 GiB read flush), speedup against the schema's baseline] lowest 1.048 at N 2624 M 2 (AMD-AGI/AKLR@90b9d50 (CI run 41976547645):aklr/ops/gemm/impls/triton/gemm_n2624_k6144/b88c8d5f28d86777.json); highest 1.292 at N 6144 M 8 (AMD-AGI/AKLR@90b9d50 (CI run 41976547645):aklr/ops/gemm/impls/triton/gemm_n6144_k2048/f806ad2b086d757d.json); case triton/checks/census_ci_speedups.py (.11; needs a checkout holding 43e112e, local:/tmp/kbmig/audit7_runs/census_ci_speedups.log)
- E5 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, AKLR CI bench (graph replay, 1 GiB read flush), speedup against the schema's baseline] rms_norm 1.006 at 4 tokens to 1.212 at 2048, 1.006–1.024 at 3–64 tokens (AMD-AGI/AKLR@90b9d50 (CI run 41976547645):aklr/ops/rms_norm/impls/triton/rms_norm_hidden_size6144/cf637bff0784c454.json); add_rms_norm 1.024 at 8 tokens to 1.093 at 256 (AMD-AGI/AKLR@43e112e (CI run 41991630592):aklr/ops/rms_norm/impls/triton/add_rms_norm_hidden_size6144/b3913402078d9226.json); at d908d01 (CI run 42000899019) rms_norm stored 1.023–1.029 at 3–18 tokens and 1.040 at 64 (case triton/checks/rt_launch_trace_gemm_norm.py (.16), local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/150_rt_launch_trace_gemm_norm_r1.log); case triton/checks/census_ci_speedups.py (.11, .12; needs a checkout holding 43e112e, local:/tmp/kbmig/audit7_runs/census_ci_speedups.log)
- E6 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, AKLR CI bench (graph replay, 1 GiB read flush), speedup against the schema's baseline] lowest 1.030 at inter 256, 128 tokens (AMD-AGI/AKLR@369a3c0 (CI run 41753978959):aklr/plugins/sglang/models/deepseek_v2/impls/triton/deepseek_v2_mxfp4_moe_shuffled_16x16_model_dim6144_n_routed256_inter_dim256/a1c4a2129730d1a0.json); highest 2.186 at inter 256, 2 tokens (AMD-AGI/AKLR@43e112e (CI run 41991630592):aklr/plugins/sglang/models/deepseek_v2/impls/triton/deepseek_v2_mxfp4_moe_shuffled_16x16_model_dim6144_n_routed256_inter_dim256/d3bbbce5c0bbc61e.json); case triton/checks/census_ci_speedups.py (.08, .16; needs a checkout holding 43e112e, local:/tmp/kbmig/audit7_runs/census_ci_speedups.log)
- E7 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, AKLR CI bench (graph replay, 1 GiB read flush), speedup against the schema's baseline] lowest 1.178 at E256, 256 tokens (AMD-AGI/AKLR@43e112e (CI run 41991630592):aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_mxfp4_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts256_inter_dim512/4da30fd46e6f82da.json); highest 3.633 at E64, 1 token (AMD-AGI/AKLR@43e112e (CI run 41991630592):aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_mxfp4_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts64_inter_dim1536/b0651fe84db2f670.json); case triton/checks/census_ci_speedups.py (.17; needs a checkout holding 43e112e, local:/tmp/kbmig/audit7_runs/census_ci_speedups.log)
- E8 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, AKLR CI bench (graph replay, 1 GiB read flush), speedup against the schema's baseline] lowest 1.102 at E64, 64 tokens (AMD-AGI/AKLR@43e112e (CI run 41991630592):aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_fp8_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts64_inter_dim1536/098200aca0238880.json); highest 4.115 at E256, 1 token (AMD-AGI/AKLR@93eba6d (CI run 41964469977):aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_fp8_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts256_inter_dim384/e75e553b34178760.json); case triton/checks/census_ci_speedups.py (.18, .19; needs a checkout holding 43e112e, local:/tmp/kbmig/audit7_runs/census_ci_speedups.log)
- E9 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, AKLR CI bench (graph replay, 1 GiB read flush), speedup against the schema's baseline] router 1.074 at 8 tokens (AMD-AGI/AKLR@43e112e (CI run 41991630592):aklr/plugins/sglang/models/qwen2_moe/impls/triton/qwen2_moe_router_model_dim4096_n_routed512_topk10/4801e216dcff290f.json) and 1.967 at 256 (AMD-AGI/AKLR@43e112e (CI run 41991630592):aklr/plugins/sglang/models/qwen2_moe/impls/triton/qwen2_moe_router_model_dim4096_n_routed512_topk10/c3889614985b700a.json); experts 1.076 at 128 (AMD-AGI/AKLR@90b9d50 (CI run 41976547645):aklr/plugins/sglang/models/qwen2_moe/impls/triton/qwen2_moe_fp8_experts_shuffled_16x16_model_dim4096_n_experts513_inter_dim256_topk10/d42593bd3c9a815e.json) and 1.208 at 8 (AMD-AGI/AKLR@93eba6d (CI run 41964469977):aklr/plugins/sglang/models/qwen2_moe/impls/triton/qwen2_moe_fp8_experts_shuffled_16x16_model_dim4096_n_experts513_inter_dim256_topk10/5c35215626bc1c87.json); router + experts 1.063 at 32 (AMD-AGI/AKLR@43e112e (CI run 41991630592):aklr/plugins/sglang/models/qwen2_moe/impls/triton/qwen2_moe_fp8_router_experts_shuffled_16x16_model_dim4096_n_routed512_inter_dim256/99f47a166d6d62fc.json) and 1.172 at 1 (AMD-AGI/AKLR@43e112e (CI run 41991630592):aklr/plugins/sglang/models/qwen2_moe/impls/triton/qwen2_moe_fp8_router_experts_shuffled_16x16_model_dim4096_n_routed512_inter_dim256/7d7e54597a7e11bd.json); case triton/checks/census_ci_speedups.py (.20; needs a checkout holding 43e112e, local:/tmp/kbmig/audit7_runs/census_ci_speedups.log)
- E10 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, AKLR CI bench (graph replay, 1 GiB read flush), speedup against the schema's baseline] lowest 1.030 at 64 heads, 8 tokens (AMD-AGI/AKLR@43e112e (CI run 41991630592):aklr/plugins/sglang/layers/dsa_backend/impls/triton/dsa_sparse_mla_decode_n_heads64_kv_dim576_topk2048/0bb2f7cbe49d1be3.json); highest 2.188 at 64 heads, 24 tokens (AMD-AGI/AKLR@43e112e (CI run 41991630592):aklr/plugins/sglang/layers/dsa_backend/impls/triton/dsa_sparse_mla_decode_n_heads64_kv_dim576_topk2048/2ee56a6dbabc48a6.json); case triton/checks/census_ci_speedups.py (.08, .21; needs a checkout holding 43e112e, local:/tmp/kbmig/audit7_runs/census_ci_speedups.log)
- E11 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, AKLR CI bench (graph replay, 1 GiB read flush), speedup against the schema's baseline] lowest 1.066 at 256 rows (AMD-AGI/AKLR@43e112e (CI run 41991630592):aklr/plugins/sglang/models/qwen3_5/impls/triton/gated_attention_decode_group_fp8_quant_qkv_dim4608_n_kv_heads1_head_dim256/dfb1ab1cdd5bf263.json); highest 2.182 at 1 row (AMD-AGI/AKLR@43e112e (CI run 41991630592):aklr/plugins/sglang/models/qwen3_5/impls/triton/gated_attention_decode_group_fp8_quant_qkv_dim4608_n_kv_heads1_head_dim256/f5ba496e04f8e989.json); case triton/checks/census_ci_speedups.py (.22; needs a checkout holding 43e112e, local:/tmp/kbmig/audit7_runs/census_ci_speedups.log)
- E12 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, AKLR CI bench (graph replay, 1 GiB read flush), speedup against the schema's baseline] lowest 1.282 at 256 rows (AMD-AGI/AKLR@a29581b (CI run 41197527705):aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_decode_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072/c2c3a42a3909684a.json); highest 2.507 at 2 rows (AMD-AGI/AKLR@90b9d50 (CI run 41976547645):aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_decode_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072/299133886d479017.json); bf16-output op 1.461 at 64 rows (AMD-AGI/AKLR@90b9d50 (CI run 41976547645):aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_decode_n_v_heads16_key_dim128_value_dim128_conv_dim3072/cd00c50e595b05bc.json); case triton/checks/census_ci_speedups.py (.23; needs a checkout holding 43e112e, local:/tmp/kbmig/audit7_runs/census_ci_speedups.log)
- E13 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, AKLR CI bench (graph replay, 1 GiB read flush), speedup against the schema's baseline] the 9 Qwen3-Next decode files, unchanged, first measured on main in d908d01; best file per Point lowest 1.457 at 256 rows (AMD-AGI/AKLR@d908d01 (CI run 42000899019):aklr/plugins/sglang/models/qwen3_next/impls/triton/qwen3_next_gdn_decode_group_fp8_quant_n_v_heads8_key_dim128_value_dim128_conv_dim2048/f44007374038041f.json), highest 3.350 at 32 rows (AMD-AGI/AKLR@d908d01 (CI run 42000899019):aklr/plugins/sglang/models/qwen3_next/impls/triton/qwen3_next_gdn_decode_group_fp8_quant_n_v_heads8_key_dim128_value_dim128_conv_dim2048/38a742cfd908c997.json); per-size file against f440 at 1, 2, 4, 8, 16, 32, 64 rows: b31b 2.443 / 2.390, bba1 2.538 / 2.450, 3719 2.370 / 2.370 and 2.506 / 2.479, 35b9 3.222 / 3.101, 38a7 3.350 / 3.091, 0c0b 2.462 / 2.401; at 128 and 256 rows 0678 1.782 and 1.425, c784 1.466 and 1.151, f440 1.755 and 1.457 (AMD-AGI/AKLR@d908d01 (CI run 42000899019):aklr/plugins/sglang/models/qwen3_next/impls/triton/qwen3_next_gdn_decode_group_fp8_quant_n_v_heads8_key_dim128_value_dim128_conv_dim2048/{b31bfeccf6dcfd7c,bba17a5608327b11,371905e46b19ee36,35b9cfd56833844f,0c0b7dfe26861b02,06780bcbbb3e3c70,c7840603329d5ed5}.json)
- E14 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, AKLR CI bench (graph replay, 1 GiB read flush), speedup against the schema's baseline] lowest 0.997 at 8193 tokens in 5 sequences (AMD-AGI/AKLR@3fb9033 (CI run 41812750710):aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_prefill_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072/c53b6d874d511c02.json) and 1.029 at 4097 in 3 (AMD-AGI/AKLR@3fb9033 (CI run 41812750710):aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_prefill_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072/2ecece9650b95f71.json); highest 2.638 at 512 tokens in 1 (AMD-AGI/AKLR@43e112e (CI run 41991630592):aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_prefill_group_fp8_quant_max_seq_len512_n_v_heads16_key_dim128_value_dim128_conv_dim3072/1d5568a78971fb62.json); case triton/checks/census_ci_speedups.py (.24; needs a checkout holding 43e112e, local:/tmp/kbmig/audit7_runs/census_ci_speedups.log)
- E15 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, AKLR CI bench (graph replay, 1 GiB read flush), speedup against the schema's baseline] the 11 Qwen3-Next prefill files, unchanged, first measured on main in d908d01; best file per Point lowest 1.233 at 13312 tokens in 13 sequences (AMD-AGI/AKLR@d908d01 (CI run 42000899019):aklr/plugins/sglang/models/qwen3_next/impls/triton/qwen3_next_gdn_prefill_group_fp8_quant_n_v_heads8_key_dim128_value_dim128_conv_dim2048/010eab6660c4e415.json), highest 1.741 at 5120 in 1 (AMD-AGI/AKLR@d908d01 (CI run 42000899019):aklr/plugins/sglang/models/qwen3_next/impls/triton/qwen3_next_gdn_prefill_group_fp8_quant_max_seq_len5120_n_v_heads8_key_dim128_value_dim128_conv_dim2048/f1374046df2fef79.json); twins, constexpr / runtime: 162f / a99f 1.718 / 1.719; c3a7 / f137 1.727 / 1.741 and 1.308 / 1.311; 4489 / d8f9 1.592 / 1.554, 1.504 / 1.501, 1.490 / 1.484, 1.288 / 1.282, 1.463 / 1.460; ed2a / 98af 1.294 / 1.283; 70f4 / 010e 1.232 / 1.233 and 1.328 / 1.329 (AMD-AGI/AKLR@d908d01 (CI run 42000899019):aklr/plugins/sglang/models/qwen3_next/impls/triton/qwen3_next_gdn_prefill_group_fp8_quant*/*.json)
- E16 [2026-11, gfx950 (MI355X), FlyDSL ? and HIP kernels beside Triton 3.8.0 ones, N=1, AKLR CI bench (graph replay, 1 GiB read flush), speedup against the schema's baseline] FlyDSL gemm_n6144_k2048 M 1 1.191, M 2–64 1.045, 0.403, 0.445, 0.392, 0.361, 0.360, M 128–4096 0.450–0.864 (AMD-AGI/AKLR@43e112e (CI run 41991630592):aklr/ops/gemm/impls/flydsl/gemm_n6144_k2048/eb749c07b94008ab.json); FlyDSL gemm_n2624_k6144 M 1 1.134, M 2, 4, 8, 16, 32, 64 0.420, 0.415, 0.419, 0.396, 0.410, 0.553, M 128–4096 0.623–0.998 (AMD-AGI/AKLR@43e112e (CI run 41991630592):aklr/ops/gemm/impls/flydsl/gemm_n2624_k6144/4001df477e6c67f6.json); HIP add_rms_norm 32 tokens 1.038, 128 tokens 1.067, 512–8192 1.040–1.072 (AMD-AGI/AKLR@43e112e (CI run 41991630592):aklr/ops/rms_norm/impls/hip/add_rms_norm_hidden_size6144/fc82f61df4a69c54.json); Triton add_rms_norm 32 tokens 1.035, 128 tokens 1.062 (AMD-AGI/AKLR@43e112e (CI run 41991630592):aklr/ops/rms_norm/impls/triton/add_rms_norm_hidden_size6144/b3913402078d9226.json); dispatch serves the highest qualifying stored speedup of any language at all 1628 Points (case triton/checks/census_ci_speedups.py (.01), local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/550_census_ci_speedups_r2.log); case triton/checks/census_ci_speedups.py (.13, .14; needs a checkout holding 43e112e, local:/tmp/kbmig/audit7_runs/census_ci_speedups.log)
- E17 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2, probe kernels reading one 4.5 GiB uint8 tensor with buffer_load and with gl.load] buffer_load returns the data up to byte 2147483645 and 0 from byte 2147483646 = 2^31 − 2 on (also at 2^31, 2^31 + 64, 2^31 + 4096 and 2^32 − 1), where gl.load returns the data; the descriptor is built with num_records 0x7ffffffe (case triton/checks/rt_buffer_2gib_range.py (.03.isa, .07, .13), local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/150_rt_buffer_2gib_range_r1.log, repeat local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/100_rt_buffer_2gib_range.log)
- E18 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3, a Gluon probe of 2048 programs × 4 warps reading bf16 chunks with buffer_load, a 1 GiB read before each launch, median of 9 launches per round, 3 rounds alternating consecutive ids and ids congruent mod 8] 8 readers per chunk, consecutive over congruent time per round: 64 KB 1.76–1.82 (21.3–21.6 against 11.7–12.1 µs), 512 KB 2.07–2.17 (154.4–160.5 against 74.1–76.8 µs); one reader per chunk (the congruent map a stride-8 permutation), median ratio over the rounds 1.005, 1.012, 0.980 by run, 25.2–26.0 against 25.4–25.9 µs (case triton/checks/bench_xcd_l2_reuse.py (bench_xcd_l2_reuse.03, .08, .12), local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/200_bench_xcd_l2_reuse.log, local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/800_bench_xcd_l2_reuse_rep2.log, local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/880_bench_xcd_l2_reuse_rep3.log)
- S1 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/7266cdc0ff1fd299.json#kernel.py:243 one unit-scale `mfma_scaled` per 128-K group into a zero accumulator, scaled and added at :253; `compute_efficient_padded_shared_layout` at :84, :89; async copies and `commit_group` at :114-116
- S2 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/bd017b0794d5db31.json#kernel.py:728 rows of A and B as (token or column, K group) pairs; the matching-group diagonal kept at :781-786; groups summed at :792
- S3 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/bd017b0794d5db31.json#kernel.py:720 contiguous N runs per XCD; the M 16 stripes in 083b493a1861424c.json#kernel.py:306-307
- S4 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/831926aed9523c0f.json#kernel.py:562 a 16-row tail tile for the last row, the bulk compiled at M - M % BM (:597)
- S5 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n7168_k2048/06755ba3ee5eb085.json#kernel.py:300 the one launch, shapes and strides as constexprs, waves_per_eu by M at :303; `assert m in (32, 64) and n == 7168 and k == 2048` at :298; `BK = 256`, `STAGES = 8` at :159-160; the 224-tile permutation `tile * 8 - quotient * 223` at :171-172
- S6 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/54f6affa5c286a70.json#kernel.py:422 `assert n == 3072 and k == 1536 and weight.shape[1] == k`; `for group in gl.static_range(12)` at :279; one A and one B LDS buffer per K group at M 16 and 32: 083b493a1861424c.json#kernel.py:334-340, fcf21622f6c91bbb.json#kernel.py:261-267
- S7 aklr/ops/gemm/impls/triton/gemm_n2624_k6144/b88c8d5f28d86777.json#kernel.py:96 `@functools.cache` `_launch_for` returns the launch; the fixed-order `_reduce_kernel` at :87, launched after the split-K GEMM at :154
- S8 aklr/ops/rms_norm/impls/triton/rms_norm_hidden_size6144/cf637bff0784c454.json#kernel.py:18 grid-stride over rows with `tl.range(..., num_stages=NS)`; `loop = m // 512 if m >= 1024` at :42
- S9 aklr/plugins/sglang/models/deepseek_v2/impls/triton/deepseek_v2_mxfp4_moe_shuffled_16x16_model_dim6144_n_routed256_inter_dim256/5751284b6965eabb.json#kernel.py:77 programs below 256 // BN * SPLITS run router tiles; the rest quantize the hidden states (:107-123)
- S10 aklr/plugins/sglang/models/deepseek_v2/impls/triton/deepseek_v2_mxfp4_moe_shuffled_16x16_model_dim6144_n_routed256_inter_dim256/d3bbbce5c0bbc61e.json#kernel.py:217 `_expert_at_rank`, rank + 1 argmax rounds, its `(256,)` expert vector at :220; `_next_expert`'s inline asm at :134-162; `_select_routes` at :166, ids and weights stored at `row * 9` at :185; `_quantize_groups` at :43; `_activate_quantize` at :367, launched at :516 with `enable_fp_fusion=False` (:517)
- S11 aklr/plugins/sglang/models/deepseek_v2/impls/triton/deepseek_v2_mxfp4_moe_shuffled_16x16_model_dim6144_n_routed256_inter_dim256/1204f286c55bb56e.json#kernel.py:351 gate rows then the matching up rows in one N tile; SwiGLU and MXFP4 encoding in the epilogue at :215-221
- S12 aklr/plugins/sglang/models/deepseek_v2/impls/triton/deepseek_v2_mxfp4_moe_shuffled_16x16_model_dim6144_n_routed256_inter_dim512/7e9c13e0f2d58a3b.json#kernel.py:486 block-diagonal down, the diagonal gathered and summed in slot order to :507; the whole call `fused_moe` at :606-692
- S13 aklr/plugins/sglang/models/deepseek_v2/impls/triton/deepseek_v2_mxfp4_moe_shuffled_16x16_model_dim6144_n_routed256_inter_dim256/1204f286c55bb56e.json#kernel.py:513 `gl.associative_scan` over the per-expert counts; `v_readfirstlane_b32` at :521
- S14 aklr/plugins/sglang/models/deepseek_v2/impls/triton/deepseek_v2_mxfp4_moe_shuffled_16x16_model_dim6144_n_routed256_inter_dim256/699f9fa93bf64b26.json#kernel.py:11 the prefill launches in order: _prologue, _select_routes, _prepare, _experts, _paired_down, _shared_down
- S15 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_mxfp4_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts256_inter_dim512/84116a96e535f174.json#kernel.py:1346 the relaxed ticket; the scatter at :1221-1227 with its bases at :1188-1194; the worst-case capacity at :1373; the early exit on Total at :443
- S16 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_mxfp4_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts64_inter_dim1536/abc7bbf7a7135f28.json#kernel.py:531 per-expert route masks from seven wave ballots, popcount histogram at :537
- S17 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_mxfp4_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts256_inter_dim512/d8610d895915c167.json#kernel.py:772 copies of the selected experts made on one wave, `owner` masks all but one; a job every 16th ticket at :792
- S18 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_fp8_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts256_inter_dim384/e75e553b34178760.json#kernel.py:134 `_down_combine`'s MFMA layout with `warps_per_cta=[8, 1, 1]`
- S19 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_fp8_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts64_inter_dim1536/82e9371086a875bd.json#kernel.py:175 SWAR counters and one scan in `_rank_counts` (:161-189); `gl.static_assert(512 % TOPK == 0 and 512 // TOPK < 256)` at :170
- S20 aklr/plugins/sglang/models/qwen2_moe/impls/triton/qwen2_moe_router_model_dim4096_n_routed512_topk10/c3889614985b700a.json#kernel.py:19 `_ShapeConfig(split, expert_tile, k_tile, lookahead, routing_group)` (to :24), `_shape_config` at :27-34; the two launches with their `llvm_fn_attrs` at :294-306
- S21 aklr/plugins/sglang/models/qwen2_moe/impls/triton/qwen2_moe_fp8_router_experts_shuffled_16x16_model_dim4096_n_routed512_inter_dim256/7d7e54597a7e11bd.json#kernel.py:282 each CTA pops its row's queue `slot` times; one CTA publishes the routes at :290
- S22 aklr/plugins/sglang/models/qwen2_moe/impls/triton/qwen2_moe_fp8_router_experts_shuffled_16x16_model_dim4096_n_routed512_inter_dim256/99f47a166d6d62fc.json#kernel.py:81 a float64 accumulator fed by `gl.amd.cdna3.mfma` on float64 operands (:86-88)
- S23 aklr/plugins/sglang/models/qwen2_moe/impls/triton/qwen2_moe_fp8_experts_shuffled_16x16_model_dim4096_n_experts513_inter_dim256_topk10/86987a66c0cdba39.json#kernel.py:110 `atomic_scatter_add` on an LDS counter array, scanned into tasks to :131
- S24 aklr/plugins/sglang/layers/dsa_backend/impls/triton/dsa_sparse_mla_decode_n_heads64_kv_dim576_topk2048/2ee56a6dbabc48a6.json#kernel.py:584 head tile and splits from the row count; the `(rows, heads // head_tile, splits)` grid at :593; `assert heads == 64 and width == 576 and (value_dim == 512)` at :581; `cache_buffer` (cache under 2**31 bytes) at :592; QK `mfma_scaled` on unscaled e4m3 into the running `scores` at :120
- S25 aklr/plugins/sglang/layers/dsa_backend/impls/triton/dsa_sparse_mla_decode_n_heads64_kv_dim576_topk2048/f60f22e6f14448c6.json#kernel.py:627 the merge: the maximum over splits, rescale at :629, guarded `1 / den` at :645
- S26 aklr/plugins/sglang/layers/dsa_backend/impls/triton/dsa_sparse_mla_decode_n_heads8_kv_dim576_topk2048/fbe6ddab5dac5958.json#kernel.py:709 `_direct_attention[m,]`, one program per token, no merge (its condition at :708)
- S27 aklr/plugins/sglang/layers/dsa_backend/impls/triton/dsa_sparse_mla_decode_n_heads64_kv_dim576_topk2048/0bb2f7cbe49d1be3.json#kernel.py:939 `q.to(torch.bfloat16)` in `run`
- S28 aklr/plugins/sglang/models/qwen3_5/impls/triton/gated_attention_decode_group_fp8_quant_qkv_dim4608_n_kv_heads1_head_dim256/f5ba496e04f8e989.json#kernel.py:43 the prologue call, the attention call at :58; the cached `torch.ones(1)` at :19-22
- S29 aklr/plugins/sglang/models/qwen3_5/impls/triton/gated_attention_decode_group_fp8_quant_qkv_dim4608_n_kv_heads1_head_dim256/dfb1ab1cdd5bf263.json#attention.py:140 the 1-D split grid; `splits = 2 if rows == 256 else 4` at :334; `assert rows in (128, 256) and heads == 8 and dims == 256` at :329
- S30 aklr/plugins/sglang/models/qwen3_5/impls/triton/gated_attention_decode_group_fp8_quant_qkv_dim4608_n_kv_heads1_head_dim256/6dd321aac1b1d85a.json#attention.py:171 "Every split publishes fresh neutral records, including inactive splits"
- S31 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_decode_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072/2c2a09e220f2706c.json#kernel.py:8 the second launch shifts history after every reader of the first; the two launches at :315-328
- S32 aklr/plugins/sglang/models/qwen3_next/impls/triton/qwen3_next_gdn_decode_group_fp8_quant_n_v_heads8_key_dim128_value_dim128_conv_dim2048/c7840603329d5ed5.json#kernel.py:34 the convolution, `gl.barrier()` at :40, the in-place shift at :41-44
- S33 aklr/plugins/sglang/models/qwen3_next/impls/triton/qwen3_next_gdn_decode_group_fp8_quant_n_v_heads8_key_dim128_value_dim128_conv_dim2048/f44007374038041f.json#gdn_decode_config.py:187 `CONFIGS`, one row per token count; the exact-match `lookup` at :323
- S34 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_prefill_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072/382853eb770e30b3.json#initial_state.py:10 `_initialize_fresh` zeroes the slots of sequences without an initial state
- S35 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_prefill_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072/24ede969ed66a87a.json#kernel.py:471 `BV = 64`; the `[heads, 2, batch]` grid at :739; the same in c53b6d874d511c02.json#kernel.py:507, :755
- S36 aklr/plugins/sglang/models/qwen3_next/impls/triton/qwen3_next_gdn_prefill_group_fp8_quant_n_v_heads8_key_dim128_value_dim128_conv_dim2048/010eab6660c4e415.json#kernel.py:57 `M` and `BATCH` as runtime ints; constexprs in 70f4be35b5e88940.json#kernel.py:57-58
- S37 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_prefill_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072/0a07f9f9342c6cf1.json#kernel.py:386 "Commit history only after all preparation readers have completed"
- S38 aklr/plugins/sglang/models/qwen3_next/impls/triton/qwen3_next_gdn_prefill_group_fp8_quant_n_v_heads8_key_dim128_value_dim128_conv_dim2048/11f62a16b4778338.json#kernel.py:151 "BF16 for logical BF16 Gram products; native FP32 for state algebra."
- S39 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_prefill_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072/24ede969ed66a87a.json#kernel.py:246 `_permute_columns_for_a`; the accumulator converted to the A operand with `assert_trivial=True` at :554-557
- S40 aklr/plugins/sglang/models/qwen3_5/impls/triton/gated_attention_decode_group_fp8_quant_qkv_dim4608_n_kv_heads1_head_dim256/d9cbde237fd34de9.json#attention.py:116 `head_tile = 16 if SPLITS == 16 else HEADS`; the split count by rows at :358; the `(rows, splits)` grid at :367, read as `row`, `split` at :113-114
- S41 aklr/plugins/sglang/models/qwen3_next/impls/triton/qwen3_next_gdn_decode_group_fp8_quant_n_v_heads8_key_dim128_value_dim128_conv_dim2048/06780bcbbb3e3c70.json#kernel.py:78 the conv layout with no replicated lanes or waves (to :81); `gl.static_assert(NW * 64 <= conv_width)` at :70; the shift right after the reads, no barrier, at :48-51
- S42 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_mxfp4_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts256_inter_dim512/655723f593921b08.json#kernel.py:361 `_project_gate_up_fold`, `mfma_scaled(a, sa, "e2m1", b, sb, "e2m1", acc)` at :425; `gate_fold` by token count at :1301
- S43 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_decode_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072/4f257a3e566b20d1.json#kernel.py:13 `_store_quantized_head`: amax floored at 1e-10, scale × (1 / 448), reciprocal, `v_med3_f32` clamp, rtne cast
- S44 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_fp8_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts256_inter_dim384/5cac8212f51edf84.json#kernel.py:33 `_activate_quantize`: split partials summed, SwiGLU, fp8 per 128, launched at :950; the Qwen2 experts' in aklr/plugins/sglang/models/qwen2_moe/impls/triton/qwen2_moe_fp8_experts_shuffled_16x16_model_dim4096_n_experts513_inter_dim256_topk10/5c35215626bc1c87.json#kernel.py:31
- S45 aklr/plugins/sglang/layers/dsa_backend/impls/triton/dsa_sparse_mla_decode_n_heads64_kv_dim576_topk2048/2f7000a8186a8ae8.json#kernel.py:702 head tile 32 and 4 splits at 32 rows (:703); `TILES = TOPK // (SPLITS * 256)` at :368, the tile loop at :413
- S46 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n7168_k2048/1db61e7b0dc84d90.json#kernel.py:458 `SLOTS = 3 if M == 64 else 4` A slots; `B_SLOTS` 6 when `M == 16 and SXM == 2048 and SXK == 1 and SXSK == 1`, else `SLOTS` (:459-460), allocated at :492-493; both operands on `SLOTS` in ae4fa8a440b98f09.json#kernel.py:186 (`4 if M == 32 else 3`, :209-210) and 06755ba3ee5eb085.json#kernel.py:161 (`3 if M == 64 else 4`, :198-199)
- S47 harness/skills/aklr-kb/triton/kernels/kernel_library.md:1 page-local Evidence E1–E25 and source citations S1–S46 retain their stated scopes.
