---
title: "Triton and Gluon kernels on gfx950: the family map"
kind: topic
ops: [gemm, grouped_gemm, moe, fused_moe, attention, mla, gdn, norm]
dtypes: [fp8, mxfp4, bf16, fp32]
regimes: [all]
gens: [gfx950]
triton: 3.8.0
---

# Triton and Gluon kernels: the family map

"The library", on this page and the other Triton and Gluon pages, is one set of 154 Triton-language kernels for gfx950 (MI355X): 150 Gluon (`@gluon.jit`) and 4 plain `@triton.jit`, in eleven families; §2–§14 describe them. The library later grew to 280 (276 Gluon), "after its growth" on these pages: the families added or grown since are listed after §2's table. Per family: what it replaces, its sizes, kernels, launches, decomposition, techniques, compiled resources and measured speedups, to start a new kernel from the closest one; packaging and checking it: [authoring](../authoring.md) §10, §11. The compiled resources come from one build of every kernel *(field, E2)*.

## Route here when

- You start a Triton or Gluon kernel for a new op or shape and look for the kernel nearest in op, dtype and token count, or need to know that no Triton kernel here covers your op or shape (§3).
- You change or tune a family and need its per-size kernels and launch structure; the ablations run on these kernels are in [tuning](../tuning.md) §3, §5, §9, §11, §12.
- You want a worked example of a technique: async LDS rings, MXFP4 `mfma_scaled`, atomic-ticket sorts, split-KV merges, fused fp8 group quantization (§4).

## 1. Using a kernel of the library as the starting point

- **How this page names a kernel.** By family and size: "the N 3072 fp8 GEMM for M = 16", "the 24-token DSA kernel"; the bf16 GEMMs by both dimensions (N 6144 × K 2048, N 2624 × K 6144); MoE kernels by the labels of [moe](../moe.md) §1: DS256 and DS512 are DeepSeek-V2 on a rank of that expert intermediate dimension, MM256-FP8, MM64-FP8, MM256-FP4 and MM64-FP4 MiniMax-M2 on a rank holding that many of its experts, with FP8 or MXFP4 weights. A kernel here is one host entry, `run`, with the `@gluon.jit` or `@triton.jit` functions it launches, written for a fixed list of sizes, a size being one value of each variable dimension of its op; the library holds 154 of them, 280 after its growth. The `Copy from` bullets show the code to start from, or link the page that shows it.
- **One design per size.** 116 of the library's 154 kernels are written for one size, the others for 2–14, and only Qwen3-Next's sizes have more than one Triton kernel (§13, §14); among the later additions the DSA indexer, DSA 8-head (32, 128 and 256 tokens), both fp8 router + experts shapes and Qwen3-Next GDN decode also have two or three kernels at a size. Both bf16 GEMM shapes also have a FlyDSL kernel at their sizes, and add + RMSNorm a HIP kernel at its token counts with another epsilon (§6). In a host-AST scan (token count read from `m`, `rows`, `tokens` or `num_tokens`), 73 of the 116 single-size kernels (the N 3072 fp8 GEMM for M = 4096 and the DS512 kernel for 256 tokens among them) keep a host branch or conditional arm for a size they are not written for. Follow your size's branch before you copy a constant.
- **The shape is in the code.** A kernel is written for one op shape and writes its constant axes as literals: N 3072's 12 K groups as `gl.static_range(12)` (the M = 64 kernel), N 7168's 8 stages of 256 K and its 224-tile order `tile * 8 - quotient * 223` (the M = 64 kernel), 256 experts as `(256,)` vectors and top-8 plus the shared expert as `row * 9` (the DS256 kernel for 2 tokens). Many kernels also assert the shape on the host (`assert n == 3072 and k == 1536`); no DeepSeek or plain-Triton kernel does, and the host code of that DeepSeek kernel and of both bf16 GEMMs ran at other shapes without an error. For another shape read every literal and assert, not only your size's branch, and recount the grid against the 256 CUs ([tuning](../tuning.md) §6). Buffer accesses take 32-bit offsets and return data only for an access, or each dword of a wider one, lying wholly below byte 2^31 − 2 from their scalar base, the descriptor's `num_records`; any other access silently reads 0 *(field, E17, E33)*; of the 137 kernels using them, 30 check a size against 2^31 on the host (the 24-token DSA kernel takes the buffer path only when its cache is under 2^31 bytes), 4 `static_assert` one, and 103 have neither kind of guard ([gluon/memory](../gluon/memory.md) §3). MoE traps: [moe](../moe.md) §12.
- **Triton 3.8 APIs.** 12 of the 15 N 3072 kernels call `llvm_fn_attrs`, `compute_efficient_padded_shared_layout` or `gl.amd.slice`, which Triton 3.7 lacks, and none of the 15 states the Triton version it needs. Under Triton 3.7.0 the M = 6144 kernel fails to compile on a missing helper (CPU, compile only). The setup these kernels were run under skips a kernel whose stated Triton requirement the installed Triton does not meet: in one GPU run, a copy requiring `>=3.9` under 3.8.0 was skipped without being built or timed, and the call kept the original at M = 6144 *(field, E19)* ([authoring](../authoring.md) §3).
- **Copy a size's whole configuration.** Its constants were chosen together: changing one alone made the 24-token DSA kernel fail to build (a `CompilationError` in its split kernel) and the 1-token MiniMax MXFP4 kernel crash (exit −11); moving a constant with what it feeds: [tuning](../tuning.md) §5. Where one could be moved both ways, the library's router split-K (256 tokens) was the fastest of the three values tried, and so was the 32-row gated-attention kernel's pair of KV split count and head tile (tied; with the 64- and 16-row kernels' values: 0.63 and 0.95); the fp8 GEMM at M = 16 lost prefetching 3 K groups instead of 5 (0.87), not 4, and 7 neither gained nor was shown equal (time ratio 7/5 0.9998 and 0.9968 in two runs) *(field, E1)*. Re-tune by measuring at your size ([tuning](../tuning.md) §2), not with a neighbour's value.
- **Helpers are copied, not shared.** Each kernel carries its own helper modules, so families copy them: the module that zeroes fresh state slots is byte-identical in all 26 GDN prefill kernels, the shared math helpers in the 10 DeepSeek kernels that use them.
- **Reading the numbers.** Resources come from compiling every kernel once on MI355X with Triton 3.8.0 *(field, E2)*; a count of compiled kernels there counts compiled-cache records, one per compilation of a `@gluon.jit` or `@triton.jit` function, so the same binary can be counted more than once; "VGPR" is a compiled kernel's `.vgpr_count`, architectural VGPRs plus AGPRs (a 4-warp 256 × 256 `tl.dot` for gfx950 with 256 of each reports 512). A speedup is the ruler's stored value for one size against that op's baseline, the path the kernel replaces (§2's table); [tuning](../tuning.md) §1 defines the ruler, its timer and the bar these runs counted a size against, and the kernels' durations here were read from Kineto. A speedup describes a whole design at one size against one op's baseline: do not rank families by it. The family ranges below come from one run of the ruler per kernel; later runs of the ruler moved some of them. Read a kernel's per-size values before you start from it, because the family ranges hide weak sizes, such as N 7168 at M = 128 (1.114), DS256 at 128 tokens (1.030) and DSA 64 heads at 8 tokens (1.030) *(field, E3, E6, E10)*.

## 2. The families

The table holds the library's eleven families. Shapes are the op's fixed dimensions. Launches are the kernel nodes of one call at the kernel's sizes, a `q.to(torch.bfloat16)` conversion included. Accuracy is the determinism level the kernel is held to ([numerics](../numerics.md) §1, §8): `bitwise`, every run on the same inputs gives the same bytes as the first, or `bounded`, nothing is asked across runs. Neither compares the output with the reference: that is the job of each output's error bound, which holds under both. The `bitwise` kernels use no atomic and each repeated bit for bit at its first size; all 31 kernels with an atomic (tickets, job counters, LDS scatters) are `bounded`. The fp8 GEMM is `bounded` too, although none of its 24 kernels uses an atomic. In the MoE rows a route is one (token, expert) pair, a ticket the position that a relaxed `atomic_add` on its expert's counter returns, a job a run of up to BM routes of one expert, the rows of one gate-up tile.

| Family (§) | Shapes | Replaces | Sizes | Source files per kernel | Launches | Language | Accuracy |
|---|---|---|---|---|---|---|---|
| fp8 block-scaled GEMM (§5) | N 3072 × K 1536; N 7168 × K 2048 | aiter `gemm_a8w8_blockscale` | 15 + 9: M 1–8193, 1–256 | 3 (entry, adapter, kernel) at N 3072, 1 at N 7168 | 1 | Gluon | bounded |
| bf16 GEMM, M ≤ 64 (§6) | N 6144 × K 2048; N 2624 × K 6144 | aiter `gemm_a16w16` | 6 + 4: M 2–64 | 1 | 1; 2 at N 2624 × K 6144, M 64 | Triton | bitwise |
| RMSNorm, add + RMSNorm (§6) | hidden size 6144 | aiter `rmsnorm2d_fwd`, `rmsnorm2d_fwd_with_add` | 14: 3–4096; 7: 4–256 | 1 | 1 | Triton | bitwise |
| DeepSeek-V2/GLM MXFP4 MoE (§7) | model dim 6144, 256 routed experts, expert intermediate dim 256 or 512 on this rank | SGLang `DeepseekV2MoE.forward_normal` | 24: 1–4192 | 1–5 | 3–6 | Gluon | bounded |
| MiniMax-M2 MXFP4 MoE (§8) | model dim 3072, 256 routed experts: all 256 on this rank at intermediate dim 512, or 64 of them at 1536 | SGLang `MiniMaxM2MoE.forward_normal` | 30: 1–16384 | 1 | 3–8 | Gluon | bounded |
| MiniMax-M2 FP8 MoE (§9) | model dim 3072, 256 routed experts: all 256 on this rank at intermediate dim 384, or 64 of them at 1536 | SGLang `MiniMaxM2MoE.forward_normal` | 28: 1–16384 | 1 | 3–8 | Gluon | bounded |
| Qwen2-MoE router, fp8 experts, fp8 router + experts (§10) | model dim 4096, 512 routed experts, top-10; experts at intermediate dim 256, 513 with the shared one | SGLang `Qwen2MoeSparseMoeBlock._forward_router_experts` | 7: 8–4096; 3: 8–128; 4: 1–32 | 1 | 2; 3–5; 4–7 | Gluon | bounded |
| DSA sparse MLA, one query token (§11) | 64 or 8 heads, cache rows of 576 values, top-k 2048 | SGLang `DeepseekSparseAttnBackend._forward_triton_decode` | 5 + 3: 8–32, 32–256 | 1–2 | 2–3 | Gluon | bitwise |
| Qwen3.5 gated attention, one query token (§12) | QKV dim 4608, 1 KV head of dim 256, fp8 group-quantized output | SGLang `Qwen3_5AttentionDecoderLayer.self_attention` | 9: 1–256 | 3 | 3 | Gluon | bitwise |
| GatedDeltaNet, one token per step (§13) | Qwen3.5: 16 value heads, key and value dim 128, conv dim 3072, fp8 group-quantized or bf16 output; Qwen3-Next: 8 value heads, conv dim 2048, fp8 group-quantized output | SGLang `Qwen3_5GatedDeltaNet.forward`, `Qwen3GatedDeltaNet.forward` at decode | 8 + 1: 1–256; 9: 1–256 | 1 (3 in the Qwen3-Next template, the kernel written for all 9 sizes: [gdn](../gdn.md) §1) | 2; 1–2 | Gluon | bitwise |
| GatedDeltaNet, chunked sequences (§14) | the same two layers, fp8 group-quantized output; an op shape per stated maximum sequence length, and one without | the same two `forward` at prefill | 21: 512–16384 tokens in 1–8 sequences; 13: 1024–16384 in 1–16 | 2–3 | 5–8 | Gluon | bitwise |

**Added or grown since** (all Gluon; kernels in the library → after its growth). This page does not describe them further, and none of the kernels added has a speedup measured on this page's ruler.

| Family | Shapes | Kernels | Sizes | Accuracy |
|---|---|---|---|---|
| GLM-5.2 DSA indexer: key prepare; query and key prepare | model dim 6144; with the query, q LoRA rank 2048 and 32 heads | 0 → 17; 0 → 16 | 1–256 tokens, 9 each, two kernels per size | bitwise |
| DSA sparse MLA, one query token, 16 heads; 8 heads (§11) | cache rows of 576 values, top-k 2048 | 0 → 8; 3 → 12 | 36 each, 1–256 tokens (8 heads: 32, 128, 256 in two kernels) | bitwise |
| Qwen3-Next gated attention, one query token | QKV dim 2560, 1 KV head of dim 256; one op shape per stated maximum context and one without (4) | 0 → 31 | 1–256 rows (1–32 at 131072, 1–64 at 65536) | bitwise |
| Qwen3-Next fp8 router + experts, dim 2048, inter 128 | model dim 2048, 512 routed experts, intermediate dim 128 | 0 → 11 | 16: 1–16384 tokens | bounded |
| Qwen3.5 fp8 router + experts (§10) | model dim 4096, 512 routed experts, intermediate dim 256 | 4 → 11 | 4 → 24: 1–16384 tokens | bounded |
| Qwen3.5 MXFP4 MoE block, one op on `Qwen2MoeSparseMoeBlock.forward` | model dim 4096, 512 routed experts, intermediate dim 256, shared expert intermediate dim 256 | 0 → 10 | 11: 1–256 tokens | bounded |
| Qwen3.5 GatedDeltaNet, one token per step (§13): fp8 op; bf16 op | 16 value heads, key and value dim 128, conv dim 3072 | 8 → 9; 1 → 12 | 9; 12 (64 rows twice): 1–256 rows | bitwise |
| Qwen3-Next GatedDeltaNet, one token per step (§13) | 8 value heads, key and value dim 128, conv dim 2048 | 9 → 14 | 9 → 36: 1–256 rows | bitwise |

The Qwen3.5 bf16-output op is the one for layers whose projections are unquantized.

## 3. Which family to start from

| Your op | Start from | What you inherit |
|---|---|---|
| GEMM, e4m3 A and B, fp32 scales per 1 × 128 of A and 128 × 128 of B | §5's N 3072 / K 1536 and N 7168 / K 2048 kernels, the only shapes (picking the nearest one for yours is untested) | one launch at each of the 24 sizes *(field, E20)*, a unit-scale `mfma_scaled` per 128-K group with the fp32 scales applied after it, async LDS copies (§4); the N 3072 kernel for M = 6144 rejects independently strided A and B *(field, E20)* |
| GEMM, bf16, M ≤ 64 | §6: N 6144 (K 2048) or N 2624 (K 6144) | the weight loaded with `.cg`, programs remapped across XCDs, no atomics; split-K with a separate fixed-order reduce kernel at N 2624 × K 6144 only (M = 64) |
| GEMM with MXFP4 operands (e2m1, e8m0 per 32) | the expert GEMMs of §7 or §8 at your token count | `mfma_scaled` on e2m1 with e8m0 scale layouts, MXFP4 activation quantization in the kernel |
| MiniMax router projection, bf16 hidden and fp32 weights | §8 or §9 at your token count | fp32 VALU dots at 1–16 tokens, fp32 or bf16 MFMAs above; sampled projections matched an exact sum at all 58 sizes *(field, E21)*, and complete logits were no less accurate than a GPU fp32 matmul in L2 on 96 of 96 inputs at three sizes *(field, E28)* and on 25 of 26 at 13 more branches (max abs: 23 of 26) *(field, E29)*: finite inputs, no guarantee for every input or for the top-8 ([numerics](../numerics.md) §5) |
| bf16 router in fp64 products (written against the former float64 reference: the fp32 one needs no fp64) | §10's router + experts | fp64 VALU products at 1–4 tokens, fp64 MFMA at 32; the experts stay mixed-precision under their own bound *(field, E30)* |
| bf16 router with top-k | §10 router, 8–256 tokens (five kernels) | a split-K projection of exactly 256 CTAs, then per-row top-k; outputs bitwise identical over 20 repeats *(field, E25)*; 2232 and 4096 tokens untested |
| MoE layer, MXFP4 experts and a shared expert | §7 | gate-up launches, grids and W1 sources per size *(field, E20)*; the shared expert as slot 256 at weight 1 (§7), its combine untested |
| MoE layer, MXFP4 experts, no shared expert, all or some experts on this rank | §8 (MM256-FP4, or MM64-FP4 with an expert offset) | gate-up sources, grids and integer buffers per size *(field, E20)*; ticket (MM256-FP4) or wave-ballot (MM64-FP4) sorts from 1024 tokens; the expert offset and job heights untested |
| MoE layer, fp8 experts with 128 × 128 block scales | §9; with a sigmoid-gated shared expert, §10's router + experts | gate-up host paths per size *(field, E20)*; relaxed-atomic job queues, an atomic-free SWAR counting sort in the MM64-FP8 kernel for 1024–16384 tokens; the job counts and the shared gate untested |
| MoE experts, routing given | §10 fp8 experts | per-route split-K (8 tokens), an LDS `atomic_scatter_add` sort (in one CTA at 64 tokens; also at 128), persistent grid-stride stages (64 tokens) |
| SwiGLU + requantization as its own launch | the activation launch of the DS256 kernel for 2 tokens (§7: K-split gate-up partials summed in order, SwiGLU, MXFP4 per 32); fp8 per 1 × 128: that of the MM256-FP8 kernel for 32–64 tokens (§9) or of the Qwen2 experts kernel for 8 tokens (§10) | not a standalone op: these are launches inside the MoE kernels ([moe](../moe.md) §8) |
| Attention of one query token per sequence over gathered rows (top-k indices) | §11 | key splits merged by a separate atomic-free launch (7 of 8 kernels; the 8-head kernel for 256 tokens runs one workgroup per token, no merge), LDS-staged K and V |
| Attention of one query token per sequence over a paged KV cache the op writes, fp8 output | §12 | a prologue task grid, split-KV partials, a finish that writes the fp8 values and their scales: three launches at all 9 sizes |
| Q/K RMSNorm, RoPE or mRoPE, fp8 paged KV-cache write | §12's prologue launch (the K norm, RoPE and cache-write task of the 32- and 256-row kernels) | a task grid of one-wave CTAs, a whole number per token (320 at 32 rows, 1,024 at 256; [gated-attention launches](../attention/gated_attention_launches.md) §3) |
| A per-slot recurrent state stepped once per token, one token per sequence | §13 | the state split over CTAs or owned by one, an in-place history shift |
| A chunked linear recurrence over long sequences | §14 | parallel chunk phases, one sequential launch, then a parallel epilogue |
| Row-wise normalization | §6 RMSNorm (plain Triton; it measured below the bar of these runs ([tuning](../tuning.md) §1) over aiter at 3–18 tokens, and at 64 in one run (1.024), where a later run of the ruler read 1.040 *(field, E5)*); in Gluon only fused: the Q/K RMSNorm of §12's prologue, the gated RMSNorm of §13 and §14 | one row per program, grid-stride from 1024 rows |
| Many sizes from one source | a host branch per size (RMSNorm: 14 sizes, the N 6144 × K 2048 bf16 GEMM: 6), or the configuration table of the Qwen3-Next GDN template (§13): the per-size kernels beat it at 7 of 9 sizes *(field, E13)* | one configuration row per token count; an unlisted count raises |
| An output quantized to fp8 per 1 × 128 for the next GEMM | the finish of the Qwen3.5 GDN prefill kernel for 16384 tokens in 2 sequences of up to 10000 (§14) or of the 32-row gated attention kernel (§12), the last launch of its call | amax, scale and e4m3 codes in the epilogue; that GDN finish matched a reference quantization of its own bf16 output bit for bit on 100000 random groups *(field, E32)* ([numerics](../numerics.md) §2) |

**Not covered by a Triton kernel.** These ops have only FlyDSL or HIP kernels, or none. Start from
[authoring](../authoring.md) §10 and borrow:

| Op (kernels in other languages) | Nearest Triton code |
|---|---|
| GEMM with MXFP4 `a` and `b` (none) | the e2m1 `mfma_scaled` step of the gate-up fold in the MM256-FP4 kernel for 1 token (§8), without its routing: that call looped over K on row-major operands and [rows, K/32] scales matched torch's fp32 matmul at M = N = 32, K = 512 (the kernel's own function not run) |
| fp8 quantization per group of 128 as its own op (HIP) | the quantizing store of the Qwen3.5 GDN kernel for 8 rows (§13): its codes and amax × fl32(1/448) scales equal the op's reference bit for bit on random bf16 groups ([numerics](../numerics.md) §2); the op stores its scales group-major, as (groups, tokens) |
| MXFP8 quantization (e4m3 per group of 32 with an e8m0 scale) as its own op (HIP) | the group quantization of the DS256 kernel for 2 tokens as its routed experts call it: the e8m0 exponent rounded up from the bits of peak / 6 (MXFP4; its shared expert takes the even rule); the op's reference rounds amax × (1/448) up the same way |
| aiter's `biased_grouped_topk` as its own op (FlyDSL) | §7's selection: sigmoid plus fp32 correction bias, top-8 of 256, renormalized × 2.5 (the selection launch of the DS256 kernel for 2 tokens) |
| aiter's `fused_moe`, and expert-parallel MoE across the ranks of a group in one call (FlyDSL) | the MXFP4 expert GEMMs and layers of §7–§8: the FlyDSL `fused_moe` kernels are `per_1x32` only (33 kernels); 7 of them take the 16 × 16 shuffled weights of §7–§8, the others take gate and up rows interleaved |
| MLA attention over a paged KV pool with FlashMLA's API (FlyDSL), sparse attention of many query rows over CSR ranges with block-scaled e4m3 (HIP) | §11's gathered-row attention of one query token; no Triton softmax-attention kernel attends many query rows over a prompt (§14's chunked GatedDeltaNet runs a linear recurrence over it) |
| fp8 GEMM with per-token scales (none), batched bf16 GEMM, `fp8_paged_mqa_logits`, per-row top-k over paged scores, the mHC residual-stream mixes | nothing close |

Price an uncovered size before you implement it: time the op's baseline at the size you target (a bf16 GEMM at N 16, K 128 took 1.924 µs at M = 1 and 2.787 µs at M = 64 *(field, E31)*); no saving of effort from this order was measured.

Within the covered ops, fp8 block-scaled GEMMs exist at N 3072 / K 1536 and N 7168 / K 2048 only, with row-major
`b`. For another N or K start from a ring whose slots do not grow with K (the 8-slot skeleton of [gemm](../gemm.md) §5
takes 32 KiB of LDS for its 8 slots at K 2048 and at K 7168, compiled offline at N 7168 and N 4096): §5's N 7168 kernels, built for K 2048, stream 8 stages of 256 K through
4 A + 6 B LDS slots at M = 16, 4 + 4 at M = 32 and 3 + 3 at M = 64, 64–72 KiB *(field, E2)*;
the M = 16 kernel takes its 6 B slots only when `a`'s row stride is 2048. The N 3072 kernels at M = 16 and 32 do not carry
over: they give each 128-K group its own LDS slot, 48 and 49 KiB at K 1536 (M = 1, 2, 4, 8: 9, 9, 12, 36 KiB)
*(field, E2)*, about 224 KiB at K 7168 (estimated), above the 160 KiB one work-group may request ([gfx950
memory](../../hardware/gfx950/memory_and_sync.md) §4). Longer K and a preshuffled `b` exist only as FlyDSL kernels
(K 256–16384); long-K evidence in plain Triton is
the bf16 K 6144 GEMM ([gemm/small_m_gemm](../gemm/small_m_gemm.md) §5). bf16 GEMMs exist at N 6144 / K 2048 and N 2624 × K 6144 /
K 6144, norms at hidden 6144 only.

## 4. Techniques across the families

Counts are kernels whose sources use the construct (a text scan of the library's 154 kernels, comments excluded, dead branches included); Example names one kernel that uses it.

| Technique | Where (kernels) | Example | Read |
|---|---|---|---|
| `buffer_load` / `buffer_store` (`gl.amd.cdna4`; the same functions as `gl.amd.cdna3` in 37 kernels): scalar base, 32-bit offsets | 137 kernels (`buffer_load` in 128, `buffer_store` in 93), every Gluon family | the 24-token DSA kernel, gated on the cache's size | [gluon/memory](../gluon/memory.md) §3 |
| async direct-to-LDS copies (`async_copy.buffer_load_to_shared`, `commit_group`, `wait_group`) | fp8 GEMM 24, DeepSeek 6, Qwen2 router 6 and experts 1, DSA 4, gated attention 3, MiniMax FP8 2; staging the 8-token router's hidden tile this way gained nothing measurable over `gl.load`s converted through LDS (8.45 µs both; the weight's load layout changed with it) *(field, E1)* ([tuning](../tuning.md) §5) | the N 3072 fp8 GEMM for M = 6144 | [gluon/memory](../gluon/memory.md) §8, §12 |
| padded LDS layouts (`PaddedSharedLayout`, `compute_efficient_padded_shared_layout`) | fp8 GEMM 14, MiniMax MXFP4 11, DSA 5, Qwen2 router 5, DeepSeek 4, gated attention 3, MiniMax FP8 2 | the N 3072 fp8 GEMM for M = 6144 | [gluon/memory](../gluon/memory.md) §9, [gluon/layouts](../gluon/layouts.md) §4 |
| `mfma_scaled` on e4m3 with `None` scales | fp8 GEMM 24 (each 128-K group's product multiplied by the fp32 block scales), MiniMax FP8 9, Qwen2 router + experts 4 | the N 3072 fp8 GEMM for M = 6144 | [gluon/mfma](../gluon/mfma.md) §2, [numerics](../numerics.md) §4 |
| `mfma_scaled` on e4m3 with `None` scales into a running accumulator | DSA 5 (the 64-head kernels) | the 24-token DSA kernel | [attention](../attention.md) §10 |
| `mfma_scaled` on e2m1 with e8m0 scales | MiniMax MXFP4 23, DeepSeek 19 | the MM256-FP4 kernel for 1 token | [gluon/mfma](../gluon/mfma.md) §2, §3, [numerics](../numerics.md) §3 |
| K groups folded into the MFMA rows at tiny M | all 8 fp8 GEMM kernels at M ≤ 8; the gate-up of the MM256-FP4 kernel for 1 token | the N 3072 fp8 GEMM for M = 1; that MiniMax kernel | [gluon/mfma](../gluon/mfma.md) §8, [gemm](../gemm.md) §4 |
| `gl.inline_asm_elementwise` | 83: MiniMax MXFP4 22, DeepSeek 19, Qwen3.5 GDN decode 8, Qwen3-Next GDN decode 7, fp8 GEMM 6, MiniMax FP8 6, Qwen2 router 5, Qwen3.5 GDN prefill 4, DSA 3, Qwen2 router + experts 3 | the argmax step of the DS256 kernel for 2 tokens (§7) | [gluon/wave_ops](../gluon/wave_ops.md) §1, §2 |
| relaxed global integer atomics (tickets, job counters, bitmasks); LDS `atomic_scatter_*` | DeepSeek 12, MiniMax MXFP4 11, MiniMax FP8 6; LDS: Qwen2 experts 2, MiniMax FP8 1 | the MM256-FP4 kernel for 16384 tokens; the Qwen2 experts kernel for 64 tokens | [moe](../moe.md) §6, [numerics](../numerics.md) §8 |
| fixed-grid graph capture/replay | four MiniMax kernels (MM256-FP4 for 16384, for 8192 and for 2048–4096 tokens; MM64-FP8 for 1024–16384): replayed with new inputs accurately, eager grids identical at two input sets (6, 6, 6 and 8 launches) *(field, E24)*; other routings, worst-case sizing and the device early exit untested | the MM256-FP4 kernel for 16384 tokens | [authoring](../authoring.md) §8 |
| several roles in one grid, split by program id | program-id branches calling different helpers (a static scan that also counts bounds branches): DeepSeek 18 of 19, MiniMax MXFP4 19 of 23, MiniMax FP8 7 of 9, Qwen2 router + experts 4 of 4, gated attention 7 of 9 kernels; a finish branch in 6 Qwen3.5 GDN prefill kernels | the DS256 kernel for 1 token | [moe](../moe.md) §3 |
| a fixed-order merge launch instead of float atomics (bitwise outputs) | bf16 GEMM at N 2624 × K 6144, M 64 (a second kernel adds the six partials in order); DSA and gated attention, which use no atomic and repeat bit for bit | the 16-token DSA kernel; the N 2624 × K 6144 bf16 GEMM (§6) | [attention](../attention.md) §2, [numerics](../numerics.md) §8 |
| program-id tile-order remaps: in a probe with eight readers per chunk, ids congruent mod 8 ran 1.76–2.17× faster than consecutive ones, three runs *(field, E18)*; at the library's sizes, below *(field, E23)* | fp8 GEMM, bf16 GEMM, DeepSeek and MiniMax kernels; the N 2624 × K 6144 bf16 GEMM streams a contiguous N run per XCD and staggers K phases. Neither the probe nor the timings below show where a program ran or that L2 caused a gain | the N 3072 fp8 GEMMs for M = 1 and M = 16 | [gluon/memory](../gluon/memory.md) §14, [../../gemm/schedule_and_locality.md](../../gemm/schedule_and_locality.md) §2 |
| fp32- or fp64-accurate products | MiniMax routers: complete logits no less accurate than a GPU fp32 matmul in L2 on 96 of 96 inputs at three sizes and 25 of 26 at 13 branches *(field, E28, E29)*; Qwen2 router + experts at 1–32 tokens: fp64 router products, all 32 SQNR checks passed *(field, E30)*; GDN prefill state algebra in fp32: the accuracy checks passed; only the random input of the Qwen3.5 kernel for 512 tokens in 2 sequences stayed within a preset limit (that of the kernel for 8193 tokens in 5 exceeded it in max-abs, 37.3 times the fp32 recurrence's error), and on the cancellation inputs the L2 error reached 721 (8193 in 5) and 1192 (512 in 2) times a GPU fp32 recurrence's *(field, E30)*. Finite inputs, no general accuracy guarantee | the Qwen2 router + experts kernel for 32 tokens | [numerics](../numerics.md) §5 |
| fused e4m3 quantization of an output or GEMM input | every kernel of gated attention, GDN decode and prefill (the bf16-output GDN kernel excepted), MiniMax FP8, the Qwen2 experts and the Qwen2 router + experts: 68 kernels | the Qwen3.5 GDN kernel for 8 rows | [numerics](../numerics.md) §2 |
| `enable_fp_fusion=False` on launches | 42: DeepSeek 19, Qwen3.5 GDN prefill 11, Qwen3-Next GDN prefill 11, DSA 1; set per size in 4 more | the DS256 kernel for 2 tokens | [numerics](../numerics.md) §6, [tuning](../tuning.md) §9 |
| `llvm_fn_attrs`; `amdgpu-ieee` only in the Qwen2 routers | 26: Qwen2 router 5, Qwen3.5 GDN prefill 5, fp8 GEMM 4, MiniMax MXFP4 4, Qwen3-Next GDN decode 4, MiniMax FP8 3, DeepSeek 1 | the Qwen2 router for 256 tokens | [tuning](../tuning.md) §9 |
| no MFMA at all | 22: GDN decode 18, gated attention at 1–2 rows 2, RMSNorm 2 | the Qwen3.5 GDN kernel for 2 rows | [gluon/mfma](../gluon/mfma.md) §2, [gdn](../gdn.md) §1 |

**Remaps timed at the library's sizes.** In one run at seven GEMM and MoE sizes of the library, each with 4 balanced AB/BA pairs and no accuracy finding, A was the library's order and B changed only the tile order. Every B sample was slower than every A sample, but only five sizes met the preset criterion: mean B/A at least 1.01, all B slower than all A, and a paired interval above 1. Mean B/A is the ratio of arm means; the interval is the 95% t interval of the paired mean ratio within the run, not across runs *(field, E23)*.

| Size | Mean B/A | Paired-ratio 95% CI | Criterion |
|---|---|---|---|
| N3072/K1536, M64 | 1.11993 | 1.0965–1.1436 | met |
| N3072/K1536, M128 | 1.05936 | 1.0566–1.0621 | met |
| N3072/K1536, M2048 | 1.00651 | 1.0054–1.0076 | resolved change, below 1% |
| N3072/K1536, M4096 | 1.01862 | 1.0181–1.0191 | met |
| N3072/K1536, M8192 | 1.00935 | 1.0085–1.0102 | interval crosses 1.01; >=1% not established |
| N7168/K2048, M128 | 1.04045 | 1.0391–1.0418 | met |
| DeepSeek inter256, 2048 tokens | 1.02654 | 1.0104–1.0428 | met; **whole MoE call**, not an expert GEMM |

A map that puts program `pid` on XCD `pid % 8` only estimates locality: by it, the N 3072 orders at M 2048, 4096 and 8192 spread each weight tile's readers over fewer XCDs but fill more A + B bytes in total (estimated), so "all readers of a weight tile on one XCD" does not describe every faster order. No L2 counter was read; the cause is unknown. At N 3072 / K 1536, M = 256, with a 1 GiB read before each call, replacing `pn ^ (pm * 2)` with `pn` was faster and accurate: 6.0875 → 5.906 µs, B/A 0.97012, paired interval 0.9689–0.9713, 4 balanced pairs *(field, E23)*; with the operands left in cache (20 back-to-back calls in a graph) it was 1.1% slower *(field, E34)*.

**One weight reader does not imply no timing effect.** Removing the permutation at N 7168 / K 2048, M = 64 was equivalent within a preset ±1.5% band: B/A 0.99305, interval 0.9856–1.0005, 4 pairs. At N 3072 / K 1536, M = 1 (B/A 1.00434, interval 0.9895–1.0196) and M = 16 (1.00803, 0.9953–1.0211), 12 pairs each, neither equivalence within ±1.5% and ±1.3% nor a difference was resolved. At N 7168, M = 2 (12 pairs) the order without the permutation was slower by more than 1%: B/A 1.02350, interval 1.0101–1.0372, although pair signs were mixed. All within one run and accurate; pair signs alone prove neither a difference nor equivalence, and none of these isolates L2 *(field, E23)*.

In the 897 compiled-cache records of the library's kernels *(field, E2)*: direct-to-LDS `buffer_load … lds` in 71 (fp8 GEMM 24 of 24, MiniMax FP8
17 of 188, DeepSeek 10 of 116, bf16 GEMM 6 of 7, Qwen2 router 6 of 14, DSA 4 of 15, gated attention 3 of 27, Qwen2
experts 1 of 13); transposing LDS reads, all `ds_read_b64_tr_b8`, only in DSA 8, gated attention 6, DeepSeek 5, MiniMax
MXFP4 3; `v_cvt_scalef32_pk_fp4_f32` in DeepSeek 36, MiniMax MXFP4 62; global or buffer atomics only in DeepSeek 15,
MiniMax FP8 17, MiniMax MXFP4 18; `v_permlane16_swap` / `v_permlane32_swap` in 144; `ds_bpermute_b32` in 291.

## 5. fp8 block-scaled GEMM

`a` (M, K) and `b` (N, K) in e4m3, fp32 scales per 1 × 128 of `a` and 128 × 128 of `b`, bf16 out; 15 kernels at N 3072,
K 1536 and 9 at N 7168, K 2048, one size each. Op card: [../../gemm/INDEX.md](../../gemm/INDEX.md); Triton side:
[gemm](../gemm.md).

| M | N 3072, K 1536 | N 7168, K 2048 |
|---|---|---|
| 1–4 | a kernel per M (1), (2), (4): 768 programs × 1 wave, 4M outputs each | a kernel per M (1), (2), (4): 448 × 1 wave, 16M outputs each |
| 8 | one kernel: 192 × 2 waves | one kernel: 448 × 1 wave |
| 16–64 | a kernel per M (16), (32), (64): 192–768 × 1 wave, 256 outputs each | a kernel per M (16), (32), (64), a staged pipeline: 224 × 2–4 waves, 32M outputs each |
| 128–256 | a kernel per M (128), (256): 192 × 4 waves, 2048 and 4096 outputs each | a kernel per M (128), (256): 224 and 448 × 4 waves, 4096 outputs each, 32 × 32 × 64 MFMA |
| 1024–8193 | a kernel per M (1024), (2048), (4096), (6144), (8192), (8193): 512–1560 × 4 waves, 6144–24576 outputs per bulk program | — |

- **Decomposition.** One launch, one output tile per program over all of K: no split-K, atomics or scratch. Each 128-K group produces a unit-scale partial, sometimes through several `mfma_scaled` calls, that the two fp32 scales multiply before it is added; the epilogue casts to bf16 and stores.
- **Techniques.** In all 8 kernels at M ≤ 8 the MFMA rows hold (token, K group) pairs and the matching-group diagonal is
  kept ([gemm](../gemm.md) §4, [../../gemm/small_m.md](../../gemm/small_m.md)). All 24 stage tiles with async
  `buffer_load_to_shared`, `commit_group` and `wait_group` ([gemm](../gemm.md) §5, [gluon/memory](../gluon/memory.md) §12).
  At N 3072, the program-id address maps under the `pid % 8` model give one contiguous N run per group at M 1–4 (N 7168 too at M 2, 4), stripes
  that keep all M tiles of an N tile in one group at M 16–64, and windows of 2 or 4 ids on one N tile at M 2048–8193
  ([gemm](../gemm.md) §8, [../../gemm/schedule_and_locality.md](../../gemm/schedule_and_locality.md) §2); N 7168 has no
  remap at M 1, 16, 32. M = 8193 compiles the bulk at M = 8192 and adds 16-row tail programs ([gemm](../gemm.md) §7).
- **Copy from.** The per-group step of the N 3072 kernel for M = 6144 (shown and explained in [gemm](../gemm.md) §2), and the N 7168 host:
  one launch with shape and strides as constexprs and `waves_per_eu` by M (the kernel for M = 64, whose host also takes M = 32):

  ```python
  m, k = x.shape
  n = weight.shape[0]
  assert m in (32, 64) and n == 7168 and k == 2048  # the shape is asserted, then written as literals in the kernel
  assert group_n == 128 and group_k == 128
  _pipeline_output[(n // 32,)](  # one launch: 224 programs, each a 32-column tile over all of K
      x, x_scales, weight, weight_scales, out, m, n,
      *x.stride(), *x_scales.stride(), *weight.stride(), *weight_scales.stride(),  # constexpr parameters
      num_warps=4, num_stages=1, waves_per_eu=1 if m == 32 else 2,
  )
  ```
- **Resources** *(field, E2)*: 24 compiled kernels; `num_warps` 1 (10), 2 (2), 4 (12); `waves_per_eu` 1–8 in 13; VGPR 40–242,
  AGPRs in 8 (8–32); LDS 9–72 KiB; no spills; `v_mfma_scale_f32_16x16x128_f8f6f4` in 21, `…_32x32x64_…` in 3.
- **Speedups** over aiter *(field, E3)*: 1.114 (N 7168, M 128) to 1.840 (N 7168, M 8) over the 24 sizes.

## 6. Plain Triton: bf16 GEMM at M ≤ 64 and RMSNorm

The four `@triton.jit` kernels. Why plain Triton fits them, and what its compiler adds (coalescing, `num_stages`
pipelining, buffer instructions): [authoring](../authoring.md) §2. Each GEMM kernel but the split-K reduce holds
`buffer_load_dwordx4 … lds`; the 6 RMSNorm kernels load with `buffer_load_dwordx4`, no `global_load` *(field, E2)*.

- **bf16 GEMM at M ≤ 64**, `out = a @ b.T` with `b` (N, K), no bias: one kernel for N 6144 × K 2048 (M = 2, 4, 8, 16, 32, 64) and one for
  N 2624 × K 6144 (M = 2, 4, 8, 64). One workgroup per N tile (192 at N 6144 × K 2048, 164 at N 2624 × K 6144), each `tl.dot` accumulating in fp32,
  the weight loaded with `.cg`; each XCD streams one contiguous run of N tiles at N 2624 × K 6144 for M ≤ 8 and at N 6144 × K 2048 for M ≤ 16,
  8-tile chunks (3 runs per XCD) at N 6144 × K 2048 for M 32, 64 (the N 2624 × K 6144 remap is a permutation for any grid size); at
  some M a per-workgroup K phase. One launch, except N 2624 × K 6144 at M = 64: split-K 6 into fp32 scratch (246 workgroups), then
  a second kernel adds the six partials in a fixed order (split 6 measured fastest of seven split counts at this size, split 4 at 1.11–1.15× its time: [gemm/small_m_gemm](../gemm/small_m_gemm.md) §6):

  ```python
  @triton.jit
  def _reduce_kernel(W, C, TOTAL, SPLIT_K: tl.constexpr, BLOCK: tl.constexpr):
      offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
      ok = offs < TOTAL
      acc = tl.load(W + offs, mask=ok, other=0.0, cache_modifier=".cg")
      for s in tl.static_range(1, SPLIT_K):  # split 0, 1, ..., 5: the same order in every run
          acc += tl.load(W + s * TOTAL + offs, mask=ok, other=0.0, cache_modifier=".cg")
      tl.store(C + offs, acc.to(C.dtype.element_ty), mask=ok)  # round to bf16 once, after the sum
  ```

  Not this: the split-K programs adding their partials into `out` with an atomic add. The sum then follows the
  arrival order, and two runs need not give the same bytes ([numerics](../numerics.md) §8). The configuration comes from a `@functools.cache`
  compile function of the two operand shapes that returns the launch closure. Tiles by M, split-K under a bitwise
  output, M = 1 and re-tuning for cold-cache timing: [gemm/small_m_gemm](../gemm/small_m_gemm.md).
- **RMSNorm**: one kernel for RMSNorm (3–4096 tokens) and one for add + RMSNorm (4–256
  tokens), hidden size 6144. The 6144-wide row as three 2048-wide chunks in registers, the sum of squares in fp32, `.cg` loads, `.wt`
  stores, one program per row *(field, E2)*; RMSNorm from 1024 rows gives each program `m // 512` rows, the next
  row's loads pipelined and the weight loaded once:

  ```python
  for it in tl.range(0, LOOP, num_stages=NS):
      row = tl.program_id(0) + it * tl.num_programs(0)
      rmask = cmask & (row < M)
      base = row.to(tl.int64) * H + cols
      x0 = tl.load(x_ptr + base, mask=rmask, other=0.0, cache_modifier=".cg").to(tl.float32)
  ```
- **Resources** *(field, E2)*: GEMM 7 compiled kernels, `num_warps` 2, 4 or 8, `waves_per_eu` 2 in 6, VGPR 26–106, LDS up to
  about 148 KiB of the pipeliner's buffers, `v_mfma_f32_16x16x32_bf16`; RMSNorm 6 compiled kernels, 4 warps, VGPR 49–79, 16 B of LDS.
- **Speedups** over aiter *(field, E4, E5)*: GEMM 1.048 (N 2624 × K 6144, M 2) to 1.292 (N 6144 × K 2048, M 8); RMSNorm
  1.006 (4 tokens) to 1.212 (2048), below the bar of these runs ([tuning](../tuning.md) §1) at 3–64 tokens (a later
  run of the ruler read 1.023–1.029 at 3–18 tokens and 1.040 at 64); add + RMSNorm 1.024 (8 tokens, also below the bar) to
  1.093 (256).
- **Other languages at these sizes.** The three op shapes also have a FlyDSL or HIP kernel, and at each size the
  kernel to beat is the best measured one of any language. A HIP add + RMSNorm kernel measured 1.038 and 1.067 at 32 and 128 tokens against the Triton kernel's 1.035 and
  1.062, so it is the better one there, and it alone covers 512–8192 tokens. FlyDSL kernels are the best at M = 1 of both GEMM
  shapes (1.191 at N 6144 × K 2048, 1.134 at N 2624 × K 6144) and measure 0.360–1.045 at the Triton kernels' sizes; at N 2624 × K 6144, M 16 and
  32, and in both shapes from M 128, the only kernels are FlyDSL ones below 1.0 (0.396–0.998), so aiter's GEMM stays
  the fastest there *(field, E16)*. A new kernel at a shared size has to beat the best one already there.

## 7. DeepSeek-V2 / GLM MXFP4 MoE

SGLang's `DeepseekV2MoE` on one rank: bf16 router logits, the sigmoid top-8 of 256 with an fp32 correction bias, weights
renormalized × 2.5, the shared expert as slot 256 at weight 1, MXFP4 experts (e2m1 with e8m0 per 32, aiter's 16 × 16
shuffled weights) and their weighted sum. This rank's expert intermediate dimension is 256 or 512 (DS256 and DS512, the labels of [moe](../moe.md) §1); 19 kernels, 24 sizes. The baseline
replays aiter's router GEMM, `biased_grouped_topk` and `fused_moe`. Triton side: [moe](../moe.md); op card:
[../../moe/INDEX.md](../../moe/INDEX.md). Kernels with their token counts in parentheses; DS512 has no kernel at 32–128.

| Tokens | DS256 | DS512 | Launches | Decomposition |
|---|---|---|---|---|
| 1–2 | a kernel per count (1), (2) | one kernel (1, 2) | 3–4 | router (split-K tiles, or one full-K program per expert at DS256, 2 tokens) beside the hidden-state MXFP4 quantization (192 + 24 programs at DS256, 1 token); the gate-up of every route split along K (4 splits at DS256, 12 at DS512 at 1 token) and finished by a small launch, or whole (DS512 at 2); a last launch of down and combine over (token, column tile): 16 columns at DS256, 32 and 64 at DS512 at 1 and 2 |
| 4–16 | two kernels (4), (8, 16) | a kernel per count (4), (8), (16) | 3 (DS512 at 16 tokens: 5) | router + quantization; one launch runs the gate-up of every route, no separate activation launch; down + combine. Inter 512 at 16 tokens: a front launch, a selection launch that writes masks, two grouped-expert launches, a combine |
| 32–256 | a kernel per count (32), (128), (256) | one kernel (256) | 5 (DS512: 6) | router, quantization and counter initialization in the first launches; a selection launch on relaxed atomics; gate-up over expert jobs (16 to 128 rows at DS512; DS256 at 128 and 256 tokens derives the job from the counts); down; a last, atomic-free launch reduces the parts and writes `out` |
| 1024–4192 | two kernels (1024, 2048, 3072), (4096, 4192) | a kernel per count (1024), (2048), (3072), (4096), (4192) | 6 | full-K router + quantization; route selection; a prepare launch writes job lists (at DS512, 1024–3072 tokens: full 128-row tiles first, then tails by falling height down to 16 rows, the mix changing with the routing); grouped gate-up; routed down; the shared expert's down, which writes `out` |

- **Techniques.** Router tiles and the MXFP4 quantization of the hidden states share one launch, split by program id
  ([moe](../moe.md) §3). At 1–16 tokens every kernel but DS512 at 16 tokens selects experts in the kernel with a rank loop; at DS256, 2 tokens,
  each gate-up program recomputes its route's expert with rank + 1 argmax rounds instead of waiting for a top-k launch
  ([moe](../moe.md) §4, [../../moe/few_tokens.md](../../moe/few_tokens.md)). A gate-up N tile holds matching gate and up
  rows; in full-K gate-up designs such as DS256 at 256 tokens, SwiGLU and the MXFP4 requantization run in its epilogue ([moe](../moe.md) §7, §8). At DS512, 1–2 tokens, the down
  multiplies a token's routes (MFMA rows) by their experts' column bands and keeps the diagonal, weighted and summed in
  slot order in registers ([moe](../moe.md) §9). At 32–256 tokens, the gate-up programs of DS256 at 256 tokens scan the per-expert
  counts with `gl.associative_scan` and make the job descriptor wave-uniform with `v_readfirstlane_b32`
  ([gluon/wave_ops](../gluon/wave_ops.md) §6). MXFP4 operands go to `mfma_scaled` with e8m0 scales in
  `get_mfma_scale_layout` layouts; activations are packed with `v_cvt_scalef32_pk_fp4_f32` inline asm
  ([numerics](../numerics.md) §3).
- **Copy from.** The DS512 kernel for 1–2 tokens is a whole call in 3–4 launches. The selection recomputed per
  program (DS256, 2 tokens):

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
- **Resources** *(field, E2)*: 116 compiled-cache records, not distinct binaries (§1); `num_warps` 1 (57), 2 (8), 4 (51);
  `waves_per_eu` unset in all; VGPR 22–280, AGPRs in 2 (16, 24); LDS 0–74.8 KiB; no spills; scaled 16 × 16 × 128 MFMA in
  54, bf16 16 × 16 × 32 in 23, scaled 32 × 32 × 64 in 8.
- **Speedups** over the baseline above *(field, E6)*: 1.030 (DS256, 128 tokens) to 2.186 (DS256, 2 tokens) over
  the 24 sizes.

## 8. MiniMax-M2 MXFP4 MoE

MiniMax-M2's MoE layer on one rank: an fp32 router (sigmoid plus correction bias, top-8 of 256, renormalized), MXFP4 experts, no shared expert. MM256-FP4 holds all 256 experts on this rank at intermediate dim 512 (12 kernels); MM64-FP4 holds the 64 from an expert offset at intermediate dim 1536, routes to other experts adding nothing (11 kernels); 30 sizes. The baseline runs the router as `tgemm.mm` with its first operand cast to fp32, then `moe_fused_gate` and `fused_moe` at `per_1x32` *(field, E20)*.

| Tokens | MM256-FP4 | MM64-FP4 | Launches | Decomposition |
|---|---|---|---|---|
| 1–16 | three kernels (1), (2), (4, 8, 16) | three kernels (1, 2), (4), (8, 16) | 3 *(field, E27)* | router on fp32 VALU dots, no MFMA, over the full K of 3072 in every kernel but MM256-FP4 for 4–16 tokens; 492 sampled projections matched an exact sum *(field, E21)*; the last launch runs the down and the weighted combine *(field, E27)* |
| 32–256 | a kernel per count (32), (64), (128), (256) | two kernels (32, 64), (128, 256) | 5 *(field, E27)* | split-K router tiles, counter and job initialization and the hidden quantization in the first two launches; per-token top-8 with relaxed atomic tickets, a job opened every 16th ticket (every 64th at MM256-FP4, 128 and 256 tokens); gate-up over jobs (MM64-FP4 at 128–256 tokens: each program finds its job by scanning the counts); down; combine |
| 1024–16384 | five kernels (1024), (2048, 4096), (6144), (8192), (16384) | a kernel per count (1024), (2048), (4096), (6144), (8192), (16384) | 6–8 *(field, E27)* | router on fp32 MFMAs at MM256-FP4 and at MM64-FP4 for 1024 tokens, on bf16 MFMAs (three bf16 terms of the weight, checked on samples) in the other five MM64-FP4 kernels; 912 sampled projections matched an exact sum *(field, E21)*; at MM64-FP4 too the router scores all 256 experts. Then selection, a sort (tickets at MM256-FP4, wave ballots at MM64-FP4), gate-up with requantization, the down (one design in every MM256-FP4 kernel but the one for 2048–4096 tokens, which has its own) and the combine *(field, E27)* |

- **Techniques.** Job queues from relaxed atomics; a publish tensor built on one wave masks its copies so each route takes one ticket
  ([moe](../moe.md) §6). At MM64-FP4 a counting sort from wave ballots and popcounts
  ([gluon/wave_ops](../gluon/wave_ops.md) §6), atomic-free from 2048 tokens *(field, E2)*. The MM256-FP4 kernel for 16384 tokens replays a captured graph with new inputs
  correctly and keeps identical grids over its 6 eager launches at two input sets *(field, E24)*; other routings,
  worst-case sizing and the device-count exit are untested ([authoring](../authoring.md) §8).
  Router accuracy: on complete logits at MM64-FP4 for 2048 and 4096 tokens and MM256-FP4 for 1024, over four input
  distributions and eight seeds, the error against an fp64 matmul of the same inputs was no larger than a GPU fp32
  matmul's in L2 in 96 of 96 cases and in max-abs in 93 *(field, E28)*; at 13 more branches (the 1–16-token VALU and the
  32-token fp32 MFMA among them), on a seed-0 random and a cancellation input each, in L2 on 25 of the 26 inputs
  (the cancellation input of MM256-FP4 for 6144 tokens 1.00009×), in max-abs on 23 and element by element on 3; on two more inputs with a
  nonzero weight residual, the fp64 correction of MM64-FP4 for 6144 and for 8192 tokens ran and both errors were 0 *(field, E29)*. Finite
  fp32-level evidence: not every tile or split specialization, nor the same top-8 for every input.
- **Copy from.** The relaxed ticket of the select launch of MM256-FP4 for 16384 tokens and the scatter that turns it into a slot (shown in
  [moe](../moe.md) §6; checked with one count block, not that kernel's several); the gate-up fold of MM256-FP4 for 1 token, which packs
  several K chunks into the gate-up's 16 MFMA rows, their number set per token count ([gluon/mfma](../gluon/mfma.md) §8).
- **Resources** *(field, E2)*: 183 compiled kernels; `num_warps` 1 (35), 2 (7), 4 (139), 8 (2); `waves_per_eu` 3 in 4, 4 in 1;
  VGPR 9–244, no AGPRs; LDS 0–35 KiB; no spills; scaled 16 × 16 × 128 MFMA in 69, fp32 16 × 16 × 4 in 12, bf16
  16 × 16 × 32 in 5, fp32 32 × 32 × 2 in 3, scaled 32 × 32 × 64 in 1.
- **Speedups** over the baseline above *(field, E7)*: 1.178 (MM256-FP4, 256 tokens) to 3.633 (MM64-FP4, 1 token) over the 30 sizes.

## 9. MiniMax-M2 FP8 MoE

The same layer with e4m3 experts and fp32 scales per 128 × 128 block, both GEMM inputs quantized per 1 × 128 in the kernels: MM256-FP8 at intermediate dim 384 (5 kernels) and MM64-FP8 at intermediate dim 1536 (4 kernels), 28 sizes. The baseline runs the router as `tgemm.mm` with its first operand cast to fp32, then `moe_fused_gate` and `fused_moe` at `per_128x128` *(field, E20)*.

| Tokens | MM256-FP8 | MM64-FP8 | Launches | Decomposition |
|---|---|---|---|---|
| 1–16 | one kernel (1, 2, 4, 8, 16) | one kernel (1, 2, 4, 8, 16) | MM256-FP8 3; MM64-FP8 4 (1–4 tokens), 6 (8–16) *(field, E27)* | router on fp32 VALU, no MFMA; 180 sampled projections matched an exact sum *(field, E21)*; MM256-FP8 runs the down and the combine in one launch of 8-warp CTAs, one per (token, column tile) *(field, E27)* |
| 32–256 | three kernels (32, 64), (128), (256) | two kernels (32, 64), (128, 256) | 5–6 *(field, E27)* | router on fp32 MFMAs at MM256-FP8 for (32, 64) and (128) and at MM64-FP8 for (32, 64), on bf16 MFMAs at MM256-FP8 for (256) and at MM64-FP8 for (128, 256); 3456 sampled projections matched an exact sum, the inline bf16 terms not checked one by one *(field, E21)*; MM64-FP8 at 256 tokens launches a persistent gate-up on a 48 × 96 grid *(field, E27)* |
| 1024–16384 | one kernel (1024, 2048, 4096, 8192, 16384) | one kernel (the same five) | 6–8 *(field, E27)* | a prepare launch first (MM64-FP8, which prepares its inputs; MM256-FP8 from 8192 tokens, which prepares the router and quantizes) or the router first (MM256-FP8 up to 4096); router; selection and sort (MM256-FP8: launches over chunks of constexpr size; MM64-FP8: an atomic-free SWAR counting sort, a rank-count launch and a prefix scatter); gate-up with fused SwiGLU + fp8 quantization; down; combine |

- **Techniques.** One warp per route under `AMDMFMALayout(..., warps_per_cta=[8, 1, 1])`, the route as the batch
  dimension ([moe](../moe.md) §9); a counting sort with four 8-bit counters per 32-bit word and one
  `gl.associative_scan` ([moe](../moe.md) §6); inline asm in 6 kernels: `v_rcp_f32` in all six, `v_exp_f32` in five,
  `v_readlane_b32` in four, `s_ff1_i32_b64` in three, `v_max_f64` / `v_min_f64` only in MM64-FP8 for 128–256 tokens
  ([gluon/wave_ops](../gluon/wave_ops.md) §4, §6). The routers on bf16 MFMAs (§8 and here) split the fp32 weight into
  three bf16 terms rounded per kernel: RTZ, RTZ, RTZ with an fp64 correction of flagged residuals in the MM64-FP4 kernels for 2048, 4096,
  6144 and 8192 tokens; RTZ, RNE, RNE in the MM256-FP8 kernel for 1024–16384; RNE, RNE, RNE without the correction in the MM64-FP8 kernel for 1024–16384 ([numerics](../numerics.md)
  §5). Random and cancellation inputs leave no residual; in two inputs built to have one (MM64-FP4 for 6144 and for 8192 tokens), 16 flags and
  256 residual entries were set, and the correction moved those logits from 0 to 2^-23, the fp64 value *(field, E29)*.
  Three bf16 terms alone do not make a router fp32-accurate for every input.
- **Copy from.** The SWAR rank of every route among same-expert routes in a 512-route block (the MM64-FP8 kernel for 1024–16384 tokens):

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
  the kernel asserts `gl.static_assert(512 % TOPK == 0 and 512 // TOPK < 256)` (TOPK 3 fails it). Over 255 routes of
  one expert in a block carry into the next expert's counter: with 300, the neighbour's ranks came out one too high.
- **Resources** *(field, E2)*: 188 compiled kernels; `num_warps` 1 (73), 2 (13), 4 (96), 8 (6); `waves_per_eu` 1 in 4; VGPR
  14–327, AGPRs in 2 (35, 71); LDS 0–131.9 KiB; an 8-SGPR spill in one route-packing kernel; scaled 16 × 16 × 128 MFMA in 47,
  bf16 16 × 16 × 32 in 13, scaled 32 × 32 × 64 in 10, fp8 16 × 16 × 32 in 9, fp32 16 × 16 × 4 in 7.
- **Speedups** over the baseline above *(field, E8)*: 1.102 (MM64-FP8, 64 tokens) to 4.115 (MM256-FP8, 1 token) over the 28 sizes. Across §7–§9 the
  32–256-token sizes are the weakest band whatever the design, and inside one 1–16-token kernel the value falls as
  tokens grow: band ranges in [moe](../moe.md) §2, band medians in [tuning](../tuning.md) §14.

## 10. Qwen2-MoE: router, fp8 experts, fp8 router + experts

Qwen3.5-397B-A17B's MoE block at TP4: model dim 4096, 512 routed experts, top-10; the
experts ops add inter dim 256, one shared expert weighted by `sigmoid(hidden · shared_gate_weight)` (513 experts in
all) and e4m3 experts with fp32 128 × 128 block scales. The router + experts op computes in one call what the router op
followed by the experts op computes.

| Op | Tokens, a kernel each | Launches | Decomposition |
|---|---|---|---|
| router: bf16 logits, top-10 ids and softmax weights | 8, 16, 64, 128, 256, 2232, 4096 | 2 | a split-K projection, exactly 256 programs at 8–256 tokens, then per-row selection over the stored bf16 logits: top-10 with lower-id ties, softmax weights within 2.3e-8 of those 10 logits, outputs bitwise identical over 20 repeats at 8–256 tokens *(field, E25)* (whether the split sums round once is untested); at 2232 and 4096 a large-M GEMM with multi-row selection (read from source, untested) |
| fp8 experts: routing given, 513 experts, aiter's 16 × 16 shuffled weights | 8, 64, 128 | 3; 5; 5 | routing given; 8 tokens: per-route split-K gate-up, activation quantization, down with the ordered 11-route sum; 64 and 128: a route sort in LDS (by one CTA at 64), then gate-up, quantization, down, ordered reduce (at 64: persistent stages over device-counted tasks) |
| fp8 router + experts, the same weights | 1, 2, 4, 32 (written against a float64 reference; with an fp32 reference these token counts need no fp64) | 4; 4; 4; 7 | router logits in fp64 (VALU), rounded once to bf16; every gate-up CTA replays the top-k of its row; activation quantization; one CTA per output tile multiplies all of a token's routes in one MFMA and keeps the diagonal; at 32 tokens fp64 MFMA router partials, a counting sort, sparse gate-up, quantization, grouped down, ordered sum |

- **Techniques.** One `@gluon.constexpr_function` gives the host its grid and the kernel its tile geometry
  ([tuning](../tuning.md) §14); `llvm_fn_attrs` on the router's launches ([tuning](../tuning.md) §9); the top-k replayed per CTA ([moe](../moe.md) §4); fp64 products for a router,
  written against a float64 reference, on the VALU at 1–4 tokens and on fp64 MFMAs at 32 ([numerics](../numerics.md)
  §5); a one-CTA counting sort with `atomic_scatter_add` on an LDS descriptor ([moe](../moe.md) §6).
- **Copy from.** The shape configuration of the router for 256 tokens: one function that returns the split, expert tile, K tile, lookahead
  and routing group per row count to both the launch and the kernel (shown in [tuning](../tuning.md) §3).
- **Resources** *(field, E2)*: router 14 compiled kernels, `num_warps` 1, 4 or 16, VGPR 41–192, LDS 0–130 KiB, bf16 MFMA;
  experts 13 compiled kernels, 1–8 warps, `waves_per_eu` 2 or 4 in 3, VGPR 16–110, LDS 0–12 KiB, bf16 and fp16 MFMA; router +
  experts 19 compiled kernels, 1–4 warps, VGPR 19–128, LDS 0–16 KiB, scaled 16 × 16 × 128, fp64 16 × 16 × 4 and fp16 MFMA. No
  AGPRs, no spills.
- **Speedups** over what SGLang runs for the block *(field, E9)*: router 1.074 (8 tokens) to 1.967 (256); experts 1.076 (128) to 1.208 (8); router +
  experts 1.063 (32) to 1.172 (1).

## 11. DSA sparse MLA, one query token per sequence

Each token's 64 or 8 heads attend 2048 rows of an fp8 latent cache (576 values: 512 latent and 64 RoPE) named by
top-k indices, q in e4m3, `kv_b_proj` absorbed, bf16 out; 8 kernels, one size each (the later additions are 9 more 8-head kernels
and a 16-head op, §2). The baseline is SGLang's
`triton_sparse_mla_decode_splitk`. Triton side: [attention](../attention.md) §3, §4, §5; op card:
[../../attention/INDEX.md](../../attention/INDEX.md).

| Tokens (heads) | Kernels | Launches | Decomposition |
|---|---|---|---|
| 8 (64) | one | 3 | `q.to(torch.bfloat16)`; a producer of 256 programs × 4 warps; a merge of 512 programs |
| 12–32 (64) | a kernel per count (12), (16), (24), (32) | 2 | 192–256 programs of 8 warps per (token, 32- or 64-head tile, key split), 256 gathered rows per LDS tile (two tiles per program at 32 tokens, which has 4 splits), QK on `mfma_scaled` e4m3 with unit scales; a merge launch |
| 32, 128 (8) | a kernel per count (32), (128) | 3 | `q.to`; a producer; a merge |
| 256 (8) | one | 2 | `q.to`; one workgroup per token over all 2048 keys, no merge |

- **Techniques.** Head tile and split count follow the token count; at 64 heads the grid is 192–256 programs, one
  LDS-heavy workgroup per CU ([../../attention/split_kv.md](../../attention/split_kv.md) §4, §5).
  Partials merged by a separate launch in a fixed order, scaled by `exp(m_s - max)` with a guarded `1 / den`
  (bitwise output, [attention](../attention.md) §2); enough tokens to fill the machine drop the merge. Four kernels
  (64 heads at 8 tokens; 8 heads at 32, 128 and 256) convert `q` in `run`, one more kernel per call.
- **Copy from.** The 64-head host of the 24-token kernel (shown in [attention](../attention.md) §4): head tile and split count
  from the row count, the partial records sized from them, the buffer path gated on the cache's size.
- **Resources** *(field, E2)*: 15 compiled kernels; `num_warps` 1 (3), 2 (2), 4 (6), 8 (4); `waves_per_eu` 1 in 2, 2 in 3;
  VGPR 23–228, AGPRs in 1 (22); LDS 0–160 KiB; no spills; bf16 32 × 32 × 16 MFMA in 4, scaled 32 × 32 × 64 in 4, bf16
  16 × 16 × 32 in 3, scaled 16 × 16 × 128 in 1, fp16 32 × 32 × 16 in 1.
- **Speedups** over the baseline above *(field, E10)*: 1.030 (64 heads, 8 tokens) to 2.188 (64 heads, 24 tokens).

## 12. Qwen3.5 gated attention, one query token per sequence

One token per row, 8 query heads on 1 KV head of dim 256: Q and K RMS-normalized, mRoPE on 64 channels, K and V written
to the e4m3 paged cache, attention over the row's slots, `sigmoid(gate)`, then e4m3 per 128 channels with fp32 scales; 9
kernels (three source files each: the entry, the prologue, the attention), one size each, 1–256 rows. The baseline chains SGLang's fused QK
norm, Triton mRoPE and FP8 KV write, aiter's `paged_attention_ragged`, `fused_sigmoid_mul` and aiter's per-1 × 128
quantization. Triton side: [attention](../attention.md) §6, §7 and [gated-attention launches](../attention/gated_attention_launches.md) §1, §2.

| Rows | Kernels | Decomposition (3 launches: prologue, attention, finish) |
|---|---|---|
| 1–2 | a kernel per count (1), (2) | one CTA per (row, head, split), 32 splits, FMA dot products, no MFMA |
| 4 | one | one program per (row, split), 4 × 32, all 8 query heads in it on MFMA |
| 8 | one | MFMA partials, async copies into 34,816 bytes of LDS |
| 16–64 | a kernel per count (16), (32), (64) | a (row, split) grid, rows × splits = 512 (32, 16 and 8 splits), tiles dealt to the splits cyclically; K and V gathered into registers, V through a padded LDS layout; a 16-row head tile at 16 splits, else 8 |
| 128–256 | a kernel per count (128), (256) | a 1-D grid of 4 and 2 splits (rows × splits = 512), one contiguous chunk per split, async copies into LDS |

- **Techniques.** The prologue is a per-token task grid of one-wave CTAs (K norm, RoPE and cache write; V cache write; Q
  heads and the gate copy; the kernels for 1, 2, 8, 32 and 128 rows give the cache-write tasks the low program ids). The
  attention launch is flash-decoding ([../../attention/split_kv.md](../../attention/split_kv.md) §5): per
  split an unnormalized numerator with its (max, sum); the finish merges in a fixed order, rounds to bf16, applies the
  gate and quantizes, bit for bit as emulated in the 32-row kernel ([numerics](../numerics.md) §2). At 256 rows the 1-D
  grid is a bijection whose blocks of 128 CTAs hold both splits of 64 rows. Enumerated for rows 1–1024 and 2,
  3, 4, 8, 16 or 32 splits, the map is a bijection exactly when the row count is a multiple of 64, each block of
  64 × SPLITS consecutive CTAs then covering every split of the same 64 rows (6144 grids; no proof for other split
  counts) *(field, E26)*. Every split writes a record, inactive
  ones neutral, and a replayed graph matched the eager call (32-row kernel). `run` is two calls, one into the prologue module and one into the attention module.
- **Copy from.** The split-to-row map of the 256-row kernel:

  ```python
  # Flatten the launch into groups containing all splits of 64 rows.
  cta = gl.program_id(0)
  row = cta % 64 + cta // (64 * SPLITS) * 64
  split = cta // 64 % SPLITS
  ```

  With one split the map is a bijection at every row count tested (1024 more grids), so the multiple-of-64 condition
  holds for 2 splits or more. At 96 rows and 2 splits it misses 32 (row, split) pairs, launches 32 CTAs out of range
  and reaches row 127; at 256 rows it is a bijection. The source expressions of the 256- and 128-row kernels matched the
  enumeration, and five sample grids ran as enumerated *(field, E26)*. This does not widen what the kernels accept: the 128- and 256-row kernels assert `rows in (128, 256)`, and at 16–64 rows the kernels launch a
  (row, split) grid instead.
- **Resources** *(field, E2)*: 27 compiled kernels; `num_warps` 1 (9), 2 (1), 4 (17); `waves_per_eu` 2 in 2, 3 in 1; VGPR
  25–192, AGPRs in 1 (63); LDS 0–70 KiB; no spills; bf16 16 × 16 × 32 MFMA in 7, fp16 in 1.
- **Speedups** over the baseline above: 1.066 (256 rows) to 2.182 (1 row); a later run of the ruler read 1.085 to 2.221
  *(field, E11)*.

## 13. GatedDeltaNet, one token per sequence step

Per row: a width-4 causal convolution over the projected Q, K, V with the conv history shifted in place, the gates, the gated delta-rule update of a 128 × 128 fp32 state per (slot, value head), a gated RMSNorm and e4m3 per 128 with fp32 scales. Qwen3.5 (16 value heads): 8 kernels, 1–256 rows, plus one with bf16 output (64 rows); Qwen3-Next (8 value heads, grouped projections): 9 kernels over 9 sizes, one of them the template, table-driven and written for all 9. After the library's growth these are 9, 12 and 14 kernels (§2). The baseline chains a conv update, the recurrence, the norm and the quantization *(field, E20)*. No kernel uses MFMA *(field, E2)*. Triton side: [gdn](../gdn.md) §2–6.

| Model, rows | Kernels | Launches | Decomposition |
|---|---|---|---|
| Qwen3.5, 1–4 | a kernel per count (1), (2), (4) | 2 | each head's 128 value rows split over CTAs, 256 at 1, 2 and 4 rows; then a launch of 40 programs per row; the conv history shifted in place |
| Qwen3.5, 8–256 | a kernel per count (8), (16), (32), (64, bf16 output), (128), (256) | 2 | convolution with exclusive channel ownership, history shifted in place; one CTA per (row, head) owns the state, 4 waves at 8–32 rows, 16 from 64, the norm (and quantization) in its epilogue |
| Qwen3-Next, 1–8 | three kernels (1), (2), (4, 8) | 2 | producers of one or two waves per (row, value head or head pair, block of value rows), then a finisher |
| Qwen3-Next, 16–256 | a kernel per count (16), (32), (64); a one-owner kernel (128, 256) and a kernel with replicated lanes behind a barrier (128, 256) | 1 | one workgroup per (row, key head) owns the conv history and both value heads' states |

- **Techniques.** In the Qwen3.5 kernels for 1, 2 and 4 rows, moving the history shift before the reader launch failed 20 of 20 trials each; shifting inside the reader failed 14, 5 and 17 of 20, while the kernels' own shift in the second launch passed 20 of 20 *(field, E22)* ([gdn](../gdn.md) §3): a race need not fail every trial. The one-owner kernels, Qwen3.5 at 8 rows and Qwen3-Next at 128, passed 40 of 40 each; the Qwen3-Next kernel with replicated lanes (128) passed 100 of 100 with its barrier and failed 100 of 100 without it. A 4-wave kernel with a delayed reader was exact 40 of 40 with one owner or a barrier and mismatched 40 of 40 with neither; a probe without the delay had shown no race *(field, E22)*. The table-driven kernel keeps one configuration row per row count and raises on an unlisted count. Inline asm supplies `v_med3_f32` (all 8 Qwen3.5 kernels, the ±448 clamp; `gl.clamp` emits the same instruction) and, in Qwen3-Next, `v_rcp_f32`, `v_exp_f32`, `v_log_f32`, `v_div_fixup_f32`, `v_pk_fma_f32`, `v_cvt_scalef32_pk_fp8_f32`, `s_setreg_imm32_b32` and DPP wave reductions (`v_max_f32` / `v_add_f32` with `row_shr`, then `v_readlane_b32`) ([gluon/wave_ops](../gluon/wave_ops.md) §3, §4, §6, §7).
- **Copy from.** The exclusive-owner shift of the Qwen3-Next one-owner kernel for 128 and 256 rows: one 1-D tile over the key group's 512 conv channels with `gl.static_assert(NW * 64 <= conv_width)`, so no lane or wave holds a copy and each thread shifts the history it read, with no barrier:

  ```python
  conv_layout: gl.constexpr = gl.BlockedLayout([1], [64], [NW], [0])
  # This packed tile has no replicated lanes/waves. Every history read and
  # write has one owner, even though Q/K are subsequently shared by two heads.
  c = gl.arange(0, conv_width, layout=conv_layout)
  ```

  The other Qwen3-Next design for 128 and 256 rows keeps replicated lanes and orders the reads before the shift with `gl.barrier()`:

  ```python
  h2 = gl.load(history + base + 2).to(gl.float32)  # replicated lanes: several threads read the same entry
  acc = gl.fma(x, w3, acc)
  gl.barrier()  # every wave has read the history before any wave shifts it
  if live:
      gl.store(history + base, h1)
      gl.store(history + base + 1, h2)
      gl.store(history + base + 2, x)
  ```

  Not this: the same stores with replicated lanes and no barrier, which failed 100 of 100 trials *(field, E22)*. The barrier design
  measured the slowest of the three 128- and 256-row designs, 1.466 and 1.151 against the one-owner kernel's 1.782 and
  1.425 *(field, E13)*; the designs differ in more than the shift ([gdn](../gdn.md) §2).
- **Resources** *(field, E2)*: Qwen3.5 18 compiled kernels, `num_warps` 1, 2, 4 or 16, `waves_per_eu` 1, 6 or 8 in 5, VGPR
  16–146, LDS 0–2 KiB; Qwen3-Next 27 compiled kernels, 1–8 warps, VGPR 17–208, LDS 0–2.5 KiB. No AGPRs, no spills.
- **Speedups** over the baseline above: Qwen3.5 1.282 (256 rows) to 2.507 (2 rows), the bf16-output kernel 1.461 *(field, E12)*.
  Qwen3-Next, in a later run of the ruler: its 9 kernels 1.151 (replicated lanes, 256 rows) to 3.350 (32 rows), the best
  kernel per size 1.457 (256 rows) to 3.350, and the per-size kernels above the table-driven one at 7 of 9 sizes (equal at 4 rows,
  the table-driven one ahead at 256) *(field, E13)*.

## 14. GatedDeltaNet, chunked sequences of many tokens

Packed sequences of the same layer: Qwen3.5, 15 kernels over 21 sizes (512–16384 tokens in 1–8 sequences); Qwen3-Next, 11 kernels over 13 sizes (1024–16384 in 1–16). Source files: the kernel, a module that zeroes fresh state slots and, in 9 Qwen3.5 kernels, a module that repacks the projections. The baseline chains `causal_conv1d_fn`, `fused_gdn_gating`, `chunk_gated_delta_rule`, the norm and the quantization (`per_token_group_quant_fp8` in Qwen3.5) *(field, E20)*. Triton side: [gdn](../gdn.md) §7–12.

Every call zeroes the state slots of sequences without an initial state (tested on the Qwen3.5 kernel for 15361–16384 tokens in 8 sequences), and 9 Qwen3.5 kernels repack the projections into grouped order. In launch order at all 34 sizes: prepare, chunk factors, summaries for long sequences, one sequential launch, a parallel output epilogue; 5–8 launches *(field, E27)*. At the 21 Qwen3.5 sizes the conv history left after the call was the last three entries of the old history (zeros when fresh) followed by the sequence's inputs, so a sequence shorter than three tokens keeps old or zero entries *(field, E27)*. Committing the history early in a Qwen3.5 kernel (2049 tokens in 2 sequences) and a Qwen3-Next kernel (3072 in 3, 3072 in 2) changed 12,693–29,318 fp8 values and failed accuracy on all three inputs in each of two executions, while the unchanged kernels passed and the final conv and delta states were unchanged: only the outputs expose the early read *(field, E22)*.

Kernels with their sizes as tokens/sequences, then the op's stated maximum sequence length after the semicolon (∅: none stated).

| Sequences | Qwen3.5 | Qwen3-Next | Sequential pass |
|---|---|---|---|
| up to about 2K tokens per sequence | five kernels: (512/2; 256), (512/1, 1024/2; 512), (1024/1; 1024), (1025/1; 1025), (2048/1; ∅) | one kernel (1024/1, 2048/2; ∅) | 64-token chunks, chunk-start checkpoints, output in parallel per (head, chunk) |
| several sequences | four kernels: (2049/2, 4096/2), (4097/3, 8192/4), (8193/5, 15360/8), (15361/8, 16384/8), all ∅ | two twin pairs: (3072/3, 4096/4, 8192/8, 12288/12, 16384/16; ∅), (13312/13, 15360/15; ∅) | per (sequence, head, value-row strip); the Qwen3.5 pass has 160–256 programs (at 8 sequences: 16 heads × 2 strips of 64 value rows × 8 sequences = 256; at 5 sequences: 160) |
| long sequences, 4096–16384 tokens in 1–4 | maps from an identity seed: (8192/1; 8192), (10000/1; 10000); reverse summaries: (5000/1, 15000/3; 5000), (16384/2; 8192), (16384/2; 10000), (16384/1; 16384) | three twin pairs: (4096/1; 4096), (5120/1, 10240/2; 5120), (16384/4; 5120) | a parallel summary launch per run of chunks (320–864 CTAs), one sequential launch (64–384), a parallel replay of the chunks (320–1024) |

- **Techniques.** bf16 MFMA for the Gram products of the bf16-rounded Q and K, fp32 MFMA for the state algebra
  ([gdn](../gdn.md) §9, [numerics](../numerics.md) §5). The Qwen3.5 kernel for 15361–16384 tokens in 8 sequences permutes K inside every 16-block of both operands
  so that an fp32 accumulator becomes the next MFMA's A operand by a trivial `convert_layout` ([gdn](../gdn.md)
  §11, [gluon/layouts](../gluon/layouts.md) §7). Every Qwen3-Next kernel but the one for 1024–2048 tokens has a twin, the same source with the token count
  and the sequence count (in the 13312–15360-token pair also a row total) as constexprs in one and runtime ints in the other: the runtime twin can reuse one binary only across sizes
  whose complete JIT specialization key is the same. Measured, the twins came within 2.4% of each other,
  either way *(field, E15)*.
- **Copy from.** The per-sequence pass of the Qwen3.5 kernel for 15361–16384 tokens in 8 sequences (the grid of head × 2 strips × sequence above), and the launch that zeroes fresh state slots, one program per (sequence, block of the slot):

  ```python
  @gluon.jit
  def _initialize_fresh(
      STATE, INDICES, INITIAL, WIDTH: gl.constexpr, BLOCK: gl.constexpr
  ):
      seq = gl.program_id(0)
      if not gl.load(INITIAL + seq):  # this sequence brought no initial state
          slot = gl.load(INDICES + seq)
          offset = gl.program_id(1) * BLOCK + gl.arange(
              0, BLOCK, gl.BlockedLayout([1], [64], [4], [0])
          )
          gl.store(STATE + slot * WIDTH + offset, 0.0, offset < WIDTH)
  ```
- **Resources** *(field, E2)*: Qwen3.5 123 compiled kernels, `num_warps` 1, 2, 4 or 8, `waves_per_eu` 2, 4 or 5 in 15, VGPR
  3–256, no AGPRs, LDS 0–32 KiB, spills in 2 (64 VGPRs in a kernel that builds paired chunk tables, 1 SGPR); Qwen3-Next 117
  compiled kernels, 1–4 warps, VGPR 2–316, AGPRs in 5 (20–60), LDS 0–32 KiB, no spills. fp32 16 × 16 × 4 MFMA in 57 compiled kernels
  of each family, bf16 16 × 16 × 32 in 24 of each; fp32 32 × 32 × 2 in 4 (Qwen3.5) and 2 (Qwen3-Next), bf16
  32 × 32 × 16 in 2 (Qwen3.5). All 26 kernels hold both bf16 16 × 16 × 32 and fp32 16 × 16 × 4 MFMAs.
- **Speedups** over the baseline above: Qwen3.5 0.997 (8193 tokens in 5 sequences) to 2.638 (512 in 1); 0.997 and 1.029
  (4097 in 3) are below the bar of these runs ([tuning](../tuning.md) §1) *(field, E14)*. Qwen3-Next, in a later
  run of the ruler: the best kernel per size 1.233 (13312 tokens in 13 sequences) to 1.741
  (5120 in 1) *(field, E15)*.

## Evidence

The E lines were measured on the library's 154 kernels unless a line says otherwise; "the ruler" is [tuning](../tuning.md) §1's, its kernel durations read from Kineto.

- E1 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1 (router split-K: N=3), the ruler on one GPU, copy and original alternated, 4 rounds per arm] copies of library kernels with one constant changed, ratio of the copy's speedup to the original's: gated attention, 32 rows, KV splits 16 → 8 0.63, 16 → 32 0.95 (the head tile following the split count, 16 → 8 rows in both); Qwen2 router, 256 tokens, split-K 4 → 2 0.75 (9.91 → 13.29 µs a call), 4 → 8 0.97 (9.97 → 10.25 µs, all 4 copy runs below all 4 originals), repeated at 0.748 / 0.967 and 0.749 / 0.961; Qwen2 router, 8 tokens, hidden tile by `gl.load` converted through LDS instead of async copies (the weight's load layout changing with it): 8.447 → 8.45 µs, ratio 0.99; fp8 GEMM N 3072, M 16, K groups prefetched 5 → 3 0.87, 5 → 4 1.00, 5 → 7 1.00 (time ratio 7/5 0.9998 and 0.9968 in two runs); DSA 24 tokens, KV splits 8 → 4 alone: the build failed, a `CompilationError` in the split kernel (`NameError: cache_shared is not defined`) in all 4; MM256-FP4, 1 token, gate fold 4 → 1 alone: the kernel crashed, exit −11, in all 4
- E2 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, every kernel built and run once, each with its own `TRITON_CACHE_DIR`, on 4 GPUs; the cached `.amdgcn` and `.json` read] `num_warps`, `waves_per_eu`, VGPR (`.vgpr_count`, AGPRs included), AGPR, LDS, spills, MFMA opcodes and the count of every opcode (loads, stores and atomics included, buffer loads into LDS apart) of every kernel function of the library (a `@gluon.jit` or `@triton.jit` def of a kernel's sources) compiled while running each of the library's 154 kernels once: 897 compiled-cache records among 1709 compiled kernels, one per compilation, not deduplicated; N 3072 fp8 GEMM LDS 9, 9, 12, 36, 48, 49 KiB at M 1, 2, 4, 8, 16, 32; N 7168 at M 16, 32, 64 64.0, 66.1, 72.0 KiB; MM64-FP4 at 2048, 4096, 6144, 8192, 16384 tokens: 0 atomic instructions (global, buffer, flat, ds), popcounts (`s_bcnt`) 4, 2, 4, 8, 8; every GDN prefill kernel (26) holds `v_mfma_f32_16x16x32_bf16` and `v_mfma_f32_16x16x4_f32`; 45 GDN decode compiled kernels, 0 MFMA; RMSNorm loads with the nt bit, stores sc0 sc1 only, 16 B of LDS (all 154 Triton kernels)
- E3 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, a reference run of the ruler, stored speedup against the op's baseline] fp8 GEMM: lowest 1.114 at N 7168, M 128, highest 1.840 at N 7168, M 8, in two runs of the ruler (fp8 block-scaled GEMM, N 7168 × K 2048)
- E4 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, a reference run of the ruler, as in E3] bf16 GEMM at M ≤ 64: lowest 1.048 at N 2624 × K 6144, M 2, highest 1.292 at N 6144 × K 2048, M 8, both in one run
- E5 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, a reference run of the ruler, as in E3] RMSNorm 1.006 at 4 tokens to 1.212 at 2048, 1.006–1.024 at 3–64 tokens; add + RMSNorm 1.024 at 8 tokens to 1.093 at 256, in another run; a later run of the ruler: RMSNorm 1.023–1.029 at 3–18 tokens and 1.040 at 64 (hidden size 6144)
- E6 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, a reference run of the ruler, as in E3] DeepSeek-V2 MXFP4 MoE: lowest 1.030 at DS256, 128 tokens, highest 2.186 at DS256, 2 tokens, in two runs of the ruler (model dim 6144, 256 routed experts, aiter's 16 × 16 shuffled weights)
- E7 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, a reference run of the ruler, as in E3] MiniMax-M2 MXFP4 MoE, both in one run: lowest 1.178 at MM256-FP4 (intermediate dim 512), 256 tokens, highest 3.633 at MM64-FP4 (intermediate dim 1536), 1 token (model dim 3072, 256 routed experts)
- E8 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, a reference run of the ruler, as in E3] MiniMax-M2 FP8 MoE: lowest 1.102 at MM64-FP8 (intermediate dim 1536), 64 tokens, highest 4.115 at MM256-FP8 (intermediate dim 384), 1 token, in two runs of the ruler (model dim 3072, 256 routed experts)
- E9 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, a reference run of the ruler, as in E3] Qwen2-MoE: router 1.074 at 8 tokens and 1.967 at 256, in one run; fp8 experts 1.076 at 128 and 1.208 at 8, in two other runs; router + experts 1.063 at 32 and 1.172 at 1, in the router's run (model dim 4096, 512 routed experts, top-10)
- E10 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, a reference run of the ruler, as in E3] DSA sparse MLA decode, both in one run: lowest 1.030 at 64 heads, 8 tokens, highest 2.188 at 64 heads, 24 tokens (64 heads, cache rows of 576 values, top-k 2048)
- E11 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, a reference run of the ruler, as in E3] Qwen3.5 gated attention decode: in one run lowest 1.066 at 256 rows, highest 2.182 at 1 row; a later run of the ruler: 1.085 and 2.221 (QKV dim 4608, 1 KV head of dim 256, fp8 group-quantized output)
- E12 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, a reference run of the ruler, as in E3] Qwen3.5 GDN decode: lowest 1.282 at 256 rows, highest 2.507 at 2 rows, in two runs of the ruler; bf16-output op 1.461 at 64 rows, in the second of them (16 value heads, key and value dim 128, conv dim 3072)
- E13 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, a reference run of the ruler, as in E3] the 9 Qwen3-Next GDN decode kernels, one later run of the ruler: 1.151 (the replicated-lane kernel, 256 rows) to 3.350 (the 32-row kernel); best kernel per size 1.457 at 256 rows (the table-driven kernel) to 3.350; per-size kernel against the table-driven one at 1, 2, 4, 8, 16, 32, 64 rows: 2.443 / 2.390, 2.538 / 2.450, 2.370 / 2.370, 2.506 / 2.479, 3.222 / 3.101, 3.350 / 3.091, 2.462 / 2.401; at 128 and 256 rows the one-owner kernel 1.782 and 1.425, the replicated-lane kernel 1.466 and 1.151, the table-driven kernel 1.755 and 1.457 (8 value heads, key and value dim 128, conv dim 2048)
- E14 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, a reference run of the ruler, as in E3] Qwen3.5 GDN prefill: lowest 0.997 at 8193 tokens in 5 sequences and 1.029 at 4097 in 3, both in one run; highest 2.638 at 512 tokens in 1 (maximum sequence length 512), in another run (16 value heads, conv dim 3072)
- E15 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, a reference run of the ruler, as in E3] the 11 Qwen3-Next GDN prefill kernels, one later run of the ruler: best kernel per size lowest 1.233 at 13312 tokens in 13 sequences (the runtime twin), highest 1.741 at 5120 in 1 (maximum sequence length 5120, the runtime twin); twins, constexpr / runtime: 4096 tokens in 1 sequence 1.718 / 1.719; 5120 in 1 and 10240 in 2 1.727 / 1.741 and 1.308 / 1.311; 3072 in 3, 4096 in 4, 8192 in 8, 12288 in 12 and 16384 in 16 1.592 / 1.554, 1.504 / 1.501, 1.490 / 1.484, 1.288 / 1.282, 1.463 / 1.460; 16384 in 4 1.294 / 1.283; 13312 in 13 and 15360 in 15 1.232 / 1.233 and 1.328 / 1.329 (8 value heads, conv dim 2048)
- E16 [2026-11, gfx950 (MI355X), FlyDSL ? and HIP kernels beside Triton 3.8.0 ones, N=1, a reference run of the ruler, as in E3] one run: the FlyDSL bf16 GEMM N 6144 × K 2048 M 1 1.191, M 2, 4, 8, 16, 32, 64 1.045, 0.403, 0.445, 0.392, 0.361, 0.360, M 128–4096 0.450–0.864; the FlyDSL bf16 GEMM N 2624 × K 6144 M 1 1.134, M 2, 4, 8, 16, 32, 64 0.420, 0.415, 0.419, 0.396, 0.410, 0.553, M 128–4096 0.623–0.998; the HIP add + RMSNorm (hidden size 6144) 1.038 at 32 tokens, 1.067 at 128, 1.040–1.072 at 512–8192; the Triton add + RMSNorm 1.035 at 32, 1.062 at 128
- E17 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2, probe kernels reading one 4.5 GiB uint8 tensor with `buffer_load` and with `gl.load`] `buffer_load` returns the data up to byte 2147483645 and 0 from byte 2147483646 = 2^31 − 2 on (also at 2^31, 2^31 + 64, 2^31 + 4096 and 2^32 − 1), where `gl.load` returns the data; the descriptor is built with `num_records` 0x7ffffffe (a probe, not a library kernel)
- E18 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3, a Gluon probe of 2048 programs × 4 warps reading bf16 chunks with `buffer_load`, a 1 GiB read before each launch, median of 9 launches per round, 3 rounds alternating consecutive ids and ids congruent mod 8] 8 readers per chunk, consecutive over congruent time per round: 64 KB 1.76–1.82 (21.3–21.6 against 11.7–12.1 µs), 512 KB 2.07–2.17 (154.4–160.5 against 74.1–76.8 µs); one reader per chunk (the congruent map a stride-8 permutation): median ratio over the rounds 1.005, 1.012 and 0.980 by run, 25.2–26.0 against 25.4–25.9 µs (a probe, not a library kernel)
- E19 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1 GPU run; CPU under Triton 3.7.0, compile only] a copy of the kernel requiring Triton `>=3.9` under 3.8.0, the requirement checked before the build: the run stops with no matching toolchain, no traceback, no size timed and no copy built, and M = 6144 stays with the original kernel; requiring `>=3.8`: one size timed, one build; under 3.7.0 a `>=3.8` requirement is reported unmet, and without it the kernel fails to compile on a missing helper; no GPU run under 3.7 (fp8 GEMM N 3072 / K 1536, M = 6144)
- E20 [2026-11, CPU (target gfx950), Triton 3.8.0, N=1, each kernel's `run` replayed on the host with its launches recorded, no GPU] the 24 fp8 GEMM sizes one launch each, the N 3072 kernel for M = 6144 rejecting independently strided A and B; at the 89 MoE sizes of 58 kernels the gate-up launch, its grid and the source of its W1; the baselines' calls: MiniMax-M2 routers `tgemm.mm` with the first operand cast to fp32, then `moe_fused_gate` and `fused_moe` at `per_1x32` (MXFP4) or `per_128x128` (FP8); GDN decode a conv update, the recurrence, the norm and the quantization; GDN prefill `causal_conv1d_fn`, `fused_gdn_gating`, `chunk_gated_delta_rule`, the norm and the quantization (`per_token_group_quant_fp8` in Qwen3.5)
- E21 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, one input set per size, the router launch's projection sampled against `math.fsum` with a tolerance per sample, its ISA read; not timed] 58 MiniMax-M2 router projections (hidden bf16 [M, 3072], weight fp32 [256, 3072], M from 1 to 16384): all 8520 sampled comparisons within tolerance, none non-finite; by subset (overlapping): MXFP4 1–16 tokens 10 sizes / 492 comparisons, MXFP4 1024–16384 12 / 912, FP8 1–16 10 / 180, FP8 32–256 8 / 3456, bf16-MFMA routers 18 / 1896; no MFMA in the routers at 1–16 tokens; bf16 MFMA in the MM64-FP4 kernels for 2048, 4096, 6144, 8192 and 16384 tokens, the MM256-FP8 kernels for 256 and for 1024–16384, and the MM64-FP8 kernels for 128–256 and for 1024–16384; fp32 MFMA in every other router above 16 tokens; the three bf16 terms of the MM64-FP4 routers checked on samples, the FP8 routers' inline terms not (MiniMax-M2 MoE kernels)
- E22 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2 executions, 20–100 trials each within an execution] Qwen3.5 GDN decode at 1, 2 and 4 rows: the kernels as they are 20/20 pass each, history shift before the reader launch 20/20 fail each, shift inside the reader 14/20, 5/20, 17/20 fail; the one-owner kernels, Qwen3.5 at 8 rows and Qwen3-Next at 128, 40/40 pass; the Qwen3-Next replicated-lane kernel at 128 rows 100/100 pass with its barrier, 100/100 fail without; a 4-wave kernel with a delayed reader: replicated without a barrier 40/40 mismatch, with a barrier or one owner 40/40 exact (an earlier probe without the delay showed no race); GDN prefill, a Qwen3.5 kernel at 2049 tokens / 2 sequences and a Qwen3-Next kernel at 3072/3 and 3072/2: history committed early fails all three inputs in each execution, 12693–29318 fp8 values changed, final conv and delta states unchanged; the unchanged kernels pass
- E23 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1 run, the ruler with a separate compile cache per arm, 4 balanced AB/BA pairs per size (12 at N 3072 M 1 and M 16 and at N 7168 M 2), 95% t interval of the paired mean ratio] B is A with only the tile order changed: 12 sizes, 58 kernels compiled, no accuracy finding; mean B/A and intervals per size as tabled in §4; N 3072 M 256 B/A 0.97012 (6.0875 → 5.906 µs, interval 0.9689–0.9713); N 7168 M 64 B/A 0.99305 (0.9856–1.0005), inside ±1.5%; N 3072 M 1 1.00434 (0.9895–1.0196) and M 16 1.00803 (0.9953–1.0211), unresolved against ±1.5% and ±1.3%; N 7168 M 2 1.02350 (1.0101–1.0372); tile maps under `pid % 8` estimated on CPU, no L2 counter read (fp8 GEMM and DeepSeek-V2 MoE kernels)
- E24 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2, each kernel captured into a CUDA graph, replayed and checked for accuracy; eager launches recorded by the profiler] MM256-FP4 for 16384, for 8192 and for 2048–4096 tokens and MM64-FP8 for 1024–16384: captured graphs replayed with new inputs, accurate; eager grids identical at two input sets, 6, 6, 6 and 8 launches; other routings, worst-case sizing and the device early exit not tested
- E25 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3, the library's routers run 20 times on equal inputs with ties in every row] Qwen2-MoE routers at 8–256 tokens (five kernels): projection grid 256 programs; outputs bitwise identical over the 20 repeats; ids the top-10 of the stored bf16 logits, lower id first on ties; softmax weights within 2.3e-8 of those 10 logits; 2232 and 4096 tokens and once-rounded split sums not tested
- E26 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, the map enumerated on CPU, five sample grids run on the GPU] the 256-row kernel's split-to-row map over rows 1–1024 and SPLITS 1, 2, 3, 4, 8, 16, 32 (7168 grids): for SPLITS ≥ 2 a bijection exactly at multiples of 64, each block of 64 × SPLITS consecutive CTAs covering every split of the same 64 rows; for SPLITS = 1 a bijection at every row count; at 96 rows and 2 splits 32 pairs missed, 32 CTAs out of range, row 127 reached; the expressions of the 256- and 128-row kernels agree with the enumeration (Qwen3.5 gated attention decode)
- E27 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, one call per size with its device launches recorded by the profiler] MiniMax-M2 MXFP4: 3 launches at the 10 sizes of 1–16 tokens, the last a down with the weighted combine, the router over K = 3072 in every kernel but MM256-FP4 for 4–16 tokens; 5 at 32–256 tokens; 6–8 at 1024–16384, the down one design in every MM256-FP4 kernel but the one for 2048–4096 tokens, which has its own; MiniMax-M2 FP8: MM256-FP8 3 at 1–16 tokens, its down-and-combine launch 8 warps on a (token, column tile) grid, MM64-FP8 4 at 1–4 and 6 at 8–16, 5–6 at 32–256 with MM64-FP8 at 256 launching a persistent gate-up on 48 × 96 programs, MM256-FP8 6 at 1024–4096 and 7 at 8192–16384, MM64-FP8 8; GDN prefill at all 34 sizes 5–8 launches in the order prepare, chunk factors, summaries for long sequences, one sequential launch, epilogue; at the 21 Qwen3.5 sizes the final conv history the last three entries of the old history (zeros when fresh) followed by the inputs (MoE and GDN prefill kernels)
- E28 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1 per input, complete router logits of the library kernel, of a GPU fp32 matmul (IEEE fp32 MFMA, no TF32) and of an fp64 matmul on the same inputs] MM64-FP4 at 2048 and at 4096 tokens and MM256-FP4 at 1024, 4 input distributions × 8 seeds each: the kernel's L2 error against fp64 no larger than the fp32 matmul's in 96 of 96 inputs, its max-abs error in 93 (MiniMax-M2 MXFP4 MoE kernels)
- E29 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1 per input, as in E28] 13 router branches at selected sizes (among them the VALU routers of MM256-FP8 at 1, 2, 4 and 16 tokens, the fp32 MFMA of MM256-FP8 at 32 and of MM256-FP4 for 2048–4096, for 6144 and for 8192 tokens, the bf16 three-term MFMA of MM64-FP4 for 6144 and for 8192 and of MM256-FP8 and MM64-FP8 for 1024–16384), a seed-0 random and a cancellation input each (26 inputs), plus two inputs built with nonzero weight residuals; on the 26 the kernel's error against fp64 no larger than the fp32 matmul's in L2 on 25 (the cancellation input of MM256-FP4 for 6144 tokens 1.0000934×), in max-abs on 23 (the cancellation inputs of MM256-FP4 for 2048–4096, for 6144 and for 8192 tokens 1.0319×, 1.0018×, 1.0451×), element by element on 3; in the two residual inputs (MM64-FP4 at 6144 and at 8192 tokens) 16 flags and 256 residual entries set, the fp64 correction ran, logits 0 without it and 2^-23 with it, equal to fp64 and to the fp32 matmul: error 0 on both (MiniMax-M2 MoE kernels)
- E30 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, the library kernels on their ops' input sets; for GDN prefill also a GPU fp32 eager recurrence and an fp64 one on the same inputs] Qwen2-MoE router + experts at 1, 2, 4 and 32 tokens: all 32 SQNR comparisons passed, the once-rounded bf16 logits matched at 1–4 tokens and the fp64 MFMA router partials at 32 had zero measured error; GDN: all 168 recorded accuracy comparisons passed; GDN prefill delta state of the Qwen3.5 kernels for 512 tokens in 2 sequences and for 8193 in 5, a random and a cancellation input each: within a preset limit of 4 × the fp32 recurrence's error + 4 × eps32 × the fp64 norm only on the random input at 512 in 2 (L2 error 1.75 and max-abs 1.93 times the fp32 recurrence's), the random input at 8193 in 5 over it in max-abs only (4.69 and 37.3 times); on the cancellation inputs L2 and max-abs error 721 and 6116 times the fp32 recurrence's at 8193 in 5, 1192 and 9171 times at 512 in 2; all four accuracy checks passed
- E31 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=12 timing blocks, the op's baseline timed at the size] bf16 GEMM at N 16, K 128, shapes with no library kernel: 1.924 µs at M = 1, 2.787 µs at M = 64
- E32 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2, the finish of the Qwen3.5 GDN prefill kernel for 16384 tokens in 2 sequences of up to 10000, on random bf16 groups against an fp32 mirror of the reference quantization] 100000 groups of 1 × 128: 0 scale and 0 code mismatches in each of two runs
- E33 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3 repeats, `buffer_load` over a 4.25 GiB uint8 tensor from three scalar bases, `num_records` 0x7ffffffe] `_ubyte` and `_ushort` returned data up to byte 2^31 − 3, `_dword` and `_dwordx4` up to 2^31 − 5 and 0 from 2^31 − 4, a `dwordx4` across that edge its first 12 bytes; no width read data at or above 2^31 − 2; the edge moved with the scalar base (a probe, not a library kernel)
- E34 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=15 rounds, CUDA events over a graph of 20 back-to-back calls (operands in cache), orders rotated] the fp8 GEMM N 3072 × K 1536 at M 256 with `pn` (B) against `pn ^ (pm * 2)` (A): B/A 1.011 (interval 1.004–1.017); B bitwise equal to A
