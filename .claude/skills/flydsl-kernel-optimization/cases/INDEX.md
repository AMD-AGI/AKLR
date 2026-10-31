---
kind: index
---

# Cases

Delivered kernels: what made them fast, under which conditions, and how to reproduce them.

<!-- generated:start (tools/kb_index.py) -->
| Page | Ops | Dtypes | Regimes | Contract | Basis |
|---|---|---|---|---|---|
| [fp8 block-scaled GEMM, N 4096 × K 256–8192 and N 8192 / 32768 × K 1024, shuffled B — one structure per K; at K 512 and 1024 most Points up to m = 1024 are not served](a8w8_n4096_k_sweep.md) | gemm | fp8 | M 1-64, M 128-512, M 1024-4096 | bounded | measured N=1 |
| [fp8 block-scaled GEMM with a row-major a_scale, shuffled B — read the scales in the caller's layout; the margin at m ≥ 2 includes the baseline's relayout](a8w8_row_major_a_scale.md) | gemm | fp8 | M 1-64, M 128-512, M 1024-4096 | bounded | measured N=1 |
| [fp8 block-scaled GEMM, N 512–1536 × K 4096, shuffled B — K split over the waves of one workgroup](a8w8_small_n_large_k.md) | gemm | fp8 | M 1-64, M 128-512, M 1024-4096 | bounded | measured N=1 |
| [bf16 GEMM, narrow N (32–256) × K 6144 — whole dot products per wave or workgroup at M ≤ 8, split-K with a fixed-order fold at M ≥ 2048](bf16_glm52_narrow_n.md) | gemm | bf16 | M 2-8, M 16-1024, M 2048-4096 | bitwise | measured N=1 |
| [bf16 GEMM, wide N (2624–6144) × K 2048–6144 — one wave per output column on the unpacked weight at M ≤ 2; a weight repack per call loses above](bf16_glm52_wide_n.md) | gemm | bf16 | M 1-2, M 4-4096 | bitwise | measured N=1 |
| [bf16 GEMM, N 256–2048 × K 4096 — K split over waves or workgroups, folded in a fixed order](bf16_k4096_splitk_hgemm.md) | gemm | bf16 | M 1-256, M 512-4096 | bitwise | measured N=1 |
| [fp8 paged MQA logits (DeepSeek-V4 indexer), 64 heads, page 64, 4097-page tables — 16 fp8 16x16x128 MFMAs per page with the head sum kept in registers, a merge-free split over pages, and a band-sorted grid from batch 64](fp8_paged_mqa_logits.md) | mla, indexer_logits | fp8 | decode batch 1-4, decode batch 8-32, decode batch 64-256 | bounded | measured N=1 |
| [MXFP4 fused MoE with clamped SwiGLU on gate/up-interleaved weights, hidden 4096, top-6 (256 experts, or 32 / 64 local of 256) — one workgroup per route and 16 columns, two launches, at 1–8 tokens](fused_moe_interleave_swiglu.md) | moe | mxfp4, bf16 | tokens 1-2 (expert parallel), tokens 1-8 (all experts local) | bounded | measured N=1 |
| [MXFP4 fused MoE, hidden 6144, top-9, 16×16-shuffled weights (257, 65 and 33 experts) — no sort and two launches at 1 token, the sort folded into stage 1 at 8–64, sorted row tiles with fixed-order route sums from 64 tokens](fused_moe_shuffled_16x16.md) | moe | mxfp4 | tokens 1-2, tokens 4-64, tokens 128-4096 | bounded | measured N=1 |
| [bf16 GEMM, N 7168 × K 768–4224 and N 18432 / 20480 × K 1536 / 7168 (Kimi-K3) — one column band of B per workgroup at m ≤ 112, a tiled 8-wave grid that turns persistent at m 9397 and 16384](k3_bf16_band_stream.md) | gemm | bf16 | M 7-112, M 893-16384 | bitwise | measured N=1 |
| [bf16 GEMM, short K or large M only (Kimi-K3: N 1536 × K 128, N 163840 × K 256, N 2304 × K 1536, N 3072 × K 512) — the whole K in one tile or one workgroup's waves, persistent grids at the largest M](k3_bf16_short_k_large_m.md) | gemm | bf16 | M 1-14, M 8-112, M 893-16384, M 35107-342073 | bitwise | measured N=1 |
| [bf16 GEMM, N 144–3584 × K 7168 (Kimi-K3) — K split 4–16 ways across workgroups at m ≤ 112 and folded in a fixed order inside the launch; tiled or lightly split grids at m ≥ 893](k3_bf16_splitk_narrow_n.md) | gemm | bf16 | M 7-112, M 893-16384 | bitwise | measured N=1 |
| [bf16 GEMM, N 6016–8448 × K 7168 and N 7168 × K 35840 (Kimi-K3) — split K 2–8 ways at m ≤ 112 with an in-launch fixed-order fold, a 4-wave AGPR tile or a split tail at large m](k3_bf16_splitk_wide_n.md) | gemm | bf16 | M 8-112, M 893-16384 | bitwise | measured N=1 |
| [mHC fused post + pre, 4 streams × hidden 4096, with norm weight — the post combine, the 24-logit projection and a 4×4 Gram in one pass over the residual; the RMSNorm comes from the Gram, so the finish is element-wise](mhc_fused_post_pre.md) | mhc | bf16, fp32 | tokens 1-64, tokens 128-512, tokens 1024-4096 | bounded | measured N=1 |
| [Sparse MLA decode on DeepSeek-V4's padded fp8 KV pool, 64 heads, 128 window slots plus 0, 512 or 8256 compressed-cache slots — one dequantised 32-key LDS tile feeds both GEMMs for all 64 heads; a second launch merges the partials](mla_sparse_fp8_decode.md) | mla, attention | fp8, bf16 | decode batch 1-16, decode batch 32-64, decode batch 128-256 | bounded | measured N=1 |
<!-- generated:end -->
