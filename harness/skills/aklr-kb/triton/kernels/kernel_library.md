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
compiled resources and shape-specific selection conditions. Packaging:
[authoring](../authoring.md) §10, §11. Original schema sources and compiler references are listed under Evidence.
A file is a starting template, not a certified nearest-shape or fastest-design rule.

## Route here when

- You start a Triton or Gluon kernel for a new schema and look for the committed kernel nearest in op, dtype and token count, or need to know that AKLR has no Triton kernel for your op or shape (§3).
- You change or tune a family and need its per-Point files and launch structure; the ablations run on these files are in [tuning](../tuning.md) §3, §5, §9, §11, §12.
- You want a shipped example of a technique: async LDS rings, MXFP4 `mfma_scaled`, atomic-ticket sorts, split-KV merges, fused fp8 group quantization (§4).

## 1. Using a committed kernel as the starting point

- **Find it.** A kernel is one schema file, `aklr/<ops|plugins/sglang/...>/impls/triton/<scenario>/<uuid>.json`; its files are the `content` of each `kernel.sources` entry, its entry is `kernel.entry_point` (`kernel.py::run`, or `main.py::run` in the 15 N 3072 fp8 GEMMs). The format is in `aklr/ops/README.md` § The kernel section and ABI. This page names a file by the first four hex digits of its uuid, unique among the 154 Triton files: `find aklr -path '*/impls/triton/*' -name '8411*.json'` finds it. A citation `<schema>#kernel.py:<line>` is a line of a source the schema stores; `python -c 'import json,sys,pathlib; [pathlib.Path(s["path"]).write_text(s["content"]) for s in json.load(open(sys.argv[1]))["kernel"]["sources"]]' <schema.json>` writes the sources into the current directory.
- **One design per Point.** 116 of the 154 files list one Point, the others 2–14; only Qwen3-Next's Points are held by more than one Triton file (§13, §14); both bf16 GEMM scenarios share Points with a FlyDSL file, and `add_rms_norm` has a HIP file at its token counts with another epsilon (§6). Many single-Point files retain host branches for other token counts. Follow the actual Point's branch
  before copying a constant; a dead branch is not a tested configuration.

- **The shape is in the code.** A file serves one scenario and writes its constant axes as literals: N 3072's 12 K groups as `gl.static_range(12)` (`54f6`, S6), N 7168's `STAGES = 8` stages of 256 K and its 224-tile order `tile * 8 - quotient * 223` (`0675`, S5), 256 experts as `(256,)` vectors and top-8 plus the shared expert as `row * 9` (`d3bb`, S10). Many files also assert the shape on the host (`assert n == 3072 and k == 1536`, S6); some other files omit equivalent guards, so a successful host call is not proof the new shape is legal. For another shape read every literal and assert, not only your Point's branch, and recount the grid against the 256 CUs ([tuning](../tuning.md) §6). Buffer accesses take 32-bit offsets and reach only byte offsets from their scalar base below 2^31 − 2, the descriptor's `num_records`; of the 137 files using them, 30 check a size against 2^31 on the host (`cache_buffer` in `2ee5`, S24), 4 `static_assert` one, and 103 have neither kind of guard ([gluon/memory](../gluon/memory.md) §3). MoE traps: [moe](../moe.md) §12.
- **Triton 3.8 APIs.** Sources using `llvm_fn_attrs`, `compute_efficient_padded_shared_layout` or
  `gl.amd.slice` need the matching Triton API. State the minimum supported toolchain in `requires`;
  an unmet requirement should prevent dispatch from building or timing the incompatible source.
- **Copy a Point's whole configuration.** Tile height, splits, warps, scratch shapes and pipeline depth
  are coupled. The 24-token DSA head64 path only allocates its LDS cache for one tile; changing splits
  alone can make that buffer undefined. MiniMax `6557` gate-fold changes can trigger an LLVM crash
  with its narrow K tile and scheduler attribute. Start from a compiling combination and retune the full call.

- **Helpers are copied, not shared.** A schema imports its schema-local helpers only from its own files, so families copy their helpers: `initial_state.py` is byte-identical in all 26 GDN prefill files, `_math.py` in the 10 DeepSeek files that use it.
- **Reading the numbers.** Resources come from compiling every schema once on MI355X with Triton 3.8.0; "VGPR" is a kernel's `.vgpr_count`, architectural VGPRs plus AGPRs (a 4-warp 256 × 256 `tl.dot` for gfx950 with 256 of each reports 512). A CI speedup is the lower bound CI stored for one Point against that op's baseline; dispatch serves a Point from `AKLR_MIN_SPEEDUP` up (`aklr/ops/README.md` § Environment variables). It describes a whole design at one Point against one op's baseline: do not rank families by it. A file's own values are its `workloads[].speedup`; read them before you start from a file, because the family ranges below hide weak Points, such as N 7168 at M = 128 (1.114), DeepSeek inter 256 at 128 tokens (1.030) and DSA 64 heads at 8 tokens (1.030).

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

| Your op | Starting source | Mechanism and selection condition |
|---|---|---|
| e4m3 GEMM, FP32 scales per 1×128 A and 128×128 B | §5 N3072/K1536 or N7168/K2048 | unit-scale fp8 MFMA per K group, FP32 group scaling, async LDS pipeline; prove stride and scale indexing for another shape |
| BF16 GEMM, M≤64 | §6 `f806`, `b88c` | pointer `tl.dot`, weight streaming; `b88c` M64 adds a fixed-order split-K merge |
| MXFP4 GEMM | expert projections in §7/§8 | e2m1 `mfma_scaled` with E8M0 per32 and shuffled operands; remove routing, preserve packed-scale geometry |
| BF16 hidden × FP32 router weights | MiniMax §8/§9 | small-M VALU or FP32/three-BF16 MFMA branches; validate actual logits, ties and top-k rather than assume FP32 equivalence |
| once-BF16-rounded router with exact top-k | Qwen2 §10 | FP32 split partials summed before one BF16 rounding; some router+expert sources use FP64 products and midpoint correction |
| shared-expert MXFP4 MoE | DeepSeek §7 | top8 plus shared ninth route, small-M recomputation or grouped jobs, shared down/combined output |
| all or subset of routed MXFP4 experts | MiniMax §8 | ticket queues or padded sort; normalize global top8, mask nonlocal routes using runtime expert_offset |
| FP8 MoE with 128×128 block scales | MiniMax §9, Qwen2 §10 | quantize activations per1×128; route grouping and down hand-off depend on batch and shard |
| experts with routing supplied | Qwen2 §10 | per-route split-K at M8; one-CTA LDS sort and grouped/persistent stages at M64/128 |
| SwiGLU and requantization | `d3bb` `_activate_quantize`, `5cac`/`5c35` FP8 version | separate launch for K-split sums; full-K gate-up can fuse it if it owns a complete quant group |
| gathered-row sparse decode | DSA §11 | atomic-free split-KV merge; `fbe6` M256 uses direct one-program-per-token attention |
| paged KV decode with FP8 output | Qwen3.5 §12 | norm/RoPE/cache prologue, split-KV producer, gate+quant merge |
| Q/K RMSNorm, mRoPE, cache write | §12 `d9cb`/`dfb1` prologue | one-wave role grid, wave-local partner gather and constexpr axis masks |
| per-slot decode recurrence | GDN §13 | split-state CTAs or whole-head ownership; history shift only after old readers finish |
| chunked recurrence over sequences | GDN §14 | parallel factors, sequence/checkpoint or affine-summary scan, parallel epilogue; tune sequence count and length together |
| row normalization | §6 `cf63`/`b391` | one row per program, grid-stride at large row counts; Gluon equivalents are fused in attention/GDN |
| many Points from one source | `cf63`, `f806`, GDN `f440` | host branches or exact-match table; runtime annotations reuse binaries only with an unchanged complete specialization key |
| per1×128 FP8 output for next GEMM | §12 finish or §14 `3b99` | amax/scale/codes from the BF16-rounded output; preserve output and scale layout contracts |

**Not covered by a Triton kernel.** These contracts have only FlyDSL or HIP kernels on main, or none. Start from
[authoring](../authoring.md) §10 and borrow:

Before implementing an uncovered Point, measure its legal baseline and any served implementation.
A nearby source supplies a mechanism, not a transfer guarantee.

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
4 A + 6 B LDS slots at M = 16 (`1db6`), 4 + 4 at M = 32 (`ae4f`) and 3 + 3 at M = 64 (`0675`), 64–72 KiB;
`1db6` takes its 6 B slots only when `a`'s row stride is 2048 (S5, S46). The N 3072 files at M = 16 and 32 do not carry
over: they give each 128-K group its own LDS slot (S6), 48 and 49 KiB at K 1536 (M = 1, 2, 4, 8: 9, 9, 12, 36 KiB)
  about 224 KiB at K 7168 (estimated), above the 160 KiB one work-group may request ([gfx950
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
| async direct-to-LDS copies (`async_copy.buffer_load_to_shared`, `commit_group`, `wait_group`) | fp8 GEMM 24, DeepSeek 6, Qwen2 router 6 and experts 1, DSA 4, gated attention 3, MiniMax FP8 2; staging the 8-token router's hidden tile this way gained nothing measurable over `gl.load`s converted through LDS (8.45 µs both; the weight's load layout changed with it)  ([tuning](../tuning.md) §5) | `7266` (S1) | [gluon/memory](../gluon/memory.md) §8, §12 |
| padded LDS layouts (`PaddedSharedLayout`, `compute_efficient_padded_shared_layout`) | fp8 GEMM 14, MiniMax MXFP4 11, DSA 5, Qwen2 router 5, DeepSeek 4, gated attention 3, MiniMax FP8 2 | `7266` (S1) | [gluon/memory](../gluon/memory.md) §9, [gluon/layouts](../gluon/layouts.md) §4 |
| `mfma_scaled` on e4m3 with `None` scales | fp8 GEMM 24 (each 128-K group's product multiplied by the fp32 block scales), MiniMax FP8 9, Qwen2 router + experts 4 | `7266` (S1) | [gluon/mfma](../gluon/mfma.md) §2, [numerics](../numerics.md) §4 |
| `mfma_scaled` on e4m3 with `None` scales into a running accumulator | DSA 5 (`0bb2`, `1148`, `2ee5`, `2f70`, `f60f`) | `2ee5` (S24) | [attention](../attention.md) §10 |
| `mfma_scaled` on e2m1 with e8m0 scales | MiniMax MXFP4 23, DeepSeek 19 | `6557` (S42) | [gluon/mfma](../gluon/mfma.md) §2, §3, [numerics](../numerics.md) §3 |
| K groups folded into the MFMA rows at tiny M | all 8 fp8 GEMM files at M ≤ 8; the MiniMax MXFP4 decode file `6557` (`gate_fold`) | `bd01` (S2), `6557` (S42) | [gluon/mfma](../gluon/mfma.md) §8, [gemm](../gemm.md) §4 |
| `gl.inline_asm_elementwise` | 83: MiniMax MXFP4 22, DeepSeek 19, Qwen3.5 GDN decode 8, Qwen3-Next GDN decode 7, fp8 GEMM 6, MiniMax FP8 6, Qwen2 router 5, Qwen3.5 GDN prefill 4, DSA 3, Qwen2 router + experts 3 | `d3bb`'s `_next_expert` (S10) | [gluon/wave_ops](../gluon/wave_ops.md) §1, §2 |
| relaxed global integer atomics (tickets, job counters, bitmasks); LDS `atomic_scatter_*` | DeepSeek 12, MiniMax MXFP4 11, MiniMax FP8 6; LDS: Qwen2 experts 2, MiniMax FP8 1 | `8411` (S15), `8698` (S23) | [moe](../moe.md) §6, [numerics](../numerics.md) §8 |
| fixed-grid graph capture/replay | shape-based worst-case grids, device-written job totals, early exits and initialized scratch | `8411` (S15) | [authoring](../authoring.md) §8 |
| several roles in one grid, split by program id | program-id branches calling different helpers (a static scan that also counts bounds branches): DeepSeek 18 of 19, MiniMax MXFP4 19 of 23, MiniMax FP8 7 of 9, Qwen2 router + experts 4 of 4, gated attention 7 of 9 files; a finish branch in 6 Qwen3.5 GDN prefill files | `5751` (S9) | [moe](../moe.md) §3 |
| a fixed-order merge launch instead of float atomics (bitwise contracts) | bf16 GEMM at N 2624, M 64 (a second kernel adds the six partials in order); DSA and gated attention, which use no atomic and repeat bit for bit | `f60f` (S25), `b88c` (S7) | [attention](../attention.md) §2, [numerics](../numerics.md) §8 |
| program-id tile-order remaps | reorder N/M tiles or K phases; tune at the actual Point, without assuming XCD placement or unique L2 causality | `bd01`, `083b` (S3) | [gluon/memory](../gluon/memory.md) §14 |
| router precision alternatives | FP32 VALU/MFMA, BF16 split terms or FP64 routing products; finite logit accuracy does not certify exact top-k or whole-op FP64 equivalence | `99f4` (S22) | [numerics](../numerics.md) §5 |
| fused e4m3 quantization of an output or GEMM input | every file of gated attention, GDN decode and prefill (the bf16 op's `cd00` excepted), MiniMax FP8 and the Qwen2 experts: 64 files | `4f25` (S43) | [numerics](../numerics.md) §2 |
| `enable_fp_fusion=False` on launches | 42: DeepSeek 19, Qwen3.5 GDN prefill 11, Qwen3-Next GDN prefill 11, DSA 1; set per Point in 4 more | `d3bb` (S10) | [numerics](../numerics.md) §6, [tuning](../tuning.md) §9 |
| `llvm_fn_attrs`; `amdgpu-ieee` only in the Qwen2 routers | 26: Qwen2 router 5, Qwen3.5 GDN prefill 5, fp8 GEMM 4, MiniMax MXFP4 4, Qwen3-Next GDN decode 4, MiniMax FP8 3, DeepSeek 1 | `c388` (S20) | [tuning](../tuning.md) §9 |
| no MFMA at all | 22: GDN decode 18, gated attention at 1–2 rows 2, RMSNorm 2 | `2991` | [gluon/mfma](../gluon/mfma.md) §2, [gdn](../gdn.md) §1 |

**Remapping is a Point-specific scheduling knob, not a universal L2 rule.** Changing tile order helped
some actual GEMM/MoE Points and hurt others; removing the `199c` M256 XOR improved the tested call,
while other sources benefited from their committed order. A `pid % 8` locality model does not measure
CTA placement or L2 fills. Even one-reader-per-weight designs can have timing effects from ordering.
Compare the actual Point, and do not attribute a measured gain uniquely to XCD-local shared-weight reuse.

In the 897 compiled AKLR kernels : direct-to-LDS `buffer_load … lds` in 71 (fp8 GEMM 24 of 24, MiniMax FP8
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
- **Resources** : 24 kernels; `num_warps` 1 (10), 2 (2), 4 (12); `waves_per_eu` 1–8 in 13; VGPR 40–242,
  AGPRs in 8 (8–32); LDS 9–72 KiB; no spills; `v_mfma_scale_f32_16x16x128_f8f6f4` in 21, `…_32x32x64_…` in 3.
- **CI speedups** : 1.114 (N 7168, M 128) to 1.840 (N 7168, M 8) over the 24 Points.

## 6. Plain Triton: bf16 decode GEMM and RMSNorm

The four `@triton.jit` files. Why plain Triton fits them, and what its compiler adds (coalescing, `num_stages`
pipelining, buffer instructions): [authoring](../authoring.md) §2. Each GEMM kernel but the split-K reduce holds
`buffer_load_dwordx4 … lds`; the 6 RMSNorm kernels load with `buffer_load_dwordx4`, no `global_load`.

- **bf16 decode GEMM**, `trans_b`, no bias: `gemm_n6144_k2048/` (`f806`, M = 2, 4, 8, 16, 32, 64) and `gemm_n2624_k6144/`
  (`b88c`, M = 2, 4, 8, 64). One workgroup per N tile (192 at N 6144, 164 at N 2624), each `tl.dot` accumulating in fp32,
  the weight loaded with `.cg`; each XCD streams one contiguous run of N tiles in `b88c` at M ≤ 8 and `f806` at M ≤ 16,
  8-tile chunks  in `f806` at M 32, 64 (`b88c`'s `_xcd_remap` is a permutation for any grid size); at
  some M a per-workgroup K phase. One launch, except `b88c` at M = 64: split-K 6 into fp32 scratch (246 workgroups), then
  a second kernel adds the six partials in a fixed order. The configuration comes from a `@functools.cache` compile
  function, `_launch_for(a_shape, b_shape)`, that returns the launch closure (S7). Tiles by M, split-K under the bitwise
  contract, M = 1 and re-tuning for cold-cache timing: [gemm/decode_gemm](../gemm/decode_gemm.md).
- **RMSNorm**: `rms_norm_hidden_size6144/` (`cf63`, 3–4096 tokens), `add_rms_norm_hidden_size6144/` (`b391`, 4–256
  tokens). The 6144-wide row as three 2048-wide chunks in registers, the sum of squares in fp32, `.cg` loads, `.wt`
  stores, one program per row; `rms_norm` from 1024 rows gives each program `m // 512` rows, the next
  row's loads pipelined and the weight loaded once (S8):

```python
for it in tl.range(0, LOOP, num_stages=NS):
    row = tl.program_id(0) + it * tl.num_programs(0)
    rmask = cmask & (row < M)
    base = row.to(tl.int64) * H + cols
    x0 = tl.load(x_ptr + base, mask=rmask, other=0.0, cache_modifier=".cg").to(tl.float32)
```
- **Resources** : GEMM 7 kernels, `num_warps` 2, 4 or 8, `waves_per_eu` 2 in 6, VGPR 26–106, LDS up to
  about 148 KiB of the pipeliner's buffers, `v_mfma_f32_16x16x32_bf16`; RMSNorm 6 kernels, 4 warps, VGPR 49–79, 16 B of LDS.
- **CI speedups** : GEMM 1.048 (N 2624, M 2) to 1.292 (N 6144, M 8); `rms_norm` 1.006 (4 tokens) to
  1.212 (2048), below the default `AKLR_MIN_SPEEDUP` and so not served at 3–64 tokens; `add_rms_norm` 1.024 (8 tokens,
  not served) to 1.093 (256).
- **Other languages at these Points.** The three scenarios also hold a FlyDSL or HIP file, and dispatch serves each
  Point with the best measured file of any language (`aklr/ops/README.md`, the paragraph on files that share a
  `<name>`). The HIP `add_rms_norm` (`fc82`) stored 1.038 and 1.067 at 32 and 128 tokens against `b391`'s 1.035 and
  1.062, so it is the one served there, and it alone holds 512–8192 tokens. FlyDSL files serve M = 1 of both GEMM
  scenarios (1.191 at N 6144, 1.134 at N 2624) and store 0.360–1.045 at the Triton files' Points; at N 2624, M 16 and
  32, and in both scenarios from M 128, the only files are FlyDSL ones below 1.0 (0.396–0.998), so no AKLR kernel is
  served there. A new kernel at a shared Point has to beat the file served there.

## 7. DeepSeek-V2 / GLM MXFP4 MoE

SGLang's `DeepseekV2MoE` on one rank: bf16 router logits, the sigmoid top-8 of 256 with an fp32 correction bias, weights
renormalized × 2.5, the shared expert id 256 as route slot 8 at weight 1, MXFP4 experts (e2m1 with e8m0 per 32, aiter's 16 × 16
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
- **Resources** : 116 kernels; `num_warps` 1 (57), 2 (8), 4 (51); `waves_per_eu` unset in all; VGPR
  22–280, AGPRs in 2 (16, 24); LDS 0–74.8 KiB; no spills; scaled 16 × 16 × 128 MFMA, BF16 16 × 16 × 32 and

  scaled 32 × 32 × 64 in 8.
- **CI speedups** : 1.030 (inter 256, 128 tokens) to 2.186 (inter 256, 2 tokens) over the 24 Points.

## 8. MiniMax-M2 MXFP4 MoE

MiniMax-M2's MoE layer on one rank: an fp32 router (sigmoid plus correction bias, top-8 of 256, renormalized), MXFP4 experts, no shared expert. `n_experts256_inter_dim512` holds all 256 experts (12 files); `n_experts64_inter_dim1536` holds the 64 from `expert_offset`, routes to other experts adding nothing (11 files); 30 Points. CPU baseline-closure inspection found router `tgemm.mm` with its first operand cast to fp32, `moe_fused_gate` and `fused_moe` at `per_1x32`; vendor execution was not measured).

| Tokens | E256 | E64 | Launches | Decomposition |
|---|---|---|---|---|
| 1–16 | `6557` (1) `fbd9` (2) `96e5` (4, 8, 16) | `b065` (1, 2) `f82c` (4) `3df5` (8, 16) | 3 | VALU router with hidden quantization; per-route gate-up with repeated selection, or local-expert token masks at E64 M8/16; activation in a separate K-split finish or full-K epilogue; down+weighted combine |

| 32–256 | `0101` (32) `d861` (64) `01ce` (128) `4da3` (256) | `dfa8` (32, 64) `1d27` (128, 256) | 5 | split-K router tiles, counter and job initialization and the hidden quantization in the first two launches; per-token top-8 with relaxed atomic tickets, a job opened every 16th ticket (every 64th in `01ce`, `4da3`); gate-up over jobs (`1d27`: each program finds its job by scanning the counts); down; combine |
| 1024–16384 | `bb66` (1024) `147b` (2048, 4096) `e6fc` (6144) `51ae` (8192) `8411` (16384) | `abc7` (1024) `cb88` (2048) `b94f` (4096) `62df` (6144) `f839` (8192) `8b22` (16384) | 6 (E256), 7–8 (E64) | router and quantization; top8; padded counting-sort/scatter; gate-up+SwiGLU/MXFP4; down; slot-order combine. E64 is the expert shard, while routing still spans 256 experts |

- **Techniques.** Relaxed-ticket job queues; mask explicit replicated selection lanes so each route takes
  one ticket (S17). E64 prefill counting-sort uses ballots/popcounts (S16), with an atomic-free path from
  2048 tokens. `8411` uses shape-based worst-case capacity and a device-written total (S15).
  Router precision is branch-specific: FP32 VALU at small M, FP32 MFMA in E256 prefill, and three-BF16
  terms in several E64 sources. Finite logit comparisons do not guarantee arbitrary-input top8 equivalence;
  residual-correction branches require inputs that actually make the residual nonzero.

- **Copy from.** The relaxed ticket of `8411`'s select launch and the scatter that turns it into a slot (S15; shown in
  [moe](../moe.md) §6; checked with one count block, not `8411`'s several); the gate-up fold of `6557` (S42), which packs
  `gate_fold` K chunks into the gate-up's 16 MFMA rows, set per token count.
- **Resources** : 183 kernels; `num_warps` 1 (35), 2 (7), 4 (139), 8 (2); `waves_per_eu` 3 in 4, 4 in 1;
  VGPR 9–244, no AGPRs; LDS 0–35 KiB; no spills; scaled 16 × 16 × 128 MFMA in 69, fp32 16 × 16 × 4 in 12, bf16
  16 × 16 × 32 in 5, fp32 32 × 32 × 2 in 3, scaled 32 × 32 × 64 in 1.
- **CI speedups** : 1.178  tokens) to 3.633  token) over the 30 Points.

## 9. MiniMax-M2 FP8 MoE

The same layer with e4m3 experts and fp32 scales per 128 × 128 block, both GEMM inputs quantized per 1 × 128 in the kernels: `n_experts256_inter_dim384` (5 files) and `n_experts64_inter_dim1536` (4 files), 28 Points. The baseline source closure contains router `tgemm.mm` with a first-operand fp32 cast, `moe_fused_gate` and `fused_moe` at `per_128x128`; the vendor baseline must be measured separately.

| Tokens | E256 | E64 | Launches | Decomposition |
|---|---|---|---|---|
| 1–16 | `e75e` (1, 2, 4, 8, 16) | `1c17` (1, 2, 4, 8, 16) | 3 (E256), 4–6 (E64) | full-K VALU router and input quantization; per-route gate-up with selection recomputation, or local-expert programs at E64 M8/16; down+combine where ownership permits |
| 32–256 | `5cac` (32, 64) `b43d` (128) `8a5a` (256) | `0982` (32, 64) `94af` (128, 256) | 5–6 | router, selection and job queues; expert gate-up; BF16 activation hand-off or FP8 quantization; grouped down and combine. `94af` uses a persistent gate-up grid |

| 1024–16384 | `f770` (1024, 2048, 4096, 8192, 16384) | `82e9` (the same five) | 6–8 | a prepare launch first (`82e9`'s `_prepare_inputs`; `f770`'s `_prepare_router_and_quantize` from 8192 tokens) or the router first (`f770` up to 4096); router; selection and sort (`f770`: launches with a `CHUNK` constexpr; `82e9`: an atomic-free SWAR counting sort, `_rank_counts` and `_prefix_scatter`); gate-up with fused SwiGLU + fp8 quantization; down; combine |

- **Techniques.** One warp per route under `AMDMFMALayout(..., warps_per_cta=[8, 1, 1])`, the route as the batch
  dimension (S18; [moe](../moe.md) §9); a counting sort with four 8-bit counters per 32-bit word and one
  `gl.associative_scan` (S19; [moe](../moe.md) §6); inline asm in 6 files: `v_rcp_f32` in all six, `v_exp_f32` in five,
  `v_readlane_b32` in four, `s_ff1_i32_b64` in three, `v_max_f64` / `v_min_f64` only in `94af`
  ([gluon/wave_ops](../gluon/wave_ops.md) §4, §6). Router branches use different BF16 split/rounding recipes. Some MXFP4 E64 variants apply flagged FP64
  residual correction, while FP8 `f770`/`82e9` use their own BF16 term sequences. Do not treat three terms
  alone as an accuracy theorem; validate actual top8 and the original whole-op bound.

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
- **Resources** : 188 kernels; `num_warps` 1 (73), 2 (13), 4 (96), 8 (6); `waves_per_eu` 1 in 4; VGPR
  14–327, AGPRs in 2 (35, 71); LDS 0–131.9 KiB; an 8-SGPR spill in `_pack_routes`; scaled 16 × 16 × 128 MFMA in 47,
  bf16 16 × 16 × 32 in 13, scaled 32 × 32 × 64 in 10, fp8 16 × 16 × 32 in 9, fp32 16 × 16 × 4 in 7.
- **CI speedups** : 1.102  tokens) to 4.115  token) over the 28 Points. Across §7–§9 the
  32–256-token Points are the weakest band whatever the design, and inside one 1–16-token file the value falls as
  tokens grow: band ranges in [moe](../moe.md) §2, band medians in [tuning](../tuning.md) §14.

## 10. Qwen2-MoE: router, fp8 experts, fp8 router + experts

Qwen3.5-397B-A17B's MoE block at TP4 (by the contract's docstring): model dim 4096, 512 routed experts, top-10; the
experts schemas add inter dim 256, one shared expert weighted by `sigmoid(hidden · shared_gate_weight)` (513 experts in
all) and e4m3 experts with fp32 128 × 128 block scales. The route tries the router + experts op first, then the router
followed by the experts op (`aklr/plugins/sglang/models/qwen2_moe/api.py`).

| Contract, directory `qwen2_moe_…` | Tokens: files | Launches | Decomposition |
|---|---|---|---|
| `router_model_dim4096_n_routed512_topk10/` | 8 `4801`, 16 `5f20`, 64 `a878`, 128 `cd47`, 256 `c388`, 2232 `4c2b`, 4096 `ca82` | 2 | a split-K projection (S20), exactly 256 programs at 8–256 tokens, then per-row selection over the stored bf16 logits: top-10 with lower-id ties, softmax weights over the selected logits; FP32 split sums round once to BF16. Prefill Points use a GEMM and multi-row selection |
| `fp8_experts_shuffled_16x16_…_n_experts513_…/` | 8 `5c35`, 64 `8698`, 128 `d425` | 3; 5; 5 | routing given; 8 tokens: per-route split-K gate-up, activation quantization, down with the ordered 11-route sum; 64 and 128: a route sort in LDS (by one CTA in `8698`), then gate-up, quantization, down, ordered reduce (`8698`: persistent stages over device-counted tasks) |
| `fp8_router_experts_shuffled_16x16_…/` | 1 `7d7e`, 2 `04e9`, 4 `ddb8`, 32 `99f4` | 4; 4; 4; 7 | router logits in fp64 (VALU), rounded once to bf16; every gate-up CTA replays the top-k of its row; activation quantization; one CTA per output tile multiplies all of a token's routes in one MFMA and keeps the diagonal; at 32 tokens fp64 MFMA router partials, a counting sort, sparse gate-up, quantization, grouped down, ordered sum |

- **Techniques.** One `@gluon.constexpr_function` gives the host its grid and the kernel its tile geometry (S20;
  [tuning](../tuning.md) §14); CPU AST inspection records `llvm_fn_attrs` arguments at the actual Qwen2 launch sites,
  not GPU scheduler effects ([tuning](../tuning.md) §9); the top-k replayed per CTA (S21; [moe](../moe.md) §4); fp64 products for a router that
  must round like an fp64 reference, on the VALU at 1–4 tokens and on fp64 MFMAs at 32 (S22; [numerics](../numerics.md)
  §5); a one-CTA counting sort with `atomic_scatter_add` on an LDS descriptor (S23; [moe](../moe.md) §6).
- **Copy from.** `c388`'s `_shape_config`, which returns `_ShapeConfig(split, expert_tile, k_tile, lookahead,
  routing_group)` per row count to both the launch and the kernel (S20; shown in [tuning](../tuning.md) §3).
- **Resources** : router 14 kernels, `num_warps` 1, 4 or 16, VGPR 41–192, LDS 0–130 KiB, bf16 MFMA;
  experts 13 kernels, 1–8 warps, `waves_per_eu` 2 or 4 in 3, VGPR 16–110, LDS 0–12 KiB, bf16 and fp16 MFMA; router +
  experts 19 kernels, 1–4 warps, VGPR 19–128, LDS 0–16 KiB, scaled 16 × 16 × 128, fp64 16 × 16 × 4 and fp16 MFMA. No
  AGPRs, no spills.
- **CI speedups** : router 1.074 (8 tokens) to 1.967 (256); experts 1.076 (128) to 1.208 (8); router +
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
- **Resources** : 15 kernels; `num_warps` 1 (3), 2 (2), 4 (6), 8 (4); `waves_per_eu` 1 in 2, 2 in 3;
  VGPR 23–228, AGPRs in 1 (22); LDS 0–160 KiB; no spills; bf16 32 × 32 × 16 MFMA in 4, scaled 32 × 32 × 64 in 4, bf16
  16 × 16 × 32 in 3, scaled 16 × 16 × 128 in 1, fp16 32 × 32 × 16 in 1.
- **CI speedups** : 1.030 (64 heads, 8 tokens) to 2.188 (64 heads, 24 tokens).

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
  grid groups all splits of 64 rows (S29). Its row map requires rows divisible by 64 when splits exceed
  one; a single split degenerates to the identity. Empty splits still write fresh neutral records (S30).
  Preserve the prologue's actual output strides: the `dfb1` below-256-row prologue returns interleaved
  views, but its attention reads contiguous rows, so copying that file to M128 is unsafe without repair.

- **Copy from.** The split-to-row map of the 256-row file (S29):

```python
# Flatten the launch into groups containing all splits of 64 rows.
cta = gl.program_id(0)
row = cta % 64 + cta // (64 * SPLITS) * 64
split = cta // 64 % SPLITS
```

  For 96 rows and two splits, this map misses 32 (row,split) pairs and emits 32 out-of-range programs.
  Use it only on compatible row counts, or a masked/general row-split grid. The source's asserted
  128/256 counts (S29) do not excuse inconsistent prologue strides; use the matching source pipeline.

- **Resources** : 27 kernels; `num_warps` 1 (9), 2 (1), 4 (17); `waves_per_eu` 2 in 2, 3 in 1; VGPR
  25–192, AGPRs in 1 (63); LDS 0–70 KiB; no spills; bf16 16 × 16 × 32 MFMA in 7, fp16 in 1.
- **CI speedups** : 1.066 (256 rows) to 2.182 (1 row).

## 13. GatedDeltaNet decode

Per row: a width-4 causal convolution over the projected Q, K, V with the conv history shifted in place, the gates, the gated delta-rule update of a 128 × 128 fp32 state per (slot, value head), a gated RMSNorm and e4m3 per 128 with fp32 scales. Qwen3.5 (16 value heads): 8 files, 1–256 rows, plus `gdn_decode_…/cd00` (64 rows, bf16 output). Qwen3-Next (8 value heads, grouped projections): 9 files over 9 Points, `f440` listing all of them. The baseline chains convolution update, recurrence, norm and quantization. No kernel uses MFMA. Triton side: [gdn](../gdn.md) §2–6.

| Model, rows | Files | Launches | Decomposition |
|---|---|---|---|
| Qwen3.5, 1–4 | `2c2a` (1) `2991` (2) `409c` (4) | 2 | each head's 128 value rows split over CTAs, 256 at 1, 2 and 4 rows; then a launch of 40 programs per row; the conv history shifted in place |
| Qwen3.5, 8–256 | `4f25` (8) `a4f8` (16) `20f6` (32) `cd00` (64, bf16 op) `fc26` (128) `c2c3` (256) | 2 | convolution with exclusive channel ownership, history shifted in place; one CTA per (row, head) owns the state, 4 waves at 8–32 rows, 16 from 64, the norm (and quantization) in its epilogue |
| Qwen3-Next, 1–8 | `b31b` (1) `bba1` (2) `3719` (4, 8) | 2 | producers of one or two waves per (row, value head or head pair, block of value rows), then a finisher |
| Qwen3-Next, 16–256 | `35b9` (16) `38a7` (32) `0c0b` (64) `0678` (128, 256) `c784` (128, 256) | 1 | one workgroup per (row, key head) owns the conv history and both value heads' states |

- **Techniques.** Split-head decode files defer the history shift until every first-launch reader finishes
  (S31). Moving it before or inside that reader launch can corrupt history/state, not necessarily on every
  input. Whole-key-group ownership permits an inline shift; replicated layouts such as `c784` use a CTA
  barrier (S32). A delayed reader exposes the hazard when neither ownership nor a barrier orders accesses.
  Final states alone are insufficient to validate a pipeline: old-history hazards can change only output.

The template keeps one configuration row per row count and raises on an unlisted count (S33). Inline asm supplies `v_med3_f32` (all 8 Qwen3.5 files, the ±448 clamp; `gl.clamp` emits the same instruction) and, in Qwen3-Next, `v_rcp_f32`, `v_exp_f32`, `v_log_f32`, `v_div_fixup_f32`, `v_pk_fma_f32`, `v_cvt_scalef32_pk_fp8_f32`, `s_setreg_imm32_b32` and DPP wave reductions (`v_max_f32` / `v_add_f32` with `row_shr`, then `v_readlane_b32`) ([gluon/wave_ops](../gluon/wave_ops.md) §3, §4, §6, §7).
- **Copy from.** The exclusive-owner shift of `0678`: one 1-D tile over the key group's 512 conv channels with `gl.static_assert(NW * 64 <= conv_width)`, so no lane or wave holds a copy and each thread shifts the history it read, with no barrier (S41):

```python
conv_layout: gl.constexpr = gl.BlockedLayout([1], [64], [NW], [0])
# This packed tile has no replicated lanes/waves. Every history read and
# write has one owner, even though Q/K are subsequently shared by two heads.
c = gl.arange(0, conv_width, layout=conv_layout)
```

  `c784` keeps replicated lanes and orders the reads before the shift with `gl.barrier` (S32). In one CI run on
  main in `d908d01` it was the slowest of the three 128- and 256-row designs, 1.466 and 1.151 against `0678`'s 1.782 and
  1.425; the designs differ in more than the shift ([gdn](../gdn.md) §2).
- **Resources** : Qwen3.5 18 kernels, `num_warps` 1, 2, 4 or 16, `waves_per_eu` 1, 6 or 8 in 5, VGPR
  16–146, LDS 0–2 KiB; Qwen3-Next 27 kernels, 1–8 warps, VGPR 17–208, LDS 0–2.5 KiB. No AGPRs, no spills.
- **CI speedups**: Qwen3.5 1.282 (256 rows) to 2.507 (2 rows), the bf16 op 1.461. Qwen3-Next stores no
  speedup on main at `43e112e`, so dispatch does not serve it at the default `AKLR_MIN_SPEEDUP`; a CI run of the same
  files on main in `d908d01` gave the best file per Point 1.457 (256 rows) to 3.350 (32 rows), and the per-size files
  above `f440` at 7 of 9 Points (equal at 4 rows, `f440` ahead at 256).

## 14. GatedDeltaNet chunked prefill

Packed sequences of the same layer: Qwen3.5, 15 files over 21 Points (512–16384 tokens in 1–8 sequences); Qwen3-Next, 11 files over 13 Points (1024–16384 in 1–16). Files: `kernel.py`, `initial_state.py` and, in 9 Qwen3.5 files, `packing.py`; the baseline chains convolution, gating, chunk recurrence, normalization and quantization. Triton side: [gdn](../gdn.md) §7–12.

Every call zeroes state slots marked fresh (S34), and nine Qwen3.5 sources repack projections into grouped
order. Prepare → chunk factors → optional summaries → sequential scan → parallel epilogue uses 5–8 launches.
Conv history must become the last three entries of [old history or fresh zeros; sequence inputs], retaining
old/zero entries for sequences shorter than three tokens. Commit it only after preparation readers finish
(S37): early commit can corrupt FP8 output while both final states remain unchanged.

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
  and `BATCH` (in `70f4` / `010e` also `TOTAL`) as constexprs in one and runtime ints in the other, permitting binary reuse only when the complete JIT specialization key is unchanged (S36); changing the annotations alone does not establish reuse across all Points. Shape, strides, launch options and other constexprs still determine the key. Runtime twins are
  candidates to reduce compilation, not a guarantee of binary reuse or speed at every Point.

- **Copy from.** The per-sequence pass of `24ed` (S35) and `_initialize_fresh` (S34).
- **Resources** : Qwen3.5 123 kernels, `num_warps` 1, 2, 4 or 8, `waves_per_eu` 2, 4 or 5 in 15, VGPR
  3–256, no AGPRs, LDS 0–32 KiB, spills in 2 (64 VGPRs in `_build_chunk_tables_paired`, 1 SGPR); Qwen3-Next 117
  kernels, 1–4 warps, VGPR 2–316, AGPRs in 5 (20–60), LDS 0–32 KiB, no spills. fp32 16 × 16 × 4 MFMA in 57 kernels
  of each family, bf16 16 × 16 × 32 in 24 of each; fp32 32 × 32 × 2 in 4 (Qwen3.5) and 2 (Qwen3-Next), bf16
  32 × 32 × 16 in 2 (Qwen3.5). All 26 files hold both bf16 16 × 16 × 32 and fp32 16 × 16 × 4 MFMAs.
- **CI speedups**: Qwen3.5 0.997 (8193 tokens in 5 sequences) to 2.638 (512 in 1); 0.997 and 1.029 (4097 in 3) are
  below the default `AKLR_MIN_SPEEDUP`. Qwen3-Next has none at `43e112e`; the same files' CI run on main in `d908d01` gave the best file per Point 1.233 (13312 tokens in 13 sequences) to 1.741 (5120 in 1).

## Evidence

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
- S32 aklr/plugins/sglang/models/qwen3_next/impls/triton/qwen3_next_gdn_decode_group_fp8_quant_n_v_heads8_key_dim128_value_dim128_conv_dim2048/c7840603329d5ed5.json#kernel.py:34 the convolution, `gl.barrier` at :40, the in-place shift at :41-44
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
