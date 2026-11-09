---
title: "MoE on gfx950: fused MoE (MXFP4 experts) and the expert grouped GEMM — package map"
kind: op
ops: [moe, fused_moe, grouped_gemm]
dtypes: [mxfp4, bf16, fp8]
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=1
---

# MoE — package map: fused MoE and the expert grouped GEMM

Where to start on a fused MoE kernel for AKLR (its FlyDSL side: `../flydsl/moe.md`): what the op computes, what
bounds it at each token count, which cards to read, which kernels to start from, and the AKLR files that govern
it. The expert GEMMs are grouped GEMMs in the NT layout: `grouped_sorted_tokens.md` covers the sorted-token form
AKLR's kernels use, and the other grouped-GEMM cards apply to the same GEMMs through the NT layout; § Techniques
that apply routes each problem, at any token count, to the card section that covers it. "m" is a Point's
`n_tokens`; a route is one (token, expert) pair.

Unless a sentence names another language or a vendor library, the measured numbers (`*(field, E<n>)*`) come from
FlyDSL kernels; each evidence line names its toolchain. The mechanisms, stated in ISA and LLVM terms or by the
FlyDSL, aiter or Turbo source they cite, should carry over to Triton or HIP; the numbers have not been checked there.

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
  reference (float32 in validation) rounded to the output dtype, with no bias test. Atomic accumulation passes it, and so do fp8
  partial rows and bf16 truncation ([fused_moe_shuffled_16x16.md](../cases/fused_moe_shuffled_16x16.md), Counter-signals).
- Every AKLR kernel computes it in two GEMM stages. How the routes reach their expert's rows is the main choice at a
  few tokens per call (m 1–64 below): a separate sort kernel, a workgroup set per route with no sort, or a sort
  folded into stage 1.

## Bottleneck by regime

| Regime | What bounds the time | Read |
|---|---|---|
| m 1–2 (6–18 routes) | Little work per call, so the launches and the sort weigh (estimated). At m 1 every AKLR kernel runs two launches, no sort and a workgroup set per route, and all four are served, at 1.26–3.36× *(field, E1)*; at m 2 the 33-expert kernel is served with four launches | the two cases |
| m 8–64 with 257 experts | 72–576 routes over 257 experts, so most hit experts are read for one or two routes (estimated); each graph node adds a fixed cost (`few_tokens.md` §1). Folding the sort into stage 1 saved 3.5–6.4 µs of 39–102 µs *(field, E2)* | `few_tokens.md` §1–§2 |
| m 10–60 with 64 local experts | Measured on aiter's FlyDSL MXFP4 kernels: bandwidth. At 60 tokens a 75-layer step read 5.77 TB/s against a 6.27 TB/s read roof measured in the same run (92%); from 10 to 60 tokens the time grew 2.38× while the experts hit grew 2.6× *(field, E3)* | `grouped_sorted_tokens.md` §3–§5 |
| Kimi-K3 SiTUv2, fp8 × MXFP4, m 8–112: 896 experts (TP) or 112 of 896 local (EP8), top-16 | Bandwidth: the weights of the experts the routes touch, 2.19 MB each at inter 384 and 17.5 MB at inter 3072 (estimated from the layout); count the touched experts from AKLR's own inputs (`../patterns/memory_bound.md`, Rule out first). From 32–40 tokens up the delivered kernels read them at 6.0–6.45 TB/s against a 6.5 TB/s read roof, aiter at 5.0–5.97, so at most 5–8% is left; at 8–24 tokens the per-launch costs around the GEMMs weigh, and EP8 measured 1.28–1.59 there *(field, E5)* | `../patterns/memory_bound.md`; `grouped_sorted_tokens.md` §2, §7; `few_tokens.md` |
| Kimi-K3 SiTUv2 at m 893–16384 | Every expert is touched; from ~9k tokens the GEMMs are compute-bound. At 16384 tokens the delivered kernels ran stage 1 at 1970 (TP) and 1889 (EP8) TF/s of useful FLOPs (padding rows not counted) and stage 2 at 1347 and 1764, where aiter ran 1431/1459 and 929/1267 *(field, E6)*; dense fp8 GEMMs of squat shapes reached ~2170–2400 TF/s in FlyDSL and ~2470–2660 in hipBLASLt on MI355X *(field, E7)*. TP's per-route rows and their reduce kernel took 190 of 1533 µs | `grouped_sorted_tokens.md` §2, §6; `../gemm/mainloop.md` §7; why a grouped GEMM trails a dense one: `grouped_tile_pricing.md` §6, `grouped_scheduling.md` §9 |
| m 64–4096 with 33 local experts | Rows repeat per expert: tile-M 32–128, several row tiles of one expert per weight sweep, activations staged through LDS. Served at 1.030–1.609 over aiter at every m from 64 *(field, E4)* | [fused_moe_shuffled_16x16.md](../cases/fused_moe_shuffled_16x16.md) |

## Techniques that apply

Problem → card section, for the expert GEMMs of a fused MoE (stage 1 and stage 2, both NT) and the kernels around
them. A row applies at every token count unless its problem names a range or a shape. The grouped cards other than
`grouped_sorted_tokens.md` were measured mostly on Primus-Turbo's kernels over contiguous rows, at hundreds to tens
of thousands of tokens per group (most at 4096, G = 32); `grouped_sorted_tokens.md` on MoE kernels at 8–16384
tokens. Reuse the mechanism and re-derive the number.

| Area | Problem | Read |
|---|---|---|
| Launches | A call that lasts tens of µs (a few tokens): decide the launches, count the graph nodes | `few_tokens.md` §1 |
| Launches | Remove the sort launch at 8–64 tokens | `few_tokens.md` §2 |
| Launches | Sort and quantization kernels around the GEMMs; fold per-token kernels into the sort launch | `grouped_sorted_tokens.md` §7 |
| Launches | Price a helper kernel (sort, quantizer, scale repack, reduce) by launching it again, not by its trace duration | `grouped_host_autotune.md` §7, `grouped_measurement.md` §6 |
| Launches | Kernel-argument preload; rows bound by launch and latency | `../gemm/small_m.md` §6, `../patterns/launch_floor_small_m.md` |
| Sorted-token mapping | Map sorted tokens to expert blocks, exit on the device, persistent stage 2 | `grouped_sorted_tokens.md` §2 |
| Sorted-token mapping | Under expert parallelism: sort and size the grid by the local experts, cap the grid | `grouped_sorted_tokens.md` §2 |
| Sorted-token mapping | An over-launched grid: exit on the raw workgroup id, then remap over the live count | `grouped_group_mapping.md` §3–§4, `grouped_sorted_tokens.md` §2 |
| Sorted-token mapping | Find a tile's expert on the device from per-expert counts or offsets (a sort folded into stage 1, or no per-block expert table): wave scan and ballot | `grouped_group_mapping.md` §5, `few_tokens.md` §2 |
| Sorted-token mapping | An expert id, row base or record count that feeds a descriptor; 64-bit per-expert weight bases at large expert counts | `grouped_group_mapping.md` §6–§7 |
| Sorted-token mapping | Experts that receive no route (most of them at a few tokens) | `grouped_group_mapping.md` §8; record-count clamp: `grouped_correctness.md` §3 |
| Per-size configuration | Grid shape and tile-N when the tokens fill only a few M-blocks | `grouped_sorted_tokens.md` §3 |
| Per-size configuration | Tile-M against the sort unit; halving a 256-row tile paid only while its grid fit one dispatch round | `grouped_sorted_tokens.md` §5, `grouped_tile_pricing.md` §3 |
| Per-size configuration | Split K over the waves of one workgroup | `../gemm/small_m.md` §5, `grouped_sorted_tokens.md` §3 |
| Per-size configuration | Cache policy of the weight loads: one block per expert against repeated blocks | `grouped_sorted_tokens.md` §4, §8 |
| Per-size configuration | A configuration per token count: race offline on balanced and skewed routing at every token count it covers; where one compile serves many token counts, key on the static shape | `grouped_host_autotune.md` §3, §5; § In AKLR below |
| Tile and occupancy | Tile size and waves per SIMD for the NT layout; the register wall at 256×256 | `grouped_tile_pricing.md` §1–§2 |
| Tile and occupancy | N (2 · inter dim, or model dim) not a multiple of tile-N: the half-tile body at the N edge | `grouped_tile_pricing.md` §4 |
| Tile and occupancy | An expert's rows not a multiple of tile-M: tail bodies and their hit rate under skew; skip the padding rows' MFMAs | `grouped_tile_pricing.md` §5, `grouped_sorted_tokens.md` §5 |
| Tile and occupancy | Few K iterations per tile (a short K, as stage 2 has at a small inter dim): the per-tile fixed cost, not the main loop | `grouped_tile_pricing.md` §6, `grouped_workflow.md` §5 |
| Tile and occupancy | Persistent grid, one tile per workgroup, or several tiles per workgroup | `grouped_scheduling.md` §6–§7, `grouped_sorted_tokens.md` §2 |
| Tile and occupancy | MFMA atom: scaled fp4, or bf16 on dequantized fp4 | `../gemm/mainloop.md` §5 |
| XCD placement | Tile order across XCDs, the dense rules | `../gemm/schedule_and_locality.md` §2 |
| XCD placement | Keep an M-block's N tiles on one XCD; race any remap against the hardware order | `grouped_sorted_tokens.md` §2–§3 |
| XCD placement | Several M-blocks per expert: an expert's weight slab on one XCD, group-aligned XCD runs, band width against K | `grouped_scheduling.md` §2–§4 |
| XCD placement | Fewer M-blocks per expert than XCDs × band (thin experts): every XCD sees another expert; `nt` on C | `grouped_scheduling.md` §2, `grouped_lowprec.md` §7 |
| XCD placement | Skewed routing: is the skew in the tile count or in the cost per tile | `grouped_scheduling.md` §1, `grouped_workflow.md` §3 |
| XCD placement | One grouped launch trails a dense GEMM of the same size: L2 locality across experts | `grouped_scheduling.md` §9 |
| Low precision | Preshuffled operands and MX scales | `../gemm/datapath.md` §10, §11 |
| Low precision | MX scales of the NT layout: four E8M0 per dword, the scale-load floor, tile heights fixed by the scale layout | `grouped_lowprec.md` §3–§4 |
| Low precision | A per-call scale repack or activation quantizer: what bounds it, its metadata in one prologue, all expert weights in one launch | `grouped_lowprec.md` §2, §5 |
| Low precision | A row pitch ≡ 64 mod 128 B on the contraction axis: pad K | `grouped_lowprec.md` §6 |
| Low precision | The gate/up activation fused into stage 1's epilogue; padded gate and up halves | `grouped_lowprec.md` §10, `../gemm/epilogue.md` §8 |
| Low precision | The C store and its cache bits | `grouped_lowprec.md` §7 |
| Low precision | MXFP4 short of 2× MXFP8: the feed per wave at one wave per SIMD | `grouped_lowprec.md` §9 |
| Stage 2 | Atomics or a fixed-order reduce; MXFP4 route rows; the reduce as its own launch | `grouped_sorted_tokens.md` §6, `../gemm/correctness.md` §8 |
| Correctness | Judge a port whose stage 2 uses atomics against its source | `few_tokens.md` §3 |
| Correctness | Gate inputs: skewed routing, experts with no route, many experts (64-bit offsets), row counts off the tile grid | `grouped_correctness.md` §1–§4 |
| Correctness | The SNR band per dtype; a reference that can see the error; prove a ragged body ran | `grouped_correctness.md` §6–§7 |
| Correctness | Deleting a barrier or a wait; a graded `vmcnt` | `grouped_correctness.md` §8 |
| Measurement | A few tokens per call: cold weights, graph replay, the per-node floor, drift between batches | `grouped_sorted_tokens.md` §8, `../gemm/small_m.md` §1 |
| Measurement | The operating point: local experts per GPU, tokens per expert per call, the routing histogram, graph or eager | `grouped_measurement.md` §1–§2 |
| Measurement | A model trace: per-call sizes, FLOPs from tokens × top-k per layer, contention | `grouped_measurement.md` §3 |
| Measurement | Noise per cell under skewed routing; arm order | `grouped_measurement.md` §5 |
| Host path | No device→host reads, graph capture, caches, ragged dims handled in the kernel instead of per-call padding | `grouped_host_autotune.md` §1–§4; § In AKLR below |
| K varying per group | A grouped GEMM that contracts over each group's tokens (a per-expert weight gradient; not part of `fused_moe`): run-time K loop, K tails, split-K over deep groups, beta = 1 | `grouped_variable_k.md` |
| Working order | Where to start on a grouped kernel, and what is usually dead | `grouped_workflow.md` |
| Between GPUs | EP or TP for the MoE layer; the dispatch and combine | `../communication/ep_dispatch_combine.md` |

## Kernels to start from

- **AKLR's own.** `../flydsl/kernels/flydsl_kernel_library.md` § AKLR's own FlyDSL MoE kernels lists the five
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
- **aiter's FlyDSL MoE families** and their knobs: `../flydsl/kernels/flydsl_kernel_library.md` § Other
  FlyDSL families in the baseline aiter (MoE). aiter is the baseline; price it through `fused_moe_baseline`, which
  calls `aiter.fused_moe.fused_moe` (`grouped_sorted_tokens.md` §8).
- **FlyDSL v0.3.2's `kernels/moe/`** and how aiter's copies differ: `../flydsl/moe.md` § grouped_sorted_tokens
  §1.
- **Primus-Turbo's FlyDSL grouped GEMMs** (group offsets over contiguous rows; NT, NN and K varying per group):
  `../flydsl/moe.md` § INDEX Kernel map, and § Expert GEMM (grouped GEMM) below.

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

## Expert GEMM (grouped GEMM)

The grouped-GEMM package, folded in from its own folder: AKLR has no grouped-GEMM op, and its MoE op runs
these GEMMs. Pinned sources: AMD-AGI/Primus-Turbo@ca1fdd67 (primus_turbo/flydsl/{grouped_gemm,utils}); ROCm/FlyDSL@dd8373349 (v0.3.2; kernels/moe).

How to write and optimize a grouped GEMM on gfx950 (MI350/MI355X; the FlyDSL side: `../flydsl/moe.md`): one launch
that runs G independent GEMMs whose sizes differ per group, as in a mixture-of-experts layer. The cards cover three
layouts, in bf16, fp8 with per-tensor scales, MXFP8 and MXFP4, with GLU/dGLU epilogue fusion:
- M split per group, each group with its own weight: NT, `A[M_total, K] × B[G, N, K]`. Every MoE expert GEMM
  (stage 1 and stage 2, at any token count) has this layout, and so does a forward pass.
- M split per group, contracting the weight along its other axis: NN, or NT on a transposed copy (an input
  gradient, dgrad).
- K varying per group: TN, the contraction runs over each group's tokens (a weight gradient, wgrad).

Expert GEMMs whose rows are gathered by sorted token ids have one card of their own, `grouped_sorted_tokens.md`;
the other cards apply to them wherever they treat the NT layout, at the token counts each section states.

A grouped GEMM is a dense GEMM plus a group dimension. The main loop, MFMA atoms, LDS layout, DMA, scheduling
hints, epilogue basics and the measurement protocol are the same as for dense GEMM and live in `../gemm/`; the
cards here cover what the group dimension adds: mapping tiles to groups, ordering them under token skew, the
per-tile fixed cost that small groups expose, K varying per group, scales and quantization per group, the host
path, and the gates and rulers that a skewed distribution needs.

**Three sources, used together.**
- **Primus-Turbo's FlyDSL grouped GEMMs** (`primus_turbo/flydsl/grouped_gemm/` at main `ca1fdd67`), which take
  group offsets over contiguous rows rather than sorted token ids. They keep group metadata on the device, launch
  a static upper-bound grid, find each tile's group
  on the device (a lane-resident prefix table in the MXFP4 kernels and in the fp8 kernels above eight groups),
  order tiles across XCDs, split deep variable-K groups with a deterministic reduce, and race most
  configurations on balanced synthetic data. The cards cite them by file and line.
- **FlyDSL's own MoE GEMMs** (`kernels/moe/` in v0.3.2): sorted-token, per-M-block expert ids, persistent
  stage 2.
- **Measured grouped-GEMM work**, marked `*(field, E<n>)*`: what moved wall time and what did not.

- FlyDSL facts are stated for v0.3.2.
- Numbers marked `*(field, E<n>)*` are measured values; `E<n>` names the line under the card's `## Evidence` that
  gives the conditions. Read them as magnitudes, not targets.
- Constants in the Turbo and FlyDSL kernels the cards cite (tile tables, band widths, group-count limits, adoption
  margins) were tuned on specific shapes and distributions. Reuse the mechanism and re-derive the number.
- Signed percentages of wall time, throughput or a score are speed changes (+ = faster) unless the sentence says
  faster or slower; a sign next to a named quantity (cycles, sclk, a counter, bytes, registers) is the change of
  that quantity.
- Occupancy is stated as waves per SIMD or workgroups per CU, explicitly.

In FlyDSL: `../flydsl/moe.md` § INDEX Expert GEMM (grouped GEMM).

### In AKLR

AKLR has no grouped-GEMM op. Its MoE op is `fused_moe` (the sglang plugin's MoE ops take a whole MoE block): start
at § Techniques that apply above, which routes each problem at any token count to its card section (at a few tokens
first `few_tokens.md`, then `grouped_sorted_tokens.md`); AKLR's own FlyDSL MoE kernels are listed
in `../flydsl/kernels/flydsl_kernel_library.md` § AKLR's own FlyDSL MoE kernels. Where a card disagrees with
`aklr/ops/README.md` (§ The kernel section and ABI), the README wins
(`../SKILL.md` § In AKLR). What it changes here:
- **Cache keys.** A compile function caches by the Point's full shape, M included: one compile per Point. The rule
  against keying on `M_total` (`grouped_host_autotune.md` §3) is for callers that compile once for many token
  counts, whose token count changes from call to call. Bound a cache with `@functools.lru_cache(maxsize=...)`, not
  with module-level state such as MX4's `_bound_caches`.
- **No tuner in launch code.** A Point compiles on its first, uncaptured call, every `beta` variant included, and
  later calls may run inside a graph capture. The capture-time tuner of `grouped_host_autotune.md` §2, the
  races on a call's own data of §5 and the beta ranking of §6 become a configuration chosen from the shape in the
  compile function, raced offline; §8's logged build errors stay in that offline script (sources may not `print`
  or do I/O).
- **Nothing derived from an argument survives a call.** The `(id(weight), weight._version)` cache of
  `grouped_host_autotune.md` §3 and quantizing weights once per weight update (`grouped_lowprec.md`
  §1) are not available: repack on every call. The harness overwrites weights with `.data.copy_()`, which keeps
  both the address and `_version`. Handle ragged dimensions in the kernel (`grouped_host_autotune.md` §4).
- **Scratch.** Per-shape scratch must be "rewritten whole by every call before it is read" (README). The persistent
  split-K bands of `grouped_variable_k.md` §9, sized for the deepest geometry, do not meet that as written:
  allocate them per call, or have every call write every band the kernel can read.
- **Every output element is checked.** Write all of the output, zeros for empty groups included; the "fingerprint
  only the region the contract writes" workaround of `grouped_correctness.md` §3 does not apply.

### Start here

| You are… | Read |
|---|---|
| starting on a grouped GEMM, or deciding what to try next | `grouped_workflow.md` |
| mapping tiles to groups: grid size, group lookup, empty groups, 64-bit offsets | `grouped_group_mapping.md` |
| ordering tiles across XCDs and groups, handling token skew, choosing persistent or not | `grouped_scheduling.md` |
| choosing a tile, occupancy or boundary body per layout; pricing small groups | `grouped_tile_pricing.md` |
| working on a grouped GEMM whose K varies per group (the contraction runs over each group's tokens), split-K over deep groups, beta=1 accumulation | `grouped_variable_k.md` |
| adding fp8, MXFP8 or MXFP4: scale preshuffle and packing, quantization, stores, GLU fusion | `grouped_lowprec.md` |
| launching without host syncs, CUDA graphs, caching, autotuning across distributions | `grouped_host_autotune.md` |
| benchmarking grouped kernels: shapes, distributions, deployment conditions, noise | `grouped_measurement.md` (with `../gemm/measurement.md`) |
| writing or trusting a correctness gate for grouped kernels | `grouped_correctness.md` (with `../gemm/correctness.md`) |
| working on a MoE expert GEMM over sorted token ids (fused MoE stage 1 or 2, any token count): the sort, work mapping, cache policy, stage 2 | `grouped_sorted_tokens.md`, then § Techniques that apply above for the other cards |
| working on the main loop, LDS, DMA, MFMA atoms or the epilogue itself | `../gemm/INDEX.md` |
| choosing EP or TP for a MoE layer at a few tokens per GPU, or working on the dispatch and combine between GPUs | `../communication/ep_dispatch_combine.md` |

### Kernel map

In FlyDSL: `../flydsl/moe.md` § INDEX Kernel map.

### Cards

| Card | One line |
|---|---|
| `grouped_workflow.md` | what the group dimension changes, working order per pass, what is usually dead |
| `grouped_group_mapping.md` | device-side group metadata, upper-bound grids, tile→group lookups and their cost, empty groups, 64-bit bases |
| `grouped_scheduling.md` | XCD and L2 order across groups, where the skew lives, longest-first, persistent vs one tile per workgroup |
| `grouped_tile_pricing.md` | tile and occupancy per layout, boundary and tail bodies, the per-tile fixed cost of small groups |
| `grouped_variable_k.md` | K varying per group: run-time K loops, scale timing, store-bound short groups, deterministic split-K, beta=1 accumulation |
| `grouped_lowprec.md` | fp8/MX scale preshuffle and packing, quantization share, K padding, store forms, GLU/dGLU fusion |
| `grouped_host_autotune.md` | no host syncs, graph vs eager launch, caching, races on balanced data, beta and ranking |
| `grouped_measurement.md` | shapes and distributions, deployment conditions, FLOP accounting, per-cell noise |
| `grouped_correctness.md` | skewed and empty groups in the gate, out-of-bounds reads, MX probes, determinism vs correctness |
| `grouped_sorted_tokens.md` | sorted-token MoE expert GEMMs at any token count: work mapping, persistent stage 2, cache policy, a few M-blocks per expert |

## Evidence

- E1 [2026-10, gfx950, FlyDSL 0.3.4.1, N=1, AKLR CI] speedup over aiter at m 1: 2.472 (257 experts, 1fee59821837c26f), 1.258 (33 experts, ef15e87894a8ae06), 1.353 (65 experts, 1bf41ccba6750300), 3.015–3.359 (clamped SwiGLU, four files; 9dcb4d796b597d0b 3.089, a5a343437ee191a7 3.015 from 102877ca0) (5571f4589, CI run 40613440602)
- E2 [2026-10, gfx950, FlyDSL 0.3.2, N=1, bench_cudagraph: graph replay, 1 GiB flush per call, Kineto kernel time, median; one run per arm] 39.09–102.04 µs per call with a separate sort node, 34.38–95.67 µs with it folded into stage 1, at m 8–64 (257 experts, inter 256; local:/workspace/code/gpt_oss_docker/scratch/glm52_sikl_port/evidence/moe_triage_AB.md:13)
- E3 [2026-09, gfx950, FlyDSL ?, N=?, wall clock of a 75-layer MoE step over 8 weight sets against a read-roof kernel in the same process; per-layer times from a minibench] aiter's MXFP4 MoE (`mfma_moe1`/`mfma_moe2`, GLM-5.2 TP4/EP4: 64 local experts, inter 2048, hidden 6144) at 60 tokens: 5.77 TB/s against a 6.27 TB/s roof, 92.0%; 10 → 60 tokens 82.2 → 195.6 µs per layer (2.38×) with 20 → 52 experts hit (2.6×) (local:/workspace/code/gpt_oss_docker/.claude/memory/reference_glm52_moe_bandwidth_bound.md:21; :60; :81)
- E4 [2026-10, gfx950, FlyDSL 0.3.4.1, N=1, AKLR CI] speedup over aiter at m 64–4096: 1.030, 1.147, 1.109, 1.257, 1.609, 1.081, 1.070 (fused_moe_shuffled_16x16_model_dim6144_n_experts33_topk9_inter_dim2048, file ef15e87894a8ae06; 5571f4589, CI run 40613440602)
- E5 [2026-10, gfx950, FlyDSL ?, N=1, `python -m aklr validate --base origin/main` run locally on one MI355X: graph replay, 1 GiB flush per call, one-sided lower bound against aiter; touched experts counted from the same inputs] speedup over aiter at m 8–112: TP 1.373, 1.287, 1.243, 1.194, 1.168, 1.084, 1.069, 1.061, 1.058, 1.054, 1.043, 1.039, 1.038; EP8 1.592, 1.427, 1.347, 1.281, 1.220, 1.278, 1.282, 1.276, 1.262, 1.253, 1.245, 1.229, 1.240; weight bytes of the touched experts over the median candidate time 6.04–6.20 TB/s (TP, m 40–112) and 6.20–6.45 TB/s (EP8, m 32–112), aiter 5.0–5.97 TB/s; read roof 6.5 TB/s from a read-only kernel (the eight K3 kernel files of branch kyle/k3-moe-kernels; local:/shared_nfs/kyle/moe/k3rebase/val2/report.json; local:/shared_nfs/kyle/moe/kf/scan/touched.json; local:/workspace/code/gpt_oss_docker/.claude/memory/reference_mi355_measured_ceilings.md)
- E6 [2026-10, gfx950, FlyDSL 0.3.2, N=1, eager calls on the validate inputs, L2 flushed between calls, torch.profiler kernel times; useful FLOPs 2 · routes · N · K per GEMM] at 16384 tokens, the delivered kernels against aiter: TP stage 1 732.4 µs = 1970 TF/s (aiter 1008.2 µs = 1431), stage 2 535.8 µs = 1347 (776.8 µs = 929), other kernels 264.5 µs with the reduce 190.4 (480.3 with 374.8); EP8 stage 1 762.6 µs = 1889 (987.2 µs = 1459), stage 2 408.1 µs = 1764 (568.3 µs = 1267), other 57.5 µs (508.5); at 9397 tokens TP 1701 / 1200, EP8 1660 / 1511 TF/s (local:/shared_nfs/kyle/moe/kf/flops_run/r2/tp8.txt; local:/shared_nfs/kyle/moe/kf/flops_run/r2/ep8.txt; script local:/shared_nfs/kyle/moe/kf/flops_run/prof_flops.py)
- E7 [2026-09, gfx950, FlyDSL ?, N=?, FlyDSL's own GEMM bench and hipBLASLt on one idle MI355X] dense fp8 GEMM at M=131072: N5632 K2560 2288 / 2474, N6144 K3072 2396 / 2602, N2560 K2560 2167 / 2488, N3072 K3072 2369 / 2664 TF/s (FlyDSL / hipBLASLt); fp8 16384³ 3319 / 2813 (local:/workspace/code/gpt_oss_docker/.claude/memory/reference_mi355_measured_ceilings.md)
- S1 aklr/ops/README.md:76 a schema file with empty `sources` is a definition: loaded and audited, not served or benched
