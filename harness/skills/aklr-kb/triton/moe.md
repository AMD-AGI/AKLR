---
title: "MoE layer plugins in Gluon on gfx950: decomposition by token count, routing, sorting, expert GEMMs, combine"
kind: topic
ops: [moe]
dtypes: [mxfp4, fp8, bf16]
regimes: [decode, prefill]
gens: [gfx950]
triton: 3.8.0
---

# MoE layer plugins in Gluon (gfx950)

How AKLR's Gluon kernels compute a rank's router-and-expert portion of a SGLang MoE layer call in a few launches, and which design to copy at which
token count: the Triton side of [../moe/INDEX.md](../moe/INDEX.md), whose measured MoE-kernel numbers are from FlyDSL unless it explicitly names another language or a vendor-library comparison; the numbers here
are these Gluon kernels' on MI355X. Per-file templates:
[kernels/kernel_library.md](kernels/kernel_library.md) §7–§10. Here m is `n_tokens`, a route one (token, expert)
pair, and a four-digit file name abbreviates its source UUID. Configurations are shape-specific examples;
original kernel and compiler citations appear under Evidence.

## Route here when

- You write or tune a kernel for `deepseek_v2_mxfp4_moe@1.0`, `minimax_m2_fp8_moe@1.0`, `minimax_m2_mxfp4_moe@1.0`
  or one of the three `qwen2_moe_*` contracts.
- You decide a MoE call's launches per token count and where top-k, sort, activation, combine and shared expert go.
- You port a `../moe/` mechanism to Gluon, or need the Gluon form of atomic tickets, a device-side job list or a
  recomputed top-k (preshuffled MXFP4 weights and shuffled E8M0 scales: [gluon/mfma.md](gluon/mfma.md) §7,
  [gluon/memory.md](gluon/memory.md) §13).
- A constant change in one of these files fails to compile or crashes, or a knob you expected to matter does nothing.

## 1. The plugins and what they replace

| Plugin (`aklr/plugins/sglang/models/`) | Replaces (SGLang 0.5.22) | Contract, scenario | Points (`n_tokens`) | Files | Launches per call (≤ 16, 32–256, ≥ 1024 tokens)  |
|---|---|---|---|---|---|
| `deepseek_v2` | `DeepseekV2MoE.forward_normal` (S1) | `deepseek_v2_mxfp4_moe@1.0`, `…_inter_dim256` (rank intermediate dimension 256) | 1, 2, 4, 8, 16, 32, 128, 256, 1024, 2048, 3072, 4096, 4192 | 9 | 3–4, 5, 6 |
| | | same contract, `…_inter_dim512` (rank intermediate dimension 512) | 1, 2, 4, 8, 16, 256, 1024, 2048, 3072, 4096, 4192 | 10 | 3–4 (5 at 16), 6, 6 |
| `minimax_m2` | `MiniMaxM2MoE.forward_normal` (S1) | `minimax_m2_fp8_moe@1.0`, `…_n_experts256_inter_dim384` | 1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 2048, 4096, 8192, 16384 | 5 | 3, 5–6, 6–7 |
| | | same, `…_n_experts64_inter_dim1536` | same 14 | 4 | 4 (6 at 8–16), 5–6, 8 |
| | | `minimax_m2_mxfp4_moe@1.0`, `…_n_experts256_inter_dim512` | 1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 2048, 4096, 6144, 8192, 16384 | 12 | 3, 5, 6 |
| | | same, `…_n_experts64_inter_dim1536` | same 15 | 11 | 3, 5, 7–8 |
| `qwen2_moe` | `Qwen2MoeSparseMoeBlock._forward_router_experts` (S2) | `qwen2_moe_fp8_router_experts@1.0` | 1, 2, 4, 32 | 4 | 4, 7 |
| | | `qwen2_moe_router@1.0` | 8, 16, 64, 128, 256, 2232, 4096 | 7 | 2 |
| | | `qwen2_moe_fp8_experts@1.0` | 8, 64, 128 | 3 | 3, 5 |

- **Terms.** A route's *slot* is its place 0 … k−1 in its token's top-k, the reference's order (the kernels' `rank`;
  DeepSeek's shared expert is slot 8, Qwen2's slot 10); an expert's global *id* indexes the router; its local `w1` index (W13 below; the
  contracts' "expert slot") is `id - expert_offset` on an expert-parallel rank, otherwise `id`; a route's place in the sorted buffer is its *row*; a *rank* is an expert-parallel rank.
- **Shapes and weights.** DeepSeek (model dim 6144): 256 routed experts, top-8 by sigmoid(bf16 logit) + correction bias,
  each route weighed by its unbiased sigmoid, renormalized over the 8, times `routed_scaling_factor` 2.5 (the bias only
  selects); the shared expert, id 256, is a ninth route at weight 1; MXFP4 weights, fp32 activations in the reference
  (quantizing them to MXFP4 moves `out` by rel L2 0.21). MiniMax (3072): 256 experts, fp32 router weight,
  top-8 by sigmoid + bias, weighed by the unbiased sigmoids renormalized to sum 1 (no factor, no shared expert). Qwen2
  (4096): 512 experts, top-10 by bf16 logit, the lower id first on a tie, weighed by the softmax of the 10 chosen
  logits; the shared expert, id 512, by sigmoid(hidden · `shared_gate_weight`); fp8 weights, 128 × 128 block scales.
- **The `n_experts64` scenarios** are an expert-parallel rank holding 64 of MiniMax's 256 experts; `out` is the rank's
  share (`_RANK` in `aklr/plugins/sglang/models/minimax_m2/contract.py`): top-k over all 256, weights normalized over
  all 8 routes, the routes outside `expert_offset` … `expert_offset` + 63 adding nothing (a token keeps about 2;
  estimated, uniform routing). b065 compacts a token's local routes with a scan, 0982 flags them beside each weight for
  a masked load in its combine, 3df5 runs one program per local expert (S34, S21). `expert_offset` is 192 in every
  workload, but every input set except one numeric set runs its default, 0, and a kernel must be right over its declared
  range (`aklr/ops/README.md` § Schema fields): take it from `run` ([authoring.md](authoring.md) §5).
- **Qwen2 splits the layer over three ops.** The route tries the router+experts op, then the router op and the
  experts op, then SGLang's experts (S2); at 8 tokens a call is the router's 2 launches plus the experts' 3.
- **What they beat.** Each baseline reconstructs SGLang's aiter path (`baseline.py` of each plugin); MiniMax explicitly does not replay SGLang's process-wide small-batch sort patch. DeepSeek and MiniMax: a
  router GEMM (MiniMax's on fp32 activations), a top-k kernel, the shared-expert append (DeepSeek) and aiter's
  `fused_moe`, which sorts and runs its two stages. Qwen2: when `router_weight.shape == (512, 4096)` and the imported SGLang `smallm_router_supported(hidden_states)` accepts the call, its gfx950 small-M router appends
  the gated shared expert itself; otherwise the gate, `fused_topk` and the append; then SGLang's gfx950 small-M MoE kernel
  where it accepts the call, else `fused_moe` (S2). Configurations the ops do not take stay on SGLang (all-to-all
  backends, EPLB, deterministic inference: S1).
- **One file per Point or a few neighbours.** The configuration is a Python function of m (a constexpr in 335 of 471
  launches): each Point compiles its own binaries on its first call  ([tuning.md](tuning.md) §14). Most
  files keep branches for other token counts (§12); Qwen2's experts files 5c35, 8698 and d425 assert their one Point.
- **All 65 schemas are `bounded`**: runs may differ, accuracy is each contract's output bound's job (the MoE `out` uses SQNR; Qwen2's router weights use a tolerance bound and its ids must match exactly) ([numerics.md](numerics.md) §1,
  §8); the kernels let atomic order place routes and sum values in a fixed order (§6, §10).

## 2. Pick the decomposition by token count

`gl.barrier` synchronizes a CTA, not a grid (S24). A later launch gives a simple producer/consumer
boundary. Same-launch last-arriver or cooperative counter schemes need a correct publication protocol and
forward-progress guarantees; their benefit depends on the dependency and timer, not merely launch count.
The served source decompositions are

| Tokens | One program does | Launch sequence of one served file  |
|---|---|---|
| ≤ 16 | one route × one column tile (× one K split), no sort; at 8–16 tokens 3df5 and 1c17 (64-expert ranks) and 3c6e (DeepSeek `inter_dim512`, 16) run one local expert × one column tile over all its tokens instead (§4.1) | DeepSeek d3bb, 2 tokens (S4): `_router_pair_packet` (304 programs: a full-K router program per expert, the rest MXFP4 quantization) → `_select_and_up` (2306: M = 2 selection + 2304 gate-up programs; the routed-route ones recompute their expert, §4.1) → `_activate_quantize` (72) → `_token_local_down` (768: down and weighted combine) |
| 32–256 | one expert job × one column tile: 16–128-row bodies in the DeepSeek/MiniMax files and Qwen2 experts-only files; Qwen2 `99f4` at 32 tokens uses 4-row routed jobs and 16-row shared-expert jobs | DeepSeek 1204, 256 tokens (S5): `_router_projection` → `_select_and_quantize` (top-8, relaxed atomic tickets, hidden quantization) → `_expert_tiles` gate-up (each program derives its job, §4.2) → `_expert_tiles` down → `_reduce_parts` (fixed-order combine) |
| ≥ 1024 | one tile of a sorted, padded expert segment (all 32 Points sort before the gate-up; no unsorted design was measured) × one column tile: nominally 64 or 128 rows (MiniMax; f770 can execute a 32- or 64-row tail body at run time), or a height of 16–256 rows picked per tile at run time (DeepSeek, §7) | MiniMax MXFP4 8411, 16384 tokens: `_router_projection_7` (also zeroes counters, fills sentinels) → `_select_partitioned_2` (top-8 + tickets) → `_scatter_quantize_inline_2` (rows + hidden quantization) → `_gate_up_scale_preload_fp4` → `_down_reuse_fp4_2` → `_combine_rows_3` |

Use r = m·k/E as the mean routes per expert. The identity m = r·E/k converts coordinates, not a
performance threshold: skewed routing changes touched experts, tile fill and tails.

- **Small decode: measure fusions on the whole call.** Selection recomputation, full-K SwiGLU/quantization
  epilogues and down+combine can remove boundaries. Recomputing in every consumer can cost more than the
  launch saved. Full-K ownership enables an epilogue but can lose to a K-split gate-up; combined gains are
  not the sum of isolated launch savings. Synthetic fp8 VALU results do not establish production MXFP4 MFMA gains.
- **32–256 tokens: group routes by expert.** Take relaxed atomic tickets and open a job every BM tickets,
  or let GEMM programs derive jobs from counts (§4.2, §6). Uniform top-8-of-256 routing touches about
  `1 - (31/32)^m` of experts, an estimate rather than a fact for the schema's biased routing distribution.
- **Prefill: sorted grouped GEMM.** Counting-sort into padded expert segments; consider variable-height
  tails, operand reuse in down and fixed-order combine (§6–§10). Compare these candidates rather than infer
  a sorted-design optimum from the existing dispatch.
- **Use the right launch ruler.** A trivial added kernel reads about 4 µs on AKLR's profiler-based timer,
  versus about 1.8 µs wall time in a bare graph replay (`authoring.md` §7). Small calls can be
  launch/latency dominated; at larger m price router, expert weight reads and route-row write/read traffic
  separately. The full MoE call, not an MFMA-rate estimate, decides the fusion trade.

## 3. Put independent work in one grid

When two phases read only call inputs, they can share a program-id-role grid; the scalar role test is uniform.
Fewer launches can help the profiler-based timer without helping graph replay. Role-grid fusion is a
candidate for independent router/quantization work, not a guaranteed scheduling win.

No DeepSeek or MiniMax kernel is named as a quantizer alone
. In 5751 (1 token, S3) 192 one-wave programs each compute a 16 × 16 router tile over one K split of 512
(16 expert tiles × 12 splits) beside 24 programs that quantize the token to MXFP4:

```python
program = gl.program_id(0)
if program < 256 // BN * SPLITS:
    tile = program % (256 // BN)
    split = program // (256 // BN)
    ...
    gl.store(Y + (om[:, None] * SPLITS + split) * 256 + on[None, :], acc, om[:, None] < M)
else:
    quant = program - 256 // BN * SPLITS - (1 if GROUPED else 0)
    ...
    routed, routed_scale = _quantize_groups(values, False)
    shared, shared_scale = _quantize_groups(values, True)
```

- Other role splits: programs `< M` select and publish the routes while the rest run the gate-up (d3bb, S4); one CTA
  sorts in LDS while the others quantize (Qwen2 d425; 8698 splits its roles by program id too); the router programs of
  an early launch also zero the counters and fill the sentinels a later launch reads (8411: each router program
  initializes a 256-entry stripe after its tile, S6; 1204: router program 0, S5).
- One binary serves every role and shares its resource allocation. A cheap role can inherit high VGPR/LDS
  use from a heavy one; this is a resource constraint, not proof it will run slower. Inspect the binary and time
  the actual mixed grid.

- DeepSeek quantizes the token twice, once per exponent rule (routed, shared): [numerics.md](numerics.md) §3.

## 4. Drop a launch by recomputing its result

Recompute only where the duplicated decision is cheaper than storing it and adding a dependency launch.
Gate-up-only recomputation can publish ids/weights for later stages; repeating top-k in gate-up, down and
combine can instead regress. Benefit depends on expert count, k, number of consumers and GEMM decomposition.
Deriving jobs from counts is another form of recomputation, with its own costs (§4.2).

### 4.1 Top-k in every gate-up program (decode)

At 1 token 5751 runs 1153 programs in `_select_and_up`: program 0 publishes `Ids`/`Weights` for the down launch, the
1024 routed-route programs (8 routes × 4 K splits × 32 column tiles) each recompute their route's expert, the 128
shared-expert ones take id 256 (S3). A recomputing program sums the router's 12 fp32 split partials in split order,
rounds once to bf16 (the reference's rounding point, §5), adds the bias to the sigmoid and runs slot + 1 elections:

```python
@gluon.jit
def _expert_at_rank_peeled(Logits, Bias, row, rank, SPLITS: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([4], [64], [1], [0])
    e = gl.arange(0, 256, layout)
    logits = gl.load(Logits + row * SPLITS * 256 + e)
    for part in gl.static_range(1, SPLITS):
        logits += gl.load(Logits + (row * SPLITS + part) * 256 + e)
    logits = logits.to(gl.bfloat16).to(gl.float32)
    probability = 1.0 / (1.0 + gl.exp(-logits))
    score = probability + gl.load(Bias + e).to(gl.float32)
    available = gl.full((256,), True, gl.int1, layout)
    idx = _elect_rank(score, available, e)
    for j in range(rank):
        available = available & (e != idx)
        score = gl.where(e == idx, -float("inf"), score)
        idx = _elect_rank(score, available, e)
    return idx
```

`_elect_rank` (S3) is the ballot election of §5.3; writing it: [gluon/wave_ops.md](gluon/wave_ops.md) §6.2.

- **Cost and order.** A program pays slot + 1 rounds. At 1 token 5751 maps job j to slot `7 - j`, so the costliest
  selections start first (`route = gl.where(job < 8, 7 - job, 8)`, S3); d3bb at 2 tokens and 96e5 at 4–16
  (`RANK_MAJOR`) run slot 0 first, equal-slot routes adjacent (S4, S37). No file says why; no Point compares them.
- **Where.** The DeepSeek and MiniMax decode files, and the Qwen2 router+experts files at 1–4 tokens, whose W13
  programs replay the top-k pops on the 512 router keys (S26). Variants at 8–16 tokens: the MiniMax MXFP4 64-expert
  rank runs one program per local expert that keeps the tokens for which fewer than 8 experts beat it (S21), the FP8
  one adds a selection launch and runs one gate-up program per local expert (S30); DeepSeek `inter_dim512` at 16 builds
  per-expert token bitmasks with `buffer_atomic_or` in a selection launch (S20).
- **Copy the mechanism, not the speedup association.** High stored small-M ratios occur in files that
  recompute routing, but that does not attribute the gain to recomputation. Compare a separate selection
  launch against the same complete expert path at your Point.

### 4.2 GEMM programs that derive their own job

At 16–256 tokens five files skip the job-list launch: each gate-up program finds its tile from the per-expert counts.
1204 (256 tokens, S5) scans the counts; `PAD` (m rounded up to the tile height `TM`) is an expert's region (§6):

```python
e = gl.arange(0, 256, scan_layout)
count = gl.load(Counts + e * (32 if M <= 128 else _NATIVE_COUNT_STRIDE))
chunks = ((count + TM - 1).to(gl.uint32) // TM).to(gl.int32)
end = gl.associative_scan(chunks + (count << 10), 0, _add)
begin = (end & 1023) - chunks
row_begin = (end >> 10) - count
chunk = tile - shared_jobs - begin
packed = (
    (e * (PAD // TM) + chunk) * 32 + gl.minimum(TM, count - chunk * TM) - 1 + (row_begin + chunk * TM << 17)
)
descriptor = gl.max(gl.where((chunk >= 0) & (chunk < chunks), packed, -1), 0)
```

Two prefix sums share one `gl.associative_scan` by bit packing, guarded by `gl.static_assert` on the ranges, and
`v_readfirstlane_b32` in inline assembly makes the result wave-uniform. The descriptor holds live rows − 1 in bits 0–4,
the tile `e · PAD/TM + chunk` in bits 5–16, its first activation row from bit 17; the column-0 program stores it for the
down launch (S5). The others: wave ballots and popcounts over per-expert token bitmasks (DeepSeek `inter_dim512` 3c6e at 16 tokens,
`_mask_descriptor`, S20), a DPP prefix in inline assembly (DeepSeek a1c4), a scan and `gl.min` (MiniMax 1d27), a
persistent W13 grid (MiniMax FP8 94af at 256); 5cac reads its overflow tiles at 32 tokens, and every tile at 64, from a
list its selection launch writes. These Points measured 1.030–1.194, inside the 1.030–1.275 of all
32–256 designs: no evidence that deriving the job beats a job list written by the selection launch. Lookup forms and the
`readfirstlane` rule: [../moe/grouped_group_mapping.md](../moe/grouped_group_mapping.md) §5, §6.

### 4.3 The global prefix in every scatter program

A route's row needs the exclusive prefix of every expert's padded count: each program of 8411's scatter launch reloads
all count chunks, pads, scans and adds the earlier chunks' counts; program 0 stores `Total` (S6):

```python
padded = gl.cdiv(gl.sum(counts, 0), BM) * BM
vector: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
padded_vector = gl.convert_layout(padded, vector)
start_vector = gl.associative_scan(padded_vector, 0, _add) - padded_vector
start = gl.convert_layout(start_vector, gl.SliceLayout(0, layout))
chunk = gl.program_id(0) * 256 // CHUNK_BLOCK
bases = start + gl.sum(gl.where(c[:, None] < chunk, counts, 0), 0)
if gl.program_id(0) == 0:
    gl.store(Total, gl.sum(padded_vector, 0))
```

Both `convert_layout`s are needed: `gl.associative_scan` on the `SliceLayout` that `gl.sum` returns fails to lower in
Triton 3.8.0 ("error: TODO: unsupported scan layout"); scan a 1-D `BlockedLayout`
([debugging.md](debugging.md) §4). An atomic-ticket sort thus costs two launches (selection, scatter), not three. The
The FlyDSL `4fbd` analog repeats the full M×9 route scan, expert histogram and padded-block prefix in every stage-1 workgroup; each live workgroup then ranks only its own expert’s items and retains its 16-row block: `../moe/decode.md` §2.

## 5. Router GEMV and top-k

### 5.1 Compute the logits at the reference's rounding point

| Contract's routing | What the kernels compute | Where |
|---|---|---|
| DeepSeek: logits in bf16 (what SGLang's gate returns on ROCm), sigmoid, plus an fp32 bias | bf16 MFMA tiles (16×16×32) write fp32 split-K partials; the consumer sums them in split order and rounds once to bf16 | 5751 (S3) |
| | at 2 tokens a full-K dot per expert with `gl.dot_fma`, both tokens sharing the weight load (read from the source); `_router_pair_packet` compiles to 48 `v_dot2c_f32_bf16` and no MFMA  | d3bb (S4) |
| Qwen2: bf16 logits, top-10 by logit, the lower id on a tie | the same split-order sum and one bf16 rounding | c388 (S9) |
| | router+experts: float64 products (f64 VALU and no f64 MFMA at 1–4 tokens, `v_mfma_f64_16x16x4_f64` at 32), rounded once to bf16 with a midpoint fix against double rounding (on 512 crafted near-midpoint logits f64 → f32 → bf16 erred on 260, 7d7e's helper on none) | 7d7e (S26), 99f4 (S11) |
| MiniMax: fp32 router weight, fp32 logits | full-K VALU at small M; fp32 MFMA or three BF16 terms at larger M, with source-specific residual correction | e75e (S27), 8a5a (S10), 5cac, 8411 |

Preserve the reference's rounding point and tie order (`numerics.md` §5). Qwen2 requires actual ids and
slot order, not just similar logits or weights. Rounding each split before summing can change selected ids;
sum fp32 partials in the specified order and round once. A more accurate logit can still cross a top-k tie,
and skipping a rounding point need not improve every output.

**Three BF16 terms are an arithmetic candidate, not an accuracy theorem.** Splitting FP32 router weights
and using three BF16 products can reduce matrix-core work, but the splitting/memory cost must be included.
Finite full-logit comparisons support some source branches, not arbitrary inputs or exact top-8 equivalence.
Near-tie inputs can reject a split router even when its logit error is lower than the executed FP32 baseline.
Some source variants include a flagged FP64 residual correction: ordinary random inputs with zero residual
cannot exercise that path. Validate the complete router's ids/weights and measure the whole MoE call before
choosing a precision path.

### 5.2 Size the router grid for bytes in flight

At decode and mid sizes the router streams a few MB of weights. Program count, look-ahead and vector
loads jointly expose bytes in flight; matching nominal bytes alone does not predict the rate, especially
with small grids. Treat a streaming roofline as an estimate, not a router measurement.

One design splits K over small MFMA tiles whose fp32 partials every consumer sums in split
order: 5751's 192 one-wave tiles at 1 token, 1024 one-wave programs in a1c4 and 1204 at 128–256 tokens, MiniMax's 512
CTAs at 128–256 (01ce, 4da3, 8a5a, 94af), the Qwen2 router's 256 CTAs from 8 to 256 tokens (4801 at 8: 1 × 8 splits × 32
expert tiles of one wave; c388 at 256: 8 token tiles × 8 expert tiles × 4 splits of four waves, S9). The other gives
each expert (each token block and expert in e75e, fbd9, 3df5 above 1 token) one full-K program that writes a finished
probability or ordered key, sparing every top-k consumer of §4.1 the split sum (d3bb and the MiniMax decode files e75e,
1c17, 6557, fbd9, b065, f82c, 3df5; e75e `_prepare_keys`, S27). No isolated comparison establishes a universal winner between these forms. The committed 256-CTA Qwen2
router split was faster or tied among tested neighbors, but more look-ahead could improve it. Tune both
split count and look-ahead, preserving the split-sum rounding and scratch dimensions.

Not speed knobs here: the
`amdgpu-sched-strategy` attribute, `amdgpu-ieee false` (accuracy unchanged) and, at 8 tokens (4801), the file's register
path for the hidden tile in place of async staging through LDS (it also changes the weight-load form). At
prefill K is split fewer ways or not at all (DeepSeek full-K, Qwen2 2 splits at 2232 tokens and none at 4096).

### 5.3 Top-k with cross-lane operations

Keep a token's top-k reduction within one wave where practical. Four waves on one token need cross-wave
synchronization every round; separate tokens per wave avoid it. For 256 experts this gave a large measured
selector advantage, but 512 experts and other layouts need their own comparison. The four-wave `147b`
selector emits LDS exchanges and barriers, while `8411` and `b94f` keep their reductions wave-local (S6, S13).
Gluon 3.8.0 has no ballot, lane-read or DPP builtin; sources use inline assembly (`gluon/wave_ops.md` §6).

| Design | One round | Where |
|---|---|---|
| Election by ballot | `gl.max` over the wave; `v_cmp_*_e64` writes the lanes holding the maximum to an SGPR pair, `s_ff1_i32_b64` takes the first, `v_readlane_b32` broadcasts its id; lanes hold consecutive ids, so the lowest id wins a tie | DeepSeek 5751 `_elect_rank` (S3) |
| Keyed min, no inline asm | with retired experts' scores at −inf (as its caller sets them), `gl.max` of the scores, then `gl.min(where(available, where(score == max, e, e + 256), 512)) & 255`: the lowest available id holding the maximum; retired maxima must stay masked | DeepSeek 3c6e `_next_expert` (S20) |
| Per-lane sorted queues | sort each lane's keys once (a 5-compare network on 4 keys, or two 4-deep queues on 8); a round compares only the queue heads and pops the winner's queue | MiniMax FP8 5cac (S25) and e75e `_rank_stable_queue` (S27); Qwen2 c388, whose ballot winner goes into lane `slot` with `v_writelane_b32` (S9) |
| Order-preserving integer keys | the score's bits made monotonic, −0 folded into 0, the inverted expert id in the low bits: one `gl.max` returns the score and the lowest id | Qwen2 7d7e (16-bit bf16 key above a 9-bit id, S26). Two reductions per round instead: MiniMax e75e keys the fp32 score alone (NaN to the largest key, S27) and takes the lowest id with a second `gl.min`; 5cac packs the inverted id into 64 bits (S25) but at 256 experts still reduces the head scores with `gl.max`, then the ids with `gl.min` |
| Packed float64 keys | for finite scores, the same order inside a double's mantissa in [1, 2), so one float compare decides (`v_max_f64` in inline assembly in 94af); fbd9 maps NaNs to NaN-valued keys outside that interval | MiniMax fbd9 (S28), 94af (S29) |
| Packed uint64 keys, partitioned reduce | keys 4 per lane in a [4, 16] layout (a wave's four 16-lane rows); a round pre-reduces each lane's 4 in registers, runs a 16-lane `gl.reduce(…, _argmax_packed)` that compares the scores as floats and the ids on a tie, gathers the 4 row roots from lane 15 of each row and finishes 4-way; the winner's key becomes a tombstone | MiniMax 8411 `_select_partitioned_2` (S6); b94f `_select_wave_rows` on (value, index) pairs (S13) |
| Rank counting, no rounds | one program per candidate expert counts, per token, the experts that beat it (higher score, or equal score and lower id); a count below k selects it and names its slot | MiniMax FP8 1c17 `_select_rank` (S30), exact at 8 and 16 tokens; MXFP4 3df5 (S21) returns a token bitmask: measured M2 keeps 8 via its id tie rule, M8 counts only strictly higher scores and a tied 8th/9th selects 9; M16 not run  |

Election executes slot + 1 argmax rounds; elapsed time need not grow linearly with this operation count.
Queues amortize local sorting, while rank counting trades rounds for expert-wise work. Selection also
renormalizes weights and takes tickets in the same launch. Choose using the actual shape and tie semantics.

- **One call, one tie rule.** Every §4.1 design selects at least twice per call: the program that publishes
  `Ids`/`Weights` and each gate-up program that recomputes its expert (5751: `_select_routes`, `_expert_at_rank_peeled`,
  S3; e75e: a `gl.min` round and per-lane queues, S27; f82c both in one launch, S31). The copies share the score
  (split-order sum, bf16 rounding) and the lower-id rule.
  Keep publishing and recomputing copies consistent in sum order, BF16 rounding, tie rule and producer/
  consumer expert slots. If only the gate-up copy changes, the down stage can interpret a different expert
  as the same route even though each local selection looks reasonable.

- **Read the winner from fixed lanes.** `gl.reduce` combines lane pairs in a butterfly; with a non-commutative combine
  (NaN scores) lanes disagree on the winner (on NaN rows only, in a test), so f82c reads it from lanes 60–63 with
  `ds_bpermute_b32` (S31). e75e/fbd9 canonicalize signed zero and NaN score fields; 7d7e folds signed zero but retains distinct BF16 NaN fields. Treat NaN ordering as a source-specific condition rather than assume a numeric max is commutative.

## 6. Sorting and bucketing (32 tokens and up)

- **Tickets.** In the selection launch each route takes `ticket = gl.atomic_add(Counts + …, 1, sem="relaxed")` and
  stores `ticket << 8 | expert`; the scatter launch turns prefix (§4.3) plus ticket into its row of `Sorted`, and
  segments are padded to the tile height so a tile never mixes experts (8411, S6):

```python
ticket = gl.atomic_add(
    Counts + row * TOPK // COUNT_BLOCK * 256 + ids, 1, sem="relaxed"
)
record = ticket << 8 | ids
...
offset = ticket + gl.gather(base_table, expert, 1)
gl.store(Sorted + offset, route, valid)
gl.store(Experts + offset // BM, expert, valid & (offset % BM == 0))
```

  The 8-bit field holds ids 0–255, MiniMax's 256 experts, and in an int32 record the ticket must stay below 2^23. With
  more ids widen it as the DeepSeek and Qwen2 job words do (`expert | live << 9 | block << 17`, decoded `& 511`, 2422,
  S16; `expert | count << 16`, `& 1023`, 8698, S36), and bound every field with `gl.static_assert` as 1204 does (S5).
- **Pass `sem="relaxed"`.** The consumers run in a later launch. The default `acq_rel` makes the AMD lowering add
  cache invalidates and L2 write-backs around the atomics (S23).
- **Which atomic you get.** Under Gluon `gl.atomic_add` compiles to `global_atomic_add`: `gluon_to_ttgir` has no
  buffer-op conversion (S23). `gl.amd.cdna4.buffer_atomic_add` takes a uint32 or uint64 counter, `buffer_atomic_or`,
  `and`, `xor` int32 or int64 (3c6e's masks are int32): a mismatch fails at trace time, `xchg` later .
- **One atomic per distinct element.** Copies that the layout makes (redundant lanes) are predicated off (S23): 8411's
  `[8]` `ids` in a 64-lane layout take 8 tickets . Copies you build yourself are distinct elements: d861
  widens its 8 winners to an `(8, 8)` tensor with `+ replica[:, None] * 0` to fill a wave, and masks all but one copy
  with `owner = replica[:, None] == 0` (S7), or each copy takes a ticket.
- **Job queues (32–256).** Each expert owns a fixed region of the sorted buffer, as long as the routes it can receive:
  `PAD`, m rounded up to the tile (1204, S5), or `SORTED_STRIDE`, the routes rounded up to 16 (d861, S7). A route's
  row is `expert · stride + ticket`, written in the selection launch with no scan; a second relaxed atomic opens a job
  every 16th (or BM-th) ticket of an expert and writes it to `Jobs` (d861).
- **Persistence is rare.** Qwen2 8698 at 64 tokens loops its W13 programs over a device-counted list on a 3072-program grid
  (complete 16-program route tiles assigned in 256-program cohorts, S36); its down stage uses a separate
  4096-program grid; MiniMax FP8 94af at 256 runs
  `_w13_persistent` on a 48 × 96 grid (S29), the only kernel named persistent at the 96 Points .
- **Sorts without global atomics.** The MiniMax MXFP4 64-expert rank at prefill gives lane j the 64-bit mask of the
  routes in its stripe whose expert is j with seven wave ballots, then popcounts (abc7, S8); the FP8 64-expert rank
  packs four 8-bit counters per word. Both fix the row order; Qwen2's LDS atomics at 64 and 128 tokens
  (`atomic_scatter_add` in LDS, one CTA) do not (8698, S36). No Point compares them with a ticket sort.
- **Graph capture.** Grids depend on shapes only (same grids on two routings at all 82 Points ): a
  worst-case tile bound on the host (MiniMax `cdiv(min(routes · bm, routes + E · (bm − 1)), bm) · bm`; DeepSeek `A +
  (8m − A) // block_m` with `A = min(256, 8m)`, the most experts that can hold a partial tile, S6, S18) and an exit on a
  device-written total (S6; what else breaks capture: [authoring.md](authoring.md) §8). Scratch is `torch.empty`, filled
  by an earlier launch of the call (S6): no file uses `torch.zeros` or a fill, a kernel more at the dispatch floor (§2).
  In FlyDSL: [../moe/grouped_group_mapping.md](../moe/grouped_group_mapping.md) §3, §4.
- **Ticket order is not deterministic.** Use tickets only to place rows whose result does not depend on the row, and sum
  a token's routes in slot order (§10): tickets differed in 19 of 20 runs, the slot-order sum kept one bit pattern
  . `bounded` allows run-to-run differences, `bitwise` would not ([numerics.md](numerics.md) §1, §8).

## 7. Gate-up GEMM

A gate-up program loads its expert's preshuffled weight words straight into the MFMA operand layout
(`assert_trivial=True` compiles for 7e9c: no data moves between lanes ; S12, S24) and issues `mfma_scaled`.
Its E8M0 scale words go through LDS (7e9c's gate-up; 8411's activation scales, preloaded per tile and gathered per K
step, S6), are shifted out of their word per element and converted to the scale layout (d3bb, S4), or load in the scale
layout directly (7e9c's down, S12). At prefill the activation tile the CTA's waves share goes through padded LDS (8411).
Writing those loads, the `shuffled_16x16` offsets and the scale layout: [gluon/mfma.md](gluon/mfma.md) §7,
[gluon/memory.md](gluon/memory.md) §13, [gluon/layouts.md](gluon/layouts.md) §6.3. Particular to MoE:

- **Instruction per weight format** ([gluon/mfma.md](gluon/mfma.md) §2.1–§2.3). MXFP4: `mfma_scaled` on e2m1 with both
  scales (S4), 16×16×128 and no 32×32×64 in the 14 DeepSeek and MiniMax MXFP4 schemas whose Points are all ≤ 16 tokens
  . FP8 at 128 × 128 block scales (MiniMax): `mfma_scaled(a, None, "e4m3", b, None, "e4m3", zeros)` per
  128-K group, the fp32 scales applied after each group (e75e's 4–16-token gate-ups, S27; the dense GEMMs' form,
  [gemm.md](gemm.md) §2). On a rank-3 layout (routes or K splits batched over warps), which `mfma_scaled` aborts on, the
  files run the fold on plain fp8 `mfma` [16, 16, 32] (`v_mfma_f32_16x16x32_fp8_fp8`, half the MACs per cycle)
  , compiled into e75e's `_down_combine` and `_project_w13_internal_packed2` and 5cac's `_w13_local_split`.
  One 0982 `_down_projection_2` variant also uses plain fp8 MFMA, but its `NATIVE=m == 64` path has a rank-2
  layout and is not evidence of a rank-3 requirement  (layouts and Points: S27, S34). In 5c35 (Qwen2 experts, m=8), source AST inspection finds bf16 casts on both MFMA operands
  in `_project_gate_up` and `_project_down_and_sum`. CPU conversion of all 256 fp8 codes preserves 254 finite values
  exactly, both NaNs and signed zero (N=1 CPU run); family GPU execution was not tested  (S19).
- **Gate and up in one column tile.** A BN-column tile holds BN/2 gate rows and the matching BN/2 up rows
  (`n = column * (BN // 2) + n % (BN // 2) + n // (BN // 2) * (N // 2)`, S5), so the epilogue applies SwiGLU (§8).
- **Few rows on a 16-row instruction.** For one token, replicate it over the 16 rows and store row 0 (d3bb, S4); fold K
  into the rows (6557: with `gate_fold` 4, row partition p and column block p read the same K chunk and the diagonal
  blocks are summed, S14); put the routes' experts side by side along N and keep the diagonal (§9); or swap A and B so
  the weights fill M (Qwen2 5c35). The 4-row instruction ran 1.155× the useful MACs of 16 padded rows :
  Qwen2 99f4 at 32 tokens runs routed tasks of at most 4 rows on `instr_shape=[4, 64, 64]` (`v_mfma_f32_4x4x4_16b_f16`),
  the shared expert on 16×16×32 (S11). Register maps: [gluon/mfma.md](gluon/mfma.md) §8, §4.
- **Gate-up grid at 1–2 tokens: not simply wider.** d3bb sums split-K partials in `_activate_quantize` (S4). At m=2, splits 4→1 (2306→578 one-wave programs) took 1.079× call time; 4→8 (4610) 1.050×. At m=1, 5751 1153→577 took 1.020×, →2305 1.028×; 7e9c 1728→864 took 1.035×; e75e column tile 8→16 (385→193 four-wave programs) 1.031× (N=1 run, 4 alternated rounds/arm; scratch and reducing launches changed consistently, all arms accuracy-passed). The older d3bb 4→2 (2306→1154) comparison took 1.004× (2.183→2.175 speedup), with paired CI 0.991–1.001 on the reciprocal ratio: unresolved, not proof from sample spread. These Points establish neither a universal widest-grid rule nor MoE bytes-in-flight sizing; the new case's byte count stops in expert-selection loads . - **Tile height at run time (DeepSeek).** A descriptor carries the live row count and the program branches to a body compiled for 16, 32, 64 or 128 rows; the job list puts full tiles first, then the tails by height, tallest first (2422, S16). c524 pairs two bodies instead of padding to the next height: 65–96 rows as 64 + 16 or 32 at 1024 tokens, 129–192 as 128 + 16, 32 or 64 at 3072 (S32). MiniMax 01ce and 4da3 (128–256 tokens) drop to 16- or 32-row bodies for short tails, f770 runs a 128-row tile with an empty second half as 64 rows from 4096 tokens (S33); the other MiniMax or Qwen2 gate-ups do not select a different compiled tile height from their live rows. MiniMax b94f, cb88, f839, 8b22 and 62df still branch on `live == 64` to select an unmasked full-tile path versus a masked tail path at the same 64-row height. No Point compares variable and padded heights. - **The weights' cache policy is per Point.** Dropping `.cg` from d3bb at m=2 cost 0.9–1.2% in two of three runs (every pair; speedup 2.205→2.180, 2.217→2.195), but nothing in the third (2.171→2.171, pairs both ways). Dropping it from 1d27/m256 cost 2.6–3.5% (1.177–1.189→1.144–1.150); adding it to 2422/m256 cost 0.2–0.7% (1.012–1.040→1.009–1.034), each in every pair of four runs . 2422 leaves `.cg` off its 128-row gate-up tiles above 128 tokens, at 256 the shared expert's two tiles (256 rows; a routed expert's about 8, estimated), whose weight address has no row-tile term (S16), so both read the same columns; a cache hit on the second is inferred (a toy in which two programs read each block ran 4 % faster without `.cg` ). Put `.cg` (`cache_modifier=".cg"` on `gl.load`, `cache=".cg"` on `buffer_load`; it sets `nt`) on weights the call reads once, leave it off weights a second tile re-reads, settle it per Point ([gluon/memory.md](gluon/memory.md) §6). In one run over 12 Points of seven kernels, `.cg` won at seven of ten predominantly read-once Points (1.6–9.6%; six had every pair favorable on both rulers, e75e/m2 only the aggregate direction), with no consistent effect at three m=1 Points and no measured loss. It lost at both re-read Points: 2422/m256 (0.2–0.7%, N=4) and 8411/m16384 (17%, N=1). Reuse was classified from the schedule and measured routing, not cache-hit counters; the reuse side covers only two Points . Store hints are unmeasured here: c524 writes cross-launch partials `.wt` and its output `.cs` at 2048 and 3072 tokens (S32), the Qwen2 routers their partials `.wt`, 4801 through `gl.store` (S9), which, unlike `buffer_store`, waits after each `.wt` store . - **Program order.** e9fd's decode down at 8–16 tokens makes a column tile's program ids congruent modulo 8 (S38); treating that as one-XCD placement assumes workgroup i maps to XCD i mod 8, which was not measured here, 8a5a walks its job list in 8 contiguous stripes (S10), f770 keeps an M tile's N tiles on one XCD (S33), placement derived from the formulas. Unmeasured on MoE kernels: [gluon/memory.md](gluon/memory.md) §14. - **`waves_per_eu` did not change the time where measured.** The gate-up of 51ae and 8411 (MiniMax MXFP4, 256 experts) sets 3 above 3072 tokens (S15); removing 51ae's at 8192 tokens left the speedup at 1.44 (1.441 → 1.440) ; 146 VGPRs with it, 150 without, no spills  ([tuning.md](tuning.md) §8, [debugging.md](debugging.md) §2). - **A Point's constants are coupled.** Changing 6557's `gate_fold` from 4 to 1 alone at 1 token crashed every bench run : LLVM's code generation dies with SIGSEGV on `gate_fold` 1 with 4 warps, a 128-wide K tile and the `amdgpu-sched-strategy=iterative-ilp` attribute the file sets at m ≤ 2 (S14); `gate_fold` 1 compiles with 1 warp, with a 256-wide K tile, or without that attribute . Change constants together, from a combination that compiles elsewhere (here 6557's 16-token branch) ([debugging.md](debugging.md) §6).

## 8. Activation and requantization

| Situation | Where SwiGLU and requantization go | Examples |
|---|---|---|
| One program owns full K for its gate+up columns and a whole quantization group of the output | the gate-up epilogue | MXFP4 queued and sorted designs (b94f, S13), DeepSeek from 4 tokens and 7e9c at 2 |
| The gate-up is split along K over programs | a small launch that sums the split partials in order, then SwiGLU and quantization | d3bb and 5751 `_activate_quantize` (S4), 7e9c at 1 token (`_finish_gate_up`), Qwen2 5c35  |
| The down operand is fp8 per 1 × 128 | the gate-up stores bf16; the down kernel quantizes each 128-group as it loads it | MiniMax FP8 e75e (S27), 0982 |

b94f's epilogue rounds the accumulator and the SwiGLU product to bf16 and quantizes groups of 32 to MXFP4 (S13):

```python
rounded = acc.to(gl.bfloat16).to(gl.float32)
pairs = gl.permute(gl.reshape(rounded, (BM, 2, BN // 2)), (0, 2, 1))
gate, up = gl.split(pairs)
value = (gate * _sigmoid(gate) * up).to(gl.bfloat16)
groups: gl.constexpr = BM * BN // 64
grouped = gl.convert_layout(gl.reshape(value, (groups, 32)), layout).to(gl.float32)
q, scale = _quantize_groups_3(grouped)
```

- **Take the activation from the reference.** The plugins split `x @ W13ᵀ` into gate (columns `[0, inter_dim)`)
  and up (the rest) and compute `silu(gate) · up`, SwiGLU on this page. `fused_moe@1.0` also has 'gelu', 'situv2',
  gpt-oss's 'swiglu' (`gate · sigmoid(1.702 · gate) · (up + 1)`) and silu with `gate_mode` 'interleave' (`w1`
  alternating 16-row gate and up tiles); these two first clamp gate to ≤ L and up to [−L, L], L = `swiglu_limit` (S40).
- Write the consumer's operand format: 147b stores the quantized activation in the weights' 16 × 16 shuffle and its
  scales in the shuffled E8M0 map (at 4096 tokens padding rows get code 127, at 2048 they stay unwritten), so the down
  GEMM loads A like B, where row-major words cost LDS  (S17; [gluon/mfma.md](gluon/mfma.md) §7).
- The MXFP4 and fp8 encoders (exponent rule, NaN guard, `v_cvt_scalef32_pk_fp4_f32` packing, saturation) and the
  little room the MXFP4 bound leaves beyond the activation quantization: [numerics.md](numerics.md) §1–§3. FlyDSL's
  GLU-in-epilogue numbers: [../moe/grouped_lowprec.md](../moe/grouped_lowprec.md) §10.
- Most DeepSeek launches (86 of 100 sites, all three of a2bb's, S22) pass `enable_fp_fusion=False`, MiniMax and Qwen2
  only 96e5's down (16 tokens). The switch did not change the tested a2bb assembly at 8 tokens;
  inspect whether it changes the actual arithmetic before treating it as a tuning axis.

## 9. Down GEMM

- **Decode: fuse down+combine only with compatible ownership.** A program can keep every route's output
  tile and sum weighted slots locally, removing route-row scratch and a launch. The larger combined
  MFMA/layout can cost more than it saves; synthetic full-call comparisons have shown regressions.
  The source form in `7e9c` puts routes in A rows and experts' column bands in B, then keeps only the diagonal
  and sums in slot order without partials or atomics (S12)

```python
acc = _tiny_projection(X, XS, W, WS, Ids, token, column, N, K, BN, BK, WARPS, False, CACHE)
vn = gl.arange(0, 8 * BN, gl.SliceLayout(0, mma))
diagonal = gl.gather(acc, vn[None, :] // BN, 0).reshape((8, BN))
...
for rank in gl.static_range(8):
    part = gl.gather(diagonal, gl.full((1, BN), rank, gl.int32, ep), 0)
    weight = gl.load(Weights + token * 8 + rank)
    value += part * weight
shared_acc = _tiny_projection(X, XS, W, WS, Ids, token, column, N, K, BN, BK, WARPS, True, CACHE)
```

  d3bb puts all 9 routes (shared expert included) into one 16 × 256 MFMA tile per program; MiniMax FP8 (256 experts)
  gives each of 8 waves one route (`AMDMFMALayout(..., warps_per_cta=[8, 1, 1])`, e75e) on plain fp8 `mfma` (§7).
- **Prefill: operand reuse across tiles (no gain where ablated).** MiniMax 8411 holds a 64-row activation tile in
  registers across a strip of column tiles (`_down_reuse_fp4_2`); DeepSeek loads one expert's weight panel and scales
  once and streams activation tiles through it (`_paired_down`, `_routed_down`). c524's schedule leaves that off at 2048
  tokens (S32). Its tested weight-reuse change had no established whole-call gain; activation-tile reuse
  needs a separate comparison. Atomics against a

  fixed-order reduce, on FlyDSL kernels: [../moe/grouped_moe_inference.md](../moe/grouped_moe_inference.md) §6.

## 10. Combine and the shared expert

- **Combine in slot order.** `out[t] = Σ_slot w[t, slot] · down[route]`, slot 0 to 7, from bf16 route rows (8411
  `_combine_rows_3`, S6). The route rows are a `routes × H` bf16 scratch the down writes and the combine reads: 805 MB
  each way at 16384 tokens for MiniMax MXFP4 with 256 experts (estimated from its `torch.empty` shape). Other files
  reorder it: the MiniMax MXFP4 64-expert prefill files store the down rows in sorted order, read through an inverse
  map kept in the ids buffer (b94f, S13); e6fc writes them by sorted row in 256-column chunks, read through a position
  map; DeepSeek's prefill stores dense expert-sorted rows and points each route's record at its row (3432) (S35). No
  Point compares the hand-offs. Decode designs fuse the combine into the down (§9); 1c17 and 3c6e keep a launch for it.
- **No float atomics** in the 65 files : integer counters, cursors, job indices, bitmasks, LDS markers.
  Qwen2's 8698 and d425 reorder rows across calls, yet they and 29 global-ticket files repeat outputs.
- **DeepSeek's shared expert (id 256, weight 1)** changes the plan by size: at decode it is the token's ninth
  route; at 32–256 tokens its jobs run first, its down result is stored straight into `out`, and the reduce adds the
  8 routed partials on top of it (1204, S5); at prefill the shared expert's down epilogue gathers the 8 routed
  partials of each token and writes `out`, so there is no combine launch (699f, S18).
- **Qwen2's shared expert (id 512)** is weighted by `sigmoid(hidden · shared_gate_weight)`; its dot products come
  from the shared expert's own gate-up programs (slot 10, tile 0), which add one partial dot per K group (5c35, S19).

## 11. Interpreting stored performance

A schema's `workloads[].speedup` describes its complete call at one Point against that contract's baseline.
Do not rank designs across Points using those ratios: the baseline changes too. Small-M launch fusions,
mid-size weight streaming and prefill matrix-core work have different costs. Re-measure both sides with
matched inputs, the same timing ruler and an identical-arm control (`tuning.md` §2).

## 12. Traps when editing these files

- **Dead branches.** Most files carry paths for other token counts (5751 a 2-token projection and a non-peeled top-k,
  c388 configurations for 64 and 128 rows): trace your Point's constexpr branch first.
- **Scenario constants are baked in, and `run` checks few of them.** DeepSeek multiplies the routed weights by 2.5 in
  device code, e9fd sums 12 router splits in `_rank_expert`; the MiniMax 256-expert files ignore `expert_offset`
  (right while the rank holds all 256); the MXFP4 64-expert prefill files read an expert from 6 bits
  (`descriptor & 63`); Qwen2's router+experts `run` ignores `topk` and `out_dtype`; cd47's `(256,)` router grid with
  `split = 8 if rows == 64 else 4` holds for 64, 128 and 256 rows only (S39). Re-derive them for another scenario.
- **Write every token's row at a 64-expert rank.** About 10 % of tokens have no route to a held expert (estimated,
  uniform routing: C(192, 8)/C(256, 8)); their `out` row is zero and must still be written: bench's outputs start as NaN
  ([authoring.md](authoring.md) §8), and an abc7 copy that skipped them failed. The committed combines
  loop over all 8 slots of every token and add nothing for a non-local route (abc7 `_combine_14`, S8; 0982, S34).
- **Buffer ops stop at 2 GiB from one base.** MiniMax MXFP4's e6fc, 51ae and 8411 have host extent guards for their router/scale-buffer paths
  (8411: three `… < 2**31` tests, S6); their data/output buffer choices also depend on fixed-Point policy flags.
  147b guards in device code and bb66 by token count. These are not a universal guard for every buffer base. Route-row scratch is routes × H: a
  larger Point or H would cross it (a buffer load read 0 past 2 GiB on a generic tensor; memory.md §3).
- **Docstring launch lists can be short.** c3e3 lists two launches; at its 4-token Point the host code runs three
  (`_padded_down` follows), as 4 other lists differ from the trace : follow the host function.
- **A worst-case grid may reach descriptor or sorted-row entries that no route fills; a device-total exit can prevent some descriptor reads.** Initialize the sentinels (out-of-range route ids
  the consumer masks: 8411 fills `Sorted` with `R`, the route count, S6; d861 with `CAPACITY`, S7) in an earlier
  launch of the same call, also the tails no route fills (without its fill d861 crashed).
- **Scratch aliasing requires completed readers.** `699f` aliases router logits with activations and quantized
  inputs with partials (S18). The host launch order must end every old reader before the overwrite;
  source order alone does not prove safe device accesses or race freedom.

## Evidence

- S1 aklr/plugins/sglang/models/deepseek_v2/api.py:32 the route on `DeepseekV2MoE.forward_normal`: router, top-k and experts, the shared one last, as one op call; MiniMax's route on `MiniMaxM2MoE.forward_normal` at aklr/plugins/sglang/models/minimax_m2/api.py:45, the experts' format picking the fp8 or MXFP4 op; aklr/plugins/sglang/models/moe_blocks.py:16 the runtime configurations that keep a MoE call on SGLang
- S2 aklr/plugins/sglang/models/qwen2_moe/api.py:52 the route on `_forward_router_experts`: router+experts op, else router op (:82), then experts op (:91), then SGLang's experts; aklr/plugins/sglang/models/qwen2_moe/baseline.py:4 the baseline: SGLang's small-M router of 1 to 8 tokens, which appends the gated shared expert itself, else the gate, `fused_topk` and the append; then SGLang's gfx950 small-M kernel where it serves, else `fused_moe` (to :12); `_smallm_router` (:45-55), `_experts` (:100-140); MiniMax's gate on fp32 activations at aklr/plugins/sglang/models/minimax_m2/baseline.py:44
- S3 aklr/plugins/sglang/models/deepseek_v2/impls/triton/deepseek_v2_mxfp4_moe_shuffled_16x16_model_dim6144_n_routed256_inter_dim256/5751284b6965eabb.json#kernel.py:76 `_router_linear`: split-K router tiles and the token's two MXFP4 quantizations in one grid (to :121); `_select_routes` (:126-180) sums the split partials, rounds once to bf16 (:133-135) and publishes the routes; `_elect_rank` (:247-275), the election in inline assembly, and `_expert_at_rank_peeled` (:279-294) repeat the sum, the rounding and the election; gate-up program → (route, tile, split) at :321, the slot-7 route first (:324), the shared-expert route on id 256 without an election (:333-334); program 0 selects (:518-519); grid `m + up_jobs * splits * (2 * intermediate // up_width)` (:708)
- S4 aklr/plugins/sglang/models/deepseek_v2/impls/triton/deepseek_v2_mxfp4_moe_shuffled_16x16_model_dim6144_n_routed256_inter_dim256/d3bbbce5c0bbc61e.json#kernel.py:494 the 2-token launch sequence (to :521); `_router_pair_packet` (:62): one full-K program per expert, `gl.dot_fma` for both tokens on one weight load (:87, :90), logits rounded to bf16; programs `< M` select, the rest run the gate-up (:360-363), slot-major job order `route = job % M * 9 + job // M` (:252); the route replicated over the 16 MFMA rows (:274), only row 0 stored (:337); shuffled E8M0 scale words shifted per element, then `mfma_scaled` on e2m1 (:320-333); `_activate_quantize` (:367): the split partials summed in order, SwiGLU, MXFP4
- S5 aklr/plugins/sglang/models/deepseek_v2/impls/triton/deepseek_v2_mxfp4_moe_shuffled_16x16_model_dim6144_n_routed256_inter_dim256/1204f286c55bb56e.json#kernel.py:498 `_job_for_tile`: range asserts (:499-504), scan and `v_readfirstlane_b32` (:509-523); `_decode_job`: the tile from bits 5-16, live rows − 1 from bits 0-4, the first activation row from bit 17 (:527-533); the column-0 program publishes the descriptor for the down launch (:580-582); a route's row `selected_id * PAD + ticket` (:169), `pad = triton.cdiv(m, tile_rows) * tile_rows` (:626); the router launch's program 0 zeroes the counts (:91-94); router grid `triton.cdiv(m, 32) * 8 * router_splits`, 16 splits at 256 tokens (:646-650, :53); gate rows and the matching up rows in one column tile (:351), split back with reshape/permute (:215); `_reduce_parts` (:597): the 8 routed parts in slot order added to the shared expert's result already in `out` (:615), the shared store (:255-258); the 256-token launch sequence (:648-717)
- S6 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_mxfp4_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts256_inter_dim512/84116a96e535f174.json#kernel.py:1346 a relaxed atomic ticket per route in the selection launch, record `ticket << 8 | ids`; row = ticket + base, `Sorted` and the expert of each 64-row tile (:1225-1227); packed (score, id) keys (:1320), 4 per lane in a [4, 16] layout, pre-reduced in registers (:1327-1329), `gl.reduce(..., _argmax_packed)` (:1330-1331), the row roots gathered from lane 16·c + 15 by `_collect_partition_roots` (:1281-1289), `_argmax_packed` (:89-95); `_scatter_bases_2` (:1170): every scatter program recomputes the padded prefix, program 0 stores `Total` (:1196), scatter programs beside quantization programs (:1253-1270); every router program, after its tile, fills a 256-entry stripe of `Sorted` with the sentinel `R` and zeroes `Counts` (:1069-1080); worst-case capacity on the host (:1373); the gate-up that runs at 16384 tokens, `_gate_up_scale_preload_fp4` (`preload_scales=not compact`, :943), exits on the device-written `Total` (:626-627; `_gate_up_fp4_3` at :443-444), loads the weight words (:709) and converts them with `assert_trivial=True` (:767, :777), its weight scales with a plain `convert_layout` (:768), preloads the tile's activation scales into LDS (:686-691), gathers them per K step (:762) and stages the shared activation tile in padded LDS (:748-756); `_combine_rows_3` (:114): weighted route rows summed in slot order (:136), the bf16 route-row scratch (:836); buffer ops gated in the host: `gate_buffer` (:803), `down_buffer` (:837-842), `router_buffer` (:1361-1368), each `< 2**31`; "expert_offset is 0: every routed expert is on this rank" (:1517)
- S7 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_mxfp4_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts256_inter_dim512/d8610d895915c167.json#kernel.py:764 explicit copies masked by `owner` (:772-774); a job opened every 16th ticket (:792-801); `sorted_stride = triton.cdiv(routes, 16) * 16` (:839); `Sorted` filled with the sentinel `CAPACITY` (:327)
- S8 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_mxfp4_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts64_inter_dim1536/abc7bbf7a7135f28.json#kernel.py:503 wave ballot in inline assembly; `_route_masks` (:515-539) builds per-expert route masks and a histogram; `_combine_14` (:37) adds all 8 slots of every token (:71)
- S9 aklr/plugins/sglang/models/qwen2_moe/impls/triton/qwen2_moe_router_model_dim4096_n_routed512_topk10/c3889614985b700a.json#kernel.py:28 `_shape_config`: split, expert tile, K tile, lookahead per row count (to :34); the async-copy ring at :113-129; the projection grid `rows // 32 * (512 // expert_tile) * split` and the routing launch (:294-306); `_wave_winner` (:185-203); queues and rounds in `_stable_topk` (:207-246); split-order sum and one bf16 rounding (:269-272); split partials stored `.wt` by `buffer_store` (:134); 4801e216dcff290f.json in the same directory launches `(triton.cdiv(rows, block_m), splits, 32)` one-wave CTAs (kernel.py:213) and stores its partials with `gl.store(..., cache_modifier=".wt")` (:76-81)
- S10 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_fp8_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts256_inter_dim384/8a5a03b603ec1b70.json#kernel.py:909 the fp32 router weight split into three bf16 terms, three bf16 MFMAs (to :916); `_project_w13` walks its job list in 8 contiguous stripes, `stripe = logical % stripes` (:656-672)
- S11 aklr/plugins/sglang/models/qwen2_moe/impls/triton/qwen2_moe_fp8_router_experts_shuffled_16x16_model_dim4096_n_routed512_inter_dim256/99f47a166d6d62fc.json#kernel.py:19 float64 → bf16 rounding with the midpoint fix; float64 MFMA at :81-88; routed tasks of at most 4 rows (`sparse_size`, :255-256) on `instr_shape=[4, 64, 64]`, the shared expert on [16, 16, 32] (:317-330)
- S12 aklr/plugins/sglang/models/deepseek_v2/impls/triton/deepseek_v2_mxfp4_moe_shuffled_16x16_model_dim6144_n_routed256_inter_dim512/7e9c13e0f2d58a3b.json#kernel.py:370 preshuffled words converted to the A and B operand layouts with `assert_trivial=True` (:370, :374); the gate-up's scale tiles go through LDS (:342-343, :377-378, :394-395); block-diagonal down and weighted combine in one program (:486-507), its operand built in decode_tiny_projection.py:38-60, which loads the scale words in the scale layouts (:70-83)
- S13 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_mxfp4_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts64_inter_dim1536/b94f5777fd5d2d88.json#kernel.py:237 SwiGLU at the baseline's bf16 rounding points and MXFP4 requantization in the gate-up epilogue (to :257; called at :731); `_select_wave_rows` (:131): a 16-lane reduce of (value, index) pairs and the row roots at lane 16·c + 15 (:190-197); the combine reads the down rows through `Inverse` (:567-572), `inverse = ids` (:1099), with `compact_down = m >= 2048` (:1017)
- S14 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_mxfp4_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts256_inter_dim512/655723f593921b08.json#kernel.py:1301 `gate_fold`, K tile, warps and row constants chosen per token count (to :1336); the K fold at :361-437; `gate_attrs` = `iterative-ilp` at m ≤ 2 (:1306), passed at :1336
- S15 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_mxfp4_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts256_inter_dim512/51ae499426db920d.json#kernel.py:941 `waves_per_eu=3 if not compact else 0` (compact: m ≤ 3072)
- S16 aklr/plugins/sglang/models/deepseek_v2/impls/triton/deepseek_v2_mxfp4_moe_shuffled_16x16_model_dim6144_n_routed256_inter_dim512/2422a87b3b814299.json#kernel.py:598 the descriptor's live row count picks a body compiled for 16, 32, 64 or 128 rows (:610-689); `_write_height_descriptors` places full tiles first, then tails by height, tallest first (:109-130); `WEIGHT_CACHE` without `.cg` on 128-row gate-up tiles above 128 tokens at :434; `_load_packed_weight` addresses the weight by expert, column and K offset, with no row-tile term (:286-304); `_pack_job`: `expert | live << 9 | block << 17` (:105), decoded `info & 511` (:602)
- S17 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_mxfp4_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts256_inter_dim512/147b861bd4c406d2.json#kernel.py:509 the activation stored with the weights' 16 × 16 offset and its scales with the shuffled E8M0 map, padding rows at code 127 when `R // TOPK` is at most 1536 or above 3072 (:523-539), else a masked store (:540-545)
- S18 aklr/plugins/sglang/models/deepseek_v2/impls/triton/deepseek_v2_mxfp4_moe_shuffled_16x16_model_dim6144_n_routed256_inter_dim256/699f9fa93bf64b26.json#kernel.py:674 `_finish_output`: the shared expert's down epilogue adds the 8 routed parts in rank order (:712-724); tile bound at :1285-1286; aliased scratch at :1296, :1303
- S19 aklr/plugins/sglang/models/qwen2_moe/impls/triton/qwen2_moe_fp8_experts_shuffled_16x16_model_dim4096_n_experts513_inter_dim256_topk10/5c35215626bc1c87.json#kernel.py:152 slot-10, tile-0 programs write the shared gate's partial dot products; fp8 promoted exactly to bf16 (docstring :2-6)
- S20 aklr/plugins/sglang/models/deepseek_v2/impls/triton/deepseek_v2_mxfp4_moe_shuffled_16x16_model_dim6144_n_routed256_inter_dim512/3c6e26ffcce7196d.json#kernel.py:158 `_select_masks`: `buffer_atomic_or` of `1 << token` into each chosen expert's mask (:162); `_next_expert`: the keyed `gl.min` (:119-123); `_mask_descriptor`: wave ballots (:210-213) and popcounts (:214-216) over the masks, the column-0 program stores the descriptor (:242-243)
- S21 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_mxfp4_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts64_inter_dim1536/3df5254a048c3dab.json#kernel.py:176 `_expert_membership`: per token, count the experts with a higher score, the lower-id rule only at M == 2 (:211-215); member when fewer than TOPK; a token bitmask returned (:220)
- S22 aklr/plugins/sglang/models/deepseek_v2/impls/triton/deepseek_v2_mxfp4_moe_shuffled_16x16_model_dim6144_n_routed256_inter_dim512/a2bb3d83284b20ff.json#kernel.py:342 `enable_fp_fusion=False` on each of the 3 launches (:342, :375, :404)
- S23 triton-lang/triton@c01b677:third_party/amd/backend/compiler.py:327 `gluon_to_ttgir`: no buffer-op conversion and no pipeliner (to :348; `make_ttgir` converts at :306); `launch_cooperative_grid: bool = False` (:95), launched with `hipModuleLaunchCooperativeKernel` (third_party/amd/backend/driver.c:674-677); triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/LoadStoreOpToLLVM.cpp:2278 the redundant-thread predicate of an atomic, ANDed into its mask at :2338; `acq_rel`, the default of `atomic_add` (python/triton/language/core.py:2731), makes the backend insert buffer invalidates and L2 write-backs (:2354)
- S24 triton-lang/triton@c01b677:python/triton/experimental/gluon/language/_semantic.py:245 `convert_layout(..., assert_trivial=True)` raises unless the conversion is trivial; triton-lang/triton@c01b677:python/triton/experimental/gluon/language/_core.py:703 `gl.barrier` synchronizes the threads of one CTA; a cluster only with `cluster=True` and `num_ctas > 1` (:712-714), which the HIP backend rejects on gfx950 (third_party/amd/backend/compiler.py:162-163)
- S25 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_fp8_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts256_inter_dim384/5cac8212f51edf84.json#kernel.py:455 64-bit order-preserving keys with the inverted id; a 5-compare network on each lane's 4 keys (:465-475); 8 rounds popping the lane-queue heads (:476-503); each round `gl.max` over the head scores, then `gl.min` over the ids (:477-482), at `B = triton.next_power_of_2(e)` = 256 (:892)
- S26 aklr/plugins/sglang/models/qwen2_moe/impls/triton/qwen2_moe_fp8_router_experts_shuffled_16x16_model_dim4096_n_routed512_inter_dim256/7d7e54597a7e11bd.json#kernel.py:63 `_routing_keys`: sign-flipped bf16 bits, −0 folded, `<< 9` above `511 - experts`; decoded by `_routing_value` (:71); router products in float64 at :94-96, rounded by the midpoint-corrected `_round_to_bf16` (:22); each W13 program replays the top-k pops (:282), one program publishes ids and weights (:290-291)
- S27 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_fp8_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts256_inter_dim384/e75e553b34178760.json#kernel.py:191 `_ordered_score`: monotonic int32 key, −0 folded, NaN to the largest key; fp32 multiply and `gl.sum` router dot at decode (:228); `_down_combine` quantizes each 128-group of the bf16 activation to fp8 as it loads it (:157-161); `_prepare_keys` (:204): one program per (row, expert), a full-K dot (:228), the key and the probability stored (:249-250), beside the quantization programs (:753); `_rank_stable_queue`: per-lane sorted queues (:301-320); the publishing rounds take `gl.max` of the keys, then `gl.min` of the ids holding it (:386-388); `_down_combine` on a rank-3 [16, 16, 32] layout of 8 warps (:134-135), plain fp8 `mfma` (:166); `mfma_scaled` in the 4-16-token gate-ups (:477, :573)
- S28 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_mxfp4_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts256_inter_dim512/fbd9fa8970f5ac47.json#kernel.py:58 `_order_key`: ordered score and `255 - expert` in the mantissa of a double in [1, 2); NaN to a fixed NaN key (:64)
- S29 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_fp8_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts64_inter_dim1536/94af11a072175a38.json#kernel.py:520 `_key_maximum`: `v_max_f64` on packed keys in inline assembly (`_key_minimum` at :527); `_w13_persistent[2 * f // 64, 96]` (:1030) loops over its tiles in a static range (:741)
- S30 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_fp8_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts64_inter_dim1536/1c1747a25af67b8e.json#kernel.py:993 `_select_rank`: one program per candidate expert; rank = experts with a higher score, or an equal one and a lower id; selected when rank < TOPK; at m ≥ 8 the gate-up and down run one program per local expert, grid `(tiles, local_e, splits)` (:632-639)
- S31 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_mxfp4_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts64_inter_dim1536/f82c87618c3b5cd4.json#kernel.py:1542 one gate-up launch (`_gate_up_pairs_2` at 4 tokens, :1645) runs `_select_routes` (publishing, :1544, defined at :775) and `_select_route_pair` (:1552, recomputing slots, defined at :720); winner read from lanes 60–63 by `ds_bpermute_b32` (:671-680, :708-715)
- S32 aklr/plugins/sglang/models/deepseek_v2/impls/triton/deepseek_v2_mxfp4_moe_shuffled_16x16_model_dim6144_n_routed256_inter_dim256/c5244d02949f27b4.json#kernel.py:1003 the live row count picks a 16-, 32- or 64-row body, two paired bodies (`_native_pair_w13`) for 65-96 rows at a 64-row tile and for 129-192 rows when `PAIR_MIN` > 0, else a 128- or 256-row body (to :1028); `block_m` (:1123) and `pair_min` (:1137) per Point; cross-launch partials stored `.wt` (:296, :696), the output `.cs` at 2048 and 3072 tokens (:690)
- S33 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_fp8_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts256_inter_dim384/f77013cc201d92d5.json#kernel.py:1187 `tile = gl.program_id(1) * GROUP + gl.program_id(0) % GROUP`, `n_tile = gl.program_id(0) // GROUP` (:1188); a 128-row tile flagged in its metadata runs a 32-row body (:1192-1217), one whose second 64-row half is empty a 64-row body (:1218); `SHRINK_TAIL=m > 2048` (:1463), `exact_tail = 2048 < m <= 4096` (:39)
- S34 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_mxfp4_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts64_inter_dim1536/b0651fe84db2f670.json#kernel.py:1029 `_store_routes`: `owned` (:1046), the local routes compacted by `gl.associative_scan` (:1051-1056); aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_fp8_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts64_inter_dim1536/098200aca0238880.json#kernel.py:568 the owned flag in bit 32 beside the weight (:568-570), the combine's masked load of every slot (:32-40); plain fp8 `mfma` on a rank-2 [16, 16, 32] layout when `NATIVE` (:48-49, :94-97), `NATIVE=m == 64` (:706)
- S35 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_mxfp4_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts256_inter_dim512/e6fc7b071ab748e5.json#kernel.py:230 the combine reads the route rows through a position map in 256-column chunks (:230-236); `output_rows = capacity if p.output_chunk else routes` (:804), `output_chunk=0 if middle else 256` (:902); aklr/plugins/sglang/models/deepseek_v2/impls/triton/deepseek_v2_mxfp4_moe_shuffled_16x16_model_dim6144_n_routed256_inter_dim512/3432dd09808b8428.json#kernel.py:280 dense expert-sorted offsets, each route's record pointed at its dense row (:280-288), the routed partials stored there (:566)
- S36 aklr/plugins/sglang/models/qwen2_moe/impls/triton/qwen2_moe_fp8_experts_shuffled_16x16_model_dim4096_n_experts513_inter_dim256_topk10/86987a66c0cdba39.json#kernel.py:105 LDS counters in one CTA reserved with `atomic_scatter_add`, "Integer reservations may reorder rows" (:108-112); descriptor `expert | (... << 16)` (:127), decoded `& 1023` (:81); the W13 grid loops over a device count, "balanced across passes in 256-program cohorts" (:15-16, :183-189)
- S37 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_mxfp4_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts256_inter_dim512/96e51c1f85a71b7e.json#kernel.py:671 `RANK_MAJOR`: `route % M * TOPK + route // M` (:672), set at 4, 8 and 16 tokens (:813)
- S38 aklr/plugins/sglang/models/deepseek_v2/impls/triton/deepseek_v2_mxfp4_moe_shuffled_16x16_model_dim6144_n_routed256_inter_dim256/e9fd0e91468ca721.json#kernel.py:727 `_shared_a_down_combine` remaps (tile, token) at 8 and 16 tokens so that one column tile's programs are congruent mod 8 (:728-736); `_rank_expert` sums 12 router splits (:345-347)
- S39 aklr/plugins/sglang/models/qwen2_moe/impls/triton/qwen2_moe_router_model_dim4096_n_routed512_topk10/cd47a5e858af6ab4.json#kernel.py:263 `split = 8 if rows == 64 else 4` with the router grid `(256,)` (:267); in the files of S12, S38, S26, S13 and S6: `selected_prob / total * 2.5` (7e9c :81), 12 router splits (e9fd :345-347), `run` passing neither `topk` nor `out_dtype` (7d7e :755-779), `expert = descriptor & 63` (b94f :777-778), "expert_offset is 0" (8411 :1517)
- S40 aklr/plugins/sglang/models/deepseek_v2/reference.py:40 `_mlp`: "down(silu(gate) * up), gate and up the halves of x gate_up^T" (:41, the split at :44); the same split and `silu(gate) * up` at aklr/plugins/sglang/models/minimax_m2/reference.py:89-93 and aklr/plugins/sglang/models/qwen2_moe/reference.py:98-99; `w1` rows "gate and up concatenated", `2 * inter_dim` (deepseek_v2/contract.py:111); aklr/ops/fused_moe/reference.py:110 `projected.chunk(2, dim=-1)`, "w1 stores [gate | up]"; `gate.clamp(max=limit)`, `up.clamp(-limit, limit)` whenever `swiglu_limit` is passed (:111-112); silu, gelu, swiglu `gate * torch.sigmoid(1.702 * gate) * (up + 1)` and, otherwise, situv2 with `beta` and `linear_beta` (:113-121); aklr/ops/fused_moe/contract.py:91 the activations silu, gelu, situv2, swiglu (to :99); `gate_mode` separated or interleave, "Where w1 keeps the up rows" (:108-114); interleave only with silu on the `gate_up_interleaved` layout (:154); `swiglu_limit` passed exactly for interleave and swiglu (:155); aklr/core/contract/semantics/_layouts.py:158 `GateUpInterleavedWeight`: 16-row tiles, "the gate and up halves alternate tile by tile" (:159-160)
