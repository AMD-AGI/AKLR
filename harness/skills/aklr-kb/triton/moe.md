---
title: "MoE layer plugins in Gluon on gfx950: decomposition by token count, routing, sorting, expert GEMMs, combine"
kind: topic
ops: [moe, fused_moe, grouped_gemm]
dtypes: [mxfp4, fp8, bf16]
regimes: ["tokens 1-16", "tokens 32-256", "tokens 1024-16384", sorted-tokens]
gens: [gfx950]
triton: 3.8.0
---

# MoE layer plugins in Gluon (gfx950)

How AKLR's Gluon kernels compute a rank's router-and-expert portion of a SGLang MoE layer call in a few launches, and which design to copy at which
token count: the Triton side of [../moe/INDEX.md](../moe/INDEX.md), whose measured MoE-kernel numbers are from FlyDSL unless it explicitly names another language or a vendor-library comparison; the numbers here
are these Gluon kernels' on MI355X; the times on AKLR's timer were measured under Kineto, before #33. The page covers the 65 MoE files AKLR held at 43e112e and quotes the CI speedups stored then *(field, E1)*. Added since, none storing a speedup at 0e02a81: Qwen3-Next's FP8 router+experts scenario at model dim 2048, inter 128 (11 files, #55), seven more files in the scenario at model dim 4096 (11, #74) and a fourth contract, `qwen2_moe_mxfp4_moe` (10 files, #77, on `Qwen2MoeSparseMoeBlock.forward`, S2). Per file:
[kernels/kernel_library.md](kernels/kernel_library.md) §7–§10. m is a Point's `n_tokens`, a route one (token, expert) pair; files go by their uuid's first four hex digits (Evidence).

## Route here when

- You write or tune a kernel for `deepseek_v2_mxfp4_moe@1.0`, `minimax_m2_fp8_moe@1.0`, `minimax_m2_mxfp4_moe@1.0`
  or one of the `qwen2_moe_*` contracts (three at 43e112e, four since #77).
- You decide a MoE call's launches per token count and where top-k, sort, activation, combine and shared expert go.
- You write expert GEMMs as a grouped GEMM: per-route or per-expert jobs at few tokens (§4), sorted, padded expert segments at many (§6, §7, §9), low-precision weights and scales (§7, §8).
- You port a `../moe/` mechanism to Gluon, or need the Gluon form of atomic tickets, a device-side job list or a
  recomputed top-k (preshuffled MXFP4 weights and shuffled E8M0 scales: [gluon/mfma.md](gluon/mfma.md) §7,
  [gluon/memory.md](gluon/memory.md) §13).
- A constant change in one of these files fails to compile or crashes, or a knob you expected to matter does nothing.

## 1. The plugins and what they replace

| Plugin (`aklr/plugins/sglang/models/`) | Replaces (SGLang 0.5.22) | Contract, scenario | Points (`n_tokens`) | Files at 43e112e | Launches per call (≤ 16, 32–256, ≥ 1024 tokens) *(field, E11)* |
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
  (quantizing them to MXFP4 moves `out` by rel L2 0.21 *(field, E4)*). MiniMax (3072): 256 experts, fp32 router weight,
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
  launches): each Point compiles its own binaries on its first call *(field, E11)* ([tuning.md](tuning.md) §14). Most
  files keep branches for other token counts (§12); Qwen2's experts files 5c35, 8698 and d425 assert their one Point.
- **All 65 schemas are `bounded`** (so are the 28 added since): runs may differ, accuracy is each contract's output bound's job (the MoE `out` uses SQNR; Qwen2's router weights use a tolerance bound and its ids must match exactly) ([numerics.md](numerics.md) §1,
  §8); the kernels let atomic order place routes and sum values in a fixed order (§6, §10).

## 2. Pick the decomposition by token count

Gluon has no grid-barrier primitive (`gl.barrier()` synchronizes one CTA, S24), and none of AKLR's Triton schemas uses cooperative launch (`launch_cooperative_grid`, S23; 154 schemas at 43e112e, 280 at 0e02a81). Ending a launch is not an unconditional dependency rule: on toy Gluon kernels (256 experts, H=6144, m=1,2,16; one GPU run, 9 rotated rounds per Point), two launches for logits before top-k took 0.26–0.51× the replay time of a last-arriving program doing top-k, and 0.15–0.48× a cooperative counter-barrier launch. For counts before rows, a last arriver scanning 256 counts beat two launches at m=1,2: two launches took 1.16× its replay time and 1.48–1.50× on AKLR's timer (before #33), with all 1000 calls bitwise equal per Point. At m=16 the rulers disagreed (two launches: 0.89× replay, 1.28× AKLR). Intermediate rows before down and routes before combine were not measured *(field, E12)*. Under the cooperative flag, a bounded counter barrier completed over 256/512/1024 one-wave workgroups (all 1000 eager and replayed calls per Point; one run), and over 256 workgroups of 1024 threads (0.20–0.22 ms; N=2); 4096 of those large workgroups were refused. No plain-launch comparison established that the flag caused co-residency, nor was preservation of the cooperative graph-node attribute checked; the tested barrier took 2.1–6.8× two-launch replay time *(field, E4, E12)*. Served-file decompositions below are observations:

| Tokens | One program does | Launch sequence of one served file *(field, E11)* | CI speedups at 43e112e, DeepSeek and MiniMax |
|---|---|---|---|
| ≤ 16 | one route × one column tile (× one K split), no sort; at 8–16 tokens 3df5 and 1c17 (64-expert ranks) and 3c6e (DeepSeek `inter_dim512`, 16) run one local expert × one column tile over all its tokens instead (§4.1) | DeepSeek d3bb, 2 tokens (S4): `_router_pair_packet` (304 programs: a full-K router program per expert, the rest MXFP4 quantization) → `_select_and_up` (2306: M = 2 selection + 2304 gate-up programs; the routed-route ones recompute their expert, §4.1) → `_activate_quantize` (72) → `_token_local_down` (768: down and weighted combine) | 1.097–4.115 over 30 Points *(field, E1)* |
| 32–256 | one expert job × one column tile: 16–128-row bodies in the DeepSeek/MiniMax files and Qwen2 experts-only files; Qwen2 `99f4` at 32 tokens uses 4-row routed jobs and 16-row shared-expert jobs | DeepSeek 1204, 256 tokens (S5): `_router_projection` → `_select_and_quantize` (top-8, relaxed atomic tickets, hidden quantization) → `_expert_tiles` gate-up (each program derives its job, §4.2) → `_expert_tiles` down → `_reduce_parts` (fixed-order combine) | 1.030–1.275 over 20 Points *(field, E1)* |
| ≥ 1024 | one tile of a sorted, padded expert segment (all 32 Points sort before the gate-up; no unsorted design was measured *(field, E11)*) × one column tile: nominally 64 or 128 rows (MiniMax; f770 can execute a 32- or 64-row tail body at run time), or a height of 16–256 rows picked per tile at run time (DeepSeek, §7) | MiniMax MXFP4 8411, 16384 tokens: `_router_projection_7` (also zeroes counters, fills sentinels) → `_select_partitioned_2` (top-8 + tickets) → `_scatter_quantize_inline_2` (rows + hidden quantization) → `_gate_up_scale_preload_fp4` → `_down_reuse_fp4_2` → `_combine_rows_3` | 1.152–1.785 over 32 Points *(field, E1)* |

In routes per expert, r = m·k/E: a source census computed r for 89 MoE Points from their `n_tokens`, `topk` and `n_routed` and traced their gate-up launches and grids, without classifying the job forms, so it sets no r boundary between per-route, grouped and sorted designs, nor one for other E or k *(field, E15)*.

The token count for a given r is N = r·E/k; this converts the coordinate, not a measured threshold, to another E, top-k or routing distribution, and how many experts a call touches depends on the routing distribution (below).

- **≤ 16 tokens: candidate fusions, not a guaranteed combined speedup.** In a toy top-4-of-32 MoE (H 256, I 128) at m = 2 and 16 each fusion saved 3.30–4.64 µs on AKLR's timer before #33 (top-k 3.30–4.19, SwiGLU 4.04–4.64, combine 4.19–4.43); all three took 0.667–0.751× the replay time and 0.426–0.462× on AKLR's timer (4 runs); single fusions were not timed replayed, requantization was not in the toy and full-K ownership was not varied. Synthetic fp8 VALU shapes (256 experts, top-8; H6144 with inter 256, H3072 with inter 512; m = 1, 2, 8, 16; one run, 16 rotated rounds per Point) put SwiGLU and 1×128 requantization in a full-K gate-up epilogue (§8): against a full-K gate-up with a separate activation launch it saved 1.76–2.76 µs replayed and 3.05–3.73 µs on AKLR's timer, but it lost to the unfused K-split design, and all three fusions took 1.08–2.30× that design's replay time. At H6144/m16 one fp8 code flipped at a rounding boundary (relative L2 1.122e-5 against the reference, above 1e-5); it passed only against a reference using its own activation. A/A controls of the K-split design showed replay bias at some Points, the full-K comparison had none, and MXFP4 MFMA kernels were not timed this way *(field, E4, E13)*. Gate-up recomputation (§4.1) and down+combine (§9) need their own comparisons. Narrow column bands give one token about a thousand one-wave programs (5751: 1153, S3).
- **32–256 tokens: group routes per expert.** Take a relaxed atomic ticket per route in the selection launch and either open a job every 16 (or BM, the tile height) tickets there, or let each GEMM program derive its job from the counts (§6, §4.2; neither form timed); sum the routes in a separate fixed-order launch *(field, E4)*. Uniform top-8-of-256 routing touches about 40 % of the experts at 16 tokens, 64 % at 32, 87 % at 64, 98 % from 128 (estimated for uniform routing: 1 − (31/32)^m). With the schema's router distribution, d861's 64-token call touched 77.2% of the experts on average (74.6% at the least), not 86.9%, and read those 495 MB at 4.87–5.00 TB/s, the baseline at 3.88–3.90 *(field, E19)*.
- **≥ 1024 tokens: a sorted grouped GEMM.** Counting sort into tile-padded segments (DeepSeek: variable-height tails, tallest first), a down GEMM that reuses one operand across tiles, a fixed-order combine (§6, §7, §9, §10); none timed against another design.
- **What a launch costs.** AKLR's timer read each kernel at about 4 µs or more under Kineto, before #33, a bare graph replay about 1.8 µs per extra kernel *(field, E11)* ([../pitfalls/aklr_dispatch_floor.md](../pitfalls/aklr_dispatch_floor.md), [authoring.md](authoring.md) §7). The FlyDSL trade: `../moe/few_tokens.md` §1–2.
- **Price the Point before you design.** At 2 tokens d3bb touches about 17 experts, 42 MB of weights and scales, a 6–7 µs read (estimated from the layouts and the 1 GiB stream rates, 6.0 TB/s plain and 6.8 TB/s with the `nt` bit a load's `.cg` sets), yet the call took 19.06–19.61 µs in 4 kernel launches, 4.76–4.90 µs per launch on average (4 runs of 2 validates; whole-call time, before #33) *(field, E11)* against the separately measured 4.03–4.07 µs launch floor (3 runs) *(field, E10)*: the per-launch reading fits a launch or latency cost but does not isolate it; fusion is a candidate to measure on the whole call, not a guaranteed speedup. At 256 tokens 2422 reads all experts' weights, about 1.29 GB, in 225.5–236.8 µs, MiniMax 1d27 its 64 experts', about 481 MB, in 103.6–105.4 µs (4 runs) *(field, E11)*: about 80–95% and 67–77% of the cited 6.0–6.8 TB/s stream rate (estimated), little left over a stream the baseline pays too. At thousands of tokens price each stage: MiniMax MXFP4 51ae takes 505.7–518.6 µs at 8192 tokens (4 runs) *(field, E11)*. An earlier batch of runs read 225.8, 102.9 and 515.6 µs for 2422, 1d27 and 51ae *(field, E11)*. At the rates of [tuning.md](tuning.md) §13 `51ae`'s expert GEMMs (0.62 TFLOP) need about 66 µs, its fp32 router (12.9 GFLOP) 82 µs on fp32 MFMAs or 16 µs as three bf16 MFMAs (§5.1), its 642 MB of weights and scales 94–107 µs, and the 403 MB of bf16 route rows its down writes and its combine reads 118–134 µs (all estimated): the fp32 router and that round trip each exceed the expert GEMMs' MFMA floor.

## 3. Put independent work in one grid

When two phases read only call inputs, they can share a program-id-role grid; the scalar role test is uniform.
A toy router GEMV (64 programs, 2 rows, K2048) beside a 4-program elementwise phase gave equal outputs in 4 runs; one grid took 0.495–0.506× the time of two launches on AKLR's timer (before #33) in all four, but 0.755–0.835× the replay time in three and 1.154× (9.360 against 8.110 µs) in the fourth: a gain on AKLR's timer but not in every replay run *(field, E10)*. No DeepSeek or MiniMax kernel is named as a quantizer alone
*(field, E11)*. In 5751 (1 token, S3) 192 one-wave programs each compute a 16 × 16 router tile over one K split of 512
(16 expert tiles × 12 splits) beside 24 programs that quantize the token to MXFP4:

```python
    program = gl.program_id(0)
    if program < 256 // BN * SPLITS:
        tile = program % (256 // BN)
        split = program // (256 // BN)
        ...
        gl.store(Y + (om[:, None] * SPLITS + split) * 256 + on[None, :], acc, om[:, None] < M)
    ...
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
- One binary serves every role, so a cheap role shares that binary's resource allocation: in a tested mixed-role kernel the binary used 256 VGPRs where the streaming role alone used 16, yet its streaming role took 0.966–0.974× the standalone stream's time (the heavy role not executed; latency- or compute-bound cheap roles not measured) *(field, E3, E10)*.
- DeepSeek quantizes the token twice, once per exponent rule (routed, shared): [numerics.md](numerics.md) §3.

## 4. Drop a launch by recomputing its result

Recomputing a small decision is a candidate, not an unconditional rule. In the toy MoE (32 experts, top-4) it saved 2.89–3.40 µs at m=2 and 3.22–3.87 µs at m=16 on AKLR's timer before #33, and 1.24–1.80 µs replayed at m=2 (4 runs). In the synthetic fp8 VALU shapes (256 experts, top-8; H6144 with inter 256, H3072 with inter 512; m = 1, 2, 8, 16; one run, 16 rotated rounds per Point), recomputation in every gate-up, down and combine consumer was slower on both rulers. Recomputing only in the gate-up, which publishes ids and weights in that launch, helped at H6144 m = 1 and 8 (replay savings 2.87 and 2.45 µs, AKLR's timer 2.06 and 2.44 µs; the A/A CIs of both include zero); the apparent replay savings at m = 2 matched the bias of A/A controls in both shapes, H3072/m1 also had replay bias, m = 16 was slower in both shapes, and at H3072/m8 the two rulers disagreed in sign. These are paired CIs within one run, not 16 independent experiments; deriving GEMM jobs from counts (§4.2) was not timed *(field, E10, E13)*.

### 4.1 Top-k in every gate-up program (≤ 16 tokens)

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
- **Where.** The DeepSeek and MiniMax files at 1–16 tokens, and the Qwen2 router+experts files at 1–4 tokens, whose W13
  programs replay the top-k pops on the 512 router keys (S26). Variants at 8–16 tokens: the MiniMax MXFP4 64-expert
  rank runs one program per local expert that keeps the tokens for which fewer than 8 experts beat it (S21), the FP8
  one adds a selection launch and runs one gate-up program per local expert (S30); DeepSeek `inter_dim512` at 16 builds
  per-expert token bitmasks with `buffer_atomic_or` in a selection launch (S20).
- **Measured.** At 43e112e all 15 DeepSeek and MiniMax Points above 2.0, at 1–8 tokens, are served by files that recompute the routing inside the gate-up launch *(field, E1)*. No file runs the same design with a separate selection launch; on a toy top-4/32 MoE dropping it saved 2.89–3.40 µs at 2 tokens and 3.22–3.87 at 16 on AKLR's timer before #33, 1.24–1.80 replayed at 2 (4 runs; top-8 of 256 and deriving jobs from counts not tested) *(field, E10)*.

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
list its selection launch writes. These files stored 1.030–1.194 at 43e112e *(field, E1)*, but no Point compares
deriving the job with a job list the selection launch writes, and ratios across Points cannot stand in for that (§11). Lookup forms and the
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
Triton 3.8.0 ("error: TODO: unsupported scan layout") *(field, E9)*; scan a 1-D `BlockedLayout`
([debugging.md](debugging.md) §4). An atomic-ticket sort thus costs two launches (selection, scatter), not three.
The FlyDSL `4fbd` analog repeats the full M×9 route scan, expert histogram and padded-block prefix in every stage-1 workgroup; each live workgroup then ranks only its own expert’s items and retains its 16-row block: `../moe/few_tokens.md` §2.

## 5. Router GEMV and top-k

### 5.1 Compute the logits at the reference's rounding point

| Contract's routing | What the kernels compute | Where |
|---|---|---|
| DeepSeek: logits in bf16 (what SGLang's gate returns on ROCm), sigmoid, plus an fp32 bias | bf16 MFMA tiles (16×16×32) write fp32 split-K partials; the consumer sums them in split order and rounds once to bf16 | 5751 (S3) |
| | at 2 tokens a full-K dot per expert with `gl.dot_fma`, both tokens sharing the weight load (read from the source); `_router_pair_packet` compiles to 48 `v_dot2c_f32_bf16` and no MFMA *(field, E8)* | d3bb (S4) |
| Qwen2: bf16 logits, top-10 by logit, the lower id on a tie | the same split-order sum and one bf16 rounding | c388 (S9) |
| | router+experts: float64 products (f64 VALU and no f64 MFMA at 1–4 tokens, `v_mfma_f64_16x16x4_f64` at 32 *(field, E8)*), rounded once to bf16 with a midpoint fix against double rounding (on 512 crafted near-midpoint logits f64 → f32 → bf16 erred on 260, 7d7e's helper on none *(field, E4)*) | 7d7e (S26), 99f4 (S11) |
| MiniMax: fp32 router weight, fp32 logits | fp32 multiply and `gl.sum` at 1–16 tokens (read from the source; FP8's e75e `_prepare_keys`, 1–16 tokens, compiles to no MFMA); the fp32 weight split into three bf16 terms on bf16 MFMAs (the bf16 hidden makes each product exact; the split read from the sources): the routers of MXFP4's 64-expert files at 2048–16384 tokens (cb88, b94f, 62df, f839, 8b22: 1.318–1.735 at 43e112e) and FP8's f770, 82e9 run bf16 MFMAs and no fp32 MFMA; or fp32 MFMA: 5cac's `_prepare` (32–64 tokens) `v_mfma_f32_16x16x4_f32`, 8411's `_router_projection_7` `v_mfma_f32_32x32x2_f32`, the MXFP4 256-expert files' routers at 1024–16384 tokens 16×16×4 (bb66, 147b) or 32×32×2 (e6fc, 51ae, 8411: 1.261–1.423 for the five) *(field, E1, E8)*; much else differs between these files: no ablation. On 13 of these router branches, each with a random and a cancellation input, the logits' error against fp64 was no larger than an fp32 computation's in L2 on 25 of the 26 inputs (e6fc's cancellation input 1.00009×) and in max abs on 23 of 26; on two more inputs with a nonzero residual, 62df's and f839's fp64 correction ran and both errors were 0 *(field, E20)*. | e75e (S27), 8a5a (S10); 5cac, 8411 |

Check the reference's rounding points first ([numerics.md](numerics.md) §5), then compare each output against its
contract. On `qwen2_moe_router@1.0`, c388 at m = 256 (one input set): swapping only token 0's first two ids, every other
id and `topk_weights` bitwise unchanged, failed validate with `topk_ids` `numeric_mismatch` (violation fraction 2/2560)
while the committed file passed twice; in three earlier runs on the same input, sending ties to the higher id (174 of
256 rows changed) or rounding the split partials to bf16 before the sum (66 rows) failed, and split-K 4 → 2 changed no
id and passed *(field, E3, E18)*. Keep the ids and their order; this does not show that every change to logit rounding
fails, and replacing a single id was not tested.

Skipping a rounding step need not fail or help: a2bb (DeepSeek-V2, 8 tokens) ranking the unrounded logits passed
validate twice and changed no token's experts, with no resolved change in error; d9cb (gated attention, 32 rows)
keeping the attention in fp32 before the gate also passed twice, while its scales' relative L2 against a float64
reference without that rounding rose from 0.003233 to 0.003269, a paired difference too small to resolve *(field, E18)*.

At 5751 (1 token), on four saved inputs with near or exact ties, replacing only slot 7's expert in both gate-up and down moved `out` 6.21, 112.2, 6.13 and 9.47 times as far (L2) as removing only the bf16 rounding of the activation product; against the reference, the two total errors differed by 1.22, 3.03, 1.23 and 1.28× *(field, E21)*: on these inputs a wrong expert cost more than a skipped rounding.

**Three bf16 terms are a candidate, not a preference rule at many tokens.** Eight fixed register-MFMA probes
(grid 1536, 3072 iterations) give estimated fp32/router-to-MXFP4-expert matrix-core time ratios of 1.244–1.263,
and three-bf16 ratios of 0.244–0.246, for K=3072, 256 experts, top-8 and inter=512. These omit memory and weight
splitting costs, are not router or expert GEMM latencies and do not establish performance at other token counts
*(field, E17)*; four earlier runs on the same shapes gave 1.255–1.284 and 0.2386–0.2452 *(field, E2)*.

At K=3072, (M,E)=(1024,64),(4096,64),(16384,64),(1024,256),(4096,256),(8192,256), each synthetic logits arm
was selected independently from a bounded search (BK32, stages1, fp-fusion on; BM/BN64/128/256, warps4/8,
BN≤E, BM·BN≤16384; 6 candidates/arm at E64, 12 at E256), not a global optimum. On random input (seed 1201),
three-bf16 logits kernels took 0.457–0.512× the two fp32-MFMA baselines' time, including W splitting each call
but excluding sigmoid, bias, top-8, normalization and expert GEMMs (N=1 run/Point, 32 balanced AB/BA pairs).
Across eight Points × eight seeds × three input modes × three arms, 217 of 576 accuracy comparisons failed the fp32
top-8 contract (all 192 near-tie inputs, 25 random or dynamic ones). At M16384/E64, random seed 705, both fp32
arms passed while split failed, despite lower split logit error (max abs 4.165e-6 vs 1.149e-5 against fp64).
Lower logit error does not guarantee unchanged routing ids. M8192/E64 and M16384/E256 had no contract-correct
split tuning candidate on seed 1201: their accuracy inputs were all checked, but they were not timed.
Finite-sample noninferiority held for dynamic at M1024/4096,E64 and random/dynamic at M1024,E256; dynamic
speed was not measured, so these accuracy results are not joint speed/accuracy recommendations. Small M,
AITER, complete-router and whole-MoE performance remain unmeasured *(field, E17)*.

### 5.2 Size the router grid for bytes in flight

At up to 256 tokens the router streams a 3–4 MB weight (estimated), and a streaming probe approximately tracked bytes in
flight across its main 256–8192-program matrix; it also measured 64/96/128-program controls and 192-program
held-out arms, where matching nominal bytes did not reproduce the larger-grid rate *(field, E10)* ([tuning.md](tuning.md) §4;
inferred for a router). One design splits K over small MFMA tiles whose fp32 partials every consumer sums in split
order: 5751's 192 one-wave tiles at 1 token, 1024 one-wave programs in a1c4 and 1204 at 128–256 tokens, MiniMax's 512
CTAs at 128–256 (01ce, 4da3, 8a5a, 94af), the Qwen2 router's 256 CTAs from 8 to 256 tokens (4801 at 8: 1 × 8 splits × 32
expert tiles of one wave; c388 at 256: 8 token tiles × 8 expert tiles × 4 splits of four waves, S9). The other gives
each expert (each token block and expert in e75e, fbd9, 3df5 above 1 token) one full-K program that writes a finished
probability or ordered key, sparing every top-k consumer of §4.1 the split sum (d3bb and the MiniMax files at 1–16 tokens: e75e,
1c17, 6557, fbd9, b065, f82c, 3df5; e75e `_prepare_keys`, S27) *(field, E11)*. No Point compares the two router forms. Among half/committed/double splits, the committed
256 four-wave CTAs were fastest or tied on three Qwen2 routers: c388/m256 half 1.33–1.34× call time, double
1.02–1.04× (N=8 runs); cd47/m128 1.22×/1.03×; a878/m64 1.04×/0.99× (double tied; N=1 each).
Bytes in flight alone did not size c388: 128 CTAs four K tiles ahead (nominal bytes equal to 256 CTAs two ahead)
took 1.26×, and 512 CTAs one ahead 1.03× (N=1, 4 alternated rounds). At 256 CTAs one tile ahead took 1.12–1.14×
(N=6), four ahead 0.97× (N=1); committed is not best across all look-ahead settings *(field, E10, E14)*.
Not speed knobs here: the
`amdgpu-sched-strategy` attribute, `amdgpu-ieee false` (accuracy unchanged) and, at 8 tokens (4801), the file's register
path for the hidden tile in place of async staging through LDS (it also changes the weight-load form) *(field, E6)*. At
1024 tokens and up K is split fewer ways or not at all (DeepSeek full-K, Qwen2 2 splits at 2232 tokens and none at 4096).

### 5.3 Top-k with cross-lane operations

For the measured top-8 of 256 (4 scores/lane, 16384 tokens), one wave/token took 12.96–13.18 µs over 3 runs,
four waves 37.68–38.32 µs, the same ids, no scratch, 0 and 31 `s_barrier`; 512 experts at 8 per lane and one CTA serving four tokens were not tested *(field, E11)*. Four waves on one token synchronize each round: 147b's `_select_5` compiles to
41 `s_barrier`, 25 LDS writes and 25 reads, the one-wave-per-token selectors of 8411 and b94f (b94f's CTA serves 4
tokens) to no `s_barrier` (LDS not counted) *(field, E8)*; no Point times it. Gluon 3.8.0 has no ballot, lane-read or
DPP builtin, so the kernels write them in inline assembly ([gluon/wave_ops.md](gluon/wave_ops.md) §6).

| Design | One round | Where |
|---|---|---|
| Election by ballot | `gl.max` over the wave; `v_cmp_*_e64` writes the lanes holding the maximum to an SGPR pair, `s_ff1_i32_b64` takes the first, `v_readlane_b32` broadcasts its id; lanes hold consecutive ids, so the lowest id wins a tie | DeepSeek 5751 `_elect_rank` (S3) |
| Keyed min, no inline asm | with retired experts' scores at −inf (as its caller sets them), `gl.max` of the scores, then `gl.min(where(available, where(score == max, e, e + 256), 512)) & 255`: the lowest available id holding the maximum (0 wrong of 4000 tie-heavy rows; unmasked retired maxima untested *(field, E11)*) | DeepSeek 3c6e `_next_expert` (S20) |
| Per-lane sorted queues | sort each lane's keys once (a 5-compare network on 4 keys, or two 4-deep queues on 8); a round compares only the queue heads and pops the winner's queue | MiniMax FP8 5cac (S25) and e75e `_rank_stable_queue` (S27); Qwen2 c388, whose ballot winner goes into lane `slot` with `v_writelane_b32` (S9) |
| Order-preserving integer keys | the score's bits made monotonic, −0 folded into 0, the inverted expert id in the low bits: one `gl.max` returns the score and the lowest id | Qwen2 7d7e (16-bit bf16 key above a 9-bit id, S26). Two reductions per round instead: MiniMax e75e keys the fp32 score alone (NaN to the largest key, S27) and takes the lowest id with a second `gl.min`; 5cac packs the inverted id into 64 bits (S25) but at 256 experts still reduces the head scores with `gl.max`, then the ids with `gl.min` |
| Packed float64 keys | for finite scores, the same order inside a double's mantissa in [1, 2), so one float compare decides (`v_max_f64` in inline assembly in 94af); fbd9 maps NaNs to NaN-valued keys outside that interval | MiniMax fbd9 (S28), 94af (S29) |
| Packed uint64 keys, partitioned reduce | keys 4 per lane in a [4, 16] layout (a wave's four 16-lane rows); a round pre-reduces each lane's 4 in registers, runs a 16-lane `gl.reduce(…, _argmax_packed)` that compares the scores as floats and the ids on a tie, gathers the 4 row roots from lane 15 of each row and finishes 4-way; the winner's key becomes a tombstone | MiniMax 8411 `_select_partitioned_2` (S6); b94f `_select_wave_rows` on (value, index) pairs (S13) |
| Rank counting, no rounds | one program per candidate expert counts, per token, the experts that beat it (higher score, or equal score and lower id); a count below k selects it and names its slot | MiniMax FP8 1c17 `_select_rank` (S30), exact at 8 and 16 tokens; MXFP4 3df5 (S21) returns a token bitmask: measured M2 keeps 8 via its id tie rule, M8 counts only strictly higher scores and a tied 8th/9th selects 9; M16 not run *(field, E11)* |

No Point compares designs. Election runs slot + 1 argmax rounds (from the code); over 16384 one-wave programs d3bb's
took 9.7–10.7 µs at slots 0–4 and 11.3–11.4, 12.8–13.0 and 14.3–14.4 µs at slots 5, 6, 7 in 3 runs, not linear in the
slot (R² 0.724–0.774) *(field, E11)*; the queue and rank-counting costs are estimated from the code, and the counting
that was timed ran in a test kernel, not in 1c17. The selecting program also
renormalizes weights and, in the ticket designs, takes each route's ticket (§6), in the same launch. Traps:

- **One call, one tie rule.** Every §4.1 design selects at least twice per call: the program that publishes
  `Ids`/`Weights` and each gate-up program that recomputes its expert (5751: `_select_routes`, `_expert_at_rank_peeled`,
  S3; e75e: a `gl.min` round and per-lane queues, S27; f82c both in one launch, S31). The copies share the score
  (split-order sum, bf16 rounding) and the lower-id rule.
  Keep the copies consistent: at 5751 (1 token), changing only the gate-up copy's bf16 rounding or its tie rule made producer and consumer disagree on a slot's expert on 12 of 13 inputs, and changing both alike restored agreement on 36 of 36; summing the 12 split partials in reverse order changed no id, weight or output on four saved inputs, which did not exercise that order *(field, E21)*. Through validate, the copy without the bf16 rounding failed (`capture_unsafe`, `numeric_mismatch`) and the copy with the higher-id tie rule passed *(field, E18)*.
- **Read the winner from fixed lanes.** `gl.reduce` combines lane pairs in a butterfly; with a non-commutative combine
  (NaN scores) lanes disagree on the winner (on NaN rows only, in a test), so f82c reads it from lanes 60–63 with
  `ds_bpermute_b32` (S31). Tested e75e/fbd9 keys fold ±0 and 8 fp32 NaN payloads to one score field each; 7d7e folds ±0 but gives 7 bf16 NaN patterns 7 keys *(field, E11)*.

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
  `and`, `xor` int32 or int64 (3c6e's masks are int32): a mismatch fails at trace time, `xchg` later *(field, E9)*.
- **One atomic per distinct element.** Copies that the layout makes (redundant lanes) are predicated off (S23): 8411's
  `[8]` `ids` in a 64-lane layout take 8 tickets *(field, E4)*. Copies you build yourself are distinct elements: d861
  widens its 8 winners to an `(8, 8)` tensor with `+ replica[:, None] * 0` to fill a wave, and masks all but one copy
  with `owner = replica[:, None] == 0` (S7), or each copy takes a ticket.
- **Job queues (32–256).** Each expert owns a fixed region of the sorted buffer, as long as the routes it can receive:
  `PAD`, m rounded up to the tile (1204, S5), or `SORTED_STRIDE`, the routes rounded up to 16 (d861, S7). A route's
  row is `expert · stride + ticket`, written in the selection launch with no scan; a second relaxed atomic opens a job
  every 16th (or BM-th) ticket of an expert and writes it to `Jobs` (d861).
- **Persistence is rare.** Qwen2 8698 at 64 tokens loops its W13 programs over a device-counted list on a 3072-program grid
  (complete 16-program route tiles assigned in 256-program cohorts, S36); its down stage uses a separate
  4096-program grid; MiniMax FP8 94af at 256 runs
  `_w13_persistent` on a 48 × 96 grid (S29), the only kernel named persistent at the 96 Points *(field, E11)*.
- **Sorts without global atomics.** The MiniMax MXFP4 64-expert rank at ≥ 1024 tokens gives lane j the 64-bit mask of the
  routes in its stripe whose expert is j with seven wave ballots, then popcounts (abc7, S8); the FP8 64-expert rank
  packs four 8-bit counters per word. Both fix the row order; Qwen2's LDS atomics at 64 and 128 tokens
  (`atomic_scatter_add` in LDS, one CTA) do not (8698, S36). No Point compares them with a ticket sort.
- **Graph capture.** Grids depend on shapes only (same grids on two routings at all 82 Points *(field, E11)*): a
  worst-case tile bound on the host (MiniMax `cdiv(min(routes · bm, routes + E · (bm − 1)), bm) · bm`; DeepSeek `A +
  (8m − A) // block_m` with `A = min(256, 8m)`, the most experts that can hold a partial tile, S6, S18) and an exit on a
  device-written total (S6; what else breaks capture: [authoring.md](authoring.md) §8). Scratch is `torch.empty`, filled
  by an earlier launch of the call (S6): no file uses `torch.zeros` or a fill, a kernel more at the dispatch floor (§2).
  In FlyDSL: [../moe/grouped_group_mapping.md](../moe/grouped_group_mapping.md) §3, §4.
- **Ticket order is not deterministic.** Use tickets only to place rows whose result does not depend on the row, and sum
  a token's routes in slot order (§10): tickets differed in 19 of 20 runs, the slot-order sum kept one bit pattern
  *(field, E4)*. `bounded` allows run-to-run differences, `bitwise` would not ([numerics.md](numerics.md) §1, §8).

## 7. Gate-up GEMM

A gate-up program loads its expert's preshuffled weight words straight into the MFMA operand layout
(`assert_trivial=True` compiles for 7e9c: no data moves between lanes *(field, E9)*; S12, S24) and issues `mfma_scaled`.
Its E8M0 scale words go through LDS (7e9c's gate-up; 8411's activation scales, preloaded per tile and gathered per K
step, S6), are shifted out of their word per element and converted to the scale layout (d3bb, S4), or load in the scale
layout directly (7e9c's down, S12). At ≥ 1024 tokens the activation tile the CTA's waves share goes through padded LDS (8411).
Writing those loads, the `shuffled_16x16` offsets and the scale layout: [gluon/mfma.md](gluon/mfma.md) §7,
[gluon/memory.md](gluon/memory.md) §13, [gluon/layouts.md](gluon/layouts.md) §6.3. Particular to MoE:

- **Instruction per weight format** ([gluon/mfma.md](gluon/mfma.md) §2.1–§2.3). MXFP4: `mfma_scaled` on e2m1 with both
  scales (S4), 16×16×128 and no 32×32×64 in the 14 DeepSeek and MiniMax MXFP4 schemas whose Points are all ≤ 16 tokens
  *(field, E8)*. FP8 at 128 × 128 block scales (MiniMax): `mfma_scaled(a, None, "e4m3", b, None, "e4m3", zeros)` per
  128-K group, the fp32 scales applied after each group (e75e's 4–16-token gate-ups, S27; the dense GEMMs' form,
  [gemm.md](gemm.md) §2). On a rank-3 layout (routes or K splits batched over warps), which `mfma_scaled` aborts on, the
  files run the fold on plain fp8 `mfma` [16, 16, 32] (`v_mfma_f32_16x16x32_fp8_fp8`, half the MACs per cycle)
  *(field, E2)*, compiled into e75e's `_down_combine` and `_project_w13_internal_packed2` and 5cac's `_w13_local_split`.
  One 0982 `_down_projection_2` variant also uses plain fp8 MFMA, but its `NATIVE=m == 64` path has a rank-2
  layout and is not evidence of a rank-3 requirement *(field, E8)* (layouts and Points: S27, S34). 5c35 (Qwen2 experts, m = 8) casts both MFMA operands to bf16 in
  `_project_gate_up` and `_project_down_and_sum` (S19); all 254 finite e4m3 codes, both NaNs and signed zero convert to
  bf16 exactly (on the CPU) *(field, E15)*; the kernels were not run for this.
- **Gate and up in one column tile.** A BN-column tile holds BN/2 gate rows and the matching BN/2 up rows
  (`n = column * (BN // 2) + n % (BN // 2) + n // (BN // 2) * (N // 2)`, S5), so the epilogue applies SwiGLU (§8).
- **Few rows on a 16-row instruction.** For one token, replicate it over the 16 rows and store row 0 (d3bb, S4); fold K
  into the rows (6557: with `gate_fold` 4, row partition p and column block p read the same K chunk and the diagonal
  blocks are summed, S14); put the routes' experts side by side along N and keep the diagonal (§9); or swap A and B so
  the weights fill M (Qwen2 5c35). The 4-row instruction ran 1.155× the useful MACs of 16 padded rows *(field, E11)*:
  Qwen2 99f4 at 32 tokens runs routed tasks of at most 4 rows on `instr_shape=[4, 64, 64]` (`v_mfma_f32_4x4x4_16b_f16`),
  the shared expert on 16×16×32 (S11). Register maps: [gluon/mfma.md](gluon/mfma.md) §8, §4.
- **Gate-up grid at 1–2 tokens: not simply wider.** d3bb sums split-K partials in `_activate_quantize` (S4). At m=2, splits 4→1 (2306→578 one-wave programs) took 1.079× call time; 4→8 (4610) 1.050×. At m=1, 5751 1153→577 took 1.020×, →2305 1.028×; 7e9c 1728→864 took 1.035×; e75e column tile 8→16 (385→193 four-wave programs) 1.031× (one run, 4 alternated rounds per arm; scratch and reducing launches changed with the split, every arm accurate) *(field, E14)*. An earlier d3bb comparison of 4 → 2 splits (2306 → 1154 programs) took 1.004× (speedup 2.183 → 2.175; paired 95% CI of the reciprocal ratio 0.991–1.001), unresolved *(field, E11)*. These Points set neither a widest-grid rule nor a bytes-in-flight sizing for MoE; the bytes in flight were counted only up to the expert-selection loads *(field, E14)*.
- **Tile height at run time (DeepSeek).** A descriptor carries the live row count and the program branches to a body compiled for 16, 32, 64 or 128 rows; the job list puts full tiles first, then the tails by height, tallest first (2422, S16). c524 pairs two bodies instead of padding to the next height: 65–96 rows as 64 + 16 or 32 at 1024 tokens, 129–192 as 128 + 16, 32 or 64 at 3072 (S32). MiniMax 01ce and 4da3 (128–256 tokens) drop to 16- or 32-row bodies for short tails, f770 runs a 128-row tile with an empty second half as 64 rows from 4096 tokens (S33); the other MiniMax or Qwen2 gate-ups do not select a different compiled tile height from their live rows. MiniMax b94f, cb88, f839, 8b22 and 62df still branch on `live == 64` to select an unmasked full-tile path versus a masked tail path at the same 64-row height. No Point compares variable and padded heights. Over 100 calls on one fixed input 2422's tile stayed in its 16-row body with one bit pattern, and on tiles of 15–17 and 31–33 live rows the neighbouring body sizes gave the same bits *(field, E22)*.
- **The weights' cache policy is per Point.** Dropping `.cg` from d3bb at m=2 cost 0.9–1.2% in two of three runs (every pair; speedup 2.205→2.180, 2.217→2.195), but nothing in the third (2.171→2.171, pairs both ways). Dropping it from 1d27/m256 cost 2.6–3.5% (1.177–1.189→1.144–1.150); adding it to 2422/m256 cost 0.2–0.7% (1.012–1.040→1.009–1.034), each in every pair of four runs *(field, E7, E16)*. 2422 leaves `.cg` off its 128-row gate-up tiles above 128 tokens, at 256 the shared expert's two tiles (256 rows; a routed expert's about 8, estimated), whose weight address has no row-tile term (S16), so both read the same columns; a cache hit on the second is inferred (a toy in which two programs read each block ran 4 % faster without `.cg` *(field, E7)*). Put `.cg` (`cache_modifier=".cg"` on `gl.load`, `cache=".cg"` on `buffer_load`; it sets `nt`) on weights the call reads once, leave it off weights a second tile re-reads, settle it per Point ([gluon/memory.md](gluon/memory.md) §6). In one run over 12 Points of seven kernels, `.cg` won at seven of ten predominantly read-once Points (1.6–9.6%; six had every pair favorable on both rulers, e75e/m2 only the aggregate direction), with no consistent effect at three m=1 Points and no measured loss. It lost at both re-read Points: 2422/m256 (0.2–0.7%, N=4) and 8411/m16384 (17%, N=1). Reuse was classified from the schedule and measured routing, not cache-hit counters; the reuse side covers only two Points *(field, E16)*. Store hints are unmeasured here: c524 writes cross-launch partials `.wt` and its output `.cs` at 2048 and 3072 tokens (S32), the Qwen2 routers their partials `.wt`, 4801 through `gl.store` (S9), which, unlike `buffer_store`, waits after each `.wt` store *(field, E9)*.
- **Program order.** e9fd's down at 8–16 tokens makes a column tile's program ids congruent mod 8 (S38), so with round-robin dispatch they share an XCD on any stream (the absolute XCD index rotates on a non-default stream). 8a5a walks its job list in 8 contiguous stripes (S10); f770 keeps an M tile's N tiles on one XCD (S33). All placement here is derived from the formulas; none was measured on MoE kernels: [gluon/memory.md](gluon/memory.md) §14.
- **`waves_per_eu` did not change the time where measured.** The gate-up of 51ae and 8411 (MiniMax MXFP4, 256 experts) sets 3 above 3072 tokens (S15); removing 51ae's at 8192 tokens left the speedup at 1.44 (1.441 → 1.440) *(field, E7)*; 146 VGPRs with it, 150 without, no spills *(field, E9)* ([tuning.md](tuning.md) §8, [debugging.md](debugging.md) §2).
- **A Point's constants are coupled.** Changing 6557's `gate_fold` from 4 to 1 alone at 1 token crashed every bench run *(field, E7)*: LLVM's code generation dies with SIGSEGV on `gate_fold` 1 with 4 warps, a 128-wide K tile and the `amdgpu-sched-strategy=iterative-ilp` attribute the file sets at m ≤ 2 (S14); `gate_fold` 1 compiles with 1 warp, with a 256-wide K tile, or without that attribute *(field, E9)*. Change constants together, from a combination that compiles elsewhere (here 6557's 16-token branch) ([debugging.md](debugging.md) §6).

## 8. Activation and requantization

| Situation | Where SwiGLU and requantization go | Examples |
|---|---|---|
| One program owns full K for its gate+up columns and a whole quantization group of the output | the gate-up epilogue | MXFP4 queued and sorted designs (b94f, S13), DeepSeek from 4 tokens and 7e9c at 2 |
| The gate-up is split along K over programs | a small launch that sums the split partials in order, then SwiGLU and quantization | d3bb and 5751 `_activate_quantize` (S4), 7e9c at 1 token (`_finish_gate_up`), Qwen2 5c35 *(field, E11)* |
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
  GEMM loads A like B, where row-major words cost LDS *(field, E9)* (S17; [gluon/mfma.md](gluon/mfma.md) §7).
- The MXFP4 and fp8 encoders (exponent rule, NaN guard, `v_cvt_scalef32_pk_fp4_f32` packing, saturation) and the
  little room the MXFP4 bound leaves beyond the activation quantization: [numerics.md](numerics.md) §1–§3. FlyDSL's
  GLU-in-epilogue numbers: [../moe/grouped_lowprec.md](../moe/grouped_lowprec.md) §10.
- Most DeepSeek launches (86 of 100 sites, all three of a2bb's, S22) pass `enable_fp_fusion=False`, MiniMax and Qwen2
  only 96e5's down (16 tokens); on at 8 tokens, a2bb's code is unchanged: 1.147 → 1.146, accuracy passed *(field, E7)*.

## 9. Down GEMM

- **≤ 16 tokens: down+combine is a candidate, not a uniform win.** In the toy top-4/32 MoE at m = 2 and 16, fusing down and combine saved 2.94–3.73 µs on AKLR's timer before #33 (4 runs; its replay saving and output equality alone not measured; all fusions against none: §2) *(field, E10)*. In the synthetic fp8 VALU shapes at m = 1, 2, 8, 16 (one run, 16 rotated rounds per Point; slot order as the reference sums), it added 0.25–8.18 µs of replay time on average (savings −8.18 to −0.25 µs); at H6144/m2 the CI crossed zero and some small differences fell within the bias of A/A controls, at H3072/m8 and m16 it was slower on AKLR's timer too, and at none of the Points did all fusions together reach 0.85× the unfused time *(field, E13)*. Neither shows how MXFP4 MFMA kernels fare. The source form in 7e9c (S12) puts routes in MFMA A rows and experts' column slices side by side in B, keeps the diagonal blocks and sums them weighted in slot order, without partials or atomics:

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
- **≥ 1024 tokens: operand reuse across tiles (no gain where ablated).** MiniMax 8411 holds a 64-row activation tile in
  registers across a strip of column tiles (`_down_reuse_fp4_2`); DeepSeek loads one expert's weight panel and scales
  once and streams activation tiles through it (`_paired_down`, `_routed_down`). c524's schedule leaves that off at 2048
  tokens, which stored 1.329 against 1.402 at 1024 and 1.407 at 3072 at 43e112e *(field, E1)*; turning it on changed the time by
  −0.04 % to −0.5 % in 3 runs *(field, E3)*: no gain (activation-tile reuse not ablated). Atomics against a
  fixed-order reduce, on FlyDSL kernels: [../moe/grouped_sorted_tokens.md](../moe/grouped_sorted_tokens.md) §6.

## 10. Combine and the shared expert

- **Combine in slot order.** `out[t] = Σ_slot w[t, slot] · down[route]`, slot 0 to 7, from bf16 route rows (8411
  `_combine_rows_3`, S6). The route rows are a `routes × H` bf16 scratch the down writes and the combine reads: 805 MB
  each way at 16384 tokens for MiniMax MXFP4 with 256 experts (estimated from its `torch.empty` shape). Other files
  reorder it: the MiniMax MXFP4 64-expert files at ≥ 1024 tokens store the down rows in sorted order, read through an inverse
  map kept in the ids buffer (b94f, S13); e6fc writes them by sorted row in 256-column chunks, read through a position
  map; DeepSeek at ≥ 1024 tokens stores dense expert-sorted rows and points each route's record at its row (3432) (S35). No
  Point compares the hand-offs. Designs at ≤ 16 tokens fuse the combine into the down (§9); 1c17 and 3c6e keep a launch for it.
- **No float atomics** in the 65 files *(field, E8)*: integer counters, cursors, job indices, bitmasks, LDS markers.
  Qwen2's 8698 and d425 reorder rows across calls, yet they and 29 global-ticket files repeat outputs *(field, E11)*.
- **DeepSeek's shared expert (id 256, weight 1)** changes the plan by size: at ≤ 16 tokens it is the token's ninth
  route; at 32–256 tokens its jobs run first, its down result is stored straight into `out`, and the reduce adds the
  8 routed partials on top of it (1204, S5); at ≥ 1024 tokens the shared expert's down epilogue gathers the 8 routed
  partials of each token and writes `out`, so there is no combine launch (699f, S18).
- **Qwen2's shared expert (id 512)** is weighted by `sigmoid(hidden · shared_gate_weight)`; its dot products come
  from the shared expert's own gate-up programs (slot 10, tile 0), which add one partial dot per K group (5c35, S19).

## 11. What the measurements say

The committed-file ablations use matched copies and alternating original/candidate measurements on one MI355X
with `python -m aklr validate`; their run counts and rounds are stated per result, and the probes state their own.
Committed-against-itself runs spread 0.9–1.0 % on the two Points where it was measured *(field, E5)*. CI's stored
speedups at 43e112e, per Point, against SGLang's path on aiter; d908d01 (CI run 42000899019) re-stored 57 of the 96
Points, by −6.3% to +7.0% (median 0.9%), the values 0e02a81 still holds *(field, E1)*:

| Scenario | ≤ 16 tokens | 32–256 | ≥ 1024 |
|---|---|---|---|
| DeepSeek rank inter 256 (`inter_dim256`) | 1.245 (16) – 2.186 (2) *(field, E1)* | 1.030 (128) – 1.160 (256) | 1.329 (2048) – 1.420 (4096) |
| DeepSeek rank inter 512 (`inter_dim512`) | 1.097 (16) – 2.093 (1) *(field, E1)* | 1.043 (256) | 1.152 (1024) – 1.391 (2048) |
| MiniMax FP8, 256 experts | 1.732 (16) – 4.115 (1) *(field, E1)* | 1.106 (128) – 1.173 (64) | 1.248 (2048) – 1.388 (16384) |
| MiniMax FP8, 64 of 256 | 1.325 (16) – 3.137 (1) *(field, E1)* | 1.102 (64) – 1.167 (32) | 1.216 (1024) – 1.785 (16384) |
| MiniMax MXFP4, 256 experts | 1.595 (16) – 3.444 (1) *(field, E1)* | 1.178 (256) – 1.230 (64) | 1.261 (4096) – 1.423 (8192) |
| MiniMax MXFP4, 64 of 256 | 1.555 (16) – 3.633 (1) *(field, E1)* | 1.190 (128) – 1.275 (64) | 1.252 (1024) – 1.735 (8192) |
| Qwen2 router+experts | 1.119 (4) – 1.172 (1) *(field, E1)* | 1.063 (32) | — |
| Qwen2 router | 1.074 (8) – 1.219 (16) *(field, E1)* | 1.295 (64) – 1.967 (256) | 1.226 (4096) – 1.416 (2232) |
| Qwen2 experts | 1.208 (8) *(field, E1)* | 1.076 (128) – 1.171 (64) | — |

Each value belongs to the whole file that serves the Point, and CI's runs of unchanged sources move a stored ratio by
2 % at the median, 6 % at p90 *(field, E1)* ([tuning.md](tuning.md) §1): such gaps say nothing about designs. In every
DeepSeek and MiniMax scenario the smallest ratio sits at 32–256 tokens, where both sides stream nearly every expert
(§2), the largest at 1–2 tokens, where the baseline runs 8–11 device operations to AKLR's 3–4 *(field, E11)* (the links
inferred). 51ae (8192 tokens, 1.423) and 8411 (16384, 1.389) share their kernel source but for the docstring, as do f839
(8192, 1.735) and 8b22 (16384, 1.619) *(field, E1)*.

## 12. Traps when editing these files

- **Dead branches.** Most files carry paths for other token counts (5751 a 2-token projection and a non-peeled top-k,
  c388 configurations for 64 and 128 rows): trace your Point's constexpr branch first
  (`harness/skills/aklr-port-generated-kernels/references/porting-notes.md`).
- **Scenario constants are baked in, and `run` checks few of them.** DeepSeek multiplies the routed weights by 2.5 in
  device code, e9fd sums 12 router splits in `_rank_expert`; the MiniMax 256-expert files ignore `expert_offset`
  (right while the rank holds all 256); the MXFP4 64-expert files at ≥ 1024 tokens read an expert from 6 bits
  (`descriptor & 63`); Qwen2's router+experts `run` ignores `topk` and `out_dtype`; cd47's `(256,)` router grid with
  `split = 8 if rows == 64 else 4` holds for 64, 128 and 256 rows only (S39). Re-derive them for another scenario.
- **Write every token's row at a 64-expert rank.** About 10 % of tokens have no route to a held expert (estimated,
  uniform routing: C(192, 8)/C(256, 8)); their `out` row is zero and must still be written: bench's outputs start as NaN
  ([authoring.md](authoring.md) §8), and an abc7 copy that skipped them failed *(field, E4)*. The committed combines
  loop over all 8 slots of every token and add nothing for a non-local route (abc7 `_combine_14`, S8; 0982, S34).
- **Buffer ops stop at 2 GiB from one base.** MiniMax MXFP4's e6fc, 51ae and 8411 have host extent guards for their router/scale-buffer paths
  (8411: three `… < 2**31` tests, S6); their data/output buffer choices also depend on fixed-Point policy flags.
  147b guards in device code and bb66 by token count. These are not a universal guard for every buffer base. Route-row scratch is routes × H: a
  larger Point or H would cross it (a buffer load read 0 past 2 GiB on a generic tensor *(field, E4)*; memory.md §3).
- **Docstring launch lists can be short.** c3e3 lists two launches; at its 4-token Point the host code runs three
  (`_padded_down` follows), as 4 other lists differ from the trace *(field, E11)*: follow the host function.
- **A worst-case grid may reach descriptor or sorted-row entries that no route fills; a device-total exit can prevent some descriptor reads.** Initialize the sentinels (out-of-range route ids
  the consumer masks: 8411 fills `Sorted` with `R`, the route count, S6; d861 with `CAPACITY`, S7) in an earlier
  launch of the same call, also the tails no route fills (without its fill d861 crashed *(field, E4)*).
- **Scratch aliasing: source order, not a GPU safety proof.** 699f aliases router logits with the activation buffer and quantized input with partials (S18); in its host code each aliased buffer is written, read and overwritten in launch order (kernel.py:1307, :1324, :1338), read from the source *(field, E15)*; no GPU run tested the aliasing.

## Evidence

AKLR paths are on `AMD-AGI/AKLR@0e02a81` unless a line names another commit; a committed kernel is cited as `<schema path>#<source file>:<line>`, and every schema cited here holds the same sources at 43e112e. Triton paths are `triton-lang/triton@c01b677` (tag v3.8.0). Each E line names the commit or CI run its numbers come from; those marked only #89 were taken for #89 with AKLR's code as of 43e112e, whose bench timed calls with Kineto (AKLR's rocprofiler-sdk collector since #33).

- E1 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, AKLR CI bench (graph replay, 1 GiB read flush), speedup against the schema's baseline] stored speedups at 43e112e: DeepSeek-V2/GLM rank inter 256: 1 token 2.139 (5751), 2 2.186 (d3bb), 4 1.867 (c3e3), 8 1.521 and 16 1.245 (e9fd), 32 1.086 (f8cb), 128 1.030 (a1c4), 256 1.160 (1204), 1024 1.402, 2048 1.329 and 3072 1.407 (c524), 4096 1.420 and 4192 1.378 (699f); rank inter 512: 1 token 2.093 and 2 1.829 (7e9c), 4 1.324 (4798), 8 1.184 (a2bb), 16 1.097 (3c6e), 256 1.043 (2422), 1024 1.152 (3432), 2048 1.391 (a841), 3072 1.235 (67d9), 4096 1.161 (3b66), 4192 1.166 (b058); MiniMax-M2 FP8, 256 experts: 1, 2, 4, 8, 16 tokens 4.115, 3.755, 2.452, 1.841, 1.732 (e75e); 32 1.158 and 64 1.173 (5cac); 128 1.106 (b43d); 256 1.133 (8a5a); 1024–16384 1.257, 1.248, 1.348, 1.372, 1.388 (f770); FP8, 64 of 256 experts (expert_offset 192): 1, 2, 4, 8, 16 tokens 3.137, 2.912, 1.672, 1.466, 1.325 (1c17); 32 1.167 and 64 1.102 (0982); 128 1.146 and 256 1.164 (94af); 1024–16384 1.216, 1.485, 1.681, 1.770, 1.785 (82e9); MXFP4, 256 experts: 1 token 3.444 (6557), 2 3.309 (fbd9), 4, 8, 16 2.549, 1.977, 1.595 (96e5), 32 1.179 (0101), 64 1.230 (d861), 128 1.210 (01ce), 256 1.178 (4da3), 1024 1.352 (bb66), 2048 1.272 and 4096 1.261 (147b), 6144 1.415 (e6fc), 8192 1.423 (51ae), 16384 1.389 (8411); MXFP4, 64 of 256 experts (expert_offset 192): 1 token 3.633 and 2 2.797 (b065), 4 2.554 (f82c), 8 2.113 and 16 1.555 (3df5), 32 1.231 and 64 1.275 (dfa8), 128 1.190 and 256 1.194 (1d27), 1024 1.252 (abc7), 2048 1.318 (cb88), 4096 1.590 (b94f), 6144 1.689 (62df), 8192 1.735 (f839), 16384 1.619 (8b22); Qwen2-MoE router+experts: 1 token 1.172 (7d7e), 2 1.145 (04e9), 4 1.119 (ddb8), 32 1.063 (99f4); router: 8 tokens 1.074 (4801), 16 1.219 (5f20), 64 1.295 (a878), 128 1.417 (cd47), 256 1.967 (c388), 2232 1.416 (4c2b), 4096 1.226 (ca82); fp8 experts: 8 tokens 1.208 (5c35), 64 1.171 (8698), 128 1.076 (d425); each value from the CI run of the commit that last stored it before 43e112e: 93eba6d for f8cb, e75e, 5f20, 5c35; 369a3c0 for a1c4, 4798; 90b9d50 for 1204, 7e9c, b43d, f82c, 1d27, b94f, cd47, d425; 3fb9033 for 3c6e, 2422, 5cac, 94af, 6557, f839, ddb8; a29581b for e6fc, 51ae, 62df; 62a5fe8 for cb88; f23abc6 for 8698; 43e112e for the rest; d908d01 re-stored 57 of the 96 Points, by −6.3% (e75e at 1 token, 4.115 → 3.855) to +7.0% (5f20 at 16 tokens, 1.219 → 1.304), median change 0.9%, the values 0e02a81 holds; over AKLR's CI history, stored ratios of unchanged sources at the same Point differ by 2.03 % at the median, 6.2 % at p90 and by more than 5 % in 16 % of 1716 pairs at 170 Points, at most 54.8 % (7d7e at 1 token) (aklr/plugins/sglang/models/{deepseek_v2,minimax_m2,qwen2_moe}/impls/triton/; AKLR@43e112e CI run 41991630592, AKLR@93eba6d CI run 41964469977, AKLR@369a3c0 CI run 41753978959, AKLR@90b9d50 CI run 41976547645, AKLR@3fb9033 CI run 41812750710, AKLR@a29581b CI run 41197527705, AKLR@62a5fe8 CI run 41778680267, AKLR@f23abc6 CI run 41418106915, AKLR@d908d01 CI run 42000899019)
- E2 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=4 (the rank-3 compile in a fifth run too), Gluon MFMA loops on register operands; the MiniMax MXFP4 schema's shapes, H 3072, 256 experts, top-8, inter 512] plain fp8 16 × 16 × 32 at 0.498, 0.500, 0.498, 0.501 of the rate of `mfma_scaled` 16 × 16 × 128 without scales; at the measured MFMA rates the fp32-MFMA router over the expert GEMMs, per token, 1.255, 1.262, 1.270, 1.284 and the three-bf16 split 0.2386, 0.2392, 0.2410, 0.2452 (1.33 and 0.25 at the nominal MACs per cycle); three bf16 MFMA chains over the fp32 MFMA time 0.1888, 0.1887, 0.1890, 0.1891, and on 64 × 64 logits at K = 3072 (one seed) the max error against fp64 3.6e-6 for the split, 1.4e-5 for fp32 MFMA, the same in every run; `mfma_scaled` on a rank-3 e4m3 layout of two batched 16 × 16 × 128 tiles aborts the compiling child while tracing (exit −6, LLVM ERROR "Dimensions must match") and plain fp8 `mfma` 16 × 16 × 32 on that layout lowers to `v_mfma_f32_16x16x32_fp8_fp8`, in all 5 runs (probe kernels; #89)
- E3 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3, `python -m aklr validate`, one GPU, A/B 4 alternated rounds per arm; AKLR's timer before #33 (Kineto) on a test kernel] c388 at 256 tokens: ties sent to the higher id fail validate (ids differ in 174 of 256 rows, 176 rows hold ties), fp32 split partials rounded to bf16 before the sum fail (66 rows), split-K 4 → 2 passes with no id changed, identical in the 3 runs; c524 at 2048 tokens with the weight-resident down on (`down_group` 1 → 2: one expert's weight panel held across two 128-row activation tiles): 234.6 → 234.2, 230.8 → 229.6, 235.9 → 235.8 µs (−0.17 %, −0.52 %, −0.04 %), 12 alternated pairs B/A 0.974–1.006, stored speedup 1.311–1.321 off and 1.309–1.326 on, the 1024- and 3072-token controls within 1 %; a two-role test kernel at 256 VGPRs launched without heavy work: a flushed 128 MiB fp32 read and write stream 44.03, 44.00, 43.93 µs in it against 45.56, 45.19, 45.35 µs alone (0.967–0.974×) (`c3889614985b700a`, `c5244d02949f27b4`; test kernel; #89)
- E4 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2 (the toy MoE N=4), test kernels and copies of committed code on one GPU] DeepSeek's reference with both GEMMs' activations MXFP4-quantized moves `out` by rel L2 0.2103, max abs 31.19; a d861-form copy at 32, 128 and 256 tokens puts each route in a job of up to 16 rows of one expert, and over 20 runs at 256 tokens the tickets differ in 19 while the slot-order combine keeps one bit pattern (a float `atomic_add` in arrival order gives 20); 1000 programs of 8 ids in a 64-lane layout leave every counter at 1000; 512 crafted near-midpoint logits: f64 → f32 → bf16 wrong on 260, with the midpoint fix on 0, 7d7e's helper 0 of 1024; an abc7 copy storing only the rows with a local route: 239616 non-finite mismatches in `out`; a d861 copy without its fill of `Sorted`: candidate crash, runner exit −6; under `launch_cooperative_grid` a counter barrier over 256 workgroups of 1024 threads finished in 0.2 ms, every program seeing every store; 4096 workgroups were refused (`hipErrorCooperativeLaunchTooLarge`); on a 2.5 GiB tensor `buffer_load` read the data at byte 2^31 − 64 and 0 at 2^31 + 64 with no fault, `gl.load` both; a toy decode MoE (32 experts, top-4, H 256, I 128, every gate-up program owning full K, no requantization) at 2 and 16 tokens against no fusion, 4 runs: each fusion alone saves on AKLR's timer before #33 (Kineto) top-k recomputed in the gate-up and down programs 3.30–4.19 µs, SwiGLU in the gate-up epilogue 4.04–4.64 µs, down and combine in one program 4.19–4.43 µs (not timed replayed); the three together give call-time ratios of 0.667–0.751 replayed and 0.426–0.462 on AKLR's timer, rel L2 against no fusion 4.5e-8 and 5.1e-8 (`d8610d895915c167`, `abc7bbf7a7135f28`, `7d7e54597a7e11bd`, `84116a96e535f174`; test kernels; #89)
- E5 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2 sessions, `python -m aklr validate`, one GPU, the committed file against itself and an identical copy] first session: c388 at 256 tokens 1.93, 1.92, 1.94; d3bb at 2 tokens 2.18, 2.19, 2.17; second, (max − min)/mean of the first 3 runs: c388 0.91 %, d3bb 0.94 %; over all 8 runs (A and the copy) 0.95 % and 1.27 % (`c3889614985b700a`, `d3bbbce5c0bbc61e`; #89)
- E6 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2 sessions, 4 alternated rounds per arm, `python -m aklr validate`, one GPU] Qwen2 router c388 at 256 tokens: `_ShapeConfig` split 4 → 2 (grid 256 → 128 CTAs) 1.923 → 1.449 and 1.927 → 1.441, candidate time 9.93 → 13.21 and 9.906 → 13.29 µs; split 4 → 8 (512 CTAs) 1.931 → 1.874 and 1.915 → 1.864 (9.91 → 10.22 and 9.971 → 10.25 µs, all 4 B runs slower in the second); look-ahead 2 → 1 1.925 → 1.696 and 1.917 → 1.699 (9.93 → 11.26 and 10.01 → 11.28 µs); `amdgpu-sched-strategy` removed from both launches 1.927 → 1.922 and 1.923 → 1.917; `amdgpu-ieee false` removed from the routing launch 1.935 → 1.924 and 1.921 → 1.928; Qwen2 router 4801 at 8 tokens, the hidden loaded into registers (the file's register branch) instead of async to LDS 1.179 → 1.197 and 1.169 → 1.172; accuracy passed in every run (`c3889614985b700a`, `4801e216dcff290f`; #89)
- E7 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2 sessions (the `.cg` A/Bs three runs in the second), 4 alternated rounds per arm, `python -m aklr validate`, one GPU] first session: d3bb at 2 tokens, `.cg` → none on the weight loads 2.164 → 2.170 (18.98 → 19.05 µs); 1d27 at 256 tokens `.cg` → none 1.180 → 1.156, all 4 pairs (102.93 → 104.65 µs), its unchanged 128-token Point 1.183 → 1.185; 2422 at 256 tokens `.cg` added on the `TM == 128` gate-up tiles 1.043 → 1.033, all 4 pairs (225.81 → 227.67 µs); a2bb at 8 tokens `enable_fp_fusion=False` → `True` on all 3 launches 1.150 → 1.150 (56.63 → 56.66 µs), accuracy passed; 6557 at 1 token `gate_fold` 4 → 1 alone: every changed run crashed (exit −11), committed runs 3.50–3.59; 51ae at 8192 tokens gate-up `waves_per_eu` 3 → 0 1.442 → 1.443 (515.56 → 514.69 µs); second session, three runs: d3bb without `.cg` 2.171 → 2.171 (19.30 → 19.32 µs, pairs both ways), 2.205 → 2.180 and 2.217 → 2.195 (×1.009 and ×1.010, every pair slower); 1d27 without `.cg` 1.189 → 1.150, 1.181 → 1.145, 1.180 → 1.144 (×1.029–1.035); 2422 with `.cg` 1.040 → 1.034, 1.039 → 1.033, 1.032 → 1.027 (×1.005–1.007), each slower in all 4 pairs; a toy kernel whose 64 KiB blocks are each read by two adjacent programs 39.44–39.95 µs unhinted against 41.35–41.74 µs with `.cg`, one read of each block 27.17–27.93 against 24.19–25.39 µs; a2bb 1.147 → 1.146 (B/A 0.9994), accuracy passed in all 8 runs, its 3 launches compiled for gfx950 with and without the switch giving the same code (633, 1554 and 440 instructions); 51ae 1.441 → 1.440 (510.9 → 511.8 µs); the `gate_fold` crash in 4 of 4 changed runs, the committed file 3.535–3.570 (`d3bbbce5c0bbc61e`, `1d277d76892ee692`, `2422a87b3b814299`, `a2bb3d83284b20ff`, `655723f593921b08`, `51ae499426db920d`; #89)
- E8 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, every Triton schema validated once with `python -m aklr validate` on its own Triton cache, each compiled kernel's `.amdgcn` parsed; d3bb's router also compiled at its host code's launch constexprs] d3bb `_router_pair_packet` 48 `v_dot2c_f32_bf16`, no MFMA; Qwen2 router+experts f64 VALU ops (`v_fma`, `v_mul`, `v_add_f64`) 23 in 7d7e (1 token), 38 in 04e9 (2), 69 in ddb8 (4), no `v_mfma_f64`; 99f4 (32) `_router_and_shared_matrix` 64 `v_mfma_f64_16x16x4_f64`; MiniMax FP8 e75e `_prepare_keys` (5 variants, 1–16 tokens) no MFMA, 5cac `_prepare` (32–64) `v_mfma_f32_16x16x4_f32`, MXFP4 8411 `_router_projection_7` (16384) only `v_mfma_f32_32x32x2_f32`; the MFMA-bearing router kernels of cb88, b94f, 62df, f839, 8b22, f770, 82e9 `v_mfma_f32_16x16x32_bf16` and no fp32 MFMA, of bb66, 147b `v_mfma_f32_16x16x4_f32`, of e6fc, 51ae, 8411 `v_mfma_f32_32x32x2_f32`; 147b `_select_5` (both variants 4 warps) 41 `s_barrier`, 25 `ds_write*`, 25 `ds_read*`; 8411 `_select_partitioned_2` (1 warp) and b94f `_select_wave_rows` (4 warps) no `s_barrier`; the 14 DeepSeek-V2 and MiniMax-M2 MXFP4 schemas whose Points are all ≤ 16 tokens contain `v_mfma_scale_f32_16x16x128_f8f6f4` and no `v_mfma_scale_f32_32x32x64_f8f6f4`; `v_mfma_f32_16x16x32_fp8_fp8` in e75e `_down_combine` (12 or 24 per variant) and `_project_w13_internal_packed2` (24), 5cac `_w13_local_split` (24), one 0982 `_down_projection_2` variant (48); the 65 MoE files' kernels hold no float atomic, their atomic instructions `global_atomic_add` 57, `ds_add_rtn_u32` 27, `buffer_atomic_add` 8, `ds_wrxchg_rtn_b32` 4, `buffer_atomic_or` 1 (the 65 MoE schemas, sources as at AKLR@43e112e; #89)
- E9 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, compile only: `triton.compile` for gfx950 without a GPU, the Point's constexprs and launch options; rechecked in later compiles] 6557 `_gate_up` at 1 token: FOLD 4 compiles (160 VGPRs, 24 scaled MFMAs); FOLD 1 with 4 warps and K tile 128 kills the compiling process with SIGSEGV (with 16 or 4 rows, transposed or not); FOLD 1 compiles with 1 warp (K tile 128 or 256), with 4 warps and K tile 256, or without `llvm_fn_attrs` (29 VGPRs, 24 scaled MFMAs), and every crashing variant carried the `iterative-ilp` attribute 6557 passes at m ≤ 2; 51ae `_gate_up_scale_preload_fp4` at 8192 tokens 146 VGPRs with `waves_per_eu` 3 and 150 without, no spills (two compiles); `gl.atomic_add` gives one `global_atomic_add`, no buffer atomic; `gl.amd.cdna4.buffer_atomic_add` on uint32 or uint64 and `buffer_atomic_or`, `and`, `xor` on int32 or int64 compile, all 10 dtype mismatches raise at trace time, `buffer_atomic_xchg` on int32 or int64 traces and then fails in LLVM translation (no `llvm.amdgcn.raw.ptr.buffer.atomic.exch`); `gl.associative_scan` on the `SliceLayout` of 8411's `_scatter_bases_2` fails to lower ("TODO: unsupported scan layout") and compiles after a `convert_layout` to a 1-D blocked layout, equal to `torch.cumsum`; 7e9c's `_expert` at 2 tokens compiles with `assert_trivial` on both operand conversions, row-major words raise; shuffled MXFP4 words become the [16, 16, 128] A operand with no instruction and no LDS, row-major ones take 8 `ds_write_b128`, 8 `ds_read_b128` and 8192 B of LDS; a `.wt` `gl.store` is followed by a wait after each of 4 stores, a `.wt` `buffer_store` by none (`655723f593921b08`, `51ae499426db920d`, `84116a96e535f174`, `7e9c13e0f2d58a3b`; #89)
- E10 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3 (the toy MoE and the one-grid router N=4), one GPU: 1 GiB stream probes timed as 2^30 bytes over the median of 9 rounds of 10 launches; AKLR's timer before #33 (Kineto) and bare graph replay on test kernels and on d3bb; `python -m aklr validate` A/B, 4 alternated rounds per arm] a 1 GiB bf16 tensor read once per launch by 2048 programs of 4 warps, 16 bytes per lane: 6013, 6009, 6009 GB/s plain with `buffer_load` (5994, 6000, 5993 with `gl.load`), 6823, 6805, 6802 GB/s with `.cg` (6803, 6791, 6785), so 6.0 and 6.8 TB/s; d3bb at 2 tokens: 4 launches (`_router_pair_packet`, `_select_and_up`, `_activate_quantize`, `_token_local_down`) in every trace, call 19.00, 19.17, 19.04 µs, 4.75, 4.79, 4.76 µs per kernel against a per-launch floor (slope over chains of 1–8 trivial kernels) of 4.041, 4.029, 4.066 µs, so the 4 floors are 84–85 % of the call; 4 chained trivial kernels 16.41, 16.44, 16.42 µs, one kernel doing their 4 updates 4.30, 4.30, 4.32 µs; a 1 GiB `.cg` stream over 256–8192 programs of 1, 2, 4 or 8 warps with 1–8 independent 16-byte loads per lane: configurations with equal bytes in flight within 1.104× of each other (the widest at 4 MB: 256 programs × 4 warps × 4 loads 5697–5756 GB/s, 512 × 4 × 2 6287–6333 GB/s), 256 × 4 × 8 over 2048 × 4 × 1 1.067–1.070; at 3 MB 384 × 4 × 2 4920–4962 GB/s, 768 × 2 × 2 4898–4947, 96 × 4 × 8 3945–3995 (0.80×), 192 × 1 × 16 2454–2467 (0.495–0.499×); c388 at 256 tokens, B/A speedup: split 4 → 2 (256 → 128 CTAs) 0.749, 0.748, 0.753 (9.93–10.03 → 13.26–13.32 µs), 4 → 8 (512 CTAs) 0.961, 0.967, 0.978, look-ahead 2 → 1 0.889, 0.881, 0.890; a toy decode MoE (32 experts, top-4, H 256, I 128, every gate-up program owning full K, no requantization), on AKLR's timer: top-k recomputed in the gate-up and down programs instead of a selection launch saves 3.07, 2.89, 3.22, 3.40 µs at 2 tokens and 3.44, 3.22, 3.41, 3.87 µs at 16 (replayed at 2 tokens 1.24–1.80 µs, call-time ratio 0.843–0.889), down and combine in one program per (token, 64-column tile) instead of a down and a combine launch 2.94, 2.97, 3.32, 3.23 µs at 2 tokens and 3.16, 3.18, 3.38, 3.73 µs at 16 (not replayed); a toy router GEMV (64 programs, 2 rows, K 2048) beside a 4-program elementwise phase in one grid: outputs equal, one grid over two launches 0.499, 0.495, 0.506, 0.498 on AKLR's timer, 0.835, 0.775, 0.755 and 1.154 (9.360 against 8.110 µs) replayed; a two-role test kernel compiles at 256 VGPRs and 32 B of LDS, its streaming role alone at 16 VGPRs and no LDS, its heavy role at 256 and 32 B (`d3bbbce5c0bbc61e`, `c3889614985b700a`; probe and toy kernels; #89)
- E11 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1 (the committed call times N=4, the top-k probes N=3), one GPU: launch traces with torch.profiler, AKLR's timer before #33 (Kineto) and bare graph replay on test kernels, `python -m aklr validate` on committed files and on copies with one change (A/B: 4 alternated rounds per arm)] launch traces of the 96 MoE Points: launches per call as in §1; every Point compiles on its first call, none on its second, and 136 of 471 launches carry no token-count constexpr; d3bb at 2 tokens grids 304, 2306, 72, 768; 5751 at 1 token `_router_linear` 216 = 192 + 24 programs, `_select_and_up` 1153; all 32 DeepSeek and MiniMax Points from 1024 tokens sort before the gate-up; none of 185 kernel names is a quantizer alone; routers of 512 programs in 01ce, 4da3, 8a5a, 94af and of one full-K program per (token block, expert) in e75e, fbd9, 3df5 above 1 token; 8698 at 64 grids of 3072 and 4096, 94af at 256 `_w13_persistent` on 48 × 96; the same grids on two routings and graph replay equal to eager at 82 Points; split-K finish launches in d3bb, 5751, 7e9c at 1 token (`_finish_gate_up`) and 5c35; the baseline enqueues 8–11 device operations at 1–2 tokens, AKLR 3–4; docstring lists differ from the trace at 7 Points in 5 of 11 files; one trivial kernel reads 4.21 µs on AKLR's timer, each added one 4.06 µs, and a bare graph replay pays 1.815–1.844 µs per extra kernel; committed call times, mean of 2 validates, 4 runs: d3bb at 2 tokens 19.06–19.61 µs (4.76–4.90 µs per launch), 2422 at 256 225.5–236.8 µs, 1d27 at 256 103.6–105.4 µs, 51ae at 8192 505.7–518.6 µs, an earlier batch of runs 225.8, 102.9 and 515.6 µs for 2422, 1d27 and 51ae; d3bb at 2 tokens, gate-up K splits 4 → 2 (grid 2306 → 1154): 2.183 → 2.175, 19.20 → 19.28 µs, runs 2.167–2.195 against 2.157–2.191, paired 95 % CI of the ratio 0.991–1.001; top-8 of 256 at 16384 tokens: one wave per token 12.96–13.18 µs with no `s_barrier`, four waves 37.68–38.32 µs with 31, the same ids, no scratch in either; the keyed min (retired scores at −inf) 0 wrong of 4000 rows; 1c17 exact at 8 and 16 tokens; 3df5 at 2 tokens keeps 8 through its id tie rule, at 8 tokens selects 9 on a tied 8th and 9th (16 tokens not run); d3bb's `_expert_at_rank` over 16384 programs 9.7–10.7 µs at slots 0–4, 11.3–11.4, 12.8–13.0 and 14.3–14.4 µs at slots 5–7, R² of a linear fit 0.724–0.774; with b94f's combine lanes disagree on 10000 of 10000 NaN rows and 0 of 10000 finite ones; e75e and fbd9 give ±0 and every NaN payload one key each, 7d7e gives 7 bf16 NaNs 7 keys; an MFMA-only fp16 loop with 4 useful rows: `instr_shape=[4, 64, 64]` 2432 µs, [16, 16, 32] 2810 µs; every finite E4M3FN code converts exactly to bf16; integer-scratch patterns over 100 calls: 8698 at 64 tokens 9 and d425 at 128 4 in one run, 15 and 8 in another; 31 atomic files, 20 runs each, one output each (the 96 MoE Points of 43e112e; `d3bbbce5c0bbc61e`, `5751284b6965eabb`, `2422a87b3b814299`, `1d277d76892ee692`, `51ae499426db920d`; probe kernels; #89)
- E12 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, 9 rotated rounds per dependency and Point at m = 1, 2, 16, CUDA-graph replay and AKLR's timer before #33 (Kineto); 1000 correctness calls per Point] toy Gluon kernels, 256 experts, H 6144: logits → top-k: two launches took 0.26–0.51× the replay time of a last-arriving program doing top-k and 0.15–0.48× a cooperative counter-barrier launch; counts → rows, a last arriver scanning 256 counts against two launches: at m = 1, 2 two launches 1.16× its replay time and 1.48–1.50× on AKLR's timer, at m = 16 0.89× and 1.28×; all 1000 calls bitwise equal per Point; under `launch_cooperative_grid` a bounded counter barrier completed over 256, 512 and 1024 one-wave workgroups in all 1000 eager and replayed calls, at 2.1–6.8× the two-launch replay time; co-residency against a plain launch and the captured node's cooperative attribute not checked (toy kernels; #89)
- E13 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, 16 rotated rounds per Point, 10 replays per round, CUDA-graph replay and AKLR's timer before #33 (Kineto), paired CIs with A/A controls] synthetic fp8 VALU MoE, 256 experts, top-8, H6144 with inter 256 and H3072 with inter 512, m = 1, 2, 8, 16: routing recomputed in every gate-up, down and combine consumer slower on both rulers; recomputed only in the gate-up, which publishes ids and weights: H6144 m = 1 and 8 saved 2.87 and 2.45 µs replayed, 2.06 and 2.44 µs on AKLR's timer (A/A CIs including zero); SwiGLU and 1×128 requantization in a full-K gate-up epilogue saved 1.76–2.76 µs replayed and 3.05–3.73 µs on AKLR's timer against a full-K gate-up with a separate activation launch, but lost to the unfused K-split design; down and combine fused in slot order: replay saving −8.18 to −0.25 µs on average (a loss); all fusions together 1.0841–2.2992× the K-split design's replay time; H6144/m16: one fp8 code flipped, relative L2 1.122e-5 against the reference; some A/A controls of the K-split design showed replay bias (toy kernels; #89)
- E14 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1 (c388's half/double N=8 and one-tile look-ahead N=6, with the runs of E6 and E10), `python -m aklr validate`, 4 alternated rounds per arm, one GPU] Qwen2 routers against half and double splits: c388 at 256 tokens half 1.33–1.34×, double 1.02–1.04× the committed call time; cd47 at 128 1.22× and 1.03×; a878 at 64 1.04× and 0.99×; c388 with 128 CTAs four K tiles ahead 1.26×, 512 CTAs one ahead 1.03×, 256 CTAs one ahead 1.12–1.14×, four ahead 0.97×; MoE gate-up splits at 1–2 tokens: d3bb at 2 tokens 4 → 1 (2306 → 578 one-wave programs) 1.079×, 4 → 8 (4610) 1.050×; 5751 at 1 token 1153 → 577 programs 1.020×, → 2305 1.028×; 7e9c 1728 → 864 1.035×; e75e column tile 8 → 16 (385 → 193 four-wave programs) 1.031×; every arm accurate (`c3889614985b700a`, `cd47a5e858af6ab4`, `d3bbbce5c0bbc61e`, `5751284b6965eabb`, `7e9c13e0f2d58a3b`, `e75e553b34178760`; #89)
- E15 [2026-11, CPU (no GPU), Triton 3.8.0, N=1, source and AST census of the host code; fp8 codes converted in torch] 89 MoE Points of 58 schemas: r = m·k/E computed from each schema, gate-up launches and grids traced from the host code, job forms not classified; 5c35's MFMA operands cast to bf16 in `_project_gate_up` and `_project_down_and_sum`; all 256 e4m3 codes: the 254 finite values, both NaNs and signed zero convert to bf16 exactly; 699f's aliased buffers written, read and overwritten in launch order (kernel.py:1307, :1324, :1338) (the MoE schemas as at AKLR@43e112e; #89)
- E16 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, `python -m aklr validate`, 4 alternated rounds per arm, 12 Points of seven kernels, A/A controls at four Points; reuse classified from the schedule and the measured routing, not from cache counters] `.cg` on the weight loads against none: read once at 1c17 1, 4, 8, 16 tokens, 5751 1, e75e 1, 2, 1d27 128, 256 and c524 2048, re-read at 2422 256 and 8411 16384; `.cg` faster in every pair on both rulers at six read-once Points (1c17 at 4, 8, 16, 1d27 at 128, 256, c524 at 2048; 1.6–9.6%), e75e at 2 tokens in aggregate only, no consistent effect at the three 1-token Points; without `.cg` faster in every pair at both re-read Points (2422 1.012 → 1.009, 0.2–0.5%; 8411 17%); A/A offsets within 0.37%; every arm accurate (`1c1747a25af67b8e`, `5751284b6965eabb`, `e75e553b34178760`, `1d277d76892ee692`, `c5244d02949f27b4`, `2422a87b3b814299`, `84116a96e535f174`; #89)
- E17 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1 per Point over eight sessions, graph-replay events after a 1 GiB flush and a 90 s ramp, 32 balanced AB/BA pairs per Point, A/A controls; register-MFMA probes, grid 1536, 3072 iterations] synthetic router logits at K 3072, (M, E) = (1024, 64), (4096, 64), (16384, 64), (1024, 256), (4096, 256), (8192, 256), each arm's tiling from a bounded search: the three-bf16-term kernel took 0.457–0.512× the time of the two fp32-MFMA kernels (random input, seed 1201, W split in each call; sigmoid, bias, top-8 and the expert GEMMs excluded); 18 A/A controls within ±log(1.02); against the fp32 top-8 contract over 8 Points × 8 seeds × 3 input modes × 3 arms, 217 of 576 comparisons failed, all 192 near-tie inputs among them; at M16384/E64, random seed 705, both fp32 arms passed and the split failed with lower logit error (max abs 4.165e-6 against 1.149e-5, against fp64); M8192/E64 and M16384/E256 had no correct split candidate on seed 1201 and were not timed; probes: estimated matrix-core time of the fp32 router over the MXFP4 expert GEMMs 1.244–1.263, of the three-bf16 split 0.244–0.246 (K 3072, 256 experts, top-8, inter 512) (toy kernels; #89)
- E18 [2026-11, gfx950 (MI355X), Triton not recorded (MI355X image), N=1, `python -m aklr validate` (the committed files twice), one input set per Point, copies of committed kernels with one change; the selection copies also run directly] c388 at 256 tokens with token 0's first two ids swapped, everything else bitwise equal: `numeric_mismatch` on `topk_ids` (violation fraction 2/2560), the committed file passed twice; a2bb at 8 tokens ranking the unrounded logits: passed twice, no token's experts changed, error change unresolved; d9cb at 32 rows with the attention kept in fp32 before the gate: passed twice, its scales' relative L2 against the float64 reference without that rounding 0.0032334 → 0.0032690; 5751 at 1 token, the gate-up copy without the bf16 rounding of the summed logits: 6 of 16 tokens took another expert, validate failed (`capture_unsafe`, `numeric_mismatch`); with the higher id winning ties: 4 tokens, validate passed (`c3889614985b700a`, `a2bb3d83284b20ff`, `d9cbde237fd34de9`, `5751284b6965eabb`; #89)
- E19 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3, `python -m aklr validate` twice per run; routing simulated from the schema's router distribution and bias with random normal hidden states] MiniMax MXFP4, 256 experts, at 64 tokens: experts touched per call 77.19% on average, 74.61% at the least (uniform-routing estimate 86.9%), 495.3 MB of expert weights; call 99.09–101.7 µs against the baseline's 127.0–127.8 µs (speedup 1.249–1.285), so 4.87–5.00 TB/s against 3.88–3.90 TB/s for those bytes (`d8610d895915c167`; #89)
- E20 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, the router prefix (logits) of 13 MiniMax router branches at legal Points run on the GPU, a random and a cancellation input each (seed 0), plus two inputs with a nonzero fp64 residual; error against fp64 on the same input, compared with an fp32 GPU computation] the routers' logit error no larger than fp32's in L2 on 25 of the 26 inputs of the 13 branches (the exception e6fc's cancellation input, 1.0000934×) and in max abs on 23 of 26 (the exceptions the cancellation inputs of 147b, e6fc and 51ae); on the two residual inputs the conditional fp64 correction of 62df and f839 ran, logits of 2^-23 equal to fp64, the routers' and the fp32 computation's errors both 0 (`62df6de0b64b800d`, `f83985dc8066c00c`, `e6fc7b071ab748e5` among the 13 branches; #89)
- E21 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, copies of 5751's selection and expert code run directly at 1 token on natural and saved inputs, frozen reference ids and slot weights] four saved inputs with near or exact ties: replacing only slot 7's expert (by an unselected one) in both gate-up and down moved `out` 6.21, 112.2, 6.13 and 9.47 times as far in L2 as removing only the bf16 rounding of the activation product, total errors against the reference 1.22, 3.03, 1.23 and 1.28×; changing only the gate-up copy's bf16 rounding or tie rule: producer and consumer disagreed on a slot's expert on 12 of 13 natural inputs, both sides changed alike: agreement on 36 of 36; the 12 split partials summed in reverse order on four saved inputs: ids, weights and outputs unchanged (`5751284b6965eabb`; #89)
- E22 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, 100 synchronized calls on one fixed input; bodies forced at 15, 16, 17, 31, 32 and 33 live rows] 2422 at 256 tokens: the observed tile ran its 16-row body in all 100 calls with one route-partial hash; forcing the body sizes changed no bit of its up-projection codes, scales or down partials (`2422a87b3b814299`; #89)
- S1 aklr/plugins/sglang/models/deepseek_v2/api.py:32 the route on `DeepseekV2MoE.forward_normal`: router, top-k and experts, the shared one last, as one op call; MiniMax's route on `MiniMaxM2MoE.forward_normal` at aklr/plugins/sglang/models/minimax_m2/api.py:45, the experts' format picking the fp8 or MXFP4 op; aklr/plugins/sglang/models/moe_blocks.py:16 the runtime configurations that keep a MoE call on SGLang
- S2 aklr/plugins/sglang/models/qwen2_moe/api.py:105 the route on `_forward_router_experts`: router+experts op (:131), else router op (:135), then experts op (:144), then SGLang's experts; since #77 also a route on `Qwen2MoeSparseMoeBlock.forward` for `qwen2_moe_mxfp4_moe` (:69); aklr/plugins/sglang/models/qwen2_moe/baseline.py:4 the baseline: SGLang's small-M router of 1 to 8 tokens, which appends the gated shared expert itself, else the gate, `fused_topk` and the append; then SGLang's gfx950 small-M kernel where it serves, else `fused_moe` (to :12); `_smallm_router` (:50-60), `_experts` (:110-150); MiniMax's gate on fp32 activations at aklr/plugins/sglang/models/minimax_m2/baseline.py:44
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
- S24 triton-lang/triton@c01b677:python/triton/experimental/gluon/language/_semantic.py:245 `convert_layout(..., assert_trivial=True)` raises unless the conversion is trivial; triton-lang/triton@c01b677:python/triton/experimental/gluon/language/_core.py:703 `gl.barrier()` synchronizes the threads of one CTA; a cluster only with `cluster=True` and `num_ctas > 1` (:712-714), which the HIP backend rejects on gfx950 (third_party/amd/backend/compiler.py:162-163)
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
- S40 aklr/plugins/sglang/models/deepseek_v2/reference.py:40 `_mlp`: "down(silu(gate) * up), gate and up the halves of x gate_up^T" (:41, the split at :44); the same split and `silu(gate) * up` at aklr/plugins/sglang/models/minimax_m2/reference.py:89-93 and aklr/plugins/sglang/models/qwen2_moe/reference.py:103-104; `w1` rows "gate and up concatenated", `2 * inter_dim` (deepseek_v2/contract.py:111); aklr/ops/fused_moe/reference.py:110 `projected.chunk(2, dim=-1)`, "w1 stores [gate | up]"; `gate.clamp(max=limit)`, `up.clamp(-limit, limit)` whenever `swiglu_limit` is passed (:111-112); silu, gelu, swiglu `gate * torch.sigmoid(1.702 * gate) * (up + 1)` and, otherwise, situv2 with `beta` and `linear_beta` (:113-121); aklr/ops/fused_moe/contract.py:91 the activations silu, gelu, situv2, swiglu (to :99); `gate_mode` separated or interleave, "Where w1 keeps the up rows" (:108-114); interleave only with silu on the `gate_up_interleaved` layout (:154); `swiglu_limit` passed exactly for interleave and swiglu (:155); aklr/core/contract/semantics/_layouts.py:158 `GateUpInterleavedWeight`: 16-row tiles, "the gate and up halves alternate tile by tile" (:159-160)
