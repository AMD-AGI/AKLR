---
title: "Whole MoE layers in Gluon on gfx950: decomposition by token count, routing, sorting, expert GEMMs, combine"
kind: topic
ops: [moe, fused_moe, grouped_gemm]
dtypes: [mxfp4, fp8, bf16]
regimes: ["tokens 1-16", "tokens 32-256", "tokens 1024-16384", sorted-tokens]
gens: [gfx950]
triton: 3.8.0
---

# Whole MoE layers in Gluon (gfx950)

How Gluon kernels compute a rank's router-and-expert portion of a SGLang MoE layer call in a few launches, and which design to copy at which
token count: the Triton side of [../moe/INDEX.md](../moe/INDEX.md), whose measured MoE-kernel numbers are from FlyDSL unless it explicitly names another language or a vendor-library comparison; the numbers here
are these Gluon kernels' on MI355X. Two timers recur. *The ruler's timer* is the per-call time of the ruler ([tuning.md](tuning.md) §1), the times of a call's kernels summed over flushed CUDA-graph replays, read here with Kineto, which holds every short kernel at that timer's floor (§2); *replay time* is the time of a bare graph replay. The page covers 65 kernels of the library (154 Triton and Gluon kernels for gfx950: [kernels/kernel_library.md](kernels/kernel_library.md) §1), called implementations here because each is a host function that launches a few Gluon kernels, and quotes each one's speedup, the ruler's stored value for one size against SGLang's path on aiter *(field, E1)*. Per implementation:
[kernels/kernel_library.md](kernels/kernel_library.md) §7–§10. m is a call's token count, a route one (token, expert) pair. An implementation is named by its layer and its token count: DS256 and DS512 are DeepSeek on a rank of that intermediate dimension; MM256-FP8, MM64-FP8, MM256-FP4 and MM64-FP4 MiniMax on a rank that holds that many of its experts, with FP8 or MXFP4 weights; Q2-RE, Q2-R and Q2-E Qwen2's router+experts, router and experts. DS256/m2 is the DS256 implementation at m = 2; a range, as in DS256/m8–m16, names one implementation built for several token counts.

## Route here when

- You write or tune a fused replacement for an SGLang MoE layer: DeepSeek-V2 with MXFP4 experts, MiniMax-M2 with FP8
  or MXFP4 experts, or Qwen2-MoE's router and FP8 experts.
- You decide a MoE call's launches per token count and where top-k, sort, activation, combine and shared expert go.
- You write expert GEMMs as a grouped GEMM: per-route or per-expert jobs at few tokens (§4), sorted, padded expert segments at many (§6, §7, §9), low-precision weights and scales (§7, §8).
- You port a `../moe/` mechanism to Gluon, or need the Gluon form of atomic tickets, a device-side job list or a
  recomputed top-k (preshuffled MXFP4 weights and shuffled E8M0 scales: [gluon/mfma.md](gluon/mfma.md) §7,
  [gluon/memory.md](gluon/memory.md) §13).
- A constant change in one of these kernels fails to compile or crashes, or a knob you expected to matter does nothing.

## 1. The layers and what they replace

| Layer | Replaces (SGLang 0.5.22) | Op, rank | Token counts | Implementations | Launches per call (≤ 16, 32–256, ≥ 1024 tokens) *(field, E11)* |
|---|---|---|---|---|---|
| DeepSeek-V2 | `DeepseekV2MoE.forward_normal` | router, top-k and MXFP4 experts, the shared one last; rank intermediate dimension 256 (DS256) | 1, 2, 4, 8, 16, 32, 128, 256, 1024, 2048, 3072, 4096, 4192 | 9 | 3–4, 5, 6 |
| | | the same op, rank intermediate dimension 512 (DS512) | 1, 2, 4, 8, 16, 256, 1024, 2048, 3072, 4096, 4192 | 10 | 3–4 (5 at 16), 6, 6 |
| MiniMax-M2 | `MiniMaxM2MoE.forward_normal` | router, top-k and FP8 experts; all 256 experts on the rank, intermediate dimension 384 (MM256-FP8) | 1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 2048, 4096, 8192, 16384 | 5 | 3, 5–6, 6–7 |
| | | the same, 64 of the experts on the rank, intermediate dimension 1536 (MM64-FP8) | same 14 | 4 | 4 (6 at 8–16), 5–6, 8 |
| | | router, top-k and MXFP4 experts; all 256 experts on the rank, intermediate dimension 512 (MM256-FP4) | 1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 2048, 4096, 6144, 8192, 16384 | 12 | 3, 5, 6 |
| | | the same, 64 of the experts on the rank, intermediate dimension 1536 (MM64-FP4) | same 15 | 11 | 3, 5, 7–8 |
| Qwen2-MoE | `Qwen2MoeSparseMoeBlock._forward_router_experts` | router and FP8 experts in one op, intermediate dimension 256 (Q2-RE) | 1, 2, 4, 32 | 4 | 4, 7 |
| | | the router alone (Q2-R) | 8, 16, 64, 128, 256, 2232, 4096 | 7 | 2 |
| | | the FP8 experts alone (Q2-E) | 8, 64, 128 | 3 | 3, 5 |

- **Terms.** A route's *slot* is its place 0 … k−1 in its token's top-k, the reference's order (the kernels' `rank`;
  DeepSeek's shared expert is slot 8, Qwen2's slot 10); an expert's global *id* indexes the router; its local `w1` index (W13 below)
  is `id - expert_offset` on an expert-parallel rank, otherwise `id`; a route's place in the sorted buffer is its *row*; a *rank* is an expert-parallel rank.
- **Shapes and weights.** DeepSeek (model dim 6144): 256 routed experts, top-8 by sigmoid(bf16 logit) + correction bias,
  each route weighed by its unbiased sigmoid, renormalized over the 8, times `routed_scaling_factor` 2.5 (the bias only
  selects); the shared expert, id 256, is a ninth route at weight 1; MXFP4 weights, fp32 activations in the reference
  (quantizing them to MXFP4 moves `out` by rel L2 0.21 *(field, E4)*). MiniMax (3072): 256 experts, fp32 router weight,
  top-8 by sigmoid + bias, weighed by the unbiased sigmoids renormalized to sum 1 (no factor, no shared expert). Qwen2
  (4096): 512 experts, top-10 by bf16 logit, the lower id first on a tie, weighed by the softmax of the 10 chosen
  logits; the shared expert, id 512, by sigmoid(hidden · `shared_gate_weight`); fp8 weights, 128 × 128 block scales.
- **The 64-expert ranks** (MM64-FP8, MM64-FP4) are an expert-parallel rank holding 64 of MiniMax's 256 experts; `out` is the rank's
  share: top-k over all 256, weights normalized over
  all 8 routes, the routes outside `expert_offset` … `expert_offset` + 63 adding nothing (a token keeps about 2;
  estimated, uniform routing). MM64-FP4/m1–m2 compacts a token's local routes with a scan, MM64-FP8/m32–m64 flags them beside each weight for
  a masked load in its combine, MM64-FP4/m8–m16 runs one program per local expert. `expert_offset` is 192 in every
  timed workload, but every accuracy input set except one runs 0: a kernel must be right for every offset its caller
  can pass, so take it from the call's arguments ([authoring.md](authoring.md) §5).
- **Qwen2 splits the layer over three ops**: the router and experts in one, the router alone, and the experts
  alone; at 8 tokens a call is the router's 2 launches plus the experts' 3.
- **What they beat.** Each baseline reconstructs SGLang's aiter path; MiniMax's explicitly does not replay SGLang's process-wide small-batch sort patch. DeepSeek and MiniMax: a
  router GEMM (MiniMax's on fp32 activations), a top-k kernel, the shared-expert append (DeepSeek) and aiter's
  `fused_moe`, which sorts and runs its two stages. Qwen2: when `router_weight.shape == (512, 4096)` and the imported SGLang `smallm_router_supported(hidden_states)` accepts the call, its gfx950 small-M router appends
  the gated shared expert itself; otherwise the gate, `fused_topk` and the append; then SGLang's gfx950 small-M MoE kernel
  where it accepts the call, else `fused_moe`. Not covered, and left on SGLang's path: all-to-all
  backends, EPLB, deterministic inference.
- **One implementation per token count or a few neighbours.** The configuration is a Python function of m (a constexpr in 335 of 471
  launches): each token count compiles its own binaries on its first call *(field, E11)* ([tuning.md](tuning.md) §14). Most
  implementations keep branches for other token counts (§12); Qwen2's experts-only Q2-E/m8, Q2-E/m64 and Q2-E/m128 assert their one token count.
- **All 65 are held to an output bound, not to bitwise equality**: runs may differ, accuracy is each output bound's job (the MoE `out` uses SQNR; Qwen2's router weights use a tolerance bound and its ids must match exactly) ([numerics.md](numerics.md) §1,
  §8); the kernels let atomic order place routes and sum values in a fixed order (§6, §10).

## 2. Pick the decomposition by token count

Gluon has no grid-barrier primitive (`gl.barrier()` synchronizes one CTA, S24), and no kernel of the library uses cooperative launch (`launch_cooperative_grid`, S23; none of its 154 kernels, nor of the 280 after its growth). Ending a launch is not an unconditional dependency rule: on toy Gluon kernels (256 experts, H=6144, m=1,2,16; one GPU run, 9 rotated rounds per size), two launches for logits before top-k took 0.26–0.51× the replay time of a last-arriving program doing top-k, and 0.15–0.48× a cooperative counter-barrier launch. For counts before rows, a last arriver scanning 256 counts beat two launches at m=1,2: two launches took 1.16× its replay time and 1.48–1.50× on the ruler's timer, with all 1000 calls bitwise equal per size. At m=16 the two timers disagreed (two launches: 0.89× replay, 1.28× the ruler's timer). Intermediate rows before down and routes before combine were not measured *(field, E12)*. Under the cooperative flag, a bounded counter barrier completed over 256/512/1024 one-wave workgroups (all 1000 eager and replayed calls per size; one run), and over 256 workgroups of 1024 threads (0.20–0.22 ms; N=2); 4096 of those large workgroups were refused. No plain-launch comparison established that the flag caused co-residency, nor was preservation of the cooperative graph-node attribute checked; the tested barrier took 2.1–6.8× two-launch replay time *(field, E4, E12)*. The decompositions below are observations of the measured implementations:

| Tokens | One program does | Launch sequence of one implementation *(field, E11)* | Speedups over the baseline, DeepSeek and MiniMax |
|---|---|---|---|
| ≤ 16 | one route × one column tile (× one K split), no sort; at 8–16 tokens MM64-FP4/m8–m16 and MM64-FP8/m1–m16 (64-expert ranks) and DS512/m16 (DeepSeek, 16) run one local expert × one column tile over all its tokens instead (§4.1) | DS256/m2, 2 tokens: router and quantization (304 programs: a full-K router program per expert, the rest MXFP4 quantization) → selection and gate-up (2306: M = 2 selection + 2304 gate-up programs; the routed-route ones recompute their expert, §4.1) → activation and requantization (72) → down (768: down and weighted combine) | 1.097–4.115 over 30 sizes *(field, E1)* |
| 32–256 | one expert job × one column tile: 16–128-row bodies in the DeepSeek/MiniMax and Qwen2 experts-only implementations; Q2-RE/m32 at 32 tokens uses 4-row routed jobs and 16-row shared-expert jobs | DS256/m256, 256 tokens: router projection → selection (top-8, relaxed atomic tickets, hidden quantization) → gate-up over expert tiles (each program derives its job, §4.2) → down over expert tiles → reduce (fixed-order combine) | 1.030–1.275 over 20 sizes *(field, E1)* |
| ≥ 1024 | one tile of a sorted, padded expert segment (all 32 sizes sort before the gate-up; no unsorted design was measured *(field, E11)*) × one column tile: nominally 64 or 128 rows (MiniMax; MM256-FP8/m1024–m16384 can execute a 32- or 64-row tail body at run time), or a height of 16–256 rows picked per tile at run time (DeepSeek, §7) | MM256-FP4/m16384, 16384 tokens: router projection (also zeroes counters, fills sentinels) → selection (top-8 + tickets) → scatter (rows + hidden quantization) → gate-up → down → combine | 1.152–1.785 over 32 sizes *(field, E1)* |

In routes per expert, r = m·k/E: a source census computed r for 89 MoE sizes from their token count, top-k and routed-expert count and traced their gate-up launches and grids, without classifying the job forms, so it sets no r boundary between per-route, grouped and sorted designs, nor one for other E or k *(field, E15)*.

The token count for a given r is N = r·E/k; this converts the coordinate, not a measured threshold, to another E, top-k or routing distribution, and how many experts a call touches depends on the routing distribution (below).

- **≤ 16 tokens: candidate fusions, not a guaranteed combined speedup.** In a toy top-4-of-32 MoE (H 256, I 128) at m = 2 and 16 each fusion saved 3.30–4.64 µs on the ruler's timer (top-k 3.30–4.19, SwiGLU 4.04–4.64, combine 4.19–4.43); all three took 0.667–0.751× the replay time and 0.426–0.462× on the ruler's timer (4 runs); single fusions were not timed replayed, requantization was not in the toy and full-K ownership was not varied. Synthetic fp8 VALU shapes (256 experts, top-8; H6144 with intermediate dimension 256, H3072 with 512; m = 1, 2, 8, 16; one run, 16 rotated rounds per size) put SwiGLU and 1×128 requantization in a full-K gate-up epilogue (§8): against a full-K gate-up with a separate activation launch it saved 1.76–2.76 µs replayed and 3.05–3.73 µs on the ruler's timer, but it lost to the unfused K-split design, and all three fusions took 1.08–2.30× that design's replay time. At H6144/m16 one fp8 code flipped at a rounding boundary (relative L2 1.122e-5 against the reference, above 1e-5); it passed only against a reference using its own activation. A/A controls of the K-split design showed replay bias at some sizes, the full-K comparison had none, and MXFP4 MFMA kernels were not timed this way *(field, E4, E13)*. Gate-up recomputation (§4.1) and down+combine (§9) need their own comparisons. Narrow column bands give one token about a thousand one-wave programs (DS256/m1: 1153).
- **32–256 tokens: group routes per expert.** Take a relaxed atomic ticket per route in the selection launch and either open a job every 16 (or BM, the tile height) tickets there, or let each GEMM program derive its job from the counts (§6, §4.2; neither form timed); sum the routes in a separate fixed-order launch *(field, E4)*. Uniform top-8-of-256 routing touches about 40 % of the experts at 16 tokens, 64 % at 32, 87 % at 64, 98 % from 128 (estimated for uniform routing: 1 − (31/32)^m). With routing simulated from the workload's router distribution, MM256-FP4/m64's 64-token call touched 77.2% of the experts on average (74.6% at the least), not 86.9%, and read those 495 MB at 4.87–5.00 TB/s, the baseline at 3.88–3.90 *(field, E19)*.
- **≥ 1024 tokens: a sorted grouped GEMM.** Counting sort into tile-padded segments (DeepSeek: variable-height tails, tallest first), a down GEMM that reuses one operand across tiles, a fixed-order combine (§6, §7, §9, §10); none timed against another design.
- **What a launch costs.** The ruler's timer read each kernel at about 4 µs or more under Kineto, a bare graph replay about 1.8 µs per extra kernel *(field, E11)* (the floor of the ruler's timer: [../pitfalls/profiled_timer_floor.md](../pitfalls/profiled_timer_floor.md), [authoring.md](authoring.md) §7). The FlyDSL trade: `../moe/few_tokens.md` §1–2.
- **Price the size before you design.** At 2 tokens DS256/m2 touches about 17 experts, 42 MB of weights and scales, a 6–7 µs read (estimated from the layouts and the 1 GiB stream rates, 6.0 TB/s plain and 6.8 TB/s with the `nt` bit a load's `.cg` sets), yet the call took 19.06–19.61 µs in 4 kernel launches, 4.76–4.90 µs per launch on average (4 runs of 2 measurements each; whole-call time on the ruler's timer) *(field, E11)* against the separately measured 4.03–4.07 µs launch floor (3 runs) *(field, E10)*: the per-launch reading fits a launch or latency cost but does not isolate it; fusion is a candidate to measure on the whole call, not a guaranteed speedup. At 256 tokens DS512/m256 reads all experts' weights, about 1.29 GB, in 225.5–236.8 µs, MM64-FP4/m128–m256 its 64 experts', about 481 MB, in 103.6–105.4 µs (4 runs) *(field, E11)*: about 80–95% and 67–77% of the cited 6.0–6.8 TB/s stream rate (estimated), little left over a stream the baseline pays too. At thousands of tokens price each stage: MM256-FP4/m8192 takes 505.7–518.6 µs at 8192 tokens (4 runs) *(field, E11)*. An earlier batch of runs read 225.8, 102.9 and 515.6 µs for DS512/m256, MM64-FP4/m128–m256 and MM256-FP4/m8192 *(field, E11)*. At the rates of [tuning.md](tuning.md) §13 MM256-FP4/m8192's expert GEMMs (0.62 TFLOP) need about 66 µs, its fp32 router (12.9 GFLOP) 82 µs on fp32 MFMAs or 16 µs as three bf16 MFMAs (§5.1), its 642 MB of weights and scales 94–107 µs, and the 403 MB of bf16 route rows its down writes and its combine reads 118–134 µs (all estimated): the fp32 router and that round trip each exceed the expert GEMMs' MFMA floor.

## 3. Put independent work in one grid

When two phases read only call inputs, they can share a program-id-role grid; the scalar role test is uniform.
A toy router GEMV (64 programs, 2 rows, K2048) beside a 4-program elementwise phase gave equal outputs in 4 runs; one grid took 0.495–0.506× the time of two launches on the ruler's timer in all four, but 0.755–0.835× the replay time in three and 1.154× (9.360 against 8.110 µs) in the fourth: a gain on the ruler's timer but not in every replay run *(field, E10)*. No DeepSeek or MiniMax kernel is named as a quantizer alone
*(field, E11)*. In DS256/m1 (1 token) 192 one-wave programs each compute a 16 × 16 router tile over one K split of 512
(16 expert tiles × 12 splits) beside 24 programs that quantize the token to MXFP4:

```python
    program = gl.program_id(0)
    if program < 256 // BN * SPLITS:
        tile = program % (256 // BN)
        split = program // (256 // BN)
        # ... (a 16 × BN router tile over this split's K on bf16 MFMAs, accumulated in acc)
        gl.store(Y + (om[:, None] * SPLITS + split) * 256 + on[None, :], acc, om[:, None] < M)
    elif GROUPED and program == 256 // BN * SPLITS:
        # ... (one program initializes the per-expert group masks)
    else:
        quant = program - 256 // BN * SPLITS - (1 if GROUPED else 0)
        # ... (this program's row and K groups loaded as fp32 values)
        routed, routed_scale = _quantize_groups(values, False)
        shared, shared_scale = _quantize_groups(values, True)
        # ... (both quantizations and their scales stored)
```

- Other role splits: programs `< M` select and publish the routes while the rest run the gate-up (DS256/m2); one CTA
  sorts in LDS while the others quantize (Q2-E/m128; Q2-E/m64 splits its roles by program id too); the router programs of
  an early launch also zero the counters and fill the sentinels a later launch reads (MM256-FP4/m16384: each router program
  initializes a 256-entry stripe after its tile; DS256/m256: router program 0).
- One binary serves every role, so a cheap role shares that binary's resource allocation: in a tested mixed-role kernel the binary used 256 VGPRs where the streaming role alone used 16, yet its streaming role took 0.966–0.974× the standalone stream's time (the heavy role not executed) *(field, E3, E10)*; HIP streams held to 2 waves per SIMD by 256 VGPRs were not slower at 32–64 B per thread reading and writing 512 MiB, but took 1.09× the time at 32 B over 128 MiB, 1.26× read-only at 32 B and 1.46× at 16 B per thread (latency- or compute-bound cheap roles not measured) *(field, E26)*.
- DeepSeek quantizes the token twice, once per exponent rule (routed, shared): [numerics.md](numerics.md) §3.

## 4. Drop a launch by recomputing its result

Recomputing a small decision is a candidate, not an unconditional rule. In the toy MoE (32 experts, top-4) it saved 2.89–3.40 µs at m=2 and 3.22–3.87 µs at m=16 on the ruler's timer, and 1.24–1.80 µs replayed at m=2 (4 runs). In the synthetic fp8 VALU shapes (256 experts, top-8; H6144 with intermediate dimension 256, H3072 with 512; m = 1, 2, 8, 16; one run, 16 rotated rounds per size), recomputation in every gate-up, down and combine consumer was slower on both timers. Recomputing only in the gate-up, which publishes ids and weights in that launch, helped at H6144 m = 1 and 8 (replay savings 2.87 and 2.45 µs, the ruler's timer 2.06 and 2.44 µs; the A/A confidence intervals of both include zero); the apparent replay savings at m = 2 matched the bias of A/A controls in both shapes, H3072/m1 also had replay bias, m = 16 was slower in both shapes, and at H3072/m8 the two timers disagreed in sign. These are paired confidence intervals within one run, not 16 independent experiments; deriving GEMM jobs from counts (§4.2) was not timed *(field, E10, E13)*.

### 4.1 Top-k in every gate-up program (≤ 16 tokens)

At 1 token DS256/m1 runs 1153 programs in its selection and gate-up launch: program 0 publishes `Ids`/`Weights` for the down launch, the
1024 routed-route programs (8 routes × 4 K splits × 32 column tiles) each recompute their route's expert, the 128
shared-expert ones take id 256. A recomputing program sums the router's 12 fp32 split partials in split order,
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

`_elect_rank` is the ballot election of §5.3; writing it: [gluon/wave_ops.md](gluon/wave_ops.md) §6.2.

- **Cost and order.** A program pays slot + 1 rounds. At 1 token DS256/m1 maps job j to slot `7 - j`, so the costliest
  selections start first (`route = gl.where(job < 8, 7 - job, 8)`); DS256/m2 at 2 tokens and MM256-FP4/m4–m16 at 4–16
  run slot 0 first, equal-slot routes adjacent. No source says why; no measurement compares them.
- **Where.** The DeepSeek and MiniMax implementations at 1–16 tokens, and the Qwen2 router+experts ones at 1–4 tokens, whose W13
  programs replay the top-k pops on the 512 router keys. Variants at 8–16 tokens: the MiniMax MXFP4 64-expert
  rank runs one program per local expert that keeps the tokens for which fewer than 8 experts beat it, the FP8
  one adds a selection launch and runs one gate-up program per local expert; DS512/m16 at 16 builds
  per-expert token bitmasks with `buffer_atomic_or` in a selection launch.
- **Measured.** All 15 DeepSeek and MiniMax sizes above 2.0 over the baseline, at 1–8 tokens, belong to implementations that recompute the routing inside the gate-up launch *(field, E1)*. None runs the same design with a separate selection launch; on a toy top-4/32 MoE dropping it saved 2.89–3.40 µs at 2 tokens and 3.22–3.87 at 16 on the ruler's timer, 1.24–1.80 replayed at 2 (4 runs; top-8 of 256 and deriving jobs from counts not tested) *(field, E10)*.

### 4.2 GEMM programs that derive their own job

At 16–256 tokens five implementations skip the job-list launch: each gate-up program finds its tile from the per-expert counts.
DS256/m256 (256 tokens) scans the counts; `PAD` (m rounded up to the tile height `TM`) is an expert's region (§6):

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
down launch. The others: wave ballots and popcounts over per-expert token bitmasks (DS512/m16 at 16 tokens),
a DPP prefix in inline assembly (DS256/m128), a scan and `gl.min` (MM64-FP4/m128–m256), a
persistent W13 grid (MM64-FP8/m128–m256 at 256); MM256-FP8/m32–m64 reads its overflow tiles at 32 tokens, and every tile at 64, from a
list its selection launch writes. These implementations measured 1.030–1.194 over the baseline *(field, E1)*, but no measurement compares
deriving the job with a job list the selection launch writes, and ratios across sizes cannot stand in for that (§11). Lookup forms and the
`readfirstlane` rule: [../moe/grouped_group_mapping.md](../moe/grouped_group_mapping.md) §5, §6.

### 4.3 The global prefix in every scatter program

A route's row needs the exclusive prefix of every expert's padded count: each program of MM256-FP4/m16384's scatter launch reloads
all count chunks, pads, scans and adds the earlier chunks' counts; program 0 stores `Total`:

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
The FlyDSL analog repeats the full M×9 route scan, expert histogram and padded-block prefix in every stage-1 workgroup; each live workgroup then ranks only its own expert’s items and retains its 16-row block: `../moe/few_tokens.md` §2.

## 5. Router GEMV and top-k

### 5.1 Compute the logits at the reference's rounding point

| The reference's routing | What the kernels compute | Where |
|---|---|---|
| DeepSeek: logits in bf16 (what SGLang's gate returns on ROCm), sigmoid, plus an fp32 bias | bf16 MFMA tiles (16×16×32) write fp32 split-K partials; the consumer sums them in split order and rounds once to bf16 | DS256/m1 |
| | at 2 tokens a full-K dot per expert with `gl.dot_fma`, both tokens sharing the weight load (read from the source); the router kernel compiles to 48 `v_dot2c_f32_bf16` and no MFMA *(field, E8)* | DS256/m2 |
| Qwen2: bf16 logits, top-10 by logit, the lower id on a tie | the same split-order sum and one bf16 rounding | Q2-R/m256 |
| | router+experts: float64 products (f64 VALU and no f64 MFMA at 1–4 tokens, `v_mfma_f64_16x16x4_f64` at 32 *(field, E8)*), rounded once to bf16 with a midpoint fix against double rounding (below; on 512 crafted near-midpoint logits f64 → f32 → bf16 erred on 260, Q2-RE/m1's helper on none *(field, E4)*) | Q2-RE/m1, Q2-RE/m32 |
| MiniMax: fp32 router weight, fp32 logits | fp32 multiply and `gl.sum` at 1–16 tokens (read from the source; MM256-FP8/m1–m16's key-preparing router, 1–16 tokens, compiles to no MFMA); the fp32 weight split into three bf16 terms on bf16 MFMAs (the bf16 hidden makes each product exact; the split read from the sources, below): the routers of the MM64-FP4 implementations at 2048–16384 tokens (MM64-FP4/m2048, MM64-FP4/m4096, MM64-FP4/m6144, MM64-FP4/m8192, MM64-FP4/m16384: 1.318–1.735 over the baseline) and of MM256-FP8/m1024–m16384 and MM64-FP8/m1024–m16384 run bf16 MFMAs and no fp32 MFMA; or fp32 MFMA: MM256-FP8/m32–m64's router (32–64 tokens) `v_mfma_f32_16x16x4_f32`, MM256-FP4/m16384's `v_mfma_f32_32x32x2_f32`, the MM256-FP4 routers at 1024–16384 tokens 16×16×4 (MM256-FP4/m1024, MM256-FP4/m2048–m4096) or 32×32×2 (MM256-FP4/m6144, MM256-FP4/m8192, MM256-FP4/m16384: 1.261–1.423 for the five) *(field, E1, E8)*; much else differs between these implementations: no ablation. On 13 of these router branches, each with a random and a cancellation input, the logits' error against fp64 was no larger than an fp32 computation's in L2 on 25 of the 26 inputs (MM256-FP4/m6144's cancellation input 1.00009×) and in max abs on 23 of 26; on two more inputs with a nonzero residual, MM64-FP4/m6144's and MM64-FP4/m8192's fp64 correction ran and both errors were 0 *(field, E20)*. | MM256-FP8/m1–m16, MM256-FP8/m256; MM256-FP8/m32–m64, MM256-FP4/m16384 |

The midpoint fix (Q2-RE/m1) and the three-term split (MM256-FP8/m256), as the kernels write them:

```python
# Q2-RE/m1: float64 value -> bf16; an fp32 halfway between two bf16 values first moves one fp32 step toward the float64 value
@gluon.jit
def _round_to_bf16(value):
    approximate = value.to(gl.float32)
    bits = approximate.to(gl.int32, bitcast=True)
    midpoint = (bits & 65535) == 32768
    inexact = value != approximate.to(gl.float64)
    toward_positive = value > approximate.to(gl.float64)
    step = gl.where(toward_positive == (approximate > 0), 1, -1)
    corrected = (bits + step).to(gl.float32, bitcast=True)
    return gl.where(midpoint & inexact, corrected, approximate).to(gl.bfloat16)
```

```python
# MM256-FP8/m256, the router's K loop: the fp32 weight tile b as three bf16 terms, one bf16 MFMA accumulator each
    for step in gl.static_range(CHUNK):
        # ... (a and b loaded and converted to the MFMA operand layouts)
        b0 = b.to(gl.bfloat16, fp_downcast_rounding="rtz")
        residual = b - b0.to(gl.float32)
        b1 = residual.to(gl.bfloat16, fp_downcast_rounding="rtz")
        b2 = (residual - b1.to(gl.float32)).to(gl.bfloat16)
        high = gl.amd.cdna4.mfma(a, b0, high)
        middle = gl.amd.cdna4.mfma(a, b1, middle)
        low = gl.amd.cdna4.mfma(a, b2, low)
    acc = high + (middle + low)
```

Check the reference's rounding points first ([numerics.md](numerics.md) §5), then compare each output against its
bound. On the Qwen2 router, whose ids must match exactly, Q2-R/m256 at m = 256 (one input set): swapping only token 0's first two ids, every other
id and `topk_weights` bitwise unchanged, failed the accuracy check on `topk_ids` (violation fraction 2/2560)
while the unchanged implementation passed twice; in three earlier runs on the same input, sending ties to the higher id (174 of
256 rows changed) or rounding the split partials to bf16 before the sum (66 rows) failed, and split-K 4 → 2 changed no
id and passed *(field, E3, E18)*. Keep the ids and their order; this does not show that every change to logit rounding
fails, and replacing a single id was not tested.

Skipping a rounding step need not fail or help: DS512/m8 (DeepSeek-V2, 8 tokens) ranking the unrounded logits passed
the accuracy check twice and changed no token's experts, with no resolved change in error; a gated-attention kernel at 32 rows
keeping the attention in fp32 before the gate also passed twice, while its scales' relative L2 against a float64
reference without that rounding rose from 0.003233 to 0.003269, a paired difference too small to resolve *(field, E18)*.

At DS256/m1 (1 token), on four saved inputs with near or exact ties, replacing only slot 7's expert in both gate-up and down moved `out` 6.21, 112.2, 6.13 and 9.47 times as far (L2) as removing only the bf16 rounding of the activation product; against the reference, the two total errors differed by 1.22, 3.03, 1.23 and 1.28× *(field, E21)*: on these inputs a wrong expert cost more than a skipped rounding.

**Three bf16 terms are a candidate, not a preference rule at many tokens.** Eight fixed register-MFMA probes
(grid 1536, 3072 iterations) give estimated fp32/router-to-MXFP4-expert matrix-core time ratios of 1.244–1.263,
and three-bf16 ratios of 0.244–0.246, for K=3072, 256 experts, top-8 and intermediate dimension 512. These omit memory and weight
splitting costs, are not router or expert GEMM latencies and do not establish performance at other token counts
*(field, E17)*; four earlier runs on the same shapes gave 1.255–1.284 and 0.2386–0.2452 *(field, E2)*.

At K=3072, (M,E)=(1024,64),(4096,64),(16384,64),(1024,256),(4096,256),(8192,256), M tokens and E experts of a synthetic router (written M16384/E64 below: an expert count, not an evidence id or a layer label of §1), each synthetic logits arm
was selected independently from a bounded search (BK32, stages1, fp-fusion on; BM/BN64/128/256, warps4/8,
BN≤E, BM·BN≤16384; 6 candidates/arm at E64, 12 at E256), not a global optimum. On random input (seed 1201),
three-bf16 logits kernels took 0.457–0.512× the two fp32-MFMA baselines' time, including W splitting each call
but excluding sigmoid, bias, top-8, normalization and expert GEMMs (N=1 run/size, 32 balanced AB/BA pairs).
Across eight sizes × eight seeds × three input modes × three arms, 217 of 576 accuracy comparisons failed the exact-id
check against the fp32 top-8 (all 192 near-tie inputs, 25 random or dynamic ones). At M16384/E64, random seed 705, both fp32
arms passed while split failed, despite lower split logit error (max abs 4.165e-6 vs 1.149e-5 against fp64).
Lower logit error does not guarantee unchanged routing ids. M8192/E64 and M16384/E256 had no id-exact
split tuning candidate on seed 1201: their accuracy inputs were all checked, but they were not timed.
Finite-sample noninferiority held for dynamic at M1024/4096,E64 and random/dynamic at M1024,E256; dynamic
speed was not measured, so these accuracy results are not joint speed/accuracy recommendations. Small M,
AITER, complete-router and whole-MoE performance remain unmeasured *(field, E17)*.

### 5.2 Size the router grid for bytes in flight

At up to 256 tokens the router streams a 3–4 MB weight (estimated), and a streaming probe approximately tracked bytes in
flight across its main 256–8192-program matrix; it also measured 64/96/128-program controls and 192-program
held-out arms, where matching nominal bytes did not reproduce the larger-grid rate *(field, E10)* ([tuning.md](tuning.md) §4;
inferred for a router). One design splits K over small MFMA tiles whose fp32 partials every consumer sums in split
order: DS256/m1's 192 one-wave tiles at 1 token, 1024 one-wave programs in DS256/m128 and DS256/m256 at 128–256 tokens, MiniMax's 512
CTAs at 128–256 (MM256-FP4/m128, MM256-FP4/m256, MM256-FP8/m256, MM64-FP8/m128–m256), the Qwen2 router's 256 CTAs from 8 to 256 tokens (Q2-R/m8 at 8: 1 × 8 splits × 32
expert tiles of one wave; Q2-R/m256 at 256: 8 token tiles × 8 expert tiles × 4 splits of four waves). The other gives
each expert (each token block and expert in MM256-FP8/m1–m16, MM256-FP4/m2, MM64-FP4/m8–m16 above 1 token) one full-K program that writes a finished
probability or ordered key, sparing every top-k consumer of §4.1 the split sum (DS256/m2 and the MiniMax implementations at 1–16 tokens: MM256-FP8/m1–m16,
MM64-FP8/m1–m16, MM256-FP4/m1, MM256-FP4/m2, MM64-FP4/m1–m2, MM64-FP4/m4, MM64-FP4/m8–m16) *(field, E11)*. No measurement compares the two router forms. Among half, as-written and double splits, the as-written
256 four-wave CTAs were fastest or tied on three Qwen2 routers: Q2-R/m256 half 1.33–1.34× call time, double
1.02–1.04× (N=8 runs); Q2-R/m128 1.22×/1.03×; Q2-R/m64 1.04×/0.99× (double tied; N=1 each).
Bytes in flight alone did not size Q2-R/m256: 128 CTAs four K tiles ahead (nominal bytes equal to 256 CTAs two ahead)
took 1.26×, and 512 CTAs one ahead 1.03× (N=1, 4 alternated rounds). At 256 CTAs one tile ahead took 1.12–1.14×
(N=6), four ahead 0.97× (N=1); the setting as written is not best across all look-ahead settings *(field, E10, E14)*.
Not speed knobs here: the
`amdgpu-sched-strategy` attribute, `amdgpu-ieee false` (accuracy unchanged) and, at 8 tokens (Q2-R/m8), its register
path for the hidden tile in place of async staging through LDS (it also changes the weight-load form) *(field, E6)*. At
1024 tokens and up K is split fewer ways or not at all (DeepSeek full-K, Qwen2 2 splits at 2232 tokens and none at 4096).

### 5.3 Top-k with cross-lane operations

For the measured top-8 of 256 (4 scores/lane, 16384 tokens), one wave/token took 12.96–13.18 µs over 3 runs,
four waves 37.68–38.32 µs, the same ids, no scratch, 0 and 31 `s_barrier`; 512 experts at 8 per lane and one CTA serving four tokens were not tested *(field, E11)*. Four waves on one token synchronize each round: MM256-FP4/m2048–m4096's four-wave selector compiles to
41 `s_barrier`, 25 LDS writes and 25 reads, the one-wave-per-token selectors of MM256-FP4/m16384 and MM64-FP4/m4096 (MM64-FP4/m4096's CTA serves 4
tokens) to no `s_barrier` (LDS not counted) *(field, E8)*; no measurement times it. Gluon 3.8.0 has no ballot, lane-read or
DPP builtin, so the kernels write them in inline assembly ([gluon/wave_ops.md](gluon/wave_ops.md) §6).

| Design | One round | Where |
|---|---|---|
| Election by ballot | `gl.max` over the wave; `v_cmp_*_e64` writes the lanes holding the maximum to an SGPR pair, `s_ff1_i32_b64` takes the first, `v_readlane_b32` broadcasts its id; lanes hold consecutive ids, so the lowest id wins a tie | DS256/m1 (`_elect_rank` in §4.1) |
| Keyed min, no inline asm | with retired experts' scores at −inf (as its caller sets them), `gl.max` of the scores, then `gl.min(where(available, where(score == max, e, e + 256), 512)) & 255`: the lowest available id holding the maximum (0 wrong of 4000 tie-heavy rows; unmasked retired maxima untested *(field, E11)*) | DS512/m16 |
| Per-lane sorted queues | sort each lane's keys once (a 5-compare network on 4 keys, or two 4-deep queues on 8); a round compares only the queue heads and pops the winner's queue | MM256-FP8/m32–m64 and MM256-FP8/m1–m16; Q2-R/m256, whose ballot winner goes into lane `slot` with `v_writelane_b32` |
| Order-preserving integer keys | the score's bits made monotonic, −0 folded into 0, the inverted expert id in the low bits: one `gl.max` returns the score and the lowest id | Q2-RE/m1 (16-bit bf16 key above a 9-bit id, below). Two reductions per round instead: MM256-FP8/m1–m16 keys the fp32 score alone (NaN to the largest key) and takes the lowest id with a second `gl.min`; MM256-FP8/m32–m64 packs the inverted id into 64 bits but at 256 experts still reduces the head scores with `gl.max`, then the ids with `gl.min` |
| Packed float64 keys | for finite scores, the same order inside a double's mantissa in [1, 2), so one float compare decides (`v_max_f64` in inline assembly in MM64-FP8/m128–m256); MM256-FP4/m2 maps NaNs to NaN-valued keys outside that interval | MM256-FP4/m2, MM64-FP8/m128–m256 |
| Packed uint64 keys, partitioned reduce | keys 4 per lane in a [4, 16] layout (a wave's four 16-lane rows); a round pre-reduces each lane's 4 in registers, runs a 16-lane `gl.reduce` whose combine compares the scores as floats and the ids on a tie, gathers the 4 row roots from lane 15 of each row and finishes 4-way; the winner's key becomes a tombstone | MM256-FP4/m16384; MM64-FP4/m4096 on (value, index) pairs |
| Rank counting, no rounds | one program per candidate expert counts, per token, the experts that beat it (higher score, or equal score and lower id); a count below k selects it and names its slot | MM64-FP8/m1–m16, exact at 8 and 16 tokens; MM64-FP4/m8–m16 returns a token bitmask: measured M2 keeps 8 via its id tie rule, M8 counts only strictly higher scores and a tied 8th/9th selects 9; M16 not run *(field, E11)* |

The integer key of Q2-RE/m1, from the bf16 logit's bits and the expert ids:

```python
    bits = logits.to(gl.uint16, bitcast=True).to(gl.uint32)
    bits = gl.where((bits & 32767) == 0, 0, bits)  # -0 folded into 0
    ordered = gl.where((bits & 32768) != 0, 65535 - bits, bits | 32768)  # negative scores below positive ones
    return (ordered << 9) | (511 - experts).to(gl.uint32)  # the inverted id below the score: the lower id wins a tie
```

No measurement compares designs. Election runs slot + 1 argmax rounds (from the code); over 16384 one-wave programs DS256/m2's
took 9.7–10.7 µs at slots 0–4 and 11.3–11.4, 12.8–13.0 and 14.3–14.4 µs at slots 5, 6, 7 in 3 runs, not linear in the
slot (R² 0.724–0.774) *(field, E11)*; the queue and rank-counting costs are estimated from the code, and the counting
that was timed ran in a test kernel, not in MM64-FP8/m1–m16. The selecting program also
renormalizes weights and, in the ticket designs, takes each route's ticket (§6), in the same launch. Traps:

- **One call, one tie rule.** Every §4.1 design selects at least twice per call: the program that publishes
  `Ids`/`Weights` and each gate-up program that recomputes its expert (DS256/m1: the publishing selection and `_expert_at_rank_peeled`;
  MM256-FP8/m1–m16: a `gl.min` round and per-lane queues; MM64-FP4/m4 both in one launch). The copies share the score
  (split-order sum, bf16 rounding) and the lower-id rule.
  Keep the copies consistent: at DS256/m1 (1 token), changing only the gate-up copy's bf16 rounding or its tie rule made producer and consumer disagree on a slot's expert on 12 of 13 inputs, and changing both alike restored agreement on 36 of 36; summing the 12 split partials in reverse order changed no id, weight or output on four saved inputs, which did not exercise that order *(field, E21)*. In the accuracy check, the copy without the bf16 rounding failed and the copy with the higher-id tie rule passed *(field, E18)*.
- **Read the winner from fixed lanes.** `gl.reduce` combines lane pairs in a butterfly; with a non-commutative combine
  (NaN scores) lanes disagree on the winner (on NaN rows only, in a test), so MM64-FP4/m4 reads it from lanes 60–63 with
  `ds_bpermute_b32`. Tested MM256-FP8/m1–m16 and MM256-FP4/m2 keys fold ±0 and 8 fp32 NaN payloads to one score field each; Q2-RE/m1 folds ±0 but gives 7 bf16 NaN patterns 7 keys *(field, E11)*.

## 6. Sorting and bucketing (32 tokens and up)

- **Tickets.** In the selection launch each route takes `ticket = gl.atomic_add(Counts + …, 1, sem="relaxed")` and stores `ticket << 8 | expert`; the scatter launch turns prefix (§4.3) plus ticket into its row of `Sorted`, and segments are padded to the tile height so a tile never mixes experts (MM256-FP4/m16384, its selection kernel, then its scatter kernel):

  ```python
      ticket = gl.atomic_add(
          Counts + row * TOPK // COUNT_BLOCK * 256 + ids, 1, sem="relaxed"
      )
      record = ticket << 8 | ids
  ```

  ```python
      offset = ticket + gl.gather(base_table, expert, 1)
      gl.store(Sorted + offset, route, valid)
      gl.store(Experts + offset // BM, expert, valid & (offset % BM == 0))
  ```

  The 8-bit field holds ids 0–255, MiniMax's 256 experts, and in an int32 record the ticket must stay below 2^23. With
  more ids widen it as the DeepSeek and Qwen2 job words do (`expert | live << 9 | block << 17`, decoded `& 511`, DS512/m256;
  `expert | count << 16`, `& 1023`, Q2-E/m64), and bound every field with `gl.static_assert` as DS256/m256 does.
- **Pass `sem="relaxed"`.** The consumers run in a later launch. The default `acq_rel` makes the AMD lowering add
  cache invalidates and L2 write-backs around the atomics (S23).
- **Which atomic you get.** Under Gluon `gl.atomic_add` compiles to `global_atomic_add`: `gluon_to_ttgir` has no
  buffer-op conversion (S23). `gl.amd.cdna4.buffer_atomic_add` takes a uint32 or uint64 counter, `buffer_atomic_or`,
  `and`, `xor` int32 or int64 (DS512/m16's masks are int32): a mismatch fails at trace time, `xchg` later *(field, E9)*.
- **One atomic per distinct element.** Copies that the layout makes (redundant lanes) are predicated off (S23): MM256-FP4/m16384's
  `[8]` `ids` in a 64-lane layout take 8 tickets *(field, E4)*. Copies you build yourself are distinct elements: MM256-FP4/m64
  widens its 8 winners to an `(8, 8)` tensor with `+ replica[:, None] * 0` to fill a wave, and masks all but one copy
  with `owner = replica[:, None] == 0`, or each copy takes a ticket.
- **Job queues (32–256).** Each expert owns a fixed region of the sorted buffer, as long as the routes it can receive:
  `PAD`, m rounded up to the tile (DS256/m256), or the routes rounded up to 16 (MM256-FP4/m64). A route's
  row is `expert · stride + ticket`, written in the selection launch with no scan; a second relaxed atomic opens a job
  every 16th (or BM-th) ticket of an expert and writes it to the job list (MM256-FP4/m64).
- **Persistence is rare.** Q2-E/m64 at 64 tokens loops its W13 programs over a device-counted list on a 3072-program grid
  (complete 16-program route tiles assigned in 256-program cohorts); its down stage uses a separate
  4096-program grid; MM64-FP8/m128–m256 at 256 runs
  a persistent W13 kernel on a 48 × 96 grid, the only kernel named persistent at the 96 sizes *(field, E11)*.
- **Sorts without global atomics.** The MiniMax MXFP4 64-expert rank at ≥ 1024 tokens gives lane j the 64-bit mask of the
  routes in its stripe whose expert is j with seven wave ballots, then popcounts (MM64-FP4/m1024); the FP8 64-expert rank
  packs four 8-bit counters per word. Both fix the row order; Qwen2's LDS atomics at 64 and 128 tokens
  (`atomic_scatter_add` in LDS, one CTA) do not (Q2-E/m64). No measurement compares them with a ticket sort.

  ```python
      # each ballot is a wave's 64-bit lane mask; a wave holds a stripe of 64 routes, and lane j also stands for expert j
      mask = _wave_ballot_2((r < R) & (ids >= 0) & (ids < 64))  # ballot 1: the routes to an expert of this rank
      for bit in gl.static_range(6):  # ballots 2-7, one per bit of the expert id
          bits = _wave_ballot_2(ids & 1 << bit != 0)
          mask &= gl.where(expert[None, :] & 1 << bit != 0, bits, ~bits)  # lane j keeps the routes whose expert is j
  ```
- **Graph capture.** Grids depend on shapes only (same grids on two routings at all 82 sizes *(field, E11)*): a
  worst-case tile bound on the host (MiniMax `cdiv(min(routes · bm, routes + E · (bm − 1)), bm) · bm`; DeepSeek `A +
  (8m − A) // block_m` with `A = min(256, 8m)`, the most experts that can hold a partial tile) and an exit on a
  device-written total (what else breaks capture: [authoring.md](authoring.md) §8). Scratch is `torch.empty`, filled
  by an earlier launch of the call: no implementation uses `torch.zeros` or a fill, one kernel more at the floor of the ruler's timer (§2).
  In FlyDSL: [../moe/grouped_group_mapping.md](../moe/grouped_group_mapping.md) §3, §4.
- **Ticket order is not deterministic.** Use tickets only to place rows whose result does not depend on the row, and sum
  a token's routes in slot order (§10): tickets differed in 19 of 20 runs, the slot-order sum kept one bit pattern
  *(field, E4)*. An output bound allows run-to-run differences, a bitwise requirement would not ([numerics.md](numerics.md) §1, §8).

## 7. Gate-up GEMM

A gate-up program loads its expert's preshuffled weight words straight into the MFMA operand layout
(`assert_trivial=True` compiles for DS512/m1–m2: no data moves between lanes *(field, E9)*; S24) and issues `mfma_scaled`.
Its E8M0 scale words go through LDS (DS512/m1–m2's gate-up; MM256-FP4/m16384's activation scales, preloaded per tile and gathered per K
step), are shifted out of their word per element and converted to the scale layout (DS256/m2), or load in the scale
layout directly (DS512/m1–m2's down). At ≥ 1024 tokens the activation tile the CTA's waves share goes through padded LDS (MM256-FP4/m16384).
Writing those loads, the `shuffled_16x16` offsets and the scale layout: [gluon/mfma.md](gluon/mfma.md) §7,
[gluon/memory.md](gluon/memory.md) §13, [gluon/layouts.md](gluon/layouts.md) §6.3. Particular to MoE:

- **Instruction per weight format** ([gluon/mfma.md](gluon/mfma.md) §2.1–§2.3). MXFP4: `mfma_scaled` on e2m1 with both
  scales, 16×16×128 and no 32×32×64 in the 14 DeepSeek and MiniMax MXFP4 implementations whose token counts are all ≤ 16
  *(field, E8)*. FP8 at 128 × 128 block scales (MiniMax): `mfma_scaled(a, None, "e4m3", b, None, "e4m3", zeros)` per
  128-K group, the fp32 scales applied after each group (MM256-FP8/m1–m16's 4–16-token gate-ups; the dense GEMMs' form,
  [gemm.md](gemm.md) §2). On a rank-3 layout (routes or K splits batched over warps), which `mfma_scaled` aborts on, the
  kernels run the fold on plain fp8 `mfma` [16, 16, 32] (`v_mfma_f32_16x16x32_fp8_fp8`, half the MACs per cycle)
  *(field, E2)*, compiled into MM256-FP8/m1–m16's down+combine and one of its W13 projections and MM256-FP8/m32–m64's split W13 projection.
  One MM64-FP8/m32–m64 down variant also uses plain fp8 MFMA, but its path, taken at m = 64, has a rank-2
  layout and is not evidence of a rank-3 requirement *(field, E8)*. Q2-E/m8 (Qwen2 experts, m = 8) casts both MFMA operands to bf16 in
  its gate-up and down kernels; all 254 finite e4m3 codes, both NaNs and signed zero convert to
  bf16 exactly (on the CPU) *(field, E15)*; the kernels were not run for this.
- **Gate and up in one column tile.** A BN-column tile holds BN/2 gate rows and the matching BN/2 up rows
  (`n = column * (BN // 2) + n % (BN // 2) + n // (BN // 2) * (N // 2)`), so the epilogue applies SwiGLU (§8).
- **Few rows on a 16-row instruction.** For one token, replicate it over the 16 rows and store row 0 (DS256/m2); fold K
  into the rows (MM256-FP4/m1: with a K fold of 4, row partition p and column block p read the same K chunk and the diagonal
  blocks are summed); put the routes' experts side by side along N and keep the diagonal (§9); or swap A and B so
  the weights fill M (Q2-E/m8). In a Gluon fp16 loop with random padding rows the 4-row instruction ran 1.155× the useful MACs of 16 padded rows *(field, E11)*; in HIP loops it ran 1.08–1.09× with random padding rows, through a higher clock (estimated), but 0.91× with zero padding rows and 0.79× as one dependent chain at one wave per SIMD *(field, E27)*. Q2-RE/m32 at 32 tokens runs routed tasks of at most 4 rows on `instr_shape=[4, 64, 64]` (`v_mfma_f32_4x4x4_16b_f16`), the shared expert on 16×16×32. Register maps: [gluon/mfma.md](gluon/mfma.md) §8, §4.
- **Gate-up grid at 1–2 tokens: not simply wider.** DS256/m2 sums split-K partials in its activation launch. At m=2, splits 4→1 (2306→578 one-wave programs) took 1.079× call time; 4→8 (4610) 1.050×. At m=1, DS256/m1 1153→577 took 1.020×, →2305 1.028×; DS512/m1–m2 1728→864 took 1.035×; MM256-FP8/m1–m16 column tile 8→16 (385→193 four-wave programs) 1.031× (one run, 4 alternated rounds per arm; scratch and reducing launches changed with the split, every arm accurate) *(field, E14)*. An earlier DS256/m2 comparison of 4 → 2 splits (2306 → 1154 programs) took 1.004× (speedup 2.183 → 2.175; paired 95% confidence interval of the reciprocal ratio 0.991–1.001), unresolved *(field, E11)*. These sizes set neither a widest-grid rule nor a bytes-in-flight sizing for MoE; the bytes in flight were counted only up to the expert-selection loads *(field, E14)*.
- **Tile height at run time (DeepSeek).** A descriptor carries the live row count and the program branches to a body compiled for 16, 32, 64 or 128 rows; the job list puts full tiles first, then the tails by height, tallest first (DS512/m256). DS256/m1024–m3072 pairs two bodies instead of padding to the next height: 65–96 rows as 64 + 16 or 32 at 1024 tokens, 129–192 as 128 + 16, 32 or 64 at 3072. MM256-FP4/m128 and MM256-FP4/m256 (128–256 tokens) drop to 16- or 32-row bodies for short tails, MM256-FP8/m1024–m16384 runs a 128-row tile with an empty second half as 64 rows from 4096 tokens; the other MiniMax or Qwen2 gate-ups do not select a different compiled tile height from their live rows. MM64-FP4/m4096, MM64-FP4/m2048, MM64-FP4/m8192, MM64-FP4/m16384 and MM64-FP4/m6144 still branch on `live == 64` to select an unmasked full-tile path versus a masked tail path at the same 64-row height. No measurement compares variable and padded heights. Over 100 calls on one fixed input DS512/m256's tile stayed in its 16-row body with one bit pattern, and on tiles of 15–17 and 31–33 live rows the neighbouring body sizes gave the same bits *(field, E22)*.
- **The weights' cache policy is per size.** Dropping `.cg` from DS256/m2 at m=2 changed call time by +0.1% to +1.0% over four runs *(field, E7)* and by +0.2% over three sessions on the rocprofiler-sdk timer *(field, E23)*, less than one run of the ruler resolves at this size (§11). Dropping it from MM64-FP4/m256 cost 1.7–4.0% and adding it to DS512/m256 0.2–0.8%, each slower in every pair of five runs *(field, E7, E16, E23)*. DS512/m256 leaves `.cg` off its 128-row gate-up tiles above 128 tokens, at 256 the shared expert's two tiles (256 rows; a routed expert's about 8, estimated), whose weight address has no row-tile term, so both read the same columns; a cache hit on the second is inferred (a toy in which two programs read each block ran 4 % faster without `.cg` *(field, E7)*). Put `.cg` (`cache_modifier=".cg"` on `gl.load`, `cache=".cg"` on `buffer_load`; it sets `nt`) on weights the call reads once; leave it off weights that a program on another XCD re-reads in the same pass, and settle other re-reads per size ([gluon/memory.md](gluon/memory.md) §6); in a read-only probe `.cg` cost 4.8–6.5% when the adjacent program id (another XCD) re-read each block and showed no consistent sign when the re-reader shared the XCD *(field, E24)*. In one run over 12 sizes of seven implementations, `.cg` won at seven of ten predominantly read-once sizes (1.6–9.6%; six had every pair favorable in call time and in speedup, MM256-FP8/m2 only the aggregate direction), with no consistent effect at three m=1 sizes and no measured loss. It lost at both re-read sizes: DS512/m256 (above) and MM256-FP4/m16384 (17%, N=1). Reuse was classified from the schedule and measured routing, not cache-hit counters; the reuse side covers only two sizes *(field, E16)*. Store hints are unmeasured here: DS256/m1024–m3072 writes cross-launch partials `.wt` and its output `.cs` at 2048 and 3072 tokens, the Qwen2 routers their partials `.wt`, Q2-R/m8 through `gl.store`, which, unlike `buffer_store`, waits after each `.wt` store *(field, E9)*.
- **Program order.** DS256/m8–m16's down at 8–16 tokens makes a column tile's program ids congruent mod 8, so with round-robin dispatch they share an XCD on any stream (the absolute XCD index rotates on a non-default stream). MM256-FP8/m256 walks its job list in 8 contiguous stripes; MM256-FP8/m1024–m16384 keeps an M tile's N tiles on one XCD. All placement here is derived from the formulas; none was measured on MoE kernels: [gluon/memory.md](gluon/memory.md) §14.
- **`waves_per_eu` did not change the time where measured.** The gate-up of MM256-FP4/m8192 and MM256-FP4/m16384 (MiniMax MXFP4, 256 experts) sets 3 above 3072 tokens; removing MM256-FP4/m8192's at 8192 tokens left the speedup at 1.44 (1.441 → 1.440) *(field, E7)*; 146 VGPRs with it, 150 without, no spills *(field, E9)* ([tuning.md](tuning.md) §8, [debugging.md](debugging.md) §2).
- **A size's constants are coupled.** Changing MM256-FP4/m1's K fold from 4 to 1 alone at 1 token crashed every run of the ruler *(field, E7)*: LLVM's code generation dies with SIGSEGV on a K fold of 1 with 4 warps, a 128-wide K tile and the `amdgpu-sched-strategy=iterative-ilp` attribute the implementation sets at m ≤ 2; a K fold of 1 compiles with 1 warp, with a 256-wide K tile, or without that attribute *(field, E9)*. Change constants together, from a combination that compiles elsewhere (here the same source's 16-token branch) ([debugging.md](debugging.md) §6).

## 8. Activation and requantization

| Situation | Where SwiGLU and requantization go | Examples |
|---|---|---|
| One program owns full K for its gate+up columns and a whole quantization group of the output | the gate-up epilogue | MXFP4 queued and sorted designs (MM64-FP4/m4096), DeepSeek from 4 tokens and DS512/m1–m2 at 2 |
| The gate-up is split along K over programs | a small launch that sums the split partials in order, then SwiGLU and quantization | DS256/m2 and DS256/m1, DS512/m1–m2 at 1 token, Q2-E/m8 *(field, E11)* |
| The down operand is fp8 per 1 × 128 | the gate-up stores bf16; the down kernel quantizes each 128-group as it loads it | MM256-FP8/m1–m16, MM64-FP8/m32–m64 |

MM64-FP4/m4096's epilogue rounds the accumulator and the SwiGLU product to bf16 and quantizes groups of 32 to MXFP4:

```python
    rounded = acc.to(gl.bfloat16).to(gl.float32)
    pairs = gl.permute(gl.reshape(rounded, (BM, 2, BN // 2)), (0, 2, 1))
    gate, up = gl.split(pairs)
    value = (gate * _sigmoid(gate) * up).to(gl.bfloat16)
    groups: gl.constexpr = BM * BN // 64
    grouped = gl.convert_layout(gl.reshape(value, (groups, 32)), layout).to(gl.float32)
    q, scale = _quantize_groups_3(grouped)
```

- **Take the activation from the reference.** These layers split `x @ W13ᵀ` into gate (columns `[0, I)`, I the intermediate dimension)
  and up (the rest) and compute `silu(gate) · up`, SwiGLU on this page. Other fused MoE ops also have 'gelu', 'situv2',
  gpt-oss's 'swiglu' (`gate · sigmoid(1.702 · gate) · (up + 1)`) and silu on an interleaved `w1` (`w1`
  alternating 16-row gate and up tiles); these two first clamp gate to ≤ L and up to [−L, L], L = `swiglu_limit`.
- Write the consumer's operand format: MM256-FP4/m2048–m4096 stores the quantized activation in the weights' 16 × 16 shuffle and its
  scales in the shuffled E8M0 map (at 4096 tokens padding rows get code 127, at 2048 they stay unwritten), so the down
  GEMM loads A like B, where row-major words cost LDS *(field, E9)* ([gluon/mfma.md](gluon/mfma.md) §7).
- The MXFP4 and fp8 encoders (exponent rule, NaN guard, `v_cvt_scalef32_pk_fp4_f32` packing, saturation) and the
  little room the MXFP4 bound leaves beyond the activation quantization: [numerics.md](numerics.md) §1–§3. FlyDSL's
  GLU-in-epilogue numbers: [../moe/grouped_lowprec.md](../moe/grouped_lowprec.md) §10.
- Most DeepSeek launches (86 of 100 sites, all three of DS512/m8's) pass `enable_fp_fusion=False`, MiniMax and Qwen2
  only MM256-FP4/m4–m16's down (16 tokens); on at 8 tokens, DS512/m8's code is unchanged: 1.147 → 1.146, accuracy passed *(field, E7)*.

## 9. Down GEMM

- **≤ 16 tokens: down+combine is a candidate, not a uniform win.** In the toy top-4/32 MoE at m = 2 and 16, fusing down and combine saved 2.94–3.73 µs on the ruler's timer (4 runs; its replay saving and output equality alone not measured; all fusions against none: §2) *(field, E10)*. In the synthetic fp8 VALU shapes at m = 1, 2, 8, 16 (one run, 16 rotated rounds per size; slot order as the reference sums), it added 0.25–8.18 µs of replay time on average (savings −8.18 to −0.25 µs); at H6144/m2 the confidence interval crossed zero and some small differences fell within the bias of A/A controls, at H3072/m8 and m16 it was slower on the ruler's timer too, and at none of the sizes did all fusions together reach 0.85× the unfused time *(field, E13)*. Neither shows how MXFP4 MFMA kernels fare. The source form in DS512/m1–m2 puts routes in MFMA A rows and experts' column slices side by side in B, keeps the diagonal blocks and sums them weighted in slot order, without partials or atomics:

  ```python
      acc = _tiny_projection(X, XS, W, WS, Ids, token, column, N, K, BN, BK, WARPS, False, CACHE)
      vn = gl.arange(0, 8 * BN, gl.SliceLayout(0, mma))
      diagonal = gl.gather(acc, vn[None, :] // BN, 0).reshape((8, BN))
      # ... (diagonal converted to a layout ep that holds one row per route; value starts at 0 in it)
      for rank in gl.static_range(8):
          part = gl.gather(diagonal, gl.full((1, BN), rank, gl.int32, ep), 0)
          weight = gl.load(Weights + token * 8 + rank)
          value += part * weight
      shared_acc = _tiny_projection(X, XS, W, WS, Ids, token, column, N, K, BN, BK, WARPS, True, CACHE)
  ```

  DS256/m2 puts all 9 routes (shared expert included) into one 16 × 256 MFMA tile per program; MiniMax FP8 (256 experts)
  gives each of 8 waves one route (`AMDMFMALayout(..., warps_per_cta=[8, 1, 1])`, MM256-FP8/m1–m16) on plain fp8 `mfma` (§7).
- **≥ 1024 tokens: operand reuse across tiles (no gain where ablated).** MM256-FP4/m16384 holds a 64-row activation tile in
  registers across a strip of column tiles; DeepSeek loads one expert's weight panel and scales
  once and streams activation tiles through it. DS256/m1024–m3072's schedule leaves that off at 2048
  tokens, which measured 1.329 over the baseline against 1.402 at 1024 and 1.407 at 3072 *(field, E1)*; turning it on changed the time by
  −0.04 % to −0.5 % in 3 runs *(field, E3)*: no gain (activation-tile reuse not ablated). Atomics against a
  fixed-order reduce, on FlyDSL kernels: [../moe/grouped_sorted_tokens.md](../moe/grouped_sorted_tokens.md) §6.

## 10. Combine and the shared expert

- **Combine in slot order.** `out[t] = Σ_slot w[t, slot] · down[route]`, slot 0 to 7, from bf16 route rows
  (MM256-FP4/m16384). The route rows are a `routes × H` bf16 scratch the down writes and the combine reads: 805 MB
  each way at 16384 tokens for MiniMax MXFP4 with 256 experts (estimated from its `torch.empty` shape). Other implementations
  reorder it: the MM64-FP4 ones at ≥ 1024 tokens store the down rows in sorted order, read through an inverse
  map kept in the ids buffer (MM64-FP4/m4096); MM256-FP4/m6144 writes them by sorted row in 256-column chunks, read through a position
  map; DeepSeek at ≥ 1024 tokens stores dense expert-sorted rows and points each route's record at its row (DS512/m1024). No
  measurement compares the hand-offs. Designs at ≤ 16 tokens fuse the combine into the down (§9); MM64-FP8/m1–m16 and DS512/m16 keep a launch for it.
- **No float atomics** in the 65 implementations *(field, E8)*: integer counters, cursors, job indices, bitmasks, LDS markers.
  Q2-E/m64 and Q2-E/m128 reorder rows across calls, yet they and 29 global-ticket implementations repeat outputs *(field, E11)*.
- **DeepSeek's shared expert (id 256, weight 1)** changes the plan by size: at ≤ 16 tokens it is the token's ninth
  route; at 32–256 tokens its jobs run first, its down result is stored straight into `out`, and the reduce adds the
  8 routed partials on top of it (DS256/m256); at ≥ 1024 tokens the shared expert's down epilogue gathers the 8 routed
  partials of each token and writes `out`, so there is no combine launch (DS256/m4096–m4192).
- **Qwen2's shared expert (id 512)** is weighted by `sigmoid(hidden · shared_gate_weight)`; its dot products come
  from the shared expert's own gate-up programs (slot 10, tile 0), which add one partial dot per K group (Q2-E/m8).

## 11. What the measurements say

The ablations of these implementations use matched copies and alternating original/candidate measurements on one MI355X
with the ruler ([tuning.md](tuning.md) §1); their run counts and rounds are stated per result, and the probes state their own.
Runs of an unchanged implementation against itself spread 0.9–2.5 % (Q2-R/m256) and 0.9–3.2 % (DS256/m2) as (max − min)/mean over 3 runs in four sessions *(field, E5)*, and 1.8 % and 3.05 % over 14 runs on the rocprofiler-sdk timer *(field, E25)*: a difference of about 1 % at these sizes is not resolved by one run of the ruler. The speedups below are per size, from one run of the ruler, against SGLang's path on aiter; a later run re-measured 57 of the 96 sizes and moved them by −6.3% to +7.0% (median 0.9%) *(field, E1)*:

| Scenario | ≤ 16 tokens | 32–256 | ≥ 1024 |
|---|---|---|---|
| DS256 (DeepSeek, rank intermediate dimension 256) | 1.245 (16) – 2.186 (2) *(field, E1)* | 1.030 (128) – 1.160 (256) | 1.329 (2048) – 1.420 (4096) |
| DS512 (DeepSeek, rank intermediate dimension 512) | 1.097 (16) – 2.093 (1) *(field, E1)* | 1.043 (256) | 1.152 (1024) – 1.391 (2048) |
| MM256-FP8 (MiniMax FP8, 256 experts) | 1.732 (16) – 4.115 (1) *(field, E1)* | 1.106 (128) – 1.173 (64) | 1.248 (2048) – 1.388 (16384) |
| MM64-FP8 (MiniMax FP8, 64 of 256) | 1.325 (16) – 3.137 (1) *(field, E1)* | 1.102 (64) – 1.167 (32) | 1.216 (1024) – 1.785 (16384) |
| MM256-FP4 (MiniMax MXFP4, 256 experts) | 1.595 (16) – 3.444 (1) *(field, E1)* | 1.178 (256) – 1.230 (64) | 1.261 (4096) – 1.423 (8192) |
| MM64-FP4 (MiniMax MXFP4, 64 of 256) | 1.555 (16) – 3.633 (1) *(field, E1)* | 1.190 (128) – 1.275 (64) | 1.252 (1024) – 1.735 (8192) |
| Q2-RE (Qwen2 router+experts) | 1.119 (4) – 1.172 (1) *(field, E1)* | 1.063 (32) | — |
| Q2-R (Qwen2 router) | 1.074 (8) – 1.219 (16) *(field, E1)* | 1.295 (64) – 1.967 (256) | 1.226 (4096) – 1.416 (2232) |
| Q2-E (Qwen2 experts) | 1.208 (8) *(field, E1)* | 1.076 (128) – 1.171 (64) | — |

Each value belongs to the whole implementation at that size, and repeat runs of unchanged sources move a ratio by
2 % at the median, 6 % at p90 *(field, E1)* ([tuning.md](tuning.md) §1): such gaps say nothing about designs. In every
DeepSeek and MiniMax scenario the smallest ratio sits at 32–256 tokens, where both sides stream nearly every expert
(§2), the largest at 1–2 tokens, where the baseline runs 8–11 device operations to these implementations' 3–4 *(field, E11)* (the links
inferred). MM256-FP4/m8192 (8192 tokens, 1.423) and MM256-FP4/m16384 (16384, 1.389) share their kernel source but for the docstring, as do MM64-FP4/m8192
(8192, 1.735) and MM64-FP4/m16384 (16384, 1.619) *(field, E1)*.

## 12. Traps when changing one constant or branch

What broke, or silently would, when one constant or branch of these implementations is changed alone; each is a condition to check in a kernel of your own.

- **Dead branches.** An implementation grown from a multi-size source keeps paths its token count never takes (DS256/m1 a 2-token projection and a non-peeled top-k, Q2-R/m256 configurations for 64 and 128 rows). Only the branch its constexprs select was compiled and checked: follow your token count's constexpr branch before you copy or change a constant; an edit in another branch changes nothing at your size.
- **Scenario constants are baked in, and the entry function checks few of them.** A call with another scenario then computes the wrong thing without an error. DeepSeek multiplies the routed weights by 2.5 in device code, DS256/m8–m16 sums 12 router splits in its top-k; the MM256-FP8 and MM256-FP4 implementations ignore `expert_offset` (right while the rank holds all 256); the MM64-FP4 ones at ≥ 1024 tokens read an expert from 6 bits (`descriptor & 63`); Qwen2's router+experts entry function ignores `topk` and `out_dtype`; Q2-R/m128's `(256,)` router grid with `split = 8 if rows == 64 else 4` holds for 64, 128 and 256 rows only. Re-derive them for another scenario.
- **Write every token's row at a 64-expert rank.** About 10 % of tokens have no route to a held expert (estimated, uniform routing: C(192, 8)/C(256, 8)); their `out` row is zero and must still be written: an accuracy check whose outputs start as NaN catches a skipped row ([authoring.md](authoring.md) §8), and an MM64-FP4/m1024 copy that skipped them failed *(field, E4)*. The combines as written loop over all 8 slots of every token and add nothing for a non-local route (MM64-FP4/m1024; MM64-FP8/m32–m64).
- **Buffer ops stop at 2 GiB from one base.** MM256-FP4/m6144, MM256-FP4/m8192 and MM256-FP4/m16384 have host extent guards for their router/scale-buffer paths (MM256-FP4/m16384: three `… < 2**31` tests); their data/output buffer choices also depend on policy flags fixed per token count. MM256-FP4/m2048–m4096 guards in device code and MM256-FP4/m1024 by token count. These are not a universal guard for every buffer base. Route-row scratch is routes × H: a larger token count or H would cross it (a buffer load read 0 past 2 GiB on a generic tensor *(field, E4)*; memory.md §3).
- **A written launch list can be short.** DS256/m4's own description lists two launches; a trace of the call at 4 tokens shows three (a padded down follows), and 4 other descriptions differ from the trace *(field, E11)*: count a call's launches from a trace or the host function, not from a description.
- **A worst-case grid may reach descriptor or sorted-row entries that no route fills; a device-total exit can prevent some descriptor reads.** Initialize the sentinels (out-of-range route ids the consumer masks: MM256-FP4/m16384 fills `Sorted` with the route count; MM256-FP4/m64 with its capacity) in an earlier launch of the same call, also the tails no route fills (without its fill MM256-FP4/m64 crashed *(field, E4)*).
- **Scratch aliasing: source order, not a GPU safety proof.** DS256/m4096–m4192 aliases router logits with the activation buffer and quantized input with partials; in its host code each aliased buffer is written, read and overwritten in launch order, read from the source *(field, E15)*; no GPU run tested the aliasing.

## Evidence

Triton paths are `triton-lang/triton@c01b677` (tag v3.8.0). "The ruler" and "the ruler's timer", its per-call time alone, are those of [tuning.md](tuning.md) §1; a speedup is the ruler's stored value against the baseline its line names. Kernel times were read with Kineto unless a line names rocprofiler-sdk.

- E1 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, the ruler (graph replay, 1 GiB read flush, Kineto), speedup over the baseline of SGLang's path on aiter] speedup per size, the implementation in parentheses: DeepSeek-V2/GLM DS256: 1 token 2.139 (DS256/m1), 2 2.186 (DS256/m2), 4 1.867 (DS256/m4), 8 1.521 and 16 1.245 (DS256/m8–m16), 32 1.086 (DS256/m32), 128 1.030 (DS256/m128), 256 1.160 (DS256/m256), 1024 1.402, 2048 1.329 and 3072 1.407 (DS256/m1024–m3072), 4096 1.420 and 4192 1.378 (DS256/m4096–m4192); DS512: 1 token 2.093 and 2 1.829 (DS512/m1–m2), 4 1.324 (DS512/m4), 8 1.184 (DS512/m8), 16 1.097 (DS512/m16), 256 1.043 (DS512/m256), 1024 1.152 (DS512/m1024), 2048 1.391 (DS512/m2048), 3072 1.235 (DS512/m3072), 4096 1.161 (DS512/m4096), 4192 1.166 (DS512/m4192); MiniMax-M2 FP8, 256 experts: 1, 2, 4, 8, 16 tokens 4.115, 3.755, 2.452, 1.841, 1.732 (MM256-FP8/m1–m16); 32 1.158 and 64 1.173 (MM256-FP8/m32–m64); 128 1.106 (MM256-FP8/m128); 256 1.133 (MM256-FP8/m256); 1024–16384 1.257, 1.248, 1.348, 1.372, 1.388 (MM256-FP8/m1024–m16384); FP8, 64 of 256 experts (expert_offset 192): 1, 2, 4, 8, 16 tokens 3.137, 2.912, 1.672, 1.466, 1.325 (MM64-FP8/m1–m16); 32 1.167 and 64 1.102 (MM64-FP8/m32–m64); 128 1.146 and 256 1.164 (MM64-FP8/m128–m256); 1024–16384 1.216, 1.485, 1.681, 1.770, 1.785 (MM64-FP8/m1024–m16384); MXFP4, 256 experts: 1 token 3.444 (MM256-FP4/m1), 2 3.309 (MM256-FP4/m2), 4, 8, 16 2.549, 1.977, 1.595 (MM256-FP4/m4–m16), 32 1.179 (MM256-FP4/m32), 64 1.230 (MM256-FP4/m64), 128 1.210 (MM256-FP4/m128), 256 1.178 (MM256-FP4/m256), 1024 1.352 (MM256-FP4/m1024), 2048 1.272 and 4096 1.261 (MM256-FP4/m2048–m4096), 6144 1.415 (MM256-FP4/m6144), 8192 1.423 (MM256-FP4/m8192), 16384 1.389 (MM256-FP4/m16384); MXFP4, 64 of 256 experts (expert_offset 192): 1 token 3.633 and 2 2.797 (MM64-FP4/m1–m2), 4 2.554 (MM64-FP4/m4), 8 2.113 and 16 1.555 (MM64-FP4/m8–m16), 32 1.231 and 64 1.275 (MM64-FP4/m32–m64), 128 1.190 and 256 1.194 (MM64-FP4/m128–m256), 1024 1.252 (MM64-FP4/m1024), 2048 1.318 (MM64-FP4/m2048), 4096 1.590 (MM64-FP4/m4096), 6144 1.689 (MM64-FP4/m6144), 8192 1.735 (MM64-FP4/m8192), 16384 1.619 (MM64-FP4/m16384); Qwen2-MoE router+experts: 1 token 1.172 (Q2-RE/m1), 2 1.145 (Q2-RE/m2), 4 1.119 (Q2-RE/m4), 32 1.063 (Q2-RE/m32); router: 8 tokens 1.074 (Q2-R/m8), 16 1.219 (Q2-R/m16), 64 1.295 (Q2-R/m64), 128 1.417 (Q2-R/m128), 256 1.967 (Q2-R/m256), 2232 1.416 (Q2-R/m2232), 4096 1.226 (Q2-R/m4096); fp8 experts: 8 tokens 1.208 (Q2-E/m8), 64 1.171 (Q2-E/m64), 128 1.076 (Q2-E/m128); the values come from eight runs of the ruler, each implementation's from one run; a later run re-measured 57 of the 96 sizes, by −6.3% (MM256-FP8 at 1 token, 4.115 → 3.855) to +7.0% (Q2-R at 16 tokens, 1.219 → 1.304), median change 0.9%; over repeat runs of the ruler on unchanged sources, the ratios at the same size differ by 2.03 % at the median, 6.2 % at p90 and by more than 5 % in 16 % of 1716 pairs at 170 sizes, at most 54.8 % (Q2-RE at 1 token)
- E2 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=4 (the rank-3 compile in a fifth run too), Gluon MFMA loops on register operands; the MiniMax MXFP4 layer's shapes, H 3072, 256 experts, top-8, intermediate dimension 512] plain fp8 16 × 16 × 32 at 0.498, 0.500, 0.498, 0.501 of the rate of `mfma_scaled` 16 × 16 × 128 without scales; at the measured MFMA rates the fp32-MFMA router over the expert GEMMs, per token, 1.255, 1.262, 1.270, 1.284 and the three-bf16 split 0.2386, 0.2392, 0.2410, 0.2452 (1.33 and 0.25 at the nominal MACs per cycle); three bf16 MFMA chains over the fp32 MFMA time 0.1888, 0.1887, 0.1890, 0.1891, and on 64 × 64 logits at K = 3072 (one seed) the max error against fp64 3.6e-6 for the split, 1.4e-5 for fp32 MFMA, the same in every run; `mfma_scaled` on a rank-3 e4m3 layout of two batched 16 × 16 × 128 tiles aborts the compiling child while tracing (exit −6, LLVM ERROR "Dimensions must match") and plain fp8 `mfma` 16 × 16 × 32 on that layout lowers to `v_mfma_f32_16x16x32_fp8_fp8`, in all 5 runs (probe kernels)
- E3 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3, the ruler, one GPU, A/B 4 alternated rounds per arm; the ruler's timer (Kineto) on a test kernel] Q2-R/m256 at 256 tokens: ties sent to the higher id fail the accuracy check (ids differ in 174 of 256 rows, 176 rows hold ties), fp32 split partials rounded to bf16 before the sum fail (66 rows), split-K 4 → 2 passes with no id changed, identical in the 3 runs; DS256/m1024–m3072 at 2048 tokens with the weight-resident down on (its down group 1 → 2: one expert's weight panel held across two 128-row activation tiles): 234.6 → 234.2, 230.8 → 229.6, 235.9 → 235.8 µs (−0.17 %, −0.52 %, −0.04 %), 12 alternated pairs B/A 0.974–1.006, speedup 1.311–1.321 off and 1.309–1.326 on, the 1024- and 3072-token controls within 1 %; a two-role test kernel at 256 VGPRs launched without heavy work: a flushed 128 MiB fp32 read and write stream 44.03, 44.00, 43.93 µs in it against 45.56, 45.19, 45.35 µs alone (0.967–0.974×) (test kernel)
- E4 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2 (the toy MoE N=4), test kernels and copies of the implementations' code on one GPU] DeepSeek's reference with both GEMMs' activations MXFP4-quantized moves `out` by rel L2 0.2103, max abs 31.19; a MM256-FP4/m64-form copy at 32, 128 and 256 tokens puts each route in a job of up to 16 rows of one expert, and over 20 runs at 256 tokens the tickets differ in 19 while the slot-order combine keeps one bit pattern (a float `atomic_add` in arrival order gives 20); 1000 programs of 8 ids in a 64-lane layout leave every counter at 1000; 512 crafted near-midpoint logits: f64 → f32 → bf16 wrong on 260, with the midpoint fix on 0, Q2-RE/m1's helper 0 of 1024; an MM64-FP4/m1024 copy storing only the rows with a local route: 239616 non-finite mismatches in `out`; a MM256-FP4/m64 copy without its fill of `Sorted`: the copy crashes, exit −6; under `launch_cooperative_grid` a counter barrier over 256 workgroups of 1024 threads finished in 0.2 ms, every program seeing every store; 4096 workgroups were refused (`hipErrorCooperativeLaunchTooLarge`); on a 2.5 GiB tensor `buffer_load` read the data at byte 2^31 − 64 and 0 at 2^31 + 64 with no fault, `gl.load` both; a toy decode MoE (32 experts, top-4, H 256, I 128, every gate-up program owning full K, no requantization) at 2 and 16 tokens against no fusion, 4 runs: each fusion alone saves on the ruler's timer (Kineto) top-k recomputed in the gate-up and down programs 3.30–4.19 µs, SwiGLU in the gate-up epilogue 4.04–4.64 µs, down and combine in one program 4.19–4.43 µs (not timed replayed); the three together give call-time ratios of 0.667–0.751 replayed and 0.426–0.462 on the ruler's timer, rel L2 against no fusion 4.5e-8 and 5.1e-8 (test kernels)
- E5 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=4 sessions, the ruler, one GPU, the implementation against itself and an identical copy] first session: Q2-R/m256 at 256 tokens 1.93, 1.92, 1.94; DS256/m2 at 2 tokens 2.18, 2.19, 2.17; second, (max − min)/mean of the first 3 runs: Q2-R/m256 0.91 %, DS256/m2 0.94 %; over all 8 runs (A and the copy) 0.95 % and 1.27 %; third and fourth, first 3 runs: Q2-R/m256 0.98 % and 2.46 %, DS256/m2 2.39 % and 3.17 %
- E6 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2 sessions, 4 alternated rounds per arm, the ruler, one GPU] Qwen2 router Q2-R/m256 at 256 tokens: K split 4 → 2 (grid 256 → 128 CTAs) 1.923 → 1.449 and 1.927 → 1.441, its time 9.93 → 13.21 and 9.906 → 13.29 µs; split 4 → 8 (512 CTAs) 1.931 → 1.874 and 1.915 → 1.864 (9.91 → 10.22 and 9.971 → 10.25 µs, all 4 B runs slower in the second); look-ahead 2 → 1 1.925 → 1.696 and 1.917 → 1.699 (9.93 → 11.26 and 10.01 → 11.28 µs); `amdgpu-sched-strategy` removed from both launches 1.927 → 1.922 and 1.923 → 1.917; `amdgpu-ieee false` removed from the routing launch 1.935 → 1.924 and 1.921 → 1.928; Qwen2 router Q2-R/m8 at 8 tokens, the hidden loaded into registers (its register branch) instead of async to LDS 1.179 → 1.197 and 1.169 → 1.172; accuracy passed in every run
- E7 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2 sessions (the `.cg` A/Bs three runs in the second), 4 alternated rounds per arm, the ruler, one GPU] first session: DS256/m2 at 2 tokens, `.cg` → none on the weight loads 2.164 → 2.170 (18.98 → 19.05 µs); MM64-FP4/m128–m256 at 256 tokens `.cg` → none 1.180 → 1.156, all 4 pairs (102.93 → 104.65 µs), its unchanged 128-token size 1.183 → 1.185; DS512/m256 at 256 tokens `.cg` added on the 128-row gate-up tiles 1.043 → 1.033, all 4 pairs (225.81 → 227.67 µs); DS512/m8 at 8 tokens `enable_fp_fusion=False` → `True` on all 3 launches 1.150 → 1.150 (56.63 → 56.66 µs), accuracy passed; MM256-FP4/m1 at 1 token K fold 4 → 1 alone: every changed run crashed (exit −11), unchanged runs 3.50–3.59; MM256-FP4/m8192 at 8192 tokens gate-up `waves_per_eu` 3 → 0 1.442 → 1.443 (515.56 → 514.69 µs); second session, three runs: DS256/m2 without `.cg` 2.171 → 2.171 (19.30 → 19.32 µs, pairs both ways), 2.205 → 2.180 and 2.217 → 2.195 (×1.009 and ×1.010, every pair slower); MM64-FP4/m128–m256 without `.cg` 1.189 → 1.150, 1.181 → 1.145, 1.180 → 1.144 (×1.029–1.035); DS512/m256 with `.cg` 1.040 → 1.034, 1.039 → 1.033, 1.032 → 1.027 (×1.005–1.007), each slower in all 4 pairs; a toy kernel whose 64 KiB blocks are each read by two adjacent programs 39.44–39.95 µs unhinted against 41.35–41.74 µs with `.cg`, one read of each block 27.17–27.93 against 24.19–25.39 µs; DS512/m8 1.147 → 1.146 (B/A 0.9994), accuracy passed in all 8 runs, its 3 launches compiled for gfx950 with and without the switch giving the same code (633, 1554 and 440 instructions); MM256-FP4/m8192 1.441 → 1.440 (510.9 → 511.8 µs); the K-fold crash in 4 of 4 changed runs, the unchanged implementation 3.535–3.570
- E8 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, every Triton implementation run once on the ruler on its own Triton cache, each compiled kernel's `.amdgcn` parsed; DS256/m2's router also compiled at its host code's launch constexprs] DS256/m2's router kernel 48 `v_dot2c_f32_bf16`, no MFMA; Qwen2 router+experts f64 VALU ops (`v_fma`, `v_mul`, `v_add_f64`) 23 in Q2-RE/m1 (1 token), 38 in Q2-RE/m2 (2), 69 in Q2-RE/m4 (4), no `v_mfma_f64`; Q2-RE/m32 (32), its router and shared-gate kernel, 64 `v_mfma_f64_16x16x4_f64`; MM256-FP8/m1–m16's key-preparing router (5 variants, 1–16 tokens) no MFMA, MM256-FP8/m32–m64's router (32–64) `v_mfma_f32_16x16x4_f32`, MM256-FP4/m16384's router (16384) only `v_mfma_f32_32x32x2_f32`; the MFMA-bearing router kernels of MM64-FP4/m2048, MM64-FP4/m4096, MM64-FP4/m6144, MM64-FP4/m8192, MM64-FP4/m16384, MM256-FP8/m1024–m16384, MM64-FP8/m1024–m16384 `v_mfma_f32_16x16x32_bf16` and no fp32 MFMA, of MM256-FP4/m1024, MM256-FP4/m2048–m4096 `v_mfma_f32_16x16x4_f32`, of MM256-FP4/m6144, MM256-FP4/m8192, MM256-FP4/m16384 `v_mfma_f32_32x32x2_f32`; MM256-FP4/m2048–m4096's selector (both variants 4 warps) 41 `s_barrier`, 25 `ds_write*`, 25 `ds_read*`; MM256-FP4/m16384's selector (1 warp) and MM64-FP4/m4096's (4 warps) no `s_barrier`; the 14 DeepSeek-V2 and MiniMax-M2 MXFP4 implementations whose token counts are all ≤ 16 contain `v_mfma_scale_f32_16x16x128_f8f6f4` and no `v_mfma_scale_f32_32x32x64_f8f6f4`; `v_mfma_f32_16x16x32_fp8_fp8` in MM256-FP8/m1–m16's down+combine (12 or 24 per variant) and one of its W13 projections (24), MM256-FP8/m32–m64's split W13 projection (24), one MM64-FP8/m32–m64 down variant (48); the 65 MoE implementations' kernels hold no float atomic, their atomic instructions `global_atomic_add` 57, `ds_add_rtn_u32` 27, `buffer_atomic_add` 8, `ds_wrxchg_rtn_b32` 4, `buffer_atomic_or` 1 (the 65 MoE implementations)
- E9 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, compile only: `triton.compile` for gfx950 without a GPU, each size's constexprs and launch options; rechecked in later compiles] MM256-FP4/m1's gate-up at 1 token: K fold 4 compiles (160 VGPRs, 24 scaled MFMAs); K fold 1 with 4 warps and K tile 128 kills the compiling process with SIGSEGV (with 16 or 4 rows, transposed or not); K fold 1 compiles with 1 warp (K tile 128 or 256), with 4 warps and K tile 256, or without `llvm_fn_attrs` (29 VGPRs, 24 scaled MFMAs), and every crashing variant carried the `iterative-ilp` attribute MM256-FP4/m1 passes at m ≤ 2; MM256-FP4/m8192's gate-up at 8192 tokens 146 VGPRs with `waves_per_eu` 3 and 150 without, no spills (two compiles); `gl.atomic_add` gives one `global_atomic_add`, no buffer atomic; `gl.amd.cdna4.buffer_atomic_add` on uint32 or uint64 and `buffer_atomic_or`, `and`, `xor` on int32 or int64 compile, all 10 dtype mismatches raise at trace time, `buffer_atomic_xchg` on int32 or int64 traces and then fails in LLVM translation (no `llvm.amdgcn.raw.ptr.buffer.atomic.exch`); `gl.associative_scan` on the `SliceLayout` of MM256-FP4/m16384's scatter prefix (§4.3) fails to lower ("TODO: unsupported scan layout") and compiles after a `convert_layout` to a 1-D blocked layout, equal to `torch.cumsum`; DS512/m1–m2's expert kernel at 2 tokens compiles with `assert_trivial` on both operand conversions, row-major words raise; shuffled MXFP4 words become the [16, 16, 128] A operand with no instruction and no LDS, row-major ones take 8 `ds_write_b128`, 8 `ds_read_b128` and 8192 B of LDS; a `.wt` `gl.store` is followed by a wait after each of 4 stores, a `.wt` `buffer_store` by none
- E10 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3 (the toy MoE and the one-grid router N=4), one GPU: 1 GiB stream probes timed as 2^30 bytes over the median of 9 rounds of 10 launches; the ruler's timer (Kineto) and bare graph replay on test kernels and on DS256/m2; the ruler A/B, 4 alternated rounds per arm] a 1 GiB bf16 tensor read once per launch by 2048 programs of 4 warps, 16 bytes per lane: 6013, 6009, 6009 GB/s plain with `buffer_load` (5994, 6000, 5993 with `gl.load`), 6823, 6805, 6802 GB/s with `.cg` (6803, 6791, 6785), so 6.0 and 6.8 TB/s; DS256/m2 at 2 tokens: 4 launches (router, selection and gate-up, activation, down) in every trace, call 19.00, 19.17, 19.04 µs, 4.75, 4.79, 4.76 µs per kernel against a per-launch floor (slope over chains of 1–8 trivial kernels) of 4.041, 4.029, 4.066 µs, so the 4 floors are 84–85 % of the call; 4 chained trivial kernels 16.41, 16.44, 16.42 µs, one kernel doing their 4 updates 4.30, 4.30, 4.32 µs; a 1 GiB `.cg` stream over 256–8192 programs of 1, 2, 4 or 8 warps with 1–8 independent 16-byte loads per lane: configurations with equal bytes in flight within 1.104× of each other (the widest at 4 MB: 256 programs × 4 warps × 4 loads 5697–5756 GB/s, 512 × 4 × 2 6287–6333 GB/s), 256 × 4 × 8 over 2048 × 4 × 1 1.067–1.070; at 3 MB 384 × 4 × 2 4920–4962 GB/s, 768 × 2 × 2 4898–4947, 96 × 4 × 8 3945–3995 (0.80×), 192 × 1 × 16 2454–2467 (0.495–0.499×); Q2-R/m256 at 256 tokens, B/A speedup: split 4 → 2 (256 → 128 CTAs) 0.749, 0.748, 0.753 (9.93–10.03 → 13.26–13.32 µs), 4 → 8 (512 CTAs) 0.961, 0.967, 0.978, look-ahead 2 → 1 0.889, 0.881, 0.890; a toy decode MoE (32 experts, top-4, H 256, I 128, every gate-up program owning full K, no requantization), on the ruler's timer: top-k recomputed in the gate-up and down programs instead of a selection launch saves 3.07, 2.89, 3.22, 3.40 µs at 2 tokens and 3.44, 3.22, 3.41, 3.87 µs at 16 (replayed at 2 tokens 1.24–1.80 µs, call-time ratio 0.843–0.889), down and combine in one program per (token, 64-column tile) instead of a down and a combine launch 2.94, 2.97, 3.32, 3.23 µs at 2 tokens and 3.16, 3.18, 3.38, 3.73 µs at 16 (not replayed); a toy router GEMV (64 programs, 2 rows, K 2048) beside a 4-program elementwise phase in one grid: outputs equal, one grid over two launches 0.499, 0.495, 0.506, 0.498 on the ruler's timer, 0.835, 0.775, 0.755 and 1.154 (9.360 against 8.110 µs) replayed; a two-role test kernel compiles at 256 VGPRs and 32 B of LDS, its streaming role alone at 16 VGPRs and no LDS, its heavy role at 256 and 32 B (probe and toy kernels)
- E11 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1 (the implementations' call times N=4, the top-k probes N=3), one GPU: launch traces with torch.profiler, the ruler's timer (Kineto) and bare graph replay on test kernels, the ruler on the implementations and on copies with one change (A/B: 4 alternated rounds per arm)] launch traces of the 96 MoE sizes: launches per call as in §1; every size compiles on its first call, none on its second, and 136 of 471 launches carry no token-count constexpr; DS256/m2 at 2 tokens grids 304, 2306, 72, 768; DS256/m1 at 1 token router launch 216 = 192 + 24 programs, selection and gate-up launch 1153; all 32 DeepSeek and MiniMax sizes from 1024 tokens sort before the gate-up; none of 185 kernel names is a quantizer alone; routers of 512 programs in MM256-FP4/m128, MM256-FP4/m256, MM256-FP8/m256, MM64-FP8/m128–m256 and of one full-K program per (token block, expert) in MM256-FP8/m1–m16, MM256-FP4/m2, MM64-FP4/m8–m16 above 1 token; Q2-E/m64 at 64 grids of 3072 and 4096, MM64-FP8/m128–m256 at 256 a persistent W13 kernel on 48 × 96; the same grids on two routings and graph replay equal to eager at 82 sizes; split-K finish launches in DS256/m2, DS256/m1, DS512/m1–m2 at 1 token and Q2-E/m8; the baseline enqueues 8–11 device operations at 1–2 tokens, these implementations 3–4; docstring lists differ from the trace at 7 sizes in 5 of 11 implementations; one trivial kernel reads 4.21 µs on the ruler's timer, each added one 4.06 µs, and a bare graph replay pays 1.815–1.844 µs per extra kernel; call times of the unchanged implementations, mean of 2 measurements of the ruler's timer, 4 runs: DS256/m2 at 2 tokens 19.06–19.61 µs (4.76–4.90 µs per launch), DS512/m256 at 256 225.5–236.8 µs, MM64-FP4/m128–m256 at 256 103.6–105.4 µs, MM256-FP4/m8192 at 8192 505.7–518.6 µs, an earlier batch of runs 225.8, 102.9 and 515.6 µs for DS512/m256, MM64-FP4/m128–m256 and MM256-FP4/m8192; DS256/m2 at 2 tokens, gate-up K splits 4 → 2 (grid 2306 → 1154): 2.183 → 2.175, 19.20 → 19.28 µs, runs 2.167–2.195 against 2.157–2.191, paired 95 % confidence interval of the ratio 0.991–1.001; top-8 of 256 at 16384 tokens: one wave per token 12.96–13.18 µs with no `s_barrier`, four waves 37.68–38.32 µs with 31, the same ids, no scratch in either; the keyed min (retired scores at −inf) 0 wrong of 4000 rows; MM64-FP8/m1–m16 exact at 8 and 16 tokens; MM64-FP4/m8–m16 at 2 tokens keeps 8 through its id tie rule, at 8 tokens selects 9 on a tied 8th and 9th (16 tokens not run); DS256/m2's election (the form of §4.1) over 16384 programs 9.7–10.7 µs at slots 0–4, 11.3–11.4, 12.8–13.0 and 14.3–14.4 µs at slots 5–7, R² of a linear fit 0.724–0.774; with MM64-FP4/m4096's combine lanes disagree on 10000 of 10000 NaN rows and 0 of 10000 finite ones; MM256-FP8/m1–m16 and MM256-FP4/m2 give ±0 and every NaN payload one key each, Q2-RE/m1 gives 7 bf16 NaNs 7 keys; an MFMA-only fp16 loop with 4 useful rows: `instr_shape=[4, 64, 64]` 2432 µs, [16, 16, 32] 2810 µs; every finite E4M3FN code converts exactly to bf16; integer-scratch patterns over 100 calls: Q2-E/m64 at 64 tokens 9 and Q2-E/m128 at 128 4 in one run, 15 and 8 in another; 31 implementations with atomics, 20 runs each, one output each (the 96 MoE sizes; probe kernels)
- E12 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, 9 rotated rounds per dependency and size at m = 1, 2, 16, CUDA-graph replay and the ruler's timer (Kineto); 1000 correctness calls per size] toy Gluon kernels, 256 experts, H 6144: logits → top-k: two launches took 0.26–0.51× the replay time of a last-arriving program doing top-k and 0.15–0.48× a cooperative counter-barrier launch; counts → rows, a last arriver scanning 256 counts against two launches: at m = 1, 2 two launches 1.16× its replay time and 1.48–1.50× on the ruler's timer, at m = 16 0.89× and 1.28×; all 1000 calls bitwise equal per size; under `launch_cooperative_grid` a bounded counter barrier completed over 256, 512 and 1024 one-wave workgroups in all 1000 eager and replayed calls, at 2.1–6.8× the two-launch replay time; co-residency against a plain launch and the captured node's cooperative attribute not checked (toy kernels)
- E13 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, 16 rotated rounds per size, 10 replays per round, CUDA-graph replay and the ruler's timer (Kineto), paired confidence intervals with A/A controls] synthetic fp8 VALU MoE, 256 experts, top-8, H6144 with intermediate dimension 256 and H3072 with 512, m = 1, 2, 8, 16: routing recomputed in every gate-up, down and combine consumer slower on both timers; recomputed only in the gate-up, which publishes ids and weights: H6144 m = 1 and 8 saved 2.87 and 2.45 µs replayed, 2.06 and 2.44 µs on the ruler's timer (A/A confidence intervals including zero); SwiGLU and 1×128 requantization in a full-K gate-up epilogue saved 1.76–2.76 µs replayed and 3.05–3.73 µs on the ruler's timer against a full-K gate-up with a separate activation launch, but lost to the unfused K-split design; down and combine fused in slot order: replay saving −8.18 to −0.25 µs on average (a loss); all fusions together 1.0841–2.2992× the K-split design's replay time; H6144/m16: one fp8 code flipped, relative L2 1.122e-5 against the reference; some A/A controls of the K-split design showed replay bias (toy kernels)
- E14 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1 (Q2-R/m256's half/double N=8 and one-tile look-ahead N=6, with the runs of E6 and E10), the ruler, 4 alternated rounds per arm, one GPU] Qwen2 routers against half and double splits: Q2-R/m256 at 256 tokens half 1.33–1.34×, double 1.02–1.04× the unchanged call time; Q2-R/m128 at 128 1.22× and 1.03×; Q2-R/m64 at 64 1.04× and 0.99×; Q2-R/m256 with 128 CTAs four K tiles ahead 1.26×, 512 CTAs one ahead 1.03×, 256 CTAs one ahead 1.12–1.14×, four ahead 0.97×; MoE gate-up splits at 1–2 tokens: DS256/m2 at 2 tokens 4 → 1 (2306 → 578 one-wave programs) 1.079×, 4 → 8 (4610) 1.050×; DS256/m1 at 1 token 1153 → 577 programs 1.020×, → 2305 1.028×; DS512/m1–m2 1728 → 864 1.035×; MM256-FP8/m1–m16 column tile 8 → 16 (385 → 193 four-wave programs) 1.031×; every arm accurate
- E15 [2026-11, CPU (no GPU), Triton 3.8.0, N=1, source and AST census of the host code; fp8 codes converted in torch] 89 MoE sizes of 58 implementations: r = m·k/E computed from each one's token count, top-k and routed-expert count, gate-up launches and grids traced from the host code, job forms not classified; Q2-E/m8's MFMA operands cast to bf16 in its gate-up and down kernels; all 256 e4m3 codes: the 254 finite values, both NaNs and signed zero convert to bf16 exactly; DS256/m4096–m4192's aliased buffers written, read and overwritten in launch order
- E16 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, the ruler, 4 alternated rounds per arm, 12 sizes of seven implementations, A/A controls at four sizes; reuse classified from the schedule and the measured routing, not from cache counters] `.cg` on the weight loads against none: read once at MM64-FP8/m1–m16 1, 4, 8, 16 tokens, DS256/m1 1, MM256-FP8/m1–m16 1, 2, MM64-FP4/m128–m256 128, 256 and DS256/m1024–m3072 2048, re-read at DS512/m256 256 and MM256-FP4/m16384 16384; `.cg` faster in every pair in call time and in speedup at six read-once sizes (MM64-FP8/m1–m16 at 4, 8, 16, MM64-FP4/m128–m256 at 128, 256, DS256/m1024–m3072 at 2048; 1.6–9.6%), MM256-FP8/m1–m16 at 2 tokens in aggregate only, no consistent effect at the three 1-token sizes; without `.cg` faster in every pair at both re-read sizes (DS512/m256 1.012 → 1.009, 0.2–0.5%; MM256-FP4/m16384 17%); A/A offsets within 0.37%; every arm accurate
- E17 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1 per size over eight sessions, graph-replay events after a 1 GiB flush and a 90 s ramp, 32 balanced AB/BA pairs per size, A/A controls; register-MFMA probes, grid 1536, 3072 iterations] synthetic router logits at K 3072, (M, E) = (1024, 64), (4096, 64), (16384, 64), (1024, 256), (4096, 256), (8192, 256), each arm's tiling from a bounded search: the three-bf16-term kernel took 0.457–0.512× the time of the two fp32-MFMA kernels (random input, seed 1201, W split in each call; sigmoid, bias, top-8 and the expert GEMMs excluded); 18 A/A controls within ±log(1.02); against the exact-id check on the fp32 top-8 over 8 sizes × 8 seeds × 3 input modes × 3 arms, 217 of 576 comparisons failed, all 192 near-tie inputs among them; at M16384/E64, random seed 705, both fp32 arms passed and the split failed with lower logit error (max abs 4.165e-6 against 1.149e-5, against fp64); M8192/E64 and M16384/E256 had no correct split candidate on seed 1201 and were not timed; probes: estimated matrix-core time of the fp32 router over the MXFP4 expert GEMMs 1.244–1.263, of the three-bf16 split 0.244–0.246 (K 3072, 256 experts, top-8, intermediate dimension 512) (toy kernels)
- E18 [2026-11, gfx950 (MI355X), Triton not recorded (MI355X image), N=1, the ruler with its accuracy check (the unchanged implementations twice), one input set per size, copies of the implementations with one change; the selection copies also run directly] Q2-R/m256 at 256 tokens with token 0's first two ids swapped, everything else bitwise equal: a mismatch on `topk_ids` (violation fraction 2/2560), the unchanged implementation passed twice; DS512/m8 at 8 tokens ranking the unrounded logits: passed twice, no token's experts changed, error change unresolved; a gated-attention kernel at 32 rows with the attention kept in fp32 before the gate: passed twice, its scales' relative L2 against the float64 reference without that rounding 0.0032334 → 0.0032690; DS256/m1 at 1 token, the gate-up copy without the bf16 rounding of the summed logits: 6 of 16 tokens took another expert, the accuracy check failed; with the higher id winning ties: 4 tokens, the accuracy check passed
- E19 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3, the ruler twice per run; routing simulated from the workload's router distribution and bias with random normal hidden states] MiniMax MXFP4, 256 experts, at 64 tokens: experts touched per call 77.19% on average, 74.61% at the least (uniform-routing estimate 86.9%), 495.3 MB of expert weights; call 99.09–101.7 µs against the baseline's 127.0–127.8 µs (speedup 1.249–1.285), so 4.87–5.00 TB/s against 3.88–3.90 TB/s for those bytes
- E20 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, the router prefix (logits) of 13 MiniMax router branches at legal sizes run on the GPU, a random and a cancellation input each (seed 0), plus two inputs with a nonzero fp64 residual; error against fp64 on the same input, compared with an fp32 GPU computation] the routers' logit error no larger than fp32's in L2 on 25 of the 26 inputs of the 13 branches (the exception MM256-FP4/m6144's cancellation input, 1.0000934×) and in max abs on 23 of 26 (the exceptions the cancellation inputs of MM256-FP4/m2048–m4096, MM256-FP4/m6144 and MM256-FP4/m8192); on the two residual inputs the conditional fp64 correction of MM64-FP4/m6144 and MM64-FP4/m8192 ran, logits of 2^-23 equal to fp64, the routers' and the fp32 computation's errors both 0
- E21 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, copies of DS256/m1's selection and expert code run directly at 1 token on natural and saved inputs, frozen reference ids and slot weights] four saved inputs with near or exact ties: replacing only slot 7's expert (by an unselected one) in both gate-up and down moved `out` 6.21, 112.2, 6.13 and 9.47 times as far in L2 as removing only the bf16 rounding of the activation product, total errors against the reference 1.22, 3.03, 1.23 and 1.28×; changing only the gate-up copy's bf16 rounding or tie rule: producer and consumer disagreed on a slot's expert on 12 of 13 natural inputs, both sides changed alike: agreement on 36 of 36; the 12 split partials summed in reverse order on four saved inputs: ids, weights and outputs unchanged
- E22 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, 100 synchronized calls on one fixed input; bodies forced at 15, 16, 17, 31, 32 and 33 live rows] DS512/m256 at 256 tokens: the observed tile ran its 16-row body in all 100 calls with one route-partial hash; forcing the body sizes changed no bit of its up-projection codes, scales or down partials
- E23 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3 sessions × 10 rounds, the ruler's timer (rocprofiler-sdk), 1 GiB read before each call, an unchanged copy A2/A −0.04 to +0.11%] time of a copy with one source line changed over the unchanged implementation: `.cg` removed from DS256/m2's weight loads at 2 tokens +0.20%; removed from MM64-FP4/m128–m256's at 256 tokens +3.76% (sessions +3.59 to +4.03%, slower in 30 of 30 rounds); added to DS512/m256's 128-row gate-up tiles at 256 tokens +0.29%; outputs bitwise equal
- E24 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2 runs × 20 rounds, CUDA events around one call after a 1 GiB read, arm order rotated] a read-only Gluon kernel, each program summing one 32 KiB block, `.cg` (all loads `nt`) over none, time: every block read once 0.892–0.945; re-read by the adjacent program id (another XCD) 1.048–1.065; by the id 8 apart (the same XCD) 0.973–1.021 (test kernels)
- E25 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=14 runs per implementation, the ruler (rocprofiler-sdk), each run a new process, the implementation against itself] speedup: Q2-R/m256 at 256 tokens 1.972–2.008, (max − min)/mean 1.81 %; DS256/m2 at 2 tokens 2.169–2.236, 3.05 %, its time 18.9–19.3 µs and the baseline's 41.8–42.5 µs
- E26 [2026-11, gfx950 (MI355X), HIP 7.15, N=1 run, 9 rounds × 5 launches per arm, hipEvent after a 512 MiB flush read, arm order rotated] HIP stream kernels compiled alone (occupancy 8) and with a never-taken branch writing v255 (256 VGPRs, occupancy 2), time at 256 VGPRs over alone: `y = 2x + 1` over 512 MiB 0.972 at 32 B and 0.962 at 64 B per thread, 1.461 at 16 B; 1.089 at 32 B over 128 MiB; a read-only sum at 32 B 1.256 (test kernels)
- E27 [2026-11, gfx950 (MI355X), HIP 7.15, N=1 run, 9 rounds × 5 launches per arm, arms alternated, hipEvent; clock estimated from `s_memtime` against the 100 MHz wall clock] MFMA-only loops with the same useful MACs, `v_mfma_f32_16x16x32_f16` on rows padded to 16 over `v_mfma_f32_4x4x4_16b_f16`, time: random padding rows 1.082–1.092 (4 chains, or one chain at 2 waves per SIMD), the 4-row at about 2140–2170 MHz against 1910–2000 MHz; zero padding rows 0.906; one chain at 1 wave per SIMD 0.788 (test kernels)
- S23 triton-lang/triton@c01b677:third_party/amd/backend/compiler.py:327 `gluon_to_ttgir`: no buffer-op conversion and no pipeliner (to :348; `make_ttgir` converts at :306); `launch_cooperative_grid: bool = False` (:95), launched with `hipModuleLaunchCooperativeKernel` (third_party/amd/backend/driver.c:674-677); triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/LoadStoreOpToLLVM.cpp:2278 the redundant-thread predicate of an atomic, ANDed into its mask at :2338; `acq_rel`, the default of `atomic_add` (python/triton/language/core.py:2731), makes the backend insert buffer invalidates and L2 write-backs (:2354)
- S24 triton-lang/triton@c01b677:python/triton/experimental/gluon/language/_semantic.py:245 `convert_layout(..., assert_trivial=True)` raises unless the conversion is trivial; triton-lang/triton@c01b677:python/triton/experimental/gluon/language/_core.py:703 `gl.barrier()` synchronizes the threads of one CTA; a cluster only with `cluster=True` and `num_ctas > 1` (:712-714), which the HIP backend rejects on gfx950 (third_party/amd/backend/compiler.py:162-163)
