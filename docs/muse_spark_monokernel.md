# Case study: a 1.25-1.44x faster whole-layer monokernel for Muse Spark on MI355X

[中文](muse_spark_monokernel.zh.md)

Muse Spark's decode on MI355X runs each target layer as one kernel launch, a monokernel from [sgl-amd-mono](https://github.com/sgl-project/sgl-amd-mono). About 70% of decode device time goes to that launch. An AI agent campaign made it **1.25-1.44x faster with every output bit-identical**. In end-to-end serving, p90 interactivity rose **16-27%**. The change is [sgl-amd-mono#135](https://github.com/sgl-project/sgl-amd-mono/pull/135).

This page describes what limited the kernel, what changed, how correctness was kept, and which lessons carry over to other monokernels.

## The kernel

The kernel is `muse_spark_layer_s8/s16`, in `sgl_amd_mono/muse_spark/gfx950/`. It is a persistent FlyDSL kernel: 256 CTAs × 512 threads, one per CU, all resident for the whole launch. It runs one rank of TP8 / EP8 for the 816B model. A verify step makes 60 launches, one per layer, at 8 or 16 rows (1 or 2 requests × 8 draft tokens).

One launch does the whole layer:

1. attention over a paged fp8 KV cache;
2. o-norm, gate and o_proj, then the TP all-reduce in the kernel, over HIP IPC peer buffers and mailbox flags;
3. post-attention norm, gated residual and pre-FFN norm;
4. the FP32 router with top-8 selection, and the pooler that projects width 8192 to the experts' 4096 (MXFP4 activation quantization);
5. gate-up, SiLU and down over 32 local MXFP4 experts (EP8), then the expert norm, the weighted finalize and the cross-rank reduce;
6. unpool, FFN norm and gated residual;
7. the next layer's QKV, gate, QK-norm, RoPE and KV-cache write.

Stages hand data to each other through flags in memory: one CTA stores, another polls. Some hand-offs cross ranks.

## Why it was slow: latency, not bandwidth

A byte floor gives a first estimate. Reading each weight once at about 5.6 TB/s costs:

| Weight | Bytes |
|---|---|
| o_proj | 16 MB |
| pooler | 64 MB |
| unpool | 64 MB |
| next layer's QKV | about 38 MB |
| touched MXFP4 experts, 8 rows | about 3-6 experts × 25 MB |

That is about 45-60 µs at 8 rows. The launch took 129 µs.

A per-stage trace confirmed the gap. The kernel can be built with `trace=True`, which records a timestamp per CTA per stage. At 8 rows on TP8:

- HBM averaged about 2.4 TB/s over the launch.
- Only the expert window ran near the HBM roofline.
- The rest was a serial chain: store→poll hops across CTAs and ranks, mailbox hand-offs, and weight loads issued only after a critical poll returned.

So the work was to shorten the chain, not to move fewer bytes.

## What changed

Each item below is a technique, with the part of the layer it was applied to.

**Fewer bytes on the link per all-reduce (o_proj).**
- The one-shot all-reduce became an owner-sliced reduce-scatter followed by an all-gather. This halves the bytes each peer link carries.
- Each rank sums its partials locally in fp32, in part order, before pushing. The additions happen in the same order as before, so results stay bit-exact.
- Peer order is rotated per rank, so all ranks do not hit the same peer at once.

**Work moved off the critical chain.**
- Router weights are loaded before the poll for x, not after it.
- The route runs on CTAs that stream weights, not on the CTAs on the dependency chain.
- The expert finalize, the cross-rank reduce and the unpool hand-offs moved off the path of the most-loaded rank.
- The tail CTAs of the expert stage prefetch the next layer's QKV weights.

**Cheaper hand-offs.**
- Each CTA polls its own copy of the x flags (`X_FLAG_COPIES`), so many CTAs no longer poll one word.
- Gathers use sent-word mailboxes (`poll_sent`, `sent_sum`, `rearm_sent`).
- A CTA now polls before it issues its own remote pushes. Loads and stores share one in-order `vmcnt` counter on gfx950, so a poll issued after the pushes also waits for the pushes' acknowledgements.

**Attention merge.**
- One merge level covers up to 64 partitions. The fold order is unchanged, so results stay bit-exact.
- The 4-lane fold uses a DPP quad instead of a round trip through LDS.
- The partial blocks are laid out so that merge waves load contiguous 256 B and 1 KB lines.

**Expert weight loads.**
- MXFP4 weights are loaded in full lines with non-temporal hints, then unswizzled in registers.
- Gate-up items are assigned round-robin, which improves DRAM locality.
- Down items are batched, two 64 KB items per barrier.
- LDS bank conflicts were removed in the K1 row sums and in the unpool MFMA B reads.

## How correctness was kept

The campaign ran under fixed rules, enforced by an acceptance script that the agents could not edit:

- **One launch per call.** No host sync, safe under CUDA-graph capture and replay (flags carry an epoch, so a flag from an earlier replay never matches), and no compile on replay. The launch arguments did not change, so the serving code needed no change.
- **Numerics.** Every output had to stay within a tolerance of the original kernel on the same inputs, and routing ids had to match exactly. No new approximations were allowed: same MXFP4 activation quantization points, bf16 rounding points and RMS eps.
- **Real conditions.** Tests ran on 1 GPU and on TP8 with 8 ranks and a real peer all-reduce. The benchmark chained 8 distinct layers so that weights were cold in L2 and MALL, as in serving.
- **A hang is a failure.** Every wait has a deadline. A wait that times out sets an error word, and the run fails.

The delivered kernel is stricter than the tolerance: all 60 outputs per configuration match main bit for bit. That holds for ctx 1k, 131k and 197k, at 8 and 16 rows, under both the default configuration and `SGLANG_AMD_MONO_MUSE_TPX/PSHARD/UG_DYN=0`. Because the outputs are identical, model accuracy is unchanged by construction. As a serving check, GSM8K scored 97.8% ± 0.4 (all 1319 questions) with the new kernel.

## How the campaign ran

- **Who did the work.** The optimization was done by Claude Code agents driven by GEAK's kernel workflow. Engineer agents proposed changes in rounds, and a change was kept only if it passed the full acceptance run.
- **Where and how long.** It ran for about 48 hours on one 8-GPU MI355X node. Single-GPU work shared the GPUs through locks; TP8 runs took all 8.
- **What counted as a gain.** The primary metric was a paired, interleaved A/B against a frozen copy of the original kernel. Its A/A noise is about ±0.5%, so only gains that held up on a re-run were kept.
- **Restarts.** The node expired several times. Each new attempt resumed from the committed winners, and read the earlier attempts' notes so that it would not retry dead ends.

The same campaign also worked on the draft-model layer. The PR covers the target layer only.

## Results

### Kernel

TP8, 8 cold MXFP4 layers chained. Each value is µs per launch, the median of 7 × 20 graph replays.

| rows / ctx | main | this PR | speed-up |
|---|---:|---:|---:|
| 8 / 1024 | 108.9 | 78.7 | 1.38x |
| 16 / 1024 | 156.4 | 108.5 | 1.44x |
| 8 / 131072 | 124.1 | 97.3 | 1.28x |
| 16 / 131072 | 183.7 | 138.6 | 1.33x |
| 8 / 196608 | 129.6 | 103.3 | 1.25x |

These are the default configuration. With `TPX/PSHARD/UG_DYN=0` the speed-up is 1.35-1.50x.

### End to end

The workload was InferenceX AgentX, TP8 / EP8, DFlash2 speculative decoding at a simulated acceptance length of 4.05, for 60 minutes. Each run was identical to an InferenceX CI run (CI run 42879019953) except for the kernel. p90 interactivity is the slow end, about 1000 / ITL p90.

| | c6 CI | c6 new kernel | c10 CI | c10 new kernel |
|---|---:|---:|---:|---:|
| p90 interactivity (tok/s/user) | 500.4 | 579.4 (+15.8%) | 373.1 | 472.7 (+26.7%) |
| p50 interactivity | 558.2 | 657.7 (+17.8%) | 429.6 | 564.8 (+31.5%) |
| total throughput per GPU (tok/s) | 5,275 | 5,654 (+7.2%) | 8,774 | 9,087 (+3.6%) |

The c6 run used `TPX/PSHARD/UG_DYN=0`; the c10 run used the default configuration.

## Lessons that carry over

- **Do a byte floor before tuning.** If the launch takes twice the floor or more, the time is in dependencies and synchronization, and faster loads alone will not close the gap.
- **Trace each stage.** A per-CTA, per-stage timestamp trace shows which waits are on the critical path. Reasoning from the code alone does not.
- **Order polls before pushes on gfx950.** Loads and stores retire through one in-order `vmcnt`, so a poll issued after a CTA's remote stores also waits for those stores' acks.
- **Issue weight loads before the poll that guards the data they combine with.** The weights do not depend on the poll.
- **Change the reduction shape, keep the summation order.** Reduce-scatter plus all-gather, with local fp32 partial sums in part order, halves link bytes and stays bit-exact. Bit-exactness makes the change easy to verify and review.
- **Freeze the judge.** A fixed acceptance script with paired A/B and a known noise floor let the agents work for two days without drifting into gains that were really noise.
