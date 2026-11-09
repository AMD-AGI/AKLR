# Direction survey, iteration 6

## Task

Survey: list the directions this kernel should try, from the Analysis evidence and the skill's lever taxonomy. Cover the taxonomy broadly (grid and split-K, main loop, data path and LDS, scheduling and locality, epilogue, tuning, and whatever else the evidence names) rather than many variants of one idea. Do not repeat a direction the ledger already holds.

Campaign state (JSON):
{
  "mode": "survey",
  "ledger": [
    {
      "id": "replace-the-torch-path-with-a-fused-flyd",
      "title": "Replace the torch path with a fused FlyDSL a4w4 MoE: route/sort, x\u2192MXFP4 quant, stage 1 (gate+up, clamped SwiGLU, MXFP4 requant), stage 2 (down \u00d7 routing weight), using native scaled fp4 MFMA on the 16x16-preshuffled weights",
      "category": "structure / fusion",
      "status": "KEPT",
      "attempts": 2,
      "keeps": 2,
      "attempts_since_keep": 0,
      "gain_pct": 6718.24,
      "expected_gain_pct": 5000.0,
      "expected_gain_reason": "Report \u00a75: 26\u201390\u00d7 headroom against the candidate. A structurally correct fused kernel takes the score from ~0.016 to ~0.8\u20131.1. Native fp4 MFMA is needed to match or beat aiter at 3112\u201332768, where the bf16-dequant practical floor is at or above the baseline.",
      "hypothesis": "The 12.5 ms all-expert bf16 dequant, the ~20-pass torch _mxfp4 and the glue (slab gathers, fp32 combine matmul) make up 95%+ of the 16\u201363 ms. A 3\u20134 launch FlyDSL pipeline that reads the shuffled fp4 weights straight into mfma_scale_f32_16x16x128_f8f6f4 operands should reach about the baseline's 0.18\u20132.5 ms. It must reproduce the reference rounding exactly: x rounded to MXFP4 per 1\u00d732, E8M0 = floor(log2 amax) \u2212 2 (+1 if mantissa > 0.75), round-to-nearest-even on e2m1, and y rounded to bf16 then MXFP4. Port aiter's mxfp4_gemm1/mxfp4_gemm2/moe_sorting as the starting point.",
      "retired": ""
    },
    {
      "id": "stepping-stone-replace-unpack-s-6-pass-d",
      "title": "Stepping stone: replace _unpack's 6-pass dequant chain and the ~20-pass torch _mxfp4 with one vectorized FlyDSL kernel each, keeping the rest of the torch path",
      "category": "fusion (stepping stone / hedge)",
      "status": "OPEN",
      "attempts": 0,
      "keeps": 0,
      "attempts_since_keep": 0,
      "gain_pct": 0.0,
      "expected_gain_pct": 200.0,
      "expected_gain_reason": "Report \u00a76.5: each piece is a 3\u20135\u00d7 gain at most. It does not reach 0.1 speedup but is a quick, low-risk landing if the fused kernel slips.",
      "hypothesis": "One FlyDSL dequant kernel at \u22654 TB/s instead of six torch passes at 1\u20132 TB/s saves about 11 ms per call. A single-pass fused MXFP4 quant kernel with exact reference rounding saves 1\u201324 ms. Together they cut 16\u2192~5 ms at n=256 and 63\u2192~25 ms at 32768. The quant kernel is reusable as stage (b) of direction 1, and its bit-exact rounding test de-risks direction 1.",
      "retired": ""
    },
    {
      "id": "per-regime-tile-m-equal-to-the-sort-unit",
      "title": "Per-regime tile-M equal to the sort unit: BM=32 at n\u2264640, BM=128 at 3112/3816, BM=128\u2013256 at \u226510000",
      "category": "grid / tile-M",
      "status": "OPEN",
      "attempts": 0,
      "keeps": 0,
      "attempts_since_keep": 0,
      "gain_pct": 0.0,
      "expected_gain_pct": 6.0,
      "expected_gain_reason": "Neutral on S (aiter is already at BM=32). Large on the 5 M/L Points, where halving weight sweeps against a 2\u20136.6\u00d7-off-roof baseline could give 1.2\u20132\u00d7 on each. That is ~5\u20138% of the mean score.",
      "hypothesis": "Weight traffic is M-tiles \u00d7 weight slab. At n\u2264640, BM=32 gives exactly one 0.96 GB sweep (\u226431 routes per expert), while BM=16 sweeps up to 1.68 GB. At 3816, BM=128 sweeps 1.15 GB against 2.11 GB at BM=64, and at 32768 8.2 GB against 15.9 GB. aiter runs untuned heuristic BM (t32/t128/t64) everywhere, notably BM=64 at 32768.",
      "retired": ""
    },
    {
      "id": "compute-leaning-main-loop-for-large-n-ga",
      "title": "Compute-leaning main loop for large n: gathered A rows through an XOR-swizzled multi-stage LDS ring prefetched two K-steps ahead, B straight to VGPR, 32x32x64 vs 16x16x128 scaled MFMA, s_setprio around the MFMA block",
      "category": "main loop / LDS",
      "status": "OPEN",
      "attempts": 0,
      "keeps": 0,
      "attempts_since_keep": 0,
      "gain_pct": 0.0,
      "expected_gain_pct": 5.0,
      "expected_gain_reason": "L Points have 2.6\u20133.4\u00d7 paper headroom (report \u00a75). Even 1.3\u00d7 on 3\u20135 Points is ~4\u20136% of the mean.",
      "hypothesis": "At 32768 aiter is at 1.5 PF/s against ~4 PF practical MXFP4. A loop that hides the gathered-A latency and the B stream behind back-to-back scaled MFMAs (32x32 atom halves the A/B register traffic per FLOP) should lift stage throughput toward 2\u20133 PF/s. The delivered K3 MoE ran stage 1 at 1.38\u00d7 aiter's TF/s.",
      "retired": ""
    },
    {
      "id": "xcd-aware-tile-order-plus-group-m-keep-a",
      "title": "XCD-aware tile order plus GROUP_M: keep an expert's row tiles and an M-block's N tiles on one XCD, several row tiles of one expert per column sweep",
      "category": "scheduling / locality",
      "status": "OPEN",
      "attempts": 0,
      "keeps": 0,
      "attempts_since_keep": 0,
      "gain_pct": 0.0,
      "expected_gain_pct": 3.0,
      "expected_gain_reason": "Field result: 511\u2192475 \u00b5s (\u22127%) on a fused-MoE stage 1 from the XCD placement. Applies mainly to 5 M/L Points.",
      "hypothesis": "At 10000\u201332768 each expert has 272\u20131110 routes (3\u20139 BM=128 tiles), so expert slabs are re-read through L2. Dealing M-blocks round-robin over XCDs with their N tiles together, and grouping 2\u20139 row tiles per expert as the 33-expert kernel does at m\u2265512, turns repeats into L2 hits. Exit on the raw id before the remap.",
      "retired": ""
    },
    {
      "id": "stage-2-combine-race-bf16x2-coalesced-at",
      "title": "Stage-2 combine: race bf16x2 coalesced atomics into out against fixed-order per-route bf16/fp32 rows plus a separate reduce launch, per regime",
      "category": "epilogue / stage 2 reduction",
      "status": "OPEN",
      "attempts": 0,
      "keeps": 0,
      "attempts_since_keep": 0,
      "gain_pct": 0.0,
      "expected_gain_pct": 3.0,
      "expected_gain_reason": "Field numbers show a 10\u201340% swing in stage-2 cost depending on routes per token. Stage 2 is about a third of the call.",
      "hypothesis": "With topk=4 every route is local. With 16 local routes the atomic stage 2 was 31\u201342% slower than split+reduce, but with about 2 routes it was faster. topk=4 sits between, and the answer likely flips between S (a launch matters at ~180 \u00b5s) and L (contention on 32768\u00d76144 outputs). Rows must stay bf16/fp32 because of the per-element BaselineRelativeBound, so no fp8 or MXFP4 rows.",
      "retired": ""
    },
    {
      "id": "cut-graph-nodes-fold-x-quantization-and-",
      "title": "Cut graph nodes: fold x quantization and the output zeroing into the sort launch (or the sort into stage 1), preload kernargs, read the expert id before the A loads",
      "category": "launch overhead / small kernels",
      "status": "OPEN",
      "attempts": 0,
      "keeps": 0,
      "attempts_since_keep": 0,
      "gain_pct": 0.0,
      "expected_gain_pct": 3.0,
      "expected_gain_reason": "On 20 S Points with \u22641.16\u20131.24\u00d7 headroom, saving 1\u20132 nodes (4\u20138 \u00b5s of 180\u2013190 \u00b5s) is 2\u20134%, which may decide serving at 1.03.",
      "hypothesis": "Each graph node costs ~4 \u00b5s under AKLR's timer, about 2% of a ~180 \u00b5s S-regime call. aiter runs sort + quant + stage1 + stage2. Folding per-token quant and zeroing into the sort launch removed 6.9\u20138.0 \u00b5s in the field, and merging sort phases gained 1.1%. The route kernel must scale past R=1024 (multi-CTA or looping), unlike the prior art.",
      "retired": ""
    },
    {
      "id": "offline-raced-per-shape-configuration-ta",
      "title": "Offline-raced per-shape configuration table (BM, BN, waves, cache policy, atomic vs reduce, MFMA atom) chosen from n_tokens inside the functools.cache compile function",
      "category": "tuning",
      "status": "OPEN",
      "attempts": 0,
      "keeps": 0,
      "attempts_since_keep": 0,
      "gain_pct": 0.0,
      "expected_gain_pct": 3.0,
      "expected_gain_reason": "Tune last. Per-Point picks typically add 2\u20135% over one config per regime, and it guards against wins on some Points losing on others (a lost workload rejects the candidate).",
      "hypothesis": "aiter runs untuned heuristic configs at every Point. The best tile, cache and reduce choice flips between S, M and L, and across the 256\u2013640 band as the >16-route expert count grows 0\u219296. A table raced offline per Point, with no timing in run, captures the per-Point optimum without a shape-divergence loss.",
      "retired": ""
    },
    {
      "id": "s-regime-grid-shape-race-tile-n-downward",
      "title": "S-regime grid shape: race tile-N downward (256\u2192128\u219264) in both stages so 128 experts \u00d7 (N/BN) blocks fill \u2265256 CUs with enough bytes in flight",
      "category": "grid / N-split",
      "status": "OPEN",
      "attempts": 0,
      "keeps": 0,
      "attempts_since_keep": 0,
      "gain_pct": 0.0,
      "expected_gain_pct": 2.0,
      "expected_gain_reason": "The S regime has at most 1.16\u20131.24\u00d7 to the hard roof. Grid fill is one of the few levers left on 20 of 25 Points.",
      "hypothesis": "At n\u2264640 every expert has about one 32-row block, so the N blocks are the parallelism. Narrower tile-N gave +0.8\u20131.0% per halving on an a4w4 atomic stage 2. Count workgroups against the 256-CU dispatch round to avoid a mostly empty last round.",
      "retired": ""
    },
    {
      "id": "weight-load-cache-policy-per-regime-and-",
      "title": "Weight-load cache policy per regime and per M-block: non-temporal at n\u2264640, cached when a sorted neighbour block has the same expert, scale loads always cached",
      "category": "data path",
      "status": "KEPT",
      "attempts": 1,
      "keeps": 1,
      "attempts_since_keep": 0,
      "gain_pct": 1.87,
      "expected_gain_pct": 2.0,
      "expected_gain_reason": "Field numbers: +1.8% per-block selection, a 6\u201326% loss if set wrong at multi-block. This is a correctness-free knob with a clear regime flip.",
      "hypothesis": "At S nothing reuses weights, so non-temporal streaming frees L2 and MALL for activations. At M/L (3\u20139 blocks per expert), non-temporal W1/W2 cost 26% and 6\u20139% in the field, so cached loads with same-expert neighbours on one XCD are better. Per-block selection gained 1.8% on an a4w4 decode score.",
      "retired": ""
    },
    {
      "id": "stage-1-fused-epilogue-clamp-swiglu-in-f",
      "title": "Stage-1 fused epilogue: clamp/SwiGLU in fp32, bf16 round, MXFP4 quantize with shuffled E8M0 in sorted-row order, staged through LDS for 16-byte stores",
      "category": "epilogue",
      "status": "OPEN",
      "attempts": 0,
      "keeps": 0,
      "attempts_since_keep": 0,
      "gain_pct": 0.0,
      "expected_gain_pct": 1.5,
      "expected_gain_reason": "The y traffic is small next to the weights, but at L (R=131k routes) store efficiency and the per-lane quant cost become visible.",
      "hypothesis": "Writing y as fp4 plus scales (\u2248R\u00b7408 B), produced in registers and stored with wide coalesced stores, keeps the intermediate off the critical path. Writing it in the exact layout stage 2's A DMA consumes removes any relayout. Poorly vectorized stores of y cost noticeably at large R.",
      "retired": ""
    },
    {
      "id": "persistent-or-capped-grids-at-large-n-la",
      "title": "Persistent or capped grids at large n: launch min(work, k\u00b7NUM_CU) workgroups that stride over the device-side tile bound instead of the padded upper-bound grid",
      "category": "grid / persistence",
      "status": "OPEN",
      "attempts": 0,
      "keeps": 0,
      "attempts_since_keep": 0,
      "gain_pct": 0.0,
      "expected_gain_pct": 1.5,
      "expected_gain_reason": "A field gain of 3\u20135% on the stage it applies to, restricted to the 3\u20135 large Points.",
      "hypothesis": "The padded upper bound (routes + experts\u00b7(BM\u22121))/BM \u00d7 N blocks over-launches many empty workgroups at 10000\u201332768. A persistent loop with a per-tile barrier (or C staged per wave) avoids dispatch waste and tail rounds, and beat one tile per workgroup by 3\u20135% on a dense GEMM.",
      "retired": ""
    },
    {
      "id": "skip-padding-work-send-padding-rows-a-lo",
      "title": "Skip padding work: send padding rows' A loads out of range (buffer bound returns zero) and skip MFMAs on all-padding sub-tiles",
      "category": "data path / main loop",
      "status": "OPEN",
      "attempts": 0,
      "keeps": 0,
      "attempts_since_keep": 0,
      "gain_pct": 0.0,
      "expected_gain_pct": 1.0,
      "expected_gain_reason": "Small field gains of 1\u20133% per stage, but cheap and helpful on the near-roof S Points.",
      "hypothesis": "With BM=32 at n\u2264640, experts have 2\u201331 routes, so a large fraction of tile rows are padding. Out-of-range A loads for padding rows gained ~1% and 1.7 \u00b5s on a stage 2. Skipping MFMAs on padding rows gained 1.6% on a stage 1.",
      "retired": ""
    },
    {
      "id": "parallel-routing-sort-for-n-1024-first-t",
      "title": "parallel routing sort for n > 1024 (first), then per-regime-tile-m-equal-to-the-sort-unit (a time-boxed race)",
      "category": "dispatch / routing sort",
      "status": "KEPT",
      "attempts": 1,
      "keeps": 1,
      "attempts_since_keep": 0,
      "gain_pct": 6.05,
      "expected_gain_pct": null,
      "expected_gain_reason": "",
      "hypothesis": "",
      "retired": ""
    }
  ],
  "progress": {
    "window_hours": 3.26,
    "gain_pct": 7266.04,
    "flat": false
  },
  "analysis_reports": [
    "/home/ruibzhan/aklr-moe-m31-20261105/run-large/workspace/forge_experiments/analysis/5bfbea64c0d72d191756b25f14d171e78e120703/generation-001",
    "/home/ruibzhan/aklr-moe-m31-20261105/run-large/workspace/forge_experiments/analysis/5bfbea64c0d72d191756b25f14d171e78e120703/generation-001/report.md"
  ],
  "optimization_history": "/home/ruibzhan/aklr-moe-m31-20261105/run-large/workspace/forge_experiments/optimization_history.md",
  "program": "# Program: implement fused_moe_swiglu_has_swiglu_limit_shuffled_16x16_model_dim6144_n_experts128_topk4_inter_dim768 in flydsl\n\n## Objective\n\nMake `run` in `kernel.py` faster than AKLR's baseline on as many of the 25 workloads below as\npossible, keeping every one of them correct: a candidate that wins on time and loses a workload is rejected.\n\n## Contract\n\n- `fused_moe`: Route each token to its top-k experts, run both projections, combine.\n- Attributes: `{\"activation\": \"swiglu\", \"quant_type\": \"per_1x32\", \"doweight_stage1\": false, \"gate_mode\": \"separated\", \"has_expert_mask\": false, \"has_swiglu_limit\": true, \"act_quant\": 4}`. Inputs: `hidden_states`, `w1`, `w2`, `topk_weights`, `topk_ids`, `w1_scale`, `w2_scale`, `swiglu_limit`. Outputs: `out`.\n- Fixed axes: `model_dim=6144, n_experts=128, topk=4, inter_dim=768, gate_up_rows=1536, model_blocks=192, inter_blocks=24`. Varying axes: `n_tokens`.\n- `run` takes AKLR's call keys as keyword arguments and writes each output into its preallocated argument;\n  it returns nothing and allocates no output.\n- `kernel.py` may import modules beside it relatively (`from .helpers import f`): those modules are part of the\n  kernel AKLR publishes; a file `kernel.py` does not reach, such as a probe, is not.\n- AKLR compares the outputs with `{\"max_violation_fraction\": 0.001, \"determinism\": \"bounded\", \"element_order\": \"fixed\"}`.\n- Every shape, dtype, layout and error bound is in the schema: `/home/ruibzhan/aklr-moe-m31-20261105/run-large/schema.json`.\n- What a kernel may and may not do (the audit, the ABI, determinism) is AKLR's rule:\n  `/home/ruibzhan/AKLR-wt-minimax-moe/aklr/ops/README.md`.\n\n## Workloads\n\n| case | `n_tokens` |\n|---|---|\n| `n_tokens=256` | 256 |\n| `n_tokens=272` | 272 |\n| `n_tokens=288` | 288 |\n| `n_tokens=304` | 304 |\n| `n_tokens=320` | 320 |\n| `n_tokens=336` | 336 |\n| `n_tokens=352` | 352 |\n| `n_tokens=368` | 368 |\n| `n_tokens=384` | 384 |\n| `n_tokens=400` | 400 |\n| `n_tokens=416` | 416 |\n| `n_tokens=424` | 424 |\n| `n_tokens=432` | 432 |\n| `n_tokens=448` | 448 |\n| `n_tokens=464` | 464 |\n| `n_tokens=480` | 480 |\n| `n_tokens=496` | 496 |\n| `n_tokens=512` | 512 |\n| `n_tokens=576` | 576 |\n| `n_tokens=640` | 640 |\n| `n_tokens=3112` | 3112 |\n| `n_tokens=3816` | 3816 |\n| `n_tokens=10000` | 10000 |\n| `n_tokens=13152` | 13152 |\n| `n_tokens=32768` | 32768 |\n\n## Where the numbers come from\n\n`driver.py` judges correctness with `python -m aklr validate` and times the candidate with AKLR's graph\ntimer on AKLR's timing inputs. It is the measurement boundary and is not editable.\n\nA score is AKLR's baseline time over the candidate's, per workload, averaged over the workloads: above 1.0 is faster than the baseline on average, not on every workload.\n",
  "knowledge_index": "# Knowledge base (maps for this task; full cards on disk \u2014 Read on demand)\n\nKnowledge root (KB): /home/ruibzhan/Hyperloom/src/kernelforge/knowledge/local_wiki/resources\n\nThe curated knowledge maps for this kernel task are below. Open a card with the\n`Read` tool using an ABSOLUTE path \u2014 a bare relative path resolves against the\nkernel's working directory (NOT the KB) and will miss. Build the absolute path:\n- a path a map lists relative to its own folder (e.g. `overall/\u2026`,\n  `skills/optimize/\u2026`) \u2192 prepend that section's `base:` shown below;\n- a cross-reference written as `<pillar>/\u2026` or `local_knowledge/<pillar>/\u2026`\n  (e.g. `hardware/\u2026`, `framework/aiter/\u2026`) \u2192 it lives at `/home/ruibzhan/Hyperloom/src/kernelforge/knowledge/local_wiki/resources/<pillar>/\u2026`.\nRead a card only when it is relevant \u2014 decide for yourself what to read.\n\n## hardware/  \u2014  base: /home/ruibzhan/Hyperloom/src/kernelforge/knowledge/local_wiki/resources/hardware\n\nMap not inlined \u2014 MI350X / MI355X hardware \u2014 knowledge map. `Read` `/home/ruibzhan/Hyperloom/src/kernelforge/knowledge/local_wiki/resources/hardware/INDEX.md` for this pillar's map before opening any card under it.\n\n## common_methodology/  \u2014  base: /home/ruibzhan/Hyperloom/src/kernelforge/knowledge/local_wiki/resources/common_methodology\n\nMap not inlined \u2014 Common optimization methodology \u2014 knowledge map. `Read` `/home/ruibzhan/Hyperloom/src/kernelforge/knowledge/local_wiki/resources/common_methodology/INDEX.md` for this pillar's map before opening any card under it.\n\n## languages/flydsl/  \u2014  base: /home/ruibzhan/Hyperloom/src/kernelforge/knowledge/local_wiki/resources/languages/flydsl\n\nMap not inlined \u2014 FlyDSL \u2014 knowledge map. `Read` `/home/ruibzhan/Hyperloom/src/kernelforge/knowledge/local_wiki/resources/languages/flydsl/INDEX.md` for this pillar's map before opening any card under it.\n\n## languages/assembly/  \u2014  base: /home/ruibzhan/Hyperloom/src/kernelforge/knowledge/local_wiki/resources/languages/assembly\n\nMap not inlined \u2014 AMDGPU assembly knowledge map. `Read` `/home/ruibzhan/Hyperloom/src/kernelforge/knowledge/local_wiki/resources/languages/assembly/INDEX.md` for this pillar's map before opening any card under it.\n\n## skill aklr-kb/  \u2014  base: /home/ruibzhan/AKLR-wt-minimax-moe/harness/skills/aklr-kb\n\nIts entry file `/home/ruibzhan/AKLR-wt-minimax-moe/harness/skills/aklr-kb/SKILL.md` is inlined below and applies to every round: follow it, and before planning or editing `Read` the cards it routes to for the direction you take; the cards are not inlined. A path it lists relative to its own folder resolves against the `base:` above, and a sibling skill it links as `../<name>/SKILL.md` is `/home/ruibzhan/AKLR-wt-minimax-moe/harness/skills/<name>/SKILL.md`.\nA path it gives relative to its repository resolves against `/home/ruibzhan/AKLR-wt-minimax-moe`. A section it scopes to that repository applies only when this task lives in that repository.\n\n### SKILL.md (inlined)\n\n# AKLR kernel knowledge base\n\nWhat makes a kernel fast on AMD GPUs inside AKLR, by op and by symptom. The `flydsl_` cards and the cases come\nfrom FlyDSL kernels: their mechanisms carry to other languages, their numbers were measured on FlyDSL code (re-derive\nthem: \u00a7 Rules that hold everywhere), and their code is FlyDSL. How to write it in\na language is in that language's folder: `flydsl/SKILL.md`, `triton/SKILL.md`, `hip/SKILL.md`.\n\nPaths below are relative to this folder, except those starting with `aklr/`, relative to the repository root, and\n`aiter/` or `csrc/`, relative to aiter's. File and line citations are to ROCm/FlyDSL `v0.3.2` (`dd8373349`),\nAMD-AGI/Primus-Turbo main `7615a5a2` (`ca1fdd67` for the expert GEMM in `moe/`) and ROCm/aiter `dbd8bf5`\n(`AKLR_BASELINE_AITER`); check out those revisions to follow them. FlyDSL's own skills are linked at `v0.3.2`, not\ncopied: `flydsl/guides.md` and `profile.md` give each link with what this repository corrects in it; a\n`\u00a7` marked as FlyDSL's is that skill's own numbering.\n\n## Stop: what a AKLR kernel cannot do\n\n`aklr/ops/README.md` \u00a7 The kernel section and ABI is the authority; where this skill disagrees, the README\nwins. Check a plan against this table before writing code.\n\n| Rule | What it rules out here |\n|---|---|\n| Outputs pass the definition's `compare` block; under `determinism: bitwise`, the default, two runs are byte-identical (`runs_agree`, `aklr/core/contract/semantics/_compare.py`) | Reducing in arrival order (atomic split-K, stream-K fix-ups): reduce in a fixed order (`gemm/flydsl_gemm_correctness.md` \u00a78, `gemm/flydsl_gemm_grid_pricing.md` \u00a72) |\n| A Point is served by the file with the best measured `speedup` over the baseline, and only from `AKLR_MIN_SPEEDUP` (1.03) | Judging on a mean: compare each Point with the faster of the kernel AKLR serves there and the baseline |\n| Launch code (`run` and what it calls) does not synchronize, call `.item()` / `.cpu()`, print, read a clock or the environment, or change torch state (SA3) | Timing or racing configurations in `run` |\n| Compilation and the choice of configuration happen in a module-level `@functools.cache` function on a Point's first call; later calls may be inside a CUDA graph capture | The first-call race of `gemm/flydsl_gemm_autotune.md` \u00a71 as written: choose from the shape, with a table raced offline |\n| Inputs other than the mutable ones are read-only (SA2); nothing derived from an argument is kept across calls, and there is no `global` or `nonlocal` (SA1) | Caching a repacked weight: repack on every call, or serve the contract's shuffled layout variant where it has one |\n| Scratch is allocated in the call (`torch.empty`: contents undefined) or once per shape in the compile function and rewritten before it is read; no call depends on what an earlier call left | Counters that live across calls: zero workspaces in each call before any read (`gemm/flydsl_gemm_small_m.md` \u00a74) |\n| No workgroup waits for another: nothing polls a flag or data that another workgroup of the launch writes, since a launch's workgroups need not all be resident (a stream limited to part of the CUs, another kernel holding CUs) | A split-K fold or split-KV merge whose owner polls its partners: merge in a separate launch, or have each publish and count in on an atomic counter and the last to arrive merge, in a fixed order |\n| `run(*, <kernel_call_keys>)` writes the preallocated outputs and returns nothing; the entry file's docstring describes the kernel, with no history or speedup (SA4) | Allocating outputs; importing sibling files other than relatively (the package name is a hash) |\n\n## How to look things up\n\n- Before opening a page, write down the diagnosis so far (the regime and the trace or counter that shows it),\n  the plan, and the one question whose answer would change the plan.\n- Pick one page and one section with the tables below, and read that, not the folder.\n- Then note what changed in the plan, and leave one line in the round's notes: `KB: matched <page> \u00a7<n>` or\n  `KB: no-match <question>`. The no-match lines are what this skill is missing.\n- Confirm a change by what its page says to check (final ISA, kernel descriptor, a counter), not by the total\n  time alone.\n- Before starting on an op, look in [cases/INDEX.md](cases/INDEX.md) for a delivered kernel of the same op and\n  regime, and in [pitfalls/INDEX.md](pitfalls/INDEX.md) for its traps.\n\n## By task\n\n| Task | Read in this order |\n|---|---|\n| Write or optimize a dense GEMM (fp8 / MXFP4 / MXFP8 / bf16) | `gemm/INDEX.md` \u2192 the card it routes to |\n| A GEMM with M \u2264 256 (decode, speculative verify) | `gemm/flydsl_gemm_small_m.md` |\n| A GEMM on gfx942 (MI300X) | `gemm/flydsl_gemm_gfx942.md` |\n| A MoE: fused MoE at decode sizes (MXFP4, routing sort, stage 1 / stage 2), or its expert grouped GEMM (forward, dgrad, variable-K wgrad) | `moe/INDEX.md` \u2192 the card it routes to; the expert GEMM is its \u00a7 Expert GEMM (grouped GEMM), with main loop and datapath in `gemm/` |\n| Write or optimize an attention kernel (prefill, backward, decode with split KV) | `attention/INDEX.md` \u2192 the card it routes to |\n| Dump a kernel's IR and final ISA; read registers, LDS, spills, waits and hot-loop counts | `gemm/flydsl_gemm_mainloop.md` \u00a710 (`FLYDSL_DUMP_IR=1`); `analysis/isa_diff.py` compares two dumps |\n| Capture a kernel trace and find the hotspot, or find the commit behind a performance regression | `profile.md` |\n| What an instruction, encoding, counter, wait state or hardware limit really is | `hardware/gfx950/INDEX.md` \u2192 `hardware/gfx950/lookup.md`, `hardware/gfx950/matrix_core.md`, `hardware/gfx950/memory_and_sync.md`; gfx942 (MI300X): `hardware/gfx942/INDEX.md` |\n| The language: its API, compile errors, debugging, its own guides | the language's folder: `flydsl/SKILL.md`, `triton/SKILL.md`, `hip/SKILL.md` |\n\n## By symptom\n\n| What the timings, trace or counters show | Read (then the symptom page, for what to rule out first) |\n|---|---|\n| Not yet known whether it is compute-, memory- or launch-bound | `gemm/flydsl_gemm_attribution.md` \u00a71\u20133 |\n| Fewer workgroups than CUs, or a last dispatch round mostly empty | `gemm/flydsl_gemm_grid_pricing.md` \u00a71\u20132, \u00a710; `patterns/low_cu_utilization.md` |\n| M \u2264 256 on a row bound by launch, reduction and latency, not by reading B | `gemm/flydsl_gemm_small_m.md` \u00a72\u20133; `patterns/launch_floor_small_m.md` |\n| LDS bank conflicts | `gemm/flydsl_gemm_datapath.md` \u00a73; `patterns/lds_bank_conflicts.md` |\n| MFMAs wait on global loads; bytes / time near the measured stream bandwidth | `gemm/flydsl_gemm_datapath.md` \u00a75, \u00a77; `patterns/memory_bound.md` |\n| Spills, VGPR / AGPR pressure, one wave per SIMD | `gemm/flydsl_gemm_mainloop.md` \u00a74, \u00a77, `gemm/flydsl_gemm_grid_pricing.md` \u00a74; `patterns/register_pressure.md` |\n| The store costs more than expected | `gemm/flydsl_gemm_epilogue.md` \u00a72\u20133 |\n| Memory-bound with a linear tile order: tiles that share a panel land on different XCDs | `gemm/flydsl_gemm_schedule_and_locality.md` \u00a72 |\n| Stalls on `s_waitcnt` | `gemm/flydsl_gemm_schedule_and_locality.md` \u00a74, `hardware/gfx950/memory_and_sync.md` \u00a71 |\n| A fixed cost per tile: time not proportional to K | `gemm/flydsl_gemm_attribution.md` \u00a77, `gemm/flydsl_gemm_grid_pricing.md` \u00a76 |\n| Clocks drop under load; slower than the instruction count says | `gemm/flydsl_gemm_attribution.md` \u00a74 |\n| A gain that does not reproduce, or an A/B inside the noise | `gemm/flydsl_gemm_measurement.md` \u00a71, \u00a74 |\n| Outputs that differ between runs | `gemm/flydsl_gemm_correctness.md` \u00a73, \u00a78 |\n| A change wins on some shapes and loses on others | `gemm/flydsl_gemm_autotune.md` \u00a71\u20132; `patterns/shape_divergence.md` |\n\n## Rules that hold everywhere\n\n- **Structure before parameters.** Fusion, pipelining and layout changes beat knob tuning; tune last.\n- **Reuse the mechanism, re-derive the number.** Constants in the cited kernels (tile and preload tables,\n  wait counts, adoption margins, cache thresholds) were tuned on specific shapes.\n- Numbers marked *(field)* are measured values: use them for direction and order of magnitude, not as\n  targets. *(field, E1)* points to the line under the page's `## Evidence` that gives the conditions.\n\n## Also in AKLR\n\n- **The aiter these cards describe is the baseline**, ROCm/aiter at `AKLR_BASELINE_AITER`\n  (`.github/workflows/ci.yaml`): AKLR does not ship, patch or fall back to it, so its `aiter/` paths are that\n  revision's, read as reference code. Price aiter through the op's `<contract>_baseline`\n  (`aklr/ops/<op>/baseline.py`), the call bench times. At a Point no kernel serves, the api returns a\n  `FallbackReason` and its caller runs its own path. AKLR's plugins route a model to plugin ops only\n  (`aklr/plugins/README.md`), so an `aklr/ops` kernel runs in a model only where the engine calls its api.\n- **Workflow** (`aklr/ops/README.md` \u00a7 Workflow): export the Points, fill `kernel.sources`, then\n  `python -m aklr export check` and `python -m aklr validate --base origin/main`. The local result does not\n  count; CI measures. A `platform_failure` \"Kineto recorded no kernels for a single call\" is a tracer fault: on a\n  busy node, wait a few minutes before you rerun (`pitfalls/profiler_misses_kernels.md`).\n- **AKLR does not time below about 4 \u00b5s.** Its bench runner forces `GPU_FORCE_QUEUE_PROFILING`, and a profiled\n  dispatch costs about that however little it does, so a short kernel and its baseline read about the same. A\n  Point whose baseline sits near that floor shows no gain in CI: check `baseline.json` before working on it.\n\n## Folders and adding a page\n\n`CONTRIBUTING.md`: what each folder holds, the page kinds, and how a page gets in.\n\n## skill aklr-kb/flydsl/  \u2014  base: /home/ruibzhan/AKLR-wt-minimax-moe/harness/skills/aklr-kb/flydsl\n\nIts entry file `/home/ruibzhan/AKLR-wt-minimax-moe/harness/skills/aklr-kb/flydsl/SKILL.md` is inlined below and applies to every round: follow it, and before planning or editing `Read` the cards it routes to for the direction you take; the cards are not inlined. A path it lists relative to its own folder resolves against the `base:` above, and a sibling skill it links as `../<name>/SKILL.md` is `/home/ruibzhan/AKLR-wt-minimax-moe/harness/skills/aklr-kb/<name>/SKILL.md`.\nA path it gives relative to its repository resolves against `/home/ruibzhan/AKLR-wt-minimax-moe`. A section it scopes to that repository applies only when this task lives in that repository.\n\n### SKILL.md (inlined)\n\n# FlyDSL kernels\n\nHow to write in FlyDSL what `../SKILL.md` says to do; check a plan against its Stop table first. Paths\nbelow are relative to this folder. FlyDSL's own skills are linked at `v0.3.2`, not copied: `guides.md` gives each\nlink and what differs on gfx950 or what this repository adds. Read one with `curl -sSfL <link>` (it needs network);\nwithout network, `guides.md` still has every correction.\n\nEvery API statement here is for FlyDSL v0.3.2; AKLR requires `flydsl>=0.3.2` (`pyproject.toml`). CI measures on the\nFlyDSL that the baseline aiter installs (`AKLR_BASELINE_AITER` in `.github/workflows/ci.yaml`; its `setup.py` pins\n0.3.2 today), so a new baseline can change it: check it before trusting a version difference\n(`../pitfalls/measured_a_different_build.md`).\n\n## By task\n\n| Task | Read |\n|---|---|\n| Your first FlyDSL GEMM (FlyDSL's walkthrough), before `../gemm/INDEX.md` | `guides.md` \u00a7 gemm-optimization |\n| The language: layouts, `@flyc.kernel` / `@flyc.jit`, copy atoms, MFMA, loops | `guides.md` \u00a7 flydsl-kernel-authoring |\n| Author and optimize any other kernel, structure first | `kernels/flydsl_authoring_method.md` |\n| Read aiter's FlyDSL kernels (dense GEMM, MoE, attention and the rest) and their knobs | `kernels/flydsl_kernel_library.md` \u2192 `kernels/flydsl_knob_space.md` |\n| LDS bank conflicts, double buffering, swizzle | `guides.md` \u00a7 lds-optimization |\n| Overlap loads with compute, prefetch, software pipelining; MFMAs wait on global loads | `guides.md` \u00a7 prefetch-data-load |\n| Modernize or clean up an existing kernel's code | `guides.md` \u00a7 kernel-code-cleanup |\n| Wrong output, NaN, won't compile, a fix that didn't take | `debug-flydsl-kernel.md` (clear the cache first); an error message: `language/compile_errors.md` |\n| Code from another FlyDSL version, or FlyDSL doing what you did not expect | `language/version_notes.md`, `language/limits.md` |\n| Run or extend FlyDSL's own tests and benchmarks, inside a FlyDSL checkout | `flydsl_repo_tests.md` |\n\n## Rules that hold everywhere\n\n- **Clear the compile cache** (`rm -rf \"${FLYDSL_RUNTIME_CACHE_DIR:-$HOME/.flydsl/cache}\"`) before believing\n  any result; `debug-flydsl-kernel.md` \u00a7 What the cache key sees tells you when a result was stale."
}

Reply with your reasoning, then one fenced JSON block, exactly:
```json
{"new_directions": [{"title": "...", "category": "...", "hypothesis": "...", "expected_gain_pct": 3.0,
  "reason": "why that much", "evidence": ["<absolute path of a file that supports it>", "..."]}]}
```
Each direction needs at least one evidence path to an existing file; one without is dropped. expected_gain_pct is
the gain you expect it to bring, in percent of the best score, as a number.

## Reply

Nine of the 14 ledger directions are still OPEN, and I didn't repeat any of them. I read the generation-001 report on 5bfbea64 and the campaign history. Since the parent analysis, iterations 2–5 were all kept. The kernel is now a 4–5 launch FlyDSL MoE that beats aiter on all 25 Points, from 1.053× to 1.134× (mean 1.071×). Every open ledger item came from the analysis of the old torch kernel, so I matched the new report's measurements against the ledger to find what it doesn't cover.

**What the new evidence says that the ledger misses:**
- **Small n (20 of the 25 Points, so 80% of the score).** Stage 1 reads W1 at 6.3–6.6 TB/s. Stage 2 reads W2 at only 6.0–6.4 TB/s, spread over 6144 short-lived workgroups in 4 rounds. Before the GEMMs there is a 7.6–9.1 µs routing sort that runs on one workgroup, so 255 CUs sit idle. The ledger only parallelised the sort for n > 1024, and its only small-n grid idea makes tiles narrower. Here the open questions are keeping weight loads in flight between tiles and a parallel sort at small n.
- **Large n, stage 2 (10000–32768 tokens).** A probe that swapped atomic adds for plain stores removed 30% of stage-2 time at 32768. That cost comes from read-modify-writes into a 402 MB output that is far bigger than the caches. Nothing in the ledger changes where those writes land in the cache.
- **Large n, stage 2 per-tile cost.** About 70% of stage 2 is fixed cost per tile: K is only 768 (3 k-steps), so loading A, the LDS reshuffle and the epilogue are paid for every 12.6 MFLOP. Stage 2 also holds 124 VGPRs at 2 workgroups per CU, where LDS would allow 4. The knowledge base's matched fix (reading A one 16-row group ahead, which cut registers by 8–26% in two fused-MoE kernels) appears nowhere in the ledger.
- **Stage 1 at 3112–32768 tokens.** It uses 324 VGPRs and 99 KB of LDS, so only one workgroup fits per CU. At 3816 the 918 workgroups need 3.59 rounds, which is why stage 1 there is 10.5 µs slower than aiter's. The ledger's main-loop item is about compute pipelining, not occupancy.
- **Output zero-fill (needed only because stage 2 adds atomically).** It is about half of what the x-quant kernel writes, and at large n x-quant plus zero-fill is 7–9% of the call. It can run inside stage 1, which is compute-bound there and would hide it.

**What I left out as already covered:**
- Folding launches together and decoupling quantization from the sort: covered by "cut graph nodes".
- Stage-2 tile-M of 128: the tile-M item.
- Atomics versus split-and-reduce: the stage-2 combine item.
- XCD tile order: the XCD/GROUP_M item.
- Capped grids at large n: the persistent-grid item.
- Tuning: the offline config table.
- The stage-1 epilogue: already has its own item.

I deliberately left out a non-atomic stage 2 written as store-then-combine; the report estimates it at only 0–3% of stage 2.

```json
{"new_directions": [
{"title": "Small-n stage 2 as a streaming persistent kernel: resident-slot grid (6 WG/CU × 256) looping over the 6144 tiles, with the next tile's W2 loads issued before the current tile's epilogue/atomics, so the W2 stream never drains between tiles",
 "category": "grid / persistence (S regime, stage 2 streaming)",
 "hypothesis": "At n≤640, stage 2 runs 6144 short-lived workgroups in exactly 4 rounds (48 KB of W2 each, persist_m=1). It reads W2 at 6.0–6.4 TB/s while stage 1 reaches 6.5–6.6. Each workgroup's ramp-up and epilogue leaves the HBM queue empty, and the round boundaries add tail time. A persistent loop with cross-tile B prefetch (no barrier, C staged per wave, the form that beat one-tile-per-WG by 3–5% on a dense GEMM) should bring W2 up to stage 1's rate: −3 to −5 µs on ~55 µs. This is the opposite axis to the ledger's tile-N-downward race, and both should be raced against the current setting.",
 "expected_gain_pct": 1.5,
 "reason": "Report §6.2 estimates −3 to −5 µs of a ~170–180 µs call (+2–3%) on all 20 S Points. Discounted because the stage is already near the stream rate.",
 "evidence": ["/home/ruibzhan/aklr-moe-m31-20261105/run-large/workspace/forge_experiments/analysis/5bfbea64c0d72d191756b25f14d171e78e120703/generation-001/report.md", "/home/ruibzhan/AKLR-wt-minimax-moe/harness/skills/aklr-kb/moe/flydsl_grouped_moe_inference.md"]},
{"title": "Parallel routing sort at n≤640: replace the one-workgroup serial sort (7.6–9.1 µs on one CU) with a wave-per-expert / bitmask-popcount sort spread over many workgroups",
 "category": "dispatch / routing sort (S regime)",
 "hypothesis": "The S-regime sort is a single workgroup looping over R/1024 routes and writing 31 padding rows per expert serially while 255 CUs idle. It sits on the critical path in front of x-quant and stage 1. Field data: a wave per expert instead of a workgroup took a sort from 7.3–7.8 to 6.5 µs, and per-expert bitmasks (LDS atomic OR + popcount + one packed prefix scan) took another from 7.65 to 5.32 µs against a ~4.4 µs node floor. Doing the same here should cut 2–4 µs per call. The kept multi-WG sort for n>1024 is a starting point, but it must not add a launch at small n.",
 "expected_gain_pct": 1.2,
 "reason": "2–4 µs of 168–182 µs is 1–2% on 20 of 25 Points. Capped by the ~4 µs per-node floor.",
 "evidence": ["/home/ruibzhan/aklr-moe-m31-20261105/run-large/workspace/forge_experiments/analysis/5bfbea64c0d72d191756b25f14d171e78e120703/generation-001/report.md", "/home/ruibzhan/AKLR-wt-minimax-moe/harness/skills/aklr-kb/moe/flydsl_grouped_moe_inference.md"]},
{"title": "Stage-2 register plan for M/L: issue A ds_reads one 16-row group ahead to cut VGPRs below 124, then raise resident workgroups from 2 to 3–4 per CU to hide atomic and B latency",
 "category": "register pressure / occupancy (stage 2)",
 "hypothesis": "Stage 2 at large n uses 124 VGPRs and runs 2 WG/CU (cu_num_mul 2) where LDS (16.6 KB) would allow 4. It reaches only 0.96 PF/s, and ~25–30% of its time is waiting on atomics that miss L2. The KB-matched mechanism (A read per 16-row slice one phase ahead) took a fused-MoE MXFP4 stage 2 from 146 to 122 VGPRs at −3%, and gave −8% on another. Raising cu_num_mul alone did nothing (1278.5 vs 1285.9 µs), so registers are the binding limit. With registers freed, more waves can cover the atomic and B-load latency.",
 "expected_gain_pct": 1.2,
 "reason": "Report §6.3 estimates −8 to −15% of stage 2, which is 40–55% of the call at 10000–32768 and ~45% at 3112/3816. That is ~4–7% on 5 Points.",
 "evidence": ["/home/ruibzhan/aklr-moe-m31-20261105/run-large/workspace/forge_experiments/analysis/5bfbea64c0d72d191756b25f14d171e78e120703/generation-001/report.md", "/home/ruibzhan/AKLR-wt-minimax-moe/harness/skills/aklr-kb/gemm/flydsl_gemm_mainloop.md"]},
{"title": "Stage-1 occupancy at M/L: bring the t128 build from 324 VGPRs / 99 KB LDS (1 WG per CU) to 2 WG per CU, and choose tiles so 3112/3816 fill whole dispatch rounds",
 "category": "register pressure / occupancy (stage 1) / wave quantization",
 "hypothesis": "Stage 1 at n>1024 runs one wave per SIMD at 2.0–2.9 PF/s (31% of nominal). At 3816 its 918 WGs take 3.59 rounds with the last 59% full, which is why stage 1 there loses 10.5 µs to aiter. Reading A per 16-row slice one phase ahead (KB-matched: 202→148 VGPRs, +7.5% on a t64 MXFP4 stage 1) plus a shallower or narrower LDS stage would let 2 WGs share a CU and overlap one's epilogue with the other's MFMAs. A tile choice that makes WGs a multiple of resident slots removes the partial round at 3816.",
 "expected_gain_pct": 0.9,
 "reason": "Report §6.4 estimates −5 to −10% of stage 1, which is ~45% of the call, on 5 Points: ~2–4.5% each.",
 "evidence": ["/home/ruibzhan/aklr-moe-m31-20261105/run-large/workspace/forge_experiments/analysis/5bfbea64c0d72d191756b25f14d171e78e120703/generation-001/report.md", "/home/ruibzhan/AKLR-wt-minimax-moe/harness/skills/aklr-kb/gemm/flydsl_gemm_grid_pricing.md"]},
{"title": "Stage-2 tile order for output-atomic locality at L: sweep column slabs outermost (all experts' M-blocks for one band of N tiles), so the bf16 read-modify-write working set of out fits in L2/MALL instead of HBM",
 "category": "scheduling / locality (output side)",
 "hypothesis": "At 32768 the atomic RMW into a 402 MB out (bigger than the 256 MB MALL and the 4 MB per-XCD L2) costs ~390 µs, 30% of stage 2. Each token's 4 routes sit in 4 distinct expert blocks far apart in the N-fastest order, so every RMW line is fetched from HBM up to 4 times. With an outer band of 1–8 N tiles, the out slab touched is 8–64 MB (32768×BN×2 B), the W2 band per expert is small, and the gathered A (50 MB total) stays in MALL, so the atomics hit cache. Race band widths against the current order and check that error stays within bounds.",
 "expected_gain_pct": 0.8,
 "reason": "Recovering half the measured atomic cost is ~8% at 32768, ~5% at 13152, ~4% at 10000 and ~1–2% at M. That is ~0.8% of the mean.",
 "evidence": ["/home/ruibzhan/aklr-moe-m31-20261105/run-large/workspace/forge_experiments/analysis/5bfbea64c0d72d191756b25f14d171e78e120703/generation-001/report.md", "/home/ruibzhan/AKLR-wt-minimax-moe/harness/skills/aklr-kb/gemm/flydsl_gemm_schedule_and_locality.md"]},
{"title": "Stage-2 A-stationary N sweep: each workgroup keeps its gathered A tile (≤64×768 fp4 + scales, ~25 KB) in LDS and walks several N tiles, double-buffering B across tile boundaries, to amortize the short-K per-tile fixed cost",
 "category": "main loop / stage-2 per-tile overhead",
 "hypothesis": "With K=768 a 64×128 stage-2 tile has only 3 k-steps of 256, so the A gather, B fill, LDS c-shuffle and epilogue dominate. Even without atomics stage 2 reaches only 1.38 PF/s vs stage 1's 2.9. Today the gathered A rows are re-fetched for each of the 48 N tiles (~2.4 GB of gather traffic at 32768). Loading A once per workgroup and sweeping 4–8 N tiles, with the next B tile's loads issued during the current tile's atomics, removes the repeated gathers and the pipeline drain. This differs from the ledger's persistent-grid item, which targets dispatch waste, not A reuse or cross-tile pipelining.",
 "expected_gain_pct": 1.0,
 "reason": "Report §4.3 puts the per-tile overhead as the larger portable share of stage 2. A 10–20% stage-2 cut on 5 M/L Points is ~4–8% each, discounted for occupancy risk.",
 "evidence": ["/home/ruibzhan/aklr-moe-m31-20261105/run-large/workspace/forge_experiments/analysis/5bfbea64c0d72d191756b25f14d171e78e120703/generation-001/report.md", "/home/ruibzhan/AKLR-wt-minimax-moe/harness/skills/aklr-kb/gemm/flydsl_gemm_epilogue.md"]},
{"title": "Hide the out zero-fill at M/L: move it from the x-quant kernel into compute-bound stage 1 (each workgroup zeroes a slice after its main loop), and vectorize x-quant's 1-byte E8M0 scale stores",
 "category": "data path / small kernels (large n)",
 "hypothesis": "At large n, x-quant plus zero-fill is 7–9% of the call at 4.0–4.6 TB/s, and the zero-fill is half its writes (402 MB at 32768, ~80 µs). It exists only for the atomic stage 2. Stage 1 runs before stage 2 on the same stream and is compute-bound at 2–2.9 PF/s, so streaming stores issued there overlap MFMA work at nearly no cost. Packing the four 1-byte scale stores per 32-element block into wider stores trims the rest. The order stays correct because stage 1 finishes before stage 2 starts, and nothing waits across workgroups.",
 "expected_gain_pct": 0.5,
 "reason": "~2.5–3.5% on 3 L Points and ~2% on 2 M Points, ~0 at S. That is ~0.5% of the mean.",
 "evidence": ["/home/ruibzhan/aklr-moe-m31-20261105/run-large/workspace/forge_experiments/analysis/5bfbea64c0d72d191756b25f14d171e78e120703/generation-001/report.md", "/home/ruibzhan/AKLR-wt-minimax-moe/harness/skills/aklr-kb/moe/flydsl_grouped_moe_inference.md"]},
{"title": "S-regime stage-1 memory-level parallelism: deepen W1 prefetch (more outstanding 16-byte buffer_loads per wave, an extra B stage) in the single-round 768-WG grid to push 6.3–6.6 TB/s toward ~7 TB/s",
 "category": "data path / bytes in flight (S regime, stage 1)",
 "hypothesis": "At n≤640, stage 1 is exactly one round (768 WGs at 3 WG/CU, 142 VGPRs), so the grid cannot add parallelism. The only way to keep more HBM requests in flight is more outstanding loads per wave. Stage 1 is ~100 µs of a ~175 µs call against a 146 µs practical floor. A 5% rate gain is ~5 µs per Point. Check against waitcnt stalls in the ISA and the VGPR count, so the change does not drop occupancy.",
 "expected_gain_pct": 0.8,
 "reason": "Up to ~2–3% per S Point if the rate rises 5%. Discounted because 6.6 TB/s already beats every stream kernel measured here (≤5.8 TB/s).",
 "evidence": ["/home/ruibzhan/aklr-moe-m31-20261105/run-large/workspace/forge_experiments/analysis/5bfbea64c0d72d191756b25f14d171e78e120703/generation-001/report.md", "/home/ruibzhan/AKLR-wt-minimax-moe/harness/skills/aklr-kb/gemm/flydsl_gemm_mainloop.md"]}
]}
```
