# Kimi K3's AKLR definitions: where they come from

A AKLR definition describes one op at one shape; its workloads are the sizes the
benchmark times it at. Neither file says which deployment it was taken from: a
definition's description and tags are its template's, the same for every shape.
This page records it for every definition K3 runs, and for every row count its
workloads sweep (origin/main's `aklr/benchmarks/api/config.py`).

## The deployments

All on 8x MI355X, BF16 dense weights, MXFP4 routed experts with fp8
activations (`AITER_SITUV2_A8W4=1`), fp8 KV cache.

| | image | sglang | aiter | layout | how the calls were recorded |
|---|---|---|---|---|---|
| **A** | `v0.5.18-rocm724-mi35x-20260903` | - | - | TP8, Triton attention | AKLR schema dump during K3 forwards at 1 to 64 concurrent requests and one 16,384-token prefill chunk |
| **B** | `v0.5.19-rocm720-mi35x-20260916` (the cookbook's) | `00a9a81` (`e7f7447` plus one NPU-test-only commit) | `4ad998328` | the cookbook's MI355X recipe: TP8 + decode context parallelism over 8 (DCP8), aiter attention; also with `--ep-size 8` | this directory's forward driver (`profile_decode.py` at `--CONC` 1 and 16, `profile_prefill.py` at one 16,384-token chunk) with every op call's arguments logged; top-k and both MoE definitions also by a live schema dump (`--CONC` 16 and 56, and EP8 in decode and prefill), as were the one-token MLA call's shapes |
| **C** | `v0.5.20-rocm720-mi35x-20260926` | `fc9e1c8` | `acf8fdf` | B's recipe as `sglang.launch_server`, without speculative decoding and with DSPARK (`RadixArk/Kimi-K3-DSpark`, block sizes 7, 5, 4, 3, static verify) | every op call's arguments logged on the server, while it captured its CUDA graphs and under `sglang.bench_serving` load: 100,000-token random prompts, 1 to 16 concurrent requests |

Where a definition was written from logged arguments rather than dumped live,
AKLR's own recorder wrote it from those arguments on CPU tensors: the recorder
reads only shapes, dtypes and scalars, which are what was logged. GEMM files
written that way match a live dump byte for byte.

In C, the large batches come from CUDA graph capture: the server records one
graph per batch size, and at run time pads a batch of n requests to the
smallest captured size of at least n and replays that graph. The graph-replayed
ops therefore run at exactly the captured sizes, whatever the load.

## Definitions

GEMMs are bf16, the weight `[N, K]`, no bias. "Seen" means the same shape and
dtypes were logged there; only the column "from" wrote the file.

| definition | what it computes | from | seen |
|---|---|---|---|
| `gemm_a16w16_nt_n6016_k7168` | MoE front: router, latent down and shared-expert gate_up in one GEMM (torch) | B | C |
| `gemm_a16w16_nt_n7168_k768` | shared-expert down (torch) | B | C |
| `gemm_a16w16_nt_n6144_k7168` | KDA in-projection (aiter) | A | B prefill; C every step |
| `gemm_a16w16_nt_n6288_k7168` | KDA in-projection in decode, B only; C splits it into n6144 and the two below | B | - |
| `gemm_a16w16_nt_n144_k7168`, `gemm_a16w16_nt_n1536_k128` | KDA's beta and forget-gate projections (sglang's tiny GEMM below 16-24 rows, torch above) | B | C |
| `gemm_a16w16_nt_n1536_k7168` | MLA q_a (aiter) | A | B, C |
| `gemm_a16w16_nt_n2112_k7168` | MLA q_a and kv_a fused (aiter) | A | B, C |
| `gemm_a16w16_nt_n18432_k1536` | MLA q_b over all 96 heads, decode and verify (torch) | B | C |
| `gemm_a16w16_nt_n2304_k1536` | MLA q_b, prefill (aiter) | A | B, C prefill |
| `gemm_a16w16_nt_n3072_k512` | MLA kv_b over the prefix, prefill (aiter) | A | B, C prefill |
| `bmm_a16w16_nn_n512_k128`, `bmm_a16w16_nt_n128_k512` | MLA's two absorb products, batch 96 and 12 (torch) | B | C |
| `gemm_a16w16_nt_n7168_k1536` | MLA and KDA o_proj (aiter) | A | B, C |
| `gemm_a16w16_nt_n7168_k3584` | latent up (aiter) | A | B, C |
| `gemm_a16w16_nt_n7168_k4224`, `gemm_a16w16_nt_n8448_k7168` | dense layer 0's down and gate_up (aiter) | A | B, C |
| `gemm_a16w16_nt_n20480_k7168` | lm_head; DSPARK's draft projects through the same head (torch) | B | C |
| `gemm_a16w16_nt_n768_k7168`, `gemm_a16w16_nt_n3584_k7168`, `gemm_a16w16_nt_n7168_k1792` | DSPARK draft's qkv, gate_up, down (aiter) | C | - |
| `gemm_a16w16_nt_n7168_k35840` | DSPARK's KV inject: five target layers' hidden states projected (torch) | C | - |
| `gemm_a16w16_nt_n1280_k7168` | DSPARK's KV inject: the stacked context K/V (torch) | C | - |
| `gemm_a16w16_nt_n163840_k256` | DSPARK's Markov head over the full vocabulary (torch) | C | - |
| `flash_mla_with_kvcache_fp8_e4m3_varlen_lse_query_length1_n_heads96_page_size1` | MLA decode without speculation, aiter `mla_gluon` | B | C |
| `flash_mla_with_kvcache_fp8_e4m3_table_lse_query_length1_n_heads96_page_size1` | DSPARK verify over the committed prefix, aiter `mla_gluon` | C | - |
| `flash_mla_with_kvcache_bf16_varlen_causal_lse_query_length8_n_heads96_page_size1` | DSPARK verify over the 8-token window, aiter `mla_gluon` | C | - |
| `biased_grouped_topk_bf16_renorm_n_experts896_topk16` | router top-16 of 896 experts, aiter | B | C |
| `fused_moe_situv2_per_1x32_model_dim3584_n_experts896_topk16_inter_dim384` | routed experts under TP8, aiter | A; B live dump identical | C |
| `fused_moe_situv2_ep_per_1x32_model_dim3584_n_experts112_topk16_inter_dim3072_global_experts896` | routed experts under EP8: 112 of 896, whole | B with `--ep-size 8` | - |

The draft's o_proj, `gemm_a16w16_nt_n7168_k512`, was shipped for another model
and keeps the default sweep.

Not shipped, though logged in C: the vision encoder's GEMMs (they run once, at
startup), and the ops AKLR cannot state today: KDA's recurrence and the
ReplaySSM commit (they update state in place), the draft's attention, the LSE
merge after split verify, DCP's exchange kernels.

## Workloads

The workloads size every K3 definition for sglang's recipe (C's layout and
graph sizes) serving InferenceX's AgentX benchmark: agentic coding sessions
replayed at fixed concurrency. Three sources:

- **AgentX lengths**, measured. InferenceX publishes every client request of
  its official K3 run on 8x MI355X (run `36082123873`, 2026-09-25, vLLM, 9
  concurrency points). A request's context while it decodes, and its new and
  cached tokens when it prefills (from the AgentX dataset's 64-token hash
  blocks, assuming an unbounded prefix cache), follow from the traffic, not the
  engine. Every point weighs equally.
- **sglang batch sizes**, estimated: no sglang AgentX run exists. Each vLLM
  point keeps its non-decode time per request cycle (prefill, queueing, the
  replayed tool and think gaps); its decode time is replaced by sglang's
  (step time per accepted token, below); the share of time a client decodes,
  and so the mean batch, follows. The batch over time is the vLLM point's,
  scaled to that mean, capped at sglang's `max_running_requests` and padded up
  to its CUDA-graph sizes. The same estimate, made for GLM-5.2 on sglang before
  a fix lengthened its outputs, came within -8% to +16% of the mean batch
  measured after it.
- **The server's own facts**, logged in C: graph sizes, pool sizes, scales,
  rows per request in DSPARK.

| | value | source |
|---|---|---|
| decode context per request, mean | 189,427 tokens; 23,678 per rank over DCP8 | AgentX |
| decode context p10 / p50 / p90 / p99 | 55k / 140k / 469k / 678k | AgentX |
| spread within a decode batch | log of the length: standard deviation 0.98 in batches of 8 or more | AgentX |
| prefill new tokens p10 / p50 / p90 / p99 | 893 / 2,732 / 9,397 / 67,000 | AgentX |
| prefill cached + new, p10 / p50 / p90 | 35,107 / 104,093 / 342,073 | AgentX |
| sglang step, DSPARK block size 7 | 28.6 / 38.0 / 51.2 ms at 1 / 4 / 8 requests, 3.84 tokens accepted per step | C, under the torch profiler, at 12,501 rows per rank; MLA's share scaled to AgentX's |
| sglang step, no speculation | 17.5 / 27.1 / 46.4 ms at 1 / 16 / 56 requests | B, likewise |
| DSPARK batches, 1 to 14 clients | 1..8, 10, 12, 14 (99.9% of decode steps); 1 request in 29% | estimated |
| batches without speculation, 44 to 70 clients | 8, 12, 16, 24, 32, 40, 48, 56, 60 (99.3%) | estimated |

Speculation follows InferenceX's split for K3: DSPARK at 1 to 14 clients, none
from 44. A graph size enters the sweep if it carries at least 0.5% of decode
steps. DSPARK rows per step are C's: 8 verify rows, 7 draft rows and one
Markov-head row per request.

How the MLA initializers use the lengths: each request's length is drawn
log-normal with sigma 1, clipped at two sigmas. In batches of 35 to 70 that
gives a log standard deviation of 0.95 and a longest request 6.5 to 7.3 times
the median; AgentX's are 0.91 to 1.20 and 5.6 to 6.7. One-token decode splits `total_kv` (batch x 23,678) that way. The
verify prefix sets the longest possible request to the table width, so the
mean is 0.2106 of it and the sweep sizes the table at 23,678 / 0.2106; the 8
rows of a request share their table row, as deployed, though nothing checks
that a kernel does not rely on it.

Which definition takes which rows is in origin/main's `config.py`. Derived there, neither
measured nor estimated:

- **`gemm_a16w16_nt_n6288_k7168`** fires only on B, which ran without DSPARK;
  it takes the decode and prefill rows.
- **The EP8 experts** take the target's rows, DSPARK included, though B ran
  EP8 without it. Without an all-to-all backend, attention stays TP8 and every
  rank's experts receive every token, so the rows are the TP8 experts'.
- **MLA kv_b in prefill** runs over a request's cached prefix and new tokens.
  C ran it over the whole prefix in one call, up to 115,808 rows; the sweep
  takes that to hold to AgentX's p90 of 342,073.
- **At 60 requests without speculation**, AgentX's mean context outgrows the
  1,376,021-row pool, so the workload repeats slots; a real server would be
  short of KV there.

Not swept: DSPARK at block sizes 5, 4 and 3 (C logged their shapes; InferenceX's
vLLM run lowers the block size as concurrency grows, sglang's recipe does not).
The scales are `sm_scale` 1/sqrt(192) and `kv_scale` 1.0, as logged; the pools
are C's, 1,376,021 rows without speculation and 760,149 with DSPARK, whose
verify window reads its own 8 rows per request.
