# Muse Spark's AKLR definitions: where they come from

A AKLR definition describes one op at one shape; its workloads are the sizes the
benchmark times it at. This page records the deployment the Muse Spark
definitions were taken from, and which calls of the model they are.

## The deployment

8x MI355X, `Muse-Spark-1.2-816B-A42B` with Quark MXFP4 routed experts, BF16
everything else; vLLM's MI355X recipe:

| | value |
|---|---|
| image | `vllm/vllm-openai-rocm:nightly-rocm72-83c078d955cd6e0c63d3b12abe669c8de936a370` (vLLM `0.32.1rc1.dev361+g83c078d95`, the first nightly with Muse Spark; no release has it yet) |
| aiter | this repository at `1d4dab9b` with the gemm n % 8, vLLM plugin and `swiglu_limit` changes, installed in place of the image's `amd-aiter` (`GPU_ARCHS=gfx950`) |
| environment | `VLLM_ROCM_USE_AITER=1 VLLM_ROCM_MUSE_MOE_REDUCE_DTYPE=bf16 VLLM_USE_BREAKABLE_CUDAGRAPH=1` |
| `vllm serve` | `--tensor-parallel-size 8 [--enable-expert-parallel] --dtype bfloat16 --distributed-executor-backend mp --language-model-only --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' --no-enable-prefix-caching --enable-chunked-prefill --attention-backend ROCM_AITER_FA --max-model-len 32768 --max-num-seqs 128 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.90` |
| layouts | **TP8**, and **EP8**: the same with `--enable-expert-parallel` (attention TP8, experts 32 per rank, no all-to-all) |
| load | `vllm bench serve`, random prompts: 1,024 in / 256 out at 1, 4, 16, 64 and 128 concurrent requests, then 16,384 in / 64 out at 4 |

Recorded with `AKLR_EXPORT=1` on all 8 ranks; the plugin entry point initializes
AKLR in each worker. The image's `/usr/share/libdrm/amdgpu.ids` predates MI355X,
so torch reports no device name and AKLR cannot name the target; the run
mounts a current `amdgpu.ids` (from AKLR CI's image) over it.

## Definitions

GEMMs are bf16, the weight `[N, K]`, no bias, through aiter's `gemm_a16w16`.
Hidden size 8192; the experts run at `moe_hidden_size` 4096 between the
pooler and unpooler. Every definition is seen in both layouts unless noted.

| definition | what it computes |
|---|---|
| `gemm_n2304_k8192` | attention's fused q, k, v and output gate: (8192 + 1024 + 1024 + 8192) / 8 |
| `gemm_n8192_k1024` | o_proj: 128 heads x 64 / 8 in |
| `gemm_n25256_k8192` | lm_head: 202,048 / 8 |
| `fused_moe_shuffled_16x16_model_dim4096_n_experts256_topk1_inter_dim512` | routed experts under TP8: 256, inter 4096 / 8; vLLM expands each token into its 8 routes, one row each |
| `fused_moe_has_expert_mask_shuffled_16x16_model_dim4096_n_experts32_topk2_inter_dim4096_global_experts257` | routed experts under EP8: this rank's 32 of 256, whole, by `expert_mask`; each row's second route is aiter's always-masked 257th slot; EP8 only |

Not shipped, though they run: attention, norms and rotary embeddings, the
FP32 router and its top-k, the pooler and unpooler (torch `F.linear` by design),
vLLM's own skinny GEMMs for small batches, and the experts of layers 0 and 61,
which the checkpoint keeps in BF16: none reaches a AKLR route.

## Workloads

As recorded: every row count a call was made at, under both layouts merged. In
decode, vLLM pads a batch to its CUDA-graph sizes (multiples of 8 up to 256
here) and replays the graph; prefill runs eagerly at its chunk sizes, up to the
8,192-token budget, 65,536 expert rows. The load is synthetic: these are the
sizes this deployment can produce, not a weighting of a real trace. Call
counts are captures, not throughput.

| definition | rows | workloads |
|---|---|---|
| `gemm_n2304_k8192` | 8 to 8,192 | 57 |
| `gemm_n8192_k1024` | 8 to 8,192 | 56 |
| `gemm_n25256_k8192` | 6 to 256 | 64 |
| `fused_moe_..._n_experts256_topk1_inter_dim512` | 8 to 65,536 | 51 |
| `fused_moe_has_expert_mask_..._global_experts257` | 8 to 65,536 | 53 |

To be redone on vLLM's first release image with Muse Spark (v0.33.0): the
nightly's recipe may change before it.
