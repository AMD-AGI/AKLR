# Kimi-K3 model-level micro-bench

Times every operator on its own, sums them as `per-layer cost × layer count`, checks that sum against a full-forward measurement.

`ratio = roll-up / full forward`. Near 1 means nothing was missed — not that the split inside a layer is right.

`profile_decode.py` times one decode step. `profile_prefill.py` times every chunk of one request's TTFT.

K3 is hybrid: 69 of its 93 layers run Kimi Delta Attention (KDA), a linear attention whose context is a fixed-size recurrent state, and 24 run MLA over a paged KV cache. Rows are therefore keyed by layer kind — `moe+kda`, `moe+mla`, `dense+kda` — and the roll-up is a sum of three per-layer costs, not one.

The `layer` row is additionally timed once per **residual depth**. K3 banks a snapshot of the residual stream every 12 layers and every layer mixes the rows banked so far back in, on both sides of attention, so a layer's cost grows with its position in the stack — 0 rows at layer 0, 8 at the deepest — while everything inside attention and the MLP sees the same shapes at every depth. Timing one layer per kind would price all 24 MLA layers at the cost of the shallowest.

A layer that banks a row is its own group, written `3w`: it snapshots at its pre-attention point, so its second aggregation mixes one row more than its first, where every other layer mixes the same count at both. That is 24 groups over the 93 layers. The report shows the per-depth measurements in their own section; every other table shows the count-weighted mean, which multiplied by the layer count reproduces the sum over the real stack.

## Usage

```bash
cd aklr/benchmarks/models/kimi_k3    # required, the scripts do `import model_bench`
M=/data/models/Kimi-K3               # local checkout; the default is a hub id and a 1.5 TB fetch
```

The environment of the K3 cookbook's MI355X recipe (image `lmsysorg/sglang-rocm:v0.5.19-rocm720-mi35x-20260916`). Everything below assumes it is exported; the report prints it back as a **Kernel environment** table, because `AITER_SITUV2_A8W4` decides the arithmetic inside `mlp.experts` and nothing in the call itself says which was used.

```bash
export SGLANG_USE_AITER=1 SGLANG_AITER_K3_OPT=1 AITER_FLYDSL_FORCE=1 \
       AITER_SITUV2_A8W4=1
# not in the recipe, but it does not start without it -- see below
export SGLANG_AITER_HONOR_EXPLICIT_MEM_FRACTION=1
# optional: fuses f_b + conv + recurrence + gated RMSNorm into one KDA decode kernel,
# which makes attn.o_norm stop firing and its cost move into attn.<rest>
export SGLANG_K3_KDA_FUSED_BACKEND=aiter
```

The recipe's server flags, which the bench takes as they are. `profile_prefill.py` takes the same flags minus `--CONC`.

```bash
# TP8 + DCP8, the recipe's single-node layout
python3 profile_decode.py --model $M --tp-size 8 \
    --dcp-size 8 --dcp-comm-backend a2a \
    --prefill-attention-backend aiter --decode-attention-backend aiter \
    --mem-fraction-static 0.85 \
    --ISL 100000 --CONC 16 --reps 20

# TP8 without DCP, for comparison: the same flags minus the first line
python3 profile_decode.py --model $M --tp-size 8 \
    --prefill-attention-backend aiter --decode-attention-backend aiter \
    --mem-fraction-static 0.85 \
    --ISL 100000 --CONC 16 --reps 20

# EP8 -- mori is the only a2a backend that loads on ROCm here. The dispatch cap
# sizes every mori call: at 4096 the experts are handed 32,768 padded rows and
# the step runs 4x slower. It must be >= --chunked-prefill-size, so both drop.
SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=64 MORI_SHMEM_MODE=ISOLATION \
python3 profile_decode.py --model $M --tp-size 8 --ep-size 8 \
    --moe-a2a-backend mori --moe-dense-tp-size 1 --enable-dp-lm-head \
    --chunked-prefill-size 64 \
    --prefill-attention-backend aiter --decode-attention-backend aiter \
    --mem-fraction-static 0.85 \
    --ISL 100000 --CONC 8 --reps 20

# prefill, one chunked request per attention rank
python3 profile_prefill.py --model $M --tp-size 8 \
    --dcp-size 8 --dcp-comm-backend a2a \
    --prefill-attention-backend aiter --decode-attention-backend aiter \
    --mem-fraction-static 0.85 \
    --ISL 100000 --reps 20 [--prefix-cache-hit 0.97]
```

**The recipe does not start as published.** Under the aiter attention backend and a context over 8192, sglang multiplies `--mem-fraction-static` by 0.85, including a value passed explicitly: the recipe's 0.85 becomes 0.7225, the 194 GB of weights per rank take the whole budget, and sglang stops in its state-pool sizing (`total_rest_memory=-0.59 GB`) whether it is the bench or `sglang.launch_server` with the recipe's flags verbatim. `SGLANG_AITER_HONOR_EXPLICIT_MEM_FRACTION=1` keeps the value given; export it with the rest. The bench prints a line when the fraction it was given is not the one sglang resolved.

Pass the attention backends explicitly too. Left unset, sglang resolves `triton` on this image; the recipe is `aiter`, and at long context they are not close: at TP8, ISL 100k, `--CONC` 8 the MLA decode kernel took 13.9 ms of a 35.0 ms step under Triton and 2.4 ms of 23.0 ms under aiter. The report's `attention` row names the backend classes sglang built, so a run on the wrong ones shows it.

The bench measures whichever `aiter` Python imports. This repository carries its own `aiter` package, so running with the repository root on `PYTHONPATH` times this tree's kernels rather than the image's; the report's Versions table says which. The scripts need only their own directory on the path.

### What fits, measured on 8x MI355X (288 GB each)

2.8T parameters at MXFP4 is ~1.5 TB of weights before any cache.

The first three rows are on the recipe's image and flags, `--mem-fraction-static 0.85` kept as given (EP8 with the mori flags above); the rest were measured on the 20260903 image and not re-run.

| layout | at ISL 100k |
| --- | --- |
| TP8 + DCP8 | weights 193.8 GB/rank. KV pool 10.98M tokens of context (1.37M rows per rank): 109 requests. State pool 302 slots: 60 requests, so the state pool binds and `--CONC` 64 is refused at the default split; see below |
| TP8 | KV pool 1.70M tokens: 16 requests, so `--CONC` ends at 16. State pool 374 slots |
| EP8 (mori) | KV pool 871k tokens: `--CONC` ends at 8. Step 37.7 ms at 8 (38.2 at 4, padded to the same 8 rows); 157.6 ms at 4 with the dispatch cap at 4096 |
| TP8 + DP8 | does not load. `attn_tp` is 1, so the attention weights replicate: 254.5 GB/rank, and sglang's state-pool sizing goes negative before a bench flag is read |
| TP4, EP4, TP4 + DP4 | do not load. 1.5 TB over 4 ranks is 375 GB/rank |

The state-pool limit at TP8 + DCP8 is a split, not a ceiling. `--mamba-full-memory-ratio` r gives the state pool r/(1+r) of the memory left after weights; at the default the KV side holds 109 requests and the state side 60. Each request at ISL 100k costs a rank about 0.17 GB of KV rows and 0.27 GB of state (5 slots), so moving the split toward the state pool raises `--CONC` until the two meet:

| `--mem-fraction-static` | `--mamba-full-memory-ratio` | `--CONC` | step ms | tok/s |
| --- | --- | --- | --- | --- |
| 0.85 | default | 60 max | | |
| 0.85 | 1.2 | 64 | 47.2 | 1,354 |
| 0.85 | 1.5 | 72 | 49.2 | 1,464 |
| 0.85 | 1.6 | 78 max | 50.1 | 1,557 |
| 0.87 | 1.6 | 84 | 51.2 | 1,641 |
| 0.89 | 1.6 | 90 | 58.1 | 1,548 |
| 0.89 | 1.6 | 100 | 60.1 | 1,665 |

At r = 1.6 the pools meet at 78 requests; at 1.7 the KV side refuses 80. Past that, only a larger `--mem-fraction-static` adds room. Throughput stops rising near `--CONC` 84: from 84 to 90 `attn.attn_mqa_for_dcp_decode` goes from 12.7 to 18.9 ms per step, and `--mem-fraction-static` does not move it (84 reads 51.1 ms at 0.89 too). These are bench points only. A server needs headroom for its graphs and activations that the bench does not, so 0.87 and 0.89 are not shown to start a server.

DCP buys capacity, not step time. Measured at ISL 100k, the two layouts' steps are within 3% of each other at every `--CONC` both can run (1 to 16), plain TP8 the faster at 1 to 4: DCP8 shrinks the MLA kernel (4.2 to 3.3 ms per step at `--CONC` 16) and its exchange adds about as much back (the MLA `attn.<rest>` goes from 0.8 to 2.3 ms). What it adds is `--CONC` 32, which plain TP8 cannot hold. Prefill pays for it barely: one 100k request, 7 chunks, TTFT 6.81 s under DCP8 and 6.82 s without.

DCP's pools are smaller than plain TP8's (33.5 GB against 41.5 GB per rank; sglang holds back more outside them), yet hold 6.5 times the context, because each rank stores one token in eight.

The ceilings are pool arithmetic, not a policy: `--CONC x --ISL` must fit the
MLA KV pool, and the KDA state pool must hold every request's slots at any
context length. sglang reserves more than one slot per live request (five here)
and resolves how many while it sizes the pools, so the bench reads the request
count off the request pool rather than counting slots. Both pools are checked
before the batch is built and the failure names the pool that was short.

TP8 + DP2 (`attn_tp` 4) and TP8 + DP4 (`attn_tp` 2) shard the attention weights
instead of replicating them, so they are the DP layouts worth trying. Neither
has been run.

Under DP attention `--CONC` is the global batch and must divide by `--dp-size`, so a DP4 sweep starts at 4, not 1.

Under an all-to-all backend, sglang pads a decode batch up to a multiple of the attention TP size before attention runs, because the MoE region reduce-scatters over that group. At EP8 that means **a batch of 1 is a batch of 8 rows**. The padding rows point at the request pool's padding row and are given `seq_len = 1`, so they read no context. The bench sizes `--max-running-requests` for the padded count, so no run stops on it.

The padding is not free, and the attention rows of an EP report at low `--CONC` show what it costs — which the server pays too, since it is the server's own padding:

- **KDA** runs its recurrence per row, so eight rows is eight states advanced instead of one. Real work, though well under 8× at these sizes: the kernel is launch-bound at one row, so the extra rows are partly free.
- **MLA** reads the same bytes either way (`seq_lens_sum` goes from 100001 to 100008), yet `attn.attn_mqa` still scales close to the row count. At batch 1 over a long context the decode kernel does not fill the device, so its cost tracks the grid rather than the bytes, and the grid is the padded batch.

So at EP8 the attention rows at `--CONC` 1 are not comparable with TP8's; they are the cost of a batch of 8 with one real row in it. The comparison that means something is against TP8 at `--CONC` 8.

`ServerArgs` flags pass through, except that the bench forces `--disable-cuda-graph` (unless `--full-step`, since the hooks need sglang's graphs off) and `--trust-remote-code` (the checkpoint ships its config class and processor as code, so the tokenizer will not load without it), rewrites `--kv-cache-dtype auto` to `fp8_e4m3`, and refuses `--speculative-algorithm`. Ours:

| flag | default | script | meaning |
| --- | --- | --- | --- |
| `--ISL` | 4096 | both | input length per request |
| `--reps` | 50 | both | graph replays per operator |
| `--repeat` | 1 | both | timing passes, fastest wins |
| `--out` | auto | both | report path |
| `--CONC` | 1 | decode | **total** concurrency, split across `dp_size`, must divide evenly |
| `--full-step N` | 0 | decode | time N whole steps under sglang's own graphs and exit |
| `--trace` | off | decode | print one step's kernels instead of timing operators |
| `--no-kernels` | off | decode | skip the per-kernel breakdown under each hotspot row |
| `--prefix-cache-hit` | 0 | prefill | fraction of `--ISL` already cached |

DSPARK, K3's speculative decoding, is not modelled: a step would run a draft model this bench neither times nor sizes a pool for, so every row would describe the target stack while the summary claimed a step. Both scripts exit on `--speculative-algorithm` rather than load the draft weights and quietly shrink the pools every other row is measured against.

## Memory: two pools

K3 splits static memory in two and the halves are priced in different units, so the **Memory** section of a report has a `per` column and two capacities:

- **MLA KV pool** — bytes per token, and only the 24 MLA layers hold any; it is what a long ISL runs out of. Plain TP replicates it on every rank. DCP shards it along the sequence, each rank storing every `dcp_size`-th token, so the same bytes hold `dcp_size` times the context. The report counts tokens of context, not rows: under DCP that is `max_total_num_tokens × dcp_size`.
- **KDA state pool** — bytes per request, a fixed state per slot however long the request is. Neither DP, EP nor DCP shards it; only the attention-TP width, `--mamba-ssm-dtype` and `--mamba-radix-cache-strategy` change what a rank pays. This is the concurrency ceiling, and it is what a large `--CONC` runs out of.

`--mamba-full-memory-ratio` divides the static budget between them, so raising one capacity lowers the other. A run has to fit both pools; whichever binds depends on the shape, and both scripts refuse the run with the name of the pool that was short rather than letting sglang's allocator fail deep inside a batch.

## Reading a report

Rows are `kind|op`, summed as per-layer cost × layer count.

In a decode report each hotspot row is followed by the GPU kernels its replay launched, with calls and GPU time per step and the Python or dispatcher call that launched each one. They come from one eager Kineto replay per op after all timing is done; a parent's replay leaves out its timed children, so its `<rest>` lists only its own kernels. The row's summed `GPU ms/step` sits beside the graph-timed `ms/step` and runs above it where launches or a collective fall inside the eager call. ⧗ marks a collective: its GPU time includes waiting for the slowest peer and, outside a graph, a copy into the all-reduce buffer, so at TP8 + DCP8 `--CONC` 16 the rows holding one read 18–40% over their graph time. The graph-timed column is the measurement; the kernels say what it is made of.

| row | contents |
| --- | --- |
| `attn.fused_qkvg_proj` | KDA, attention TP == TP: q, k, v and the output gate in **one** GEMM |
| `attn.qkv_proj`, `attn.g_proj` | KDA under DP attention: the same projections as separate modules |
| `attn.b_proj`, `attn.f_a_proj`, `attn.f_b_proj` | KDA under DP attention only. Where the fused projection is built these exist but are replaced after weight load by a tiny GEMM over a merged `[f_a\|b]` tensor, so they never fire |
| `attn.attn` | KDA only — the conv + recurrence kernel |
| `attn.attn_mqa` | MLA's attention kernel on the absorbed path: decode, and prefill chunks past the dense threshold. On a long context it is the row that grows fastest with the batch |
| `attn.attn_mqa_for_dcp_decode` | the same kernel in a decode step under DCP, over this rank's share of the context. The exchange of partial results that follows it is in the MLA `attn.<rest>` |
| `attn.attn_mha` | MLA's attention kernel on the dense path: shallow prefill chunks only |
| `attn.o_norm` | KDA's gated RMSNorm. Folded into the decode kernel when `SGLANG_K3_KDA_FUSED_BACKEND=aiter` arms, and then absent from a decode report |
| `attn.fused_qkv_a_proj_with_mqa` | MLA: q_a and kv_a are one GEMM |
| `attn.kv_b_proj` | MLA, prefill only, dense chunks only |
| `mlp.experts` | MXFP4, 16 of 896 routed experts per token; under an a2a this row is comms + expert GEMMs |
| `mlp.routed_expert_down_proj`, `mlp.routed_expert_up_proj`, `mlp.routed_expert_norm` | the latent MoE: experts run in a 3584-wide space, not the 7168-wide hidden one |
| `mlp.shared_experts` | K3's two shared experts, a separate module rather than extra slots in the routed set |
| `self_attention_res_*`, `mlp_res_*` | the attention-residual stream, scored and normed on both sides of attention in every layer |
| `attn_res_out` | the final aggregation, once per forward: it folds the bank into the last layer's output and applies the final norm. Timed through `AttnResidual.forward` rather than through its three modules, which it reads as weights; `output_attn_res_proj`, `output_attn_res_norm` and `norm` are its children and drop out of the roll-up when it is measured |
| `layer.<rest>` | the attn-residual aggregation, completion of `o_proj`'s deferred reduction, residual adds. K3 has no `LayerCommunicator`; this is where its collectives live |

Under plain TP the MoE front is one merged GEMM over the router gate, the latent down-projection and the shared experts' `gate_up`, so those three rows are **absent** and their cost is in `mlp.<rest>`. Under an all-to-all they are three modules and get rows of their own.

On ROCm the attention-residual aggregation is one Triton kernel that reads `input_layernorm`, `post_attention_layernorm`, the two score projections and their norms as weights rather than calling them, so **all six rows are absent on every layer but layer 0** and their cost is in `layer.<rest>`. (Layer 0's pre-attention point has no banked rows to mix yet, so it applies its norm directly.) The **Coverage** section names each group of absent rows against the one kernel that took its work, rather than listing the names.

A `<rest>` row is a **difference of two measurements, not a measurement**. Error in the parent or any child lands there at full size and reversed sign, so a small one is mostly that error and can read negative.

Two sections name what a report could not measure. **Coverage**: hooked but never fired, folded into a `<rest>` row. **Not replayable**: fired but could not be timed alone; the table names where its cost landed — a `<rest>` row, which is then an upper bound, or nowhere, leaving the roll-up short.

Every row is a minimum over timed passes, so a row can read high but never low. That makes the tree its own check: a parent runs its children, so a parent measured under their sum is a disturbed measurement, not a decomposition. The report detects that and banners it above the tables rather than rendering the `<rest>` row it implies — which would be the error at full size and reversed sign, multiplied by the layer count. **On a device shared with anything else, pass `--repeat 3`**: each row then takes the fastest of three passes, and one disturbed pass stops setting a row.

`ratio` catches missing work, not mismeasured work — a residual is a subtraction, so the roll-up balances either way. It does not catch a disturbed measurement either: a parent too high and its residual too low cancel in the sum. Below 1.0 is work no row covers. **Above 1.0 is a row that costs more alone than it does in the step**, and under an all-to-all backend `mlp.experts` is that row: replayed on its own it is the only thing every rank is running, so each exchange waits out the slowest peer with no neighbouring compute to overlap, where in the step the next layer's attention covers part of the skew. Read it as an upper bound. The external check is `bench_serving`'s **median** ITL at the same shape. Against the recipe's server at TP8 + DCP8, ISL 100k and 512 output tokens, the step time came within 5% of it at `--CONC` 1 to 32 (17.5 ms against 17.9 at 1, 33.4 against 32.1 at 32). Every report carries a `bench` row in Versions, because two reports of the same shape and the same sglang can still disagree in the ranking.

In a prefill report, the growth table sums each op over both attentions. A climbing attention row is the 24 MLA layers moving under 69 flat KDA ones: a KDA layer's cost is a function of its own chunk's token count whatever the prefix behind it, because the context it reads is a fixed-size state. The per-kind table separates them.

`--prefix-cache-hit` models the MLA side exactly — to a layer, a hit and a finished chunk are the same state, KV already in the pool. On the KDA side a real hit is served by `--mamba-radix-cache-strategy`, which restores a saved state into the request's slot; here the slot holds random values instead. No chunk's shape or kernel changes, so the chunk times hold; what is outside the TTFT is the restore, a state copy per request.

`--load-format dummy` runs, but on K3 the router is what it breaks: collapsed hidden states rank every token the same way, so a layer reads 16 of 896 experts instead of the several hundred a batch spreads over a trained router. `balance_moe_routing` overrides the top-k to restore the byte count, not the distribution. The report banners it and the filename carries a `dummy` tag.

## Model structure

| item | value |
| --- | --- |
| `architectures` | `KimiK3ForConditionalGeneration`; the text stack is `language_model` (`KimiK3LinearForCausalLM` → `KimiK3LinearModel`) and the vision tower is out of scope |
| layers | 93: layer 0 dense, 1–92 MoE (`first_k_dense_replace 1`, `moe_layer_freq 1`) |
| attention | 69 KDA, 24 MLA (`linear_attn_config`; the layer lists there are one-based) |
| hidden / dense FFN / MoE intermediate | 7168 / 33792 / 3072 |
| experts | 896 routed, `top-k 16`, 2 shared; latent MoE at `routed_expert_hidden_size 3584` |
| MLA | `q_lora 1536`, `kv_lora 512`, `qk_nope 128`, `qk_rope 64`, `v_head 128`, 96 heads, nope + output gate |
| KDA | 96 heads × 128, short conv 4, full-rank gate, `gate_lower_bound -5` |
| attn residual | `attn_res_block_size 12`: a residual bank scored on both sides of every layer, a fresh block written every 12th |
| activation | `situ`, β 4.0, linear β 25.0 |
| MXFP4 | routed experts only; attention, shared experts, dense MLP and lm_head are BF16 |

| layer kind | ids | count |
| --- | --- | --- |
| dense + KDA | 0 | 1 |
| MoE + KDA | the rest | 68 |
| MoE + MLA | 3/7/11/…/91, 92 | 24 |

Counts are read off the built model, not hardcoded: whether a layer is MoE and whether it is KDA are both decisions the config only implies.

## AKLR workloads

The shapes this model runs are recorded by a run with `AKLR_ENABLE_TRACING=1`; the scenarios that have a kernel are checked in under `aklr/ops/<op>/impls/<language>/` (`aklr/ops/README.md`).

Record the **Kernel environment** table with any capture: aiter picks the SiTUv2 activation dtype from `AITER_SITUV2_A8W4` / `AITER_SITUV2_A4W4` and no argument of the call carries the choice, so the same shapes captured under a different setting are a different arithmetic.

## TODO

- [ ] split `mlp.experts` into dispatch / experts / combine under an all-to-all
- [ ] a check that can catch a wrong per-operator row; `ratio` provably cannot
- [ ] time the attn-residual score rows per depth too, for the torch fallback path where they fire (on ROCm they never do, so today they are timed once per kind and any depth error lands in `layer.<rest>`)
- [ ] under `SGLANG_K3_SP_ATTN_RES=1`, when the final aggregation's fused all-gather declines a shape the model falls back to the plain path, and `attn_res_out` then records only that fallback — the declined all-gather's own cost is in neither row
