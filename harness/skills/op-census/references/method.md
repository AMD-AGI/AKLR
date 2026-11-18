# How a census is made

## A point, bound as InferenceX submits it

`inferencex.bind` runs InferenceX's own code inside its checkout: `infx.matrix.generate test-config` gives each
config key's matrix rows (one per arm and concurrency); `infx.srt_slurm.generate.bound_variant` composes the recipe
file with InferenceX's shared AgentX block and selects the variant for the row's concurrency; and
`synthetic_acceptance.build_overrides` lists the `srtctl apply` overrides that set the synthetic acceptance an AgentX
speculative run forces. `apply_overrides` applies them. The result is the recipe the cluster runs.

`serving.server_command` turns `roles.agg.args` into flags the way `srtctl` does for a single-node aggregated worker:
`python3 -m sglang.launch_server --model-path <checkpoint>` or `vllm serve <checkpoint>`; `true` is a bare flag,
`false` and `null` are omitted, a mapping is JSON. Only local paths (checkpoint, tokenizer, draft model) and
`--host`/`--port` change. `test_serving.py` checks this against the AgentX recipes AKLR reviewed
(`benchmarks/agentx/recipes`): same flags, same server environment, same client environment.

For vLLM the runner adds `--profiler-config` (torch profiler, no stacks, no shapes), without which vLLM does not serve
`/start_profile`. Nothing else is added.

## The run

1. Server in the config's image, `--cleanenv`, the recipe's environment, the point's GPUs.
2. `/v1/models` lists the served name; one chat request returns.
3. The AgentX client: InferenceX's `python -m infx.bench agentic` with the matrix environment, the recipe's
   `benchmark.env` and the runtime fields InferenceX sets; `DURATION` covers the capture.
4. The client's warmup primes every trajectory's cache; its profiling phase replays them at the point's
   concurrency. The runner waits for AIPerf's `Credit phase start: ... profiling`, then `--settle-s`.
5. Windows: `/start_profile`, `--window-s` seconds, `/stop_profile`; the server's running and queued request
   gauges are logged at both edges. SGLang writes one trace per rank into the window's directory; vLLM writes into
   its profiler directory and the runner moves them.
6. Client and server stop by process group; the GPUs must hold no memory before the next point.

## From a trace to a census

Both servers name each forward step with a CPU span while profiling: SGLang `step[<STAGE> bs=.. toks=..]`
(with detailed annotations, `c_`/`g_` aggregates), vLLM `execute_context_<n>(<t>)_generation_<m>(<u>)`. A GPU
activity (kernel, memcpy, memset) carries the correlation id of the runtime call that launched it; that call lies
inside a step span on its thread. This ties graph-replayed kernels (one `hipGraphLaunch`) and work an overlap
scheduler queued early to the right step, which GPU timestamps alone cannot. A launch after a span and before the
next one on its thread (vLLM samples and drafts in `sample_tokens`) is that step's tail; a launch before a thread's
first step is `(no step)`; an activity whose launch is not in the trace is `(no launch record)`.

Phases: SGLang EXTEND is prefill; DECODE and VERIFY are decode; MIXED is mixed. A draft model's step (`DRAFT`) is a
draft decode (single-query, `g_`) or a draft extend (context-shaped, `c_`); a draft extend has the phase of the
target step before it (after EXTEND: the draft side of a prefill; after VERIFY: of a decode round). vLLM steps with
only context requests are prefill, only generation requests decode, both mixed: a mixed step's kernels serve both
and are not split.

Shares are of summed activity time per phase, per rank (a sum over ranks divided by their number). `per_rank` keeps
each rank's phase time with and without communication.

Host-cache traffic launched after a step's span and before a prefill step (the new request's cached prefix loaded
back, pages written back to make room for it) is moved to that prefill step as `/before`: left in the previous
step's tail it made Qwen3.8's c80 decode look 3.8% heavier. `checks.cache_traffic_moved_share` says how much moved.

A single call more than 50 times its kernel's median and longer than 1 s is a GPU stall (seen while other servers
on the host loaded weights), not work: it is listed under `stalls` and left out. A point with stalls is rerun.

## Classes

`classify.RULES` maps a kernel name to an operator class, first match wins; the order is part of the rule (a MoE
GEMM is `moe`, not `gemm`; a whole decoder layer in one launch is `fused_decoder_layers`). A name that is only a
DSL's default (TileLang's `kernel_kernel`) or a compiler's serial name (`triton_tem_fused_..._0`) is attributed per
deployment in `classify.ATTRIBUTED`, with the source file it comes from. Check the rules against the model: calls
per step of each class divided by the layers that run it should be whole numbers (attention per attention layer,
router per MoE layer); a class whose count matches no layer count holds a misplaced kernel.

## Load

An agentic client's load varies within and between runs: the decode batch size and the prefill lengths a window
catches change the mix (a batch-1 decode step may run a fused decoder-layer kernel a larger batch does not).
`ops_by_load` keeps the classes per decode batch size and per prefill token range (rounded up to a power of two),
and `compare` judges a phase within those buckets when the phase as a whole differs. `inventory` puts every point
of a deployment side by side (classes by concurrency, every kernel any point ran, all of them in the CSV).
