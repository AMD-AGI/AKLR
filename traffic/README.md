# Traffic

Recorded serving runs that `bench` uses to build realistic kernel inputs.

## Why

Dispatch selects kernels by their `bench` speedups, so those speedups should match what users get. By default,
`bench` draws each input tensor independently, at sizes the schema chose. Real calls depend on the traffic and
the server: how many requests a step batches, how long each sequence is, which rows share KV pages, and how
wide the server pre-allocated its tables. A kernel's latency can depend on any of these.

For example, a server that allows a 1M-token context allocates 1M-wide tables even when its sequences are
much shorter. A kernel that splits its work by table width rather than sequence length looks fast on synthetic
tables sized to the sequences, and slow in serving.

## How

1. **Record:** A recording holds a serving run's startup settings and the requests of every forward step.
2. **Configure:** `schemas.json` maps each schema to the recordings it runs against, and sets how many
   calls a workload is measured on (`samples`, for every schema unless its entry sets its own).
3. **Sample:** For each workload of a listed schema, `bench` takes the recorded calls with the workload's
   dispatch key, orders them by work, splits them into `samples` parts of equal count, and takes each
   part's median call. Each call sets the server-allocated sizes and the request-metadata tensors
   (sequence lengths, page tables, sequence start offsets); `bench` draws the other inputs as usual.
4. **Measure:** Each call is checked and timed on its own. The workload's speedup is that of the calls'
   total time: in each round, the baseline's times summed over the calls against the kernel's.
5. **Fallback:** If no recorded call has the workload's key, `bench` draws all inputs as before.

The thresholds and the report stay the same; one call (`samples: 1`) measures a workload as often as before.

## Files

- `<name>.traffic.jsonl.xz`: a recording (`xzcat` to read).
- `schemas.json`: the recordings each schema uses, and `samples`.

## Recordings

| Recording | Run |
| --- | --- |
| `glm52-fp8-tp8-mi355x-agentx-c10` | AgentX at concurrency 10 for 0.26 h: `zai-org/GLM-5.2-FP8`, TP8 on 8x MI355X, SGLang 0.5.19, EAGLE with 6 draft tokens. 26,714 steps, 26,160 of them decode. |
