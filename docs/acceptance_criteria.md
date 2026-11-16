# Acceptance criteria for kernels and plugins

[中文](acceptance_criteria.zh.md)

Your kernel PR or plugin PR must meet these criteria:

| Criterion | Applies to | Requirement | How it is checked |
|---|---|---|---|
| Kernel | every kernel file | at least one workload is 5% or more faster than the baseline | CI checks it automatically |
| End to end | a new plugin route | the model is 2% or more faster on a workload you name | you report it in the PR |

## Kernel criterion: 5%

- CI measures each workload and writes its speedup over the baseline into the kernel file.
- A kernel file can be merged when at least one of its workloads is correct and 5% or more faster. Otherwise
  validation fails with `no_qualifying_workload`.
- At run time, AKLR uses the kernel only for workloads that are 5% or more faster. Other workloads run the
  original code.
- The baseline is defined in the op's `baseline.py`. For ops in `aklr/ops/`, it is usually the AITER op. For plugin
  ops, it is the original SGLang or vLLM code.

## End-to-end criterion: 2%

A plugin route replaces a function in SGLang or vLLM. CI checks the kernel, but it does not run the model. So you
measure the gain on the whole model and report it in the PR.

### When to report it

| PR | End-to-end data |
|---|---|
| Adds a route (`@sglang_route`, `@vllm_route` or `@sglang_call_route`) | required |
| Makes an existing route handle more cases, such as a new branch or a new serving configuration | required |
| Adds or changes kernels or workloads of an existing op or route | recommended, not required |
| Adds an op to `aklr/ops/` | recommended, not required |

### What to report

Put this in the `End to end` section of the PR description:

- **Setup**: the model, the GPU type and count, and the workload (input and output lengths, concurrency).
- **Result**: run with this PR's ops turned off (`AKLR_DISABLE=<contract id>,...`) and turned on, with everything
  else the same. Give the throughput or latency of both, and the gain.
- **Commands**: the commands that start the server and run the benchmark.
