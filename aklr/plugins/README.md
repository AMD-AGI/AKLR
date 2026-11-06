# `aklr/plugins/` — replace inference-framework code with AKLR implementations

A plugin replaces a function of an inference framework (for example, an MoE layer's forward) with an AKLR implementation. When AKLR has a faster op implementation, the AKLR implementation runs; otherwise the call falls back to the original path.

To add an optimization, follow the [plugin tutorial](../../docs/plugin_tutorial.md). This page is the reference: the layout, the call flow, the rules for routes, validation, and enabling and upgrading.

## Architecture

A plugin has three parts:

- Entry and hooking: `__init__.py` and `hook.py` start AKLR and hook the routes onto the framework's functions.
- Ops: each made of a contract, a reference, a baseline and kernels, written the same way as the ops in `aklr/ops` (see `aklr/ops/README.md`).
- Routes: code that replaces a framework function and hands the call to an op; when the op does not take the call, it falls back to the framework's original implementation.

```
aklr/plugins/
├── sglang/                   the SGLang plugin
│   ├── __init__.py           entry: called at SGLang start-up; initializes AKLR and hooks the routes
│   ├── hook.py               @sglang_route declares a route; install() hooks the routes onto SGLang's functions
│   ├── models/               mirrors SGLang's srt/models/
│   │   └── deepseek_v2/      one package per SGLang source file (here srt/models/deepseek_v2.py)
│   │       ├── contract.py   the op's interface: inputs, outputs, shapes, dtypes and error bounds
│   │       ├── api.py        the op's entry function, and the route that replaces the SGLang function
│   │       ├── reference.py  reference implementation written from the math, used to verify kernel correctness
│   │       ├── baseline.py   SGLang's original implementation, the baseline for kernel performance
│   │       └── impls/        kernel implementations, one JSON file per kernel
│   └── layers/               mirrors SGLang's srt/layers/, with the same layout
├── vllm/                     planned, with the same layout as sglang/
└── aiter/                    planned
```

## Call flow

1. When SGLang calls a replaced function, the route runs first.
2. The route checks SGLang's runtime configuration and falls back on any configuration AKLR does not support.
3. The route passes the arguments to the op. The op selects a kernel by shape, dtype and so on; if no kernel has passed validation and is faster than SGLang, it also falls back.
4. Otherwise the kernel computes the result, and the route returns it to SGLang.

In addition, a route is hooked only on the SGLang releases it declares, and inside a torch.compile region the call always falls back.

## Rules for routes

- Declare a route with `@sglang_route("<target>", versions=(...))` on a function with the target's exact parameters; `versions` lists the SGLang releases it was tested on.
- A route replaces the whole original function and must return exactly what it returns: steps the op does not cover (such as the projections before and after, or the all-reduce) are done by the route, calling SGLang's existing functions.
- Shapes, dtypes, token counts and GPU models are the op's to judge; the route does not check them again.
- When preparing the op's inputs is costly (for example, computing projections first), call `op.fallback_reason(...)` to confirm that the op takes the call before preparing them.
- Raise exceptions for errors; return a `FallbackReason` for unsupported cases, never an exception.
- Import `sglang` only inside functions: environments without SGLang import `api.py` too.
- A plugin package's name is its op's `op_type` and must be unique across `aklr/ops` and `aklr/plugins`.

## Validation and tests

Validation (`python -m aklr validate`) checks each kernel in two stages:

- audit: checks the kernel code against the rules, for example that it keeps no state between calls and does not modify read-only inputs.
- bench: draws several sets of inputs from the contract, and compares the kernel's results with the reference for correctness and with the baseline for performance. After a PR merges, CI writes these results into its JSON files; at runtime, only kernels that passed and are faster than SGLang are used.

CI runs audit, bench and e2e checks to verify kernel correctness, performance and model accuracy.

Unit tests:

- The small inputs in `tests/plugins/sglang/cases.py` feed the shared tests in `tests/ops`.
- `tests/plugins/sglang/test_hook.py` covers hooking, version checks, fallback, and exiting the process when start-up fails, and automatically checks every route: its target exists, has the same parameters, and is not replaced by another route.

## Enabling the plugin and upgrading SGLang

With AKLR installed, SGLang loads the plugin at start-up; routes are hooked only when `AKLR_SGLANG_PLUGIN=1`. `AKLR_DISABLE=<contract id,...>` disables the listed ops; `AKLR_DISABLE=1` disables AKLR entirely.

Only official SGLang releases are supported. To upgrade:

1. Install the new release and run `test_hook.py`. It lists the routes whose parameters changed or that do not declare the new release yet.
2. Adapt each route, and check it by running the model (step 7 of the plugin tutorial).
3. Add the new release to each route's `versions`.
