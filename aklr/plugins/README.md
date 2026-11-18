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
├── vllm/                     the vLLM plugin (AKLR_VLLM_PLUGIN=1)
│   ├── __init__.py           entry: hooks the routes in each worker once its device is bound
│   ├── hook.py               @vllm_route wraps the function a vLLM function looks up at run time
│   ├── warmup.py             compiles the AKLR kernels vLLM's own warmup does not reach
│   ├── stack.py              runs each vLLM step from a CPython frame chunk of its own
│   ├── layers/               routes from vLLM's layers to the ops of aklr/ops, mirrors vLLM's model_executor/layers/
│   └── models/               routes from the model files vLLM runs uncompiled, mirrors vLLM's models/
│       └── muse_spark/       Muse Spark on ROCm (vLLM's models/muse_spark/amd)
└── aiter/                    AKLR on aiter's side: aiter's routed calls turned into AKLR op calls
    └── provider/             AKLR as a kernel plugin of aiter: one module per op family selects the calls of aiter's routed ops
```

## aiter's kernel plugin interface

aiter routes some of its public functions through its kernel plugin interface (`aiter.plugin`, RFC ROCm/aiter#7240): before running its own kernel, aiter asks each enabled provider whether it serves the call. `aklr/plugins/aiter/provider` is AKLR's provider, declared by the `aiter.plugins` entry point. It replaces no function: aiter keeps the call, and the provider answers from AKLR's dispatch table without running anything.

- No route of a plugin wraps an aiter function aiter routes, or a framework function that only calls one: sglang and vLLM reach AKLR on those ops only through this provider, so one call is offered to AKLR once. Where the provider needs what only the framework knows, a route that runs nothing notes it for the provider (sglang's paged top-k calls `provider.mla.note_target_verify`). aiter functions aiter does not route (its Triton `unified_attention`) keep their framework routes.
- Enabling: with `AKLR_AITER_PLUGIN=1` the sglang and vLLM plugins call `aklr.plugins.aiter.provider.enable()` at start-up, which puts AKLR first among the providers aiter asks and keeps the others; another host calls it itself. aiter's `AITER_PLUGINS=0` turns all routing off. An aiter without the interface leaves its ops to aiter.
- Selecting: one selector per aiter op, registered with `@selector("<aiter op>", version=...)` at the version of aiter's contract. It unpacks the call by aiter's parameter names, declines what the AKLR contract does not take, and asks the op's api `serving(...)`: the kernel `dispatch` would run, or the `FallbackReason`, counted and traced as a fallback. Its `run` takes aiter's parameters exactly, since aiter checks them before the call, and returns what aiter's op returns.
- Not falling back: once `run` starts, aiter does not fall back, so `run` asks the same table `select` asked and raises if it fell back. Inside a CUDA graph capture, a Point never run outside one is declined, so aiter's kernel is captured.
- Validation: AKLR's baselines call aiter's functions, routed ones included, so validate runs with no provider enabled (it never calls `enable()`).

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

In CI, audit always runs, and bench needs the PR to carry the `ci:gpu` label (a push to main always qualifies). Validation does not run the model, so it does not check routes: check a route by running the model with the plugin on (step 7 of the plugin tutorial).

Unit tests:

- The small inputs in `tests/plugins/sglang/cases.py` feed the shared tests in `tests/ops`.
- `tests/plugins/sglang/test_hook.py` covers hooking, version checks, fallback, and exiting the process when start-up fails, and automatically checks every route: its target exists, has the same parameters, and is not replaced by another route.

## Enabling the plugin and upgrading SGLang

With AKLR installed, SGLang loads the plugin at start-up; routes are hooked only when `AKLR_SGLANG_PLUGIN=1`. `AKLR_DISABLE=<contract id,...>` disables the listed ops; `AKLR_DISABLE=1` disables AKLR entirely.

To see which routes and ops a run actually reaches, set `AKLR_TRACE=plugin` (the vLLM and sglang routes), `AKLR_TRACE=op` (every op dispatch, including the ops aiter calls) or `AKLR_TRACE=all` (both). Each process logs each distinct call once, served or fallen back:

```
[aklr plugin] pid=<pid> framework=<vllm|sglang> target=<hooked function> route=<route function> result=<served|fallback> reason=<FallbackReason|->
[aklr op] pid=<pid> contract=<contract id> scenario=<schema name|-> point=<axes|-> result=<served|fallback> kernel=<language:schema name:hash|-> reason=<FallbackReason|->
```

A vLLM route declared for several names of one function (aiter's compatibility module paths) shares one hook, so its `target` lists them all, comma-separated.

- The ops AKLR actually ran: `grep '\[aklr op\].*result=served'`.
- The calls AKLR did not serve, and why: `grep 'result=fallback'`, then read `reason`.

A CUDA graph replay runs no Python and logs nothing: the graph replays the kernels its capture logged.

Only official SGLang releases are supported, and the development build a model's image ships (Kimi-K3.1-Pro's `0.5.24.dev809`): a route declared only for such builds is tested on that image, not on CI's release. To upgrade:

1. Install the new release and run `test_hook.py`. It lists the routes whose parameters changed or that do not declare the new release yet.
2. Adapt each route, and check it by running the model (step 7 of the plugin tutorial).
3. Add the new release to each route's `versions`.
