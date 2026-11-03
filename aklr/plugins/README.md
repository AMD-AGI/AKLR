# `aklr/plugins/` — replace inference-framework code with AKLR implementations

A plugin replaces a function of an inference framework (for example, an MoE layer's forward) with an AKLR implementation. When AKLR has a faster op implementation, the AKLR implementation runs; otherwise the call falls back to the original path.

## Architecture

A plugin has three parts:

- Entry and hooking: `__init__.py` and `hook.py` start AKLR and hook the routes onto the framework's functions.
- Ops: each made of a contract, a reference, a baseline and kernels, written the same way as the ops in `aklr/ops` (see `aklr/ops/README.md`).
- Routes: code that replaces a framework function and hands the call to an op; when the op does not take the call, it falls back to the framework's original implementation.

```
aklr/plugins/
├── sglang/                   the sglang plugin
│   ├── __init__.py           entry: called at sglang start-up; initializes AKLR and hooks the routes
│   ├── hook.py               @sglang_route declares a route; install() hooks the routes onto sglang's functions
│   ├── models/               mirrors sglang's srt/models/
│   │   └── deepseek_v2/      one package per sglang source file (here srt/models/deepseek_v2.py)
│   │       ├── contract.py   the op's interface: inputs, outputs, shapes, dtypes and error bounds
│   │       ├── api.py        the op's entry function, and the route that replaces the sglang function
│   │       ├── reference.py  reference implementation written from the math, used to verify kernel correctness
│   │       ├── baseline.py   sglang's original implementation, the baseline for kernel performance
│   │       └── impls/        kernel implementations, one JSON file per kernel
│   └── layers/               mirrors sglang's srt/layers/, with the same layout
├── vllm/                     planned, with the same layout as sglang/
└── aiter/                    planned
```

## Call flow

1. When sglang calls a replaced function, the route runs first.
2. The route checks sglang's runtime configuration and falls back on any configuration AKLR does not support.
3. The route passes the arguments to the op. The op selects a kernel by shape, dtype and so on; if no kernel has passed validation and is faster than sglang, it also falls back.
4. Otherwise the kernel computes the result, and the route returns it to sglang.

In addition, a route is hooked only on the sglang releases it declares, and inside a torch.compile region the call always falls back.

## Tutorial: adding an optimization

To replace an sglang function with an AKLR implementation:

1. Determine what the function computes, its inputs and outputs, and the sglang source file it is in.
2. Create a package named after that source file under `sglang/models/` (or `sglang/layers/`). The package name is the op's `op_type` and must be unique across `aklr/ops` and `aklr/plugins`.
3. Write the op: `contract.py`, `reference.py`, `baseline.py`, and the op's entry function (`@api`) in `api.py`.
4. Write the route in `api.py`. The example below is from `models/deepseek_v2/api.py`, with some parameters and helpers omitted:

   ```python
   @sglang_route("sglang.srt.models.deepseek_v2.DeepseekV2MoE.forward_normal", versions=("0.5.22",))
   def _moe_forward_normal(self, hidden_states, ...):  # the replaced function's exact parameters
       if _forward_normal_falls_back(self, ...):  # an sglang configuration AKLR does not support
           return FallbackReason.OUT_OF_CONTRACT
       out = deepseek_v2_mxfp4_moe(**_moe_arguments(self, hidden_states))  # the module's weights and input as op arguments
       if isinstance(out, FallbackReason) or not self.reduce_results:  # the op declined, or the original skips the all-reduce
           return out
       from sglang.srt.layers.moe import post_experts_all_reduce

       return post_experts_all_reduce(out)  # the original's last step, which the route must run itself
   ```

   Rules for routes:
   - A route replaces the whole original function and must return exactly what it returns: steps the op does not cover (such as the projections before and after, or the all-reduce) are done by the route, calling sglang's existing functions.
   - Shapes, dtypes, token counts and GPU models are the op's to judge; the route does not check them again.
   - When preparing the op's inputs is costly (for example, computing projections first), call `op.fallback_reason(...)` to confirm that the op takes the call before preparing them.
   - Raise exceptions for errors; return a `FallbackReason` for unsupported cases, never an exception.
   - Import sglang only inside functions: environments without sglang import `api.py` too.
5. Add a few small inputs for the op to `tests/plugins/sglang/cases.py`, for the shared tests in `tests/ops`; then run `AKLR_UPDATE_NAMES=1 pytest tests/ops/test_contracts.py` to update the op names in `tests/plugins/sglang/names.txt`.
6. Write the kernels under `impls/`; the JSON format is described in `aklr/ops/README.md`.
7. Run validation locally (next section), then open a PR.

## Correctness: validation

Correctness rests mainly on validation (`python -m aklr validate`), which has three stages. In CI, audit always runs and bench needs the PR to carry the `ci:gpu` label (a push to main always qualifies); CI does not run e2e.

1. audit: checks that kernel code follows the rules, for example that it keeps no state between calls and does not modify read-only inputs.
2. bench: for each kernel, draws several sets of inputs from the contract, compares the results with the reference for correctness and with the baseline for performance, and writes the results back to the JSON. At runtime, only kernels that passed and are faster than sglang are used.
3. e2e: runs the whole model with AKLR on and off. For now it compares performance only; a correctness comparison will be added, and route correctness will then be verified at this stage.

Unit tests come in two kinds:

- Ops: the shared tests in `tests/ops`, using the inputs added in step 5 of the tutorial.
- The plugin framework: `tests/plugins/sglang/test_hook.py` covers hooking, version checks, fallback, and exiting the process when start-up fails, and automatically checks every route: its target exists, has the same parameters, and is not replaced by another route.

## Enabling the plugin and upgrading sglang

- With AKLR installed, sglang loads the plugin at start-up; routes are hooked only when `AKLR_SGLANG_PLUGIN=1`. `AKLR_DISABLE=<contract id,...>` disables the listed ops; `AKLR_DISABLE=1` disables AKLR entirely.
- Only official sglang releases are supported. To upgrade, install the new release and run `test_hook.py`: it lists the routes whose parameters changed or that do not declare the new release yet. Adapt each route, pass the whole-model tests, then add the new release to each route's `versions`.
