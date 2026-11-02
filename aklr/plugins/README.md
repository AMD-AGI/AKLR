# `aklr/plugins/` — route framework layers to AKLR ops

A plugin wraps a framework method so that its call reaches a AKLR op: a measured kernel
serves it, or the framework's own code runs.

## sglang

sglang calls the entry point `aklr` of `sglang.srt.plugins` (`aklr.plugins.sglang:register`)
once per process. `register()` starts AKLR once torch.distributed is up and, unless
`AKLR_DISABLE=1`, proxies aiter's routed entry points (`load_routes()`). With
`AKLR_SGLANG_PLUGIN=1` (`0` or unset: off; another value is an error), unless
`AKLR_DISABLE=1`, it also imports each plugin op package's `api.py`, registers
the model backends (`BACKENDS`), hooks the routes' targets (`install()`) and names sglang's
TP group (its gloo group) as the peer group of kernels that sum over the ranks
(`peer_memory.set_group`). `AKLR_DISABLE` also takes contract ids (`name@major.minor`, those
ops off, plugin ops included), the namespace `aklr.plugins` (the whole plugin) and backend
names (one monokernel spec, e.g. `aklr_deepseek_v41_dspark_draft`).

A model the plugin does not know, an sglang outside a backend's tested versions, a backend
that fails to register and a setup a spec refuses all keep sglang's own code, with one
warning naming why (on TP rank 0 where the ranks agree first); nothing else changes.

```
aklr/plugins/sglang/
  __init__.py           register()
  hook.py               @sglang_route, install()
  layers/<family>/      shared layers, after sglang's srt/layers (norms/: the one to copy)
  models/<model file>/  one model file's layers, after srt/models
  speculative/<file>/   one speculative worker file's steps, after srt/speculative
```

An op package is laid out as in `aklr/ops/README.md`, with a baseline of its own:

| file | holds |
|---|---|
| `contract.py` | the `OpContract`s, in `CONTRACTS` |
| `api.py` | the `@api` functions and the `@sglang_route` functions that reach them |
| `reference.py` | `<contract>_reference`: what the contract says, in torch, in fp32, not as sglang's code does it (sglang may be wrong); it quantizes only where the checkpoint's scheme does (an fp8 linear's input). Kernels are checked against it. |
| `baseline.py` | `<contract>_baseline`: what sglang runs today. Kernels are timed against it. |
| `impls/` | the kernels' schemas |

A package's last name is its `op_type`, unique across `aklr/ops` and `aklr/plugins`.

## Routes

```python
@sglang_route(
    "sglang.srt.layers.layernorm.RMSNorm.forward_aiter", versions=">=0.5.22,<0.5.23"
)
def _rms_norm_forward_aiter(
    self, x, residual=None, post_residual_addition=None, quant_linear=None
): ...
```

- The target is a sglang method's dotted path. The route runs in its place (an AROUND hook)
  and takes its parameters: names, order and defaults.
- A route only translates: the layer's state becomes the op's arguments, and the route
  finishes the layer as sglang does. It returns a `FallbackReason`, and sglang's method runs,
  only for what the op cannot express; whether a kernel serves the call is for the contract
  and the schemas. An error raises: nothing in AKLR falls back on one. A route whose input
  costs work to form (a sum over the ranks) asks the op first, `op.fallback_reason(...)` with
  a tensor shaped as that input, and forms it only once the op serves.
- `versions` is a PEP 440 range with an upper bound (else the import fails): the sglang
  versions the route was tested on, a dev or rc build counting as its release. A changed
  sglang API gets a new route. Unless exactly one route fits the running sglang, the target
  warns once and runs sglang's own code.
- Import sglang inside functions, so that bench and tests load the ops without it, and read
  sglang's runtime config in the route: sglang publishes it after `register()`.
- `@sglang_route(target, versions, around=True)` wraps a step instead of replacing it: the route
  is called as `route(orig, *args, **kwargs)`, runs sglang's method itself (to wrap its arguments
  or its result) and counts nothing. For lifecycle steps (load, after-load setup) and steps a
  served forward skips or replaces, not for layers an op serves.
- A route may return `RouteFallback(reason, detail)` instead of a `FallbackReason`: sglang's
  method runs and the call is counted under `reason` (`AKLR_EXPORT_DIR`'s `route_calls_*.json`).

## Monokernel backends

`models/deepseek_v4/` is not an op package: it registers sglang monokernel specs (backend `aklr`,
whole-layer kernels for DeepSeek-V4.1-Flash decode) through sglang's monokernel plugin API; an
sglang without it runs the model itself. See `docs/sglang_plugin_dsv41flash.md`.

On sglang 0.5.22 (`VERSIONS`):

1. `load_weights` of the target and its DSpark draft: the spec keeps copies of the checkpoint
   tensors it needs before sglang converts them.
2. After load, before the KV cache is sized (sglang's `install_monokernel_forward`): every TP rank
   checks the server (features, TP size, dtype, decode attention backend, server args, HF config,
   sglang env) and the model, the ranks agree, and the spec builds the widths the server's decode /
   verify steps need. The draft builds only over a built target. A refusal keeps sglang's forward
   (`runtime_routes.py` turns the native runtime's error into that), with the ranks' reasons in one
   warning; a build that fails raises, as the ranks may stand in different collectives.
3. `forward` of both models: a DECODE or TARGET_VERIFY step whose (batch size, tokens per request)
   was built runs the monokernels; any other step runs sglang's, counted with its reason.

`tests/plugins/test_routes.py` (a GPU test, on the installed sglang) checks that every route's
target exists and that the route binds the arguments sglang calls the method with.
