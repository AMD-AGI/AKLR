# `aklr/plugins/` — route framework layers to AKLR ops

A plugin wraps a framework method so that its call reaches a AKLR op: a measured kernel
serves it, or the framework's own code runs.

## sglang

sglang calls the entry point `aklr` of `sglang.srt.plugins` (`aklr.plugins.sglang:register`)
once per process. `register()` starts AKLR once torch.distributed is up and, unless
`AKLR_DISABLE=1`, proxies aiter's routed entry points (`load_routes()`). With
`AKLR_SGLANG_PLUGIN=1` (`0` or unset: off; another value is an error), unless
`AKLR_DISABLE=1`, it also imports each plugin op package's `api.py`, hooks its routes'
targets (`install()`) and names sglang's TP group (its gloo group) as the peer group of kernels
that sum over the ranks (`peer_memory.set_group`). `AKLR_DISABLE=<contract ids>` turns those
ops off, plugin ops included.

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
