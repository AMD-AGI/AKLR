# Plugin tutorial: replacing vLLM functions

When an optimization has to cover more than one op, a plugin hands several vLLM calls to one AKLR op. This tutorial's example is `muse_spark_moe`, which takes Muse Spark's MoE layer from the pre-expert RMSNorm to each route's expert output; the aiter `fused_moe` that vLLM calls in this layer covers only its experts.

Before you start, read the [ops tutorial](ops_tutorial.md): a plugin's op is written the same way. The SGLang plugin is covered in the [SGLang plugin tutorial](plugin_tutorial.md), and the plugin's layout and rules are in [aklr/plugins/README.md](../aklr/plugins/README.md).

## Why a plugin

vLLM computes Muse Spark's MoE layer in `MuseSpark1xMoE` (`models/muse_spark/amd/model.py`), in several calls. `fused_moe` covers only some of its steps:

| Step | What it does | `fused_moe` | `muse_spark_moe` |
|---|---|---|---|
| 1 | Routing: the FP32 router and its top-k; the pooler projects the hidden states from width 8192 to the experts' width 4096 | — | — |
| 2 | Pre-expert RMSNorm (fp32, times `norm_weight + gain_center`), each row expanded into topk rows, one per route | — | ✓ |
| 3 | Routes grouped by expert; the routes to experts this GPU does not hold give zeros | ✓ | ✓ |
| 4 | Input quantized to MXFP4 | ✓ | ✓ |
| 5 | GEMM: gate and up | ✓ | ✓ |
| 6 | Activation SiLU(gate) × up, quantized to MXFP4 | ✓ | ✓ |
| 7 | GEMM: down, giving each route's output (not multiplied by the router weights) | ✓ | ✓ |
| 8 | Reduction across GPUs, each route's post-expert norm, the router weights and the sum over the routes | — | — |
| 9 | Unpooler | — | — |

vLLM runs step 2 on its own before `fused_moe`, in `fused_rmsnorm_expand`: it writes the normalized rows out, topk copies of each, and the experts read them back. To fold the norm into the experts' loading of their input, one op has to take steps 2–7; and no single vLLM call covers them: step 2 is in `MuseSpark1xMoE._pool`, steps 3–7 in `MuseSparkAiterExperts.apply`. So the plugin replaces both calls, and defers the norm to the experts' call.

So:

- To make one op faster, write an op in `aklr/ops`.
- When the optimization crosses one op's boundary, write a plugin.

## How the plugin runs

With `AKLR_VLLM_PLUGIN=1`, each worker hooks the routes once its device is bound. From then on, each call of a replaced function first runs the route, which hands the call to the op or returns a `FallbackReason` to fall back to vLLM's code (see the [call flow](../aklr/plugins/README.md#call-flow)).

## Steps

### 1. Find the functions to replace

These two vLLM calls:

- `fused_rmsnorm_expand`, called by `MuseSpark1xMoE._pool` (`models/muse_spark/amd/model.py`): computes step 2 above and returns the normalized, expanded rows.
- `MuseSparkAiterExperts.apply`, called by `FusedMoEKernelModularImpl._fused_experts` (`models/muse_spark/amd/muse_moe_runner.py`): computes steps 3–7 above and hands each route's output to vLLM's later steps (step 8).

Their parameters in vLLM 0.33.0rc1.dev910, which the routes copy exactly, `self` included for the method:

```python
def fused_rmsnorm_expand(x, weight, eps, gain_center, input_scale, repeats): ...

def apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids, activation, global_num_experts,
          expert_map, a1q_scale, a2_scale, workspace13, workspace2, expert_tokens_meta,
          apply_router_weight_on_input): ...
```

### 2. Create the package

Create `aklr/plugins/vllm/models/muse_spark/`, named after vLLM's source directory `models/muse_spark/`. It holds the same files as an op in `aklr/ops`: `__init__.py`, `contract.py`, `reference.py`, `baseline.py`, `api.py` and `impls/`; and `_norm_expand.py`, a copy of vLLM's norm kernel, so that the baseline runs without vLLM. The package name is the op's `op_type` (`muse_spark`) and must be unique across `aklr/ops` and `aklr/plugins`.

### 3. Write the op

As in step 1. A of the ops tutorial. For this op:

- `contract.py` defines `MUSE_SPARK_MOE` (`muse_spark_moe@1.0`):
  - inputs: the pooled rows `hidden_states`, not yet normalized; the norm's `norm_weight`; each row's routes `topk_ids`; `expert_mask`, the experts this GPU holds (its last slot is vLLM's always-masked one); the experts' `w1`, `w1_scale`, `w2` and `w2_scale`; and the scalars `eps` and `gain_center`;
  - layouts: `row_major` and `shuffled_16x16`, the latter as vLLM holds the experts for aiter;
  - output: `out`, of shape `[n_tokens, topk, model_dim]`: each route's expert output, not multiplied by the router weights; the routes to experts this GPU does not hold give zeros. The router weights, the post-expert norm and the sum over the routes are vLLM's, after it.
- `reference.py`: `muse_spark_moe_reference`, the contract's computation: the RMSNorm in fp32, rounded to `hidden_states`' dtype as vLLM's `fused_rmsnorm_expand` writes it, then each route through `fused_moe`'s reference at weight 1.
- `baseline.py`: `muse_spark_moe_baseline`, what vLLM runs today for steps 2–7: `fused_rmsnorm_expand`, then the aiter `fused_moe` (per_1x32) that `MuseSparkAiterExperts.apply` calls.
- `api.py`: the entry function `@api(MUSE_SPARK_MOE)`, whose body is `...`.

### 4. Write the routes

The routes go in `api.py`, beside the op's entry function `muse_spark_moe`. Together they hand vLLM's two calls to this one op:

- The contract's inputs are the rows not yet normalized and the norm's `norm_weight`, `eps` and `gain_center`: the op computes the norm. So `_defer_norm` skips vLLM's norm and leaves its arguments to the op.
- `_experts_apply` turns vLLM's arguments into the op's and calls `muse_spark_moe`.

The first argument of `@vllm_route` is the caller, the second the replaced function; the two routes are in one `group`. Some parameters and checks are omitted below:

```python
@vllm_route(
    "vllm.models.muse_spark.amd.model.MuseSpark1xMoE._pool",
    "vllm.models.muse_spark.amd.model.fused_rmsnorm_expand",
    versions=("0.33.0rc1.dev910",),
    group="muse_spark.norm_experts",
)
def _defer_norm(x, weight, eps, gain_center, input_scale, repeats):  # exactly the replaced function's parameters
    # an input AKLR does not support
    if weight is None or input_scale or x.dtype != torch.bfloat16 or x.shape[0] == 0:
        return FallbackReason.OUT_OF_CONTRACT
    # the op declines this layer at this token count: vLLM runs its norm
    layer = id(weight)
    serves = _SERVES.get((layer, x.shape[0]))
    if serves is None and layer in _EXPERTS:
        serves = _would_serve(x, weight, eps, gain_center, layer)
    if serves is False:
        return FallbackReason.OUT_OF_CONTRACT
    # no norm: its arguments are kept on the tensor, returned as it is, for the experts' route
    setattr(x, _DEFERRED_NORM, (weight, eps, gain_center))
    _PENDING.add(layer)
    return x


def _would_serve(x, weight, eps, gain_center, layer):
    # the layer's experts as _experts_apply recorded them; None for a vLLM configuration it does not support
    if _EXPERTS[layer] is None:
        return False
    topk, ids_dtype, expert_map, w1, w1_scale, w2, w2_scale = _EXPERTS[layer]
    # asks the op, with its arguments, whether it takes the call, without computing; ids are zeros as topk_ids
    reason = muse_spark_moe.fallback_reason(
        x, weight, ids[: x.shape[0]], expert_map, w1, w1_scale, w2, w2_scale,
        eps=float(eps), gain_center=float(gain_center),
    )
    _SERVES[(layer, x.shape[0])] = reason is None
    return reason is None


@vllm_route(
    "vllm.model_executor.layers.fused_moe.modular_kernel.FusedMoEKernelModularImpl._fused_experts",
    "vllm.models.muse_spark.amd.muse_moe_runner.MuseSparkAiterExperts.apply",
    versions=("0.33.0rc1.dev910",),
    group="muse_spark.norm_experts",
)
def _experts_apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids, ...):
    deferred = getattr(hidden_states, _DEFERRED_NORM, None)
    # the layer's norm was not deferred: back to vLLM
    if deferred is None:
        return FallbackReason.OUT_OF_CONTRACT
    weight, eps, gain_center = deferred
    quant = self.quant_config
    route_output = FallbackReason.OUT_OF_CONTRACT
    _EXPERTS[id(weight)] = None
    # a vLLM configuration AKLR supports
    if expert_map is not None and activation.name == "SILU" and quant.use_mxfp4_w4a4 and ...:
        # the layer's experts, for _defer_norm to check later calls with
        _EXPERTS[id(weight)] = (topk_ids.shape[1], topk_ids.dtype, expert_map, w1, quant.w1_scale, w2, quant.w2_scale)
        # the op that fuses the norm and fused_moe
        route_output = muse_spark_moe(
            hidden_states,  # the pooled rows, not yet normalized
            weight,  # norm_weight
            topk_ids,
            expert_map,  # expert_mask
            w1,
            quant.w1_scale,
            w2,
            quant.w2_scale,
            eps=float(eps),
            gain_center=float(gain_center),
            out=output,
        )
    # the op declined: the deferred norm first, then vLLM's own apply
    if isinstance(route_output, FallbackReason):
        normalized = _original(_defer_norm)(hidden_states, weight, eps, gain_center, False, topk_ids.shape[1])
        return _original(_experts_apply)(self, output, normalized, w1, w2, topk_weights, topk_ids, ...)
    # each route's output, for vLLM's later steps (step 8)
    self._park_route_output(route_output.view_as(output))
    return None
```

When the op takes the call, it returns `out`, each route's output of shape `[n_tokens, topk, model_dim]`, which the route hands to `_park_route_output` in `output`'s shape, for vLLM's later steps (step 8); when it declines, it returns a `FallbackReason`, and the route runs the deferred norm and falls back to vLLM.

### 5. Add test inputs

Add a small Point of the op to `SMALL_POINTS` in `tests/plugins/vllm/cases.py`:

```python
PointSpec(
    contract_id="muse_spark_moe@1.0",
    axes={"n_tokens": 4, "model_dim": 2048, "topk": 2, "n_experts": 2, "inter_dim": 256, "global_experts": 5},
    dtypes={"hidden_states": "bfloat16", "norm_weight": "bfloat16", "topk_ids": "int32", "expert_mask": "int32",
            "w1": "float4_e2m1", "w1_scale": "float8_e8m0", "w2": "float4_e2m1", "w2_scale": "float8_e8m0",
            "out": "bfloat16"},
    attrs={},
    layout_variant_name="shuffled_16x16",
),
```

Its baseline runs only on gfx950, so its contract id also goes into `GFX950_CONTRACT_IDS` in the same file. Then update `tests/plugins/vllm/names.txt`:

```bash
AKLR_UPDATE_NAMES=1 pytest tests/ops/test_contracts.py
```

`tests/plugins/vllm/test_hook.py` checks every route: that the function it replaces exists in the installed vLLM with the same parameters, and that no other route replaces it. The deferred norm is an agreement between the two routes that `test_hook.py` cannot check, so it is tested on its own in `tests/plugins/vllm/test_muse_spark.py`: rows without a deferred norm fall back to vLLM, a deferred norm lost on the way raises, and the baseline's copy of the norm gives vLLM's bits.

### 6. Write the kernel and validate it

Follow steps 1. B to 4 of the ops tutorial, with this op's scenario: Muse Spark's MXFP4 MoE layer at EP8, where each GPU holds 32 of the 256 routed experts and `global_experts` is the 256 plus the always-masked slot. The script differs in the contract, the Point and the workloads, where `eps` and `gain_center` are scalars given with each workload; this op's committed kernel is written in FlyDSL:

```python
contract = ContractRegistry.default().get("muse_spark_moe@1.0")
base = PointSpec(
    contract_id="muse_spark_moe@1.0",
    axes={"n_tokens": 1, "model_dim": 4096, "topk": 8, "n_experts": 32, "inter_dim": 4096, "global_experts": 257},
    dtypes={"hidden_states": "bfloat16", "norm_weight": "bfloat16", "topk_ids": "int32", "expert_mask": "int32",
            "w1": "float4_e2m1", "w1_scale": "float8_e8m0", "w2": "float4_e2m1", "w2_scale": "float8_e8m0",
            "out": "bfloat16"},
    attrs={},
    layout_variant_name="shuffled_16x16")
workloads = [contract.workload_for(base.model_copy(update={"axes": {**base.axes, "n_tokens": t}}),
                                   {"eps": 1e-6, "gain_center": 1.0}) for t in (8, 256, 8192)]
kernel = {"language": "flydsl", "targets": [{"arch": "gfx950", "hardware": "MI355X"}],
          "entry_point": "kernel.py::run", "sources": []}
```

The schema is written beside the committed kernel of this scenario, in `aklr/plugins/vllm/models/muse_spark/impls/flydsl/muse_spark_moe_shuffled_16x16_model_dim4096_topk8_n_experts32_inter_dim4096_global_experts257/`. Where these shapes and token counts come from is in [Muse Spark's definitions](../aklr/benchmarks/models/muse_spark/DEFINITIONS.md). Validate it and open the PR as in the ops tutorial.

### 7. Run it in vLLM

Validation checks the kernel against the reference, not the routes; check the routes by running the model. The experts' route takes calls only under expert parallelism, so start vLLM with Muse Spark (MXFP4) at TP8 with `--enable-expert-parallel`:

```bash
VLLM_ROCM_USE_AITER=1 VLLM_ROCM_MUSE_MOE_REDUCE_DTYPE=bf16 VLLM_USE_BREAKABLE_CUDAGRAPH=1 AKLR_VLLM_PLUGIN=1 \
    vllm serve <Muse Spark MXFP4 checkpoint> --tensor-parallel-size 8 --enable-expert-parallel \
    --language-model-only --attention-backend ROCM_AITER_FA
```

The full launch options are in [Muse Spark's definitions](../aklr/benchmarks/models/muse_spark/DEFINITIONS.md). The log reports the hooked calls (`aklr hooks N calls of vLLM 0.33.0rc1.dev910`), and the model's replies must stay correct.

## Next

- The plugin's layout and rules for routes: [aklr/plugins/README.md](../aklr/plugins/README.md).
- Letting agents write and tune a kernel: [forge_task](forge_task.md).
