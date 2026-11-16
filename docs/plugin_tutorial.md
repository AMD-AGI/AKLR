# Plugin tutorial: replacing an SGLang function

When an optimization has to cover more than one op, a plugin replaces a whole SGLang function with an AKLR op. This tutorial's example is `deepseek_v2_mxfp4_moe`, which takes GLM-5's whole MoE layer: the layer whose experts the [ops tutorial](ops_tutorial.md) optimizes with `fused_moe`.

Before you start, read the ops tutorial: a plugin's op is written the same way. The plugin's layout and rules are in [aklr/plugins/README.md](../aklr/plugins/README.md).

## Why a plugin

SGLang computes GLM-5's MoE layer in `DeepseekV2MoE.forward_normal` (`srt/models/deepseek_v2.py`), in several calls. `fused_moe` covers only some of its steps:

| Step | What it does | `fused_moe` | `deepseek_v2_mxfp4_moe` |
|---|---|---|---|
| 1 | Router GEMM: each token's scores for the 256 routed experts | — | ✓ |
| 2 | Expert selection: sigmoid plus the correction bias, the top 8, their weights renormalized and multiplied by 2.5 | — | ✓ |
| 3 | The shared expert appended as a 9th expert at weight 1 | — | ✓ |
| 4 | Tokens grouped by expert | ✓ | ✓ |
| 5 | Input quantized to MXFP4 | ✓ | ✓ |
| 6 | GEMM: gate and up | ✓ | ✓ |
| 7 | Activation SiLU(gate) × up, quantized to MXFP4 | ✓ | ✓ |
| 8 | GEMM: down | ✓ | ✓ |
| 9 | Weighted sum over each token's experts | ✓ | ✓ |
| 10 | All-reduce across GPUs | — | — |

SGLang runs steps 1–3 as separate calls before `fused_moe`. In decode, where a call has few tokens, their launches and the intermediate results they write and read back take a large share of the layer's time. To optimize them as well, one op has to take the whole layer; since no single SGLang call covers it, the plugin replaces the SGLang function itself.

So:

- To make one op faster, write an op in `aklr/ops`.
- When the optimization crosses one op's boundary, write a plugin.

## How the plugin runs

With `AKLR_SGLANG_PLUGIN=1`, each call of `DeepseekV2MoE.forward_normal` first runs the route, which hands the call to the op or falls back to SGLang's code (see the [call flow](../aklr/plugins/README.md#call-flow)).

## Steps

### 1. Find the function to replace

`DeepseekV2MoE.forward_normal` in SGLang's `srt/models/deepseek_v2.py` computes steps 1–10 above and returns the layer's output, summed over the GPUs when `self.reduce_results` is set. Its parameters in SGLang 0.5.22, which the route copies exactly:

```python
def forward_normal(self, hidden_states, gemm_output_zero_allocator=None, input_ids=None, input_ids_global=None,
                   skip_shared_experts=False, num_token_non_padded=None, use_vision_topk=False,
                   return_moe_output=False):
```

### 2. Create the package

Create `aklr/plugins/sglang/models/deepseek_v2/`, named after SGLang's source file. It holds the same files as an op in `aklr/ops`: `__init__.py`, `contract.py`, `reference.py`, `baseline.py`, `api.py` and `impls/`. The package name is the op's `op_type` and must be unique across `aklr/ops` and `aklr/plugins`.

### 3. Write the op

As in step 1. A of the ops tutorial. For this op:

- `contract.py` defines `DEEPSEEK_V2_MXFP4_MOE` (`deepseek_v2_mxfp4_moe@1.0`):
  - inputs: what `forward_normal` reads, namely `hidden_states`; the router's `router_weight` and `correction_bias`; the experts' `w1`, `w1_scale`, `w2` and `w2_scale`, with the shared expert in the last slot (`n_experts = n_routed + 1`); and the scalar `routed_scaling_factor`;
  - attrs: the routing settings `topk`, `num_expert_group`, `topk_group`, `renormalize` and `scoring_func`;
  - output: `out`, the layer's output on this GPU, before the all-reduce.
- `reference.py`: `deepseek_v2_mxfp4_moe_reference`, the contract's computation in fp32.
- `baseline.py`: `deepseek_v2_mxfp4_moe_baseline`, what SGLang runs today for steps 1–9: the router GEMM, AITER's grouped top-k, SGLang's shared-expert append and AITER's `fused_moe`. A kernel has to beat this whole path, not `fused_moe` alone.
- `api.py`: the entry function `@api(DEEPSEEK_V2_MXFP4_MOE)`, whose body is `...`.

### 4. Write the route

The route goes in `api.py`, beside the op's entry function. Some parameters are omitted below:

```python
@sglang_route("sglang.srt.models.deepseek_v2.DeepseekV2MoE.forward_normal", versions=("0.5.22",))
def _moe_forward_normal(self, hidden_states, ...):  # exactly the replaced function's parameters
    # an SGLang configuration AKLR does not support
    if _forward_normal_falls_back(self, ...):
        return FallbackReason.OUT_OF_CONTRACT
    # the op for the experts' format, with the module's weights and input as its arguments
    op, arguments = _moe_call(self, hidden_states)
    out = op(**arguments)
    # the op declined, or the original skips the all-reduce here
    if isinstance(out, FallbackReason) or not self.reduce_results:
        return out
    # the original's last step (step 10), which the route runs itself
    from sglang.srt.layers.moe import post_experts_all_reduce

    return post_experts_all_reduce(out)
```

It calls two helpers:

- `_forward_normal_falls_back` is true for SGLang configurations the op does not support: some of the call's own arguments (`skip_shared_experts`, `return_moe_output`, ...), graph modes and expert-parallel settings (checks shared by the MoE routes, in `models/moe_blocks.py`), and experts held in another format.
- `_moe_call` picks the op for the experts' format, `deepseek_v2_mxfp4_moe` for Quark's MXFP4 or `deepseek_v2_fp8_moe` for fp8 at a scale per 128 x 128 block, and takes its arguments from the module: `hidden_states`, `mlp.gate.weight` as `router_weight`, the experts' weights and scales, and the routing settings.

Shapes, dtypes, token counts and the GPU are left to the op. All the rules for routes are in [aklr/plugins/README.md](../aklr/plugins/README.md#rules-for-routes).

### 5. Add test inputs

Add a small Point of the op to `SMALL_POINTS` in `tests/plugins/sglang/cases.py`:

```python
PointSpec(
    contract_id="deepseek_v2_mxfp4_moe@1.0",
    axes={"n_tokens": 4, "model_dim": 256, "n_routed": 8, "inter_dim": 256},
    dtypes={"hidden_states": "bfloat16", "router_weight": "bfloat16", "correction_bias": "float32",
            "w1": "float4_e2m1", "w1_scale": "float8_e8m0", "w2": "float4_e2m1", "w2_scale": "float8_e8m0",
            "out": "bfloat16"},
    attrs={"topk": 8, "num_expert_group": 1, "topk_group": 1, "renormalize": True, "scoring_func": "sigmoid"},
    layout_variant_name="shuffled_16x16",
),
```

Its baseline runs only on gfx950, so its contract id also goes into `GFX950_CONTRACT_IDS` in the same file. Then update `tests/plugins/sglang/names.txt`:

```bash
AKLR_UPDATE_NAMES=1 pytest tests/ops/test_contracts.py
```

The route needs no test of its own: `tests/plugins/sglang/test_hook.py` checks every route, that its target exists in the installed SGLang with the same parameters and that no other route replaces it.

### 6. Write the kernel and validate it

Follow steps 1. B to 4 of the ops tutorial, with this op's scenario: GLM-5's MXFP4 MoE layer at TP8. The script differs in the contract, the Point and the workloads, where `routed_scaling_factor` is a scalar given with each workload; this op's committed kernels are written in Triton:

```python
contract = ContractRegistry.default().get("deepseek_v2_mxfp4_moe@1.0")
base = PointSpec(
    contract_id="deepseek_v2_mxfp4_moe@1.0",
    axes={"n_tokens": 1, "model_dim": 6144, "n_routed": 256, "inter_dim": 256},
    dtypes={"hidden_states": "bfloat16", "router_weight": "bfloat16", "correction_bias": "float32",
            "w1": "float4_e2m1", "w1_scale": "float8_e8m0", "w2": "float4_e2m1", "w2_scale": "float8_e8m0",
            "out": "bfloat16"},
    attrs={"topk": 8, "num_expert_group": 1, "topk_group": 1, "renormalize": True, "scoring_func": "sigmoid"},
    layout_variant_name="shuffled_16x16")
workloads = [contract.workload_for(base.model_copy(update={"axes": {**base.axes, "n_tokens": t}}),
                                   {"routed_scaling_factor": 2.5}) for t in (32, 128, 256)]
kernel = {"language": "triton", "targets": [{"arch": "gfx950", "hardware": "MI355X"}],
          "entry_point": "kernel.py::run", "sources": []}
```

The schema is written beside the committed kernels of this scenario, in `aklr/plugins/sglang/models/deepseek_v2/impls/triton/deepseek_v2_mxfp4_moe_shuffled_16x16_model_dim6144_n_routed256_inter_dim256/`. Validate it and open the PR as in the ops tutorial.

### 7. Run it in SGLang

Validation checks the kernel against the reference, not the route; check the route by running the model. Start SGLang with GLM-5 (MXFP4) at TP8:

```bash
SGLANG_USE_AITER=1 AKLR_SGLANG_PLUGIN=1 python -m sglang.launch_server \
    --model-path <GLM-5 MXFP4 checkpoint> --tensor-parallel-size 8 --trust-remote-code
```

The log reports the hooked functions (`aklr hooks N sglang functions on sglang 0.5.22`), and the model's replies must stay correct.

A new route also reports its end-to-end gain: benchmark the same server with the op off (`AKLR_DISABLE=deepseek_v2_mxfp4_moe@1.0`) and on, and put the result in the PR's `End to end` section. The criterion is 2% on a stated workload ([acceptance criteria](acceptance_criteria.md)).

## Next

- The plugin's layout, rules for routes, and upgrading SGLang: [aklr/plugins/README.md](../aklr/plugins/README.md).
- Letting agents write and tune a kernel: [forge_task](forge_task.md).
