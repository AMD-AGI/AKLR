# Plugin tutorial: replacing vLLM functions

A plugin replaces a vLLM function that no aiter op covers with an AKLR op: a function of the model's own, or one that runs a whole stretch of vLLM's computation, which the op fuses. This tutorial's example is `kimi_k3_attn_res`, which takes Kimi-K3's attention residual: `attn_res`, vLLM's own Triton kernel for the model, run before each layer's attention and MLP and once before the final norm.

Before you start, read the [ops tutorial](ops_tutorial.md): a plugin's op is written the same way. The SGLang plugin is covered in the [SGLang plugin tutorial](sglang_plugin_tutorial.md), and the plugin's layout and rules are in [aklr/plugins/README.md](../aklr/plugins/README.md).

## Why a plugin

vLLM computes Kimi-K3's attention residual in one Triton kernel, `attn_res` (`models/kimi_k3/amd/ops/attn_res.py`), which `_apply_attn_res` (`models/kimi_k3/amd/linear.py`) calls with the layer's weights. Each token keeps a bank of rows, the embedding and then the sum of each block of 12 layers' outputs, beside the running prefix of the current block. For each token the kernel computes:

| Step | What it does | vLLM's `attn_res` | `kimi_k3_attn_res` |
|---|---|---|---|
| 1 | The pending layer output `delta` added to the prefix, rounded to bf16 and stored back | ✓ | ✓ |
| 2 | On a block's first layer, the prefix written as the bank's next row | ✓ | ✓ |
| 3 | Each source scored, the bank's written rows and the prefix: RMS-normalized, then projected to one score | ✓ | ✓ |
| 4 | Softmax over the scores; the sources' weighted sum is the mix | ✓ | ✓ |
| 5 | The layer's RMSNorm on the mix: `input_layernorm` before attention, `post_attention_layernorm` before the MLP, none before the final norm | ✓ | ✓ |
| 6 | The output quantized to fp8 per token, when the next linear takes fp8 input | ✓ | — |

vLLM already runs these steps as one kernel, but the kernel is the model's own: it lives in Kimi-K3's model directory, not in aiter. No aiter op covers it, so AKLR's aiter provider never sees the call, and no op in `aklr/ops` has its computation. To run a faster kernel in its place, the plugin replaces the model's function: the `attn_res` that `_apply_attn_res` calls. The op takes steps 1–5; the fp8 output of step 6 is in no contract, so there the route falls back to vLLM's kernel.

What a plugin may replace: a route never replaces an aiter function, nor a framework function that only wraps one aiter call, such as vLLM's `_rocm_aiter_fused_moe_impl` (`_aiter_ops.py`) around aiter's `fused_moe`. Calls of aiter's ops reach AKLR through aiter's own plugin interface, whose AKLR provider is `aklr/plugins/aiter/provider` ([aiter's kernel plugin interface](../aklr/plugins/README.md#aiters-kernel-plugin-interface)). A route does one of two things: it replaces a model's own function, as here, or it fuses a whole stretch of the framework's computation, aiter calls included, into one op.

So:

- To make one op faster, write an op in `aklr/ops`; if aiter routes that op, the aiter provider hands its calls to AKLR.
- When the function is the model's own, or the optimization fuses a stretch of the framework's computation across ops, write a plugin.

## How the plugin runs

With `AKLR_VLLM_PLUGIN=1`, each worker hooks the routes once its device is bound. From then on, each call of a replaced function first runs the route, which hands the call to the op or returns a `FallbackReason` to fall back to vLLM's code (see the [call flow](../aklr/plugins/README.md#call-flow)).

## Steps

### 1. Find the function to replace

This vLLM call:

- `attn_res`, called by `_apply_attn_res` (`models/kimi_k3/amd/linear.py`): computes steps 1–6 above and returns the output, having updated the prefix and the bank in place. `_apply_attn_res` runs twice in each decoder layer, before attention and before the MLP (`KimiDecoderLayer.forward_attn_residual`), and once before the final norm (`KimiLinearModel.forward`): 187 calls a forward over Kimi-K3's 93 layers.

`_apply_attn_res` only unpacks the layer's modules into tensors and, for an fp8 output, wraps the result; the arguments of `attn_res` are already the op's, so the route replaces `attn_res`. Its parameters in vLLM 0.34.0, which the route copies exactly:

```python
def attn_res(prefix, delta, blocks, norm_weight, qk_weight, output_norm_weight, num_blocks, block_write_idx,
             eps, output_norm_eps, *, quant_dtype=None): ...
```

### 2. Create the package

Create `aklr/plugins/vllm/models/kimi_k3/attn_res/`: `kimi_k3/` is named after vLLM's source directory `models/kimi_k3/`, `attn_res/` after the kernel's file `ops/attn_res.py`. It holds the same files as an op in `aklr/ops`: `__init__.py`, `contract.py`, `reference.py`, `baseline.py`, `api.py` and `impls/`; and `_triton.py`, a copy of vLLM's Triton kernel and its launcher without the fp8 output, so that the baseline runs without vLLM. The package name is the op's `op_type` (`attn_res`) and must be unique across `aklr/ops` and `aklr/plugins`.

### 3. Write the op

As in section 6 of the [ops tutorial](ops_tutorial.md). For this op:

- `contract.py` defines `KIMI_K3_ATTN_RES` (`kimi_k3_attn_res@1.0`):
  - inputs: the running `prefix`; the pending layer output `addend`, present with `has_addend`; each token's `bank`; the score's two weights `score_norm_weight` and `score_proj_weight`; the layer norm's `norm_weight`, present with `has_norm`; and the scalars `score_eps` and `norm_eps`;
  - attrs: `n_valid_rows`, the bank rows written so far (0 to 8), all of which the mix reads; `write_bank`, whether the prefix also becomes bank row `n_valid_rows`; and `has_addend` and `has_norm`. Each combination vLLM calls is a scenario of its own;
  - layout: `row_major` only;
  - output: `out`, of shape `[n_tokens, hidden_size]`: the mix, normalized and scaled by `norm_weight` when there is one. `prefix` and `bank` are mutable inputs, which the op updates in place as vLLM's kernel does; their error bound is exact.
- `reference.py`: `KimiK3AttnResReference`, the contract's computation in float32. It rounds `prefix + addend` to the prefix's dtype, as vLLM's kernel stores it back, and its outputs to that dtype. It returns `out`, `prefix` and `bank` as the call leaves them.
- `baseline.py`: `KimiK3AttnResBaseline`, what vLLM runs today: its Triton kernel, one launch per call, with the launch configuration of its `attn_res`, from the copy in `_triton.py`.
- `api.py`: the entry function `@api(KIMI_K3_ATTN_RES)`, whose body is `...`.

### 4. Write the route

The route goes in `api.py`, beside the op's entry function `kimi_k3_attn_res`. It turns vLLM's arguments into the op's and calls `kimi_k3_attn_res`:

- `blocks`, `norm_weight` and `qk_weight` are the op's `bank`, `score_norm_weight` and `score_proj_weight`; `delta` and `output_norm_weight` its optional `addend` and `norm_weight`; `eps` and `output_norm_eps` its `score_eps` and `norm_eps`.
- `num_blocks` is `n_valid_rows`, and a `block_write_idx` of 0 or more is `write_bank`: vLLM writes the bank only at row `num_blocks`, the row the contract writes.
- What no contract states goes back to vLLM before the op is called: the fp8 output (`quant_dtype`), and a bank write at any other row.

The first argument of `@vllm_route` is the caller, the second the replaced function, named as the caller looks it up at run time: `linear.py` imports `attn_res` from `ops/attn_res.py`, so the route replaces `vllm.models.kimi_k3.amd.linear.attn_res`. `versions` lists the vLLM releases the route was tested on:

```python
@vllm_route(
    "vllm.models.kimi_k3.amd.linear._apply_attn_res",
    "vllm.models.kimi_k3.amd.linear.attn_res",
    versions=("0.32.1rc1.dev276", "0.33.0rc1.dev910", "0.33.0", "0.34.0"),
)
def _attn_res(
    prefix,
    delta,
    blocks,
    norm_weight,
    qk_weight,
    output_norm_weight,
    num_blocks,
    block_write_idx,
    eps,
    output_norm_eps,
    *,
    quant_dtype=None,
):
    """Every attention residual of Kimi-K3's decoder layers and the one before its final norm.
    vLLM writes the bank only at row num_blocks."""
    # the fp8 output for a linear that takes per-token fp8 input, which no contract states
    if quant_dtype is not None or block_write_idx not in (-1, num_blocks):
        return FallbackReason.OUT_OF_CONTRACT
    return kimi_k3_attn_res(
        prefix,
        blocks,
        norm_weight,
        qk_weight,
        n_valid_rows=num_blocks,
        write_bank=block_write_idx >= 0,
        addend=delta,
        norm_weight=output_norm_weight,
        score_eps=eps,
        norm_eps=output_norm_eps,
    )
```

When the op takes the call, it returns `out`, which the route returns to `_apply_attn_res` as vLLM's `attn_res` would, with `prefix` and `bank` updated in place; when it declines, it returns a `FallbackReason`, and vLLM's `attn_res` runs. Token counts, row counts and GPU models are the op's to judge, not the route's: the call before the first layer's attention, with no bank row yet, reaches the op, which has no kernel for it and declines.

### 5. Add test inputs

Add small Points of the op to `SMALL_POINTS` in `tests/plugins/vllm/cases.py`, the contract's base Point first. This op has four, one for each kind of call: folding the pending output into the prefix, writing the bank with and without the addend, and the residual before the final norm, without a layer norm. The first:

```python
PointSpec(
    contract_id="kimi_k3_attn_res@1.0",
    axes={"n_tokens": 4, "hidden_size": 256, "n_bank_rows": 4},
    dtypes={
        "prefix": "bfloat16",
        "addend": "bfloat16",
        "bank": "bfloat16",
        "score_norm_weight": "bfloat16",
        "score_proj_weight": "bfloat16",
        "norm_weight": "bfloat16",
        "out": "bfloat16",
    },
    attrs={"n_valid_rows": 2, "write_bank": False, "has_addend": True, "has_norm": True},
    layout_variant_name="row_major",
),
```

`GFX950_CONTRACT_IDS` in the same file lists the contracts whose baseline runs only on gfx950; vLLM's Triton kernel is not one of them, so this contract is not listed. Then update `tests/plugins/vllm/names.txt`:

```bash
AKLR_UPDATE_NAMES=1 pytest tests/ops/test_contracts.py
```

`tests/plugins/vllm/test_hook.py` checks every route: that the function it replaces exists in the installed vLLM with the same parameters, and that no other route replaces it. How the route maps vLLM's arguments `test_hook.py` cannot check, so it is tested on its own in `tests/plugins/vllm/test_kimi_k3_attn_res.py`: the fp8 output and a bank write before the last row go back to vLLM without the op called, a supported call hands each operand to the op under its name, and the baseline's copy of vLLM's Triton kernel gives vLLM's bits for the output, the updated prefix and the bank.

### 6. Write the kernel and validate it

Follow sections 2 to 5 of the [ops tutorial](ops_tutorial.md), with this op's scenario: Kimi-K3's attention residual at hidden size 7168, with a bank of 8 rows for its 93 layers in blocks of 12 (see [Kimi-K3's model structure](../aklr/benchmarks/models/kimi_k3/README.md#model-structure)). Of the 187 calls a forward makes, 170 fold the pending output in and apply the layer norm, at 1 to 8 bank rows; 7 also write the bank; 8 have no addend, the MLP's call on a block's first layer; one, before the final norm, has no layer norm; and the one before the first layer's attention has no bank row yet. Each combination of the attrs is a scenario of its own. The script differs in the contract, the Point and the workloads, where `score_eps` and `norm_eps` are scalars given with each workload (`_apply_attn_res` passes a `norm_eps` of 0.0 without a layer norm); the token counts are decode steps of 1 to 4 requests, 3 tokens each, and this op's committed kernels are written in FlyDSL (`language` is `flydsl` in the script, and `requires` is `{"flydsl": ">=0.3.2"}`):

```python
contract = ContractRegistry.default().get("kimi_k3_attn_res@1.0")
point = PointSpec(
    contract_id="kimi_k3_attn_res@1.0",
    axes={"n_tokens": 1, "hidden_size": 7168, "n_bank_rows": 8},
    dtypes={"prefix": "bfloat16", "addend": "bfloat16", "bank": "bfloat16", "score_norm_weight": "bfloat16",
            "score_proj_weight": "bfloat16", "norm_weight": "bfloat16", "out": "bfloat16"},
    attrs={"n_valid_rows": 8, "write_bank": False, "has_addend": True, "has_norm": True},
    layout_variant_name="row_major")
workloads = [contract.workload_for(point.model_copy(update={"axes": {**point.axes, "n_tokens": t}}),
                                   {"score_eps": 1e-5, "norm_eps": 1e-5}) for t in (3, 6, 9, 12)]
```

The schema is written beside the committed kernel of this scenario, in `aklr/plugins/vllm/models/kimi_k3/attn_res/impls/flydsl/kimi_k3_attn_res_hidden_size7168_n_bank_rows8/`; the other scenarios carry their attrs in the name, such as `kimi_k3_attn_res_rows3_write_hidden_size7168_n_bank_rows8`. The call without a bank row has no schema and stays on vLLM's kernel. Validate it and open the PR as in the ops tutorial.

### 7. Run it in vLLM

Validation checks the kernel against the reference, not the route; check the route by running the model. Start vLLM with Kimi-K3's MXFP4 checkpoint at TP8, with the deployment's own options and the plugin on. Its attention and dense MLP are BF16, so no call asks for the fp8 output the route leaves to vLLM:

```bash
AKLR_VLLM_PLUGIN=1 vllm serve <Kimi-K3 MXFP4 checkpoint> --tensor-parallel-size 8
```

The log reports the hooked calls (`aklr hooks N calls of vLLM 0.34.0`), and the model's replies must stay correct. With `AKLR_TRACE=plugin`, each route logs once per distinct call whether it served or fell back, and why ([aklr/plugins/README.md](../aklr/plugins/README.md#enabling-the-plugin-and-upgrading-sglang)).

## Next

- The plugin's layout and rules for routes: [aklr/plugins/README.md](../aklr/plugins/README.md).
- Letting agents write and tune a kernel: [forge_task](forge_task.md).
