# MiniMax M3.1 TP4 schema handoff

This branch provides 10 schema definitions and 679 workloads for
`MiniMax-M3.1-Flash-Quark-MXFP4` with TP4 and DSpark on `gfx950/MI350X`.
Every definition has `kernel.sources=[]`; fill those sources to implement the
kernel. The files contain no accuracy or speedup measurements.

The definitions come from C32/C64 calls captured on 2026-11-05, using AKLR
base `0f2036bc5106a70d74cab671b52c28cf05166126` and container
`vllm/vllm-openai-rocm@sha256:ae8681212aa5291d8c27f06e90016435e3fca144a68fd5c439231f6475f74569`.
The checkpoint revision was `0a00e0e39fc00499b23efccc18a8ea63fce5c44c`.
Both captures used `VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0` and retained
the checkpoint's MXFP8 shared experts.

The [capture manifest](minimax-m31-tp4/manifest.json) records the schema
hashes, observed token sizes, source hashes and runtime configuration.
Its schema paths are relative to `aklr/ops/`. Other paths inside the
recorded reports identify files in the original capture environment.

## Definitions

The main model contributes seven definitions and 448 workloads. DSpark
contributes three definitions and 231 workloads. All GEMMs have BF16 output,
no bias, and `trans_b=True`.

| Role | Contract | N | K | Workloads | Schema |
| --- | --- | ---: | ---: | ---: | --- |
| Routed experts | `fused_moe@1.0` | — | — | 64 | [9b4802c967de53f9](../aklr/ops/fused_moe/impls/flydsl/fused_moe_swiglu_has_swiglu_limit_shuffled_16x16_model_dim6144_n_experts128_topk4_inter_dim768/9b4802c967de53f9.json) |
| Shared experts gate/up | `gemm_a8w8_mxfp8@1.0` | 1536 | 6144 | 64 | [c72746d8aae22e24](../aklr/ops/gemm/impls/flydsl/gemm_a8w8_mxfp8_n1536_k6144/c72746d8aae22e24.json) |
| Shared experts down | `gemm_a8w8_mxfp8@1.0` | 6144 | 768 | 64 | [daf9cc42dd4bc9e9](../aklr/ops/gemm/impls/flydsl/gemm_a8w8_mxfp8_n6144_k768/daf9cc42dd4bc9e9.json) |
| First three dense MLPs gate/up | `gemm_a8w8_mxfp8@1.0` | 6144 | 6144 | 64 | [117607496a05eeb8](../aklr/ops/gemm/impls/flydsl/gemm_a8w8_mxfp8_n6144_k6144/117607496a05eeb8.json) |
| First three dense MLPs down | `gemm_a8w8_mxfp8@1.0` | 6144 | 3072 | 64 | [8c369b14abe6461e](../aklr/ops/gemm/impls/flydsl/gemm_a8w8_mxfp8_n6144_k3072/8c369b14abe6461e.json) |
| Main attention QKV | `gemm_a8w8_mxfp8@1.0` | 2560 | 6144 | 64 | [b6ed977fd10e992b](../aklr/ops/gemm/impls/flydsl/gemm_a8w8_mxfp8_n2560_k6144/b6ed977fd10e992b.json) |
| Main attention output | `gemm_a8w8_mxfp8@1.0` | 6144 | 2048 | 64 | [5895fc6d7c70a996](../aklr/ops/gemm/impls/flydsl/gemm_a8w8_mxfp8_n6144_k2048/5895fc6d7c70a996.json) |
| DSpark merged context K/V | `gemm_a8w8_mxfp8@1.0` | 1280 | 6144 | 82 | [e1964d79107fd6e4](../aklr/ops/gemm/impls/flydsl/gemm_a8w8_mxfp8_n1280_k6144/e1964d79107fd6e4.json) |
| DSpark merged hidden-state fc | `gemm@1.0` | 6144 | 30720 | 82 | [be082e22c749d95f](../aklr/ops/gemm/impls/flydsl/gemm_n6144_k30720/be082e22c749d95f.json) |
| DSpark Markov vocabulary projection | `gemm@1.0` | 200064 | 256 | 67 | [69cc30902c5b4f8a](../aklr/ops/gemm/impls/flydsl/gemm_n200064_k256/69cc30902c5b4f8a.json) |

`m` / `n_tokens` remains a variable axis. Each file lists the distinct observed
values, spanning 1–32768 except the Markov projection, which spans 1–128.
The workloads do not enumerate every integer in those ranges. To extend
coverage, add explicit workloads for the desired values and check the
canonical schema; changing `any_size` does not enumerate M.

DSpark's N=1280 projection merges context K/V for five layers and reuses the
first layer's `qkv_proj` prefix. The N=200064 Markov weight is fully replicated
on each rank. The captured Python entry points do not cover all operations
lowered into compiled code. The 298,160 recorded calls include startup,
eager execution and graph construction; graph replays are not counted, and
these counts are not workload weights.

## Operator semantics

The seven MXFP8 files require the added
[`gemm_a8w8_mxfp8`](../aklr/ops/gemm/api.py) API, contract and reference.
Inputs are row-major E4M3 tensors A[M,K] and B[N,K], with row-major E8M0
scales A[M,K/32] and B[N,K/32]. They compute
`dequant(A) @ dequant(B).T`. Input quantization and intervening MLP
activations belong to the caller. The new contract retains the strict
dot-product error bound and bitwise determinism; it has no registered
performance baseline yet.

The MoE uses the existing contract: hidden size 6144, 128 local routed
experts, top-k 4, intermediate size 768 per rank, MXFP4 weights in
`shuffled_16x16` layout, and separated gate/up rows. Its SwiGLU clamps gate
to at most 7 and up to [-7,7], then computes
`gate * sigmoid(1.702 * gate) * (up + 1)`. Both stages keep BF16 activations
below 256 tokens and quantize to MXFP4 from 256 tokens onward. Preserve the
rounding points in the [reference](../aklr/ops/fused_moe/reference.py).
Router computation, routed-output multiplication by 2, shared experts and
TP reduction are outside this MoE definition.

For `activation="swiglu"`, the current MoE contract uses
`BaselineRelativeBound()`. The SQNR gate described in older GLM MoE cases
does not apply to this variant. Read the current
[contract](../aklr/ops/fused_moe/contract.py) before reusing those kernels.

The capture used a separate probe plugin around AITER's fused MoE, vLLM's
native MXFP8 GEMM and unquantized linear entry points. This branch provides
core-op definitions; production vLLM routing to their APIs still needs to
be integrated when the kernels are ready.

## Filling and checking kernels

Follow [the schema workflow and kernel ABI](../aklr/ops/README.md).
Add source objects under `kernel.sources`, implement the keyword-only
`kernel.py::run` entry, write the supplied output buffers, and return nothing.
After changing sources, rename the JSON to the UUID reported by
`export check`; leave measurement fields for CI.

```bash
python -m aklr export check
python -m aklr validate --base origin/main
```

The relevant [harness knowledge base](../harness/skills/aklr-kb/SKILL.md)
has been reviewed for this handoff:

- [GEMM workflow](../harness/skills/aklr-kb/gemm/flydsl_gemm_workflow.md):
  choose the layout, grid and pipeline before tuning constants.
- [Small-M GEMM](../harness/skills/aklr-kb/gemm/flydsl_gemm_small_m.md):
  separate weight-bandwidth limits from launch and reduction costs; include
  cold weights and graph execution in measurements.
- [GEMM correctness](../harness/skills/aklr-kb/gemm/flydsl_gemm_correctness.md):
  test distinct E8M0 scales, repeat calls, and edge shapes. Bitwise GEMMs
  need fixed-order reduction. Workgroups must not poll other workgroups.
- [MoE decode](../harness/skills/aklr-kb/moe/flydsl_moe_decode.md) and
  [shuffled MoE cases](../harness/skills/aklr-kb/cases/fused_moe_shuffled_16x16.md):
  route-direct decode and fused routing are candidates to measure. Recheck
  their layouts and arithmetic against this contract before reuse.

The KB's FlyDSL examples target v0.3.2; the capture container had v0.4.0.
Recheck API details and measure constants on the actual runtime.

## Validation recorded before handoff

- Schema generation was byte-identical with C32/C64 input order reversed.
  AKLR `export check` passed [locally](minimax-m31-tp4/local-export-check.json)
  and in the [capture container](minimax-m31-tp4/node-export-check.json).
- New contract/registration tests: 27 passed. Related reference/fast-key
  checks: 4 passed, 3 skipped. Commands:

  ```bash
  python -B -m pytest tests/ops/test_mxfp8_gemm.py tests/ops/test_contracts.py -q
  python -B -m pytest tests/ops/test_reference.py tests/core/contract/test_fastkey.py \
    -k gemm_a8w8_mxfp8 -m 'not gpu' -q
  ```

- [MoE weight and scale layouts](minimax-m31-tp4/moe-layout-validation.json)
  matched AITER/vLLM byte for byte.
- [Synthetic native MXFP8 GPU checks](minimax-m31-tp4/mxfp8-gpu-validation.json)
  passed 29 of 30 cases. At
  `(M,N,K)=(4096,6144,768)`, 2 of 25,165,824 output elements exceeded the
  default error bound; the maximum error/bound ratio was 1.03324.
  [Exact rational summation](minimax-m31-tp4/mxfp8-failure-diagnostic.json)
  confirmed the reference for both elements; the
  discrepancy was already present in native FP32 output. The cause remains
  unresolved, and the tolerance has not been loosened. Repeat outputs were
  bitwise equal in every tested case.
- A broader CPU run had four pre-existing MXFP4 failures caused by packed-FP4
  copy support in local Torch 2.9.1. They are not passing results. The GPU
  checks used Torch 2.13.0 with ROCm 10.

These checks establish the definitions and record native-path limitations.
The empty-source files have no implemented kernel to qualify or benchmark.
