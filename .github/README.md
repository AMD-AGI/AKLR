<h1 align="center">AKLR</h1>

<p align="center"><b>Agentic Kernel Library for ROCm</b></p>

<div align="center">

[![CI](https://github.com/AMD-AGI/AKLR/actions/workflows/ci.yaml/badge.svg)](https://github.com/AMD-AGI/AKLR/actions/workflows/ci.yaml)
[![License](https://img.shields.io/badge/license-MIT%20%2B%20Apache--2.0-blue)](../LICENSE)

[Repository map](#repository-map) | [Rosetta Stone](#the-kernel-rosetta-stone) | [Install](#install) | [Usage](#usage) | [Development](#development) | [Testing](#testing) | [Contributing](#contributing)

</div>

**AKLR** is a kernel library for LLM inference on ROCm, where agents write the kernels and every kernel is validated automatically before it serves. It is a step toward AI improving the systems that run AI.

We believe kernel performance will no longer come from rare, hand-tuned efforts but from continuous, automatic improvement, with AI driving each speedup on ROCm.

<p align="center"><img src="../docs/images/aklr_architecture.svg" alt="AKLR architecture" width="820"></p>

## Repository map

AKLR has two main parts: `aklr/` runs kernels, while `harness/` helps agents build and deliver them.

| Path | What is here |
|---|---|
| [`aklr/ops/`](../aklr/ops/README.md) | Each operator's API, correctness check, speed-test baseline, and kernel files. |
| [`aklr/core/`](../aklr/core/) | Shared code that checks inputs, loads kernels, picks one for each workload, and validates results. |
| [`aklr/plugins/`](../aklr/plugins/README.md) | Plugins for other projects. SGLang works today; vLLM and AITER are planned. |
| [`aklr/benchmarks/`](../aklr/benchmarks/) | Model workloads and benchmark code. |
| [`harness/`](../harness/README.md) | Guides and tools for agents to write, test, review, and submit kernels. [`forge_task`](../docs/forge_task.md) sends an AKLR problem to KernelForge. |
| [`tests/`](../tests/) and [`.github/workflows/`](workflows/) | Local tests and CI checks. |

An engine calls an AKLR operator. AKLR checks the inputs and picks a kernel that passed validation. If no kernel fits, the engine uses its original code. AKLR uses AITER as a speed-test baseline for library kernels; it does not ship AITER as a runtime fallback.

## The Kernel Rosetta Stone

The **Kernel Rosetta Stone** is a guide for agents writing AMD kernels. It brings together tips from handwritten kernels, AMD architecture and CDNA4 ISA documents, and real examples. Start at [`SKILL.md`](../harness/skills/aklr-kb/SKILL.md). As of 2026-11-05, it has **123 pages** and **1,204 findings with sources**.

It aims to help agents reach the same speed sooner, use fewer tokens, and find faster kernels. Below is our first measurement on one GEMM ([details](../docs/kernel_rosetta_stone.md)):

<p align="center"><img src="../docs/images/kernel_rosetta_stone_speedup.svg" alt="Kernel Rosetta Stone: GEMM comparison" width="820"></p>

- **Same 1.28x speedup: 5h20m and ~$52 with the AKLR knowledge base, against 14h46m and ~$205 without**, about a third of the time and a quarter of the spend.
- **After 8 hours: 1.286x against 1.177x.**
- **At the end: 1.300x for ~$133 with the AKLR knowledge base; without, the best was 1.289x for ~$230.**

## Install

| Framework | Development image |
|---|---|
| SGLang | `lmsysorg/sglang:v0.5.22-rocm10-mi35x` |
| vLLM | TODO |

Inside the image, install AKLR, one pure-Python package `aklr` (Python 3.11 or later), versioned `0.1.devN+g<commit>` before the first tag. It uses the image's torch, Triton and SGLang and replaces none of them:

```bash
git clone https://github.com/AMD-AGI/AKLR.git && cd AKLR
pip install -e .
```

Validation times AKLR's kernels against a baseline pinned at `AKLR_BASELINE_AITER` in [ci.yaml](workflows/ci.yaml); install that commit before validating: `AITER_USE_SYSTEM_TRITON=1 pip install "amd-aiter @ git+https://github.com/ROCm/aiter.git@<that commit>"`.

## Usage

SGLang loads AKLR at startup and, with `AKLR_SGLANG_PLUGIN=1`, runs AKLR's kernels through the SGLang plugin. vLLM also loads AKLR, but no vLLM code calls AKLR's ops yet. Other code starts AKLR once (`DispatcherRegistry.default().init()`, from `aklr.core.runtime.dispatch`), then calls the ops in `aklr.ops` and runs its own code when one returns a `FallbackReason`.

| Variable | Effect |
|---|---|
| `AKLR_DISABLE=1` | Turn AKLR off |
| `AKLR_DISABLE=<contract id>,...` | Turn off the listed ops, e.g. `gemm@1.0` |
| `AKLR_SGLANG_PLUGIN=1` | Hook SGLang's functions to the plugin's ops ([aklr/plugins/README.md](../aklr/plugins/README.md)) |
| `AKLR_MIN_SPEEDUP=<x>` | Serve only kernels with a measured speedup of at least x (default 1.03) |

## Development

Read in this order:

1. [Ops tutorial](../docs/ops_tutorial.md) ([中文](../docs/ops_tutorial.zh.md)): add a kernel to an op in `aklr/ops/`. The example is `fused_moe`, the experts of GLM-5's MoE layer.
2. [Plugin tutorial](../docs/plugin_tutorial.md) ([中文](../docs/plugin_tutorial.zh.md)): when an optimization crosses one op's boundary, replace a whole SGLang function. The example, `deepseek_v2_mxfp4_moe`, takes the same MoE layer whole.
3. [forge_task](../docs/forge_task.md): let agents write and tune a kernel.

For reference: [aklr/ops/README.md](../aklr/ops/README.md) (schema fields, the kernel interface, names, commands), [aklr/plugins/README.md](../aklr/plugins/README.md) (the plugin layout, rules for routes, upgrading SGLang), [harness/README.md](../harness/README.md) (agent skills and the kernel lifecycle).

## Testing

```bash
pytest tests                                                # unit tests
python -m aklr validate --stages audit,bench,e2e --devices 0-7  # full validation, as CI runs it
```

The e2e stage reads `aklr/core/validation/stages/_e2e/scenarios.yaml`, prints the scenario,
cookbook SHA, AKLR mode and full command, then starts each server directly with subprocess.
It compares AKLR off/on accuracy using this run's measured schemas over committed schemas.
E2E results and executed commands are included in `aklr_validation/report.{json,md}`.
Weights must already be available in the configured Hugging Face cache, as in CI.

## Contributing

- Commit hooks and the baseline: [CONTRIBUTING.md](../CONTRIBUTING.md)
- PR descriptions: follow the [template](pull_request_template.md)
