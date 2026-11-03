<h1 align="center">AKLR</h1>

<p align="center"><b>Agentic Kernel Library for ROCm</b></p>

<div align="center">

[![CI](https://github.com/AMD-AGI/AKLR/actions/workflows/ci.yaml/badge.svg)](https://github.com/AMD-AGI/AKLR/actions/workflows/ci.yaml)
[![License](https://img.shields.io/badge/license-MIT%20%2B%20Apache--2.0-blue)](../LICENSE)

[Repository map](#repository-map) | [Rosetta Stone](#the-kernel-rosetta-stone) | [Install](#install) | [Usage](#usage) | [Testing](#testing) | [Contributing](#contributing)

</div>

**AKLR** is a kernel library for LLM inference on ROCm, where agents write the kernels and every kernel is validated automatically before it serves. It is a step toward AI improving the systems that run AI.

We believe kernel performance will no longer come from rare, hand-tuned efforts but from continuous, automatic improvement, with AI driving each speedup on ROCm.

## Repository map

| Path | What it owns |
|---|---|
| [`aklr/ops/`](../aklr/ops/README.md) | Operator contracts, reference implementations, benchmark baselines, public APIs, and kernel schemas. A schema names the workloads a kernel may serve. |
| [`aklr/core/`](../aklr/core/) | Contract types, kernel loading and compilation, per-workload dispatch, and validation. |
| [`aklr/plugins/`](../aklr/plugins/README.md) | Inference-engine routes. The SGLang plugin is implemented; vLLM routes are planned. |
| [`aklr/benchmarks/`](../aklr/benchmarks/) | Model workload definitions and benchmark support. |
| [`harness/`](../harness/README.md) | Agent skills and the kernel delivery lifecycle, including [`forge_task`](../docs/forge_task.md), which sends an AKLR problem to KernelForge. |
| [`tests/`](../tests/) and [`.github/workflows/`](workflows/) | Tests and CI validation. |

An engine plugin or another caller invokes an AKLR op. The op checks its contract and selects a validated kernel for that workload. If none qualifies, AKLR returns a `FallbackReason` and the caller runs its original path. AITER is a validation baseline, not a runtime fallback bundled with AKLR.

## The Kernel Rosetta Stone

The **Kernel Rosetta Stone** is the agent-readable AMD kernel optimization guide in [`harness/skills/flydsl-kernel-optimization/`](../harness/skills/flydsl-kernel-optimization/SKILL.md). It is knowledge for authors and agents, not a runtime component. It turns experience from handwritten AMD kernels and CDNA4 ISA and architecture references into task-specific Markdown pages: GEMM, grouped GEMM, attention and MoE techniques; debugging and measurement guides; pitfalls; and [case studies](../harness/skills/flydsl-kernel-optimization/cases/INDEX.md) of delivered kernels.

Agents start at the skill's [`SKILL.md`](../harness/skills/flydsl-kernel-optimization/SKILL.md), follow its task or symptom table to one relevant page, apply a technique, and validate the result with AKLR. As of 2026-11-03, the guide contains **131 Markdown pages** (including indexes) and **1,192 sourced findings**. Measured findings record their workload, GPU, toolchain, method, and source so readers can check when a lesson applies. New pages pass [evidence and review rules](../harness/skills/flydsl-kernel-optimization/CONTRIBUTING.md).

The intended benefits are less time and fewer agent tokens to reach the same validated performance, and a higher best speedup at an equal optimization budget. AKLR does not yet have a controlled run with and without this guide, so these are goals, not measured gains. A comparison should hold the problem, agent, GPU, and budget fixed, then record time and input/output tokens to the first qualifying kernel and the best validated speedup at the same budget.

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

SGLang and vLLM load AKLR at startup; SGLang reaches AKLR's kernels only with `AKLR_SGLANG_PLUGIN=1`. Other code starts AKLR once (`DispatcherRegistry.default().init()`, from `aklr.core.runtime.dispatch`), then calls the ops in `aklr.ops` and runs its own code when one returns a `FallbackReason`.

| Variable | Effect |
|---|---|
| `AKLR_DISABLE=1` | Turn AKLR off |
| `AKLR_DISABLE=<contract id>,...` | Turn off the listed ops, e.g. `gemm@1.0` |
| `AKLR_SGLANG_PLUGIN=1` | Also hook SGLang's own layers to the plugin's ops ([aklr/plugins/README.md](../aklr/plugins/README.md)) |
| `AKLR_MIN_SPEEDUP=<x>` | Serve only kernels with a measured speedup of at least x (default 1.03) |

## Testing

```bash
pytest tests                                                # unit tests
python -m aklr validate --stages audit,bench --devices 0-7  # full validation
```

## Contributing

- Commit hooks and the baseline: [CONTRIBUTING.md](../CONTRIBUTING.md)
- Optimizing a kernel with forge_task: [docs/forge_task.md](../docs/forge_task.md)
- PR descriptions: follow the [template](pull_request_template.md)
