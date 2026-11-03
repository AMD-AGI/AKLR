<h1 align="center">AKLR</h1>

<p align="center"><b>Agentic Kernel Library for ROCm</b></p>

<div align="center">

[![CI](https://github.com/AMD-AGI/AKLR/actions/workflows/ci.yaml/badge.svg)](https://github.com/AMD-AGI/AKLR/actions/workflows/ci.yaml)
[![License](https://img.shields.io/badge/license-MIT%20%2B%20Apache--2.0-blue)](/LICENSE)

[Install](#install) | [Usage](#usage) | [Testing](#testing) | [Contributing](#contributing)

</div>

**AKLR** is a kernel library for LLM inference on ROCm, where agents write the kernels and every kernel is validated automatically before it serves. It is a step toward AI improving the systems that run AI.

We believe kernel performance will no longer come from rare, hand-tuned efforts but from continuous, automatic improvement, with AI driving each speedup on ROCm.

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

Validation times AKLR's kernels against a baseline pinned at `AKLR_BASELINE_AITER` in [ci.yaml](/.github/workflows/ci.yaml); install that commit before validating: `AITER_USE_SYSTEM_TRITON=1 pip install "amd-aiter @ git+https://github.com/ROCm/aiter.git@<that commit>"`.

## Usage

SGLang and vLLM load AKLR at startup; SGLang reaches AKLR's kernels only with `AKLR_SGLANG_PLUGIN=1`. Other code starts AKLR once (`DispatcherRegistry.default().init()`, from `aklr.core.runtime.dispatch`), then calls the ops in `aklr.ops` and runs its own code when one returns a `FallbackReason`.

| Variable | Effect |
|---|---|
| `AKLR_DISABLE=1` | Turn AKLR off |
| `AKLR_DISABLE=<contract id>,...` | Turn off the listed ops, e.g. `gemm@1.0` |
| `AKLR_SGLANG_PLUGIN=1` | Also hook SGLang's own layers to the plugin's ops ([aklr/plugins/README.md](/aklr/plugins/README.md)) |
| `AKLR_MIN_SPEEDUP=<x>` | Serve only kernels with a measured speedup of at least x (default 1.03) |

## Testing

```bash
pytest tests                                                # unit tests
python -m aklr validate --stages audit,bench --devices 0-7  # full validation
```

## Contributing

- Commit hooks and the baseline: [CONTRIBUTING.md](/CONTRIBUTING.md)
- PR descriptions: follow the [template](/.github/pull_request_template.md)
