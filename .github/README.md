<h1 align="center">AKLR</h1>

<p align="center"><b>Agentic Kernel Library for ROCm</b></p>

<div align="center">

[![CI](https://github.com/AMD-AGI/AKLR/actions/workflows/ci.yaml/badge.svg)](https://github.com/AMD-AGI/AKLR/actions/workflows/ci.yaml)
[![License](https://img.shields.io/badge/license-MIT%20%2B%20Apache--2.0-blue)](/LICENSE)

[Install](#install) | [Usage](#usage) | [Testing](#testing) | [Contributing](#contributing) | [AITER](https://github.com/ROCm/aiter)

</div>

**AKLR** is a kernel library for LLM inference on ROCm, where agents write the kernels and every kernel is validated automatically before it serves. It is a step toward AI improving the systems that run AI.

We believe kernel performance will no longer come from rare, hand-tuned efforts but from continuous, automatic improvement, with AI driving each speedup on ROCm.

## Install

| Framework | Development image |
|---|---|
| SGLang | `lmsysorg/sglang:v0.5.22-rocm10-mi35x` |
| vLLM | TODO |

Inside the image, install AKLR. It ships as `amd-aiter`: the AITER commit AKLR's routes describe (the `3rdparty/aiter` submodule, built from source in a few minutes) plus the `aklr` package, versioned `<aiter version>+aklr.g<commit>`. pip replaces the image's AITER, and `pip uninstall amd-aiter` removes both:

```bash
git clone --recursive https://github.com/AMD-AGI/AKLR.git && cd AKLR
pip install -e .
```

## Usage

SGLang and vLLM load AKLR automatically at startup; no code changes are needed.

| Variable | Effect |
|---|---|
| `AKLR_DISABLE=1` | Turn AKLR off and use AITER only |
| `AKLR_DISABLE=<contract id>,...` | Turn off the listed ops, e.g. `gemm@1.0` |
| `AKLR_SGLANG_PLUGIN=1` | Also serve whole SGLang layers and models (GLM-5, DeepSeek-V4.1-Flash) |
| `AKLR_MIN_SPEEDUP=<x>` | Serve only kernels with a measured speedup of at least x (default 1.03) |

## Testing

```bash
pytest tests                                                    # unit tests
python -m aklr validate --stages audit,bench,e2e --devices 0-7  # full validation
```

## Contributing

- Commit hooks and how to change AITER: [CONTRIBUTING.md](/CONTRIBUTING.md)
- PR descriptions: follow the [template](/.github/pull_request_template.md)
