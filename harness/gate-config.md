# Gate Configuration

This file records the categories of configuration required to execute the
gates. Concrete values will move to `config/` when the execution environment is
defined.

## Required Configuration

- AKLR base and candidate revisions.
- AITER performance baseline (ROCm/aiter at `AKLR_BASELINE_AITER` in `.github/workflows/ci.yaml`).
- Supported AMD GPU architectures and node topology.
- ROCm, Python, PyTorch, and compiler versions.
- Correctness tolerances by operator and data type.
- Representative models, shapes, batch sizes, and sequence lengths.
- Warm-up, repetition, timeout, and performance-regression policy.
- Supported SGLang, vLLM, and ATOM revisions.
- Artifact retention and reporting locations.

No unspecified value should be silently treated as an approved default.
