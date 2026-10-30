# AKLR Agent Harness

This directory is the home of AKLR's agent-facing kernel lifecycle. It defines
how generated kernels enter AKLR, how selected AITER improvements are synced,
how kernel pull requests are reviewed, and how releases are qualified for
downstream engines.

The initial layout is intentionally small. The eight skills are usable guidance;
the `tools`, `schemas`, `config`, and `tests` directories currently document
their future ownership boundaries without introducing implementation or CI.

## Lifecycle

1. `aklr-monitor-upstreams` identifies a candidate.
2. `aklr-port-generated-kernels` or `aklr-sync-aiter` integrates it.
3. `aklr-review-kernel-pr` validates the pull request.
4. `aklr-build-runtime-image` creates a reproducible runtime.
5. `aklr-qualify-downstream` checks SGLang, vLLM, and ATOM compatibility.

`aklr-open-pr` opens the pull request of any step, with the evidence its kinds need.
`aklr-read-validation` reads a validation result, local or from CI, per Point.

The lifecycle is governed by [rules.md](rules.md), [gates.md](gates.md), and
[gate-config.md](gate-config.md).

## Layout

- `skills/`: eight independent Codex skills and their supporting references.
- `tools/`: future reusable implementation tools.
- `schemas/`: future machine-readable result schemas.
- `config/`: future repository, GPU, and downstream configuration.
- `tests/`: future skill, tool, and workflow tests.
- `install-agent-skills.py`: local installer for the eight skills.

This first version does not add or modify GitHub Actions workflows.
