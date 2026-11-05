# AKLR Agent Harness

This directory is the home of AKLR's agent-facing kernel lifecycle. It defines
how generated kernels enter AKLR, how kernel pull requests are reviewed, and
how releases are qualified for downstream engines.

The initial layout is intentionally small: the skills are usable guidance, and
`scripts` and `tests` hold the scripts they run and those scripts' tests, without CI.

## Lifecycle

1. `aklr-monitor-upstreams` identifies a candidate.
2. `aklr-port-generated-kernels` integrates it.
3. `aklr-review-kernel-pr` validates the pull request.
4. `aklr-build-runtime-image` creates a reproducible runtime.
5. `aklr-qualify-downstream` checks SGLang, vLLM, and ATOM compatibility.

`aklr-open-pr` opens the pull request of any step, with the evidence its kinds need.
`aklr-read-validation` reads a validation result, local or from CI, per Point.

The lifecycle is governed by [rules.md](rules.md), [gates.md](gates.md), and
[gate-config.md](gate-config.md).

## Layout

- `skills/`: every AKLR agent skill, `skills/<name>/SKILL.md`, kept only here. Agents find them
  through symlinks: `.claude/skills` for Claude Code, `.agents/skills` for Codex and Cursor.
- `agents/`: subagent definitions the skills start, `agents/<name>.md`, found through
  `.claude/agents`, which Claude Code and Cursor read. Codex reads only TOML from
  `.codex/agents`, a symlink to `agents/codex/`; each `<name>.toml` there points back to
  `agents/<name>.md`, so the prompt is written once.
- `scripts/`: scripts the skills run, such as `forge_task/`.
- `tests/`: tests of the scripts.

This first version does not add or modify GitHub Actions workflows.
