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
`aklr-bad-cases` lists mistakes agents have made in AKLR, checked when reviewing a change or opening a pull request.

The lifecycle is governed by [rules.md](rules.md), [gates.md](gates.md), and
[gate-config.md](gate-config.md).

## Layout

`harness/` is the `aklr` plugin. `.claude-plugin/plugin.json` is its manifest for Claude Code
(Codex also reads it), `.cursor-plugin/plugin.json` its manifest for Cursor.

- `skills/`: every AKLR agent skill, `skills/<name>/SKILL.md`, kept only here.
- `agents/`: subagent definitions the skills start, `agents/<name>.md`. Codex reads only TOML
  from `.codex/agents`, a symlink to `agents/codex/`; each `<name>.toml` there points back to
  `agents/<name>.md`, so the prompt is written once.

How each agent loads it inside this repository:

- Claude Code loads the plugin from `.claude/skills/aklr`, a symlink to `harness/`, once the
  repository is trusted (open it interactively once; worktrees share the main checkout's trust).
  Start Claude Code at the repository root. Skills are `/aklr:<skill>`, the subagent is
  `aklr:something-important`; after editing a skill run `/reload-plugins`.
- Codex and Cursor read the skills natively from `.agents/skills`, a symlink to `skills/`, so
  edits take effect at once. Cursor reads each agent from `.cursor/agents/<name>.md`, a symlink to
  `agents/<name>.md`; add one for every new agent. Codex installs plugins as cached copies that do not follow edits,
  and Cursor would load each skill twice, so neither loads the plugin here.

- `scripts/`: scripts the skills run, such as `forge_task/`.
- `tests/`: tests of the scripts.

This first version does not add or modify GitHub Actions workflows.
