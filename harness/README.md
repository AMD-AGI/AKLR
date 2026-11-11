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

## Installation

`harness/` is the `aklr` plugin, holding the AKLR skills and subagents. The
repository root holds its marketplace, in Claude Code's format, which Codex and Cursor also read.
Install it separately in each agent you use. Inside an AKLR checkout you need none of this; see
[Inside this repository](#inside-this-repository).

### Claude Code

```text
/plugin marketplace add AMD-AGI/AKLR
/plugin install aklr@amd-aklr
```

Skills run as `/aklr:<skill>`, for example `/aklr:aklr-kb`, and subagents
as `aklr:<name>`. Update with `claude plugin update aklr@amd-aklr`, then `/reload-plugins`.
On a machine without a GitHub SSH key, set `CLAUDE_CODE_PLUGIN_PREFER_HTTPS=1` first.

### Codex

```bash
codex plugin marketplace add AMD-AGI/AKLR --sparse .claude-plugin --sparse harness
codex plugin add aklr@amd-aklr
```

The `--sparse` paths fetch only the plugin instead of the whole repository. Start a new session,
then run skills as `$aklr:<skill>`. Update with `codex plugin marketplace upgrade amd-aklr`. Codex
plugins carry no subagents, so subagents are available only inside an AKLR checkout.

### Inside this repository

Nothing to install. Claude Code, Codex and Cursor load the harness straight from the checkout, so
skill edits take effect right away.

- Claude Code: trust the repository once in an interactive session and start at its root. Skills
  are `/aklr:<skill>`; run `/reload-plugins` after editing one.
- Codex: skills are `$aklr:<skill>`; subagents need the project to be trusted.
- Cursor: skills keep their bare names; there are no subagents.

If you also installed the plugin, Claude Code uses the installed copy (start with
`claude --plugin-dir harness` to use the checkout instead), Codex uses the checkout, and Cursor
shows each skill twice.

## Layout

`.claude-plugin/plugin.json` is the plugin's manifest, and `.claude-plugin/marketplace.json` at the
repository root its marketplace. Claude Code, Codex and Cursor all read Claude Code's format, so
there is one of each.

- `skills/`: every AKLR agent skill, `skills/<name>/SKILL.md`, kept only here.
- `agents/`: subagent definitions the skills start, `agents/<name>.md`. Codex reads only TOML
  from `.codex/agents`, a symlink to `agents/codex/`; each `<name>.toml` there points back to
  `agents/<name>.md`, so the prompt is written once.
- `scripts/`: scripts the skills run, such as `forge_task/`.
- `tests/`: tests of the scripts.

This first version does not add or modify GitHub Actions workflows.
