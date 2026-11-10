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

`harness/` is the `aklr` plugin: 14 skills and the `something-important` subagent. The
repository root holds its marketplace, in Claude Code's format, which Codex and Cursor also read.
Install it separately in each agent you use. Inside an AKLR checkout you need none of this; see
[Inside this repository](#inside-this-repository).

### Claude Code

```text
/plugin marketplace add AMD-AGI/AKLR
/plugin install aklr@aklr
```

Skills run as `/aklr:<skill>`, for example `/aklr:aklr-kb`, and the subagent is
`aklr:something-important`. Update with `claude plugin update aklr@aklr`, then `/reload-plugins`.
On a machine without a GitHub SSH key, set `CLAUDE_CODE_PLUGIN_PREFER_HTTPS=1` first.

### Codex

```bash
codex plugin marketplace add AMD-AGI/AKLR --sparse .claude-plugin --sparse harness
codex plugin add aklr@aklr
```

The `--sparse` paths fetch only the plugin instead of the whole repository. Start a new session,
then run skills as `$aklr:<skill>`. Update with `codex plugin marketplace upgrade aklr`. Codex
plugins carry no subagents, so `something-important` is available only inside an AKLR checkout.

### Inside this repository

Nothing to install: each agent loads the harness from the checkout, so edits take effect at once.

- Claude Code loads the plugin from `.claude/skills/aklr`, a symlink to `harness/`, as
  `/aklr:<skill>`, and only once the repository is trusted. `claude -p` and SDK sessions never
  show the trust dialog, so open the repository interactively once first; worktrees share the
  main checkout's trust. Start Claude Code at the repository root, since it does not look for
  plugins in parent directories. After editing a skill run `/reload-plugins`.
- Codex reads the skills from `.agents/skills`, a symlink to `skills/`, as `$aklr:<skill>`,
  because the link leads into the plugin. It reads agents from `.codex/agents` once the project
  is trusted.
- Cursor reads the skills from `.claude/skills/aklr` by bare name. It gets no subagents here; a
  skill that names one falls back to giving a general subagent its prompt.
- If you also installed the plugin:
  - Claude Code loads the installed copy instead of the checkout, since an installed plugin takes
    precedence over a project's `.claude/skills/` plugin of the same name. To use the checkout,
    for example while editing a skill, start with `claude --plugin-dir harness`, which takes
    precedence over both.
  - Codex turns the installed copy off here through `.codex/config.toml` and keeps the
    checkout's skills. In an untrusted project it reads no project config, so each skill appears
    twice.
  - Cursor loads both, so each skill appears twice. This includes a plugin installed only in
    Claude Code, because Cursor imports Claude Code's plugins. Uninstall the plugin, or turn off
    Cursor's "Include Third-Party Plugins, Skills, and Other Configs" setting for a plugin
    installed only in Claude Code.

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
