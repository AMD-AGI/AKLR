# Roadmap: merging `refactor/core-validation` into main

The branch is one commit on top of `8ad787a4d`: about 870 paths, 519 of them schema JSON.
It goes to main as three pull requests, in this order.

| PR | Title | What is in it | Status |
|----|-------|---------------|--------|
| 1 | `test(agentx): pytest style, tests/AGENTS.md rules and the tdd skill` | `tests/benchmarks/**`, `tests/AGENTS.md`, root `AGENTS.md`, `harness/skills/tdd/**`, `tests/check_layout.py`, the `tests-layout` pre-commit hook | on the branch, to be cut out |
| 2 | `refactor(core): contract removal, schema rewrite, op variants and validation bench` | everything else on the branch: `aklr/core`, `aklr/ops`, `aklr/seam`, their tests, harness, CI, docs | on the branch |
| 3 | `refactor(plugins): move plugins to the schema and the seam` | `aklr/plugins/**`, `tests/plugins/**`, the leftovers listed in step 5 | not written |

## Why three

- **PR 1 is independent and unblocks PR 2.** The agentx tests import nothing from `aklr.core` or
  `aklr.ops`. The new `tests-layout` hook fails on the old agentx tests, so they have to be converted
  before PR 2 can pass lint. Main's #208 rewrote the same files; keeping them out of PR 2 keeps that
  conflict out of the core rebase.
- **PR 2 cannot be split further.** The JSON format, the loader and the op code change together: the
  old loader cannot read the new files and the new one cannot read the old. Op variants and the bench
  are built on the new schema. Any split needs a hand-made middle state that the next PR deletes, and
  every PR that touches `aklr/core` re-benches every schema.
- **PR 3 is what PR 2 breaks.** After PR 2 the plugin entry points live in
  `aklr.core.runtime.dispatch`, while `aklr/plugins` still imports the removed `aklr.core.contract`
  and its 440 JSON files are in the old format, loaded by nothing. See `TODO.md`.

## Steps

1. **Clean up locally.**
   - Update local `main`; it is 58 commits behind the branch base.
   - Keep `ref/` out of git (add it to `.git/info/exclude`). It is 1.5 GB of cloned repositories.
   - Decide what happens to `fix/core-validation-gaps`: 12 commits, most of them not on this branch.

2. **Open PR 1.** Redo the agentx conversion on current `origin/main`, on top of #208. Leave
   `tests/ruff.toml` for PR 2: the tests on main do not pass it yet.

3. **Rebase the branch onto `origin/main`** once PR 1 is merged. Main is 11 commits ahead and
   250 files overlap. Expect conflicts from:
   - #193, which adds a `microseconds` field to bench files this branch deletes;
   - new work under `aklr/core/contract/`, which this branch deletes;
   - #147, which renamed the harness skills (dropped the `aklr-` prefix);
   - #220, which reworked CI;
   - 199 schema JSON files updated by the measurement bot.

4. **Open PR 2.**
   - Recommit by path so it can be reviewed in order: schema and common, runtime, validation,
     op code and seam, the JSON files, tests, then harness, CI and docs.
   - The JSON commit is about 70% of the diff and is produced by `migrate_v1.py`. Reviewers read that
     script and the CI validation result instead of the files.
   - The description must say it changes `aklr/core/runtime/` and `aklr/core/schema/`, both protected.
   - Say in the description that every `evaluations` entry is reset and needs a full re-bench, that
     about 6.9k lines of tests are removed, and that four ops now check with flashinfer tolerances.

5. **Write PR 3, stacked on PR 2.**
   - Move `aklr/plugins` to the schema and the seam; migrate its 440 JSON files.
   - Restore `tests/plugins/**`.
   - Fix the remaining `aklr.core.contract` imports: `aklr/core/runtime/dispatch/_trace.py`,
     `harness/scripts/forge_task/assets/aklr_forge_driver.py` and `aklr_api_params.py`.
   - Fix the `DispatcherRegistry` import in `aklr/benchmarks/models/deepseekv4/profile_decode.py`.
   - Add seams for the model-specific routes that have none.

6. **Merge PR 2 and PR 3 back to back.** PR 2 on its own leaves main without working plugins.

## Open decisions

- Does PR 2 wait for PR 3? The plan above says yes.
- Are the fixes on `fix/core-validation-gaps` replaced by the variants work, or missing from it?

## Not verified yet

- No tests were run for this plan. That each PR leaves main importable and green is read from the
  code, not measured.
- The rebase has not been tried; the conflict list comes from overlapping files.
- That plugins stop working after PR 2 is read from imports and entry points. sglang and vLLM were
  not started.
