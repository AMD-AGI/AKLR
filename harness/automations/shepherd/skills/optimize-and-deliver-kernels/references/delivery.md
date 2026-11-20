# Delivery

Requires `allow_draft_pr`, current evidence, a frozen commit and confirmed cleanup.

Re-fetch main. If freshness from the validation base is stale (see [validation](validation.md)), or the
head does not merge cleanly into main (in a scratch worktree of main, `git merge --no-commit --no-ff
<head>` exits non-zero; remove the worktree after), mark evidence stale and hand the candidate to the
owner. Otherwise follow `harness/skills/open-pr/SKILL.md` with `branch_prefix` and `--draft`. The PR
head is the validated commit, never rebased; its evidence comes from that commit merged into the
validation base, which open-pr allows, and the body says so, with the
validation base and the main the last freshness check passed on. For a regime candidate the body states
the crossover and the workloads on each side.

- Save push and PR-create intents separately. Put a campaign marker in the body and look up the head
  branch's PRs, closed ones included, before creating: a lost reply means reconcile, never a second PR.
- Add `ci:gpu` once, only with `allow_ci_gpu_label`; never `ci:priority-high`. Comment only with `allow_pr_comments`, deduplicated
  by content hash; a comment is the only way shepherd tells an open PR's readers something.
- What needs the owner goes to `publication.attention_issue`, when set, the same way: intent first,
  deduplicated by content hash. It needs no `allow_pr_comments`: naming the issue authorizes it.
