#!/usr/bin/env bash
# code-lint's lint range and validation base, from the event. Reads PUSH_BEFORE, FULL_LINT,
# AKLR_BOT_LOGIN, AKLR_BOT_PR and GH_TOKEN; writes validation_base, measured_by_bot and the
# pre-commit args to $GITHUB_OUTPUT.
set -eo pipefail

head=$(git rev-parse HEAD)
# The validation base: incumbents and the change set start here; empty validates everything.
validation_base=
# Whether the bot wrote the change's measured fields (the workloads' evaluations).
measured_by_bot=false
if [[ "$GITHUB_EVENT_NAME" == pull_request ]]; then
  # Checkout tests the PR merge commit; its first parent is the
  # base the merge was made on (the merge-base rule), not the payload's base.sha.
  base=$(git rev-parse HEAD^1)
  validation_base=$base
  measured_by_bot=$AKLR_BOT_PR
elif [[ -n "$PUSH_BEFORE" ]] && git cat-file -e "${PUSH_BEFORE}^{commit}" 2>/dev/null; then
  base=$PUSH_BEFORE
  # A replaced (cancelled) run benched nothing; [skip ci] commits have no run; an unknown status benches everything.
  replaced=$(gh run list -R "$GITHUB_REPOSITORY" -w ci.yaml -e push -c "$base" --json conclusion --jq 'length > 0 and all(.[]; .conclusion == "cancelled")') || replaced=true
  # A force push or a replaced run validates everything.
  if git merge-base --is-ancestor "$base" HEAD && [[ $replaced != true ]]; then validation_base=$base; fi
  # A push of the one commit that merged the bot PR: the PR run at the PR's last head,
  # whose AKLR_BOT_PR also checked who pushed it, passed the audit as the bot's.
  # A failed lookup leaves measured_by_bot=false: the audit stays strict.
  if [[ -n $validation_base && -n $AKLR_BOT_LOGIN && $(git rev-list --count "$base..HEAD") == 1 ]]; then
    pr_head=$(gh api "repos/$GITHUB_REPOSITORY/commits/$head/pulls" | jq -r --arg sha "$head" --arg bot "$AKLR_BOT_LOGIN" --arg repo "$GITHUB_REPOSITORY" \
      'first(.[] | select(.merge_commit_sha == $sha and .user.login == $bot and .head.ref == "aklr-perf/main" and .head.repo.full_name == $repo) | .head.sha) // ""') ||
      { echo '::warning::Could not look up the PR that merged this push; measured fields stay audited'; pr_head=; }
    if [[ -n $pr_head ]]; then
      # A run is matched by its branch, not head_sha alone: another PR can share the bot's commit.
      pr_run=$(gh api "repos/$GITHUB_REPOSITORY/actions/workflows/ci.yaml/runs?head_sha=$pr_head&event=pull_request" | jq -r --arg repo "$GITHUB_REPOSITORY" \
        'first(.workflow_runs[] | select(.head_branch == "aklr-perf/main" and .head_repository.full_name == $repo)) | "\(.conclusion) \(.html_url)"') ||
        { echo '::warning::Could not look up the bot PR run; measured fields stay audited'; pr_run=; }
      echo "🤖 Merged the bot PR at $pr_head; its latest PR run: $pr_run" | tee -a "$GITHUB_STEP_SUMMARY"
      [[ $pr_run != success\ * ]] || measured_by_bot=true
    fi
  fi
elif git rev-parse --verify HEAD^ >/dev/null 2>&1; then
  base=$(git rev-parse HEAD^)
else
  base=$(git hash-object -t tree /dev/null)
fi
echo "validation_base=$validation_base" >> "$GITHUB_OUTPUT"
echo "measured_by_bot=$measured_by_bot" >> "$GITHUB_OUTPUT"
if [[ "$FULL_LINT" == true ]]; then
  echo '🔎 Lint scope: all files, including inherited formatting debt.' | tee -a "$GITHUB_STEP_SUMMARY"
  echo 'args=--all-files --show-diff-on-failure' >> "$GITHUB_OUTPUT"
  exit 0
fi
echo "🔎 Lint range: $base → $head" | tee -a "$GITHUB_STEP_SUMMARY"
echo '::group::📄 Files in the lint range (hook filters still apply)'
git diff --name-only --diff-filter=ACMRTUXB "$base" "$head"
echo '::endgroup::'
echo "args=--from-ref $base --to-ref $head --show-diff-on-failure" >> "$GITHUB_OUTPUT"
