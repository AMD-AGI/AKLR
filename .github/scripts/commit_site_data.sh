#!/usr/bin/env bash
# Rebuild one section of the site-data branch and push it as one commit.
#
# Usage: commit_site_data.sh SECTION MESSAGE COMMAND [ARGS...]
#   COMMAND runs with SECTION's directory in a fresh checkout of the branch tip appended as its last
#   argument and edits only that directory. A rejected push reruns COMMAND on the new tip, so a
#   concurrent publisher's commit is kept rather than rebased over. The first publish creates the
#   branch with no history.
# Reads GH_TOKEN (the bot App token, which may push site-data) and BOT_LOGIN.
set -euo pipefail

section=$1 message=$2
shift 2
[[ $section =~ ^[a-z][a-z0-9-]*$ ]] || { echo "invalid section: $section" >&2; exit 2; }
branch=site-data
work="$RUNNER_TEMP/site-data"
bot_id=$(gh api "users/$BOT_LOGIN" --jq .id)
auth="AUTHORIZATION: basic $(printf 'x-access-token:%s' "$GH_TOKEN" | base64 -w0)"

checkout_tip() {
  rm -rf "$work"
  git init -q "$work"
  cd "$work"
  git remote add origin "$GITHUB_SERVER_URL/$GITHUB_REPOSITORY.git"
  git config http.extraheader "$auth"
  git config user.name "$BOT_LOGIN"
  git config user.email "$bot_id+$BOT_LOGIN@users.noreply.github.com"
  local status=0
  git ls-remote --exit-code --heads origin "$branch" >/dev/null || status=$?
  if [[ $status == 0 ]]; then
    git fetch -q --depth=1 origin "$branch"
    git checkout -q -B "$branch" FETCH_HEAD
  elif [[ $status == 2 ]]; then
    git checkout -q --orphan "$branch"
  else
    return "$status"
  fi
  cd - >/dev/null
}

for attempt in 1 2 3 4 5; do
  checkout_tip
  mkdir -p "$work/$section"
  "$@" "$work/$section"
  git -C "$work" add --all -- "$section"
  if git -C "$work" diff --cached --quiet; then
    echo "site-data/$section is unchanged"
    exit 0
  fi
  git -C "$work" commit -q -m "$message"
  if git -C "$work" push -q origin "HEAD:refs/heads/$branch"; then
    echo "Pushed site-data/$section: $(git -C "$work" rev-parse --short HEAD)"
    exit 0
  fi
  echo "::warning::site-data moved during publish (attempt $attempt); rebuilding on the new tip"
  sleep $((attempt * 5))
done
echo "::error::could not push site-data/$section after 5 attempts" >&2
exit 1
