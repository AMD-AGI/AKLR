#!/usr/bin/env bash
# publish_measurement.sh REPORT_DIR...: rebuild the bot branch aklr-perf/main on main from the
# lanes' validation reports. Carries the bot branch's still-current measurements, then this
# run's; the newest measurement of a file wins. The commit body lists "<sha> <path>" per
# published file (the next run's carry) and one Measured-SHA trailer per sha (lint's check).
# %q escapes newlines, so an artifact name cannot inject workflow commands.
# Reads BOT_LOGIN, GH_TOKEN, AITER_IMAGE, SGLANG_IMAGE and AKLR_MEASUREMENT_CURRENT; writes
# committed and prev to $GITHUB_OUTPUT.
set -eo pipefail

reports=()
for report in "$@"; do
  [[ ! -d $report/schemas ]] || reports+=("$report")
done
if [[ ${#reports[@]} == 0 ]]; then
  echo 'No measured schema.' >> "$GITHUB_STEP_SUMMARY"
  echo 'committed=false' >> "$GITHUB_OUTPUT"
  exit 0
fi
schema_path='^aklr/(ops|plugins(/[^/]+)+)/[^/]+/impls/[^/]+/[^/]+/[0-9a-f]{16}\.json$'
authored='del(.workloads[] | .speedup, .accuracy, .microseconds)'
# Every workload measured; speedup absent only on a failed one, microseconds wherever no baseline was timed.
shape='(.workloads | length > 0) and all(.workloads[];
  all(.speedup, .microseconds; . == null or (type == "number" and . > 0)) and
  (.accuracy == "passed" and .speedup != null or .accuracy == "failed"))'
manifest=$RUNNER_TEMP/manifest
# accept <file> <path> <sha>: copy over main's file iff only measured fields change.
accept() {
  if [[ $2 =~ $schema_path ]] &&
    [[ -f $1 && ! -L $1 && -f $2 && ! -L $2 ]] &&
    git --literal-pathspecs ls-files --error-unmatch -- "$2" >/dev/null 2>&1 &&
    a=$(jq -eS "$authored" "$2") && b=$(jq -eS "$authored" "$1") && [[ $a == "$b" ]] &&
    jq -e "$shape" "$1" >/dev/null; then
    cp -- "$1" "$2"
    echo "$3 $2" >> "$manifest"
  else
    printf '::warning::Dropped %q: not a measured-only change to a schema on main\n' "$2"
  fi
}
# current <sha>: the current rule, memoised per sha (the manifest has many lines but few shas).
declare -A current_of
current() {
  [[ -v current_of[$1] ]] ||
    { current_of[$1]=0; PYTHONPATH=. python -c "$AKLR_MEASUREMENT_CURRENT" "$1" || current_of[$1]=$?; }
  return "${current_of[$1]}"
}
prev=
if git fetch --no-tags origin aklr-perf/main; then
  prev=$(git rev-parse FETCH_HEAD)
  while read -r sha path; do
    [[ $sha =~ ^[0-9a-f]{40}$ && $path =~ $schema_path ]] || continue
    if ! current "$sha"; then
      printf '::warning::Dropped %q: measured at %s, before aklr/core changes on main\n' "$path" "$sha"
      continue
    fi
    git show "$prev:$path" > "$RUNNER_TEMP/carried.json"
    accept "$RUNNER_TEMP/carried.json" "$path" "$sha"
  done < <(git log -1 --format=%b "$prev")
fi
if current "$GITHUB_SHA"; then
  for report in "${reports[@]}"; do
    while IFS= read -r -d '' new; do
      accept "$new" "${new#"$report/schemas/"}" "$GITHUB_SHA"
    done < <(find "$report/schemas" -mindepth 1 \( -type f -o -type l \) -print0)
  done
else
  echo "::warning::Dropped this run's measurements: main has aklr/core changes since $GITHUB_SHA"
fi
if git diff --quiet; then
  echo 'No measured schema to publish.' >> "$GITHUB_STEP_SUMMARY"
  echo 'committed=false' >> "$GITHUB_OUTPUT"
  exit 0
fi
stray=$(comm -23 <(git diff --name-only | sort) <(cut -d' ' -f2 "$manifest" | sort -u))
[[ -z $stray ]] || { printf '::error::Changed outside the accepted schemas: %q\n' "$stray"; exit 1; }
python -m aklr export check
# The last measurement per changed path; a carried value already on main drops out.
published=$(awk 'NR == FNR {changed[$0]; next} $2 in changed {sha[$2] = $1} END {for (p in sha) print sha[p], p}' \
  <(git diff --name-only) "$manifest" | sort -k2)
echo "$(wc -l <<< "$published") measured schemas published" >> "$GITHUB_STEP_SUMMARY"
# jq -e fails the step on a missing or unsafe hardware name
hardware=$(jq -er '.platform.hardware | select(test("^[A-Za-z0-9_. -]+$"))' "${reports[0]}/report.json")
bot_id=$(gh api "users/$BOT_LOGIN" --jq .id)
git config user.name "$BOT_LOGIN"
git config user.email "$bot_id+$BOT_LOGIN@users.noreply.github.com"
image=$SGLANG_IMAGE
[[ $AITER_IMAGE == "$SGLANG_IMAGE" ]] || image="$AITER_IMAGE (aklr/ops), $SGLANG_IMAGE (plugins)"
{
  echo "perf: measurements on $hardware (run $GITHUB_RUN_ID, image $image)"
  echo
  echo "$published"
  echo
  cut -d' ' -f1 <<< "$published" | sort -u | sed 's/^/Measured-SHA: /'
} | git commit -aF -
echo 'committed=true' >> "$GITHUB_OUTPUT"
echo "prev=$prev" >> "$GITHUB_OUTPUT"
