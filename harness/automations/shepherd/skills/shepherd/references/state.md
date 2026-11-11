# Persistent round state

Run the helper on the same host with the same `--root` every time. It expands `$VAR` and `~`, needs only
Python 3.10+ on Linux, and refuses a root that another user owns or could swap. Set `AKLR_STATE_DIR` from
`execution.state_dir` and `AKLR_ROUND_ID` to this invocation's unique identity.

```bash
STATE="python harness/automations/shepherd/skills/shepherd/scripts/state.py --root ${AKLR_STATE_DIR:?Set from execution.state_dir}"
$STATE init     # setup only
$STATE acquire --owner "${AKLR_ROUND_ID:?Set the unique round identity}"
$STATE show
$STATE save --token "$AKLR_ROUND_TOKEN" --revision "$AKLR_REVISION" --input "$AKLR_NEXT_STATE"
$STATE release --token "$AKLR_ROUND_TOKEN"
```

`acquire` returns a token and `{revision, state}`. To save, write the complete next **state object** (not
the envelope) to a temporary file and pass it with the current revision; use the returned revision for
the next save. A stale revision or wrong token is rejected. Never edit `state.json` or `owner.json` by
hand, and keep tokens out of shared logs.

## Ledger contents

`init` creates `campaigns: []`, `committed: {model_usd: 0, node_hours: 0}` and `last_main_sha: null`.
Setup adds `repository`, `budget_authorization` and the cumulative caps. Changing the authorization
erases no commitment.

Each campaign records:

- Identity: ID, repository, schema path, SHA-256 of the input, base SHA, config snapshot, creation time.
- Bounds: model, admitted model/node bounds, absolute deadline, automatic resume count.
- Work: phase, local/remote run and checkout paths, exact allocation handle, task handles.
- Operations: ID, kind, intent time, unique external name, deadline, result or `unknown`, last observation.
- Evidence: candidate digest, intended head, validator/environment versions, report paths, per-Point verdicts.
- Cleanup: stop intent, process/container/allocation observations, confirmed completion time.
- Delivery: branch, campaign marker, PR number/URL, head/base, CI/review state, merge commit, comment hashes.
- Next action, failure or blocked reason, delivered input identity, cooldown-until time.

Store each campaign's snapshots and reports under `runs/<campaign_id>/`. The helper checks none of this;
the entry skill and the owner keep it correct.

## Crash and recovery

The round lock has no expiry. If a round crashes, pause the scheduled task. An operator on that host
confirms the owner invocation has stopped, reconciles its saved intents against the external resources,
saves that evidence, and only then runs `release` with the token from `owner.json`. Never release on age
or add a force-unlock.

An empty, missing or corrupt directory does not mean nothing is running. Restore a consistent backup
while paused and reconcile resources created since it; never initialize over a lost ledger.

Back up `state.json`, `owner.json` and the evidence while paused, to another machine as well. Recovery
covers one host, not multi-host availability.
