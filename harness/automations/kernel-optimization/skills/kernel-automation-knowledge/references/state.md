# Persistent round state

Run the helper on the configured control host with the same absolute `--root` every time. Python 3.10+
standard library on Linux is enough; no GPU packages are imported. Set `AKLR_STATE_DIR` from the local
configuration's `execution.state_dir`, and set `AKLR_ROUND_ID` to this invocation's unique identity.
Keep deployment paths and machine names in local configuration, never in the shared skill files.

```bash
: "${AKLR_STATE_DIR:?Set from local execution.state_dir}"
python harness/automations/kernel-optimization/skills/kernel-automation-knowledge/scripts/state.py --root "$AKLR_STATE_DIR" init
python harness/automations/kernel-optimization/skills/kernel-automation-knowledge/scripts/state.py --root "$AKLR_STATE_DIR" acquire --owner "${AKLR_ROUND_ID:?Set the unique round identity}"
python harness/automations/kernel-optimization/skills/kernel-automation-knowledge/scripts/state.py --root "$AKLR_STATE_DIR" show
```

`acquire` returns an opaque token and snapshot `{revision, state}`. Keep that token through this round.
To save, write the complete next **state object** (not the envelope) to a temporary JSON file.
Set `AKLR_NEXT_STATE` to its path, and retain `AKLR_ROUND_TOKEN` and `AKLR_REVISION` from the helper result:

```bash
python harness/automations/kernel-optimization/skills/kernel-automation-knowledge/scripts/state.py --root "$AKLR_STATE_DIR" save \
  --token "${AKLR_ROUND_TOKEN:?Use the acquired token}" --revision "${AKLR_REVISION:?Use the current revision}" \
  --input "${AKLR_NEXT_STATE:?Set the next snapshot path}"
python harness/automations/kernel-optimization/skills/kernel-automation-knowledge/scripts/state.py --root "$AKLR_STATE_DIR" release --token "${AKLR_ROUND_TOKEN:?Use the acquired token}"
```

Use the returned revision for the next save. It rejects a stale revision or wrong token. Helpers write via
atomic rename and fsync; do not edit `state.json` or `owner.json` directly during normal operation. Keep input
files and tokens out of shared logs. The token coordinates cooperating agents; it is not an authentication boundary.

## State contract maintained by Knowledge

Initial `state` contains `campaigns: []`, `committed: {model_usd: 0, node_hours: 0}`, and `last_main_sha: null`.
On first setup add `repository`, `budget_authorization` and the approved cumulative caps. Changing the budget
authorization does not erase historical commitments. Never refund a failed campaign or reset counters on a new day.

Each campaign records:

- Identity: ID, repository, schema path, SHA-256 of input bytes, fixed base SHA, config snapshot, creation time.
- Bounds: model, admitted model/node upper bounds, absolute deadline, automatic resume count.
- Work: phase, local/remote run and checkout paths, exact allocation handle, stage/task handles.
- Operations: stable operation ID, kind, intent time, unique external name, deadline, result/unknown and last observation.
- Candidate/evidence: candidate digest, intended head, validator/environment versions, report paths and per-Point verdicts.
- Cleanup: stop intent, process/container/allocation observations and confirmed completion time. Not just a boolean from the agent.
- Delivery: deterministic branch, campaign marker, PR number/URL, head/base, CI/review state, merge commit, comment hashes.
- Next action, failure/blocked reason, delivered input identity, cooldown-until timestamp.

Record operation intent and budget commitment in the same snapshot before the side effect. Keep completed
campaigns to preserve deduplication, PR history and spend; active work includes all records whose cleanup is
not confirmed, even when their phase says failed or stopped. Store snapshots/reports under `runs/<campaign_id>/`.
The script does not enforce this domain schema or verify agent observations. The Knowledge skill and owner
are responsible for those facts until the trusted service described in the M1 design is implemented.

## Crash and recovery

The per-round owner survives process exit; it has **no time-based expiry**. `release` ends only the current
round's lock, not a remote task or campaign. All snapshots retain remote occupancy until cleanup is proven.
An empty/missing/corrupt directory is not evidence that there is no running work.

If a round crashes, pause its Cursor Automation. On the control host, an operator must establish that its
owner invocation is no longer running, inspect saved intents and all corresponding external resources, and
save evidence of the reconciliation. Only then use `release --token` with the token from `owner.json`.
Do not release a lock on age alone and do not add a force-unlock/retry timer. Resume the same state directory;
unknown external actions remain blocked. If state is damaged, restore a consistent backup while paused and
reconcile resources created since the snapshot before enabling admission. Never initialize over a lost ledger.

Back up `state.json`, `owner.json` and referenced evidence with the automation paused and no live round writer.
Keep a copy outside the control machine. The first version provides single-host recovery, not multi-host availability.
