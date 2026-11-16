# Persistent round state

Run the helper on the same host with the same `--root` every time. It expands `$VAR` and `~`, needs only
Python 3.10+ on Linux, and refuses a root that another user owns or could swap.

```bash
STATE="python3 -I harness/automations/shepherd/skills/optimize-and-deliver-kernels/scripts/state.py --root <execution.state_dir>"
$STATE init                                                   # setup only
$STATE round --config .shepherd/configuration.yaml -- <agent CLI> <prompt>   # the scheduler's command
$STATE show
$STATE save --revision "$REVISION" --input "$NEXT_STATE"     # inside a round only
```

`round` owns the ledger for one agent run:

1. If `owner.json` names a round that may still be running, it prints `{"busy": <id>}`, appends a line to
   `reports.md` and exits 0 without starting the agent. A round counts as running while its wrapper
   process (pid and start time) exists, or any process this user can see carries its `SHEPHERD_ROUND_ID`.
   Another PID namespace on the same boot is undecidable and also counts as running. Age never counts.
2. Otherwise, if the last round ended cleanly (its agent exited 0) and `--now` is not given, it skips the trigger with `{"waiting_until": ...}`
   while the ledger's `wake_at` is ahead, at most two hours ahead, and the configuration file is unchanged
   since the last round.
3. It records itself in `owner.json` and starts the agent with `SHEPHERD_ROUND_ID` and `SHEPHERD_STATE_DIR`, plus
   `SHEPHERD_RECOVERED_FROM=<id>` when the previous round died (its `owner.json` was left behind) or ended
   without exit code 0 (failed, killed by a signal, or its agent never started). SIGTERM or SIGINT to the
   wrapper terminates the agent, and kills it 10 seconds later if it is still running.
4. When the agent exits, it kills every remaining process carrying its round id, writes `last_round.json`
   with the agent's exit code, removes `owner.json`, and exits with the agent's code.

`save` accepts a write only from the round named in `owner.json`, read from `SHEPHERD_ROUND_ID`. Write the
complete next **state object** (not the envelope) to a temporary file and pass it with the current
revision; use the returned revision for the next save. A stale revision is rejected, and so is a state
that drops a campaign, gives two the same `id`, or changes a campaign's `bounds` once saved: the
committed budget is derived from them. Never edit `state.json`, `owner.json` or `last_round.json` by hand.

## Ledger contents

`init` creates `campaigns: []` and `last_main_sha: null`, and any missing parent directories (mode 700).
The caps and the authorization stay in the configuration. The committed budget is not stored: model USD
is the sum over campaigns, node hours the sum over `pools`, each its settled use, or its admitted bounds
until settled. Changing the
authorization or a cap erases no commitment. Each round may set the top-level `wake_at` its wrapper reads.

Each campaign records:

- Identity: `id` (unique, required), repository, schema path, SHA-256 of the input, regime (M range and workloads, or none),
  base SHA, config snapshot, creation time.
- Bounds: `bounds`, fixed at admission (model, admitted model USD, absolute deadline); the automatic resume
  count; and once cleanup is confirmed the settled model USD with the observations it came from, outside
  `bounds`.
- Work: phase, pool id and GPU index, local/remote run and checkout paths, task handles.
- Operations: ID, kind, intent time, unique external name, deadline, result or `unknown`, last observation.
- Evidence: candidate digest, intended head, validation base, validator/environment versions, report paths,
  per-Point verdicts, and the last freshness check (main SHA, relevant paths).
- Cleanup: stop intent, the `release` output, each node's `cleaned.<node>` marker, the `ls` observation,
  and whether it is confirmed or unconfirmed-but-bounded, with the time.
- Delivery: branch, campaign marker, PR number/URL, head/base, CI/review state, merge commit, comment hashes.
- Next action, failure or blocked reason, delivered input identity, cooldown-until time.

Each pool (`pools`, created on first use) records its `id` (unique), `bounds` fixed before allocating
(node hours as nodes times the time limit, and its end), the allocation intent and exact handle(s),
`CANCEL_FAILED` handles, scheduler start, the GPU each campaign holds, the release observations, and once
released the settled node hours. `save` rejects dropping a pool or changing its `bounds`, as for campaigns.

A starting-kernel failure also records the blocked op contract with the tree hashes (the op directory
without `impls/`, and `harness/scripts/forge_task`) and the runtime revision that lift it.

Store each campaign's snapshots and reports under `runs/<campaign_id>/`. The helper checks none of this;
`optimize-and-deliver-kernels` and the owner keep it correct.

## Crash and recovery

A crashed round needs no operator: the next trigger takes over once none of its processes remain on this
host, and that round reconciles the saved intents first. A wrapper killed while its agent lives stays busy
until the agent exits.

A round that hangs while alive stays busy. To end it, an operator on that host finds its processes by id
(`grep -l "SHEPHERD_ROUND_ID=<id>" /proc/[0-9]*/environ`) and kills them; the next trigger takes over. Never
release on age, delete `owner.json`, or add a force-unlock.

An empty, missing or corrupt directory does not mean nothing is running. Restore a consistent backup
while paused and reconcile resources created since it; never initialize over a lost ledger.

Back up `state.json`, `owner.json`, `last_round.json` and the evidence while paused, to another machine as well. Recovery
covers one host, not multi-host availability.
