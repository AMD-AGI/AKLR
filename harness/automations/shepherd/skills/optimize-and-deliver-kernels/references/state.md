# Persistent round state

Run the helper on the same host with the same `--root` every time. It expands `$VAR` and `~`, needs
Python 3.10+ with PyYAML on Linux, and refuses a root that another user owns or could swap.

```bash
STATE="python3 -I harness/automations/shepherd/skills/optimize-and-deliver-kernels/scripts/state.py --root <execution.state_dir>"
$STATE init                                                         # setup only
$STATE begin --config .shepherd/configuration.yaml [--now]          # first step of a round
$STATE show
$STATE save --round "$ROUND" --revision "$REVISION" --input "$NEXT_STATE"
$STATE end --round "$ROUND" --exit-code 0                           # last step of a round; 1 if it failed
```

`begin` takes the ledger for one round with a lease of `execution.round_hours` (default 2):

1. If `owner.json` names a round whose lease has not expired, it prints `{"busy": <id>}`, appends a line
   to `reports.md` and starts nothing.
2. Otherwise, if the last round ended cleanly (`end --exit-code 0`) and `--now` is not given, it prints
   `{"waiting_until": ...}` while the ledger's `wake_at` is ahead, at most two hours ahead, and the
   configuration file is unchanged since the last round.
3. Otherwise it records the new round in `owner.json` and prints `{"round", "lease_until",
   "recovered_from"}`. `recovered_from` names the previous round when it overran its lease without `end`
   or ended with a non-zero code.

The lease is a fencing token, not a liveness check. Once it expires, that round's `save` is refused, so
under the save-intent-first rule it cannot start any new side effect, even if its agent is still running.
`end` releases the ledger and writes `last_round.json` with the exit code.

`save` accepts a write only from the round named in `owner.json`, with an unexpired lease. Write the
complete next **state object** (not the envelope) to a temporary file and pass it with the current
revision; use the returned revision for the next save. It refuses a stale revision, and a state that:

- drops a campaign or pool, gives two the same `id`, lacks `bounds.model_usd` (campaign) or
  `bounds.node_hours` (pool), or changes `bounds` once saved: the committed budget is derived from them;
- saves a new campaign in any phase but `admitted`, moves a phase backward, leaves `ended`, or reaches
  `ended` from a phase other than `admitted`, `cleaning`, `ready_to_deliver` or `pr_open`;
- has more campaigns in `admitted`…`cleaning` than `policy.max_parallel_campaigns`, any of them without
  `pool` and `gpu`, or two on the same GPU;
- has a campaign above `campaign.max_model_usd`, or settles one between 0 and its bound;
- raises the committed model USD or node hours above `budget.total_model_usd` or
  `budget.total_node_hours` (a save that does not raise them passes, so lowering a cap never blocks
  settling).

Never edit `state.json`, `owner.json` or `last_round.json` by hand.

## Ledger contents

`init` creates `campaigns: []` and `last_main_sha: null`, and any missing parent directories (mode 700).
The caps and the authorization stay in the configuration. The committed budget is not stored: model USD
is the sum over campaigns, node hours the sum over `pools`, each its settled use, or its admitted bounds
until settled. Changing the
authorization or a cap erases no commitment. Each round may set the top-level `wake_at` that `begin` reads.

Each campaign records:

- Identity: `id` (unique, required), repository, schema path, SHA-256 of the input, regime (M range and workloads, or none),
  base SHA, config snapshot, creation time.
- Bounds: `bounds`, fixed at admission (`model`, `model_usd`, absolute `deadline`); the automatic resume
  count; and once cleanup is confirmed `settled_model_usd` with the observations it came from, outside
  `bounds`.
- Work: `phase`, `pool` id and `gpu` index, local/remote run and checkout paths, task handles.
- Operations: ID, kind, intent time, unique external name, deadline, result or `unknown`, last observation.
- Evidence: candidate digest, intended head, validation base, validator/environment versions, report paths,
  per-Point verdicts, and the last freshness check (main SHA, relevant paths).
- Cleanup: stop intent, the `release` output, each node's `cleaned.<node>` marker, the `ls` observation,
  and whether it is confirmed or unconfirmed-but-bounded, with the time.
- Delivery: branch, campaign marker, PR number/URL, head/base, CI/review state, merge commit, comment hashes.
- Next action, `outcome` once `ended` (delivered, merged, closed, failed, no gain, blocked, stopped), failure or blocked reason, delivered input identity, cooldown-until time.

Each pool (`pools`, created on first use) records its `id` (unique), `bounds` fixed before allocating
(`node_hours` as nodes times the time limit, and its `end`), the allocation intent and exact handle(s),
`CANCEL_FAILED` handles, scheduler start, the GPU each campaign holds, the release observations, and once
released `settled_node_hours`. `save` rejects dropping a pool or changing its `bounds`, as for campaigns.

## Phases

A campaign moves through
`admitted → preparing → optimizing → finalizing → validating → cleaning → ready_to_deliver → pr_open`
only on a confirmed result, never on a log line alone; skipping ahead is allowed, going back is not. It
leaves for `ended` (with its `outcome` and time) only from `admitted` before any container, from
`cleaning` once cleanup is confirmed, or from `ready_to_deliver` or `pr_open`. The deadline is absolute
across resumes; `max_auto_resumes` caps automatic resumes.

## Settling

Settle once, in the same save that confirms cleanup or release, from facts the worker cannot write:

- a campaign's model USD (`settled_model_usd`): 0 if no model-calling step (loop, finalize, distill) ever
  had a saved intent, else the admitted bound. Usage the worker reports is never used to lower it.
- a pool's node hours (`settled_node_hours`): for each allocation, from its scheduler start time (the
  allocation intent's time when the start was never observed) to the first observation of it gone, times
  its node count.

A campaign or pool not yet settled keeps its admitted bounds.

## Blocked contracts

A starting-kernel failure also records the blocked op contract with the tree hashes (the op directory
without `impls/`, and `harness/scripts/forge_task`) and the runtime revision that lift it.

Store each campaign's snapshots and reports under `runs/<campaign_id>/`. Beyond the rules above the helper
checks none of this; `optimize-and-deliver-kernels` and the owner keep it correct.

## Crash and recovery

A crashed or cut-off round needs no operator: once its lease expires, the next trigger takes over and
reconciles its saved intents first. Set `execution.round_hours` no longer than the automation's measured
run limit, so a round the runner kills is taken over soon after.

A round that hangs stays busy until its lease expires. Never delete `owner.json` or add a force-unlock.

An empty, missing or corrupt directory does not mean nothing is running. Restore a consistent backup
while paused and reconcile resources created since it; never initialize over a lost ledger.

Back up `state.json`, `owner.json`, `last_round.json` and the evidence while paused, to another machine as well. Recovery
covers one host, not multi-host availability.
