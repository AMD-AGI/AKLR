# Persistent round state

Run the helper on the same host with the same `--root` every time. It expands `$VAR` and `~`, needs
Python 3.10+ with PyYAML on Linux, and refuses a root that another user owns or could swap.

```bash
STATE="python3 -I harness/automations/shepherd/skills/optimize-and-deliver-kernels/scripts/state.py --root <execution.state_dir>"
$STATE init                                                         # setup only
$STATE begin --config .shepherd/configuration.yaml [--now] [--observation "$SEEN"]   # first step of a round; --root defaults to execution.state_dir
$STATE show
$STATE save --round "$ROUND" --revision "$REVISION" --patch "$PATCH"
$STATE end --round "$ROUND" --exit-code 0                           # last step of a round; 1 if it failed
```

`begin` takes the ledger for one round with a lease of `execution.round_hours` (default 2):

1. If `owner.json` names a round whose lease has not expired, it prints `{"busy": <id>}` and starts
   nothing.
2. Otherwise, if the last round ended cleanly (`end --exit-code 0`) and `--now` is not given, it prints
   `{"waiting_until": ...}` while the ledger's `wake_at` is ahead, at most two hours ahead, and the
   configuration file is unchanged since the last round. The limit is 24 hours when `--observation`
   equals the ledger's `idle_observation` (see [idle rounds](#idle-rounds)).
3. Otherwise it records the new round in `owner.json` and prints `{"round", "lease_until",
   "recovered_from"}`. `recovered_from` names the previous round when it overran its lease without `end`
   or ended with a non-zero code. A round whose `end` was cut off after recording its exit code has
   ended.

The lease is a fencing token, not a liveness check. Once it expires, that round's `save` is refused, so
under the save-intent-first rule it cannot start any new side effect, even if its agent is still running.
`end` releases the ledger and writes `last_round.json` with the exit code.

`save` accepts a write only from the round named in `owner.json`, with an unexpired lease. Write a
**patch** to a temporary file and pass it with the current revision; it prints the next revision. The
patch is a JSON merge patch of the state object: only the keys it names change, a mapping merges key by
key, `null` removes a key, and a list is replaced whole. `campaigns` and `pools` are addressed by id, and
one patch may change several entries, which are then saved together:

```json
{"campaigns": {"<id>": {"phase": "optimizing", "operations": {"loop": {"intent_at": "...", "deadline": "..."}}}},
 "pools": {"<id>": {"end": "2026-11-12T08:30:00+00:00"}},
 "wake_at": "2026-11-11T08:30:00+00:00"}
```

The top-level `wake_at` and `idle_observation` are the exception: every save removes them unless its
patch sets them, so a round that sets no `wake_at` leaves none behind.

It refuses every save unless `mode` is `run` or `drain`, a stale revision, and a patch whose result:

- removes a campaign or pool, has an `id` that is not letters, digits, `_` or `-`, lacks
  `bounds.model_usd` (campaign) or `bounds.node_hours` (pool), or changes `bounds` once saved: the
  committed budget is derived from them;
- removes an operation, or adds one that already has a `result`: the intent is saved first, in an
  earlier save than its result;
- sets or changes an entry's `archive`: the helper alone writes it;
- sets `idle_observation` while a campaign has not ended or a pool is not settled, or to anything but
  the observation this round's `begin` was given;
- saves a new campaign in any phase but `admitted`, moves a phase backward, leaves `ended`, or takes a
  campaign off its GPU (to `ready_to_deliver`, `pr_open` or `ended`) from a phase other than `cleaning`,
  except `admitted` straight to `ended`;
- has a campaign in `admitted`…`cleaning` with `settled_model_usd`, or one past them without it; settles
  one between 0 and its bound, or changes a settled amount;
- settles a pool in the save that creates it, while a campaign holds one of its GPUs, at a negative or
  non-numeric amount, or a second time;
- has a campaign in `admitted`…`cleaning` without `pool` and `gpu`, or two on the same GPU: the
  campaigns alone record who holds a GPU, a pool keeps no copy;
- places a campaign on a pool that has no `end` yet (not started), is settled, or ends at or before the
  campaign's `bounds.deadline`;
- adds a campaign in `mode: drain`, or above `campaign.max_model_usd`, or a holder of a GPU beyond
  `policy.max_parallel_campaigns`;
- adds a campaign or pool that takes the committed model USD or node hours above
  `budget.total_model_usd` or `budget.total_node_hours`.

The caps hold only against a save that adds a campaign, a pool or a holder of a GPU, so lowering one
blocks new admissions and never the saves that clean up, settle or end what was already admitted. A cap
left `null` blocks them the same way, and the refusal names it. A pool
settles at what it used, even above its bound.

Never edit `state.json`, `owner.json`, `last_round.json` or an archive by hand.

## Archive

The save that ends a campaign or settles a pool writes the whole entry to `runs/<id>/campaign.json` or
`runs/<id>/pool.json` and keeps in the ledger only its `archive` path and what deduplication, cooldown and
the budget read: `id`, `bounds`, the settled use, and for a campaign `phase`, `outcome`, `schema_path`,
`input_sha256`, `regime`, `contract`, `backend`, `branch`, `delivered_input`, `cooldown_until`,
`observed_model_usd`. So put
everything worth keeping in the entry before or in that save. Read the archive when a finished entry's
history is needed; a later patch may still add a key to what the ledger kept, such as a lifted cooldown.

## Idle rounds

A round that found nothing to do, with every campaign ended and every pool settled, waits only on the
base branch's head, the open PRs' heads, this checkout's commit, `readiness.md` and the issue requests.
`scripts/observation.py --config .shepherd/configuration.yaml` prints their fingerprint, taken before
the round begins. Such a round saves the fingerprint it gave `begin` (never one taken later, which would
cover a change the round did not look at) as the top-level `idle_observation`, with a `wake_at` at the
next time-based change (a cooldown's end), at most 24 hours ahead; `begin --observation` with the same
fingerprint then skips until that `wake_at`.

## Ledger contents

`init` creates `campaigns: []` and `last_main_sha: null`, and any missing parent directories (mode 700).
The envelope records the layout's `schema`. `begin` migrates an older ledger before the round starts,
keeping the files it replaces in `schema-<old>/`, and refuses to start if the result would break a rule
above; a ledger written by a newer helper is refused. This happens in every mode, `observe` included.
Update the helper while no round is running: `save` is refused on a ledger not yet migrated, and a round
of the earlier helper is taken over as overrun. If `begin` refuses to migrate, change nothing by hand:
check out the earlier helper, let its rounds finish and settle what the message names, then update.
The caps and the authorization stay in the configuration. The committed budget is not stored: model USD
is the sum over campaigns, node hours the sum over `pools`, each its settled use, or its admitted bounds
until settled. Changing the
authorization or a cap erases no commitment. Each round may set the top-level `wake_at` that `begin` reads; `save` keeps one beyond `begin`'s limit
(two hours, 24 with an `idle_observation`) as that limit.

Each campaign records:

- Identity: `id` (unique, required), repository, schema path, SHA-256 of the input, regime (M range and workloads, or none),
  base SHA, the `contract_key` its candidate was listed with, config snapshot, creation time.
- Bounds: `bounds`, fixed at admission (`model`, `model_usd`, absolute `deadline`); the automatic resume
  count; and once cleanup is confirmed `settled_model_usd` with the observations it came from, outside
  `bounds`.
- Work: `phase`, `pool` id and `gpu` index, local/remote run and checkout paths, task handles.
- Operations: `operations`, a mapping from each operation's ID to its kind, intent time, unique external
  name, deadline, `result` (or `unknown`) and last observation. A pool records its allocation and release
  the same way.
- Evidence: candidate digest, intended head, validation base, validator/environment versions, report paths,
  per-Point verdicts, and the last freshness check (main SHA, relevant paths).
- Cleanup: stop intent, the `release` output, each node's `cleaned.<node>` marker, the `ls` observation,
  and whether it is confirmed or unconfirmed-but-bounded, with the time.
- Request: `issue`, the number of the issue whose [request](requests.md) admitted it; `pr_url` and a
  one-line `summary` of the result, which that issue's report shows.
- Delivery: branch, campaign marker, PR number/URL, head/base, CI/review state, merge commit, comment hashes.
- Next action, `outcome` once `ended` (delivered, merged, closed, failed, no gain, blocked, stopped), failure or blocked reason, delivered input identity, cooldown-until time.

Each pool (`pools`, created on first use) records its `id` (unique), `bounds` fixed before allocating
(`node_hours` as nodes times the time limit), the allocation intent and exact handle(s),
`CANCEL_FAILED` handles, scheduler start and, once started, its `end` (ISO 8601 with a UTC offset; a
campaign is placed only on a pool that has one), the release observations, and once released
`settled_node_hours`. Which campaign holds which GPU is read from the campaigns' `pool` and `gpu`. `save` rejects dropping a pool or changing its `bounds`, as for campaigns.

## Phases

A campaign moves through
`admitted → preparing → optimizing → finalizing → validating → cleaning → ready_to_deliver → pr_open`
only on a confirmed result, never on a log line alone; skipping ahead is allowed up to `cleaning`, going
back is not. It holds its GPU through `cleaning` and leaves it only from there, once cleanup is
confirmed: to `ready_to_deliver`, or to `ended` (with its `outcome` and time). It may also end from
`admitted` before any container, and from `ready_to_deliver` or `pr_open`. The deadline is absolute
across resumes; `max_auto_resumes` caps automatic resumes.

## Settling

Settle once, from facts the worker cannot write: a campaign in the save that confirms its cleanup and
moves it from `cleaning` to `ready_to_deliver` or `ended` (or ends it from `admitted`), a pool in the
save that confirms its release:

- a campaign's model USD (`settled_model_usd`): 0 if no model-calling step (loop, finalize, distill) ever
  had a saved intent, else the admitted bound. Usage the worker reports is never used to lower it.
- a pool's node hours (`settled_node_hours`): for each allocation, from its scheduler start time (the
  allocation intent's time when the start was never observed) to the first observation of it gone, times
  its node count.

A campaign or pool not yet settled keeps its admitted bounds.

When this host can read the provider's own billing for the campaign's key or gateway, also save it as
`observed_model_usd` with its source and time. It is what the report shows as spent; it never lowers
the commitment, and the worker's own usage report is never its source.

## Blocked contracts

A starting-kernel failure adds an entry to the top-level `blocked_contracts` list (a list is replaced
whole, so send every entry): `contract`, `op`, `at`, `campaigns`, `cause`, and `lifts_when_changed`, the
campaign's `contract_key`. The key holds `op_dir_tree_without_impls` (the op directory on main, whose
measurement updates under `impls/` are expected), `forge_task_tree` (this checkout's
`harness/scripts/forge_task`) and `runtime_revision`. `candidate.py list` skips every candidate of that
`op` while its current key equals a recorded one, so the block lifts when any of the three changes and
never by time.

Store each campaign's snapshots and reports under `runs/<campaign_id>/`, under any name but
`campaign.json`, which is the helper's archive. Beyond the rules above the helper
checks none of this; `optimize-and-deliver-kernels` and the owner keep it correct.

## Crash and recovery

A crashed or cut-off round needs no operator: once its lease expires, the next trigger takes over and
reconciles its saved intents first. Set `execution.round_hours` no longer than the automation's measured
run limit, so a round the runner kills is taken over soon after.

A round that hangs stays busy until its lease expires. Never delete `owner.json` or add a force-unlock.

An empty, missing or corrupt directory does not mean nothing is running. Restore a consistent backup
while paused and reconcile resources created since it; never initialize over a lost ledger.

Back up `state.json`, `owner.json`, `last_round.json`, `runs/` and the evidence while paused, to another machine as well. Recovery
covers one host, not multi-host availability.
