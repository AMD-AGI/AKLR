# Pools

A pool is one node of `execution.cluster`, shared by up to one campaign per GPU. Use
`harness/skills/remote-gpu/SKILL.md` for every command. With `execution.cluster: local`, read
[local](local.md) instead of the sections on allocating and releasing.

## Allocating

Request with `--cluster <execution.cluster> alloc --no-wait -N 1 -t <execution.pool_hours>` and `--idle`
above the longest gap between rounds (180 minutes, since a ledger `wake_at` delays a round by at most two
hours), so the node is not released while its campaigns wait. `--no-wait` turns off alloc's race over the
cluster's accounts, so queue one request per `ACCOUNT:QOS` pair of the cluster's manifest (`ACCOUNTS` in
`~/.remote-gpu/clusters/<cluster>/cluster.env`), each with `-A <account> -q <qos>`, all saved under the one
pool's `operations` before submitting.

A full cluster leaves them queued, keeping their places: save each handle as `pending` and in later rounds
observe each with `remote-gpu ready -j <handle> --wait 60` (exit 205: still pending, keep waiting; never
cancel and resubmit one). The first to print OK starts the pool; release the pool's other requests at
once, confirmed gone by `ls`, and settle each from its own start if it ever started. Its OK line starts
the pool, and its end is the scheduler start plus the time limit: save it as the pool's `end`, without
which no campaign is placed on it. Its bounds are its node hours (nodes times the time limit).

`alloc` may print `CANCEL_FAILED job=<handle>` for a request it could not withdraw: that job is the
pool's too; record it and release it like the kept one before settling.

## Placing campaigns

Admit a campaign only onto a started pool with a free GPU whose end is after the campaign's deadline, so
a deadline never waits on the queue; request a new pool only when no started or pending one can take the
next candidate. Never put two campaigns on one GPU.

## Releasing

Release a pool once no campaign holds one of its GPUs and none will be placed on it this round: release
the exact allocation, then pull the job directory and run `remote-gpu ls`.

A pool's release is confirmed only when the containers and the exact allocation are observed gone:
`release` printed `OK released` without a partial-confirmation warning, or the job ended by itself and
every node left its `cleaned.<node>` marker in the pulled job directory; and a later successful
`remote-gpu ls` no longer lists the job. If those markers never come, it is unconfirmed but bounded once
a successful `ls` no longer lists the job and the pool's end has passed: free its GPUs (its campaigns
settle and leave `cleaning` as after a confirmed cleanup), do not settle the pool, and report the nodes
to the owner. Until the owner resolves it no round is skipped as idle. A failed `ls` observes nothing.

Settle the pool in the same save that confirms its release (see [state](state.md#settling)).
