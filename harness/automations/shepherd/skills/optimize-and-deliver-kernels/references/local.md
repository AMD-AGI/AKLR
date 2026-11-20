# Local pool

With `execution.cluster: local` the pool is this host: the round runs on the GPU machine itself (the
`aklr-gpu` runner of `ci:gpu-direct`), and its campaigns are detached Docker containers here. Never use
`remote-gpu`, even when its skill is installed: nothing is allocated. [pools](pools.md) and
[worker](worker.md) hold, with the `remote-gpu` commands replaced as below.

## The pool

- Its GPUs are `execution.gpus` (indices) and nothing else: the others are left to GPU CI, whose test
  containers see every device. `policy.max_parallel_campaigns` is at most their count.
- There is no allocation and no queue. Save the pool with `bounds.node_hours` (`execution.pool_hours`)
  and its `end` (now plus `pool_hours`) in one save, with no allocation operation. When it has ended and
  no campaign holds a GPU, settle it at the hours from its creation to that observation, then make the
  next one.
- Its release is confirmed when `docker ps -a --filter name=<campaign>-` lists nothing for every
  campaign that was on it. A reboot ends every container: reconcile each running operation as `unknown`.

## Campaign steps

Campaign directories are `<execution.work_dir>/<campaign>/{worker,secrets,validate}`, made by this user
before any container (what a container writes is root's; delete it in a container mounting `<campaign>`).

| On a cluster | Here |
|---|---|
| `remote-gpu bg -j <pool> --name <task> --image <image> -C <dir> -e K=V --mount <dir>` | `docker run -d --init --name <task> --label shepherd=<campaign> --network host --ipc host --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=<gpu> -e TMPDIR=/tmp -v <dir>:<dir> -w <dir> <image> timeout -k 60 <limit> bash -c '<step>'`, with `-v <campaign>/secrets:<campaign>/secrets:ro` only for a model-calling step |
| `remote-gpu logs -j <pool> <task>` | `docker logs <task>`; `docker inspect -f '{{.State.Status}} {{.State.ExitCode}}' <task>` |
| `remote-gpu exec -j <pool> -- <command>` | the command, in this shell |
| `remote-gpu push` / `pull` | `cp -P`, of the named regular files only (`test -f` and not `test -L`: a link is refused, never followed), checked as [worker](worker.md#isolation) says |
| `remote-gpu ls`, `release` | `docker ps -a --filter label=shepherd=<campaign>`; `docker stop -t 30 <task>`, then `docker rm <task>` |

The container sees every GPU, and `HIP_VISIBLE_DEVICES` alone picks the campaign's: a step never sets
it again (the `forge_task` examples set `0`; drop that), and `<limit>` plus the 60 seconds before the
kill stays within the campaign's bound.

The container name is the operation's unique external name: a lost reply is reconciled with `docker
inspect <task>`, never by starting it again. No `--rm`, so a finished step keeps its exit code until
cleanup removes it.

The worker's model access is the round's own: its `ANTHROPIC_*` environment variables when the round
has them (the workflow's gateway and key), else this user's Claude settings (`~/.claude/settings.json`,
only its `ANTHROPIC_*` variables); and this user's `claude` binary, as [worker](worker.md#isolation)
says for the cluster user. The round's GitHub token (`GH_TOKEN`) is never written to a file or passed to a container.
