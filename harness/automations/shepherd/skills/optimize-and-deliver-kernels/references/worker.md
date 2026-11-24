# Campaign worker

A campaign runs KernelForge in containers on its own GPU of a pool. Workers never run on this host, even
when it has a free card.

## Isolation

Start every step through `remote-gpu bg -j <pool> --name <campaign>-<step> --image <execution.image> -C
<campaign>/worker -e HIP_VISIBLE_DEVICES=<gpu> -e TMPDIR=/tmp` with `--mount <campaign>/worker` only
(inside, its GPU is device 0); model-calling steps (loop, finalize, distill) add `--mount
<campaign>/secrets:ro`, and no other step does. Make the campaign's directories with a plain `remote-gpu
exec -- mkdir -p` first: containers run as root, so a directory they create, and everything they write, is
root's (delete those in a container mounting `<campaign>`).

The worker gets the model only through `<campaign>/secrets/env.sh` (mode 600), made on the node from the
cluster user's Claude settings holding only their `ANTHROPIC_*` variables, which each model-calling step
sources; the image has no `claude`, so copy the cluster user's `claude` binary to `<campaign>/worker/bin/`
and put it on `PATH`.

In the worker, clone AKLR at the recorded main and Hyperloom at `execution.runtime_revision` from their
public URLs, replace the clone's `harness/scripts/forge_task` with this checkout's (pushed read-only
beside the campaign, never fetched from main: the pack runs with the forge_task it was tested with; the
validator clone does the same, and the ledger records its tree hash), and install as forge_task's README
says into a venv in the worker whose `image.pth` holds `import site; site.addsitedir("<the image python's
site-packages>")` (that python is itself a venv, so `--system-site-packages` would hide torch, and a
plain path line would skip the image's editable installs such as sglang), with the baseline AITER from
AKLR CI's `AKLR_BASELINE_AITER`.

The worker then cannot see the validator, earlier evidence, the cluster `$HOME` or GitHub and state
credentials. Nothing the worker wrote is trusted on this host or by the validator: pull only the named
files delivery needs (never the worker directory, whose `.git` would run here), refuse any that is not a
regular file (`find <pulled> ! -type f ! -type d` prints nothing), and run no code from it.

## Running the campaign

Follow `harness/skills/forge-loop-campaign/SKILL.md` for every command, as its "Called by an automation"
section says: no timers, no KernelForge GPU flags (the container's `HIP_VISIBLE_DEVICES` is the
campaign's card), and the stress gate run in the validator (see [validation](validation.md)). Prepare
with `--kernel-backend <the candidate's language>`.
For a Triton or Gluon kernel, finalize without `--review-guide` and skip distill.

Start long steps as remote background tasks and return their handle. KernelForge has no spending limit of
its own, so the model bound holds only through time: every model-calling step gets a `timeout` whose
hours times `campaign.model_usd_per_hour` fit in the model USD the campaign has left. For kernel
reasoning read `harness/skills/kernel-optimization-book/SKILL.md`, and `harness/skills/aklr-kernel-rules/SKILL.md` for
what AKLR requires of a kernel.

## Cleanup

After collecting artifacts, stop the campaign's containers (`docker stop -t 30` by name), observe them
gone, and delete `<campaign>/secrets` and the root-owned worker files. Failed and blocked runs are
cleaned too.

Cleanup is confirmed when every container the campaign started is observed gone: each of its tasks ended
(`remote-gpu logs -j <pool> <task>`) and `remote-gpu exec -j <pool> -- docker ps -a --filter name=<task
container>` lists none, or its pool's release is confirmed. Until then it holds its GPU, whatever its
phase. Save what was observed, settle the campaign and move it to `ready_to_deliver` or `ended` in that one save (see [state](state.md#settling)),
then free its GPU.
