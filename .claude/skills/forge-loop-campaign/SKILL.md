---
name: forge-loop-campaign
description: >
  Run an autonomous optimization campaign on a kernel-less AKLR schema with KernelForge on an MI355X
  (gfx950) machine and deliver the kernel: prepare the task, run forge-loop, watch and steer it, stop or
  resume it, review and squash the kept rounds, publish the kernel into AKLR, and hand over the skill
  patch. Use when asked to optimize an AKLR op or schema with KernelForge or forge-loop, or to check on,
  resume, finish or deliver such a run.
allowed-tools: Read Edit Bash Grep Glob
---

# Forge-loop campaign on an AKLR schema

`harness/tools/forge_task/README.md` § Run a campaign has the six steps and their commands: read it first and
follow it. This skill adds what you decide, and what you must not forget, when you run the campaign yourself
rather than a person at a terminal.

## Before step 1

- Check the README's "Before you start" list on this machine. `python -m kernelforge.cli forge-loop --help`
  must print help.
- Settle the budget (`--max-hours`) and the model with the user if they did not give them: the agents spend
  money from step 2 on.
- Pick one card nobody uses (`rocm-smi --showmeminfo vram`, `rocm-smi --showpids`) and use it in steps 1, 2
  and 5. Another process on the card makes every timing on it meaningless.

## Step 2: launch in the background

Read `$RUN/baseline.json` first. AKLR's timer reads every dispatch at about 4.1 µs or more, because its bench runner
forces queue profiling, which costs that much per dispatch. A workload whose baseline runs near that floor reads as
parity in CI whatever the kernel does there (a 1.7 µs kernel against a 2.6 µs baseline read 4.09 against 4.10 µs).
Tell the user which workloads those are before you launch: the campaign can win nothing AKLR will show there.

You keep no terminal open, so start the loop in a process group of its own:

```bash
set -m   # the run gets a process group of its own, which the agents it starts share
nohup $RUN/forge_loop.sh --max-hours 12 --model claude-opus-5-5 --gpu 0 \
  --lanes 1 --no-experience-kb --no-kb-warmstart --no-pr-kb > $RUN/forge_loop.log 2>&1 &
echo $! > $RUN/forge_loop.pid
```

- When someone else may take the card, add `--gpu-pool N,M,…`: before a round the run moves to a free card of
  the pool, and waits while all are taken.
- Look in every few minutes, no more: `tail $RUN/forge_loop.log`, the dashboard, and each round's own account in
  `$RUN/workspace/forge_experiments/lessons/iter_NNN.md`.

## Stopping and resuming

- After the current round: `touch $RUN/workspace/.stop`. At once: `kill -- -$(cat $RUN/forge_loop.pid)`, which
  stops forge-loop and the agents it started; killing forge-loop alone leaves them running, and spending.
- Afterwards `pgrep -fa claude` must list no agent of this run.
- Resume as the README's step 4 says, launched as above with `forge_loop_resume.sh` in place of
  `forge_loop.sh`.

## Steps 5 and 6

- Start them only once the loop has ended: `$RUN/forge_loop.log` ends with forge-loop's result.
- Give forge-ablate several `--gpus` only when this machine runs this campaign alone.
- After forge-finalize, `python $W/driver.py --stress` must end with `stress: PASS` before you publish. AKLR's
  validate gives a kernel the GPU alone, so it passes one that is right only while every workgroup of a launch is
  resident: a split-K fold that polls for its partners and gives up with NaN. On a stream limited to part of the CUs
  such a kernel returns NaN. Do not deliver it; tell the user, and have its fold made wait-free first.
- Do not apply the skill patch yourself: hand `$RUN/knowledge_patch/distill.md` and the patch to the user.
- `python .claude/skills/flydsl-kernel-optimization/tools/kb_gaps.py $RUN/workspace` lists the skill pages the
  rounds opened and the questions they found no page for.
- Report: AKLR's verdict per workload from `aklr validate`, the delivered commit, what forge-ablate said each kept
  round is worth, and the skill's gaps.

## Pitfalls

- Two runs must not share an AKLR clone or a run directory.
- Clear FlyDSL's compile cache only at `FLYDSL_RUNTIME_CACHE_DIR` (default `~/.flydsl/cache`). A glob such as
  `/tmp/flydsl*` also deletes source trees.
- "Kineto recorded no kernels for a single call" on a busy machine is the profiler's, not the kernel's: see
  `flydsl-kernel-optimization/pitfalls/profiler_misses_kernels.md`.
