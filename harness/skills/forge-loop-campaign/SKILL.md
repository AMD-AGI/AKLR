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

`harness/scripts/forge_task/README.md` § Run a campaign has the six steps and their commands: read it first and
follow it. This skill adds what you decide, and what you must not forget, when you run the campaign yourself
rather than a person at a terminal.

## Before step 1

- Check the README's "Before you start" list on this machine. `python -m kernelforge.cli forge-loop --help`
  must print help.
- Settle the budget (`--max-hours`) and the model with the user if they did not give them: the agents spend
  money from step 2 on.
- **Split small M from large M: one run per regime, never one run over both.** A run scores the mean over its
  workloads, so a schema whose small-M Points (decode: launch, latency or the weight read) sit beside large-M ones
  (prefill: compute) gets one kernel tuned for whichever Points outnumber the others. Read each workload's bound
  first: the weight bytes over the read roof against the FLOPs over the compute roof; the M where they cross
  (peak FLOP/s / read bandwidth x weight bytes per element, about 230 for bf16 on MI355X) splits a GEMM, and a
  long flat stretch of time over M marks the latency-bound end. Give each regime its own run, card and schema
  file holding only its workloads; each delivers its own kernel file, and dispatch serves every Point with the
  best one. forge_task reads a schema file at a path shaped like AKLR's, `<op>/impls/<language>/<name>/<file>.json`;
  keep each regime's file outside the AKLR clone, e.g. `~/aklr-regimes/gemm/impls/flydsl/<name>/small_m.json`. In
  the clone, `aklr export check` in step 6 rejects it: its name is not its content hash.
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
- `python harness/skills/flydsl-kernel-optimization/tools/kb_gaps.py $RUN/workspace` lists the skill pages the
  rounds opened and the questions they found no page for.
- Report: AKLR's verdict per workload from `aklr validate`, the delivered commit, what forge-ablate said each kept
  round is worth, and the skill's gaps.

## Step 7: put what the campaign taught into the skill

forge-distill's patch (`$RUN/knowledge_patch/`) is a proposal. Review every item before it goes in, as
`flydsl-kernel-optimization/CONTRIBUTING.md` says:

- Check each claim against the run's files (`$W/forge_experiments/`: `lessons/`, `analysis/`, `ablation.md`),
  FlyDSL's source or the ISA. A number carries the ruler it was taken with; a gain inside the noise, or measured by
  another ruler than AKLR's, is not a gain.
- Search the skill for a page that already says it, and extend that page rather than write a second one.
- An edit to an existing page goes in place; a new page goes to `staging/` with its `<slug>.review.md`, for a person
  to approve with `tools/kb_approve.py` and link afterwards.
- `python tools/kb_lint.py --errors-only` and `python tools/kb_index.py --check` pass, and `tools/kb_numbers_guard.py`
  accounts for every number an edited page gains or loses.
- Several campaigns at once leave patches that touch the same pages: review them together, one owner per page.

What the campaign taught about the kernel goes into the pull request that delivers the kernel's schema file; a kernel
held back keeps its knowledge back with it. What it taught about the method -- this skill, forge_task, its driver, a
ruler -- goes into a pull request of its own that changes the method.

## Pitfalls

- Two runs must not share an AKLR clone or a run directory.
- Clear FlyDSL's compile cache only at `FLYDSL_RUNTIME_CACHE_DIR` (default `~/.flydsl/cache`). A glob such as
  `/tmp/flydsl*` also deletes source trees.
- "Kineto recorded no kernels for a single call" on a busy machine is the profiler's, not the kernel's: see
  `flydsl-kernel-optimization/pitfalls/profiler_misses_kernels.md`.
