---
name: geak-campaign
description: >
  Optimize a kernel-less AKLR schema with GEAK (AMD-AGI/GEAK) instead of KernelForge, and deliver the kernel back
  into AKLR: prepare the task with forge_task, give it GEAK's task format, run GEAK's kernel_workflow on it, then
  stress-test, publish and validate the kernel GEAK ends with. Use when asked to run GEAK on an AKLR op or schema.
allowed-tools: Read Edit Bash Grep Glob
---

# GEAK on an AKLR schema

AKLR measures and GEAK optimizes. `forge_task prepare` turns the schema into a task whose `driver.py` checks a kernel
with `aklr validate` and times it with AKLR's own timer (`harness/tools/forge_task/README.md`, step 1). This skill
gives that task GEAK's format: a `config.yaml` whose compile, correctness and performance commands run
`task_runner.py`, which calls the driver. GEAK's kernel_workflow then optimizes `kernel.py` against those commands,
and `forge_task publish` writes the kernel it ends with back into `aklr/ops/`.

## Before you start

- forge_task README's "Before you start", without KernelForge: AKLR installed from your clone, FlyDSL, ruff, the
  claude CLI logged in (GEAK needs 2.1.177 or later).
- GEAK: `git clone https://github.com/AMD-AGI/GEAK.git && GEAK_SKIP_BOOTSTRAP=1 pip install -e GEAK`. Without
  `GEAK_SKIP_BOOTSTRAP` the install clones GEAK a second time and installs another claude CLI.
- Node.js 20 or later, to run GEAK without an interactive session. `node GEAK/interface/runtime/engine/selftest.mjs`
  checks GEAK's runtime, with no GPU and no key.
- Knowledge: GEAK reads its own (`kernel_workflow/knowledge/`, `perf_knowledge/`) and adds what a run learns to
  `kernel_workflow/knowledge/learned/` and `kb_artifacts/` in its checkout. Its `perf_knowledge_dir` takes only
  GEAK's layout, so AKLR's skills are not given to it. The agents read AKLR's rules through the task's `program.md`
  and `aklr/ops/README.md`.

## 1. Prepare

Split regimes and pick a card as `forge-loop-campaign` says in "Before step 1". Keep a regime's schema file outside
the clone. From the AKLR clone's root:

```bash
RUN=~/aklr-runs/<run> S=.claude/skills/geak-campaign
HIP_VISIBLE_DEVICES=0 python harness/tools/forge_task prepare --schema <name> --run-dir $RUN --kernel-backend triton
# GEAK names the kernel, and the knowledge it records, after the task folder
T=$RUN/<schema name>
mkdir -p $T/scripts && cp $RUN/workspace/*.py $RUN/workspace/program.md $S/config.yaml $T/
cp $S/task_runner.py $T/scripts/
cd $T && for m in compile correctness performance; do
  HIP_VISIBLE_DEVICES=0 python scripts/task_runner.py $m || break
done
```

The last line must print `Compilation: PASS`, `Correctness: PASS` and one `Perf:` line per workload. Read
`$RUN/baseline.json`: a workload whose baseline runs near AKLR's dispatch floor, about 4.1 µs, reads as parity in CI
whatever the kernel does (`forge-loop-campaign`, step 2). Tell the user which workloads those are.

## 2. Run GEAK

Give GEAK one card per AKLR clone. Each correctness check writes the candidate into the clone, and GEAK's GPU lock
keeps only the commands on one card from overlapping.

```bash
G=<GEAK checkout>
cat > $RUN/geak_args.json <<EOF
{"kernel_path": "$T", "workflow_dir": "$G/kernel_workflow", "budget": 6, "gpu_ids": "0", "exp_root": "$RUN/geak_exp",
 "task": "An AKLR kernel: keep every workload correct under AKLR's validate; program.md has the contract and rules."}
EOF
cd $G && set -m && GEAK_CLAUDE_MODEL=claude-opus-5-5 nohup node interface/runtime/engine/run_workflow.mjs \
  kernel_workflow/kernel_workflow.js --args-file $RUN/geak_args.json --result-file $RUN/geak_result.json \
  > $RUN/geak.log 2>&1 &
```

- `budget` is how many optimization directions GEAK may dispatch, over all rounds (`$G/kernel_workflow/README.md`).
- In an interactive Claude Code session the same run is `use <G>/kernel_workflow to optimize <T>, budget 6, gpu_ids
  0`, which the Workflow tool runs.
- Look in every few minutes. `$RUN/geak.log` prints a `=== PHASE:` line per phase and ends with `COMPLETE.` and
  `WORKFLOW_RESULT`. A round of two directions took about 30 minutes on a small GEMM.

## 3. Deliver

```bash
E=$(python -c "import json, sys; print(json.load(open(sys.argv[1]))['eval_dir'])" $RUN/geak_result.json)
cp $(ls $E/optimized/*.py | grep -v '/driver\.py$') $RUN/workspace/
HIP_VISIBLE_DEVICES=0 python $RUN/workspace/driver.py --stress
python harness/tools/forge_task publish $RUN
python -m aklr export check && python -m aklr validate --base origin/main <the path publish printed>
```

- Deliver only when `validation_status` in `geak_result.json` is `accepted` and `--stress` ends with `stress: PASS`.
  The rule and its reason are in `forge-loop-campaign`, "Steps 5 and 6".
- `optimized/` holds every file of GEAK's kernel. A module GEAK adds must be imported relatively from `kernel.py`
  (`from .gemm import f`), or publish leaves it out.
- Report AKLR's verdict per workload, GEAK's `tech_lead_report.md` (each round's directions and the verified
  speedups), and the regime's workloads near the dispatch floor.
- What the run taught AKLR's skills is not distilled for you: forge-distill reads KernelForge's runs only. Take
  findings from `tech_lead_report.md` and `round_N/*/report.md` into the skill as its `CONTRIBUTING.md` says, in the
  pull request that delivers the kernel.

## Pitfalls

- GEAK records no token or cost figures: `WORKFLOW_METRICS` gives the agent count and the wall time only.
- `Live execution tracker not started (non-fatal)` in `geak.log` is the runtime's own and changes nothing.
- GEAK's speedups are its own measurement. What decides delivery is AKLR's verdict, from validate, per workload.
- A run writes into the GEAK checkout (`kernel_workflow/knowledge/learned/`, `kb_artifacts/`). Two campaigns on one
  checkout share what they learn.
