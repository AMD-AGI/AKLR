# Optimizing a kernel with forge_task

`forge_task` takes a schema in `aklr/ops/` that has no kernel yet and hands it to KernelForge's `forge-loop`.
KernelForge's agents write and tune the kernel round by round, and `forge_task` writes the kernel they end with
back into AKLR. AKLR keeps the measurement: every candidate is checked with `python -m aklr validate` and timed with
AKLR's own timer, so a kernel that wins in the loop wins in CI too.

```
schema without a kernel ── prepare ──▶ task ── forge-loop ──▶ kept rounds
kept rounds ── forge-ablate, forge-finalize ──▶ one reviewed commit
one reviewed commit ── --stress, publish ──▶ aklr/ops ── aklr validate ──▶ pull request
```

This page is the walk-through. `harness/scripts/forge_task/README.md` has every option and the error messages. The
`forge-loop-campaign` skill in `harness/skills/` lets an AI agent run the whole campaign.

## Set up

Inside the ROCm container you run AKLR in:

1. A clone of AKLR that is yours alone, installed from that clone. The driver writes each candidate into its
   `aklr/ops/` while checking it.

   ```bash
   git clone --recursive https://github.com/AMD-AGI/AKLR.git && cd AKLR && pip install -e .
   ```

2. FlyDSL, ruff and KernelForge (from Hyperloom's AKLR branch):

   ```bash
   pip install "flydsl==0.3.2" "ruff==0.15.12"
   pip install -e "<your Hyperloom checkout>[forge]"
   python -m kernelforge.cli forge-loop --help
   ```

3. The Claude Code CLI, logged in (`claude -p "reply ok"` answers), and `git config user.name` / `user.email` set
   in the clone: they author the delivered commit.

## Choose what one run optimizes

- **One run per regime.** A run scores the mean over its workloads, so small-M Points (decode: latency or the
  weight read) and large-M Points (prefill: compute) in one run get a kernel tuned for whichever outnumber the
  others. A GEMM splits near the M where reading the weights and computing take the same time. Write each regime's
  workloads into a schema file of its own, outside the clone, at a path shaped like AKLR's, e.g.
  `~/aklr-regimes/gemm/impls/flydsl/<name>/small_m.json`, and give it to `prepare` with `--schema-file`.
- **One idle GPU per run** (`rocm-smi --showpids`): another process on the card makes every timing meaningless.
- **Mind the dispatch floor.** AKLR's timer reads every dispatch at about 4.1 µs or more. A workload whose baseline
  runs near that reads as parity in CI, whatever the kernel does there.

## Run it

Every command runs from the clone's root. The example optimizes `bmm_n128_k512`.

**1. Prepare** (minutes, no agent yet). It checks the starting kernel with AKLR, records AKLR's baseline time per
workload in `$RUN/baseline.json`, and writes the task. A GEMM starts from torch, or from its contract's AKLR baseline when it is scaled or quantized; any other op needs
`--starting-kernel <kernel.py or a directory of *.py>`.

```bash
RUN=~/aklr-runs/bmm_n128_k512
HIP_VISIBLE_DEVICES=0 python harness/scripts/forge_task prepare --schema bmm_n128_k512 --run-dir $RUN
```

**2. Optimize** (hours; the agents cost money from here, about $10 an hour on a GEMM). Run it in `tmux`:

```bash
$RUN/forge_loop.sh --max-hours 12 --model claude-opus-5-5 --gpu 0 \
  --lanes 1 --no-experience-kb --no-kb-warmstart --no-pr-kb
```

A round takes 20–70 minutes. The loop keeps a round only when it beats the kernel kept so far by more than the
measured noise, and stops early once its directions are mined out.

**3. Watch and steer.** `$RUN/workspace/forge_experiments/dashboard.html` shows the score, every round, and each
workload against AKLR's serving bar (1.03x the baseline). A message to the agents, read from the next round on:

```bash
python -m kernelforge.cli forge-say --workspace $RUN/workspace "focus on m<=64"
```

**4. Stop or resume.** `touch $RUN/workspace/.stop` stops after the current round. To go on later,
`rm -f $RUN/workspace/.stop` and run `$RUN/forge_loop_resume.sh` with the same options and `--max-hours <left>`.

**5. Review**, once the loop has ended (one to two hours):

```bash
A=$(pwd) W=$RUN/workspace
python -m kernelforge.cli forge-ablate --workspace $W --gpus 0      # what each kept round is worth
mkdir -p $RUN/review_guide && cp $A/harness/skills/aklr-kb/flydsl/guides.md $RUN/review_guide/   # the review's guides,
curl -sSfL https://raw.githubusercontent.com/ROCm/FlyDSL/v0.3.2/.claude/skills/kernel-code-cleanup/SKILL.md \
  -o $RUN/review_guide/flydsl_code_cleanup.md   # in a folder of their own: the review may edit it
python -m kernelforge.cli forge-finalize --workspace $W --gpu 0 --model claude-opus-5-5 \
  --author "$(git config user.name) <$(git config user.email)>" \
  --lint "python -m ruff check --no-fix --config $A/pyproject.toml" \
  --lint "python -m ruff format --check --config $A/pyproject.toml" \
  --review-guide $RUN/review_guide/flydsl_code_cleanup.md \
  --review-guide $RUN/review_guide/guides.md   # review, squash into one commit
HIP_VISIBLE_DEVICES=0 python $W/driver.py --stress                 # must end with "stress: PASS"
python -m kernelforge.cli forge-distill --workspace $W --model claude-opus-5-5 \
  --skill-dir $A/harness/skills/aklr-kb --out $RUN/knowledge_patch \
  --reference $A --guide CONTRIBUTING.md \
  --lint "python maintenance/kb_lint.py --errors-only --repo-root {repo}"   # what aklr-kb should learn
```

`--stress` runs every workload on streams limited to a few CUs and beside a second call of its own. AKLR's validate
gives a kernel the GPU alone, so it passes a kernel whose workgroups wait for each other; on a busy GPU that kernel
returns NaN or hangs. Do not deliver a kernel that fails it. For a Triton kernel, leave out `--review-guide` and skip
`forge-distill`.

**6. Deliver.**

```bash
python harness/scripts/forge_task publish $RUN
python -m aklr export check && python -m aklr validate --base origin/main <the path publish printed>
```

Commit the published file as `aklr/ops/README.md` describes and open a pull request. What the run taught the skill
goes into the same pull request: apply `$RUN/knowledge_patch/knowledge_patch.diff` and review it item by item, as
`harness/skills/aklr-kb/CONTRIBUTING.md` says.

## What you get

| File | What it holds |
|---|---|
| `$RUN/baseline.json` | AKLR's baseline time per workload, the time to beat |
| `$RUN/workspace/kernel.py` | the kernel; `driver.py` beside it is the measurement and the agents cannot edit it |
| `$RUN/workspace/forge_experiments/` | the dashboard, each round's lesson, `ablation.md`, `finalize.md` |
| `$RUN/knowledge_patch/` | the proposed skill patch and `distill.md`, its evidence |
| `aklr/ops/<op>/impls/<language>/<name>/<id>.json` | the delivered kernel, written by `publish` |
