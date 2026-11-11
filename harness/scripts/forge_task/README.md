# forge_task

Let KernelForge's agents write and tune the kernel of one AKLR schema, then put the kernel they end with back
into AKLR.

A schema without a kernel is a problem waiting for one: `aklr/ops/<op>/impls/<language>/<name>/*.json` with an
empty `kernel.sources`. `forge_task` turns it into a task for KernelForge's optimization loop and writes the
result back. KernelForge does the optimizing, with its own commands; `forge_task` adds only what is AKLR's:

- a **driver** that checks each candidate with `python -m aklr validate` and times it with AKLR's own timer;
- a **brief** for the agents: what the operator computes, its workloads, and where AKLR's rules are;
- AKLR's **baseline** as the time to beat, workload by workload.

```
schema without a kernel ── prepare ──▶ KernelForge task ── forge-loop ──▶ the kernel the loop kept
the kernel the loop kept ── forge-ablate, forge-finalize ──▶ one reviewed commit
one reviewed commit ── publish ──▶ aklr/ops ── aklr validate ──▶ your pull request
```

## Before you start

- An MI355X (gfx950) machine, inside the ROCm container you run AKLR in.
- A clone of AKLR that is yours alone: while a candidate is checked, the driver writes it into `aklr/ops/`.
- AKLR installed in that container from your clone, and the baseline that its validate run times against, as AKLR's
  README says (`pip install -e .`, then the `AKLR_BASELINE_AITER` install line).
- In that container:

  ```bash
  pip install "flydsl==0.3.2" "ruff==0.15.12"   # the ruff that .pre-commit-config.yaml pins
  pip install -e "<your Hyperloom checkout>[forge]"    # KernelForge, from Hyperloom's AKLR branch
  ```

- The Claude Code CLI, logged in: `claude -p "reply ok"` answers. KernelForge's agents run through it.
- `git config user.name` and `user.email` set in the AKLR clone: they author the delivered commit.

## Run a campaign

Run every command from the AKLR clone's root. The example optimizes `bmm_n128_k512`; `ls -d
aklr/ops/*/impls/*/*/` lists the schema names.

**1. Prepare the task.** A few minutes on one GPU; no agent runs yet.

```bash
RUN=~/aklr-runs/bmm_n128_k512
HIP_VISIBLE_DEVICES=0 python harness/scripts/forge_task prepare --schema bmm_n128_k512 --run-dir $RUN
```

It checks the starting kernel with AKLR, records AKLR's baseline times and writes the task into `$RUN`. A GEMM
schema starts from torch, or from its contract's AKLR baseline when torch cannot compute it (a scaled or quantized
GEMM); any other needs `--starting-kernel <kernel.py>`, or `--starting-kernel <dir>` for a kernel
of several files: every `*.py` in the directory, `kernel.py` among them.

**2. Optimize.** Hours, and the agents cost money from here: in the Kimi-K3 runs, about $10 an hour on a GEMM
and $17 on a MoE. Run it inside `tmux`:

```bash
$RUN/forge_loop.sh --max-hours 12 --model claude-opus-5-5 --gpu 0 \
  --lanes 1 --no-experience-kb --no-kb-warmstart --no-pr-kb
```

The options go to KernelForge's `forge-loop`: one agent at a time (`--lanes 1`), and no reading or writing of
KernelForge's shared knowledge stores (the three `--no-` options). A round takes 20–70 minutes; the loop stops
early once the directions it knows are mined out.

**3. Watch and steer.** `$RUN/workspace/forge_experiments/dashboard.html` shows the score, every round, and each
workload against AKLR's serving bar (1.03× the baseline). To leave the agents a message, which they read from
the next round on:

```bash
python -m kernelforge.cli forge-say --workspace $RUN/workspace "focus on m<=64"
```

**4. Stop or resume.** `touch $RUN/workspace/.stop` stops after the current round; Ctrl-C stops now. To go on
later, `rm -f $RUN/workspace/.stop` and run `$RUN/forge_loop_resume.sh` with the same options and
`--max-hours <hours left>`.

**5. Review the result**, once the loop has ended. One to two hours:

```bash
A=$(pwd) W=$RUN/workspace
# What each kept round is worth: pays, within noise, or faster without it
python -m kernelforge.cli forge-ablate --workspace $W --gpus 0
# An agent reviews the whole change against AKLR's checks and squashes it into one commit
# The review's guides: FlyDSL's own code-cleanup skill (linked, not kept in this repository) and what this
# repository corrects in it. They go in a folder of their own: the review agent may edit the guides' folder.
mkdir -p $RUN/review_guide && cp $A/harness/skills/aklr-kb/flydsl/guides.md $RUN/review_guide/
curl -sSfL https://raw.githubusercontent.com/ROCm/FlyDSL/v0.3.2/.claude/skills/kernel-code-cleanup/SKILL.md \
  -o $RUN/review_guide/flydsl_code_cleanup.md
python -m kernelforge.cli forge-finalize --workspace $W --gpu 0 --model claude-opus-5-5 \
  --author "$(git config user.name) <$(git config user.email)>" \
  --lint "python -m ruff check --no-fix --config $A/pyproject.toml" \
  --lint "python -m ruff format --check --config $A/pyproject.toml" \
  --review-guide $RUN/review_guide/flydsl_code_cleanup.md --review-guide $RUN/review_guide/guides.md
# The same answer, as the schema's determinism asks, on a stream limited to a few CUs and beside a second call of
# its own: a kernel whose workgroups wait for each other does not give it, and validate cannot tell
HIP_VISIBLE_DEVICES=0 python $W/driver.py --stress
# What aklr-kb (its flydsl/ folder included) should learn from the run, as a patch for you to review
python -m kernelforge.cli forge-distill --workspace $W --model claude-opus-5-5 \
  --skill-dir $A/harness/skills/aklr-kb --out $RUN/knowledge_patch \
  --reference $A --guide CONTRIBUTING.md \
  --lint "python maintenance/kb_lint.py --errors-only --repo-root {repo}"
```

Their reports are `$W/forge_experiments/ablation.md` and `finalize.md`, and `$RUN/knowledge_patch/distill.md`.
For a Triton kernel, leave out `--review-guide` and skip `forge-distill`.

**6. Deliver.**

```bash
python harness/scripts/forge_task publish $RUN
python -m aklr export check && python -m aklr validate --base origin/main <the path publish printed>
```

`publish` writes the reviewed kernel into `aklr/ops/` as the schema's kernel. Give validate that path: with none, it
checks only the schema files a change touches, and only when it touches nothing else; on a branch that also changes
code, it checks every schema in AKLR, which takes hours. Commit the kernel as `aklr/ops/README.md` describes and
open a pull request. What the run taught goes into the same pull request: apply
`$RUN/knowledge_patch/knowledge_patch.diff`, then review it item by item as aklr-kb's `CONTRIBUTING.md` says --
edits to existing pages in place, new pages in their folder, reviewed in the kernel's pull request. A change to the method itself
(this README, the driver, the `forge-loop-campaign` skill) goes into a pull request of its own.

An AI agent can run the whole campaign: the `forge-loop-campaign` skill in `harness/skills/` guides it.
To have GEAK (AMD-AGI/GEAK) optimize the task instead of KernelForge, follow the `geak-campaign` skill.

## What `prepare` writes

```
$RUN/
├── forge_loop.sh          starts KernelForge's loop on this task; options given to it go to forge-loop
├── forge_loop_resume.sh   resumes it after a stop or a crash
├── schema.json            the schema, as read
├── baseline.json          AKLR's baseline time per workload, the time to beat
├── aklr_validate/         AKLR's report on each kernel the driver checked
└── workspace/             KernelForge's git workspace: kernel.py, driver.py, program.md (the brief)
```

Options: `--schema-file <file>` instead of `--schema <name>`; `--starting-kernel <kernel.py|dir>`; `--kernel-backend
flydsl|triton|gluon` (default `flydsl`); `--anchor start` to score against the starting kernel instead of AKLR's
baseline; `--run-dir` (default `./aklr-runs/<schema>-<time>`).

## How a candidate is judged

- **Correct** when `python -m aklr validate --stages audit,bench` finds nothing wrong on any workload: a wrong
  output on one workload rejects the candidate.
- **Fast** by AKLR's graph timer on AKLR's timing inputs. A candidate is kept only when it beats the kernel kept so
  far by more than the measured noise.
- The score is the baseline's time over the candidate's, per workload, averaged over the workloads.
- The agents cannot edit `driver.py`.

## When something goes wrong

| Message | What to do |
|---|---|
| `AKLR does not pass the starting kernel on every workload` | fix the starting kernel on the workloads listed |
| `AKLR timed no baseline for …` | the schema has nothing to time against: prepare again with `--anchor start` |
| `run directory … is not empty` | give each task a directory of its own |
| `Kineto recorded no kernels for a single call` | the profiler lost a trace, usually on a busy machine: try again, or on an idle one |
| `stress: differs from the call alone: …` | the kernel's workgroups wait for each other (a split-K fold that polls flags), or its concurrent calls share scratch memory: make it wait-free, the last workgroup to arrive on an atomic counter doing the fold, before you publish. Counters that must start at zero are zeroed in the call: scratch from `torch.empty` holds whatever was there |
| `stress: hung, no result within 60 s: …` | the same defect, a wait with no bound: the kernel never finished on that stream. The check stops there, since the device is still running it |

## Tests

`python -m pytest harness/tests/forge_task` runs on a CPU against an AKLR stand-in. Two tests also need
KernelForge installed (they read the driver's output with KernelForge's own parsers), and two need torch (they
build the timed calls and read a report with this checkout's AKLR); each is skipped without it.
