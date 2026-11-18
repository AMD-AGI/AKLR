---
name: op-census
description: Record every GPU kernel an InferenceX AgentX point really runs on a target (MI355X by default), for each SGLang and vLLM submission in the InferenceX cookbook - by phase (prefill, decode, mixed), operator class and kernel, with each one's share of the phase's GPU time. Use before choosing which operators to optimize for a model, or to check that a kernel is on a deployed path at all.
---

# Operator census of InferenceX AgentX points

A census answers, for one deployed point: which kernels ran, in which phase, how often per step and how much of
the GPU time they took. It comes from a real run: the cookbook's server command in the cookbook's image, the
InferenceX AgentX client replaying its traces at the point's concurrency, and torch-profiler windows captured in
the client's profiling phase. Nothing is estimated or replayed in isolation, and no kernel is left out: a kernel no
classification rule knows is reported as `other` under its full name.

The code is [`harness/scripts/op_census`](../../scripts/op_census/); `python3 -m op_census -h` lists its commands.

## Required context

- The InferenceX cookbook: `inferencex-e2e/configs/amd-master.yaml` (one entry per submission, with its image, model,
  recipe file and the AgentX concurrencies) and the recipe files under `benchmarks/single_node/srt-slurm-recipes/`.
- [references/method.md](references/method.md): how a point is bound, run and captured, how a kernel is tied to a
  step and a step to a phase, and the checks.
- [references/pitfalls.md](references/pitfalls.md): what has broken a census before.

## Procedure

1. **Plan.** `python3 -m op_census plan --ref <InferenceX commit> --out plan.json` binds every single-node SGLang
   and vLLM AgentX point of the target with InferenceX's own code (matrix row, recipe variant, synthetic acceptance).
   Pin the commit: the cookbook changes daily, and the plan records which commit it is. `--keys` limits it to some
   configs. `python3 -m op_census show --plan plan.json --points <regex>` prints a point's exact server command.
2. **Prepare the host** (once): pull each image the plan names as an Apptainer sandbox (a reference with both a
   tag and a digest must be written `repo@sha256:...`); locate each repository the plan lists
   (`points[].repositories`: checkpoint, tokenizer, draft model) and check that every shard of its index is
   present; bootstrap the AgentX client runtime and download its trace corpora. Write `host.json`:
   `images` (cookbook reference -> sandbox), `paths` (repository -> directory), `infx`, `agentx_runtime`, `hf_home`.
3. **Probe each image** once on one GPU: `python3 -m op_census probe --out <dir>` inside the image. It must report
   `ok: true` (all 50 kernels of a replayed CUDA graph recorded). An image that fails cannot give decode numbers
   with the torch profiler; do not use it until another capture is in place.
4. **Run.** `python3 -m op_census run --plan plan.json --host host.json --out <dir> --points <regex>` serves each
   point, checks it with one request, starts the AgentX client, waits for its profiling phase plus `--settle-s`,
   and captures `--windows` windows of `--window-s` seconds. Points needing fewer GPUs than the host has run side by
   side. Run each point twice (`--repeat 2`) when its numbers will be used: the two censuses must agree (below).
5. **Parse and report.** `python3 -m op_census parse <run>/<point>` writes `census.json`;
   `python3 -m op_census report <census.json>...` renders it.
6. **Check before trusting a census** (the report's first section):
   - activity outside every step and without a launch record near 0 (a few tenths of a percent at most);
   - no decode step without GPU work (a graph whose kernels were not recorded);
   - rank spread small for a TP point (a large spread in decode is a collective waiting for a slow rank);
   - the two runs of a point agree: `python3 -m op_census compare <run 1>/census.json <run 2>/census.json` (exit 1
     if not) checks each operator class's share of each phase within 3 points and the same kernels above 0.5% of a
     phase (a kernel near 0.5% in both runs is only listed). Run the second runs into another `--out` directory with
     the same point ids. An agentic client's load varies, so two runs sample different mixes of decode batch sizes
     and prefill lengths; a phase that disagrees as a whole is compared again within each load bucket both runs
     sampled enough. Exit 2 (`load_mix`): the buckets agree, only the mix differs - the census is right for the load
     it saw. Exit 3 (`few_steps`): too few steps to tell (low-concurrency prefill: 10-20 steps in 8 windows) - pool
     the runs or capture longer before quoting that phase's mix. Exit 1: a bucket disagrees, look for a capture fault;
   - per-step call counts consistent with the model (e.g. one attention call per attention layer per step).
   A census failing one of these is not a result; record why and rerun.

## Output

Per point, `census.json`: `point` (config key, recipe variant, image, model, concurrency, synthetic acceptance
length), `windows` (per rank: wall time, GPU busy time, steps by phase), `phases` (steps, stages, batch sizes,
tokens, time per rank and per step, share), `ops` and `ops_by_stage` (operator class per phase and stage), `kernels`
(every kernel name with calls and time per step, p10/p50/p90 duration and share of its phase), `ops_by_load`
(operator class per decode batch size and per prefill token range, with each bucket's share of its phase), `steps`
(rank 0's steps in order) and `checks`.

Shares are of summed GPU activity time per rank (kernels on concurrent streams both count), not of wall time.
