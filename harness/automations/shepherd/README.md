# shepherd

shepherd gives you one automation that optimizes AKLR kernels unattended. Each round observes GitHub
and the GPU jobs, advances each KernelForge campaign as far as confirmed results allow, admits the schemas
its strategies pick (kernel-less ones, then the kernels with the most time over the roofline) into free GPUs, validates the kernel, opens a draft PR and follows it until a maintainer
merges or closes it.

The files in this directory are dormant setup and automation sources. They do not appear as slash skills.
The round reuses the optimization, GPU, validation, PR and reporting skills in `harness/skills/`.

## Model call tables

`models/<model>.json` says which Points a model's serving calls and how often: `calls` maps a schema's `name`
to its workloads' axes (`m=1,n=32`, as the schema writes them) and each to the calls made at it per the unit
`per` names: `forward pass` in every table, one decode step or one prefill chunk that reaches the Point, so a
GEMM every one of 61 layers calls counts 61. `source` names the run the counts are from. `policy.target_models` keeps only the schemas a target model calls, and the
`roofline` strategy weighs each workload's time over the roofline by these calls. A `1` is a seed, not a
measurement. A PR that adds a schema or a Point adds it to the table of each model whose serving calls it;
`harness/tests/shepherd/test_model_calls.py` checks that every committed schema is in some table.

To measure a table, serve the model with AKLR on and `AKLR_EXPORT=1` under the load to weigh by, then at the
repository root:

```bash
python -m aklr export --from <calls dir> --to <scratch dir> --language flydsl |
  python harness/automations/shepherd/skills/optimize-and-deliver-kernels/scripts/model_calls.py \
    record --model <model> --source "<engine, layout, load, hardware, date>" \
    --per "forward pass" --units <forward passes the run made>
```

It divides each count by `--units`, so record a run whose forward passes all reach the same Points (one batch
size, a fixed number of steps, warmup kept out). `per` is in the file because a count means nothing without
it; a run counted per another unit is refused, and the test refuses tables whose `per` differ, since counts
per different units do not add up over the target models.

It writes the calls of every Point the run reached, on any GPU: a call is counted whether or not a kernel
served it. A Point the run did not reach keeps its value.

The roofline a workload is measured against is the shepherd's own
(`skills/optimize-and-deliver-kernels/scripts/roofline.py`): the least time one call's matrix math and bytes
take on MI355X, from the schema file alone. It has formulas for the GEMM contracts and `fused_moe`, and
counts bytes alone for the memory-bound ones; a schema of any other contract is skipped by `roofline`.

## Set it up

1. Point an agent at [`setup-shepherd`](./skills/setup-shepherd/SKILL.md) in the AKLR checkout, on the
   base branch, that will run rounds. It already holds this pack; nothing is copied.
2. The Cursor automation runs each round itself, so its run shows every step. It reads the shared
   skills by path from that checkout, not from an installed plugin whose cached revision can lag.
3. Keep user-owned configuration outside the pack, in `.shepherd/configuration.yaml` (ignored by
   Git). Adapt [`configuration.example.yaml`](./templates/configuration.example.yaml).
4. Create the automation yourself, or update the existing one, with the fields setup hands over and a
   prompt adapted from [`automation-prompt.example.md`](./templates/automation-prompt.example.md) (it
   begins the round itself, so a skipped trigger costs two commands), and enable it. `mode: run` works
   from the first round. For a cautious start, enable it with `mode: observe` first, or run on probation
   (one campaign's budget, `max_parallel_campaigns: 1`, one PR) until a real campaign reaches a draft PR.
