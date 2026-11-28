# What CI Validates and Where Its Report Is

Sources: `.github/workflows/ci.yaml` (`aiter-test`, `sglang-test` and `vllm-test` jobs),
`aklr/core/validation/validators/validation_scope.py` (`ValidationScope.of_working_tree`).

## Scope

- GPU jobs run on a PR labelled `ci:gpu`: the Primus-SaFE runner `aklr-safe` submits each as a
  workload on one GPU node (`.github/scripts/safe_gpu_job.py`). An external contributor's run waits
  for a maintainer's approval.
- `python -m aklr validate --base <base>` with no schema paths, as CI runs it, picks:
  - **every committed schema** when the PR changes `aklr/core/` (but a `*.md` there) or
    `stack.json` (`ValidationScope.path_kind`; one such run: 88 schemas, 559 Points, 39 minutes on
    8 GPUs). Then another kernel's workload that fails is demoted, and another schema left with no
    qualifying workload fails the run;
  - the schema files the PR adds otherwise, whatever else it changes (an op's Python, tests, docs):
    a changed kernel is a new file, its uuid hashing the kernel.
- CI validates the merge of the PR head into its base, so main's newest kernels are in it.

## Where the result is

- The test jobs' `Run unit tests ... and validation on a GPU node` steps, whose log ends with the validation (`aiter-test`: `aklr/plugins/aiter`; `sglang-test` and `vllm-test`: their plugins, then `aklr/plugins/aiter` untimed): a line per batch (schema, Points, speedup range, accuracy),
  then the validator's findings, e.g. `no_qualifying_workload <schema>: no workload passed accuracy and cleared Entry (1.05)`,
  and its count of demotions. A finding names the schema; a demotion's Point is in the report.
- The run's artifacts `validation-report-<job>-<attempt>` for each of the three jobs, and
  `validation-check-sglang-test-<attempt>` and `validation-check-vllm-test-<attempt>` (`aklr/plugins/aiter` built and checked
  in each image, no timing, no `schemas/`): `report.md` (one row per Point: verdict,
  what it was timed against, ratio, speedup and both medians; then the demotions, the schemas
  skipped for a `kernel.requires` the image misses, and the findings), `report.json` and `schemas/`
  (the schemas with the measured fields).
- Nothing else is kept: the job deletes its GPU workload and that workload's files when it ends.
  Use the artifact. A job cancelled or timed out before its tests ended has no reports, only its
  step log.

## `report.json`

- Top level: `head_sha`, `base_sha`, `platform`, `schema_paths`, `bench_paths`, `validators`.
- `validators[]`, one entry per validator the command named, in the order written, has `name`
  (`audit`, `bench`), `status`, `findings[]` (`code`, `schema_path`,
  `workload`, `detail`; `workload` is the workload's key, `{"axes": {"M": 4}}`, or null); the bench
  validator adds `demotions[]`, `records[]`, `profile` (the measurement profile: trials, graph budget,
  the clamp), `protocol` (its digest, with the timer's sources: results under another protocol are
  not comparable), `bars` (Entry, Retain), `evaluations` (what is written back, by schema path and
  the workload key's text, `M=4`: the workload with its `speedup`, `accuracy` and `microseconds`)
  and `skipped_schemas` (by schema path, the `kernel.requires` this environment misses).
- Each validator runs whatever the others report: a failed audit does not skip the bench, so one
  run can list both (`report.md`'s first line: `failed (audit), aborted (bench)`). Exit 1 when any
  validator failed, else 3 when any aborted, else 0. `skipped` means the run had no schema.
- A record: `schema_path`, `workload`, `verdict` (its first finding's code, or `passed`),
  `is_demoted`, `versus` (`baseline`, `incumbent` or `nothing`), `versus_name` (the
  baseline class's dotted name or the Incumbent's path), `threshold` (`entry` or
  `retain`; null where nothing is timed), `kernel_hash`, and `comparison` (null when not timed):
  `ratio` (the gate's, clamped), `speedup` (unclamped), and the per-trial `candidate_ms` /
  `opponent_ms`.

```python
import json, statistics as st
r = json.load(open("report.json"))
bench = next(v for v in r["validators"] if v["name"] == "bench")
for w in bench["records"]:
    c = w["comparison"]
    if "<schema name>" in w["schema_path"] and c:
        cm, om = c["candidate_ms"], c["opponent_ms"]
        print(w["workload"]["axes"], w["verdict"], w["versus"], round(c["ratio"], 4),
              round(st.median(om) / st.median(cm), 4), max(cm) / st.median(cm))
```
