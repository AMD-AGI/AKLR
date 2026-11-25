# What CI Validates and Where Its Report Is

Sources: `.github/workflows/ci.yaml` (`aiter-test`, `sglang-test` and `vllm-test` jobs),
`aklr/core/validation/stages/validation_scope.py` (`ValidationScope.of_working_tree`).

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

- The test jobs' `Run unit tests ... and validation on a GPU node` steps, whose log ends with the validation
  (`aiter-test`: `aklr/plugins/aiter` timed; `sglang-test` and `vllm-test`: `aklr/plugins/aiter` built and checked in their images,
  untimed): a line per batch (schema, Points, speedup range, accuracy), then the stage's findings and its count
  of demotions. A finding names the schema; a demotion's Point is in the report.
- The run's artifact `validation-report-aiter-test-<attempt>`: `report.md` (one row per Point: verdict,
  what it was timed against, ratio, speedup and both medians; then the demotions, the findings and the
  schemas skipped for a `kernel.build.requires` the image misses), `report.json` and `schemas/` (the
  schemas with the measured fields). `validation-check-sglang-test-<attempt>` and
  `validation-check-vllm-test-<attempt>` hold the untimed checks' reports, with no `schemas/`.
- Nothing else is kept: the job deletes its GPU workload and that workload's files when it ends.
  Use the artifact. A job cancelled or timed out before its tests ended has no reports, only its
  step log.

## `report.json`

- Top level: `version` (3: records and findings name a `workload`, version 2's `axes`), `head_sha`, `base_sha`, `platform`, `schema_paths`, `bench_paths`, `stages`.
- `stages[]` has `stage_name` (`audit`, `bench`), `status`, `findings[]` (`code`, `schema_path`,
  `workload`, `detail`; `workload` is the workload's key, `{"axes": {"M": 4}}`, or null); the bench stage adds `demotions[]`, `records[]`, `profile` (the measurement
  profile: trials, graph budget, Entry, Retain),
  `protocol` (its digest, with the timer's sources: results under another protocol are not
  comparable) and `evaluations` (what is written back, by schema path and the workload key's text,
  `M=4`).
- A record: `schema_path`, `workload`, `verdict` (its first finding's code, or `passed`), `is_demoted`,
  `versus` (`baseline`, `incumbent` or `nothing`), `versus_name` (the Baseline's name or the Incumbent's
  path), `kernel_hash`, and `comparison` (none when not timed): `ratio` (the gate's,
  clamped), `speedup` (unclamped), and the per-trial `candidate_ms` /
  `opponent_ms`.

```python
import json, statistics as st
r = json.load(open("report.json"))
bench = next(s for s in r["stages"] if s["stage_name"] == "bench")
for w in bench["records"]:
    c = w["comparison"]
    if "<schema name>" in w["schema_path"] and c:
        cm, om = c["candidate_ms"], c["opponent_ms"]
        print(w["workload"]["axes"], w["verdict"], w["versus"], round(c["ratio"], 4),
              round(st.median(om) / st.median(cm), 4), max(cm) / st.median(cm))
```
