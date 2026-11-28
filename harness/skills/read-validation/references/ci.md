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

- The test jobs' `Run unit tests ... and validation on a GPU node` steps, whose log ends with the validation (`aiter-test`: `aklr/plugins/aiter`; `sglang-test` and `vllm-test`: their plugins, then `aklr/plugins/aiter` untimed): the progress per schema, the outcome counts,
  and the findings, e.g. `not_faster <schema>: graph vs baseline ratio_lower_bound 0.9950 < 1.05`.
  It names the schema, not the Point. `regressed (demoted)` there counts demotions, not findings.
- The run's artifacts `validation-report-<job>-<attempt>` for each of the three jobs, and
  `validation-check-sglang-test-<attempt>` and `validation-check-vllm-test-<attempt>` (`aklr/plugins/aiter` built and checked
  in each image, no timing, no `schemas/`): `report.md` (one row per Point: verdict,
  `graph/baseline`, `graph/incumbent`, `inputs`: drawn, or the recorded calls it was measured on and each one's
  speedup (`traffic/`), then the findings), `report.json`, `schemas/` (the schemas
  with the measured fields) and `baseline/` (what each Point was timed against).
- Nothing else is kept: the job deletes its GPU workload when it ends.
  Use the artifact. A job cancelled or timed out before its tests ended has no reports, only its
  step log.

## `report.json`

- Top level: `version`, `head_sha`, `base_sha`, `measurement_profile`, `platform`, `schema_paths`,
  `bench_paths`, `stages`.
- `stages[]` has `stage_name` (`audit`, `bench`), `status`, `findings[]` (`code`,
  `schema_path`, `axes`, `detail`); the bench stage adds `demotions[]` and `workload_records[]`.
- A workload record: `schema_path`, `axes`, `kernel_content_hash`, `is_timed` (the contract's
  baseline was timed; false under `--no-timing`), `is_compared` (the reference was compared with its outputs), `timings[]`,
  `batch_stopped`.
- A timing: `mode` (`graph`), `versus` (`baseline` or `incumbent`), `threshold` (`entry` or
  `retain`), `n_rounds`, `ratio_lower_bound` (the gate's, clamped), `log_ratio_mean` and `min_ratio`
  (unclamped), and the per-round `candidate_milliseconds` / `versus_milliseconds`.

```python
import json, statistics as st
r = json.load(open("report.json"))
bench = next(s for s in r["stages"] if s["stage_name"] == "bench")
for w in bench["workload_records"]:
    if "<schema name>" in w["schema_path"]:
        for t in w["timings"]:
            c, v = t["candidate_milliseconds"], t["versus_milliseconds"]
            print(w["axes"], t["versus"], round(t["ratio_lower_bound"], 4),
                  round(st.median(v) / st.median(c), 4), max(c) / st.median(c))
```
