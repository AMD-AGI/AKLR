# What CI Validates and Where Its Report Is

Sources: `.github/workflows/ci.yaml` (`validation` job),
`aklr/core/validation/stages/validation_scope.py` (`ValidationScope.of_working_tree`).

## Scope

- GPU jobs run on a PR labelled `ci:gpu`; on a fork PR, or when its author or sender lacks write
  access, they wait for a maintainer's approval (`gpu-node.yaml`).
- `python -m aklr validate --base <base>` with no schema paths, as CI runs it, picks:
  - the changed schemas, when every changed path is a schema file or a document (`docs/`, `*.md`);
  - **every committed schema** otherwise. A PR that also touches an op's Python, tests or any
    other file benches all of them (one such run: 88 schemas, 559 Points, 39 minutes on 8 GPUs)
    and fails on any finding, another kernel's included.
- CI validates the merge of the PR head into its base, so main's newest kernels are in it.

## Where the result is

- The `validation` job's `Validate (<stages>)` step: the progress per schema, the outcome counts,
  and the findings, e.g. `not_faster <schema>: graph vs baseline ratio_lower_bound 0.9950 < 1.05`.
  It names the schema, not the Point. `regressed (demoted)` there counts demotions, not findings.
- The run's artifact `validation-report-<attempt>`: `report.md` (one row per Point: verdict,
  `graph/baseline`, `graph/incumbent`, then the findings), `report.json`, `schemas/` (the schemas
  with the measured fields) and `baseline/` (what each Point was timed against).
- The runners' work directories live on the CI's nodes under the CI's account; other accounts cannot
  read them. Use the artifact.

## `report.json`

- Top level: `version`, `head_sha`, `base_sha`, `measurement_profile`, `platform`, `schema_paths`,
  `bench_paths`, `stages`.
- `stages[]` has `stage_name` (`audit`, `bench`), `status`, `findings[]` (`code`,
  `schema_path`, `axes`, `detail`); the bench stage adds `demotions[]` and `workload_records[]`.
- A workload record: `schema_path`, `axes`, `kernel_content_hash`, `baseline` (what it was timed
  against: a `<contract>_baseline` function or a committed schema), `is_timed`, `is_kept`,
  `worst_violation_fraction`, `timings[]`.
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
