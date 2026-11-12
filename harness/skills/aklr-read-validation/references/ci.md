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

- The `validation` job's `Validate (<stages>)` step: a line per batch (schema, Points, speedup
  range, accuracy), then the stage's findings and its count of demotions. A finding names the schema;
  a demotion's Point is in the report.
- The run's artifact `validation-report-<attempt>`: `report.md` (one row per Point: verdict, what it
  was timed against, ratio, speedup and both medians; then the demotions and findings),
  `report.json` and `schemas/` (the schemas with the measured fields).
- The runners' work directories live on the CI's nodes under the CI's account; other accounts cannot
  read them. Use the artifact.

## `report.json`

- Top level: `version` (3: records and findings name a `workload`, version 2's `axes`), `head_sha`, `base_sha`, `platform`, `schema_paths`, `bench_paths`, `stages`.
- `stages[]` has `stage_name` (`audit`, `bench`), `status`, `findings[]` (`code`, `schema_path`,
  `workload`, `detail`; `workload` is the workload's key, `{"axes": {"M": 4}}`, or null); the bench stage adds `demotions[]`, `records[]`, `profile` (the measurement
  profile: trials, graph budget, Entry, Retain),
  `protocol` (its digest, with the timer's sources: results under another protocol are not
  comparable) and `evaluations` (what is written back, by schema path and the workload key's text,
  `M=4`).
- A record: `schema_path`, `workload`, `verdict` (its first finding's code, or `passed`), `is_demoted`,
  `versus` (`baseline`, `incumbent` or `nothing`), `kernel_hash`, and `comparison` (none when not
  timed): `versus_name` (the Baseline's backend or the Incumbent's path), `ratio` (the gate's,
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
