---
name: aklr-read-validation
description: Read a AKLR validation result, local or from CI, and decide what it means for a kernel - which Points pass, which fail and why, whether a failure is the kernel or the measurement, and which Points to keep. Use after `python -m aklr validate` or a failed CI `validation` job.
---

# Read a AKLR Validation Result

A verdict is per Point. Before changing a kernel or its workloads, read the trials behind each
failing Point in the report: the log names the failing schema, and only the report names the Point.

## Required Context

Read [references/ruler.md](references/ruler.md) (how a Point is timed and judged),
[references/ci.md](references/ci.md) (what CI validates and where its report is) and
`aklr/ops/README.md`.

## Procedure

1. **Get the report.** Locally, `python -m aklr validate --base origin/main --stages audit,bench
   <schemas> --out <dir>` writes `<dir>/report.md` and `report.json`. From CI, download the run's
   artifact `validation-report-<attempt>`; the `validation` job's `Validate (<stages>)` log ends with
   the findings, and `Enforce the overall CI result` only names the job that failed.
2. **Separate findings from demotions.** The bench stage's `findings` decide it: `failed` on a
   candidate's or schema's fault (`no_admitted_workload`: no Point is correct and clears Entry;
   `baseline_unbound`: no seam adapter binds a Point), `aborted` on the platform's
   (`platform_fault`: a Baseline, Incumbent or Reference that raised, a tracer fault, an unreadable
   base file). `demotions` are Points that failed accuracy (`outputs_mismatch`, `wrong_outputs`,
   `candidate_raised`, `build_failed`, `task_crashed`, `task_timeout`) or performance (`regressed`,
   `not_faster`); they are written back so dispatch skips them, and fail nothing by themselves.
3. **Read the failing Point's trials.** In `report.json`, the bench stage's `records` entry with that
   `schema_path` and `workload` (its key: `{"axes": {...}}`) gives its `verdict`, `is_demoted`, `versus` (`baseline`, `incumbent` or
   `nothing`) and `comparison`: `versus_name`, per-trial `candidate_ms` and
   `opponent_ms`, `ratio` (the gate's) and `speedup` (unclamped, written back). Compare the
   trials, then look for a timing far above its side's others.
4. **Classify the failure.**
   - **Spikes**: one trial of one side well above its others, the other trials' ratios clearing the
     threshold, and spikes on either side elsewhere in the run: the node, not the kernel.
   - **Floor**: both medians near the graph-node floor (~4 µs): the Point's gain is a fixed latency
     difference that does not carry across nodes (ruler.md).
   - **Real**: the medians are close or inverted.
5. **Decide per Point** (thresholds in ruler.md). Keep a Point whose stored speedup is at least the
   dispatch threshold, 1.05, with margin; one under 1.05 can merge when another Point of the file
   qualifies, but is never served. Remove only Points the report shows failing or under 1.05, never
   others on a guess. Rerun a spike failure
   before removing a Point its medians support.
6. **Change and rerun on the new head.** Removing a workload renames the schema (its uuid hashes the
   workloads); the kernel's sources and content hash stay.

## Boundary

Read-only unless the invoking user asks for the change. Do not request GPU nodes or rerun CI on
their behalf.

## Completion

For each failing Point: the finding, its ratio, both medians, the spike trials if any, the
classification, and the change proposed with its evidence.
