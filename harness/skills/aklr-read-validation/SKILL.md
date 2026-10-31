---
name: aklr-read-validation
description: Read a AKLR validation result, local or from CI, and decide what it means for a kernel - which Points pass, which fail and why, whether a failure is the kernel or the measurement, and which Points to keep. Use after `python -m aklr validate` or a failed CI `validation` job.
---

# Read a AKLR Validation Result

A verdict is per Point. Before changing a kernel or its workloads, read the rounds behind each
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
2. **Separate findings from demotions.** A run fails on its `findings` (`not_faster`, `regressed`,
   accuracy, build, crash). `demotions` are committed kernels' Points that measured too slow and stop
   being served; they fail nothing. A committed Point that newly computes wrong is a demotion and a
   finding at once.
3. **Read the failing Point's rounds.** In `report.json`, the bench stage's `workload_records` entry
   with that `schema_path` and `axes` holds `timings[]`: per-round `candidate_milliseconds` and
   `versus_milliseconds`, `ratio_lower_bound`, `log_ratio_mean`, `min_ratio`. Compare the medians, then
   look for rounds far above either side's median.
4. **Classify the failure.**
   - **Spikes**: a few rounds of one side well above its median, the medians themselves apart, and
     spikes on either side elsewhere in the run: the node, not the kernel. The median ratio is the
     kernel's.
   - **Floor**: both medians near the graph-node floor (~4 µs): the Point's gain is a fixed latency
     difference that does not carry across nodes (ruler.md).
   - **Real**: the medians are close or inverted.
5. **Decide per Point** (thresholds in ruler.md). Keep a Point whose lower bound is at least the
   dispatch threshold, 1.03, with margin; one between 1.02 and 1.03 merges but is never served. Remove
   only Points the report shows failing or under 1.03, never others on a guess. Rerun a spike failure
   before removing a Point its medians support.
6. **Change and rerun on the new head.** Removing a workload renames the schema (its uuid hashes the
   workloads); the kernel's sources and content hash stay.

## Boundary

Read-only unless the invoking user asks for the change. Do not request GPU nodes or rerun CI on
their behalf.

## Completion

For each failing Point: the finding, its lower bound, both medians, the spike rounds if any, the
classification, and the change proposed with its evidence.
