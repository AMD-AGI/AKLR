---
name: read-validation
description: Read a AKLR validation result, local or from CI, and decide what it means for a kernel - which Points pass, which fail and why, whether a failure is the kernel or the measurement, and which Points to keep. Use after `python -m aklr validate` or a failed CI `aiter-test`, `sglang-test` or `vllm-test` job.
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
   artifact `validation-report-aiter-test-<attempt>` (`aklr/ops`), `validation-report-sglang-test-<attempt>`
   (sglang's plugins) or `validation-report-vllm-test-<attempt>` (vLLM's); the `aiter-test`, `sglang-test` and
   `vllm-test` jobs' `Validate` step logs end with the findings, and `Enforce the overall CI result` only names the job that failed.
2. **Separate findings from demotions.** A bench finding of one workload (it names the `axes`) that
   the workload computes wrong (`numeric_mismatch`, `post_timing_mismatch`, `nondeterministic`,
   `input_mutation`, `capture_unsafe`, `build_failed`, `candidate_crash`, `candidate_timeout`) or is too
   slow (`not_faster`, `regressed`) is a **demotion**: CI writes it back (`accuracy: failed`, or the low
   `speedup`), dispatch does not serve that Point, and the run does not fail on it. The run fails on
   the other **findings**: the audit's, a schema with no workload both correct and at the entry
   threshold over the baseline, 1.05 by default (`no_qualifying_workload`), a failing baseline, the
   platform. Under `--no-timing` every finding fails the run (`BenchReport.from_findings` in `aklr/core/validation/stages/_bench/_report.py`).
3. **Read an accuracy failure.** The detail names each tensor past its bound: a `tolerance` bound gives
   the `violation_fraction` (elements past it; one fails it) and the `peak_bound_multiple` (the worst
   error over the bound), an `sqnr` bound the SQNR against its minimum. The bound is the contract's
   (`error_bounds` in `aklr/ops/<op>/contract.py`), the expected values the reference's (`reference.py`),
   and the judgment its `check`, which an op may override ("Reference and baseline" in
   `aklr/ops/README.md`). Before blaming the bound, compute the baseline's error on the same inputs: when
   the engine's own code is within the bound and the kernel is not, the kernel is wrong.
4. **Read a slow Point's rounds.** In `report.json`, the bench stage's `workload_records` entry
   with that `schema_path` and `axes` holds `timings[]`: per-round `candidate_milliseconds` and
   `versus_milliseconds`, `ratio_lower_bound`, `log_ratio_mean`, `min_ratio`. Compare the medians, then
   look for rounds far above either side's median.
5. **Classify the slow Point.**
   - **Spikes**: a few rounds of one side well above its median, the medians themselves apart, and
     spikes on either side elsewhere in the run: the node, not the kernel. The median ratio is the
     kernel's.
   - **Floor**: both medians near the graph-node floor (~4 µs): the Point's gain is a fixed latency
     difference that does not carry across nodes (ruler.md).
   - **Real**: the medians are close or inverted.
6. **Decide per Point** (thresholds in ruler.md). Keep a Point whose lower bound is at least the
   dispatch threshold, 1.05, with margin; one under 1.05 can merge when another Point of the file
   qualifies, but is never served. Remove only Points the report shows failing or under 1.05, never others on a guess. Rerun a spike failure
   before removing a Point its medians support.
7. **Change and rerun on the new head.** Removing a workload renames the schema (its uuid hashes the
   workloads); the kernel's sources and content hash stay.

## Boundary

Read-only unless the invoking user asks for the change. Do not request GPU nodes or rerun CI on
their behalf.

## Completion

For each failing Point: the finding, its lower bound, both medians, the spike rounds if any, the
classification, and the change proposed with its evidence.
