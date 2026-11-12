# Bench stage requirements

What the validation bench must do, as the contract for rewriting
`aklr/core/validation/stages/bench/`. This document states behaviour, not structure: the files,
classes and functions of the rewrite belong to its design document, which is checked against these
requirements one by one.

Terms (Candidate, Workload, Reference, Baseline, Incumbent, Entry and Retain thresholds,
Admission, Demotion) are defined in [GLOSSARY.md](../GLOSSARY.md).

## 1. Scope

In scope:

- multi-rank ops: a group schema (an op whose `ExecutionPolicy` gives more than one rank; for
  mega_moe the Const axis `ep_size`) is checked and timed on all its ranks at once, one process and one device each,
  on one machine; the schema format is unchanged

- `aklr/core/validation/stages/bench/`, except `_execution/`
- `aklr/core/validation/stages/_measure.py` (only the bench uses it)
- the bench's finding codes in `aklr/core/validation/stages/report/_finding.py`
- `aklr/core/validation/command.py`, for the bench's options
- every reader of the bench's part of `report.json` (R8.2)

Out of scope:

- `bench/_execution/` beyond its runners (R7.2, R7.3). The bench uses its public API only:
  `Executor.for_devices(devices, runner=..., initializer=..., startup_timeout_secs=...)`,
  `Executor.submit(fn, timeout_secs=..., resource_request=..., runner=..., **kwargs)`,
  `ExecutionFuture.result()`.
- `aklr/core/runtime/` and `aklr/core/schema/` (protected, see AGENTS.md)
- the audit stage and its codes (`schema_invalid`, `source_rejected`, `perf_edited`)
- `ValidationScope`'s choice of which schemas to bench, and `pipeline.py` beyond the write-back
  contract (R6)

## 2. Inputs and outputs

The stage reads the scope's bench paths, the base commit's schemas, the platform the pipeline
recorded, and the seam adapters. It produces a stage report that:

- decides the stage: passed, failed (a Candidate's or schema's fault) or aborted (only platform
  faults);
- carries one record per benched workload (R8.1);
- yields each schema's Evaluations for the pipeline's write-back (R6).

## 3. Planning

- **R3.1** The bench measures each bench path whose schema has a kernel; a kernel-less schema is
  skipped.
- **R3.2** A workload's Incumbent is chosen among the base commit's schemas under the op's `impls`
  directory with the same schema name that serve the workload's dispatch key (the runtime's, today
  its axes), that either qualify on this target (`Workload.qualifies` at `AKLR_MIN_SPEEDUP` when the
  op has a Baseline, else at 0) or hold the Candidate's own kernel. Among several, prefer: the
  Candidate's own kernel, then accuracy not failed, then the higher stored speedup over the Baseline,
  then the kernel hash. Its stored speedup is carried only from its workload of an equal workload
  key; one served from a workload of another key carries none.
- **R3.3** A base file with the Candidate's schema name that is not JSON is a platform fault for
  that schema, which is then not measured; a base file that is JSON of another schema format is
  ignored.
- **R3.4** What a workload is timed against, and at which threshold:

  | Case | Timed against | Passes when the ratio | Speedup written back |
  |---|---|---|---|
  | The Incumbent holds the Candidate's own kernel | Baseline | > Retain | measured over the Baseline |
  | A different Incumbent, with a stored speedup s over the Baseline on this target | Incumbent | > Entry | ratio × s |
  | A different Incumbent, without such a speedup (or the op has no Baseline) | Incumbent | > Entry | none |
  | No Incumbent, and no base commit or a first release | Baseline | > Retain | measured over the Baseline |
  | No Incumbent otherwise | Baseline | > Entry | measured over the Baseline |
  | Neither a Baseline nor an Incumbent | nothing (accuracy only) | - | none |

  A Candidate that is not the Incumbent must be faster than it: otherwise it has no reason to merge.
- **R3.5** A schema's workloads run in batches. A batch belongs to one schema and builds the
  Candidate (and each Incumbent it needs) once. A long schema is split into even batches of at
  most 16 workloads, so it spreads over all devices; the batches of all schemas start
  costliest first (about 1 per workload plus 1 per GiB of one input set), one batch per device at
  a time.
- **R3.6** Batches leasing more devices start first; among equal widths the order of R3.5 holds.

## 4. Accuracy check (core)

- **R4.1** Each workload is checked on `num_trials` input sets (a `BenchStage` parameter and the
  `--num-trials` option, default 3). Each set is made by the workload's sampler, chosen when the
  bench plans (today the schema's `init`, on a seed derived from the schema name, the workload's
  axes and the trial index, the same in every process); the timing order's seed comes from the
  workload key.
  A group workload's trial draws one input set per rank, the seed also salted with the rank (rank
  0's unchanged); each rank computes its own Reference from the whole group's input sets.
- **R4.2** Each trial calls the Candidate once, eagerly, on fresh output buffers, the way dispatch
  calls it (through the dispatch adapter), and compares its outputs with the Reference's on the
  same inputs by the schema's `check`. A trial that fails the check, or whose outputs are not the
  declared shapes and dtypes, fails the workload's accuracy. A group trial passes only when it
  passes on every rank; otherwise its finding is the lowest failing rank's, its detail naming the
  rank.
- **R4.3** The Reference runs in the same task as the Candidate, on the same device, and the
  comparison happens in memory: no file and no process boundary between them.
- **R4.4** A Candidate call that raises fails the workload's accuracy (the Candidate's fault). A
  Reference that raises is a platform fault, never the Candidate's.

## 5. Performance check (core)

- **R5.1** Only workloads that passed accuracy are timed, each on its `num_trials` input sets: a
  trial is timed once it passes its check.
- **R5.2** Timing uses `bench_cudagraph` (CUDA graph replay, L2 flush before each call, the kernel
  tracer's GPU duration; `bench_cudagraph` warms up and repeats the call itself) with the method
  recorded in the measurement profile: per trial, each participant timed once, in a seeded
  shuffled order, a 20 ms graph budget, a tracer fault measured again once. A group times with
  `bench_cudagraph(group=...)`: each call takes the slowest rank's time.
- **R5.3** A comparison's ratio is the geometric mean of its trials' ratios (opponent time over
  Candidate time). For the pass/fail gate, each trial's log ratio is clamped to ±ln 2; the speedup
  written back uses the unclamped mean, rounded to 3 decimals and at least 0.001.
- **R5.4** Entry is 1.05 (`DEFAULT_MIN_SPEEDUP`, `AKLR_MIN_SPEEDUP`'s default), Retain is 0.99. Failing Retain is a regression; failing Entry is not
  faster. Both are performance failures.
- **R5.5** The Baseline is bound and run only for a workload timed against it. No adapter binding
  the workload is the schema's fault (the workload must go or the schema be fixed); a Baseline
  that raises is a platform fault.
- **R5.6** An Incumbent that fails to build or run is a platform fault: it passed validation
  before.
- **R5.7** The opponent's binding and its first call are agreed on by the whole group before
  timing.

## 6. Verdicts, admission, demotion and write-back

These rules hold as written. Changing one needs an entry in section 11 first.

- **R6.1** A workload's verdict is its first finding: a batch failure (build, task crash, task
  timeout) when there is one, else its accuracy findings, then its performance finding; none
  means passed.
- **R6.2** Admission: a schema is admitted when at least one workload passed accuracy and its
  published speedup over its opponent reaches Entry, compared with >= as dispatch compares a stored
  one (also where its gate used Retain); where nothing is timed (neither a
  Baseline nor an Incumbent), passing accuracy is enough.
- **R6.3** A workload's accuracy or performance failure, and a batch failure that stands for it,
  is a Demotion: recorded on the workload, and it does not fail the stage.
- **R6.4** A schema with no admitted workload fails the stage, unless another finding already
  decides that schema.
- **R6.5** Any other finding decides the stage: failed when one is a Candidate's or schema's fault,
  else aborted.
- **R6.6** Write-back, per schema whose kernel runs on this target: one Evaluation per workload,
  with the hardware, accuracy passed or failed, and the speedup of R3.4 keyed by the Baseline's
  name with the head commit. A workload that failed accuracy gets no speedup.
- **R6.7** A schema is written back whole or not at all: not when a finding other than "no
  admitted workload" decides it, nor when any workload has no result.
- **R6.8** A schema is rewritten only when a workload's accuracy or speedup keys change, or a
  speedup moves by more than 2%; Evaluations of other hardware stay as they are.

## 7. Execution

- **R7.1** One task per batch runs build, accuracy and timing of all its workloads and returns
  its result as JSON.
- **R7.2** Every task runs in child processes, never in the `aklr validate` process. Two runners,
  chosen per task through `Executor.submit(..., runner=...)`:
  1. `reuse`, the default: the processes stay for the next task on the same devices, keeping
     their imports, GPU contexts and in-process build caches. A task that fails, times out or
     crashes takes its processes with it; the next task starts fresh ones.
  2. `disposable`: fresh processes per task, for an op whose `ExecutionPolicy` isolates it
     (by default every group, whose communication state, a symmetric heap or registered process
     groups, can be set up only once per process), or one named by the `BenchStage` parameter and
     the repeatable `--isolate <op_type>` option, for ops that pollute the process (global state, a
     broken GPU context).
- **R7.3** A task leasing N devices runs on N processes, process i being rank i on device i of the
  lease, with its device current and `RANK`, `WORLD_SIZE` and `LOCAL_RANK` set. As in
  `multiprocessing.Pool`, the caller's `initializer(*initargs)` runs once in each process before any
  task: it makes the `torch.distributed` process group, or any other per-process state, so the
  runners know nothing of either. Every rank runs the task; rank 0's result is the task's, so the ranks agree
  on it inside the task. A rank that fails, times out or exits ends every rank, which a collective
  would otherwise hang.
- **R7.4** A crash or timeout is the Candidate's fault for its whole batch, in either runner.
- **R7.5** A task's timeout covers its build, accuracy (Reference included) and timing; build and
  per-workload budgets are instance parameters.
- **R7.6** When a rank's next step may differ from another rank's, the group agrees on it first;
  a failure outside an agreement ends every rank.
- **R7.7** The bench executor's startup budget covers N processes' imports and the process
  group's setup: a runner that overruns it takes its devices out of the pool.

## 8. Findings and the report

- **R8.1** Finding codes may be renamed, merged or removed. Each finding must still say:
  - whose fault it is: the Candidate's, the schema's or the platform's (this sets failed or
    aborted);
  - where: the schema, and the workload or the whole batch;
  - what: accuracy failed, performance failed, or not measured (this sets Demotion and the
    written accuracy).

  | Situation | Fault | What |
  |---|---|---|
  | Build fails, call raises, wrong output shape or dtype, task crash, task timeout | Candidate | accuracy failed (workload or batch) |
  | Outputs fail the check | Candidate | accuracy failed |
  | Ratio below its threshold | Candidate | performance failed |
  | No admitted workload | Candidate | decides the schema |
  | No adapter binds a workload timed against the Baseline | schema | decides the stage |
  | Baseline, Incumbent or Reference raises; tracer fault; unknown device; kernel targets no matching hardware; unreadable base file; unparsable task result | platform | decides the stage |

- **R8.2** The report's format is free. It must give, per workload: schema path, its workload key
  (records and findings name a `workload`, written back by the key's text), verdict
  (and whether demoted), what it was timed against, the ratios with their raw times, and the
  kernel hash; per stage: the deciding findings and the Demotions; and the measurement profile.
  Its readers change in the same change:
  `harness/scripts/forge_task/assets/aklr_forge_driver.py` and `harness/tests/forge_task/`, the
  `aklr-read-validation` skill, and `.github/workflows/ci.yaml` (which reads only `platform`).
- **R8.3** Each finished batch writes one progress line: schema, workload count, speedup range,
  accuracy summary.

## 9. Non-functional requirements

- **N1** The structure checker (`code-structure` S1–S7) reports no BLOCK on the rewritten files.
- **N2** No module-level constants or statics: parameters are fields of the instance that uses
  them, fixed sets are enums. An exception needs a reviewing agent's sign-off.
- **N3** Planning (R3), verdicts, Admission and Demotion (R6.1–R6.5) and write-back (R6.6–R6.8)
  are testable on the CPU without starting a task; GPU behaviour (R4, R5, R7.3) has GPU tests.
- **N4** Test first (the `tdd` skill), following `tests/AGENTS.md`.
- **N5** The old code of a feature is deleted before its new code is written: no parallel paths,
  flags or shims.
- **N6** `_execution/`, `aklr/core/runtime/` and `aklr/core/schema/` are unchanged.
- **N7** Code comments do not cite this document's requirement ids.

## 10. Optional checks (deferred)

Not part of the rewrite. Each is added after the design only if it stays cheap and simple there;
if it makes the design noticeably more complex, it is left out.

| Check | Catches |
|---|---|
| Poison the outputs (NaN, 0xFF) before each call | a kernel that leaves outputs partly unwritten |
| Byte-compare inputs before and after a call | a kernel that writes its inputs |
| Compile tripwire during graph capture (Triton, FlyDSL) | a kernel that compiles inside a capture |
| Replay one captured graph on new inputs copied into its buffers | a kernel that bakes input values or addresses into the graph |
| Check again after timing | a kernel whose results change after many calls |
| Reference rounded to the output dtypes passes its own check | a schema whose check rejects its own Reference (blamed on the schema, not the Candidate) |
| Cap the first call's peak and resident memory (16 GiB) | a kernel that hoards device memory |

## 11. Changes from the current behaviour

| # | Change | Why |
|---|---|---|
| C1 | Candidate and Reference in one task, compared in memory (R4.3, R7.1) | the simplest path: no safetensors handoff through the host, no second task, no task directories to clean |
| C2 | Reused processes by default; fresh ones with `--isolate` per op type (R7.2) | isolation is needed against pollution, not against the Candidate; a crash must not end the run |
| C3 | The Candidate is not treated as adversarial | the threat is a buggy or polluting kernel, not reward hacking; the anti-cheat checks become optional (section 10) |
| C4 | Accuracy on `num_trials` eager calls (R4.1, R4.2) | replaces 9 graph calls per workload (first_capture, numeric, post_timing) |
| C5 | A workload that fails accuracy is not timed and gets no speedup (R5.1, R6.6) | its speedup was never used: dispatch and Incumbent choice skip failed accuracy |
| C6 | A Candidate that is not the Incumbent is timed against the Incumbent only, at Entry (R3.4) | it must be faster than what serves today; replaces timing against both the Baseline and the Incumbent |
| C7 | Speedup derived through the Incumbent's stored speedup (R3.4) | keeps Evaluations keyed by the Baseline without timing it |
| C8 | The Baseline is bound only for workloads timed against it (R5.5) | follows from C6 |
| C9 | Finding codes redesigned (R8.1) | one code, `candidate_crash`, covers too many situations (P5) |
| C10 | Report format free; its readers change with it (R8.2) | |
| C11 | `--num-trials` and `--isolate` options | C2, C4 |
| C12 | The 16 GiB per-batch output cap and the Reference's time budget per output GiB go | outputs are no longer kept for a second task (C1) |
| C13 | The Incumbent serves the workload's dispatch key; its stored speedup is carried only from an equal workload key (R3.2) | dispatch decides what serves a workload; a speedup measured on other inputs does not describe these |
| C14 | Input sets and their seeds come from the workload's sampler, chosen when the bench plans; the timing order's seed from the workload key (R4.1) | one way to make input sets for the bench, the forge driver and op tests, seeds unchanged in value; a new sampler does not touch the bench's check loop |
| C15 | Records and findings name the workload key, `workload` in place of `axes`; report version 3 (R8.2) | a workload's identity can grow (a traffic profile) without changing the report's readers again |

## 12. Problems in the current code

- **P1** `_stage.py` mixes planning (Incumbents, thresholds, batches), running two tasks and their
  directories, and judging (`_judged_findings`).
- **P2** `_report.py` mixes rendering with Admission and Demotion (`BenchReport.from_findings`)
  and write-back (`BenchReport.measurements`).
- **P3** Judging is split between `_stage._judged_findings` and `BenchReport.from_findings`, and is
  tested only through a whole stage run with its tasks monkeypatched.
- **P4** "baseline" means two things: what is timed (the adapter's call, or the Incumbent when there
  is no adapter), and `TrialRequest.baseline` / `WorkloadRecord.baseline` (the served kernel's
  path, else the adapter's entry point).
- **P5** `candidate_crash` covers a crashed child, a raising task, wrong output shapes or dtypes, a
  missing outputs file, unusable timing rounds, and a task result that does not parse (which may
  be AKLR's own defect).
- **P6** Dead code: `WorkloadEvaluation._fail` in `_reference.py` keys on `schema_invalid`, which the
  bench never raises.
- **P7** Structure checker BLOCKs in scope: S4 on `BenchStage._incumbents`,
  `BenchStage._judged_findings` and `CandidateRun._poison_outputs`; S2 on `_bytes` in
  `_candidate.py`. (Two more are in the frozen `_execution/`.)
- **P8** 20 module-level constants in scope: 5 in `_stage.py`, 2 in `_report.py`, 2 in
  `_candidate.py`, 2 in `_trials.py`, 9 in `_measure.py`.
- **P9** The candidate copies every step's outputs to the host and a safetensors file for the
  Reference task: the slow path that #32 removed and 156493228 brought back.
- **P10** Broken readers: `aklr_forge_driver.py` imports the removed
  `_bench._evaluation._request`, and `harness/tests/forge_task/` calls `BenchReport.from_findings`
  with its old signature.
- **P11** `_bench/_planning/` is a leftover directory holding only `__pycache__`.

## 13. Acceptance

- **A1** Every requirement of sections 3–8 has a test: CPU tests for those N3 names, GPU tests for
  the rest.
- **A2** N1, N2 and N6 hold on the final diff.
- **A3** The old code is gone: the safetensors handoff, the second task, `ReferenceNotProduced`,
  the task-directory cleanup, `_planning/`, and the checks of section 10.
- **A4** The forge driver and its tests pass against the new report, and the
  `aklr-read-validation` skill describes it.
- **A5** On an MI355X, a full re-bench (a change under `aklr/core/`) completes with reused processes
  and the needed `--isolate` ops, and admits every kernel the current code admits there.
