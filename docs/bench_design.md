# Bench stage design

The structure of the rewritten `aklr/core/validation/stages/bench/`, checked against
[bench_requirements.md](bench_requirements.md). Terms are those of [GLOSSARY.md](../GLOSSARY.md).

## 1. Layers

The bench's public module is the stage; the package also exports `ExecutionPolicy`, the extension point an op
registers in, and the samplers. Behind it, private packages own one concept each: an op's execution
(`_execution_policy`), what a workload is timed against and how it is judged (`_admission`), how a workload is called
(`_workload_call`), one task's run (`_task`) and what a measurement is (`_measurement`); two private modules hold the
batch schedule and the report. The processes tasks run in are validation's own
`aklr/core/validation/_execution/`, under the bench rather than inside it.
What varies with the kind of opponent (how it binds, its threshold, how it admits, the speedup it
writes back) lives in one class per kind: the R3.4 table is the only branch on it.

```
BenchStage.run                                       level 0: plan, schedule, run, judge, report
├── _admission/  BenchPlanner.plan(scope, target)    level 1 [meaning]: what each workload is timed against, and its verdict
│                ├── _plan.py       IncumbentPool, SchemaPlan.from_schema (the R3.4 table), WorkloadPlan
│                ├── _opponent.py   Opponent {BaselineOpponent, IncumbentOpponent, Unopposed}: label, name, bind, admits, speedups
│                ├── _threshold.py  Threshold {ENTRY, RETAIN}: passes, failing_code
│                └── _verdict.py    SchemaVerdict.judge, WorkloadJudgement: findings, Admission, Demotion, write-back
├── _batch_schedule  BatchSchedule.batches/runner/timeout  level 1 [speed]: how workloads are batched, when and where batches run
│                └── SchedulePass   one rewrite of the batch list (CostliestFirst, WidestFirst); checked after each
├── _task/       Batch.run(context) -> JSON          level 1 [meaning]: one task's check and timing of a batch
│                ├── BatchRun       each workload: check (num_trials trials), then time against its bound opponent
│                └── GraphTimer     Candidate and opponent timed in CUDA graphs, a tracer fault read again once
├── _workload_call/                                  how a workload is called
│                ├── InputSampler, InitSampler       its inputs
│                ├── KernelCall                      a built kernel, called the way dispatch calls it
│                ├── ReferenceCall                   its expected outputs
│                └── OutputCheck                     the outputs against the Reference
├── _measurement/                                    shared base [meaning]
│                ├── _measurement_profile.py  MeasurementProfile {digest}: trials, graph budget, bars
│                └── _workload_result.py      Comparison, WorkloadResult
├── _report      BenchReport.from_verdicts           level 1: the stage's report, the verdicts gathered and rendered
└── _execution_policy/ ExecutionPolicy.for_schema    an op's ranks and set-up; <op_type>.py registers a subclass (mega_moe.py)

validation/_execution/                               the task executor and its runners, below the bench
```

Imports run one way: `_measurement`, `_execution`, `_execution_policy` < `_workload_call` < `_admission._threshold` <
`_admission._opponent` < `_admission._plan` < `_task` < `_batch_schedule` < `_admission._verdict` < `_report` < `stage`;
inside the bench a module imports another's leaf module. The one reference back, `Opponent.bind(batch_run, ...)`, is a
type annotation only: an opponent binds itself, `BatchRun.time` times it.

Change is kept apart by kind (from the history of the bench): what changes a number or a verdict's
meaning (the measurement protocol: timing method, trials, accuracy checks, the R3.4
table) lives in `MeasurementProfile` and the `Opponent` classes; what changes only
how fast results come (batching, order, isolation, budgets, and later sharding, deadlines, reuse)
lives in `BatchSchedule`. The protocol is named by `MeasurementProfile.digest` (the profile and the timer's
sources), written into the report. Optional accuracy checks (section 10) get a `TrialCheck` hook
when the first one is added. `BatchSchedule.batches` splits the plans evenly, then applies its `SchedulePass`es
in order; after each pass it checks that every planned workload is in exactly one batch of its own schema
and that no batch's profile changed, and raises naming the pass otherwise, so a speed change cannot change
what a result means. Later speed changes are new passes (new kernels or failures first, a probe batch per
schema, sharding, reuse of recorded results with a result log added by the first pass that reads one);
reacting while batches run (fail-fast, retry) has no place yet and needs its own requirement first.

Patterns: the opponent kind is a class hierarchy (Strategy, with `Unopposed` as Null Object) rather
than a type code; `BatchRun.result` is the fixed skeleton every workload runs through (Template
Method); a failure anywhere in a workload raises `FindingRaised`, caught once by the skeleton;
`Threshold` and `FindingCode` are enums whose members carry what they mean (failing code; fault and
outcome); the schedule's passes run in order over the batch list, each checked after it ran (a pass
manager with a verifier, as in MLIR's or Catalyst's rule executor). Plans and opponents are plain data that pickle into a task; only `BatchRun` holds live
kernels.

## 2. Structure

| file | concept | classes | public methods/functions |
|---|---|---|---|
| `bench/__init__.py` | the bench package's exports: `BenchStage`, `ExecutionPolicy`, and `InitSampler`, `InputSampler`, `ReferenceCall` for op tests and the forge driver | - | - |
| `bench/stage.py` | the bench stage | `BenchStage` | `run`, `plan`; `report_type` |
| `bench/_report.py` | the bench's stage report | `BenchReport` | `BenchReport.from_verdicts`, `record_blocks`, `console_lines` |
| `bench/_batch_schedule.py` | the bench's batch schedule and its passes | `BatchSchedule`, `SchedulePass`, `CostliestFirst`, `WidestFirst` | `batches`, `runner`, `timeout`, `apply` |
| `bench/_execution_policy/__init__.py` | an op type's execution policy | `ExecutionPolicy` | `ExecutionPolicy.for_schema`, `ranks_per_task`, `isolate`, `setup` |
| `bench/_execution_policy/mega_moe.py` | mega_moe's execution policy | `MegaMoeExecutionPolicy` | `ranks_per_task`, `setup` |
| `bench/_admission/__init__.py` | the admission package's exports | - | - |
| `bench/_admission/_plan.py` | the plan of each benched schema, its Incumbents found | `BenchPlanner`, `SchemaPlan`, `WorkloadPlan`, `Incumbent`, `IncumbentPool` | `BenchPlanner.plan`, `SchemaPlan.from_schema`, `unmeasured`, `IncumbentPool.read`, `incumbent` |
| `bench/_admission/_opponent.py` | a workload's opponent | `Opponent`, `BaselineOpponent`, `IncumbentOpponent`, `Unopposed` | `label`, `name`, `bind`, `admits`, `speedups` |
| `bench/_admission/_threshold.py` | a speed threshold | `Threshold` | `passes`, `failing_code` |
| `bench/_admission/_verdict.py` | a schema's verdict and each workload's judgement | `SchemaVerdict`, `WorkloadJudgement`, `WorkloadRecord` | `SchemaVerdict.judge`, `WorkloadJudgement.judge`, `verdict`, `is_demoted`, `is_accurate`, `is_measured`, `is_admitting`, `evaluation`, `record` |
| `bench/_measurement/__init__.py` | the measurement package's exports | - | - |
| `bench/_measurement/_measurement_profile.py` | the measurement profile | `MeasurementProfile` | `digest` |
| `bench/_measurement/_workload_result.py` | a workload's result from its task | `Comparison`, `WorkloadResult` | `Comparison.from_timings`, `WorkloadResult.failed` |
| `bench/_task/__init__.py` | the task package's exports | - | - |
| `bench/_task/_batch.py` | a batch: one task's share of a schema | `Batch` | `run`, `cost`, `failed`, `results`, `summary`; field `execution_policy` |
| `bench/_task/_batch_run.py` | a batch's run | `BatchRun` | `result`, `check`, `incumbent`, `time` |
| `bench/_task/_graph_timer.py` | the graph timer | `GraphTimer`, `TimingFault` | `time` |
| `bench/_workload_call/__init__.py` | how a workload is called: the exports | - | - |
| `bench/_workload_call/_input_sampler.py` | how a workload's input sets are made | `InputSampler` | `InputSampler.for_workload`, `draw`, `draw_group` |
| `bench/_workload_call/_init_sampler.py` | input sets drawn by the init of a schema's standard | `InitSampler` | `seed`, `draw` |
| `bench/_workload_call/_kernel_call.py` | a built kernel's dispatch-way call | `KernelCall` | `KernelCall.build`, `bind` |
| `bench/_workload_call/_reference_call.py` | a workload's Reference, called the bench's way | `ReferenceCall` | `outputs` |
| `bench/_workload_call/_output_check.py` | the outputs' check against the Reference | `OutputCheck` | `verify` |
| `report/_finding.py` | a stage finding | `Fault`, `Outcome`, `FindingCode`, `FindingRaised`, `Finding` | `FindingCode.fault`, `.outcome`, `.is_demotion`, `.is_platform_fault`, `.failing_every_rank` |

No top-level functions. A step every rank of a task fails together is `FindingCode.failing_every_rank(context)`, the
finding code it fails as: `with FindingCode.CANDIDATE_RAISED.failing_every_rank(f"trial {trial}"):`.

## 3. Level 0: the stage

`BenchStage(devices, *, isolate, num_trials)`, holding a `BatchSchedule` and a `MeasurementProfile`:

1. the target from `earlier.platform`; an incomplete one is a platform finding (unknown device);
2. `plans = BenchPlanner(min_speedup).plan(scope, target)`;
3. `BatchSchedule.batches`: every plan's workloads in even batches of at most `batch_size`, then its passes
   (`CostliestFirst` in `BenchStage`), submitted to
   `Executor.for_devices(devices, runner="reuse")` with `BatchSchedule.runner` (`disposable` when the batch's
   `execution_policy.isolate` or `--isolate` names its op type), leasing `execution_policy.ranks_per_task` devices, and `BatchSchedule.timeout` (`build_timeout_secs + workload_timeout_secs × workloads`); `Batch.results(future)` turns a raised, crashed or
   timed-out task into findings;
4. `BenchReport.from_verdicts(name, [SchemaVerdict.judge(plan, results, ...) for plan in plans], profile)`.

## 4. Level 1: planning (`_admission/_plan.py`)

`BenchPlanner.plan` returns one `SchemaPlan` per bench path with a kernel, with the op's Baseline
name: when the standard of the schema's template has a baseline, the backend of the op's seam adapters, the
name dispatch reads speedups by. A schema json holds no standard: read from json, a schema takes its template's
(`Schema.template()`), so `SchemaPlan.schema` and `Batch.schema` have it. It hides:

- `IncumbentPool.read(scope, path, name)`: the base files of the schema's name under its op's
  `impls`; a file that is not JSON is a platform finding, one of another format is ignored.
  `incumbent(...)` ranks the qualifying ones (own kernel, accuracy not failed, stored speedup,
  kernel hash); a base kernel qualifies when it runs on the target and dispatch serves it there, or it is the
  Candidate's own.
- `SchemaPlan.from_schema(...)`: the R3.4 table, the one branch on the kind of opponent: another kernel
  serving → `IncumbentOpponent(ENTRY)`; no Baseline → `Unopposed`; the own kernel serving, or no
  base or a first release → `BaselineOpponent(RETAIN)`; else `BaselineOpponent(ENTRY)`.

## 5. Level 1: the task (`_task/`)

`Batch.run(context)` calls `execution_policy.setup()` on every rank, then builds the Candidate as a `KernelCall` and, on a `BatchRun`, runs each
workload through `BatchRun.result`: `BatchRun.check`, then `plan.opponent.compare(batch_run, ...)`;
a `FindingRaised` anywhere becomes the workload's result. It returns the JSON of the results.

- `BaselineOpponent.compare` calls the standard's baseline on the call and its output buffers (None: a schema
  finding; raising, a backend not installed included: a platform finding); `IncumbentOpponent.compare` uses `BatchRun.incumbent`
  (built once; failing: a platform finding); `Unopposed.compare` times nothing.
- `BatchRun.result` times each trial once its check passes; `BatchRun.time` times both once, in a
  seeded shuffled order, and the trials' comparisons are joined into one.

## 6. Level 1: judging (`_admission/_verdict.py`)

`WorkloadJudgement.judge` adds the threshold's finding (`Threshold.failing_code`) when the ratio does
not clear it; its first finding is its verdict; `is_admitting` asks its opponent; `evaluation` takes
its opponent's `speedups`, none when inaccurate. `SchemaVerdict.judge` gathers them: deciding findings,
Demotions, "no admitted workload", and the Evaluations to write back when the schema is whole.
`BenchReport.from_verdicts` gathers verdicts; the pipeline writes back `BenchReport.evaluations[path]`,
rewriting only on a change (its existing 2% rule).

## 7. Findings

| code | fault | outcome |
|---|---|---|
| `build_failed`, `candidate_raised`, `wrong_outputs`, `task_crashed`, `task_timeout`, `outputs_mismatch` | candidate | accuracy failed |
| `regressed` (≤ Retain), `not_faster` (≤ Entry) | candidate | performance failed |
| `no_admitted_workload` | candidate | decides the schema |
| `baseline_unbound` | schema | not measured |
| `platform_fault` | platform | not measured |

A batch failure is one finding per workload of the batch; its code says it was the batch.

## 8. Execution

Every task runs in child processes spawned by `_execution`'s runners; the `aklr validate` process
makes no GPU call of its own.

- **Runners.** `disposable` spawns a task's processes and closes them after it. `reuse` is the same
  runner, kept by the supervisor after a task that succeeded, keyed by the lease's resource IDs; the
  next task on the same resources reuses it. A lease that overlaps an idle runner's resources but
  is not the same closes it first, so a device never holds two runners' contexts.
- **Ranks.** A lease of N devices is N processes; process i imports `aklr.core.utils.testing` (the
  kernel tracer registers as torch initializes the GPU, before the process's first kernel), then sets
  device i current, with `RANK`, `WORLD_SIZE` and `LOCAL_RANK` set, then calls the executor's
  `initializer(*initargs)` once, as `multiprocessing.Pool` does: the caller makes the process group
  there (a file rendezvous in the process's directory works). A failing initializer fails the
  start without taking the device out; a failing `set_device` takes it out. The supervisor waits for every rank's reply; the first rank that fails,
  exits or misses the deadline ends them all. Rank 0's result is the task's.
- **Between tasks.** A rank synchronizes its device after its task, so a late GPU fault is that
  task's. Any failure closes the runner; only a healthy one is reused.
- The kernel tracer's sessions still share one window, reference-counted, as thread mode needed;
  with one task per process now only one session is open at a time.

## 9. Requirement trace

| req | where | test |
|---|---|---|
| R3.1–R3.4 | `_admission/_plan.py` | CPU: `test_bench_stage.py` (`BenchStage.plan`) |
| R3.5, R7.2, R7.5 | `BatchSchedule.batches`, `SchedulePass` | CPU: `test_bench_stage.py` (tasks submitted by `BenchStage.run`) |
| R4.1–R4.4 | `BatchRun.check`, `KernelCall` | GPU: `test_bench_task.py` |
| R5.1, R5.5, R5.6 | `BatchRun.result`, `.time`, `Opponent.bind` | GPU: `test_bench_task.py` |
| R5.2 | `GraphTimer.time` | GPU: `test_bench_task.py` |
| R5.3, R5.4 | `Threshold.passes`, `Comparison.from_timings` | CPU: `test_bench_stage.py` (thresholds); the statistics and the tracer's re-read only end to end on a GPU |
| R6.1–R6.7 | `WorkloadJudgement`, `SchemaVerdict.judge`, `Opponent.admits`, `.speedups` | CPU: `test_bench_stage.py` (`BenchStage.run`) |
| R6.8 | `PipelineReport.measured_schemas` | CPU: `test_pipeline.py` |
| R7.1–R7.5 | `BenchStage.run`, `Batch.run`, `command.main` | GPU: `test_bench_task.py` |
| R8.1–R8.3 | `FindingCode`, `BenchReport`, `Batch.summary` | CPU: `test_bench_stage.py` |

## 10. Test seams

Tests call the bench through `BenchStage` alone, the one public module of the bench:

- `BenchStage.plan(scope, target)` on a temporary git repository: Incumbent choice, the versus/threshold table;
  misses anything a task does.
- `BenchStage.run(scope, earlier)` on the CPU, its task processes mocked (`Executor` in `bench.stage` replaced, each
  task answering with scripted results in the task's own JSON): batching, isolation, budgets, verdicts, Admission,
  Demotion, write-back and the protocol's name; misses the check, the timing and the statistics a task computes.
- `BenchStage.run` with a `reuse` executor on a GPU, for small `gemm_bf16_nt` schemas: the
  accuracy check, timing and the task JSON; misses MI355X-scale timing (A5).
