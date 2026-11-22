# PR 3 (validation 重构) 相对 PR 270 的适配清单

PR 3 把 PR 270 (`refactor/core-validation-pr2`) 里的 validation 重构单独提到 main 上。main 还是 contract 层
(`aklr.core.contract`)，没有 schema 模型 (`aklr.core.schema`、`OpVariants`、`KernelFunction`、seam)，所以凡是
validation 接模型的地方都改成了走 contract 层。本文件列出每一处，后续重构 core 的 PR 合入 schema 模型时按这里改回。

对照基准：PR 270 head `73fdf9ff7`，PR 3 分支 `refactor/validation-pr3`。查看全部差异：

```
git diff origin/refactor/core-validation-pr2 refactor/validation-pr3 -- aklr/core/validation aklr/core/common tests/core/validation
```

## 一、与 PR 270 完全一致，core PR 不用动

- `aklr/core/common/` (`Registrable`、`Singleton`、`ClassRegistry`)：代码原样带入，仅按 review 删掉了 docstring 和注释（core PR 以 #270 的版本为准即可）。
- `aklr/core/validation/_execution/` 全部文件。
- `aklr/core/validation/command.py`、`stages/__init__.py`、`__init__.py`。
- `stages/report/_markdown.py`、`stages/bench/_plan/_threshold.py`、`stages/bench/_measure/_schedule.py`、
  `stages/bench/__init__.py`。
- 只差一行 import 的文件 (`frozen_dataclass` 的来源，见二.1)：`stages/stage.py`、`stages/report/_finding.py`、
  `stages/bench/_measure/_profile.py`、`stages/bench/_measure/_result.py`。

## 二、适配点 (core PR 需要改回)

### 1. `frozen_dataclass`

- PR 270：`aklr/core/schema/_frozen_model.py`。
- PR 3：`aklr/core/schema/` 在 main 的 AGENTS.md 里是受保护目录，所以复制了一份到
  `aklr/core/validation/_frozen_model.py`，validation 里所有 `from aklr.core.schema._frozen_model import ...`
  改成 `from aklr.core.validation._frozen_model import ...`。
- core PR：删掉 `aklr/core/validation/_frozen_model.py`，import 改回。

### 2. bench 的模型适配器：`stages/bench/_measure/_contract_workload.py` (PR 3 新增，PR 270 没有)

`ContractWorkload` 是 main 上 `TrialRequest` 去掉 step 表后剩下的部分：一个 schema 的一个 workload 在 contract 层下的
Point、`schema_json` / `workload_json` / `distributions`，以及 `draw` (`semantics.row_major_inputs`)、
`assemble_call`、`input_bytes`、`ranks`、`has_error_bound`。bench 里所有需要 Point 或 semantics 的地方都经过它。

- core PR：整个文件删除。它的使用方见下面 3 到 8。
- `tests/op_case.py` 的 `OpCase.contract_workload` (原 `OpCase.request`) 也用它；contract 层删除时一并处理。

### 3. `stages/bench/_measure/_input_sampler.py`

- PR 270：`schema.standard.init(**axes, device=, seed=WorkloadKey.seed(...))`。
- PR 3：`ContractWorkload.of(schema, axes).draw(f"trial {trial}", device, rank)`，种子由 semantics 按 input set 名
  和 rank 派生；scalar 输入用 workload 自己记录的值（`ContractWorkload.draw` 在画完后替换；每个 trial 的精度和计时都用它）。

### 4. `stages/bench/_measure/_built_kernel.py`

- PR 270：`KernelFunction(schema)`，`bind(inputs, axes)` 调 `kernel(call, {**const_axes, **axes})`。
- PR 3：`BuilderRegistry` 构建 + `KernelDispatcher` / `DispatchTable.from_schemas(contract, device, [(schema, runnable,
  workloads)], min_speedup=0)`，和 main 的 `make_warmed_dispatcher` 一样；`build(schema, device, kernel_schema=None)`
  多了 `device` 和 `kernel_schema` (Incumbent 用 Candidate 的 schema 做 key、自己的 kernel 做 runnable)；
  `bind` 返回的调用给出 `contract.call_mutable_tensors` (按名字的 dict)，dispatch 回退时抛错；新增 `call(inputs, axes)`
  给 Baseline 组装调用。
- 连带：`_op_bench.py` 的 `incumbent()` 调 `BuiltKernel.build(self.candidate.schema, self.device, schema)`；
  `_task.py` 的 `run()` 调 `BuiltKernel.build(self.schema, device)`。

### 5. `stages/bench/_measure/_output_check.py`

- PR 270：`schema.standard.reference` / `schema.standard.check`，输出按位置的 list。
- PR 3：`contract.reference_outputs(point, input_sets, rank, decode)` (float32，溢出时 float64)、dot_product bound 的
  magnitude、baseline_relative bound 的 Baseline 输出，然后 `semantics.compare_outputs`；`reference()` 返回的是
  `compare_outputs` 的关键字参数 dict，`verify()` 的 `produced` 是按名字的 dict。

### 6. `stages/bench/_measure/_task.py`、`_op_bench.py`、`_ops/mega_moe.py`

- `_task.py` 的 `cost`：PR 270 用 `schema.get_input_shapes`；PR 3 用 `ContractWorkload.input_bytes`。
- `_profile.py`：`digest` 的源文件列表多了 `_contract_workload.py`。
- `_op_bench.py`：类型注解 `Schema` 指向 `OpContract`；两处 docstring 把 "init" 改成 "distributions"。
- `_ops/mega_moe.py` 的 `ranks`：PR 270 是 `schema.const_axes["ep_size"]`；PR 3 是
  `ContractWorkload.of(schema, schema.workloads[0].axes).ranks` (contract 的 `ranks` attr)。`setup` 与 PR 270 一致。
- main 上的 `aklr/ops/mega_moe/group.py`、`OpContract.group_init` / `group_init_name` 和
  `tests/core/contract/test_ranks.py::test_no_group_py_is_no_group_init` 已删除 (分组初始化改由 `MegaMoeBench.setup` 做)。

### 7. `stages/bench/_plan/`

- `_plan.py`：`OpVariants.load` 改为 `ContractRegistry.default().parse_schema`；"没有 kernel" 是
  `not schema.kernel.sources`；Baseline 名字是 contract 的 baseline 函数的点分名
  (如 `aklr.ops.gemm.baseline.gemm_baseline`)，PR 270 是 seam adapter 的 backend 名 (如 `aiter`)；
  `kernel.build.requires` / `runs_on` 改为 `kernel.requires` / `DeviceTarget.matches_target(target, kernel.targets)`；
  `untimed()` 少了 `target` 参数，按 `w.accuracy != "failed"` 过滤。
- `_incumbent_pool.py`：base 文件用 `registry.parse_schema` 读，且每个 workload 都要是 Candidate 的 contract 下的
  Point，否则跳过；`DispatchKey.from_workload` 改为 `point_with_contract_id(...).key()`；`kernel.hash()` 改为
  `kernel.content_hash`，"自己的 kernel" 用 `kernel.same_kernel`；`schema.serves(...)` 改为新加的 `_runs_here(base)`
  加 `workload.qualifies(min_speedup)`；存储的 speedup 直接是 `workload.speedup`。`_plan.py` 的 `from_schema` 多了一条：
  op 有 Baseline 而 Incumbent 没有存储的 speedup 时不选它做对手，改为对 Baseline 计时 (否则 passed 的 workload 写不出
  相对 Baseline 的 speedup)；PR 270 里这种情况是写回不带 speedup 的 Evaluation。PR 270 的 TODO 里说的
  `Schema.serves` 去重在这里不适用。
- `_opponent.py`：`BaselineOpponent.bind` 用 `contract.baseline()(**call)`，`NotImplementedError` 记为
  `BASELINE_UNBOUND` (PR 270 是 `standard.baseline(**inputs)` 返回 None 表示不接)；`IncumbentOpponent.schema` 是
  `OpContract`。

### 8. `stages/bench/_judge/` 和 `pipeline.py` (回写格式)

- PR 270：`Evaluation(hardware, accuracy, speedup={baseline: Speedup(commit, value)}, microseconds)`，按硬件存在 workload
  的 `evaluations` 里。
- PR 3：回写的就是 main 的 `WorkloadSpec` (`accuracy` / `speedup` / `microseconds`)。
  - `_verdict.py`：`WorkloadJudgement.evaluation(workload)` 返回带测量值的 `WorkloadSpec`；因为 `WorkloadSpec` 要求
    passed 必须有 speedup：op 没有 Baseline 时写相对 Incumbent 的 speedup (Incumbent 充当 Baseline)，什么都没计时则写
    1.0 (main 原有规则)；`SchemaVerdict.judge` 去掉了 `head_sha`、`target` 两个参数；`kernel_hash` 取
    `kernel.content_hash`。
  - `_report.py`：`evaluations: dict[str, dict[str, WorkloadSpec]]`。
  - `stage.py`：`SchemaVerdict.judge(p, results[p.path], self.profile)`。
  - `pipeline.py`：`measured_schemas` 用 `registry.parse_schema` + `schema.with_measurements(measured)`，比较的是
    workload 上的三个字段；`write_schemas` 写 `schema.schema_text()`。
- 文案里的 `kernel.build.requires` 都写成了 `kernel.requires` (`stage.py`、`_plan.py`、`_report.py`)。

### 9. `stages/workload_key.py`

- `Workload` 类型是 `aklr.core.contract.spec.WorkloadSpec`；docstring 去掉了 Evaluation、template 的说法。

### 10. audit 和 scope：保留 main 的逻辑，只换目录和 report 接口

这两块的规则和模型绑定 (kernel 入口是 keyword-only、call keys、mutable tensors、schema 根目录含 `aklr/plugins`、
文件名按内容寻址)，PR 270 的版本在 main 上会把现有 kernel 全判错，所以没有采用。

- `stages/audit/_hip.py`、`_hsaco.py`、`_rules.py`、`_stage.py`：内容是 main 的 `stages/_audit/*`，改动只有
  import 路径、`Finding(workload=None)` (原 `axes=None`)、去掉 `AuditReport` 改用 `StageReport`、
  `StageReport.from_findings(self.name, findings)`。
- `stages/validation_scope.py`：内容是 main 的，只把 `_SCHEMA_PATH_PATTERN` 改名为公开的 `SCHEMA_PATH_PATTERN`
  (`report/_markdown.py` 要用)。PR 270 的 `layout`、`is_measurement_only`、"schema-only 时取所有变更文件" 没有带入。
- core PR：这五个文件整体换成 PR 270 的版本。

### 11. PR 270 改了、PR 3 没带的文件

- `aklr/__main__.py`：PR 270 删了 `export` 命令；PR 3 保持 main 的原样 (contract 的 export 还在)。
- `aklr/core/utils/testing.py` 的 `capture_error_mode="thread_local"`、`aklr/core/utils`、`aklr/core/platform` 的其他改动：
  没带。PR 3 的 bench 在 main 的这些文件上实测能跑。

## 三、测试

- `tests/core/validation/test_audit.py`、`test_validation_scope.py`：main 的版本，只改了 import 路径和
  `f.code.value` → `f.code` (`frozen_dataclass` 把枚举存成字符串)。core PR 换成 PR 270 的版本。
- `test_pipeline.py`、`test_bench_stage.py`、`test_bench_task.py`：PR 270 的用例，夹具换成 contract 层的
  (`tests/op_case.py` 的 `OpCase`，gemm@1.0 和 mega_moe@1.0)。差别：
  - Baseline 名字断言是 `aklr.ops.gemm.baseline.gemm_baseline` 而不是 `aiter`；workload key 是 `m=1` 而不是 `M=1`。
  - main 上每个 contract 都有 Baseline，"没有 Baseline 的 op" 的用例用 fixture 把 `OpContract.baseline` 打成 None。
  - 玩具分组 op (`toy_group.py`) 删除：计划/调度用例改用 mega_moe 的 `ep_size` = 2/4/8；GPU 端到端的分组用例
    通过 sitecustomize 给 gemm 注册一个按环境变量 `AKLR_TEST_RANKS` / `AKLR_TEST_CLOCK_MS` 取 rank 数和时钟的
    `OpBench`，每个 rank 在自己的输入上算 gemm。
  - "只重写这一种硬件" 的用例在 main 上不成立 (一个文件一个 target)，改成 "speedup 变了就重写"；
    多了一个 `write_schemas` 的用例。
  - "什么都没计时" 时回写的是 `speedup=1.0`。
- `test_validation_cli.py`：多一行建 `aklr/ops` 目录 (main 的 scope 要列 op 目录)。
- `tests/ops/test_reference.py`：原来从 validation 内部 import 的 `is_reference_comparable` 已随旧 bench 删除，
  检查内联进了测试。
- `tests/core/contract/test_ranks.py`、`test_schema.py`：`OpCase.request` → `OpCase.contract_workload`。

## 四、PR 3 相对 main 去掉的行为 (PR 270 里本来就没有)

新协议是每个 trial 画输入、对 Reference、再对对手计时一次。main 上这些检查不再做：capture 安全 (首次 capture、
graph 内编译的 tripwire)、输入被改写、determinism、post-timing 一致性、large-magnitude 输入集、workload 自己的
scalar 输入集、首次调用的显存上限、输出缓冲区投毒、Reference 可比性自检、tracer fault 重跑策略。
`aklr_validation/baseline/` 导出 (`write_baselines`) 也没有了。

## 五、harness

`harness/scripts/forge_task/assets/aklr_forge_driver.py`、`harness/tests/forge_task/` 按新的 report 格式改了，
`fake_aklr.py`、`test_workspace.py` 与 PR 270 一致，`_outcomes`、`INCORRECT_CODES`、`UNCHECKED_CODES`、
`_platform_failed` 是 PR 270 的原文。与 PR 270 不同的地方：

- driver `_calls`：PR 270 用 `OpVariants.load`、`schema.ordered_inputs`、`schema.get_output_shapes`；PR 3 用
  `registry.parse_schema`、`InputSampler(schema, axes).draw(0, 0, device)`、`ContractWorkload.assemble_call`，
  按 `schema.kernel_call_keys` 过滤，`contract.allocated_outputs` 用 `storage_shape_of` / `semantics.torch_dtypes`
  分配。
- driver `_aklr_timer` / `_measure_candidate`：PR 270 调 `MeasurementProfile().reading(...)`，但 PR 270 分支上既没有这个
  导出也没有 `reading` 方法 (它的 bench 模式在那边是坏的)；PR 3 保留 `_aklr_timer()`，返回
  `(BenchStage(devices=()).profile.graph_budget_ms, bench_cudagraph)`，直接读 `bench_cudagraph(call_once, rep=budget)`。
  `graph_reading` (tracer fault 重读) 两边都删了。
- `test_driver_contract.py`：计时器断言保留为 `budget, bench_cudagraph = driver._aklr_timer()`；读真实报告的用例用
  `BenchStage.report_type(...)` 加 `TypeAdapter(...).dump_python`，PR 270 用的 `BenchReport`、`Versus`、
  `WorkloadRecord` 在它自己的分支上也没有从 `aklr.core.validation` 导出。

## 六、文档

validation 行为的描述 (read-validation skill、`aklr/ops/README.md`、`docs/ops_tutorial*.md`、
`docs/acceptance_criteria*.md`、`aklr/benchmarks/README.md`、PR 模板、kb 的 triton / pitfalls 页) 按 PR 270 的文案改写，
模型相关的词保持 main 的 (contract、Point、`kernel.requires`、workload 上的 `speedup` / `accuracy` /
`microseconds`)。core PR 合入 schema 模型时这些文档按 PR 270 的版本再对一遍。kb 的 case 页是历史记录，没有动，
里面仍引用旧路径 `stages/_measure.py`、`_bench/_report.py`。

## finding code 的名字（review 后改）

- `FindingCode.NO_ADMITTED_WORKLOAD = "no_admitted_workload"`（#270）在本 PR 里改回 main 的
  `NO_QUALIFYING_WORKLOAD = "no_qualifying_workload"`：report 里的 code 是对外接口，这个 PR 不改名。
  涉及 `stages/report/_finding.py`、`_judge/_verdict.py`、两个 bench 测试、`harness/tests/forge_task/fake_aklr.py` 和文档。
- 内部的 `Opponent.admits()` / `WorkloadJudgement.is_admitting()` 保持 #270 的名字。core PR 若要改用 `no_admitted_workload`，在那里连同下游一起改。

## Copilot review 后的修正（head 之后的提交）

- `ContractWorkload.draw`：scalar 用 workload 自己的值，不再是默认值。
- `MeasurementProfile.num_trials`：必须大于 0。
- `MegaMoeBench.isolate`：恒为 True（`ep_size=1` 也是新进程，`setup` 注册 mori 进程组每进程只能一次）。#270 没有这个 override。
- `IncumbentPool.incumbent`：stored speedup 并列时取较大的 kernel hash，与 `DispatchTable.from_schemas` 的 `rank_key` 一致。#270 取较小的。
- `_execution/_runner.py::ProcessExecutionRunner._run_child`：子进程入口、`import torch` 之前 pop `ROCPROFILER_REGISTER_FORCE_LOAD` / `ROCPROFILER_REGISTER_LIBRARY`（origin/main 的 runner 在 spawn 前 unset，#220；#270 的 runner 没有）。torch ≥ 2.12 的 profiler 是 rocprofiler-sdk tool，父进程初始化 GPU 后 SDK 配置并导出变量；子进程继承后 `import torch` 时 SDK 就配置完，SDK < 1.4.0 之后不再收 tool，kernel tracer 注册失败。SDK ≥ 1.4.0 后可删。同时：`testing.py` 删掉 `_links_rocprofiler_sdk()` 和 `torch.cuda._lazy_call` 路径（只在 torch 2.11 上能用，且判别两个镜像都为 True），统一 import 时注册；`_kernel_tracer.cpp::attach()` 对 `CONFIGURATION_LOCKED` 报明确错误。core PR 需要带上这三处。 另外 `tests/conftest.py` 在 GPU 探测后 `os.unsetenv` 这两个变量（pytest 进程自己的导出会传给 `test_bench_cudagraph.py` 的 worker，torch ≥ 2.12 镜像上 main 也 9 个全挂）。
