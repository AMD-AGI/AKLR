# 算子定制：bench 里因算子而异的事放在哪里（设计）

> 下文路径是写这份设计时的。`_policy/` 现在是 `stages/bench/_execution_policy/`，`BenchExecutionPolicy`
> 改名为 `ExecutionPolicy`，发现方式不变（导入该包下每个不以 `_` 开头的模块）；现行结构见
> [bench_design.md](bench_design.md) 第 2 节。

> init、reference、check 和 baseline 后来移进了 schema 的 `Standard`，baseline 不再归 seam adapter，`standard_check` 已删除（默认
> check 是 `default_check`）；见 [schema_standard_design.md](schema_standard_design.md)。

状态：第 11 节的问题已全部定下，已实现。起因：multi_rank_design.md（已删除） 的方案 G 让 bench 按 `op_type`
过滤：`Batch.n_ranks` 对 mega_moe 取 `ep_size`，`Batch.run` 对 mega_moe 调用 `mega_moe_group_init`。review 指出，这样每加
一个算子都要改 bench，冲突大，也难读。只给组算子开口子的 `GroupOp` 方案也被否决了，因为因算子而异的不只有组算子。
术语见 [GLOSSARY.md](../GLOSSARY.md)；bench 现有结构见 [bench_design.md](bench_design.md)。

结论：

1. **bench 只在一处问"这个算子有什么不同"。** 每个 batch 带一个 `ExecutionPolicy`。它是 `aklr.core.common.Registrable` 的一个
   家族，按 `op_type` 注册。基类本身就是默认行为：1 个 rank、可复用的进程、不需要事先准备。哪个算子需要不同的行为，就在
   bench 自己的子包 `_policy/` 里加一个以 op_type 命名的文件，注册一个子类，只覆盖要变的成员。子类不放进 `aklr/ops`：
   它是 bench 怎么跑的事，不是算子本身的一部分。bench 里不再出现 `op_type == ...`。
2. **它只管"在哪里、在什么样的进程里跑"**，也就是 rank 数、要不要新进程、进程里先准备什么。**它不管"测什么、怎么判"**：
   输入的值、Reference、check、种子和计时方法仍然归 schema 和 `MeasurementProfile`，Baseline 仍然归 seam adapter。所以子类没法借它放宽对某个算子的判定。
3. **`BatchRun` 里的组代码不是算子定制。** 它对任何组都一样，组大小为 1 时就退化成单 rank，所以留在原处（第 3、4.2 节）。
4. 不改 schema、runtime、`_execution` 和 `aklr/core/common`。

## 1. 结构

只列新增或改动的 Python 文件，不列私有 helper。

| 文件 | 概念 | 类 | 公开方法/函数 |
|---|---|---|---|
| **bench：`aklr/core/validation/stages/bench/`** | | | |
| `_policy/_bench_execution_policy.py`（新） | 一个算子类型在 bench 里的执行策略 | `ExecutionPolicy` | `ExecutionPolicy.for_schema(schema)`、`ranks_per_task`、`isolate`、`setup()` |
| `_policy/__init__.py`（新） | policy 子包的接口 | - | 导出 `ExecutionPolicy` |
| `_policy/mega_moe.py`（新） | mega_moe 的执行策略 | `MegaMoeExecutionPolicy` | `ranks_per_task`、`setup` |
| `_task/_batch.py`（改） | 一个 batch | `Batch` | `run`、`cost`、`failed`、`results`、`summary`；新字段 `execution_policy: ExecutionPolicy`，删掉属性 `n_ranks` |
| `_schedule/_schedule.py`（改） | bench 的 schedule | `BatchSchedule` | `batches`（每个 plan 调一次 `ExecutionPolicy.for_schema`）、`runner`（看 `execution_policy.isolate`）、`timeout` |
| `_schedule/_pass.py`（改） | 一个 schedule pass | `SchedulePass`、`CostliestFirst`、`WidestFirst` | `apply`（`WidestFirst` 按 `execution_policy.ranks_per_task` 排序） |
| `_stage.py`（改） | bench stage | `BenchStage` | `run`（每批租 `batch.execution_policy.ranks_per_task` 块设备）、`plan` |
| `bench/__init__.py`（改） | bench 包的导出 | - | 加导出 `ExecutionPolicy` |
| `aklr/core/validation/stages/__init__.py`（改） | stages 包的导出 | - | 加导出 `ExecutionPolicy` |
| `aklr/core/validation/__init__.py`（改） | validation 的公开接口 | - | 加导出 `ExecutionPolicy`；测试从这里导入 |
| **op 包** | | | |
| `aklr/ops/mega_moe/group.py`（删） | - | - | `mega_moe_group_init` 的内容移进 `MegaMoeExecutionPolicy.setup` |
| `aklr/ops/mega_moe/schema.py`（改） | mega_moe 的 schema 模板 | - | `MEGA_MOE` 不再传 `check`，改用默认的 `standard_check`（已定），删掉自定义 check |
| **测试辅助** | | | |
| `tests/core/validation/toy_group.py`（新） | bench 测试用的玩具组算子 | `ToyGroupExecutionPolicy` | `ranks_per_task`；`toy_group_schema`：S3 (a)，公开，两个测试模块共用，没有哪个类拥有它 |
| `tests/op_case.py`（改） | 测试用的算子 case | `OpCase` | `ranks` 改为读 `ExecutionPolicy.for_schema(self.schema()).ranks_per_task` |

bench 里没有新的顶层函数。`ExecutionPolicy` 是扩展点，和 `ExecutionRunner`、`SchedulingStrategy` 一样基于 `Registrable`。以下不变：
`BatchRun`、`KernelCall`、`MeasurementProfile`、`_plan/`、`_verdict/`、`_report.py`、`_execution/`。

## 2. 层次

```
BenchStage.run                                  level 0（改 1 处）：每批租 batch.execution_policy.ranks_per_task 块设备
├── _plan/      BenchPlanner.plan               不变：R3.4 的表；Baseline 来自 seam
├── _schedule/  BatchSchedule.batches/runner/timeout level 1（改）
│               ├── batches                     每个 plan 调一次 ExecutionPolicy.for_schema(plan.schema)，结果放进它的每个 batch
│               ├── runner                      execution_policy.isolate 或 --isolate → disposable，否则 reuse
│               └── WidestFirst                 按 execution_policy.ranks_per_task 排序
├── _task/      Batch.run(context) -> JSON      level 1（改）：组大小必须等于 execution_policy.ranks_per_task；execution_policy.setup()（全组约定）→ build → 逐个 workload
│               ├── BatchRun                      不变：组机制直接用 torch.distributed，组大小为 1 时退化
│               └── KernelCall                    不变
├── _policy/    ExecutionPolicy（新）       level 1：bench 问"这个算子执行上有什么不同"的唯一入口；基类就是默认行为
│               └── mega_moe.py                 MegaMoeExecutionPolicy：ranks_per_task = ep_size；setup 建 mori 对称堆
├── _measure/   MeasurementProfile、WorkloadFailed 不变
└── _verdict/、_report、_execution/              不变

因算子而异的东西按归属分三处，每处有自己的注册方式
├── bench/_policy/<op_type>.py    执行上的不同：class X(ExecutionPolicy, name="<op_type>")
├── aklr/seam/*.py                Baseline：按 op_type 注册的 seam adapter（SeamRegistry），不变
└── schema 的 init/reference/check  语义：跟着 schema 走，不经过 ExecutionPolicy
```

bench 内部的导入顺序：`_measure`、`_execution` < `_policy` < `_plan` < `_task` < `_schedule` < `_verdict` < `_report` < `_stage`。
`_policy` 只依赖 `aklr.core.common`、`aklr.core.schema` 和 `_measure`。bench 不导入 `aklr.ops`，
`aklr/ops` 也不导入 validation。

## 3. 盘点：bench 里哪些地方因算子而异

范围是 `aklr/core/validation/stages/bench/` 的全部代码，以及它直接调用的地方。行号以 1c3033c5f 为准。

| # | 地方 | 位置 | 现状 | 由谁决定 | 新机制下 |
|---|---|---|---|---|---|
| **A** | **bench 按 `op_type` 写死** | | | | |
| 1 | 一个任务的 rank 数 | `_task/_batch.py:51` `Batch.n_ranks` | `op_type == "mega_moe"` 时取 `ep_size`，否则 1 | bench 写死 | `ExecutionPolicy.ranks_per_task`；`MegaMoeExecutionPolicy` 覆盖 |
| 2 | 进程里事先准备什么 | `_task/_batch.py:121-131` `Batch.run` | `op_type == "mega_moe"` 时，在全组约定里调用 `mega_moe_group_init`（建 mori 对称堆） | bench 写死 | `ExecutionPolicy.setup()`：对每个算子都调用，默认什么都不做；`MegaMoeExecutionPolicy` 覆盖 |
| 3 | 测试里的 rank 数 | `tests/op_case.py:142-144` `OpCase.ranks` | 写死，同第 1 项 | 测试写死 | `ExecutionPolicy.for_schema(schema).ranks_per_task` |
| 4 | 玩具组算子进入组路径 | `tests/core/validation/test_bench_task.py:41-45、127-130` | 用 `sitecustomize.py` 替换私有属性 `Batch.n_ranks` | 测试补丁 | 注册 `ToyGroupExecutionPolicy`（第 6 节） |
| **B** | **由第 1 项派生，或按 `op_type` 配置** | | | | |
| 5 | 设备租约 | `_stage.py:75` | `ResourceRequest(count=batch.ranks_per_task)` | 派生 | `batch.execution_policy.ranks_per_task` |
| 6 | 宽批次先启动 | `_schedule/_pass.py:37` `WidestFirst` | 按 `ranks_per_task` 排序 | 派生 | 按 `b.execution_policy.ranks_per_task` 排序 |
| 7 | runner | `_schedule/_schedule.py:69-72` | `op_type` 在 `--isolate` 里，或 `ranks_per_task > 1`，就用 `disposable`；docstring 里写着 mori 的"symmetric heap" | CLI 的 op_type 清单 + 派生 | `execution_policy.isolate`（默认 `ranks_per_task > 1`）或 `--isolate` |
| **C** | **通用规则：对任何算子都一样，组大小为 1 时退化** | | | | |
| 8 | 进程组 | `_stage.py` `BenchStage._join_process_group`（executor 的 initializer） | 建 `cpu:gloo,cuda:nccl` 组，单 rank 也建、不启动 RCCL | bench 通用 | 不变 |
| 9 | 每个 rank 的输入 | `_task/_batch_run.py:97-110` | 每个 trial 为每个 rank 调用一次 `init`，种子加上 `rank q`（rank 0 的不变）；只有 `init` 声明了 `rank` 才传它 | bench 通用（R4.1） | 不变；`init` 的 `rank` 没有使用者（Q5） |
| 10 | Reference 的参数 | `_task/_batch_run.py:111-112` | 本 rank 的输入按名字传；另有保留名 `group_calls`、`rank`，只传 Reference 声明了的 | 约定在 bench，用不用由 Reference 的签名决定 | 不变；Reference 的 `rank` 没有使用者（Q5） |
| 11 | 全组约定 | `failing_every_rank_as`：`_batch.py:121、132`，`_batch_run.py:97、119、171、173` | 组的准备、build、每个 trial 的两步、对手的绑定与首次调用 | bench 通用（R7.6、R5.7） | 不变 |
| 12 | 组计时 | `_measure/_profile.py:94-100` | `bench_cudagraph(group=…)`，每次调用取最慢的 rank | bench 通用（R5.2） | 不变 |
| **D** | **数据驱动，本来就不在 bench 里** | | | | |
| 13 | 输入的值 | schema 的 `init` 源码 | | schema | 不变 |
| 14 | 正确的输出 | schema 的 `reference` 源码 | | schema | 不变 |
| 15 | 比较 | schema 的 `check` 源码，默认是 `standard_check` | mega_moe 现在用自定义的 SQNR ≥ 20 dB | schema | 不变；mega_moe 改用默认（已定），不放宽 |
| 16 | 输出的形状和 dtype | schema 的 `outputs` | | schema | 不变 |
| 17 | build 和调用方式 | `KernelCall.build` → `BuilderRegistry`（按 language）；`KernelCall.bind` → `adapt_and_call`（dps） | | kernel 的元数据 | 不变 |
| 18 | Baseline：选哪个、怎么绑定 | `_plan/_planner.py:42-43`、`_plan/_opponent.py:86` | `SeamRegistry.adapters(op_type)` | seam 注册表（`aklr/seam/*.py`） | 不变，不并入 `ExecutionPolicy`（4.2） |
| 19 | batch 的代价 | `Batch.cost` | 按输入字节数估算 | schema | 不变 |
| **E** | **现在对所有算子都一样，将来可能因算子而异** | | | | |
| 20 | 任务预算 | `_schedule/_schedule.py:74-76` `BatchSchedule.timeout` | build 900 s，每个 workload 再加 300 s | bench 通用 | 暂不纳入；触发条件见 4.5 |
| 21 | 启动预算 | `BatchSchedule.startup_timeout_secs` | 300 s，整个 executor 只有一个值 | bench 通用 | 做不到按算子设：它是 executor 的参数，而 `_execution` 不改 |
| 22 | 建组前的进程环境 | `BenchStage._join_process_group` | 在任务到达之前运行，不知道是哪个算子 | - | 暂不纳入；见 4.5 |
| 23 | 输入放在哪种显存 | `init` | 所有输入都在普通显存里 | - | 暂不纳入；见 4.5 |
| 24 | 计时方法 | `bench_cudagraph` | | `MeasurementProfile`（R5.2） | 不纳入：它改的是语义，要走需求变更 |

bench 之外还有一处同类问题：`harness/scripts/forge_task/assets/aklr_forge_driver.py` 的 `_calls` 只按单 rank 生成输入。
在组算子上跑 forge 时它会出错，但这不在本设计的范围内。

小结：bench 里按 `op_type` 写死的只有第 1、2 项，测试里另有第 3、4 项。其余因算子而异的东西都已经有主人：语义归 schema，
Baseline 归 seam，build 和调用方式归 kernel。`BatchRun` 里为组加的代码（第 9–12 项）不按算子分支，而是对任何组都一样的
规则（R4.1、R4.2、R5.2、R5.7、R7.6）。它们是因为 mega_moe 才写的，但第二个组算子也会原样用到（5.2）。

## 4. 机制：`ExecutionPolicy`

### 4.1 接口

```python
class ExecutionPolicy(Registrable):
    """Where and in what processes the bench runs one schema's tasks; subclassed per op_type."""

    def __init__(self, schema: Schema) -> None:
        self.schema = schema

    @staticmethod
    def for_schema(schema: Schema) -> ExecutionPolicy:
        """The subclass registered under schema.op_type, else this default."""
        import importlib
        import pkgutil

        from aklr.core.validation.stages.bench import _policy

        for module in pkgutil.iter_modules(_policy.__path__):
            if not module.name.startswith("_"):
                importlib.import_module(f"{_policy.__name__}.{module.name}")
        try:
            kind = ExecutionPolicy.resolve(schema.op_type)
        except ValueError:  # no subclass registered under this op type
            kind = ExecutionPolicy
        return kind(schema)

    @property
    def ranks_per_task(self) -> int:
        """Ranks that run a task together, one device each."""
        return 1

    @property
    def isolate(self) -> bool:
        """Whether each task runs in new processes, like --isolate."""
        return self.ranks_per_task > 1

    def setup(self) -> None:
        """Prepare each rank's process before the build; idempotent unless isolate."""
```

`for_schema` 是 `staticmethod`，回退时显式用 `ExecutionPolicy`：即使有人在子类上调用 `for_schema`，也不会把那个子类当成别的算子的默认。

bench 里所有的调用点：

```python
# BatchSchedule.batches：每个 plan 调一次
execution_policy = ExecutionPolicy.for_schema(plan.schema)
batches += [Batch(plan.path, plan.schema_text, workloads, profile, execution_policy) for ...]

# BatchSchedule.runner
return "disposable" if batch.execution_policy.isolate or batch.schema.op_type in self.isolate else "reuse"

# WidestFirst.apply
return sorted(batches, key=lambda b: b.execution_policy.ranks_per_task, reverse=True)

# BenchStage.run
resource_request=ResourceRequest(count=batch.execution_policy.ranks_per_task)

# Batch.run：替换现在 mega_moe 的分支
if dist.get_world_size() != self.execution_policy.ranks_per_task:
    raise RuntimeError(f"{self.schema_path} runs on {self.execution_policy.ranks_per_task} ranks, its task on {dist.get_world_size()}")
with failing_every_rank_as(FindingCode.PLATFORM_FAULT, "the op's setup"):  # no kernel ran yet
    self.execution_policy.setup()
```

### 4.2 边界：能改什么，不能改什么

- **`isolate` 和 CLI 的 `--isolate` 是同一个开关的两个入口**：`BatchSchedule.runner` 在
  `execution_policy.isolate or op_type in --isolate` 时用 disposable runner。策略里声明的是算子已知的需要（组默认为 True）；
  CLI 的 `--isolate` 留给操作者临时排查。两者都只决定要不要新进程，不影响测什么。
- **`setup` 在复用进程里的规则**：`isolate` 为 False 时，同一个进程会先后跑多个任务，每个任务都会调一次 `setup`。所以
  `setup` 必须幂等，也不能留下影响其他算子的进程状态；做不到就声明 `isolate = True`。

- **能改**：任务在哪里跑、跑在什么样的进程里，也就是 rank 数、要不要新进程、进程里先准备什么。这些只决定"能不能跑起来、
  跑多快"，和 bench_design 里 `BatchSchedule` 那一类改动同属一侧。
- **不能改**：输入的值、Reference、check、种子、计时方法和阈值。它们是语义：schema 和 flashinfer、flashinfer-bench 共用，
  同一个 schema 在哪个 bench 上都应得出同一个结论；计时协议由 `MeasurementProfile.digest` 命名。如果执行策略能改这些，它就能
  让某个 kernel 过关，比如放宽 check，而这正是 mega_moe 的自定义 check 要删掉的原因。
- **备选方案 B（否决）：把输入生成和 Reference 调用也做成 `ExecutionPolicy` 的方法。** 那样 `ExecutionPolicy` 的默认实现就是今天
  `BatchRun.check` 里的同一段代码，只是换了个地方，读代码时要跳两处；而且它会开一扇改语义的门。第 9、10 项的组规则
  对 mega_moe、玩具组算子和 5.2 的 gemm_allreduce 完全一样，没有算子需要不同的写法。
- **Baseline 不并入。** seam adapter 在运行时也要用（它 patch 后端的入口），已经按 `op_type` 注册在 `SeamRegistry`，
  bench 只读取它。把它并进 `ExecutionPolicy`，等于把一个概念拆到两处。
- **`setup` 的规则**：只建立 kernel 需要的进程状态，比如通信、库的初始化；不改全局数值设置，比如 TF32，否则 Reference
  也会跟着变。这一条写进 docstring，靠 review 把关（C3：Candidate 不当作对手）。

### 4.3 注册、查找，以及怎么进入 rank 进程

- **放在哪里**：bench 的子包 `aklr/core/validation/stages/bench/_policy/`，每个需要定制的 op_type 一个文件，文件名就是
  op_type（`mega_moe.py`）。不放进 `aklr/ops`：执行策略是 bench 怎么跑的事，不是算子的一部分；serving 也不会碰到它。
  每个算子一个文件，加算子只新增文件，不改共享文件，不会和别人冲突。
- **注册在什么时候发生**：在 `_policy/<op_type>.py` 被导入的时候。class 语句带着 `name="<op_type>"`，
  `Registrable.__init_subclass__` 把这个类登记进 `ExecutionPolicy` 家族的 `ClassRegistry`。两个类注册同一个 op_type，
  导入时就抛 `ValueError`。
- **bench 怎么找到它**：`ExecutionPolicy.for_schema(schema)` 先导入 `_policy/` 下每个不以 `_` 开头的模块，再调用
  `ExecutionPolicy.resolve(schema.op_type)`；没注册的 op_type 得到基类。扫描的是 bench 自己的子包，不需要中心清单。
- **在哪个进程查找**：只在 validate 进程里查，由 `BatchSchedule.batches` 每个 plan 调一次。查到的实例作为
  `Batch.execution_policy` 字段，和 `Opponent` 一样作为"计划好的数据"随任务 pickle 进 rank 进程。rank 进程 unpickle 时按类的
  模块路径导入它，注册随之发生，但 rank 进程不查表。好处：rank 进程里不做发现；测试注册的策略（不在 `_policy/` 里）也能
  到达 rank 进程；同一个 batch 的所有 rank 拿到的一定是同一个策略。前提是子类定义在可导入模块的顶层。
- **导入方向**：全在 bench 内部（见第 2 节），不涉及 `aklr.ops`。重的库（mori 等）只在 `setup` 里导入。
- **注册是进程级的**：测试注册的类在整个 pytest 会话里都有效，所以玩具算子要用不会和真实算子重名的 op_type。

### 4.4 默认行为就是现在的单 rank 行为，只有一条路径

对一个没有定制的算子：`ExecutionPolicy.for_schema` 返回基类；`ranks_per_task` 是 1，所以租 1 块设备，进程组的大小也是 1。这时 `failing_every_rank_as`
原样抛出本 rank 的失败，`bench_cudagraph` 走非分布式分支。`setup` 照样在约定里被调用，但什么都不做；
`isolate` 是 False，所以用 reuse runner（除非 `--isolate`）；`BatchRun` 和今天一样。mega_moe 走的是同一段代码，
只是这几个值不同。bench 里没有任何 `if op_type`，也没有"有定制"和"没定制"两条路。

### 4.5 怎样加一种新的定制

规则：一种新的定制就是给 `ExecutionPolicy` 加一个成员，它的默认值要复现今天的行为，再在 bench 里加一个调用点。bench 只为每一
**种**定制改一次；之后任何算子覆盖这个成员，都不用改 bench。下面这些现在没有算子需要，所以不加：

| 定制 | 形状 | 触发条件 |
|---|---|---|
| 任务预算 | `ExecutionPolicy` 给出每个 workload 的预算，默认用 `BatchSchedule` 的；`BatchSchedule.timeout` 读它 | mega_moe 阶段 3 实测超过 300 s/workload，或某个算子的 Reference 特别慢 |
| 输入放在哪 | 默认原样返回的 `place(call)`；`BatchRun` 在调用 Candidate 前用它 | 出现要求输入在对称堆或注册缓冲里的 kernel |
| 建组前的环境 | 把 `BenchStage._join_process_group` 从 executor 的 initializer 挪进 `Batch.run`，排在 `setup` 之前，由 `ExecutionPolicy` 给出建组前的环境变量 | RCCL 或 mori 需要在建组之前设置的变量 |

## 5. 走一遍

### 5.1 mega_moe（`_policy/mega_moe.py`）

```python
"""How the bench runs mega_moe."""

from __future__ import annotations

from aklr.core.validation.stages.bench._execution_policy import ExecutionPolicy


class MegaMoeExecutionPolicy(ExecutionPolicy, name="mega_moe"):
    @property
    def ranks_per_task(self) -> int:
        """ep_size ranks."""
        return self.schema.const_axes["ep_size"]

    def setup(self) -> None:
        """mori's symmetric heap over the group, where the kernels allocate their buffers."""
        import os

        # Baseline and kernel buffers together exceed mori's 4 GiB default
        os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "16G")
        import mori
        import torch
        import torch.distributed as dist

        torch._C._distributed_c10d._register_process_group("mori", dist.group.WORLD)
        mori.shmem.shmem_torch_process_group_init("mori")
```

`isolate` 用默认值：ep8 时为 True，也就是 R7.2 和 Q3 定下的规则。

| bench 的决定 | 由谁给出 | mega_moe ep8 |
|---|---|---|
| 租几块设备，WidestFirst 怎么排 | `execution_policy.ranks_per_task` | 8 |
| runner | `execution_policy.isolate` | disposable |
| 进程里先准备什么 | `execution_policy.setup()` | mori 对称堆，把 WORLD 注册为 "mori" |
| 输入 | schema 的 `init`，每个 rank 调一次（通用规则） | 每个 rank 有自己的 token 和 expert 权重 |
| Reference | schema 的 `reference(…, group_calls)` | 按 owner rank 对 fused_moe 求和 |
| check | schema 的默认 `standard_check` | 不放宽 |
| Baseline（阶段 4） | seam adapter `MegaMoEV2.forward` | 在 `setup` 之后才绑定，此时对称堆已经建好 |
| 计时 | 通用的组计时 | 取最慢的 rank |

和现在比：删掉 `group.py`，也删掉 `Batch` 里的两处 `op_type == "mega_moe"`。行为不变：mori 注册的就是 WORLD。

### 5.2 假想的组算子：融合 all-reduce 的 GEMM

- schema：`op_type="gemm_allreduce"`；Var 轴 `M`；Const 轴 `tp_size`（abbrev `tp`，即组大小）、`N`、`K`（每个 rank 的
  K 切片）；输入 `a [M, K]`、`b [N, K]`；输出 `out [M, N] = Σ_r a_r b_rᵀ`，每个 rank 上都一样。
- `init`：和普通单 rank 的写法一样，每个 rank 的种子不同，切片也就不同。`reference(a, b, group_calls)` 返回
  `sum(c["a"].double() @ c["b"].double().T for c in group_calls)`。check 用默认。
- `_policy/gemm_allreduce.py`：

  ```python
  class GemmAllreduceExecutionPolicy(ExecutionPolicy, name="gemm_allreduce"):
      @property
      def ranks_per_task(self) -> int:
          return self.schema.const_axes["tp_size"]
  ```

  如果它的 kernel 像 custom all-reduce 那样，要在每个进程里注册一次 IPC 缓冲，就再覆盖 `setup`；`isolate` 默认为
  True。
- bench 的改动：无。逐项看：租 tp 块设备 → disposable → `setup` → 每个 rank 一份输入 → Reference 读 `group_calls` →
  默认 check → Baseline 来自它的 seam adapter → 组计时，每一步都是第 3 节已有的。

### 5.3 假想的单 rank 算子：需要先准备进程

算子 `x` 只用一个 rank。它的 kernel 来自一个库：第一次 import 之前要设一个进程级的环境变量（比如 workspace 大小），而且
它会留下进程级的状态，后面的任务不该继承。今天要支持它，得在 bench 里再加一个 `op_type == "x"` 分支来做准备，CI 也得每次
记得传 `--isolate x`。有了 `ExecutionPolicy`：

```python
class XBench(ExecutionPolicy, name="x"):
    @property
    def isolate(self) -> bool:
        return True  # the library's process state outlives a task

    def setup(self) -> None:
        import os

        os.environ.setdefault("X_WORKSPACE_MB", "512")  # read once, at the library's first import
```

`ranks_per_task` 用默认值 1。bench 不用改，也不用再为它传 `--isolate`。

"特殊的输入准备"分两种情况：

- **改的是值**：预先打乱的权重、特定分布的长度、合法的索引等。这些写在 schema 的 `init` 里，比如 fused_moe 的 shuffled
  权重今天就是这样生成的，bench 不用改。它们属于语义，和 flashinfer-bench 共用，不能放进 `ExecutionPolicy`。
- **改的是位置**：kernel 要求输入放在对称堆或注册过的缓冲里，值本身不变，属于执行环境。现在还没有这样的 kernel。出现时，
  按 4.5 给 `ExecutionPolicy` 加一个成员。

### 5.4 够用，也不多余

| 成员 | 默认 | 谁覆盖 |
|---|---|---|
| `ranks_per_task` | 1 | mega_moe、玩具组算子、gemm_allreduce |
| `setup` | 什么都不做 | mega_moe、x（gemm_allreduce 视 kernel 而定） |
| `isolate` | `ranks_per_task > 1` | x |

这三个例子都不需要更多的成员，所以够用。`ranks_per_task` 和 `setup` 今天就有使用者。`isolate` 今天还没有覆盖者，它的
默认值就是 R7.2 的规则；把它放进 `ExecutionPolicy`，是因为组要用新进程的原因在算子本身（mori 的堆每个进程只能建一次，玩具组算子
就不需要新进程），而且这样单 rank 中会污染进程的算子就不必依赖 CI 传 `--isolate`。预算、输入位置这些推测性的定制都没有加。

## 6. 阶段 2 的玩具组算子测试

- `tests/core/validation/toy_group.py` 定义 `class ToyGroupExecutionPolicy(ExecutionPolicy, name="toy_group")`，其中
  `ranks_per_task = schema.const_axes["world"]`。玩具 schema 的 `init`、`reference`、`check` 和构造函数从 `test_bench_task.py`
  移到这里，由两个测试模块共用。
- CPU 测试（`test_bench_plan.py`）：导入 `toy_group` 就完成了注册。对玩具 plan 调 `BatchSchedule.batches`，期望 `batch.execution_policy` 是
  `ToyGroupExecutionPolicy`，`ranks_per_task` 等于 `world`，runner 是 disposable，WidestFirst 的顺序不变。另外检查默认情况：对 gemm，
  `ExecutionPolicy.for_schema` 得到基类，`ranks_per_task` 为 1，runner 为 reuse。mega_moe 自己的测试移到 `tests/ops/test_mega_moe.py`（CPU）：
  `ExecutionPolicy.for_schema(mega_moe 的 schema)` 应得到 `MegaMoeExecutionPolicy`，`ranks_per_task == ep_size`，`isolate` 为 True。
- GPU 测试（`test_bench_task.py::TestGroup`）：仍经 `python -m aklr validate` 子进程跑，和模块里其他测试一样。
  删掉 `_N_RANKS_FROM_WORLD`，`sitecustomize.py` 只留一行 `import tests.core.validation.toy_group`：走公开的扩展点，不再
  替换私有属性。这样 2 卡机器上也有一个从 CLI 入口走到组路径的测试。
- 约束：`toy_group.py` 在 import 时不能碰 GPU。`sitecustomize` 在 rank 进程设置设备之前执行，import 时调用 GPU 会让
  kernel tracer 挂不上；要用设备的代码只放在构造 schema 的函数里。

## 7. `BatchRun.time` 的 `bind` 参数（32d1ac9a4）

- **目的仍然成立，而且和 `ExecutionPolicy` 无关。** 绑定对手（Baseline adapter，或者 Incumbent 的 build）可能只在一个 rank 上
  失败。如果不先约定，其他 rank 会进入 `opponent()` 或 `bench_cudagraph` 的集合通信，然后一直卡住。这属于组机制（R5.7、
  R7.6）。新机制不改变这一点：`ExecutionPolicy` 不碰对手；mega_moe 的 `setup` 在 build 之前就完成了，所以阶段 4 绑定 MegaMoEV2
  时，对称堆已经建好。
- **把约定放在 `BatchRun.time` 里是对的。** 它是模板方法，任何 `Opponent` 子类都不会漏掉这一步约定。
- **不改成 `Opponent.bind` 方法（Q4 已定）。** 那样 `Unopposed` 要实现一个永远不会被调用的 `bind`，两个子类的名字字段
  （`backend`、`path`）还要再抽一个属性，代码总量反而变多；现在的闭包已经让所有绑定都经过 `time` 里的约定。

## 8. 受保护代码

不需要改。`ExecutionPolicy` 在 bench 里；schema 不加字段，mega_moe 的 check 用的是 `Schema.check` 字段已有的默认值；runtime 的
`adapt_and_call`、`SeamRegistry`、`BuilderRegistry` 都不动；`aklr/core/common` 不动，"未注册就用默认"由 `ExecutionPolicy.for_schema` 捕获
`resolve` 的 `ValueError` 来实现。multi_rank_design.md 里阶段 4 的 P3（seam 支持 `Class.method`）和本设计无关，保持原样。

## 9. 落地

每一步都先写测试（`tdd` skill，`tests/AGENTS.md`），并先删旧代码再写新代码。

1. **红**：按第 6 节写 CPU 测试，包括 `ExecutionPolicy` 的默认行为、玩具组算子的 batch、runner、WidestFirst、租约宽度，以及
   mega_moe 的 `ExecutionPolicy`；`OpCase.ranks` 改走 `ExecutionPolicy`；GPU 的玩具组测试只把 sitecustomize 改成一行 import。
0. **单独一个 commit：删保留名 `rank`**（Q5）：`_batch_run.py` 里给 `init` 传 `rank` 的一行和给 Reference 的 `rank` 键，
   `tests/op_case.py` 里对应的键；`multi_rank_design.md` 6.3 和 `bench_requirements.md` 相应的说法。以后有组算子需要时再加，
   改名 `group_rank`。
2. **删**：`Batch.n_ranks`；`Batch.run` 里的 mega_moe 分支；`aklr/ops/mega_moe/group.py`；测试里的 `_N_RANKS_FROM_WORLD`
   和 sitecustomize 里替换私有属性的代码；`OpCase.ranks` 的 op_type 判断；mega_moe 的自定义 check。
3. **写**：`_policy/`（`_bench_execution_policy.py`、`__init__.py`、`mega_moe.py`）；`Batch.execution_policy`；让
   `BatchSchedule.batches`、`BatchSchedule.runner`、`WidestFirst`、`BenchStage.run` 读 `execution_policy`；三处导出；
   `tests/core/validation/toy_group.py`。
4. 重新生成 mega_moe 3 个 kernel JSON 里的 check 源码。
5. **文档**：
   - `bench_requirements.md`：范围那一行的"today only `op_type == "mega_moe"`"改为"an op whose ExecutionPolicy gives more than
     one rank"；R7.2 改为"an op whose ExecutionPolicy isolates it (by default every group), or one named by --isolate"；
   - `bench_design.md`：层次树和结构表加上 `ExecutionPolicy`；
   - `multi_rank_design.md`：第 1、2、4.5、6.1 节里按算子过滤的说法改为 `ExecutionPolicy`；本次只加了一句指向本文；
   - `aklr/ops/README.md`：删掉 `group.py` 的约定，指向 `_policy/`（这份 README 还停留在 contract 时代的写法）；
   - `ops_tutorial.md`：加一节"需要定制 bench 的算子"。
6. **验证**：CPU 全量测试；2 卡 GPU 上的玩具组测试；8×MI355X 上的 `tests/ops/test_mega_moe.py`，期望不变，仍是 19 个
   passed（如果在 `standard_check` 下不过，那是 kernel 或 Reference 的问题，不能放宽）；单 rank 回归测试，结论应与改动前
   相同。

## 10. 风险

| # | 风险 | 后果 | 缓解 |
|---|---|---|---|
| R1 | `_policy/` 里某个模块导入失败，或两个类注册了同一个 op_type | 每次 validate 都在分批时失败 | 任何调用 `ExecutionPolicy.for_schema` 的 CPU 测试都会导入整个 `_policy/`；模块顶层不导入重的库，放到 `setup` 里 |
| R2 | `setup` 改了数值环境，比如 TF32 | Reference 跟着变，结论失真 | docstring 写明；靠 review 把关 |
| R3 | 子类无法按模块路径导入（嵌套类、定义在 `__main__` 里） | rank 进程 unpickle 失败，整批记为 `platform_fault` | 约定子类必须是可导入模块的顶层类（注册本来就要靠 class 语句） |
| R4 | `toy_group.py` 在 import 时碰了 GPU | sitecustomize 早于设备设置执行，kernel tracer 挂不上，计时全错 | 设备相关代码只放在构造 schema 的函数里（第 6 节） |
| R5 | mega_moe 在 `standard_check` 下不通过 | 19 个 workload 被记为 `outputs_mismatch` | 已定不放宽；去查 kernel 和 Reference 的舍入点 |

## 11. 需要用户决定的问题

- **Q1 已定**：子类放在 bench 的 `_policy/<op_type>.py`，不放进 `aklr/ops`；bench 扫描 `_policy/` 完成注册。
- **Q2 已定**：保留 CLI 的 `--isolate`；策略上的 `fresh_processes` 改名为 `isolate`，两者任一为真就用 disposable runner。
- **Q3 已定**：GPU 组测试保留 `python -m aklr validate` 子进程，sitecustomize 只留一行 import。
- **Q4 已定**：`bind` 闭包不改。
- **Q5 已定**：删掉没有使用者的保留名 `rank`，单独一个 commit。
- **Q6 已定**：基类 `ExecutionPolicy`，`for_schema`，`ranks_per_task`，`Batch` 上的字段 `execution_policy`。
