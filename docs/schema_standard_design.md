# Schema 的测试标准（Standard）：init、reference、check、baseline 只在代码里（设计）

状态：已批准（含第 8 节的受保护代码），已实现，待 review。实现与设计的出入见第 12 节。
术语见 [GLOSSARY.md](../GLOSSARY.md)；bench 现有结构见 [bench_design.md](bench_design.md)。

起因：

1. schema JSON 里存着 init、reference、check 的独立源码，uuid 也由它们算出。代码一改，所有相关 JSON 都要重新生成：
   换成 flashinfer 标准 check 的那次改动（已并入 3ae39bae0）重写了 25 个 JSON。
2. baseline 挂在 seam adapter 上（`Adapter.baseline_fn(original, **call)`），和 adapter 本来的职责（把 backend 的调用绑到
   AKLR api）混在一起；要知道一个 schema 拿什么当 Baseline，得去 seam 里按 op_type 找。
3. tracing dump 只有 JSON。用户合入代码、agent 优化 kernel 时，拿不到一份能独立运行的测试标准。

结论：

1. **一个定义，两部分。** Schema 在代码里是一个聚合体：它的数据（axes、inputs、outputs、kernel、workloads 等）加上它的
   测试标准 `Standard`（init 生成输入，reference 和 check 定义精度，baseline 定义性能）。定义 schema 时两部分一起写。
   JSON 只存数据，`standard` 不进 JSON。合入仓库的只有 JSON。
2. **四个函数同一种写法。** 都是普通函数，用 schema 的名字作关键字参数调用，按同一种方式渲染成独立源码。baseline 和
   reference 一样是一个函数，自己导入并调用 backend；只是它返回待计时的调用，而不是输出（第 3 节）。
3. **只有 tracing 导出和 bench 用 Standard。** dispatch、seam 绑定、dataset 读写都不碰它。
4. **读 JSON 时自动补上 Standard，靠名字。** 从 JSON 读出一个 schema 时，按 dispatch 的命名规则找到代码里同名的 schema
   模板，取它的 standard；没有模板对应的照常读入，standard 为空。代码里构造的 schema 不受影响。
5. **函数在代码里就是普通 callable。** bench 直接调用；只有 tracing 导出时才渲染成独立源码。从源码加载函数的
   `source_to_function`、`FunctionSource.load/validate` 一并删除。
6. **不留兼容层。** Schema 删掉三个字段；它是 `extra="forbid"`，旧 JSON 会读不进，所以 223 个已提交 JSON 在同一个提交里迁移。

## 1. 结构

只列新增或改动的 Python 文件，不列私有 helper。🔒 是受保护代码。

| 文件 | 概念 | 类 | 公开方法/函数 |
|---|---|---|---|
| **`aklr/core/schema/`（🔒）** | | | |
| `_standard.py`（新） | 一个 schema 的测试标准 | `Standard` | 字段 `reference`、`init`、`check`、`baseline`；`save(directory)` |
| `_schema.py`（改） | 一个 schema | `Schema` | 新字段 `standard`（不进 JSON，读 JSON 时从模板补上）；`templates()`、`declare_template(template)`、`template()`；删字段 `reference`、`check`、`init`，删 `source_to_function` 和两个源码校验器 |
| `_function_source.py`（改） | 函数的独立源码 | `FunctionSource` | `render`、`dependencies`；删 `validate`、`load`、`FUNCTION_SOURCES` |
| `_check.py`（改） | 默认的精度检查 | - | `default_check`、`default_tolerances`（S3(a)：各 op 的 check 复用，没有类是它们的归属）；删纯转发的 `standard_check` |
| `__init__.py`（改） | schema 包的公开接口 | - | 加导出 `Standard`，删 `standard_check` |
| **`aklr/core/runtime/`（🔒）** | | | |
| `seam/_adapter.py`（改） | backend 入口到 AKLR api 的绑定 | `Adapter` | 删 `baseline_fn`、`baseline()`；`original()` 留着，`PatchManager` 在用 |
| `tracing/_runtime.py`（改） | tracing 运行时 | `TracingRuntime` | `flush` 写出 `standards/<name>/` |
| **`aklr/core/validation/`** | | | |
| `stages/bench/_admission/_plan.py`（改） | bench 的计划 | `SchemaPlan`、`BenchPlanner` | 有 `standard.baseline` 才有 Baseline，名字取服务该 op 的 seam adapter 的 `backend` |
| `stages/bench/_admission/_opponent.py`（改） | workload 的计时对手 | `BaselineOpponent` | `bind` 调 `schema.standard.baseline(**call)`，不再遍历 `SeamRegistry` |
| `stages/bench/_workload_call/_init_sampler.py`（改） | 由 init 抽取的输入 | `InitSampler` | `draw` 用 `schema.standard.init` |
| `stages/bench/_workload_call/_reference_call.py`（改） | 一个 workload 的期望输出 | `ReferenceCall` | 用 `schema.standard.reference` |
| `stages/bench/_workload_call/_output_check.py`（改） | 对照期望输出的产出检查 | `OutputCheck` | 用 `schema.standard.check` |
| `stages/audit/_stage.py`（改） | audit stage | `AuditStage` | `run` 逐个 JSON 对照模板（第 5 节） |
| **`aklr/seam/`** | | | |
| `__init__.py`（改） | seam adapter 的声明入口 | - | `adapter(entry, api)`，去掉 `baseline` 参数 |
| `aiter.py`（改） | aiter 的 adapter | - | baseline 函数搬到各 op 的 `baseline.py` |
| `sglang.py`（改） | sglang 的 adapter | - | baseline 函数搬到各 op 的 `baseline.py` |
| **`aklr/ops/`** | | | |
| `<op>/schema.py` ×11（改） | 该 op 的 schema 模板 | - | init、reference、check、baseline 收进 `standard=Standard(...)` |
| `<op>/baseline.py` ×10（新） | 该 op 的 Baseline 调用 | - | 每个模板一个 `<name>_baseline(**call)`（S3(a)：由 `schema.py` 导入） |
| **harness** | | | |

没有 baseline 的是 mega_moe，所以 `baseline.py` 是 10 个，不是 11 个。

## 2. 分层

```
level 0  使用方（只有这两处用 Standard）
  tracing  TracingRuntime.flush ─► <dump>/schemas/<op_type>/<name>/<uuid>.json      数据
                                └► <dump>/standards/<name>/*.py                     测试标准，独立源码
  bench    读 JSON（standard 自动补上）─► schema.standard.{init, reference, check, baseline}
  audit    读 JSON ─ schema.template() ─► 模板存在、有 init 与 reference、签名一致

level 1  核心类型（aklr/core/schema）
  Schema ──字段 standard──► Standard ── save 用 ──► FunctionSource.render
                             └ check 默认 default_check

level 2  每个 op 的定义（aklr/ops/<op>/）
  schema.py     Schema(..., standard=Standard(init=..., reference=..., check=..., baseline=...))
  reference.py  reference 函数本体（不变）
  baseline.py   baseline 函数本体（从 aklr/seam 搬来，改成自己导入 backend）
```

导入方向：

- `aklr/core/schema` 不导入 runtime、validation、ops。
- `Schema.templates()` 第一次调用时延迟导入 `aklr.ops`：底层的 core/schema 由此在运行时依赖 ops，这是读 JSON 时自动补上
  standard 的代价（第 4 节）。
- `aklr/ops/<op>/baseline.py` 不能导入 `aklr.seam`：seam 导入 ops，反过来会成环。bind 和 baseline 共用的 helper
  （目前只有 `_aiter_moe_act_quant`）随 baseline 搬到 `aklr/ops/fused_moe/baseline.py`，seam 从那里导入。
- baseline 在函数体里导入 backend，`aklr.ops` 被导入时不碰 aiter、sglang。

## 3. Standard

```python
@frozen_dataclass
class Standard:
    """A schema's test standard: the inputs it draws, the outputs it must match, the time it must beat."""

    reference: Callable[..., Any]
    init: Optional[Callable[..., Dict[str, Any]]] = None
    check: Callable[..., bool] = default_check
    baseline: Optional[Callable[..., Optional[Callable[[], Any]]]] = None
```

Schema 上的字段：

```python
standard: Optional[Standard] = Field(default=None, exclude=True, repr=False)
```

- `exclude=True`：standard 不进 JSON，因而也不进 uuid，不进 `SchemaCollection` 的同名一致性比较。`for_call` 用
  `replace`，模板的 standard 原样带到每个具体 schema。
- `save(directory)`：tracing 导出用，见第 6 节。Standard 没有别的方法，bench 直接调它的四个函数。

四个函数：

| 函数 | 调用 | 返回 | 可空 |
|---|---|---|---|
| init | `init(**axes, device, seed)` | 一组输入，按 schema 输入名 | 可以；现有 37 个模板中 11 个没有，它们带 kernel 时 audit 会拦 |
| reference | `reference(**inputs)`，声明了 `group_calls` 时另给全组输入 | 期望输出 | 不可以；现有模板都有 |
| check | `check(reference_outputs, actual_outputs)` | bool | 默认 `default_check` |
| baseline | `baseline(**inputs, **outputs)` | 待计时的无参调用；backend 不支持这个调用时返回 None | 可以；没有就没有 Baseline |

baseline 和 reference 只差在返回值：它返回"待计时的调用"，而不是输出。原因有二，都来自现有的 22 个 baseline 函数：

- 11 个在计时前要做准备（fused_moe 有 9 条语句，paged_attention_v1 有 6 条，例如权重重排、metadata 规划），这些不能算进
  Baseline 的时间；
- 11 个会对它不支持的调用返回 None。

定义示例：

```python
# aklr/ops/mhc/baseline.py
def mhc_pre_baseline(residual, proj_weight, mix_scale, mix_bias, *, rms_eps=1e-6, ..., **_):
    from aiter.ops.mhc import mhc_pre

    return lambda: mhc_pre(residual, proj_weight, ...)


# aklr/ops/mhc/schema.py
MHC_PRE = Schema(
    op_type="mhc_pre",
    axes={...},
    inputs={...},
    outputs={...},
    standard=Standard(reference=mhc_pre_reference, check=_mhc_check, baseline=mhc_pre_baseline),
)
```

Baseline 的名字（speedup 记在它下面）不由 Standard 给出，而是取服务该 op 的 seam adapter 的 `backend`：dispatch 判断
一个 workload 能否服务时查的就是这个名字，两边自然一致。已查：没有哪个 op_type 的 adapter 跨两个 backend。

## 4. 读 JSON 时补上 Standard：Schema 的模板

模板就是 op api 声明的 Schema，即 `aklr_api(schema)` 的 schema，或 resolver 的 `.schemas`。Schema 自己记着它们（类变量，
按名字），没有另设注册表：

- `Schema.templates()`：第一次调用时导入每个 `aklr.ops.<pkg>.api`，把模块里所有带 `.schemas` 属性的对象的模板记下来。
  它取代 `tests/ops/test_op_schemas.py` 里手写的 `_TEMPLATES`；那个列表要手动补上不在 `aklr.ops.__all__` 里的 api。
- `Schema.declare_template(template)`：同一个对象重复声明时忽略；不同对象同名时抛 `ValueError`。这把原来的测试
  `test_names_are_unique` 变成运行时保证。测试的 toy op 用它声明自己的模板。
- `schema.template()`：返回 op_type 相同、且 `t.schema_name(schema.const_axes) == schema.name` 的模板，没有就返回 None。
  这就是 dispatch 的命名规则。
- 一个 model validator：只在从 JSON 读时（pydantic 的 `info.mode == "json"`，即 `validate_json`、`load_json_file`）且
  `standard` 为空时，补上 `template().standard`。代码里构造、`replace`、`for_call` 都是 python 模式，不受影响。

找不到模板的 JSON 照常读入，standard 为空：dispatch、tracing 读数据集不受影响；audit 报 SCHEMA_INVALID；bench 万一
碰到，`InitSampler.draw` 报"没有 init"。代价：所有读 JSON 的地方都会查一次模板，第一次会导入全部 ops。

多进程：bench 的 worker 用 spawn 启动，`Batch.schema` 在 worker 里读 JSON 时就补上 standard，模板也在 worker 里导入，
函数不需要 pickle。toy op 在 `tests/core/validation/toy_group.py` 导入时声明模板；worker 反序列化
`ToyGroupExecutionPolicy` 时会导入这个模块（实现时确认）。

## 5. bench 与 audit

解析 JSON 的地方：

| 位置 | 改动 |
|---|---|
| `SchemaPlan.schema`、`Batch.schema`、`BenchPlanner.plan`、forge driver | 不变：读 JSON 时 standard 已自动补上 |
| `IncumbentPool`、`BatchRun.incumbent`、`PipelineReport.measured_schemas` | 不变：同样会补上，但用不到 |

用到函数的地方：

- `InitSampler.draw`、`ReferenceCall`、`OutputCheck` 分别读 `schema.standard` 的 init、reference、check。
- `BaselineOpponent.bind` 调 `schema.standard.baseline(**call)`，`call` 带上输出 buffer，和现在一样由它分配。返回 None
  记 BASELINE_UNBOUND（同现在）；抛异常记 PLATFORM_FAULT（同现在）。backend 没装时，baseline 里的导入会失败，于是记
  PLATFORM_FAULT；现在记的是 BASELINE_UNBOUND（算 schema 的错）。没装 backend 是环境的问题，改后更准确。
- `BenchPlanner.plan`：`schema.standard.baseline` 为 None 就没有 Baseline；否则 Baseline 名取
  `SeamRegistry.default().adapters(op_type)` 的 `backend`。删掉 `_plan.py` 里按 `a.baseline_fn is not None` 挑选的那条规则。

### 每个 workload 的计时对手

规则不变，`SchemaPlan.from_schema` 仍是唯一决定对手的地方。只换"有没有 Baseline"的来源：现在是"op 有带
`baseline_fn` 的 seam adapter"，改后是"`schema.standard.baseline` 不为 None"，名字仍取 seam adapter 的 backend。

先找 Incumbent（`IncumbentPool.incumbent`）。没有 base commit 时没有 Incumbent。

- 范围：base commit 里 `aklr/ops/<op>/impls/*/<同名>/` 下的 schema。要求能按当前格式解析、带 kernel，并且有一个 workload
  和本 workload 的 dispatch key 相同；axes 不必完全相同。
- 资格：满足其一即可。
  - kernel 和 Candidate 相同（"自己"）。
  - base 时 dispatch 会服务它：`served.qualifies(target, 有 Baseline ? AKLR_MIN_SPEEDUP : 0, Baseline 名)`。具体是：精度没有
    失败，且对 Baseline 的 speedup 达标或没有记录；如果在本硬件上没有测量值，只有门槛 ≤ 0（也就是没有 Baseline）时才算数。
- 排序：自己优先；其次精度没有失败的；再按存的 speedup 从高到低；最后按 kernel hash、路径。

再查表。表中"base 已发布"指有 base commit，且不是首次发布（`is_released`）：

| Incumbent | Baseline | base 已发布 | 对手 | 门槛 |
|---|---|---|---|---|
| 别的 kernel | 有或无 | 任意 | Incumbent | Entry |
| 无，或是自己 | 无 | 任意 | 无对手，只看精度 | - |
| 自己 | 有 | 任意 | Baseline | Retain |
| 无 | 有 | 否 | Baseline | Retain |
| 无 | 有 | 是 | Baseline | Entry |

写回的 speedup：

- 对手是 Baseline：写实测值，记在 Baseline 名下。
- 对手是 Incumbent：写"相对 Incumbent 的倍数 × Incumbent 在同一 workload key 上存的 speedup"，记在 Baseline 名下。
  没有 Baseline，或 Incumbent 在这个 key 上没存值，就不写。
- 无对手：不写。

已知问题，不在本设计内：Incumbent 的资格判断没有检查 `kernel.build.runs_on(target)`，而 dispatch 的 `filter_schemas`
会检查。有 Baseline 时这一点被掩盖了：门槛大于 0，必须在本硬件上有测量值。没有 Baseline（或门槛 ≤ 0）时，一个面向别的
硬件、在本硬件上没有测量值的 base kernel 也会被选成 Incumbent；它建不起来时，这个 workload 会记 PLATFORM_FAULT。
目前已提交的 168 个 kernel 都只面向 gfx950，所以还没有发生过。

audit 对每个 JSON：

- 找不到模板：SCHEMA_INVALID。
- 带 kernel，但模板的 standard 没有 init 或 reference：SCHEMA_INVALID（就是现在那两条检查，改为看模板）。
- inputs、outputs 的名字和模板不同，或 axes 不是模板 axes 的子集：SCHEMA_INVALID。这防的是代码改了签名、JSON 没跟上；
  不拦的话 bench 会在 init 或 reference 调用时崩，报错难懂。axes 只要求子集，因为 `for_call` 会丢掉没取到值的、
  带 extract 的 Const axis。
- JSON 里残留 reference、check、init：不用单独检查，`extra="forbid"` 已经会报 SCHEMA_INVALID。

## 6. tracing 导出

`TracingRuntime.flush` 处理每个有新 workload 的名字时，在 `add_workloads` 之后，如果布局是 `TraceDumpLayout` 且这个
名字的定义带 standard，就调用 `definition.standard.save(layout.root / "standards" / name)`。

- 用的是 `for_call` 得到的定义（带 standard），不是 dataset 里已有的 JSON（它们没有 standard）。
- 每次 flush 都覆盖写，保证和做 tracing 的那一版代码一致。
- 直接写进 checkout（`DefaultLayout`）时跳过：代码本来就在。

```
<AKLR_SCHEMA_PATH>/
├── schemas/<op_type>/<name>/<uuid>.json   合入仓库的只有这个
└── standards/<name>/
    ├── init.py        run(**axes, device, seed) -> dict                          没有 init 就不写
    ├── reference.py   run(**inputs) -> outputs
    ├── check.py       run(reference_outputs, actual_outputs) -> bool
    └── baseline.py    run(**inputs, **outputs) -> Callable[[], Any] | None       没有 baseline 就不写
```

- 四个文件写法相同：preamble、函数连同它调用的项目内函数（`FunctionSource.render`）、`run = <函数名>`。不导入 aklr；
  baseline.py 运行时还要它的 backend。限制和现在一样：不能是 lambda，不能有自由变量，不能用 preamble 之外的全局名。
- preamble 改为在 `save` 里按文件构造，不再是模块级常量：reference 用 `math, torch, torch.nn.functional as F`；
  check 用 `torch`；init 用 `math, torch`；baseline 用 `os, torch`。
- 已实测：现有 22 个 baseline 函数中 21 个能直接渲染，`_fused_moe_baseline` 用到 `os`，加进 preamble 后通过。
  搬家后它们去掉 `original` 参数，改为在函数体里导入 backend，渲染限制不受影响，由第 9 节的测试再确认一次。

## 7. 迁移

- 223 个已提交 JSON 原地 load 再 save，删掉 reference、check、init 三个 key，文件名不变：`SCHEMA_PATH_PATTERN` 只要求
  文件名是 16 位 hex，没有地方要求它等于 uuid。之后新写的 JSON 的 uuid 不再包含源码，和旧文件名不同，这没有影响。
- 旧的 tracing dump 用同一个脚本处理。
- 代价：迁移 PR 自己的 base 版 JSON 解析会失败，这一次 bench 没有 Incumbent，只和 Baseline 比。audit 把解析不了的
  base 当作没有测量值，这是现有行为。
- baseline 函数从 `aklr/seam/aiter.py`、`sglang.py` 搬到各 op 的 `baseline.py`：
  - 去掉 `original` 参数，在函数体里导入对应的 backend 函数；改为公开名字（去掉开头的 `_`）。
  - 一个模板只有一个 baseline。现在按 entry 拆成多个 adapter 的，要合成一个函数：`gemm_fp8_nt_blockscaled*` 的 ck、
    triton、bpreshuffle 三个，按现在的 adapter 注册顺序依次尝试，返回第一个非 None。
  - `_aiter_moe_act_quant` 随 fused_moe 的 baseline 一起搬，seam 改为从 `aklr/ops/fused_moe/baseline.py` 导入。
- seam：`adapter(entry, api)` 去掉 baseline 参数。

注意一处行为差别：`Adapter.original()` 会把 AKLR 装上的替身解开，拿到 backend 原函数；baseline 自己导入 backend 时
不解。如果某个进程装了 seam，又启用了 dispatch 或 tracing，baseline 拿到的就是替身，计的不是 backend 的时间。
bench 进程两者都不启用（现状），所以没有影响；这条前提写进 `BaselineOpponent` 的 docstring。

## 8. 受保护代码（需批准）

| 文件 | 改动 | 为什么绕不开 |
|---|---|---|
| `core/schema/_schema.py` | 删 `reference`、`check`、`init` 字段，删 `source_to_function` 和两个校验器；加 `standard` 字段；加模板查找（`templates`、`declare_template`、`template`）和读 JSON 时补 standard 的校验器 | 需求本身：JSON 不带函数，Schema 聚合 Standard，读 JSON 时自动补上 |
| `core/schema/_standard.py`（新） | `Standard` | `standard` 字段的类型必须能被 core/schema 导入；放到别处会让 core/schema 反向依赖它 |
| `core/schema/_function_source.py` | 删 `validate`、`load`、`FUNCTION_SOURCES` | JSON 不再存源码后，它们没有调用方（先删旧代码） |
| `core/schema/_check.py` | 删 `standard_check` | 它只转发 `default_check`；留着会和 `Standard.check` 混淆。这一项不删也能实现 |
| `core/schema/__init__.py` | 导出 `Standard`，去掉 `standard_check` | 随上面几项 |
| `core/runtime/seam/_adapter.py` | 删 `baseline_fn`、`baseline()` | baseline 归 Standard；留着就是两条路径 |
| `core/runtime/tracing/_runtime.py` | `flush` 写 `standards/<name>/` | 需求要求导出发生在 tracing 时；改成单独命令就要用户多跑一步 |

## 9. 测试（先写，确认失败）

| 测试文件 | 测什么 |
|---|---|
| `tests/core/schema/test_schema.py` | 带 standard 的 Schema 存成 JSON 后没有 standard，也没有 reference/check/init；standard 不同的两个 schema uuid 相同；含 reference 的 JSON 读入时报错 |
| `tests/ops/test_op_schemas.py` | 用 `Schema.templates()` 替换 `_TEMPLATES`；每个已提交 JSON 读出来都带着它模板的 standard；没有模板的 JSON 读出来 standard 为空；代码里构造的不受影响；每个模板的 standard 都能 `save` 到临时目录；每个有 baseline 的模板，服务它的 seam adapter 恰好来自一个 backend |
| `tests/ops/test_reference.py`、`tests/op_case.py` | `template.source_to_function(x)` 改为 `template.standard.x` |
| `tests/core/validation/toy_group.py`、`test_bench_task.py`、`test_bench_stage.py` | toy 用 `standard=Standard(...)` 并注册模板；现有 bench 测试在 JSON 去掉函数后先变红，切换后恢复 |
| `tests/core/validation/test_audit.py` | 第 5 节的三种 SCHEMA_INVALID |
| `tests/core/runtime/test_seam.py` | 删掉 `adapter.baseline` 的断言（`test_absent_entry_has_no_baseline`）；`TestTracing` 新增：走一次 fallback 调用后 `standards/<name>/` 下有对应文件，隔离加载后 init、reference、check 在 CPU 上能跑，结果和模板的函数一致 |

## 10. 实施顺序

在 worktree `.claude/worktrees/yardstick`（分支 `refactor/yardstick`）里做，先删后写，不留新旧两条路径。

1. 写第 9 节的测试，确认失败。
2. 删：Schema 的三个字段和 `source_to_function`，`FunctionSource.load/validate`，`standard_check`，`Adapter.baseline_fn`；
   迁移 223 个 JSON。
3. 写：`Standard`、Schema 的模板查找与读 JSON 时的补全；`aklr/ops/<op>/schema.py` 改用 `standard=`，新增 `baseline.py`；seam 去掉 baseline；
   切换 bench、audit 和 forge driver。
4. tracing 导出。
5. 文档：
   - GLOSSARY：新增 Standard；Reference 改为"一个 schema 的 Standard 里定义正确输出的高精度函数"；Baseline 改为
     "Standard 的 baseline 调用的 backend 函数，也就是 seam adapter 包住的那个原函数"。
   - `bench_design.md`：`_workload_call` 各部分的说明改为读 `schema.standard`。
   - `ops_tutorial.md`、`ops_tutorial.zh.md`：新增 op 时要写 `standard=` 和 `baseline.py`。
   - `op_customization_design.md`：把"Baseline 仍然归 seam adapter"改为归 Standard。

## 11. 不改的

dispatch；`Workload.qualifies` 用的 Baseline 名（仍然是 `Adapter.backend`，bench 也取它）；`SchemaCollection`、
`TraceDumpLayout`、`DefaultLayout`；`ExecutionPolicy`；`MeasurementProfile`；`aklr/core/common`。

## 12. 实现与设计的出入

- tracing 导出的测试放进了 `tests/core/runtime/test_seam.py` 已有的 `TestTracing`，没有新建 `test_tracing.py`。
- baseline 返回 None 记 BASELINE_UNBOUND、抛异常记 PLATFORM_FAULT：这段在 bench 的任务进程里，只有 GPU 测试
  （`test_bench_task.py`）走得到，没有新增 CPU 测试。`tests/ops/test_reference.py` 里只在 gfx950 上跑的
  `test_backend_matches_the_reference` 改为调用 `standard.baseline`。
- topk 的 baseline 不再用 `importlib` 找 `plan_topk_v2`，直接从同一模块导入。
- `tests/ops/test_op_schemas.py` 只看 `aklr/ops` 自己的模板：测试会往默认注册表里加 toy 模板。
- 选 Incumbent 时补上了 `runs_on(target)`（标了 `WORKAROUND:`），根治记在 `TODO.md`。
- 实现初版用 validation 里的 `SchemaTemplates` 让调用方显式补 standard；review 时改为读 JSON 时自动补上，模板查找放进
  `Schema` 自己（`_schema.py`），不新增文件。
