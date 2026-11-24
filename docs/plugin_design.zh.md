# AKLR Plugin 详细设计（已确认；core 和架子已实施，族待移植）

需求见 [plugin_requirements.zh.md](plugin_requirements.zh.md)。本文写 `Plugin`、AITER 的 `Provider`、contract 语言的一处改动和试点族怎么实现。

依据的宿主源码：SGLang 0.5.22 源码和锁定的 0.5.23 镜像，vLLM 0.34.0 源码，AITER 锁定提交 `a6c759b1`。AKLR 库里旧的接入代码不作依据。

各项决定由 owner 在 2026-11-24 逐条确认，列在第 10 节。实施当天 owner 又定了六条（决定 24–29），本文已按它们改过；实施状态见第 9 节。

## 1. 结构

| 文件 | 概念 | 类 | 公共方法 |
|---|---|---|---|
| `aklr/core/plugin.py` | 一个宿主名下的注册 | `Plugin` | `declare`、`register`、`targets`、`dispatchers` |
| `aklr/plugins/<host>/__init__.py` | 一个宿主的 `Plugin` 实例 | 无 | 无（`plugin = Plugin("<host>")`） |
| `aklr/plugins/<host>/<族>/contract.py` | 一个族的 contract | 无 | 无（`CONTRACTS`，最下面是 `declare` 语句） |
| `aklr/plugins/aiter/provider.py` | AKLR 在 AITER 里的 kernel provider | `Provider` | `select`、`enable`，以及协议规定的属性 `name`、`api_versions`、`op_versions` |
| `aklr/core/contract/spec/_args.py`（已有） | contract 参数的规格 | `DTypeSpec` 增加字段 `param_name` | 无 |

没有顶层函数。`<host>` 是宿主名：`sglang`、`vllm`、`aiter`。

族目录里 core 规定的其余文件不变：`reference.py`、`baseline.py`、`group.py` 和 `impls/<语言>/<schema 名>/<uuid>.json`。族里没有 `api.py`。

## 2. 术语

这些词也记在仓库根目录的 `GLOSSARY.md` 里。

| 词 | 含义 | 不再用 |
|---|---|---|
| 宿主 | sglang、vllm、aiter 之一 | 目标库 |
| target | SGLang、vLLM 里一个函数的点分全名；AITER 里一个 op 的名字 | api、plugined api、route |
| device target | kernel 面向的 GPU 架构和硬件（core 的 `DeviceTarget`、schema 的 `kernel.targets`） | 单独说 target |
| 声明 | 一条 `declare`：target、contract、宿主版本 | |
| 族 | `aklr/plugins/<host>/` 下的一个目录，目录名是 contract 的 `op_type` | |
| 注册函数 | 宿主 entry point 调的 `Plugin.register` | hook 函数 |
| 截获函数 | 替换一个 target 的那个函数 | hook（只指 SGLang `HookRegistry` 的 hook） |
| 宿主版本 | `versions` 里的值。SGLang、vLLM 是发行版本，AITER 是 op 版本 | 版本（不加限定） |
| 回退 | 未命中时运行宿主自己的实现 | |

## 3. `Plugin`

### 3.1 类定义

```python
# aklr/core/plugin.py
class Plugin:
    """One host library's registrations: the targets AKLR serves in it and the contracts describing them."""

    def __init__(self, host: str) -> None:
        """``host`` names the host library and its directory ``aklr/plugins/<host>/``."""

    def declare(self, target: str, contract: OpContract, path: str | None = None, *,
                versions: tuple[str, ...] | tuple[int, ...]) -> None:
        """``target``, a host function's dotted full name (an op's name in aiter), is described by
        ``contract`` on the host versions ``versions``, and the contract's schema JSON, its kernels, are
        under ``path``, relative to the family's directory; ``None``: it has none yet. A statement at the
        bottom of a family's ``contract.py``: heard when the plugin imports that file, nothing when another
        does."""

    def register(self) -> None:
        """The host's entry point calls it: collect the declarations, check them, intercept each target."""

    def targets(self) -> Mapping[str, tuple[str, ...] | tuple[int, ...]]:
        """Every declared target and the host versions it is declared for."""

    def dispatchers(self, target: str, version: str | int) -> tuple[KernelDispatcher, ...]:
        """The dispatchers of the contracts declared for ``target`` at host version ``version``, in
        declaration order; empty when no declaration lists ``version``."""
```

三个宿主用这一个类。`Plugin` 不新增调度类，`dispatchers` 返回的是 core 的 `KernelDispatcher`。

### 3.2 `declare`

`declare` 记下 target、contract、`versions`。它是族的 `contract.py` 最下面的语句，没有单独的 `plugin.py`（决定 35）：语句属于哪个族不看调用栈（决定 27），看 contract：`CONTRACTS` 里有这个 contract 对象的那个族。

`contract.py` 也会被 core 的 registry 导入，谁先导入都一样：语句在文件被导入时记下来，收集时（3.4）`Plugin` 导入每个族的 `contract.py`（已经导入过的不会重新执行），再把记下的语句按 contract 归到各族。不重新执行文件，是因为 main 的 #322 之后 contract 上带着 reference、baseline 对象，重新执行会得到与 registry 手里不相等的 contract，`DispatcherRegistry` 会拒绝。

语句执行时就抛 `ValueError` 的情况：

| 情况 | 对应 |
|---|---|
| `target` 不是名字（点分的标识符）；`versions` 为空，或混用字符串和整数 | R2、R4a |
| 同一个 (target, contract) 已经声明过 | 需求第 3 节“重复注册报错” |

下面几种要到收集时才知道，报的错写明族的文件和那条 `declare`，没有行号：`contract` 不在任何族的 `CONTRACTS` 里（这条语句不生效，决定 10）；`target` 已在另一个族里声明过，或 `path` 不是存在的目录（这个族的声明都不生效）。

同一个 target 可以声明多次，有两种用途，写法相同：

```python
plugin.declare(TARGET, FP8, versions=("0.5.23",))    # 重载：按语句顺序尝试，第一个命中的生效
plugin.declare(TARGET, MXFP4, versions=("0.5.23",))
plugin.declare(TARGET, OLD, versions=("0.5.22",))    # R4b：宿主改了签名，旧 contract 留给旧版本
```

`versions` 对 `Plugin` 是不透明的值，只做“在不在里面”的比较。target 对它也是不透明的名字：SGLang、vLLM 的点分全名在注册时解析，解析不到按 3.7 处理；AITER 的 op 名由 `Provider` 使用。

### 3.3 schema 文件的目录

没有单独的 `provide`：`declare` 的第三个参数 `path` 就是这个 contract 的 schema JSON 所在的目录（决定 37），相对于本族的目录，写 `"impls"`。不写表示还没有 kernel。

`path` 不改变 core 加载什么：core 照旧扫描族目录下的 `impls/`。它是一条被核对的陈述。

| 时机 | 核对 |
|---|---|
| 收集时 | 写了 `path` 时它是存在的目录 |
| 收集结束时 | 有声明的族里，`impls/` 下每个 JSON 都要有一条声明：声明的是它的 contract，`path` 包含它 |

第二条同时覆盖两种错误：JSON 不属于任何已声明的 contract；已声明 contract 的 JSON 不在声明写的目录下。它要读每个 JSON 的 `contract_id`。核对不过的后果见 3.7。

由此得出两点：

- 不写 `path` 的声明是合法的：target 会被截获，全部未命中。用 `AKLR_EXPORT=1` 采集调用时就是这种状态。
- 没有宿主 target 的 contract（需求第 7 节，只参加验证）要放在没有任何声明的族里。放进有声明的族，它的 JSON 会让第二条核对失败。

### 3.4 收集

第一次调用 `register`、`targets` 或 `dispatchers` 时，`Plugin` 对 `aklr/plugins/<host>/` 下每个族（按路径排序）导入它的 `contract.py`，取它的 `CONTRACTS`，然后把记下的语句按 contract 对象归到各族，核对 3.2 里收集时的几条，再对有声明的族做 3.3 的第二条核对。持锁，每个进程只做一次。

没有公开的 `collect`。新增一个族只需要新建目录，不改注册函数、entry point 或任何清单（R6）。导入 `contract.py` 不导入宿主库，因为 target 是字符串。

### 3.5 `targets` 和 `dispatchers`

这两个查询对三个宿主通用，`register` 和 `Provider` 都用它们（R19）。

- `targets()`：target 到它所有声明的 `versions` 的并集。
- `dispatchers(target, version)`：`versions` 含 `version` 的那些声明的 contract，按声明顺序，各取 `DispatcherRegistry.default().dispatcher(contract)`。没有声明列出 `version` 时返回空元组（R7a）。`target` 没声明过抛 `KeyError`。

dispatch 启动之前取到的 dispatcher 对每次调用都返回未命中，启动之后同一个对象开始服务。所以截获函数可以早于设备就绪装上（R13）。

### 3.6 `register`

`register` 是 SGLang、vLLM 的注册函数。它不带参数，返回值被宿主忽略，在宿主的每个进程里都会被调用。它不向宿主抛异常（3.7）。

**步骤**

1. 读开关（`RuntimeEnv`）。`AKLR_DISABLE=1`，或本宿主的开关（`AKLR_SGLANG_PLUGIN`、`AKLR_VLLM_PLUGIN`）不是 `1`：什么都不做（R9）。
2. 本进程已经注册过：返回（R8）。
3. 收集（3.4）。
4. 取运行中的宿主版本：`importlib.metadata.version(host)`，去掉 `+` 之后的本地标签。
5. 安排 dispatch 在设备就绪后启动（`DispatcherRegistry.default().init_once()`）。`register` 自己不碰 CUDA。
6. 对每个声明过的 target：
   - `dispatchers(target, 版本)` 为空：不截获，记入“版本不符”的名单（R7a）。
   - 解析出宿主函数（`pkgutil.resolve_name`）。
   - 对每个 dispatcher 调 `make_api_fn(宿主函数)`，得到这个 contract 的派发函数。core 在这一步核对参数表。
   - 装上截获函数。
7. 记一行日志：截获了哪些 target，哪些因版本不符没有截获。

`Plugin("aiter").register()` 什么都不装，记一条 error：AITER 不经注册函数接入（第 4 节）。

**截获函数**

用法与 core 的 `@api` 相同，只是不用 `@api` 另外定义一个函数：target 就是宿主的函数。命中由 core 的 dispatcher 派发，未命中调宿主的函数。

```python
dispatch_fns = [(d.contract.contract_id, d.make_api_fn(fn)) for d in self.dispatchers(target, version)]
returns_none = inspect.signature(fn).return_annotation in (None, "None")   # fn 是宿主函数
tracer = trace()                                       # AKLR_TRACE 关闭时是 None

def intercept(original, *args, **kwargs):
    if torch.compiler.is_compiling():                  # R12：追踪期间不运行 kernel
        return original(*args, **kwargs)
    reached.add(target)                                # R15：这个 target 到达过 dispatch
    for contract_id, dispatch_fn in dispatch_fns:      # 声明顺序
        result = dispatch_fn(*args, **kwargs)          # core 生成：组实参，调 dispatch
        served = not isinstance(result, FallbackReason)
        if tracer is not None and tracer.is_plugin:    # AKLR_TRACE=plugin
            tracer.plugin(host, target, contract_id, None if served else result)
        if served:
            return None if returns_none else result    # R11：宿主函数返回 None 时返回 None
    return original(*args, **kwargs)                   # 全部未命中，回退
```

`dispatch_fn` 来自 core 现有的 `KernelDispatcher.make_api_fn`，也就是 `@api` 背后的那个方法。它按宿主函数的签名生成一个函数，做 R11 的前两件事：位置实参对到参数名；调用方没传的参数先用宿主函数的默认值，contract 有而宿主函数没有的参数用 contract 的默认值。然后调 `dispatch`。key 的生成、命中判断、输出分配、运行 kernel、计数和 op 层的 trace 都在 `dispatch` 里（R10）。kernel 抛出的异常原样传出（R14）。

`make_api_fn` 现在不能用于宿主函数，只因为 `OpContract.check_api_signature` 多做了三条给 `@api` 桩函数定的检查：函数名等于 contract 名、位于 `<族>/api.py`、函数体是 `...`。本设计删掉这三条（第 7 节）。

宿主函数被装饰过时，传给 `make_api_fn` 的是 `inspect.unwrap` 的结果，因为 core 从 `__code__` 读参数名。

返回注解是 `"None"`（字符串）也算，SGLang 的不少模块用了 `from __future__ import annotations`。

**“参数表一致”的含义**

就是 `check_api_signature` 剩下的规则，不一致时 `make_api_fn` 抛 `TypeError`（R3）：

1. 宿主函数没有 positional-only 参数、`*args`、`**kwargs`。
2. contract 的每个必需参数都是宿主函数的参数，且在宿主函数里没有默认值，或默认值是 `None`。
3. 宿主函数多出来的参数必须有默认值。调用时它不是默认值，这次调用算未命中。

第 3 条接住了 contract 语言写不了的参数（枚举、`dict`、对象）：不写进 contract，只服务它取默认值的调用。

**SGLang**

| 事 | 做法 | 依据 |
|---|---|---|
| 设备就绪 | `HookRegistry.register("sglang.srt.model_executor.model_runner.ModelRunner.init_torch_distributed", …, HookType.AFTER)` | `ModelRunner.__init__` 先取设备（`model_runner.py:366`），再调它（`:452`），之后才加载模型 |
| 装截获函数 | `HookRegistry.register(target, intercept, HookType.AROUND)` | AROUND 的调用形式是 `hook(original_fn, *args, **kwargs)` |
| target 解析不到 | `register` 自己先解析 | `apply_hooks` 解析失败只记它自己的日志，不指出声明 |
| 更早取走的引用 | 不用处理 | `apply_hooks` 会把 `sys.modules` 里指向原函数的模块属性一并换掉 |
| 捕获前预热 | 不用处理 | 捕获每个形状前先跑两遍前向（`breakable_cuda_graph_backend.py:121`、`tc_piecewise_cuda_graph_backend.py:240`） |

**vLLM**

| 事 | 做法 | 依据 |
|---|---|---|
| 设备就绪 | 包住 `vllm.v1.worker.worker_base.WorkerWrapperBase.init_device`，原方法返回后启动 dispatch | 单进程和多进程 executor 都经它。worker 类可配置，所以不包 `gpu_worker.Worker` |
| 装截获函数 | 把 target 所指的模块属性换成截获函数，在 `register` 被调用的当下完成 | vLLM 没有 hook 机制。fork 出来的子进程继承补丁，但不会再调注册函数 |
| 从未到达 dispatch 的 target（R15） | 包住 worker 实例的 `compile_or_warm_up_model`，它返回后把一次都没到达 dispatch 的 target 列在一条 warning 里 | 预热和捕获到这里结束，每个形状都已 eager 跑过 |
| 捕获前预热（R16） | 不用处理 | 捕获每个 graph 前，vLLM 用同样的 token 数先 eager 跑一次同一个前向（`v1/worker/gpu/cudagraph_utils.py:502`） |

vLLM 上有一条 `Plugin` 解决不了的限制。vLLM 默认用 `torch.compile`，只追踪一次。从被编译的 forward 里直接调用的普通 Python 函数会被内联，函数体之后不再执行，而追踪期间截获函数走原函数（R12）。所以这类调用点永远不会被服务，R15 的 warning 会把它列出来。能被服务的 target 是在不透明的 custom op 实现内部、或编译区域之外被调用的函数。`vllm._custom_ops` 里的函数大多属于前一种。

### 3.7 出错时

三个宿主一条规则：AKLR 的错误不让宿主失败。出错的部分不生效，对应的 target 走宿主自己的实现，用 error 级别的日志记一次。语句执行时出的错写明 `contract.py` 和行号（取自异常的 traceback）；注册时才发现的错（target 解析不到、参数表不一致）写明 `contract.py` 和那条 `declare` 的 target。这条日志不依赖 `AKLR_TRACE`。

| 出错的地方 | 不生效的范围 |
|---|---|
| 开关的值不合法 | 这个宿主的全部声明 |
| 导入某个族的 `contract.py` 时出错，包括 3.2、3.3 里语句执行时的 `ValueError` | 这个族的全部声明 |
| 收集结束时某个族的 JSON 核对不过（3.3） | 这个族的全部声明 |
| target 解析不到，或某个 contract 与宿主函数的参数表不一致 | 这个 target |
| dispatch 启动失败（schema 不合法、认不出设备） | 全部 target 都未命中，本进程不再重试 |

唯一的例外是 R14：kernel 开始运行后抛出的异常原样传出，不回退。

声明写错在线上只表现为没有加速，所以要靠两样东西发现：这条 error 日志，和 CI 里对真宿主的核对（第 8 节的 `test_upstream.py`）。

### 3.8 两种控制方向

| | SGLang、vLLM | AITER |
|---|---|---|
| 谁把调用交给 `KernelDispatcher` | 截获函数调 `dispatch` | AITER 调 `Provider.select`，它调 `serving` |
| 命中 | 截获函数返回结果 | `select` 返回候选，AITER 调候选的 `run`，它调 `dispatch` |
| 未命中 | 截获函数调原函数 | `select` 返回 `None`，AITER 运行自己的 kernel |
| `torch.compile` 追踪期间 | 截获函数调原函数 | `select` 返回 `None` |
| 校验的时机 | 注册函数被调用时，每个进程 | AITER 第一次为这个 op 来问时，只在用到 AITER 的进程 |

## 4. `Provider`

### 4.1 类定义

```python
# aklr/plugins/aiter/provider.py
class Provider:
    """AKLR as an AITER kernel provider: it answers AITER from the declarations on Plugin("aiter")."""

    name = "aklr"
    api_versions = (1,)                          # aiter.plugin.API_VERSION this is written against
    op_versions: Mapping[str, tuple[int, ...]]   # AITER op name -> the op versions declared

    def __init__(self) -> None:
        """AITER instantiates the entry point's class when it discovers providers."""

    def select(self, op: Op, args: tuple, kwargs: Mapping) -> KernelCandidate | None:
        """The kernel serving this call, else None; nothing runs."""

    @staticmethod
    def enable() -> None:
        """SGLang's and vLLM's entry points call it: enable AKLR in AITER when AKLR_AITER_PLUGIN=1."""
```

`Provider` 不收声明，不查表，不运行 kernel。声明在 `Plugin("aiter")` 上，查表和运行是 `KernelDispatcher.serving` 和 `dispatch`。`aklr/core/plugin.py` 不 import 这个文件（R17、R18）。

### 4.2 生命周期

| 阶段 | 谁触发 | AKLR 做什么 |
|---|---|---|
| 安装 | `pip install aklr` | 什么都不做 |
| 启用 | SGLang、vLLM 的 entry point `aklr_aiter` 调 `Provider.enable`；单独使用 AITER 的用户自己调 `aiter.plugin.configure` | 4.6 |
| 发现 | AITER。启用之后第一次调到被路由的 op 时 | `Provider()`，4.3 |
| 询问 | AITER，每次调到被路由的 op | `select`，4.4 |
| 执行 | AITER 选中候选后 | 候选的 `run`，4.5 |

### 4.3 发现：`Provider()`

AITER 的 target 就是 op 名（决定 28），所以发现时只做一件事：`op_versions = plugin.targets()`，op 名到声明的 op 版本。AITER 只为列在里面的 op 和版本来问。

每个 op 的核对推迟到 AITER 第一次为它来问时（4.4 第 3 步），不在发现时做。原因：op 名解析不出模块，而 40 个被路由的函数里有 9 个模块不随 `import aiter` 导入，发现 provider 时它们的 op 还没有登记到 `aiter.plugin._routed_ops`。

op 的签名取自 `aiter.plugin._routed_ops[op.name].contract`。这是 AITER 的一个私有名：10 个带 `fake` 的 op 注册成 torch custom op 之后，模块属性是 `OpOverloadPacket`，`inspect.signature` 只给 `(*args, **kwargs)`，而 AITER 没有公开的取签名接口。40 个 op 统一从这里取。

`Provider()` 自己不抛异常。两个后果：

- 声明了一个 AITER 没有路由的 op 名：AITER 永远不会为它来问，AKLR 这边没有任何日志。靠 CI 里对真 AITER 的核对（第 8 节的 `test_upstream.py`）发现。
- 运行中的 op 版本没有被声明（R7a）：AITER 读 `op_versions` 后不来问，由它自己的 `status()`、`explain()` 报告；AKLR 不另记日志。

### 4.4 `select`

AITER 传来的 `args` 是按 op 签名排好、补齐默认值的位置实参，仅限关键字的参数在 `kwargs` 里。`select` 不抛异常。

1. 正在 `torch.compile` 追踪中：返回 `None`（R12）。
2. dispatch 还没启动：`DispatcherRegistry.default().init_once()`。只开了 `AKLR_AITER_PLUGIN` 时没有别人启动它。启动失败按 3.7 处理，本进程此后一律返回 `None`。
3. 第一次为这个 op 来问时，对 `plugin.dispatchers(op.name, op.version)` 的每个 contract：按 3.6 的规则核对签名，用的是同一个 `OpContract.check_api_signature`（它收的是函数，这里手里是 `Signature`，所以传一个带 `__signature__` 的占位函数）；记下 op 的参数名顺序；生成一个固定的 `run`。出错按 3.7 处理：记一次 error，这个 op 此后一律返回 `None`。
4. 按声明顺序逐个 contract：
   - 实参按参数名组成这个 contract 的调用。op 签名多出来的参数不是默认值：跳过。contract 多出来的参数用 contract 的默认值。
   - `dispatcher.serving(call)` 命中：返回候选。
   - `AKLR_TRACE=plugin` 时，用 `Trace.plugin` 记下这个 contract 命中或回退的原因，与 3.6 相同。
5. 全部未命中：返回 `None`，AITER 运行自己的 kernel。

```python
KernelCandidate(
    provider="aklr",
    kernel_id=metadata.kernel_ref,
    artifact_digest=metadata.content_hash,
    run=<这个 op 的这个 contract 的 run>,
)
```

`latency_us`、`baseline_us` 等比较字段不填。AKLR 只在测量合格时才提供候选。

### 4.5 `run`

每个 (op, contract) 只有一个 `run`，第一次被问到时生成，一直复用。AITER 按 `id(run)` 缓存它对参数个数的检查。

1. 实参组成调用，调这个 contract 的 `dispatcher.dispatch(call)`。
2. 结果是 `FallbackReason`：抛 `RuntimeError`。`select` 刚问过同一张表，这里未命中只能是 bug，而 AITER 规定 `run` 开始后不回退。
3. kernel 抛出的异常原样传出（R14）。
4. op 签名的返回注解是 `None` 时返回 `None`，否则返回 `dispatch` 的结果（R11）。

### 4.6 `enable`

1. `AKLR_DISABLE=1`，或 `AKLR_AITER_PLUGIN` 不是 `1`：返回。
2. `aiter.plugin.configure(source="aklr: AKLR_AITER_PLUGIN=1", enable=["aklr"])`。导入 `aiter.plugin` 失败按 3.7 处理。

不传 `exclude`，所以不会触发 AITER 的发现，也就不会在没有 GPU 工作的进程里导入 AITER 的各个模块。`configure` 会覆盖之前的启用列表。SGLang 0.5.22 和 vLLM 0.34 的源码里都没有调 `aiter.plugin.configure` 的地方，所以现在没有可被覆盖的东西。

## 5. contract 语言的改动

现在“调用方可以选输出的 dtype”绑在名字上：输出必须叫 `out`，参数必须叫 `out_dtype`，而且 `out` 不能与别的输出并存。宿主函数的参数名不由 AKLR 定，所以改成显式声明。

```python
# aklr/core/contract/spec/_args.py
class DTypeSpec(FrozenModel):
    any_of: tuple[DType, ...] | None = None
    same_as_tensor: str | None = None
    default: DType | None = None
    # new: the call param through which the caller chooses this output's dtype; None: not the caller's choice
    param_name: str | None = None
```

字段名和含义与 `AttrSpec.param_name` 一致。`out` 这个名字不再有特殊含义。

| 原来的规则 | 现在 |
|---|---|
| 输出要么只有一个 `out`，要么都不叫 `out` | 取消，输出可以叫任何名字 |
| `out` 的 dtype 由 `out_dtype` 选；别的输出只有一个 dtype，或与某个输入相同 | 写了 `param_name` 的输出由这个参数选 dtype；没写的输出只有一个 dtype，或与某个输入相同 |
| `out` 总能由 dispatch 分配，不能是某个轴的唯一来源 | 写了 `param_name` 的输出同样 |
| 有输出 `out` 时 `api_params` 里才有 `out_dtype` | 每个写了 `param_name` 的输出在 `api_params` 里对应一个同名的参数，位置不变：可选输入之后，非必需输出之前 |
| `out_dtype` 是保留名 | 不再保留。`param_name` 参加“名字不重复”的检查；输入写 `param_name` 是错误 |

改动的位置：

| 文件 | 改动 |
|---|---|
| `aklr/core/contract/spec/_args.py` | `DTypeSpec.param_name`；`TensorSpec.narrow` 保留它，所以 schema 的 `kernel_call_keys` 里仍有这个参数 |
| `aklr/core/contract/op.py` | `_RESERVED_NAMES`、`_check_names`、`_check_tensors` 的两处判断、`api_params` |
| `aklr/core/contract/fastkey/_fast_key_fn.py` | 生成输出部分的方法里，三处“名字是 `out`”的判断和写死的 `"out_dtype"` |
| `aklr/core/validation/stages/_bench/_evaluation/_request.py` | 给每个写了 `param_name` 的输出填 dtype 参数，原来只填 `out_dtype` |

这些都不在受保护目录里。schema JSON 的格式不变：JSON 里的输出只有形状、dtype 和描述，不含 `DTypeSpec` 的字段，`schema_version` 不动。

AITER 的三种写法都能描述了：

```python
# aiter.ops.gemm_op_a8w8.gemm_a8w8_blockscale_bpreshuffle(XQ, WQ, x_scale, w_scale, dtype=dtypes.bf16, out=None)
outputs={"out": TensorSpec(…, dtype=DTypeSpec(any_of=(DType.BFLOAT16, …), default=DType.BFLOAT16, param_name="dtype"))}

# aiter.ops.triton.gemm.basic.gemm_a16w16._gemm_a16w16_routed(x, w, bias=None, dtype=torch.bfloat16, y=None, …)
outputs={"y": TensorSpec(…, dtype=DTypeSpec(any_of=(DType.BFLOAT16, …), default=DType.BFLOAT16, param_name="dtype"))}

# aiter.ops.rmsnorm.rmsnorm2d_fwd_with_add(out, input, residual_in, residual_out, …)：没有 dtype 参数
outputs={"out": _output(…), "residual_out": _output(…)}
```

旧 contract 迁回时，靠名字 `out` 得到 `out_dtype` 的要显式写：宿主函数有 dtype 参数就写它的名字；没有就去掉可选的 dtype，改成与某个输入相同，或按 dtype 拆成几个 contract。kernel 入口里的 `out_dtype` 参数跟着改名。

## 6. 试点族

**这一节没有实施**（决定 24：先把核心写好，族之后移植）。仓库里现在只有 `aklr/plugins/sglang/kimi_k3_tiny_gemm` 的 `contract.py` 和 `reference.py`，作为 core 测试用的 contract（决定 25），没有 `plugin.py` 和 kernel JSON，所以三个宿主里都没有声明。下面保留当时的设计，移植时照此做。

四个 target 的签名都对着源码核过。每个族的 contract 和 JSON 从 `c8e0e1534^` 取回，把 contract 的输入输出改成宿主函数的参数名。JSON 的键和 kernel 入口的参数名同步改，UUID 随之变化，测量字段清空，由 CI 重测。

### 6.1 SGLang

宿主函数（0.5.22 和 0.5.23 相同，没有装饰器）：

```python
def kimi_k3_tiny_gemm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:   # x @ w.T
```

```python
# aklr/plugins/sglang/__init__.py（全文）
from aklr.core.plugin import Plugin

plugin = Plugin("sglang")
```

```python
# aklr/plugins/sglang/kimi_k3_tiny_gemm/contract.py（节选）
KIMI_K3_TINY_GEMM = OpContract(
    contract_id="kimi_k3_tiny_gemm@1.0",
    op_type="kimi_k3_tiny_gemm",                      # 族目录名
    inputs={"x": TensorSpec(shape=("m", "k"), …), "w": TensorSpec(shape=("n", "k"), …)},
    outputs={"out": TensorSpec(shape=("m", "n"), …)},
    …
)
CONTRACTS = (KIMI_K3_TINY_GEMM,)
```

```python
# aklr/plugins/sglang/kimi_k3_tiny_gemm/contract.py 的最下面（文件开头 from aklr.plugins.sglang import plugin）
TARGET = "sglang.kernels.ops.gemm.kimi_k3_tiny_gemm"
plugin.declare(TARGET, KIMI_K3_TINY_GEMM, "impls", versions=("0.5.22", "0.5.23"))
```

两个调用点都在函数体内 `from sglang.kernels.ops.gemm import kimi_k3_tiny_gemm`，调用时才取名字，能走到截获函数。

### 6.2 vLLM

宿主函数（0.34.0）：

```python
def wvSplitKrc(a: torch.Tensor, b: torch.Tensor, cu_count: int, bias: torch.Tensor = None) -> torch.Tensor:
```

```python
# aklr/plugins/vllm/wv_split_krc/contract.py 的最下面
TARGET = "vllm._custom_ops.wvSplitKrc"
plugin.declare(TARGET, WV_SPLIT_KRC, "impls", versions=("0.34.0",))
```

contract 的输入是 `a`（`[n, k]`）、`b`（`[m, k]`）、`cu_count`（整数标量，kernel 不用）、`bias`（可选），输出 `out`（`[n, m]`）。`cu_count` 没有默认值，按 3.6 的规则 3 必须写进 contract。

实施时试过：`cu_count` 不能写成只有 `interval` 的 `ScalarSpec`。这种标量在 bench 里的默认值是浮点 `1.0`，fast key 拒绝给整数标量传浮点；而且 `cu_count=1` 也不是 vLLM kernel 的合法调用（它要求 `CuNeeded <= cu_count`）。写成 `values=(256,)`，即 MI355X 的 CU 数。

调用点是 `rocm_unquantized_gemm_impl` 里的 `ops.wvSplitKrc(...)`。这个实现注册成了 torch custom op，对 `torch.compile` 不透明，所以每次 eager 执行和捕获都会走到截获函数。0.33.0 没有核对，所以没写进 `versions`。

### 6.3 AITER

一个族，两个 target。第二个同时走到三条难走的路径：带 `fake`（签名只能从 `_routed_ops` 取）、返回 `None`、`out` 与另一个输出并存（第 5 节的改动）。

```python
@_route(op="rmsnorm", version=2)
def _rms_norm_fwd_dispatch(input, weight, epsilon, use_model_sensitive_rmsnorm) -> Tensor:

@_route(op="rmsnorm_with_add", version=1, fake=…, mutates_args=["out", "residual_out"])
@torch_compile_guard(…)
def rmsnorm2d_fwd_with_add(out: Tensor, input: Tensor, residual_in: Tensor, residual_out: Tensor,
                           weight: Tensor, epsilon: float, gemma_norm: bool = False,
                           use_model_sensitive_rmsnorm: int = 0) -> None:
```

```python
# aklr/plugins/aiter/rms_norm/contract.py（节选）
RMSNORM = OpContract(
    contract_id="rmsnorm@1.0",
    op_type="rms_norm",
    attrs={
        "use_model_sensitive_rmsnorm": attr(
            "AITER's model-sensitive variant; AKLR serves only 0.",
            0, default=0, param_name="use_model_sensitive_rmsnorm", optional=True,
        ),
    },
    inputs={"input": TensorSpec(…), "weight": _WEIGHT, "epsilon": _EPSILON},
    outputs={"normalized": _output("The normalized rows, scaled by weight.")},
    …
)
RMSNORM_WITH_ADD = OpContract(
    contract_id="rmsnorm_with_add@1.0",
    op_type="rms_norm",
    inputs={"input": TensorSpec(…), "residual_in": TensorSpec(…), "weight": _WEIGHT, "epsilon": _EPSILON},
    outputs={"out": _output("The normalized sum, scaled by weight."), "residual_out": _output("input + residual_in.")},
    …
)
CONTRACTS = (RMSNORM, RMSNORM_WITH_ADD)
```

```python
# aklr/plugins/aiter/rms_norm/contract.py 的最下面
# AITER 的 target 是 op 名（决定 28）
plugin.declare("rmsnorm", RMSNORM, "impls", versions=(2,))
plugin.declare("rmsnorm_with_add", RMSNORM_WITH_ADD, "impls", versions=(1,))
```

两个参数的两种处理：

- `_rms_norm_fwd_dispatch` 的 `use_model_sensitive_rmsnorm` 没有默认值，所以写成只允许 `0` 的 attr。调用时不是 `0`，fast key 返回空，这次调用未命中。
- `rmsnorm2d_fwd_with_add` 的 `gemma_norm` 和 `use_model_sensitive_rmsnorm` 有默认值，不写进 contract。调用时不是默认值就未命中（3.6 的规则 3）。

## 7. 其余改动

| 文件 | 改动 | 受保护 |
|---|---|---|
| `aklr/core/plugin.py` | 新增 | 否 |
| `aklr/core/contract/op.py` | `check_api_signature` 删三条桩函数检查；第 5 节的改动；2 处提到 `aklr/ops` 的注释 | 否 |
| `aklr/core/contract/spec/_args.py`、`fastkey/_fast_key_fn.py`、`aklr/core/validation/stages/_bench/_evaluation/_request.py` | 第 5 节的改动 | 否 |
| `aklr/core/contract/locate.py` | 删 `aklr.ops` 的分支：两个正则、`SCHEMA_ROOTS`、`op_packages`、`op_package_of` 的回落 | 否 |
| `aklr/core/validation/stages/validation_scope.py` | 删遍历 `aklr/ops` 的一行 | 否 |
| `aklr/core/contract/registry.py`、`export.py`、`aklr/core/validation/command.py` | 各一处提到 `aklr/ops` 的文字 | 否 |
| `aklr/core/runtime/dispatch/_sglang.py`、`_vllm.py`、`aklr/core/runtime/_vllm_hook_registry.py` | 删除 | 是，owner 已批准（决定 8） |
| `aklr/core/runtime/dispatch/__init__.py` | 删四个导出（`register_sglang_plugin`、`register_vllm_plugin`、`SGLANG_HOOKED_METHOD`、`VLLM_HOOKED_METHOD`）；加一个导出 `trace` | 是，owner 已批准（决定 8、21） |
| `aklr/core/runtime/dispatch/` 其余文件 | 不动。`api` 装饰器不再有调用方，留着 | 是 |
| `pyproject.toml` | 五条 entry point（见下）；package-data 的 glob 按新布局改，去掉 `aklr.ops` | 否 |
| `.github`、`harness` 里的 `aklr/ops` 路径，三份教程 | 随族迁移逐步替换，不在本设计里 | |

受保护目录的两处改动要在 PR 描述里写明经 owner 批准。

```toml
[project.entry-points."sglang.srt.plugins"]
aklr = "aklr.plugins.sglang:plugin.register"
aklr_aiter = "aklr.plugins.aiter.provider:Provider.enable"

[project.entry-points."vllm.general_plugins"]
aklr = "aklr.plugins.vllm:plugin.register"
aklr_aiter = "aklr.plugins.aiter.provider:Provider.enable"

[project.entry-points."aiter.plugins"]
aklr = "aklr.plugins.aiter.provider:Provider"
```

entry point 的值可以带点分属性，我用标准库的 `EntryPoint.load` 试过。设了 `SGLANG_PLUGINS` 或 `VLLM_PLUGINS` 白名单的部署要把 `aklr_aiter` 也列进去。

## 8. 测试

每步先写测试并看到失败。测试只调 `Plugin`、`Provider`、`OpContract` 的公共方法，以及宿主自己的函数。

| 测试 | 在哪跑 | 覆盖 |
|---|---|---|
| `tests/core/contract` 里补 | CPU | 第 5 节：输出不叫 `out` 也能由参数选 dtype；`out` 与别的输出并存；`param_name` 与别的名字重复、写在输入上都报错；fast key 按该参数取 dtype；`check_api_signature` 接受宿主函数 |
| `tests/core/test_plugin.py` | CPU，假的 `sglang`、`vllm` 模块和假的宿主函数 | `declare` 语句的各种报错；收集后的核对；`targets`；`dispatchers` 按版本和声明顺序；开关关闭时什么都不装；第二次 `register` 不重复；版本不符时跳过并记日志。3.7 的每一行：不生效的范围对、有 error 日志、`register` 不抛异常。截获函数：命中、未命中调原函数、调用方省略带默认值的参数、多出的参数不是默认值时未命中、多个 contract 的顺序、返回注解为 `None` 和 `"None"`、追踪期间调原函数、`AKLR_TRACE=plugin` 的输出 |
| `tests/plugins/aiter/test_provider.py` | CPU，假的 `aiter.plugin` | `op_versions` 由声明生成；发现之后才被路由的 op 也能服务；参数表不一致的 op 在第一次被问到时记一次 error，之后一律回退，`Provider()` 和 `select` 都不抛异常；候选的字段；同一个 (op, contract) 的 `run` 是同一个对象；`run` 未命中时抛错；追踪期间返回 `None`；dispatch 启动失败后一律返回 `None`；`enable` 在开关关闭时不调 `configure` |
| `tests/plugins/<host>/test_upstream.py` | 宿主镜像（GPU） | 每个声明过的 target 在真宿主上能解析、签名核对通过；试点 target 调一次，结果来自 AKLR 的 kernel，并与 reference 一致 |

`tests/core` 里原来用 `gemm@1.0` 的测试改写到 `kimi_k3_tiny_gemm@1.0` 上；需要已删除的族、或需要已提交的 schema 文件的测试先跳过，随族移植恢复。`test_upstream.py` 随第一个族一起写。

被删代码的测试一起删：`tests/plugins/{sglang,vllm}/test_hook.py`，以及 `tests/core/runtime/test_dispatch.py` 里测 `register_vllm_plugin` 的部分。

## 9. 实施顺序

| 步骤 | 内容 | 验证 | 状态 |
|---|---|---|---|
| 0 | 第 7 节的 core 清理 | `tests/core`（CPU） | 已完成 |
| 1 | 第 5 节 contract 语言的改动 | `tests/core/contract`（CPU） | 已完成 |
| 2 | `Plugin` 和 `tests/core/test_plugin.py` | CPU | 已完成 |
| 3 | `Provider` 和 `tests/plugins/aiter/test_provider.py` | CPU | 已完成 |
| 4 | 三个试点族，`pyproject.toml` 的 entry point | sglang-test、vllm-test、aiter-test（GPU） | entry point 已完成；试点族未做（决定 24） |
| 5 | 其余族逐个迁移，每个族先选定目标函数。第一个迁带 dtype 参数的 op，在 GPU 上验证第 5 节 | 每族一次 GPU job | 未做 |

其余族的迁移表不在本文里。它需要对 `c8e0e1534^` 里的 30 个 `contract.py` 清点一遍，还没有做。AITER 一侧能声明的 40 个 target 见附录 A。

## 10. 已确认的决定

owner 在 2026-11-24 逐条确认。带引号的是 owner 的原话。

| # | 决定 | 结论 |
|---|---|---|
| 1 | 登记以谁为准 | core 的发现不变，作者照旧写 `CONTRACTS`；`declare` 是 target 到 contract 的唯一登记；`provide` 是被核对的陈述 |
| 2 | `provide` 的 `path` | 相对于写这一行的 `plugin.py` 所在目录，写 `"impls"` |
| 3 | AITER 的 `versions` | op 版本（整数） |
| 4 | SGLang、vLLM 的版本比较 | 去掉 `+` 之后的本地标签再完全相等 |
| 5 | 截获函数 | “直接用 core 的实现即可。用法跟 @api 类似……plugin 不需要用 @api 定义 api，因为 api 就是宿主的 api，若命中则直接用内部 dispatcher 派发，若不命中直接调用宿主的 api。”我的理解：用 `make_api_fn`，为此删 `check_api_signature` 的三条桩函数检查 |
| 6 | 收集 | 第一次用到 `Plugin` 时自动做，没有公开的 `collect` |
| 7 | AITER 的启用入口 | SGLang、vLLM 的插件组里各加一个 entry point `aklr_aiter` |
| 8 | core 里旧的接入代码 | 删除，owner 以 code owner 身份批准 |
| 9 | 范围 | 接受 R4；本文只写 `Plugin`、`Provider` 和试点族 |
| 10 | 族的规则 | 族的 `plugin.py` 只声明本族的 contract；一个 target 的全部声明写在同一个 `plugin.py` 里 |
| 11 | 术语 | 第 2 节，记入 `GLOSSARY.md` |
| 12 | R16 | `Plugin` 不做预热 |
| 13 | vLLM 的校验和替换 | 在 `register` 被调用的当下做，每个进程都做 |
| 14 | R15 的 warning | `compile_or_warm_up_model` 返回后发一次 |
| 15 | AITER 的签名 | 取自私有的 `aiter.plugin._routed_ops` |
| 16 | AITER 的校验时机 | AITER 发现 provider 时。被决定 28 取代：第一次为这个 op 来问时 |
| 17 | 出错时 | AITER：“trace 记录 error，fallback 宿主的 api”。SGLang、vLLM 与 AITER 一致。见 3.7 |
| 18 | “参数表一致” | 3.6 的三条规则 |
| 19 | 返回 `None` | 按返回注解判断 |
| 20 | `Provider.enable` | 覆盖之前的 `configure()` |
| 21 | 记录方式 | 声明出错记 error 日志；`AKLR_TRACE=plugin` 保留，用 core 的 `Trace.plugin`；批准在受保护的 `dispatch/__init__.py` 加导出 `trace` |
| 22 | contract 语言 | 这次一起改，显式写 dtype 参数名。见第 5 节 |
| 23 | target 这个词 | 宿主里被接管的函数叫 target，不叫 api（“api 我认为叫做 target 更合适”）。`Plugin` 的参数名和方法名跟着改：`target`、`targets()`。硬件那个意思一律说 device target |

以下六条是 owner 在实施当天（2026-11-24）定的。

| # | 决定 | 结论 |
|---|---|---|
| 24 | 这次的范围 | “先不用写aklr/plugins具体的plugins和target，可以先搭好aklr/plugins的架子，具体的部分这是后续要移植的部分，先把核心写好”。做了 core、`Provider`、三个 `<host>/__init__.py` 和 entry point |
| 25 | core 测试用的 contract | 留一个最小真实族 `aklr/plugins/sglang/kimi_k3_tiny_gemm`，只有 `contract.py` 和 `reference.py` |
| 26 | 命名 | `Plugin` 的参数和属性叫 `host`，不叫 `ns`（“plugin不应该引用namespace的概念，应该是宿主名”） |
| 27 | `declare`、`provide` 怎么知道本族 | 不用 `sys._getframe`。收集时 `Plugin` 自己导入各族的模块，由它给出本族；`provide` 的 `path` 相对于族目录，写法不变 |
| 28 | AITER 的 target | op 名，不是全名（“aiter的provider target应该只有op的名字而不是fullname”）。取代需求第 9 节的决定 1 和本节的决定 16 |
| 29 | `register` 是否按宿主拆开 | 不拆，留在 `Plugin` 里（owner 提出解耦后又说“不用解耦了”） |

以下十一条是 owner 在移植族时（2026-11-24）定的。

| # | 决定 | 结论 |
|---|---|---|
| 30 | 族怎么来 | 每个宿主一个适配脚本，从 main 的代码直接生成（“写一个aklr/plugins/下aiter、sglang和vllm的各一个适配脚本，用脚本将main分支的代码直接适配生成。这样后续main分支更新也只需要更新即可”）。见第 12 节 |
| 31 | 现在从哪个版本生成 | #322 之前的 main（`c8e0e1534^`）：本分支的 core 还没有 #322 的 reference/baseline 类 |
| 32 | 一个 contract 经多个 aiter op 到达 | 每个 op 一个 contract，kernel JSON 在每个 op 下各放一份 |
| 33 | 只有一种布局的 target | 删掉 core 里“注册的 contract 第一个 layout variant 必须是 row_major”的检查，这种 contract 只有一个 variant，直接写它的 layout |
| 34 | schema 名里的张量名 | 宿主的参数名可以是大写（aiter 的 `XQ`），`name_at` 把它写成小写（我按已有的“schema 名全小写”测试改的，不是 owner 的原话） |
| 35 | 声明写在哪 | 不要单独的 `plugin.py`，`declare`、`provide` 写在族的 `contract.py` 最下面（“不需要一个plugin.py写在contract.py里最下面就行”）。取代决定 2、10、27 里说的 `plugin.py`；语句在文件被导入时记下、收集时按 contract 归族（不重新执行文件），是我为此定的做法 |
| 36 | 生成的 `contract.py` 的正文 | 写成构造调用，和手写的 `contract.py` 一个形式，不写成 `OpContract.model_validate({…})` 的数据（owner 在对话框里选的） |
| 37 | `declare` 和 `provide` | 合成一个 `declare`（“plugin的declare + provide可以合并成一个declare”）：schema JSON 的目录是它的第三个参数。取代决定 1、2 里的 `provide`；参数叫 `path`、可以不写，是我定的 |
| 38 | 生成脚本放哪 | 仓库根下独立的目录（“生成脚本独立放到 AKLR/ 一个目录下”），我取名 `codegen/`，不在 `aklr` 包里 |
| 39 | 数据从哪来 | 每次运行都从 main 取（“每次从main上拿数据codegen”）：不给版本时读 `origin/main` |
| 40 | 跟上 main | 把 origin/main（含 #322）合进本分支，生成器改成读 #322 之后的 contract 和 reference/baseline 类（owner 在对话框里选的）。取代决定 31：现在不给版本就能从 `origin/main` 生成 |

## 11. 核实过和没有核实的事

实施时在本机（MI300X，gfx942）跑过的：

- `tests/core`、`tests/plugins/aiter/test_provider.py` 在 SGLang 0.5.22 镜像里通过。`Plugin`、`Provider` 和第 5 节的改动各做过变异检查。
- 真实 SGLang 0.5.22：用一条临时声明，`make_api_fn` 能用在 `sglang.kernels.ops.gemm.kimi_k3_tiny_gemm` 上；`HookRegistry.apply_hooks` 装上了截获函数和 `ModelRunner.init_torch_distributed` 的 AFTER hook；命中返回 AKLR kernel 的结果，未命中回到 SGLang 的实现。kernel 是 CPU 上的假 kernel。
- 真实 vLLM 0.34.0（锁定镜像）：`vllm._custom_ops.wvSplitKrc` 被替换且签名核对通过；`register` 没有初始化 CUDA；用一个假 worker 驱动真的 `WorkerWrapperBase`，`init_device` 返回后 dispatch 启动，预热后的包装返回原值。
- AITER 锁定提交的 `aiter/plugin/__init__.py`（单独加载，没有装整个 AITER）：经 entry point 发现并由 `Provider.enable` 启用后，普通被路由的 op 和带 `fake` 的 custom op 都由 AKLR 服务，未命中走 native，`aiter.plugin.validate("aklr")` 没有报问题。

仍然没有核实的：

- 没有在 gfx950 上跑过，没有任何真实 kernel 经过这条路径，也没有跑过真的 AITER op。
- `torch.compile` 内联普通函数的行为是按 PyTorch 语义推断的。
- 在 worker 实例上包 `compile_or_warm_up_model` 只用假 worker 试过。vLLM 的 Ray executor 路径没有看。
- vLLM 里 `register` 会在每个进程导入 `vllm._custom_ops` 和 `worker_base`，包括为避免初始化 CUDA 而起的模型注册子进程。除了“不初始化 CUDA”之外的副作用没有验证。
- SGLang 的设备就绪点只核对了 `ModelRunner.__init__` 里的先后顺序，没有核对那时 `torch.cuda.current_device()` 已经是这个 runner 的设备。
- 锁定的 SGLang 0.5.23 镜像里的 aiter（`0.1.26.dev144`）没有 `aiter/plugin`，AITER 一侧只能在装了锁定提交的环境里测。
- 无 GPU 设备的容器里 `import aiter` 抛 `RuntimeError`（SGLang 0.5.22 镜像，aiter `0.1.23.dev183`，实施时再次见到）。影响是：在这种机器上开 `AKLR_AITER_PLUGIN=1`，`Provider.enable` 记一条 error，AITER 一侧不启用。锁定提交上是否相同没有跑过。
- 通用路径的 Python 开销没有量过。
- `harness` 和教程里提到 `out_dtype` 的地方没有清点。

## 附录 A：AITER 被路由的 40 个函数

锁定提交 `a6c759b1`。都是模块级函数，都没有 `*args`、`**kwargs`，都没有用 `supersedes`。声明里写的 target 是 `op` 一列（决定 28）；第一列是它所在的函数，供查找签名用。“guard 在外”指 `torch_compile_guard` 套在 `route` 外面，`Op` 在模块属性的 `__wrapped__` 上。

| 函数 | op（声明的 target） | op 版本 | `fake` | guard 在外 |
|---|---|---|---|---|
| `aiter.tuned_gemm._gemm_a16w16` | `gemm_a16w16` | 1 | | |
| `aiter.mla.mla_decode_fwd` | `mla_decode_fwd` | 1 | | |
| `aiter.mla.mla_decode_fwd_v4_nm` | `mla_decode_fwd_v4_nm` | 1 | | |
| `aiter.mla.mla_decode_fwd_v4_nm_ps` | `mla_decode_fwd_v4_nm_ps` | 1 | | |
| `aiter.fused_moe.fused_moe_` | `fused_moe` | 1 | | 是 |
| `aiter.ops.attention._pa_fwd_routed` | `pa_fwd` | 1 | | |
| `aiter.ops.attention._pa_decode_bf16_routed` | `pa_decode_bf16` | 1 | | |
| `aiter.ops.attention.pa_persistent_fwd` | `pa_persistent_fwd` | 1 | | |
| `aiter.ops.attention.mla_prefill_ps_asm_fwd` | `mla_prefill_ps` | 1 | | |
| `aiter.ops.attention.mla_reduce_v1` | `mla_reduce` | 1 | | |
| `aiter.ops.batched_gemm_op_a8w8.batched_gemm_a8w8_mxscale` | `batched_gemm_a8w8_mxscale` | 1 | | 是 |
| `aiter.ops.batched_gemm_op_a8w8.batched_gemm_a8w8_mxscale_bpreshuffle` | `batched_gemm_a8w8_mxscale_bpreshuffle` | 1 | | 是 |
| `aiter.ops.fused_qk_norm_rope_cache_quant.fused_qk_norm_rope_cache_quant_shuffle` | `fused_qk_norm_rope_cache_quant_shuffle` | 1 | 有 | |
| `aiter.ops.gemm_fusion.sp_qkv.op.sp_qkv_fused` | `sp_qkv_fused` | 1 | | |
| `aiter.ops.gemm_op_a8w8.gemm_a8w8_bpreshuffle` | `gemm_a8w8_bpreshuffle` | 1 | | 是 |
| `aiter.ops.gemm_op_a8w8.gemm_a8w8_blockscale` | `gemm_a8w8_blockscale` | 1 | 有 | |
| `aiter.ops.gemm_op_a8w8.gemm_a8w8_blockscale_bpreshuffle` | `gemm_a8w8_blockscale_bpreshuffle` | 1 | | |
| `aiter.ops.inverse_rope_group_quant.inverse_rope_group_quant` | `inverse_rope_group_quant` | 1 | | |
| `aiter.ops.mha.flash_attn_varlen_func` | `flash_attn_varlen` | 1 | | |
| `aiter.ops.mhc.mhc_pre` | `mhc_pre` | 1 | | 是 |
| `aiter.ops.mhc.mhc_post` | `mhc_post` | 1 | | 是 |
| `aiter.ops.mhc._mhc_fused_post_pre_routed` | `mhc_fused_post_pre` | 1 | 有 | |
| `aiter.ops.mla_sparse_prefill._mla_sparse_prefill_routed` | `mla_sparse_prefill` | 1 | | |
| `aiter.ops.mla_sparse_prefill._mla_sparse_prefill_v4_routed` | `mla_sparse_prefill_v4` | 1 | | |
| `aiter.ops.mla_sparse_prefill._mla_sparse_prefill_v41_routed` | `mla_sparse_prefill_v41` | 2 | | |
| `aiter.ops.quant.pertoken_quant` | `pertoken_quant` | 1 | | |
| `aiter.ops.quant._per_group_quant_routed` | `per_group_quant` | 1 | 有 | |
| `aiter.ops.rmsnorm._rms_norm_fwd_dispatch` | `rmsnorm` | 2 | | |
| `aiter.ops.rmsnorm.rmsnorm2d_fwd_with_add` | `rmsnorm_with_add` | 1 | 有 | |
| `aiter.ops.topk.topk_gating` | `topk_gating` | 1 | 有 | |
| `aiter.ops.topk.biased_grouped_topk` | `biased_grouped_topk` | 1 | 有 | |
| `aiter.ops.topk.top_k_per_row_decode` | `top_k_per_row_decode` | 1 | 有 | |
| `aiter.ops.triton.attention.kda.fused_recurrent_kda` | `fused_recurrent_kda` | 1 | | |
| `aiter.ops.triton.attention.kda.fused_recurrent_kda_packed_decode` | `fused_recurrent_kda_packed_decode` | 1 | | |
| `aiter.ops.triton.attention.pa_mqa_logits.deepgemm_fp8_paged_mqa_logits` | `fp8_paged_mqa_logits` | 1 | | |
| `aiter.ops.triton.gated_delta_net.causal_conv1d_prefill.causal_conv1d_split_qkv_triton_tile_fn` | `causal_conv1d_split_qkv` | 1 | | |
| `aiter.ops.triton.gemm.basic.gemm_a16w16._gemm_a16w16_routed` | `gemm_a16w16_activation` | 1 | | |
| `aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale._gemm_a8w8_blockscale_routed` | `gemm_a8w8_blockscale_out` | 1 | | |
| `aiter.ops.triton.gluon.mla_gluon._mla_paged_decode_routed` | `mla_paged_decode` | 1 | 有 | |
| `aiter.ops.triton.quant.fused_mxfp8_quant._fused_dual_rmsnorm_mxfp8_quant_routed` | `fused_dual_rmsnorm_mxfp8_quant` | 1 | 有 | |

## 12. 适配脚本

脚本在仓库根下的 `codegen/` 里（决定 38），不属于 `aklr` 包，测试在 `codegen/tests/`。

族不手写，由每个宿主的适配脚本从 main 生成（决定 30）。`python -m codegen.aiter [REV]` 读 main 在 `REV` 的 `aklr/ops`，重新生成 `aklr/plugins/aiter/` 下的全部族和 `tests/plugins/aiter/cases.py`；不给 `REV` 时用 `origin/main`。

| 文件 | 概念 | 类 | 公共方法 |
|---|---|---|---|
| `codegen/adaptation.py` | main 的一个 contract 按宿主 target 重切 | `Adaptation` | `contract`、`point`、`schema`、`reference_source`、`baseline_source` |
| `codegen/kernel_entry.py` | kernel 源码的入口函数 | `KernelEntry` | `retargeted` |
| `codegen/contract_source.py` | 写成 python 源码的 contract | `ContractSource` | `text` |
| `codegen/generation.py` | 一个宿主的族，由 main 生成 | `Generation` | `write` |
| `codegen/<host>.py` | 这个宿主的适配表 | 无 | `main`（S3 (a)：脚本入口，没有类拥有它），`ADAPTATIONS` |

一条 `Adaptation` 写明：main 的哪个 contract（`source`）、宿主的哪个 target 和版本、放进哪个族、main 的名字到 target 参数名的对照（`rename`）、target 固定下来的 attr 和布局（`pin`，固定后不再是 contract 的参数）、固定的布局里调用仍然带着的 fact（`facts`）、以及名字之外的差别（`override`，合并到重切后 contract 的 json 上）。`target` 为空表示没有宿主函数接它，contract 原样搬过来，只参加验证。

生成的内容：

| 文件 | 来源 |
|---|---|
| `<族>/contract.py` | main 的 contract 重切后，写成 `OpContract(…)`、`TensorSpec(…)` 这样的构造调用，默认值不写（决定 36）；最下面是每条有 target 的 `Adaptation` 一行 `declare`。写完后脚本会导入这个文件，核对它构造出的 contract 与重切的结果相等 |
| `<族>/reference.py`、`baseline.py` | 每个改过名的 contract 一个类，继承 main 的 reference/baseline 类：把 target 的调用换成 main 的名字去调父类，再把结果的名字换回 target 的。只是搬过来的 contract 直接用 main 的类 |
| `_main/<main 的族>/*.py` | main 的 reference、baseline 等模块原样拷贝，只改 import 路径 |
| `<族>/impls/**.json` | main 的 schema 文件：键和 kernel 入口的参数改成 target 的名字，被固定的参数在入口里绑成常量，workload 和测量值不变 |
| `tests/plugins/<host>/cases.py` | main 的 small Point 按同样的规则换名 |

描述文字里，main 的名字只在公式里（挨着运算符、括号，或者是单字母的名字）换成 target 的名字；被固定下来的参数在描述里仍会被提到（如“unless trans_b”），需要时在 `override` 里改写。

reference 和 baseline 用的是 main 自己的代码，所以 JSON 里的测量值仍然是对着同一个 baseline 量的。生成的文件开头写明“Generated by … do not edit”，重新生成时只删带这个标记的族。

现状（2026-11-24）：只有 aiter 的脚本。从 `origin/main`（`4415c9a6f`）生成 6 个族、13 个 contract、433 个 schema 文件；main 的 361 个 schema 文件里还有 123 个没有对应的 `Adaptation`（fused_moe、mla、mla_prefill、mla_absorb、quant、kda_conv、mega_moe、attention、situv2_a4w4_moe、mhc 的 fused_post_pre、gemm 的一个布局）。sglang 和 vllm 的脚本没有写。

还做不到的事：

- target 的张量形状与 main 的不同（转置、多一维）时，kernel 入口、reference、baseline 都要先变形。`Adaptation` 还没有这一项，所以 `mhc_fused_post_pre`、`per_group_quant`、`causal_conv1d_split_qkv`、`flash_attn_varlen` 这类没有做。
- 旧 route 靠环境变量或进程内状态决定接不接的（`AITER_FLYDSL_USE_HERD`、SiTUv2 的激活 dtype）在 contract 里写不出来。`biased_grouped_topk`、`topk_gating` 现在不看 HERD。
- 旧 route 把 `gemm_a16w16` 的三维 `A` 摊平成二维再派发，现在三维的调用回退到 aiter。
- 分成几个 rank 一起调用的 op（mega_moe）和 reference 自带 `check` 的 op 只能原样搬，改名的还没做。
