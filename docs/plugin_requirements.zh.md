# AKLR Plugin 需求说明

依据：owner 在 `req.txt` 里的描述，以及三个宿主库自己的插件源码（AITER `aiter/plugin/__init__.py`、SGLang 0.5.22 `sglang/srt/plugins/`、vLLM 0.34 `vllm/plugins/`）。库里旧的接入代码不作依据。本文只写 Plugin 要解决什么问题和各部分的边界，不写实现。2026-11-24 实施当天 owner 改了几处（宿主名叫 `host`、AITER 的 target 用 op 名、`path` 相对于族目录），本文已按它们改过，出处见设计文档第 10 节的决定 24–29。

## 1. 一句话

族作者写下“宿主库的某个 target（字符串：SGLang、vLLM 里是函数全名，AITER 里是 op 名）由这个 contract 描述，它的 kernel 是这些 schema JSON”，注册就完成了。安装 aklr 之后，宿主启动时调 AKLR 的 hook 函数，由 `Plugin` 统一把每个声明过的 target 截获下来。调用到达时，生成 key、判断命中、派发、回退都由 core 的 dispatcher 完成，族里没有任何手写的派发函数。

## 2. 结构

| 文件 | 概念 | 类 | 公共方法 |
|---|---|---|---|
| `aklr/core/plugin.py` | 一个宿主库名下的注册 | `Plugin` | `declare`、`register`、`targets`、`dispatchers` |
| `aklr/plugins/<host>/__init__.py` | 一个宿主库的 `Plugin` 实例 | 无 | 无（`plugin = Plugin("<host>")`） |
| `aklr/plugins/<host>/…/<族>/contract.py` | 一个族的契约 | 无 | 无 |
| `aklr/plugins/<host>/…/<族>/contract.py` 的最下面 | 一个族的声明 | 无 | 无（只有 `declare` 语句；没有单独的 `plugin.py`，也没有单独的 `provide`） |
| `aklr/plugins/aiter/provider.py` | AKLR 在 AITER 里的 kernel provider | `Provider` | AITER 协议规定的成员，本文不展开 |

`<host>` 是宿主名：`aiter`、`sglang`、`vllm`。没有顶层函数。方法名是建议，签名在设计文档里定。

## 3. 参照 torch 的 `Library`

| torch | AKLR | 需求 |
|---|---|---|
| `Library(ns, kind)` | `Plugin(host)` | 一个宿主库一个 `Plugin`，以宿主名区分（不叫命名空间）。三个宿主库用同一个类，没有按宿主分的子类 |
| `lib.define("name(args) -> ret")` | `plugin.declare(target, contract, versions=…)` | `target` 是宿主库里一个具体函数的全名字符串（AITER 里是 op 名），`contract` 是它的 `OpContract`，`versions` 是这条声明适用的版本（torch 没有对应物，见 4.2） |
| 重载（`div.Tensor`、`div.Scalar`） | 同一个 `target` 多次 `declare` | 调用时按声明顺序尝试，第一个命中的生效 |
| `lib.impl(name, fn, dispatch_key)` | `plugin.declare` 的 `path` 参数 | `path`（相对于族目录）下的 schema JSON 是这个 target 的 kernel。设备和 Point 写在 JSON 里，相当于 dispatch key |
| 全局 dispatcher 按 key 选 kernel | core 的 `KernelDispatcher` | `Plugin` 不重写任何调度逻辑，也不新增调度类 |
| 没有 kernel 时走 fallback | 未命中时走宿主自己的实现 | 回退不需要声明 |
| 重复注册报错 | 同上 | 同一个 (target, contract) 声明两次是错误 |
| `torch.ops.ns.name(...)` | 没有 | `aklr/plugins/*` 不导出任何可调用的 api，用户只调宿主库自己的函数 |

## 4. Plugin 要覆盖的问题

### 4.1 声明

- R1 一条声明由三样组成：target（R2）、contract、schema JSON。族里不再有别的注册代码：没有 route 函数、selector、`@api` 函数、op 函数。
- R2 SGLang、vLLM 的 target 是能导入并解析到一个函数的点分路径。AITER 的 target 是它的 op 名：AITER 的 provider 协议问的就是 op 名。
- R3 contract 的参数表就是目标函数的参数表，中间没有适配层。对不上是 contract 或 JSON 的 bug，在那里修。
- R4a 每条声明必须写明它适用的版本（`versions`），因为宿主的 target 会有不兼容的改动。版本是逐个列出的、验证过的版本，不支持范围：新版本可能就是改动发生的那一版，没验证过就不该命中。
  - SGLang、vLLM：宿主的发行版本字符串，如 `("0.5.22", "0.5.23")`。
  - AITER：AITER 给每个被路由的 op 编的版本号（整数），它正是 AITER 用来标记不兼容改动的。
- R4b 同一个 target 可以按版本声明不同的 contract：宿主改了签名或语义之后，旧 contract 留给旧版本，新 contract 声明给新版本。
- R4 由此得出的范围：只有参数是张量和标量的函数能被声明。输入藏在 `self` 属性里、或 kernel 之外还有别的步骤的方法不在范围内，要换成更窄的目标函数。

### 4.2 注册（安装后自动完成）

- R5 用户只需 `pip install aklr` 并打开开关。宿主的 entry point 调 AKLR 的 hook 函数，这个 hook 函数属于 `Plugin`（`Plugin.register`），不再放在 `aklr/core/runtime/dispatch`。
- R6 hook 函数做三件事：收集 `aklr/plugins/<host>/` 下全部声明；校验；把每个 target 截获下来。新增一个族不需要改 hook 函数、entry point 或任何清单。
- R7 校验在注册时完成。出错的声明不生效，对应的 target 走宿主自己的实现，并用 error 级别的日志指出是哪条声明；进程照常启动，三个宿主相同。要核对的错误：target 解析不到；contract 与目标函数签名不一致；`provide` 的路径不存在；JSON 不属于已声明的 contract。AITER 的 op 在 AITER 第一次为它来问时校验：op 名解析不出模块，而它的一部分模块要到用时才导入。
- R7a 运行中的版本不在某条声明的 `versions` 里：这条声明不生效，该 target 走宿主自己的实现，并在日志里列出。这不算错误，进程照常启动。R7 的签名核对只对版本匹配的声明做，两者互补：版本管“没验证过的不碰”，签名管“验证过的版本里写错了”。
- R8 hook 函数在宿主的每个进程里都会被调用，包括没有 GPU 的进程，可能被调多次。它必须幂等，自己不初始化 CUDA；core 的 dispatch 在设备就绪后再启动（`DispatcherRegistry.init_once`）。
- R9 开关沿用现有的：`AKLR_DISABLE`、`AKLR_SGLANG_PLUGIN`、`AKLR_VLLM_PLUGIN`、`AKLR_AITER_PLUGIN`。关掉时什么都不截获。

### 4.3 截获与派发

- R10 截获之后一次调用的流程固定：实参按参数名交给这个 target 的 `KernelDispatcher`，命中则返回 kernel 的结果，全部未命中则调宿主原函数。key 的生成、命中判断、输出分配、运行 kernel、计数和 trace 都是 core 现有逻辑。
- R11 `Plugin` 在这条路上只做不含判断的三件事：位置实参对到参数名；调用方没传的参数补默认值（先用目标函数的，再用 contract 的）；目标函数返回 `None` 时返回 `None`。
- R12 `torch.compile` 追踪期间不运行 kernel，直接走原函数。
- R13 dispatch 启动之前、关闭之后，被截获的 target 行为与没有 AKLR 时相同。
- R14 kernel 开始运行后抛出的异常原样传出，不再回退到原函数。

### 4.4 各宿主的截获手段

| 宿主 | 宿主提供什么 | Plugin 怎么截获 | 谁回退 |
|---|---|---|---|
| SGLang | entry point 组 `sglang.srt.plugins`，`HookRegistry` 按字符串路径挂 hook | 每个 target 注册一个 AROUND hook | AKLR 调原函数 |
| vLLM | entry point 组 `vllm.general_plugins`，没有 hook 机制，没有“设备就绪”回调 | 自己替换 target 所指的模块属性；自己找设备就绪的时机 | AKLR 调原函数 |
| AITER | entry point 组 `aiter.plugins`，kernel provider 协议 | 不截获，见第 5 节 | AITER |

vLLM 的两个已知限制要在需求里接住：

- R15 被更早取走的引用（`from X import Y`）替换不到；vLLM 默认用 `torch.compile`，从被编译的 forward 里直接调用的普通函数会被内联，也不会被服务。一个声明过的 target 到预热结束时从未到达过 dispatch，要告警，不能静默失效。只在追踪时进过截获函数不算到达。
- R16 core 在捕获期间拒绝没有 eager 跑过的 Point。vLLM 0.34 在捕获每个 CUDA graph 前会用同样的形状先 eager 跑一次前向，所以 Plugin 不需要补预热。声明新的宿主版本时要重新核对这一点。

## 5. AITER：kernel provider 与 Plugin 的边界

AITER 的 kernel provider 与本文的概念一一对应，只是控制方向相反：SGLang、vLLM 是 AKLR 截获调用，AITER 是它来问 provider。

| AITER | AKLR | 
|---|---|
| `@route(op=…, version=…)` 的 op 名 | `declare` 的 `target` |
| provider 的 `op_versions` | `Plugin` 里声明过的 target 的集合 |
| `select(op, args, kwargs)`：只问不跑 | `KernelDispatcher.serving`：生成 key，判断命中 |
| `KernelCandidate.run` | `KernelDispatcher.dispatch` |
| 没有候选时运行 native | 未命中时走原函数 |
| `configure(enable=[…])` | `AKLR_AITER_PLUGIN` |

需求：

- R17 `Plugin` 上没有任何 AITER 协议的成员（`select`、`op_versions`、`KernelCandidate`…），`aklr/core/plugin.py` 不 import `aklr.plugins.aiter`。
- R18 AITER 族的声明与另外两个宿主写法相同（R1–R4），声明在 `Plugin("aiter")` 上。`Provider` 不收声明，也不含查表或运行 kernel 的逻辑。
- R19 `Provider` 只靠 `Plugin` 的公共查询就能写出来。`Plugin` 为此提供两样：声明过的 target 及其版本（`targets`），和某个版本下服务一个 target 的各 contract 的 `KernelDispatcher`（`dispatchers`）。版本对 `Plugin` 是不透明的值，只做“在不在 `versions` 里”的比较；运行中的版本是多少由接入方给出（SGLang、vLLM 读已安装的发行版本，AITER 由 `Provider` 给出 op 版本）。这两样对三个宿主通用，`Plugin.register` 自己也用它们。
- R20 target 就是 AITER 的 op 名，不需要换算。op 的签名和运行中的 op 版本由 `Provider` 向 AITER 取（签名在 `aiter.plugin._routed_ops` 里，版本随 `select` 的 `op` 传来）。
- R21 AITER 规定“安装不等于启用”，启用由框架做。在 SGLang、vLLM 里由 AKLR 在 `AKLR_AITER_PLUGIN=1` 时替框架启用；单独使用 AITER 的用户自己调 `aiter.plugin.configure`。这段代码在 `Provider` 里。

## 6. 三个宿主各一条声明

```python
# aklr/plugins/sglang/kimi_k3_tiny_gemm/contract.py 的最下面
plugin.declare("sglang.kernels.ops.gemm.kimi_k3_tiny_gemm", KIMI_K3_TINY_GEMM, "impls", versions=("0.5.22", "0.5.23"))

# aklr/plugins/vllm/wv_split_krc/contract.py 的最下面
plugin.declare("vllm._custom_ops.wvSplitKrc", WV_SPLIT_KRC, "impls", versions=("0.34.0",))

# aklr/plugins/aiter/rms_norm/contract.py 的最下面
plugin.declare("rmsnorm", RMSNORM, "impls", versions=(2,))      # AITER：op 名
```

## 7. 不做

- 适配层、route 函数、selector、`@api`、op 函数。
- `aklr/plugins/*` 的对外 api；对旧路径（`aklr/ops`、`.api`、`.contract` 转发）的兼容。
- 新的调度类；按宿主分的 `Plugin` 子类。
- 版本范围（`>=0.5.22` 这类写法）。
- 没有宿主 target 可服务的 contract 不写 `declare`，只参加验证。

## 8. 约束

- `aklr/core/runtime/`、`aklr/core/schema/` 受保护，改动要 code owner 批准。
- 当前 HEAD 上 `aklr/ops`、`aklr/plugins` 已删除，但 core 还引用它们：`locate.op_packages()` 导入 `aklr.ops`，`register_vllm_plugin` 导入 `aklr.plugins.vllm._register`，`pyproject.toml` 的三个 entry point 指向不存在的模块。这些都要随本需求处理。
- core 规定族目录名等于 contract 的 `op_type`，并且在整个 `aklr/plugins` 下唯一。“每个 target 一份 contract”之后族会变多。
- contract 语言的限制会直接变成能服务的范围。没有“任意对象、忽略”这种参数：这类参数不写进 contract，只服务它取默认值的调用。输出 dtype 的参数名原来写死为 `out_dtype`（AITER 叫 `otype`、`dtype`），本次改成由 contract 指定，见设计文档第 5 节。

## 9. 需要你决定

这六条已在 2026-11-24 定下，结论见 [plugin_design.zh.md](plugin_design.zh.md) 第 10 节。下面保留当时的问题。

1. （已被取代：2026-11-24 实施时 owner 定为 AITER 的 target 用 op 名，见设计文档决定 28。）**AITER 的全名指被 `route` 装饰的函数，其中一部分是私有名**（`aiter.ops.rmsnorm._rms_norm_fwd_dispatch`，不是公开的 `aiter.rms_norm`）。按你的描述本文用全名。锁定提交上被路由的函数有 40 个，其中 13 个是私有名。代价是 AITER 改名时声明失效（按 R7 记 error 并回退）。
2. **`declare`、`provide` 是不是 core 读取的唯一登记**。现在 core 自己发现 contract（各族 `contract.py` 的 `CONTRACTS`）并扫描 `impls/`。
   - 建议：是。否则作者要写两遍（`CONTRACTS` 和 `declare`），与“声明即完成注册”不符。要改 `aklr/core/contract/registry.py` 和 `locate.py`，不在受保护目录里，但超出了“core 只动 `plugin.py`”。`locate.py` 因 `aklr.ops` 已删除本来就得改。
   - 另一种：core 不变，`declare`、`provide` 只做登记和一致性核对。
3. **`provide` 的 `path` 写到哪一层**。建议写到 `impls`，每个 target 一行，新增 kernel 不用改声明。
4. **AITER 的 `versions` 用 op 版本还是 aiter 的发行版本**。建议 op 版本：它是 AITER 自己为不兼容改动编的号，provider 协议的 `op_versions` 要的也是它；AITER 多按提交锁定，发行版本区分不了两个提交。代价是三个宿主的 `versions` 类型不同（AITER 是整数，另两个是字符串）。
5. **core 里现有的接入函数**（`register_sglang_plugin`、`register_vllm_plugin`、`_vllm_hook_registry`）在受保护目录里，按 R5 不再被调用。删除需要你批准；不批准就留着不调用。
6. **R4 的范围你是否接受**。上一轮读下来，AKLR 当时服务的约 21 个 AITER 目标里 16 个、SGLang/vLLM 约 38 个里 10 个是张量级函数，其余要换目标函数，kernel 边界跟着变。AITER 被路由的函数共 40 个。

## 10. 没有核实的事

没有运行任何代码。宿主一侧我这次复核了：AITER 的 `KernelProvider` 协议、`route`、`_dispatch`；SGLang、vLLM 加载插件的函数和调用点。以下没有核实：

- R16 的预热、带 `fake` 的 AITER op 这两条已在设计阶段读源码核实：vLLM 0.34 自己会预热；带 `fake` 的 op 带 `__aiter_plugin_op__`，但签名读不到，只能从 AITER 私有的 `_routed_ops` 取。其余未核实的事见设计文档第 11 节。
- 决定 6 里的条数来自上一轮读代码的分类，那些代码已随 `aklr/ops`、`aklr/plugins` 删除。
