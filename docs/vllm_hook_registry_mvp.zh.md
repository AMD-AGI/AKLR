# vLLM HookRegistry MVP：Muse Spark 示例

该 MVP 在 AKLR 内统一管理现有 vLLM 运行时包装，不修改 vLLM 源码或安装文件。它沿用 `vllm.general_plugins` 入口及现有开关，默认只修改明确声明的目标绑定。

## 运行入口

在已配置好的 Muse Spark vLLM 环境中，继续使用原启动命令，并设置：

```bash
export AKLR_VLLM_PLUGIN=1
```

当前计算 routes 的版本白名单仍为 `0.33.0rc1.dev910`；没有新增 `AKLR_VLLM_MODE`、Worker 类参数或 SGLang 依赖。禁用 routes 时仍保留既有 core runtime 初始化行为。

安装有两个阶段：

```text
vllm.general_plugins
  → 校验配置
  → bootstrap：安装 Worker hooks

原 Worker.init_device 成功
  → DispatcherRegistry.init_once()
  → device_ready：校验并安装计算 routes
  → 后续模型加载、预热和执行
```

Worker 的补充 warmup 和 `roomy` 包装也由 Registry 安装，返回值与异常继续按原行为传递。现有六条 routes 均通过兼容 facade 接入；本次以 Muse 的 norm/experts 作为关联安装和原生回退的案例。

公开的 `aklr.core.runtime.dispatch.register_vllm_plugin()` 和 `aklr.plugins.vllm.register()` 使用同一 bootstrap 编排：先校验配置，再收齐 core 与已启用插件的声明，最后统一提交。在配置不变的前提下，两种调用顺序及重复调用均幂等，包括设备已经初始化后的重复注册；不会为了补充声明而临时恢复已安装的 hooks。阶段冻结后改变开关不属于热切换接口。

## 最小 API

[HookRegistry](../aklr/core/runtime/hook_registry.py) 只依赖标准库，提供：

```python
from aklr.core.runtime.hook_registry import (
    KEEP_RESULT,
    HookRegistry,
    HookSpec,
    HookType,
)

registry = HookRegistry.default()

# 应在所属阶段 apply 前收齐声明；这里仅演示接口。
def after_call(result, *args, **kwargs):
    return KEEP_RESULT

registry.register(HookSpec(
    hook_id="example.observer",
    target="example.Model.forward",
    hook=after_call,
    hook_type=HookType.AFTER,
    phase="device_ready",
    source=__name__,
))
```

- `register(spec)` 收集声明；同 ID 同定义幂等，改变定义会报错。
- `apply(phase)` 先解析并检查完整批次，再安装；失败回滚本批属性写入。成功后阶段冻结。可选的 `HookSpec.validate(original)` 在预检时检查实际会被调用的原函数，以抛异常表示不兼容；它不进入调用路径。
- `original(hook_id)` 返回安装前保存的 callable；它与 AROUND 收到的内层 `next_fn` 不同。
- `report()` 返回声明与安装状态，包括目标、来源、组、阶段、别名、`exclusive` 和 PID。应在相应 worker 内读取，不代表 kernel 命中或 graph 覆盖。
- `restore(phase=None)` 用于启动失败/受控测试清理，检查绑定所有权后恢复属性；不支持 serving 期间热切换。

支持 BEFORE、AFTER、AROUND 和函数 REPLACE。AFTER 返回 `KEEP_RESULT` 才保持原结果；返回 `None` 表示显式将结果变为 `None`。非 REPLACE hooks 按 `(order, hook_id)` 从内到外构建。

`exclusive=True` 的 hook 独占该目标：任何其他 BEFORE、AFTER、AROUND 或 REPLACE 都会使预检失败，且整批不写入。当前计算 routes 均使用该规则；需要组合的通用 hooks 必须显式使用非独占声明。包装闭包在安装时按 hook 类型构建，运行时不再逐次判断类型。

支持普通方法、直接/继承的 classmethod 和 staticmethod，以及现有 AITER singleton 和 `ModuleType` 子类实例的方法。类上的 `math.sqrt`、`functools.partial`、已绑定方法和 callable 对象保留不额外绑定 `self` 的行为。singleton 实例方法保存绑定后的原 callable，恢复时删除本次添加的实例属性以恢复类查找。classmethod route 必须声明 `cls`，singleton route 则按其绑定后的签名省略 `self`；签名检查使用 Registry 解析得到的同一个原 callable。

同批安装父类、继承方法的子类以及 singleton 时，各目标独立包装批次开始前解析到的原 callable。提交从更具体的目标到祖先，恢复顺序相反，结果不依赖目标路径名称。例如同批声明 `Base.fn` 和未覆盖 `fn` 的 `Child` 时，`Child.fn` 包装原函数，不隐式叠加本批 `Base.fn` 的 hook；若只声明 `Base.fn`，子类仍按正常继承查找使用它。类整体 REPLACE、异步/generator、自定义 descriptor，以及仅由 metaclass 提供的目标不在本版支持范围内；后者在预检时明确拒绝。

## Muse 算子如何接入

现有 [Muse routes](../aklr/plugins/vllm/models/muse_spark/api.py) 保留数学计算范围：

1. `MuseSparkTorchLinearMethod.apply` 继续调用 `gemm`。
2. `_defer_norm` 和 `_experts_apply` 标记相同的 `muse_spark.norm_experts` group；组内任一 route 不支持当前版本时整组跳过，任一目标检查失败时整批不安装。
3. 命中时，`muse_spark_moe` 的逐路未加权结果继续交给原生 finalize。
4. miss 时，使用安装前保存的原 norm 与原 experts，完成一次原生归一化后调用原 experts。

回退不再从可能被修改的 canonical 模块路径动态导入 norm，也不再依赖当前全局 `apply.__wrapped__`：

```python
from aklr.plugins.vllm.hook import original

normalized = original(_defer_norm)(
    hidden_states, weight, eps, gain_center, False, topk_ids.shape[1]
)
return original(_experts_apply)(self, output, normalized, ...)
```

这段是省略完整参数的示意。fallback 按 route/callee 直接索引，不扫描所有声明；每次仍通过 Registry 取得原函数，以便 `restore` 后拒绝旧句柄查找、重新安装后使用新保存的原函数。route/group 元数据存于同一声明结构中，日志区分 route 自身版本未测试与被同组成员阻止安装。安装期检查与 alias 扫描不会进入正常 kernel 调用路径。

默认不扫描全部 `sys.modules` 传播别名。通用 Registry 可通过 `HookSpec.aliases` 显式指定模块属性，并检查其仍指向目标原函数；主目标与 aliases 一起提交、一起恢复。

## 验证

CPU 协议与宿主替身测试：

```bash
python -m pytest tests/core/runtime/test_hook_registry.py \
  tests/plugins/vllm \
  tests/core/runtime/test_dispatch.py::TestVllmPlugin -q
```

Muse 测试使用实际 AKLR route 函数，配合 CPU Tensor 和原生宿主替身，检查 `NOT_WARMED`/`NOT_QUALIFIED` 回退、归一化只执行一次、原入口不受全局别名变化影响、命中时保持逐路输出交接，以及缺少 experts 时不会只安装 norm。

Registry 测试还检查重复安装、独占冲突、无绑定 callable、继承 descriptor、多层类与 singleton 同批安装、显式别名、跨阶段冲突、修改后抛异常的提交回滚和第三方修改后的恢复保护。facade 测试覆盖 classmethod 的实际调用、原生回退与错误签名的整批拒绝。完整 Worker 测试覆盖两个公开入口的调用顺序，没有把 core 注册整体 mock。

这些测试不替代实际 vLLM/AITER 镜像的 ABI 检查或 GPU 集成。目标镜像仍需验证真实 Muse 的精度、并行归约、eager/capture/replay 和端到端性能。

## 保留的边界

- 本版规范安装与回退入口，仍保留 Muse deferred norm 的跨调用协议、Tensor 属性和全局缓存；它不是此前完整 module 适配方案的实现。
- 同一实际目标、alias 或关联组不能跨阶段；阶段提交后不支持新增或改变 hooks。
- 显式 aliases 仅覆盖模块属性，不处理闭包、局部变量、容器内部引用、已保存的 bound method 或已编译图。
- 回滚只恢复 Registry 管理的属性写入，不恢复 import 副作用、GPU 资源或已执行的业务计算；预检失败的声明仍留在 Registry，修正错误声明需在受控启动/测试中重建 Registry。
- `restore` 在写入前检查整个所选批次的所有权；任一绑定被第三方改写会使恢复失败，不会留下已恢复一半的关联 hooks。
- `restore` 不重置 dispatcher、Muse 缓存或插件 `_install_once` 等业务状态；测试需要使用隔离实例并自行管理这些状态。
- 新 Registry 不会补齐 MoE pre-capture warmup；原补充 warmup 的时机和范围保持不变。

逐项审查修复与验证记录见[2026-11-10 修复说明](vllm_review_fixes_20261110.zh.md)。kernel 数值修复及混合后端的覆盖说明见[MiniMax M3.1 kernels](minimax_m31_kernel_coverage.zh.md)。
