# vLLM 插件接入：vllm_route 与 Muse Spark 示例

AKLR 的 vLLM 插件对外只提供 `vllm_route` 声明 Hook。插件内部统一管理安装与回退，只修改明确声明的目标绑定，不修改 vLLM 源码或安装文件。扩展代码使用装饰器即可，Registry 与安装 helper 均为私有实现。

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

Worker 的补充 warmup 和 `roomy` 包装由插件内部安装，返回值与异常继续按原行为传递。现有六条 routes 均通过 `vllm_route` 接入；本页以 GEMM 演示公开声明，以 Muse 的 norm/experts 说明关联安装和原生回退。

`aklr.plugins.vllm.register()` 是 vLLM 插件加载入口，既有 `aklr.core.runtime.dispatch.register_vllm_plugin()` 是兼容启动入口。两者使用同一 bootstrap 编排：先校验配置，再收齐内部声明，最后统一安装。它们负责启动插件；新增计算 route 只需要使用下面的装饰器。在配置不变的前提下，启动入口的两种调用顺序及重复调用均幂等，包括设备已经初始化后的重复调用。阶段冻结后改变开关不属于热切换接口。

## 唯一公开 Hook 声明 API

下面摘自现有 [GEMM route](../aklr/plugins/vllm/layers/gemm.py)，展示将 vLLM 使用的 AITER GEMM 入口接到 AKLR：

```python
from aklr.core.runtime.dispatch import FallbackReason
from aklr.ops.gemm.api import gemm
from aklr.plugins.vllm.hook import vllm_route


@vllm_route(
    "vllm.model_executor.layers.utils.rocm_unquantized_gemm_impl",
    "aiter.tuned_gemm.tgemm.mm",
    versions=("0.33.0rc1.dev910",),
)
def _tgemm_mm(inp, weights, bias=None, otype=None, scale_a=None, scale_b=None, scale_c=None, out=None):
    if scale_a is not None or scale_b is not None or scale_c is not None:
        return FallbackReason.OUT_OF_CONTRACT
    return gemm(inp, weights, bias=bias, out_dtype=otype, out=out)
```

- 第一个参数 `caller` 用于安装时的源码检查。它不限制运行时调用者；任何查找到被替换 callee 的调用都会受影响。
- 后续 `callees` 是框架运行时实际查找的完整属性路径。只修改这些显式绑定，不自动传播其他模块的别名。
- `versions` 是经过测试的 vLLM 版本白名单。`group=...` 可将关联 routes 一起选择：任一成员不支持当前版本时整组跳过。
- route 的参数名称、顺序和 kind 必须与实际原 callable 一致。默认值与类型注解不参与兼容检查。classmethod route 包含 `cls`，singleton route 使用绑定后的签名，省略 `self`。
- 返回 `FallbackReason` 表示当前调用由原生实现处理；普通返回值，包括 `None`，表示已处理。计算异常继续抛出。`torch.compile` tracing 期间，包装层直接调用保存的原函数。

装饰器只收集声明，并原样返回 route 函数。声明模块由插件启动流程导入；设备就绪后，内部安装函数完成版本选择、目标及参数预检，然后安装整批 routes。扩展代码不需要另行调用 `registry.register` 或安装函数。

所有计算 routes 在设备就绪阶段安装，并独占各自 callee。同一目标出现冲突或任一目标预检失败时，整批不安装。内部的绑定、回滚、恢复和 Worker 生命周期协议见 [设计文档](vllm_plugin_design.zh.md)；这些协议不作为另一套扩展 API。

## Muse 算子如何接入

现有 [Muse routes](../aklr/plugins/vllm/models/muse_spark/api.py) 保留数学计算范围：

1. `MuseSparkTorchLinearMethod.apply` 继续调用 `gemm`。
2. `_defer_norm` 和 `_experts_apply` 标记相同的 `muse_spark.norm_experts` group；组内任一 route 不支持当前版本时整组跳过，任一目标检查失败时整批不安装。
3. 命中时，`muse_spark_moe` 的逐路未加权结果继续交给原生 finalize。
4. 参数不支持或 op miss 时，使用安装前保存的原 norm 与原 experts，完成一次原生归一化后调用原 experts。

Muse 内置适配器通过私有 `_original` helper 查询两个保存的原函数，协调延迟 norm 和 experts 的回退。这是内置适配器的内部协作，普通扩展 route 返回 `FallbackReason` 即可触发原生回退。原函数查询按 route/callee 直接索引，不扫描所有声明，也不依赖可能变化的 canonical 模块属性或 `__wrapped__` 链；受控恢复后会拒绝新查询，重新安装后使用新保存的原函数。

route/group 元数据存于同一声明结构中，日志区分 route 自身版本未测试与被同组成员阻止安装。安装期的版本、caller 和参数检查不会进入正常 kernel 调用路径。

## 验证

CPU 协议与宿主替身测试：

```bash
python -m pytest tests/core/runtime/test_hook_registry.py \
  tests/plugins/vllm \
  tests/core/runtime/test_dispatch.py::TestVllmPlugin -q
```

Muse 测试使用实际 AKLR route 函数，配合 CPU Tensor 和原生宿主替身，检查 `NOT_WARMED`/`NOT_QUALIFIED` 回退、归一化只执行一次、原入口不受全局别名变化影响、命中时保持逐路输出交接，以及缺少 experts 时不会只安装 norm。

私有 Registry 测试检查重复安装、独占冲突、无绑定 callable、继承 descriptor、多层类与 singleton 同批安装、显式别名、跨阶段冲突、修改后抛异常的提交回滚和第三方修改后的恢复保护。facade 测试覆盖 classmethod 的实际调用、原生回退与错误签名的整批拒绝。完整 Worker 测试通过公开 `vllm_route` 声明计算入口，调用真实内部安装流程，覆盖两个启动入口的调用顺序、设备就绪时机、版本选择和原生回退。

这些测试不替代实际 vLLM/AITER 镜像的 ABI 检查或 GPU 集成。目标镜像仍需验证真实 Muse 的精度、并行归约、eager/capture/replay 和端到端性能。

## 保留的边界

- 本版规范声明、安装与回退，仍保留 Muse deferred norm 的跨调用协议、Tensor 属性和全局缓存；完整 module 适配需要另行设计。
- 所有 route 声明应在设备就绪安装前收齐；安装完成后不支持新增或改变 routes，也不提供 serving 期间的热切换接口。
- 只修改明确声明的 callee 属性，不追踪闭包、局部变量、容器引用、已保存的 bound method 或已编译图。
- 回滚只恢复私有 Registry 管理的属性写入，不恢复 import 副作用、GPU 资源或已执行的业务计算。内部恢复流程检查绑定所有权，但自定义 setter 本身失败时仍可能恢复不完整。
- 内部安装状态与 dispatcher、Muse 缓存及插件 `_install_once` 等业务状态有各自的生命周期；受控测试需要隔离并管理这些状态。
- 现有补充 warmup 没有补齐 MoE pre-capture 覆盖，仍需在目标镜像验证。

逐项审查修复与验证记录见[2026-11-10 修复说明](vllm_review_fixes_20261110.zh.md)。kernel 数值修复及混合后端的覆盖说明见[MiniMax M3.1 kernels](minimax_m31_kernel_coverage.zh.md)。
