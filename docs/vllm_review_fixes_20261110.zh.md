# 2026-11-10：分支审查逐项修复

本记录对应 `dev/minimax-m31-vllm-plugin`，审查时的 HEAD 为 `c1eac07cb0dc1dbc69ec6e487c1d6ee61149b983`，范围包括当时的 kernel 提交与尚未提交的 HookRegistry/Muse MVP。六项已确认问题均已实施代码修复；下表保留原审查的 R1–R5、K1 编号，说明行为和验证证据。

| 审查项 | 修复后的行为 | 回归证据 |
| --- | --- | --- |
| R1：类属性 callable 被额外绑定 `self` | 对原本不绑定实例的 callable 使用 `staticmethod` 包装，保留 builtin、partial、已绑定方法及 callable 对象的调用约定；恢复后还原对象身份与继承查找。 | 直接/继承类属性各四类 callable、类/实例调用、显式模块 alias 均覆盖。 |
| R2：父子类批量安装依赖路径排序 | 全批解析完成后按继承关系先安装更具体的目标，再安装祖先；每个目标使用本批开始前的原函数，逆序恢复。 | 正常/反向别名、instance/classmethod/staticmethod、多层继承与 singleton；父类赋值已修改再抛异常时仍完整回滚。 |
| R3：classmethod 的签名检查与调用不一致 | `HookSpec.validate` 使用 Registry 实际保存的 callable 做安装期 ABI 检查。classmethod route 包含 `cls`，singleton 使用已绑定签名。 | 正确 route 的类、子类、实例调用及原生回退/恢复通过；缺少 `cls` 的 route 在写入前整批拒绝。 |
| R4：公开 core 注册提前冻结 bootstrap | 两个公开入口共享插件的 bootstrap 编排，先收齐 core 和已启用插件声明，再提交。 | 配置保持不变时，两种入口顺序及设备初始化前后的重复调用均保持同一 wrapper；禁用、无效配置、缺失目标和设备失败均覆盖。 |
| R5：exclusive route 可被其他 hook 跳过 | 同一目标存在 `exclusive=True` 时拒绝任何其他 hook；`report()` 显示独占标志。 | BEFORE、AFTER、AROUND、REPLACE 在两种排序位置下均在预检失败，不写入目标。 |
| K1：FP24 partial 压缩引入 Inf/NaN | 大 GEMM producer 和 fold 使用完整 FP32 中间结果与四字节布局，删除 pack/unpack。 | CPU 执行真实内嵌写入与 fold 代码，保留 `+FLT_MAX/-FLT_MAX` 相消为零及相消后一个 FP32 ulp；旧实现会产生 NaN。真实 gfx950 kernel 测试已加入，当前环境跳过。 |

Registry 修复后的独立复核还检查了菱形继承与 singleton，并发现两个相邻的既有绑定边界，已一并处理：`ModuleType` 子类方法按绑定后的 module receiver 调用；只存在于 metaclass 的方法在预检时拒绝，避免安装后改变类及实例两套查找语义。两者均有回归测试。

实现与用法见 [HookRegistry](../aklr/core/runtime/hook_registry.py)、[route facade](../aklr/plugins/vllm/hook.py) 和 [MVP 指南](vllm_hook_registry_mvp.zh.md)。

## 冗余、兼容处理与可观测性

- 小 MoE 删除约 78 行不可达的 Torch 辅助/计算代码，保留全部 39 个小 token workloads。直接以至少 256 tokens 调用会明确报错；大 token workloads 仍由独立 schema 负责。
- Registry 在安装时构造各 hook 类型的专用闭包，移除每次调用的类型分支。facade 的原函数查找按 route/callee 直接索引，继续检查 Registry 生命周期；restore 后不返回已失效的旧原函数。
- route、版本及 group 使用同一份声明结构。兼容性预检直接检查将用于安装的原 callable，删除 facade 中重复的 callee 解析；AITER 首次导入替换 `sys.modules` 的兼容处理集中保留在 Registry。
- 跳过日志区分自身版本未测试与被同组成员阻止；已有 WARNING 安装摘要与 INFO 逐目标日志继续保留。
- 明确记录 5 个小 GEMM 和另外 6 个大 GEMM workloads 使用 Torch。真实入口的后端选择已由 CPU 测试遍历核对，未把这些 Point 的 AKLR 调用计作 FlyDSL 命中。

三个有源码变化的 JSON 已按 authored content 重新生成文件名，workloads、设备条件和契约均保留：

| 修复前文件名 | 修复后文件名 |
| --- | --- |
| `cf4fb32f9a5b131e.json` | [f6791d637fe1bf47.json](../aklr/ops/gemm/impls/flydsl/gemm_n6144_k30720/f6791d637fe1bf47.json) |
| `cf656a46ea5f73c4.json` | [16e11e5d19578ead.json](../aklr/ops/gemm/impls/flydsl/gemm_n6144_k30720/16e11e5d19578ead.json) |
| `90eaf5c456ed45f4.json` | [40b0de61e4f6959c.json](../aklr/ops/fused_moe/impls/flydsl/fused_moe_swiglu_has_swiglu_limit_shuffled_16x16_model_dim6144_n_experts128_topk4_inter_dim768/40b0de61e4f6959c.json) |

原文件均没有测量记录，修复后仍没有填写未经验证的 speedup 或 accuracy。默认未测量 Point 的调度门槛继续有效。[kernel 覆盖说明](minimax_m31_kernel_coverage.zh.md) 记录完整后端分配和测量要求。

## 复核后保留的设计

第三方改写任一绑定时，`restore` 仍会在恢复整个所选批次前失败。这是明确的所有权保护，避免关联 hooks 只恢复一部分。

属性安装与声明集合是不同生命周期：预检或提交失败后，属性未安装或已回滚，声明仍保留；错误声明需在受控启动/测试中重建 Registry。阶段冻结、`KEEP_RESULT` 的 AFTER 语义与严格 ABI 检查均保留，并已写入使用说明。

Muse deferred norm、Tensor 属性、业务全局缓存、`roomy` 和既有 warmup 时机属于 MVP 保留的计算协议。本轮修复没有重新设计 module 边界、EP 归约或 graph replay 协议，也没有新增已确认的安全漏洞修复项。

## 验证

相关测试命令：

```bash
python -m pytest tests/core/runtime tests/plugins tests/core/contract \
  tests/core/platform tests/ops/test_contracts.py \
  tests/ops/test_minimax_gemm.py tests/ops/test_minimax_moe.py -q
python -m aklr export check
```

最终结果为 **408 passed、171 skipped**。12 个修改或新增的 Python 文件以及三个 JSON 中解码出的 Python 源码均通过 Ruff lint/format；内嵌源码也通过 Python 编译检查。schema 导出检查通过，文档的 36 个本地链接及 `git diff --check` 均通过。

当前使用 CPU PyTorch 环境，真实 vLLM/AITER ABI 与 GPU 用例按测试条件跳过。测试出现一条环境原有的 PyTorch 缺少 NumPy 警告，没有导致失败。新增 gfx950 数值测试直接构建并执行真实 `m=268` kernel，已准备好供目标环境运行。

大 GEMM 的 FP32 workspace 变为原来的 **4/3**，partial 存取流量增加。其当前 FlyDSL 表内最大 workspace 为 66 MiB。目标 GPU 仍需重新验证数值、显存、延迟、eager/capture/replay 和模型集成；CPU 结果不能用于填写 GPU 性能或准确率测量。
