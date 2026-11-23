# AKLR Plugin 的目录与 AITER 接口

AITER、SGLang、vLLM 的 28 个算子族共用 `aklr.core.plugin.Plugin`。每个族在 `plugin.py` 中发布自己的契约、绑定 kernel 调用签名，并声明这个族能接管的宿主调用。调用方继续使用宿主自身的 API。下面的 AITER 清单按 `stack.json` 的提交 `a6c759b1fc70` 核对。[Plugin](../aklr/core/plugin.py) · [锁定版本](../stack.json) · [上游 provider ABI 清单](https://github.com/ROCm/aiter/blob/a6c759b1fc700cbe856ebc57770c549ac010ea3b/tests/plugin_contracts.json)

## 1. 作者在哪个文件工作

```text
aklr/
  core/plugin.py                          Plugin 的定义、登记、检查与冻结
  plugins/
    aiter/
      __init__.py                         宿主 Plugin 实例
      api.py                              按需导出上游公开函数
      provider/__init__.py                aiter.plugins 入口、enable 和 select
      rms_norm/
        plugin.py                         契约、kernel 签名、两个版本化的 provide
        reference.py                      数值正确性
        baseline.py                       宿主对照实现
        impls/<语言>/<场景>/<uuid>.json    实现及其 workload
      mla/_paged_logits.py                分页打分、verify 分组和写回
      mega_moe/group.py                   多 rank 验证的初始化
    sglang/
      __init__.py                         宿主 Plugin 与 register()
      hook.py                             按宿主版本安装 wrap
      layers/dsa_backend/plugin.py         族的契约和对 DSA 的接入
      layers/logits_processor_torch_gemm.py 适配 lm_head，复用已有 GEMM
      models/qwen3_5/plugin.py             发布八个算子及要包装的模型方法
    vllm/
      __init__.py                         宿主 Plugin 与 register()
      hook.py                             安装及恢复 intercept
      layers/skinny_gemm.py                适配已有 GEMM
      layers/fused_moe_router.py           声明 Kimi-K3 路由 GEMM 的执行与资格查询
      models/muse_spark/plugin.py          算子与模型的三条 intercept
```

新增 kernel 时只添加族内的实现 JSON。新增算子时在族的 `plugin.py` 创建 `OpContract`，用 `family.define(contract)` 登记。所有契约登记完调用一次 `family.build()`，然后用 `@family.op` 绑定具名函数。新增宿主接入时，在同文件的 `declare(host)` 中调用 `host.provide`、`host.wrap` 或 `host.intercept`。首次接入的新模块加入本宿主的清单。纯适配使用一个按功能命名的模块。大族可以按数学功能将契约定义或适配实现提到同级私有文件。Qwen3.5 按 GDN、gated attention 拆分，声明入口仍可读出八个签名。[RMSNorm](../aklr/plugins/aiter/rms_norm/plugin.py) · [Qwen3.5](../aklr/plugins/sglang/models/qwen3_5/plugin.py) · [skinny GEMM](../aklr/plugins/vllm/layers/skinny_gemm.py)

一次宿主调用经过三层。**宿主接入**决定谁拥有调用权，AITER 仲裁候选，SGLang 拥有原方法，vLLM 拥有 worker 内被替换的绑定。**族适配**检查宿主参数、构造 AKLR 调用、按宿主约定写回或返回。**kernel 分派**根据契约与 Point 查询或执行已验证的实现。参考实现、baseline 和多 rank 初始化供离线验证加载。三个宿主分别定义 28、36、2 个 kernel 契约。[AITER 入口](../aklr/plugins/aiter/provider/__init__.py) · [SGLang 入口](../aklr/plugins/sglang/__init__.py) · [vLLM 入口](../aklr/plugins/vllm/__init__.py)

## 2. AITER 调用方使用哪个 API

```python
from aiter.ops.rmsnorm import rmsnorm2d_fwd_with_add
from aklr.plugins.aiter.provider import enable

enable("my framework")
rmsnorm2d_fwd_with_add(
    out=out, input=x, residual_in=residual, residual_out=residual_out,
    weight=weight, epsilon=1e-5,
)
```

这次调用写入 `out` 和 `residual_out`，返回 `None`。在 SGLang/vLLM 中启用 `AKLR_AITER_PLUGIN=1` 时，框架的 `register()` 会启用 provider。单独使用 AITER 时由调用方显式启用。`aklr.plugins.aiter.api.rmsnorm2d_fwd_with_add` 与示例导入的是同一个上游函数，所以参数名、默认值、关键字限制、原地写入以及编译用的装饰器都一致。仅导入 AKLR、枚举契约或读取 kernel 资源时可以不安装 AITER。[上游 RMSNorm](https://github.com/ROCm/aiter/blob/a6c759b1fc700cbe856ebc57770c549ac010ea3b/aiter/ops/rmsnorm.py) · [AKLR 导出](../aklr/plugins/aiter/api.py)

下表列出 provider 服务的 21 个 op ID 对应的 Python 入口。除 `rmsnorm` 为 v2，其余均为 v1。AITER 在询问 provider 前规范化调用、补齐默认值。族的 selector 只查询资格；获选的 `run` 负责执行一次。两个模块都有同名 GEMM，调用方以完整的上游模块路径选择相应签名。AKLR 门面中的同名函数分别取自 `aiter.tuned_gemm` 和 `aiter.ops.gemm_op_a8w8`。

| Provider op ID | 上游 Python 模块 | 调用函数 |
|---|---|---|
| `rmsnorm` | `aiter.ops.rmsnorm` | `rms_norm`、`rmsnorm2d_fwd` |
| `rmsnorm_with_add` | 同上 | `rmsnorm2d_fwd_with_add` |
| `gemm_a16w16` | `aiter.tuned_gemm` | `gemm_a16w16`、`gemm_a16w16_out` |
| `gemm_a16w16_activation` | `aiter.ops.triton.gemm.basic.gemm_a16w16` | `gemm_a16w16` |
| `gemm_a8w8_bpreshuffle`、`gemm_a8w8_blockscale`、`gemm_a8w8_blockscale_bpreshuffle` | `aiter.ops.gemm_op_a8w8` | 各自同名函数 |
| `gemm_a8w8_blockscale_out` | `aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale` | `gemm_a8w8_blockscale` |
| `fused_moe` | `aiter.fused_moe` | `fused_moe` |
| `per_group_quant` | `aiter.ops.quant` | `per_group_quant_hip`、`per_1x32_mx_quant_hip` 的相应量化分支 |
| `mhc_pre`、`mhc_post`、`mhc_fused_post_pre` | `aiter.ops.mhc` | 各自同名函数。post 写入 `out`，返回 `None` |
| `mla_decode_fwd` | `aiter.mla` | `mla_decode_fwd`，含 4 个仅限关键字的参数 |
| `mla_paged_decode` | `aiter.ops.triton.gluon.mla_gluon` | `mla_gluon` |
| `fp8_paged_mqa_logits` | `aiter.ops.triton.attention.pa_mqa_logits` | `deepgemm_fp8_paged_mqa_logits` |
| `mla_sparse_prefill_v4` | `aiter.ops.mla_sparse_prefill` | `mla_sparse_prefill_v4` |
| `flash_attn_varlen` | `aiter.ops.mha` | `flash_attn_varlen_func` |
| `biased_grouped_topk`、`topk_gating` | `aiter.ops.topk` | 各自同名函数 |
| `causal_conv1d_split_qkv` | `aiter.ops.triton.gated_delta_net.causal_conv1d_prefill` | `causal_conv1d_split_qkv_triton_tile_fn` |

`aklr.plugins.aiter.api.fused_moe` 现在接受上游的 `hidden_states, w1, w2, topk_weight, topk_ids, ...`，默认值由 AITER 提供。原来从 `aklr.plugins.aiter` 包根导入同名函数的代码迁到 `from aiter.fused_moe import fused_moe`，或从 `aklr.plugins.aiter.api` 导入。包根的 `fused_moe` 名称现在用于定位算子族子包。

框架自行决定回退的适配器以及离线验证使用 `<family>.plugin.<op>`。例如 vLLM 的 skinny GEMM 与 SGLang 的 lm_head 使用 `aklr.plugins.aiter.gemm.plugin.gemm`，能查询 `.serving` 或拿到 `FallbackReason` 后执行框架原调用。这个函数的 `trans_b`、`out_dtype` 属于 AKLR kernel 契约。普通 AITER 调用可按上表使用 `gemm_a16w16`，由 AITER 仲裁回退。[SGLang lm_head](../aklr/plugins/sglang/layers/logits_processor_torch_gemm.py) · [GEMM 参数转换](../aklr/plugins/aiter/gemm/plugin.py)

## 3. 暂存的 TODO

专用变体和缺少等价 provider ABI 的算子保留契约 ID、已提交 kernel、原有验证和当前有效调用路径。后续统一决定是否归属框架、复用已有 ABI，或与 AITER 新增版本化接口；本轮不为它们单设公开函数名。

| 内部契约 | 当前路径与待解决的问题 |
|---|---|
| `fp8_paged_mqa_logits_verify`、`mla_decode_fwd_cprr` | 现有 logits / MLA decode ABI 在满足 verify 或 context-parallel 条件时选中它们。独立公开需明确请求分组和跨 rank 布局 |
| `mla_prefill_varlen`、`mla_prefill_varlen_lse` | SGLang 的 prefill 适配决定精度及是否需要 LSE。带 LSE 的契约还服务 `flash_attn_varlen` 的一部分调用。单独公开需确定精度、输出组合与回退所有者 |
| `situv2_a4w4_moe` | `fused_moe` selector 可按权重和激活格式选中，已有路径供兼容；单独公开需上游定义数值与布局条件 |
| `paged_attention_decode`、`unified_attention` | 基线用 `paged_attention_v1` 和 Triton `unified_attention`。两者在锁定版本没有对应的同语义路由；vLLM 仍可直接调用 unified attention |
| `bmm`、`gemm_a4w4_mxfp4` | 对照函数分别是 `batched_gemm_bf16_CK`、`gemm_a4w4`。当前 provider 没有与这些布局及数据类型对应的 ABI |
| `mega_moe` | EP 的 collective 和每 rank 初始化属于当前契约。需要带相同多卡语义的宿主入口 |
| `topk_transform_paged` | 对照 SGLang 的 `topk_transform_paged_v2`，需要页表和输出页索引；AITER 的 `top_k_per_row_decode` 返回的索引含义不同 |

## 4. 发现、发布与兼容

族的 `plugin.py` 是作者入口。定位器在传入的 `aklr/plugins` 根枚举族，跳过 `impls/`。独立 Git checkout 和已安装 wheel 使用同一规则。读取契约时，注册表导入族并读取 `family.contracts`。宿主启动时用同一组对象构建自己的实例，重复启动和构建失败重试沿用 `Plugin.build` 的语义。[族定位](../aklr/core/contract/locate.py) · [注册表](../aklr/core/contract/registry.py) · [独立 checkout 验证](../aklr/core/validation/stages/validation_scope.py)

已发布的 SGLang、vLLM 和 SiTUv2 族保留 `.api` 和 `.contract` 的转发，旧导入和 `.plugin` 引用相同契约、常量和 dispatcher。由 `aklr/ops` 迁入的 11 个族在新位置只写 `plugin.py`。`OpContract.definition_name` 记录实际签名位置，`api_name` 提供可导入的 kernel 调用路径。尚未迁移的外部族仍可用原 `contract.py` 发现；双标记目录只注册一次。原来只借用 vLLM skinny GEMM 的 `layers.utils.api` 继续转发到具名适配模块。[vLLM 兼容入口](../aklr/plugins/vllm/layers/utils/api.py) · [发现和绑定回归](../tests/core/test_plugin.py)

## 5. 迁移验收

与主线 `b5907967fe17` 对照，原 `aklr/ops` 的 351 份 JSON 中，350 份在新位置保留相同的 Git blob。一份 `fused_moe` 的内嵌源码引用了旧包，迁移后使用新 UUID `c9d92edef1e6c882`。原 20 个 Point 随实现迁入，测量字段留给新实现重测。历史测量报告按原族和相同文件名关联到新路径，新的计时记录优先。[迁移回归](../tests/plugins/aiter/test_kernel_assets.py) · [历史报告回归](../tests/benchmarks/test_kernels_report.py)

| 验收对象 | 核对方式 |
|---|---|
| 作者与调用方 | 从干净安装的 wheel 发现 28 族，比较全部 66 个内部契约签名。公开的 22 个 AITER 函数与锁定上游逐一比较对象身份。另行核对两个与公开函数同名的 Triton GEMM 入口 |
| 宿主注册 | 通过 AITER 自身的 `validate("aklr")` 核对 21 个版本化 ABI。SGLang 声明 30 个包装，vLLM 声明 8 个 intercept、共 9 个被调路径 |
| 资源与计时归属 | wheel 内的 Python、JSON、HIP 和 C++ 资源与源码逐字节比较。837 份 schema 均有唯一的 GPU 计时作业，AITER 351 份（含 `mega_moe`），SGLang 458 份（含 5 份 SiTUv2），vLLM 28 份 |
| GPU 与整模 | `mhc_post` 的仲裁、原地写回和原生回退可独立于性能验证。gfx950 数值与计时以此提交的三项 GPU CI 为准，新路由的整模收益需另做相同配置的开关对照 |
