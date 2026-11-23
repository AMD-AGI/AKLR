# AKLR 插件体系与 AITER 算子迁移需求

本文定义迁移必须交付的行为。`Plugin` 公共接口已定稿，运行时行为在实施阶段验收。盘点基线为 AKLR `766a200e367f` 和 `stack.json` 锁定的 AITER `a6c759b1fc70`。[公共接口和实施顺序](aiter_refactor_design.zh.md)

## 1. 需要分别验收的三个结果

1. **算子归属**。移除整个 `aklr/ops/`。每个算子族在 `aklr/plugins/aiter/` 下有唯一的定义位置，包含契约、Python 调用入口、参考实现、性能基线、kernel 和专用辅助代码。已有的 SGLang、vLLM 私有算子仍由对应插件拥有。
2. **调用归属**。AITER 拥有上游函数的签名、版本、原生实现和 provider 仲裁。凡是通过 AITER 暴露的加速，都从 AITER 的 `aiter.plugins` 入口接入。一个算子族迁入插件，并不自动获得一条 AITER 路由。框架自己的复合操作、直接调用 AKLR 的应用、导出和离线验证仍有各自的入口。
3. **统一的插件作者接口**。算子契约和宿主接入的 Python 声明都在具名的 `Plugin` 实例上登记。算子族导出冻结的契约，kernel 作者在族内提交一份带 `contract_id` 的实现 JSON。宿主 entry point 消费本进程包级实例的声明。AITER 可以只查询候选，SGLang 可以包装整个方法，vLLM 可以替换特定 caller 查找的 callee。适配器按实参复用一个或多个算子，各宿主持有自己的启动、安装和回退时机。

例如，迁移后 kernel 作者可以给 `mega_moe` 增加一个 EP=8 的实现，而不必修改 provider。反过来，给 AITER 的另一个 GEMM 名称增加适配，可以复用已有 `gemm@1.0`，无需复制契约和 kernel。

## 2. 现有资产和调用范围

下表逐族列出旧目录的全部 27 个契约。名字取自 `@api` 入口，JSON 数量包含只有场景定义的文件。前三列构成搬运清单，第四列记录基线提交已声明的 AITER 路由。当前逐项实现和 TODO 见[现行接口清单](plugin_layout.zh.md)。

| 算子族 | AKLR 契约名 | JSON | AITER 路由名 |
|---|---|---:|---|
| `attention` | `paged_attention_decode`、`unified_attention` | 5 | 无绑定。vLLM 有直接调用 `unified_attention` 的路由 |
| `fused_moe` | `fused_moe` | 59 | `fused_moe`，部分调用使用下文的 SiTUv2 契约 |
| `gemm` | `gemm`、`gemm_a8w8`、`gemm_a8w8_blockwise_scaled`、`gemm_a4w4_mxfp4`、`bmm` | 232 | `gemm_a16w16`、`gemm_a16w16_activation`、`gemm_a8w8_bpreshuffle`、`gemm_a8w8_blockscale`、`gemm_a8w8_blockscale_bpreshuffle`、`gemm_a8w8_blockscale_out` |
| `kda_conv` | `kimi_k3_kda_conv` | 2 | `causal_conv1d_split_qkv` |
| `mega_moe` | `mega_moe` | 3 | 无绑定。另有每个 rank 的 `group.py` 初始化 |
| `mhc` | `mhc_pre`、`mhc_post`、`mhc_fused_post_pre` | 5 | `mhc_pre`、`mhc_fused_post_pre`。上游也有 `mhc_post`，AKLR 当前未声明 |
| `mla` | `flash_mla_with_kvcache`、`fp8_paged_mqa_logits`、`fp8_paged_mqa_logits_verify`、`mla_decode_fwd`、`mla_decode_fwd_cprr` | 23 | `mla_paged_decode`、`fp8_paged_mqa_logits`、`mla_decode_fwd` |
| `mla_prefill` | `sparse_prefill_mxfp8`、`mla_prefill_varlen`、`mla_prefill_varlen_lse` | 9 | `mla_sparse_prefill_v4`、`flash_attn_varlen`。族内还有 `ps_plan.py` |
| `quant` | `per_group_quant_fp8`、`per_group_quant_mxfp8` | 3 | `per_group_quant` |
| `rms_norm` | `rms_norm`、`add_rms_norm` | 3 | `rmsnorm` v2、`rmsnorm_with_add` v1 |
| `topk` | `biased_grouped_topk`、`topk_transform_paged` | 6 | `biased_grouped_topk`、`topk_gating` |

旧树共 **350 个 JSON**，其中 279 个含源码、71 个仅定义场景，228 个文件含 CI 写入的测量字段。另有 58 个 Python 文件和一份算子作者手册。插件内已有的 `situv2_a4w4_moe` 独立拥有 1 个契约和 5 个 JSON。最终 AITER 插件管理 **12 族、28 个契约**。盘点或比较须使用同一 Git 基线，新增工作应独立计入。

上游锁定版本的契约清单有 40 条，AKLR 已声明其中 20 条，名称和版本全部对应。除 `rmsnorm` 为 v2 外，此处的绑定均为 v1。该数值说明接口覆盖范围，实际能否接管还取决于本次输入、设备和 Point 的资格。[锁定的上游路由契约](https://github.com/ROCm/aiter/blob/a6c759b1fc700cbe856ebc57770c549ac010ea3b/tests/plugin_contracts.json) · [AITER provider 协议源码](https://github.com/ROCm/aiter/blob/a6c759b1fc700cbe856ebc57770c549ac010ea3b/aiter/plugin/__init__.py)

## 3. 当前结构造成的具体问题

仓库中插件根为 `aklr/plugins/`。下面盘点其全部三个宿主。契约数按族的 `CONTRACTS` 统计，路由数按源码声明统计，每个 release 实际安装的子集另外检查。

| 宿主 | 算子所有权 | 宿主集成必须覆盖的内容 |
|---|---|---|
| AITER | 插件现有 1 族、1 个契约，接收旧目录后共 12 族、28 个契约 | 20 条版本化 provider 路由、安装发现、显式启用、按调用提供候选 |
| SGLang | `layers/`、`models/` 共 14 族、36 个契约 | 30 条函数或方法 hook 声明（含 DSA 提示用的观察 hook），版本与 eager 门控、分布式初始化后的 runtime 启动、跨插件复用 GEMM 和 MLA prefill 的 BF16 回退 |
| vLLM | `models/kimi_k3/attn_res` 和 `models/muse_spark` 共 2 族、2 个契约 | 6 条路由声明、7 个显式被调路径（unified attention 含两个别名），成组安装、原 callable、设备就绪后的安装、warmup 与执行包装 |

迁移后可独立核对 66 个契约的唯一所有者。SGLang/vLLM 的私有源码、测试、数据资产和计时环境仍需纳入总体验收。[三种宿主的加载入口](../pyproject.toml) · [SGLang 路由安装](../aklr/plugins/sglang/hook.py) · [vLLM 调用点与分组](../aklr/plugins/vllm/hook.py)

| 现状 | 对维护者或调用方的影响 |
|---|---|
| 同一次 fused MoE 调用的上游版本适配、量化策略、AKLR 参数分散在 `aiter/fused_moe.py`、`provider/moe.py` 和两个契约包 | 修改一种激活类型时，选择与执行可能走向不同契约。新目录还会与现有同名 Python 模块冲突 |
| provider 在 import 时扫描模块，装饰器写全局 `SELECTORS`，再派生 `op_versions` | 新增绑定的生效顺序隐含在导入流程中，难以逐条核对它声明的 ABI |
| AITER 按 op 名登记 selector，SGLang 收集目标函数，vLLM 按被调路径与 group 安装 | 单看“目标与处理函数”无法判断谁能运行 kernel、谁负责原调用或哪组改动需一起生效。接口要给出各宿主的调用约定、冲突键和失败行为 |
| MLA 的 verify 提示存在进程级列表中，SGLang 从 provider 的 MLA 实现模块写入，`select` 可能只被 `explain` 或多 provider 比选调用 | 只用相同行数无法区分并发请求或过期提示。一次查询也未必对应一次 AKLR 执行 |
| 契约发现会导入 `aklr.ops`，全量验证直接遍历旧目录。CI、wheel、Forge、测量覆盖目录和调用示例也使用物理路径 | 只搬 Python 文件会使发现、测量继承、产物发布或安装后加载失败，其中有些会无声地漏掉文件 |
| 一个 FlyDSL kernel 的 JSON 源码内直接导入旧版 `fused_moe_baseline`，超过 70 tokens 会调用 routed AITER MoE | 替换 Python 文件里的 import 仍会留下无法加载的 kernel。改其源码会改变 UUID，候选执行中回调 AITER 还可能重新进入同一路由 |

现有调用行为可对照[当前 provider](https://github.com/AMD-AGI/AKLR/blob/766a200e367fac608a91db74b22e67fbf140397e/aklr/plugins/aiter/provider/__init__.py)、[MoE 选择](https://github.com/AMD-AGI/AKLR/blob/766a200e367fac608a91db74b22e67fbf140397e/aklr/plugins/aiter/provider/moe.py)、[MLA 选择](https://github.com/AMD-AGI/AKLR/blob/766a200e367fac608a91db74b22e67fbf140397e/aklr/plugins/aiter/provider/mla.py)和[契约定位](../aklr/core/contract/locate.py)。

## 4. 可验收需求

| 编号 | 谁需要什么 | 验收条件 |
|---|---|---|
| R1 资产迁移 | 算子作者能在唯一位置维护全族 | 旧目录删除。按族核对契约 ID、名称、签名、场景、Point、源码、测量、辅助文件。每个物理路径变化都有确定映射。内容变更单独审计 |
| R2 上游接入 | AITER 能发现、启用、询问和执行 AKLR | 安装后可通过 `aiter.plugins` 发现 `aklr`。启用由宿主决定，已有 provider 优先序和排除项得到保留。对已有 20 条声明及新增的 `mhc_post` v1 逐条核对上游版本和参数种类。对外保留 AITER 函数名、默认值、仅限关键字参数和写回约定。未声明或不适用的调用交回 AITER 仲裁 |
| R3 正确执行 | 引擎得到和 AITER 一样的可见结果 | 区分返回 tensor、tuple、`None`、写入调用方的 `out`、更新 residual/cache 和布局转换。查询阶段不运行候选、不改输入或输出。开始执行后的异常传播，绝不重复写入 |
| R4 路由边界 | SGLang、vLLM 和独立宿主均有明确入口 | 一个已路由的 AITER 调用只向 AKLR 提议一次。框架仍可在自己的复合操作中直调契约。verify/decode 交错、嵌套、异常退出、其他 provider 获选均不会串用框架上下文 |
| R5 接口稳定 | 算子族、模型和宿主作者使用同一个类登记声明 | 在 `Plugin(name)` 实例上用 `define` 声明契约。族构建后，各算子入口用 `@family.op` 绑定本族契约，保留调用签名、`serving` 和 `fallback_reason`。宿主构建时汇集本包各族发布的契约对象，用 `provide` 声明 AITER 候选查询，用 `wrap` 声明 SGLang 替换或包住原调用的回调，用 `intercept` 声明 vLLM caller 查找的一个或多个 callee。`build(*declarations)` 组装并冻结实例，`name` 和构建后的 `contracts` 只读。外部扩展在宿主首次构建前向包级实例登记，构建失败保留预登记并可重试。新增一个 ABI 或框架调用点只添一项声明，新增 kernel 只添族内有效 JSON 即可被 `read_schemas` 发现 |
| R6 数据与工具 | 用户能用既有记录继续导出、验证和发版 | 旧 calls 导出同一场景。历史 Git 基线和测量覆盖可读取。实际安装的 wheel 含完整文件。Forge 能找到无源码场景并向新目录发布。CI 对同一 Point 只有一个计时归属，保留测量防篡改规则 |
| R7 运行边界 | 启用、禁用、冷启动、图捕获行为可预期 | 覆盖 AITER 缺失、总开关、按契约禁用、其他 provider、初始化失败、未预热/已预热图捕获。候选身份可由 AITER 决策追溯到 AKLR kernel。记录选择与实际执行的区别 |
| R8 结构约束 | 维护者能从一次调用追到负责它的代码 | 执行链按宿主接入、调用适配、算子分派划分，至多三层。插件声明可在未安装三个宿主的环境中检查。契约沿用 `OpContract`，合格实现由已有 dispatcher 选择，AITER 候选沿用上游 `KernelCandidate`。DSA 提示、BF16 回退和 worker 生命周期由相应宿主适配承担 |
| R9 三宿主接入与复用 | 插件作者能让另一个宿主复用已有算子，也能单独扩展宿主 | 从原有三个 entry point 验证实例声明的调用权。AITER 只在候选获选后调用 `run`。SGLang 的 `replace` 回调放弃时由包装器调用一次原函数，`around` 回调控制原调用作用域。vLLM 在设备就绪时成组安装并能恢复原绑定。实例在发布前检查重复声明，安装器在写入前检查上游版本、实际路径与签名。两个框架调用同一 GEMM 时共用同一算子定义与实现资产 |

## 5. 本次范围和评审口径

本次交付覆盖三个宿主的需求与设计。后续实现以重构前的公开行为和逐族清单为基线。新增其余 19 条 AITER 上游路由、调整精度或速度阈值、改变 kernel ABI，各自需要独立的需求和证据。

切换到新 API 的目标发行版本时，直调方迁到新包名，使用 `sglang_route`、`vllm_route` 的外部作者分别迁到包级实例的 `wrap`、`intercept`。同次切换移除旧包和旧登记入口，发布说明给出实际版本与示例。含旧 import 的内嵌 kernel 需要单独验证允许未测量实现时的行为，以及历史测量的继承条件。[外部扩展示例与内嵌 kernel 迁移方案](aiter_refactor_design.zh.md)
