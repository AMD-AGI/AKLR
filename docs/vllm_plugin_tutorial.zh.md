# 插件教程：替换 vLLM 的函数

当一项优化需要覆盖不止一个算子时，插件会把 vLLM 的几次调用交给 AKLR 的一个算子。本教程的示例是 `muse_spark_moe`，它接管 Muse Spark 的 MoE 层从 pre-expert RMSNorm 到各路专家输出的部分；vLLM 在这一层调用的 aiter `fused_moe`，只是其中的专家计算部分。

开始之前，请先阅读 [Ops 教程](ops_tutorial.zh.md)：插件里的算子写法与之相同。SGLang 插件的做法见 [SGLang 插件教程](plugin_tutorial.zh.md)，插件的结构和规范见 [aklr/plugins/README.md](../aklr/plugins/README.md)。

## 为什么需要插件

vLLM 在 `MuseSpark1xMoE`（`models/muse_spark/amd/model.py`）中计算 Muse Spark 的 MoE 层，分成多次调用完成。`fused_moe` 只覆盖其中的一部分步骤：

| 步骤 | 内容 | `fused_moe` | `muse_spark_moe` |
|---|---|---|---|
| 1 | 路由：FP32 router 和 top-k；pooler 把宽 8192 的 hidden 投影到专家的宽 4096 | — | — |
| 2 | pre-expert RMSNorm（fp32，乘以 `norm_weight + gain_center`），并把每行展开为 topk 行，每路一行 | — | ✓ |
| 3 | 按专家对各路分组；本卡不持有的专家，其路的输出为零 | ✓ | ✓ |
| 4 | 输入量化为 MXFP4 | ✓ | ✓ |
| 5 | GEMM：gate 和 up | ✓ | ✓ |
| 6 | 激活 SiLU(gate) × up，并量化为 MXFP4 | ✓ | ✓ |
| 7 | GEMM：down，得到每一路的输出（不乘 router 权重） | ✓ | ✓ |
| 8 | 多卡归约、逐路 post-expert norm、乘 router 权重并对各路求和 | — | — |
| 9 | unpooler | — | — |

vLLM 在调用 `fused_moe` 之前，用 `fused_rmsnorm_expand` 单独执行第 2 步：它把归一化后的行展开 topk 份写出，专家再把它们读回。要把 norm 合进专家读取输入的过程，就需要一个算子接管第 2–7 步；而 vLLM 中没有哪一次调用覆盖这几步：第 2 步在 `MuseSpark1xMoE._pool` 中，第 3–7 步在 `MuseSparkAiterExperts.apply` 中。所以插件替换这两次调用，并把 norm 推迟到专家那一次调用中执行。

因此：

- 只需让某一个算子更快时，在 `aklr/ops` 中写算子。
- 优化需要越过单个算子的边界时，写插件。

## 插件如何运行

设置 `AKLR_VLLM_PLUGIN=1` 后，每个 worker 在绑定设备后挂载 route。此后每次调用被替换的函数，都会先执行 route：route 把调用交给算子，或者返回 `FallbackReason`，把调用交回 vLLM 原有的代码（见 [调用流程](../aklr/plugins/README.md#call-flow)）。

## 步骤

### 1. 确定要替换的函数

即 vLLM 的这两次调用：

- `MuseSpark1xMoE._pool` 调用的 `fused_rmsnorm_expand`（`models/muse_spark/amd/model.py`）：完成上表第 2 步，返回归一化并展开后的行。
- `FusedMoEKernelModularImpl._fused_experts` 调用的 `MuseSparkAiterExperts.apply`（`models/muse_spark/amd/muse_moe_runner.py`）：完成上表第 3–7 步，把逐路的输出交给 vLLM 的后续步骤（第 8 步）。

它们在 vLLM 0.33.0rc1.dev910 中的参数如下，route 要原样照抄，方法要带上 `self`：

```python
def fused_rmsnorm_expand(x, weight, eps, gain_center, input_scale, repeats): ...

def apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids, activation, global_num_experts,
          expert_map, a1q_scale, a2_scale, workspace13, workspace2, expert_tokens_meta,
          apply_router_weight_on_input): ...
```

### 2. 建包

新建 `aklr/plugins/vllm/models/muse_spark/`，以 vLLM 的源目录 `models/muse_spark/` 命名。包内的文件与 `aklr/ops` 中的算子相同：`__init__.py`、`contract.py`、`reference.py`、`baseline.py`、`api.py` 和 `impls/`；另有 `_norm_expand.py`，是 vLLM norm kernel 的副本，让 baseline 不依赖 vLLM 也能运行。包名即算子的 `op_type`（`muse_spark`），在 `aklr/ops` 和 `aklr/plugins` 中都不能重名。

### 3. 编写算子

做法同 ops 教程的 1. A。针对这个算子：

- `contract.py` 定义 `MUSE_SPARK_MOE`（`muse_spark_moe@1.0`）：
  - 输入：尚未归一化的 pooled rows `hidden_states`；norm 的 `norm_weight`；每行的路由 `topk_ids`；本卡持有哪些专家的 `expert_mask`（最后一位是 vLLM 恒被屏蔽的槽位）；专家的 `w1`、`w1_scale`、`w2`、`w2_scale`；以及标量 `eps` 和 `gain_center`；
  - 布局：`row_major` 和 `shuffled_16x16`，后者即 vLLM 为 aiter 保存专家权重的方式；
  - 输出：`out`，形状 `[n_tokens, topk, model_dim]`，即每一路的专家输出，不乘 router 权重；本卡不持有的专家，其路写零。router 权重、post-expert norm 和各路求和由 vLLM 在之后完成。
- `reference.py`：`muse_spark_moe_reference`，按 contract 的计算实现：RMSNorm 在 fp32 下计算，按 vLLM 的 `fused_rmsnorm_expand` 写出的方式舍入为 `hidden_states` 的 dtype，再让每一路以权重 1 经过 `fused_moe` 的 reference。
- `baseline.py`：`muse_spark_moe_baseline`，即 vLLM 目前执行第 2–7 步的方式：`fused_rmsnorm_expand`，然后 `MuseSparkAiterExperts.apply` 调用的 aiter `fused_moe`（per_1x32）。
- `api.py`：算子的入口函数 `@api(MUSE_SPARK_MOE)`，函数体为 `...`。

### 4. 编写 route

route 写在 `api.py` 中，放在算子入口函数 `muse_spark_moe` 旁边。两个 route 合起来，把 vLLM 的两次调用交给这一个算子：

- contract 的输入是尚未归一化的行，以及 norm 的 `norm_weight`、`eps` 和 `gain_center`，norm 由算子计算。所以 `_defer_norm` 跳过 vLLM 的 norm，把 norm 的参数留给算子。
- `_experts_apply` 把 vLLM 的参数整理为算子的参数，调用 `muse_spark_moe`。

`@vllm_route` 的第一个参数是调用方，第二个是被替换的函数；两个 route 属于同一个 `group`。下面省略了部分参数和检查：

```python
@vllm_route(
    "vllm.models.muse_spark.amd.model.MuseSpark1xMoE._pool",
    "vllm.models.muse_spark.amd.model.fused_rmsnorm_expand",
    versions=("0.33.0rc1.dev910",),
    group="muse_spark.norm_experts",
)
def _defer_norm(x, weight, eps, gain_center, input_scale, repeats):  # 参数与被替换的函数完全一致
    # AKLR 不支持的输入
    if weight is None or input_scale or x.dtype != torch.bfloat16 or x.shape[0] == 0:
        return FallbackReason.OUT_OF_CONTRACT
    # 算子不接这一层在这个 token 数下的调用：vLLM 照常做 norm
    layer = id(weight)
    serves = _SERVES.get((layer, x.shape[0]))
    if serves is None and layer in _EXPERTS:
        serves = _would_serve(x, weight, eps, gain_center, layer)
    if serves is False:
        return FallbackReason.OUT_OF_CONTRACT
    # 不做 norm：把它的参数记在张量上，原样返回，交给专家的 route
    setattr(x, _DEFERRED_NORM, (weight, eps, gain_center))
    _PENDING.add(layer)
    return x


def _would_serve(x, weight, eps, gain_center, layer):
    # _experts_apply 记下的这一层的专家参数；None 表示这一层的 vLLM 配置不受支持
    if _EXPERTS[layer] is None:
        return False
    topk, ids_dtype, expert_map, w1, w1_scale, w2, w2_scale = _EXPERTS[layer]
    # 用算子的参数问算子是否接这次调用，不计算；ids 是全零的 topk_ids，只用来判断
    reason = muse_spark_moe.fallback_reason(
        x, weight, ids[: x.shape[0]], expert_map, w1, w1_scale, w2, w2_scale,
        eps=float(eps), gain_center=float(gain_center),
    )
    _SERVES[(layer, x.shape[0])] = reason is None
    return reason is None


@vllm_route(
    "vllm.model_executor.layers.fused_moe.modular_kernel.FusedMoEKernelModularImpl._fused_experts",
    "vllm.models.muse_spark.amd.muse_moe_runner.MuseSparkAiterExperts.apply",
    versions=("0.33.0rc1.dev910",),
    group="muse_spark.norm_experts",
)
def _experts_apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids, ...):
    deferred = getattr(hidden_states, _DEFERRED_NORM, None)
    # 这一层的 norm 没有推迟：交回 vLLM
    if deferred is None:
        return FallbackReason.OUT_OF_CONTRACT
    weight, eps, gain_center = deferred
    quant = self.quant_config
    route_output = FallbackReason.OUT_OF_CONTRACT
    _EXPERTS[id(weight)] = None
    # AKLR 支持的 vLLM 配置
    if expert_map is not None and activation.name == "SILU" and quant.use_mxfp4_w4a4 and ...:
        # 记下这一层的专家参数，供 _defer_norm 之后判断
        _EXPERTS[id(weight)] = (topk_ids.shape[1], topk_ids.dtype, expert_map, w1, quant.w1_scale, w2, quant.w2_scale)
        # 调用融合了 norm 和 fused_moe 的算子
        route_output = muse_spark_moe(
            hidden_states,  # 尚未归一化的 pooled rows
            weight,  # norm_weight
            topk_ids,
            expert_map,  # expert_mask
            w1,
            quant.w1_scale,
            w2,
            quant.w2_scale,
            eps=float(eps),
            gain_center=float(gain_center),
            out=output,
        )
    # 算子不接：先补做推迟的 norm，再交给 vLLM 原有的 apply
    if isinstance(route_output, FallbackReason):
        normalized = _original(_defer_norm)(hidden_states, weight, eps, gain_center, False, topk_ids.shape[1])
        return _original(_experts_apply)(self, output, normalized, w1, w2, topk_weights, topk_ids, ...)
    # 把逐路的输出交给 vLLM 的后续步骤（第 8 步）
    self._park_route_output(route_output.view_as(output))
    return None
```

算子接这次调用时，返回 `out`，即形状为 `[n_tokens, topk, model_dim]` 的逐路输出，route 按 `output` 的形状把它交给 `_park_route_output`，供 vLLM 的后续步骤（第 8 步）使用；不接时返回 `FallbackReason`，route 补做 norm 后交回 vLLM。

### 5. 添加测试输入

在 `tests/plugins/vllm/cases.py` 的 `SMALL_POINTS` 中，为算子添加一个小规模的 Point：

```python
PointSpec(
    contract_id="muse_spark_moe@1.0",
    axes={"n_tokens": 4, "model_dim": 2048, "topk": 2, "n_experts": 2, "inter_dim": 256, "global_experts": 5},
    dtypes={"hidden_states": "bfloat16", "norm_weight": "bfloat16", "topk_ids": "int32", "expert_mask": "int32",
            "w1": "float4_e2m1", "w1_scale": "float8_e8m0", "w2": "float4_e2m1", "w2_scale": "float8_e8m0",
            "out": "bfloat16"},
    attrs={},
    layout_variant_name="shuffled_16x16",
),
```

它的 baseline 只能在 gfx950 上运行，所以还要把它的 contract id 加入同一文件的 `GFX950_CONTRACT_IDS`。然后更新 `tests/plugins/vllm/names.txt`：

```bash
AKLR_UPDATE_NAMES=1 pytest tests/ops/test_contracts.py
```

`tests/plugins/vllm/test_hook.py` 会检查每个 route，确认它替换的函数在当前安装的 vLLM 中存在、参数一致，且没有其他 route 替换同一个函数。推迟 norm 是这两个 route 之间的约定，`test_hook.py` 检查不到，所以要在 `tests/plugins/vllm/test_muse_spark.py` 中单独测试：没有推迟标记的行交回 vLLM，推迟的 norm 在途中丢失时报错，baseline 所用的 norm 副本与 vLLM 的结果逐位一致。

### 6. 编写 kernel 并验证

按 ops 教程的 1. B 到第 4 步进行，场景换成这个算子的：Muse Spark MXFP4 MoE 层在 EP8 下的形状，每张卡持有 256 个路由专家中的 32 个，`global_experts` 为 256 加上恒被屏蔽的槽位。脚本中 contract、Point 和 workload 不同，其中 `eps` 和 `gain_center` 是标量，随每个 workload 给出；这个算子已提交的 kernel 用 FlyDSL 编写：

```python
contract = ContractRegistry.default().get("muse_spark_moe@1.0")
base = PointSpec(
    contract_id="muse_spark_moe@1.0",
    axes={"n_tokens": 1, "model_dim": 4096, "topk": 8, "n_experts": 32, "inter_dim": 4096, "global_experts": 257},
    dtypes={"hidden_states": "bfloat16", "norm_weight": "bfloat16", "topk_ids": "int32", "expert_mask": "int32",
            "w1": "float4_e2m1", "w1_scale": "float8_e8m0", "w2": "float4_e2m1", "w2_scale": "float8_e8m0",
            "out": "bfloat16"},
    attrs={},
    layout_variant_name="shuffled_16x16")
workloads = [contract.workload_for(base.model_copy(update={"axes": {**base.axes, "n_tokens": t}}),
                                   {"eps": 1e-6, "gain_center": 1.0}) for t in (8, 256, 8192)]
kernel = {"language": "flydsl", "targets": [{"arch": "gfx950", "hardware": "MI355X"}],
          "entry_point": "kernel.py::run", "sources": []}
```

schema 会写到这个场景已提交的 kernel 旁边，即 `aklr/plugins/vllm/models/muse_spark/impls/flydsl/muse_spark_moe_shuffled_16x16_model_dim4096_topk8_n_experts32_inter_dim4096_global_experts257/`。这些形状和 token 数的来源见 [Muse Spark 的 definitions](../aklr/benchmarks/models/muse_spark/DEFINITIONS.md)。验证和提 PR 的做法同 ops 教程。

### 7. 在 vLLM 中运行

validation 检查的是 kernel 与 reference 是否一致，不检查 route；route 要靠运行整个模型来检查。专家的 route 只在专家并行下接管调用，所以用 Muse Spark（MXFP4）在 TP8 加 `--enable-expert-parallel` 下启动 vLLM：

```bash
VLLM_ROCM_USE_AITER=1 VLLM_ROCM_MUSE_MOE_REDUCE_DTYPE=bf16 VLLM_USE_BREAKABLE_CUDAGRAPH=1 AKLR_VLLM_PLUGIN=1 \
    vllm serve <Muse Spark MXFP4 checkpoint> --tensor-parallel-size 8 --enable-expert-parallel \
    --language-model-only --attention-backend ROCM_AITER_FA
```

完整的启动参数见 [Muse Spark 的 definitions](../aklr/benchmarks/models/muse_spark/DEFINITIONS.md)。日志会报告挂载了哪些调用（`aklr hooks N calls of vLLM 0.33.0rc1.dev910`），模型的回答必须保持正确。

## 下一步

- 插件的结构和 route 规范：[aklr/plugins/README.md](../aklr/plugins/README.md)。
- 让 agent 编写和调优 kernel：[forge_task](forge_task.md)。
