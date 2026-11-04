# 插件教程：替换 SGLang 的函数

当一项优化需要覆盖不止一个算子时，插件会把 SGLang 的整个函数替换为 AKLR 的一个算子。本教程的示例是 `deepseek_v2_mxfp4_moe`，它接管 GLM-5 的整个 MoE 层；[Ops 教程](ops_tutorial.zh.md) 用 `fused_moe` 优化的，正是这一层的专家计算部分。

开始之前，请先阅读 ops 教程：插件里的算子写法与之相同。插件的结构和规范见 [aklr/plugins/README.md](../aklr/plugins/README.md)。

## 为什么需要插件

SGLang 在 `DeepseekV2MoE.forward_normal`（`srt/models/deepseek_v2.py`）中计算 GLM-5 的 MoE 层，分成多次调用完成。`fused_moe` 只覆盖其中的一部分步骤：

| 步骤 | 内容 | `fused_moe` | `deepseek_v2_mxfp4_moe` |
|---|---|---|---|
| 1 | 路由 GEMM：每个 token 对 256 个路由专家的打分 | — | ✓ |
| 2 | 选专家：sigmoid 加 correction bias，选出前 8 个，权重归一化后乘以 2.5 | — | ✓ |
| 3 | 共享专家作为第 9 个专家追加进来，权重为 1 | — | ✓ |
| 4 | 按专家对 token 分组 | ✓ | ✓ |
| 5 | 输入量化为 MXFP4 | ✓ | ✓ |
| 6 | GEMM：gate 和 up | ✓ | ✓ |
| 7 | 激活 SiLU(gate) × up，并量化为 MXFP4 | ✓ | ✓ |
| 8 | GEMM：down | ✓ | ✓ |
| 9 | 对每个 token 选中的专家加权求和 | ✓ | ✓ |
| 10 | 多卡归约（all-reduce） | — | — |

SGLang 在调用 `fused_moe` 之前，把第 1–3 步作为单独的调用执行。在 decode 中，每次调用的 token 很少，这几次调用的启动开销以及中间结果的写出和读回，占了这一层相当大的时间。要把这几步也一起优化，就需要一个算子接管整个 MoE 层；而 SGLang 中没有哪一次调用覆盖整层，所以插件直接替换 SGLang 的这个函数。

因此：

- 只需让某一个算子更快时，在 `aklr/ops` 中写算子。
- 优化需要越过单个算子的边界时，写插件。

## 插件如何运行

设置 `AKLR_SGLANG_PLUGIN=1` 后，每次调用 `DeepseekV2MoE.forward_normal` 都会先执行 route：route 把调用交给算子，或者交回 SGLang 原有的代码（见 [调用流程](../aklr/plugins/README.md#call-flow)）。

## 步骤

### 1. 确定要替换的函数

即 SGLang `srt/models/deepseek_v2.py` 中的 `DeepseekV2MoE.forward_normal`。它完成上表第 1–10 步，返回这一层的输出；`self.reduce_results` 为真时，输出已在各卡之间求和。它在 SGLang 0.5.22 中的参数如下，route 要原样照抄：

```python
def forward_normal(self, hidden_states, gemm_output_zero_allocator=None, input_ids=None, input_ids_global=None,
                   skip_shared_experts=False, num_token_non_padded=None, use_vision_topk=False,
                   return_moe_output=False):
```

### 2. 建包

新建 `aklr/plugins/sglang/models/deepseek_v2/`，以 SGLang 的源文件命名。包内的文件与 `aklr/ops` 中的算子相同：`__init__.py`、`contract.py`、`reference.py`、`baseline.py`、`api.py` 和 `impls/`。包名即算子的 `op_type`，在 `aklr/ops` 和 `aklr/plugins` 中都不能重名。

### 3. 编写算子

做法同 ops 教程的 1. A。针对这个算子：

- `contract.py` 定义 `DEEPSEEK_V2_MXFP4_MOE`（`deepseek_v2_mxfp4_moe@1.0`）：
  - 输入：`forward_normal` 读取的内容，即 `hidden_states`；路由的 `router_weight` 和 `correction_bias`；专家的 `w1`、`w1_scale`、`w2`、`w2_scale`，共享专家放在最后一个位置（`n_experts = n_routed + 1`）；以及标量 `routed_scaling_factor`；
  - attr：路由设置 `topk`、`num_expert_group`、`topk_group`、`renormalize` 和 `scoring_func`；
  - 输出：`out`，即这一层在本卡上的输出，也就是多卡归约之前的结果。
- `reference.py`：`deepseek_v2_mxfp4_moe_reference`，按 contract 的计算在 fp32 下实现。
- `baseline.py`：`deepseek_v2_mxfp4_moe_baseline`，即 SGLang 目前执行第 1–9 步的方式：路由 GEMM、AITER 的分组 top-k、SGLang 追加共享专家、AITER 的 `fused_moe`。kernel 要快过的是这整条路径，而不只是 `fused_moe`。
- `api.py`：算子的入口函数 `@api(DEEPSEEK_V2_MXFP4_MOE)`，函数体为 `...`。

### 4. 编写 route

route 写在 `api.py` 中，放在算子入口函数旁边。下面省略了部分参数：

```python
@sglang_route("sglang.srt.models.deepseek_v2.DeepseekV2MoE.forward_normal", versions=("0.5.22",))
def _moe_forward_normal(self, hidden_states, ...):  # 参数与被替换的函数完全一致
    # AKLR 不支持的 SGLang 配置
    if _forward_normal_falls_back(self, ...):
        return FallbackReason.OUT_OF_CONTRACT
    # 把模块的权重和输入整理为算子参数
    out = deepseek_v2_mxfp4_moe(**_moe_arguments(self, hidden_states))
    # 算子不接，或原函数此时不做多卡归约
    if isinstance(out, FallbackReason) or not self.reduce_results:
        return out
    # 原函数的最后一步（第 10 步），由 route 自己执行
    from sglang.srt.layers.moe import post_experts_all_reduce

    return post_experts_all_reduce(out)
```

它调用两个辅助函数：

- `_forward_normal_falls_back`：遇到算子不支持的 SGLang 配置时为真，包括这次调用的某些参数（`skip_shared_experts`、`return_moe_output` 等）、图模式和专家并行设置（几个 MoE route 共用的检查，在 `models/moe_blocks.py` 中），以及其他格式的专家权重。
- `_moe_arguments`：从模块中取出算子的参数：`hidden_states`、作为 `router_weight` 的 `mlp.gate.weight`、专家的权重和 scale，以及路由设置。

形状、dtype、token 数和 GPU 型号交给算子判断。route 的完整规范见 [aklr/plugins/README.md](../aklr/plugins/README.md#rules-for-routes)。

### 5. 添加测试输入

在 `tests/plugins/sglang/cases.py` 的 `SMALL_POINTS` 中，为算子添加一个小规模的 Point：

```python
PointSpec(
    contract_id="deepseek_v2_mxfp4_moe@1.0",
    axes={"n_tokens": 4, "model_dim": 256, "n_routed": 8, "inter_dim": 256},
    dtypes={"hidden_states": "bfloat16", "router_weight": "bfloat16", "correction_bias": "float32",
            "w1": "float4_e2m1", "w1_scale": "float8_e8m0", "w2": "float4_e2m1", "w2_scale": "float8_e8m0",
            "out": "bfloat16"},
    attrs={"topk": 8, "num_expert_group": 1, "topk_group": 1, "renormalize": True, "scoring_func": "sigmoid"},
    layout_variant_name="shuffled_16x16",
),
```

它的 baseline 只能在 gfx950 上运行，所以还要把它的 contract id 加入同一文件的 `GFX950_CONTRACT_IDS`。然后更新 `tests/plugins/sglang/names.txt`：

```bash
AKLR_UPDATE_NAMES=1 pytest tests/ops/test_contracts.py
```

route 不需要单独写测试：`tests/plugins/sglang/test_hook.py` 会检查每个 route，确认它替换的函数在当前安装的 SGLang 中存在、参数一致，且没有其他 route 替换同一个函数。

### 6. 编写 kernel 并验证

按 ops 教程的 1. B 到第 4 步进行，场景换成这个算子的：GLM-5 MXFP4 MoE 层在 TP8 下的形状。脚本中 contract、Point 和 workload 不同，其中 `routed_scaling_factor` 是标量，随每个 workload 给出；这个算子已提交的 kernel 用 Triton 编写：

```python
contract = ContractRegistry.default().get("deepseek_v2_mxfp4_moe@1.0")
base = PointSpec(
    contract_id="deepseek_v2_mxfp4_moe@1.0",
    axes={"n_tokens": 1, "model_dim": 6144, "n_routed": 256, "inter_dim": 256},
    dtypes={"hidden_states": "bfloat16", "router_weight": "bfloat16", "correction_bias": "float32",
            "w1": "float4_e2m1", "w1_scale": "float8_e8m0", "w2": "float4_e2m1", "w2_scale": "float8_e8m0",
            "out": "bfloat16"},
    attrs={"topk": 8, "num_expert_group": 1, "topk_group": 1, "renormalize": True, "scoring_func": "sigmoid"},
    layout_variant_name="shuffled_16x16")
workloads = [contract.workload_for(base.model_copy(update={"axes": {**base.axes, "n_tokens": t}}),
                                   {"routed_scaling_factor": 2.5}) for t in (32, 128, 256)]
kernel = {"language": "triton", "targets": [{"arch": "gfx950", "hardware": "MI355X"}],
          "entry_point": "kernel.py::run", "sources": []}
```

schema 会写到这个场景已提交的 kernel 旁边，即 `aklr/plugins/sglang/models/deepseek_v2/impls/triton/deepseek_v2_mxfp4_moe_shuffled_16x16_model_dim6144_n_routed256_inter_dim256/`。验证和提 PR 的做法同 ops 教程。

### 7. 在 SGLang 中运行

validation 检查的是 kernel 与 reference 是否一致，不检查 route；route 要靠运行整个模型来检查。用 GLM-5（MXFP4）在 TP8 下启动 SGLang：

```bash
SGLANG_USE_AITER=1 AKLR_SGLANG_PLUGIN=1 python -m sglang.launch_server \
    --model-path <GLM-5 MXFP4 checkpoint> --tensor-parallel-size 8 --trust-remote-code
```

日志会报告挂载了哪些函数（`aklr hooks N sglang functions on sglang 0.5.22`），模型的回答必须保持正确。

## 下一步

- 插件的结构、route 规范、升级 SGLang：[aklr/plugins/README.md](../aklr/plugins/README.md)。
- 让 agent 编写和调优 kernel：[forge_task](forge_task.md)。
