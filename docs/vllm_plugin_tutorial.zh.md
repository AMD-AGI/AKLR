# 插件教程：替换 vLLM 的函数

vLLM 里有些函数，aiter 没有对应的算子：可能是模型自己的函数，也可能是跑完一整段计算、要被融成一个算子的函数。插件就是把这样的函数换成 AKLR 的算子。本教程的示例是 `kimi_k3_attn_res`，它接管 Kimi-K3 的 attention residual。vLLM 为这个模型自带了一个 Triton kernel `attn_res`，每层的 attention 和 MLP 之前各跑一次，最后的 norm 之前再跑一次。

开始之前，请先阅读 [Ops 教程](ops_tutorial.zh.md)：插件里的算子写法与之相同。SGLang 插件的做法见 [SGLang 插件教程](sglang_plugin_tutorial.zh.md)，插件的结构和规范见 [aklr/plugins/README.md](../aklr/plugins/README.md)。

## 为什么需要插件

Kimi-K3 的 attention residual 在 vLLM 里是一个 Triton kernel，即 `attn_res`（`models/kimi_k3/amd/ops/attn_res.py`），由 `_apply_attn_res`（`models/kimi_k3/amd/linear.py`）带上该层的权重来调用。每个 token 有一个 bank：第一行是 embedding，之后每 12 层算一块，每块各层输出之和占一行；当前这一块到目前为止的累加另外存着，叫 prefix。对每个 token，这个 kernel 做下面几步：

| 步骤 | 内容 | vLLM 的 `attn_res` | `kimi_k3_attn_res` |
|---|---|---|---|
| 1 | 把还没并入的上一层输出 `delta` 加到 prefix 上，舍入为 bf16 后写回 prefix | ✓ | ✓ |
| 2 | 在每块的第一层，把 prefix 写成 bank 的下一行 | ✓ | ✓ |
| 3 | 给每个来源打分（bank 中已写的各行，再加上 prefix）：先做 RMS 归一化，再投影成一个分数 | ✓ | ✓ |
| 4 | 对分数做 softmax，按权重把各来源加起来，得到 mix | ✓ | ✓ |
| 5 | 对 mix 做该层的 RMSNorm：attention 前用 `input_layernorm`，MLP 前用 `post_attention_layernorm`，最后的 norm 之前不做 | ✓ | ✓ |
| 6 | 下一个 linear 吃 fp8 输入时，把输出按 token 量化为 fp8 | ✓ | — |

这几步在 vLLM 里本来就是一个 kernel，但这是模型自己的 kernel，放在 Kimi-K3 的模型目录里，不在 aiter 里。aiter 没有对应的算子，AKLR 的 aiter provider 也就碰不到这次调用；`aklr/ops` 里也没有做这个计算的算子。想换上更快的 kernel，只能由插件替换模型自己的这个函数，也就是 `_apply_attn_res` 调用的 `attn_res`。算子接管第 1–5 步；第 6 步的 fp8 输出没有 contract 描述，遇到时 route 把调用交回 vLLM 的 kernel。

插件能替换什么：route 不替换 aiter 的函数，也不替换只是给一次 aiter 调用包了层壳的框架函数，比如 vLLM `_aiter_ops.py` 里的 `_rocm_aiter_fused_moe_impl`，它只是调用 aiter 的 `fused_moe`。aiter 算子的调用，走 aiter 自己的插件接口进入 AKLR，AKLR 在那边的 provider 是 `aklr/plugins/aiter/provider`（见 [aiter 的 kernel 插件接口](../aklr/plugins/README.md#aiters-kernel-plugin-interface)）。route 只做两种事：要么替换模型自己的函数，本例就是这种；要么把框架里连同 aiter 调用在内的一整段计算融成一个算子。

因此：

- 只需让某一个算子更快时，在 `aklr/ops` 中写算子；如果 aiter 对这个算子做了路由，它的调用会经 aiter provider 交给 AKLR。
- 要替换的是模型自己的函数，或者优化要把框架里跨越多个算子的一段计算融到一起时，写插件。

## 插件如何运行

设置 `AKLR_VLLM_PLUGIN=1` 后，每个 worker 在绑定设备后挂载 route。此后每次调用被替换的函数，都会先执行 route：route 把调用交给算子，或者返回 `FallbackReason`，把调用交回 vLLM 原有的代码（见 [调用流程](../aklr/plugins/README.md#call-flow)）。

## 步骤

### 1. 确定要替换的函数

即 vLLM 的这次调用：

- `_apply_attn_res` 调用的 `attn_res`（`models/kimi_k3/amd/linear.py`）：完成上表第 1–6 步，返回输出，同时原地更新 prefix 和 bank。在每个 decoder 层里，`_apply_attn_res` 于 attention 前和 MLP 前各调用一次（`KimiDecoderLayer.forward_attn_residual`），最后的 norm 之前再调用一次（`KimiLinearModel.forward`）。Kimi-K3 有 93 层，一次 forward 共调用 187 次。

`_apply_attn_res` 做的只是把该层模块里的张量取出来，输出为 fp8 时再给结果包一层；`attn_res` 的参数本身就是算子要的参数，所以 route 替换的是 `attn_res`。它在 vLLM 0.34.0 中的参数如下，route 要原样照抄：

```python
def attn_res(prefix, delta, blocks, norm_weight, qk_weight, output_norm_weight, num_blocks, block_write_idx,
             eps, output_norm_eps, *, quant_dtype=None): ...
```

### 2. 建包

新建 `aklr/plugins/vllm/models/kimi_k3/attn_res/`：`kimi_k3/` 对应 vLLM 的源目录 `models/kimi_k3/`，`attn_res/` 对应 kernel 所在的文件 `ops/attn_res.py`。包内的文件与 `aklr/ops` 中的算子相同：`__init__.py`、`contract.py`、`reference.py`、`baseline.py`、`api.py` 和 `impls/`；另有 `_triton.py`，是 vLLM 这个 Triton kernel 及其启动函数的副本（去掉了 fp8 输出），让 baseline 不依赖 vLLM 也能运行。包名即算子的 `op_type`（`attn_res`），在 `aklr/ops` 和 `aklr/plugins` 中都不能重名。

### 3. 编写算子

做法同 [Ops 教程](ops_tutorial.zh.md)的第 6 节。针对这个算子：

- `contract.py` 定义 `KIMI_K3_ATTN_RES`（`kimi_k3_attn_res@1.0`）：
  - 输入：当前的 `prefix`；还没并入 prefix 的上一层输出 `addend`（`has_addend` 时才有）；每个 token 的 `bank`；打分用的两个权重 `score_norm_weight` 和 `score_proj_weight`；该层 norm 的 `norm_weight`（`has_norm` 时才有）；以及标量 `score_eps` 和 `norm_eps`；
  - attr：`n_valid_rows` 是 bank 里已经写了几行（0 到 8），mix 要把这些行全部读进来；`write_bank` 表示是否同时把 prefix 写成 bank 的第 `n_valid_rows` 行；另有 `has_addend` 和 `has_norm`。vLLM 实际用到的每种组合各算一个场景；
  - 布局：只有 `row_major`；
  - 输出：`out`，形状 `[n_tokens, hidden_size]`。有 `norm_weight` 时，是归一化后再乘以 `norm_weight` 的 mix，没有时就是 mix 本身。`prefix` 和 `bank` 是可变输入，算子像 vLLM 的 kernel 一样原地更新它们，误差界为零，要求逐位一致。
- `reference.py`：`KimiK3AttnResReference`，在 float32 下按 contract 计算。它把 `prefix + addend` 舍入为 prefix 的 dtype，和 vLLM 的 kernel 写回时一样，输出也舍入为这个 dtype。它返回调用结束时的 `out`、`prefix` 和 `bank`。
- `baseline.py`：`KimiK3AttnResBaseline`，即 vLLM 目前的做法：用 `_triton.py` 里的副本，按 vLLM `attn_res` 的启动配置，每次调用启动一次这个 Triton kernel。
- `api.py`：算子的入口函数 `@api(KIMI_K3_ATTN_RES)`，函数体为 `...`。

### 4. 编写 route

route 写在 `api.py` 中，放在算子入口函数 `kimi_k3_attn_res` 旁边。它把 vLLM 的参数换成算子的参数，再调用 `kimi_k3_attn_res`：

- `blocks`、`norm_weight`、`qk_weight` 分别对应算子的 `bank`、`score_norm_weight`、`score_proj_weight`；`delta` 和 `output_norm_weight` 对应可选的 `addend` 和 `norm_weight`；`eps` 和 `output_norm_eps` 对应 `score_eps` 和 `norm_eps`。
- `num_blocks` 就是 `n_valid_rows`；`block_write_idx` 不小于 0 就是 `write_bank`。vLLM 写 bank 时只写第 `num_blocks` 行，正好是 contract 写的那一行。
- contract 没有描述的情况，不调用算子，直接交回 vLLM：要求 fp8 输出（`quant_dtype`），或者要写的不是第 `num_blocks` 行。

`@vllm_route` 的第一个参数是调用方，第二个是被替换的函数，名字要写成调用方在运行时查找它所用的那个：`linear.py` 从 `ops/attn_res.py` 导入了 `attn_res`，所以 route 替换的是 `vllm.models.kimi_k3.amd.linear.attn_res`。`versions` 列出测过这个 route 的 vLLM 版本：

```python
@vllm_route(
    "vllm.models.kimi_k3.amd.linear._apply_attn_res",
    "vllm.models.kimi_k3.amd.linear.attn_res",
    versions=("0.32.1rc1.dev276", "0.33.0rc1.dev910", "0.33.0", "0.34.0"),
)
def _attn_res(
    prefix,
    delta,
    blocks,
    norm_weight,
    qk_weight,
    output_norm_weight,
    num_blocks,
    block_write_idx,
    eps,
    output_norm_eps,
    *,
    quant_dtype=None,
):
    """Every attention residual of Kimi-K3's decoder layers and the one before its final norm.
    vLLM writes the bank only at row num_blocks."""
    # the fp8 output for a linear that takes per-token fp8 input, which no contract states
    if quant_dtype is not None or block_write_idx not in (-1, num_blocks):
        return FallbackReason.OUT_OF_CONTRACT
    return kimi_k3_attn_res(
        prefix,
        blocks,
        norm_weight,
        qk_weight,
        n_valid_rows=num_blocks,
        write_bank=block_write_idx >= 0,
        addend=delta,
        norm_weight=output_norm_weight,
        score_eps=eps,
        norm_eps=output_norm_eps,
    )
```

算子接这次调用时返回 `out`，route 把它交还给 `_apply_attn_res`，和 vLLM 的 `attn_res` 返回的东西一样，`prefix` 和 `bank` 也已原地更新；不接时返回 `FallbackReason`，由 vLLM 的 `attn_res` 照常执行。token 数、行数、GPU 型号由算子来判断，route 不管：第一层 attention 前的那次调用还没有 bank 行，route 照样交给算子，算子没有对应的 kernel，就不接。

### 5. 添加测试输入

在 `tests/plugins/vllm/cases.py` 的 `SMALL_POINTS` 中为算子添加小规模的 Point，contract 的基准 Point 放在最前面。这个算子有四个，每种调用一个：把上一层输出并入 prefix；写 bank，带 addend 和不带 addend 各一个；最后的 norm 之前那次，不做该层 norm。第一个是：

```python
PointSpec(
    contract_id="kimi_k3_attn_res@1.0",
    axes={"n_tokens": 4, "hidden_size": 256, "n_bank_rows": 4},
    dtypes={
        "prefix": "bfloat16",
        "addend": "bfloat16",
        "bank": "bfloat16",
        "score_norm_weight": "bfloat16",
        "score_proj_weight": "bfloat16",
        "norm_weight": "bfloat16",
        "out": "bfloat16",
    },
    attrs={"n_valid_rows": 2, "write_bank": False, "has_addend": True, "has_norm": True},
    layout_variant_name="row_major",
),
```

同一文件里的 `GFX950_CONTRACT_IDS` 列的是 baseline 只能在 gfx950 上运行的 contract；vLLM 的这个 Triton kernel 不属于这种情况，所以不用加。然后更新 `tests/plugins/vllm/names.txt`：

```bash
AKLR_UPDATE_NAMES=1 pytest tests/ops/test_contracts.py
```

`tests/plugins/vllm/test_hook.py` 会检查每个 route，确认它替换的函数在当前安装的 vLLM 中存在、参数一致，且没有其他 route 替换同一个函数。route 怎样把 vLLM 的参数对到算子上，`test_hook.py` 检查不到，所以要在 `tests/plugins/vllm/test_kimi_k3_attn_res.py` 中单独测试：要求 fp8 输出、或者在最后一行之前写 bank 的调用，不经过算子就交回 vLLM；受支持的调用，每个操作数都按名字交给算子；baseline 所用的 Triton kernel 副本，输出、更新后的 prefix 和 bank 都与 vLLM 逐位一致。

### 6. 编写 kernel 并验证

按 [Ops 教程](ops_tutorial.zh.md)的第 2 到第 5 节进行，场景换成这个算子的：Kimi-K3 的 attention residual，hidden size 为 7168；93 层每 12 层一块，bank 共 8 行（见 [Kimi-K3 的模型结构](../aklr/benchmarks/models/kimi_k3/README.md#model-structure)）。一次 forward 的 187 次调用里，170 次带 addend、做该层 norm，bank 行数从 1 到 8；7 次还要写 bank；8 次没有 addend，是每块第一层 MLP 前的那次；最后的 norm 之前那 1 次不做该层 norm；第一层 attention 前那 1 次还没有 bank 行。attr 的每种组合各算一个场景。脚本中 contract、Point 和 workload 不同，其中 `score_eps` 和 `norm_eps` 是标量，随每个 workload 给出（不做该层 norm 时，`_apply_attn_res` 传入的 `norm_eps` 是 0.0）；token 数取 1 到 4 个请求的 decode step，每个请求 3 个 token。这个算子已提交的 kernel 用 FlyDSL 编写（脚本里 `language` 写 `flydsl`，`requires` 写 `{"flydsl": ">=0.3.2"}`）：

```python
contract = ContractRegistry.default().get("kimi_k3_attn_res@1.0")
point = PointSpec(
    contract_id="kimi_k3_attn_res@1.0",
    axes={"n_tokens": 1, "hidden_size": 7168, "n_bank_rows": 8},
    dtypes={"prefix": "bfloat16", "addend": "bfloat16", "bank": "bfloat16", "score_norm_weight": "bfloat16",
            "score_proj_weight": "bfloat16", "norm_weight": "bfloat16", "out": "bfloat16"},
    attrs={"n_valid_rows": 8, "write_bank": False, "has_addend": True, "has_norm": True},
    layout_variant_name="row_major")
workloads = [contract.workload_for(point.model_copy(update={"axes": {**point.axes, "n_tokens": t}}),
                                   {"score_eps": 1e-5, "norm_eps": 1e-5}) for t in (3, 6, 9, 12)]
```

schema 会写到这个场景已提交的 kernel 旁边，即 `aklr/plugins/vllm/models/kimi_k3/attn_res/impls/flydsl/kimi_k3_attn_res_hidden_size7168_n_bank_rows8/`；其他场景的名字里带着各自的 attr，例如 `kimi_k3_attn_res_rows3_write_hidden_size7168_n_bank_rows8`。没有 bank 行的那次调用没有 schema，留给 vLLM 的 kernel。验证和提 PR 的做法同 ops 教程。

### 7. 在 vLLM 中运行

validation 检查的是 kernel 与 reference 是否一致，不检查 route；route 要靠运行整个模型来检查。用 Kimi-K3 的 MXFP4 checkpoint 在 TP8 下启动 vLLM，带上部署本身的参数，并打开插件。这个 checkpoint 的 attention 和 dense MLP 都是 BF16，不会有调用要求 fp8 输出（那种调用 route 会交回 vLLM）：

```bash
AKLR_VLLM_PLUGIN=1 vllm serve <Kimi-K3 MXFP4 checkpoint> --tensor-parallel-size 8
```

日志会报告挂载了哪些调用（`aklr hooks N calls of vLLM 0.34.0`），模型的回答必须保持正确。设置 `AKLR_TRACE=plugin` 后，每种不同的调用，route 都会记一次日志：接没接管，没接管的原因是什么（见 [aklr/plugins/README.md](../aklr/plugins/README.md#enabling-the-plugin-and-upgrading-sglang)）。

## 下一步

- 插件的结构和 route 规范：[aklr/plugins/README.md](../aklr/plugins/README.md)。
- 让 agent 编写和调优 kernel：[forge_task](forge_task.md)。
