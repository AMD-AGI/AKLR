# Ops 教程：给 AKLR 的算子加一个 kernel

本教程讲怎样给 `aklr/ops/` 里已有的算子写一个 kernel：生成文件、写 kernel、验证、提 PR，直到推理时用上它。例子是 bf16 矩阵乘（`gemm@1.0`），n=1024、k=4096。最后一节讲没有现成算子时怎么新建。

完整规则见 [`aklr/ops/README.md`](../aklr/ops/README.md)。

## 0. 基本概念

一个算子由五部分组成：

| 部分 | 在哪里 | 是什么 |
|---|---|---|
| contract | `aklr/ops/<op>/contract.py`，一个 `OpContract` | 算子的定义：输入、输出、形状、dtype，每个输出允许多大误差，以及下面的 reference 和 baseline |
| reference | `reference.py`，一个 `OpReference` 类 | 用 torch 按数学定义算出的正确答案 |
| baseline | `baseline.py`，一个 `OpBaseline` 类 | 推理引擎（aiter、sglang、vLLM）现在对这个算子实际跑的代码 |
| api | `api.py`，一个 `@api` 函数 | 推理引擎调用 AKLR 的入口 |
| schema | `impls/<语言>/<场景名>/<uuid>.json` | 一个 kernel：它的源码，以及它负责哪些形状 |

kernel 要过两关：

- **对**：输出和 reference 的差别，不超过 contract 规定的范围（叫误差界）。
- **快**：比 baseline 快。验证时默认要快 1.05 倍以上；推理时的门槛由环境变量 `AKLR_MIN_SPEEDUP` 设置，默认也是 1.05。

几个词：

- **轴**：描述形状的整数，比如 gemm 的 `m`、`n`、`k`。每次调用都可能变的叫 **var 轴**（gemm 的 `m`，即 token 数）。
- **场景**：除了 var 轴，其余都定下来的一组取值（n、k、dtype、有没有 bias 等）。一个 schema 文件对应一个场景。
- **workload**：var 轴的一个取值，比如 `m=256`。schema 里列出 kernel 负责的所有 workload。

推理时，api 收到一次调用，看它是哪个场景、哪个 workload：有对、又够快的 kernel，就运行它；没有，就返回一个 `FallbackReason`，推理引擎运行自己的代码。

## 1. 找到算子和场景

在 `aklr/ops/<op>/contract.py` 里找到对应的 contract。gemm 在 `aklr/ops/gemm/contract.py`：

```python
GEMM = OpContract(
    contract_id="gemm@1.0",
    description="16-bit gemm: out = a @ b.T + bias (b is (k, n) unless trans_b).",
    axes={"m": ..., "n": ..., "k": ...},  # m 是 var 轴
    attrs={"trans_b": ..., "has_bias": ...},  # b 是不是 (n, k)；有没有 bias
    inputs={"a": ..., "b": ..., "bias": ...},
    outputs=...,  # out；bf16 输出的误差界：信噪比（SQNR）不低于 45 dB
    ...
)
```

然后定场景。本教程用：`n=1024`、`k=4096`；`a`、`b`、`out` 都是 bf16；`b` 的形状是 `(n, k)`（`trans_b=True`）；没有 bias；`m` 取 1、2、4……4096。

实际工作中，场景和 `m` 应该来自真实模型：设 `AKLR_EXPORT=1` 跑一次模型，再运行 `python -m aklr export --language <语言>`，就能得到模型实际用到的所有场景（见 `aklr/ops/README.md` 的 Workflow）。本教程直接写死。

没有对应的算子，先看第 6 节。

## 2. 生成 schema 文件

schema 文件不手写，用脚本从 contract 生成。把下面的脚本存成 `~/make_gemm_schema.py`（放在仓库外）：

```python
import sys
from pathlib import Path

from aklr.core.contract.locate import op_dir
from aklr.core.contract.registry import ContractRegistry
from aklr.core.contract.spec import PointSpec, SourceFile

contract = ContractRegistry.default().get("gemm@1.0")

# 场景：除了 m，其余全部定下来
point = PointSpec(
    contract_id="gemm@1.0",
    axes={"m": 1, "n": 1024, "k": 4096},  # m 是 var 轴，这里的 1 只是占位
    dtypes={"a": "bfloat16", "b": "bfloat16", "bias": None, "out": "bfloat16"},  # None：没有这个输入
    attrs={"trans_b": True, "has_bias": False},
    layout_variant_name="row_major",  # 数据布局：普通的按行存放
)

# 要负责的 m：每个值一个 workload
workloads = [
    contract.workload_for(point.model_copy(update={"axes": {**point.axes, "m": m}}), {})
    for m in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
]

# kernel：带一个目录时，读入目录下的 .py；不带时 sources 为空
sources = []
if len(sys.argv) > 1:
    sources = [SourceFile(path=f.name, content=f.read_text(encoding="utf-8")) for f in sorted(Path(sys.argv[1]).glob("*.py"))]
kernel = {
    "language": "triton",
    "targets": [{"arch": "gfx950", "hardware": "MI355X"}],
    "entry_point": "kernel.py::run",
    "requires": {"triton": ">=3.8"} if sources else {},  # 没有 kernel 时不写
    "author": "<你的名字>",
    "sources": sources,
}
schema = contract.narrow(point, workloads=workloads, kernel=kernel)  # 用 contract 生成这个场景的 schema
print("run 的参数：", ", ".join(schema.kernel_call_keys))
if not sources:
    sys.exit()  # 还没有 kernel：只打印参数，不写文件

path = op_dir(contract.op_package) / schema.relative_path()
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(schema.schema_text(), encoding="utf-8")
print(path)
```

在仓库根目录运行，**一定要加 `PYTHONPATH=.`**。不加的话，Python 可能用上别处安装的 aklr，文件也会写到那里去。

```bash
PYTHONPATH=. python ~/make_gemm_schema.py
# run 的参数： a, b, trans_b, bias, out_dtype, out
```

`kernel` 里各项：

- `language`：`triton`、`flydsl`、`hip` 或 `hsaco`。
- `targets`：kernel 能跑的 GPU。别的 GPU 上不会加载它。
- `entry_point`：入口函数，写成 `<文件>::<函数>`，一般是 `kernel.py::run`。
- `requires`：必填。写 kernel 用的语言（`triton`、`flydsl`……）的最低版本，填你开发时用的版本；import 了 torch 以外的库，也要写上。机器上的版本不够时，不会加载这个 kernel。
- `author`：作者，可以不写。

## 3. 写 kernel

在 `~/gemm_kernel/` 下写 `kernel.py`。下面是一个能跑的 Triton kernel：把 `out` 切成 64×64 的块，每块由 GPU 上的一份并行计算负责，沿 `k` 每次累加 64 列。

```python
"""bf16 gemm out = a @ b.T, b as (n, k): one program per 64 x 64 tile of out, k in steps of 64."""

import triton
import triton.language as tl


@triton.jit
def _gemm(a, b, out, m, n, k, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for start in range(0, k, BLOCK_K):
        ks = start + tl.arange(0, BLOCK_K)
        x = tl.load(a + rows[:, None] * k + ks[None, :], mask=rows[:, None] < m, other=0.0)
        y = tl.load(b + cols[:, None] * k + ks[None, :], mask=cols[:, None] < n, other=0.0)
        acc += tl.dot(x, tl.trans(y))
    mask = (rows[:, None] < m) & (cols[None, :] < n)
    tl.store(out + rows[:, None] * n + cols[None, :], acc.to(out.dtype.element_ty), mask=mask)


def run(*, a, b, trans_b, bias, out_dtype, out):
    m, k = a.shape
    n = b.shape[0]
    grid = (triton.cdiv(m, 64), triton.cdiv(n, 64))
    _gemm[grid](a, b, out, m, n, k, BLOCK_M=64, BLOCK_N=64, BLOCK_K=64)
```

`run` 必须遵守的规则（验证时会自动检查，完整版见 `aklr/ops/README.md` 的 "The kernel section and ABI"）：

1. **参数**：就是第 2 步打印出来的那些，全部按名字传入。场景里没有的输入（这里的 `bias`）传 `None`。
2. **结果写进 `out`**：`out` 已经分配好，`run` 把结果写进去，不返回值。
3. **只写输出**：输入只能读，不能改。例外是 contract 里标了 `mutable`（可写）的输入，比如 KV cache。
4. **不留状态**：不把参数算出来的东西留到下一次调用，比如重排过的权重；不用 `global`、`nonlocal`。编译好的 kernel 可以按形状缓存（放在一个 `@functools.cache` 函数里）。
5. **不等 GPU**：只往当前的 torch stream 里提交计算，不调用要等 GPU 算完的操作（`.item()`、`.cpu()`、`torch.cuda.synchronize()` 等）。推理引擎会把调用录成 CUDA graph 反复重放，这类操作在录制时会出错。
6. **没有副作用**：不 `print`，不读写文件，不读环境变量，不改 torch 的全局设置。
7. **临时显存**在 `run` 里面分配（`torch.empty` 等）。
8. **不 import aklr**：只允许 `aklr.core.jit`。
9. **入口文件开头有一段 docstring**，说明这个 kernel 怎么算。

kernel 写好后，再运行一次脚本，这次带上 kernel 目录：

```bash
PYTHONPATH=. python ~/make_gemm_schema.py ~/gemm_kernel
# run 的参数： a, b, trans_b, bias, out_dtype, out
# aklr/ops/gemm/impls/triton/gemm_n1024_k4096/<uuid>.json
```

文件名 `<uuid>` 由文件内容算出：改了 kernel、workload 或作者后再运行，会得到新文件名，旧文件要自己删掉。仓库里已经有这个场景的一个 FlyDSL kernel（`aklr/ops/gemm/impls/flydsl/gemm_n1024_k4096/`），可以对照：除了 `kernel` 一段，以及已提交文件里 CI 写入的测量结果（`accuracy`、`speedup`、`microseconds`），两个文件完全一样。

## 4. 本地验证

验证分两步：audit 检查文件格式、和 contract 是否一致、第 3 步的规则，不需要 GPU；bench 在 GPU 上检查对不对、快不快。先只跑 audit：

```bash
python -m aklr validate --stages audit aklr/ops/gemm/impls/triton/gemm_n1024_k4096/<uuid>.json
```

再在 MI355X 上跑 audit 和 bench。bench 对每个 workload 生成输入，比较 kernel 和 reference 的输出，再把 kernel 和 baseline（这里是 aiter 的 `gemm_a16w16`）交替运行、计时：

```bash
python -m aklr validate --base origin/main --stages audit,bench aklr/ops/gemm/impls/triton/gemm_n1024_k4096/<uuid>.json
```

`--base origin/main` 表示按新 kernel 的标准判：要达到对 baseline 的门槛（默认 1.05），也不能比 main 上同一目录里已有的 kernel 慢。不加的话门槛会松一些，本地过了，CI 也可能不过。

结果写在 `aklr_validation/report.md`。上面这个 kernel 在 MI355X 上的结果（节选）：

```text
## AKLR validation: failed (bench)
| schema | axes | verdict | graph/baseline |
| triton/gemm_n1024_k4096/<uuid> | m=1    | not_faster (demoted) | 0.5000 |
| ...                              | ...    | ...                  | ...    |
| triton/gemm_n1024_k4096/<uuid> | m=4096 | not_faster (demoted) | 0.6375 |
- `no_qualifying_workload` triton/gemm_n1024_k4096/<uuid>: no workload both passes correctness and has a graph/baseline speedup >= 1.05
```

`graph/baseline` 是相对 aiter 的加速比，由多轮计时得出，取偏保守的估计；每一轮的比值限定在 0.5 到 2 之间，所以 0.5 表示只有 aiter 一半快，甚至更慢。这个 kernel 13 个 workload 的输出都对，但没有一个够快（默认要 1.05），所以全部不能用，整个文件不合格。

结果分三种：

| 结果 | 意思 | 怎么办 |
|---|---|---|
| 通过 | 每个 workload 都对，也够快 | 提 PR |
| 部分 workload 标了 `(demoted)`（降级） | 这些 workload 不对（如 `numeric_mismatch`）或不够快（`not_faster`）。推理时它们不用这个 kernel；验证本身不算失败 | 改 kernel，或者从脚本里删掉这些 `m`，重新生成 |
| `no_qualifying_workload` | 没有一个 workload 又对又快，验证失败 | 改 kernel |

精度错误怎么读、慢是不是机器波动造成的，见 `harness/skills/read-validation/SKILL.md`。怎么把 kernel 写快，见 `harness/skills/kb/SKILL.md`。

## 5. 提 PR，合入后什么时候生效

- PR 正文按 `.github/pull_request_template.md` 写，贴上第 4 步的结果，写明场景和 `m` 从哪里来。
- 加 `ci:gpu` 标签，CI 才会在 GPU 上验证。
- 不要自己在文件里写 `accuracy`、`speedup`、`microseconds`：这些只能由 CI 写，手写的话验证会报 `perf_edited`。

PR 合入后，CI 在 main 上再测一遍，由机器人在 `aklr-perf/main` 分支上开（或更新）一个 PR，把测量结果写进文件。**这个 PR 合入后**，推理时才会用上你的 kernel。

推理时，一次调用同时满足下面几条，才会运行你的 kernel：推理引擎打开了 AKLR 插件，调用才会到达 api（见 `aklr/plugins/README.md`）；机器符合 `targets` 和 `requires`；调用的形状是文件里的某个 workload；这个 workload 的 `accuracy` 不是 `failed`；`speedup` 不低于 `AKLR_MIN_SPEEDUP`（默认 1.05）。否则 api 返回 `FallbackReason`，推理引擎运行自己的代码：

| `FallbackReason` | 原因 |
|---|---|
| `no_schema` | 没有加载任何 schema |
| `disabled` | 被 `AKLR_DISABLE` 关掉了 |
| `out_of_contract` | 这次调用超出了 contract 定义的范围 |
| `no_workload` | 这个形状不在任何 schema 的 workload 里 |
| `not_measured` | 还没有测量结果 |
| `inaccurate` | 测量结果是 `failed` |
| `not_qualified` | 测过、也对，但 `speedup` 不到 `AKLR_MIN_SPEEDUP` |
| `not_warmed` | 正在录制 CUDA graph，而这个形状之前没有在录制之外运行过 |

## 6. 新建一个算子

没有对应的算子时，先新建。一个算子就是 `aklr/ops/<op>/` 下的五个文件。下面以 gemm 为例，完整代码见 `aklr/ops/gemm/`。

1. **`contract.py`**：定义 `OpContract`，并把它放进文件里的 `CONTRACTS`（AKLR 会自动找到，不用另外注册）。
   - `op_type` 写目录名 `<op>`。
   - 轴、输入、输出按数学含义命名，不按 baseline 的参数名。输出一般只有一个，叫 `out`；有多个输出时，都不能叫 `out`。
   - 每个输出、每个可写的输入，都要有误差界 `error_bounds`。尽量用 `aklr/core/contract/semantics/_error_bounds.py` 里现成的（`tolerance(...)`、`GEMM_BF16_SNR` 等）。
   - `compare=CompareSpec(determinism=...)`：同样的输入跑两次，结果是否必须完全一样。默认 `bitwise`（完全一样）；kernel 用了原子加、结果随执行顺序变化时，用 `bounded`（每次都在误差界内即可）。
   - 最后写上 `reference=` 和 `baseline=`，即下面两个类的实例。

2. **`reference.py`**：每个 contract 一个 `OpReference` 类。

   ```python
   class GemmReference(OpReference):
       def reference(self, *, a, b, trans_b, out_dtype, bias=None, **_):
           """``a @ b.T + bias``."""
           return {"out": _out(a.float() @ _weight(b, trans_b).mT, bias).to(out_dtype)}
   ```

   - 参数和 kernel 一样，按名字传入；用不到的参数由 `**_` 接住。
   - 返回一个字典：每个输出（以及被改写的可写输入）一项，**都转成 contract 规定的 dtype**。
   - 按数学定义写，不要照抄 kernel 或 baseline。目前中间一律用 float32 算，不做多余的舍入。
   - 有误差界表达不了的规则（比如 top-k 的结果可以是任意顺序），重写 `check` 方法。

3. **`baseline.py`**：每个 contract 一个 `OpBaseline` 类，就是推理引擎对这个算子的原样调用。

   ```python
   class GemmBaseline(OpBaseline):
       def baseline(self, *, a, b, trans_b=True, bias=None, out_dtype=None, out=None):
           """aiter's tuned 16-bit gemm, b as (n, k) only."""
           from aiter.tuned_gemm import gemm_a16w16

           if not trans_b:
               raise NotImplementedError("aiter's gemm_a16w16 takes b as (n, k) only")
           return {"out": gemm_a16w16(a, b, bias=bias, otype=out_dtype)}
   ```

   - 引擎调用什么函数、传什么参数，就原样调用，不重新实现，也不多做事。
   - 在方法里面 import 引擎，这样没装引擎的机器也能 import 这个文件。
   - 引擎不支持的调用，抛 `NotImplementedError`。

4. **`api.py`**：每个 contract 一个 `@api` 函数，函数体只写 `...`：

   ```python
   @api(GEMM)
   def gemm(a, b, trans_b=True, bias=None, out_dtype=None, out=None): ...
   ```

5. **`__init__.py`**：导出 api 函数。

6. **测试**：在 `tests/ops/cases.py` 的 `SMALL_POINTS` 里，按 contract 的顺序加一个小形状的 `PointSpec`（写法同第 2 步脚本里的 `point`），再运行 `AKLR_UPDATE_NAMES=1 pytest tests/ops/test_contracts.py`，重新生成 `tests/ops/names.txt`，一起提交。

多张卡一起调用的算子（比如专家并行），在 contract 里设 `ranks`，reference 和 baseline 改用 `DistOpReference`、`DistOpBaseline`，见 `aklr/ops/README.md`。

新算子的代码单独提一个 PR，比 kernel 先合入。

## 下一步

一个 kernel 只替换一个算子。想把推理引擎里的好几步合成一次计算（比如整个 MoE 层），见 [SGLang 插件教程](sglang_plugin_tutorial.zh.md) 和 [vLLM 插件教程](vllm_plugin_tutorial.zh.md)。
