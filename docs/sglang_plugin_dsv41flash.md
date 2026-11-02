# AKLR sglang 插件：DeepSeek-V4.1-Flash decode 加速

AKLR 以 sglang 插件的形式加速 DeepSeek-V4.1-Flash（下称 DSV4.1）的 decode：sglang 源码不改，装上 AKLR
即生效。DSV4.1 的 decode / DSpark verify 步骤由 Infera-Flash（IF）调过的 mono kernel 执行，每轮之间的
metadata、采样、accept / inject 等小 kernel 也尽量合并成一次 launch；任何不支持的配置或步骤都回到 sglang
自己的路径。

## 1. 架构

```
sglang 进程（launch_server / 每个 scheduler）
  └─ entry point sglang.srt.plugins: aklr = aklr.plugins.sglang:register
       ├─ AKLR_SGLANG_PLUGIN=1 且未被 AKLR_DISABLE 关掉时：
       │    BACKENDS: aklr.plugins.sglang.models.deepseek_v4.register()
       │    install(): 各 @sglang_route 以 HookRegistry AROUND 挂到 sglang 方法上
       └─ deepseek_v4.register()（sglang 需有 monokernel 插件 API sglang.srt.monokernels，没有时只打一条
            WARNING，模型由 sglang 跑）
            monokernels.spec.register(backend="aklr", DeepseekV4ForCausalLM / ...DSpark)
            包一层 handle_monokernel 解析步骤：没给 --monokernel-backend（或给的是 amd）时，
            用 sglang 自己的启动检查（monokernel_disable_reasons）验证，通过则选 aklr
            → sglang 的 runtime 在加载后 build、按 CUDA graph key 路由
```

每层两次 launch：K1（attention 的 mHC seam、wqkv、q/kv norm、KV 写入、wq_b + RoPE）和 K2（稀疏
attention、wo_a、wo_b、attention all-reduce、FFN seam、路由、共享专家、路由专家、MoE all-reduce）；
indexer 层在两者之间跑 K2i（FP4 index key 打分）+ sglang 自己的 top-k。256 个常驻 CTA（一个 CU 一个），
stage 之间用带 epoch 的 mailbox，跨卡用 uncached IPC 内存。DSpark 草稿模型的 3 个 stage 同样走 K1/K2
（128 专家 top-3）。

### 文件

| 路径 | 内容 |
|---|---|
| `aklr/plugins/sglang/__init__.py` | entry point `register()`、`BACKENDS`（某个 backend 注册失败只记一条 WARNING，撤掉它的路由）、`[aklr]` 日志 handler |
| `aklr/plugins/sglang/hook.py` | `@sglang_route`（含 `around=True` 生命周期路由）、`install()`、`RouteFallback` |
| `aklr/plugins/sglang/stats.py` | 路由计数（served / fallback + 原因）、device 计数器（graph replay 也计）、导出 |
| `aklr/plugins/sglang/models/deepseek_v4/__init__.py` | `register()`、自动选 backend |
| `.../deepseek_v4/spec.py`, `draft.py` | 目标模型 / DSpark 草稿的 MonoSpec 与 forward（来自 sgl-amd-mono b27d604 的 glue，改为 AKLR 包路径和计数器） |
| `.../deepseek_v4/routes.py` | 两个 forward 路由：记录 sglang 自己跑的步骤及原因 |
| `.../deepseek_v4/compat.py` | glue 从 sglang monokernel runtime 读到的类型和函数 |
| `.../deepseek_v4/runtime_routes.py` | 包住 sglang 的 `install_monokernel_forward`：启动检查不通过（sglang 会报错退出）时留下 sglang 的 forward 并 WARNING |
| `.../deepseek_v4/*_routes.py`, `accept.py`, `proposal.py`, `inject.py`, `prologue.py`, `embed.py`, `engram.py`, `index_q.py`, `dp_routes.py` | mono 之外每轮的融合（§3 的开关） |
| `.../deepseek_v4/kernels/` | vendored 的 SGLang-AMD-Mono / vLLM mono kernel 与 IF 的 HIP kernel，`VENDORED.md` 记录出处 |
| `tests/plugins/sglang/deepseek_v4/` | 插件单测（sglang 用 stub，`_monokernel_spec.py` 是 stub 用的 sglang 契约类型副本）；`test_routes.py`：每个路由目标在装好的 sglang 里存在、参数能绑定（需 GPU，没装 sglang 时跳过） |
| `aklr/plugins/README.md` | 插件框架说明（路由、回退、开关、加一个 backend 的步骤） |

### 为什么是插件内 kernel 包，而不是 schema 契约

AKLR 的常规做法是 schema 化的 op（契约 + 实现 JSON + 基线 + 精度/性能门槛）。DSV4.1 的 mono kernel
是有状态的整层 runner：一次 build 要拿 sglang 已加载的权重（原地读取，少数换布局的副本）、sglang 的
KV cache / 稀疏索引 / 注意力 metadata，以及跨 rank 的 IPC buffer 和 mailbox，launch 之间还要调用
sglang 自己的 compressor 和 top-k。把它拆成无状态契约既要重写 glue，又拿不到可比的单 op 基线。
sglang 0.5.22 已经提供了 monokernel 插件 API（RFC #43949），它的契约（MonoSpec：设备、TP、宽度、
特性、数值界、启动检查、weight 观察、按 graph key 路由）正好描述这种 kernel，所以这里直接用它，
AKLR 负责：kernel 与 glue 的归属、启用开关、自动选择、回退原因的记录与导出。

## 2. 如何启用

前提：gfx950（MI350X / MI355X），TP2 或 TP4，DSV4.1-Flash，DSpark（或无投机），即 IF 的 InferenceX 配方：

```bash
AKLR_SGLANG_PLUGIN=1 SGLANG_USE_AITER=1 AITER_BF16_FP8_MOE_BOUND=0 SGLANG_DSPARK_USE_GREEDY_DRAFT=1 \
python3 -m sglang.launch_server --model-path <DeepSeek-V4.1-Flash> --tp 2 \
  --speculative-algorithm DSPARK --speculative-dspark-block-size 5 \
  --enforce-shared-experts-fusion --fp8-gemm-backend aiter ...
```

需要 `AKLR_SGLANG_PLUGIN=1`（默认 0：插件不挂接 sglang，与 main 相同）。不给 `--monokernel-backend` 时自动选
`aklr`；给 `amd` 时，启动检查通过则由 `aklr` 代替（`AKLR_DSV41_TAKEOVER=0` 时保留 sgl-amd-mono）；也可以显式
`--monokernel-backend aklr`。`SGLANG_DSPARK_USE_GREEDY_DRAFT=1` 是 sglang 自己的开关
（argmax 草稿，与 IF 相同）；不设时草稿逐行采样，Markov 步骤走采样版 kernel（`AKLR_DSV41_SIM_MARKOV`，至多 8 条
序列）或 sglang 的。

- `AKLR_SGLANG_PLUGIN=0` 或不设：不挂接 sglang（`AKLR_DISABLE=1` 关掉全部 AKLR；`AKLR_DISABLE=aklr.plugins` 只关插件）。
- `AKLR_DISABLE=aklr_deepseek_v41`：只关某个 monokernel backend（逗号分隔，与 contract id、命名空间混写；
  名字不带 `@`）。关掉目标模型的 `aklr_deepseek_v41` 时草稿 `aklr_deepseek_v41_dspark_draft` 也不建（草稿依赖目标
  模型的 build）；只关草稿则目标模型照常。
- `AKLR_DSV41_MONO=0`：插件开着，但 DSV4.1 的 mono kernel 与 §3 的所有路由都不装（只有框架）。
  `AKLR_DISABLE=aklr_deepseek_v41` 只关 mono kernel，§3 的融合照常。
- `AKLR_EXPORT_DIR=<dir>`：每个进程把路由计数、device 计数器、build 信息写到 `route_calls_<pid>.json`。
- `AKLR_LOG_LEVEL`：`[aklr]` 日志级别（默认 INFO）。
- `AKLR_PLUGIN_MIN_SPEEDUP`：插件 op（`aklr.plugins.*`）单独的测量加速比门槛；不设时同 `AKLR_MIN_SPEEDUP`。
  DSV4.1 后端不是 op 包，不受它影响。

启动日志里应看到：

```
[aklr ...] INFO <n> sglang methods routed
[aklr ...] INFO DeepSeek-V4.1: monokernel backend 'aklr' selected
monokernel aklr_deepseek_v41 on gfx950: widths [6, 12, ..., 48] built in ... s, ... MiB; ...
monokernel aklr_deepseek_v41 on gfx950 runs target_verify steps of batch size [1, 2] x 6 tokens (...)
[aklr ...] WARNING sglang.srt.models.deepseek_v4.DeepseekV4ForCausalLM.forward falls back to sglang: extend
```

prefill（`extend`）本来就由 sglang 跑，每个进程记一次。

### 回退

回退而不是起不来：任何一项检查不通过（TP 不是 2/4、DP attention、unified KV、非 DSpark 的投机算法、KV cache
dtype / MoE / FP8 GEMM backend 不符、`AKLR_DISABLE` 点名、检查本身抛异常、各 rank 的结论不一致……）都在 rank 0
打一条 WARNING（列出全部原因，并写进 `AKLR_EXPORT_DIR` 的 info `<backend>:refused`），sglang 原 forward 跑所有步骤。
显式给 `--monokernel-backend aklr` 而检查不通过，sglang 自己的 `install_monokernel_forward` 会
报错退出；AKLR 包住这一步（`runtime_routes.py`），只把 `--monokernel-backend aklr: ...` 这类拒绝改成回退，其余异常
照常抛出。sglang 版本不在 `>=0.5.22,<0.5.23` 时只打一条 WARNING、不装任何 DSV4.1 路由；backend 的 `register()`
抛异常时撤掉它已登记的路由。build 成功后，宽度（batch size × 每请求 token 数）不在已 build 集合里的步骤、prefill、
mixed 等其它步骤走 sglang 原 forward 并按原因计数。

§3 的每轮融合只在"本进程服务 DeepSeek-V4.1"时生效（`deepseek_v4.serving()`）：目标模型 `__init__` 时
`model_type` 为 `deepseek_v41`、GPU 为 gfx950、TP 为 2 或 4。DeepSeek-V4 / V4-Pro（同一个模型类）、其它模型的
DSpark 草稿、其它 GPU（如 gfx942）或 TP 大小都保留 sglang 的代码，DeepSeek-V4.1 在这些条件之外时打一条 WARNING。
满足时每项还有自己的前提（DSpark、static verify graph、行数上限、HIP kernel 能否编译等），不满足时走 sglang 的
原路径。metadata、prologue、草稿采样参数、post / combine、aux 打包和 prefill JIT 在首次 eager 调用时与 sglang 的
结果逐字段 / 逐位比对，不一致则该项关闭并写进 info；accept 交换、Markov、inject 的 kernel 由单测对 sglang 逐位验证。

### 安装

DSV4.1 插件用镜像自带的 aiter 和 FlyDSL，不替换它们：`aklr` 包本身不 import aiter，只需把 `aklr/` 放上
`sys.path`（`.pth`）并写一个 `aklr-0.0.0.dist-info/entry_points.txt`（`[sglang.srt.plugins] aklr = aklr.plugins.sglang:register`）。
mono kernel 用到的 aiter FlyDSL helper 是插件自带的拷贝（`kernels/VENDORED.md`），在镜像的 FlyDSL 上编译运行（0.4.0 与 0.3.4.1 都可以）。
插件加载时 `load_routes(strict=False)` 代理镜像 aiter 的入口：`aklr.ops` 的路由描述的是 `3rdparty/aiter` 钉的
aiter，签名不同的入口（`-20261021` 镜像的 aiter 0.1.26 上有 mhc、fused_moe、gemm blockscale、paged MQA logits 共 6 个）
保留 aiter 原函数并打一条 WARNING，其余照常代理。

## 3. 每轮的融合与开关

mono kernel 之外，每轮 decode 还有 metadata、草稿采样、accept / inject、Engram、indexer 等几十个小 kernel
（MI355X 的 graph replay 里每个约 4 µs）。下面每一项都有开关，`=0` 回到 sglang 的原路径。

| 开关 | 内容 |
|---|---|
| `AKLR_DSV41_TAKEOVER` | 启动检查通过时用 `aklr` 代替 `--monokernel-backend amd` |
| `AKLR_DSV41_TARGET_MONO` | 目标模型的层走 IF 的 vllm_mono kernel；`=sgl` 时走 SGLang-AMD-Mono 的 |
| `AKLR_DSV41_SKIP_C4C128` | served verify 步骤不建 ratio-4 / ratio-128 的压缩 metadata（DSV4.1 没有这两种层），只放同 dtype 的占位 |
| `AKLR_DSV41_PROLOGUE` | `expand_extend_with_same_length` 与 `get_swa_page_indices` 各一次 Triton launch |
| `AKLR_DSV41_VERIFY_ROWS` | served verify 步骤与 DSpark 草稿 block 的全部 metadata 各一次 launch |
| `AKLR_DSV41_COUNT_IN_META` | served 计数由 metadata kernel 加一，不单独 launch |
| `AKLR_DSV41_INDEX_Q` | indexer query 一次 launch（IF 的 index_q，直接写 K2i 的 FP4 布局） |
| `AKLR_DSV41_INDEX_Q_WIDE` | 超过 8 行（bs≥2）时 indexer query 改用一次 Triton launch（`index_q.py`：GEMM 走矩阵核心，不再按 8 行一组重复读权重），首次调用与 HIP kernel 一样和 sglang 比对 |
| `AKLR_DSV41_GATE_IN_Q` | ratio-2 compressor 的 wkv_gate GEMV 放进 index_q 的 launch |
| `AKLR_DSV41_REINDEX_ONCE` | 同一步内 ratio-1 indexer 层的候选表只重排一次 |
| `AKLR_DSV41_EMBED` | token embedding 用每个 rank 上的整表（无 all-reduce；每个 rank 多占约 1.2 GiB） |
| `AKLR_DSV41_ENGRAM` | Engram 的查表和走 custom all-reduce（BF16），image 行的 where 折进 gate |
| `AKLR_DSV41_ENGRAM_IDS` | Engram 各层的 hash id 一次转成按层连续 |
| `AKLR_DSV41_ENGRAM_POST` | Engram 层之前那一层的 mHC post 在 gate kernel 里算 |
| `AKLR_DSV41_POST_COMBINE` | 最后一层（及草稿最后一个 stage）的 mHC post 与 head 的 combine 一次 launch |
| `AKLR_DSV41_COMBINE_BF16` | head 的 combine 直接读 bf16 residual（去掉 `.float()` 拷贝） |
| `AKLR_DSV41_PACK_AUX` | DSpark 目标层的 hc 均值直接写进打包的 aux 张量（去掉 head 前的 cat） |
| `AKLR_DSV41_DRAFT_HEAD_FP8` | 草稿的 vocab 投影用 FP8（行 scale） |
| `AKLR_DSV41_MARKOV` | greedy 草稿的 Markov 步骤用 IF 的 markov_step kernel |
| `AKLR_DSV41_SIM_MARKOV` | 采样草稿的 Markov 步骤每步一对 kernel（bias GEMV、与 sglang 相同的 Philox Gumbel-max、kernel 内 rank 交换），至多 8 条序列，每 4 条一个 wave 一条并行打分 |
| `AKLR_DSV41_MARKOV_BUMP` | 采样块的 Philox offset 由最后一步的 finish kernel 推进 |
| `AKLR_DSV41_DRAFT_PARAMS` | `resolve_draft_params` 的 clamp / le / where 一次 launch |
| `AKLR_DSV41_SIM_DRAFT` | 模拟接受下的采样草稿只交换每块候选，不 all-gather 整个 vocab |
| `AKLR_DSV41_ACCEPT_XCHG` | verify 的 greedy accept 在各 rank 的 vocab 分片上做 argmax 交换，不 gather 全 vocab logits |
| `AKLR_DSV41_SIM_XCHG` | 模拟接受的 verify 同样走分片 argmax 交换，模拟长度替换贪心长度 |
| `AKLR_DSV41_SIM_ONE` | 模拟长度（rank 0 的）放进交换包，commit 长度与 out tokens 在同一 kernel 里算完 |
| `AKLR_DSV41_SIM_FOLD` | 采样 batch 的模拟接受折进 verify graph，不在 graph 之后 eager 重做 |
| `AKLR_DSV41_SIM_KEEP` | 相同 batch size 的连续模拟步骤之间 commit 保持 armed |
| `AKLR_DSV41_INJECT_STORE` | DSpark verify 提交的 KV 一次 launch 写进每个草稿 stage 的 SWA cache |
| `AKLR_DSV41_INJECT_COMMITTED` | inject 的 SWA 写入 kernel 内算 committed 槽位 |
| `AKLR_DSV41_RADIX_WALK` | unified radix cache 的 insert 按偏移走，不再每步拷贝剩余 key |
| `AKLR_DSV41_DP_PAD_MASK` | DP attention 下，每个 MoE 层 DP gather 前把 pad 行清零的 arange / 比较 / masked fill 合成一次 launch（按设备上的有效行数） |
| `AKLR_DSV41_TOKEN_CACHE` | 长 prompt 按特殊 token 分段，每段编码一次并缓存（前几次与整段编码比对） |
| `AKLR_DSV41_PREFILL_JIT` | FP4 prefill MQA logits kernel 的 block-table stride 作为参数（只编译一次）；prefill schedule 的 Triton kernel 不做整数特化，启动时预热 |

结论（每项是否生效、首次比对的结果）写在 `AKLR_EXPORT_DIR` 导出的 info 里。

## 4. sglang 构建

要求 sglang 0.5.22 带 monokernel 插件 API（RFC #43949），如 InferenceX 用的
`lmsysorg/sglang-rocm:v0.5.22-rocm10-mi35x-20261021`（sglang 0.5.22.dev20261021，FlyDSL 0.4.0）。release/v0.5.22
（`lmsysorg/sglang:v0.5.22-rocm10-mi35x`，CI 镜像）没有这套 API，DSV4.1 由 sglang 自己跑。各模块的 `SGLANG_READS`
列出 glue 读到的 sglang 名字，`tests/plugins/sglang/deepseek_v4/test_routes.py` 在装好的 sglang 上检查路由目标。

## 5. 已知限制

- 只支持 gfx950、TP2/TP4、单机、无 DP attention / EP / PD、带 monokernel 插件 API 的 sglang；其余配置回退 sglang 并记录原因。
- 首次启动：mono kernel 在首次启动时编译（FlyDSL 的磁盘缓存之后复用）。
- kernel 原地读 sglang 的权重并保留少量副本，所以不支持在线权重更新（与 sgl-amd-mono 相同）。
- 与 IF 单独运行的剩余差距在 sglang 框架里：graph 之间的 eager 调度、logits 的 all-gather、DSpark 的 TP 同步。

## 6. 测试

```bash
python3 -m pytest -q tests/plugins tests/core/platform/test_env.py
```

插件单测用 stub 的 sglang，不需要模型权重；kernel 的单测（`test_deepseek_v4_*.py`）需要 gfx950，其它 GPU 上跳过。
