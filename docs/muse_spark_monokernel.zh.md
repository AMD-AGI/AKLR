# 案例：Muse Spark 整层 monokernel 在 MI355X 上提速 1.25-1.44 倍

[English](muse_spark_monokernel.md)

Muse Spark 在 MI355X 上 decode 时，每个 target 层只发一次 kernel，即 [sgl-amd-mono](https://github.com/sgl-project/sgl-amd-mono) 里的 monokernel。decode 的 device 时间约 70% 花在这次 launch 上。一轮 AI agent 优化让它**提速 1.25-1.44 倍，所有输出逐位一致**。端到端 serving 中，p90 interactivity 提高 **16-27%**。改动见 [sgl-amd-mono#135](https://github.com/sgl-project/sgl-amd-mono/pull/135)。

本文介绍这个 kernel 慢在哪里、改了什么、怎样保证正确，以及哪些经验可以用到别的 monokernel 上。

## 这个 kernel

kernel 是 `muse_spark_layer_s8/s16`，代码在 `sgl_amd_mono/muse_spark/gfx950/`。它是常驻的 FlyDSL kernel：256 个 CTA × 512 线程，每个 CU 一个，整个 launch 期间全部驻留。它跑 816B 模型 TP8 / EP8 中的一个 rank。一个 verify step 发 60 次 launch，每层一次，每次 8 行或 16 行（1 或 2 个请求 × 8 个 draft token）。

一次 launch 做完整层：

1. 在 paged fp8 KV cache 上做 attention；
2. o-norm、gate 和 o_proj，然后在 kernel 内做 TP all-reduce，走 HIP IPC 对端缓冲区和 mailbox 标志；
3. post-attention norm、gated residual 和 pre-FFN norm；
4. FP32 router 选 top-8，pooler 把宽度 8192 投影到专家的 4096（MXFP4 激活量化）；
5. 在 32 个本地 MXFP4 专家（EP8）上做 gate-up、SiLU 和 down，然后是专家 norm、加权 finalize 和跨 rank reduce；
6. unpool、FFN norm 和 gated residual；
7. 下一层的 QKV、gate、QK-norm、RoPE 和 KV cache 写入。

各阶段之间通过内存里的标志交接数据：一个 CTA 写，另一个 CTA 轮询。有些交接跨 rank。

## 慢在哪里：延迟，不是带宽

先算字节下限。每份权重按约 5.6 TB/s 读一次：

| 权重 | 字节 |
|---|---|
| o_proj | 16 MB |
| pooler | 64 MB |
| unpool | 64 MB |
| 下一层 QKV | 约 38 MB |
| 8 行时用到的 MXFP4 专家 | 约 3-6 个专家 × 25 MB |

8 行时合计约 45-60 µs，而实测 launch 是 129 µs。

逐阶段 trace 证实了这个差距。kernel 可以用 `trace=True` 构建，记录每个 CTA 每个阶段的时间戳。TP8、8 行时：

- 整个 launch 的 HBM 平均只有约 2.4 TB/s。
- 只有专家阶段接近 HBM roofline。
- 其余是一条串行链：跨 CTA、跨 rank 的写→轮询跳转，mailbox 交接，以及要等关键轮询返回后才发出的权重加载。

所以要做的是缩短这条链，而不是少搬字节。

## 改了什么

下面每一项是一种技巧，并注明用在了层的哪个部分。

**all-reduce 少走链路字节（o_proj）。**
- 一次性 all-reduce 改成按 owner 切片的 reduce-scatter 加 all-gather，每条对端链路的字节减半。
- 每个 rank 先在本地用 fp32 按 part 顺序把 partial 加好再推送。加法顺序和原来相同，结果仍逐位一致。
- 每个 rank 轮换对端顺序，避免所有 rank 同时打同一个对端。

**把工作移出关键链。**
- router 权重在轮询 x 之前就加载，而不是之后。
- route 放到负责流式读权重的 CTA 上，不占关键链上的 CTA。
- 专家 finalize、跨 rank reduce 和 unpool 的交接移出负载最重的 rank 的关键路径。
- 专家阶段末尾的 CTA 预取下一层的 QKV 权重。

**更便宜的交接。**
- 每个 CTA 轮询自己那份 x 标志副本（`X_FLAG_COPIES`），不再让很多 CTA 同时轮询一个字。
- gather 改用 sent-word mailbox（`poll_sent`、`sent_sum`、`rearm_sent`）。
- CTA 先轮询，再发自己的远端写。gfx950 上 load 和 store 共用一个按序的 `vmcnt` 计数器，所以在远端写之后发出的轮询，还要等这些写的确认。

**attention merge。**
- 一级 merge 覆盖最多 64 个分区，fold 顺序不变，结果仍逐位一致。
- 4 lane 的 fold 改用 DPP quad，不再经过 LDS。
- partial block 重新布局，让 merge wave 读连续的 256 B 和 1 KB。

**专家权重加载。**
- MXFP4 权重按整行、带 non-temporal 提示加载，在寄存器里 unswizzle。
- gate-up 任务轮转分配，改善 DRAM 局部性。
- down 任务批量处理，每个 barrier 两个 64 KB 任务。
- 消除了 K1 行求和和 unpool MFMA B 读取中的 LDS bank 冲突。

## 怎样保证正确

整轮优化遵守固定规则，由 agent 改不了的验收脚本强制执行：

- **每次调用只发一次 launch。** 不做 host 同步；在 CUDA graph 捕获和重放下安全（标志带 epoch，上一次重放留下的标志永远不会被误认）；重放时不编译。launch 参数不变，所以 serving 代码不用改。
- **数值。** 同样输入下，每个输出都要在容差内与原 kernel 一致，路由 id 必须完全相同。不允许新的近似：MXFP4 激活量化点、bf16 舍入点和 RMS eps 都不变。
- **真实条件。** 测试在单卡和 TP8（8 个 rank、真实的对端 all-reduce）上都要跑。benchmark 串联 8 个不同的层，让权重在 L2 和 MALL 中是冷的，和 serving 一样。
- **卡住算失败。** 每次等待都有截止时间，超时会写错误字，本次运行判为失败。

交付的 kernel 比容差要求更严：每种配置的 60 个输出都与 main 逐位一致。覆盖 ctx 1k、131k 和 197k，8 行和 16 行，默认配置和 `SGLANG_AMD_MONO_MUSE_TPX/PSHARD/UG_DYN=0` 两种配置。输出完全相同，所以模型精度天然不变。作为 serving 检查，换上新 kernel 后 GSM8K 得分 97.8% ± 0.4（全部 1319 题）。

## 优化是怎么跑的

- **谁来做。** 由 GEAK kernel workflow 驱动的 Claude Code agent 完成。engineer agent 按轮提出改动，只有通过完整验收的改动才会保留。
- **在哪里、多久。** 在一台 8 卡 MI355X 节点上跑了约 48 小时。单卡任务通过锁共享 GPU；TP8 测试占满 8 卡。
- **什么算收益。** 主指标是相对冻结的原 kernel 副本做的成对、交错 A/B。它的 A/A 噪声约 ±0.5%，只保留复跑后仍成立的收益。
- **重启。** 节点到期过好几次。每次新尝试都从已提交的最优版本继续，并读取之前尝试的笔记，避免重走死路。

同一轮优化也改了 draft 模型的层。这个 PR 只包含 target 层。

## 结果

### kernel

TP8，8 个冷 MXFP4 层串联。数值为每次 launch 的 µs，取 7 × 20 次 graph 重放的中位数。

| 行数 / ctx | main | 本 PR | 加速比 |
|---|---:|---:|---:|
| 8 / 1024 | 108.9 | 78.7 | 1.38x |
| 16 / 1024 | 156.4 | 108.5 | 1.44x |
| 8 / 131072 | 124.1 | 97.3 | 1.28x |
| 16 / 131072 | 183.7 | 138.6 | 1.33x |
| 8 / 196608 | 129.6 | 103.3 | 1.25x |

以上是默认配置。`TPX/PSHARD/UG_DYN=0` 时加速比为 1.35-1.50x。

### 端到端

负载是 InferenceX AgentX，TP8 / EP8，DFlash2 投机解码，模拟接受长度 4.05，跑 60 分钟。除了 kernel，每次运行都与一次 InferenceX CI 运行（CI run 42879019953）相同。p90 interactivity 是慢端，约等于 1000 / ITL p90。

| | c6 CI | c6 新 kernel | c10 CI | c10 新 kernel |
|---|---:|---:|---:|---:|
| p90 interactivity (tok/s/user) | 500.4 | 579.4 (+15.8%) | 373.1 | 472.7 (+26.7%) |
| p50 interactivity | 558.2 | 657.7 (+17.8%) | 429.6 | 564.8 (+31.5%) |
| 每 GPU 总吞吐 (tok/s) | 5,275 | 5,654 (+7.2%) | 8,774 | 9,087 (+3.6%) |

c6 用的是 `TPX/PSHARD/UG_DYN=0`，c10 用的是默认配置。

## 可以复用的经验

- **调优前先算字节下限。** launch 耗时达到下限的两倍或更多，时间就花在依赖和同步上，单靠更快的加载补不回来。
- **逐阶段 trace。** 每个 CTA、每个阶段的时间戳能看出哪些等待在关键路径上，光读代码看不出来。
- **gfx950 上先轮询再推送。** load 和 store 走同一个按序的 `vmcnt`，在 CTA 自己的远端写之后发出的轮询，也要等这些写的确认。
- **权重加载放在守护数据的轮询之前。** 权重不依赖这次轮询。
- **改归约的形状，不改求和顺序。** reduce-scatter 加 all-gather，本地 fp32 按 part 顺序求和，链路字节减半且逐位一致。逐位一致让改动容易验证和评审。
- **冻结裁判。** 固定的验收脚本，加上成对 A/B 和已知的噪声地板，让 agent 连续跑两天也不会把噪声当成收益。
