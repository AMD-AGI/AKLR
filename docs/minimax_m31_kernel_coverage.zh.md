# MiniMax M3.1 kernels：后端覆盖与验证边界

`kernel.language = "flydsl"` 说明该 schema 使用 FlyDSL builder 加载，不能单独证明每个 Point 都启动 FlyDSL kernel。下面的两个 hidden-state GEMM 实现保留混合路径；入口命中 AKLR 与命中 FlyDSL 应分别统计。

## `gemm_n6144_k30720`

两个 schema 的固定条件均为 BF16、`trans_b=True`、无 bias，计算 `out = a @ b.T`。

| Schema | 声明 workloads | FlyDSL split-K | `torch.matmul` workloads |
| --- | ---: | ---: | --- |
| [小 M](../aklr/ops/gemm/impls/flydsl/gemm_n6144_k30720/16e11e5d19578ead.json) | 47 | 42 | `m = 209, 210, 212, 216, 224` |
| [较大 M](../aklr/ops/gemm/impls/flydsl/gemm_n6144_k30720/f6791d637fe1bf47.json) | 35 | 29 | `m = 2856, 3672, 12896, 19992, 19996, 32768` |

源文件 `_SPLITK` 表决定后端；未列在表中的 M 使用 Torch。这里只统计 schema 声明的 workloads，保留其覆盖范围，不据此推断两种后端的速度。测试通过提取真实入口、保留真实配置表并替换编译/启动调用来核对全部声明 workloads 的后端选择。

较大 M 的 split-K partial 已从 FP24 压缩改为完整 FP32，producer 和 fold 使用一致的四字节元素布局。不再对 partial 先舍入，可以避免有限的 `+FLT_MAX` / `-FLT_MAX` 变成 `+Inf` / `-Inf` 后产生 NaN，也保留两片相消后剩余的低位；单纯饱和到 FP24 最大值无法保留后者。原有 FP32 累加次序保持不变。

这一改动使 split-K workspace 从 `3 × split × m × n` 字节变为 `4 × split × m × n` 字节，即原来的 **4/3**。它也增加了 partial 的存取流量，必须在目标 gfx950 环境重新测量准确率、延迟及显存占用，不能沿用旧实现的性能结论。该修复消除的是 partial 压缩引入的错误；FP32 乘加本身的溢出及求和顺序仍按原数学契约处理。

## 小 MoE

[少于 256 tokens 的 MoE schema](../aklr/ops/fused_moe/impls/flydsl/fused_moe_swiglu_has_swiglu_limit_shuffled_16x16_model_dim6144_n_experts128_topk4_inter_dim768/40b0de61e4f6959c.json) 的 39 个 workloads 最大为 248。原来 `M >= 256` 才可达的 Torch 权重展开、MXFP4 激活量化及计算路径已删除。直接以 `M >= 256` 调用此入口会明确报错，避免以小 M 的算法误算不同的激活量化语义。

从 256 tokens 起，继续由[独立的大 MoE schema](../aklr/ops/fused_moe/impls/flydsl/fused_moe_swiglu_has_swiglu_limit_shuffled_16x16_model_dim6144_n_experts128_topk4_inter_dim768/a9f3a19f544bb1c5.json) 负责。

## 验证与测量

修改的 schema 已按 authored content 重新生成文件名。三个原文件均没有 `speedup` 或 `accuracy` 测量记录，修改后仍不填写未经验证的数据。默认 `AKLR_MIN_SPEEDUP=1.03` 不会正常调度这些未测量 Point；测试/验证工具直接执行 kernel 时不经过这个门槛。

CPU 回归执行 JSON 中真实的 producer 写入和 fold 操作，覆盖 `FLT_MAX` 相消为零以及相消后保留一个 FP32 ulp 的情况。旧代码在该测试中得到 NaN。它检查的是数值位操作及缓冲区布局，不能替代 GPU 的编译、访存、MFMA、并发或 graph capture/replay 验证。

```bash
python -m pytest tests/ops/test_minimax_gemm.py tests/ops/test_minimax_moe.py -q
python -m aklr export check
```

其中 `test_gemm_finite_extreme_split_cancellation_on_gpu` 在 gfx950 上直接构建并执行真实 `m=268` kernel，以有限 BF16 输入产生上述相消案例。CPU 环境会跳过此测试。完成目标 GPU 验证及每个 workload 的测量后，才应写入准确率/速度结果并评估发布条件。
