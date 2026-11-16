# kernel 和插件的合入标准

[English](acceptance_criteria.md)

你的 kernel PR 或插件 PR 要满足以下标准：

| 标准 | 适用于 | 要求 | 怎么检查 |
|---|---|---|---|
| kernel | 每个 kernel 文件 | 至少一个 workload 比 baseline 快 5% 或以上 | CI 自动检查 |
| 端到端 | 新的插件 route | 在你说明的 workload 下，模型快 2% 或以上 | 你在 PR 里报告 |

## kernel 标准：5%

- CI 会测量每个 workload，把相对 baseline 的加速比写进 kernel 文件。
- 一个 kernel 文件至少有一个 workload 结果正确、且快 5% 或以上，才能合入。否则 validation 报
  `no_qualifying_workload`，不通过。
- 运行时，AKLR 只在快 5% 或以上的 workload 上使用这个 kernel，其他 workload 运行原来的代码。
- baseline 写在算子的 `baseline.py` 里。`aklr/ops/` 下的算子，baseline 通常是 AITER 的算子；插件算子的 baseline 是
  SGLang 或 vLLM 原来的代码。

## 端到端标准：2%

插件 route 会替换 SGLang 或 vLLM 里的一个函数。CI 只检查 kernel，不跑整个模型。所以你要测量整个模型的提升，
并在 PR 里报告。

### 什么时候要报告

| PR | 端到端数据 |
|---|---|
| 新增 route（`@sglang_route`、`@vllm_route` 或 `@sglang_call_route`） | 必须 |
| 让已有 route 处理更多情况，比如新增一个分支，或支持一种新的 serving 配置 | 必须 |
| 给已有算子或 route 新增、修改 kernel 或 workload | 建议提供，非必须 |
| 在 `aklr/ops/` 新增算子 | 建议提供，非必须 |

### 报告什么

写在 PR 描述的 `End to end` 小节里：

- **配置**：模型、GPU 型号和数量、workload（输入输出长度、并发数）。
- **结果**：分别在关闭本 PR 的算子（`AKLR_DISABLE=<contract id>,...`）和打开时运行，其他条件不变。给出两种情况的吞吐或
  延迟，以及提升多少。
- **命令**：启动 server 和跑 benchmark 的命令。
