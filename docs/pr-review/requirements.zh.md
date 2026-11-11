# PR 评审分级与正确性分层：需求文档

状态：草案，2026-11-11。范围：AMD-AGI/AKLR 的 `main`。配套设计：[design.zh.md](design.zh.md)。来源编号 [S1]… 见文末。

## 速查：各种 PR 怎样才能合并

### 所有 PR 的共同条件（不论档位）

| 条件 | 说明 |
|---|---|
| 必需检查全绿 | `code-lint`、`coverage-report`、`perf-guard`、`review-gate` |
| AI 评审已在 head 上跑完 | 每条发现要么修，要么写理由关闭 |
| 安全评审已在 head 上跑完 | 无未关闭的发现 |
| 所有讨论已关闭 | 包括 Copilot 的讨论 |
| 批准绑定当前 head | 批准后再推提交，批准失效，需要重新批准 |
| 内核类 PR 有 GPU 证据 | head 上跑过 `ci:gpu` 验证且通过 |
| 合并后 | main 上再跑 GPU 验证；测量通过前内核不被服务；出问题一步回滚（R-B5） |

### 按改动类型的人工批准数

可信作者 = 仓库 admin。名单外作者每档都需要 1 名可信成员批准。区域 owner = CODEOWNERS 里该目录的负责人。完整路径规则见 R-A1。

| 改动 | 典型路径 | 档位 | Phase 2（起步） | Phase 3（目标） | 谁能批准 |
|---|---|---|---|---|---|
| 只改文档 | `docs/**`、普通 `*.md` | Low | 1 | **0**（自动检查全绿即可） | 任一可信成员 |
| 知识库 | `harness/skills/aklr-kb/**` | Low | 1 | 0 | 同上 |
| 只新增测试 | `tests/**`，无删除行 | Low | 1 | 0 | 同上 |
| 撤回内核 | 删除 schema | Low | 1 | 0 | 同上 |
| 实测字段 | 机器人只改 `speedup`/`accuracy` | 机器人通道 | 0 | 0 | 无需人；人改会被 perf-guard 拒绝 |
| 新增或修改内核 | 带 `kernel.sources` 的 schema | Medium | 1 | **作者签字即可** | 可信成员；作者签字占 1 席 |
| 新增契约 | `kernel.sources` 为空的 schema | Medium | 1 | 签字 | 同上 |
| 算子代码 | `aklr/ops/<op>/` 的 api、contract 等 | Medium | 1 | 签字 | 同上 |
| 插件路由 | `aklr/plugins/**` | Medium | 1 | 签字 | 同上 |
| agent 流程和技能 | 其他 `harness/skills/**`、`harness/agents/**`、`harness/scripts/**` 等 | Medium | 1 | 签字 | 同上 |
| 测量方法 | `aklr/benchmarks/**`、`scripts/**` | Medium | 1 | 签字 | 同上 |
| 其余 core | `aklr/core/{jit,platform,runtime/builder,utils}` | Medium | 1 | 签字 | 同上 |
| 测试有删改 | `tests/**` 下删了行 | Medium | 1 | 签字 | 同上 |
| 已有算子的裁判 | 改已有的 `baseline.py`、`reference.py` | **High** | 1 | 1 | **区域 owner**；作者签字不算 |
| 核心裁判与分发 | `aklr/core/validation`、`runtime/dispatch`、`contract` | High | 1 | 1 | 区域 owner |
| 自动执行的代码 | `harness/hooks/**`、agent 设置文件 | High | 1 | 1 | 区域 owner |
| CI | `.github/workflows`、`scripts`、`actions` | High | 1 | 1 | 区域 owner |
| 评审系统本身 | 策略文件、门禁、`REVIEW.md`、PR 模板、CODEOWNERS、评审技能 | High | 1 | 1 | 区域 owner |
| 依赖和供应链 | `pyproject.toml`、`requirements*`、`3rdparty/**` 等 | High | 1 | 1 | 区域 owner |

### 特殊情况

| 情况 | 结果 |
|---|---|
| 一个 PR 混合多种改动（如文档 + CI） | 取最高档，CI 路径的 code owner 规则照常生效 |
| 分类器 2 命中，原本 0 人（删断言、超过 1000 行、作者首次改这个目录等） | 升为 1 人 |
| 分类器 2 命中，原本已是 1 人 | 不加第二人，改为要求区域 owner 并做 Deep 审读 |
| 熔断触发 | 相关类型升一档，或退回 Phase 2 的人数（R-B10） |
| Claude 判定失败、超时或格式不对 | 按"不确定"处理：Low 升 Medium，或加 1 人 |
| 名单外作者 | 每档需要 1 名可信成员批准；High 要区域 owner |
| agent 写的 PR，现阶段用人的账号推送 | 同人写的；签字只用于问责（Phase 3 起签字才占席位） |
| agent 写的 PR，切换到 App 身份后 | 指挥者用普通 GitHub 批准占 1 席 |
| 批准后又推了新提交 | 批准失效，重新批准 |
| 有未关闭的发现或讨论、CI 红 | 不能合并 |
| 紧急情况 | admin 绕过强制合并；之后打 `bypassed` 标签，另一名可信成员 24 小时内补审，并计入护栏 |
| Phase 1（影子期，4 周） | 以上规则只记录、不拦截；合并方式同今天 |

## 0. 背景与目标

依据 Anthropic 的两篇文章 [S1][S2]：

- 多数"读代码"的工作交给机器。人工批准的数量随风险变化，由变更本身决定，不由作者决定。
- 正确性来自多层检查加快速回滚，不靠一道人工关卡。
- 要优化的是人类注意力：每个 PR 的批准数、评论数、等人的时间。护栏是回滚率和事故率；护栏变差，说明人力只是被挪到了别处 [S1]。
- 人审的是变更的意图，不是每一行 [S2]。

AKLR 的目标沿用 [S1] 的表述：多数 PR 至多一名人类参与，这个人可以是作者，即指挥 agent 的人；回滚与事故不增加。

## 1. 术语与 AKLR 对应

| 概念 | 在 AKLR 中的含义 |
|---|---|
| 档位 Low / Medium / High | 由 diff 决定的人工批准数，目标为 0 / 1 / 1（R-A1）。High 和 Medium 的区别不在人数，而在审批人必须来自区域 owner、评审深度为 Deep、作者签字不能占席位。 |
| 人工席位 | 一次来自人类账号、绑定到当前 head 提交的批准或签字。 |
| 指挥者 | 让 agent 写出这次变更并为它负责的人。AKLR 的 agent 现在用这个人自己的 GitHub 身份推送和开 PR，所以指挥者就是 PR 作者。 |
| 可信作者 | 仓库的全部 admin（已定）。门禁按仓库 admin 角色实时判断。注意：admin 角色大多由组织继承，人数远多于 AKLR 的 owner。 |
| finding | AI 评审或安全评审针对某一行的意见，以 review thread 的形式存在。 |
| 开关 | ① 内核在 main 上测量之前不提供服务（`FallbackReason.NOT_MEASURED`）；② `AKLR_DISABLE=<contract id>` 在运行时关掉一个算子；③ 只有设了 `AKLR_SGLANG_PLUGIN=1` 才挂 SGLang；④ 只服务实测加速比达到 `AKLR_MIN_SPEEDUP` 的工作负载。 |
| 分阶段发布 | 合并 → main 推送验证 → 测量机器人 PR 写入实测字段后才服务 → AgentX 周跑与 e2e 精度检查 →（将来）打 tag，并做下游资格验证。 |
| 回滚 | GitHub 的 Revert 按钮；撤回 schema（删除后不再服务）；运行时设 `AKLR_DISABLE`；测量层的自动降级：main 验证失败后，机器人写入失败结果，该内核停止服务。 |
| 事故 | 合并后才发现、并由出问题路径的 CODEOWNERS owner 确认的问题（中等范围，已定）：已服务的内核输出错、崩溃或挂起；已服务的内核慢于基线，或被批量降级；SGLang、vLLM 等下游报告的问题；CI、runner 或密钥的安全事件。main 因非平台原因变红、`fix(...)` 跟进修复只作信号，不自动算事故。见 R-B8。 |

## 2. 现状（2026-11-11 调查）

数据来自仓库文件和 GitHub API（rulesets、PR、review、review thread）。统计窗口：2026-10-31（首次合并）到 2026-11-11。共 118 个合并 PR，其中 101 个由人提交，17 个是测量机器人的 PR；main 上共 125 个提交。

### 2.1 变更种类

已提交的 schema JSON 共 538 个，其中 482 个带 `kernel.sources`（内核），56 个为空（契约）。内核源码以 `{path, content}` 的形式嵌在 JSON 里，按扩展名判断会把 GPU 代码当成数据。

main 上 125 个提交按改动路径分类（一个提交可能属于多类）：

| 种类 | 判定 | 提交数 |
|---|---|---|
| 文档与 agent 技能 | `docs/`、`harness/`、`.claude/`、`*.md` | 56 |
| schema（内核或契约） | schema 路径正则 | 44 |
| CI | `.github/**` | 33 |
| 测试 | `tests/**` | 28 |
| 插件代码 | `aklr/plugins/**/*.py` | 18 |
| 测量机器人 | `aklr-perf[bot]` 的提交 | 17 |
| 算子代码 | `aklr/ops/*/*.py` | 12 |
| `aklr/core` 代码 | `ValidationScope.path_kind == "core"` | 12 |
| 撤回内核 | 被删除的 schema | 9 |

### 2.2 合并规则的实际状态（经 API 读取）

`main` 没有经典的分支保护，生效的是 3 个 ruleset：

| ruleset | 内容 | 可绕过者 |
|---|---|---|
| `main` | 禁止删除和强推。合并必须走 PR，需要 1 个批准。以下选项都没开：`dismiss_stale_reviews_on_push`、`require_last_push_approval`、`required_review_thread_resolution`、`require_code_owner_review`。没有按路径的 required reviewers。`require_extra_approval_for_unattributed_changes=true`：对应文档中的 "Require an additional approval for unattributed Copilot pull requests"，GitHub 默认开启，只对 Copilot 开的 PR 生效 [S3]。新 PR 自动请求 Copilot 评审。 | 仓库 admin 角色：always；aklr-perf App：pull_request |
| `main: required checks` | 必需检查只有 `code-lint`，非 strict。 | 无 |
| `aklr-perf/main: bot only` | 只有 aklr-perf App 能创建、更新、删除这个分支。 | 该 App |

另外：

- CODEOWNERS 只有一行 `*`，列 6 人。
- 环境：`gpu` 需要人工批准，供 fork、无写权限作者或 Bot 触发的 GPU 运行使用；另有 `gpu-trusted` 和 `bot`（分支策略为 main）。
- Actions 设置：`GITHUB_TOKEN` 的默认权限是 write，现有工作流都显式收窄为 read；Actions 不能批准 PR（`can_approve_pull_request_reviews=false`）。

### 2.3 review-tier：只剩引用，从未实现

- 以下位置仍引用一个"review-tier"检查或门禁：
  - `.github/pull_request_template.md:2`
  - `.github/workflows/ci.yaml:400`（"review-tier reads the stages from this name"）
  - `.github/workflows/ci.yaml:534`（"the review-tier gate decides whether it needs one"）
  - `harness/skills/aklr-open-pr/SKILL.md:3,14-15,20,51,61`：要求 agent 读取 `.github/scripts/review_tier/policy.py` 和 `change.py`，并查看 `review-tier` 状态。
- 本仓库任何分支的历史里都不存在 `.github/scripts/review_tier/`（`git log --all -- .github/scripts/review_tier` 为空）。所有引用都来自初始提交 9a2961617（2026-10-31），应是从前身仓库带过来的。ruleset 里也没有这个检查。分支 `feat/e2e-cookbook-accuracy` 上的提交 36366a121 删掉了 ci.yaml:400 那一句，但还没进 main。
- 同类的陈旧描述：
  - `ci.yaml:55-56` 说审计门禁会拒绝"mixed changes"，但 `aklr/core/validation` 里没有这段逻辑。
  - 模板说"内核和契约单独成 PR 时凭证据合并，与代码混在一起就要等代码"，但没有任何机制执行这条规则。
- 后果：
  - `aklr-open-pr` 让 agent 去读一个不存在的文件，去等一个永远不会出现的状态。
  - "是否需要 GPU 验证"被交给了不存在的门禁（`ci.yaml:530,534`）。内核 PR 不打 `ci:gpu` 就不跑 GPU 验证；打了标签而验证失败时，`coverage-report` 会变红，但它不是必需检查，挡不住合并。

### 2.4 实际评审数据（101 个由人提交的合并 PR）

| 指标 | 值 |
|---|---|
| 每个 PR 的人工批准数 | 中位数 1，p95 为 1；分布：0 个批准 29 个 PR，1 个 68 个，2 个 4 个 |
| 零批准合并（admin 绕过） | 29 个（29%），其中包括：CI 工作流 PR #126、#127、#146（gpu-node.yaml 运行在持有 Spur 账号和 runner 管理令牌的 runner 上）；`aklr/core` 改动 #130；Claude Code 钩子代码 #144；11 个带 `ci:gpu` 标签的内核 PR |
| 批准集中度 | 按 PR 去重后共 90 次批准，其中 61 次来自同一名评审者 |
| 等人时间（PR 打开或转为 ready 到第一个批准） | 中位数 1.5 h，p90 18.2 h（n=72，包含等 CI 的时间） |
| 每个 PR 的人类评论（含行内评论） | 中位数 0，均值 0.21，p95 为 1 |
| AI 评审 | Copilot 共开了 362 个 review thread，至今 346 个（96%）未解决；没有人类开的 thread |
| 批准与提交的绑定 | 72 个有批准的 PR 中，71 个的最后一个批准就在合并时的 head 上。习惯上是绑定的，但规则不强制。 |
| 回滚 | main 上没有 revert；有 `fix(...)` 形式的跟进修复（如 #130、#132、#137），是否算事故需要人确认 |

### 2.5 已有的正确性层

| 层 | 现有机制 | 是否阻断合并 |
|---|---|---|
| 提 PR 前验证 | `aklr-open-pr`（证据取自 head 上的运行）、`aklr-review-kernel-pr`、`aklr-bad-cases` | 否，只是流程约定 |
| AI 评审 | Copilot（由 ruleset 开启）；Claude 评审的 PR #149 尚未合并，它只在 PR opened/ready 时评一次，只看 `aklr/**`，只发 COMMENT | 否 |
| 安全 | 没有自动安全评审。专项防护有：`gpu-node.yaml` 的信任判定；`perf-guard`（用 base 上的代码比对实测字段）；审计阶段的 `SourceRule` | `perf-guard` 不是必需检查 |
| CI | `code-lint`（pre-commit、export check、审计）；GPU 单测和验证（需要 `ci:gpu` 标签）；`coverage-report` 汇总结果（ci.yaml:441 自称 "the required check"） | 只有 `code-lint` 是必需的 |
| 合并队列 | 无。允许 merge、squash、rebase，非 strict。 | 不采用（R-B5） |
| 开关与分阶段 | 测量之前不服务；`AKLR_DISABLE`；机器人 PR 写入实测字段；AgentX 周跑 | 合并后才起作用 |
| 发现与回滚 | 没有自动发现和告警，靠人看 main 的运行和机器人 PR | — |
| 事故变评测 / 指标 / 熔断 | 无 | — |

### 2.6 问题清单

- F1：review-tier 的引用全部失效（§2.3）。
- F2：admin 绕过让"需要 1 个批准"形同建议。29% 的人类 PR 零批准合并，其中包括 CI 工作流和 `aklr/core` 的改动。
- F3：必需检查只有 `code-lint`。下面两项都挡不住合并：
  - `perf-guard`，它是防止人改实测字段的可信副本；
  - `coverage-report`，它汇总 GPU 验证的结果。
- F4：没有任何机制判断一个 PR 是否需要 GPU 验证，这是 F1 的直接后果。
- F5：批准不绑定提交：新推送不撤销旧批准，也不要求由最后推送者以外的人批准。
- F6：AI 评审的意见没人回应（96% 未解决），不阻断合并，也不分轻重。这样的噪声在教人忽略评审。
- F7：CODEOWNERS 只有一行 `*`，表达不了"这些路径总要有人看"或"这些路径要区域 owner的人看"。
- F8：没有事故的定义，没有回滚统计，也没有人力指标，所以没有证据可以支撑"放宽"。
- F9：agent 用人的身份推送，系统分不清是 agent 推送还是人推送。本仓库的提交和 PR 也不带 agent 署名，无法据此识别 agent 写的 PR。
- F10：`GITHUB_TOKEN` 的默认权限是 write，新工作流只要忘写 `permissions:` 就会拿到写权限。

## 3. 需求 A：审批与分级

每条需求先写要求，再写在 AKLR 中如何落地，最后给出验收标准。

### R-A1 档位由变更决定，不由作者决定

**要求**

- 分类器 1 读 diff，给出 Low、Medium 或 High。
- 分类器 2 判断这次变更是否还需要一个人。即使档位说不需要，它也可以加人。人工批准数上限为 1：分类器 2 或熔断命中时，0 人升为 1 人；已经是 1 人时不再加第二人，改为要求这 1 人来自该区域 owner（CODEOWNERS），并做 Deep 级审读。
- 目标人数：Low 在自动评审和安全评审全绿后为 0 人；Medium 为 1 人；High 也为 1 人（与 [S1] 的 2 人不同，这是 AKLR 的决定）。起步更严，见 R-A3。

**AKLR 映射**

下表是建议的档位下限。PR 的档位取各文件下限的最大值。未匹配的路径默认 Medium，因为 Low 是白名单。

| 种类 | 路径或判定 | 下限 |
|---|---|---|
| 文档 | `docs/**`，以及下列例外以外的 `*.md` | Low |
| 知识库 | `harness/skills/aklr-kb/**` | Low |
| 撤回内核 | 被删除的 schema JSON | Low |
| 只增加测试 | `tests/**`、`harness/tests/**`，且没有删除任何行 | Low |
| 实测字段 | 只改 `speedup` 和 `accuracy` 的 schema | 只走机器人通道，0 人；人改这些字段会被 perf-guard 拒绝 |
| 内核 | 新增或修改的、带 `kernel.sources` 的 schema | Medium，且要求 head 上的 GPU 验证通过 |
| 契约 | `kernel.sources` 为空的 schema | Medium |
| 算子代码 | `aklr/ops/<op>/` 下的 `__init__.py`、`api.py`、`contract.py`；新算子的 `baseline.py` 和 `reference.py` | Medium |
| 插件路由 | `aklr/plugins/**` 下的代码 | Medium |
| agent 流程 | 其他 `harness/skills/**`；`harness/agents/**`、`harness/scripts/**`；`.claude/`、`.agents/`、`.codex/`、`.claude-plugin/` 下的非设置文件 | Medium |
| 测量方法 | `aklr/benchmarks/**`、`benchmarks/**`、`scripts/**` | Medium |
| 其余 `aklr/core` | `aklr/core/{jit,platform,runtime/builder,utils}/**`（改动会让验证重测全部 schema） | Medium |
| 测试有删改 | `tests/**` 下有被删除的行 | Medium |
| 人工标注的评测数据 | `.github/review_eval/**` | Medium |
| 已有算子的裁判 | 修改已有的 `baseline.py`（速度基准）或 `reference.py`（正确性参考） | High |
| 核心裁判与分发 | `aklr/core/validation/**`、`aklr/core/runtime/dispatch/**`、`aklr/core/contract/**` | High |
| 自动执行的 agent 代码 | `harness/hooks/**`；agent 设置文件（`.claude/settings*.json`、`.codex/config.toml`） | High |
| CI | `.github/workflows/**`、`.github/scripts/**`、`.github/actions/**` | High |
| 评审系统本身 | 策略文件；门禁代码；`REVIEW.md`；技能 `aklr-review-kernel-pr`、`aklr-bad-cases`、`aklr-open-pr`；PR 模板；CODEOWNERS | High |
| 供应链 | `pyproject.toml`、`requirements*.txt`、`.pre-commit-config.yaml`、`setup.py`、`MANIFEST.in`、`.gitmodules`、`3rdparty/**` | High |

[S1] 把"放在开关后面的新增变更"归为 Low。AKLR 中最接近它的是新增内核：内核合并后，要等 main 上的验证测量通过才会被服务。但新增内核仍有两个风险：内核可能针对基准作弊；内核的主机侧代码会在 GPU 节点上执行。所以新增内核起步定为 Medium，以后凭证据再降为 Low（设计 §10 的 Phase 4）。

分类器 2 的 AKLR 信号，命中任一条就加人（受 1 人上限约束，见 R-A1）：

- 契约放宽了 compare 门槛；
- 测试里的断言被删；
- PR 正文的 "Gate overrides" 不是 none；
- schema 版本或测量版本变化；
- 新挂了一个引擎函数；
- 改动了 `AKLR_MIN_SPEEDUP` 或分发优先级；
- 作者第一次改动这个目录；
- 非 JSON 的改动超过 1000 行（[S1]：超过千行的 PR 多数有评审发现）；
- 语义评审员给出的"需要加人"的概率超过阈值。

**验收**

1. 档位是三个输入的纯函数：main 上的策略、head 上的变更、评审员的输出。同样的输入重跑，得到同样的结果。每个决定都列出命中的规则 id 和证据（文件:行）。
2. 在 AKLR 历史 PR 的标注集上（至少 150 个 PR，见 R-B8）：
   - 标为 High 的 PR 被判低的个数为 0；
   - 整体判低率不超过 2%；
   - 判高率（即多花的人力）单独报告。
3. 对抗用例（设计 §8.2）全部判到不低于真实档位。
4. 只改 `docs/**` 的 PR 判 Low；改了 `.github/workflows/**` 的判 High；混合的 PR 取最高档。

### R-A2 某些路径总要人看

**要求**：策略中列为敏感的路径，以及所有 CI 工作流文件，无论档位、无论分类器结论，都至少需要 1 名非作者的人批准。这条由 CODEOWNERS 加 ruleset 的原生规则（按文件模式的 required reviewers）强制执行，不只依赖门禁代码。

**AKLR 映射**：适用于上表所有 High 行，以及"其余 `aklr/core`"。ruleset 的 required reviewers 只接受团队 [S3]；AKLR 改用 CODEOWNERS 按目录写个人账号，再开"需要 code owner 批准"，不需要组织团队。

**验收**：

- 在测试 PR 上改 `.github/workflows/*.yaml`，没有非作者批准时，除 admin 的 break-glass 绕过外，任何人都不能合并。admin 绕过保留，用于紧急强制合并，但必须留痕（N6），并在合并后由另一名可信成员补审。
- 停用门禁检查后，这条规则仍然生效，证明它是原生规则。

### R-A3 只有可信作者走轻路径

**要求**：策略文件列出可信作者。不在名单上的作者，本人的签字不占席位，每一档的席位都要由可信的独立审批人填满；因此对他们来说 Low 也需要 1 人。

**AKLR 映射**：下表是各档所需的"独立、可信、非作者、绑定当前 head"的批准数。签字见 R-A5。

| 档位 | 可信作者，Phase 2（起步） | 可信作者，Phase 3（目标） | 名单外作者 |
|---|---|---|---|
| Low | 1 | 0 | 1 |
| Medium | 1 | 0（作者或指挥者的签字占 1 席） | 1 |
| High | 1（区域 owner） | 1（区域 owner；作者或指挥者的签字不占席位） | 1（区域 owner） |

Phase 2 等同于今天名义上的"需要 1 个非作者批准"，再加上 High 的那 1 人必须来自区域 owner。任何档位都不需要第二个人。

**验收**：

- 名单外作者的 Low PR 需要 1 名可信的人批准。
- 名单只能通过 High 档的 PR 修改。
- 按账号判断是否可信，不用 `author_association`：私有组织成员会被它报成非成员（见 `gpu-node.yaml` 的 trust 作业注释）。

### R-A4 机器的批准不算人的批准

**要求**：bot、App 或 agent 的批准永远不占人工席位。自动评审可以放行 Low（它是 Low 的前提条件之一），但不能填任何人工席位。

**AKLR 映射**：适用于 `aklr-claude[bot]`、`aklr-perf[bot]`、Copilot 评审者、GitHub Actions，以及之后要切换到的 agent App 身份（设计 §12 Q4）。评审工作流提示词里的 "Never approve" 只是一句话，不是控制手段。

**验收**：

- 构造一个由 bot 批准的测试 PR，门禁对它的计数为 0。
- 仓库设置 "Allow GitHub Actions to create and approve pull requests" 保持关闭。2026-11-11 读取时已经是关闭的。

### R-A5 agent 写的 PR：指挥者算一名评审

**要求**：指挥 agent 的人算作所需评审中的一人，就像代码是他自己敲的一样。前提是他在当前 head 上签了字，确认读过证据 [S1]。

**AKLR 映射**：

- 指挥者就是 PR 作者（F9）。签字是 PR 模板里的一行，写明 head SHA，即绑定这个 SHA。
- GitHub 不允许作者批准自己的 PR，所以"作者签字占 1 席"只能由门禁计数，原生规则数不出来。
- 已定的路线（设计 §12 Q4）：Phase 1–2 agent 继续用人的账号推送，签字只用于问责；之后改为以 `aklr-claude` App 推送，指挥者用普通的 GitHub 批准占席位，原生规则可以直接表达。

**验收**：

- Phase 3 中，可信作者的 Medium PR 只要有作者在 head 上的签字，就能通过门禁。
- head 变化后，旧签字失效。
- 名单外作者的签字不占席位。

### R-A6 批准绑定到一个提交

**要求**：批准只对被批准的那个提交有效。系统无法担保的新提交会让它失效，包括 agent 的推送和手工解决冲突后的合并。

**AKLR 映射**：

- 原生开启 `dismiss_stale_reviews_on_push`：任何改变 diff 的推送都会撤销批准，比 [S1] 更严。GitHub 文档说明，手工推送的 merge commit 只有在内容与 GitHub 生成的合并完全一致时才不使批准失效 [S3]。
- 原生开启 `require_last_push_approval`。
- 门禁再核对每个计入的批准都满足 `commit_id == head`，签字同理。

**验收**：

- 批准之后再推一个提交，批准被撤销，门禁失败。
- 今天 72 个中 71 个的习惯（§2.4）变成规则。

### R-A7 有未关闭的 finding 不能合并；CI 红不能合并

**要求**：AI 评审或安全评审的任何一个 finding 没关闭，PR 就不能合并。关闭的方式有两种：

- 修复：评审员在后续轮次确认已修复后，自动关闭对应的 thread；
- 回复理由后关闭 thread。

必需的 CI 检查失败时也不能合并。

**AKLR 映射**：

- 原生开启 `required_review_thread_resolution`。
- 评审员把 finding 作为行内 thread 发出。
- 门禁要求 AI 评审和安全评审都已在当前 head 上跑完。
- 开启后，Copilot 的 thread 也会挡住合并：按今天的数据，平均每个 PR 约 3.4 个。所以要先决定 Copilot 的去留（设计 §12 Q8）。

**验收**：

- 有未解决的 thread 时，合并按钮不可用。
- 修复后，下一轮评审自动关闭对应的 thread。
- 以理由关闭的 thread 在指标里记为"误报"或"有意为之"。

### R-A8 每一档都要过同一套自动层

**要求**：所有档位都要经过必需 CI、AI 代码评审和安全评审；合并后都要过 main 上的 GPU 验证（AKLR 不用合并队列，见 R-B5）。档位只改变人工批准的数量。

**AKLR 映射**：

- 必需检查扩为 `code-lint`、`coverage-report`、`perf-guard`、`review-gate`。AI 评审和安全评审的完成情况由 `review-gate` 核对，见设计 §3.4。
- 内核、算子代码、插件、`aklr/core` 这几类要求 head 上的 `validation` 作业成功；是否满足由门禁按种类判断。
- 机器人 PR 今天就只绕过评审、不绕过检查（ci.yaml:696）。`perf-guard` 改为必需后，也会覆盖机器人 PR。

**验收**：

- 一个 Low 的文档 PR 仍要等 `review-gate` 确认 AI 评审和安全评审已完成。
- 内核 PR 不打 `ci:gpu` 时，`review-gate` 失败，并写明缺少 GPU 验证。

## 4. 需求 B：正确性分层

### R-B1 开 PR 之前验证

**要求**：agent 必须拿到可运行的东西（测试、构建、验证），贴出真实输出，而不是声称已经完成。之后由一个全新上下文的子 agent 评审 diff。反复出现的问题要在源头修掉：写进技能、lint 规则或指令文件 [S1]。

**AKLR 映射**：

- `aklr-open-pr` 已经要求证据来自 head 上的运行。
- 补一步"新上下文评审"（`aklr-review-kernel-pr`，或本地 `/code-review`、`/security-review`），把结果（发现数，已修或已答复）写进 Verification。
- 反复出现的意见沉淀到 `aklr-bad-cases`、`REVIEW.md` 或 pre-commit 规则里。

**验收**：

- `aklr-open-pr` 包含这一步。
- 门禁确定性地检查 PR 正文里每种变更所需的段落都不为空。例如内核 PR 必须有 Kernels 表，以及验证日志或 CI 链接。
- AI 评审会标出"声称已运行但没有输出"的段落。

### R-B2 每个 PR 都有 AI 评审

**要求**（均依据 [S1]）：

- 多个找 bug 的 agent 并行工作，再由投票过滤，票数足够的 finding 才发出。
- finding 标注 Important、Nit 或 Pre-existing。
- 丢弃纯风格、诊断、清理类意见。
- 每条 finding 都必须得到回应。
- 评审深度分 Quick、Standard、Deep 三级。
- 最多评 3 轮，之后只看新增的 diff，并请作者重新考虑方案。
- 不重复已经发过的 finding。
- 发现一类 bug 时，要求把同类的所有实例一起修掉。

**AKLR 映射**：在 PR #149 的 `claude-review.yml` 基础上扩展。

- 每次推送都评；今天只在 opened 或 ready 时评一次。
- 覆盖全部路径，不只 `aklr/**`：文档"说谎"和 CI 安全也要看。
- 深度按档位定：Low 用 Quick，Medium 用 Standard，High 用 Deep。AKLR 只有一个仓库，所以按档位分深度，而不是像 [S1] 那样按仓库分。
- `REVIEW.md` 写明 AKLR 里"失败"长什么样：
  - 内核：某个 Point 的输出错误、藏着慢速回退、为基准作弊；
  - 契约：为迁就内核而修改；
  - 文档和技能：与实测不符；
  - CI：在有密钥的作业里检出或执行 PR 代码。
- 风格问题交给 ruff 和 pre-commit。

**验收**：

- 每条 finding 都带标签，以及文件:行形式的证据。
- 按指纹计，同一个 PR 上重复的 finding 数为 0。
- 记录每个 PR 通过评审所需的轮数。
- 每天给出两个数（R-B8）：在已知干净的 PR 集上的误报率；在事故集上的召回率。

### R-B3 能阻断合并的安全评审

**要求**：有一个独立的自动安全评审。它的 finding 与 bug 评审一样阻断合并（R-A7）；后续提交修好之后自动关闭 [S1]。

**AKLR 映射**，重点检查以下几面：

- 工作流：`pull_request_target` 的误用；在有密钥的作业里检出或执行 PR 代码；self-hosted 标签 `aklr-claude`、`aklr-spur-host` 的使用；权限扩大。
- `.github/scripts/*.sh`。
- 内核源码里的主机侧代码，已有审计阶段的 `SourceRule`。
- `harness/scripts`。
- 依赖与版本固定，例如以 git 方式安装的 aiter。
- 反序列化，例如 `torch.load`。
- 外部下载。

**验收**：

- 在种子用例上（例如把工作流改成以 `pull_request_target` 检出 head 并运行），安全评审报出 Important 并挡住合并。
- 误报率可以统计。

### R-B4 阻断式 CI 与 PR 模板

**要求**：CI 必须通过。PR 模板问三件事：怎么测的、有什么风险、怎么撤销。Claude 可以代填，但要由发起变更的人签字 [S1]。

**AKLR 映射**：

- 模板已有对应的段落：Verification 和 Not run 回答"怎么测的"；Risk 下的 Affects、If wrong、Undo 回答"风险"和"怎么撤销"。保留这些。
- 删掉 review-tier 那一段，加一行签字。
- 必需检查见 R-A8。

**验收**：

- 模板里不再出现 review-tier。
- 门禁能解析签字，并把它绑定到 head。

### R-B5 合并后在最新 main 上重测（不采用合并队列）

**[S1] 的做法**：合并队列在"最新的 main 加上这个变更"上重跑检查，防止两个各自通过的 PR 合在一起后把 main 弄坏。

**AKLR 不采用合并队列（已定）**：

- 队列里只跑 CPU 检查对 AKLR 没有意义：AKLR 的主要风险是内核正确性和性能，只有 GPU 验证能发现。
- 队列里跑 GPU 验证代价过高：全量验证约需 225 分钟、8 张卡（ci.yaml:390 的注释，按 247 个 schema 估算；现在已有 538 个），每个入队的 PR 都要占一台 MI355X。
- AKLR 的 PR 量小（总共约 150 个，每天约 10 次合并），两个 PR 组合出错的机会很少。

**替代要求**：组合出错的风险由三样现有的东西兜底：

- main 推送后的 GPU 验证，就是"在最新 main 上重测"；
- "测量之前不服务"：内核在 main 上测量通过之前不会被服务，验证失败会写入降级；
- main 验证失败时一步回滚引入问题的 PR（R-B7）。

**验收**：

- 两个各自为绿、组合后出错的 PR 合并后，main 推送验证失败，相关内核不被服务，并在一个工作日内回滚或修复。

### R-B6 开关与分阶段发布

**要求**：新行为放在开关后面；先在内部使用，再公开；`latest` 通道先于 `stable` [S1]。

**AKLR 映射**：见 §1 的术语表。目前有两个缺口：

- 插件路由（挂在引擎函数上）和 `aklr/core` 的改动不受"测量之前不服务"的保护；
- 没有 release tag，也就没有 stable 通道。

需求如下：

1. 每个插件路由都能用运行时开关单独关掉，退回引擎原来的路径。`AKLR_DISABLE=<contract id>` 是否已经足够，待确认。
2. 从第一个 tag 起，stable 指经过 `aklr-qualify-downstream` 验证的 tag；下游固定到 tag，不固定到 main。

**验收**：

- 每个插件路由都有一种经过测试的关闭方式。
- 文档写明 latest 和 stable 的含义。

### R-B7 快速发现与一步回滚

**要求**：开关能在不重新部署的情况下关掉；坏掉的构建被回滚；事故复盘要回答"哪一层本该抓到它" [S1]。

**AKLR 映射**：

发现问题的信号：

- main 推送后的运行失败，且原因不是平台故障（`FindingCode.is_platform_fault` 已能区分）；
- 机器人 PR 显示，近期改过的 schema 被降级（accuracy 为 failed，或 speedup 跌破门槛），或出现大批降级；
- main 上的单测变红；
- AgentX 周报出现回退；
- 下游的 issue；
- e2e 精度检查失败。

一步回滚的方式：

- GitHub 的 Revert 按钮一步生成回滚 PR。门禁把"恰好是某个近期合并 PR 的逆变更"判为 Low。
- 内核级：撤回内核，即删除 schema（Low）。
- 运行时：设置 `AKLR_DISABLE`。

"谁合并谁负责"：发现问题后自动通知原 PR 的作者。

**验收**：

- 从信号出现到通知原作者不超过 24 小时（由每日任务完成）。
- 纯回滚 PR 判为 Low，只需过 CI。
- 每个事故 issue 都记录"本该抓到它的那一层"。

### R-B8 事故和回滚变成每日评测集

**要求**（均依据 [S1]）：

- 收集引发过事故的 PR，以及它们的修复或回滚；由专家确认 bug 和它的位置。
- 评审员每天在这个集合上跑一遍。
- 评审员的提示词、工具或模型每改一次都要打分：分数上升才保留。
- 每出一次新事故，就加一个用例。
- 同时在已知干净的 PR 上跑，用来统计误报。

**AKLR 映射**：

- 事故来源：R-B7 的信号；回滚提交；用 SZZ 方法找到的引入 bug 的 PR，即对 `fix(...)` PR 改动的行做 blame [S7]。
- AKLR 至今没有回滚，样本很少。所以先用全部约 150 个历史 PR 建一个档位标注集，由两名 owner 独立标注；事故集随时间积累。
- 同一个评测集也给两个分类器打分，主要看判低率。

**验收**：

- 每日任务产出一张记分卡，包括：事故召回率、干净集上的误报率、分类器的判低率和判高率、成本、延迟。
- 修改评审系统的 PR（提示词、策略、模型、阈值）必须附带记分卡，且分数不得退化，否则门禁失败。

### R-B9 指标与护栏

**要求**：

- 人力成本指标：每个 PR 的人工批准数（中位数和 p95）、人类评论数、PR 等人的时间。
- 护栏指标：回滚率、事故率 [S1]。
- AKLR 另加的护栏：绕过合并的次数、合并后的降级数、main 因非平台原因变红的次数、通过评审所需的轮数。

**基线**（来自 §2.4）：

- 批准数中位数 1，p95 为 1；
- 评论数中位数 0；
- 等人时间中位数 1.5 h，p90 18.2 h；
- 回滚 0 次；
- 绕过合并占 29%。

**验收**：

- 每周自动出一份报告，用同一套定义，结果可以复现。
- 指标的定义写在策略旁边；改定义算作修改评审系统。

### R-B10 熔断：指标变坏时自动收紧审批

[S1] 承认这一层还没有实现。

**要求**：护栏指标变坏时自动收紧审批，而不是等人发现。

**AKLR 映射**：下面的规则只是示例；所有数值写在策略文件里，不写成代码常量。

| 条件 | 动作 |
|---|---|
| 14 天内由 Low 或 Medium PR 引发的已确认事故不少于 2 起 | 相关种类升一档 |
| 任何一个 Low PR 引发了事故 | 该种类停用 Low |
| 评测集上 High 用例的判低率大于 0，或评审员的召回率下降 | 停用所有放宽，退回 Phase 2 |
| 出现一次绕过合并 | 告警，并要求 24 小时内事后补审 |

熔断的状态对门禁可读、对人可见。连续 N 天没有事故才自动恢复；手动恢复需要一个 High 档的变更。

**验收**：

- 在回放数据里注入事故，下一次每日任务就让熔断跳闸；此后新 PR 的档位提升。
- 跳闸和恢复都有记录。

## 5. 非功能需求

- **N1 出错即收紧**：分类器或评审员出错、超时，或输出不符合 schema 时，按更严的情况处理：升一档或加一个人。门禁自己出错时，检查结论为失败，而不是放行。
- **N2 可信代码读取不可信数据**：
  - 门禁代码和策略取自 main，通过 `pull_request_target`、`workflow_run` 或 `schedule` 触发。
  - PR 的内容只当数据读，永远不检出执行。
  - 用来签发检查的 App 密钥，放在只有 main 能用的环境里。
  - 必需检查绑定这个 App 的 integration id，PR 自己的工作流伪造不了 [S3]。
- **N3 抵御提示注入**：PR 的标题、正文、评论和代码都是数据。模型类的分类器只能升档，不能降档，所以注入最多让结果更严，不会让 PR 被放行 [S11]。
- **N4 成本**：确定性部分在 1 分钟内跑完，不调用模型。模型调用按 head SHA 加策略版本缓存。日预算有上限，超出后按 N1 处理。
- **N5 延迟**：CI 结束后 5 分钟内，`review-gate` 给出结论；Standard 深度的 AI 评审在 30 分钟内完成。
- **N6 可审计**：每个决定都能从检查记录里复原，包括档位、规则 id、证据、模型输出和策略版本。break-glass 绕过必须留痕，并计入护栏。
- **N7 可回退**：每个阶段都能通过改一个 ruleset 参数或策略字段，退回上一阶段。
- **N8 隔离**：不在 `aklr-claude` runner 上执行 PR 代码，因为它以个人账号运行；这台 runner 上也不放门禁 App 的密钥。
- **N9 公开仓库**：文档和检查输出里不写内部主机名、密钥 ID、预算等内部配置。

## 6. 不做的事

- 不自建评审工具或 git 服务。[S2] 的 Antique 和 Scion 面向每天数万次的合并，AKLR 每天约 10 次。
- 不用合并队列（R-B5）：只跑 CPU 没有意义，跑 GPU 代价过高。
- 现在不训练专用分类器：历史数据太少（设计 §4）。
- 不做仪表盘。周报发在一个跟踪 issue 里。
- 不允许模型批准或合并任何 PR。

## 7. 依赖与前置条件

需要以下准备，细节见设计 §12：

- 一个新的 GitHub App，用来签发门禁检查；
- 模型网关的预算；

## 来源

- [S1] Addy Osmani, Morgan Westlee Lunt, Qing Wang, "How the Claude Code team does code review", 2026-10-15. https://claude.dev/blog/solving-code-review-at-anthropic/
- [S2] Nick G. and the source control team, "Redesigning source control at Anthropic", 2026-10-13. https://www.anthropic.com/research/redesigning-source-control
- [S3] GitHub Docs, "Available rules for rulesets". https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/available-rules-for-rulesets
- [S7] J. Śliwerski, T. Zimmermann, A. Zeller, "When do changes induce fixes?", MSR 2005. https://doi.org/10.1145/1083142.1083147
- [S11] K. Greshake et al., "Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection", arXiv:2302.12173. https://arxiv.org/abs/2302.12173

编号与设计文档共用，所以这里不连续。
