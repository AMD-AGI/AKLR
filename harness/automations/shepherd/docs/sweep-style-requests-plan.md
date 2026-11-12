# 照 Sweep 的做法驱动 shepherd：方案（草案）

做法是把 Sweep 的 issue 流程照搬过来，只在三类地方改：会花 GPU 钱、会把 issue 文字交给有写权限的 agent、或者 Sweep 自己就有缺陷。

只搬设计，不搬代码。运行形态也照 Sweep 做成“一直在线的服务接收事件”（见第 2 节），但和 Sweep 有一处根本不同：任务状态只在账本里，服务本身不记任何状态，重启不丢东西。Sweep 的代码绑在它自己的 webhook 服务和线程表上，搬过来用不了。

Sweep 的做法依据它固定版本的源码：[sweepai/api.py@a8b8b67](https://github.com/sweepai/sweep/blob/a8b8b67bda4f89faac9314d34e7c7d5a64f76046/sweepai/api.py)。

## 1. 逐项对照：照抄还是改

| Sweep 的做法 | 本方案 | 理由 |
|---|---|---|
| 标题以 `sweep` 开头或含 `sweep:` 时自动加 label；`issues/labeled` 才开工 | **照抄**：标题以 `shepherd:` 开头的 issue 自动加 `shepherd` label，`labeled` 事件开工 | 用法和你的想法一致 |
| 谁都能触发 | **改**：只认一个 GitHub team 的成员（主机配置里写 team 名）。先建一个只有你的 team，跑通后再加可信的人 | 仓库公开、41 个 write 用户，Triage 权限就能加 label，GitHub 做不到只给 admin 用；用 team 管名单，加人不用改主机配置 |
| 标题和正文原样交给 agent，当作任务 | **改为两层**：正文开头的 YAML 块（`schemas`、`max_prs`、`max_campaigns`）由脚本解析，决定做什么；其余自由文本当作提示，只交给节点容器里的 KernelForge worker，不交给 round agent | round agent 跳过权限确认，手里有 App 写 token、集群访问和预算；worker 在容器里碰不到 GitHub 和账本，产出还要过可信验证 |
| 预算由服务端控制 | **照抄**：钱、模型、并发、发布权限只在主机配置里，issue 只能缩小范围 | 和 Sweep 一样，issue 定“做什么”，不定“花多少” |
| 以 `repo-issue号` 为键，一个 issue 一个任务 | **照抄键，改载体**：账本里的 `requests` 条目以 issue 号为键；一个请求可以对应多个 campaign | shepherd 本来就以账本为准，能断点续跑；进程里的线程表重启就丢 |
| 改正文就杀掉旧任务再重跑 | **改**：改正文只影响以后的准入；在跑的 campaign 不受影响，不杀也不重跑 | 一个 campaign 要几小时 GPU，重跑等于把钱扔掉；Sweep 往线程里注入异常来杀任务，本身也不可靠 |
| 评论以 `sweep:` 开头当追加指令 | **不抄** | 要改就改正文。评论没法按版本冻结，也最容易被注入 |
| 一条机器人评论原地更新进度（含“restart”勾选框） | **照抄进度评论，不抄 restart**：每轮 round 后由脚本按模板改写这一条评论，列出每个 schema 的状态和 PR 编号 | 进度看一处就够；重启语义改由关闭再重开、重新加 label 来表达 |
| PR 正文写 `Fixes #N` | **改**：写 `Refs #N` | 第一个 PR 合入就会关掉 issue，等于结束整个请求 |
| 关闭 issue 不做任何事 | **改**：关闭 issue 等于停止。在跑的 campaign 停下、清理、确认释放节点，然后结算；已开的 PR 保留 | 这是你方案里的要求，也是 Sweep 的漏洞 |
| 异常一律记日志，然后照样返回成功 | **改**：受理失败时在进度评论里写原因码；脚本出错这一轮按 `unknown` 处理，不准入，也不据此停东西 | 失败要让人看得见 |

## 2. 运行形态：runner 收事件，主机上的服务跑 round

同意做成一直在线的服务，它和本分支已有的设计（`state.py round` + 账本 + 每轮新起的 agent）不冲突：服务只决定什么时候跑一轮，每一轮怎么跑完全不变。

- **事件入口是 `aklr-claude` runner**：runner 主动连 GitHub 拉任务，主机不用开任何入站端口。这点很重要，因为这台主机在 AMD 内网，GitHub 的 webhook 打不进来。
- **workflow 只转发事件**：`shepherd.yaml` 的 job 只把 issue 号和事件类型写进主机上的事件队列目录，几秒钟就结束，不在 job 里跑 round。
- **主机上的服务跑 round**：
  - 收到新事件就跑一轮；
  - 没有事件时按账本的 `wake_at` 定时跑；
  - 同一时间只跑一轮，靠 `state.py` 现有的锁保证。

和“在 Actions job 里跑 round”相比，好处有三条：
1. round 不再占着唯一的 runner，@claude 的 job 不用排队；
2. 不受 job 60 分钟超时和 job token 有效期的限制；
3. GitHub 的定时触发会延迟或丢失，主机上的定时器不会。

服务的形式，推荐先用 systemd 的 `.path` 和 `.timer` 单元，不写 Python 常驻进程：
- `.path` 监视队列目录，有新文件就启动一次 `state.py round`；
- `.timer` 替代 GitHub 的定时触发；
- 两个单元都是 `systemctl --user`，和现有的 `aklr-claude-runner.service`、`magpie.service` 同一种做法。

这样崩溃重启、日志、同一时间只跑一个，都由 systemd 负责，不用写代码。以后真要做到 round 之间也能即时响应（例如关 issue 后几秒内就停掉 campaign），再换成 Python 常驻进程，入口不用改。

队列文件里只有 issue 号和事件类型，服务不信任这两样之外的任何内容；issue 的内容由 `request_board.py` 自己去 GitHub 读。

## 3. 一次请求的流程

| 你做什么 | shepherd 做什么 | issue 上出现什么 |
|---|---|---|
| 开 issue，标题写 `shepherd: <描述>`，正文开头写 YAML 块 | 开 issue 的人在 team 里，shepherd App 就自动加 `shepherd` label；这个 label 算作开 issue 的人批准的 | — |
| （或者）手动加 `shepherd` label | 检查最近一次加 label 的人在 team 里，正文最后编辑人也在 team 里 | — |
| — | 下一轮 round：解析 YAML 块并把这一版冻结进账本，按顺序准入 | 进度评论：已受理，列出 schema 和上限 |
| — | campaign 跑完，通过检查后开 draft PR，正文写 `Refs #N` | 进度评论更新；PR 自动出现在 issue 时间线上 |
| 你改了正文 | 以后的准入按新版本，在跑的不动 | 进度评论注明当前是第几版 |
| 不在 team 里的人改了正文 | 暂停准入，等 team 里的人重新加 label | 进度评论：已暂停及原因 |
| 去掉 label | 不再准入，在跑的做完 | 进度评论：收尾中 |
| 关闭 issue | 在跑的 campaign 停止、清理、结算 | 进度评论改成最终汇总 |
| PR 数或 campaign 数用完，或者没有可做的了 | 等 campaign 都不再占节点后，请求结束 | 最终汇总，请你关掉 issue |

## 4. 结构表

| 文件 | 概念 | 类 | 公开方法/函数 |
|---|---|---|---|
| `harness/automations/shepherd/skills/shepherd/scripts/request_board.py`（新） | issue 上的优化请求 | `RequestBoard`（仓库、label、requesters team、允许的 op）；`Request`（frozen：issue 号、状态、版本哈希、schemas、上限、提示文本） | `RequestBoard.from_config(path)`、`RequestBoard.requests(tracked)`、`RequestBoard.write_progress(request, campaigns)`、`Request.from_entry(entry)`、`main()`（子命令 `observe`、`progress`） |
| `harness/tests/shepherd/test_request_board.py`（新） | 上面脚本的测试，用假的 `gh` 回放 GraphQL | — | `test_*` |
| `.github/ISSUE_TEMPLATE/shepherd.yml`（新） | 请求的 issue 表单，预填 `shepherd:` 标题和 YAML 块 | — | — |
| `.github/workflows/shepherd.yaml`（新，上一版就要建，本版改了职责） | 把 `issues` 的 opened/labeled/unlabeled/edited/closed 事件和手动触发转发进主机的事件队列；不再跑 round，也不再定时触发 | — | job `forward` |
| `harness/automations/shepherd/templates/shepherd-round.service`、`.path`、`.timer`（新） | 主机上跑 round 的服务：队列有新事件时跑一轮，否则按定时器跑 | — | — |
| `harness/automations/shepherd/skills/setup-shepherd/SKILL.md`（改） | shepherd 的安装流程：安装三个 systemd 单元，填写 requesters team | — | — |
| `harness/automations/shepherd/skills/shepherd/SKILL.md`（改） | 一轮 shepherd | — | — |
| `harness/automations/shepherd/skills/shepherd/references/state.md`（改） | 账本新增 `requests` 条目 | — | — |
| `harness/automations/shepherd/templates/configuration.example.yaml`（改） | 新增 `requests: {label, team}` 段 | — | — |

分层：
- **issue 层（GitHub）**：表单、label、`Refs #N`、进度评论。
- **事件层（runner + 主机上的 systemd 服务）**：runner 把事件转发进队列，服务据此或按定时器跑 round。
- **受理层（主机上的 `request_board.py`）**：读时间线核对 team 成员，解析 YAML 并冻结版本，改写进度评论。
- **执行层（round agent + 节点上的 worker）**：round 只看账本里的请求，看不到 issue 文字；提示文本随 campaign 交给节点上的 worker。

## 5. 分阶段

**阶段 0：前提**
- 上一版的阶段 0：锁住 `aklr-claude` runner（A 或 B）、建 App；
- shepherd 合入 main；
- 建一个只有你的 team，并给 shepherd App 加上组织的 Members 读权限（查 team 成员要用），以及仓库的 Issues 写权限（发进度评论、自动加 label 要用）。这两项都要 org owner 批准；
- 装好 systemd 服务，`shepherd.yaml` 转发事件，在 `mode: observe` 下跑通。

**阶段 1：受理和进度评论，不花钱**
- 做：`request_board.py` 的 `observe` 和 `progress`、表单、`issues` 触发。
- 验收：你开的 issue 自动加上 label，出现进度评论；team 外的人加 label 没有反应；改正文、去掉 label、关闭 issue 后，状态分别对应。

**阶段 2：由请求启动付费 campaign**
- 做：round 按请求准入、关闭即停止、`Refs #N`、最终汇总、把提示文本交给 worker。
- 验收：第一单 1 个 schema、`max_prs: 1`，从请求走到 draft PR 再到最终汇总；另开一单跑到一半时关闭 issue，campaign 停止并确认清理。

## 6. 已定

1. 标题以 `shepherd:` 开头时自动加 label：要。
2. 名单用 GitHub team：先只有你，跑通后再建正式的组，加上可信的人。
3. 进度评论用 shepherd App 发。
4. 做成一直在线的服务：要；先用 systemd 单元实现（见第 2 节）。

## 7. 还需要你决定

1. 自由文本提示交给节点上的 worker，可以吗？还是先不用自由文本，只用 YAML 块？
2. 给 shepherd App 加组织的 Members 读权限需要 org owner 批准，谁来批？
