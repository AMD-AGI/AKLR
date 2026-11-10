# AgentX CI 使用说明

本文介绍 AKLR 的 AgentX 性能测试流程、GitHub 触发方式、本地 CLI、模型配置和 GitHub runner 注册。

## 1. Github Runner Pipeline 介绍

[agentx-ci.yaml](../../.github/workflows/agentx-ci.yaml) 包含四个 job：

```mermaid
flowchart TD
    T["手动、PR 标签或定时触发"] --> P["plan：解析模型、固定上游版本与运行配置"]
    P --> S["sweep：准备环境，逐模型逐点测试并保存结果"]
    S --> R["report：校验结果，生成报告与基线，汇总测量结果"]
    R -->|"满足发布条件"| D["pages：部署已上传的 Pages artifact"]
    P -.->|"上传"| A["agentx-plan artifact"]
    A -.->|"读取 serving.json"| S
    A -.->|"读取 manifest.json"| R
```

实线表示 job 的执行顺序；虚线表示 artifact 的上传或读取。`agentx-plan` 只上传一次，分别供 sweep 和 report 读取。report 等待 plan 和 sweep 结束后运行；sweep 失败时仍可依据计划生成失败报告。

sweep 遍历完所选数据点后，只要至少一次测量成功，job 就正常结束。有其他点失败时，摘要显示**部分成功**，全部测量失败时 sweep 才因测点结果报失败。所选模型和 target 的所有点成功完成正式测量且源码为干净状态才会更新该 target 的基线。失败、部分完成、dirty 或源码状态未知的运行不会更新基线。

### 1.1 使用 Github runner 触发

打开 [AKLR-agentx-ci workflow](https://github.com/AMD-AGI/AKLR/actions/workflows/agentx-ci.yaml)，点击 **Run workflow**，选择 AKLR 分支、填写下面的参数，再点击表单中的 **Run workflow**。runner 的注册与准备见第 4 节。

表单共有 **7 个 workflow 输入项**，另有 GitHub 提供的 **Branch / Use workflow from** 选择器，合计 8 项设置；7 个输入项均可保留默认值。

| 页面设置 / 输入名 | 默认值 | 含义 |
| --- | --- | --- |
| Branch / Use workflow from | 默认分支，通常为 `main` | 选择本次使用的 **AKLR** workflow、源码和配置版本 |
| `models` | `all` | 选择已启用的模型 ID；多个用英文逗号分隔，例如 `glm52-fp4`；`all` 选择全部已启用模型 |
| `targets` | `all` | 在所选模型中选择已启用的 target ID，例如 `mi355x`；多个用英文逗号分隔，按计划顺序串行执行 |
| `inferencex_ref` | 留空 | 覆盖 **InferenceX** 的分支、tag 或 SHA，决定测试网格、serving 配置和客户端；留空使用配置中的 ref，当前为 `main` |
| `arms` | `all` | 选择生成的服务方案 ID，例如 `tp8-ep1-none-spec-mtp`；多个用英文逗号分隔；`all` 不按 arm 过滤 |
| `points` | 留空 | 精确选择 `ARM:CONC` 测试点，例如 `tp8-ep1-none-spec-mtp:1`；多个用英文逗号分隔；留空不追加点过滤，仍受 `arms` 限制 |
| `duration_s` | 留空 | 覆盖每个点的测量时长，单位秒，必须为正整数，例如 `60`；留空继承上游生成计划中的时长，当前 AgentX 默认 `3600` 秒 |
| `publish` | `false`，不勾选 | 勾选后将报告部署到 GitHub Pages；不勾选仍会上传已生成的结果与报告 artifacts |

### 1.2 通过 PR 标签触发

对于本仓库内分支发起、目标分支为 `main` 的 PR，在 PR 页面的 **Labels** 中添加 **`ci:agentx-mi355x`** 即可触发 AgentX CI。

| 触发方式 | 模型 / target 范围 | 测量点 | AKLR off/on 每次测量时长 | 发布 Pages |
| --- | --- | --- | --- | --- |
| PR 标签 `ci:agentx-mi355x` | 全部绑定 `agentx-mi355x` 的已启用 target | 上游 Pareto 前沿 | 600 秒 | 否 |

注意：
- PR 创建、更新提交、重新打开时，如果保留该标签，就会触发运行。添加标签时，只有新增的标签正是 `ci:agentx-mi355x` 才会启动 AgentX 测试。
- 保留标签后推送新提交会触发新的 run，该 PR 正在运行或排队的 AgentX run 会被自动取消。
- `ci:agentx-mi355x` 是 **PR 标签**；`agentx-mi355x` 是 **GitHub Actions runner 标签**。PR 运行只选择配置中 `runner_label: agentx-mi355x` 的已启用模型和 target。

### 1.3 定时运行

将仓库变量 `AGENTX_SCHEDULE_ENABLED` 设为 `true`，即可启用每周五 10:17 UTC 的定时运行。定时使用默认分支和配置中的 InferenceX ref。

| 触发方式 | 模型 / target 范围 | 测量点 | AKLR off/on 每次测量时长 | 发布 Pages |
| --- | --- | --- | --- | --- |
| 定时 | 全部绑定 `agentx-mi355x` 的已启用 target | 全量网格 | 3600 秒 | 是 |

## 2. CLI 命令介绍

### 2.1 准备环境与命令入口

本节命令在仓库的 `benchmarks/agentx` 执行。

```bash
# 从仓库根目录开始
cd benchmarks/agentx
python3 -m pip install pyyaml==6.0.3
python3 -m agentx_bench pipeline --help
```

| 子命令 | 用途 |
| --- | --- |
| `pipeline` | 单模型和单 target 一键 plan → sweep → report → 清理 |
| `plan` | 解析一个或多个模型的 target，生成可供 report 校验的 manifest |
| `sweep` | 在 Docker 中安装当前 checkout 并执行 serving 计划 |
| `resolve` | 只解析单模型的 serving 配置，便于检查或使用本地 InferenceX checkout |
| `report` | 渲染单个结果；加 `--plan` 时校验计划、生成站点并管理基线 |

### 2.2 使用 pipeline 一键测试

`pipeline` 每次只测试一个模型和 target，必须使用 `--model <id>`；模型只有一个已启用 target 时可省略 `--target`。以下示例假定固定版本权重已经放在 `/data/hf/hub`；准备方式见第 4 节。

```bash
python3 -m agentx_bench pipeline \
  --model glm52-fp4 \
  --target mi355x \
  --hub /data/hf/hub \
  --out /data/agentx/run-001
```

`--out` 必须是尚不存在的新目录。完成后打开 `/data/agentx/run-001/site/index.html`。

`pipeline` 输出目录结构如下。

```text
<out>/
├── plan/
│   ├── manifest.json
│   └── models/<model-id>/targets/<target-id>/serving.json
├── results/models/<model-id>/targets/<target-id>/
│   ├── install.log
│   ├── prepare_traces.log
│   ├── results.json
│   └── attempts/<attempt-id>/<mode>/points/<exp-name>/
│       ├── server.log
│       ├── client.log
│       └── ...
├── site/
│   ├── index.html
│   ├── manifest.json
│   └── models/<model-id>/targets/<target-id>/
│       ├── index.html
│       ├── results.json
│       ├── report.json
│       └── baseline-live.json 或 baseline-pareto.json
├── work/                  # 默认临时解析、运行时与 trace 缓存
└── cache/                 # 默认临时编译等缓存
```

`<mode>` 为 `disabled`（AKLR off）或 `enabled`（AKLR on）。日志只在执行到相应阶段后出现。默认 `work/`、`cache/` 在本地 pipeline 结束后清理；显式传入 `--work`、`--cache` 的目录保留。

常用参数：

| 参数 | 说明 |
| --- | --- |
| `--config` | 自定义 YAML 路径；默认使用本目录的 `agentx.yaml` |
| `--model` | 必填，恰好一个已启用模型 ID；不接受 `all` 或逗号分隔列表 |
| `--target` | 恰好一个已启用 target ID；仅当模型只有一个已启用 target 时可省略 |
| `--inferencex-ref` | 覆盖 InferenceX 分支、tag 或 commit SHA |
| `--run-image` | 在计划阶段指定执行镜像，建议固定 digest |
| `--arms` / `--points` | 全量策略下的 smoke 过滤；`--points` 格式为 `ARM:CONC,ARM:CONC` |
| `--duration` | 每点 DURATION 覆盖，单位秒；设定后不晋升正式基线 |
| `--model-path` | 使用容器内可见的本地模型目录；不会自动新增挂载，且不晋升正式基线 |
| `--git-dir` | 生成报告中 AKLR 提交历史所用的 checkout；默认本仓库 |

### 2.3 分阶段测试

先解析计划，不运行 GPU：

```bash
python3 -m agentx_bench plan \
  --models glm52-fp4 \
  --targets mi355x \
  --run-id staged-001 \
  --work /data/agentx/staged-work \
  --out /data/agentx/staged-001/plan

python3 -m json.tool /data/agentx/staged-001/plan/manifest.json
```

确认 manifest 中该模型的 target 状态为 `resolved` 后，在 Docker 中运行该 target：

```bash
python3 -m agentx_bench sweep \
  --serving /data/agentx/staged-001/plan/models/glm52-fp4/targets/mi355x/serving.json \
  --hub /data/hf/hub \
  --run-id staged-001 \
  --work /data/agentx/staged-work/sweep-glm52 \
  --cache /data/agentx/staged-cache \
  --out /data/agentx/staged-001/results/models/glm52-fp4/targets/mi355x
```

最后生成总览与模型报告：

```bash
python3 -m agentx_bench report \
  --plan /data/agentx/staged-001/plan/manifest.json \
  --results /data/agentx/staged-001/results \
  --baseline-dir /data/agentx/run-001/site \
  --out /data/agentx/staged-001/site
```

首次没有基线时省略 `--baseline-dir`。`plan` 可通过 `--models` 和 `--targets` 生成 CI 所需的多模型、多 target 计划；本地分阶段示例只运行一个模型和 target。`--results` 指向包含 `models/` 的根目录，sweep 的 `--out` 则指向具体 target 目录。plan 与执行必须使用同一 `--run-id`。

`plan` 专用选项 `--runner-label agentx-mi355x` 可进一步筛选所选模型中绑定该 runner 标签的已启用 target，PR 标签和定时入口均使用此筛选。`--require-recipe-duration` 只校验继承的时长，不作覆盖，不能与 `--duration` 同时使用。

## 3. 配置介绍

### 3.1 agentx.yaml 字段说明

当前配置如下：

```yaml
inferencex:
  repo: SemiAnalysisAI/InferenceX
  ref: main
models:
  glm52-fp4:
    enabled: true
    display_name: GLM-5.2 MXFP4
    weights:
      repo: amd/GLM-5.2-MXFP4
      revision: 386bd0e4ec821f7b07975701cec3c3b953a5576a
    targets:
      mi355x:
        hardware:
          gpu_name_pattern: '\bMI355X\b'
          gpu_arch: gfx950
          gpu_count: 8
        runner_label: agentx-mi355x
        inferencex:
          config_key: glm5.2-fp4-mi355x-sglang-agentic-mtp
          expect:
            config_file: inferencex-e2e/configs/amd-master.yaml
            recipe: inferencex-e2e/benchmarks/single_node/srt-slurm-recipes/glm5.2/sglang/mi355x-fp4-mtp/agentic.yaml
            model: amd/GLM-5.2-MXFP4
            model-prefix: glm5.2
            framework: sglang
            precision: mxfp4
            runner_pattern: mi355x
report:
  repo_url: https://github.com/AMD-AGI/AKLR
advisory_env:
  AKLR_SGLANG_PLUGIN: '1'
```

| 字段 | 说明 |
| --- | --- |
| `inferencex.repo` | 默认上游 GitHub 仓库，格式 `owner/repo` |
| `inferencex.ref` | 默认上游分支、tag 或 SHA；plan 将其固定到 commit；CLI `--inferencex-ref` 优先 |
| `models.<id>` | 模型独立标识，用于 CLI、目录与基线隔离；1–64 位小写字母、数字、`_`、`-`，首位须为字母或数字，`all` 保留 |
| `enabled` | 模型或 target 是否启用，布尔值，省略时为 `true`；禁用项不能通过显式 ID 选择 |
| `display_name` | 报告展示名，省略时使用模型 ID |
| `models.<id>.targets.<target-id>` | 模型下的硬件目标，ID 规则与模型相同；每个 target 使用独立目录与基线，例如 `mi355x` |
| `hardware.gpu_name_pattern`（target 内） | 匹配 GPU 名称的正则表达式，例如 `\bMI355X\b` |
| `hardware.gpu_arch`（target 内） | 要求的 GPU 架构，例如 `gfx950` |
| `hardware.gpu_count`（target 内） | 目标主机的 GPU 数量；当前配置为 `8` |
| `runner_label`（target 内） | GitHub 调度该 target 使用的 runner 标签，需与注册时的 `--label` 一致 |
| `inferencex.repo` / `ref`（target 内） | 可覆盖全局上游仓库和 ref |
| `inferencex.config_key`（target 内） | 上游 master 配置中的条目键，决定该 target 的矩阵与 recipe 绑定 |
| `inferencex.expect.config_file` | 上游配置文件路径，相对 InferenceX 仓库根目录 |
| `inferencex.expect.recipe` | 预期 recipe 文件路径，相对 InferenceX 仓库根目录；解析时检查文件存在 |
| `inferencex.expect.model` | 要求上游条目的模型名称精确匹配 |
| `inferencex.expect.model-prefix` | 要求上游条目的模型前缀匹配，也用于客户端 trace 选择 |
| `inferencex.expect.framework` | 要求上游框架匹配；当前本地启动实现使用 SGLang |
| `inferencex.expect.precision` | 期望权重精度；上游通用 `fp4` 会结合模型名归一化为 `mxfp4` / `nvfp4` |
| `inferencex.expect.runner_pattern` | 匹配上游 runner 字段的正则表达式；它不是本仓库 GitHub runner 的标签 |
| `weights.repo` | Hugging Face 模型仓库，每个模型独立设置 |
| `weights.revision` | 固定的权重 commit SHA；同时用于 snapshot 路径和服务 revision |
| `advisory_env` | 全局服务器补充环境变量；当前用于启用 AKLR SGLang plugin，值建议写成字符串 |
| `models.<id>.advisory_env` | 模型级补充环境，覆盖同名全局值；若与上游 recipe 已设置的环境变量同名，会报冲突 |
| `models.<id>.targets.<target-id>.advisory_env` | target 级补充环境，覆盖同名模型级和全局值；同样不允许与上游 recipe 环境冲突 |
| `report.repo_url` | 报告中 AKLR commit 与历史链接的仓库 URL；不是 Pages 站点地址 |

上述 `inferencex.expect.*` 字段均位于 target 的 `inferencex` 中；`weights` 由同一模型的各个 target 共享。

### 3.2 新增模型

先在目标 InferenceX ref 中确认模型的 `config_key`、配置路径、recipe、模型名称、精度和 runner。当前执行器面向单机 SGLang；仅修改 `framework` 不能使其支持其他服务框架。

在已有 `models:` 下加入一个独立节点。下面是模板，尖括号内容必须替换为真实值，不能直接执行：

```yaml
  new-model:
    enabled: true
    display_name: New Model
    weights:
      repo: <Hugging Face owner/model>
      revision: <固定的40位权重commit SHA>
    targets:
      mi355x:
        hardware:
          gpu_name_pattern: '\bMI355X\b'
          gpu_arch: gfx950
          gpu_count: 8
        runner_label: agentx-mi355x
        inferencex:
          # repo/ref 可省略，继承全局值
          config_key: <上游配置条目键>
          expect:
            config_file: inferencex-e2e/configs/amd-master.yaml
            recipe: <相对上游仓库根目录的 recipe 路径>
            model: <上游模型名称>
            model-prefix: <上游模型前缀>
            framework: sglang
            precision: <期望精度，例如 bf16 或 mxfp4>
            runner_pattern: mi355x
```

按以下顺序验证：

1. 用 `plan --models new-model --targets mi355x --work /data/agentx/new-work --out /data/agentx/new-plan` 解析，检查 manifest 中的 target 为 `resolved`，并检查 serving 中生成的 GPU 数、arm、并发、权重和镜像。
2. 按第 4 节把新增模型的固定 snapshot 和 tokenizer 所需的 `refs/main` 准备到相同 hub；确认 runner 能容纳该模型及 recipe 的主机内存需求。
3. 从生成的点中选一个，按第 2 节运行 smoke pipeline；查看 `server.log`、`client.log` 和失败原因。
4. 完成配置后提交源码，再执行全量正式测量；`plan --models all` 会包含这个已启用模型的 target，定时 CI 则包含其中绑定 `agentx-mi355x` 的 target。PR 标签运行还需要上游已发布的匹配测量来选择 Pareto 点。本地用 `pipeline --model new-model --target mi355x` 单独测试。

## 4. 注册 runner

### 4.1 执行注册脚本

在目标 GitHub 仓库进入 **Settings → Actions → Runners → New self-hosted runner**，获取临时 registration token。它通过 `RUNNER_TOKEN` 环境变量传入。

```bash
# 从仓库根目录执行；交互输入不会显示 token
read -r -s -p 'Runner registration token: ' RUNNER_TOKEN
echo
export RUNNER_TOKEN

bash scripts/agentx/register_runner.sh \
  --repo AMD-AGI/AKLR \
  --label agentx-mi355x \
  --dir /opt/agentx-runner \
  --name "$(hostname -s)-agentx" \
  --hf-hub-dir /data/hf/hub

unset RUNNER_TOKEN
```

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--repo` | `AMD-AGI/AKLR` | 注册到哪个仓库；token 必须来自该仓库 |
| `--label` | `agentx-mi355x` | workflow 用来调度的自定义标签 |
| `--version` | `2.338.2` | 下载的 GitHub Actions runner 版本，可显式覆盖 |
| `--dir` | `/opt/agentx-runner` | runner 安装目录 |
| `--name` | `<hostname>-agentx` | GitHub 上显示的 runner 名称 |
| `--hf-hub-dir` | 无 | 可选：检查目录存在并打印后续变量设置提示 |

脚本会下载 Linux x64 runner、调用 `config.sh --unattended --replace` 注册，然后使用 `svc.sh install/start` 安装并启动服务。`--replace` 会替换目标仓库中同名 runner 的注册，给新主机选择独立名称。

### 4.2 设置仓库变量并验证

在 **Settings → Secrets and variables → Actions → Variables** 设置：

| 仓库变量 | 是否需要 | 值与用途 |
| --- | --- | --- |
| `AGENTX_HF_HUB_DIR` | 必填 | runner 上权重 hub 的绝对路径，例如 `/data/hf/hub` |
| `AGENTX_SCHEDULE_ENABLED` | 启用定时时填写 | 字符串 `true`；只控制定时触发，不影响手动或 PR 标签触发 |
| `AGENTX_PAGES_URL` | 自定义站点地址时填写 | 读取历史站点和基线的根 URL；默认 `https://<owner>.github.io/<repo>` |

所选 target 在 `agentx.yaml` 中的 `runner_label` 必须与注册时的 `--label` 一致；本例均为 `agentx-mi355x`。

需要发布报告时，在 **Settings → Pages** 将构建与部署来源设为 **GitHub Actions**，并确认 `github-pages` environment 的部署规则允许目标分支。

确认 runner 在 GitHub 页面显示 Online / Idle，并在主机检查服务：

```bash
cd /opt/agentx-runner
sudo ./svc.sh status
```
