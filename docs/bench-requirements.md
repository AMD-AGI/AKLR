# AKLR bench 阶段需求规格

本文对应 `refactor/core` 分支提交 `95471c5a975dab3eed69c3cdb8c172317ffb04c7`，研究范围为 `aklr/core/validation/stages/_bench/` 及其必要外部契约。目标是规定可重新实现、可验收的行为，而非要求保留现有类的内部组织。所有路径均相对于仓库根目录，GiB、MiB、KiB 均使用二进制单位。

条款分两档：

- **契约**（未标注的内容，用“必须”）：对外可观察行为，包括阶段输入输出、finding code 及归因、门槛与统计、报告和 evaluation 的 JSON、退出码、可信边界。重实现必须保持。
- **测量协议**（标注【协议】）：影响测量数值的参数与方法。可以修改，但必须同步修改 pipeline 的 `measurement_profile` 并升版本。

本文未规定的内容由实现自行决定，包括内部类型与接口、批大小与调度优先级、超时与等待时长、日志限额、finding detail 文案和展示格式。第 9 节列出本规格与上述提交代码的差异，即重实现或修复须改变的现有行为。

## 1. 范围、术语与输入

**BENCH-I-01 — 职责。** bench 必须对指定 schema 的每个候选 kernel workload 检查数值正确性、输入不变性及 CUDA Graph 可用性，并在有比较对象时测量性能。输出必须允许区分候选缺陷、平台故障、单个 workload 降级以及整个 schema 无可接纳 workload。bench 不负责替代前置 audit，也不负责实际安装运行时 dispatch。

**BENCH-I-02 — 阶段调用。** 阶段名必须为 `bench`。调用输入为 `ValidationScope` 和此前的 `PipelineReport`，构造时还须提供非空设备序列。scope 包含仓库根目录、head commit、可选 base commit、audit 的 schema 路径和 bench 路径。bench 只处理 `bench_paths`；为空时返回 `skipped`。某路径的 schema 没有 kernel 时不生成 trial。所有记录中的 schema 路径均为相对仓库根目录的 POSIX 路径。

**BENCH-I-03 — 前置约束。** 正常 pipeline 必须先运行 audit；此前阶段非 `passed` 时跳过 bench，并记录此前阶段及其状态。audit 负责 schema JSON/源码格式、文件布局、集合约束、kernel `init` 存在性、workload 轴声明、kernel 源码规则和评测字段不可私改等检查。有 kernel 的 schema 还必须由 audit 保证：`reference` 非空、所有声明输出为 tensor、至少有一个 workload。bench 不应在 audit 未通过后执行候选代码。直接调用 bench 的调用者必须提供等价的已审查输入；遗漏的 schema 契约仍按第 7 节失败规则处理。

**BENCH-I-04 — schema 和 kernel 契约。** 每个 trial 必须持有 schema 名称和 `op_type`、常量轴、workload 的变量轴、输入/输出描述、独立源码形式的 `init`/`reference`/`check`、kernel 源文件及 build 信息。build 包括语言、入口、依赖、可选 binding、目标硬件和 destination-passing-style 标记。bench 的输出检查要求输出是具备确定 shape、dtype 的 tensor。输入可以包含 tensor、scalar、可选值及 API 参数映射；参数排序/映射须遵循 schema 和运行时适配契约。

**BENCH-I-05 — workload 身份。** workload 通过 schema 路径和轴值定位。incumbent 的匹配键是 schema 名称及排序后的轴名/轴值集合；种子身份是 schema 名称和轴字典，不能使用 kernel 名称、源码、commit 或进程相关的 Python `hash()`。

**BENCH-I-06 — kernel 身份。** kernel 内容身份必须与 `Kernel.hash()` 一致，其定义归 `aklr/core/schema` 所有。判断是否为同一 kernel、是否需要 incumbent 比较及报告 `kernel_content_hash`，均使用此内容身份。

**BENCH-I-07 — scope 的产生。** CLI 必须从工作树及 Git 产生 scope：head 为 `HEAD`，`--base` 转为它与 `HEAD` 的 merge-base。显式路径必须是仓库内现存文件。没有 base 时不搜索 incumbent。base 在整个 ops 数据目录中没有任何被识别的 schema 数据文件时为首次发布。未给显式路径时的自动选取归 `ValidationScope` 所有，不属于 bench 规格。

**BENCH-I-08 — CLI。** 必须兼容 `python -m aklr validate [PATH ...] --base REF --stages audit,bench --devices SPEC --out DIR --allow-measured-edits`。默认阶段为 audit、bench，阶段必须有序、无重复且以 audit 开始。设备规格支持逗号分隔整数及闭区间，例如 `0-7`、`0,2,5`；未指定时使用全部可见 GPU。没有设备或同一设备出现多次时为参数错误。默认输出目录为 `aklr_validation`。`--allow-measured-edits` 只改变 audit 对评测字段编辑的限制，不放宽 bench 的数值或性能要求。

**BENCH-I-09 — 环境配置。** `AKLR_MIN_SPEEDUP` 默认 `1.03`，必须解析为有限浮点数，用于查找当前可服务的 incumbent；它不是 bench 的 `entry`/`retain` 门槛。无 adapter 时 incumbent 查找使用 `min_speedup=0`。bench 以显式 scope、设备和直接 kernel 调用工作，不读取其他运行时开关。GPU 可见性遵循启动环境及 PyTorch 的逻辑编号。

**BENCH-I-10 — 固定测量配置。** 此版本不提供用于改变 seed、容差、warmup、repeat、置信水平或超时的 bench CLI 开关。测量配置必须在 pipeline 的 `measurement_profile` 中公布；不能静默切换 eager 计时或热缓存计时。

## 2. baseline 与 incumbent 的确定

**BENCH-F-01 — seam baseline 来源。** 必须完成 seam adapter 声明加载，再从 `SeamRegistry.default().adapters(schema.op_type)` 获取有序 adapter 集合。集合只包含 API 声明了该 `op_type` 的 adapter。`entry` 为后端原始入口的点分名称，`backend` 为入口所属顶层 Python 包名。测量必须调用未被 AKLR 替换的原始实现。

**BENCH-F-02 — baseline 绑定。** 对具体 API call 依序尝试 adapter，选用第一个能绑定成可无参执行的 callable 的 adapter。所有 adapter 均未绑定时 workload 不能评测：没有解析异常时产生 `baseline_failed`，提示移除该 workload 或修正 schema、重跑无助。没有注册 adapter 则表示无 seam baseline；“有 adapter 但都不可用”不能自动视为没有 baseline。

**BENCH-F-03 — incumbent 搜索范围。** 有 base 时必须读取同一 op 的 base 版本 `impls/` 下所有被布局识别的 schema 数据文件，跨实现语言搜索。缺 kernel 的文件不参与。按路径属于候选 schema 名称的 base 文件无法按当前格式解析（无论是否为合法 JSON）时，该 schema 记 `platform_failure` 且不评测，避免在缺少真实 incumbent 时放宽门槛；其他名称的不可解析文件跳过。schema 格式变更时，base 中的旧文件须先迁移。

**BENCH-F-04 — 可服务性判断。** 对目标 `arch/hardware` 查找 workload evaluation：没有 evaluation 时，仅 `min_speedup<=0` 合格；accuracy 为 `failed` 时始终不合格；否则，当 `min_speedup<=0`、缺少所选 backend 的 speedup，或该 speedup 值不小于 `min_speedup` 时合格。旧 kernel 的 build 不支持目标硬件时不合格。所选 backend 是该 workload 实际绑定的 adapter 的 backend，因此 baseline 绑定须先于 incumbent 选择完成；无 adapter 时没有所选 backend。与候选内容相同的旧 kernel 即使未满足可服务性，也必须进入匹配候选集。

**BENCH-F-05 — incumbent 决胜规则。** 同名、同轴 workload 有多个候选时，按下列优先级依次取最大者：与当前 kernel 内容相同；没有失败 accuracy（无 evaluation 也算）；所选 backend 的 speedup 较大（缺失视为负无穷）；最后用任一确定性规则决胜。必须同时保留选中 schema、其 base 路径和是否满足可服务性三个结果。

**BENCH-F-06 — 比较对象和门槛。** 必须按下表产生 trial 的 `thresholds`。其中 `entry=1.02`，`retain=0.99`，比值方向均为对照耗时/候选耗时。

| 条件 | `baseline` 计时对象 | baseline 门槛 | 额外 incumbent 比较 |
| --- | --- | --- | --- |
| 有 adapter；无 base、首次发布或选中 incumbent 与候选同内容 | 成功绑定的 seam 原始实现 | retain | 仅存在不同内容 incumbent 时使用 retain |
| 有 adapter；其他情况 | 成功绑定的 seam 原始实现 | entry | 不同内容 incumbent 使用 retain |
| 无 adapter；存在不同内容 incumbent | incumbent | retain | 不再单列 |
| 无 adapter；无不同内容 incumbent | 无，不计时 | 无 | 无 |

同内容 kernel 不能与自己重复计时。不同内容 incumbent 必须按其自身 build 信息构建，但通过当前 trial 的 schema/API 输入约定调用。

**BENCH-F-07 — 报告中的 baseline 标识。** `WorkloadRecord.baseline` 必须标识 `TimingRecord.versus="baseline"` 的计时对象：seam 原始实现记 adapter 入口，incumbent 作为 baseline 时记其 base schema 路径，不计时为 null。`backend` 是成功绑定并完成该 workload 的实际 adapter backend，无 adapter 时为 null。

## 3. 输入生成、调用与正确性

**BENCH-F-08 — 输入生成接口。** 每次需要一组输入时必须从 schema 获取 `init`，合并常量轴和 workload 轴，将其显式声明的轴参数传入，并总是传入 `device=str(device)` 与 `seed`。`init` 必须返回 API call 字典。随后对每个声明输出分配指定设备、shape、dtype 的未初始化 tensor，覆盖 call 中同名值。shape 由 schema 输出维度及 workload 轴求得；没有 `init` 不能采用随机张量或其他隐式默认生成器。

**BENCH-F-09 — 可重现 seed。** 每组输入的 seed 必须只由 schema 名称、workload 轴和输入集名确定性导出，不依赖进程、kernel 或 commit。不同输入集的 seed 互不相同。candidate 和 reference 必须分别调用 `init` 重建对应数据。candidate 必须在输出证据中记录每组输入的内容指纹（各输入 tensor 字节的摘要），reference 重建输入后核对；不一致说明 `init` 不确定或跨设备不等价，该 workload 记 `platform_failure` 且 `is_compared=false`。

**BENCH-F-10 — kernel API 适配。** 输入必须按照 schema 声明顺序，以 `param` 重命名和 `tuple_idx` 提取规则适配；缺失可选输入保留为 None。DPS kernel 接受位置输入及随后的位置输出缓冲区；返回式 kernel 的 tensor 结果须在必要时复制到预分配输出缓冲区。bench 按非 inplace API 方式调用。计时必须包含实现实际执行的返回式适配复制等 GPU 工作，不只测最内层 kernel。

**BENCH-F-11 — 首调用和准备。** 必须识别当前设备的完整 arch/hardware，并检查候选 build 目标；未知设备为平台故障，不匹配为 `no_matching_device`。正确性 capture 和计时之前，必须对每个可执行的 workload 用计时输入执行一次候选并同步，完成首调用编译及惰性初始化；baseline 不可用的 workload 不执行候选。此时 incumbent 失败归平台故障。

**BENCH-F-12 — 正确性检查类别。** 每个 workload 必须执行以下四类检查，每次 capture 后 replay 才取结果。前两类在性能测量前，后两类在测量后；不计时的 workload 也执行后两类。

1. **新 capture**：新分配输入与输出，新 capture 并 replay，覆盖多组独立输入。
2. **旧图 replay**：在同一组输入地址上替换数值后 replay 首次 capture 的 graph，输出缓冲区复用。
3. **计时后检查**：在性能测量用的输入地址上替换数值，新 capture 并 replay。
4. **计时后旧图 replay**：同一计时输入地址再替换数值后，replay 第 3 类的 graph。

替换输入数值时必须保持 tensor 地址和版本计数不变，以检验依赖地址、版本或历史值的非法缓存。第 3、4 类输入必须与第 1、2 类及计时输入集独立。

**BENCH-F-13 — 输出预填毒值。** 对使用新输出的 capture，必须在捕获图中先填充输出，再调用候选：元素宽度大于 1 字节的浮点 tensor 填 NaN，其余 tensor 的每个字节填 `0xFF`。这样 replay 同一图时也重新填毒；复用输出的步骤不额外在图外填毒。此检查用于暴露输出未完全写入。

**BENCH-F-14 — capture 安全性。** 正确性图必须使用全局 capture 错误模式。预热完成后，正确性 capture 期间不得触发任何受监控的 JIT 编译入口；即便候选吞掉该编译异常，也必须判为 `capture_unsafe`。无法建立编译监控本身是平台故障。capture/replay 异常，包括这些局部区域中的非普通 `BaseException`，须转为对应失败记录。

**BENCH-F-15 — 输出结构和输入不变性。** 每个成功 replay 后必须验证所有输出为声明 shape、dtype 的 tensor。必须比较 call 中全部非输出 tensor（含 tuple、list、dict 内嵌套的 tensor）的对象身份及内容字节，不能只使用浮点近似相等；NaN payload、正负零的变化也算输入改变。出现输入变化产生 `input_mutation`。检查成功的输出必须复制为独立、连续的 CPU tensor 快照，避免后续 replay 改写已取证结果。

**BENCH-F-16 — 输出证据。** 每个 workload 的输出证据必须写成非 pickle、可安全解析的文件，在批内唯一对应该 workload，能按步骤和输出名定位 tensor。即使 workload 没有成功产出 tensor，也必须生成空证据。允许保存在失败前完成的连续步骤前缀，用于回溯先前的错误；不允许跳过前面缺失的输出而保留后续步骤，也不允许无关额外内容。

**BENCH-F-17 — reference 执行。** 候选任务结束并清理其进程后，reference 任务必须读取输出证据，为每个已存在步骤生成相应输入和 reference 期望输出。调用 schema `reference` 时，只传其签名中明确出现的 API call 参数，并禁用 autograd。单返回值按单输出处理，tuple/list 按 schema 输出顺序处理；输出数量须严格一致，期望值复制到 CPU。reference 的计算精度和设备转换由 schema reference 自己定义，bench 不自动把输入或 reference 转成 float64。

**BENCH-F-18 — reference 可比较性。** 在校验候选前，必须把 reference 输出转换为声明输出 dtype，并验证：原 reference 有限的位置在转换后仍有限；schema `check(reference_outputs, rounded_reference_outputs)` 返回真。任一不满足，应记 `platform_failure` 并将该 workload 的 `is_compared` 设为 false。不可比较的 reference 不能作为候选不正确的证据。

**BENCH-F-19 — 数值判定。** 对每个已产出步骤，先再次校验证据中的输出 shape/dtype，再按 schema 输出顺序调用 `check`，参数均为 CPU tensor 列表。`check` 为真才算该步骤数值通过。第 1 类检查失败为 `numeric_mismatch`；第 2 类失败为 `capture_unsafe`；第 3、4 类失败为 `post_timing_mismatch`。reference 自身不会被 seam baseline 或 incumbent 的结果替代；此阶段不独立检验对照实现的数值正确性。

**BENCH-F-20 — 容差。** 必须使用 schema 自带 `check`；bench 不得额外覆盖容差。schema 未指定 check 时使用 `aklr/core/schema` 提供的默认 check。

**BENCH-F-21 — reference 结果的完整性。** `is_complete` 只表示全部步骤的所有输出是否存在，不能代替正确性；`is_compared` 初始表示存在可校验步骤，reference 不可比较时变为 false，候选数值失败不会将它变为 false。同一检查类别只保留最早步骤的 finding，按步骤顺序输出；文件级错误在步骤错误之前。若没有候选 finding 解释缺失步骤，最终还须产生 `candidate_crash`。

## 4. 计时和统计判定

**BENCH-F-22 — 计时输入与参与者。** 对每个 workload 必须另生成 `timing` 输入供 candidate、seam baseline 和可选 incumbent 共同使用。每轮对照与候选必须在同一任务、同一设备上运行，使用同一计时输入地址、同一 timer 和同样的冷缓存政策。存在 baseline 才执行计时。不能缓存某次候选的好结果跨 workload 使用，也不能用 reference 时间代替 baseline。前置图检查只产出证据，不等待独立 reference 判定，因此数值错误的候选也可能已经被计时。

**BENCH-F-23 — 图外层重复。** 【协议】先执行 1 个不记录的 warmup round，再执行 32 个正式 round。每个 round 每个参与者调用一次 `bench_cudagraph(fn, rep=20.0)`，仅支持 `mode="graph"`。计时期间禁用 Python 循环 GC，结束时恢复原状态。warmup round 中每个参与者调用后必须检查计时输入的身份与字节未变：候选改变输入记 `input_mutation`，对照改变输入记 `platform_failure`，该 workload 不再计时。

**BENCH-F-24 — 轮内公平顺序。** 每轮必须对全部参与者做一次均匀随机排列，排列由 workload 与轮次确定性导出。不能让候选总在第一或最后，也不能分别采样不对应的 candidate/baseline 轮次。

**BENCH-F-25 — 冷缓存。** 【协议】每次 timer 中的 warmup、估算和图内调用前必须通过读取 1 GiB 的 int32 GPU 数据刷新 L2/MALL，并产生缩减结果，防止读操作被消除。不能以写满缓存替代读取，否则脏行回写会污染后续耗时。刷新操作保留在 graph 内，但不计入最终被测实现耗时。

**BENCH-F-26 — timer 准备。** timer 必须在同一设备专用 stream 上完成 eager warmup 与 capture，以支持按 stream 初始化的后端状态；该 stream 须等待调用方当前 stream。timer 先用带 flush 的 eager 调用估算单次耗时 `estimate_ms`，仅用于决定图内调用数，不作为最终耗时。

**BENCH-F-27 — 单调用图检查与采样数。** 必须先捕获含 1 次 flush 和 1 次被测调用的 graph，多次 replay，确认每次 kernel 身份/数量多重集相同、总记录数至少 2，且最早开始的 flush kernel 身份恰出现一次。`rep` 必须有限且为正。【协议】设每次调用包含 flush 的 dispatch 数为 `K`，正式图内调用数为 `max(1, min(floor(20.0/estimate_ms), floor(2048/(12*K))))`。

**BENCH-F-28 — 图内重复和 GPU 指标。** 正式 graph 必须依序包含上述调用数的 flush+subject。每个测量 replay 必须有与单调用图按调用数放大的 kernel 多重集，且每条 dispatch 的结束时间大于开始时间。每个 replay 的样本为“除 flush 身份外全部 kernel 的持续时间之和 / 图内调用数”，由纳秒转毫秒。该指标是 GPU dispatch duration sum，多 stream 重叠仍累加，不是 host 时间、整图墙钟时间或关键路径时间。【协议】timer 返回 10 个测量 replay 样本的中位数。

**BENCH-F-29 — profiler 完整性。** 必须在第一次 GPU 操作前配置 ROCm kernel tracer，在 replay 记录区间外完成必要同步。允许 profiler 记录异步到达并有限时等待；完整条件为预期记录总数，单调用探测则要求非空且每次 replay 的身份计数一致。记录不完整、调用间 dispatch 不稳定、无 subject GPU 工作、非法 duration、capture 错误等不能被转换成虚构的有效耗时。

**BENCH-F-30 — 原始轮次。** 正式测量正常输出恰为 32 轮；warmup 不进入统计。最终 BenchReport 保留各比较按轮次配对的对照/候选毫秒数组。

**BENCH-F-31 — 统计量。** 对每个比较对象以轮次配对，定义 `x_i = ln(t_versus_i / t_candidate_i)`。门槛用 `y_i = clamp(x_i, -ln(2), ln(2))`。对至少 2 个样本，单侧 95% 几何均值速度比下界为 `L(z) = exp(mean(z) - t_(n-1,0.95) * sample_stddev(z)/sqrt(n))`，sample_stddev 使用 n-1 自由度。`TimingRecord.ratio_lower_bound=L(y)`，`log_ratio_mean=mean(x)`，`min_ratio=exp(min(x))`。原始耗时必须保留，不可为抑制离群值而截断原始记录。

**BENCH-F-32 — workload 性能 finding。** 某 retain 比较下界严格小于 0.99 时产生 `regressed`；否则任一 entry 比较下界严格小于 1.02 时产生 `not_faster`；等于门槛不产生性能 finding。多个失败比较合并为一个 finding，列出模式、对象、下界与门槛；retain 失败优先。样本过少、缺少对照时间、非有限时间或统计计算错误，应返回空 timings 及 `candidate_crash`。不能把无效样本当作性能通过。

**BENCH-F-33 — 展示及回写 speedup。** workload 的 speedup 必须重新用未截断的 `x_i` 算 graph/baseline 比较的 `L(x)`，四舍五入至 3 位小数并下限截为 0.001。无 baseline 且批未中止时 speedup 为 1.0；否则缺少 graph/baseline 数据时为 null。该数值不同于表格内用于门槛的截断下界，不能相互替代。

## 5. 进程、调度及资源要求

**BENCH-N-01 — 批处理。** 必须按 schema 分批，批内保持 workload 顺序，不混合 schema。一个 workload 的全部步骤必须在同一批内。

**BENCH-N-02 — 调度。** 必须支持多个设备并行处理；默认每个设备资源同一时刻只租给一个任务。最终 records 顺序必须恢复到原 schema/批/workload 顺序，而不是完成顺序。

**BENCH-N-03 — 隔离单位。** candidate 和 reference 必须各自在全新 spawn 子进程中运行，每个进程只执行一个任务，不能复用候选污染过的 Python/编译器/设备状态进行 reference 校验。reference 必须等待该批 candidate 完成，但不要求两任务使用同一设备。实际 bench 的 capture/tracer/计时要求可用 ROCm GPU。

**BENCH-N-04 — 数据跨进程边界。** bench 不把 GPU tensor 借给另一个进程，也不依赖 CUDA IPC 中的 tensor 生命周期。candidate/reference 分别在自身设备生成输入，candidate 的 CPU 输出证据通过 BENCH-F-16 的文件传给 reference。候选不应获得由 reference 计算的期望输出。

**BENCH-N-05 — 超时。** 候选任务必须有随批内 workload 数增长的总执行预算，覆盖其 baseline 准备、build、首调用、检查、计时和输出保存。reference 任务有独立预算。待排队时间不计入任务执行预算。

**BENCH-N-06 — 内存。** 候选 build 和所有首调用结束后，若首调用新增的 GPU allocation 峰值或相对 build 前的常驻 allocation 增量超过 16 GiB，必须报批级 `build_failed`。该限制不是 GPU 总占用上限。

**BENCH-N-07 — 启动与健康度。** 每个子进程须在独立临时任务目录中运行并设置当前设备。GPU 启动失败或无法回收的进程必须使对应资源退出可用池。仅在任务尚未运行、启动失败且原资源不健康时，允许在其他可匹配健康资源上再试一次；任务已运行后的崩溃、超时、数值失败不得自动重试以挑选成功样本。

**BENCH-N-08 — 清理与取消。** 任务结果对父进程可见前必须完成子进程关闭和资源释放。每个子进程及其子进程组成独立进程组；关闭时须结束并回收该组。回收失败按错误报告并隔离资源。成功 candidate 的证据保留到 reference 完成；阶段结束、中断或收到 SIGINT/SIGTERM 时必须终止子进程并清理临时目录。

**BENCH-N-09 — 协议可信边界。** 父进程向子进程发送的受信命令可以包含可 pickle 的调用对象和参数；子进程回传必须为 JSON，父进程不得反序列化候选提供的 pickle。单条返回消息必须有大小上限，必须验证消息结构与 task 身份。返回的 workload 标识集合必须与请求完全相同，防止缺项、多项或冒充其他 workload。

**BENCH-N-10 — 文件可信边界。** reference 必须拒绝候选输出目录的符号链接，基于已打开目录读取证据；输出文件必须拒绝符号链接和非普通文件，且不能因 FIFO 阻塞。只能从证据文件读取张量数据，必须检查允许的键、shape、dtype。父进程不能直接信任候选 JSON 自述正确性；正确性由独立 reference 的结果决定。

**BENCH-N-11 — 日志。** 子进程 stdout/stderr 必须汇入大小受限、仅属主可读写的任务日志。进度渲染失败不得丢失已有 records。

**BENCH-N-12 — 安全保证范围。** schema 的 `init/reference/check` 是可信仓库源码；kernel 源码及其结果在可信度上较低。上述措施不是 OS 安全沙箱，不能对任意恶意 Python/native kernel 承诺不可篡改宿主、其他任务、计时数据或自身检查逻辑。

## 6. 输出、接纳与 evaluation 回写

**BENCH-O-01 — 候选结果。** 局部步骤失败只停止该 workload，其他 workload 可继续；批级 build/设备故障产生批 finding。计时轮次和 adapter 身份仅在该 workload 的检查/计时流程成功返回时保留。

**BENCH-O-02 — reference 结果。** candidate 存在批 finding 时不能运行 reference；无批 finding 时，仅排除带 `baseline_failed` 的 workload，其他局部失败 workload 仍应检查已保存证据。reference 任务整体失败时，该批不能沿用已获得候选 timings 假装完成验证。

**BENCH-O-03 — WorkloadRecord。** 每个实际 trial 必须产生下列字段，即使批失败。schema 在构造 trial 前失败时只产生 schema finding，没有 workload record。

| 字段 | 类型与含义 |
| --- | --- |
| `schema_path` | 仓库相对路径 |
| `axes` | 变量轴名到非负整数值的字典 |
| `kernel_content_hash` | 候选 kernel 内容 hash |
| `baseline` | BENCH-F-07 的展示标识或 null |
| `backend` | 实际完成评测的 adapter backend 或 null |
| `is_timed` | trial 是否要求 baseline 比较；表示计时计划，不保证已有计时结果 |
| `is_compared` | reference 结果是否形成比较；不是 accuracy passed 标志 |
| `timings` | 0 个或多个 TimingRecord |
| `batch_stopped` | 批级失败代替本 workload 自身判定时为 true |

**BENCH-O-04 — TimingRecord。** 必须输出 `mode="graph"`、`versus="baseline"|"incumbent"`、`threshold="entry"|"retain"`、`n_rounds`、原始 `log_ratio_mean`、截断后的 `ratio_lower_bound`、原始 `min_ratio`、`versus_milliseconds` 和 `candidate_milliseconds`。数组按配对轮次排列，全部单位为毫秒。reference 数值 finding 不必删除有效 timings；候选局部 finding 或批级故障则不生成最终 timing comparison。

**BENCH-O-05 — Finding。** 每条必须包含 `code`、可空 `schema_path`、可空 `axes`、`detail`。批 finding 必须复制定位到批内每个 workload，并将其 record 标为 `batch_stopped`；同一 workload 同时有 reference 与 candidate findings 时，reference findings 排在前面，性能 finding 后加。不能用省略记录代替失败。

**BENCH-O-06 — BenchReport。** 必须包含 `stage_name="bench"`、`status`、`detail`、决定阶段状态的 `findings`、`elapsed_seconds`、全部 `workload_records` 和 `demotions`。status 取 `passed/failed/skipped/aborted`；有候选责任的 deciding finding 优先得到 failed，否则有平台责任 deciding finding 得到 aborted，否则 passed。

**BENCH-O-07 — workload 降级。** 有对应 WorkloadRecord、且 axes 非空值（允许空字典）的下列 finding 必须转入 demotions，不直接决定整个阶段状态：`numeric_mismatch`、`post_timing_mismatch`、`input_mutation`、`capture_unsafe`、`build_failed`、`candidate_crash`、`candidate_timeout`、`not_faster`、`regressed`。前七类回写 accuracy failed，后两类回写 accuracy passed。平台故障、`baseline_failed` 等其他 finding 保留在 deciding findings。

**BENCH-O-08 — schema 接纳。** 每个产生 records 的 schema 必须至少有一个 workload 同时满足：`is_compared=true`；没有任何 finding（含 `not_faster`、`regressed`）；`batch_stopped=false`。性能要求完全由 BENCH-F-06 的门槛和 BENCH-F-32 的性能 finding 表达，接纳不另设门槛：首次发布或同内容 kernel 只需不慢于 seam（retain），其他情况须快于 seam（entry）且不慢于不同内容的 incumbent（retain），无 baseline 时不要求性能。未达到且不存在涉及该 schema 或全局的 deciding finding 时，新增 schema 级 `no_qualifying_workload`。

**BENCH-O-09 — verdict 与 accuracy 统计。** workload verdict 取 deciding findings 与 demotions 合并序列中的首条匹配 finding，无 finding 为 passed，demotion 须可与 deciding finding 区分。accuracy 统计中 failed 排除 batch_stopped，ok 为非该类 failed 且 is_compared；批级失败不冒充已逐项测得的 accuracy failure。

**BENCH-O-10 — 生成 evaluation 的完整性。** 对一个 schema，只要存在除 `no_qualifying_workload` 外的该 schema deciding finding，或任一声明 workload 缺 record，就不得产生整份 measurements。accuracy passed 的 workload 还必须同时有非空 speedup 与 `is_compared=true`；accuracy failed 的 workload 可在没有 timings 时记录失败。其他 schema 或整个 pipeline 的失败不能禁止本 schema 写出可信 measurements。

**BENCH-O-11 — Evaluation 数据格式。** 每个 workload 按 axes 对应一个 `Evaluation`：`hardware` 为规范 `arch/hardware`，`accuracy` 为 passed/failed，`speedup` 为 backend 键到 `{commit, value}` 的字典。commit 使用当前 scope 的 head SHA；value 使用 BENCH-F-33 的 speedup。仅当 backend 非 null、计划计时且 speedup 非 null 时添加 speedup 条目，否则为空字典。不要把 incumbent 路径作为 backend，也不要为无 adapter 的比较虚构 backend。

**BENCH-O-12 — 回写策略。** 回写由 pipeline 报告接口完成，bench 本身不覆盖源 schema。只有当前平台 target 完整、schema 有 kernel 且支持该 target 时，才考虑生成更新文件。无已有 target evaluation、accuracy 改变、backend 键集合改变，或任一 speedup 相对原值变化严格大于 2%，视为需要更新；仅 commit 改变或速度变化在 2% 内不重写。某 workload 触发更新时，必须给整个 schema 的所有 workload 替换该 target evaluation，保留其他硬件 evaluation，并按 hardware 排序。

**BENCH-O-13 — 产物和退出码。** CLI 必须输出 `<out>/report.json`、`<out>/report.md` 以及 `<out>/schemas/<原相对路径>` 下发生变化的 schema JSON；写 schema 前清空该输出子目录，防止混入旧结果，不能清空源目录。pipeline 报告 version 为 2，包含 head/base、平台软件与硬件版本、schema/bench 路径、各阶段报告和 measurement profile（当前 version 18，【协议】条款变动时必须升版）。退出码：通过为 0、候选失败为 1、平台中止为 3；failed 优先于 aborted。空 scope 或被跳过且无决定性失败的 pipeline 可返回 0。

## 7. 失败模式与归因

**BENCH-E-01 — 归因原则。** 必须区分“任务里的候选出了问题”和“承载任务的平台出了问题”。`platform_failure` 与 `no_matching_device` 是平台责任；其他定义的 deciding finding 属候选责任。按“谁的代码抛出”归因：执行候选代码时抛出的任何异常（含 `OSError`）归候选；bench 自身的文件、系统与执行基础设施操作（资源、启动、通道、清理、保存证据）失败归平台；执行 baseline、incumbent 或 reference 时的异常归平台。reference 读取候选证据时，内容或文件形态无效（缺失、符号链接、非普通文件、格式错误）归候选，底层 I/O 错误归平台。任务内崩溃、超时归候选。清理或资源释放故障不得静默丢弃。

**BENCH-E-02 — 失败映射。** 必须至少满足下表。范围“批”表示批内全部 trial 都得到同类定位 finding；“schema”表示没有 axes；“workload”表示可继续其他 workload。

| 触发条件 | 对外结果 | 范围及后续 |
| --- | --- | --- |
| 空设备序列、重复设备或非法执行设备格式 | ValueError；CLI 参数错误 | 不启动 bench |
| 空 bench_paths | skipped | 无 records |
| 当前 schema/轴/输出形状等在父进程准备时异常 | stage aborted | 顶层兜底，可能没有已收集 records |
| 候选同名 base 文件无法解析 | platform_failure | schema，其他 schema 可继续 |
| 当前 GPU arch/hardware 不完整 | platform_failure | 批 |
| 候选不支持当前硬件 | no_matching_device | 批 |
| 建立编译监控失败 | platform_failure | 批 |
| 有 adapter 但无任何绑定且没有解析异常 | baseline_failed | workload，不执行它的候选/reference |
| baseline 导入、绑定或预调用异常 | platform_failure，除明确的局部失败代码外 | workload；批其余 workload 可继续 |
| 候选 build/首调用异常或首调用内存超限 | build_failed | 批 |
| incumbent build/首调用异常 | platform_failure | 批 |
| 正确性 capture/replay、受监控 JIT 编译异常 | capture_unsafe | workload |
| 输出非声明 tensor/shape/dtype | candidate_crash | workload |
| 顶层输入 tensor 身份或内容改变 | input_mutation | workload |
| 计时中候选抛出异常 | candidate_crash | workload |
| 计时中 baseline/incumbent 抛出异常，或 profiler/timer 自身故障 | platform_failure | workload |
| warmup round 后计时输入被改变 | 候选：input_mutation；对照：platform_failure | workload，不再计时 |
| candidate 保存输出证据失败 | platform_failure | 批 |
| candidate 返回类型、axes 集合或计时轮次结构非法 | candidate_crash | 批 |
| candidate 执行超时 | candidate_timeout | 批，杀进程组 |
| candidate 进程提前退出、segfault、明确的任务失败、运行期坏消息/截断消息 | candidate_crash | 批 |
| 无健康匹配资源、执行器关闭、提交失败、启动失败、其他执行基础设施异常 | platform_failure | 批；启动失败最多按 BENCH-N-07 重试 |
| reference 任务执行失败、超时、退出、非法结果或没有结果 | platform_failure | 批，绝不把 reference 超时记为候选超时 |
| reference 无法打开候选文件，包括缺文件、符号链接、非普通文件、无效证据格式 | candidate_crash | workload，is_compared/is_complete 为 false |
| 文件键有断档后输出、部分步骤的零散输出或额外键 | candidate_crash | workload，is_compared/is_complete 为 false |
| reference 读取候选证据时底层 I/O 错误 | platform_failure | workload，is_compared=false |
| reference 读取某步骤 shape/dtype 不符 | candidate_crash | workload，可检查其他步骤 |
| reference 重建输入的指纹与候选记录不一致 | platform_failure | workload，is_compared=false |
| reference 生成/返回数量/类型或 check 自身抛出未处理异常 | platform_failure | 批 |
| reference 转声明 dtype 溢出为非有限值或自身 check 失败 | platform_failure | workload，is_compared=false |
| 候选 check 为 false | numeric_mismatch / capture_unsafe / post_timing_mismatch | workload，按步骤类别保留最早错误 |
| 候选无 finding 但步骤缺失 | candidate_crash | workload |
| 耗时统计不能使用 | candidate_crash | workload，timings 为空 |
| 合法统计低于 entry/retain | not_faster / regressed | workload，作为 demotion |
| schema 没有合格 workload | no_qualifying_workload | schema，除已有 deciding 故障解释外 |
| 其他未处理的阶段普通异常 | aborted，输出 traceback | stage 顶层兜底 |

**BENCH-E-03 — reference 故障优先级。** 一旦 reference 整任务未产出有效结果，其平台失败必须覆盖该批逐项判定；不能因 candidate 进程成功而宣告准确性通过。reference 对候选文件做出的明确无效证据判断则仍归候选。局部候选失败前已产出的 reference mismatch 要保留，不能只显示后发生的 capture 异常。

## 8. 外部接口与已有验收证据

**BENCH-X-01 — 必要依赖面。** 重实现可以改变内部组织，但必须兼容以下职责边界：

| 外部接口 | bench 所需契约 |
| --- | --- |
| `validation.command`、`ValidationScope` | CLI、工作树/base 读取、路径选择和首次发布判断 |
| `PipelineStage`、`StageReport`、`PipelineReport` | 调用顺序、跳过/中止、报告序列化及 schema 输出 |
| `Schema`、`Workload`、`Kernel`、`Evaluation`、`Speedup` | 源码加载、参数/shape、内容身份、默认 check、历史可服务性、评测数据格式 |
| `DeviceTarget` | 硬件识别/匹配 |
| `SeamRegistry`、`Adapter` | op 的 adapter 顺序、原始后端函数、call 绑定、entry/backend 标识 |
| `BuilderRegistry.build(schema)`、Runnable metadata | 构建 kernel；暴露 DPS 调用约定 |
| 运行时 dispatch 的参数适配 | API 参数映射、输出缓冲区/返回式适配 |
| `bench_cudagraph`、kernel tracer | 冷缓存 graph timer、ROCm dispatch 记录 |
| `core.utils` dtype 转换 | schema dtype 字符串与 torch dtype 双方一致 |

**BENCH-X-02 — 已有端到端验收。** `tests/core/validation/test_pipeline.py` 要求：不合法阶段顺序被拒绝；audit 失败后 bench 跳过；在 gfx950 上正确 GEMM 可写回 accuracy passed；错误 GEMM 的 bench demotions 包含 numeric_mismatch。重实现须保留这些可观察结果，但不能把这几项有限测试视为覆盖所有准入/失败组合。

**BENCH-X-03 — timer 验收。** `tests/core/validation/test_bench_cudagraph.py` 要求 timer 能测量具有按 stream 惰性状态的 callable，以及派生第二 stream 再 join 的 callable，结果为正。`test_validation_scope.py` 约束 schema/core/other 路径识别。本文为静态研究规格，没有为编写文档实际运行这些 GPU 测试。

## 9. 与现有代码的差异

第 1 至 8 节已对原待澄清问题作出决议。下列各项是 `95471c5a` 代码与本规格不符之处，修复时按此改动。

**BENCH-D-01 — 接纳门槛。** 现有代码另要求 graph/baseline 截断下界严格大于 1.02，即使 workload 使用 retain 门槛；`regressed` 也不阻止 workload 合格。按 BENCH-O-08 改为“无任何 finding 即合格”。涉及 `aklr/core/validation/stages/_bench/_report.py:82`、`:122`、`:130`；首次发布的说明见 `aklr/core/validation/stages/validation_scope.py:107`。

**BENCH-D-02 — backend 与 baseline 标识。** 现有代码用第一个已注册 adapter 的 backend 判断可服务性，不检查旧 kernel 是否支持目标硬件；`WorkloadRecord.baseline` 在 base workload 可服务时优先记其路径，与计时对象不一致。按 BENCH-F-04、BENCH-F-07 修改。涉及 `aklr/core/validation/stages/_bench/_stage.py:144`、`:182`、`:233`、`:375`；`aklr/core/validation/stages/_bench/_evaluation/_candidate.py:227`、`:302`。

**BENCH-D-03 — 输入确定性。** 现有代码不核对 candidate 与 reference 重建的输入是否一致；reference 的文档字符串称 float64，实际不做精度提升。按 BENCH-F-09 增加输入指纹核对，并把文档字符串改成“精度由 schema reference 定义”。涉及 `aklr/core/validation/stages/_bench/_evaluation/_reference.py:58`、`:126`、`:130`；`aklr/core/validation/stages/_bench/_evaluation/_trials.py:162`。

**BENCH-D-04 — 输入不变性范围。** 现有代码只检查 call 顶层 tensor，计时期间不检查输入。按 BENCH-F-15 递归检查嵌套 tensor，按 BENCH-F-23 在 warmup round 后检查计时输入。涉及 `aklr/core/validation/stages/_bench/_evaluation/_candidate.py:297`、`:347`、`:391`。不引入 OS 沙箱，保护范围以 BENCH-N-12 为准。

**BENCH-D-05 — 重复设备。** 现有 CLI 不去重设备，`--devices 0,0` 会把同一 GPU 当成两个资源。按 BENCH-I-08 报参数错误。涉及 `aklr/core/validation/command.py:65`；`aklr/core/validation/stages/_bench/_execution/_resources.py:86`。超大 workload 单独成批、图内记录预算可能超出 2048、内存只在首调用后检查，均维持现状，不再作为缺陷。

**BENCH-D-06 — 失败归因。** 现有代码把包装步骤中的任何 `OSError` 归平台，包括候选代码抛出的；baseline 在 timer 中出错会让整批记 `candidate_crash`；返回的 rounds 缺键时以 KeyError 触发顶层 aborted。按 BENCH-E-01、BENCH-E-02 改为按抛出方归因，计时异常按 workload 处理，rounds 结构非法记批级 `candidate_crash`。涉及 `aklr/core/validation/stages/_bench/_evaluation/_candidate.py:78`、`:113`、`:314`；`aklr/core/validation/stages/_bench/_evaluation/_reference.py:101`；`aklr/core/validation/stages/_measure.py:186`、`:221`；`aklr/core/validation/stages/stage.py:148`。

**BENCH-D-07 — audit 拒绝 bench 不支持的 schema。** 现有 audit 不拒绝有 kernel 但 reference 为 None、输出为 scalar 或没有 workload 的 schema，bench 到运行时才失败或不产生 records。按 BENCH-I-03 由 audit 拒绝。涉及 `aklr/core/validation/stages/_audit/_stage.py:102`；schema 定义见 `aklr/core/schema/_schema.py:87`、`:89`。

**BENCH-D-08 — 不可解析的 base 文件。** 现有代码跳过同名 schema 下格式过旧的合法 JSON，只对非 JSON 报 `UnreadableIncumbent`。按 BENCH-F-03 对同名 schema 的任何不可解析文件报 `platform_failure`。涉及 `aklr/core/validation/stages/_bench/_stage.py:215`。

**BENCH-D-09 — speedup 的 commit 语义。** 回写的 commit 是完成该测量的 head SHA（BENCH-O-11），kernel 内容身份不含目标硬件（目标支持由 BENCH-F-04 单独检查），两者维持现状。`aklr/core/schema/_workload.py:25` 的 `Speedup` 注释仍写作 baseline commit/tag，需要更正；该文件在 `aklr/core/schema/` 下，按 AGENTS.md 修改前须经 zhenhuang12 批准。
