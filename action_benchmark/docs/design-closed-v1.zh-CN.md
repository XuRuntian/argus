> 历史设计记录：仅描述旧闭集模式，路径和验证状态不代表当前使用方式。当前入口见 [使用说明](../README.zh-CN.md)。

# Argus 规范动作接入与评估实现说明

本文说明本次改造做了什么、怎样实现，以及如何运行。英文命令说明见 [README.md](../README.md)。

后续执行记录见 [外部接口测试记录](PILOT-20261031.zh-CN.md) 和 [Agent 替代测试与对比结果](AGENT-PILOT-20261031.zh-CN.md)：外部接口鉴权失败后，已用两个独立 agent 跑通同一条视频的两套流程及统一评分。该成绩不是 Qwen/DashScope 成绩。下文的验证记录描述首次实现时的状态。

## 1. 目标与完成范围

目标是让 Argus 与 `robo-subtask-alignment` 的输出处于可比较的动作和时间分段维度，而不是比较两套自由文本的相似度。

本次约定：

- 每个数据集只允许使用其规范中的**完整动作模板**，不是只限制 `pick`、`place` 等动词。
- 执行手、物体、目标和其他 slots 都属于动作身份，不能任意重新组合。
- 以 `lite_annotations/annotation_schema_v1.json` 中的人工 subtask 为参考。
- 不设准确率达标门槛，不要求先做人工质量验收或通过小样本试验。
- 仅改造 Argus；不修改参考数据和 `robo-subtask-alignment` 仓库。

已经完成：动作规范接入、约束输出、数据准备、结果校验、统一导出、离线评估、测试和数据审计。

**后续已完成单条视频的 agent 替代标注与统一评分；仍无成功的外部 API 模型推理成绩。** 详见上述执行记录。

## 2. 为什么不是简单改提示词中的 action 名称

Argus 原有 `timeline` 描述细粒度动作阶段；`subgoal` 表示持久的任务结果，不等同于这里的 subtask。
例如一次规范的 `pick` 可以包含接近、对齐、抓握等阶段，不能把这些阶段各算作一个完整 pick，也不能直接把 subgoal 改名为 subtask。

因此，本次保留 Argus 原有标注输出，增加独立字段 `constrained_subtasks`。
模型同时判断完整动作模板和时间区间；导出时再由程序填入规范动作内容，不依靠事后的自由文本相似度匹配。

## 3. 数据与规范来源

本机参考数据根目录：

```text
/home/user/data/自动标注/自动标注
```

共有三类场景，每类包含星海图、智平方两组数据，合计 6 组、60 个 episode：

- `01_左右手交换`
- `02_操作顺序不同`
- `03_同质数据`

实际读取的来源如下：

| 来源 | 用途 |
|---|---|
| `lite_annotations/annotation_schema_v1.json` 的 `skills` | 提取允许的完整动作模板 |
| 同一文件的 `episode_annotation` | 准备前检查参考数据是否可评估；评估时读取人工边界和动作，不送入模型 |
| `meta/info.json` | 读取帧率、视频路径规则、RGB 相机定义 |
| `meta/episodes.jsonl` | 读取 episode 编号和长度；不使用其中的任务文本 |
| RGB 视频文件 | 模型输入，并核验实际视频时间戳和帧数 |
| 显式传入的 `skill_spec_v1.yaml` | 读取公共动作阶段及完成规则 |

公共词表的本机路径为：

```text
/home/user/robo-subtask-alignment/robo-subtask-alignment/src/robo_subtask_alignment/vlm_annotation/resources/skill_spec_v1.yaml
```

`skill_descriptions.json` 用于前期理解数据结构，当前运行时以 `annotation_schema_v1.json.skills` 为模板来源，不同时合并两套定义。

## 4. 实现位置与调用链

| 文件 | 改动或职责 |
|---|---|
| [label/action_contract.py](../../label/action_contract.py) | 模板规范化、稳定 ID、规则来源、提示词、输出校验、盲评上下文检查 |
| [label/episode.py](../../label/episode.py) | 上下文带有 `action_contract` 时，将规范动作输出要求加入原有提示词 |
| [label/harness.py](../../label/harness.py) | 请求前检查输入隔离；记录动作契约；解析回复时校验规范 subtask |
| [action_benchmark/dataset.py](../dataset.py) | 数据审计、人工参考检查、多视角 RGB 准备、生成 manifest |
| [action_benchmark/__main__.py](../__main__.py) | `audit`、`prepare`、`export`、`evaluate` 四个命令 |
| [action_benchmark/evaluate.py](../evaluate.py) | 一对一片段匹配、帧准确率、Segment F1、边界误差、编辑距离 |
| [tests/test_action_benchmark.py](../../tests/test_action_benchmark.py) | 规范、非法输出、评分及无网络端到端测试 |
| [tests/shard_times.json](../../tests/shard_times.json) | 登记新增测试文件，保持现有 CI 分片机制可用 |
| [项目 README](../README.md) | 增加流程入口和命令参数表 |

处理链路：

```text
规范中的 skills + 公共词表
             │
             ▼
动作 catalog：固定模板、规则、来源、摘要
             │
RGB 视频 ──► prepare ──► episodes/* + manifest.json
                             │
                             ▼
                    原有 label.harness
                             │
                    原有标注 + constrained_subtasks
                             │
                          export
                             │
                       predictions.json
                             │
人工 episode_annotation ──► evaluate ──► metrics.json
```

### 动作身份与规则版本

`action_id` 是规范化后的 `subject`、`skill`、`slots` 的 SHA-256。
字典键顺序不影响 ID；自由文本描述、数据集内部数字 ID 不参与动作身份。
因此左手和右手、不同物体或目标会得到不同 ID，同一动作的描述措辞变化不会改变 ID。

`catalog_sha256` 则覆盖完整 catalog，包括规则和来源信息。修改规则后，即使动作 ID 不变，catalog 摘要也会改变。
请求记录、manifest 和预测结果必须对应同一 catalog，不能混用不同规则版本的结果。

### 补齐的默认边界规则

优先使用公共词表已有的完成规则。没有公共完成规则的以下三类动作，使用本项目的 `benchmark_defaults_v1`：

| 动作 | 开始 | 结束边界 |
|---|---|---|
| `hand_over` | 开始有目的地在执行器之间交接物体 | 接收方稳定控制物体，且交出方已释放的第一帧 |
| `hold` | 前一动作结束后，开始持续稳定持物 | 保持结束或下一有目的动作开始的第一帧 |
| `return_to_initial_pose` | 开始有目的地回到初始姿态 | 达到初始姿态并停止回位的第一帧 |

结束边界采用不包含该帧的 `end_frame`。如果动作被打断或视频先结束，就在观察到的边界结束，不虚构完成。
规则来源会写入 catalog。其他未来出现、又没有公共规则的 skill 仍会被标为 `template_text_only`，不会自动映射成别的动作。

这些默认规则是本次新增的明确约定，**不是声称另一个仓库已经采用了同样规则**，也不会重写人工参考边界。

## 5. 如何避免人工答案进入模型

原始 Parquet 已有 `subtask_annotation` 等标注字段，直接按一般数据集导入可能让答案进入模型上下文。
因此第一版选择独立的 **多视角 RGB 输入**，而不是读取所有原始数据字段再尝试删除答案。

- 不读取 Parquet 中的标注、状态或动作列作为推理输入。
- 不将人工动作顺序、数量、边界或任务描述放进提示词。
- 只使用允许的完整模板集合，并明确告知模型：这是无序的候选集合，不是预期执行序列。
- 准备阶段检查人工参考的合法性，但不把这些参考片段复制到 episode 上下文。
- 请求前拒绝示例标注、任务说明覆盖、人工 subtask 上下文和请求分片。

这是避免答案泄漏的输入隔离，不是质量验收。实际输入模态会写入 manifest 和结果来源信息。

## 6. 输出格式与异常处理

模型新增输出字段示例：

```json
{
  "constrained_subtasks": [
    {"start_frame": 0, "end_frame": 87, "action_id": "catalog 中的完整 ID"},
    {"start_frame": 87, "end_frame": 100, "action_id": null, "reason": "无法匹配任何允许的模板"}
  ]
}
```

所有时间区间使用 `[start_frame, end_frame)`。帧号为整数；当前版本要求按时间排序、互不重叠，且不超出视频范围。
显式的 `action_id: null` 必须附带原因；它与词表中的合法 `none` 动作不同。

非法 ID、缺少 ID、越界、重叠等不会被静默修复。原始回复保留，但不能作为合法规范标注导出。
导出还会检查请求契约和输入证明，防止把缺少来源绑定或输入已变化的记录作为本次结果。

缺失的预测文件不会让对应 episode 从评分分母消失，而是按空预测计算。
已经存在但不合法的记录会导致导出报错，需要明确处理，不能悄悄过滤成一个更高的成绩。

## 7. 统一评估口径

参考答案仍是人工标注，另一套系统的预测不是 Argus 的真值。
两个系统分别对同一人工参考评分，使用同一个测试 episode 清单、动作 catalog 和区间约定。

| 指标 | 实现口径 |
|---|---|
| 主指标：macro Segment F1 @ IoU 0.5 | 每个 episode 算片段 F1，再取等权平均 |
| macro Segment F1 @ IoU 0.75 | 更严格的时间重叠要求，其他口径相同 |
| 帧级完整动作准确率 | 人工覆盖的帧上，完整模板 ID 是否一致；报告逐 episode 及宏观、按帧加权汇总 |
| 起止边界 MAE | 仅统计 IoU 0.5 下已匹配片段，以秒计，并保留匹配数量；无匹配时为 null |
| 动作序列编辑距离 | 对按时间排列的动作 ID 序列计算 Levenshtein 距离 |

片段匹配要求动作 ID 相同，并满足 IoU 阈值。使用一对一匹配，先最大化有效匹配数量，再最大化总 IoU。
因此重复切碎一个人工片段不会获得多次正确匹配，多段、漏段和未知段都会影响成绩。

人工未标注区间不参与帧准确率；片段指标仍保守地计入所有预测片段，包括落在人工空缺区间的片段。
每组数据单独生成报告，保留场景和机器人的分组，当前没有另做一个六组汇总命令。

### 与 robo-subtask-alignment 对接

统一预测文件的结构见 [英文说明的 Common Prediction Format](README.md#common-prediction-format)。
外部系统应将其动作转换为同一 catalog 的完整模板 ID，并将帧边界统一为半开区间；不能仅凭 skill 动词匹配。
同时填写真实的模型、输入模态等来源信息，不应把使用了状态或示例的结果描述成纯 RGB。

当前没有实现另一个仓库专用的输出转换器，也没有修改其推理流程。
已有结果可转换格式后按同一评分规则比较；如果要声称生成时的边界规则也完全相同，另一个预测器还需使用这些相同的规则定义。

## 8. 如何运行

以下命令在项目根目录执行。清除当前终端的 `PYTHONPATH` 是为了避免本机 ROS 的 Python 3.10 包干扰项目环境。
密钥应通过 shell 或秘密管理工具提供，不写入仓库。

```bash
cd /home/user/argus
unset PYTHONPATH
ROOT='/home/user/data/自动标注/自动标注'
VOCAB='/home/user/robo-subtask-alignment/robo-subtask-alignment/src/robo_subtask_alignment/vlm_annotation/resources/skill_spec_v1.yaml'
DATASET="$ROOT/01_左右手交换/星海图/Galaxea_R1_Lite_storage_object_brown_basket_qced_hardlink"
JOB='data/action_benchmark/my-run'
```

审计和准备不会调用模型。`JOB` 必须是新目录；省略 `--episodes` 会准备该数据集的全部 episode。

```bash
uv run --frozen --offline python -m action_benchmark audit \
  --root "$ROOT" --vocabulary "$VOCAB" --out data/action_benchmark/my-audit.json

uv run --frozen --offline python -m action_benchmark prepare \
  --dataset "$DATASET" --vocabulary "$VOCAB" --out "$JOB"

uv run --frozen --offline python -m label.harness \
  --episodes-root "$JOB/episodes" --out-dir "$JOB/dry-run" \
  --cell-w 448 --dry-run
```

准备目录结构：

```text
JOB/
  manifest.json                # 数据集身份、样本清单、模态、catalog、参考摘要
  episodes/episode_000000/     # 视频来源、上下文和时钟等 sidecar
  dry-run/                    # 无付费构建的请求记录
  labels/                     # 正式标注后才有真实回复
  predictions.json            # export 的统一预测结果
  metrics.json                # evaluate 的评分报告
```

正式标注会产生费用。下面只是命令示例，需先配置密钥并确认预算；每次正式调用显式指定 `--max-spend`，不要把它理解成跨多次运行或六组数据的总预算。
不要求质量预检达标；`--dry-run` 也不是必须通过的质量门槛。

```bash
uv run --frozen --offline python -m label.harness \
  --episodes-root "$JOB/episodes" --out-dir "$JOB/labels" \
  --model openai/gpt-6-astra --reasoning medium --cell-w 448 --max-spend 5
```

这里 `uv --offline` 只约束依赖获取，**不会禁止程序调用模型 API**；禁止模型调用的是前一个命令的 `--dry-run`。
模型、采样频率和并发等使用原有 harness 参数，正式结果会记录相应来源信息。

```bash
uv run --frozen --offline python -m action_benchmark export \
  --job "$JOB" --labels "$JOB/labels" --out "$JOB/predictions.json"

uv run --frozen --offline python -m action_benchmark evaluate \
  --job "$JOB" --predictions "$JOB/predictions.json" --out "$JOB/metrics.json"
```

## 9. 已有本机产物与验证记录

| 路径 | 内容 |
|---|---|
| `data/action_benchmark/audit-defaults-v1.json` | 补齐默认规则后的 6 组审计结果，均无审计错误、无剩余缺失完成规则 |
| `data/action_benchmark/prepared-defaults-v1/` | 最新规则下重新准备的 6 组、60 个 episode，按原场景和机器人目录组织 |
| `data/action_benchmark/audit.json` | 补齐默认规则前的历史审计记录 |
| `data/action_benchmark/prepared/` | 初版准备产物；其中每组曾对 1 个 episode 构建无付费请求 |
| `data/action_benchmark/smoke/` | 初版单个真实 episode 的无付费请求验证产物 |

新实验应使用 `prepared-defaults-v1` 或重新准备的目录。不要把初版的请求记录当作新版规则的结果。
`data/` 已被项目忽略，不会随代码提交；这些路径是当前机器上的产物位置。

验证记录：

- 初版完成了合成视频的“准备 → 构建请求 → 解析模拟回复 → 导出 → 评分”无网络端到端测试。
- 初版全量回归运行中 4728 项通过、5 项跳过、6 项失败；修正本次引入的问题及重新加载代码后，6 项失败全部复跑通过。这不是一次全新的全量全绿运行。
- 补齐默认规则后，运行 benchmark、帧率约束、README 命令表和测试分片相关测试，共 **41 项通过**。
- 60 个 episode 均已完成新版规则下的视频时钟和帧数验证。
- 无付费调用；模拟回复只用于测试，不是模型实际标注结果。

复现针对性检查：

```bash
env -u PYTHONPATH uv run --frozen --offline pytest -q \
  tests/test_action_benchmark.py tests/test_frame_rate_windows.py \
  tests/test_readme_current.py tests/test_sharding.py
```

## 10. 当前边界

- 仅支持当前这种 LeRobot v2 风格、每 episode 独立视频的布局。
- 第一版输入是多视角 RGB，不含状态、控制动作、深度等模态。
- 要求相机同步、恒定帧率和一致帧数；不支持任意异步视频直接按同一帧号评分。
- 每个模板一个动作、每条时间轴不重叠；不支持并行动作轨道的通用评估。
- 不支持该模式的长视频请求分片拼接；超过现有请求窗口会拒绝处理。
- 保留格式、输入证明和数据一致性检查。这些是保证比较有效的技术检查，不是质量门槛。
- 人工标注保持原样；工具统一输出与评分规则，不保证人工答案或模型预测正确。
