# 两套标注流程的 Agent 替代测试

## 结论

2026-10-31 按用户要求，用两个独立的 Codex 子 agent 代替不可用的外部模型接口，完成同一条目标视频的两套标注流程及统一评分。

**两条链路均已跑通。这是 agent 替代测试，不是 Qwen/DashScope 成绩，也不是整体质量结论。** 之前接口返回的是 HTTP 401 `invalid_api_key`，不能据此断言模型本身无效。

## 范围与隔离

- 数据集：`02_操作顺序不同/智平方/AI2_Alphabot_2_store_toys_0_qced_hardlink`。
- 目标 episode 1：650 帧，30 FPS；人工参考有 6 个 subtask。
- alignment 使用 episode 0 作为原有流程的示例；没有向任一 agent 提供目标 episode 1 的人工标注。
- 两个 agent 均从空会话启动，不继承主会话；只允许读取各自导出的原始请求和图像，独立写出 JSON。此隔离依靠会话和访问指令，不是文件系统沙箱。
- Argus agent 看了原请求的全部 14 张图像；alignment agent 看了 6 次片段请求各自的 1 张时序拼图。均确认成功读取。
- alignment 的六次请求由同一个 agent 会话完成，要求逐请求独立判断，但不等同于六个完全无状态的服务端调用。
- 复用上一次 API 请求中保存的提示词和图像，不重新编写提示，不增加人工答案或额外图像。未再次调用 DashScope。

## 如何接回原流程

代码见 [agent_pilot.py](agent_pilot.py)。

1. 将已保存请求中的 base64 图像原样解码，保留消息和图像顺序，交给两个 agent。回答保存在各自的 `reply.json` / `replies/`。
2. Argus：把回答接回 `label.harness.read_reply`，保留完整原生标注，同时验证 `constrained_subtasks`；再用现有 `export` 校验输入证明、完整性和动作规范。
3. alignment：复用已完成的转换、DINO/状态对齐和 `alignment_only_v2` 打包结果，以 SDK 的 `run_vlm_annotation` 重跑模型响应后的原生处理。`ReplyReplay` 按**完整提示词精确匹配**回答，未知或重复请求会拒绝，不靠调用顺序猜测。
4. 两者均转换为同一 manifest/catalog 下的 `[start_frame, end_frame)` 和 `action_id`，调用现有 `evaluate`，与目标人工标注评分。
5. 输出标记为 `codex-agent-substitute` / `agent_substitution`。Argus 保留的 `annotation_policy` 是原请求配置，其中 Qwen 名称及 `reasoning=none` 不是本次实际推理参数；实际 agent 的具体模型参数未获取，顶层不宣称该 reasoning 设置。

没有根据人工答案修改 agent 回答，没有补做质量门槛，没有修改另一仓库或参考标注。

## 校验与结果

| 项目 | Argus | robo-subtask-alignment |
| --- | ---: | ---: |
| 原生响应校验 | 通过，无问题 | 6/6 通过，无修复 |
| 预测 subtask 数 | 10 | 6 |
| 无法匹配模板的显式 null 段 | 4 | 0 |
| 越界的非空 action_id | 0 | 0 |
| 帧级动作准确率 | 45.85%（298/650） | 76.46%（497/650） |
| Segment F1，IoU ≥ 0.5 | 0.3750 | 0.8333 |
| Segment F1，IoU ≥ 0.75 | 0.1250 | 0.3333 |
| 动作序列编辑距离 | 6 | 0 |
| 匹配段起点 MAE | 0.856 秒（3 对） | 0.673 秒（5 对） |
| 匹配段终点 MAE | 0.389 秒（3 对） | 0.680 秒（5 对） |

评分规则完全相同：帧准确率仅计算人工标注覆盖的帧；所有预测段参与 segment 评分；同 action 下先最大化匹配数量，再最大化 IoU；边界误差只统计 IoU ≥ 0.5 的匹配段。不同匹配数量的边界 MAE 不宜孤立排名。

Argus 的 4 个 null 段描述初始等待或空夹爪回撤，附有原因；不是新增的规范动作，也没有在评分时忽略。其余 6 段均使用目录中的完整动作模板。

**比较范围：统一输出与评分，不是相同输入的模型对决。** Argus 从整段 RGB 判断动作和边界；alignment 使用 RGB、状态和示例，模型请求已固定 subtask 动作和边界，只补充 phase 等字段。因此 alignment 本次 subtask 分数主要反映前面的对齐结果，不能归功于替代 agent 的 phase 判断；本次也没有给 phase 准确率打分。一个目标视频不足以说明总体优劣。

## 结果位置与复现

所有数据产物位于以下目录（被 Git 忽略）：

```text
data/action_benchmark/pilot-20261031/agent-substitute/
  argus/requests/call_001/         原始消息与解码图像
  argus/reply.json                agent 原始回答
  argus/labels/                   原生解析记录与请求来源
  argus/validation.json           原生校验结果
  argus/predictions.json          统一格式输出
  argus/metrics.json              统一评分
  alignment/requests/call_*/      六次原始请求与图像
  alignment/replies/call_*.json    六份 agent 原始回答
  alignment/workflow/             SDK 原生处理产物
  alignment/report.json          6/6 成功报告
  alignment/predictions.json      统一格式输出
  alignment/metrics.json          统一评分
```

在 Argus 根目录，已有回答可重新解析及评分，无外部模型调用：

```bash
env -u PYTHONPATH uv run --frozen --offline python -c \
  "from action_benchmark.agent_pilot import replay_argus; replay_argus('data/action_benchmark/pilot-20261031')"
```

alignment 原生重放需要其虚拟环境，`alignment/workflow` 必须尚不存在，避免覆盖历史产物：

```bash
env -u PYTHONPATH /home/user/robo-subtask-alignment/robo-subtask-alignment/.venv/bin/python -c \
  "from action_benchmark.agent_pilot import replay_alignment; replay_alignment('data/action_benchmark/pilot-20261031')"
```

仅重新评分可直接调用已有 `action_benchmark evaluate`，不需要重跑模型或 SDK。

验证命令：

```bash
env -u PYTHONPATH uv run --frozen --offline pytest -q \
  tests/test_action_benchmark.py tests/test_frame_rate_windows.py \
  tests/test_readme_current.py tests/test_sharding.py
```

结果：43 passed。包括新增的精确提示词绑定、未知请求、重复消费和重复提示词拒绝测试；不是本轮重新执行了整个仓库测试集。
