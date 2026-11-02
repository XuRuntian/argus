# Subtask 标注

保留 Argus 原有标注，额外输出动作、主体、物品及时间区间。只做标注和导出，不含评测。

输入为 LeRobot v2 的 `meta/info.json`、`meta/episodes.jsonl` 和从零开始、同步恒定帧率的 RGB 视频。准备过程不读取人工标注、任务文本或状态数据；目前每条 episode 必须能放入一个 Argus 请求。

```bash
uv sync --frozen
uv run python -m subtask_annotation prepare --dataset /path/to/dataset --out data/subtasks
uv run python -m label.harness --episodes-root data/subtasks/episodes \
  --out-dir data/subtasks/labels --cell-w 448 --max-spend 5
uv run python -m subtask_annotation export --job data/subtasks \
  --labels data/subtasks/labels --out data/subtasks/annotations.json
```

标注需设置 `OPENAI_API_KEY` 或 `OPENROUTER_API_KEYS`，模型沿用 Argus 的支持范围。`--dry-run` 只预览请求，不调用模型；预览不能导出。准备时使用新目录；标注可恢复运行。

导出按 episode 保存 FPS、帧数和 `subtasks`，每段恰好五个字段：

```json
{"start_frame": 0, "end_frame": 72, "action": "pick", "subject": "right_effector", "object": "plastic cup"}
```

区间为 `[start_frame,end_frame)`，按主动作连续覆盖整段视频。独立动作同时发生时只标主动作；共同操作可用 `both_effectors`。动作无法判断用 `action: null`，主体不明用 `unknown`，物品不明或不存在用 `object: null`。物品使用一致的简短英文名称，不要求额外 slots。

动作和主体优先使用规范中的名称；明确识别但没有合适选项时允许自由命名，使用一致的简短英文名（建议 snake_case），不为已有选项另造同义词。左右以主体自身为准。自定义名称原样导出，校验器不判断语义是否正确。

默认使用[精简规范](resources/skill_spec_v1.yaml)，包含 hold、hand_over 和复位，不包含 phase。来源与缺省规则见 [SOURCE.md](resources/SOURCE.md)。可用 `prepare --spec 文件.yaml` 替换；输出记录规范哈希。格式失败、缺失标注或输入变更会阻止导出，不自动修正答案。

采样默认保持 Argus 原设置；可在标注命令追加 `--sample-fps 2` 增加时间证据。输出记录实际采样时刻，帧号不代表模型逐帧看过视频。
