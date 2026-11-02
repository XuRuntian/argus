# 按统一规范标注动作

输入视频，输出动作及时间段。使用内置 [skill_spec_v1.yaml](resources/skill_spec_v1.yaml)，不读取人工答案，不依赖其他仓库；结果对比在外部进行。

需要 Python 3.11、ffmpeg/ffprobe。输入目录包含 LeRobot 的 `meta/info.json`、`meta/episodes.jsonl` 和同步、恒定帧率的 RGB 视频。

```bash
uv sync --frozen
DATASET=/path/to/dataset
JOB=data/annotation-job  # 使用新目录

uv run python -m action_benchmark prepare --dataset "$DATASET" --out "$JOB"
uv run python -m label.harness --episodes-root "$JOB/episodes" --out-dir "$JOB/labels" --cell-w 448 --max-spend 5
uv run python -m action_benchmark export --job "$JOB" --labels "$JOB/labels" --out "$JOB/annotations.json"
```

标注前设置 `OPENAI_API_KEY` 或 `OPENROUTER_API_KEYS`；`--max-spend 5` 为 5 美元上限。只预览请求时改用 `--dry-run`，预览结果不能导出。

`annotations.json` 按 episode 保存 `fps`、`frame_count` 和 `subtasks`。每段包含半开帧区间 `[start_frame, end_frame)`，以及 `primary_action` 的 `subject`、`skill`、`slots`；无法判断的片段保留原因。phase 用于理解动作边界，不单独输出为 subtask。缺失或非法标注会阻止导出。

所有标注方应使用同一版规范；可用 `prepare --vocabulary 文件路径` 指定。历史对比代码保留在 `experiments/`，不属于当前标注流程。
