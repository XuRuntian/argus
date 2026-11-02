# Shared-spec action annotation

[中文说明](README.zh-CN.md)

Annotate RGB videos using the bundled [skill specification](resources/skill_spec_v1.yaml).
No human answers or external alignment repository are required. Compare outputs externally.
Requires Python 3.11, ffmpeg/ffprobe and LeRobot metadata with synchronized CFR RGB videos.

```bash
uv sync --frozen
uv run python -m action_benchmark prepare --dataset /path/to/dataset --out data/job
uv run python -m label.harness --episodes-root data/job/episodes --out-dir data/job/labels --cell-w 448 --max-spend 5
uv run python -m action_benchmark export --job data/job --labels data/job/labels --out data/job/annotations.json
```

Use a fresh job directory. Set `OPENAI_API_KEY` or `OPENROUTER_API_KEYS` before labeling.
Replace `--max-spend 5` with `--dry-run` to preview requests without model calls; previews cannot be exported.

Output includes the specification version, per-episode FPS/frame count, and subtasks
with half-open frame intervals and `primary_action: {subject, skill, slots}`.
Unknown actions retain a reason. Missing or invalid annotations block export.
Use the same vocabulary across annotators (`prepare --vocabulary PATH`).
Historical comparison code is archived in `experiments/`, outside the annotation CLI.
