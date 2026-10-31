# Closed-Action Benchmark

This opt-in workflow adds `constrained_subtasks` to Argus labels while preserving
the normal annotation. It does not modify source data or the alignment repository.
Preparation, dry runs, export and evaluation make no model calls.

## Protocol

- Only complete templates in `lite_annotations/annotation_schema_v1.json.skills`
  are allowed. IDs are SHA-256 hashes of subject, skill and structured slots;
  free text and local numeric IDs do not define identity.
- Completion rules come from the explicitly supplied `skill_spec_v1.yaml`.
  Templates absent from that vocabulary are retained with `template_text_only`
  provenance and listed in the audit. Their boundaries require human review;
  they are not silently mapped to a different skill.
- Input is **multi-view RGB only**. No Parquet columns, task descriptions,
  episode annotations or human boundaries enter model inputs. Preparation checks
  reference eligibility separately, without copying the answers into contexts.
  Video camera names and technical metadata are retained. The catalog itself
  is a task-specific prior, not an open-vocabulary evaluation.
- V1 supports LeRobot v2-style per-episode RGB videos with synchronized constant
  frame rates, one action per template and a non-overlapping subtask track.
  Unsupported clocks, overlapping reference segments and oversized requests
  fail explicitly. Request-part stitching is not implemented for this mode.
- Keep training/reference exemplars outside the test selection. Compare input
  modalities, models, sampling and exemplar use separately from output accuracy.

## Commands

Run from the repository root. Set `DATASET` to one dataset directory, `ROOT` to
the parent containing all datasets, and `VOCAB` to the alignment repository's
`src/robo_subtask_alignment/vlm_annotation/resources/skill_spec_v1.yaml`.
Use `env -u PYTHONPATH` when the shell injects ROS packages.

```bash
uv run python -m action_benchmark audit --root "$ROOT" --vocabulary "$VOCAB" --out data/benchmark-audit.json
uv run python -m action_benchmark prepare --dataset "$DATASET" --vocabulary "$VOCAB" --episodes 0 1 --out data/benchmark-job
uv run python -m label.harness --episodes-root data/benchmark-job/episodes --out-dir data/benchmark-job/labels --cell-w 448 --dry-run
```

Preparation requires a fresh output directory. Omit `--episodes` to prepare all
source episodes. It validates each selected video's timestamps and frame count.
`manifest.json` fixes the test selection, input modality, catalog and reference
digest. A failed preparation has no completed manifest; use a fresh directory.

Only after reviewing the dry run and authorizing a budget, replace `--dry-run`
with `--max-spend 5` (USD). Configure provider credentials outside the repository.
Use the existing harness's model, sampling, concurrency and retry controls.
Do not supply example annotations or dataset-note overrides for this benchmark.

```bash
uv run python -m action_benchmark export --job data/benchmark-job --labels data/benchmark-job/labels --out data/benchmark-job/predictions.json
uv run python -m action_benchmark evaluate --job data/benchmark-job --predictions data/benchmark-job/predictions.json --out data/benchmark-job/metrics.json
```

Export refuses invalid, incomplete or changed-input records. Raw responses remain
in harness receipts. Missing prediction files are omitted from export and counted
as empty predictions, not dropped from evaluation. Invalid rows fail evaluation.

## Common Prediction Format

External systems can use the same evaluator by producing:

```json
{
  "version": 1,
  "dataset_id": "from manifest",
  "catalog_sha256": "from manifest.catalog",
  "predictor": {"name": "external-system", "input_modality": "declare actual inputs"},
  "episodes": [{
    "episode_id": 0, "frame_count": 100, "fps": 30,
    "subtasks": [{"start_frame": 0, "end_frame": 100, "action_id": "from catalog"}]
  }]
}
```

Intervals are half-open `[start_frame,end_frame)`. Null action IDs require a
nonempty `reason`, count as unknown predictions and never match human actions.
They differ from an explicitly defined `none` action template. Omitted intervals
are wrong on annotated frames. Repeated actions are not automatically merged.

The primary score is per-episode macro Segment F1 at IoU 0.5. Matching maximizes
the number of same-template one-to-one matches, then total IoU. The report also
includes F1 at 0.75, frame accuracy, start/end MAE in seconds for matches at 0.5,
match counts, and action-sequence Levenshtein distance. Frame accuracy masks
unannotated human gaps; segment metrics conservatively count **all** predicted
segments, including those in gaps. Review gaps before claiming final results.
Evaluate each dataset separately to preserve scenario/robot breakdowns; the same
folder name in different scenarios receives a different dataset ID.

Audit checks reference/template consistency and media existence; preparation
additionally verifies actual video clocks. Neither proves that human labels are
correct. Use human review for ambiguous boundaries and vocabulary-rule gaps.
