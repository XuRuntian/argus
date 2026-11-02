"""One-target alignment SDK pilot. Run in the alignment repository's environment."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from label.atomic import write_atomic


def stage_source(dataset, out, exemplar, target):
    """Materialize two episodes without any target annotation columns or JSON answers."""
    import pyarrow.parquet as pq
    from action_benchmark.dataset import read

    out.mkdir(parents=True, exist_ok=False)
    info = read(dataset / 'meta/info.json')
    removed = [key for key in info['features'] if 'annotation' in key.casefold()]
    info['features'] = {key: value for key, value in info['features'].items() if key not in removed}
    rows = [json.loads(line) for line in (dataset / 'meta/episodes.jsonl').read_text().splitlines() if line.strip()]
    rows = [dict(row, tasks=[]) for row in rows if row['episode_index'] in (exemplar, target)]
    if len(rows) != 2:
        raise ValueError('pilot requires distinct existing exemplar and target episodes')
    info['total_episodes'] = 2
    info['total_frames'] = sum(row['length'] for row in rows)
    write_atomic(out / 'meta/info.json', info)
    (out / 'meta/episodes.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    shutil.copy2(dataset / 'meta/robo_annotation.yaml', out / 'meta/robo_annotation.yaml')
    (out / 'videos').symlink_to(dataset / 'videos', target_is_directory=True)
    dropped = set(removed)
    for index in (exemplar, target):
        rel = info['data_path'].format(episode_index=index, episode_chunk=index // info.get('chunks_size', 1000))
        table = pq.read_table(dataset / rel)
        forbidden = [key for key in table.column_names if 'annotation' in key.casefold()]
        dropped.update(forbidden)
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table.drop(forbidden), dest)
    schema = read(dataset / 'lite_annotations/annotation_schema_v1.json')
    schema['data_path'] = str(out.resolve())
    schema['episode_annotation'] = [row for row in schema['episode_annotation'] if row['id'] == exemplar]
    schema['episode_count'] = 1
    next_id = max((row.get('id', -1) for row in schema['scene']['objects']), default=-1) + 1
    for row in schema['scene']['objects']:
        if 'id' not in row:
            row['id'] = next_id
            next_id += 1
    write_atomic(out / 'lite_annotations/annotation_schema_v1.json', schema)
    write_atomic(out / 'isolation.json', {'exemplar_episode': exemplar, 'target_episode': target,
                                         'removed_columns': sorted(dropped), 'target_annotations_in_input': False})


def run(dataset, root, argus_job, exemplar=0, target=1, *, model_dir):
    from robo_subtask_alignment import (
        AlignmentOptions, AutoExemplarSelection, DinoRerankConfig, FullFlowConfig, QualityGateOptions,
        VlmOptions, VlmMediaPresentationConfig, load_task_annotation,
    )
    from robo_subtask_alignment.workflow import (
        run_conversion_stage, run_alignment_stage, run_quality_gate_stage, run_vlm_stage,
    )
    from robo_subtask_alignment.vlm_annotation.clients import GenerateJsonVlmClient
    from robo_subtask_alignment.vlm_annotation.media import build_segment_video_image_loader
    from action_benchmark.dashscope_pilot import MODEL, PilotClient
    from action_benchmark.dataset import read
    from label.action_contract import action_id

    dataset, root, argus_job = Path(dataset).resolve(), Path(root).resolve(), Path(argus_job).resolve()
    source = root / 'input'
    stage_source(dataset, source, exemplar, target)
    config = FullFlowConfig(
        annotation_path=source / 'lite_annotations/annotation_schema_v1.json', dataset_dir=source,
        output_dir=root / 'workflow',
        alignment=AlignmentOptions(exemplars=[exemplar], targets=[target], strategy='visual_proposal_physical_snap',
            exemplar_selection=AutoExemplarSelection(count=1), evaluate=False,
            dino=DinoRerankConfig(model_dir=Path(model_dir),
                                 device='cuda', dtype='auto', batch_size=8, skip_panels=True)),
        quality_gate=QualityGateOptions(episodes=[target]),
        vlm=VlmOptions(model=MODEL, episodes=[target], max_repair_attempts=0, max_images_per_segment=6,
            media_presentation=VlmMediaPresentationConfig(camera_policy='primary_role',
                representation='temporal_collage', frames_per_collage=6, grid_rows=2, grid_columns=3,
                frame_label='left_gutter')),
    )
    write_atomic(root / 'config.json', config.model_dump(mode='json'))
    conversion = run_conversion_stage(config)
    print('conversion complete', flush=True)
    alignment = run_alignment_stage(config, conversion)
    print('alignment complete', flush=True)
    quality = run_quality_gate_stage(config, alignment)
    print('packaging complete', flush=True)
    client = PilotClient(root / 'receipts', max_calls=6, cap_cny=7.5, max_tokens=4096)
    adapter = GenerateJsonVlmClient(client=client,
        image_loader=build_segment_video_image_loader(dataset_dir=source, max_images_per_segment=6),
        system_prompt='Return strict JSON only.')
    vlm = run_vlm_stage(config, quality, conversion, vlm_client=adapter)
    write_atomic(root / 'report.json', vlm.model_dump(mode='json'))
    annotation = load_task_annotation(vlm.episode_annotations_file)
    manifest = read(argus_job / 'manifest.json')
    contract = manifest['catalog']
    known = {row['action_id'] for row in contract['actions']}
    ep = next(row for row in manifest['episodes'] if row['episode_id'] == target)
    predicted = next(row for row in annotation.episode_annotation if row.id == target)
    segments = []
    for row in predicted.subtasks:
        action = row.primary_action.model_dump(mode='json')
        key = action_id(action)
        segments.append({'start_frame': row.start_frame, 'end_frame': row.end_frame,
                         'action_id': key if key in known else None,
                         'reason': None if key in known else 'output template not in benchmark catalog',
                         'primary_action': action})
    output = {'version': 1, 'dataset_id': manifest['dataset_id'], 'catalog_sha256': contract['catalog_sha256'],
              'predictor': {'name': 'robo-subtask-alignment', 'model': MODEL,
                            'input_modality': 'RGB + state + exemplar', 'exemplar_episodes': [exemplar],
                            'vlm_status': vlm.status, 'successful_vlm_segments': vlm.annotation_count,
                            'strategy': config.alignment.strategy},
              'episodes': [{**ep, 'subtasks': segments}]}
    complete = vlm.status == 'success' and vlm.failure_count == 0
    write_atomic(root / ('predictions.json' if complete else 'partial-predictions.json'), output)
    print(f'VLM status={vlm.status}, successful segments={vlm.annotation_count}; '
          f'{"predictions" if complete else "partial predictions (not a completed model run)"} saved', flush=True)
