"""Historical comparison code, excluded from the annotation CLI."""
from pathlib import Path
from action_benchmark.dataset import read, reference
from action_benchmark.experiments.metrics import score_episode, summarize
from label.action_contract import digest, validate_segments, GENERIC_VERSION, action_id


def evaluate(job, predictions):
    manifest = read(job / 'manifest.json')
    if predictions.get('version') != 1 or predictions.get('dataset_id') != manifest['dataset_id'] or \
            predictions.get('catalog_sha256') != manifest['catalog']['catalog_sha256']:
        raise ValueError('prediction dataset/catalog/version mismatch')
    schema = read(Path(manifest['dataset_path']) / 'lite_annotations/annotation_schema_v1.json')
    if manifest.get('reference_sha256') is not None and digest(schema) != manifest['reference_sha256']:
        raise ValueError('human reference changed since preparation')
    rows = predictions['episodes']
    by_id = {r['episode_id']: r for r in rows}
    expected = {ep['episode_id'] for ep in manifest['episodes']}
    if len(by_id) != len(rows) or set(by_id) - expected:
        raise ValueError('duplicate or unexpected prediction episodes')
    scores = []
    for ep in manifest['episodes']:
        contract = {**manifest['catalog'], 'frame_count': ep['frame_count'], 'fps': ep['fps']}
        truth = reference(schema, contract, {'episode_index': ep['episode_id'], 'length': ep['frame_count']})
        predicted = by_id.get(ep['episode_id'])
        if predicted and (predicted.get('frame_count') != ep['frame_count'] or predicted.get('fps') != ep['fps']):
            raise ValueError('prediction frame clock mismatch')
        segments = validate_segments(predicted['subtasks'] if predicted else [], contract)
        extra = {}
        if contract['version'] == GENERIC_VERSION:
            reference_rows = next(r['subtasks'] for r in schema['episode_annotation'] if r['id'] == ep['episode_id'])
            skill_truth = [{**r, 'action_id': original['primary_action']['skill']}
                           for r, original in zip(truth, reference_rows)]
            extra['skill_only'] = score_episode(skill_truth, segments, ep['fps'])
            segments = [{**r, 'action_id': action_id(r['primary_action']) if r['action_id'] is not None else None}
                        for r in segments]
        scores.append({**extra, 'episode_id': ep['episode_id'], 'missing_prediction': predicted is None,
                       'provenance': predicted.get('provenance') if predicted else None,
                       **score_episode(truth, segments, ep['fps'])})
    return {'version': 1, 'dataset_id': manifest['dataset_id'],
            'reference_job_input_modality': manifest['input_modality'],
            'predictor': predictions.get('predictor', {'name': 'unknown', 'input_modality': 'unknown'}),
            'catalog_sha256': manifest['catalog']['catalog_sha256'],
            'reference_sha256': digest(schema), 'prediction_sha256': digest(predictions),
            'policy': {'intervals': 'half_open', 'matching': 'maximum_cardinality_then_iou',
                       'frame_mask': 'human_annotated_frames_only',
                       'segment_scope': 'all_predictions_including_unannotated_gaps',
                       'boundary_errors': 'matched_segments_at_iou_0.5_only'},
            'mode': manifest.get('mode', 'closed'),
            'skill_only_summary': summarize([{**r, **r['skill_only']} for r in scores]) if manifest['catalog']['version'] == GENERIC_VERSION else None,
            'summary': summarize(scores), 'episodes': scores}

