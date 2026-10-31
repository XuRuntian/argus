"""Audit, prepare, export and evaluate a closed-action benchmark without paid calls."""
from __future__ import annotations

import argparse
from pathlib import Path

from action_benchmark.dataset import audit, prepare, read, reference
from action_benchmark.evaluate import score_episode, summarize
from label.action_contract import digest, validate_segments
from label.atomic import write_atomic


def export(job, labels):
    from label.harness import annotation_complete
    manifest = read(job / 'manifest.json')
    templates = {r['action_id']: r['primary_action'] for r in manifest['catalog']['actions']}
    episodes = []
    for ep in manifest['episodes']:
        path = labels / (ep['directory'] + '.json')
        if not path.exists():
            continue
        record = read(path)
        if not annotation_complete(record):
            raise ValueError(f'{path}: incomplete annotation; raw reply retained, not exported')
        contract = {**manifest['catalog'], 'frame_count': ep['frame_count'], 'fps': ep['fps']}
        if record.get('action_contract') != contract:
            raise ValueError(f'{path}: request contract differs from manifest')
        from label.harness import saved_input_problem
        if not isinstance(record.get('input_identity'), dict) or not record['input_identity'].get('source_proof'):
            raise ValueError(f'{path}: missing benchmark input proof')
        problem = saved_input_problem(record, job / 'episodes' / ep['directory'])
        if problem:
            raise ValueError(f'{path}: {problem}')
        rows = validate_segments(record['labels'].get('constrained_subtasks'), contract)
        episodes.append({**ep, 'provenance': {key: record.get(key) for key in
                         ('model', 'model_served', 'reasoning_effort', 'annotation_policy', 'sampling')},
                         'subtasks': [{**r, 'primary_action': templates.get(r['action_id'])} for r in rows]})
    return {'version': 1, 'dataset_id': manifest['dataset_id'],
            'predictor': {'name': 'argus', 'input_modality': manifest['input_modality']},
            'catalog_sha256': manifest['catalog']['catalog_sha256'], 'episodes': episodes}


def evaluate(job, predictions):
    manifest = read(job / 'manifest.json')
    if predictions.get('version') != 1 or predictions.get('dataset_id') != manifest['dataset_id'] or \
            predictions.get('catalog_sha256') != manifest['catalog']['catalog_sha256']:
        raise ValueError('prediction dataset/catalog/version mismatch')
    schema = read(Path(manifest['dataset_path']) / 'lite_annotations/annotation_schema_v1.json')
    if digest(schema) != manifest['reference_sha256']:
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
        scores.append({'episode_id': ep['episode_id'], 'missing_prediction': predicted is None,
                       'provenance': predicted.get('provenance') if predicted else None,
                       **score_episode(truth, segments, ep['fps'])})
    return {'version': 1, 'dataset_id': manifest['dataset_id'],
            'reference_job_input_modality': manifest['input_modality'],
            'predictor': predictions.get('predictor', {'name': 'unknown', 'input_modality': 'unknown'}),
            'catalog_sha256': manifest['catalog']['catalog_sha256'],
            'reference_sha256': manifest['reference_sha256'], 'prediction_sha256': digest(predictions),
            'policy': {'intervals': 'half_open', 'matching': 'maximum_cardinality_then_iou',
                       'frame_mask': 'human_annotated_frames_only',
                       'segment_scope': 'all_predictions_including_unannotated_gaps',
                       'boundary_errors': 'matched_segments_at_iou_0.5_only'},
            'summary': summarize(scores), 'episodes': scores}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    audit_parser = sub.add_parser('audit')
    audit_parser.add_argument('--root', type=Path, required=True)
    audit_parser.add_argument('--vocabulary', type=Path, required=True)
    audit_parser.add_argument('--out', type=Path, required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--dataset', type=Path, required=True)
    prep.add_argument('--vocabulary', type=Path, required=True)
    prep.add_argument('--episodes', type=int, nargs='+')
    prep.add_argument('--out', type=Path, required=True)
    for name in ('export', 'evaluate'):
        p = sub.add_parser(name)
        p.add_argument('--job', type=Path, required=True)
        p.add_argument('--labels' if name == 'export' else '--predictions', type=Path, required=True)
        p.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'audit':
            paths = sorted(args.root.rglob('lite_annotations/annotation_schema_v1.json'))
            if not paths:
                raise ValueError('no reference schemas found')
            result = []
            for path in paths:
                try:
                    result.append(audit(path.parent.parent, args.vocabulary))
                except (ValueError, KeyError, TypeError, OSError) as error:
                    result.append({'dataset': str(path.parent.parent), 'errors': [str(error)]})
            write_atomic(args.out, result)
            print(f'{len(result)} datasets; {sum(bool(r["errors"]) for r in result)} with errors; {args.out}')
            return int(any(r['errors'] for r in result))
        if args.command == 'prepare':
            result = prepare(args.dataset, args.vocabulary, args.out, args.episodes)
            print(f'Prepared {len(result["episodes"])} RGB episodes; no model calls. {args.out}')
        else:
            result = export(args.job, args.labels) if args.command == 'export' else evaluate(args.job, read(args.predictions))
            write_atomic(args.out, result)
            print(args.out)
        return 0
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(2, f'{error}\n')


if __name__ == '__main__':
    raise SystemExit(main())
