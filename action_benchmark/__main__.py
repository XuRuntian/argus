"""Prepare video inputs and export annotations in the shared skill specification."""
from __future__ import annotations

import argparse
from pathlib import Path

from action_benchmark.dataset import prepare, read, DEFAULT_VOCABULARY
from label.action_contract import validate_segments, GENERIC_VERSION
from label.atomic import write_atomic


def export(job, labels):
    from label.harness import annotation_complete
    manifest = read(job / 'manifest.json')
    templates = {r['action_id']: r['primary_action'] for r in manifest['catalog'].get('actions', [])}
    episodes = []
    for ep in manifest['episodes']:
        path = labels / (ep['directory'] + '.json')
        if not path.exists():
            raise ValueError(f'{path}: missing annotation; export requires every selected episode')
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
                         'subtasks': [{**r, 'primary_action': r.get('primary_action') if manifest['catalog']['version'] == GENERIC_VERSION else templates.get(r['action_id'])} for r in rows]})
    return {'version': 1, 'annotation_spec_version': manifest['catalog'].get('annotation_spec_version', 'skill_spec_v1'),
            'interval_convention': '[start_frame, end_frame)',
            'dataset_id': manifest['dataset_id'],
            'predictor': {'name': 'argus', 'input_modality': manifest['input_modality']},
            'catalog_sha256': manifest['catalog']['catalog_sha256'], 'episodes': episodes}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--dataset', type=Path, required=True)
    prep.add_argument('--vocabulary', type=Path, default=DEFAULT_VOCABULARY)
    prep.add_argument('--episodes', type=int, nargs='+')
    prep.add_argument('--out', type=Path, required=True)
    output = sub.add_parser('export')
    output.add_argument('--job', type=Path, required=True)
    output.add_argument('--labels', type=Path, required=True)
    output.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'prepare':
            result = prepare(args.dataset, args.vocabulary, args.out, args.episodes, mode="generic")
            print(f'Prepared {len(result["episodes"])} RGB episodes; no model calls. {args.out}')
        else:
            result = export(args.job, args.labels)
            write_atomic(args.out, result)
            print(args.out)
        return 0
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(2, f'{error}\n')


if __name__ == '__main__':
    raise SystemExit(main())
