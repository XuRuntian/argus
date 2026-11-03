"""Prepare RGB episodes and export five-field subtask annotations."""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import yaml

from label.atomic import write_atomic
from label.subtasks import get_action_vocabulary, bind, digest, validate, validate_spec

DEFAULT_SPEC = Path(__file__).parent / 'resources' / 'skill_spec_v1.yaml'


def read(path):
    return json.loads(Path(path).read_text())


def rgb_files(dataset, info, index):
    from prepare.formats import names_depth
    files = {}
    for key, feature in sorted(info['features'].items()):
        if (feature.get('dtype') != 'video' or names_depth(key)
                or (feature.get('info') or {}).get('video.is_depth_map')):
            continue
        shape, names = feature.get('shape', []), feature.get('names') or []
        if len(shape) != 3:
            continue
        axis = next((i for i, n in enumerate(names) if n in ('channel', 'channels')), None)
        if axis is not None:
            rgb = axis < 3 and shape[axis] == 3
        else:
            rgb = shape[-1] == 3 or shape[0] == 3
        if not rgb:
            continue
        path = dataset / info['video_path'].format(
            episode_index=index, episode_chunk=index // info.get('chunks_size', 1000), video_key=key)
        if not path.is_file():
            raise ValueError(f'missing RGB video: {path}')
        files['exo' if not files else f'extra{len(files)}'] = (key, path.resolve())
    if not files:
        raise ValueError('no RGB videos')
    return files


def prepare(dataset, out, spec_path=DEFAULT_SPEC, selected=None, *, action_vocabulary='open'):
    from label import episode as me, policy
    from prepare.formats import probe, video_views_episode

    dataset, out = Path(dataset).resolve(), Path(out).resolve()
    spec = yaml.safe_load(Path(spec_path).read_text())
    validate_spec(spec)
    # Validate the mode before creating any output directory.
    get_action_vocabulary({'action_vocabulary': action_vocabulary})
    info = read(dataset / 'meta/info.json')
    fps = info['fps']
    if type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0:
        raise ValueError('source FPS must be positive and finite')
    episodes = [json.loads(line) for line in (dataset / 'meta/episodes.jsonl').read_text().splitlines()
                if line.strip()]
    if not episodes or any(type(e.get('episode_index')) is not int or e['episode_index'] < 0
                           or type(e.get('length')) is not int or e['length'] <= 0 for e in episodes):
        raise ValueError('invalid source episode identity or length')
    ids = {e['episode_index'] for e in episodes}
    if len(ids) != len(episodes):
        raise ValueError('duplicate source episodes')
    if selected is not None:
        if not selected or set(selected) - ids:
            raise ValueError('empty or unknown episode selection')
        episodes = [e for e in episodes if e['episode_index'] in selected]
    manifest = {'version': 1, 'dataset_id': digest(str(dataset)), 'input_modality': 'multi_view_rgb_only',
                'spec': spec, 'spec_sha256': digest(spec), 'action_vocabulary': action_vocabulary,
                'episodes': []}
    # Exclusive ownership: never reuse or remove an existing job, even after a failure.
    out.mkdir(parents=True, exist_ok=False)
    try:
        for row in episodes:
            index, count = row['episode_index'], row['length']
            files = rgb_files(dataset, info, index)
            facts = {view: probe(path) for view, (_, path) in files.items()}
            for fact in facts.values():
                times = np.asarray(fact['pts'], dtype=float) * float(fact['time_base'])
                if len(times) != count or not np.allclose(times, np.arange(count) / fps, atol=.001, rtol=0):
                    raise ValueError(f'episode {index}: requires synchronized CFR videos starting at zero')
            ep = out / 'episodes' / f'episode_{index:06d}'
            video_views_episode(ep, files, 'teleop_arms', 'subtask_annotation',
                                {'subtask_contract': bind(spec, count, fps, action_vocabulary=action_vocabulary)},
                                prs=facts, shared_clock=True, nominal_fps=fps)
            ctx = read(ep / 'context.json')
            ctx['annotation_policy'] = policy.resolve(ctx, inspection='off', field_dictionary=False)
            write_atomic(ep / 'context.json', ctx)
            policy.require_request_window(ctx, me.request_duration(me.load(ep)))
            manifest['episodes'].append({'episode_id': index, 'directory': ep.name,
                                         'frame_count': count, 'fps': fps})
        write_atomic(out / 'manifest.json', manifest)
    except BaseException:
        shutil.rmtree(out)
        raise
    return manifest


def export(job, labels):
    from label.harness import annotation_complete, saved_input_problem

    job, labels = Path(job), Path(labels)
    manifest = read(job / 'manifest.json')
    if manifest.get('version') != 1 or digest(manifest['spec']) != manifest['spec_sha256']:
        raise ValueError('invalid manifest version or specification digest')
    mode = get_action_vocabulary(manifest)
    episodes, seen = [], set()
    for ep in manifest['episodes']:
        index = ep['episode_id']
        if type(index) is not int or index < 0 or index in seen or ep['directory'] != f'episode_{index:06d}':
            raise ValueError('invalid or duplicate manifest episode')
        seen.add(index)
        path = labels / (ep['directory'] + '.json')
        record = read(path)
        if record.get('dry_run') or not annotation_complete(record):
            raise ValueError(f'{path}: missing or incomplete annotation; raw reply is retained')
        contract = bind(manifest['spec'], ep['frame_count'], ep['fps'], action_vocabulary=mode)
        saved_contract = record.get('subtask_contract')
        if (not isinstance(saved_contract, dict)
                or {'action_vocabulary': 'open', **saved_contract} != contract):
            raise ValueError(f'{path}: request contract differs from manifest')
        identity = record.get('input_identity')
        if not isinstance(identity, dict) or not identity.get('source_proof'):
            raise ValueError(f'{path}: missing input proof')
        problem = saved_input_problem(record, job / 'episodes' / ep['directory'])
        if problem:
            raise ValueError(f'{path}: {problem}')
        rows = validate(record['labels'].get('subtasks'), contract)
        episodes.append({'episode_id': index, 'frame_count': ep['frame_count'], 'fps': ep['fps'],
                         'subtasks': rows,
                         'provenance': {key: record.get(key) for key in
                                        ('model', 'model_served', 'reasoning_effort', 'annotation_policy', 'sampling')},
                         'sampled_times_s': (record.get('config') or {}).get('timesteps_s')})
    if not episodes:
        raise ValueError('manifest has no episodes')
    return {'version': 1, 'spec_version': manifest['spec']['version'], 'spec_sha256': manifest['spec_sha256'],
            'dataset_id': manifest['dataset_id'], 'interval_convention': '[start_frame, end_frame)',
            'input_modality': manifest['input_modality'], 'action_vocabulary': mode, 'episodes': episodes}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--dataset', type=Path, required=True)
    prep.add_argument('--out', type=Path, required=True)
    prep.add_argument('--spec', type=Path, default=DEFAULT_SPEC)
    prep.add_argument('--episodes', type=int, nargs='+')
    prep.add_argument('--action-vocabulary', choices=('open', 'closed'), default='open',
                      help='open allows new action names; closed restricts actions to the spec (null is allowed)')
    output = sub.add_parser('export')
    output.add_argument('--job', type=Path, required=True)
    output.add_argument('--labels', type=Path, required=True)
    output.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'prepare':
            result = prepare(args.dataset, args.out, args.spec, args.episodes,
                             action_vocabulary=args.action_vocabulary)
            print(f'Prepared {len(result["episodes"])} RGB episodes: {args.out}')
        else:
            write_atomic(args.out, export(args.job, args.labels))
            print(args.out)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(2, f'{error}\n')


if __name__ == '__main__':
    main()
