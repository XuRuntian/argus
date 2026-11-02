"""Read definitions separately from human answers; build RGB-only benchmark inputs."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import yaml

from label.action_contract import action_id, catalog, digest, validate_segments, generic_catalog, GENERIC_VERSION
from label.atomic import write_atomic


RESOURCES = Path(__file__).parent / 'resources'
DEFAULT_VOCABULARY = RESOURCES / 'skill_spec_v1.yaml'

def read(path):
    return json.loads(Path(path).read_text())


def definitions(dataset, vocabulary):
    return catalog(read(dataset / 'lite_annotations/annotation_schema_v1.json'),
                   yaml.safe_load(vocabulary.read_text()))


def source_episodes(dataset):
    info = read(dataset / 'meta/info.json')
    fps = info['fps']
    if type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0:
        raise ValueError('source FPS must be positive and finite')
    rows = [json.loads(line) for line in (dataset / 'meta/episodes.jsonl').read_text().splitlines() if line.strip()]
    ids = [r['episode_index'] for r in rows]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('empty or duplicate source episodes')
    for row in rows:
        if type(row['episode_index']) is not int or row['episode_index'] < 0 or type(row['length']) is not int or row['length'] <= 0:
            raise ValueError('invalid source episode identity or length')
    return info, rows


def video_files(dataset, info, index):
    keys = sorted(k for k, v in info['features'].items()
                  if v.get('dtype') == 'video' and len(v.get('shape', [])) == 3 and v['shape'][-1] == 3)
    if not keys:
        raise ValueError('no RGB videos')
    files = {}
    for i, key in enumerate(keys):
        path = dataset / info['video_path'].format(episode_index=index,
                episode_chunk=index // info.get('chunks_size', 1000), video_key=key)
        if not path.is_file():
            raise ValueError(f'missing video: {path}')
        files['exo' if i == 0 else f'extra{i}'] = (key, path.resolve())
    return files


def reference(schema, contract, episode):
    selected = [row for row in schema['episode_annotation'] if row['id'] == episode['episode_index']]
    if len(selected) != 1 or selected[0]['frame_count'] != episode['length']:
        raise ValueError('human/source episode identity or frame count mismatch')
    rows = [{'start_frame': row['start_frame'], 'end_frame': row['end_frame'],
             'action_id': action_id(row['primary_action'])} for row in selected[0]['subtasks']]
    if contract['version'] == GENERIC_VERSION:
        # Validate reference intervals independently of the model's vocabulary.
        actions = {r['action_id']: canonical for r, canonical in zip(rows,
                   [s['primary_action'] for s in selected[0]['subtasks']])}
        body = {'version': 'argus_actions_v1', 'vocabulary_sha256': 'reference',
                'actions': [{'action_id': key, 'primary_action': value} for key, value in actions.items()]}
        check = {**body, 'catalog_sha256': digest(body)}
    else:
        check = contract
    validate_segments(rows, {**check, 'frame_count': episode['length']})
    if not rows:
        raise ValueError('empty human annotation')
    return rows


def audit(dataset, vocabulary):
    dataset = Path(dataset).resolve()
    info, episodes = source_episodes(dataset)
    contract = definitions(dataset, vocabulary)
    schema = read(dataset / 'lite_annotations/annotation_schema_v1.json')
    result = {'dataset': str(dataset), 'catalog_sha256': contract['catalog_sha256'],
              'skills_using_default_rules': sorted({r['primary_action']['skill'] for r in contract['actions']
                                                   if r['rule_source'] == 'benchmark_defaults_v1'}),
              'skills_without_completion_rules': sorted({r['primary_action']['skill'] for r in contract['actions']
                                                        if r['rule_source'] == 'template_text_only'}),
              'templates': len(contract['actions']), 'episodes': [], 'errors': []}
    expected = {e['episode_index'] for e in episodes}
    if {e['id'] for e in schema['episode_annotation']} != expected:
        result['errors'].append('human/source episode sets differ')
    for episode in episodes:
        row = {'episode_id': episode['episode_index']}
        try:
            segments = reference(schema, contract, episode)
            files = video_files(dataset, info, episode['episode_index'])
            row.update(segments=len(segments), cameras=len(files),
                       unannotated_frames=episode['length'] - sum(s['end_frame'] - s['start_frame'] for s in segments))
        except (ValueError, KeyError, TypeError) as error:
            row['error'] = str(error)
            result['errors'].append(f'episode {episode["episode_index"]}: {error}')
        result['episodes'].append(row)
    return result


def prepare(dataset, vocabulary, out, selected=None, mode="generic"):
    if mode not in ("generic", "closed"):
        raise ValueError("unknown benchmark mode")
    from label import episode as me, policy
    from prepare.formats import probe, video_views_episode

    dataset, out = Path(dataset).resolve(), Path(out).resolve()
    if out.exists():
        raise ValueError('use a new output directory; existing benchmark artifacts are never overwritten')
    info, episodes = source_episodes(dataset)
    if mode == 'generic':
        rules = yaml.safe_load(vocabulary.read_text())
        contract = generic_catalog(rules, rules.get('phases', {}))
    else:
        contract = definitions(dataset, vocabulary)
    if selected is not None:
        if not selected or set(selected) - {r['episode_index'] for r in episodes}:
            raise ValueError('empty or unknown episode selection')
        episodes = [r for r in episodes if r['episode_index'] in selected]
    schema = None
    if mode == 'closed':
        schema = read(dataset / 'lite_annotations/annotation_schema_v1.json')
        for row in episodes:
            reference(schema, contract, row)
    manifest = {'version': 1, 'dataset_id': digest(str(dataset)), 'dataset_path': str(dataset),
                'input_modality': 'multi_view_rgb_only', 'catalog': contract,
                'mode': mode, 'reference_sha256': digest(schema) if schema is not None else None,
                'episodes': []}
    for row in episodes:
        index, count = row['episode_index'], row['length']
        files = video_files(dataset, info, index)
        facts = {v: probe(path) for v, (_, path) in files.items()}
        for fact in facts.values():
            stamps = np.asarray(fact['pts'], dtype=float) * float(fact['time_base'])
            if len(stamps) != count or not np.allclose(stamps, np.arange(count) / info['fps'], atol=0.001, rtol=0):
                raise ValueError(f'episode {index}: v1 requires synchronized CFR videos with source frame count')
        ep = out / 'episodes' / f'episode_{index:06d}'
        ep.mkdir(parents=True)
        bound = {**contract, 'frame_count': count, 'fps': info['fps']}
        video_views_episode(ep, files, 'teleop_arms', 'closed_action_benchmark',
                            {'action_contract': bound}, prs=facts, shared_clock=True, nominal_fps=info['fps'])
        ctx = read(ep / 'context.json')
        ctx['annotation_policy'] = policy.resolve(ctx, inspection='off', field_dictionary=False)
        write_atomic(ep / 'context.json', ctx)
        policy.require_request_window(ctx, me.request_duration(me.load(ep)))
        manifest['episodes'].append({'episode_id': index, 'directory': ep.name,
                                     'frame_count': count, 'fps': info['fps']})
    write_atomic(out / 'manifest.json', manifest)
    return manifest
