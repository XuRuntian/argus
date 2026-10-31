"""Closed action templates shared by benchmark requests, validation and scoring."""
from __future__ import annotations

import hashlib
import json
import math

from prepare.frame_rate import declared_fps

VERSION = 'argus_actions_v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def canonical_action(action):
    if not isinstance(action, dict):
        raise ValueError('action must be an object')
    for key in ('subject', 'skill'):
        if not isinstance(action.get(key), str) or not action[key].strip():
            raise ValueError(f'action requires {key}')
    if not isinstance(action.get('slots'), dict):
        raise ValueError('action requires structured slots')
    return {key: action[key] for key in ('subject', 'skill', 'slots')}


def action_id(action):
    return digest(canonical_action(action))


def catalog(schema, vocabulary):
    """Only definitions cross this boundary: never episode annotations or task prose."""
    if schema.get('annotation_spec_version') != vocabulary.get('version'):
        raise ValueError('schema and vocabulary versions differ')
    actions = {}
    for skill in schema['skills']:
        rows = skill.get('actions')
        if not isinstance(rows, list) or len(rows) != 1:
            raise ValueError('v1 requires exactly one action per template')
        action = canonical_action(rows[0])
        definition = vocabulary['skills'].get(action['skill'])
        definition = definition or {}
        description = rows[0].get('text') or skill.get('text')
        if not isinstance(description, str) or not description.strip():
            raise ValueError('template requires a description')
        key = action_id(action)
        actions[key] = {'action_id': key, 'primary_action': action,
                        'description': description,
                        'allowed_phases': definition.get('allowed_phases', []),
                        'completion_rule': definition.get('completion_rule'),
                        'rule_source': 'vocabulary' if definition.get('completion_rule') else 'template_text_only'}
    if not actions:
        raise ValueError('empty action catalog')
    result = {'version': VERSION, 'vocabulary_sha256': digest(vocabulary),
              'actions': [actions[key] for key in sorted(actions)]}
    return {**result, 'catalog_sha256': digest(result)}


def validate_contract(contract):
    if contract.get('version') != VERSION:
        raise ValueError('unsupported action contract')
    body = {key: contract[key] for key in ('version', 'vocabulary_sha256', 'actions')}
    if digest(body) != contract.get('catalog_sha256'):
        raise ValueError('action catalog digest mismatch')
    keys = [row['action_id'] for row in contract['actions']]
    if not keys or len(keys) != len(set(keys)):
        raise ValueError('empty or duplicate action IDs')
    for row in contract['actions']:
        if action_id(row['primary_action']) != row['action_id']:
            raise ValueError('action ID does not match its template')


def validate_segments(rows, contract):
    validate_contract(contract)
    if type(contract.get('frame_count')) is not int or contract['frame_count'] <= 0:
        raise ValueError('frame_count must be a positive integer')
    if not isinstance(rows, list):
        raise ValueError('constrained_subtasks must be a list')
    known = {row['action_id'] for row in contract['actions']}
    last = 0
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('subtask must be an object')
        if 'action_id' not in row:
            raise ValueError('subtask requires explicit action_id (null for unmatched activity)')
        start, end = row.get('start_frame'), row.get('end_frame')
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= contract['frame_count']:
            raise ValueError('invalid half-open frame interval')
        if start < last:
            raise ValueError('v1 requires ordered non-overlapping subtasks')
        last = end
        if row.get('action_id') is None:
            if not isinstance(row.get('reason'), str) or not row['reason'].strip():
                raise ValueError('unmatched subtask requires a reason')
        elif row['action_id'] not in known:
            raise ValueError('action_id outside the permitted catalog')
    return rows


def require_blind_context(ctx, *, example_dir=None, dataset_note=None):
    if not ctx.get('action_contract'):
        return
    validate_contract(ctx['action_contract'])
    forbidden = ('piece', 'instruction', 'annotation_subtasks', 'dataset_task_claims',
                 'collection_note', 'uploader_notes')
    if example_dir or dataset_note or any(ctx.get(key) for key in forbidden) or \
            (ctx.get('annotation_policy') or {}).get('dataset_note'):
        raise ValueError('closed-action benchmark requires blind whole-episode inputs without task notes or examples')


def prompt(contract):
    validate_contract(contract)
    rate = declared_fps(contract)
    if type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0:
        raise ValueError('fps must be positive and finite')
    return '''
ADDITIONAL CLOSED-ACTION BENCHMARK OUTPUT
Keep the normal annotation. Also return "constrained_subtasks", an array of objects:
{"start_frame": integer, "end_frame": integer, "action_id": string or null, "reason": string or null}.
Intervals are half-open [start_frame,end_frame). Frame zero is time zero; frame k is at k/fps seconds.
Select ONLY an exact complete template below. Subject, object, destination and all slots must match.
These are unordered possibilities, NOT an expected sequence. Infer order, repetitions and boundaries
from the video. Do not assume every template occurs. One subtask is a complete action, not a subgoal
or an individual phase: e.g. approach, align, grasp, secure and lift belong to one pick. Use the
completion rules. Keep repeated actions separate even when they use the same template.
When rule_source is template_text_only, use the template description and observable action cessation;
no shared vocabulary completion rule exists for that action. Do not substitute another skill.
Cover visible activity in temporal order without overlaps. Use action_id=null with a reason for
activity that cannot match any template; never force a match. A defined 'none' action is a real
template, not a substitute for uncertainty. Do not invent phases that were not observed.
''' + json.dumps(contract, ensure_ascii=True, allow_nan=False)
