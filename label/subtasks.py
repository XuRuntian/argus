"""Five-field subtask output, opt-in through an episode's subtask_contract."""
from __future__ import annotations

import hashlib
import json
import math


FIELDS = {'start_frame', 'end_frame', 'action', 'subject', 'object'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def validate_spec(spec):
    if not isinstance(spec, dict) or set(spec) != {'version', 'subjects', 'actions'}:
        raise ValueError('spec requires only version, subjects and actions (no phases or slots)')
    if not isinstance(spec['version'], str) or not spec['version'].strip():
        raise ValueError('spec requires a version')
    subjects = spec['subjects']
    if (not isinstance(subjects, list) or not subjects
            or any(not isinstance(s, str) or not s.strip() for s in subjects)
            or len(set(subjects)) != len(subjects) or 'unknown' not in subjects):
        raise ValueError('subjects must be unique names including unknown')
    if not isinstance(spec['actions'], dict) or not spec['actions'] or 'none' not in spec['actions']:
        raise ValueError('actions must include none for idle footage')
    for name, row in spec['actions'].items():
        if (not isinstance(name, str) or not name.strip() or not isinstance(row, dict)
                or set(row) != {'meaning', 'start', 'end'}
                or any(not isinstance(v, str) or not v.strip() for v in row.values())):
            raise ValueError('each action requires only meaning, start and end text')


def bind(spec, frame_count, fps):
    contract = {'spec': spec, 'spec_sha256': digest(spec), 'frame_count': frame_count, 'fps': fps}
    validate_contract(contract)
    return contract


def validate_contract(contract):
    if not isinstance(contract, dict) or set(contract) != {'spec', 'spec_sha256', 'frame_count', 'fps'}:
        raise ValueError('invalid subtask contract')
    validate_spec(contract['spec'])
    if digest(contract['spec']) != contract['spec_sha256']:
        raise ValueError('subtask spec digest mismatch')
    if type(contract['frame_count']) is not int or contract['frame_count'] <= 0:
        raise ValueError('frame_count must be a positive integer')
    fps = contract['fps']
    if type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0:
        raise ValueError('fps must be positive and finite')


def require_context(ctx, *, example_dir=None, dataset_note=None):
    if 'subtask_contract' not in ctx:
        return
    validate_contract(ctx['subtask_contract'])
    if (example_dir or dataset_note or (ctx.get('annotation_policy') or {}).get('dataset_note')
            or any(ctx.get(k) for k in ('piece', 'instruction', 'annotation_subtasks',
                                        'dataset_task_claims', 'collection_note', 'uploader_notes'))):
        raise ValueError('subtask annotation requires whole episodes without task notes or examples')


def validate(rows, contract):
    """Validate without rewriting or dropping a model's segments."""
    validate_contract(contract)
    if not isinstance(rows, list) or not rows:
        raise ValueError('subtasks must be a nonempty list, including none for idle footage')
    last = 0
    for row in rows:
        if not isinstance(row, dict) or set(row) != FIELDS:
            raise ValueError('each subtask requires exactly start_frame, end_frame, action, subject, object')
        start, end = row['start_frame'], row['end_frame']
        if (type(start) is not int or type(end) is not int or start != last
                or not start < end <= contract['frame_count']):
            raise ValueError('subtasks must cover the episode in ordered, contiguous half-open frame intervals')
        last = end
        action, subject, obj = row['action'], row['subject'], row['object']
        if action is not None and (not isinstance(action, str) or not action.strip() or action != action.strip()):
            raise ValueError('action must be a nonempty trimmed name or null')
        if not isinstance(subject, str) or not subject.strip() or subject != subject.strip():
            raise ValueError('subject must be a nonempty trimmed name (unknown if unidentified)')
        if obj is not None and (not isinstance(obj, str) or not obj.strip() or obj != obj.strip()):
            raise ValueError('object must be a nonempty trimmed name or null')
        if action in ('none', 'return_to_initial_pose', 'navigate') and obj is not None:
            raise ValueError('object must be null for idle, reset or navigation')
    if last != contract['frame_count']:
        raise ValueError('subtasks must reach frame_count')
    return rows


def prompt(contract):
    validate_contract(contract)
    spec = contract['spec']
    # Definitions are prose, not a second JSON object the model might copy as its answer.
    definitions = '\n'.join(f"- {name}: {row['meaning']} Start: {row['start']} End: {row['end']}"
                            for name, row in spec['actions'].items())
    return f'''

ADDITIONAL ACTION-LEVEL SUBTASK OUTPUT
Keep the normal Argus annotation. Also return a top-level "subtasks" array.
Each entry has exactly these five fields, for example:
{{"start_frame": 0, "end_frame": 1, "action": "pick", "subject": "right_effector", "object": "cup"}}
This is a format example, not an action to assume occurred.
Use whole actions, not internal motion stages or the lasting goals in tasks.
Choose the primary action when independent actions happen simultaneously; use both_effectors
for a joint two-effector action. For hand_over, subject is the giving effector.
Preferred subjects: {', '.join(spec['subjects'])}.
Prefer the listed action and subject names whenever their definitions fit the observation.
Do not invent synonyms for an existing match (use pick, not grab_object, for picking).
If the activity or actor is clearly identified but no listed definition fits, use a short,
specific English name, preferably snake_case. Keep that name consistent throughout the episode.
The lists are preferred vocabulary, not exhaustive restrictions. Never force an incorrect match.
Left and right refer to the actor's own sides, not the image's sides; use unknown when unclear.
Object is the directly manipulated item or direct interaction target, as a short consistent
English noun phrase. Include visible distinguishing attributes only when needed. Do not use
destinations or grasp parts as objects. For tool use, name the interaction target; for pour,
name the held source container. Use null for an unidentifiable or absent object; none,
return_to_initial_pose and navigate always have object=null. Never guess a hidden identity.
Use action=null for unidentifiable activity, subject=unknown for an unidentifiable actor.
Use none for observed idle activity, not uncertainty. Keep separate repetitions separate.
Cover the entire episode with ordered contiguous non-overlapping intervals [start_frame,end_frame).
Frame k is k/{contract['fps']} seconds; start at 0 and end at {contract['frame_count']}.
Follow the action-level boundary definitions below. An interrupted action ends at its observed
interruption; an action already underway starts at frame 0. Do not invent completion.
For a freely named action, start at its first purposeful motion and end at its observed
completion, cessation or interruption, capped at the end of the episode.
If a definition requires a human-specified boundary that has not been supplied, use action=null
instead of inventing that boundary. Sampling limits boundary precision; do not claim to
have observed missing frames. Do not split an action's approach or transport into a separate
action when its definition already includes that motion. Definitions are alternatives, not
an expected sequence. Return only the five-field entries in subtasks, never this dictionary.
ACTION DEFINITIONS (reference only):
{definitions}
END ACTION DEFINITIONS
'''
