"""Five-field subtask output, opt-in through an episode's subtask_contract."""
from __future__ import annotations

import hashlib
import json
import math


FIELDS = {'start_frame', 'end_frame', 'action', 'subject', 'object'}
START_FIELDS = FIELDS - {'end_frame'}


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


def get_action_vocabulary(value):
    """Absent mode means open for historical contracts and manifests."""
    mode = value.get('action_vocabulary', 'open')
    if mode not in ('open', 'closed'):
        raise ValueError('action_vocabulary must be open or closed')
    return mode


def bind(spec, frame_count, fps, *, action_vocabulary='open'):
    contract = {'spec': spec, 'spec_sha256': digest(spec), 'frame_count': frame_count, 'fps': fps,
                'action_vocabulary': action_vocabulary}
    validate_contract(contract)
    return contract


def validate_contract(contract):
    required = {'spec', 'spec_sha256', 'frame_count', 'fps'}
    if not isinstance(contract, dict) or set(contract) not in (required, required | {'action_vocabulary'}):
        raise ValueError('invalid subtask contract')
    get_action_vocabulary(contract)
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


def materialize(rows, contract):
    """Build intervals from change points; keep legacy five-field replies strictly validated.

    Never sort, clamp, drop or mutate received rows. Validate the whole candidate before
    returning it so a failed reply can retain its original payload.
    """
    validate_contract(contract)
    if isinstance(rows, list) and rows and all(
            isinstance(row, dict) and set(row) == START_FIELDS for row in rows):
        previous = -1
        for i, row in enumerate(rows):
            start = row['start_frame']
            if (type(start) is not int or not previous < start < contract['frame_count']
                    or (i == 0 and start != 0)):
                raise ValueError(f'subtask {i}: start_frame must be an integer, start at 0, '
                                 'strictly increase and remain below frame_count')
            previous = start
        rows = [{**row, 'end_frame': rows[i + 1]['start_frame'] if i + 1 < len(rows)
                 else contract['frame_count']} for i, row in enumerate(rows)]
    return validate(rows, contract)


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
        if get_action_vocabulary(contract) == 'closed' and action is not None and action not in contract['spec']['actions']:
            raise ValueError(f'action {action!r} is not in the closed action vocabulary')
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
    closed = get_action_vocabulary(contract) == 'closed'
    naming = ('''Action vocabulary is closed: action must be one of the listed action names or null.
Do not create new action names. If no listed definition fits, use action=null; never force a match.
Subject names remain extensible: if the actor is clearly identified but no preferred subject fits,
use a short consistent English name, preferably snake_case.''' if closed else '''If the activity or actor is clearly identified but no listed definition fits, use a short,
specific English name, preferably snake_case. Keep that name consistent throughout the episode.
The lists are preferred vocabulary, not exhaustive restrictions. Never force an incorrect match.''')
    free_boundary = ('' if closed else '''For a freely named action, start at its first purposeful motion and end at its observed
completion, cessation or interruption, capped at the end of the episode.
''')
    return f'''

ADDITIONAL ACTION-LEVEL SUBTASK OUTPUT
Keep the normal Argus annotation. Also return a top-level "subtasks" array.
Each entry has exactly these four fields, for example:
{{"start_frame": 0, "action": "pick", "subject": "right_effector", "object": "cup"}}
This is a format example, not an action to assume occurred.
Use whole actions, not internal motion stages or the lasting goals in tasks.
Choose the primary action when independent actions happen simultaneously; use both_effectors
for a joint two-effector action. For hand_over, subject is the giving effector.
Preferred subjects: {', '.join(spec['subjects'])}.
Prefer the listed action and subject names whenever their definitions fit the observation.
Do not invent synonyms for an existing match (use pick, not grab_object, for picking).
{naming}
Left and right refer to the actor's own sides, not the image's sides; use unknown when unclear.
Object is the directly manipulated item or direct interaction target, as a short consistent
English noun phrase. Include visible distinguishing attributes only when needed. Do not use
destinations or grasp parts as objects. For tool use, name the interaction target; for pour,
name the held source container. Use null for an unidentifiable or absent object; none,
return_to_initial_pose and navigate always have object=null. Never guess a hidden identity.
Use action=null for unidentifiable activity, subject=unknown for an unidentifiable actor.
Use none for observed idle activity, not uncertainty. Keep separate repetitions separate.
Return action change points: each entry describes the activity from its start_frame until
the next entry's start_frame. The final entry continues to the end of the video.
Do not output end_frame; the program derives it from the next start or the total frame count.
The first start_frame must be 0. All starts must be strictly increasing integers less than
{contract['frame_count']}. Frame k is k/{contract['fps']} seconds.
Explicitly include idle (none) and unidentifiable (action=null) spans, including at the beginning
or end. When an action ends, start an entry for the activity that follows; do not extend a
finished action across an idle or unknown tail. Do not add an end-of-video sentinel entry.
Follow the action-level boundary definitions below. An interrupted action ends at its observed
interruption; an action already underway starts at frame 0. Do not invent completion.
{free_boundary}If a definition requires a human-specified boundary that has not been supplied, use action=null
instead of inventing that boundary. Sampling limits boundary precision; do not claim to
have observed missing frames. Do not split an action's approach or transport into a separate
action when its definition already includes that motion. Definitions are alternatives, not
an expected sequence. Return only the four-field entries in subtasks, never this dictionary.
ACTION DEFINITIONS (reference only):
{definitions}
END ACTION DEFINITIONS
'''
